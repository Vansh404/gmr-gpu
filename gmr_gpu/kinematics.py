"""
Step 7a (part 1) of GMR-GPU: a pure-functional batched kinematics engine.

WHY THIS EXISTS (probe findings, 2026-07-02):
The repo's KinematicsModel is a fine reference but cannot be the production
engine: its FK mutates buffers in place (`joint_rot[..., j-1, :] = ...`), which
is fundamentally illegal under torch.func.vmap, and its quat ops are
@torch.jit.script, whose cold-start profiling graphs crash under functorch
("accessing `data` under vmap"). So GMR-GPU gets its own module with the rules:

  RULE 1: no in-place writes  -> build lists, torch.stack (vmap-legal)
  RULE 2: no TorchScript      -> plain torch ops (functorch/compile-legal)
  RULE 3: no branches on data -> sinc-style smooth guards (grad-safe at 0)
  RULE 4: batched-native      -> everything takes (..., d), leading dims free

Certification: this module must pass the SAME harnesses that certified the
reference stack (steps 1-3): FK parity vs MuJoCo, twist parity vs mink,
Jacobian triangulation -- run scratch/certify_batched_kin.py.

All quats xyzw. Geometry comes from the COMPILED mjModel (step-3 lesson).
"""

import numpy as np
import torch
import mujoco as mj


# --------------------------------------------------------------------------
# Plain quaternion ops (xyzw), out-of-place, batched over leading dims.
# --------------------------------------------------------------------------
def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_rotate(q, v):
    """Rotate vector v by unit quat q: v + 2*w*(u x v) + 2*(u x (u x v))."""
    u, w = q[..., :3], q[..., 3:4]
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def so3_exp(w):
    """Axis-angle (...,3) -> unit quat (...,4). Smooth at 0 (sinc form)."""
    theta = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-40)
    half = 0.5 * theta
    return torch.cat([(torch.sin(half) / theta) * w, torch.cos(half)], dim=-1)


def so3_log(q):
    """Unit quat (...,4) -> axis-angle (...,3). Shortest arc (double-cover safe
    via w-sign flip); smooth at identity (atan2/sinc-style guard)."""
    q = torch.where(q[..., 3:4] < 0, -q, q)         # canonical hemisphere
    n = torch.sqrt((q[..., :3] ** 2).sum(-1, keepdim=True) + 1e-40)
    return 2.0 * torch.atan2(n, q[..., 3:4]) / n * q[..., :3]


def hinge_quat(axis, theta):
    """Constant unit axis (...,3), angle (...,1) -> quat. Exact, no guards."""
    half = 0.5 * theta
    return torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)


# --------------------------------------------------------------------------
# SE(3) twist error (mink convention), plain ops. Mirrors se3_error.py.
# --------------------------------------------------------------------------
def _skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([
        torch.stack([o, -z, y], -1),
        torch.stack([z, o, -x], -1),
        torch.stack([-y, x, o], -1),
    ], dim=-2)


def se3_twist(cur_quat, cur_pos, tgt_quat, tgt_pos):
    """e = log(T_cur^-1 T_tgt), (...,6) = [Vinv(w)@t_rel, w]. Batched, plain."""
    q_rel = quat_mul(quat_conjugate(cur_quat), tgt_quat)
    t_rel = quat_rotate(quat_conjugate(cur_quat), tgt_pos - cur_pos)
    omega = so3_log(q_rel)

    theta2 = (omega * omega).sum(-1)
    S = _skew(omega)
    S2 = S @ S
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(S.shape)
    theta = torch.sqrt(theta2 + 1e-40)
    half = 0.5 * theta
    # coef -> 1/12 smoothly as theta -> 0 (Taylor of the closed form)
    coef = (1.0 - 0.5 * theta * torch.cos(half) / torch.sin(half).clamp_min(1e-40)) / theta2.clamp_min(1e-30)
    coef = torch.where(theta2 < 1e-10, torch.full_like(coef, 1.0 / 12.0), coef)
    Vinv = I - 0.5 * S + coef[..., None, None] * S2
    rho = (Vinv @ t_rel.unsqueeze(-1)).squeeze(-1)
    return torch.cat([rho, omega], dim=-1)


# --------------------------------------------------------------------------
# The robot: static geometry extracted ONCE from the compiled mjModel.
# --------------------------------------------------------------------------
class BatchedRobot:
    """Immutable per-robot data + a pure-functional batched FK.

    Assumes (true for all GMR robots): body 0 is the floating base; each other
    body has 0 or 1 hinge joint. All buffers are plain tensors; fk() never
    mutates them.
    """
    """
    init defines immutable specs of the robot
    1) body_names	38 strings	the links, in tree order (pelvis, left_hip_pitch_link, …)
    2) parents	38 ints	the skeleton topology — each body's parent index (-1 = pelvis/root)
    3) local_pos, local_quat	(38,3), (38,4)	the geometry — each body's fixed offset from its parent: effectively bone lengths and mounting rotations
    4) hinge_axis	(38,3)	each joint's rotation axis, in the child body's frame
    5) hinge_dof	38 ints	which of the 29 qpos entries drives each body (-1 = welded, no joint)
    6) lo, hi	(29,)	joint limits — what the clamp projects into
    7) num_dof 29, scalar

    plus two methods
    fk(pos, quat, dof) — "given this skeleton spec and this pose, where is every link in the world?"
    task_error(...) — FK + gather the 14 task links +  step-2 twist against targets.
    """
    def __init__(self, xml_path, device="cpu", dtype=torch.float64):
        self.device, self.dtype = device, dtype
        model = mj.MjModel.from_xml_path(str(xml_path))
        names, parents, hinge_axis, hinge_dof = [], [], [], []
        lo, hi = [], []
        dof_counter = 0
        # walk MuJoCo bodies 1..nbody-1 (skip world); their order is the tree order
        for b in range(1, model.nbody):
            names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b))
            parents.append(model.body_parentid[b] - 1)     # shift: world removed
            jadr, jnum = model.body_jntadr[b], model.body_jntnum[b]
            if b == 1:
                assert jnum == 1 and model.jnt_type[jadr] == mj.mjtJoint.mjJNT_FREE
                hinge_axis.append(np.zeros(3)); hinge_dof.append(-1)
            elif jnum == 0:
                hinge_axis.append(np.zeros(3)); hinge_dof.append(-1)
            else:
                assert jnum == 1 and model.jnt_type[jadr] == mj.mjtJoint.mjJNT_HINGE, \
                    f"body {names[-1]}: only 0/1 hinge joints supported"
                hinge_axis.append(model.jnt_axis[jadr].copy())
                hinge_dof.append(dof_counter); dof_counter += 1
                lo.append(model.jnt_range[jadr][0]); hi.append(model.jnt_range[jadr][1])

        tt = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device)
        self.body_names = names
        self.parents = parents                                  # python ints (static loop)
        self.num_dof = dof_counter
        self.local_pos = tt(model.body_pos[1:])                 # (J,3)  compiled geometry
        self.local_quat = tt(model.body_quat[1:][:, [1, 2, 3, 0]])  # wxyz->xyzw
        self.hinge_axis = tt(hinge_axis)                        # (J,3)
        self.hinge_dof = hinge_dof                              # python ints
        self.lo, self.hi = tt(lo), tt(hi)                       # (num_dof,)

    def fk(self, root_pos, root_rot, dof): #textbook FK
        """(...,3),(...,4 xyzw),(...,num_dof) -> body_pos (...,J,3), body_rot (...,J,4).
        Pure functional: python loop over the static tree, torch.stack at the end.
        MuJoCo tree order guarantees parent index < child index, so pos[p] always
        exists. The floating base's world pose IS (root_pos, root_rot) -- its XML
        local offset is ignored, exactly as MuJoCo treats a free joint.
        
        FK serves this kinematic mapping
        
        (SE(3)×ℝ²⁹) —> fk→ SE(3)¹⁴(world frame)

        floating base SE(3) + 29 hinge angles = 35 DOF => 14 "end effectors" ie pelvis, hips, knees, toes, torso, shoulders, elbows, wrists


        quantity                                    frame
        root_pos/root_rot (input)	                world
        dof (input)	                                joint coordinates(frameless)
        local_pos/local_quat (rob's geometry)       each parent link's frame
        hinge_axis	                                each child link's frame
        FK outputs, targets	                        world
        twist error e	                            each task link's body frame
        tangent step ξ base part	                world (translation), world left-multiply (rotation) 

        """
        pos, rot = [], []
        for i in range(len(self.parents)):
            p = self.parents[i]
            if p < 0:                                   # floating base
                pos.append(root_pos); rot.append(root_rot); continue
            parent_pos, parent_rot = pos[p], rot[p]
            world_off = quat_rotate(parent_rot, self.local_pos[i].expand_as(parent_pos))
            frame_rot = quat_mul(parent_rot, self.local_quat[i].expand_as(parent_rot))
            d = self.hinge_dof[i]
            if d >= 0:
                jq = hinge_quat(self.hinge_axis[i].expand_as(parent_pos),
                                dof[..., d:d + 1])
                frame_rot = quat_mul(frame_rot, jq)
            pos.append(parent_pos + world_off)
            rot.append(frame_rot)
        return torch.stack(pos, dim=-2), torch.stack(rot, dim=-2)

    def task_error(self, root_pos, root_rot, dof, frame_idx, tgt_quat, tgt_pos):
        """Batched stacked twist: (...,3/4/nd) + targets (...,K,4/3) -> (...,6K).
        frame_idx: LongTensor (K,) into body rows."""
        bp, br = self.fk(root_pos, root_rot, dof)
        cur_p = bp.index_select(-2, frame_idx)
        cur_q = br.index_select(-2, frame_idx)
        e = se3_twist(cur_q, cur_p, tgt_quat, tgt_pos)          # (...,K,6)
        return e.reshape(*e.shape[:-2], -1)                     # (...,6K)
