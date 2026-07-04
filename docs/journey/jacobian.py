"""
Step 3 of GMR-GPU: the task Jacobian J = de/dq, via autograd.

J linearizes the twist error e (step 2) w.r.t. the configuration: "which way do
I move the DOFs to shrink e". For K tasks it's (6K x nv). For G1, nv = 35 =
6 (floating base) + 29 (joints).

The base is a MANIFOLD (SE3), so we can't autograd w.r.t. the raw quaternion.
Instead we introduce a tangent perturbation xi in R^35, retract it onto the
configuration, and differentiate at xi=0:

    xi   = [ base_twist(6) , joint_deltas(29) ]          # starts at 0
    q(xi)= ( retract(T0, xi_base) , theta0 + xi_joints ) # onto the config
    e(xi)= se3_twist( FK(q(xi)) , targets )              # step-1 FK + step-2 twist
    J    = de/dxi  at xi=0                                # autograd

Because we differentiate the REAL e (full SE3 log), J is automatically
consistent with it -- no hand-derived jlog needed.

Validation (two oracles + a diagnostic):
  - JOINT columns (29): compared to mink's analytic FrameTask.compute_jacobian
    (exact oracle -- joint tangent is convention-free).
  - BASE columns (6): central finite differences (our tangent convention).
  - h-SWEEP on the FD: a correct autograd J makes FD error trace a U-curve
    (bottoms ~1e-10); a BUG makes it plateau at the bug's size regardless of h.
    That's what tells a real error from FD noise.

Run:  python scratch/jacobian.py
"""

import numpy as np
import torch
import mujoco as mj
import mink
from mink.lie import SE3

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.torch_utils import quat_mul, exp_map_to_quat

from fk_parity import make_qpos          # reuse step-1 qpos synthesis
from se3_error import se3_twist          # reuse step-2 twist

DEVICE = "cpu"
DTYPE = torch.float64                    # validate the math cleanly in f64
ROBOT = "unitree_g1"

# A representative set of IK task frames (all exist as bodies in the G1 xml).
TASK_FRAMES = [
    "pelvis", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link",
]


# --------------------------------------------------------------------------
# Helpers (written for you).
# --------------------------------------------------------------------------
def so3_exp_quat(w):
    """SO(3) exp: axis-angle (3,) -> unit quat (4,) xyzw. Written this way (via
    sinc) instead of torch_utils.exp_map_to_quat because the latter guards a 0/0
    with torch.where, which returns a clean value but a NaN *gradient* at w=0 --
    and we differentiate the retraction exactly at w=0. This form is smooth there
    (d(xyz)/dw = 0.5*I). This is the autograd-through-a-retraction gotcha."""
    theta = torch.sqrt((w * w).sum() + 1e-40)      # smooth, never exactly 0
    half = 0.5 * theta
    xyz = (torch.sin(half) / theta) * w            # 0.5*sinc(half)*w  ->0.5*w as theta->0
    return torch.cat([xyz, torch.cos(half).view(1)])


def to_double(kin):
    """KinematicsModel builds its buffers in float32; cast to f64 so FK runs in
    double for a clean convention check (precision is a separate concern)."""
    kin._local_translation = kin._local_translation.double()
    kin._local_rotation = kin._local_rotation.double()
    for j in kin._joints:
        if j._axis is not None:
            j._axis = j._axis.double()


def use_compiled_geometry(kin, model):
    """Source body frames from MuJoCo's COMPILED model (body_pos, body_quat)
    instead of KinematicsModel's XML-string parse. The XML writes quaternions to
    ~7 digits and doesn't renormalize -> ~2e-7 per-body error that compounds to
    ~1e-6 in FK. That -- not float32 -- is what capped step-1 parity. Compiled
    geometry is full-precision, so FK (and thus J) then matches MuJoCo to machine
    eps. The real GPU solver should load geometry this way, not from XML."""
    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in kin.body_names]
    kin._local_translation = torch.tensor(model.body_pos[ids], dtype=DTYPE, device=DEVICE)
    kin._local_rotation = torch.tensor(model.body_quat[ids][:, [1, 2, 3, 0]],  # wxyz->xyzw
                                       dtype=DTYPE, device=DEVICE)


def task_error(kin, root_pos, root_rot, dof, frame_idx, tgt_quat, tgt_pos):
    """Stacked twist e for all task frames at config (root_pos, root_rot, dof).
    Reuses step-1 FK + step-2 se3_twist. Returns (6K,)."""
    bp, br = kin.forward_kinematics(root_pos.view(1, 3), root_rot.view(1, 4), dof.view(1, -1))
    bp = bp.squeeze(0); br = br.squeeze(0)               # (Jbody,3), (Jbody,4) xyzw
    es = []
    for i, fidx in enumerate(frame_idx):
        e_i = se3_twist(br[fidx].view(1, 4), bp[fidx].view(1, 3),
                        tgt_quat[i].view(1, 4), tgt_pos[i].view(1, 3))
        es.append(e_i.view(-1))
    return torch.cat(es)                                 # (6K,)


# --------------------------------------------------------------------------
# THE THING UNDER TEST (fill the TODOs).
# --------------------------------------------------------------------------
def retract(pos0, quat0, dof0, xi):
    """Apply tangent perturbation xi (35,) onto the base config (pos0(3),
    quat0(4) xyzw, dof0(29)). Returns (pos, quat, dof)."""
    v = xi[0:3]          # base translation perturbation (world frame)
    w = xi[3:6]          # base rotation perturbation (axis-angle, 3-vec)
    dtheta = xi[6:]      # joint-angle perturbations (29)

    # retract.
    #   - translation is a flat vector space: pos = pos0 + v
    #   - joints are a flat vector space too:  dof = dof0 + dtheta
    #   - rotation is the MANIFOLD part:
    pos = pos0 + v #perturb the pos
    quat = quat_mul(so3_exp_quat(w).view(1,4),quat0.view(1,4))
    dof = dof0 + dtheta

    return pos, quat, dof


def compute_jacobian(kin, q0, frame_idx, tgt_quat, tgt_pos):
    """Return (J (6K,35), e_of_xi) where e_of_xi(xi) is the residual closure
    (also handed to the finite-difference validator)."""
    pos0, quat0, dof0 = q0

    def e_of_xi(xi):
        pos, quat, dof = retract(pos0, quat0, dof0, xi)
        return task_error(kin, pos, quat, dof, frame_idx, tgt_quat, tgt_pos)

    xi0 = torch.zeros(6 + dof0.numel(), dtype=DTYPE, device=DEVICE)

    #  J = Jacobian of e_of_xi w.r.t. xi, evaluated at xi0.
    #   torch.autograd.functional.jacobian(func, inputs) returns (out_dim, in_dim)
    #   = (6K, 35) 
    #Forward pass storing intermediates, then propagate derivatives backward through the chain rule. 
    # One reverse pass computes a vector-Jacobian product vᵀJ — i.e. it gives you one row of J (one output's gradient w.r.t. all inputs). 
    # To get the full m×n Jacobian you need m reverse passes, one per output.
    # J = torch.autograd.functional.jacobian(e_of_xi,xi0)   # loop mode: m sequential backward passes (~2 s at 84 rows)
    J = torch.func.jacrev(e_of_xi)(xi0)                      # one vmapped reverse sweep (~24x faster, bit-identical)

    return J, e_of_xi


# --------------------------------------------------------------------------
# Oracle 1: mink's analytic task Jacobian (written).
# --------------------------------------------------------------------------
def mink_jacobian(model, q0_np, frame_names, tgt_quat_xyzw, tgt_pos):
    config = mink.Configuration(model)
    config.update(q=q0_np)
    def xyzw_to_wxyz(q):
        return q[[3, 0, 1, 2]]
    blocks = []
    for i, name in enumerate(frame_names):
        task = mink.FrameTask(name, "body", position_cost=1.0, orientation_cost=1.0)
        tq = xyzw_to_wxyz(tgt_quat_xyzw[i].cpu().numpy())
        task.set_target(SE3(wxyz_xyz=np.concatenate([tq, tgt_pos[i].cpu().numpy()])))
        blocks.append(task.compute_jacobian(config))     # (6, nv)
    return np.vstack(blocks)                              # (6K, nv)


# --------------------------------------------------------------------------
# Oracle 2: central finite differences (written).
# --------------------------------------------------------------------------
def fd_jacobian(e_of_xi, xi0, h):
    n = xi0.numel()
    e0 = e_of_xi(xi0)
    J = torch.zeros(e0.numel(), n, dtype=xi0.dtype, device=xi0.device)
    for i in range(n):
        d = torch.zeros_like(xi0); d[i] = h
        J[:, i] = (e_of_xi(xi0 + d) - e_of_xi(xi0 - d)) / (2 * h)
    return J


# --------------------------------------------------------------------------
# Harness.
# --------------------------------------------------------------------------
def make_targets(kin, q0, frame_idx, rng):
    """Targets = current frame poses nudged by a small random SE3, so e is
    nonzero and well-conditioned (away from singularities)."""
    pos0, quat0, dof0 = q0
    bp, br = kin.forward_kinematics(pos0.view(1, 3), quat0.view(1, 4), dof0.view(1, -1))
    bp = bp.squeeze(0); br = br.squeeze(0)
    tq, tp = [], []
    for fidx in frame_idx:
        dw = torch.tensor(rng.uniform(-0.2, 0.2, 3), dtype=DTYPE)
        dquat = exp_map_to_quat(dw.view(1, 3)).view(1, 4)
        tq.append(quat_mul(dquat, br[fidx].view(1, 4)).view(4))
        tp.append(bp[fidx] + torch.tensor(rng.uniform(-0.1, 0.1, 3), dtype=DTYPE))
    return torch.stack(tq), torch.stack(tp)


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    kin = KinematicsModel(xml, device=DEVICE); to_double(kin)
    use_compiled_geometry(kin, model)          # match MuJoCo geometry to machine eps
    rng = np.random.default_rng(0)

    frame_idx = [kin.get_body_idx(n) for n in TASK_FRAMES]

    # a random-but-valid configuration
    qpos = make_qpos(model, "random", rng)
    qt = torch.tensor(qpos, dtype=DTYPE, device=DEVICE)
    q0 = (qt[0:3], qt[3:7][[1, 2, 3, 0]], qt[7:])        # pos, quat xyzw, dof

    tgt_quat, tgt_pos = make_targets(kin, q0, frame_idx, rng)

    J, e_of_xi = compute_jacobian(kin, q0, frame_idx, tgt_quat, tgt_pos)
    if J is None:
        print("compute_jacobian returned None -- fill the TODOs.")
        return
    xi0 = torch.zeros(6 + q0[2].numel(), dtype=DTYPE, device=DEVICE)

    print(f"=== task Jacobian validation (K={len(TASK_FRAMES)} frames, shape {tuple(J.shape)}) ===")

    # Oracle 1: mink analytic, JOINT columns (6:).
    Jm = mink_jacobian(model, qpos, TASK_FRAMES, tgt_quat, tgt_pos)
    Jm = torch.tensor(Jm, dtype=DTYPE)
    joint_err = (J[:, 6:] - Jm[:, 6:]).abs().max().item()
    print(f"  [joints] vs mink analytic : max err {joint_err:.3e}  -> {'PASS' if joint_err < 1e-9 else 'FAIL'}")

    # Oracle 2 + diagnostic: FD h-sweep, BASE columns (:6).
    print("  [base]   FD h-sweep (expect U-curve bottoming ~1e-9..1e-10):")
    best = np.inf
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        Jfd = fd_jacobian(e_of_xi, xi0, h)
        err = (J[:, :6] - Jfd[:, :6]).abs().max().item()
        best = min(best, err)
        print(f"      h={h:.0e}   max|J_auto - J_fd|(base) = {err:.3e}")
    print(f"  [base]   best over sweep  : {best:.3e}  -> {'PASS' if best < 1e-8 else 'FAIL'}")


if __name__ == "__main__":
    main()
