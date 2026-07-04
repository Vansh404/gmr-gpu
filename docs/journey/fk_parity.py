"""
Step 1 of GMR-GPU: Forward-Kinematics parity check.

Goal: prove that the repo's torch `KinematicsModel.forward_kinematics` produces
the SAME body world-positions that MuJoCo's C engine does, for the same robot
configuration. Once this passes, we can trust our torch FK inside the batched
DLS solver we're going to build. If it's wrong, every later "how far from mink?"
number is meaningless.

Two independent forward models, one shared input `qpos`:
  - MuJoCo:  set data.qpos -> mj_forward -> read data.xpos  (ground truth)
  - Torch:   split qpos -> KinematicsModel.forward_kinematics -> body_pos

Fill in the 5 TODOs. Then run:
    python scratch/fk_parity.py

Watch this: the 'neutral' pose will likely pass even if TODO 3 (quaternion
convention) is WRONG -- because the identity quaternion is the same in every
convention. The 'random' pose has a non-identity root rotation, so it will only
pass once TODO 3 is correct. That contrast is the lesson.

Success bar: max per-body position error < 1e-5 m. (It won't be tighter than
that because KinematicsModel runs in float32 while MuJoCo is float64.)
"""

import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel

DEVICE = "cpu"      # parity test needs no GPU; cpu keeps it comparable to MuJoCo
DTYPE = torch.float32
ROBOT = "unitree_g1"


# --------------------------------------------------------------------------
# Input synthesis (written for you) -- produces a valid (36,) qpos.
# --------------------------------------------------------------------------
def make_qpos(model, mode, rng):
    """mode='neutral' -> identity root, zero joints.
       mode='random'  -> random root quaternion + random in-limit joint angles.
    Note: only 'random' exercises the root-quaternion convention (TODO 3),
    because an identity quaternion is convention-agnostic."""
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, 0.8]                      # root position
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])       # identity (w,x,y,z)
    if mode == "random":
        q = rng.standard_normal(4)
        quat_wxyz = q / np.linalg.norm(q)            # random unit quaternion
    qpos[3:7] = quat_wxyz
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
            adr = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            if mode == "random":
                qpos[adr] = rng.uniform(lo, hi) if model.jnt_limited[j] else rng.uniform(-0.5, 0.5)
    return qpos


# --------------------------------------------------------------------------
# MuJoCo forward model (ground truth). Returns {body_name: pos(3,)}.
# --------------------------------------------------------------------------
def mujoco_body_positions(model, data, qpos):
    data.qpos[:] = qpos

   


    # use mj_kinematics to fill up xpos from qpos, also populates xquat
    mj.mj_kinematics(model,data)   #doesnt step sim 


    poses = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        poses[name] = data.xpos[i].copy()            # world position of body i
    return poses


# --------------------------------------------------------------------------
# Torch forward model (the thing under test). Returns {body_name: pos(3,)}.
# --------------------------------------------------------------------------
def torch_body_positions(kin, qpos):
    qpos_t = torch.tensor(qpos, dtype=DTYPE, device=DEVICE)

   
    #   root_pos  : first 3   (x, y, z)
    #   root_wxyz : next 4     (MuJoCo stores the quaternion as w, x, y, z)
    #   dof       : the rest   (the 29 hinge angles, radians)

    # the second idx is exlcusive, i think
    root_pos = qpos_t[0:3]   # >>> fix <<<
    root_wxyz = qpos_t[3:7]  # >>> fix <<<
    dof =  qpos_t[7:]      # >>> fix <<<

    

    #KinematicsModel uses x,y,z,w. It flips what mujoco gives when it takes in the xml
    # rot_w = root_wxyz[...,0].clone() #save w to tmp
    # root_rot = root_wxyz # i think pys gonna scream if i dont define root_rot before the shifting
    # root_rot[...,0:3] = root_wxyz[...,1] #shift xyz to front
    # root_rot[...,3] = rot_w
    
    #or in one line
    
    root_rot = root_wxyz[..., [1, 2, 3, 0]]
    print("root_rot:", root_rot) 

    #it will scream if you dont give it batch =1
    body_pos,body_rot = kin.forward_kinematics(root_pos.view(1,3), root_rot.view(1,4), dof.view(1,-1))
    body_pos = body_pos.squeeze(0)  
   
    poses = {}
    for i, name in enumerate(kin.body_names):
        poses[name] = body_pos[i].detach().cpu().numpy()

    # >>> your code here: fill poses[name] = body_pos[i].detach().cpu().numpy() <<<
    return poses


# --------------------------------------------------------------------------
# Comparison harness (written for you).
# --------------------------------------------------------------------------
def compare(mj_poses, torch_poses, label):
    common = [n for n in torch_poses if n in mj_poses]
    errs = {n: float(np.linalg.norm(mj_poses[n] - torch_poses[n])) for n in common}
    max_name = max(errs, key=errs.get)
    print(f"\n=== {label} ===")
    print(f"  mujoco bodies: {len(mj_poses)}  |  torch bodies: {len(torch_poses)}  |  matched: {len(common)}")
    worst = sorted(errs.items(), key=lambda kv: -kv[1])[:5]
    print("  worst 5 (name: error in meters):")
    for n, e in worst:
        print(f"    {n:24s} {e:.3e}")
    print(f"  MAX POSITION ERROR: {errs[max_name]:.3e} m  (bar: < 1e-5)")
    print(f"  -> {'PASS' if errs[max_name] < 1e-5 else 'FAIL'}")


# ==========================================================================
# HARDENING SUITE (v2). The smoke test above checks 2 poses, positions only,
# B=1. Before we build a solver on this FK, harden it three ways:
#   [1] coverage    -- N random poses, not 1 (does it hold across the space?)
#   [2] orientation -- body_rot vs xquat (never checked yet), double-cover aware
#   [3] batch axis  -- B=N in one call must equal B=1 per item (our whole design)
# NOTE: torch_fk_batch below is the batched FK we'll reuse in the real solver --
# same wxyz->xyzw reorder you debugged, just vectorized over a leading N axis.
# ==========================================================================

def torch_fk_batch(kin, qpos_batch):
    """qpos_batch: (N, nq) -> body_pos (N,J,3), body_rot (N,J,4) in xyzw."""
    q = torch.as_tensor(qpos_batch, dtype=DTYPE, device=DEVICE)
    root_pos = q[:, 0:3]
    root_rot = q[:, 3:7][:, [1, 2, 3, 0]]        # wxyz -> xyzw (batched)
    dof = q[:, 7:]
    return kin.forward_kinematics(root_pos, root_rot, dof)


def mujoco_fk_batch(model, data, qpos_batch, mj_ids):
    """Ground-truth loop. Returns pos (N,J,3), quat (N,J,4) wxyz, ordered to
    match kin.body_names via mj_ids."""
    N, J = qpos_batch.shape[0], len(mj_ids)
    pos = np.zeros((N, J, 3))
    quat = np.zeros((N, J, 4))
    for i in range(N):
        data.qpos[:] = qpos_batch[i]
        mj.mj_kinematics(model, data)
        pos[i] = data.xpos[mj_ids]
        quat[i] = data.xquat[mj_ids]             # wxyz
    return pos, quat


def hardening_suite(model, data, kin, seed=1, N=2000):
    rng = np.random.default_rng(seed)
    qpos_batch = np.stack([make_qpos(model, "random", rng) for _ in range(N)])
    mj_ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in kin.body_names])

    mj_pos, mj_quat = mujoco_fk_batch(model, data, qpos_batch, mj_ids)
    body_pos, body_rot = torch_fk_batch(kin, qpos_batch)   # ONE call over all N
    t_pos = body_pos.detach().cpu().numpy()
    t_rot = body_rot.detach().cpu().numpy()                # xyzw

    print(f"\n=== HARDENING SUITE (N={N} random poses) ===")

    # [1] position coverage over every pose x every body
    pos_err = np.linalg.norm(mj_pos - t_pos, axis=-1)      # (N, J)
    fj = np.unravel_index(np.argmax(pos_err), pos_err.shape)[1]
    ok1 = pos_err.max() < 1e-5
    print(f"  [1] position    max err: {pos_err.max():.3e} m   "
          f"(worst body: {kin.body_names[fj]})   -> {'PASS' if ok1 else 'FAIL'}")

    # [2] orientation, double-cover aware: q and -q are the SAME rotation, so
    #     compare via |dot| not ||q1-q2||. Convert torch xyzw -> wxyz to match.
    t_rot_wxyz = t_rot[..., [3, 0, 1, 2]]
    dots = np.clip(np.abs(np.sum(mj_quat * t_rot_wxyz, axis=-1)), 0.0, 1.0)  # (N,J)
    resid = 1.0 - dots                                     # ~0 when aligned
    ang_deg = np.degrees(2.0 * np.arccos(dots)).max()      # human-readable
    ok2 = resid.max() < 1e-5
    print(f"  [2] orientation max resid (1-|dot|): {resid.max():.3e}  "
          f"(~{ang_deg:.2e} deg)   -> {'PASS' if ok2 else 'FAIL'}")

    # [3] batch axis: the B=N result must match B=1 per item, or our whole
    #     batched design is unsound.
    md = 0.0
    for i in range(min(16, N)):
        bp1, _ = torch_fk_batch(kin, qpos_batch[i:i + 1])
        md = max(md, float((body_pos[i] - bp1[0]).abs().max()))
    ok3 = md < 1e-5
    print(f"  [3] batch B=N vs B=1 max diff: {md:.3e}   -> {'PASS' if ok3 else 'FAIL'}")

    print(f"  OVERALL: {'ALL PASS -- FK is trustworthy' if (ok1 and ok2 and ok3) else 'FAIL -- do not build on this yet'}")


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    data = mj.MjData(model)
    kin = KinematicsModel(xml, device=DEVICE)
    rng = np.random.default_rng(0)

    # smoke test: 2 poses, positions only, B=1
    for mode in ["neutral", "random"]:
        qpos = make_qpos(model, mode, rng)
        mj_poses = mujoco_body_positions(model, data, qpos)
        torch_poses = torch_body_positions(kin, qpos)
        compare(mj_poses, torch_poses, mode)

    # hardening: coverage + orientation + batch path
    hardening_suite(model, data, kin)


if __name__ == "__main__":
    main()
