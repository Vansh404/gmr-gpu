"""
Step 6 of GMR-GPU: real IK-config weights + the two-stage (coarse->fine) solve.

Until now W = I and the task frames were hand-picked. This step wires in the
ACTUAL retargeting problem definition from smplx_to_g1.json:

  - ik_match_table1 / ik_match_table2: robot_frame -> [human_body, pos_w, rot_w,
    pos_offset, rot_offset]. The 14 rows are the tasks; pos_w/rot_w are the
    per-task costs (100 on pelvis/feet position, 10 on orientations, ...).
  - Two stages, solved SEQUENTIALLY on the same configuration: table1 (coarse --
    mostly orientations + root/feet positions) to convergence, then table2
    (fine -- all positions on) WARM-STARTED from stage 1's result. Coarse->fine
    homotopy: stage 2 starts from a good posture instead of a cold guess.

THE WEIGHT CONVENTION (verified in mink/tasks/task.py _weighted_residual):
  mink multiplies the RESIDUAL by cost:  H = (cost*J)^T (cost*J) = J^T diag(cost^2) J.
  Our solver uses H = J^T W J. So to solve the SAME problem as mink/GMR:
      W_row = cost^2   (JSON pos_w=100 -> W=10000 on that row)
  Layout matches our twist: rows [translation(3) then rotation(3)] per task,
  cost[:3]=position_cost, cost[3:]=orientation_cost.

Validation: mink/DAQP two-stage reference (same JSON costs, same targets, run to
convergence) -> compare converged task error + per-frame poses. Plus the homotopy
check: two-stage vs solving stage-2 cold.

Run:  python scratch/two_stage.py   (needs jacobian.py + dls_step.py + gn_loop.py filled)
"""

import json
import numpy as np
import torch
import mujoco as mj
import mink
from mink.lie import SE3

from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel

from fk_parity import make_qpos
from jacobian import (
    DTYPE, DEVICE, ROBOT,
    to_double, use_compiled_geometry, retract,
)
from dls_step import frame_poses
from gn_loop import solve, enorm


# --------------------------------------------------------------------------
# Config loading 
# --------------------------------------------------------------------------
def load_ik_tables():
    """Returns (frame_names, table1_costs, table2_costs) where each costs array
    is (K, 2) = [pos_w, rot_w] per frame, in frame_names order. Rows where both
    are 0 are kept (they contribute zero weight -- same as mink skipping them)."""
    with open(IK_CONFIG_DICT["smplx"][ROBOT]) as f:
        cfg = json.load(f)
    t1, t2 = cfg["ik_match_table1"], cfg["ik_match_table2"]
    assert list(t1.keys()) == list(t2.keys()), "tables must share the frame list"
    frame_names = list(t1.keys())
    c1 = np.array([[t1[k][1], t1[k][2]] for k in frame_names], dtype=np.float64)
    c2 = np.array([[t2[k][1], t2[k][2]] for k in frame_names], dtype=np.float64)
    return frame_names, c1, c2


# --------------------------------------------------------------------------
# THE THING UNDER TEST 
# --------------------------------------------------------------------------
def weights_from_costs(costs):
    """costs: (K,2) [pos_w, rot_w] -> W: (6K,) per-row quadratic weights.
        K = 
    W exists so that our H = J^T W J equals mink's H = J^T diag(cost^2) J.
      - each task contributes 6 rows in OUR twist order: translation first
        (3 rows of pos), then rotation (3 rows of rot)
      - mink squares the cost (see module docstring) -> W rows are cost**2
    """
    K = costs.shape[0]
    W = torch.zeros(6*K)
    W = (costs ** 2).repeat_interleave(3, dim=1).reshape(-1) #this is big brain time
    return W


def two_stage_solve(kin, q0, frame_idx, tgt_quat, tgt_pos, W1, W2, lo, hi):
    """ the coarse->fine homotopy. i miss aghf :(    
      stage 1: solve() with W1 starting from q0        -> q1
      stage 2: solve() with W2 starting from q1 (WARM) -> q2
      Return (q1, q2). (solve returns (best_q, hist); keep just the q's.)
    """
    q1, _ = solve(kin, q0, frame_idx, tgt_quat, tgt_pos, W1, lo, hi, max_iter=200, tol=1e-10)
    q2, _ = solve(kin, q1, frame_idx, tgt_quat, tgt_pos, W2, lo, hi, max_iter=200, tol=1e-10)
    return q1, q2


# --------------------------------------------------------------------------
# mink/GMR two-stage reference (written for you) -- same JSON costs, same
# targets, each stage run far past GMR's 10-iter production cap.
# --------------------------------------------------------------------------
def mink_two_stage(model, qpos_init, frame_names, costs1, costs2, tgt_quat, tgt_pos,
                   iters_per_stage=80):
    config = mink.Configuration(model)
    config.update(q=qpos_init.copy())
    limits = [mink.ConfigurationLimit(model)]
    dt = config.model.opt.timestep

    def make_tasks(costs):
        tasks = []
        for i, name in enumerate(frame_names):
            pw, rw = costs[i]
            if pw == 0 and rw == 0:
                continue
            t = mink.FrameTask(name, "body", position_cost=pw, orientation_cost=rw,
                               lm_damping=1)          # GMR sets lm_damping=1
            tqw = tgt_quat[i].numpy()[[3, 0, 1, 2]]   # xyzw -> wxyz
            t.set_target(SE3(wxyz_xyz=np.concatenate([tqw, tgt_pos[i].numpy()])))
            tasks.append(t)
        return tasks

    for costs in (costs1, costs2):
        tasks = make_tasks(costs)
        for _ in range(iters_per_stage):
            vel = mink.solve_ik(config, tasks, dt, "daqp", 0.5, limits)
            config.integrate_inplace(vel, dt)
    return config.data.qpos.copy()


# --------------------------------------------------------------------------
# Harness.
# --------------------------------------------------------------------------
def qpos_to_q(qpos_np):
    qt = torch.tensor(qpos_np, dtype=DTYPE, device=DEVICE)
    return (qt[0:3], qt[3:7][[1, 2, 3, 0]], qt[7:])


def pose_gap(kin, qa, qb, frame_idx):
    """Max per-frame position gap (m) between two configs -- the honest 'same
    answer?' metric (converged qpos can differ in flat valleys)."""
    _, pa = frame_poses(kin, qa, frame_idx)
    _, pb = frame_poses(kin, qb, frame_idx)
    return max(float((x - y).norm()) for x, y in zip(pa, pb))


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    kin = KinematicsModel(xml, device=DEVICE); to_double(kin); use_compiled_geometry(kin, model)
    rng = np.random.default_rng(0)

    frame_names, c1, c2 = load_ik_tables()
    frame_idx = [kin.get_body_idx(n) for n in frame_names]
    W1 = weights_from_costs(torch.tensor(c1, dtype=DTYPE))
    W2 = weights_from_costs(torch.tensor(c2, dtype=DTYPE))
    if W1 is None:
        print("weights_from_costs returned None -- fill TODO 1."); return
    lo, hi = kin.get_dof_limits(); lo = lo.to(DTYPE); hi = hi.to(DTYPE)

    qpos = make_qpos(model, "random", rng)
    q0 = qpos_to_q(qpos)
    # reachable targets for ALL 14 config frames (real human targets = step 7)
    xi = torch.tensor(rng.uniform(-0.25, 0.25, 6 + q0[2].numel()), dtype=DTYPE)
    q_true = retract(*q0, xi)
    tq, tp = frame_poses(kin, q_true, frame_idx)
    tgt_quat, tgt_pos = torch.stack(tq), torch.stack(tp)

    q1, q2 = two_stage_solve(kin, q0, frame_idx, tgt_quat, tgt_pos, W1, W2, lo, hi)
    if q2 is None:
        print("two_stage_solve returned None -- fill TODO 2."); return

    e1 = enorm(kin, q1, frame_idx, tgt_quat, tgt_pos, W1)
    e2 = enorm(kin, q2, frame_idx, tgt_quat, tgt_pos, W2)
    print("=== ours: two-stage with real smplx_to_g1 weights ===")
    print(f"  stage-1 (coarse) final ||e||_W1 : {e1:.4e}")
    print(f"  stage-2 (fine)   final ||e||_W2 : {e2:.4e}")

    # homotopy check: does the warm start actually help?
    q2_cold, _ = solve(kin, q0, frame_idx, tgt_quat, tgt_pos, W2, lo, hi)
    e2_cold = enorm(kin, q2_cold, frame_idx, tgt_quat, tgt_pos, W2)
    print(f"  stage-2 solved COLD (no stage 1): {e2_cold:.4e}   "
          f"(two-stage {'<=' if e2 <= e2_cold else '>'} cold)")

    # mink reference on the same problem
    qpos_mink = mink_two_stage(model, qpos, frame_names, c1, c2, tgt_quat, tgt_pos)
    q_mink = qpos_to_q(qpos_mink)
    e2_mink = enorm(kin, q_mink, frame_idx, tgt_quat, tgt_pos, W2)
    gap = pose_gap(kin, q2, q_mink, frame_idx)
    print("=== vs mink/DAQP two-stage (same weights/targets, 80 it/stage) ===")
    print(f"  mink final ||e||_W2            : {e2_mink:.4e}")
    print(f"  max per-frame pose gap ours-mink: {gap:.4e} m")


if __name__ == "__main__":
    main()
