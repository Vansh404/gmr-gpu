"""
Step 7c of GMR-GPU: full-clip retargeting parity vs mink/GMR on real CMU data.

Pipeline (all pieces you built):
  clip -> SMPL-X FK -> TargetPipeline (7b) -> per-frame TWO-STAGE (6) warm-started
  solve_batched (7a) -> robot qpos trajectory   ...vs...   GMR.retarget() (mink/DAQP)

Three runs on the same clip:
  OURS-SEQ : sequential, warm-started from frame t-1, two-stage -- mink's
             structure, the parity configuration. B=1 per frame.
  OURS-COLD: every frame cold from the default pose, ALL frames in one batch --
             our GPU-native shape. Measures how much warm-start structure
             matters on real data (informs 7d batching design).
  MINK     : GMR.retarget() per frame, production budget (10 it/stage).

Metrics (distributions over all frames -- flat-valley lesson from step 6: task
error and FRAME POSES are the honest metrics; raw qpos can differ at saturated
joints without being wrong):
  - task error ||e||_W2 per frame, ours vs mink (who tracks the human better?)
  - task-frame position gap ours-vs-mink (mm): mean / p95 / max
  - dof|root diffs (informational), feasibility counts, per-frame wall time

Artifacts: saves both trajectories as GMR-format .pkl (playable with
scripts/vis_robot_motion.py) and, if a renderer is available, a side-by-side
mp4 (ours | mink).
the harness config might lowkenuinely have a tol bug
Run:  python scratch/clip_parity.py [--frames N] [--render]
"""

import argparse
import pickle
import time

import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT

from batched_kin import BatchedRobot
from batched_lm import solve_batched, enorm_b
from two_stage import load_ik_tables, weights_from_costs
from human_targets import TargetPipeline, load_frames, DTYPE, ROBOT


# --------------------------------------------------------------------------
# Provided helpers.
# --------------------------------------------------------------------------
def default_q(model, B=1):
    """The robot's default configuration (model.qpos0) as our q tuple --
    the same start mink.Configuration uses, for a fair warm-start chain."""
    q0 = torch.tensor(model.qpos0, dtype=DTYPE)
    return (q0[0:3].expand(B, 3).clone(),
            q0[3:7][[1, 2, 3, 0]].expand(B, 4).clone(),
            q0[7:].expand(B, -1).clone())


def qpos_from_q(q):
    """(pos,quat_xyzw,dof) tuple (B=1) -> mujoco qpos (nq,) wxyz."""
    pos, quat, dof = q[0][0], q[1][0], q[2][0]
    return torch.cat([pos, quat[[3, 0, 1, 2]], dof]).numpy()


def save_pkl(path, fps, qpos_traj):
    """GMR-format motion pkl (root_rot saved xyzw, like smplx_to_robot.py)."""
    qp = np.asarray(qpos_traj)
    with open(path, "wb") as f:
        pickle.dump({"fps": fps,
                     "root_pos": qp[:, :3],
                     "root_rot": qp[:, 3:7][:, [1, 2, 3, 0]],
                     "dof_pos": qp[:, 7:],
                     "local_body_pos": None, "link_body_list": None}, f)
    print(f"  saved {path}")


def run_mink(frames, height, T):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    gmr = GMR(src_human="smplx", tgt_robot=ROBOT, actual_human_height=height, verbose=False)
    out = np.zeros((T, 36))
    t0 = time.perf_counter()
    for t in range(T):
        out[t] = gmr.retarget(frames[t])
    dt = (time.perf_counter() - t0) / T * 1e3
    return out, dt


# --------------------------------------------------------------------------
# THE THING UNDER TEST, 
# --------------------------------------------------------------------------
def retarget_clip_sequential(rob, fidx, tgt_quat, tgt_pos, W1, W2, model,
                             max_iter=10, tol=1e-3):
    """TODO 1: the capstone loop -- mink's production structure with our solver.
    For each frame t (targets tgt_quat[t], tgt_pos[t], sliced as (1,K,...)):
      stage 1: solve_batched(..., W1, ...) warm-started from previous frame's q
      stage 2: solve_batched(..., W2, ...) warm-started from stage 1's result
      record qpos_from_q(q) into the trajectory; q carries over to frame t+1.
    Start q = default_q(model). Budgets mirror GMR: max_iter=10/stage, tol=1e-3.
    Returns (T,36) numpy qpos trajectory."""
    T = tgt_quat.shape[0]
    traj = np.zeros((T, 36))
    q = default_q(model)
    # <<< fill: ~6 lines >>>

    for t in range(T):
        q1,_ = solve_batched(rob, q, fidx, tgt_quat[t:t+1], tgt_pos[t:t+1], W1, max_iter=max_iter, tol=tol)
        q, _ = solve_batched(rob, q1, fidx, tgt_quat[t:t+1], tgt_pos[t:t+1], W2, max_iter=max_iter, tol=tol)
        traj[t] = qpos_from_q(q)
    return traj


def task_pose_gap(rob, fidx, qpos_a, qpos_b):
    """ geometric gap between two qpos trajectories (T,36):
    batched FK over ALL frames at once, gather the K
    task frames, return per-frame MAX position gap (T,) in meters.
    
    """
    
    def split(qpos):
        qpos = torch.tensor(qpos, dtype=DTYPE)

        root_pos = qpos[:, 0:3]
        root_quat = qpos[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
        joint_pos = qpos[:, 7:]
        return root_pos, root_quat, joint_pos
    
    pos_a, _ = rob.fk(*split(qpos_a))#unpack 
    pos_b, _ = rob.fk(*split(qpos_b))

    feet_a = pos_a.index_select(dim=-2, index=fidx)
    feet_b = pos_b.index_select(dim=-2, index=fidx)

    per_foot_gap = torch.norm(feet_a - feet_b, dim=-1)
    max_gap_per_frame = per_foot_gap.max(dim=-1).values
    return max_gap_per_frame


# --------------------------------------------------------------------------
# Harness.
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None, help="limit #frames")
    ap.add_argument("--render", action="store_true", help="write side-by-side mp4")
    args = ap.parse_args()

    frames, fps, height = load_frames()

    # frames[t] = {
    # 'pelvis':      (pos(3,), quat(4,)),     # world position [m], orientation wxyz
    # 'left_knee':   (pos(3,), quat(4,)),
    # ...                                      # all 55 SMPL-X joints:
    # } 
    # Each entry is the SE(3) pose of one human joint at one instant 
    
    T = min(args.frames or len(frames), len(frames))
    print(f"clip: {T}/{len(frames)} frames @ {fps:.0f} fps, height {height:.3f} m")


    #pos,quat: where the human joints are
    #tgt_pos, tgt_quat: where the robot links should go
    pipe = TargetPipeline(height)
    pos, quat = pipe.stack_frames(frames[:T]) #stack_frames drops stuff like fingers eyes jaws etc etc, stuff the G1 cant map to, so 14 joints
   
    tgt_pos, tgt_quat = pipe(pos, quat) #human > robot ccoordinate tranformation, scake + offset

    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    rob = BatchedRobot(xml, dtype=DTYPE)
    fidx = torch.tensor([rob.body_names.index(n) for n in pipe.robot_frames]) #these are the 14 frames the FK maps out to
    _, c1, c2 = load_ik_tables()
    W1 = weights_from_costs(torch.tensor(c1, dtype=DTYPE))
    W2 = weights_from_costs(torch.tensor(c2, dtype=DTYPE))

    # ---- ours, sequential warm-started (parity config) ----
    t0 = time.perf_counter()
    ours_seq = retarget_clip_sequential(rob, fidx, tgt_quat, tgt_pos, W1, W2, model)
    ms_seq = (time.perf_counter() - t0) / T * 1e3
    if not np.any(ours_seq):
        print("retarget_clip_sequential returned zeros ."); return

    # ---- ours, cold-batched (GPU-native shape) ----
    t0 = time.perf_counter()
    qb = default_q(model, B=T)
    
    bq, _ = solve_batched(rob, qb, fidx, tgt_quat, tgt_pos, W2, max_iter=10, tol=1e-6)
    ms_cold = (time.perf_counter() - t0) / T * 1e3
    ours_cold = np.concatenate([bq[0].numpy(), bq[1].numpy()[:, [3, 0, 1, 2]], bq[2].numpy()], axis=1)

    # ---- mink reference ----
    mink_traj, ms_mink = run_mink(frames, height, T)

    # ---- metrics ----
    def q_of(traj):
        qt = torch.tensor(traj, dtype=DTYPE)
        return (qt[:, 0:3], qt[:, 3:7][:, [1, 2, 3, 0]], qt[:, 7:])
    e_seq = enorm_b(rob, q_of(ours_seq), fidx, tgt_quat, tgt_pos, W2)
    e_cold = enorm_b(rob, q_of(ours_cold), fidx, tgt_quat, tgt_pos, W2)
    e_mink = enorm_b(rob, q_of(mink_traj), fidx, tgt_quat, tgt_pos, W2)

    gap_seq = task_pose_gap(rob, fidx, ours_seq, mink_traj)
    if gap_seq is None:
        print("task_pose_gap returned None -- fill TODO 2."); return
    gap_cold = task_pose_gap(rob, fidx, ours_cold, mink_traj)
    gap_modes = task_pose_gap(rob, fidx, ours_seq, ours_cold)

    def stats(x):
        x = np.asarray(x)
        return f"mean {x.mean():.4g}  p95 {np.percentile(x, 95):.4g}  max {x.max():.4g}"

    print("\n=== task error ||e||_W2 per frame (lower = tracks human better) ===")
    print(f"  ours seq  : {stats(e_seq.numpy())}")
    print(f"  ours cold : {stats(e_cold.numpy())}")
    print(f"  mink      : {stats(e_mink.numpy())}")
    print("=== task-frame position gap vs mink (meters) ===")
    print(f"  seq  vs mink : {stats(gap_seq.numpy())}")
    print(f"  cold vs mink : {stats(gap_cold.numpy())}")
    print(f"  seq  vs cold : {stats(gap_modes.numpy())}   <- warm-start structure effect")
    print("=== per-frame wall time (CPU f64; batching+GPU is 7d) ===")
    print(f"  ours seq {ms_seq:.1f} ms | ours cold-batched {ms_cold:.2f} ms | mink {ms_mink:.1f} ms")

    save_pkl("/home/templ/molib/retargeted/parity_ours.pkl", fps, ours_seq)
    save_pkl("/home/templ/molib/retargeted/parity_mink.pkl", fps, mink_traj)
    print("  view either:  python scripts/vis_robot_motion.py --robot unitree_g1 --robot_motion_path <pkl>")

    if args.render:
        try:
            import imageio
            ren = mj.Renderer(model, 480, 480)
            data = mj.MjData(model)
            cam = mj.MjvCamera()                      # follow-cam: clip walks out of a static view
            cam.azimuth, cam.elevation, cam.distance = 135, -15, 2.5
            vid = []
            for t in range(0, T, 2):
                panes = []
                for traj in (ours_cold, ours_seq, mink_traj):
                    data.qpos[:] = traj[t]; mj.mj_kinematics(model, data)
                    cam.lookat[:] = traj[t][:3]       # track the pelvis
                    ren.update_scene(data, camera=cam)
                    panes.append(ren.render())
                vid.append(np.concatenate(panes, axis=1))
            out = "/home/templ/molib/retargeted/parity_side_by_side.mp4"
            imageio.mimsave(out, vid, fps=fps / 2)
            print(f"  side-by-side video (ours-cold | ours-seq | mink): {out}")
        except Exception as ex:
            print(f"  [render skipped: {type(ex).__name__}: {ex} -- use vis_robot_motion.py instead]")


if __name__ == "__main__":
    main()
