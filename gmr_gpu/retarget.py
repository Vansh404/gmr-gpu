"""Top-level API: SMPL-X clips -> robot motion, batched on GPU.

    from gmr_gpu import retarget_clips
    motions = retarget_clips(["clip1.npz", ...], device="cuda")
    # motions[i] = {"fps", "root_pos", "root_rot" (xyzw), "dof_pos"}

Uses the cold-batched projected-LM solver (stage-2 weights, 40 iters) with each
frame's base initialized at its own pelvis target -- the configuration that
tracks the human measurably better than the CPU/mink production pipeline on
full CMU clips. All frames of all clips are pooled into GPU batches.
"""
import numpy as np
import torch
from general_motion_retargeting.params import ROBOT_XML_DICT

from .kinematics import BatchedRobot
from .solver import solve_batched
from .targets import TargetPipeline, load_frames
from .config import load_ik_tables, weights_from_costs


def retarget_clips(smplx_files, robot="unitree_g1", device=None,
                   batch=8192, max_iter=40, tol=1e-6, tgt_fps=30, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cuda" else torch.float64
    rob = BatchedRobot(str(ROBOT_XML_DICT[robot]), device=device, dtype=dtype)
    frame_names, _, c2 = load_ik_tables(robot)
    fidx = torch.tensor([rob.body_names.index(n) for n in frame_names], device=device)
    W2 = weights_from_costs(c2).to(device=device, dtype=dtype)

    clips = []
    for f in smplx_files:
        frames, fps, height = load_frames(f, tgt_fps=tgt_fps)
        pipe = TargetPipeline(height, robot=robot, dtype=torch.float64)
        pos, quat = pipe.stack_frames(frames)
        tp, tq = pipe(pos, quat)
        clips.append((tq.to(device, dtype), tp.to(device, dtype), fps))
        if verbose:
            print(f"loaded {f}: {tq.shape[0]} frames")

    pairs = [(ci, t) for ci, (tq, tp, _) in enumerate(clips) for t in range(tq.shape[0])]
    trajs = [np.zeros((c[0].shape[0], 36)) for c in clips]
    q0d = None
    import mujoco as mj
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    q0d = torch.tensor(model.qpos0, dtype=dtype, device=device)
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]; n = len(chunk)
        tq = torch.stack([clips[ci][0][t] for ci, t in chunk])
        tp = torch.stack([clips[ci][1][t] for ci, t in chunk])
        # base at each frame's own pelvis target (row 0): basin-safe cold start
        q0 = (tp[:, 0].clone(), tq[:, 0].clone(), q0d[7:].expand(n, -1).clone())
        bq, _ = solve_batched(rob, q0, fidx, tq, tp, W2, max_iter=max_iter, tol=tol)
        qp = torch.cat([bq[0], bq[1][:, [3, 0, 1, 2]], bq[2]], dim=1).double().cpu().numpy()
        for j, (ci, t) in enumerate(chunk):
            trajs[ci][t] = qp[j]
        if verbose:
            print(f"solved {min(s + batch, len(pairs))}/{len(pairs)} frames")

    return [{"fps": float(clips[ci][2]),
             "root_pos": trajs[ci][:, :3],
             "root_rot": trajs[ci][:, 3:7][:, [1, 2, 3, 0]],   # -> xyzw
             "dof_pos": trajs[ci][:, 7:]} for ci in range(len(clips))]
