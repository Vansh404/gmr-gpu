"""
GMR-GPU capstone: retarget ALL of CMU three ways and benchmark throughput+quality.

    mink : GMR.retarget() per frame, sequential (production CPU config)
    seq  : ours, warm-started two-stage -- LOCKSTEP-BATCHED ACROSS CLIPS on GPU
           (all clips advance frame t together, each warm from its own t-1;
           the original GMR-GPU design, finally at full scale)
    cold : ours, every frame independent, pooled into giant GPU batches
           (stage-2 weights, 40 iters -- the config that beat mink in 7c)

Phases (all resumable -- each skips work already on disk):

  python scratch/dataset_bench.py cache  [--limit N] [--workers 2]
      Load every npz (SMPL-X FK + SLERP, the expensive shared step) ONCE and
      cache raw 14-joint poses -> ~/molib/retargeted/cache/*.npz  (~350 MB).
      WARNING: each worker transiently needs ~4-5 GB (SMPL-X forward);
      keep --workers 2 on a 15 GB WSL.

  python scratch/dataset_bench.py solve --method {mink,seq,cold} [--limit N]
      Retarget every cached clip -> ~/molib/retargeted/dataset_<method>/*.pkl
      (GMR pkl format) + per-clip metrics json. Timed.

  python scratch/dataset_bench.py report
      Aggregate: frames/sec + task-error distribution per method.

Quality metric is identical for all three: ||e||_W2 computed with OUR rob/W2
from the SAME cached targets (the 7c methodology).
"""

import argparse
import json
import os
import pickle
import time
import glob

import numpy as np
import torch
import mujoco as mj

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from general_motion_retargeting.params import ROBOT_XML_DICT

CMU_ROOT = "/home/templ/molib/CMU"
OUT_ROOT = "/home/templ/molib/retargeted"
CACHE_DIR = os.path.join(OUT_ROOT, "cache")
ROBOT = "unitree_g1"
TGT_FPS = 30

# solver configs, verbatim from 7c
SEQ_CFG = dict(max_iter=10, tol=1e-3)     # per stage, two-stage, warm
COLD_CFG = dict(max_iter=40, tol=1e-6)    # stage-2 only, cold


def clip_id(npz_path):
    return os.path.splitext(os.path.basename(npz_path))[0].replace("_stageii", "")


def all_clips(limit=None):
    files = sorted(glob.glob(f"{CMU_ROOT}/**/*_stageii.npz", recursive=True))
    return files[:limit] if limit else files


# ==========================================================================
# PHASE: cache
# ==========================================================================
def cache_one(npz_path):
    from human_targets import TargetPipeline, load_frames
    cid = clip_id(npz_path)
    out = os.path.join(CACHE_DIR, cid + ".npz")
    if os.path.exists(out):
        return cid, "skip", 0.0
    t0 = time.perf_counter()
    try:
        frames, fps, height = load_frames(npz_path, tgt_fps=TGT_FPS)
        if len(frames) < 2:
            return cid, "too-short", time.perf_counter() - t0
        pipe = TargetPipeline(height)                      # only for name list
        pos, quat = pipe.stack_frames(frames)              # RAW human (T,14,·)
        np.savez_compressed(out, pos=pos.numpy().astype(np.float32),
                            quat=quat.numpy().astype(np.float32),
                            height=height, fps=fps)
        return cid, "ok", time.perf_counter() - t0
    except Exception as ex:
        return cid, f"FAIL {type(ex).__name__}: {str(ex)[:60]}", time.perf_counter() - t0


def phase_cache(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = all_clips(args.limit)
    print(f"caching {len(files)} clips with {args.workers} workers -> {CACHE_DIR}")
    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(cache_one, files):
                results.append(r)
                if len(results) % 25 == 0:
                    print(f"  {len(results)}/{len(files)}  ({time.perf_counter()-t0:.0f}s)")
    else:
        for f in files:
            results.append(cache_one(f))
            if len(results) % 25 == 0:
                print(f"  {len(results)}/{len(files)}  ({time.perf_counter()-t0:.0f}s)")
    ok = sum(1 for _, s, _ in results if s == "ok")
    skip = sum(1 for _, s, _ in results if s == "skip")
    bad = [(c, s) for c, s, _ in results if s not in ("ok", "skip")]
    print(f"done in {time.perf_counter()-t0:.0f}s: {ok} cached, {skip} already, {len(bad)} failed")
    for c, s in bad:
        print(f"  FAILED {c}: {s}")


# ==========================================================================
# shared solve-phase setup
# ==========================================================================
def load_cache(limit=None):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if limit:
        files = files[:limit]
    clips = []
    for f in files:
        d = np.load(f)
        clips.append(dict(cid=os.path.splitext(os.path.basename(f))[0],
                          pos=d["pos"], quat=d["quat"],
                          height=float(d["height"]), fps=float(d["fps"])))
    return clips


def save_traj_pkl(path, fps, traj):
    with open(path, "wb") as f:
        pickle.dump({"fps": fps, "root_pos": traj[:, :3],
                     "root_rot": traj[:, 3:7][:, [1, 2, 3, 0]],
                     "dof_pos": traj[:, 7:],
                     "local_body_pos": None, "link_body_list": None}, f)


def our_setup(device, dtype):
    from batched_kin import BatchedRobot
    from two_stage import load_ik_tables, weights_from_costs
    from human_targets import TargetPipeline
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    rob = BatchedRobot(xml, device=device, dtype=dtype)
    pipe_names = TargetPipeline(1.8).robot_frames          # height-independent
    fidx = torch.tensor([rob.body_names.index(n) for n in pipe_names], device=device)
    _, c1, c2 = load_ik_tables()
    W1 = weights_from_costs(torch.tensor(c1, dtype=torch.float64)).to(device=device, dtype=dtype)
    W2 = weights_from_costs(torch.tensor(c2, dtype=torch.float64)).to(device=device, dtype=dtype)
    return model, rob, fidx, W1, W2


def clip_targets(clip, device, dtype):
    """cache (raw human) -> solver targets, via the user's TargetPipeline."""
    from human_targets import TargetPipeline
    pipe = TargetPipeline(clip["height"], dtype=torch.float64)
    tp, tq = pipe(torch.tensor(clip["pos"], dtype=torch.float64),
                  torch.tensor(clip["quat"], dtype=torch.float64))
    return tq.to(device=device, dtype=dtype), tp.to(device=device, dtype=dtype)


def eval_quality(rob, fidx, W2, traj, tgt_quat, tgt_pos, device, dtype):
    from batched_lm import enorm_b
    qt = torch.tensor(traj, dtype=dtype, device=device)
    q = (qt[:, 0:3], qt[:, 3:7][:, [1, 2, 3, 0]], qt[:, 7:])
    return enorm_b(rob, q, fidx, tgt_quat, tgt_pos, W2).cpu().numpy()


# ==========================================================================
# PHASE: solve --method mink
# ==========================================================================
def solve_mink(args):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    device, dtype = "cpu", torch.float64
    model, rob, fidx, W1, W2 = our_setup(device, dtype)
    out_dir = os.path.join(OUT_ROOT, "dataset_mink"); os.makedirs(out_dir, exist_ok=True)
    clips = load_cache(args.limit)
    metrics, t_solve, n_frames = {}, 0.0, 0
    human_names = None
    for i, clip in enumerate(clips):
        pkl = os.path.join(out_dir, clip["cid"] + ".pkl")
        mfile = pkl + ".json"
        if os.path.exists(mfile):
            continue
        # rebuild minimal frame dicts from cache (only the 14 bodies GMR touches)
        if human_names is None:
            from human_targets import TargetPipeline
            human_names = TargetPipeline(1.8).human_names
        T = clip["pos"].shape[0]
        t0 = time.perf_counter()
        gmr = GMR(src_human="smplx", tgt_robot=ROBOT,
                  actual_human_height=clip["height"], verbose=False)
        traj = np.zeros((T, 36))
        for t in range(T):
            fr = {n: (clip["pos"][t, k].astype(np.float64),
                      clip["quat"][t, k][[3, 0, 1, 2]].astype(np.float64))  # xyzw->wxyz
                  for k, n in enumerate(human_names)}
            traj[t] = gmr.retarget(fr)
        dt = time.perf_counter() - t0
        t_solve += dt; n_frames += T
        tq, tp = clip_targets(clip, device, dtype)
        e = eval_quality(rob, fidx, W2, traj, tq, tp, device, dtype)
        save_traj_pkl(pkl, TGT_FPS, traj)
        json.dump({"frames": T, "solve_s": dt, "err_mean": float(e.mean()),
                   "err_p95": float(np.percentile(e, 95))}, open(mfile, "w"))
        if (i + 1) % 10 == 0:
            print(f"  mink {i+1}/{len(clips)}  {n_frames/max(t_solve,1e-9):.0f} fps so far")
    print(f"mink: {n_frames} frames in {t_solve:.0f}s = {n_frames/max(t_solve,1e-9):.0f} fps (solve only)")


# ==========================================================================
# PHASE: solve --method cold   (pool ALL frames -> giant GPU batches)
# ==========================================================================
def solve_cold(args):
    from batched_lm import solve_batched
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cuda" else torch.float64
    model, rob, fidx, W1, W2 = our_setup(device, dtype)
    out_dir = os.path.join(OUT_ROOT, "dataset_cold"); os.makedirs(out_dir, exist_ok=True)
    clips = [c for c in load_cache(args.limit)
             if not os.path.exists(os.path.join(out_dir, c["cid"] + ".pkl.json"))]
    if not clips:
        print("cold: nothing to do"); return
    print(f"cold: preparing targets for {len(clips)} clips...")
    tgt = [clip_targets(c, device, dtype) for c in clips]
    q0d = torch.tensor(model.qpos0, dtype=dtype, device=device)

    B = args.batch
    # flat index of (clip, frame) pairs
    pairs = [(ci, t) for ci, c in enumerate(clips) for t in range(c["pos"].shape[0])]
    trajs = [np.zeros((c["pos"].shape[0], 36)) for c in clips]
    n_frames = len(pairs); t_solve = 0.0
    print(f"cold: {n_frames} frames in chunks of {B} on {device}")
    for s in range(0, n_frames, B):
        chunk = pairs[s:s + B]; n = len(chunk)
        tq = torch.stack([tgt[ci][0][t] for ci, t in chunk])
        tp = torch.stack([tgt[ci][1][t] for ci, t in chunk])
        # init base at each frame's own PELVIS TARGET (row 0), joints at default.
        # Cold-starting every frame from the standing pose catastrophically fails
        # on extreme poses (01_01 frames 240+: 55 frames err>15, max 113); the
        # pelvis target already says where the base belongs -> basin fixed, tail
        # eliminated (6.19 mean / 7.8 max on the same clip). Zero extra cost.
        q0 = (tp[:, 0].clone(), tq[:, 0].clone(), q0d[7:].expand(n, -1).clone())
        t0 = time.perf_counter()
        bq, _ = solve_batched(rob, q0, fidx, tq, tp, W2, **COLD_CFG)
        if device == "cuda":
            torch.cuda.synchronize()
        t_solve += time.perf_counter() - t0
        qp = torch.cat([bq[0], bq[1][:, [3, 0, 1, 2]], bq[2]], dim=1).double().cpu().numpy()
        for j, (ci, t) in enumerate(chunk):
            trajs[ci][t] = qp[j]
        done = min(s + B, n_frames)
        print(f"  cold {done}/{n_frames}  {done/max(t_solve,1e-9):.0f} fps")
    for ci, c in enumerate(clips):
        pkl = os.path.join(out_dir, c["cid"] + ".pkl")
        e = eval_quality(rob, fidx, W2, trajs[ci], tgt[ci][0], tgt[ci][1], device, dtype)
        save_traj_pkl(pkl, TGT_FPS, trajs[ci])
        json.dump({"frames": int(c["pos"].shape[0]), "solve_s": None,
                   "err_mean": float(e.mean()), "err_p95": float(np.percentile(e, 95))},
                  open(pkl + ".json", "w"))
    json.dump({"frames": n_frames, "solve_s": t_solve},
              open(os.path.join(out_dir, "_timing.json"), "w"))
    print(f"cold: {n_frames} frames in {t_solve:.0f}s = {n_frames/max(t_solve,1e-9):.0f} fps (solve only)")


# ==========================================================================
# PHASE: solve --method seq  (lockstep across clips, warm within each clip)
# ==========================================================================
def solve_seq(args):
    from batched_lm import solve_batched
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cuda" else torch.float64
    model, rob, fidx, W1, W2 = our_setup(device, dtype)
    out_dir = os.path.join(OUT_ROOT, "dataset_seq"); os.makedirs(out_dir, exist_ok=True)
    clips = [c for c in load_cache(args.limit)
             if not os.path.exists(os.path.join(out_dir, c["cid"] + ".pkl.json"))]
    if not clips:
        print("seq: nothing to do"); return
    if args.max_frames:                                         # sanity-run truncation
        for c in clips:
            c["pos"] = c["pos"][:args.max_frames]; c["quat"] = c["quat"][:args.max_frames]
    clips.sort(key=lambda c: c["pos"].shape[0], reverse=True)   # cohort by length
    q0d = torch.tensor(model.qpos0, dtype=dtype, device=device)
    C = args.cohort
    n_frames = sum(c["pos"].shape[0] for c in clips); t_solve = 0.0
    print(f"seq: {len(clips)} clips, {n_frames} frames, lockstep cohorts of {C} on {device}")
    done_frames = 0
    for s in range(0, len(clips), C):
        cohort = clips[s:s + C]; n = len(cohort)
        lens = [c["pos"].shape[0] for c in cohort]; Tm = max(lens)
        tgt = [clip_targets(c, device, dtype) for c in cohort]
        trajs = [np.zeros((L, 36)) for L in lens]
        q = (q0d[0:3].expand(n, 3).clone(),
             q0d[3:7][[1, 2, 3, 0]].expand(n, 4).clone(),
             q0d[7:].expand(n, -1).clone())
        t0 = time.perf_counter()
        for t in range(Tm):
            idx = [min(t, L - 1) for L in lens]                # clamp finished clips
            tq = torch.stack([tgt[j][0][idx[j]] for j in range(n)])
            tp = torch.stack([tgt[j][1][idx[j]] for j in range(n)])
            q1, _ = solve_batched(rob, q, fidx, tq, tp, W1, **SEQ_CFG)
            q, _ = solve_batched(rob, q1, fidx, tq, tp, W2, **SEQ_CFG)
            qp = torch.cat([q[0], q[1][:, [3, 0, 1, 2]], q[2]], dim=1).double().cpu().numpy()
            for j in range(n):
                if t < lens[j]:
                    trajs[j][t] = qp[j]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0; t_solve += dt
        done_frames += sum(lens)
        print(f"  seq cohort {s//C + 1}: {n} clips, Tmax {Tm}, {dt:.0f}s   "
              f"({done_frames/max(t_solve,1e-9):.0f} fps so far)")
        for j, c in enumerate(cohort):
            pkl = os.path.join(out_dir, c["cid"] + ".pkl")
            e = eval_quality(rob, fidx, W2, trajs[j], tgt[j][0], tgt[j][1], device, dtype)
            save_traj_pkl(pkl, TGT_FPS, trajs[j])
            json.dump({"frames": lens[j], "solve_s": None, "err_mean": float(e.mean()),
                       "err_p95": float(np.percentile(e, 95))}, open(pkl + ".json", "w"))
    json.dump({"frames": n_frames, "solve_s": t_solve},
              open(os.path.join(out_dir, "_timing.json"), "w"))
    print(f"seq: {n_frames} frames in {t_solve:.0f}s = {n_frames/max(t_solve,1e-9):.0f} fps (solve only)")


# ==========================================================================
# PHASE: score  (post-hoc quality for ANY dir of GMR-format pkls, e.g. the
# repo production script's output. --align undoes the constant translation
# that smplx_to_robot_dataset.py applies (HEIGHT_ADJUST z + ROOT_ORIGIN_OFFSET
# xy) via best-fit mean shift of root vs pelvis target -- note this also
# absorbs any constant pelvis bias, slightly flattering the scored method.)
# ==========================================================================
def phase_score(args):
    device, dtype = "cpu", torch.float64
    model, rob, fidx, W1, W2 = our_setup(device, dtype)
    d = os.path.join(OUT_ROOT, args.dir)
    cache = {c["cid"]: c for c in load_cache(None)}
    scored = 0
    for pkl in sorted(glob.glob(os.path.join(d, "**", "*.pkl"), recursive=True)):
        cid = os.path.splitext(os.path.basename(pkl))[0].replace("_stageii", "")
        if cid not in cache:
            continue
        m = pickle.load(open(pkl, "rb"))
        traj = np.concatenate([m["root_pos"], m["root_rot"][:, [3, 0, 1, 2]],  # xyzw->wxyz
                               m["dof_pos"]], axis=1)
        tq, tp = clip_targets(cache[cid], device, dtype)
        T = min(traj.shape[0], tq.shape[0])
        traj, tq, tp = traj[:T], tq[:T], tp[:T]
        if args.align:
            # undo the script's constant translation by the WEIGHTED-LS optimal
            # shift (all position rows, their pos-weights): the best possible
            # constant placement. (Mean-pelvis align is WRONG: it re-centers on
            # the solver's mean pelvis residual and corrupts every 1e4 row.)
            qt = torch.tensor(traj, dtype=dtype)
            bp, _ = rob.fk(qt[:, 0:3], qt[:, 3:7][:, [1, 2, 3, 0]], qt[:, 7:])
            cur = bp.index_select(-2, fidx)                     # (T,K,3)
            wk = W2.view(-1, 6)[:, 0]                           # per-task pos weight
            delta = ((tp - cur) * wk[None, :, None]).sum((0, 1)) / (wk.sum() * T)
            traj = traj.copy(); traj[:, :3] += delta.numpy()
        e = eval_quality(rob, fidx, W2, traj, tq, tp, device, dtype)
        json.dump({"frames": T, "solve_s": None, "err_mean": float(e.mean()),
                   "err_p95": float(np.percentile(e, 95))}, open(pkl + ".json", "w"))
        scored += 1
    if args.solve_s:
        json.dump({"frames": sum(json.load(open(f))["frames"] for f in glob.glob(os.path.join(d, "**", "*.pkl.json"), recursive=True)),
                   "solve_s": args.solve_s}, open(os.path.join(d, "_timing.json"), "w"))
    print(f"scored {scored} clips in {d} (clips without cache entries skipped)")


# ==========================================================================
# PHASE: report
# ==========================================================================
def phase_report(args):
    print(f"{'method':8s} {'clips':>6s} {'frames':>8s} {'solve':>8s} {'fps':>8s} "
          f"{'err mean':>9s} {'err p95':>8s}")
    for method in ("mink", "seq", "cold", "repo"):
        d = os.path.join(OUT_ROOT, f"dataset_{method}")
        files = glob.glob(os.path.join(d, "**", "*.pkl.json"), recursive=True)
        if not files:
            print(f"{method:8s}  (no results)"); continue
        ms = [json.load(open(f)) for f in files]
        frames = sum(m["frames"] for m in ms)
        tj = os.path.join(d, "_timing.json")
        if os.path.exists(tj):
            solve_s = json.load(open(tj))["solve_s"]
        else:
            solve_s = sum(m["solve_s"] for m in ms if m["solve_s"])
        errs = np.array([m["err_mean"] for m in ms])
        p95s = np.array([m["err_p95"] for m in ms])
        fps = f"{frames/solve_s:8.0f}" if solve_s and solve_s > 0 else "     n/a"
        ss = f"{solve_s:7.0f}s" if solve_s and solve_s > 0 else "    n/a"
        print(f"{method:8s} {len(ms):6d} {frames:8d} {ss} "
              f"{fps} {errs.mean():9.3f} {p95s.mean():8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cache", "solve", "report", "score"])
    ap.add_argument("--method", choices=["mink", "seq", "cold"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--cohort", type=int, default=512)
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    ap.add_argument("--dir", default="dataset_repo")
    ap.add_argument("--solve-s", dest="solve_s", type=float, default=None)
    ap.add_argument("--align", action="store_true", default=True)
    args = ap.parse_args()
    if args.phase == "score":
        phase_score(args)
        return
    if args.phase == "cache":
        phase_cache(args)
    elif args.phase == "solve":
        {"mink": solve_mink, "seq": solve_seq, "cold": solve_cold}[args.method](args)
    else:
        phase_report(args)


if __name__ == "__main__":
    main()
