"""
Step 7d of GMR-GPU: precision, device, and compile 

Questions this answers, in order:
  [1] f32 SAFETY: solve the real 240-frame problem in f32 and f64; if the task
      errors match to ~1e-3 and qpos to ~1e-4, f32 is safe (it's mandatory on
      GeForce -- f64 runs at 1/43 rate).
  [2] THROUGHPUT SWEEP: ms/item/iteration of the LM hot loop across
      {cpu-f64, cpu-f32, cuda-f32, cuda-f32+torch.compile} x B in {256, 2048, 8192}.
      Fixed 8 iterations, no early exit -- clean numbers.
  [3] FWD vs REV on GPU: re-benchmark jacfwd vs jacrev in the batched GPU regime
      (the step-3 CPU verdict may flip here).
  [4] VERDICT: frames/sec vs mink (2.3 ms/frame = ~435 fps/core; a 32-core box
      ~10-15k fps with multiprocessing). Honest comparison, both scales.

All measurement plumbing -- no TODOs; the concepts are behind us.

Run:  python scratch/bench_7d.py          (needs a few minutes; compile warmup is slow)
"""

import time
import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT

from batched_kin import BatchedRobot
from batched_lm import solve_batched, enorm_b, retract_b, batched_jacobian, LAM0
from two_stage import load_ik_tables, weights_from_costs
from human_targets import TargetPipeline, load_frames

ROBOT = "unitree_g1"


def real_targets(dtype, device, T=240):
    frames, fps, height = load_frames()
    pipe = TargetPipeline(height, dtype=torch.float64)          # build in f64, cast after
    pos, quat = pipe.stack_frames(frames[:T])
    tgt_pos, tgt_quat = pipe(pos, quat)
    _, c1, c2 = load_ik_tables()
    W2 = weights_from_costs(torch.tensor(c2, dtype=torch.float64))
    to = lambda x: x.to(dtype=dtype, device=device)
    return to(tgt_quat), to(tgt_pos), to(W2), pipe.robot_frames


def default_q(model, B, dtype, device):
    q0 = torch.tensor(model.qpos0, dtype=dtype, device=device)
    return (q0[0:3].expand(B, 3).clone(), q0[3:7][[1, 2, 3, 0]].expand(B, 4).clone(),
            q0[7:].expand(B, -1).clone())


def make_step(rob, fidx, tgt_quat, tgt_pos, W):
    """One LM iteration as a pure function of (q, err, lam) -> same. No early
    exit, no python branches on data -- compile/graph friendly."""
    nv = 6 + rob.num_dof
    I = torch.eye(nv, dtype=W.dtype, device=W.device)

    def step(pos, quat, dof, err, lam):
        q = (pos, quat, dof)
        J = batched_jacobian(rob, q, fidx, tgt_quat, tgt_pos)
        E = rob.task_error(pos, quat, dof, fidx, tgt_quat, tgt_pos)
        WJ = W[..., None] * J
        H = J.transpose(-1, -2) @ WJ + lam[:, None, None] * I
        c = WJ.transpose(-1, -2) @ E[..., None]
        dxi = torch.cholesky_solve(-c, torch.linalg.cholesky(H)).squeeze(-1)
        qc = retract_b(pos, quat, dof, dxi)
        qc = (qc[0], qc[1], torch.clamp(qc[2], rob.lo, rob.hi))
        ec = enorm_b(rob, qc, fidx, tgt_quat, tgt_pos, W)
        accept = ec < err
        m = accept[:, None]
        pos = torch.where(m, qc[0], pos); quat = torch.where(m, qc[1], quat)
        dof = torch.where(m, qc[2], dof)
        err = torch.where(accept, ec, err)
        lam = torch.where(accept, lam * 0.5, lam * 4.0).clamp(1e-6, 1e6)
        return pos, quat, dof, err, lam

    return step


def bench_config(label, device, dtype, B, T, iters=8, compiled=False):
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    rob = BatchedRobot(xml, device=device, dtype=dtype)
    tq, tp, W, robot_frames = real_targets(dtype, device, T)
    fidx = torch.tensor([rob.body_names.index(n) for n in robot_frames], device=device)
    reps = (B + T - 1) // T
    tq = tq.repeat(reps, 1, 1)[:B]; tp = tp.repeat(reps, 1, 1)[:B]   # tile real frames to B

    q = default_q(model, B, dtype, device)
    err = enorm_b(rob, q, fidx, tq, tp, W)
    lam = torch.full((B,), LAM0, dtype=dtype, device=device)
    step = make_step(rob, fidx, tq, tp, W)
    if compiled:
        try:
            step = torch.compile(step)
        except Exception as ex:
            return f"{label:22s}  compile FAILED: {type(ex).__name__}: {str(ex)[:60]}"

    state = (*[x.clone() for x in q], err.clone(), lam.clone())
    try:
        state = step(*state)                                    # warmup (JIT/compile/TS)
        state = step(*state)
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            state = step(*state)
        if device == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        per_item_us = dt / B * 1e6
        return (f"{label:22s}  {dt*1e3:8.1f} ms/iter   {per_item_us:8.2f} us/item/iter   "
                f"~{1.0/(per_item_us*1e-6*10):>9.0f} fps@10it")
    except Exception as ex:
        return f"{label:22s}  FAILED: {type(ex).__name__}: {str(ex)[:80]}"


def main():
    print(f"torch {torch.__version__}  cuda: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # ---------- [1] f32 safety on the real problem ----------
    xml = str(ROBOT_XML_DICT[ROBOT]); model = mj.MjModel.from_xml_path(xml)
    results = {}
    for dtype in (torch.float64, torch.float32):
        rob = BatchedRobot(xml, dtype=dtype)
        tq, tp, W, rf = real_targets(dtype, "cpu", 240)
        fidx = torch.tensor([rob.body_names.index(n) for n in rf])
        q0 = default_q(model, 240, dtype, "cpu")
        bq, berr = solve_batched(rob, q0, fidx, tq, tp, W, max_iter=40, tol=1e-6)
        results[dtype] = (bq, berr)
    e64, e32 = results[torch.float64][1], results[torch.float32][1].double()
    d64, d32 = results[torch.float64][0][2], results[torch.float32][0][2].double()
    print("\n[1] f32 safety (240 real frames, cold-batched, 40 it):")
    print(f"    task error:  f64 mean {e64.mean():.5f}   f32 mean {e32.mean():.5f}   "
          f"max |delta| {(e64-e32).abs().max():.2e}")
    print(f"    dof qpos  :  max |delta| {(d64-d32).abs().max():.2e} rad")

    # ---------- [3] fwd vs rev on GPU (quick, one config) ----------
    if torch.cuda.is_available():
        import torch.func as F
        rob = BatchedRobot(xml, device="cuda", dtype=torch.float32)
        tq, tp, W, rf = real_targets(torch.float32, "cuda", 240)
        fidx = torch.tensor([rob.body_names.index(n) for n in rf], device="cuda")
        B = 2048
        tqB = tq.repeat(9, 1, 1)[:B]; tpB = tp.repeat(9, 1, 1)[:B]
        q = default_q(model, B, torch.float32, "cuda")
        def e_item(xi, p0, r0, d0, tq_i, tp_i):
            p, qq, d = retract_b(p0, r0, d0, xi)
            return rob.task_error(p, qq, d, fidx, tq_i, tp_i)
        xi0 = torch.zeros(B, 6 + rob.num_dof, device="cuda")
        print("\n[3] jacfwd vs jacrev, GPU f32, B=2048:")
        for name, fn in [("vmap(jacrev)", F.vmap(F.jacrev(e_item))), ("vmap(jacfwd)", F.vmap(F.jacfwd(e_item)))]:
            try:
                fn(xi0, *q, tqB, tpB); torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5): fn(xi0, *q, tqB, tpB)
                torch.cuda.synchronize()
                print(f"    {name}: {(time.perf_counter()-t0)/5*1e3:8.1f} ms")
            except Exception as ex:
                print(f"    {name}: FAILED {type(ex).__name__}: {str(ex)[:70]}")

    # ---------- [2] the sweep ----------
    print("\n[2] LM hot-loop sweep (8 fixed iters; fps@10it = single-stage-cold budget):")
    rows = [("cpu-f64  B=256", "cpu", torch.float64, 256, 240, False),
            ("cpu-f32  B=256", "cpu", torch.float32, 256, 240, False),
            ("cpu-f32  B=2048", "cpu", torch.float32, 2048, 240, False)]
    if torch.cuda.is_available():
        rows += [("cuda-f32 B=256", "cuda", torch.float32, 256, 240, False),
                 ("cuda-f32 B=2048", "cuda", torch.float32, 2048, 240, False),
                 ("cuda-f32 B=8192", "cuda", torch.float32, 8192, 240, False),
                 ("cuda-f32 B=2048 +compile", "cuda", torch.float32, 2048, 240, True),
                 ("cuda-f32 B=8192 +compile", "cuda", torch.float32, 8192, 240, True)]
    for label, dev, dt, B, T, comp in rows:
        print("   " + bench_config(label, dev, dt, B, T, compiled=comp))

    print("\n[4] bars: mink 2.3 ms/frame = ~435 fps/core; ~10-15k fps on a full 32-core box.")


if __name__ == "__main__":
    main()
