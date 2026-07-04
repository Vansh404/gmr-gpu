"""
Step 5 of GMR-GPU: a ROBUST projected Gauss-Newton loop (Levenberg-Marquardt).

Step 4 iterated a fixed-lambda DLS step; we saw it bounce, and worse -- when a
big step gets heavily clamped it can land at a point WORSE than it started, and
a naive "improvement < tol" stop then returns that worse point. Real motion data
(CMU) will throw infeasible targets at us, so the loop must be robust to that.

Three ingredients:
  1. PROJECTION. Clamp joints into [lo, hi] each iteration -- exact box projection
     -> projected-Newton, feasible, batchable, no active-set. (Base is free.)
  2. STEP ACCEPTANCE (LM). Only accept a (projected) step if it DECREASES the
     error. If it does, lower the damping lambda (toward Gauss-Newton, fast). If
     it doesn't, REJECT it (keep the old config) and raise lambda (toward gradient
     descent, small safe steps). This is what makes LM robust where GN overshoots.
  3. RETURN-BEST. Track and return the best config seen, never a regression.

Converged when an accepted step improves by < tol; give up if lambda explodes.
All of this is batchable later (per-item lambda + an accept mask on the GPU).

Run:  python scratch/gn_loop.py   (needs jacobian.py + dls_step.py TODOs filled)
"""

import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel

from fk_parity import make_qpos
from jacobian import (
    DTYPE, DEVICE, ROBOT, TASK_FRAMES,
    to_double, use_compiled_geometry, retract,
)
from dls_step import dls_step, frame_poses, enorm

LAM_INIT = 1e-2
LAM_MIN = 1e-6
LAM_MAX = 1e6
LAM_DOWN = 0.5      # on accept: lambda *= LAM_DOWN  (trust GN more)
LAM_UP = 4.0        # on reject: lambda *= LAM_UP    (damp harder)


# --------------------------------------------------------------------------
# THE THING UNDER TEST (fill the TODOs).
# --------------------------------------------------------------------------
def solve(kin, q0, frame_idx, tgt_quat, tgt_pos, weights, lo, hi,
          project=True, max_iter=60, tol=1e-6):
    """Robust projected LM Gauss-Newton IK. Returns (best_q, history_of_||e||)."""
    q = q0
    err = enorm(kin, q, frame_idx, tgt_quat, tgt_pos, weights)
    best_q, best_err = q, err
    lam = LAM_INIT
    hist = [err]

    for it in range(max_iter):
        dxi, e, J = dls_step(kin, q, frame_idx, tgt_quat, tgt_pos, weights, lam)
        q_cand = retract(*q, dxi)

        # TODO 1: projection. If `project`, clamp the candidate's JOINTS into
        #   [lo, hi]; leave the base (q_cand[0], q_cand[1]) untouched.
        #   q_cand = (q_cand[0], q_cand[1], torch.clamp(q_cand[2], lo, hi))
        if project:
            q_cand = (q_cand[0], q_cand[1], torch.clamp(q_cand[2], lo, hi))   # <<< fill

        err_cand = enorm(kin, q_cand, frame_idx, tgt_quat, tgt_pos, weights)

        # : LM step acceptance.
        
        if err_cand < err:   # ACCEPT
            improved = err - err_cand
            q, err = q_cand, err_cand
            if err < best_err: best_q, best_err = q, err
            lam = max(lam * LAM_DOWN, LAM_MIN)
            if improved < tol: break          # converged
        else:                # REJECT (keep q), damp harder and retry
            lam = lam * LAM_UP
            if lam > LAM_MAX: break           # stuck

        hist.append(err)

    return best_q, hist


# --------------------------------------------------------------------------
# Helpers (written for you).
# --------------------------------------------------------------------------
def saturating_targets(kin, q, frame_idx, rng, hi, over_joints, overshoot):
    """Targets = frame poses at a config whose `over_joints` are pushed `overshoot`
    PAST their upper limits -> forces the projection to engage."""
    xi = torch.tensor(rng.uniform(-0.3, 0.3, 6 + q[2].numel()), dtype=DTYPE)
    q_true = retract(*q, xi)
    dof = q_true[2].clone()
    for j in over_joints:
        dof[j] = hi[j] + overshoot
    tq, tp = frame_poses(kin, (q_true[0], q_true[1], dof), frame_idx)
    return torch.stack(tq), torch.stack(tp)


def limit_report(dof, lo, hi, tol=1e-6):
    viol = ((dof < lo - tol) | (dof > hi + tol)).sum().item()
    at = ((dof <= lo + tol) | (dof >= hi - tol)).sum().item()
    return viol, at


def monotonic(hist):
    return all(b <= a + 1e-12 for a, b in zip(hist, hist[1:]))


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    kin = KinematicsModel(xml, device=DEVICE); to_double(kin); use_compiled_geometry(kin, model)
    rng = np.random.default_rng(0)

    frame_idx = [kin.get_body_idx(n) for n in TASK_FRAMES]
    weights = torch.ones(6 * len(TASK_FRAMES), dtype=DTYPE, device=DEVICE)
    lo, hi = kin.get_dof_limits(); lo = lo.to(DTYPE); hi = hi.to(DTYPE)

    qpos = make_qpos(model, "random", rng)
    qt = torch.tensor(qpos, dtype=DTYPE, device=DEVICE)
    q0 = (qt[0:3], qt[3:7][[1, 2, 3, 0]], qt[7:])

    # two regimes CMU will actually hand us: mildly and severely infeasible
    for label, overshoot in [("mild infeasible (+0.05)", 0.05), ("severe infeasible (+0.5)", 0.5)]:
        tq, tp = saturating_targets(kin, q0, frame_idx, rng, hi, (3, 4, 5), overshoot)
        q, hist = solve(kin, q0, frame_idx, tq, tp, weights, lo, hi, project=True)
        if q is None:
            print("solve returned None -- fill the TODOs."); return
        viol, at = limit_report(q[2], lo, hi)
        print(f"=== {label} ===")
        print(f"  iters {len(hist) - 1:2d}  |  final ||e|| {hist[-1]:.4e}  |  "
              f"out-of-range {viol}  at-limit {at}  |  monotonic: {monotonic(hist)}")


if __name__ == "__main__":
    main()
