"""
Step 4 of GMR-GPU: one damped-least-squares (DLS) step -- the first time J
actually MOVES the robot toward the targets.

We minimize the weighted task error 0.5 * ||e||^2_W over the tangent step dxi.
Linearize e(xi) ~ e0 + J dxi (step-3 J), and the Gauss-Newton + Tikhonov step is:

    H  = J^T W J + lambda I          # (nv x nv) SPD Gauss-Newton Hessian + damping
    c  = J^T W e                     # (nv,)     gradient of 0.5||e||^2_W at xi=0
    H dxi = -c   ->   dxi = -(J^T W J + lambda I)^{-1} J^T W e     # solve via Cholesky

Then retract:  q <- retract(q, dxi), relinearize, repeat (that loop is step 5).
The damping lambda keeps H well-conditioned near singularities and bounds the
step; W will hold the real per-task position/orientation costs in step 6 (here
it's identity).

Cholesky (not a generic inverse) because H is symmetric positive-definite -- and
it's the batched-GPU-friendly solve (torch.linalg.cholesky_solve over a (B,nv,nv)
stack later).

Run:  python scratch/dls_step.py    (needs jacobian.py TODO 2 filled)
"""

import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel

from fk_parity import make_qpos
from jacobian import (
    DTYPE, DEVICE, ROBOT, TASK_FRAMES,
    to_double, use_compiled_geometry, retract, compute_jacobian, task_error,
)

LAMBDA = 1e-2      # Tikhonov damping (mink uses 0.5; smaller = faster here)


# --------------------------------------------------------------------------
# THE THING UNDER TEST (fill the TODOs).
# --------------------------------------------------------------------------
def dls_step(kin, q, frame_idx, tgt_quat, tgt_pos, weights, lam):
    """One DLS step at config q. Returns (dxi (nv,), e (6K,), J (6K,nv))."""
    J, e_of_xi = compute_jacobian(kin, q, frame_idx, tgt_quat, tgt_pos)
    if J is None:
        raise RuntimeError("J is None -- fill jacobian.py TODO 2 first.")
    e = e_of_xi(torch.zeros(J.shape[1], dtype=DTYPE, device=DEVICE))   # error at xi=0
    nv = J.shape[1]

    # assemble the normal-equations pieces with diagonal weights W=weights.
    #   H = J^T W J + lam * I      (nv x nv)  -> so not pure GN, Levenberg-Marquadt
    #   c = J^T W e                (nv,)
    #   
    H = J.T @ (weights.unsqueeze(-1)  * J) + lam*torch.eye(nv, dtype = DTYPE, device = DEVICE)
    c = J.T @ (weights*e)   

    # TODO 2: solve H dxi = -c for dxi, using Cholesky (H is SPD).
    #   L = torch.linalg.cholesky(H)
    #   torch.cholesky_solve(B, L) solves H X = B(gradQP = 0 essentially); B must be (nv, 1) -> squeeze.
    L =  torch.linalg.cholesky(H)
    dxi = torch.cholesky_solve((-c).unsqueeze(-1), L).squeeze(-1)

    return dxi, e, J


# --------------------------------------------------------------------------
# Helpers 
# --------------------------------------------------------------------------
def frame_poses(kin, q, frame_idx):
    pos0, quat0, dof0 = q
    bp, br = kin.forward_kinematics(pos0.view(1, 3), quat0.view(1, 4), dof0.view(1, -1))
    bp = bp.squeeze(0); br = br.squeeze(0)
    return [br[f].clone() for f in frame_idx], [bp[f].clone() for f in frame_idx]


def reachable_targets(kin, q, frame_idx, rng, scale=0.3):
    """Targets = frame poses at a nearby REACHABLE config q_true = retract(q, xi_true),
    so a solution exists and ||e|| can actually be driven to ~0."""
    xi_true = torch.tensor(rng.uniform(-scale, scale, 6 + q[2].numel()), dtype=DTYPE)
    q_true = retract(*q, xi_true)
    tq, tp = frame_poses(kin, q_true, frame_idx)
    return torch.stack(tq), torch.stack(tp)


def enorm(kin, q, frame_idx, tgt_quat, tgt_pos, weights):
    # error only -- no Jacobian (task_error is FK + twist, far cheaper than autograd)
    e = task_error(kin, q[0], q[1], q[2], frame_idx, tgt_quat, tgt_pos)
    return float(torch.sqrt((weights * e * e).sum()))


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml)
    kin = KinematicsModel(xml, device=DEVICE); to_double(kin); use_compiled_geometry(kin, model)
    rng = np.random.default_rng(0)

    frame_idx = [kin.get_body_idx(n) for n in TASK_FRAMES]
    nv = 6 + (model.nq - 7)
    weights = torch.ones(6 * len(TASK_FRAMES), dtype=DTYPE, device=DEVICE)   # W = I for now

    qpos = make_qpos(model, "random", rng)
    qt = torch.tensor(qpos, dtype=DTYPE, device=DEVICE)
    q = (qt[0:3], qt[3:7][[1, 2, 3, 0]], qt[7:])
    tgt_quat, tgt_pos = reachable_targets(kin, q, frame_idx, rng)

    # --- one step, validated ---
    dxi, e, J = dls_step(kin, q, frame_idx, tgt_quat, tgt_pos, weights, LAMBDA)
    if dxi is None:
        print("dls_step returned None -- fill the TODOs.")
        return

    H = J.T @ (weights.unsqueeze(-1) * J) + LAMBDA * torch.eye(nv, dtype=DTYPE)
    c = J.T @ (weights * e)
    print("=== one DLS step ===")
    print(f"  normal-eqn residual ||H dxi + c||_inf : {(H @ dxi + c).abs().max():.3e}  (should be ~0)")
    e0 = float(torch.sqrt((weights * e * e).sum()))
    q1 = retract(*q, dxi)
    e1 = enorm(kin, q1, frame_idx, tgt_quat, tgt_pos, weights)
    print(f"  ||e|| before step : {e0:.4e}")
    print(f"  ||e|| after  step : {e1:.4e}   -> {'descent OK' if e1 < e0 else 'NOT a descent!'}")

    # --- iterate (preview of the step-5 loop): watch ||e|| fall ---
    print("\n=== iterating the step (no limits yet -- that's step 5) ===")
    qk = q
    for it in range(12):
        dxi, e, J = dls_step(kin, qk, frame_idx, tgt_quat, tgt_pos, weights, LAMBDA)
        qk = retract(*qk, dxi)
        print(f"  iter {it:2d}   ||e|| = {enorm(kin, qk, frame_idx, tgt_quat, tgt_pos, weights):.4e}")


if __name__ == "__main__":
    main()
