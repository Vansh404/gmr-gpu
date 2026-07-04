"""
Step 7a (part 2) of GMR-GPU: the BATCHED projected-LM solver.

Everything from gn_loop.solve survives batching untouched -- except one thing:
`if err_cand < err:` is a PYTHON branch. A batch of B items needs B independent
accept/reject decisions in the SAME instruction stream. The GPU idiom is
BRANCHLESS MASKING:

    accept = err_cand < err                      # (B,) bool -- per-item verdict
    q      = torch.where(accept[:, None], q_cand, q)      # accepted items move
    lam    = torch.where(accept, lam*DOWN, lam*UP)        # per-item damping!

Every item computes both outcomes; the mask selects. No divergence, no active
sets -- and per-item adaptive lambda is exactly the thing SNS-style methods
can't do without branching. This file is the whole thesis in one loop:

    vmap(jacrev)  ->  batched J          (B,6K,35)
    batched H,c   ->  cholesky_solve     (B,35,35) SPD stack
    retract+clamp ->  projected candidates
    masks         ->  per-item LM accept/reject/convergence

Validation (the harness below):
  [A] batched(B=8) == loop of batched(B=1): batching is pure data parallelism,
      so results must match to ~f64 noise. (Same code path, so no binary
      accept-flip risk -- unlike comparing against the old single-item solver.)
  [B] all items: error decreased, joints feasible.
  [C] timing: ms/item/iter at B=256 (CPU f64 -- GPU/f32 is step 7d).

Problem definition: the REAL stage-2 retargeting problem -- 14 frames + squared
weights from smplx_to_g1.json (via your two_stage.py fills).

Run:  python scratch/batched_lm.py
"""

import time
import numpy as np
import torch
from .kinematics import quat_mul, so3_exp

LAM0, LAM_MIN, LAM_MAX = 1e-2, 1e-6, 1e6
LAM_DOWN, LAM_UP = 0.5, 4.0


# --------------------------------------------------------------------------
# Provided plumbing.
# --------------------------------------------------------------------------
def retract_b(pos0, quat0, dof0, xi):
    """Batched retraction, plain ops (vmap/autograd-safe). xi (...,6+nd)."""
    return (pos0 + xi[..., 0:3],
            quat_mul(so3_exp(xi[..., 3:6]), quat0),
            dof0 + xi[..., 6:])


def enorm_b(rob, q, fidx, tgt_quat, tgt_pos, W):
    """(B,) weighted error norms."""
    E = rob.task_error(q[0], q[1], q[2], fidx, tgt_quat, tgt_pos)   # (B,6K)
    return torch.sqrt((W * E * E).sum(-1))


def batched_jacobian(rob, q, fidx, tgt_quat, tgt_pos):
    """(B,6K,nv) via vmap(jacrev) at xi=0 -- certified in certify_batched_kin [4]."""
    def e_item(xi, p0, r0, d0, tq, tp):
        p, qq, d = retract_b(p0, r0, d0, xi)
        return rob.task_error(p, qq, d, fidx, tq, tp)
    B, nv = q[0].shape[0], 6 + q[2].shape[-1]
    xi0 = torch.zeros(B, nv, dtype=q[0].dtype, device=q[0].device)
    return torch.func.vmap(torch.func.jacrev(e_item))(xi0, q[0], q[1], q[2], tgt_quat, tgt_pos)


# --------------------------------------------------------------------------
# THE THING UNDER TEST (fill the TODOs).
# --------------------------------------------------------------------------
def solve_batched(rob, q0, fidx, tgt_quat, tgt_pos, W, max_iter=40, tol=1e-6):
    """Batched projected-LM IK. q0 = (pos (B,3), quat (B,4), dof (B,nd));
    targets (B,K,4)/(B,K,3); W (6K,) shared. Returns (best_q, best_err (B,))."""
    B = q0[0].shape[0]
    dt, dev = q0[0].dtype, q0[0].device
    nv = 6 + q0[2].shape[-1]

    q = tuple(x.clone() for x in q0)
    err = enorm_b(rob, q, fidx, tgt_quat, tgt_pos, W)               # (B,)
    best_q = tuple(x.clone() for x in q)
    best_err = err.clone()
    lam = torch.full((B,), LAM0, dtype=dt, device=dev)              # per-item!
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    for it in range(max_iter):
        J = batched_jacobian(rob, q, fidx, tgt_quat, tgt_pos)       # (B,6K,nv)
        E = rob.task_error(q[0], q[1], q[2], fidx, tgt_quat, tgt_pos)  # (B,6K)

        WJ = W[..., None] * J                  # diagonal weights, row-scaled
        I = torch.eye(nv, dtype=dt, device=dev)
        H = J.transpose(-1, -2) @ WJ + lam[:, None, None] * I
        c = WJ.transpose(-1, -2) @ E[..., None]#dont use .T on batched dims, transpose the matrix part, leave batch dims alone.
        dxi = torch.cholesky_solve(-c, torch.linalg.cholesky(H)).squeeze(-1)

        # projected candidate (provided): retract, clamp joints into the box
        q_cand = retract_b(q[0], q[1], q[2], dxi)
        q_cand = (q_cand[0], q_cand[1], torch.clamp(q_cand[2], rob.lo, rob.hi))
        err_cand = enorm_b(rob, q_cand, fidx, tgt_quat, tgt_pos, W)  # (B,)

       
        accept = (err_cand<err) & ~done
        improve = err - err_cand
        m = accept[:, None]
        q = (torch.where(m, q_cand[0], q[0]), torch.where(m, q_cand[1], q[1]), torch.where(m, q_cand[2], q[2]))
        err = torch.where(accept, err_cand, err)
        lam = torch.where(accept, (lam * LAM_DOWN).clamp_min(LAM_MIN),
              torch.where(~done, (lam * LAM_UP).clamp_max(LAM_MAX), lam))
        better = err < best_err
        bm = better[:, None]
        best_q = (torch.where(bm, q[0], best_q[0]), torch.where(bm, q[1], best_q[1]), torch.where(bm, q[2], best_q[2]))
        best_err = torch.where(better, err, best_err)
        done = done | (accept & (improve < tol)) | (lam >= LAM_MAX)
        if bool(done.all()):
            break

    return best_q, best_err


