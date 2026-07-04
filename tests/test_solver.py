"""Certification: batching is pure data parallelism; solver descends + stays feasible."""
import numpy as np
import torch
from conftest import random_qpos
from gmr_gpu import solve_batched, enorm_b, retract_b, load_ik_tables, weights_from_costs


def _problem(model, rob, rng, B):
    qs = np.stack([random_qpos(model, rng) for _ in range(B)])
    q = torch.tensor(qs, dtype=torch.float64)
    q0 = (q[:, 0:3], q[:, 3:7][:, [1, 2, 3, 0]], q[:, 7:])
    xi = torch.tensor(rng.uniform(-0.25, 0.25, (B, 6 + rob.num_dof)), dtype=torch.float64)
    frame_names, _, c2 = load_ik_tables()
    fidx = torch.tensor([rob.body_names.index(n) for n in frame_names])
    bp, br = rob.fk(*retract_b(*q0, xi))
    return q0, fidx, br.index_select(-2, fidx), bp.index_select(-2, fidx), weights_from_costs(c2)


def test_batched_equals_loop(model, rob):
    rng = np.random.default_rng(3)
    q0, fidx, tq, tp, W = _problem(model, rob, rng, 4)
    bq, berr = solve_batched(rob, q0, fidx, tq, tp, W, max_iter=20)
    for i in range(4):
        qi = tuple(x[i:i + 1] for x in q0)
        bqi, be = solve_batched(rob, qi, fidx, tq[i:i + 1], tp[i:i + 1], W, max_iter=20)
        assert abs(float(berr[i] - be[0])) < 1e-10
        assert (bq[2][i] - bqi[2][0]).abs().max() < 1e-8


def test_descends_and_feasible(model, rob):
    rng = np.random.default_rng(5)
    q0, fidx, tq, tp, W = _problem(model, rob, rng, 8)
    e0 = enorm_b(rob, q0, fidx, tq, tp, W)
    bq, berr = solve_batched(rob, q0, fidx, tq, tp, W, max_iter=30)
    assert bool((berr < e0).all())
    assert bool(((bq[2] >= rob.lo - 1e-9) & (bq[2] <= rob.hi + 1e-9)).all())
