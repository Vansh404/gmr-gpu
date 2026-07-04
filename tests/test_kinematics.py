"""Certification: FK vs MuJoCo, twist vs mink, vmap(jacrev) legality."""
import numpy as np
import pytest
import torch
import mujoco as mj
from conftest import random_qpos


def test_fk_matches_mujoco(model, rob):
    rng = np.random.default_rng(7)
    N = 100
    qs = np.stack([random_qpos(model, rng) for _ in range(N)])
    q = torch.tensor(qs, dtype=torch.float64)
    bp, br = rob.fk(q[:, 0:3], q[:, 3:7][:, [1, 2, 3, 0]], q[:, 7:])
    data = mj.MjData(model)
    ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in rob.body_names])
    for i in range(N):
        data.qpos[:] = qs[i]; mj.mj_kinematics(model, data)
        assert np.abs(bp[i].numpy() - data.xpos[ids]).max() < 1e-12
        dots = np.abs((br[i].numpy()[:, [3, 0, 1, 2]] * data.xquat[ids]).sum(-1))
        assert (1 - np.clip(dots, 0, 1)).max() < 1e-12


def test_twist_matches_mink():
    mink = pytest.importorskip("mink")
    from mink.lie import SE3
    from gmr_gpu import se3_twist
    rng = np.random.default_rng(0)

    def rand(n):
        ax = rng.standard_normal((n, 3)); ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
        an = rng.uniform(0, 2.0, (n, 1))
        return (torch.tensor(np.concatenate([ax * np.sin(an / 2), np.cos(an / 2)], -1)),
                torch.tensor(rng.uniform(-1, 1, (n, 3))))

    cq, cp = rand(500); tq, tp = rand(500)
    ours = se3_twist(cq, cp, tq, tp).numpy()
    for i in range(500):
        f = SE3(wxyz_xyz=np.concatenate([cq[i].numpy()[[3, 0, 1, 2]], cp[i].numpy()]))
        t = SE3(wxyz_xyz=np.concatenate([tq[i].numpy()[[3, 0, 1, 2]], tp[i].numpy()]))
        assert np.abs(ours[i] - t.minus(f)).max() < 1e-12


def test_vmap_jacrev_equals_loop(model, rob):
    from gmr_gpu.solver import batched_jacobian, retract_b
    rng = np.random.default_rng(3)
    B = 4
    qs = np.stack([random_qpos(model, rng) for _ in range(B)])
    q = torch.tensor(qs, dtype=torch.float64)
    q0 = (q[:, 0:3], q[:, 3:7][:, [1, 2, 3, 0]], q[:, 7:])
    xi = torch.tensor(rng.uniform(-0.2, 0.2, (B, 6 + rob.num_dof)), dtype=torch.float64)
    qt = retract_b(*q0, xi)
    fidx = torch.arange(6)
    bp, br = rob.fk(*qt)
    tp, tq = bp[:, :6], br[:, :6]
    JB = batched_jacobian(rob, q0, fidx, tq, tp)
    for i in range(B):
        qi = tuple(x[i:i + 1] for x in q0)
        Ji = batched_jacobian(rob, qi, fidx, tq[i:i + 1], tp[i:i + 1])
        assert (JB[i] - Ji[0]).abs().max() < 1e-12
