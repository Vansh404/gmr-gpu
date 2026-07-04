"""
Certification of batched_kin.py against every oracle that certified the old
stack (steps 1-3), plus the property it exists for: vmap(jacrev) legality.

  [1] FK parity vs MuJoCo (N random configs)              -- step-1 harness idea
  [2] twist parity vs mink SE3.minus                      -- step-2 oracle
  [3] Jacobian vs old validated stack (jacrev)            -- step-3 result
  [4] vmap(jacrev): works, and item i == single-item J    -- THE new property
  [5] timing taste: batched J at B=256, CPU f64

Run:  python scratch/certify_batched_kin.py
"""

import numpy as np
import torch
import time
import mujoco as mj

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel

import jacobian as JM          # old validated stack (user-filled)
import dls_step as DS
from se3_error import mink_twist, rand_poses
from fk_parity import make_qpos
from batched_kin import BatchedRobot, se3_twist as new_twist, so3_exp, quat_mul as new_qmul

DTYPE = torch.float64
ROBOT = "unitree_g1"


def new_retract(pos0, quat0, dof0, xi):
    """Same retraction as jacobian.py's, but on plain ops (vmap-safe)."""
    return (pos0 + xi[..., 0:3],
            new_qmul(so3_exp(xi[..., 3:6]), quat0),
            dof0 + xi[..., 6:])


def main():
    xml = str(ROBOT_XML_DICT[ROBOT])
    model = mj.MjModel.from_xml_path(xml); data = mj.MjData(model)
    rob = BatchedRobot(xml, dtype=DTYPE)
    rng = np.random.default_rng(7)

    # ---------- [1] FK parity vs MuJoCo ----------
    N = 500
    qpos_batch = np.stack([make_qpos(model, "random", rng) for _ in range(N)])
    q = torch.tensor(qpos_batch, dtype=DTYPE)
    bp, br = rob.fk(q[:, 0:3], q[:, 3:7][:, [1, 2, 3, 0]], q[:, 7:])
    ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in rob.body_names])
    mj_pos = np.zeros((N, len(ids), 3)); mj_quat = np.zeros((N, len(ids), 4))
    for i in range(N):
        data.qpos[:] = qpos_batch[i]; mj.mj_kinematics(model, data)
        mj_pos[i] = data.xpos[ids]; mj_quat[i] = data.xquat[ids]
    perr = np.abs(bp.numpy() - mj_pos).max()
    dots = np.abs((br.numpy()[..., [3, 0, 1, 2]] * mj_quat).sum(-1))
    oerr = (1 - np.clip(dots, 0, 1)).max()
    ok1 = perr < 1e-12 and oerr < 1e-12
    print(f"[1] FK vs MuJoCo (N={N}): pos {perr:.2e}  ori {oerr:.2e}   -> {'PASS' if ok1 else 'FAIL'}")

    # ---------- [2] twist parity vs mink ----------
    cq, cp = rand_poses(rng, 2000); tq, tp = rand_poses(rng, 2000)
    terr = np.abs(new_twist(cq, cp, tq, tp).numpy() - mink_twist(cq, cp, tq, tp)).max()
    print(f"[2] twist vs mink SE3.minus:            {terr:.2e}   -> {'PASS' if terr < 1e-12 else 'FAIL'}")

    # ---------- shared problem for [3][4][5] ----------
    kin = KinematicsModel(xml, device="cpu"); JM.to_double(kin); JM.use_compiled_geometry(kin, model)
    frame_names = JM.TASK_FRAMES
    fidx_old = [kin.get_body_idx(n) for n in frame_names]
    fidx_new = torch.tensor([rob.body_names.index(n) for n in frame_names])
    qp = make_qpos(model, "random", rng); qt = torch.tensor(qp, dtype=DTYPE)
    q0 = (qt[0:3], qt[3:7][[1, 2, 3, 0]], qt[7:])
    tq2, tp2 = DS.frame_poses(kin, JM.retract(*q0, torch.tensor(rng.uniform(-0.2, 0.2, 35), dtype=DTYPE)), fidx_old)
    tq2, tp2 = torch.stack(tq2), torch.stack(tp2)

    def e_old(xi):
        p, qq, d = JM.retract(*q0, xi)
        return JM.task_error(kin, p, qq, d, fidx_old, tq2, tp2)

    def e_new(xi):
        p, qq, d = new_retract(*q0, xi)
        return rob.task_error(p, qq, d, fidx_new, tq2, tp2)

    xi0 = torch.zeros(35, dtype=DTYPE)
    for _ in range(3): e_old(xi0)                      # old stack needs TS warm-up
    J_old = torch.func.jacrev(e_old)(xi0)
    J_new = torch.func.jacrev(e_new)(xi0)
    jerr = (J_old - J_new).abs().max().item()
    eerr = (e_old(xi0) - e_new(xi0)).abs().max().item()
    print(f"[3] e/J vs old validated stack:  e {eerr:.2e}  J {jerr:.2e}   -> {'PASS' if max(jerr, eerr) < 1e-12 else 'FAIL'}")

    # ---------- [4] vmap(jacrev): the property this module exists for ----------
    B = 16
    q0b = (q0[0].expand(B, 3).clone(), q0[1].expand(B, 4).clone(), q0[2].expand(B, 29).clone())
    tqb, tpb = tq2.expand(B, -1, -1).clone(), tp2.expand(B, -1, -1).clone()
    xis = torch.randn(B, 35, dtype=DTYPE) * 0.05

    def e_item(xi, p0, r0, d0, tq_i, tp_i):
        p, qq, d = new_retract(p0, r0, d0, xi)
        return rob.task_error(p, qq, d, fidx_new, tq_i, tp_i)

    try:
        JB = torch.func.vmap(torch.func.jacrev(e_item))(xis, *q0b, tqb, tpb)
        ref = torch.stack([torch.func.jacrev(lambda x: e_item(x, q0[0], q0[1], q0[2], tq2, tp2))(xis[i]) for i in range(B)])
        verr = (JB - ref).abs().max().item()
        print(f"[4] vmap(jacrev) B={B}: shape {tuple(JB.shape)}, vs per-item loop {verr:.2e}   -> {'PASS' if verr < 1e-12 else 'FAIL'}")
    except Exception as ex:
        print(f"[4] vmap(jacrev) FAILED: {type(ex).__name__}: {str(ex)[:150]}")

    # ---------- [5] timing taste ----------
    B = 256
    xis = torch.randn(B, 35, dtype=DTYPE) * 0.05
    q0b = (q0[0].expand(B, 3).clone(), q0[1].expand(B, 4).clone(), q0[2].expand(B, 29).clone())
    tqb, tpb = tq2.expand(B, -1, -1).clone(), tp2.expand(B, -1, -1).clone()
    f = torch.func.vmap(torch.func.jacrev(e_item))
    f(xis, *q0b, tqb, tpb)                             # warm
    t0 = time.perf_counter(); f(xis, *q0b, tqb, tpb); t1 = time.perf_counter()
    per = (t1 - t0) / B * 1e3
    print(f"[5] batched J, B={B} CPU f64: {(t1-t0)*1e3:.1f} ms total = {per:.3f} ms/item "
          f"(single-item jacrev was ~44 ms at K=8)")


if __name__ == "__main__":
    main()
