"""
Step 2 of GMR-GPU: the SE(3) task error (the residual `e`).

Each IK task says "robot frame X should sit at target pose T_tgt". The error
that drives the solver is NOT T_tgt - T_cur (you can't subtract poses) -- it's
the log map of the relative transform, a 6-vector "body twist":

    e = log( T_cur^{-1} . T_tgt )   expressed in the current/body frame

We mirror mink's exact convention so our `e` equals mink's `e` (parity):
  - direction : log(T_cur^{-1} T_tgt)         (mink: target (-) frame, right-minus)
  - ordering  : [ translation-part(3) , rotation-part(3) ]   (translation FIRST)
  - the translation part is V^{-1}(omega) @ t_rel, NOT raw t_rel -- the SE(3) log
    couples rotation into translation. omega is the rotation part = SO(3) log.

Reference = mink's own lie math: SE3(...).minus(...), i.e. exactly what
FrameTask.compute_error uses. Fill the 5 TODOs, then run:
    python scratch/se3_error.py

Quaternions here are xyzw (KinematicsModel / torch_utils convention). MuJoCo &
mink are wxyz -- we convert only when calling the reference.
"""

import numpy as np
import torch

from general_motion_retargeting.torch_utils import (
    quat_conjugate, quat_mul, quat_rotate, quat_to_exp_map,
)
from mink.lie import SE3

# Validate the MATH cleanly in f64 (this is our own code, not tied to
# KinematicsModel's f32). The real solver will run f32 on GPU; precision is a
# separate concern from convention, which is what we're checking here.
DTYPE = torch.float64


# --------------------------------------------------------------------------
# Helpers (written for you).
# --------------------------------------------------------------------------
def skew(v):
    """(...,3) -> (...,3,3) skew-symmetric matrix [v]_x."""
    ox, oy, oz = v[..., 0], v[..., 1], v[..., 2]
    z = torch.zeros_like(ox)
    r0 = torch.stack([z, -oz, oy], dim=-1)
    r1 = torch.stack([oz, z, -ox], dim=-1)
    r2 = torch.stack([-oy, ox, z], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def se3_log_Vinv(omega):
    """(N,3) -> (N,3,3). The V^{-1}(omega) matrix from the SE(3) log, matching
    mink's SE3.log exactly (small-angle series below the threshold)."""
    theta = omega.norm(dim=-1)
    S = skew(omega)
    S2 = S @ S
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(S.shape)
    small = theta * theta < 1e-10
    theta_safe = torch.where(small, torch.ones_like(theta), theta)   # avoid /0
    half = 0.5 * theta_safe
    coef_big = (1.0 - 0.5 * theta_safe * torch.cos(half) / torch.sin(half)) / (theta_safe ** 2)
    coef = torch.where(small, torch.full_like(theta, 1.0 / 12.0), coef_big)
    return I - 0.5 * S + coef[..., None, None] * S2


# --------------------------------------------------------------------------
# THE THING UNDER TEST (fill the TODOs).
# --------------------------------------------------------------------------
def se3_twist(cur_quat, cur_pos, tgt_quat, tgt_pos):
    """Body twist e = log(T_cur^{-1} . T_tgt), batched.
    Define twist to be a 6vector in SE(3)->[linear(3), angular(3)]. N is batch size
    Inputs: cur_quat,tgt_quat (N,4) xyzw ; cur_pos,tgt_pos (N,3).
    Returns: (N,6) = [ V^{-1}(omega)@t_rel , omega ]  (translation part first)."""

    # TODO 1: relative rotation quaternion of T_cur^{-1} . T_tgt, i.e. R_cur^{-1} R_tgt.
    #   quat_conjugate(q) is the inverse of a unit quat; quat_mul(a,b) = R_a R_b.
    
    q_rel = quat_mul(quat_conjugate(cur_quat),tgt_quat) #undo my current frame, tell me where target is
    
    # TODO 2: relative translation of T_cur^{-1} . T_tgt.
    #   SE(3) inverse-compose gives t_rel = R_cur^{-1} ( t_tgt - t_cur ).
    #   quat_rotate(q, v) applies rotation q to vector v.
    t_rel = quat_rotate(quat_conjugate(cur_quat),tgt_pos-cur_pos)

    # TODO 3: rotation part of the twist = SO(3) log of q_rel (axis-angle 3-vec).
    #   quat_to_exp_map(q) returns axis*angle, with double-cover handled.
    omega = quat_to_exp_map(q_rel)

    # TODO 4: translation part rho = V^{-1}(omega) @ t_rel.
    #   se3_log_Vinv(omega) is (N,3,3); t_rel is (N,3). You need a batched
    #   matrix-vector product (torch.einsum or unsqueeze/@/squeeze).
    rho = torch.einsum('nij,nj->ni',se3_log_Vinv(omega), t_rel)

    # TODO 5: stack into the (N,6) twist in mink's order -- translation part FIRST,
    #   then rotation part. (torch.cat along the last dim.)
    twist = torch.cat([rho,omega], dim = -1) 

    return twist


# --------------------------------------------------------------------------
# Reference: mink's own lie math (ground truth). SE3.minus == compute_error.
# --------------------------------------------------------------------------
def mink_twist(cur_quat, cur_pos, tgt_quat, tgt_pos):
    def xyzw_to_wxyz(q):
        return q[[3, 0, 1, 2]]
    cq = cur_quat.detach().cpu().numpy(); cp = cur_pos.detach().cpu().numpy()
    tq = tgt_quat.detach().cpu().numpy(); tp = tgt_pos.detach().cpu().numpy()
    N = cq.shape[0]
    out = np.zeros((N, 6))
    for i in range(N):
        frame = SE3(wxyz_xyz=np.concatenate([xyzw_to_wxyz(cq[i]), cp[i]]))
        target = SE3(wxyz_xyz=np.concatenate([xyzw_to_wxyz(tq[i]), tp[i]]))
        out[i] = target.minus(frame)         # = log(T_cur^{-1} T_tgt), mink's e
    return out


# --------------------------------------------------------------------------
# Harness.
# --------------------------------------------------------------------------
def rand_poses(rng, n):
    """Random unit quats (xyzw) with BOUNDED angle (< 2 rad) to stay clear of the
    theta=pi axis-sign ambiguity -- that's a known double-cover edge, not a
    convention bug, and it would add spurious noise to a convention check."""
    axis = rng.standard_normal((n, 3)); axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    ang = rng.uniform(0.0, 2.0, (n, 1))
    quat = np.concatenate([axis * np.sin(ang / 2), np.cos(ang / 2)], axis=-1)  # xyzw
    pos = rng.uniform(-1.0, 1.0, (n, 3))
    return torch.tensor(quat, dtype=DTYPE), torch.tensor(pos, dtype=DTYPE)


def main():
    rng = np.random.default_rng(0)
    N = 4000
    cur_quat, cur_pos = rand_poses(rng, N)
    tgt_quat, tgt_pos = rand_poses(rng, N)

    ours = se3_twist(cur_quat, cur_pos, tgt_quat, tgt_pos)
    if ours is None:
        print("se3_twist returned None -- fill the TODOs.")
        return
    ours = ours.detach().cpu().numpy()
    ref = mink_twist(cur_quat, cur_pos, tgt_quat, tgt_pos)

    err = np.abs(ours - ref)
    print(f"=== SE(3) twist parity vs mink (N={N}) ===")
    print(f"  translation half  max err: {err[:, :3].max():.3e}")
    print(f"  rotation half     max err: {err[:, 3:].max():.3e}")
    print(f"  OVERALL           max err: {err.max():.3e}   -> {'PASS' if err.max() < 1e-8 else 'FAIL'}")

    # structure sanity: identical poses must give the zero twist
    z = se3_twist(cur_quat, cur_pos, cur_quat, cur_pos)
    print(f"  identity twist norm (should be ~0): {float(z.abs().max()):.3e}")


if __name__ == "__main__":
    main()
