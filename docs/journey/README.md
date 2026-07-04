# The journey

These are the actual build-and-certify steps, in order, as they were written --
each one a fill-in-the-blank exercise validated against an independent oracle
(MuJoCo, mink, finite differences) before the next step was allowed to begin:

1. `fk_parity.py` -- forward kinematics vs MuJoCo (found: XML quats are truncated; use compiled geometry)
2. `se3_error.py` -- SE(3) body-twist error vs mink's SE3.minus (2e-15)
3. `jacobian.py` -- manifold Jacobian via autograd, triangulated vs finite differences AND mink (found: NaN gradients in where-guarded quat ops)
4. `dls_step.py` -- one damped-least-squares step (batched Cholesky)
5. `gn_loop.py` -- robust projected Levenberg-Marquardt (clamp = exact box projection)
6. `two_stage.py` -- real IK-config weights (found: mink squares its costs) + coarse-to-fine
7. `certify_batched_kin.py` / `clip_parity.py` -- the batched engine certification and the full-clip parity harness vs mink on real CMU data

They import from the original GMR working tree and are kept verbatim as
documentation, not as package code. The distilled result is `gmr_gpu/`.
