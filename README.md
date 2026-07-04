# gmr-gpu

**Batched, differentiable GPU motion retargeting for humanoid robots** — a GPU
backend for [GMR](https://github.com/YanjieZe/GMR) that solves the *same*
retargeting problem (same IK configs, weights, and preprocessing, validated to
machine precision) as a batched projected Levenberg-Marquardt iteration in pure
PyTorch.

<!-- TODO: hero video — side-by-side (ours | mink), visually indistinguishable -->

## Why

CPU retargeting (mink/DAQP) solves one frame at a time. Motion-dataset
generation for RL doesn't need one frame at a time — it needs a million frames,
independently. Batching changes the economics:

- **Quality**: at production budgets, the batched solver converges deeper per
  frame and tracks the human measurably better than the CPU pipeline
  (task error 6.19 vs 6.46 on CMU 01_01, full clip, all statistics) — while
  being visually indistinguishable.
- **Throughput**: ~40 µs per frame-iteration at batch 8192 on a laptop RTX 4090
  (~2,600 fps at a 10-iteration budget).
- **Robustness**: memory-bounded chunked preprocessing survives clips that
  OOM-kill the production pipeline (100% CMU coverage vs ~70%).
- **Differentiable end-to-end**: pure-torch kinematics and solver — backprop
  through retargeting, learn IK weights, embed in training loops.

<!-- TODO: full-CMU 4-way benchmark table (repo / mink / seq / cold) -->

## How it works

- Pure-functional batched kinematics (`BatchedRobot`): no in-place writes, no
  TorchScript, no data-dependent branches — vmap/autograd/compile-legal by
  construction. Geometry from the compiled MuJoCo model.
- SE(3) body-twist error matching mink's convention to 2e-15.
- `vmap(jacrev)` task Jacobians; batched Cholesky normal equations with
  per-item Levenberg-Marquardt damping; joint limits as exact box projection;
  branchless accept/reject via masks.
- Cold-start with each frame's base initialized at its own pelvis target —
  basin-safe, no warm-start chain, so every frame of every clip solves in
  parallel.

## Quickstart

```python
from gmr_gpu import retarget_clips

motions = retarget_clips(["path/to/clip_stageii.npz"], robot="unitree_g1")
# motions[0] = {"fps", "root_pos" (T,3), "root_rot" (T,4 xyzw), "dof_pos" (T,29)}
```

Requires the SMPL-X body models (registration required, not redistributable) in
`$SMPLX_FOLDER` or GMR's `assets/body_models/` — same convention as GMR.

## Validation

Every layer is certified against an independent oracle (`tests/`):
forward kinematics vs MuJoCo (1e-15), twist error vs mink (1e-15), Jacobians
triangulated against finite differences *and* mink's analytic Jacobian,
batched-vs-sequential equivalence (1e-14), and full-clip output parity vs the
GMR/mink production pipeline on real CMU data. The build-and-certify journey is
preserved in [`docs/journey/`](docs/journey/).

## License

MIT. Builds on [GMR](https://github.com/YanjieZe/GMR) (MIT) by Yanjie Ze et al.
