"""
Step 7b of GMR-GPU: real SMPL-X human targets, batched over frames.

GMR's update_targets() preprocesses each human frame before the IK sees it:

  1. SCALE (motion_retarget.scale_human_data): shrink the human to robot
     proportions, in the ROOT-LOCAL frame: every body's position is expressed
     relative to the root, scaled per-body (legs/torso 0.9, arms 0.8 for G1 --
     table pre-multiplied by actual_height/1.8), then re-attached to the scaled
     root. Quaternions untouched.
  2. OFFSET (motion_retarget.offset_human_data): per body, rotate the frame into
     the robot's link convention (q <- q (x) q_off, RIGHT-multiply = local frame)
     and shift the position by the offset expressed in the NEW frame
     (p <- p + R(q_new) @ pos_off). QUIRK (verified motion_retarget.py:154):
     only TABLE-1 offsets are ever applied -- table-2's are built and unused.
     We replicate the actual behavior.
  3. ground offset: p -= [0,0,ground_offset] (0.0 unless set; kept for parity).

This module does the same thing to ALL T frames at once: (T,K,3)/(T,K,4)
tensors in, targets out, ready for solve_batched. Quats are xyzw internally
(the human data arrives wxyz from get_smplx_data_offline_fast -- converted at
ingestion).

Validation: run GMR's own update_targets() on real CMU frames and compare its
retarget.scaled_human_data against our batched output. Bar ~1e-12 (same math,
numpy/scipy vs torch f64).

Run:  python scratch/human_targets.py
"""

import json
import numpy as np
import torch

from general_motion_retargeting.params import IK_CONFIG_DICT
from .kinematics import quat_mul, quat_rotate

import os

DTYPE = torch.float64
ROBOT = "unitree_g1"
# SMPL-X body models cannot be redistributed -- point this at your download
# (defaults to the GMR package's assets dir, same convention as GMR itself).
def _default_smplx_folder():
    import general_motion_retargeting as gmr
    return os.environ.get("SMPLX_FOLDER",
                          os.path.join(os.path.dirname(os.path.dirname(gmr.__file__)),
                                       "assets", "body_models"))
SMPLX_FOLDER = _default_smplx_folder()


def load_frames(smplx_file, tgt_fps=30, chunk=2000):
    """Real clip -> (frames list of {name: (pos wxyz-quat)}, fps, human_height).

    The repo's load_smplx_file runs the SMPL-X forward over the WHOLE clip in
    one batch: memory scales with clip length (normal clip ~4.3 GB transient;
    a 20k-frame monster wants ~30 GB -> OOM-killed five repo-script workers on
    the 15 GB WSL). Short clips use the validated repo path unchanged; long
    clips run the SAME forward in bounded chunks and are stitched together."""
    import types
    import smplx as smplx_lib
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    with torch.no_grad():
        d = np.load(smplx_file, allow_pickle=True)
        N = d["pose_body"].shape[0]
        if N <= 2 * chunk:                       # validated original path
            smplx_data, body_model, smplx_output, height = load_smplx_file(smplx_file, SMPLX_FOLDER)
            frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
            return frames, fps, height

        # chunked path: identical math, bounded memory
        body_model = smplx_lib.create(SMPLX_FOLDER, "smplx",
                                      gender=str(d["gender"]), use_pca=False)
        joints, glob, full = [], [], []
        for s in range(0, N, chunk):
            n = min(chunk, N - s)
            out = body_model(
                betas=torch.tensor(d["betas"]).float().view(1, -1),
                global_orient=torch.tensor(d["root_orient"][s:s + n]).float(),
                body_pose=torch.tensor(d["pose_body"][s:s + n]).float(),
                transl=torch.tensor(d["trans"][s:s + n]).float(),
                left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
                jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3),
                reye_pose=torch.zeros(n, 3), return_full_pose=True)
            joints.append(out.joints); glob.append(out.global_orient); full.append(out.full_pose)
        smplx_output = types.SimpleNamespace(joints=torch.cat(joints),
                                             global_orient=torch.cat(glob),
                                             full_pose=torch.cat(full))
        betas = d["betas"]
        height = 1.66 + 0.1 * (betas[0] if betas.ndim == 1 else betas[0, 0])
        frames, fps = get_smplx_data_offline_fast(d, body_model, smplx_output, tgt_fps=tgt_fps)
    return frames, fps, height


class TargetPipeline:
    """Batched port of GMR's scale+offset preprocessing for one (human, robot)
    pair. Built once from the IK config; applies to (T,K,·) tensors."""

    def __init__(self, actual_human_height, robot=ROBOT, dtype=DTYPE, device="cpu"):
        with open(IK_CONFIG_DICT["smplx"][robot]) as f:
            cfg = json.load(f)
        t1 = cfg["ik_match_table1"]
        self.robot_frames = list(t1.keys())                      # K robot links
        self.human_names = [t1[k][0] for k in self.robot_frames] # K human bodies
        self.root_idx = self.human_names.index(cfg["human_root_name"])

        ratio = actual_human_height / cfg["human_height_assumption"]
        scale = [cfg["human_scale_table"][n] * ratio for n in self.human_names]
        ground = np.array([0.0, 0.0, cfg["ground_height"]])
        pos_off = [np.array(t1[k][3]) - ground for k in self.robot_frames]   # table-1 ONLY (the quirk)
        rot_off_wxyz = np.array([t1[k][4] for k in self.robot_frames])

        tt = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device)
        self.scale = tt(scale)                                   # (K,)
        self.pos_off = tt(pos_off)                               # (K,3)
        rot_off = tt(rot_off_wxyz[:, [1, 2, 3, 0]])              # (K,4) xyzw
        # JSON quats are ~2e-9 off unit; scipy R.from_quat normalizes silently,
        # so we must too (same lesson as step 3's XML-vs-compiled geometry).
        self.rot_off = rot_off / rot_off.norm(dim=-1, keepdim=True)
        self.ground_offset = 0.0                                 # GMR default

    def stack_frames(self, frames):
        """frames (dicts, wxyz quats) -> pos (T,K,3), quat (T,K,4) xyzw."""
        T, K = len(frames), len(self.human_names)
        pos = np.zeros((T, K, 3)); quat = np.zeros((T, K, 4))
        for t, fr in enumerate(frames):
            for k, name in enumerate(self.human_names):
                p, q = fr[name]
                pos[t, k] = p; quat[t, k] = np.asarray(q)[[1, 2, 3, 0]]   # wxyz->xyzw
        return (torch.tensor(pos, dtype=self.scale.dtype, device=self.scale.device),
                torch.tensor(quat, dtype=self.scale.dtype, device=self.scale.device))

    def __call__(self, pos, quat):
        """(T,K,3),(T,K,4 xyzw) -> scaled+offset targets, same shapes."""
        r = self.root_idx

        #  A: SCALE in the root-local frame (port of scale_human_data).
        
       
       
        root_pos = pos[:, r:r+1, :]
        scaled_root = self.scale[r] * root_pos
        pos = (pos - root_pos) * self.scale[None, :, None] + scaled_root

        # B: OFFSET into robot link conventions (port of offset_human_data).
        
        
        quat = quat_mul(quat, self.rot_off)
        pos = pos + quat_rotate(quat, self.pos_off.expand_as(pos))

        pos = pos - torch.tensor([0.0, 0.0, self.ground_offset],
                                 dtype=pos.dtype, device=pos.device)
        return pos, quat


