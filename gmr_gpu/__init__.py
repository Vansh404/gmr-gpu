"""gmr-gpu: batched, differentiable GPU motion retargeting for humanoids.

A GPU backend for GMR (github.com/YanjieZe/GMR): the same retargeting problem
(same IK configs, weights, preprocessing -- validated to machine precision),
solved as a batched projected Levenberg-Marquardt iteration in pure torch.
"""
from .kinematics import BatchedRobot, quat_mul, quat_rotate, quat_conjugate, so3_exp, so3_log, se3_twist
from .solver import solve_batched, enorm_b, retract_b, batched_jacobian
from .targets import TargetPipeline, load_frames
from .config import load_ik_tables, weights_from_costs
from .retarget import retarget_clips

__version__ = "0.1.0"
