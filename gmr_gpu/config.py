"""IK problem definition, loaded from GMR's per-robot JSON configs."""
import json
import numpy as np
import torch
from general_motion_retargeting.params import IK_CONFIG_DICT


def load_ik_tables(robot="unitree_g1", src_human="smplx"):
    """(frame_names, table1_costs (K,2), table2_costs (K,2)) in table order."""
    with open(IK_CONFIG_DICT[src_human][robot]) as f:
        cfg = json.load(f)
    t1, t2 = cfg["ik_match_table1"], cfg["ik_match_table2"]
    assert list(t1.keys()) == list(t2.keys()), "tables must share the frame list"
    frame_names = list(t1.keys())
    c1 = np.array([[t1[k][1], t1[k][2]] for k in frame_names], dtype=np.float64)
    c2 = np.array([[t2[k][1], t2[k][2]] for k in frame_names], dtype=np.float64)
    return frame_names, c1, c2


def weights_from_costs(costs):
    """(K,2) [pos_w, rot_w] -> (6K,) quadratic weights, mink-equivalent.
    mink multiplies the residual by cost (H = J^T diag(cost^2) J), so our W rows
    are cost**2, laid out [pos^2 x3, rot^2 x3] per task (twist order)."""
    costs = torch.as_tensor(costs, dtype=torch.float64)
    return (costs ** 2).repeat_interleave(3, dim=1).reshape(-1)
