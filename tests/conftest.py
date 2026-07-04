import numpy as np
import pytest
import torch
import mujoco as mj
from general_motion_retargeting.params import ROBOT_XML_DICT

XML = str(ROBOT_XML_DICT["unitree_g1"])


@pytest.fixture(scope="session")
def model():
    return mj.MjModel.from_xml_path(XML)


@pytest.fixture(scope="session")
def rob():
    from gmr_gpu import BatchedRobot
    return BatchedRobot(XML, dtype=torch.float64)


def random_qpos(model, rng):
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0, 0, 0.8]
    q = rng.standard_normal(4); qpos[3:7] = q / np.linalg.norm(q)
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
            lo, hi = model.jnt_range[j]
            qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
    return qpos
