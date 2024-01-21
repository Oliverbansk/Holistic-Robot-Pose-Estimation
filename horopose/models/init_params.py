import json
import numpy as np
from dataset.const import INITIAL_JOINT_ANGLE


def get_regression_init(robot_type,init_pose_from_mean,cam_params=None):
    
    """ Initialize the parameters for the iterative regression network
        robot_type: str --- "panda"/"kuka"/"baxter"
        cam_params: array(float) 4x4 transformation matrix (base to camera)
        init_pose_from_mean: bool --- whether to use mean value to initialize the joints
    """
    pose_params = INITIAL_JOINT_ANGLE
    cam_matrix = np.eye(4,dtype=float)
    if cam_params is not None:
        cam_matrix = cam_params
    input_init_dict = {
        "robot_type" : robot_type,
        "pose_params": pose_params,
        "cam_params": cam_matrix,
        "init_pose_from_mean": init_pose_from_mean
    }
    return input_init_dict