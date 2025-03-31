import os
import numpy as np
import pybullet as p
from collections import namedtuple

ur5_robotiq_configs = {
    "EEF_LINK_INDEX": 0,
    "OPEN_POSITION": [0] * 6,
    "graspit_approach_dir": "x",
    "PYBULLET_LINK_COM": [0.0, 0.0, 0.031451],
    "robot_urdf": os.path.abspath("assets/NONE"),
    "PYBULLET_LINK_TO_COM": ([0.0, 0.0, 0.031451], [0.0, 0.0, 0.0, 1.0]),
    "CLOSED_POSITION": (0.72 * np.array([1, 1, -1, 1, 1, -1])).tolist(),
    "gripper_urdf": os.path.abspath(
        "assets/robotiq_2f_85_hand/robotiq_arg2f_85_model.urdf"
    ),
    "GRASPIT_LINK_TO_MOVEIT_LINK": (
        [0, 0, 0],
        [0.7071067811865475, 0.0, 0.0, 0.7071067811865476],
    ),
    "GRASPIT_LINK_TO_PYBULLET_LINK": (
        [0.0, 0.0, 0.0],
        [0.0, 0.706825181105366, 0.0, 0.7073882691671998],
    ),
    "GRIPPER_JOINT_NAMES": [
        "finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ],
}

robot_configs = {
    "ur5_robotiq": namedtuple("RobotConfigs", ur5_robotiq_configs.keys())(
        *ur5_robotiq_configs.values()
    ),
}


def back_off(grasp_pose, offset=0.05, approach_dir="z"):
    if approach_dir == "x":
        translation = (-offset, 0, 0)
        rotation = (0, 0, 0, 1)
    if approach_dir == "z":
        translation = (0, 0, -offset)
        rotation = (0, 0, 0, 1)
    pos, quat = p.multiplyTransforms(
        grasp_pose[0], grasp_pose[1], translation, rotation
    )
    pre_grasp_pose = [list(pos), list(quat)]
    return pre_grasp_pose


def change_end_effector_link_pose(grasp_pose, old_link_to_new_link):
    pos, quat = p.multiplyTransforms(
        grasp_pose[0], grasp_pose[1], old_link_to_new_link[0], old_link_to_new_link[1]
    )
    pre_grasp_pose = [list(pos), list(quat)]
    return pre_grasp_pose


def load_grasp_database(grasp_database_path, object_name):
    actual_grasps = np.load(
        os.path.join(grasp_database_path, object_name, "actual_grasps.npy")
    )
    graspit_grasps = np.load(
        os.path.join(grasp_database_path, object_name, "graspit_grasps.npy")
    )
    return actual_grasps, graspit_grasps


def convert_grasp_in_object_to_world(object_pose, grasp_in_object):
    pos, quat = p.multiplyTransforms(
        object_pose[0], object_pose[1], grasp_in_object[0], grasp_in_object[1]
    )
    grasp_in_world = [list(pos), list(quat)]
    return grasp_in_world
