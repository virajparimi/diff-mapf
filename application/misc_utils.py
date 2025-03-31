import os
import csv
import sys
import quaternion
import numpy as np
import pybullet as p
import pybullet_data
from math import radians
from os.path import join
from trimesh import load_mesh
import application.grasp_utils as gu
from contextlib import contextmanager
import application.pybullet_utils as pu
from collections import namedtuple, OrderedDict


ConfigInfo = namedtuple(
    "ConfigInfo", ["joint_config", "ee_pose", "pos_distance", "quat_distance"]
)


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def configure_pybullet(
    rendering=False,
    debug=False,
    yaw=46.39,
    pitch=-55.00,
    dist=1.9,
    target=(0.0, 0.0, 0.0),
):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pu.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


# rpy is in degrees
def load_object(object_name, xy_position, surface_height, rpy):
    model_dir = "application/assets/models/"
    rpy = [radians(i) for i in rpy]
    object_mesh_filepath = join(
        model_dir, "{}".format(object_name), "{}.obj".format(object_name)
    )
    target_urdf = join(
        model_dir, "{}".format(object_name), "{}.urdf".format(object_name)
    )
    target_mesh = load_mesh(object_mesh_filepath)
    target_z = -target_mesh.bounds.min(0)[2] + surface_height
    target_initial_pose = [
        [xy_position[0], xy_position[1], target_z],
        pu.quaternion_from_euler(rpy),
    ]
    with suppress_stdout():
        return p.loadURDF(
            target_urdf,
            basePosition=target_initial_pose[0],
            baseOrientation=target_initial_pose[1],
        )


class TargetObject:
    def __init__(self, object_name, xy_position, surface_height, rpy=(0, 0, 0)):
        self.object_name = object_name
        self.id = load_object(object_name, xy_position, surface_height, rpy)
        self.robot_configs = gu.robot_configs["ur5_robotiq"]
        self.back_off = 0.1
        self.initial_pose = self.get_pose()
        self.urdf_path = join(
            "application/assets/models/",
            "{}".format(object_name),
            "{}.urdf".format(object_name),
        )

        grasp_database_path = "application/assets/filtered_grasps"
        actual_grasps, graspit_grasps = gu.load_grasp_database(
            grasp_database_path, self.object_name
        )
        use_actual = False
        self.graspit_grasps = actual_grasps if use_actual else graspit_grasps

        self.graspit_pregrasps = [
            pu.merge_pose_2d(
                gu.back_off(
                    pu.split_7d(g),
                    self.back_off,
                    self.robot_configs.graspit_approach_dir,
                )
            )
            for g in self.graspit_grasps
        ]
        self.grasps_eef = [
            pu.merge_pose_2d(
                gu.change_end_effector_link_pose(
                    pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK
                )
            )
            for g in self.graspit_grasps
        ]
        self.grasps_link6_ref = [
            pu.merge_pose_2d(
                gu.change_end_effector_link_pose(
                    pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK
                )
            )
            for g in self.graspit_grasps
        ]
        self.pre_grasps_eef = [
            pu.merge_pose_2d(
                gu.change_end_effector_link_pose(
                    pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK
                )
            )
            for g in self.graspit_pregrasps
        ]
        self.pre_grasps_link6_ref = [
            pu.merge_pose_2d(
                gu.change_end_effector_link_pose(
                    pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK
                )
            )
            for g in self.graspit_pregrasps
        ]

    def get_pose(self):
        return pu.get_body_pose(self.id)

    def check_lift(self):
        return pu.get_body_pose(self.id)[0][2] > 0.1


class OtherObject:
    def __init__(self, urdf, initial_pose):
        self.urdf = urdf
        self.initial_pose = initial_pose
        with suppress_stdout():
            self.id = p.loadURDF(
                urdf, basePosition=initial_pose[0], baseOrientation=initial_pose[1]
            )

    def get_pose(self):
        return pu.get_body_pose(self.id)


class Robotiq2F85Target:
    def __init__(self, pose=[[0, 0, 0], [0, 0, 0, 1]]):
        with suppress_stdout():
            self.body_id = p.loadURDF(
                "application/assets/gripper/robotiq_2f_85_no_colliders.urdf",
                pose[0],
                pose[1],
                useFixedBase=1,
            )
        self.set_pose(pose)
        for i in range(p.getNumJoints(self.body_id)):
            p.changeVisualShape(
                self.body_id, i, textureUniqueId=-1, rgbaColor=(0, 0, 0, 0.5)
            )

    def transform_orientation(self, orientation):
        A = quaternion.quaternion(*orientation)
        B = quaternion.quaternion(*p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2]))
        C = B * A
        return quaternion.as_float_array(C)

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(
            self.body_id, pose[0], self.transform_orientation(pose[1])
        )

    def __del__(self):
        p.removeBody(self.body_id)


def write_csv_line(result_file_path, result):
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
