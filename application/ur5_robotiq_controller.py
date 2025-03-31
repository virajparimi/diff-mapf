import time
import numpy as np
import pyquaternion
import pybullet as p
from collections import namedtuple

import application.grasp_utils as gu
import application.pybullet_utils as pu
from application.misc_utils import suppress_stdout
from core.environment.rrt.rrt_connect import birrt

Grasp = namedtuple(
    "Grasp",
    [
        "grasp_pose",
        "grasp_jv",
        "pre_grasp_pose",
        "pre_grasp_jv",
        "pos_distance",
        "quat_distance",
        "pre_pos_distance",
        "pre_quat_distance",
    ],
)


class UR5RobotiqPybulletController(object):
    GROUPS = {
        "arm": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "gripper": [
            "finger_joint",
            "left_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "right_inner_finger_joint",
        ],
    }

    GROUP_INDEX = {"arm": [1, 2, 3, 4, 5, 6], "gripper": [9, 11, 13, 14, 16, 18]}

    LINK_COUNT = 10

    LIE = [0, 0, 0, 0, 0, 0]
    UP = [0, -1.5707, 0, -1.5707, 0, 0]
    HOME = [0, -0.8227210029571718, -0.130, -0.660, 0, 1.62]
    RESET = [-np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.4 * np.array([1, 1, -1, 1, 1, -1])

    ARM = "arm"
    GRIPPER = "gripper"
    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    EE_LINK_NAME = "ee_link"

    NEXT_AVAILABLE_COLOR = 0
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    def __init__(
        self,
        urdf="application/assets/ur5/ur5_robotiq.urdf",
        base_pose=((0, 0, 0), (0, 0, 0, 1)),
        initial_arm_joint_values=None,
    ):
        self.urdf = urdf
        self.base_pose = base_pose
        self.initial_arm_joint_values = (
            initial_arm_joint_values if initial_arm_joint_values else self.RESET
        )

        with suppress_stdout():
            self.id = p.loadURDF(
                urdf,
                basePosition=base_pose[0],
                baseOrientation=base_pose[1],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
            )

        joint_infos = [
            p.getJointInfo(self.id, joint_index)
            for joint_index in range(p.getNumJoints(self.id))
        ]
        self.JOINT_INDICES_DICT = {entry[1]: entry[0] for entry in joint_infos}
        self.EEF_LINK_INDEX = pu.link_from_name(self.id, self.EE_LINK_NAME)

        self.arm_distance_fn = pu.get_distance_fn(self.id, self.GROUP_INDEX["arm"])
        self.arm_sample_fn = pu.get_sample_fn(self.id, self.GROUP_INDEX["arm"])
        self.arm_extend_fn = pu.get_extend_fn(self.id, self.GROUP_INDEX["arm"])
        self.link_pairs = pu.get_self_link_pairs(self.id, self.GROUP_INDEX["arm"])
        self.arm_difference_fn = pu.get_difference_fn(self.id, self.GROUP_INDEX["arm"])

        self.attach_cid = None
        self.attach_object_id = None
        self.max_distance_from_others = 0.5
        self.closest_points_to_self = []
        self.closest_points_to_others = []

        self.color = pu.rgb_colors_1[UR5RobotiqPybulletController.NEXT_AVAILABLE_COLOR]
        UR5RobotiqPybulletController.NEXT_AVAILABLE_COLOR = (
            UR5RobotiqPybulletController.NEXT_AVAILABLE_COLOR + 1
        ) % len(pu.rgb_colors_1)

        self.reset()

    def reset(self):
        self.set_arm_joints(self.initial_arm_joint_values)
        self.set_gripper_joints(self.OPEN_POSITION)
        if self.attach_cid is not None:
            p.removeConstraint(self.attach_cid)
        self.attach_cid = None
        self.attach_object_id = None

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX["arm"], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX["arm"], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(
            body=self.id, joints=self.GROUP_INDEX["arm"], positions=joint_values
        )

    def control_arm_joints_delta(self, delta_joint_values):
        self.control_arm_joints(self.get_arm_joint_values() + delta_joint_values)

    def set_gripper_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX["gripper"], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX["gripper"], joint_values)

    def control_gripper_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX["gripper"], joint_values)

    def close_gripper(self, realtime=False, duration=2):
        waypoints = self.plan_gripper_joint_values(
            self.CLOSED_POSITION, duration=duration
        )
        self.execute_gripper_plan(waypoints, realtime)

    def open_gripper(self, realtime=False, duration=2):
        waypoints = self.plan_gripper_joint_values(
            self.OPEN_POSITION, duration=duration
        )
        self.execute_gripper_plan(waypoints, realtime)

    def plan_gripper_joint_values(
        self, goal_joint_values, start_joint_values=None, duration=1.0
    ):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = int(duration * 240)
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        discretized_plan = np.vstack(
            (discretized_plan, np.repeat(discretized_plan[-1, None], 240, axis=0))
        )
        return discretized_plan

    def get_arm_fk_pybullet(self, joint_values):
        return pu.forward_kinematics(
            self.id, self.GROUP_INDEX["arm"], joint_values, self.EEF_LINK_INDEX
        )

    def get_end_effector_pose(self):
        return self.get_arm_fk_pybullet(self.get_arm_joint_values())

    def get_arm_ik_pybullet(
        self, pose_2d, arm_joint_values=None, gripper_joint_values=None
    ):
        joint_values = p.calculateInverseKinematics(
            self.id,
            self.EEF_LINK_INDEX,
            pose_2d[0],
            pose_2d[1],
            maxNumIterations=10000,
            residualThreshold=0.0001,
        )
        ik_result = list(joint_values[:6])
        actual_pose = self.get_arm_fk_pybullet(ik_result)
        pos_distance = np.linalg.norm(np.array(actual_pose[0]) - np.array(pose_2d[0]))
        quat_distance = pyquaternion.Quaternion.absolute_distance(
            pyquaternion.Quaternion(pu.change_quat_rep(pose_2d[1])),
            pyquaternion.Quaternion(pu.change_quat_rep(actual_pose[1])),
        )
        return ik_result, pos_distance, quat_distance

    def plan_arm_joint_values_simple(
        self, goal_joint_values, start_joint_values=None, duration=None
    ):
        """Linear interpolation between joint_values"""
        start_joint_values = (
            self.get_arm_joint_values()
            if start_joint_values is None
            else start_joint_values
        )

        diffs = self.arm_difference_fn(goal_joint_values, start_joint_values)
        diffs[-1] = (diffs[-1] + np.pi / 2) % np.pi - np.pi / 2
        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240
        num_steps = int(max(steps))
        if duration is not None:
            num_steps = int(duration * 240)

        goal_joint_values = np.array(start_joint_values) + np.array(diffs)
        waypoints = np.linspace(start_joint_values, goal_joint_values, num_steps)

        waypoints = np.vstack((waypoints, np.repeat(waypoints[-1, None], 240, axis=0)))
        return waypoints

    @staticmethod
    def convert_range(joint_values):
        circular_idx = [0, 3, 4, 5]
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if i in circular_idx:
                new_joint_values.append(pu.wrap_angle(v))
            else:
                new_joint_values.append(v)
        return new_joint_values

    def execute_arm_plan(self, plan, realtime=False):
        """
        Execute a discretized arm plan (list of waypoints)
        """
        for wp in plan:
            self.control_arm_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1.0 / 240.0)
        pu.step(1)

    def execute_gripper_plan(self, plan, realtime=False):
        """
        Execute a discretized gripper plan (list of waypoints)
        """
        for wp in plan:
            self.control_gripper_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1.0 / 240.0)
        pu.step(1)

    def equal_conf(self, conf1, conf2, tol=0):
        adapted_conf2 = self.adapt_conf(conf2, conf1)
        return np.allclose(conf1, adapted_conf2, atol=tol)

    def adapt_conf(self, conf2, conf1):
        diff = self.arm_difference_fn(conf2, conf1)
        adapted_conf2 = np.array(conf1) + np.array(diff)
        return adapted_conf2.tolist()

    def get_arm_joint_values(self):
        return pu.get_joint_positions(self.id, self.GROUP_INDEX["arm"])

    def get_gripper_joint_values(self):
        return pu.get_joint_positions(self.id, self.GROUP_INDEX["gripper"])

    def get_eef_pose(self):
        return pu.get_link_pose(self.id, self.EEF_LINK_INDEX)

    def update_closest_points(self, excluded_objects=[]):
        others_id = [
            p.getBodyUniqueId(i)
            for i in range(p.getNumBodies())
            if p.getBodyUniqueId(i) != self.id
        ]
        for i in excluded_objects:
            others_id.remove(i)
        self.closest_points_to_others = [
            sorted(
                list(
                    p.getClosestPoints(
                        bodyA=self.id,
                        bodyB=other_id,
                        distance=self.max_distance_from_others,
                    )
                ),
                key=lambda contact_points: contact_points[8],
            )
            for other_id in others_id
        ]
        self.closest_points_to_self = [
            p.getClosestPoints(
                bodyA=self.id,
                bodyB=self.id,
                distance=0,
                linkIndexA=link1,
                linkIndexB=link2,
            )
            for link1, link2 in self.link_pairs
        ]

    def check_collision_with_info(
        self,
        conf=None,
        collision_distance=0.0,
        excluded_objects=[],
        self_collision=True,
    ):
        if conf is not None:
            self.set_arm_joints(conf)
        self.update_closest_points(excluded_objects)
        for _, closest_points_to_other in enumerate(self.closest_points_to_others):
            for point in closest_points_to_other:
                if point[8] < collision_distance:
                    return True, (point[1], point[2], point[3], point[4])
        if self_collision:
            for closest_points_to_self_link in self.closest_points_to_self:
                if len(closest_points_to_self_link) > 0:
                    point = closest_points_to_self_link[0]
                    return True, (point[1], point[2], point[3], point[4])
        return False, None

    def plan_grasp(self, target):
        current_jv = self.get_arm_joint_values()
        grasps = []
        for g, p_g in zip(target.grasps_eef, target.pre_grasps_eef):
            g_pose = pu.split_7d(g)
            pre_g_pose = pu.split_7d(p_g)
            g_pose = gu.convert_grasp_in_object_to_world(target.get_pose(), g_pose)
            pre_g_pose = gu.convert_grasp_in_object_to_world(
                target.get_pose(), pre_g_pose
            )
            jv, pos_distance, quat_distance = self.get_arm_ik_pybullet(g_pose)
            pre_jv, pre_pos_distance, pre_quat_distance = self.get_arm_ik_pybullet(
                pre_g_pose
            )
            self.set_arm_joints(jv)
            grasp_collision, _ = self.check_collision_with_info(excluded_objects=[target.id])
            self.set_arm_joints(pre_jv)
            pre_grasp_collision, _ = self.check_collision_with_info()
            collision = pre_grasp_collision or grasp_collision
            if (
                not collision
                and pos_distance < 1e-3
                and quat_distance < 1e-3
                and pre_pos_distance < 1e-3
                and pre_quat_distance < 1e-3
            ):
                grasps.append(
                    Grasp(
                        grasp_pose=g_pose,
                        grasp_jv=jv,
                        pre_grasp_pose=pre_g_pose,
                        pre_grasp_jv=pre_jv,
                        pos_distance=pos_distance,
                        quat_distance=quat_distance,
                        pre_pos_distance=pre_pos_distance,
                        pre_quat_distance=pre_quat_distance,
                    )
                )
        self.set_arm_joints(current_jv)
        return grasps

    def cartesian_control(
        self,
        axis,
        simple=True,
        value=0.1,
        realtime=False,
        excluded_objects=[],
        duration=2,
    ):
        target_pose = self.get_eef_pose()
        if axis == "x":
            i = 0
        elif axis == "y":
            i = 1
        elif axis == "z":
            i = 2
        else:
            raise TypeError("unsupported axis")
        target_pose[0][i] += value
        target_jv = self.get_arm_ik_pybullet(target_pose)[0]
        plan = (
            self.plan_arm_joint_values_simple(target_jv, duration=duration)
            if simple
            else self.plan_arm_joint_values(
                target_jv, excluded_objects=excluded_objects
            )
        )
        self.execute_arm_plan(plan, realtime)
        return plan

    def move_arm_to(
        self, target_jv, simple=True, realtime=False, excluded_objects=[], duration=2
    ):
        if simple:
            plan = self.plan_arm_joint_values_simple(target_jv, duration=duration)
        else:
            plan = self.plan_arm_joint_values(target_jv, excluded_objects)
        self.execute_arm_plan(plan, realtime=realtime)
        return plan

    def plan_arm_joint_values(
        self,
        goal_conf,
        excluded_objects=[],
        start_conf=None,
        planner="birrt",
        smooth=200,
        greedy=True,
        iterations=2000,
        restarts=10,
    ):
        current_conf = self.get_arm_joint_values()
        start_conf = current_conf if start_conf is None else start_conf

        def collision_fn(conf):
            return self.check_collision_with_info(conf, excluded_objects=excluded_objects)[0]

        extend_fn = self.arm_extend_fn

        if planner == "birrt":
            for i in range(restarts):
                iter_start = time.time()
                path_conf = birrt(
                    start_config=start_conf,
                    goal_config=goal_conf,
                    distance_fn=self.arm_difference_fn,
                    sample_fn=self.arm_sample_fn,
                    extend_fn=extend_fn,
                    collision_fn=collision_fn,
                    iterations=iterations,
                    smooth=smooth,
                    visualize=True,
                    fk=self.get_arm_fk_pybullet,
                    group=False,
                    greedy=greedy,
                )
                iter_time = time.time() - iter_start
                self.set_arm_joints(current_conf)
                if path_conf is None:
                    print(
                        "Trial {} ({} iterations) fails in {:.2f} seconds".format(
                            i + 1, iterations, iter_time
                        )
                    )
                    pu.remove_all_markers()
                else:
                    return path_conf
        else:
            raise ValueError("Planner must be 'birrt'")

    def attach_object(self, target_id):
        target_pose = pu.get_body_pose(target_id)
        eef_pose = self.get_eef_pose()
        eef_P_world = p.invertTransform(eef_pose[0], eef_pose[1])
        eef_P_target = p.multiplyTransforms(
            eef_P_world[0], eef_P_world[1], target_pose[0], target_pose[1]
        )
        self.attach_cid = p.createConstraint(
            parentBodyUniqueId=target_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.id,
            childLinkIndex=self.EEF_LINK_INDEX,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=eef_P_target[0],
            childFrameOrientation=eef_P_target[1],
        )
        self.attach_object_id = target_id

    def detach(self):
        p.removeConstraint(self.attach_cid)
        self.attach_cid = None
        self.attach_object_id = None

    def global_to_ur5_frame(self, position, rotation=None):
        self_pos, self_rot = p.getBasePositionAndOrientation(self.id)
        invert_self_pos, invert_self_rot = p.invertTransform(self_pos, self_rot)
        ur5_frame_pos, ur5_frame_rot = p.multiplyTransforms(
            invert_self_pos,
            invert_self_rot,
            position,
            invert_self_rot if rotation is None else rotation,
        )
        return ur5_frame_pos, ur5_frame_rot

    def get_link_global_positions(self):
        linkstates = [
            p.getLinkState(self.id, link_id, computeForwardKinematics=True)
            for link_id in range(UR5RobotiqPybulletController.LINK_COUNT - 2)
        ]
        linkstates.append(p.getLinkState(self.id, p.getNumJoints(self.id) - 2, computeForwardKinematics=True))
        linkstates.append(p.getLinkState(self.id, p.getNumJoints(self.id) - 1, computeForwardKinematics=True))
        link_world_positions = [world_pos for world_pos, _, _, _, _, _, in linkstates]
        return link_world_positions

    def get_pose(self):
        return p.getBasePositionAndOrientation(self.id)
