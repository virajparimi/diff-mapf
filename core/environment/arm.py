import random
import quaternion
import numpy as np
import pybullet as p
from time import sleep
from threading import Thread


from core.environment.rrt.pybullet_utils import (
    get_link_pose,
    get_sample_fn,
    get_extend_fn,
    control_joints,
    get_distance_fn,
    violates_limits,
    get_difference_fn,
    inverse_kinematics,
    forward_kinematics,
    get_joint_positions,
    get_self_link_pairs,
    set_joint_positions,
)


class HemisphereWorkspace:
    def __init__(self, radius, origin):
        self.radius = radius
        self.origin = np.array(origin)
        random.seed()

    def point_in_workspace(self, i=None, j=None, k=None):
        if i is None or j is None or k is None:
            i = random.uniform(0, 1)
            j = random.uniform(0, 1)
            k = random.uniform(0, 1)
        j = j**0.5
        output = np.array(
            [
                self.radius * j * np.cos(2 * np.pi * i) * np.cos(k * np.pi / 2.0),
                self.radius * j * np.sin(2 * np.pi * i) * np.cos(k * np.pi / 2.0),
                self.radius * j * np.sin(k * np.pi / 2.0),
            ]
        )
        output += self.origin
        assert np.linalg.norm(output - self.origin) < self.radius
        return output


class Robotiq2F85:
    NORMAL = 0
    TOUCHED = 1

    def __init__(self, arm, color, replace_textures=True):
        self.arm = arm
        self.color = color
        self.urdf = "assets/gripper/robotiq_2f_85_no_colliders.urdf"
        pose = arm.get_end_effector_pose()
        self.replace_textures = replace_textures

        self.body_id = p.loadURDF(
            self.urdf,
            pose[0],
            pose[1],
        )

        self.tool_joint_index = 7
        self.tool_offset = [0, 0, 0.02]
        self.gripper_offset_orn = p.getQuaternionFromEuler([0, -np.pi / 2.0, 0.0])
        self.reverse_gripper_offset_orn = p.getQuaternionFromEuler(
            [0, np.pi / 2.0, 0]
        )
        self.tool_constraint = p.createConstraint(
            arm.body_id,
            self.tool_joint_index,
            self.body_id,
            0,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.tool_offset,
            childFrameOrientation=self.gripper_offset_orn,
        )

        self.setup()

    def setup(self):
        for i in range(p.getNumJoints(self.body_id)):
            p.changeDynamics(
                self.body_id,
                i,
                lateralFriction=1.0,
                spinningFriction=1.0,
                rollingFriction=0.0001,
                frictionAnchor=True,
            )
            if self.replace_textures:
                p.changeVisualShape(
                    self.body_id,
                    i,
                    textureUniqueId=-1,
                    rgbaColor=(0, 0, 0, 0.5),
                )

        p.changeVisualShape(
            self.body_id,
            0,
            textureUniqueId=-1,
            rgbaColor=(self.color[0], self.color[1], self.color[2], 0.5),
        )

        self._mode = Robotiq2F85.NORMAL
        self.normal()

        self.joints = [
            p.getJointInfo(self.body_id, i) for i in range(p.getNumJoints(self.body_id))
        ]
        self.joints = [
            joint_info[0]
            for joint_info in self.joints
            if joint_info[2] == p.JOINT_REVOLUTE
        ]
        p.setJointMotorControlArray(
            self.body_id,
            self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.0] * len(self.joints),
            positionGains=[1.0] * len(self.joints),
        )

        self.open()

        self.constraints_thread = Thread(target=self.step_daemon_fn)
        self.constraints_thread.daemon = True
        self.constraints_thread.start()

    def normal(self):
        if self._mode != Robotiq2F85.NORMAL:
            p.changeVisualShape(
                self.body_id,
                0,
                textureUniqueId=-1,
                rgbaColor=(self.color[0], self.color[1], self.color[2], 0.5),
            )
            self._mode = Robotiq2F85.NORMAL

    def open(self):
        p.setJointMotorControl2(
            self.body_id,
            1,
            p.VELOCITY_CONTROL,
            targetVelocity=-5.0,
            force=10000.0,
        )

    def step(self):
        gripper_joint_positions = p.getJointState(self.body_id, 1)[0]
        p.setJointMotorControlArray(
            self.body_id,
            [6, 3, 8, 5, 10],
            p.POSITION_CONTROL,
            [
                gripper_joint_positions,
                -gripper_joint_positions,
                -gripper_joint_positions,
                gripper_joint_positions,
                gripper_joint_positions,
            ],
            positionGains=np.ones(5),
        )

    def step_daemon_fn(self):
        while True:
            self.step()
            sleep(0.01)

    def transform_orientation(self, orientation):
        A = quaternion.quaternion(*orientation)
        B = quaternion.quaternion(*p.getQuaternionFromEuler([0, np.pi / 2.0, 0]))
        C = B * A
        return quaternion.as_float_array(C)

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(
            self.body_id, pose[0], self.transform_orientation(pose[1])
        )

    def update_eef_pose(self):
        self.set_pose(self.arm.get_end_effector_pose())

    def touched(self):
        if self._mode != Robotiq2F85.TOUCHED:
            p.changeVisualShape(
                self.body_id,
                0,
                textureUniqueId=-1,
                rgbaColor=(0.4, 1.0, 0.4, 0.8),
            )
            self._mode = Robotiq2F85.TOUCHED


class Robotiq2F85Target(Robotiq2F85):
    def __init__(self, pose, color):
        self.color = color
        self.replace_textures = True
        self.body_id = p.loadURDF(
            "assets/gripper/robotiq_2f_85_no_colliders.urdf",
            pose[0],
            pose[1],
            useFixedBase=True,
        )
        self.set_pose(pose)
        self.setup()


class UR5:
    LINK_COUNT = 10
    JOINTS_COUNT = 6
    JOINT_EPSILON = 0.01
    WORKSPACE_RADIUS = 0.85

    NEXT_AVAILABLE_COLOR = 0
    COLORS = [
        (230, 25, 75),  # red
        (60, 180, 75),  # green
        (255, 225, 25),  # yellow
        (0, 130, 200),  # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190),  # pink
        (0, 128, 128),  # teal
        (230, 190, 255),  # lavender
        (170, 110, 40),  # brown
        (255, 250, 200),  # beige
        (128, 0, 0),  # maroon
        (170, 255, 195),  # lavender
        (128, 128, 0),  # olive
        (255, 215, 180),  # apricot
        (0, 0, 128),  # navy
        (128, 128, 128),  # grey
        (0, 0, 0),  # white
        (255, 255, 255),  # black,
    ]
    COLORS = np.array(COLORS) / 255.0

    GROUPS = {
        "arm": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "gripper": None,
    }

    GROUP_INDEX = {"arm": [1, 2, 3, 4, 5, 6], "gripper": None}

    UPPER_LIMITS = [2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]
    LOWER_LIMITS = [-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi]

    MAX_VELOCITY = [3.15, 3.15, 3.15, 3.2, 3.2, 3.2]
    MAX_FORCE = [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]

    EEF_LINK_INDEX = 7
    HOME = [0, 0, 0, 0, 0, 0]
    RESET = [0, -1, 1, 0.5, 1, 0]
    UP = [0, -np.pi / 2, 0, -np.pi / 2.0, 0, 0]

    def __init__(
        self,
        pose,
        home_config=None,
        velocity=1.0,
        enabled=True,
        acceleration=2.0,
        training=True,
    ):
        self.pose = pose
        self.enabled = enabled
        self.velocity = velocity
        self.target_joint_values = None
        self.acceleration = acceleration
        self.color = self.COLORS[UR5.NEXT_AVAILABLE_COLOR]
        UR5.NEXT_AVAILABLE_COLOR = (UR5.NEXT_AVAILABLE_COLOR + 1) % len(self.COLORS)

        self.workspace = HemisphereWorkspace(
            radius=self.WORKSPACE_RADIUS, origin=self.pose[0]
        )

        if not hasattr(self, "arm_urdf"):
            self.arm_urdf = (
                "assets/ur5/ur5_training.urdf" if training else "assets/ur5/ur5.urdf"
            )

        if training:
            self.end_effector = None
            self.body_id = p.loadURDF(
                self.arm_urdf,
                self.pose[0],
                self.pose[1],
                flags=p.URDF_USE_SELF_COLLISION,
            )
            p.changeVisualShape(
                self.body_id,
                self.EEF_LINK_INDEX,
                textureUniqueId=-1,
                rgbaColor=(self.color[0], self.color[1], self.color[2], 0.5),
            )
        else:
            self.body_id = p.loadURDF(
                self.arm_urdf,
                self.pose[0],
                self.pose[1],
                flags=p.URDF_USE_SELF_COLLISION,
            )
            self.end_effector = Robotiq2F85(arm=self, color=self.color)

        robot_joint_info = [
            p.getJointInfo(self.body_id, i) for i in range(p.getNumJoints(self.body_id))
        ]

        self._robot_joint_indices = [
            joint_info[0]
            for joint_info in robot_joint_info
            if joint_info[2] == p.JOINT_REVOLUTE
        ]

        self._robot_joint_lower_limits = [
            joint_info[8]
            for joint_info in robot_joint_info
            if joint_info[2] == p.JOINT_REVOLUTE
        ]
        self._robot_joint_upper_limits = [
            joint_info[9]
            for joint_info in robot_joint_info
            if joint_info[2] == p.JOINT_REVOLUTE
        ]

        self.home_config = (
            [-np.pi, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, 0]
            if home_config is None
            else home_config
        )

        self.arm_extend_fn = get_extend_fn(self.body_id, self.GROUP_INDEX["arm"])
        self.arm_sample_fn = get_sample_fn(self.body_id, self.GROUP_INDEX["arm"])
        self.link_pairs = get_self_link_pairs(self.body_id, self.GROUP_INDEX["arm"])
        self.arm_distance_fn = get_distance_fn(self.body_id, self.GROUP_INDEX["arm"])
        self.arm_difference_fn = get_difference_fn(
            self.body_id, self.GROUP_INDEX["arm"]
        )

        self.max_distance_from_others = 0.5
        self.closest_points_to_self = []
        self.closest_points_to_others = []

    def set_pose(self, pose):
        self.pose = pose
        self.workspace.origin = self.pose[0]
        p.resetBasePositionAndOrientation(self.body_id, pose[0], pose[1])
        if self.end_effector is not None:
            self.end_effector.update_eef_pose()

    def get_pose(self):
        return p.getBasePositionAndOrientation(self.body_id)

    def control_arm_joints(self, joint_values, velocity=None):
        velocity = self.velocity if velocity is None else velocity
        self.target_joint_values = joint_values
        control_joints(
            self.body_id,
            self.GROUP_INDEX["arm"],
            self.target_joint_values,
            velocity,
            self.acceleration,
        )

    def control_arm_joints_delta(self, delta_joint_values, velocity=None):
        self.control_arm_joints(
            self.get_arm_joint_values() + delta_joint_values, velocity=velocity
        )

    def set_arm_joints(self, joint_values):
        set_joint_positions(self.body_id, self.GROUP_INDEX["arm"], joint_values)
        self.control_arm_joints(joint_values)
        if self.end_effector is not None:
            self.end_effector.update_eef_pose()

    def reset(self):
        self.set_arm_joints(self.home_config)

    def get_arm_joint_values(self):
        return np.array(get_joint_positions(self.body_id, self.GROUP_INDEX["arm"]))

    def calc_next_subtarget_joints(self):
        current_joints = self.get_arm_joint_values()
        if not isinstance(self.target_joint_values, np.ndarray):
            self.target_joint_values = np.array(self.target_joint_values)
        delta_time = 1.0 / 240.0
        delta_joint = delta_time * self.velocity
        subtarget_joint_values = self.target_joint_values - current_joints
        max_joint = max(abs(subtarget_joint_values))
        if max_joint < delta_joint:
            return subtarget_joint_values
        subtarget_joint_values = subtarget_joint_values * delta_joint / max_joint
        return subtarget_joint_values + current_joints

    def step(self):
        if self.end_effector is not None:
            self.end_effector.step()

    def disable(self, idx=0):
        self.enabled = False
        self.set_pose([[idx, 20, 0], [0.0, 0.0, 0.0, 1.0]])
        self.reset()
        self.step()

    def enable(self):
        self.enabled = True

    def get_end_effector_pose(self, link=None):
        link = link if link is not None else self.EEF_LINK_INDEX
        return get_link_pose(self.body_id, link)

    def update_closest_points(self):
        others_id = [
            p.getBodyUniqueId(i)
            for i in range(p.getNumBodies())
            if p.getBodyUniqueId(i) != self.body_id
        ]
        self.closest_points_to_others = [
            (
                sorted(
                    list(
                        p.getClosestPoints(
                            bodyA=self.body_id,
                            bodyB=other_id,
                            distance=self.max_distance_from_others,
                        )
                    ),
                    key=lambda contact_points: contact_points[8],
                )
                if other_id != 0
                else []
            )
            for other_id in others_id
        ]
        self.closest_points_to_self = [
            p.getClosestPoints(
                bodyA=self.body_id,
                bodyB=self.body_id,
                distance=0,
                linkIndexA=link1,
                linkIndexB=link2,
            )
            for link1, link2 in self.link_pairs
        ]

    def check_collision(self, collision_distance=0.0):
        self.update_closest_points()
        for i, closest_points_to_other in enumerate(self.closest_points_to_others):
            if i == 0:
                for point in p.getClosestPoints(
                    bodyA=self.body_id, bodyB=0, distance=0.0
                ):
                    if point[8] < collision_distance:
                        self.collision_point = point
                        return True
            else:
                for point in closest_points_to_other:
                    if point[8] < collision_distance:
                        self.collision_point = point
                        return True
        for closest_points_to_self_link in self.closest_points_to_self:
            for point in closest_points_to_self_link:
                if len(point) > 0:
                    self.collision_point = point
                    return True

        self.collision_point = None  # Used for visualization only!
        return False

    def check_collision_with_info(self, collision_distance=0.0):
        self.update_closest_points()
        for i, closest_points_to_other in enumerate(self.closest_points_to_others):
            if i == 0:
                for point in p.getClosestPoints(
                    bodyA=self.body_id, bodyB=0, distance=0.0
                ):
                    if point[8] < collision_distance:
                        self.collision_point = point
                        return True, (point[1], point[2], point[3], point[4])
            else:
                for point in closest_points_to_other:
                    if point[8] < collision_distance:
                        self.collision_point = point
                        return True, (point[1], point[2], point[3], point[4])
        for closest_points_to_self_link in self.closest_points_to_self:
            for point in closest_points_to_self_link:
                if len(point) > 0:
                    self.collision_point = point
                    return True, (point[1], point[2], point[3], point[4])

        self.collision_point = None  # Used for visualization only!
        return False, None

    def violates_limits(self):
        return violates_limits(
            self.body_id, self.GROUP_INDEX["arm"], self.get_arm_joint_values()
        )

    def global_to_arm_frame(self, position, rotation=None):
        self_position, self_rotation = p.getBasePositionAndOrientation(self.body_id)
        invert_self_position, invert_self_rotation = p.invertTransform(
            self_position, self_rotation
        )
        arm_frame_position, arm_frame_rotation = p.multiplyTransforms(
            invert_self_position,
            invert_self_rotation,
            position,
            invert_self_rotation if rotation is None else rotation,
        )
        return arm_frame_position, arm_frame_rotation

    def get_link_global_positions(self):
        linkStates = [
            p.getLinkState(self.body_id, link_id, computeForwardKinematics=True)
            for link_id in range(self.LINK_COUNT)
        ]
        link_world_positions = [world_position for world_position, *_ in linkStates]
        return link_world_positions

    def on_untouch_target(self):
        if self.end_effector is not None:
            self.end_effector.normal()

    def on_touch_target(self):
        if self.end_effector is not None:
            self.end_effector.touched()

    def forward_kinematics(self, joint_values):
        return forward_kinematics(
            self.body_id, self.GROUP_INDEX["arm"], joint_values, self.EEF_LINK_INDEX
        )

    def inverse_kinematics(self, position, orientation=None):
        return inverse_kinematics(
            self.body_id,
            self.EEF_LINK_INDEX,
            position,
            orientation,
        )

    def set_target_end_effector_position(self, position):
        self.set_arm_joints(self.inverse_kinematics(position=position))
