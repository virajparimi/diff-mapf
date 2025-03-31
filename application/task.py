import quaternion
import numpy as np
from typing import List, Union

import application.pybullet_utils as pu
from application.misc_utils import Robotiq2F85Target
from application.ur5_robotiq_controller import UR5RobotiqPybulletController as UR5


class UR5RobotiqTask:
    def __init__(self, ur5: UR5):
        self.ur5 = ur5

    def set_action(self, action):
        pass

    def start(self):
        pass

    def step(self):
        pass

    def is_done(self):
        return True


class UR5AsyncTaskRunner:
    def __init__(self, ur5: UR5, tasks: Union[List[UR5RobotiqTask], None] = None):
        self.ur5 = ur5
        self.tasks = tasks if tasks is not None else []
        self.current_task_idx = 0

    def current_task(self):
        if self.is_done():
            return self.tasks[-1]
        return self.tasks[self.current_task_idx]

    def step(self):
        if self.is_done():
            return True, None
        self.current_task().step()
        if isinstance(self.current_task(), PolicyTask):
            collision, info = self.ur5.check_collision_with_info(
                excluded_objects=(
                    []
                    if self.ur5.attach_object_id is None
                    else [self.ur5.attach_object_id]
                ),
                self_collision=False,
            )
            if collision:
                return False, (
                    self.current_task().__class__.__name__,
                    (
                        "Unknown"
                        if info is None or info[0] is None
                        else pu.get_body_name(info[0])
                    ),
                    (
                        "Unknown"
                        if info is None or info[1] is None
                        else pu.get_body_name(info[1])
                    ),
                )
        if self.current_task().is_done():
            self.current_task_idx += 1
        return True, None

    def is_done(self):
        return self.current_task_idx == len(self.tasks)


class AttachToGripperTask(UR5RobotiqTask):
    def __init__(self, ur5: UR5, target_id):
        super().__init__(ur5=ur5)
        self.target_id = target_id
        self.attached = False

    def step(self):
        if self.is_done():
            return
        self.ur5.attach_object(self.target_id)
        self.attached = True

    def is_done(self):
        return self.attached


class DetachToGripperTask(UR5RobotiqTask):
    def __init__(self, ur5: UR5):
        super().__init__(ur5=ur5)
        self.detached = False

    def step(self):
        if self.is_done():
            return
        self.ur5.detach()
        self.detached = True

    def is_done(self):
        return self.detached


class ControlArmTask(UR5RobotiqTask):
    def __init__(self, ur5: UR5, target_config, duration=0.5):
        super().__init__(ur5=ur5)
        self.target_config = target_config
        self.duration = duration
        self.current_waypoint_idx = 0
        self.started = False

    def step(self):
        if not self.started:
            self.waypoints = self.ur5.plan_arm_joint_values_simple(
                self.target_config, duration=self.duration
            )
            self.started = True
        self.ur5.control_arm_joints(
            joint_values=self.waypoints[self.current_waypoint_idx]
        )
        self.current_waypoint_idx += 1

    def is_done(self):
        if not self.started:
            return False
        return self.current_waypoint_idx == len(self.waypoints)


class CartesianControlTask(ControlArmTask):
    def __init__(self, ur5: UR5, axis: str, value: float):
        super().__init__(ur5=ur5, target_config=None, duration=0.5)
        self.initialized = False
        self.axis = axis
        self.value = value

    def step(self):
        if not self.initialized:
            target_pose = self.ur5.get_eef_pose()
            if self.axis == "x":
                i = 0
            elif self.axis == "y":
                i = 1
            elif self.axis == "z":
                i = 2
            else:
                raise TypeError("unsupported axis")
            target_pose[0][i] += self.value
            self.target_config = self.ur5.get_arm_ik_pybullet(target_pose)[0]
            self.initialized = True
        super().step()


class PolicyTask(UR5RobotiqTask):
    def __init__(
        self,
        ur5: UR5,
        target_pose,
        position_tolerance=0.05,
        orientation_tolerance=0.2,
        visualize=False,
    ):
        super().__init__(ur5=ur5)
        self.visualize = visualize
        self.target_pose = target_pose
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        if self.visualize:
            self.target_visual = Robotiq2F85Target(pose=self.target_pose)

        self.action = None

    def set_action(self, action):
        self.action = action

    def is_done(self):
        target_position = np.array(self.target_pose[0])
        target_orientation = quaternion.quaternion(*self.target_pose[1])

        ur5_end_eff = self.ur5.get_end_effector_pose()
        eef_position = np.array(ur5_end_eff[0])
        eef_orientation = quaternion.quaternion(*ur5_end_eff[1])

        position_residual = np.linalg.norm(target_position - eef_position)
        orientation_residual = (target_orientation * eef_orientation.inverse()).angle()
        if orientation_residual > np.pi:
            orientation_residual = 2 * np.pi - orientation_residual

        reached_position = position_residual < self.position_tolerance
        reached_orientation = orientation_residual < self.orientation_tolerance
        reached = reached_position and reached_orientation
        if reached and self.visualize:
            del self.target_visual
        return reached

    def step(self):
        assert self.action is not None
        self.ur5.control_arm_joints_delta(self.action)


class GripperTask(UR5RobotiqTask):
    def __init__(self, ur5: UR5, joint_value, duration=0.5):
        super().__init__(ur5=ur5)
        self.target_joint_value = joint_value
        self.duration = duration
        self.current_waypoint_idx = 0
        self.started = False

    def step(self):
        if not self.started:
            self.waypoints = self.ur5.plan_gripper_joint_values(
                self.target_joint_value, duration=self.duration
            )
            self.started = True
        self.ur5.control_gripper_joints(self.waypoints[self.current_waypoint_idx])
        self.current_waypoint_idx += 1

    def is_done(self):
        if not self.started:
            return False
        return self.current_waypoint_idx == len(self.waypoints)


class CloseGripperTask(GripperTask):
    def __init__(self, ur5: UR5):
        super().__init__(ur5=ur5, joint_value=ur5.CLOSED_POSITION)


class OpenGripperTask(GripperTask):
    def __init__(self, ur5: UR5):
        super().__init__(ur5=ur5, joint_value=ur5.OPEN_POSITION)


class SetTargetTask(UR5RobotiqTask):
    def __init__(self, ur5: UR5, target_id, initial_pose):
        super().__init__(ur5=ur5)
        self.target_id = target_id
        self.initial_pose = initial_pose
        self.started = False

    def step(self):
        if not self.started:
            pu.set_pose(self.target_id, self.initial_pose)
        self.started = True

    def is_done(self):
        if not self.started:
            return False
        return True
