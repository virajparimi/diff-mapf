import numpy as np
import pybullet as p


def split(arr, n):
    k, m = divmod(len(arr), n)
    return (
        arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)  # noqa
    )


class ArmGroup:
    def __init__(self, create_arms_fn, collision_distance):
        self.active_controllers = []
        self.all_controllers = create_arms_fn()
        self.collision_distance = collision_distance

    def disable_all_arms(self):
        for arm_index, arm in enumerate(self.all_controllers):
            arm.disable(idx=arm_index)
        self.active_controllers = []

    def enable_arms(self, count=None):
        self.disable_all_arms()
        for arm_index, arm in enumerate(self.all_controllers):
            if count is not None and arm_index >= count:
                break
            arm.enable()
            self.active_controllers.append(arm)

    def setup(self, start_poses, start_joints):
        self.disable_all_arms()
        self.enable_arms(count=len(start_poses))
        for controller, pose, joints in zip(
            self.active_controllers, start_poses, start_joints
        ):
            controller.set_pose(pose)
            controller.set_arm_joints(joints)
        p.stepSimulation()
        return None

    def compute_dof(self):
        return sum(
            [
                len(controller.GROUP_INDEX["arm"])
                for controller in self.active_controllers
            ]
        )

    def set_joint_positions(self, joint_values):
        assert len(joint_values) == self.compute_dof()
        robot_joint_values = split(joint_values, len(self.active_controllers))
        for controller, joint_value in zip(self.active_controllers, robot_joint_values):
            controller.set_arm_joints(joint_value)

    def get_collision_fn(self):
        def collision_fn(q=None):
            if q is not None:
                self.set_joint_positions(q)
            return any(
                [
                    controller.check_collision(
                        collision_distance=self.collision_distance
                    )
                    for controller in self.active_controllers
                ]
            )

        return collision_fn

    def difference_fn(self, q2, q1):
        difference = []
        split_q1 = split(q1, len(self.active_controllers))
        split_q2 = split(q2, len(self.active_controllers))
        for controller, q1_, q2_ in zip(self.active_controllers, split_q1, split_q2):
            difference += controller.arm_difference_fn(q2_, q1_)
        return difference

    def get_extend_fn(self, resolutions=None):
        dof = self.compute_dof()
        if resolutions is None:
            resolutions = 0.05 * np.ones(dof)

        def fn(q1, q2):
            diffs = self.difference_fn(q2, q1)
            steps = np.abs(np.divide(diffs, resolutions))
            num_steps = int(max(steps))
            waypoints = []
            for i in range(num_steps):
                waypoints.append(
                    list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + q1)
                )
            return waypoints

        return fn

    def distance_fn(self, q1, q2):
        diff = np.array(self.difference_fn(q2, q1))
        return np.sqrt(np.dot(diff, diff))

    def sample_fn(self):
        values = []
        for controller in self.active_controllers:
            values += controller.arm_sample_fn()
        return values

    def forward_kinematics(self, q):
        poses = []
        split_q = split(q, len(self.active_controllers))
        for controller, q_ in zip(self.active_controllers, split_q):
            poses.append(controller.forward_kinematics(q_))
        return poses
