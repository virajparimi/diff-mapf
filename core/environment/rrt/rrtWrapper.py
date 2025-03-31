import pybullet as p
from time import sleep
from itertools import chain

from core.environment.rrt.rrt_connect import birrt
from core.environment.rrt.arm_group import ArmGroup
from core.environment.utils import Target, create_arms
from core.environment.rrt.pybullet_utils import (
    draw_line,
    configure_pybullet,
    remove_all_markers,
)


class RRTWrapper:
    def __init__(self, env_config, gui=False):
        self.gui = gui
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        configure_pybullet(debug=False, yaw=0, pitch=0, dist=1.0, target=(0, 0, 0.3))

        plane = p.loadURDF(
            "plane.urdf", [0, 0, -env_config["collision_distance"] - 0.01]
        )
        self.obstacles = [plane]

        def create_arms_fn():
            return create_arms(
                radius=0.8,
                count=env_config["max_arms_count"],
                speed=env_config["arm_speed"],
                arm_type=env_config["arm_type"],
            )

        self.arm_group = ArmGroup(
            create_arms_fn=create_arms_fn,
            collision_distance=env_config["collision_distance"],
        )

        self.targets = [
            Target(pose=[[0, 0, 0], [0, 0, 0, 1]], color=arm.color)
            for arm in self.arm_group.all_controllers
        ]

    def birrt_from_task(self, task):
        return self.birrt(
            arm_poses=task.base_poses,
            goal_config=task.goal_config,
            start_config=task.start_config,
            target_eef_poses=task.target_eef_poses,
        )

    def demo_path(self, path):
        for i in range(len(path)):
            if i != len(path) - 1:
                for pose1, pose2 in zip(
                    self.arm_group.forward_kinematics(path[i]),
                    self.arm_group.forward_kinematics(path[i + 1]),
                ):
                    draw_line(pose1[0], pose2[0], rgb_color=(1, 0, 0), width=6)
        for i, q in enumerate(path):
            self.arm_group.set_joint_positions(q)
            sleep(0.01)

    def birrt(
        self,
        start_config,
        goal_config,
        arm_poses,
        target_eef_poses,
        resolutions=0.1,
        timeout=100000,
    ):
        if self.gui:
            remove_all_markers()
            for pose, target in zip(target_eef_poses, self.targets):
                target.set_pose(pose)

        self.arm_group.setup(arm_poses, start_config)

        collision_fn = self.arm_group.get_collision_fn()
        extend_fn = self.arm_group.get_extend_fn(resolutions=resolutions)

        goal_config = list(chain.from_iterable(goal_config))
        start_config = list(chain.from_iterable(start_config))

        path = birrt(
            start_config=start_config,
            goal_config=goal_config,
            smooth=5,
            group=True,
            greedy=True,
            timeout=timeout,
            iterations=10000,
            visualize=self.gui,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            sample_fn=self.arm_group.sample_fn,
            fk=self.arm_group.forward_kinematics,
            distance_fn=self.arm_group.distance_fn,
        )

        if path is None:
            return None

        if self.gui:
            self.demo_path(path)

        return path
