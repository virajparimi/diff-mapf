from time import sleep

from core.environment.utils import create_arms
from core.environment.multiArmEnv import MultiArmEnv


class BenchmarkMultiArmEnv(MultiArmEnv):
    EPISODE_CLOCK_TIME_LENGTH = 20

    def __init__(self, env_config, training_config, gui, logger):
        env_config["arm_speed"] = 0.008
        env_config["episode_length"] = int(
            BenchmarkMultiArmEnv.EPISODE_CLOCK_TIME_LENGTH
            / env_config["action_interval"]
        )

        print("[BenchmarkMultiArmEnv] Configuration: ")
        print("\tEpisode Length: ", env_config["episode_length"])
        print("\tAction Interval: ", env_config["action_interval"])
        print("\tSimulation Time: ", BenchmarkMultiArmEnv.EPISODE_CLOCK_TIME_LENGTH)

        super().__init__(env_config, training_config, gui, logger)

        self.reset_score()
        self.position_tolerance = 0.02
        self.orientation_tolerance = 0.1
        self.stop_arm_after_reach = False
        self.terminate_on_collision = True
        self.terminate_on_collectively_reach_target = True

        self.min_task_difficulty = env_config["min_task_difficulty"]
        self.max_task_difficulty = env_config["max_task_difficulty"]

        print("\tPosition Tolerance: ", self.position_tolerance)
        print("\tOrientation Tolerance: ", self.orientation_tolerance)

    def reset_score(self):
        self.current_episode_score = {"time": 0.0}

    def on_reset(self):
        super().on_reset()
        self.reset_score()

    def set_level(self, _):
        pass

    def set_position_tolerance(self, tolerance):
        self.position_tolerance = tolerance

    def set_orientation_tolerance(self, tolerance):
        self.orientation_tolerance = tolerance

    def setup_arms(self, env_config):
        self.active_arms = []
        self.all_arms = create_arms(
            radius=1.0,
            count=self.max_arms_count,
            training=False,
            speed=env_config["arm_speed"],
            arm_type=env_config["arm_type"],
        )
        self.enable_arms()

    def reset_stats(self):
        super().reset_stats()

    def on_step_simulation(self, t_sim, t_max, state):
        super().on_step_simulation(t_sim, t_max, state)
        self.current_episode_score["time"] = (
            BenchmarkMultiArmEnv.EPISODE_CLOCK_TIME_LENGTH * t_sim / t_max
        )

    def on_episode_end(self):
        if self.gui:
            sleep(1)

        stats = self.get_stats_to_log()
        for key in stats.keys():
            self.current_episode_score[key] = stats[key]

        collision_free = self.current_episode_score["collisions"] == 0
        self.current_episode_score["success"] = (
            self.current_episode_score["collective_reach"] if collision_free else 0
        )

        task = self.get_current_task()
        assert task is not None
        self.current_episode_score["task"] = {  # type: ignore
            "id": task.id,
            "task_path": task.task_path,
            "arm_count": task.arm_count,
            "difficulty": task.difficulty,
        }

        assert self.task_manager is not None
        target_poses = self.task_manager.get_target_end_effector_poses()
        transformed_target_poses = [
            arm.global_to_arm_frame(pose[0], pose[1])
            for pose, arm in zip(target_poses, self.active_arms)
        ]

        position_residuals, orientation_residuals = self.get_arm_eef_residuals()

        self.current_episode_score["debug"] = {  # type: ignore
            "target_eef_poses": target_poses,
            "target_joint_config": task.goal_config,
            "position_residuals": position_residuals,
            "initial_joint_config": task.start_config,
            "orientation_residuals": orientation_residuals,
            "transformed_target_eef_poses": transformed_target_poses,
            "final_eef_poses": [
                list(arm.get_end_effector_pose() for arm in self.active_arms)
            ],
            "final_joint_config": [
                list(arm.get_arm_joint_values() for arm in self.active_arms)
            ],
            "positions_reached": [
                residual < self.position_tolerance for residual in position_residuals
            ],
            "orientations_reached": [
                residual < self.orientation_tolerance
                for residual in orientation_residuals
            ],
        }

        print(
            "[BenchmarkMultiArmEnv] Episode Score for Task ID {}: {}".format(
                task.id, self.current_episode_score["success"]
            )
        )
        self.logger.add_stats(self.current_episode_score)
