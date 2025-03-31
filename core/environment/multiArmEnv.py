import torch
import quaternion
import numpy as np
import pybullet as p
import pybullet_data
from time import sleep
from itertools import chain

from core.environment.arm import UR5
from core.environment.tasks import TaskManager
from core.environment.utils import (
    create_arms,
    pose_to_high_frequency_pose,
    position_to_high_frequency_position,
)


class MultiArmEnv:
    DEBUG_LOG = False
    SIMULATION_TIMESTEP = 1.0 / 240.0

    def __init__(
        self,
        env_config,
        training_config,
        gui,
        logger,
    ):
        self.logger = logger
        self.setup(env_config, training_config, gui)
        self.task_manager = TaskManager(
            config={"environment": env_config, "training": training_config},
            task_loader=training_config["task_loader"],
            colors=[arm.color for arm in self.all_arms],
        )

    def setup(self, env_config, training_config, gui):

        self.action_type = env_config["action_type"]
        self.episode_length = env_config["episode_length"]
        self.action_interval = env_config["action_interval"]
        self.simulation_steps_per_action_step = int(
            self.action_interval / MultiArmEnv.SIMULATION_TIMESTEP
        )

        self.gui = gui
        self.observations = None
        self.task_manager = None
        self.all_arms = []
        self.num_episodes_elapsed = 0

        self.max_arms_count = env_config["max_arms_count"]
        self.max_task_arms_count = env_config["max_arms_count"]
        self.min_task_arms_count = env_config["min_arms_count"]

        self.workspace_radius = env_config["workspace_radius"]
        self.collision_distance = env_config["collision_distance"]
        self.collision_penalty = env_config["reward"]["collision_penalty"]
        self.individual_reach_reward = env_config["reward"]["individually_reach_target"]
        self.collective_reach_reward = env_config["reward"]["collectively_reach_target"]

        self.terminate_on_collision = env_config["terminate_on_collision"]
        self.terminate_on_collectively_reach_target = env_config[
            "terminate_on_collectively_reach_target"
        ]

        self.position_tolerance = env_config["position_tolerance"]
        self.orientation_tolerance = env_config["orientation_tolerance"]

        self.failed_in_task = False
        self.finish_task_in_episode = False
        self.stop_arm_after_reach = env_config["stop_arm_after_reach"]

        self.min_task_difficulty = 0.0
        self.max_task_difficulty = 100.0

        self.current_step = 0
        self.terminate_episode = False

        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            "plane.urdf",
            [0, 0, -self.collision_distance - 0.01],
        )
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)

        if self.gui:
            self.real_time_debug = p.addUserDebugParameter("real-time", 0.0, 1.0, 0.0)

        self.on_setup(env_config, training_config)
        self.setup_action_observation(training_config["observations"])
        self.setup_arms(env_config)

    def setup_action_observation(self, observation_config):
        self.action_dim = UR5.JOINTS_COUNT
        self.obs_key = [obs_item["name"] for obs_item in observation_config["items"]]
        self.observation_items = observation_config["items"]

    def setup_arms(self, env_config):
        if env_config["arms_position_picker"] == "evenly_spaced":
            pass
        else:
            print(
                "[MultiArmEnv] Position picker not supported: {}".format(
                    env_config["arms_position_picker"]
                )
            )
            exit(-1)
        self.all_arms = create_arms(
            radius=0.5,
            count=self.max_arms_count,
            speed=env_config["arm_speed"],
            arm_type=env_config["arm_type"],
        )
        self.active_arms = []
        self.enable_arms()

    def enable_arms(self, count=None):
        if count == len(self.active_arms):
            return
        self.disable_all_arms()
        for i, arm in enumerate(self.all_arms):
            if count is not None and i == count:
                break
            arm.enable()
            self.active_arms.append(arm)

    def disable_all_arms(self):
        for i, arm in enumerate(self.all_arms):
            arm.disable(idx=i)
        self.active_arms = []

    def should_get_next_task(self):
        if not self.failed_in_task:
            return True
        return False

    def on_reset(self):
        pass

    def get_current_task(self):
        assert self.task_manager is not None
        return self.task_manager.current_task

    def setup_task(self):
        if self.should_get_next_task() and self.task_manager is not None:
            self.task_manager.setup_next_task(
                max_task_arms_count=self.max_task_arms_count,
                min_task_arms_count=self.min_task_arms_count,
                max_task_difficulty=self.max_task_difficulty,
                min_task_difficulty=self.min_task_difficulty,
            )

        current_task = self.get_current_task()
        assert current_task is not None
        self.enable_arms(count=current_task.arm_count)
        for arm, arm_task in zip(self.active_arms, current_task):
            arm.set_pose(arm_task["base_pose"])
            arm.set_arm_joints(arm_task["start_config"])
            arm.step()

    def reset_stats(self):
        self.stats = {
            "collective_reach_count": 0,
            # Number of steps arm spends in reached state
            "reached": [0] * len(self.active_arms),
            # Number of steps arm spends in collision state
            "collisions": [0] * len(self.active_arms),
        }

    def preprocess_observation(self, observation):
        output = []
        for arm_observation in observation["arms"]:
            arm_output = np.array([])
            for key in self.obs_key:
                key = key.split("_high_freq")[0]
                item = arm_observation[key]
                for history_frame in item:
                    arm_output = np.concatenate((arm_output, history_frame))
            output.append(arm_output)
        output = torch.FloatTensor(np.array(output))
        return output

    def get_observation(self, this_arm, state, arm_list=None):
        observations = {"arms": []}

        if arm_list is None:
            position = np.array(this_arm.get_pose()[0])
            arm_list = [
                arm
                for arm in self.active_arms
                if np.linalg.norm(position - np.array(arm.get_pose()[0]))
                < 2 * self.workspace_radius
            ]
            arm_list.sort(
                reverse=True,
                key=lambda arm: np.linalg.norm(position - np.array(arm.get_pose()[0])),
            )

        for arm in arm_list:

            observations["arms"].append({})
            arm_index = self.active_arms.index(arm)

            for item in self.observation_items:
                value = None
                key = item["name"]
                high_freq = "high_freq" in key
                key = key.split("_high_freq")[0]
                if key == "joint_values":
                    value = [state["arms"][arm_index][key]]
                elif "link_positions" in key:
                    value = [
                        list(
                            chain.from_iterable(
                                [
                                    (
                                        position_to_high_frequency_position(
                                            this_arm.global_to_arm_frame(
                                                position=np.array(link_position),
                                                rotation=None,
                                            )[0]
                                        )
                                        if high_freq
                                        else this_arm.global_to_arm_frame(
                                            position=np.array(link_position),
                                            rotation=None,
                                        )[0]
                                    )
                                    for link_position in state["arms"][arm_index][key]
                                ]
                            )
                        )
                    ]
                elif (
                    "end_effector_pose" in key
                    or "target_pose" in key
                    or key == "pose"
                    or key == "pose_high_freq"
                ):
                    value = [
                        list(
                            chain.from_iterable(
                                pose_to_high_frequency_pose(
                                    this_arm.global_to_arm_frame(
                                        position=state["arms"][arm_index][key][0],
                                        rotation=state["arms"][arm_index][key][1],
                                    )
                                )
                                if high_freq
                                else this_arm.global_to_arm_frame(
                                    position=state["arms"][arm_index][key][0],
                                    rotation=state["arms"][arm_index][key][1],
                                )
                            )
                        )
                    ]
                else:
                    value = [
                        (
                            pose_to_high_frequency_pose(
                                this_arm.global_to_arm_frame(
                                    state["arms"][arm_index][key],
                                )
                            )
                            if high_freq
                            else this_arm.global_to_arm_frame(
                                state["arms"][arm_index][key],
                            )
                        )
                    ]
                observations["arms"][-1][key] = value
        return observations

    def get_pose_residuals(self, poseA, poseB):
        positionA = np.array(poseA[0])
        xA, yA, zA, wA = poseA[1]
        orientationA = quaternion.quaternion(wA, xA, yA, zA)

        positionB = np.array(poseB[0])
        xB, yB, zB, wB = poseB[1]
        orientationB = quaternion.quaternion(wB, xB, yB, zB)  # type: ignore

        position_residual = np.linalg.norm(positionA - positionB)
        orientation_residual = (orientationA * orientationB.inverse()).angle()
        orientation_residual = orientation_residual % (2 * np.pi)
        if orientation_residual > np.pi:
            orientation_residual = 2 * np.pi - orientation_residual

        return position_residual, orientation_residual

    def on_target_reach(self, arm, arm_index):
        assert self.task_manager is not None

        arm.on_touch_target()
        self.task_manager.target_visuals[arm_index].touched()
        self.stats["reached"][self.active_arms.index(arm)] += 1
        if self.stop_arm_after_reach:
            arm.control_arm_joints(arm.get_arm_joint_values())

    def check_arm_reached_target(self, arm_index, arm, target_eef_pose):
        assert self.task_manager is not None
        position_residual, orientation_residual = self.get_pose_residuals(
            target_eef_pose, arm.get_end_effector_pose()
        )
        reached_position = position_residual < self.position_tolerance
        reached_orientation = orientation_residual < self.orientation_tolerance
        reached_target = reached_position and reached_orientation
        if reached_target:
            self.on_target_reach(arm, arm_index)
        else:
            arm.on_untouch_target()
            self.task_manager.target_visuals[arm_index].normal()
        return reached_target

    def visualize_collision(self):
        if self.gui:
            points = set(
                [
                    arm.collision_point[5]
                    for arm in self.active_arms
                    if arm.collision_point is not None
                ]
            )
            sphere_viz_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=(1, 0, 0, 0.5),
            )
            point_visuals = [
                p.createMultiBody(
                    basePosition=point,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sphere_viz_id,
                )
                for point in points
            ]
            for point_visual in point_visuals:
                p.removeBody(point_visual)

    def on_setup(self, env_config, training_config):
        _ = env_config
        _ = training_config
        pass

    def on_collision(self):
        visualize_collision = True
        if visualize_collision:
            self.visualize_collision()

        if self.terminate_on_collision:
            self.terminate_episode = True

    def on_all_arms_reach_target(self):
        self.finish_task_in_episode = True
        self.stats["collective_reach_count"] += 1
        self.terminate_episode = (
            self.terminate_episode or self.terminate_on_collectively_reach_target
        )

    def get_state(self):
        assert self.task_manager is not None
        colliding = any([arm.check_collision() for arm in self.active_arms])
        if colliding:
            self.on_collision()
        self.state = {"arms": []}

        self.state["arms"] = [
            {
                "arm": arm,
                "pose": arm.get_pose(),
                "target_pose": target_eef_pose,
                "joint_values": arm.get_arm_joint_values(),
                "end_effector_pose": arm.get_end_effector_pose(),
                "link_positions": arm.get_link_global_positions(),
                "colliding": False if not colliding else arm.check_collision(),
                "reached_target": self.check_arm_reached_target(
                    i, arm, target_eef_pose
                ),
            }
            for i, (arm, target_eef_pose) in enumerate(
                zip(self.active_arms, self.task_manager.get_target_end_effector_poses())
            )
        ]

        self.state["reach_count"] = sum(  # type: ignore
            [
                1 if arm_state["reached_target"] else 0
                for arm_state in self.state["arms"]
            ]
        )

        if self.state["reach_count"] == len(self.active_arms):
            self.on_all_arms_reach_target()

        return self.state

    def get_observations(self, state=None):
        if state is None:
            state = self.get_state()

        return [
            self.get_observation(this_arm=arm, state=state) for arm in self.active_arms
        ]

    def reset(self):

        self.current_step = 0
        self.terminate_episode = False
        self.finish_task_in_episode = False

        self.num_episodes_elapsed += 1

        self.on_reset()
        self.setup_task()
        self.reset_stats()

        self.episode_reward_sum = np.zeros(len(self.active_arms))

        for _ in range(50):
            p.stepSimulation()

        self.observations = [
            self.preprocess_observation(obs) for obs in self.get_observations()
        ]
        return self.observations

    def action_to_robots(self, actions):
        if len(actions) != len(self.active_arms):
            print(
                "[MultiArmEnv] Action count does not match arm count: {} != {}".format(
                    len(actions), len(self.active_arms)
                )
            )
            exit(1)
        for arm, action in zip(self.active_arms, actions):
            if not isinstance(action, np.ndarray):
                action = action.data.numpy()
            if self.action_type == "delta":
                arm.control_arm_joints_delta(action)
            else:
                raise NotImplementedError(
                    "[MultiArmEnv] Action type not supported: {}".format(
                        self.action_type
                    )
                )

    def handle_actions(self, actions):
        assert self.task_manager is not None

        if self.stop_arm_after_reach:
            for arm_index, (_, arm, target_eef_pose) in enumerate(
                zip(
                    actions,
                    self.active_arms,
                    self.task_manager.get_target_end_effector_poses(),
                )
            ):
                if self.check_arm_reached_target(arm_index, arm, target_eef_pose):
                    actions[arm_index] = np.zeros(UR5.JOINTS_COUNT)

        self.action_to_robots(actions)

    def get_arm_eef_residuals(self):
        assert self.task_manager is not None

        residuals = [
            self.get_pose_residuals(target_pose, arm.get_end_effector_pose())
            for arm, target_pose in zip(
                self.active_arms, self.task_manager.get_target_end_effector_poses()
            )
        ]
        position_residuals = np.array([residual[0] for residual in residuals])
        orientation_residuals = np.array([residual[1] for residual in residuals])
        return position_residuals, orientation_residuals

    def get_rewards(self, state):
        collision_penalty = np.array(
            [
                (self.collision_penalty if arm_state["colliding"] else 0)
                for arm_state in state["arms"]
            ]
        )
        individually_reached_target_rewards = np.array(
            [
                (self.individual_reach_reward if arm_state["reached_target"] else 0)
                for arm_state in state["arms"]
            ]
        )
        collectively_reached_targets = state["reach_count"] == len(self.active_arms)
        collectively_reached_target_rewards = np.array(
            [
                (self.collective_reach_reward if collectively_reached_targets else 0)
                for _ in range(len(self.active_arms))
            ]
        )
        arm_rewards_sum = (
            collision_penalty
            + individually_reached_target_rewards
            + collectively_reached_target_rewards
        )
        return arm_rewards_sum

    def on_step_simulation(self, t_sim, t_max, state):
        assert self.task_manager is not None
        for arm_index, _ in enumerate(self.active_arms):
            if state["arms"][arm_index]["colliding"]:
                self.stats["collisions"][arm_index] += 1
        self.task_manager.set_timer(t_sim / t_max)

    def get_stats_to_log(self):
        position_residuals, orientation_residuals = self.get_arm_eef_residuals()
        output = {
            "rewards": np.mean(self.episode_reward_sum),
            # Mean number of steps the arms spent in reached state
            "individual_reach": np.mean(self.stats["reached"]),
            # Number of times all arms reached the target
            "collective_reach": self.stats["collective_reach_count"],
            # Mean number of simulation steps the arms spent in collision state
            "collisions": np.mean(self.stats["collisions"]),
            "success": (
                self.stats["collective_reach_count"]
                if sum(self.stats["collisions"]) == 0
                else 0
            ),
            "mean_position_residual": np.mean(position_residuals),
            "mean_orientation_residual": np.mean(orientation_residuals),
            "max_position_residual": np.max(position_residuals),
            "max_orientation_residual": np.max(orientation_residuals),
            "num_episodes": self.num_episodes_elapsed,
        }
        return output

    def send_stats_to_logger(self):
        if self.logger is not None:
            self.logger.add_stats(self.get_stats_to_log())

    def on_episode_end(self):
        self.failed_in_task = (
            self.stats["collective_reach_count"] == 0
            or sum(self.stats["collisions"]) != 0
        )
        self.send_stats_to_logger()

    def step(self, actions, recorder=None):
        if self.terminate_episode:
            return self.reset()

        if recorder is not None:
            recorder.add_keyframe()

        self.current_step += 1

        rewards = np.zeros(len(self.active_arms))
        self.handle_actions(actions)

        for t_sim in range(self.simulation_steps_per_action_step):
            p.stepSimulation()

            for arm in self.active_arms:
                arm.step()

            self.state = self.get_state()
            rewards += self.get_rewards(self.state)

            self.on_step_simulation(
                self.current_step * self.simulation_steps_per_action_step + t_sim,
                self.episode_length * self.simulation_steps_per_action_step,
                self.state,
            )

            if self.gui and p.readUserDebugParameter(self.real_time_debug) == 1.0:
                sleep(MultiArmEnv.SIMULATION_TIMESTEP)

            rKey = ord("r")
            keys = p.getKeyboardEvents()
            if rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
                self.terminate_episode = True

            if self.terminate_episode:
                break

        self.terminate_episode = (
            self.terminate_episode or self.current_step >= self.episode_length
        )

        self.observations = [
            self.preprocess_observation(observation)
            for observation in self.get_observations(state=self.state)
        ]
        self.episode_reward_sum += rewards

        if self.terminate_episode:
            self.on_episode_end()

        return self.observations
