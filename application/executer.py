import os
import numpy as np
import pybullet as p
from typing import List
from random import randint
from itertools import chain
from torch import FloatTensor
from collections import deque

from core.environment.tasks import Task
from core.recorder import PybulletRecorder
from core.planner.cbs import ConflictBasedSearch
from core.planner.agent_planners import Agent, ResolveDualConflict
from application.task import (
    PolicyTask,
    UR5AsyncTaskRunner as TaskRunner,
)


class Executer:
    def __init__(
        self,
        task_runners: List[TaskRunner],
        recorder: PybulletRecorder,
        recorder_dir,
        parameters,
        limit=10000,
    ):
        self.limit = limit
        self.recorder_dir = recorder_dir
        self.recorder = recorder
        self.task_runners = task_runners
        self.ur5s = [tr.ur5 for tr in task_runners]
        self.simulation_output_path = os.path.join(
            self.recorder_dir, f"simulation_{randint(0,1000)}.pkl"
        )

        self.observation_items = [
            {"name": "joint_values", "dimensions": 6, "history": 0},
            {"name": "end_effector_pose", "dimensions": 7, "history": 0},
            {"name": "target_pose", "dimensions": 7, "history": 0},
            {"name": "link_positions", "dimensions": 30, "history": 0},
            {"name": "pose", "dimensions": 7, "history": 0},
        ]
        self.obs_key = [obs_item["name"] for obs_item in self.observation_items]

        self.parameters = parameters
        self.num_agents = len(self.ur5s)

        self.single_agent_planners = {
            arm_id: Agent(id=arm_id, arm=self.ur5s[arm_id], parameters=parameters)
            for arm_id in range(self.num_agents)
        }
        self.dual_agent_planner = ResolveDualConflict(
            self.ur5s, self.get_observation, self.preprocess_observation, parameters
        )
        self.state_deque = deque(
            [self.get_state()] * self.parameters["observation_horizon"],
            maxlen=self.parameters["observation_horizon"],
        )
        for agent in self.single_agent_planners.values():
            agent.set_task(
                Task(
                    target_eef_poses=self.get_target_end_effector_poses(),
                    base_poses=np.zeros((self.num_agents, 7)),
                    start_config=np.zeros((self.num_agents, 6)),
                    goal_config=np.zeros((self.num_agents, 6)),
                )
            )

    def get_target_end_effector_poses(self):
        return [
            getattr(tr.current_task(), "target_pose", tr.ur5.get_end_effector_pose())
            for tr in self.task_runners
        ]

    def act(self):
        result = self.predict_joint_plan()
        if result is None:
            return None, None
        final_plan, earliest_collision_time = result
        action_horizon = self.parameters["action_horizon"]
        if (
            earliest_collision_time > action_horizon
            and earliest_collision_time != float("inf")
        ):
            action_horizon = earliest_collision_time
        # action_horizon = self.parameters["prediction_horizon"]

        final_plan = [plan[:action_horizon] for plan in final_plan]
        actions = np.zeros((self.num_agents, action_horizon, 6))
        keep_indices = [i for i in range(self.num_agents)]
        actions[keep_indices] = final_plan
        return actions, action_horizon

    def predict_joint_plan(self):
        plans = [
            self.single_agent_planners[aid].predict_plan()
            for aid in range(self.num_agents)
        ]

        if self.parameters["search"] == "cbs":
            planner = ConflictBasedSearch(
                plans,
                self.parameters,
                (
                    self.single_agent_planners,
                    self.dual_agent_planner,
                ),
                sim_steps=1.0,
            )
            result = planner.find_plans(self.state_deque)
        else:
            raise ValueError(f"Unknown search method: {self.parameters['search']}")

        return result

    def get_state(self):
        self.state = {"ur5s": []}

        self.state["ur5s"] = [
            (
                {
                    "ur5": ur5,
                    "pose": ur5.get_pose(),
                    "target_pose": target_pose,
                    "joint_values": ur5.get_arm_joint_values(),
                    "end_effector_pose": ur5.get_end_effector_pose(),
                    "link_positions": ur5.get_link_global_positions(),
                }
            )
            for ur5, target_pose in zip(self.ur5s, self.get_target_end_effector_poses())
        ]
        return self.state

    def is_done(self):
        return all([tr.is_done() for tr in self.task_runners])

    def run(self):
        print_step = 1000
        last_print_step = -print_step
        step_count = 0
        while step_count < self.limit:
            if step_count - last_print_step >= print_step:
                # print(f"Step Count: {step_count}")
                last_print_step = step_count
            if self.is_done():
                self.try_save()
                return True, step_count, None

            compute_actions = False
            state = self.get_state()
            self.state_deque.append(state)
            for arm_id, tr in enumerate(self.task_runners):
                if isinstance(tr.current_task(), PolicyTask):
                    compute_actions = True
                    obs = self.get_single_observation(tr.ur5, arm_id, state)
                    observation = {"ur5s": [obs]}
                    observation = self.preprocess_observation(observation)
                    self.single_agent_planners[arm_id].update_deque(observation[-1])
                    self.single_agent_planners[arm_id].set_task(
                        Task(
                            target_eef_poses=self.get_target_end_effector_poses(),
                            base_poses=np.zeros((self.num_agents, 7)),
                            start_config=np.zeros((self.num_agents, 6)),
                            goal_config=np.zeros((self.num_agents, 6)),
                        )
                    )
            if compute_actions:
                actions, horizon = self.act()

            for tr in self.task_runners:
                if isinstance(tr.current_task(), PolicyTask):
                    tr.ur5.control_arm_joints_delta(np.zeros(6))

            step_happened = [False] * self.num_agents
            for arm_id, tr in enumerate(self.task_runners):
                if isinstance(tr.current_task(), PolicyTask):
                    if actions is None:
                        self.try_save()
                        return False, step_count, "No valid plan found"
                    action = actions[arm_id]
                    for execution_step in range(len(action)):
                        action_step = action[execution_step]
                        tr.current_task().set_action(action_step)
                        success, info = tr.step()
                        p.stepSimulation()
                        state = self.get_state()
                        obs = self.get_single_observation(tr.ur5, arm_id, state)
                        observation = {"ur5s": [obs]}
                        observation = self.preprocess_observation(observation)
                        self.single_agent_planners[arm_id].update_deque(observation[-1])
                        self.state_deque.append(state)
                        self.recorder.add_keyframe()
                        step_happened[arm_id] = True
                        if not success:
                            self.try_save()
                            return False, step_count, info
                    tr.current_task().set_action(None)
                    tr.ur5.control_arm_joints_delta(np.zeros(6))
                else:
                    success, info = tr.step()
                if not success:
                    self.try_save()
                    return False, step_count, info
            if not all(step_happened):
                p.stepSimulation()
                self.recorder.add_keyframe()
                step_count += 1
            if any(step_happened):
                assert actions is not None
                step_count += len(actions[0])
        self.try_save()
        return False, step_count, "Out of time"

    def get_single_observation(self, this_ur5, ur5_idx, state):
        obs_entry = {}
        for item in self.observation_items:
            key = item["name"]
            if key == "joint_values":
                val = [state["ur5s"][ur5_idx][key]]
            elif key == "link_positions":
                val = [
                    list(
                        chain.from_iterable(
                            [
                                this_ur5.global_to_ur5_frame(
                                    position=np.array(link_pos), rotation=None
                                )[0]
                                for link_pos in state["ur5s"][ur5_idx][key]
                            ]
                        )
                    )
                ]
            elif key in {"end_effector_pose", "target_pose", "pose"}:
                val = [
                    list(
                        chain.from_iterable(
                            this_ur5.global_to_ur5_frame(
                                position=state["ur5s"][ur5_idx][key][0],
                                rotation=state["ur5s"][ur5_idx][key][1],
                            )
                        )
                    )
                ]
            else:
                val = [this_ur5.global_to_ur5_frame(state["ur5s"][ur5_idx][key])]
            obs_entry[key] = val
        return obs_entry

    def get_observation(self, this_ur5, state, arm_list=None):
        workspace_radius = 0.85
        obs = {"ur5s": []}

        if arm_list is None:
            pos = np.array(this_ur5.get_pose()[0])
            arm_list = [
                ur5
                for ur5 in self.ur5s
                if np.linalg.norm(pos - np.array(ur5.get_pose()[0]))
                < 2 * workspace_radius
            ]
            arm_list.sort(
                reverse=True,
                key=lambda ur5: np.linalg.norm(pos - np.array(ur5.get_pose()[0])),
            )

        for ur5 in arm_list:
            ur5_idx = self.ur5s.index(ur5)
            obs["ur5s"].append(self.get_single_observation(this_ur5, ur5_idx, state))
        return obs

    def preprocess_observation(self, observation):
        output = []
        for ur5_obs in observation["ur5s"]:
            ur5_output = np.array([])
            for key in self.obs_key:
                item = ur5_obs[key]
                for history_frame in item:
                    ur5_output = np.concatenate((ur5_output, history_frame))
            if len(ur5_output) == 0:
                continue
            output.append(ur5_output)
        output = FloatTensor(output)
        return output

    def try_save(self):
        if self.recorder:
            self.recorder.save(self.simulation_output_path)
            self.recorder.reset()
