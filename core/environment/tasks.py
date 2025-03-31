import os
import random
import numpy as np
from time import sleep
from pathlib import Path
from os.path import abspath
from json import load, dump
from decimal import Decimal
from itertools import product

from core.environment.utils import Target
from core.environment.arm import Robotiq2F85Target


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += Decimal(jump)


class Task:
    def __init__(
        self,
        target_eef_poses,
        base_poses,
        start_config,
        goal_config,
        difficulty=None,
        dynamic_speed=None,
        start_goal_config=None,
        task_path=None,
    ):

        self.arm_count = len(start_config)

        self.base_poses = base_poses
        self.goal_config = goal_config
        self.start_config = start_config
        self.target_eef_poses = target_eef_poses

        self.task_path = task_path
        if self.task_path is not None:
            self.id = str(os.path.basename(self.task_path)).split(".")[0]
        else:
            self.id = -1

        self.dynamic_speed = dynamic_speed

        self.start_goal_config = start_goal_config  # For dynamic tasks only!
        if self.start_goal_config is None:
            self.start_goal_config = [None for _ in range(self.arm_count)]

        self.difficulty = difficulty
        if self.difficulty is None:
            self.difficulty = Task.compute_task_difficulty(self)

    def to_json(self):
        assert self.start_goal_config is not None
        output = {
            "id": self.id,
            "task_path": self.task_path,
            "arm_count": self.arm_count,
            "base_poses": self.base_poses,
            "difficulty": self.difficulty,
            "dynamic_speed": self.dynamic_speed,
            "target_eef_poses": self.target_eef_poses,
            "goal_config": [list(config) for config in self.goal_config],
            "start_config": [list(config) for config in self.start_config],
            "start_goal_config": [
                None if config is None else list(config)
                for config in self.start_goal_config
            ],
        }
        return output

    def save(self):
        if self.task_path is not None:
            dump(self.to_json(), open(self.task_path, "w"), indent=4)

    @staticmethod
    def compute_task_difficulty(task, workspace_radius=0.85):
        workspace_positions = [np.array(pose[0]) for pose in task.base_poses]
        if len(workspace_positions) < 2:
            return 0.0

        distances = [
            np.linalg.norm(position1 - position2)
            for position1, position2 in product(workspace_positions, repeat=2)
        ]
        distances = [distance for distance in distances if distance > 0.0]
        if all([distance > 2.0 * workspace_radius for distance in distances]):
            return 0.0

        resolution = 0.05
        workspace_intersection_percentages = []

        for workspace_position in workspace_positions:
            workspace_volume = 0
            workspace_intersection = 0
            others = [
                position
                for position in workspace_positions
                if np.linalg.norm(position - workspace_position) != 0.0
            ]

            x_min = Decimal(workspace_position[0] - workspace_radius)
            x_max = Decimal(workspace_position[0] + workspace_radius)
            y_min = Decimal(workspace_position[1] - workspace_radius)
            y_max = Decimal(workspace_position[1] + workspace_radius)

            for x in drange(x_min, x_max, str(resolution)):
                for y in drange(y_min, y_max, str(resolution)):
                    for z in drange(0, workspace_radius, str(resolution)):
                        point = np.array([x, y, z])
                        inside_workspace = (
                            np.linalg.norm(point - workspace_position)
                            < workspace_radius
                        )
                        if not inside_workspace:
                            continue
                        workspace_volume += 1
                        if any(
                            [
                                np.linalg.norm(point - other) < workspace_radius
                                for other in others
                            ]
                        ):
                            workspace_intersection += 1
            workspace_intersection_percentages.append(
                workspace_intersection / workspace_volume
            )
        return np.max(workspace_intersection_percentages)

    @staticmethod
    def from_file(task_path):
        try:
            task = load(open(task_path))
        except Exception as e:
            print(f"Error loading task from {task_path}: {e}")
            return None
        return Task(
            task_path=task_path,
            base_poses=task["base_poses"],
            goal_config=task["goal_config"],
            start_config=task["start_config"],
            target_eef_poses=task["target_eef_poses"],
            difficulty=None if "difficulty" not in task else task["difficulty"],
            dynamic_speed=(
                None if "dynamic_speed" not in task else task["dynamic_speed"]
            ),
            start_goal_config=(
                None if "start_goal_config" not in task else task["start_goal_config"]
            ),
        )

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        iter_index = self._iter_index
        if iter_index > self.arm_count:
            raise StopIteration

        self._iter_index = iter_index + 1
        assert self.start_goal_config is not None
        output = {
            "target_eef_pose": self.target_eef_poses[iter_index],
            "base_pose": self.base_poses[iter_index],
            "start_config": self.start_config[iter_index],
            "goal_config": self.goal_config[iter_index],
            "start_goal_config": self.start_goal_config[iter_index],
        }
        return output


class TaskManager:
    def __init__(self, config, colors, task_loader=None, training=True):
        self.current_task = None
        self.colors = colors
        self.use_task_loader = False
        self.static_tasks = config["environment"]["task"]["type"] == "static"

        if task_loader is not None:
            self.use_task_loader = True
            self.task_loader = task_loader
        else:
            raise NotImplementedError("[Task Manager] Task loader was not provided!")

        visual_class = Target if training else Robotiq2F85Target
        self.target_visuals = [
            visual_class(pose=[[0, 0, 0], [0, 0, 0, 0]], color=color)
            for color in self.colors
        ]

    def __getitem__(self, index):
        assert self.current_task is not None
        return self.current_task.target_eef_poses[index]

    def set_timer(self, _):
        if self.static_tasks:
            return
        raise NotImplementedError("Dynamic tasks not implemented yet!")

    def setup_visuals(self, target_poses):
        poses_count = len(target_poses)
        target_visuals_count = len(self.target_visuals)
        for i in range(target_visuals_count):
            if i >= poses_count:
                self.target_visuals[i].set_pose(
                    [
                        [i, 5, 0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            else:
                self.target_visuals[i].set_pose(target_poses[i])

    def set_current_task(self, task):
        self.current_task = task

    def get_target_end_effector_poses(self):
        if self.static_tasks:
            assert self.current_task is not None
            return self.current_task.target_eef_poses
        raise NotImplementedError("Dynamic tasks not implemented yet!")

    def setup_next_task(
        self,
        max_task_arms_count,
        min_task_arms_count,
        max_task_difficulty,
        min_task_difficulty,
    ):
        if self.use_task_loader:
            while True:
                self.current_task = self.task_loader.get_next_task()
                task_difficulty = self.current_task.difficulty
                if (
                    self.current_task.arm_count <= max_task_arms_count
                    and self.current_task.arm_count >= min_task_arms_count
                    and task_difficulty <= max_task_difficulty
                    and task_difficulty >= min_task_difficulty
                ):
                    break
        else:
            raise NotImplementedError("Random task generation not implemented yet!")
        self.set_timer(0.0)
        self.setup_visuals(self.get_target_end_effector_poses())


class TaskLoader:
    def __init__(self, root_dir, repeat=True, shuffle=False, only_tasks=None):
        self.current_idx = 0
        self.repeat = repeat
        if not self.repeat:
            print("[TaskLoader] WARNING: Tasks will not be repeated")

        file_names = Path(root_dir).rglob("*.json")
        if not shuffle:
            file_names = sorted(file_names)

        self.files = []
        for file_name in file_names:
            if only_tasks is not None:
                if file_name.as_posix().split("/")[-1][:-5] not in only_tasks:
                    continue
            if "config" in str(file_name):
                continue
            self.files.append(abspath(file_name))

        self.count = len(self.files)
        assert self.count > 0, "No tasks found in {}".format(root_dir)
        print("[TaskLoader] Found {} tasks".format(self.count))

        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        self.current_idx += 1
        if self.current_idx >= len(self.files):
            raise StopIteration
        return Task.from_file(task_path=self.files[self.current_idx - 1])

    def get_next_task(self):
        if self.current_idx >= len(self.files) and not self.repeat:
            print("[TaskLoader] No more tasks to load")
            while True:
                sleep(5)  # Force user to quit to trigger exit handlers
        current_file = self.files[self.current_idx]
        self.current_idx += 1
        if self.repeat:
            self.current_idx %= len(self.files)
        return Task.from_file(task_path=current_file)
