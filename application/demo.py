import os
from pathlib import Path
import time
import pickle
import argparse
import numpy as np
import pybullet as p
from tqdm import tqdm
from math import degrees
from collections import namedtuple
from itertools import combinations
from dataclasses import dataclass, asdict
from core.recorder import PybulletRecorder
from random import seed as random_seed, uniform, choice

import application.pybullet_utils as pu
from application.executer import Executer
from application.ur5_robotiq_controller import UR5RobotiqPybulletController
from application.misc_utils import (
    OtherObject,
    TargetObject,
    write_csv_line,
    suppress_stdout,
    configure_pybullet,
)
from application.task import (
    PolicyTask,
    SetTargetTask,
    ControlArmTask,
    OpenGripperTask,
    CloseGripperTask,
    AttachToGripperTask,
    DetachToGripperTask,
    CartesianControlTask,
    UR5AsyncTaskRunner as TaskRunner,
)

ConfigInfo = namedtuple(
    "ConfigInfo", ["joint_config", "ee_pose", "pos_distance", "quat_distance"]
)

d_robot = 0.6
d_target = 1.0
robot_base_poses = [
    [[d_robot, d_robot, 0.01], [0, 0, 0, 1]],
    [[d_robot, -d_robot, 0.01], [0, 0, 0, 1]],
    [[-d_robot, d_robot, 0.01], pu.quaternion_from_euler([0, 0, np.pi])],
    [[-d_robot, -d_robot, 0.01], pu.quaternion_from_euler([0, 0, np.pi])],
]
dump_positions = [
    [0.2, 0.2, 0.5],
    [0.2, -0.2, 0.5],
    [-0.2, 0.2, 0.5],
    [-0.2, -0.2, 0.5],
]
target_xys = [[d_target, 0], [0, -d_target], [0, d_target], [-d_target, 0]]


@dataclass
class Parameters:
    search: str = "cbs"
    timeout: int = 600
    action_dim: int = 6
    num_samples: int = 10
    n_timesteps: int = 100
    action_horizon: int = 1
    observation_dim: int = 57
    prediction_horizon: int = 16
    observation_horizon: int = 2
    dual_agent_model: str = "runs/plain_diffusion/mini_custom_diffusion_2.pth"
    single_agent_model: str = "runs/plain_diffusion/mini_custom_diffusion_1.pth"


def set_initial_random_configs(ur5s, randomization_magnitude=0.4):
    above_threshold = 0.2
    while True:
        for ur5 in ur5s:
            ur5.reset()
            curr = np.array(ur5.get_arm_joint_values())
            ur5.set_arm_joints(
                curr
                + np.array(
                    [
                        uniform(-randomization_magnitude, randomization_magnitude)
                        for _ in range(6)
                    ]
                )
            )

        if not any([ur5.check_collision_with_info()[0] for ur5 in ur5s]) and all(
            [ur5.get_eef_pose()[0][2] > above_threshold for ur5 in ur5s]
        ):
            break


def prepare_task_runners(ur5s, targets):
    task_runners = []
    for i, (ur5, robot_targets) in enumerate(zip(ur5s, targets)):
        folder_path = os.path.join("application/tasks", "robot" + str(i))
        grasps = [
            pickle.load(open(os.path.join(folder_path, t.object_name, "grasp.p"), "rb"))
            for t in robot_targets
        ]
        grasp_configs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "grasp_config.p"), "rb")
            )
            for t in robot_targets
        ]
        dump_configs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "dump_config.p"), "rb")
            )
            for t in robot_targets
        ]
        dump_jvs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "dump_jv.p"), "rb")
            )
            for t in robot_targets
        ]
        initial_poses = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "initial_pose.p"), "rb")
            )
            for t in robot_targets
        ]

        ur5_tasks = []
        for target, grasp, grasp_config, dump_config, dump_jv, initial_pose in zip(
            robot_targets, grasps, grasp_configs, dump_configs, dump_jvs, initial_poses
        ):
            gcee_pose = grasp_config.ee_pose
            if gcee_pose[0][2] < 0.2:
                gcee_pose[0][2] += 0.2
            dcee_pose = dump_config.ee_pose
            if dcee_pose[0][2] < 0.2:
                dcee_pose[0][2] += 0.2
            ur5_single_target_task = [
                SetTargetTask(ur5=ur5, target_id=target.id, initial_pose=initial_pose),
                PolicyTask(
                    ur5=ur5,
                    target_pose=gcee_pose,
                    position_tolerance=0.2,
                    orientation_tolerance=0.5,
                    # visualize=True,
                ),
                # ControlArmTask(ur5=ur5, target_config=grasp_config.joint_config),
                ControlArmTask(ur5=ur5, target_config=grasp.pre_grasp_jv),
                ControlArmTask(ur5=ur5, target_config=grasp.grasp_jv),
                CloseGripperTask(ur5=ur5),
                AttachToGripperTask(ur5=ur5, target_id=target.id),
                CartesianControlTask(ur5=ur5, axis="z", value=0.3),
                PolicyTask(
                    ur5=ur5,
                    target_pose=dcee_pose,
                    position_tolerance=0.2,
                    orientation_tolerance=0.5,
                    # visualize=True,
                ),
                ControlArmTask(ur5=ur5, target_config=dump_jv),
                CartesianControlTask(ur5=ur5, axis="z", value=-0.05),
                DetachToGripperTask(ur5=ur5),
                OpenGripperTask(ur5=ur5),
                CartesianControlTask(ur5=ur5, axis="z", value=0.05),
            ]
            ur5_tasks += ur5_single_target_task
        task_runners.append(
            TaskRunner(
                ur5=ur5,
                tasks=ur5_tasks + [ControlArmTask(ur5=ur5, target_config=ur5.RESET)],
            )
        )
    return task_runners


def create_target_xyss(d_target, delta, num_objects):
    result = []
    result.append([[d_target + delta * i, 0] for i in range(num_objects)])
    result.append([[0, -d_target - delta * i] for i in range(num_objects)])
    result.append([[0, d_target + delta * i] for i in range(num_objects)])
    result.append([[-d_target - delta * i, 0] for i in range(num_objects)])
    return result


def create_scene(
    random=True, target_object_names=None, initial_configs=None, num_targets_per_arm=1
):
    with suppress_stdout():
        _ = p.loadURDF("plane.urdf")
    plastic_bin = OtherObject(
        "application/assets/tote/tote.urdf", initial_pose=[[0, 0, 0], [0, 0, 0, 1]]
    )

    if target_object_names is None:
        pickable_objects = [
            os.listdir(os.path.join("application/tasks", robot))
            for robot in ["robot0", "robot1", "robot2", "robot3"]
        ]
        target_object_names = [
            choice(list((combinations(po, num_targets_per_arm))))
            for po in pickable_objects
        ]

    target_xyss = create_target_xyss(
        d_target, delta=0.2, num_objects=num_targets_per_arm
    )

    ur5s = [UR5RobotiqPybulletController(base_pose=rb) for rb in robot_base_poses]
    [
        p.addUserDebugText(str(i), [pose[0][0], pose[0][1], 0.6], (1, 0, 0), textSize=2)
        for i, pose in enumerate(robot_base_poses)
    ]
    targets = []
    for i, (names, target_xys) in enumerate(zip(target_object_names, target_xyss)):
        initial_poses = [
            pickle.load(
                open(os.path.join("application/tasks", "robot" + str(i), n, "initial_pose.p"), "rb")
            )
            for n in names
        ]
        rpys = [
            [degrees(e) for e in pu.euler_from_quaternion(pose[1])]
            for pose in initial_poses
        ]
        ur5_targets = [
            TargetObject(n, target_xy, 0, rpy)
            for n, target_xy, rpy in zip(names, target_xys, rpys)
        ]
        targets.append(ur5_targets)

    if random and initial_configs is None:
        set_initial_random_configs(ur5s)
    else:
        assert initial_configs is not None
        for ur5, c in zip(ur5s, initial_configs):
            ur5.set_arm_joints(c)

    pu.step(3)
    return ur5s, targets, plastic_bin, dump_positions


def check_success(targets, bin):
    bbox = p.getAABB(bin.id, -1)
    target_in_bins = []
    for tt in targets:
        for t in tt:
            target_pose = t.get_pose()
            target_in_bin = (
                bbox[0][0] < target_pose[0][0] < bbox[1][0]
                and bbox[0][1] < target_pose[0][1] < bbox[1][1]
            )
            target_in_bins.append(target_in_bin)
    return all(target_in_bins)


def demo_with_seed(seed, result_dir, recorder_dir, parameters):
    configure_pybullet(rendering=False, debug=False)
    random_seed(seed)
    result_filepath = os.path.join(result_dir, "results.csv")

    benchmark_dir = os.path.join("application/tasks", "benchmark", str(seed))
    if not os.path.exists(benchmark_dir):
        raise ValueError("Benchmark dataset does not exist!")

    target_object_names = pickle.load(
        open(os.path.join(benchmark_dir, "target_object_names.p"), "rb")
    )
    intitial_configs = pickle.load(
        open(os.path.join(benchmark_dir, "intitial_configs.p"), "rb")
    )

    ur5s, targets, plastic_bin, _ = create_scene(
        target_object_names=target_object_names, initial_configs=intitial_configs
    )

    recorder = PybulletRecorder()
    for ur5 in ur5s:
        recorder.register_object(
            body_id=ur5.id, urdf_path="application/assets/ur5/ur5_robotiq.urdf", color=ur5.color
        )
    for ur5_targets in targets:
        for target in ur5_targets:
            recorder.register_object(body_id=target.id, urdf_path=target.urdf_path)
    recorder.register_object(body_id=plastic_bin.id, urdf_path=plastic_bin.urdf)

    task_runners = prepare_task_runners(ur5s=ur5s, targets=targets)
    executer = Executer(
        task_runners=task_runners,
        recorder=recorder,
        recorder_dir=recorder_dir,
        parameters=asdict(parameters),
    )
    executer_success, step_count, info = executer.run()
    success = False if not executer_success else check_success(targets, plastic_bin)
    p.disconnect()

    result = {
        "info": info,
        "experiment": seed,
        "success": success,
        "limit": executer.limit,
        "step_count": step_count,
        "simulation_output_path": executer.simulation_output_path,
    }
    write_csv_line(result_filepath, result)
    return result, executer.simulation_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_dim", type=int, default=6, help="Action dimension")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument(
        "--n_timesteps", type=int, default=100, help="Number of timesteps"
    )
    parser.add_argument("--action_horizon", type=int, default=1, help="Action horizon")
    parser.add_argument(
        "--observation_dim", type=int, default=57, help="Observation dimension"
    )
    parser.add_argument(
        "--prediction_horizon", type=int, default=16, help="Prediction horizon"
    )
    parser.add_argument(
        "--observation_horizon", type=int, default=2, help="Observation horizon"
    )
    parser.add_argument(
        "--dual_agent_model",
        type=str,
        default="runs/plain_diffusion/mini_custom_diffusion_2.pth",
        help="Dual agent model path",
    )
    parser.add_argument(
        "--single_agent_model",
        type=str,
        default="runs/plain_diffusion/mini_custom_diffusion_1.pth",
        help="Single agent model path",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="cbs",
        choices=["cbs"],
        help="Search algorithm to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for the application (in seconds)",
    )
    args = parser.parse_args()

    parameters = Parameters(
        search=args.search,
        timeout=args.timeout,
        action_dim=args.action_dim,
        num_samples=args.num_samples,
        n_timesteps=args.n_timesteps,
        action_horizon=args.action_horizon,
        observation_dim=args.observation_dim,
        dual_agent_model=args.dual_agent_model,
        single_agent_model=args.single_agent_model,
        prediction_horizon=args.prediction_horizon,
        observation_horizon=args.observation_horizon,
    )

    parent_dir = Path(parameters.dual_agent_model).parent
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = parent_dir.as_posix() + "/results" + "_" + timestr
    recorder_dir = parent_dir.as_posix() + "/simulation" + "_" + timestr
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(recorder_dir):
        os.makedirs(recorder_dir)

    num_valids = 0
    num_successes = 0
    experiment_id = 0
    num_experiments = 100
    with tqdm(
        total=num_experiments - experiment_id, dynamic_ncols=True, desc="Running Application"
    ) as pbar:
        while experiment_id < num_experiments:
            result, simulation_pkl_path = demo_with_seed(
                seed=experiment_id,
                result_dir=result_dir,
                recorder_dir=recorder_dir,
                parameters=parameters,
            )
            info = result["info"]
            # print(f"Info: {info}:")
            if info is not None:
                robot_collided = info[1] == "ur5_robotiq" and info[2] == "ur5_robotiq"
                robot_collided = robot_collided or info[2] == "plane"
                is_valid = robot_collided or result["success"]
            else:
                is_valid = True
            num_valids += int(is_valid)
            num_successes += int(result["success"])
            if is_valid:
                print(f'Experiment {result["experiment"]}:')
                print(f'\tSuccess: {result["success"]}')
                print(f"\tPath: {simulation_pkl_path}")
            pbar.update()
            experiment_id += 1
            if num_valids > 0:
                pbar.set_description(f"Success Rate: {num_successes/num_valids:.04f}")
    print(f"Success Rate: {num_successes/num_valids:.04f}")
    print(f"Total Valid: {num_valids}")
    print(f"Total Success: {num_successes}")
    print(f"Total Experiments: {num_experiments}")
