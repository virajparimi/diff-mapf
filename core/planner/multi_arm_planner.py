import torch
import numpy as np
from tqdm import tqdm
from collections import deque


from core.environment.utils import Target
from core.recorder import PybulletRecorder
from core.environment.tasks import TaskLoader
from core.planner.cbs import ConflictBasedSearch
from core.utils import prepare_logger, setup_problem
from core.planner.agent_planners import Agent, ResolveDualConflict
from core.environment.benchmarkMultiArmEnv import BenchmarkMultiArmEnv

LOCAL_OBS_INDEX = -1


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device, args, config = setup_problem()
    assert args.mode == "benchmark", "[Planner] Error: --mode must be benchmark"
    logger = prepare_logger(args, config)

    env_config = config["environment"]
    testing_config = config["testing"]
    training_config = config["training"]
    parameters = testing_config["parameters"]

    assert args.tasks_path is not None, "[Planner] Error: --tasks_path is required"

    task_loader = TaskLoader(
        root_dir=args.tasks_path, repeat=args.mode != "benchmark", shuffle=False
    )
    training_config["task_loader"] = task_loader

    num_agents = args.num_agents
    env_config["min_arms_count"] = num_agents
    env_config["max_arms_count"] = num_agents

    env_config["arm_type"] = args.arm_type

    if args.task_difficulty == "easy":
        env_config["min_task_difficulty"] = 0
        env_config["max_task_difficulty"] = 0.35
    elif args.task_difficulty == "medium":
        env_config["min_task_difficulty"] = 0.35
        env_config["max_task_difficulty"] = 0.45
    elif args.task_difficulty == "hard":
        env_config["min_task_difficulty"] = 0.45
        env_config["max_task_difficulty"] = 0.50
    else:
        raise ValueError("Invalid task difficulty")

    parameters["num_samples"] = args.num_samples
    parameters["n_timesteps"] = args.num_timesteps

    environment = BenchmarkMultiArmEnv(
        env_config=env_config,
        training_config=training_config,
        gui=args.gui,
        logger=logger,
    )

    recorder = PybulletRecorder()
    for arm in environment.all_arms:
        recorder.register_object(
            body_id=arm.body_id, urdf_path=arm.arm_urdf, color=arm.color
        )
        if arm.end_effector is not None:
            recorder.register_object(
                body_id=arm.end_effector.body_id,
                urdf_path=arm.end_effector.urdf,
                color=arm.color,
            )
    assert environment.task_manager is not None
    for idx, target in enumerate(environment.task_manager.target_visuals):
        if isinstance(target, Target):
            recorder.register_visual_only(
                name=f"target_{idx}",
                body_id=target.body_id,
                mesh_path="assets/cylinder.obj",
                mesh_scale=[target.radius, target.radius, 0.1],
                color=target.color,
            )

    single_agent_planners = {}
    for arm_id in range(num_agents):
        arm = environment.all_arms[arm_id]
        single_agent_planners[arm_id] = Agent(id=arm_id, arm=arm, parameters=parameters)

    dual_agent_planner = ResolveDualConflict(
        environment.all_arms,
        environment.get_observation,
        environment.preprocess_observation,
        parameters,
    )

    testing_config["eval"]["num_tests"] = args.eval_tests
    assert logger is not None, "[Planner] Error: logger is None"
    for test_id in tqdm(range(testing_config["eval"]["num_tests"])):

        episode_search_metrics = {
            "num_expanded": [],
            "num_generated": [],
            "planning_time": [],
            "num_collision_checks": [],
        }

        observations = environment.reset()
        if test_id < len(logger.benchmark_scores):
            continue

        agents_state_deque = deque(
            [environment.get_state()] * parameters["observation_horizon"],
            maxlen=parameters["observation_horizon"],
        )
        current_task = environment.get_current_task()

        assert current_task is not None, "No task found"
        for arm_id in range(num_agents):
            single_agent_planners[arm_id].set_task(current_task)

        while not environment.terminate_episode:

            for arm_id in range(num_agents):
                agent_observation = observations[arm_id][LOCAL_OBS_INDEX]
                single_agent_planners[arm_id].update_deque(agent_observation)

            plans = []
            for arm_id in range(num_agents):
                plans.append(single_agent_planners[arm_id].predict_plan())

            if args.search == "cbs":
                solver = ConflictBasedSearch(
                    plans,
                    parameters,
                    (single_agent_planners, dual_agent_planner),
                    sim_steps=environment.simulation_steps_per_action_step,
                    pos_tol=environment.position_tolerance,
                    ori_tol=environment.orientation_tolerance,
                )
                final_plan = solver.find_plans(agents_state_deque)
            elif args.search == "no_search":
                final_plan = [
                    plans[arm_id][0] for arm_id in range(num_agents)
                ], parameters["prediction_horizon"]
            else:
                raise ValueError(f"Unknown search method: {args.search}")
            if final_plan is None:
                print("No valid plan found")
                environment.on_episode_end()
                break
            else:
                final_plan, earliest_collision_time = final_plan
                if args.search != "no_search":
                    for key in episode_search_metrics:
                        if key in solver.metrics:
                            episode_search_metrics[key].append(solver.metrics[key])

            action_horizon = parameters["action_horizon"]
            if (
                earliest_collision_time > action_horizon
                and earliest_collision_time != float("inf")
            ):
                action_horizon = earliest_collision_time

            actions = []
            for arm_id in range(num_agents):
                # Executing the first action_horizon actions of the plan
                action = final_plan[arm_id][:action_horizon]
                actions.append(action)

            for execution_step in range(int(action_horizon)):
                actions_at_step = [action[execution_step] for action in actions]
                actions_at_step = np.stack(actions_at_step, axis=0)
                observations = environment.step(actions_at_step)
                for arm_id in range(num_agents):
                    agent_observation = observations[arm_id][LOCAL_OBS_INDEX]
                    single_agent_planners[arm_id].update_deque(agent_observation)
                agents_state_deque.append(environment.get_state())
                recorder.add_keyframe()

                if environment.terminate_episode:
                    break

        recorder.save(
            path=environment.logger.logdir
            + "/recordings/"
            + args.name
            + f"_{current_task.id}.pkl"
        )
        recorder.reset()
        avg_metrics = {
            key: np.sum(value) if len(value) > 0 else 0
            for key, value in episode_search_metrics.items()
        }
        for key in avg_metrics:
            environment.logger.benchmark_scores[-1][key] = avg_metrics[key]
        environment.logger.save()

    success_rate = sum(
        [float(score["success"]) for score in environment.logger.benchmark_scores]
    )
    success_rate /= len(environment.logger.benchmark_scores)
    print(f"Success Rate: {success_rate:.2f}")
