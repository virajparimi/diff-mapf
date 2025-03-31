import os
import torch
import argparse
from pathlib import Path
from json import dump, load
from datetime import datetime

from core.logger import Logger
from core.models.diffusion import DiffusionLearner
from core.models.diffusionQL import DiffusionQLLearner
from core.models.diffusionNet import DiffusionActor, DiffusionCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collaborative Multi-Arm Trainer using Diffusion Policies"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment",
        default="collaborative_multi_arm",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the configuration file",
        default=None,
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to the policy to load for testing a trained policy or start retraining an old one",
        default=None,
    )
    parser.add_argument(
        "--tasks_path", type=str, help="Path to the tasks to load", default=None
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI", default=False)
    parser.add_argument(
        "--mode",
        choices=["train", "benchmark"],
        default="train",
        help="Mode of the run",
    )
    parser.add_argument(
        "--offline_dataset", type=str, help="Path to the offline dataset"
    )

    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs to train", default=int(1000)
    )

    parser.add_argument(
        "--grad_norm", type=float, help="Gradient norm clipping", default=0.0
    )

    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping",
        default=False,
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        help="Number of agents in the environment",
        default=4,
    )

    parser.add_argument(
        "--task_difficulty",
        type=str,
        help="Difficulty of the tasks",
        default="hard",
    )

    parser.add_argument(
        "--eval_tests",
        type=int,
        help="Number of evaluation tests",
        default=1000,
    )

    parser.add_argument(
        "--search",
        type=str,
        help="Search algorithm to use",
        choices=["cbs", "no_search"],
        default="cbs",
    )

    parser.add_argument(
        "--arm_type",
        type=str,
        help="Type of arm to use",
        choices=["UR5"],
        default="UR5",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of diffusion samples to generate",
        default=10,
    )

    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of diffusion timesteps",
        default=100,
    )

    parser.add_argument(
        "--single_agent_model",
        type=str,
        help="Path to the single agent model",
    )
    parser.add_argument(
        "--dual_agent_model",
        type=str,
        help="Path to the dual agent model",
    )

    args = parser.parse_args()

    if args.mode == "benchmark":
        if args.single_agent_model is None:
            print("[Main] Error: --single_agent_model is required in benchmark mode")
            parser.print_help()
            exit(1)
        if args.dual_agent_model is None:
            print("[Main] Error: --dual_agent_model is required in benchmark mode")
            parser.print_help()
            exit(1)
        if args.config is None:
            args.config = "{}/default.json".format(os.path.dirname(args.single_agent_model))

    if args.mode == "train":
        if args.config is None:
            print("[Main] Error: --config is required in train mode")
            parser.print_help()
            exit(1)

    return args


def load_config(path):
    config = load(open(path))
    return config


def setup_problem():
    device = None
    args = parse_args()
    config = load_config(args.config)

    return device, args, config


def prepare_logger(args, config):
    logger = None
    if args.mode == "train":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = f"runs/{args.name}/{timestamp}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        logger = Logger(logdir=logdir)
        dump(config, open(f"{logdir}/config.json", "w"), indent=4)
    elif args.mode == "benchmark":
        parent_dir = Path(args.single_agent_model).parent
        logger = Logger(
            logdir=(
                parent_dir.as_posix()
                if args.single_agent_model is not None
                else "runs/" + parent_dir.as_posix()
            ),
            benchmark_mode=True,
            benchmark_name=args.name,
        )
    return logger


def get_device():
    if not torch.cuda.is_available():
        print("[Setup] Using CPU")
        return "cpu"
    return "cuda"


def create_agent(training_config, device, logger, args, len_dataloader):
    action_dim = 6
    hyperparameters = training_config["hyperparameters"]
    observation_horizon = hyperparameters["observation_horizon"]
    observation_dim = get_observation_dimensions(training_config["observations"])

    if device is None:
        device = get_device()

    def create_actor():
        actor_observation_dim = observation_dim * observation_horizon * args.num_agents
        return DiffusionActor(
            observation_dim=actor_observation_dim,
            action_dim=action_dim,
            network_config=training_config["network"]["actor"],
        )

    def create_critic():
        critic_action_dim = action_dim
        critic_observation_dim = observation_dim * observation_horizon * args.num_agents
        return DiffusionCritic(
            observation_dim=critic_observation_dim + critic_action_dim,
            action_dim=critic_action_dim,
            horizon_config=hyperparameters,
            network_config=training_config["network"]["critic"],
        )

    policy_key = "single_agent_model" if args.num_agents == 1 else "dual_agent_model"
    if training_config["algo"] == "diffusionQL":
        network_fns = {
            "actor": create_actor,
            "critic": create_critic,
        }
        motion_planning_learner = DiffusionQLLearner(
            policy_key=policy_key,
            network_fns=network_fns,
            algo_config=hyperparameters,
            writer=logger,
            device=device,
            load_path=args.load,
            grad_norm=args.grad_norm,
            early_stop=args.early_stop,
            len_dataloader=len_dataloader,
        )
    elif training_config["algo"] == "diffusionNet":
        network_fns = {
            "actor": create_actor,
        }
        motion_planning_learner = DiffusionLearner(
            policy_key=policy_key,
            network_fns=network_fns,
            algo_config=hyperparameters,
            writer=logger,
            device=device,
            load_path=args.load,
            grad_norm=args.grad_norm,
            early_stop=args.early_stop,
            len_dataloader=len_dataloader,
        )
    else:
        raise NotImplementedError(
            f"Algo {training_config['algo']} not implemented yet!"
        )

    return motion_planning_learner


def get_observation_dimensions(observation_config):
    observation_dim = 0
    for observation_item in observation_config["items"]:
        observation_dim += observation_item["dimensions"] * (
            observation_item["history"] + 1
        )
    return observation_dim


def train_agent(agent, dataloader):
    agent.train(dataloader)
    agent.save()


def exit_handler(exit_handlers):
    print("[Main] Exiting gracefully")
    if exit_handlers is not None:
        for handler in exit_handlers:
            if handler is not None:
                handler()
    exit(0)
