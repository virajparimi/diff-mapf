import torch
import numpy as np
from itertools import islice
from collections import deque
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from core.dataset import normalize_data, unnormalize_data
from core.models.diffusionNet import ConditionalUnet1D, DiffusionCritic
from application.ur5_robotiq_controller import UR5RobotiqPybulletController


class Agent:
    def __init__(self, id, arm, parameters):
        self.id = id
        self.pybullet_id = arm.id if isinstance(arm, UR5RobotiqPybulletController) else arm.body_id
        self.arm = arm
        self.current_task = None
        self.parameters = parameters
        self.observation_deque = deque(maxlen=self.parameters["observation_horizon"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConditionalUnet1D(
            input_dim=self.parameters["action_dim"],
            global_cond_dim=self.parameters["observation_dim"]
            * self.parameters["observation_horizon"],
        )
        _ = self.model.to(self.device)
        ckpt = torch.load(
            self.parameters["single_agent_model"], map_location=self.device
        )
        networks = ckpt["networks"]
        self.model.load_state_dict(networks["noise_predictor_network"])
        if "critic" in networks.keys():
            self.critic = DiffusionCritic(
                observation_dim=self.parameters["observation_dim"]
                * self.parameters["observation_horizon"]
                + self.parameters["action_dim"],
                action_dim=self.parameters["action_dim"],
                network_config={"mlp_layers": [128, 64]},
                horizon_config=self.parameters,
            )
            self.critic.to(self.device)
            self.critic.load_state_dict(networks["critic_target"])
            self.gamma = 0.99
            self.weights = self.gamma * torch.arange(self.parameters["prediction_horizon"], device=self.device).float()
            self.weights = self.weights / self.weights.sum()
        self.noise_scheduler = DDPMScheduler(
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
            num_train_timesteps=self.parameters["n_timesteps"],
        )
        self.noise_scheduler.set_timesteps(self.parameters["n_timesteps"])
        self.stats = np.load(
            self.parameters["single_agent_model"].replace(".pth", ".npz"),
            allow_pickle=True,
        )
        action_stats = dict(self.stats["actions"].flatten()[0])
        observation_stats = dict(self.stats["obs"].flatten()[0])
        self.stats = dict(obs=observation_stats, action=action_stats)

    def update_deque(self, observation):
        if not self.observation_deque:
            self.observation_deque.extend(
                [observation] * self.parameters["observation_horizon"]
            )
        else:
            self.observation_deque.append(observation)

    def set_task(self, task):
        self.current_task = task

    def predict_plan(self):
        # Assumption:
        # Observation is a stacked np array of shape (Obs-Horizon, Obs-Dim)
        observation = np.stack(self.observation_deque)
        observation = normalize_data(observation, stats=self.stats["obs"])
        observation = torch.from_numpy(observation).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # Reshape the observation to (1, Obs-Horizon x Obs-Dim)
            rpt_samples = (
                self.parameters["num_samples"] if not hasattr(self, "critic") else 50
            )
            observation = observation.unsqueeze(0).flatten(start_dim=1)
            observation = observation.repeat(rpt_samples, 1)
            naction = torch.randn(
                (
                    rpt_samples,
                    self.parameters["prediction_horizon"],
                    self.parameters["action_dim"],
                ),
                device=self.device,
            )

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    sample=naction, timestep=k, global_cond=observation
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction  # type: ignore
                ).prev_sample  # type: ignore

            if hasattr(self, "critic"):
                # Output of critic is of the shape (Num-Samples, Predicion-Horizon, Action-Dim)
                q_values = self.critic.q_min(observation, naction)
                q_values = (q_values.squeeze(-1) * self.weights).sum(dim=1)
                best_indices = torch.topk(
                    q_values, k=self.parameters["num_samples"], dim=0
                ).indices
                naction = naction[best_indices]

        # Action shape is going to be (Num-Samples, Pred-Horizon, Act-Dim)
        naction = naction.detach().to("cpu").numpy()
        naction = unnormalize_data(naction, stats=self.stats["action"])
        return naction


class ResolveDualConflict:
    def __init__(self, arms, get_observation_fn, preprocess_observation_fn, parameters):
        self.arms = arms
        self.parameters = parameters
        self.get_observation_fn = get_observation_fn
        self.preprocess_observation_fn = preprocess_observation_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConditionalUnet1D(
            input_dim=self.parameters["action_dim"],
            global_cond_dim=self.parameters["observation_dim"]
            * self.parameters["observation_horizon"]
            * 2,
        )
        _ = self.model.to(self.device)

        ckpt = torch.load(self.parameters["dual_agent_model"], map_location=self.device)
        networks = ckpt["networks"]
        self.model.load_state_dict(networks["noise_predictor_network"])
        if "critic" in networks.keys():
            self.critic = DiffusionCritic(
                observation_dim=self.parameters["observation_dim"]
                * self.parameters["observation_horizon"]
                * 2
                + self.parameters["action_dim"],
                action_dim=self.parameters["action_dim"],
                network_config={"mlp_layers": [128, 64]},
                horizon_config=self.parameters,
            )
            self.critic.to(self.device)
            self.critic.load_state_dict(networks["critic_target"])
            self.gamma = 0.99
            self.weights = self.gamma * torch.arange(self.parameters["prediction_horizon"], device=self.device).float()
            self.weights = self.weights / self.weights.sum()
        self.noise_scheduler = DDPMScheduler(
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
            num_train_timesteps=self.parameters["n_timesteps"],
        )
        self.noise_scheduler.set_timesteps(self.parameters["n_timesteps"])
        self.stats = np.load(
            self.parameters["dual_agent_model"].replace(".pth", ".npz"),
            allow_pickle=True,
        )
        action_stats = dict(self.stats["actions"].flatten()[0])
        observation_stats = dict(self.stats["obs"].flatten()[0])
        self.stats = dict(obs=observation_stats, action=action_stats)
        self.conflict_cache = set()

    def predict_plan(self, conflict, agents_deque):
        # if conflict in self.conflict_cache:
        #     return None
        self.conflict_cache.add(conflict)
        ego_agent, other_agent = conflict
        ego_arm = self.arms[ego_agent]
        other_arm = self.arms[other_agent]

        latest_observations = islice(
            agents_deque,
            max(0, len(agents_deque) - self.parameters["observation_horizon"]),
            len(agents_deque),
        )
        ego_observation_list = []
        for obs_state in latest_observations:
            obs = self.get_observation_fn(ego_arm, obs_state, [other_arm, ego_arm])
            obs = self.preprocess_observation_fn(obs)
            ego_observation_list.append(obs)
        ego_observation = np.concatenate(ego_observation_list, axis=-1)
        ego_observation = normalize_data(ego_observation, stats=self.stats["obs"])
        ego_observation = torch.from_numpy(ego_observation).to(
            self.device, dtype=torch.float32
        )

        with torch.no_grad():
            # Reshape the observation to (1, Obs-Horizon x Obs-Dim x 2)
            rpt_samples = (
                self.parameters["num_samples"] if not hasattr(self, "critic") else 50
            )
            ego_observation = ego_observation.unsqueeze(0).flatten(start_dim=1)
            ego_observation = ego_observation.repeat(rpt_samples, 1)
            naction = torch.randn(
                (
                    rpt_samples,
                    self.parameters["prediction_horizon"],
                    self.parameters["action_dim"],
                ),
                device=self.device,
            )

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    sample=naction, timestep=k, global_cond=ego_observation
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction  # type: ignore
                ).prev_sample  # type: ignore

            if hasattr(self, "critic"):
                # Output of critic is of the shape (Num-Samples, Predicion-Horizon, Action-Dim)
                q_values = self.critic.q_min(ego_observation, naction)
                q_values = (q_values.squeeze(-1) * self.weights).sum(dim=1)
                best_indices = torch.topk(
                    q_values, k=self.parameters["num_samples"], dim=0
                ).indices
                naction = naction[best_indices]

        # Action shape is going to be (Num-Samples, Pred-Horizon, Act-Dim)
        naction = naction.detach().to("cpu").numpy()
        naction = unnormalize_data(naction, stats=self.stats["action"])
        return naction
