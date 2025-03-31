import torch
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from torch.optim import AdamW
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from core.models.utils import EarlyStopping
from core.models.basePolicyAlgo import BasePolicyAlgo

DEBUG_LOG = False


class DiffusionQLLearner(BasePolicyAlgo):
    def __init__(
        self,
        policy_key,
        network_fns,
        algo_config,
        writer,
        device="cuda",
        grad_norm=0.0,
        load_path=None,
        save_interval=1,
        early_stop=False,
        len_dataloader=None,
    ):
        super().__init__(
            policy_key=policy_key,
            writer=writer,
            load_path=load_path,
            save_interval=save_interval,
            device=device,
        )

        self.step = 0
        self.grad_norm = grad_norm
        self.early_stop = early_stop
        self.network_fns = network_fns

        self.actor_lr = algo_config["pi_lr"]
        self.critic_lr = algo_config["q_lr"]
        self.lr_decay = algo_config["lr_decay"]

        self.eta = algo_config["eta"]  # Policy loss QL weight term
        self.tau = algo_config["tau"]  # Polyak averaging coefficient
        self.discount = algo_config["discount"]

        self.save_interval = save_interval
        self.batch_size = algo_config["batch_size"]
        self.num_epochs = algo_config["num_epochs"]

        self.prediction_horizon = algo_config["prediction_horizon"]
        self.observation_horizon = algo_config["observation_horizon"]

        assert len_dataloader is not None
        self.len_dataloader = len_dataloader

        self.setup(load_path)

    def setup(self, load_path):
        self.stats = {"update_steps": 0}

        self.actor = self.network_fns["actor"]()
        self.actor.ema_model.to(self.device)
        self.actor.noise_predictor_network.to(self.device)
        # self.actor.noise_predictor_network = torch.compile(
        #     self.actor.noise_predictor_network
        # )
        self.actor_optimzer = AdamW(
            self.actor.noise_predictor_network.parameters(),
            lr=self.actor_lr,
            weight_decay=1e-6,
        )

        self.critic = self.network_fns["critic"]().to(self.device)
        # self.critic = torch.compile(self.critic)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.critic_optimizer = AdamW(
            self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-6
        )

        if self.lr_decay:
            self.actor_lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.actor_optimzer,
                num_warmup_steps=1000,
                num_training_steps=self.len_dataloader * self.num_epochs,
            )
            self.critic_lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.critic_optimizer,
                num_warmup_steps=1000,
                num_training_steps=self.len_dataloader * self.num_epochs,
            )

        if load_path is not None:
            print("[DiffusionQLLearner] Loading model from {}".format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            networks = checkpoint["networks"]

            self.actor.noise_predictor_network.load_state_dict(
                networks["noise_predictor_network"]
            )
            self.actor.ema_model.load_state_dict(networks["ema_model"])
            self.actor_optimzer.load_state_dict(networks["actor_optimizer"])
            self.actor_lr_scheduler.load_state_dict(networks["actor_lr_scheduler"])

            self.critic.load_state_dict(networks["critic"])
            self.critic_target.load_state_dict(networks["critic_target"])
            self.critic_optimizer.load_state_dict(networks["critic_optimizer"])
            self.critic_lr_scheduler.load_state_dict(networks["critic_lr_scheduler"])

            self.stats["update_steps"] = checkpoint["stats"]["update_steps"]

            if "success_rate" in checkpoint["stats"]:
                self.stats["success_rate"] = checkpoint["stats"]["success_rate"]

            print(
                "[DiffusionQLLearner] Continue training from update steps: {}".format(
                    self.stats["update_steps"]
                )
            )

        self.update_time = time()

    def update_targets(self):
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def log_scalars(self, scalars, timestamp):
        self.writer.add_scalars(scalars, timestamp)

    def flatten_obs(self, obs, horizon):
        return obs[:, :horizon, :].flatten(start_dim=1)

    def train_batch(self, batch):

        observations = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).float()
        next_observations = batch["nobs"].to(self.device)
        is_terminals = batch["is_terminal"].to(self.device).int()

        observations_cond = self.flatten_obs(observations, self.observation_horizon)
        next_observations_cond = self.flatten_obs(
            next_observations, self.observation_horizon
        )

        rewards = torch.unsqueeze(rewards, dim=-1)
        is_terminals = torch.unsqueeze(is_terminals, dim=-1)

        """Q Training"""
        current_q1, current_q2 = self.critic(observations_cond, actions)
        next_action = torch.randn_like(actions, device=self.device)
        next_action = self.actor.sample_action(next_action, next_observations_cond)
        target_q1, target_q2 = self.critic_target(next_observations_cond, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = (rewards + (1.0 - is_terminals) * self.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad()
        # Retained the graph here to potentially fix the actor_loss issue downstream
        # critic_loss.backward(retain_graph=True)
        if self.grad_norm > 0.0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=1.0
            )
        self.critic_optimizer.step()

        """Actor Training"""
        bc_loss = self.actor.loss(observations_cond, actions)
        new_action = torch.randn_like(actions, device=self.device)
        new_action = self.actor.sample_action(new_action, observations_cond)
        q1_new_action, q2_new_action = self.critic(observations_cond, new_action)
        if np.random.uniform() > 0.5:
            q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = bc_loss + self.eta * q_loss
        self.actor_optimzer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0.0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.noise_predictor_network.parameters(), max_norm=1.0
            )
        self.actor_optimzer.step()

        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        self.actor.ema_model.step(self.actor.noise_predictor_network.parameters())

        result = {
            "Training/BC_Loss": bc_loss.item(),
            "Training/QL_Loss": q_loss.item(),
            "Training/Actor_Loss": actor_loss.item(),
            "Training/Critic_Loss": critic_loss.item(),
            "Training/Target_Q_Mean": target_q.mean().item(),
        }
        if self.grad_norm > 0.0:
            result["Training/Actor_Grad_Norm"] = actor_grad_norm.item()
            result["Training/Critic_Grad_Norm"] = critic_grad_norm.item()

        return result

    def train(self, dataloader, return_stats=False):
        scalar_summaries = {
            "Training/BC_Loss": 0.0,
            "Training/QL_Loss": 0.0,
            "Training/Actor_Loss": 0.0,
            "Training/Critic_Loss": 0.0,
            "Training/Target_Q_Mean": 0.0,
        }
        if self.grad_norm > 0.0:
            scalar_summaries["Training/Actor_Grad_Norm"] = 0.0
            scalar_summaries["Training/Critic_Grad_Norm"] = 0.0

        bc_loss = np.inf
        stop_check = EarlyStopping(tolerance=1.0, min_delta=0.0)

        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for _ in tglobal:
                actor_loss = list()
                critic_loss = list()
                diffusion_loss = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        scalars = self.train_batch(nbatch)
                        for key, value in scalars.items():
                            scalar_summaries[key] += value
                        actor_loss.append(scalars["Training/Actor_Loss"])
                        diffusion_loss.append(scalars["Training/BC_Loss"])
                        critic_loss.append(scalars["Training/Critic_Loss"])
                        tepoch.set_postfix(
                            {
                                "Critic Loss": scalars["Training/Critic_Loss"],
                                "Actor Loss": scalars["Training/Actor_Loss"],
                            }
                        )
                        self.update_targets()

                    self.step += 1
                    self.stats["update_steps"] += 1
                    for key in scalar_summaries:
                        scalar_summaries[key] /= self.len_dataloader
                    self.log_scalars(
                        scalar_summaries, timestamp=self.stats["update_steps"]
                    )

                    if self.stats["update_steps"] % self.save_interval == 0:
                        self.save()

                    if DEBUG_LOG:
                        output = "\r[DiffusionQLLearner] Update Steps: {}".format(
                            self.stats["update_steps"]
                        )
                        output += " | BC: {:.4f} | QL: {:.4f} | Actor Loss: {:.4f} | Critic Loss: {:.4f}".format(
                            scalar_summaries["Training/BC_Loss"],
                            scalar_summaries["Training/QL_Loss"],
                            scalar_summaries["Training/Actor_Loss"],
                            scalar_summaries["Training/Critic_Loss"],
                        )
                        output += " | Time: {:.2f}".format(
                            float(time() - self.update_time)
                        )
                        self.update_time = time()
                        print(output)

                    if self.early_stop:
                        if stop_check(bc_loss, np.mean(diffusion_loss)):
                            print(
                                "[DiffusionQLLearner] Early stopping at update steps: {}".format(
                                    self.stats["update_steps"]
                                )
                            )
                            break
                    bc_loss = np.mean(diffusion_loss)

        if return_stats:
            return scalar_summaries

    def get_state_dicts_to_save(self):
        return {
            "critic": self.critic.state_dict(),
            "ema_model": self.actor.ema_model.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimzer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            "noise_predictor_network": self.actor.noise_predictor_network.state_dict(),
        }
