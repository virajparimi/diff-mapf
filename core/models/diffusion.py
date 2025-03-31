import torch
import numpy as np
from tqdm import tqdm
from time import time
from torch.optim import AdamW
from diffusers.optimization import get_scheduler

from core.models.utils import EarlyStopping
from core.models.basePolicyAlgo import BasePolicyAlgo

DEBUG_LOG = False


class DiffusionLearner(BasePolicyAlgo):
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
        self.lr_decay = algo_config["lr_decay"]

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
        self.optimzer = AdamW(
            self.actor.noise_predictor_network.parameters(),
            lr=self.actor_lr,
            weight_decay=1e-6,
        )

        if self.lr_decay:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimzer,
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
            self.optimzer.load_state_dict(networks["optimizer"])
            self.lr_scheduler.load_state_dict(networks["lr_scheduler"])

            self.stats["update_steps"] = checkpoint["stats"]["update_steps"]

            if "success_rate" in checkpoint["stats"]:
                self.stats["success_rate"] = checkpoint["stats"]["success_rate"]

            print(
                "[DiffusionLearner] Continue training from update steps: {}".format(
                    self.stats["update_steps"]
                )
            )

        self.update_time = time()

    def log_scalars(self, scalars, timestamp):
        self.writer.add_scalars(scalars, timestamp)

    def flatten_obs(self, obs, horizon):
        return obs[:, :horizon, :].flatten(start_dim=1)

    def train_batch(self, batch):

        observations = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)

        observations_cond = self.flatten_obs(observations, self.observation_horizon)

        """Diffusion Model Training"""
        bc_loss = self.actor.loss(observations_cond, actions)
        self.optimzer.zero_grad()
        bc_loss.backward()
        if self.grad_norm > 0.0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.noise_predictor_network.parameters(), max_norm=1.0
            )
        self.optimzer.step()

        self.lr_scheduler.step()
        self.actor.ema_model.step(self.actor.noise_predictor_network.parameters())

        result = {
            "Training/BC_Loss": bc_loss.item(),
        }
        if self.grad_norm > 0.0:
            result["Training/Actor_Grad_Norm"] = actor_grad_norm.item()

        return result

    def train(self, dataloader, return_stats=False):
        scalar_summaries = {
            "Training/BC_Loss": 0.0,
        }
        if self.grad_norm > 0.0:
            scalar_summaries["Training/Actor_Grad_Norm"] = 0.0

        bc_loss = np.inf
        stop_check = EarlyStopping(tolerance=1.0, min_delta=0.0)

        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for _ in tglobal:
                diffusion_loss = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        scalars = self.train_batch(nbatch)
                        for key, value in scalars.items():
                            scalar_summaries[key] += value
                        diffusion_loss.append(scalars["Training/BC_Loss"])
                        tepoch.set_postfix(
                            {
                                "BC Loss": scalars["Training/BC_Loss"],
                            }
                        )

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
                        output = "\r[DiffusionLearner] Update Steps: {}".format(
                            self.stats["update_steps"]
                        )
                        output += " | BC: {:.4f} | Time: {:2f}".format(
                            scalar_summaries["Training/BC_Loss"],
                            float(time() - self.update_time),
                        )
                        self.update_time = time()
                        print(output)

                    if self.early_stop:
                        if stop_check(bc_loss, np.mean(diffusion_loss)):
                            print(
                                "[DiffusionLearner] Early stopping at update steps: {}".format(
                                    self.stats["update_steps"]
                                )
                            )
                            break
                    bc_loss = np.mean(diffusion_loss)

        if return_stats:
            return scalar_summaries

    def get_state_dicts_to_save(self):
        return {
            "optimizer": self.optimzer.state_dict(),
            "ema_model": self.actor.ema_model.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "noise_predictor_network": self.actor.noise_predictor_network.state_dict(),
        }
