import math
import torch
import torch.nn as nn
from typing import Union
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim:                  Dimension of actions.
        global_cond_dim:            Dimension of global conditioning applied with FiLM in addition to diffusion
                                    step embedding. This is observation_horizon * observation_dimensions
        diffusion_step_embed_dim:   Size of positional encoding for diffusion iteration k
        down_dims:                  Channel size for each UNet level. The length of this array determines the
                                    number of levels.
        kernel_size:                Convolution kernel size
        n_groups:                   Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x:           (B,T,input_dim)
        timestep:    (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output:      (B,T,input_dim)
        """
        # (B, T, C)
        sample = sample.moveaxis(-1, -2)
        # (B, C, T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:  # type: ignore
            timesteps = timesteps[None].to(sample.device)  # type: ignore
        timesteps = timesteps.expand(sample.shape[0])  # type: ignore

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)  # type: ignore

        x = sample
        h = []
        for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # type: ignore
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for _, (resnet, resnet2, upsample) in enumerate(self.up_modules):  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, C, T)
        x = x.moveaxis(-1, -2)
        # (B, T, C)
        return x


class DiffusionActor(object):
    def __init__(self, observation_dim, action_dim, network_config):
        self.action_dim = action_dim
        self.network_config = network_config
        self.observation_dim = observation_dim
        self.noise_predictor_network = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=observation_dim,
            down_dims=network_config["unet_layers"],
            diffusion_step_embed_dim=network_config["time_dim"],
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=network_config["n_timesteps"],
            beta_schedule=network_config["beta_schedule"],
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.ema_model = EMAModel(
            parameters=self.noise_predictor_network.parameters(), power=0.75
        )

    @torch.no_grad()
    def sample_action(self, noise, observations):
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_predictor_network(
                sample=noise, timestep=k, global_cond=observations
            )
            noise = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noise  # type: ignore
            ).prev_sample  # type: ignore
        return noise

    def loss(self, observations, actions):
        device = actions.device
        noise = torch.randn_like(actions, device=device, dtype=actions.dtype)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config["num_train_timesteps"],
            (len(actions),),
            device=device,
            dtype=torch.long,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore
        noise_pred = self.noise_predictor_network(
            noisy_actions, timesteps, global_cond=observations
        )
        return nn.functional.mse_loss(noise_pred, noise)


class DiffusionCritic(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        horizon_config,
        network_config,
        activation_func=nn.Mish,
    ):
        super(DiffusionCritic, self).__init__()
        self.action_dim = action_dim
        self.activation_func = activation_func
        self.network_config = network_config
        self.observation_dim = observation_dim
        self.prediction_horizon = horizon_config["prediction_horizon"]

        self.q1_model = nn.Sequential(*self.get_layers())
        self.q2_model = nn.Sequential(*self.get_layers())

    def get_layers(self):
        mlp_input_dim = self.observation_dim
        if len(self.network_config["mlp_layers"]) == 0:
            return [nn.Linear(mlp_input_dim, 1)]
        else:
            layers = [
                nn.Linear(mlp_input_dim, self.network_config["mlp_layers"][0]),
                self.activation_func(),
            ]
            for i in range(len(self.network_config["mlp_layers"]) - 1):
                layers.append(
                    nn.Linear(
                        self.network_config["mlp_layers"][i],
                        self.network_config["mlp_layers"][i + 1],
                    )
                )
                layers.append(self.activation_func())
            layers.append(nn.Linear(self.network_config["mlp_layers"][-1], 1))
        return layers

    def forward(self, obs, actions):
        """
        obs: (B, observation_dim)
        actions: (B, prediction_horizon, action_dim)
        observation_dim: observation_dimensions * observation_horizon * num_agents
        """
        obs_expanded = obs.unsqueeze(1).expand(-1, self.prediction_horizon, -1)
        x = torch.cat([obs_expanded, actions], dim=-1)
        q1_op = self.q1_model(x.view(-1, x.shape[-1])).view(
            -1, self.prediction_horizon, 1
        )
        q2_op = self.q2_model(x.view(-1, x.shape[-1])).view(
            -1, self.prediction_horizon, 1
        )
        return q1_op, q2_op

    def q1(self, obs, actions):
        obs_expanded = obs.unsqueeze(1).expand(-1, self.prediction_horizon, -1)
        x = torch.cat([obs_expanded, actions], dim=-1)
        return self.q1_model(x.view(-1, x.shape[-1])).view(
            -1, self.prediction_horizon, 1
        )

    def q2(self, obs, actions):
        obs_expanded = obs.unsqueeze(1).expand(-1, self.prediction_horizon, -1)
        x = torch.cat([obs_expanded, actions], dim=-1)
        return self.q2_model(x.view(-1, x.shape[-1])).view(
            -1, self.prediction_horizon, 1
        )

    def q_min(self, obs, actions):
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        return torch.min(q1, q2)
