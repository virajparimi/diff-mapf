import torch
from torch.utils.data import DataLoader

from core.dataset import MultiArmDataset
from core.utils import (
    train_agent,
    create_agent,
    setup_problem,
    prepare_logger,
)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device, args, config = setup_problem()
    logger = prepare_logger(args, config)

    assert (
        args.offline_dataset is not None
    ), "[DiffusionPolicy] Error: --offline_dataset is required"

    env_config = config["environment"]
    training_config = config["training"]
    hyperparameters = training_config["hyperparameters"]

    dataset = MultiArmDataset(
        dataset_path=args.offline_dataset,
        action_horizon=hyperparameters["action_horizon"],
        pred_horizon=hyperparameters["prediction_horizon"],
        obs_horizon=hyperparameters["observation_horizon"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    motion_planning_learner = create_agent(
        training_config, device, logger, args, len(dataloader)
    )
    train_agent(motion_planning_learner, dataloader)
