import torch


class BasePolicyAlgo:
    def __init__(
        self,
        policy_key,
        writer,
        load_path=None,
        save_interval=0,
        device="cuda",
    ):
        self.device = device
        self.writer = writer
        self.policy_key = policy_key
        self.load_path = load_path
        self.save_interval = save_interval

        if self.writer is not None:
            self.logdir = self.writer.get_logdir()

        self.stats = {}

    def get_stats_to_save(self):
        return self.stats

    def get_state_dicts_to_save(self):
        raise NotImplementedError

    def save(self, eval=None, best_model_params=None):
        if self.logdir is not None:
            if eval is not None:
                output_path = "{}/best_ckpt_{}".format(self.logdir, self.policy_key)
                if best_model_params is not None:
                    self.stats["success_rate"] = best_model_params[1]
            else:
                output_path = "{}/ckpt_{}_{:05d}".format(
                    self.logdir,
                    self.policy_key,
                    int(self.stats["update_steps"] / self.save_interval),
                )
            torch.save(
                {
                    "networks": self.get_state_dicts_to_save(),
                    "stats": self.get_stats_to_save(),
                },
                output_path,
            )

            print(
                "[BasePolicyAlgo] {}: Saved model to {}".format(
                    self.policy_key, output_path
                )
            )
