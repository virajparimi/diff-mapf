import os
import pickle
import numpy as np
from tabulate import tabulate
from tensorboardX import SummaryWriter


class Logger:
    def __init__(
        self,
        logdir,
        benchmark_mode=False,
        benchmark_name=None,
    ):
        self.benchmark_mode = benchmark_mode
        self.benchmark_name = benchmark_name
        if self.benchmark_mode:
            self.logdir = logdir
            self.benchmark_scores = []
            benchmark_name = (
                self.benchmark_name
                if self.benchmark_name is not None
                else "benchmark_score"
            )
            output_path = self.logdir + "/{}.pkl".format(benchmark_name)
            if os.path.exists(output_path):
                print(
                    "[Logger] Loading benchmark scores from {}".format(output_path)
                )
                self.benchmark_scores = pickle.load(open(output_path, "rb"))
        else:
            self.writer = SummaryWriter(logdir)
            self.stats_history = {}
            self.success_rate_history = []

    def get_logdir(self):
        if self.benchmark_mode:
            return None
        return self.writer.logdir

    def save(self):
        benchmark_name = (
            self.benchmark_name
            if self.benchmark_name is not None
            else "benchmark_score"
        )
        output_path = self.logdir + "/{}.pkl".format(benchmark_name)
        print("[Logger] Saving benchmark scores to {}".format(output_path))
        pickle.dump(self.benchmark_scores, open(output_path, "wb"))

    def at_exit(self):
        if self.benchmark_mode:
            self.save()

    def get_average_success_rate(self):
        if len(self.success_rate_history) < 50:
            return 0.0
        return np.mean(self.success_rate_history)

    def print_summary(self):
        for key in self.benchmark_scores[0].keys():
            if key == "task" or key == "debug":
                continue
            scores = [score[key] for score in self.benchmark_scores]
            print(key)
            print("\tMean: {}".format(np.mean(scores)))
            print("\tStd: {}".format(np.std(scores)))

    def add_stats(self, stats):
        if self.benchmark_mode:
            self.benchmark_scores.append(stats)
            self.save()
            if len(self.benchmark_scores) % 50 == 0:
                self.print_summary()
            return
        for key in stats:
            if key not in self.stats_history:
                self.stats_history[key] = []
            self.stats_history[key].append(stats[key])
            if key == "success":
                self.success_rate_history.append(stats[key])
                if len(self.success_rate_history) > 100:
                    self.success_rate_history.pop(0)
                self.stats_history["recent_average_success_rate"] = (
                    self.get_average_success_rate()
                )
            self.success_rate_history = []

    def add_scalars(self, scalars, timestamp):
        if self.benchmark_mode:
            return
        for key in scalars:
            self.writer.add_scalar(key, scalars[key], timestamp)

        metric_table = []
        headers = ["Metric", "Value"]
        for key in self.stats_history:
            if (
                key != "recent_average_success_rate"
                and len(self.stats_history[key]) == 0
            ):
                continue
            elif key == "recent_average_success_rate":
                self.writer.add_scalar(key, self.stats_history[key], timestamp)
                metric_table.append([key, self.stats_history[key]])
                continue
            metric_table.append([key, np.mean(self.stats_history[key])])
            self.writer.add_scalar(
                key + "/mean", np.mean(self.stats_history[key]), timestamp
            )
            self.writer.add_scalar(
                key + "/std", np.std(self.stats_history[key]), timestamp
            )
        if len(self.stats_history) > 0:
            print(tabulate(metric_table, headers=headers, tablefmt="grid"))
        self.stats_history = {}
