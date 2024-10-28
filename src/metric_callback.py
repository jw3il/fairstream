
import numpy as np
import json
import pathlib
from collections import defaultdict

from typing import Dict, Optional, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from pathlib import Path
import pandas as pd


def callback_factory(results_path: Path, export_episodes: bool):
    class MetricCB2(MetricCallback):
        def __init__(self):
            super().__init__(results_path, export_episodes)
    return MetricCB2


WORKER_EXPORT_KEY = "worker_export_episode"


def worker_should_export(worker: RolloutWorker):
    if WORKER_EXPORT_KEY in worker.global_vars and worker.global_vars[WORKER_EXPORT_KEY]:
        return True
    return False


class MetricCallback(DefaultCallbacks):
    sum_metrics = ["quality_switch"]
    """
    Callback to save aggregated metrics and individual episodes.

    Warning: Episode-wise output only works with a single algorithm instance. 
    """
    # here we are only interested in the last entry of the metric
    metric_keys_last = ["t", "total_duration", "sim_time", "sim_step"]

    def __init__(self, results_path: Path, export_episodes: bool, legacy_callbacks_dict=None):
        self.results_path = results_path
        self.export_episodes = export_episodes
        self.eval_iteration = 0
        self.algorithm = None

    def get_path(self, algorithm: Algorithm = None) -> Path:
        if algorithm is None:
            return self.results_path

        # reconstruct ray tune experiment path
        path = self.results_path / Path(algorithm.logdir).name
        path.mkdir(exist_ok=True, parents=True)
        return path

    def on_episode_start(self, *,
                         worker: RolloutWorker,
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: EpisodeV2,
                         env_index,
                         **kwargs) -> None:
        # userdata: {metric: {agent: [values]}}
        episode.user_data = defaultdict(lambda: defaultdict(list))
        if worker.config["in_evaluation"]:
            episode.trace_name = None

    def get_episode_dir(self, iteration) -> Path:
        return self.get_path() / f"eval_episodes_{iteration}"

    def get_current_episode_dir(self) -> Path:
        return self.get_episode_dir(
            self.get_current_eval_iteration_for_eps_export()
        )

    def get_current_eval_iteration_for_eps_export(self) -> Path:
        # TODO: self.eval_iteration is always 0 when trying to export episodes
        #       probably because different instances are used for the calls...
        #       => we manually check where to save the episodes
        eval_nr = 0
        while self.get_episode_dir(eval_nr).exists():
            eval_nr += 1

        return eval_nr - 1

    def get_next_episode_dir(self) -> Path:
        return self.get_episode_dir(
            self.get_current_eval_iteration_for_eps_export() + 1
        )

    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        if self.export_episodes:
            self.get_next_episode_dir().mkdir(exist_ok=True, parents=True)

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        if not worker.config["in_evaluation"]:
            return
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        # fill metric info
        # {
        #   metric_0: {agent_0: ..., ..., metricN},
        #   ...
        #   metric_N: {metric0: ..., ..., metricN}
        # }
        agents = episode.get_agents()
        for agent in agents:
            info = episode._last_infos.get(agent)
            if info is None:
                continue
            for metric, v in info.items():
                if metric == "trace":
                    if info["trace"] is not None and episode.trace_name is None:
                        episode.trace_name = info["trace"]
                    continue

                # init time is only considered in the first step
                if metric == "init_time":
                    if info["t"] == 1:
                        episode.user_data[metric][agent].append(v)
                    continue

                # rebuffer time is considered in all subsequent steps
                if metric == "rebuffer_time":
                    if info["t"] > 1:
                        episode.user_data[metric][agent].append(v)
                    continue

                episode.user_data[metric][agent].append(v)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Convert to simple dict[str, float] ({metric_agent: value})
        for metric, agents in episode.user_data.items():
            for agent, values in agents.items():
                if metric in MetricCallback.metric_keys_last:
                    episode.custom_metrics[f"{metric}_{agent}"] = values[-1]
                    continue

                # only consider quality diff when agents actually switched quality
                if metric == "quality_diff":
                    switches = np.array(episode.user_data["quality_switch"][agent], dtype=np.float32)
                    sum_switches = switches.sum()
                    if sum_switches > 0:
                        masked_diffs_sum = (switches * np.array(values)).sum()
                        val = (masked_diffs_sum / sum_switches)
                    else:
                        val = 0
                    episode.custom_metrics[f"{metric}_{agent}"] = val
                    continue

                episode.custom_metrics[f"{metric}_{agent}"] = np.mean(values)
                if metric in MetricCallback.sum_metrics:
                    episode.custom_metrics[f"{metric}_sum_{agent}"] = np.sum(values)

                episode.hist_data[f"{metric}_{agent}"] = values

        if worker.config["in_evaluation"] and (self.export_episodes or worker_should_export(worker)):
            save_episode(episode, self.get_current_episode_dir())

    def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs) -> None:
        """
        Saves evaluation metrics in file.

        Args:
            algorithm: evaluated algorithm
            evaluation_metrics: collected metrics
        """
        filename = self.get_path(algorithm)\
            / f"eval_results_{self.eval_iteration}_it_{algorithm.iteration}.json"

        with open(filename, "w") as f:
            json.dump(evaluation_metrics, f, indent=2)

        self.eval_iteration += 1


def save_episode(episode, path: Path):
    metrics = list(episode.user_data.keys())
    metrics.remove("sim_time")
    metrics.remove("rebuffer_time")
    metrics.remove("init_time")
    metrics = ["agent", "sim_time"] + metrics + ["rebuffer"]

    agents = episode.user_data["sim_time"].keys()

    all_data = []
    for agent in agents:
        timestamps = episode.user_data["sim_time"][agent]
        agent_id = [int(agent.split("_")[-1]) for _ in timestamps]
        agent_data = [agent_id, timestamps]
        for metric in metrics[2:-1]:
            data = episode.user_data[metric][agent]
            if isinstance(data[0], np.ndarray):
                data = [d.item() for d in data]
            agent_data.append(data)

        all_rebuffer = episode.user_data["init_time"][agent] + episode.user_data["rebuffer_time"][agent]
        agent_data.append(all_rebuffer)

        all_data.append(np.array(agent_data).T)

    df = pd.DataFrame(np.unique(np.concatenate(all_data), axis=0), columns=metrics)
    df['agent'] = df['agent'].astype(int)

    path.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        ef = path / f"{episode.trace_name}_{i}.csv"
        if not ef.exists():
            break
        i += 1
    df.to_csv(ef, index=False)

    #csv_data = np.unique(np.concatenate(all_data), axis=0)
    #np.savetxt(ef, csv_data, delimiter=",", header=",".join(metrics), comments='')
