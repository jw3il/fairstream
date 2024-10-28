import collections
import numpy as np
import itertools
from collections import defaultdict
from functools import partial
from itertools import repeat

from metric_callback import WORKER_EXPORT_KEY
from env import StreamingEnv
from traces.load import bw_type
from ray.rllib.evaluation.worker_set import WorkerSet
from tqdm import tqdm


def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def to_plain_python(d):
    if isinstance(d, defaultdict) or isinstance(d, dict):
        d = {k: to_plain_python(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray) or isinstance(d, list):
        d = [to_plain_python(v) for v in d]
    elif isinstance(d, np.number):
        d = d.item()
    return d


def eval_single_episode_fun(bw, bw_name, media_qualities, iters, export):
    def set_env_attributes(env):
        env.set_bandwidths([bw], [bw_name])
        env.set_media_qualities(media_qualities)

    def fun(worker):
        worker.foreach_env(set_env_attributes)
        if export:
            worker.global_vars[WORKER_EXPORT_KEY] = True
        for _ in range(iters):
            # Calling .sample() runs exactly one episode per worker due to how the eval workers are configured.
            worker.sample()
        if export:
            worker.global_vars[WORKER_EXPORT_KEY] = False
        return {
            "metrics": worker.get_metrics(),
            "bw_name": bw_name,
        }

    return fun


def custom_eval_function(algorithm, eval_workers: WorkerSet, env_config, iters, num_export_per_type):
    """
    Evaluation of the passed algorithm on specific traces.
    This function evaluates all traces specified by the config iters times per evaluation worker.
    The evalaution metrics will be aggregated per bandwidth category and agent leading to this structure:
    
    {trace_category:
      {
        "reward": {"mean": x, "values": [...], "min": x, "max": x},
        "episode_length": {"mean": x, "values": [...], "min": x, "max": x},
        "metric": {
            "mean": x,
            "agent_0": {"mean": x, "values": [...]},
            ...
            "agent_n": {"mean": x, "values": [...]}
        }
      }
    }
                             

    
    Args:
        algorithm: The RL algorithm to evaluate.
        eval_workers: The set of evaluation workers.
        env_config: Environment config for the evaluation.
        iters: Number of iterations to run for each setting.

    Returns:
        Nested Dictionary with aggregated evaluation metrics
    """
    data = nested_defaultdict(list, 3)

    media_qualities = env_config["media_qualities"]
    all_bw, all_bw_name = StreamingEnv.load_bandwidths_from_config(env_config)

    available_worker_ids = eval_workers.healthy_worker_ids()
    queries = []

    exported_per_type = defaultdict(lambda: 0)

    # distribute all bandwidths over all workers
    for (bw, bw_name) in zip(all_bw, all_bw_name):
        export = False
        if num_export_per_type > 0:
            type = bw_type(bw_name)
            if exported_per_type[type] < num_export_per_type:
                export = True
                exported_per_type[type] += 1
        queries.append(
            eval_single_episode_fun(bw, bw_name, media_qualities, iters, export)
        )

    remaining = len(queries)
    print(f"Evaluating on {remaining} traces..")
    pbar = tqdm(total=remaining)
    while remaining > 0:
        # submit queries to free workers
        if len(available_worker_ids) > 0 and len(queries) > 0:
            for wid in available_worker_ids.copy():
                if len(queries) <= 0:
                    break
                available_worker_ids.remove(wid)
                # select single worker for query
                eval_workers.foreach_worker_async(
                    queries.pop(),
                    remote_worker_ids=[wid]
                )

        # fetch results async
        fetched = eval_workers.fetch_ready_async_reqs(timeout_seconds=0.01)

        n_results = len(fetched)
        remaining -= n_results
        pbar.update(n_results)

        for (worker_id, result) in fetched:
            # mark worker as free
            available_worker_ids.append(worker_id)

            # collect results
            bw_name = result["bw_name"]
            type = bw_type(bw_name)
            episodes = result["metrics"]

            data[type]["trace"]["names"].append(bw_name)
            data[type]["reward"]["values"].extend(
                [e.episode_reward for e in episodes]
            )
            data[type]["episode_length"]["values"].extend(
                [e.episode_length for e in episodes]
            )

            for episode in episodes:
                for metric, value in episode.custom_metrics.items():
                    data[type][metric]["values"].append(value)

                for (agent_id, policy_id), reward in episode.agent_rewards.items():
                    data[type][f"reward_{agent_id}"]["values"].append(reward)

    pbar.close()

    # compute summary stats
    for type, metrics in data.items():
        for metric in metrics.keys():
            if "values" not in data[type][metric]:
                continue
            vals = data[type][metric]["values"]
            data[type][metric]["mean"] = np.mean(vals)
            data[type][metric]["min"] = np.min(vals)
            data[type][metric]["max"] = np.max(vals)
            data[type][metric]["std"] = np.std(vals)

    return to_plain_python(data)
