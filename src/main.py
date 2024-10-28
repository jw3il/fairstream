import argparse
import json
from pathlib import Path
import time

import ray
import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.tune.logger import TBXLoggerCallback

from ray.rllib.models.catalog import ModelCatalog
from env import StreamingEnv, BandwidthSharing
from policies import GreedyKPolicy, LowestBitratePolicy, HighestBitratePolicy
from metric_callback import callback_factory
from eval_function import custom_eval_function
from model import TorchFrameStackingModel
from mpd import SimpleDictMediaQuality
from quality import VMAF_PHONE, VMAF_HD, VMAF_4K, QoE_POINT_CLOUD, to_bits
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPO


def main():
    parser = argparse.ArgumentParser(description='Streaming agent training')
    # environment config vars
    parser.add_argument('--clients', nargs='*', help='list of clients from ["phone", "hd", "4k", "pcv"]', default=["phone", "hd", "4k", "pcv"])
    parser.add_argument('--bw-sharing', type=str, help='Bandwidth sharing weight mode', default="proportional", choices=[e.value for e in BandwidthSharing])
    parser.add_argument('--agents', nargs='*', help='list of agents from ["ppo", "random", "min", "max", "greedy"]', default=["ppo", "ppo", "ppo", "ppo"])
    parser.add_argument('--greedy-k', default=8, type=int, help='Argument k for the greedy agent')
    parser.add_argument('--segments', default=100, type=int, help="Number of segments considered for each content type.")
    parser.add_argument('--buffer-size', default=10, type=int, help="Buffer size of each client.")
    parser.add_argument('--quality-fairness-coeff', default=0.25, type=float, help="Quality-fairness coefficient alpha (reward = alpha * QoE + (1 - alpha) * fairness)")

    # TODO: Maybe add traces scale factor?
    parser.add_argument('--traces', default='cooked_traces/traces_full_train.tar.gz', type=str, help="Folder or .tar.gz archive with traces used for training.")
    parser.add_argument('--traces-include', nargs='*', help='list of trace substrings to include', default=None)
    parser.add_argument('--traces-exclude', nargs='*', help='list of trace substrings to exclude', default=None)

    parser.add_argument('--traces-val', default='cooked_traces/traces_full_val.tar.gz', type=str, help="Folder or .tar.gz archive with traces used for validation.")
    parser.add_argument('--traces-val-include', nargs='*', help='list of trace substrings to include', default=None)
    parser.add_argument('--traces-val-exclude', nargs='*', help='list of trace substrings to exclude', default=None)

    parser.add_argument('--traces-test', default='cooked_traces/traces_full_test.tar.gz', type=str, help="Folder or .tar.gz archive with traces used for testing.")
    parser.add_argument('--traces-test-include', nargs='*', help='list of trace substrings to include', default=None)
    parser.add_argument('--traces-test-exclude', nargs='*', help='list of trace substrings to exclude', default=None)

    parser.add_argument('--rollout-workers', default=0, type=int, help='Number of rollout workers (a reasonable choice is number of available cpu cores)')
    parser.add_argument('--eval-workers', default=8, type=int, help='Number of evaluation workers (a reasonable choice is number of available cpu cores)')
    parser.add_argument('--eval-interval', default=5, type=int)
    parser.add_argument('--eval-only', default=False, action=argparse.BooleanOptionalAction, help='Only evaluate (no training)')
    parser.add_argument('--eval-validation', default=False, action=argparse.BooleanOptionalAction, help='Whether to use the validation set instead of the test set for the final evaluation')
    parser.add_argument('--test-export-traces', default=0, type=int, help='Number of exported traces at test time (the final eval) for each class')
    parser.add_argument('--from-checkpoint', default=None, type=str, help='Checkpoint used to load results or resume training')

    parser.add_argument('-it', '--iterations', default=10, type=int, 
                        help="""
                            Number of Ray training iterations. Equal to iterations*batchsize steps.\n
                            For training over a fixed amount of steps,  adjust the stop condition from 'training_iteration=...' to 'timesteps_total=...'. (See code comments in train())
                            """)
    parser.add_argument('-o', '--output', default="~/qoe-fair-streaming/results/ray", type=str, 
                        help="""
                            Output directory for the ray training output. 
                            Can be kept the same over multiple experiments.
                            The model, training infos (tensorboard logfile) as well as normal evaluation results will be saved here within one folder per ray run.
                        """)
    parser.add_argument('--comment', default=None, type=str, help='Optional: run comment')
    # seeding
    parser.add_argument('--eval-seed', default=1234, type=int, help='seed for choosing eval traces')
    parser.add_argument('--eval-episode-export', default=False, action=argparse.BooleanOptionalAction, help='Export detailed episode stats during eval')
    parser.add_argument('--trace-seed', default=1234, type=int, help='seed for sampling env traces. \nUsed to keep a specific trace order across multiple experiments.')
    # training vars
    parser.add_argument('--num-gpus', default=0., type=float, help='number of gpus passed to ray')
    parser.add_argument('--learning-rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Float specifying the discount factor of the Markov Decision process')
    parser.add_argument('--train-batch-size', default=4000, type=int, help='training batch size')
    parser.add_argument('--mini-batch', default=128, type=int, help='mini batch size')
    parser.add_argument('--num-sgd-iter', default=30, type=int, help='Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).')
    parser.add_argument('--frame-stacking', default=8, type=int, help='Number of frames that are stacked for one observation (or sequence length when using an LSTM)')
    parser.add_argument('--lstm', default=False, action=argparse.BooleanOptionalAction, help='Whether to wrap the model with an LSTM.')
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction, help='Enable debug (local) mode')
    parser.add_argument('--parameter-sharing', default=True, action=argparse.BooleanOptionalAction, help='Parameter sharing of PPO agents')

    args = parser.parse_args()
    assert len(args.clients) == len(args.agents), \
        f"Number of agents ({len(args.agents)}) must equal "\
        f"number of clients {len(args.clients)}."

    assert args.eval_workers > 0, "At least one eval worker is required."

    ray.init(local_mode=args.debug)
    config = create_config(args)
    register_env("StreamingEnv", lambda config: StreamingEnv(config))
    # IMPORTANT: While it looks like this variable is not used, this line is
    # required as-is for rllib to find the remote actor
    trace_sampler = (  # noqa: F841,E303
        TraceSampler.options(name="trace_sampler").remote(args.trace_seed)
    )
    if args.eval_only:
        if args.from_checkpoint is None:
            algo = config.build()
            name = "init"
        else:
            # TODO: maybe use the policy mapping function from the config
            #       => would allow to evaluate new agents
            algo = Algorithm.from_checkpoint(args.from_checkpoint)
            name = f"checkpoint-{args.from_checkpoint}"
    else:
        results = train(config, args)
        best_result = results.get_best_result()
        algo = Algorithm.from_checkpoint(best_result.checkpoint)
        name = best_result.path

    evaluate(config, args, algo, name)
    ray.shutdown()


def get_experiment_name(args) -> str:
    if not hasattr(get_experiment_name, "time"):
        get_experiment_name.time = time.strftime("%Y_%m_%d_%H_%M_%S")

    name = get_experiment_name.time
    if args.comment is not None:
        name += f"_{args.comment}"
    return name


def get_storage_path(args) -> Path:
    return Path(args.output).expanduser().absolute()


def get_results_path(args) -> Path:
    return get_storage_path(args) / get_experiment_name(args)


@ray.remote
class TraceSampler():
    """
    Ray actor called by the environment to sample random integers used for trace selection.
    """
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, high):
        return self.rng.integers(low=0, high=high)


def create_config(args) -> AlgorithmConfig:
    """
        Creates a ray config given parsed arguments
        :args: arguments from Streming Agent Training argparser
        :returns: ray param_space config (here: for PPO)
    """

    # define policies to be used
    # PolicySpec indicates the usage of the policy of the config, e.g. PPO for PPO config
    # other preimplemented policies for basic experiments: LowestBitratePolicy, HighestBitratePolicy
    # If training of multiple policies is wanted or LowestBitratePolicy, HighestBitratePolicy should be used
    # please also change the .multiagent() part of the ray config
    # LowestBitratePolicy, HighestBitratePolicy should not be in the list of policies to train.
    # Also adjusting the policy mapping funcion might be desired then.

    # Create media qualities from quality mappings
    quality_mapping = {
        "hd": VMAF_HD,
        "phone": VMAF_PHONE,
        "4k": VMAF_4K,
        "pcv": QoE_POINT_CLOUD
    }
    for k, qm in quality_mapping.items():
        # TODO: Instead use bytes in original mapping?
        quality_mapping[k] = to_bits(qm)

    media_qualities = [
        SimpleDictMediaQuality(quality_mapping[name], 1, args.segments)
        for name in list(args.clients)
    ]

    dummy_trace = np.zeros((1, 10))

    # training environment config
    env_config = {
        'traces': Path(args.traces).absolute(),
        'traces_include': args.traces_include,
        'traces_exclude': args.traces_exclude,
        'media_qualities': media_qualities,
        'buffer_size': args.buffer_size,
        'time_interval': 1000,
        'bw_sharing': args.bw_sharing,
        'quality_fairness_coeff': args.quality_fairness_coeff,
        # A TraceSampler (ray actor) with the name "trace_sampler" has to
        # be created to use this flag
        'remote_rng': True,
    }
    if args.eval_only:
        env_config["traces"] = None
        env_config["bandwidths"] = dummy_trace
        env_config["bw_names"] = ["dummy_trace"]

    # evaluation environment config
    # note that we use the validation trace set, not the test trace set
    eval_env_config = {
        'traces': Path(args.traces_val).absolute(),
        'traces_include': args.traces_val_include,
        'traces_exclude': args.traces_val_exclude,
        'media_qualities': media_qualities,
        'buffer_size': args.buffer_size,
        'time_interval': 1000,
        'bw_sharing': args.bw_sharing,
        'quality_fairness_coeff': args.quality_fairness_coeff,
        'explore': True,
        'remote_rng': False,
    }

    if args.eval_only:
        eval_env_config["traces"] = None
        eval_env_config["bandwidths"] = dummy_trace
        eval_env_config["bw_names"] = ["dummy_trace"]

    if args.eval_validation:
        test_traces = Path(args.traces_val).absolute()
    else:
        test_traces = Path(args.traces_test).absolute()

    test_env_config = {
        'traces': test_traces,
        'traces_include': args.traces_test_include,
        'traces_exclude': args.traces_test_exclude,
        'media_qualities': media_qualities,
        'buffer_size': args.buffer_size,
        'time_interval': 1000,
        'bw_sharing': args.bw_sharing,
        'quality_fairness_coeff': args.quality_fairness_coeff,
        'explore': True,
        'remote_rng': False,
    }

    # model config
    if args.lstm:
        model_config = {
            "use_lstm": True,
            "max_seq_len": args.frame_stacking
        }
    elif args.frame_stacking >= 1:
        model_config = {
            "custom_model": "frame_stack_model",
            "custom_model_config": {"num_frames": args.frame_stacking},
        }
    else:
        model_config = {}

    policies={
        "random": PolicySpec(policy_class=RandomPolicy),
        "min": PolicySpec(policy_class=LowestBitratePolicy),
        "max": PolicySpec(policy_class=HighestBitratePolicy),
        "greedy": PolicySpec(policy_class=GreedyKPolicy, config={"k": args.greedy_k}),
    }
    if args.parameter_sharing:
        policies["ppo"] = PolicySpec()
        policies_to_train = ["ppo"]
        policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: args.agents[int(agent_id)]
    else:
        policies_to_train = []
        agent_policy_map = {}
        for i, agent in enumerate(args.agents):
            if agent != "ppo":
                agent_policy_map[i] = agent
                continue
            agent_name = f"ppo-{i}"
            agent_policy_map[i] = agent_name
            policies[agent_name] = PolicySpec()
            policies_to_train.append(agent_name)
        policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: agent_policy_map[int(agent_id)]

    ModelCatalog.register_custom_model("frame_stack_model", TorchFrameStackingModel)

    # Algo Config
    config = (
        PPOConfig()
        .environment(env="StreamingEnv", env_config=env_config)
        .rollouts(num_rollout_workers=args.rollout_workers)
        .resources(num_gpus=args.num_gpus * 0.25, num_cpus_per_worker=0.1)
        .framework("torch")
        .training(
            lr=args.learning_rate,
            gamma=args.gamma,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.mini_batch,
            num_sgd_iter=args.num_sgd_iter,
            vf_clip_param=1_000,
            # grad_clip=1,
            # grad_clip_by="norm",
            # kl_coeff=0,
            # kl_target=0,
            model=model_config)
        .callbacks(
            callback_factory(
                get_results_path(args),
                args.eval_episode_export
            )
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_num_workers=args.eval_workers,
            evaluation_config=PPOConfig.overrides(env_config=eval_env_config),
            custom_evaluation_function=(
                lambda x, y: custom_eval_function(x, y, eval_env_config, 1, 0)
            )
        )
        .multi_agent(
            policies=policies,
            policies_to_train=policies_to_train,
            policy_mapping_fn=policy_mapping_fn
        )
    )
    # also store test env config
    config.test_env_config = test_env_config
    return config


def monkey_patch_init_eval(algo: Algorithm):
    """
    Monkey-patches given algorithm to perform initial evaluation before training.

    Note that this will be logged as part of the FIRST iteration at the corresponding step,
    although technically the evaluation happens at step 0.

    If you have eval_interval=1, the result after the first training iteration will be
    OVERWRITTEN with the initial eval on TENSORBOARD. However, you can find both, the
    true initial eval and the eval after step 0 as .json result files in the run directory.
    """
    def monkey_step(self: Algorithm):
        eval_interval = self.config.evaluation_interval
        force_eval = eval_interval is not None and eval_interval > 0 and self.iteration == 0
        if force_eval:
            eval_results = self._run_one_evaluation(train_future=None)
        results = self.original_step()
        if force_eval:
            results.update(eval_results)
        return results

    algo.original_step = algo.step
    algo.step = monkey_step
    return algo


def train(config: AlgorithmConfig, args):
    # training with ray tune
    stop = {
        "training_iteration": args.iterations,
        # "timesteps_total": 500_000,
        # "episode_reward_mean": 150,
    }

    run_config = air.RunConfig(
        stop=stop,
        storage_path=get_storage_path(args),
        name=get_experiment_name(args),
        callbacks=[TBXLoggerCallback()],
        checkpoint_config=air.CheckpointConfig(
            num_to_keep=10,
            checkpoint_frequency=100,
            checkpoint_at_end=True
        )
    )

    if args.from_checkpoint is None:
        tuner = tune.Tuner(
            monkey_patch_init_eval(PPO),
            run_config=run_config,
            param_space=config,
        )
    else:
        print("WARNING: Restoring from checkpoint with old (!) config.")
        tuner = tune.Tuner.restore(args.from_checkpoint, "PPO")

    return tuner.fit()


def evaluate(config: AlgorithmConfig, args, algorithm: Algorithm, name):
    # patch test env config
    algorithm.config._is_frozen = False
    algorithm.config.env_config = config.test_env_config
    algorithm.config.custom_evaluation_function = \
        lambda x, y: custom_eval_function(x, y, config.test_env_config, 1, args.test_export_traces)

    final_eval_metrics = algorithm.evaluate()

    results_path = get_results_path(args)
    results_path.mkdir(exist_ok=True, parents=True)
    filename = results_path / "evalresults_best_test.json"
    # save path of best run (important if there are multiple runs)
    final_eval_metrics["name"] = name
    with open(filename, 'w') as f:
        json.dump(final_eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
