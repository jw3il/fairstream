import random
import numpy as np
from quality import VMAF_4K, VMAF_HD, VMAF_PHONE, QoE_POINT_CLOUD, to_bits
from env import BandwidthSharing, StreamingEnv
from mpd import SimpleDictMediaQuality, HDQuality, FourKQuality
from collections import defaultdict


def display_values(ds, keys):
    """extract keys from a dict and return values per agent"""
    extracted = defaultdict(list)
    used = []
    for d in ds:
        ks = [k for k in keys if k not in used]
        for agent, v in d.items():
            for k in ks:
                if not k in v.keys():
                    continue
                used.append(k)
                if not np.isscalar(v[k]) and len(v[k]) == 1:
                    extracted[agent].append((k, v[k][0]))
                else:
                    extracted[agent].append((k, v[k]))

    for agent, v in extracted.items():
        print(f"Agent {agent}", v)

"""
This script runs the environment with two agents for one episode.
Both agents always choose the highest bitrate (action=5)
The environment is initialized with traces from the folder passed to tracepath
Client resolutions and values to display can be set in the bottom of the script.
"""

def run_episode(env, keys_to_display):
    obs, infos = env.reset()

    done = False
    truncated = False
    total_rewards = {k: 0 for k in obs.keys()}

    print(f"Initial timestamp[ms]: {0}, env step {0} total_bw: {int(env.bandwidths[0]) / 1_000_000}")
    display_values([infos, obs], keys_to_display)

    # simple loop
    step = 0
    while not (done or truncated):
        # create actions
        actions = {}
        for id in obs.keys():
            actions[id] = np.random.randint(0, 6)

        #for k in obs.keys():
        #    print(env.timestamp_ms, k, obs[k], "-->", actions[k])
        print(f"Actions: {actions}")
        print()

        # perform one environment step
        obs, rewards, dones, truncateds, infos = env.step(actions)
        step += 1

        # output requested values
        print(f"timestamp[ms]: {env.time_ms}, env step {step}, total_bw: {int(env.bandwidths[int((env.time_ms-1)/1000)] / 1_000_000):.4f}")
        display_values([infos, obs], keys_to_display)
        print(f"Dones: {dones}")
        print(f"Truncateds: {truncateds}")
        print(f"Rewards: {rewards}")
        print(f"Previous expected events at {env._next_expected_event} (bw at {env.next_bandwidth_ms}) | t = {[s.t for s in env.sessions]}")
        print(f"Dones from step: {dones}")

        # sum up rewards
        for k,v in rewards.items():
            total_rewards[k] += v

        # update termination requirenment
        done = dones["__all__"]
        truncated = sum(truncateds.values())>0

    print("Total Reward", total_rewards)
    print("Dones", dones)
    print("Truncateds", truncateds)


def main(keys_to_display, media_qualities, bandwidths: np.ndarray, bw_sharing: str):
    # Create environment
    env_config = {
        'bandwidths': bandwidths,
        'bw_names': ["static_example"],
        'time_interval': 1000,
        'bw_sharing': bw_sharing,
        'media_qualities': media_qualities,
        'buffer_size': 30,
        'quality_fairness_coeff': 0.25
    }
    env = StreamingEnv(env_config=env_config)
    run_episode(env, keys_to_display)

if __name__=="__main__":
    np.random.seed(42)
    random.seed(42)
    segments_duration = 1
    n_segments = 50
    trace_length = 100
    trace_bw = 15_000_000
    media_qualities = [
        SimpleDictMediaQuality(to_bits(VMAF_PHONE), segments_duration, n_segments),
        SimpleDictMediaQuality(to_bits(VMAF_HD), segments_duration, n_segments),
        SimpleDictMediaQuality(to_bits(VMAF_4K), segments_duration, n_segments),
        SimpleDictMediaQuality(to_bits(QoE_POINT_CLOUD), segments_duration, n_segments),
    ]
    # media_qualities = [SimpleDictMediaQuality(b2q, segments_duration, n_segments)] * 2
    keys_to_display = ["quality", "quality_diff"] # ["init_time", "sim_time", "total_duration", "bw", "sim_step", "bitrate", "quality", "qoe", "qoe_ema", "qoe_ema_corrected", "qoe_ema_uncorrected", "fairness", "buffer", "download_time", "init_time", "rebuffer_time", "t"]


    for i, bw_sharing in enumerate([BandwidthSharing.Minerva]):
        print(f"bw_sharing = {bw_sharing}")
        print()
        main(
            keys_to_display=keys_to_display,
            media_qualities=media_qualities,
            bandwidths=np.random.normal(loc=trace_bw, scale=1_000_000, size=(1, 1000)),
            bw_sharing=bw_sharing
        )
        if i < 1:
            print("-------------------------------------------")
