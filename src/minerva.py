from typing import Optional
from quality import NPPD_4K, NPPD_HD, VMAF_4K, VMAF_HD, VMAF_PHONE, QoE_POINT_CLOUD, to_bits
import numpy as np


def interpolate_utility(utility_dict, pseudo_key):
    leq_key = None
    geq_key = None
    # find closest keys in utility dict
    for real_key in utility_dict:
        if real_key <= pseudo_key and (leq_key is None or real_key > leq_key):
            leq_key = real_key
        elif real_key >= pseudo_key and (geq_key is None or real_key < geq_key):
            geq_key = real_key

    if leq_key == geq_key and leq_key is not None:
        return utility_dict[leq_key]

    if geq_key is None:
        # edge case: assume that value is 1 after exceeding bounds
        geq_key = pseudo_key
        geq_val = 1
        # return utility_dict[leq_key]
    else:
        geq_val = utility_dict[geq_key]

    if leq_key is None:
        # edge case: assume that 0 bandwidth has quality 0
        leq_key = 0
        leq_val = 0
        # return utility_dict[geq_key]
    else:
        leq_val = utility_dict[leq_key]

    # linear interpolation based on distance to pseudo key
    diff = geq_key - leq_key
    leq_weight = (geq_key - pseudo_key) / diff
    geq_weight = (pseudo_key - leq_key) / diff
    return leq_weight * leq_val + geq_weight * geq_val


def calc_rates(total_bw, weights):
    bw_distribution = []
    total_demand = sum(weights)
    if total_demand == 0 or total_bw == 0:
        return np.zeros(len(weights))
    for bw_demand in weights:
        bw_percent = bw_demand / total_demand
        bw_distribution.append(total_bw * bw_percent)
    return np.array(bw_distribution)


def minerva_weights(utilities, total_bw: float, init_weights: Optional[np.ndarray] = None, stop_epsilon=1e-5, max_iterations=1_000, clip=(0.5, 20), verbose=False) -> np.ndarray:
    """
    Implements the basic idea of the minerva weight selection from 
    "End-to-end transport for video QoE fairness" by Nathan et al.
    https://dl.acm.org/doi/10.1145/3341302.3342077.

    Args:
        utilities: bitrate-quality utilities (WARNING: should be in Mbit/s, bit/s leads to instability)
        total_bw: total bandwidth
        init_weights: initial weights for all clients. Defaults to None.
        stop_epsilon: stop algorithm if weights change less than stop_epsilon. Defaults to 1e-5.
        max_iterations: max number of iterations. Defaults to 1_000.
        clip: clip the weights by a tuple (min, max). Defaults to None.
        verbose: print status information for each iteration. Defaults to False.

    Returns:
        Weights for all clients that lead to download rates with equal (interpolated) utility
    """
    if init_weights is None:
        init_weights = np.ones(len(utilities))

    weights = init_weights.copy()
    for i in range(max_iterations):
        rates = calc_rates(total_bw, weights)
        rate_utilities = np.array([interpolate_utility(u, r) for u, r in zip(utilities, rates)])
        old_weights = weights
        # clip to avoid division by 0 (will be clipped by max if clip param is not None)
        weights = rates / np.clip(rate_utilities, a_min=0.00001, a_max=np.inf)
        if clip is not None:
            weights = np.clip(weights, a_min=clip[0], a_max=clip[1])
        if verbose:
            print(f"It {i}: weights {old_weights}")
            print(f"It {i}: rates {rates}")
            print(f"It {i}: rate-utilities {rate_utilities}")
        if stop_epsilon is not None and np.sum(np.abs(weights - old_weights)) < stop_epsilon:
            break

    return weights


if __name__ == "__main__":
    utilities = [VMAF_PHONE, VMAF_HD, VMAF_4K, QoE_POINT_CLOUD]
    for i, u in enumerate(utilities):
        # WARNING: Using bytes is numerically unstable
        # utilities[i] = to_bytes(u)
        print(f"Client {i} utilities: {u}")

    weights = minerva_weights(utilities, 15, verbose=True)
    weights = minerva_weights(utilities, 15.5, init_weights=weights, verbose=True, stop_epsilon=1e-2)
    # weights = minerva_weights(utilities, 100, init_weights=None, verbose=True)
    # weights = minerva_weights(utilities, 50, init_weights=weights, verbose=True)
    # weights = minerva_weights(utilities, 30, init_weights=weights, verbose=True)
    # weights = minerva_weights(utilities, 15, init_weights=weights, verbose=True)
    # weights = minerva_weights(utilities, 1, init_weights=weights, verbose=True)
    # weights = minerva_weights(utilities, 5, init_weights=weights, verbose=True)
    # weights = minerva_weights(utilities, 10, init_weights=weights, verbose=True)
