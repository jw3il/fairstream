from dataclasses import dataclass
import dataclasses
import itertools
from random import sample
from typing import Dict, List, NamedTuple, Optional, Tuple
import matplotlib
from matplotlib.ticker import LogFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import z3
import tqdm
from quality import NPPD_4K, NPPD_HD, VMAF_4K, VMAF_HD, VMAF_PHONE, QoE_POINT_CLOUD
import plots.matplotlib_settings as matplotlib_settings
# from ortools.linear_solver import pywraplp
from collections import defaultdict
from plots.constants import plots_dir
import lzma
import pickle

matplotlib_settings.init_plt()
plt.rc("font", **{"family": "serif"})
matplotlib_settings.set_matplotlib_font_size(15, 17, 18)

fairness_dir = plots_dir / "fairness"
fairness_dir.mkdir(exist_ok=True, parents=True)

@np.vectorize
def fairness_real(bw_total, client_br, fixed_br=None, old=False, n_clients=2):
    if bw_total == 0:
        return 0
    
    if fixed_br:
        bw_remaining = bw_total - client_br - fixed_br
        if old:
            bw_cumulative = (n_clients-1) * client_br
        else:
            bw_cumulative = (n_clients-1) * client_br + fixed_br
    else:
        bw_remaining = bw_total - n_clients * client_br
        if old:
            bw_cumulative = client_br
        else:
            bw_cumulative = n_clients * client_br

    if bw_remaining >= 0:
        return bw_remaining/bw_total
    else:
        return bw_remaining/bw_cumulative


@np.vectorize
def nu(bw_total, client_br, b_max=9, b_min=1,  n_clients=2):
    b_fair = n_clients * max(b_min, min(client_br, b_max))
    b_total = bw_total
    b_smax = n_clients * b_max
    b_smin = n_clients * b_min
    b_x = max(b_min, min(client_br, b_max))

    if b_fair >= b_total:
        if b_smax<=b_total:
            return 0
            
        return (b_fair-b_total)/(b_smax-b_total) * (b_x-b_min)/(b_max-b_min)


    return (b_fair-b_total)/(b_total-b_smin) * (b_max-b_x)/(b_max-b_min)


def nu_discrete(b_total, client_bitrate2quality):
    n_clients = len(client_bitrate2quality)

    opt = z3.Optimize()

    client_bitrate_idx = [z3.Int(f'b_idx_{i}') for i in range(n_clients)]
    client_bitrate = [z3.Real(f'b_{i}') for i in range(n_clients)]
    client_quality = [z3.Real(f'q_{i}') for i in range(n_clients)]
    min_quality = z3.Real("min_quality")

    client_x = []
    client_y = []

    for c_1 in range(n_clients):
        x = np.array(list(client_bitrate2quality[c_1].keys()))
        x = np.sort(x)
        client_x.append(x)
        y = np.array([client_bitrate2quality[c_1][k] for k in x])
        client_y.append(y)
        # only choose valid client bitrate indices
        opt.add(0 <= client_bitrate_idx[c_1], client_bitrate_idx[c_1] < len(x))
        # when selecting a client bitrate index,
        # set the corresponding bitrate and quality
        for bitrate_idx, bitrate in enumerate(x):
            opt.add(
                z3.Implies(
                    client_bitrate_idx[c_1] == bitrate_idx,
                    z3.And(
                        client_bitrate[c_1] == bitrate,
                        client_quality[c_1] == y[bitrate_idx]
                    )
                )
            )

        opt.add(min_quality <= client_quality[c_1])

    opt.add(z3.Sum(client_bitrate) <= b_total)

    quality_mean = z3.Sum(client_quality) / n_clients
    quality_var = z3.Sum([(q - quality_mean) ** 2 for q in client_quality]) / n_clients
    quality_abs_diff = z3.Sum([z3.Abs(q - quality_mean) for q in client_quality]) / n_clients

    # NOTE: z3.Sqrt and this trick both have issues in some configurations
    # z3_sample_std = z3.Real("std")
    # opt.add(z3_sample_std * z3_sample_std == sample_var)

    # objective

    # opt.minimize(quality_var)
    opt.maximize(min_quality)
    opt.maximize(quality_mean)
    # opt.maximize(quality_mean - quality_abs_diff)

    opt_check = opt.check()
    found_solution = opt_check == z3.sat

    # for all clients
    solved = True
    res_bitrates = np.zeros(n_clients)
    for c in range(n_clients):
        if found_solution:
            client_br = float(opt.model().evaluate(client_bitrate[c]).as_fraction())
        elif sum([client_x[i][0] for i in range(n_clients)]) > b_total:
            # bandwidth is not sufficient, clients should simply choose lowest quality
            solved = False
            client_br = np.min(client_x[c])
        else:
            unknown_string = f" Reason unknown: {opt.reason_unknown()}" if opt_check == z3.unknown else ""
            raise ValueError(f"Could not find solution: opt.check() == {opt_check}.{unknown_string}")

        res_bitrates[c] = client_br

    return res_bitrates, solved


def fairness_jain(qualities, axis=0):
    return qualities.sum(axis=axis) ** 2 / (len(qualities) * (qualities ** 2).sum(axis=axis))


def fairness_F(qualities, axis=0):
    return 1 - 2 * qualities.std(axis=axis)


def nu_discrete_full_enum(b_total, client_bitrate2quality, alpha=0.5, optimum_selection="max_br", objective="QF"):
    n_clients = len(client_bitrate2quality)

    client_x = []
    client_y = []

    for c_1 in range(n_clients):
        x = np.array(list(client_bitrate2quality[c_1].keys()))
        x = np.sort(x)
        client_x.append(x)
        y = np.array([client_bitrate2quality[c_1][k] for k in x])
        client_y.append(y)

    def constraints_fn(bitrates):
        """
        Condition function based on bitrates of all clients.

        :param bitrates: Bitrates of all clients
        :return: whether the given bitrates satisfy the constraints
        """
        bitrates = np.array(bitrates)
        return bitrates.sum() <= b_total

    def objective_fn(bitrates):
        """
        Objective function based on bitrates of all clients.

        :param bitrates: Bitrates of all clients
        :return: comparable value for given input bitrates
        """
        bitrates = np.array(bitrates)
        qualities = np.array([client_bitrate2quality[c][bitrates[c]] for c in range(n_clients)])

        if objective == "debug":
            # debugging objective, only use for debugging
            return qualities.mean() - 2 * qualities.std()

        # weighted F
        if objective == "alpha-qf":
            return alpha * qualities.mean() + (1 - alpha) * fairness_F(qualities)

        if objective == "alpha-qjain":
            return alpha * qualities.mean() + (1 - alpha) * fairness_jain(qualities)

        if objective == "alpha-qmin":
            return alpha * qualities.mean() + (1 - alpha) * qualities.min()

        if objective == "f,q":
            return fairness_F(qualities), qualities.mean()

        if objective == "fxq":
            return fairness_F(qualities), qualities.mean()

        if objective == "-std,q":
            # std fairness with maximum mean qualities (seldomly considered)
            return -qualities.std(), qualities.mean()

        if objective == "max-min":
            # max-min fairness with maximum mean qualities
            return qualities.min(), qualities.mean()

        raise ValueError(f"No valid objective given: '{objective}'.")

    # simply perform full enumeration with the objective function

    # all optimal bitrates
    best_bitrates = None
    best_obj = None
    solutions = defaultdict(list)

    for bitrates in itertools.product(*client_x):
        if not constraints_fn(bitrates):
            continue

        obj = objective_fn(bitrates)

        if optimum_selection == "none":
            solutions[obj].append(bitrates)
        else:
            if best_obj is not None and obj == best_obj:
                # add solution
                best_bitrates.append(bitrates)

            if best_obj is None or obj > best_obj:
                # initialize new best bitrates
                best_obj = obj
                best_bitrates = [bitrates]

    # select a random optimal solution
    if best_bitrates is not None:
        solution_sample_idx = 0
        if len(best_bitrates) > 1:
            if optimum_selection == "max_br":
                solution_sample_idx = np.argmax(np.array(best_bitrates).sum(axis=-1))
            elif optimum_selection == "min_br":
                solution_sample_idx = np.argmin(np.array(best_bitrates).sum(axis=-1))
            elif optimum_selection == "random":
                solution_sample_idx = np.random.randint(len(best_bitrates))
            else:
                raise ValueError(f"Unknown optimum selection mode {optimum_selection}")
            print(f"WARNING: Problem for b_total={b_total:.2f} has {len(best_bitrates)} optima: {best_bitrates}. Chose idx {solution_sample_idx}.")

    if optimum_selection == "none":
        return solutions

    # for all clients
    solved = 0 if best_bitrates is None else len(best_bitrates)
    res_bitrates = np.zeros(n_clients)
    for c in range(n_clients):
        if best_bitrates is not None:
            # sample best bitrate
            client_br = best_bitrates[solution_sample_idx][c]
        elif sum([client_x[i][0] for i in range(n_clients)]) > b_total:
            # bandwidth is not sufficient, clients should simply choose lowest quality
            client_br = np.min(client_x[c])
        else:
            raise ValueError("Could not find any valid solution.")

        res_bitrates[c] = client_br

    return res_bitrates, solved


def nu_linear(b_total, client_bitrates, clients_q_factors):
    # Assumption: All clients can choose from the same continuous bit rates in range [b_min, b_max] 
    b_min, b_max = np.min(client_bitrates), np.max(client_bitrates)

    # Goal: find fair (and equal) QoE that is less than or equal to the total bandwidth.
    # For the continuous case, we know that the quality can be perfectly fair
    #   Assuming a linear QoE function with q_i * b_i, this means that q_i * b_i = q_j * b_j for all pairs of clients i, j
    # We also know that the total bandwidth can be reached exactly
    #   This means that sum_i b_i = b_total
    # This results in a simple linear system with # num clients linear equations of the form 
    # sum_i n_i * b_i = m_i that has to be solved for parameters b (n and m are given).
    # Example with 3 clients 0,1,2:
    # b_0 b_1  b_2  | ordinate  | comment
    # 1   1    1    | b_total   | total bandwidth
    # q_0 -q_1 0    | 0         | client 0 and 1 have same quality
    # 0   q_1  -q_2 | 0         | client 1 and 2 have same quality
    # if individual bitrates exceed the max/min bitrate, fix them and solve again

    n_clients = len(clients_q_factors)
    max_br_mask = np.zeros(len(clients_q_factors), dtype=bool)
    min_br_mask = np.zeros(len(clients_q_factors), dtype=bool)

    while True:
        active_clients = list(range(len(clients_q_factors)))
        for i in range(len(clients_q_factors)):
            if max_br_mask[i] or min_br_mask[i]:
                active_clients.remove(i)
                
        if len(active_clients) == 0:
            break
            
        coefficient_matrix = np.zeros((len(active_clients), len(active_clients)))
        coefficient_matrix[0, :] = 1
        for idx in range(1, len(active_clients)):
            coefficient_matrix[idx, idx] = -clients_q_factors[active_clients[idx]]
            coefficient_matrix[idx, idx - 1] = clients_q_factors[active_clients[idx] - 1]

        ordinate_values = np.zeros(len(active_clients))
        ordinate_values[0] = b_total - np.sum(max_br_mask) * b_max - np.sum(min_br_mask) * b_min

        # solve equation system
        b_opt_active = np.linalg.solve(coefficient_matrix, ordinate_values)
        b_opt = max_br_mask.astype(float) * b_max + min_br_mask.astype(float) * b_min
        b_opt[active_clients] = b_opt_active
        
        # check if there are changes
        new_max_br_mask = max_br_mask | (b_opt >= b_max)
        new_min_br_mask = min_br_mask | (b_opt <= b_min)
        if np.all(new_max_br_mask == max_br_mask) and np.all(new_min_br_mask == min_br_mask):
            # there are no changes, we are done
            break
        
        # solve reduced equation system with new mask
        max_br_mask = new_max_br_mask
        min_br_mask = new_min_br_mask

    solved = True
    res_bitrates = np.zeros(n_clients)
    for idx in range(n_clients):
        # NOTE: Not sure if this is correct
        solved = solved and b_min <= b_opt[idx] <= b_max
        res_bitrates[idx] = max(b_min, min(b_opt[idx], b_max))

    return res_bitrates, solved


def get_bitrate_distance(bitrates, target_bitrate):
    bitrates = np.array(list(bitrates))
    # result is relative distance to optimal solution of client bitrate
    return (bitrates - target_bitrate) / (bitrates.max() - bitrates.min())


def plot_linear(clients_config, steps):
    """
    Create contour plot for NU metric.

    :param fixed_br: _description_, defaults to None
    :param name: plot name, defaults to "contour_fairness.png"
    :param n_clients: number of clients with equal demand OR list of quality factors (quality = factor * bitrate), defaults to 2
    :param title: figure title, defaults to None
    """
    
    assert isinstance(clients_config, LinearConfig)

    n_clients = len(clients_config.client_q_factors)
    xlims = [clients_config.b_min, clients_config.b_max]
    ylims = [0, (xlims[1] + xlims[0]) * n_clients]
    bitrates = np.arange(xlims[0], xlims[1], (xlims[1] - xlims[0]) / steps)
    bandwidths = np.arange(ylims[0], ylims[1], (ylims[1] - ylims[0]) / steps)

    extent = (xlims[0], xlims[1], ylims[0], ylims[1])
    br, bw = np.meshgrid(bitrates, bandwidths)
    
    # create triples of client_A_br, bw, fairness
    #fairness = fairness_real(bw, br, fixed_br, old, n_clients=n_clients)
    print(f"Generating {clients_config.name}")
    fairness = np.zeros((n_clients, len(bandwidths), len(bitrates)))
    for (j, bw_j) in tqdm.tqdm(enumerate(bandwidths), total=len(bandwidths)):
        res_bitrates, _ = nu_linear(bw_j, bitrates, clients_q_factors=clients_config.client_q_factors)
        for c in range(n_clients):
            fairness[c, j, :] = get_bitrate_distance(bitrates, res_bitrates[c])
    
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    fig, axs = plt.subplots(nrows=1, ncols=n_clients, sharey=True, width_ratios=[1] * n_clients)

    divnorm = colors.TwoSlopeNorm(vcenter=0., vmin=np.min(fairness), vmax=np.max(fairness))
    
    for c in range(n_clients):
        ax = axs[c]
        ims = ax.imshow(fairness[c], interpolation='none', cmap=cm.coolwarm, norm=divnorm, extent=extent, origin='lower', aspect='auto')

        CS = ax.contour(br, bw, fairness[c], colors="k", norm=divnorm, levels=20)
        ax.clabel(CS, fontsize=9, inline=True, inline_spacing=5)
        
        ax.set_xlabel(f"Bitrate client {c}")
        if c == 0:
            ax.set_ylabel("Bandwidth")

    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(ims, shrink=0.5, aspect=5, cax=cbar_ax)
    fig.suptitle(clients_config.name)
    
    plt.tight_layout()
    plt.savefig(fairness_dir / (clients_config.name + ".pdf"), bbox_inches='tight')


def get_client_quality(clients_config, client_id, bitrates):
    assert isinstance(clients_config, DiscreteConfig)
    if not isinstance(bitrates, np.ndarray):
        bitrates = np.array([bitrates])
    
    b2q = clients_config.client_bitrate2quality[client_id]
    client_bitrates = np.array(list(b2q.keys()))
    client_bitrates.sort()

    qualities = np.zeros_like(bitrates)
    for i, b in enumerate(bitrates):
        # get closest bitrate
        dist = np.abs(client_bitrates - b)
        closest_idx = np.argsort(dist)[0]
        qualities[i] = b2q[client_bitrates[closest_idx]]

    return qualities


def plot_discrete(clients_config, steps, figsize=(6.4, 4.8), util_ylim=None, quality_ylim=None, legend=True, with_marker=False):
    """
    Create contour plot for NU metric.

    :param fixed_br: _description_, defaults to None
    :param name: plot name, defaults to "contour_fairness.png"
    :param n_clients: number of clients with equal demand OR list of quality factors (quality = factor * bitrate), defaults to 2
    :param title: figure title, defaults to None
    """
    assert isinstance(clients_config, DiscreteConfig)
    n_clients = len(clients_config.client_bitrate2quality)
    
    client_labels = clients_config.client_labels

    all_bitrates = list(itertools.chain(*[list(bit2qual.keys()) for bit2qual in clients_config.client_bitrate2quality]))
    xlims = [np.min(all_bitrates), np.max(all_bitrates)]
    ylims = [0, (xlims[1] + xlims[0]) * n_clients]
    bandwidths = np.arange(ylims[0], ylims[1], ylims[1] / steps)
    
    # create triples of client_A_br, bw, fairness
    #fairness = fairness_real(bw, br, fixed_br, old, n_clients=n_clients)
    client_bitrates = []
    fairness_metric = []
    opt_bitrates = np.zeros((n_clients, len(bandwidths)))
    min_total_bitrate = 0
    for c in range(n_clients):
        bitrates = np.array(list(clients_config.client_bitrate2quality[c].keys()))
        bitrates = np.sort(bitrates)
        min_total_bitrate += bitrates[0]
        client_bitrates.append(bitrates)
        fairness_metric.append(np.zeros((len(bandwidths), len(bitrates))))

    print(f"Generating {clients_config.name}")
    for (j, bw_j) in tqdm.tqdm(enumerate(bandwidths), total=len(bandwidths)):
        res_bitrates, _ = nu_discrete_full_enum(bw_j, clients_config.client_bitrate2quality, alpha=clients_config.alpha, optimum_selection=clients_config.optimum_selection, objective=clients_config.objective)
        for c in range(n_clients):
            fairness_metric[c][j] = get_bitrate_distance(client_bitrates[c], res_bitrates[c])
            opt_bitrates[c, j] = res_bitrates[c]
            
            
    # calculate F
    client_qualities = []
    for c in range(n_clients):
        client_qualities.append(get_client_quality(clients_config, c, opt_bitrates[c]))
    
    fairness_f_vals = fairness_F(np.array(client_qualities))
        
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    fig, axs = plt.subplots(nrows=1, ncols=n_clients, sharey=True, width_ratios=[1] * n_clients)

    vmin = np.min([np.min(f) for f in fairness_metric])
    vmax = np.max([np.max(f) for f in fairness_metric])
    print(f"debug info: vmin={vmin:.2f}, vmax={vmax:.2f}")
    divnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    
    for c in range(n_clients):
        ax = axs[c]
        ims = ax.imshow(fairness_metric[c], interpolation='none', cmap=cm.coolwarm, norm=divnorm, origin='lower', aspect='auto')
        x_ticks = np.arange(len(client_bitrates[c]))
        ax.set_xticks(x_ticks, [f"{v:.2f}" for v in client_bitrates[c]])
        img_coord_to_bandwidth_factor = ylims[1] / len(bandwidths)
        yticks_stepsize = 2 / (ylims[1] / steps)
        yticks = np.arange(0, len(bandwidths), yticks_stepsize)
        yticks_labels = [f"{v:.2f}" for v in yticks * img_coord_to_bandwidth_factor]
        ax.set_yticks(yticks, yticks_labels)
        ax.set_xlabel(f"Bitrate client {c}")
        if c == 0:
            ax.set_ylabel("Bandwidth")

    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(ims, shrink=0.5, aspect=5, cax=cbar_ax)
    fig.suptitle(clients_config.name)
    
    plt.tight_layout()
    plt.savefig(fairness_dir / (clients_config.name + ".pdf"), bbox_inches='tight')

    plt.clf()

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, height_ratios=[1, 0.5, 1], figsize=figsize)

    all_client_qualities = np.zeros((n_clients, *bandwidths.shape))
    for c in range(n_clients):
        all_client_qualities[c] = get_client_quality(clients_config, c, opt_bitrates[c])

    valid_bandwidths = bandwidths >= min_total_bitrate

    # axs[0].vlines(min_total_bitrate, np.min(opt_bitrates), np.max(opt_bitrates), linestyles="dotted", color="k")
    # axs[2].vlines(min_total_bitrate, np.min(all_client_qualities), np.max(all_client_qualities), linestyles="dotted", color="k")
    client_p = []
    for c in range(n_clients):
        label = f"Client {c}" if client_labels is None else client_labels[c]
        linestyle = "-" if clients_config.client_linestyle is None else clients_config.client_linestyle[c]
        x = bandwidths[valid_bandwidths]
        axs[0].plot(bandwidths[valid_bandwidths], opt_bitrates[c][valid_bandwidths], label=label, linestyle=linestyle)
        p = axs[2].plot(bandwidths[valid_bandwidths], all_client_qualities[c][valid_bandwidths], label=label, linestyle=linestyle, linewidth=0.9 if with_marker else None)
        client_p.append(p)

    if with_marker:
        for c in range(n_clients):
            p = client_p[c]
            y = all_client_qualities[c][valid_bandwidths]
            quality_diff = np.diff(y, 1, prepend=y[0])
            quality_switch_up = quality_diff > 0
            axs[2].scatter(x[quality_switch_up], y[quality_switch_up], marker="^", color=p[0].get_color(), s=10, zorder=200)
            quality_switch_down = quality_diff < 0
            axs[2].scatter(x[quality_switch_down], y[quality_switch_down], marker="v", color=p[0].get_color(), s=10, zorder=199)

    axs[0].set_ylabel("Bitrate [Mbps]", labelpad=8)
    axs[2].set_ylabel("Quality", labelpad=6)
    if quality_ylim is not None:
        axs[2].set_ylim(quality_ylim)

    for c in range(n_clients):
        axs[1].plot([])

    # utilization = np.clip(np.sum(opt_bitrates, axis=0) / np.clip(bandwidths, a_min=0.01, a_max=None), a_min=0, a_max=1) * 100
    axs[1].plot(bandwidths[valid_bandwidths], fairness_f_vals[valid_bandwidths])
    # axs[1].plot(bandwidths[valid_bandwidths], utilization[valid_bandwidths])
    if util_ylim is not None:
        axs[1].set_ylim(util_ylim)
    axs[1].set_ylabel("Fairness", labelpad=6)
    # axs[1].set_ylabel("Utilization [\%]")

    # all_qualities_array = np.stack(all_client_qualities)
    # F = fairness_F(all_qualities_array)
    # mean_quality = all_qualities_array.mean(axis=0)

    # axs[2].plot(bandwidths[valid_bandwidths], F[valid_bandwidths])
    # axs[2].set_ylabel("F")

    # axs[3].plot(bandwidths[valid_bandwidths], mean_quality[valid_bandwidths])
    # axs[3].set_ylabel("Mean Quality")
    # axs[3].set_ylim((0.5, 1))

    axs[-1].set_xlabel("Bandwidth [Mbps]")
    if legend:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fairness_dir / (clients_config.name + "_line.pdf"), bbox_inches='tight')


def plot_discrete_all_alpha(clients_config, steps):
    assert isinstance(clients_config, DiscreteConfig)
    assert "alpha" in clients_config.objective
    n_clients = len(clients_config.client_bitrate2quality)
    
    all_bitrates = list(itertools.chain(*[list(bit2qual.keys()) for bit2qual in clients_config.client_bitrate2quality]))
    xlims = [np.min(all_bitrates), np.max(all_bitrates)]
    ylims = [0, (xlims[1] + xlims[0]) * n_clients]
    bandwidths = np.arange(ylims[0], ylims[1], ylims[1] / steps)
    alphas = np.arange(0, 1 + 0.5 / steps, 1 / steps)  

    min_total_bitrate = 0
    for c in range(n_clients):
        bitrates = np.array(list(clients_config.client_bitrate2quality[c].keys()))
        bitrates = np.sort(bitrates)
        min_total_bitrate += bitrates[0]

    res_bitrates = np.zeros((n_clients, len(bandwidths), len(alphas)))
    res_quality = np.zeros((n_clients, len(bandwidths), len(alphas)))
    res_num_optima = np.zeros((len(bandwidths), len(alphas)))
    for (j, bw_j), (k, alpha_k) in tqdm.tqdm(itertools.product(enumerate(bandwidths), enumerate(alphas)), total=len(bandwidths) * len(alphas)):
        res_bitrates[:, j, k], res_num_optima[j, k] = nu_discrete_full_enum(bw_j, clients_config.client_bitrate2quality, alpha=alpha_k, optimum_selection=clients_config.optimum_selection, objective=clients_config.objective)
        for c in range(n_clients):
            res_quality[c, j, k] = get_client_quality(clients_config, c, res_bitrates[c, j, k])

    xticks_stepsize = int(0.2 / (1 / steps))
    xticks = np.arange(0, len(alphas), xticks_stepsize)
    xticks_labels = [f"{v:.1f}" for v in alphas[xticks]]

    img_coord_to_bandwidth_factor = ylims[1] / len(bandwidths)
    yticks_stepsize = 10 / (ylims[1] / steps)
    yticks = np.arange(0, len(bandwidths), yticks_stepsize)
    yticks_labels = [f"{round(v):d}" for v in yticks * img_coord_to_bandwidth_factor]

    def masked_feasible(np_arr):
        return np.ma.masked_array(np_arr, mask=res_num_optima == 0)

    def enable_yticks():
        plt.gca().tick_params(axis='y', which='major', size=4)
        plt.gca().tick_params(axis='y', which='major', width=1)

    cmap = matplotlib.cm.viridis
    cmap.set_bad(color='white')

    plt.clf()
    plt.figure()
    plt.grid(False)
    im = plt.imshow(masked_feasible(res_quality.mean(axis=0)), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    plt.xlabel("Weight $\\alpha$")
    plt.ylabel("Bandwidth [Mbps]")
    # plt.axhline(y=min_total_bitrate / img_coord_to_bandwidth_factor, color='white', linestyle='--')
    plt.xticks(xticks, xticks_labels)
    plt.yticks(yticks, yticks_labels)
    enable_yticks()
    col = plt.colorbar(im)
    col.set_label("Mean quality")
    plt.savefig(fairness_dir / (clients_config.name + "_mean_quality.pdf"), bbox_inches="tight")
    
    res_f_measures = {}
    
    for f_measure_name, f_measure_label, f_measure_fun in [("F", "Fairness", fairness_F), ("jain", "Jain's Index", fairness_jain), ("min", "Min Quality", lambda q: q.min(axis=0))]:
        plt.clf()
        plt.grid(False)
        f_values = f_measure_fun(res_quality)
        res_f_measures[f_measure_name] = f_values
        im = plt.imshow(masked_feasible(f_values), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
        plt.xlabel("Weight $\\alpha$")
        plt.ylabel("Bandwidth [Mbps]")
        plt.xticks(xticks, xticks_labels)
        plt.yticks(yticks, yticks_labels)
        enable_yticks()
        # plt.axhline(y=min_total_bitrate / img_coord_to_bandwidth_factor, color='white', linestyle='--')
        col = plt.colorbar(im)
        col.set_label(f"{f_measure_label}")
        plt.savefig(fairness_dir / (clients_config.name + "_fairness_" + f_measure_name + ".pdf"), bbox_inches="tight")

    with lzma.open(fairness_dir / (clients_config.name + f"_res_{steps}.pk.lz"), "wb") as f:
        pickle.dump(
            {
                "bandwidths": bandwidths,
                "alphas": alphas,
                "res_bitrates": res_bitrates,
                "res_quality": res_quality,
                "res_num_optimal": res_num_optima,
                "res_f_measures": res_f_measures,
            },
            f
        )

    plt.clf()
    plt.grid(False)
    n = int(res_num_optima.max()) + 1
    cmap = plt.get_cmap("viridis", n)
    im = plt.imshow(res_num_optima, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    plt.xlabel("Weight $\\alpha$")
    plt.ylabel("Bandwidth [Mbps]")
    plt.xticks(xticks, xticks_labels)
    plt.yticks(yticks, yticks_labels)
    enable_yticks()
    # plt.axhline(y=min_total_bitrate / img_coord_to_bandwidth_factor, color='white', linestyle='--')
    # discrete colorbar based on
    # https://stackoverflow.com/questions/48253810/show-a-discrete-colorbar-next-to-a-plot-as-a-legend-for-the-automatically-cho
    # https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
    col = plt.colorbar(im)
    c_labels = np.arange(n)
    c_loc = (n - 1) / n * (c_labels + 0.5)
    col.set_ticks(c_loc, labels=c_labels)
    col.set_label("\# of optimal solutions")
    plt.savefig(fairness_dir / (clients_config.name + "_num_optima.pdf"), bbox_inches="tight")
    

def plot(fixed_br=None, name="fairness.pdf", old=False, n_clients=2):
    step = 0.5
    bws = np.arange(0.1, 14,step)
    brs = np.arange(1, 7.5,step)

    br, bw = np.meshgrid(brs, bws)
    # create triples of client_A_br, bw, fairness
    fairness = fairness_real(bw, br, fixed_br, old, n_clients=n_clients)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    vmin = -2 if old else -1
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=1)
    surf = ax.plot_surface(br, bw, fairness, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False, norm=divnorm)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(fairness_dir / name)



@dataclass
class LinearConfig:
    name: str
    # minimum client bitrate
    b_min: float
    # maximum client bitrate
    b_max: float
    # client quality factors (q * bitrate = quality)
    # high factor => higher quality per bitrate
    client_q_factors: List[float]


@dataclass
class DiscreteConfig:
    name: str
    # list of mappings from bitrates to quality levels (for each client)
    client_bitrate2quality: List[Dict[float, float]]
    client_labels: Optional[list[str]] = None
    client_linestyle: Optional[list[str]] = None
    optimum_selection: str = "min_br"
    objective: str = "alpha-qf"
    alpha: float = 0.5


full_config = DiscreteConfig(
    name="discrete_phone_hd_4k_pcv",
    client_bitrate2quality=[
        VMAF_PHONE,
        VMAF_HD,
        VMAF_4K,
        QoE_POINT_CLOUD
    ],
    client_labels=[
        "Phone",
        "HDTV",
        "4KTV",
        "PCV"
    ],
    # client_linestyle = [
    #     ":",
    #     "-",
    #     "--",
    #     "-."
    # ],
    objective="alpha-qf",
    optimum_selection="max_br",
)

def main():
    base_config = full_config
    line_plot_steps = 500
    alpha_plot_steps = 500
    alphas = [0.25, 0.5, 0.75]
    alphas_legend = [0.75, 0.875]

    cloned_config = dataclasses.replace(base_config)
    cloned_config.name = f"{cloned_config.name}_fq"
    cloned_config.objective = "alpha-qf"

    # plot_discrete(
    #     cloned_config,
    #     line_plot_steps,
    #     figsize=(6.4 / 2, 4.8),
    #     util_ylim=(0.75, 1.05),
    #     quality_ylim=(0.5, 1.05),
    #     legend=True,
    #     with_marker=False
    # )

    matplotlib_settings.set_matplotlib_font_size(12, 14, 16)
    for alpha in alphas:
        continue
        cloned_config = dataclasses.replace(base_config)
        cloned_config.name = f"{cloned_config.name}_{alpha:.2f}"
        cloned_config.alpha = alpha
        plot_discrete(
            cloned_config,
            line_plot_steps,
            figsize=(6.4 / 2, 4.8),
            util_ylim=(0.75, 1.05),
            quality_ylim=(0.5, 1.05),
            legend=alpha in alphas_legend,
            with_marker=False
        )

    matplotlib_settings.set_matplotlib_font_size(22, 24, 24)
    plot_discrete_all_alpha(base_config, alpha_plot_steps)


if __name__ == '__main__':
    main()
