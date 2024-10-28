import matplotlib
from matplotlib.ticker import LogFormatter
import numpy as np
import matplotlib.pyplot as plt
from plots.constants import plots_dir

from plots.plot_fairness import nu_discrete_full_enum, full_config
import plots.matplotlib_settings as matplotlib_settings

matplotlib_settings.init_plt()
plt.rcParams["axes.grid.axis"] = "both"
matplotlib_settings.set_matplotlib_font_size(12, 14)


# def all_less_or_equal(a, b):
#     return all(a[i] <= b[i] for i in range(len(a)))


# def all_greater(a, b):
#     return all(a[i] > b[i] for i in range(len(a)))


def filter_pareto(keys, z=None):
    front = []
    for k in keys:
        # if abs(z[k] - 3.73) < 0.1:
        #     debug_this_iter = True
        #     print(k, z[k])
        # else:
        #     debug_this_iter = False

        k_is_dominated_or_equal = False
        for f in front.copy():
            k_is_dominated_or_equal = all([k_e <= f_e for k_e, f_e in zip(k, f)]) and (z is None or z[k] >= z[f])
            k_dominates_or_is_equal = all([k_e >= f_e for k_e, f_e in zip(k, f)]) and (z is None or z[k] <= z[f])

            # if debug_this_iter and (k_dominates or k_is_dominated):
            #     print(f"{k} vs {f}: ", end='')
            #     if k_dominates:
            #         print("dominates")
            #     if k_is_dominated:
            #         print("is dominated")

            if k_dominates_or_is_equal:
                # f is not needed anymore, as k is at least as good as f
                front.remove(f)

            if k_is_dominated_or_equal:
                # no need to add k, as f in the current front is at least as good as k
                break

        if k_is_dominated_or_equal:
            continue

        # k is not dominated by any element in the front => add it
        front.append(k)
    return front


sols = nu_discrete_full_enum(100, full_config.client_bitrate2quality, alpha=0.5, optimum_selection="none", objective="f,q")
xy = list(sols.keys())
x = list(map(lambda x: x[0], xy))
y = list(map(lambda x: x[1], xy))
bitrates = list(sols.values())
min_bws = {}

for k in sols:
    v_min_bw = min([sum(v) for v in sols[k]])
    min_bws[k] = v_min_bw

bws = [min_bws[k] for k in xy]
norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
format = LogFormatter(10, labelOnlyBase=False)

plt.scatter(y, x, c=bws, cmap="viridis", norm=norm)
plt.ylabel("Mean fairness")
plt.xlabel("Mean quality")
plt.colorbar(label="Total bitrate [Mbps]", format=format)
plt.savefig(plots_dir / "feasible_solutions.pdf", bbox_inches="tight")

plt.clf()

circle_size = 55

plt.scatter(y, x, fc="grey", ec="none", alpha=0.25, s = circle_size)

xy = filter_pareto(list(sols.keys()), z = min_bws)
bws = [min_bws[k] for k in xy]
pareto_comb_factor = 0.25
optimal_sols = []
num_same_bws = []
for k in xy:
    best_sol = True
    num_same = 0
    for other_k in xy:
        if k == other_k:
            continue
        if min_bws[other_k] != min_bws[k]:
            continue
        # same bandwidth
        num_same += 1
        # 1, 0 because first fairness then qoe
        k_val = k[1] * pareto_comb_factor + k[0] * (1 - pareto_comb_factor)
        other_val = other_k[1] * pareto_comb_factor + other_k[0] * (1 - pareto_comb_factor)
        if other_val > k_val:
            best_sol = False

    num_same_bws.append(num_same)
    if best_sol:
        optimal_sols.append(True)
    else:
        optimal_sols.append(False)


sorted_bw_indices = np.argsort(bws)
for i in range(len(sorted_bw_indices) - 1):
    pos_from = xy[sorted_bw_indices[i]]
    pos_to = xy[sorted_bw_indices[i + 1]]
    pos_diff = np.array(pos_to) - np.array(pos_from)
    plt.arrow(pos_from[1], pos_from[0], pos_diff[1], pos_diff[0], width=0.0001, head_length=0, head_width=0, color="black")

x = list(map(lambda x: x[0], xy))
y = list(map(lambda x: x[1], xy))
# plt.scatter(x, y, c=bws, cmap="viridis", linewidths=list(map(lambda x: 2 if x else 0, optimal_sols)), edgecolor="black", norm=norm)
plt.scatter(y, x, c=bws, s = circle_size, cmap="viridis", norm=norm)
plt.colorbar(label="Total bitrate [Mbps]", ticks=[1, 5, 10, 25, 50, 100], format='%d')
plt.ylabel("Mean fairness")
plt.xlabel("Mean quality")



# print("Solutions on pareto-front")
# for i, k in enumerate(xy):
#     # print(f"{k}: min {min_bws[k]}, sols: {sols[k]}")
#     # plt.text(k[0], k[1], f"{sols[k][0][-1]}")
#     plt.text(k[1], k[0], f"{num_same_bws[i]}")

plt.savefig(plots_dir / "pareto.pdf", bbox_inches="tight")
