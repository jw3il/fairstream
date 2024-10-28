import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plots.matplotlib_settings import adjust_lightness, adjust_lightness_relative
from scipy.interpolate import interp1d
from pathlib import Path
import plots.matplotlib_settings as matplotlib_settings
from plots.constants import plots_dir

from traces.load import load_traces


matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(22, 24, 16)

plt.figure(figsize=(4.8 * 2, 6.4))

search_trace_in_dataset = "cooked_traces/traces_full_test.tar.gz"

client_labels = [
    "Phone",
    "HDTV",
    "4KTV",
    "PCV"
]

def plot_eval_episode(trace_path, filename, legend=False):
    plt.clf()
    df = pd.read_csv(trace_path)

    trace_name = Path(trace_path).stem
    trace_name = trace_name[0:trace_name.index(".txt")]
    loaded_trace_bw, loaded_trace_name = load_traces(search_trace_in_dataset, includes=[trace_name])
    # print(loaded_trace_bw, loaded_trace_name)

    bw_color = "black"
    plt.plot([], [], color=bw_color, label="Bandwidth")

    def get_step_prev(x: np.ndarray, xp: np.ndarray, yp: np.ndarray):
        idx = np.searchsorted(xp, x, side='left')
        return np.concatenate((yp, np.zeros(1)))[idx]

    xs = []
    ys = []
    x_orig = []
    y_orig = []
    ps = []
    idxs = []
    last_x = None
    last_y = None
    max_x = 0
    print(f"Agent rebuffering:")
    for a in range(4):
        df_a = df[df["agent"] == a]
        print(f"{a} > mean {df_a['rebuffer'].mean()}, max {df_a['rebuffer'].mean()}, 0 {df_a['rebuffer'].iloc[0]}, mean without 0 {df_a['rebuffer'].iloc[1:].mean()}")
        x = np.array([0] + list(df_a["sim_time"].to_numpy())) / 1_000
        len_x = len(x)
        y = list(df_a["bitrate"].to_numpy())
        y = np.array([y[0]] + y)
        x_orig.append(x.copy())
        y_orig.append(y.copy())
        if last_x is not None:
            # step 1: get y values of layer below for x values of this layer
            y_on_old_layer = get_step_prev(x, last_x, last_y)
            # step 2: get y values of this layer for x values of layer below
            y_on_new_layer = get_step_prev(last_x, x, y)
            # step 3: combine the points pointwise
            y = np.concatenate((y_on_old_layer, last_y)) + np.concatenate((y, y_on_new_layer))
            # step 4: cat x values as well
            x = np.concatenate((x, last_x))
            # step 5: sort x
            x_sort_idx = np.argsort(x)
            inv_x_sort_idx = np.argsort(x_sort_idx)
            idxs.append(inv_x_sort_idx[:len_x])
            x_new = x[x_sort_idx]
            # step 6: sort y correspondingly
            y_new = y[x_sort_idx]
            # interpolate old y values in terms of new x values
            # TODO: correct for step function
            # y += np.interp(x, last_x, last_y)
        else:
            idxs.append(np.arange(len_x))
            x_new = x
            y_new = y

        ps.append(plt.plot([], []))
        xs.append(x_new)
        max_x = max(max_x, x_new[-1])
        ys.append(y_new)

        last_x = x_new
        last_y = y_new

    for i, (x, y, xo, yo, ix, p) in enumerate(reversed(list(zip(xs, ys, x_orig, y_orig, idxs, ps)))):
        c = p[0].get_color()
        c_fill = adjust_lightness_relative(c, 0.3)
        plt.fill_between(x, y * 0, y, step="pre", color=c_fill, zorder=i*3)#, alpha=0.3)
        plt.step(x, y, where="pre", label=client_labels[-(i+1)], color=c, zorder=i*3+1)

        diffs = yo[1:] - yo[:-1]
        switches = np.abs(diffs) >= 0.001
        a = ix[:-1][switches[:]]
        n_a = ix[:-1][~switches[:]]
        plt.scatter(x[n_a], y[n_a], marker="|", color=c, zorder=i*3+2, s=50, linewidths=0.8)
        plt.scatter(x[ix[-1]], y[ix[-1]], marker="|", color=c, zorder=i*3+2, s=50, linewidths=0.8)
        plt.scatter(x[a], y[a], marker=".", color=c_fill, zorder=i*3+2, s=90, edgecolor=c, linewidth=1)
        
        # for i, d in enumerate(diffs):
        #     if d != 0:
        #         plt.text(x[i], y[i], yo[i])
        
        #switches_pos = diffs > 0
        #switches_neg = diffs < 0
        #plt.scatter(x[:-1][switches_pos], y[:-1][switches_pos], marker=".", color=c_fill, zorder=i*3+2, s=20, edgecolor=c, linewidth=1)
        #plt.scatter(x[:-1][switches_neg], y[:-1][switches_neg], marker=".", color=c_fill, zorder=i*3+2, s=20, edgecolor=c, linewidth=1)

        # plt.text(x[:-1][s], y[:-1][s], d[s])
        #pos_switch = s * (d > 0)
        #plt.scatter(x[:-1][pos_switch], y[:-1][pos_switch], marker="^", color=c, zorder=i*3+2)
        #neg_switch = s * (d < 0)
        #plt.scatter(x[:-1][neg_switch], y[:-1][neg_switch], marker="v", color=c, zorder=i*3+2)
        
        # if i == 0:
        #     print(yo)
        #     print(y)

    # TODO: get correct bandwidth at all points in time... using agent stats is not enough :/
    # total = list(df[df["agent"] == 0]["bw_total"])
    # total = np.array([total[0]] + total)
    # sim_time = df[df["agent"] == 0]["sim_time"]
    # sim_time= np.array([0] + list(sim_time))
    # plt.step(sim_time / 1_000, total / 1_000_000, where="pre", color="black")

    total = loaded_trace_bw[0]
    total = np.array([total[0]] + list(total))
    sim_time = list(range(int(np.ceil(max_x))))
    sim_time = np.array(sim_time)
    plt.step(sim_time, total[sim_time] / 1_000_000, where="pre", color=bw_color, zorder=1_000_000)

    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative bitrate [Mbps]")
    if legend:
        plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    # x = [0, 1, 2, 3, 4]
    # y = [1, 1, 2, 3, 4]
    # plt.step(x, y, where="pre")
    # plt.savefig("tmp_ep_step_test.pdf", bbox_inches="tight")
    
    
#2024_05_29_12_18_13_eval-only-greedy-4-minerva
# fluctuation_curr_webget-2023-apr_3603005_www-bing-com_0.txt_652052754142632896
#2024_05_29_12_17_06_eval-only-greedy-32
trace = "fluctuation_curr_webget-2023-apr_80307501_www-msn-com_0.txt_0.csv"
plot_eval_episode(f"results/ray/2024_06_02_12_03_10_eval-only-greedy-8-minerva/eval_episodes_-1/{trace}", plots_dir / "episode_greedy8_minerva.pdf")
plot_eval_episode(f"results/ray/2024_06_06_14_32_45_ppo-500-lr1e-5-nosharing-fs8-2/eval_episodes_-1/{trace}", plots_dir / "episode_ppo.pdf")

handles, labels = plt.gca().get_legend_handles_labels()
plt.clf()
plt.legend(handles, labels, ncol=len(labels))
plt.axis('off')
plt.savefig(plots_dir / "episode_legend.pdf", bbox_inches='tight')
