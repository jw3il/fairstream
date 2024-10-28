import itertools
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import plots.matplotlib_settings as matplotlib_settings
import numpy as np
import pandas as pd
import seaborn as sns
import re
from plots.util import load, all_classes
from plots.constants import best_ppo_agent_dir, plots_dir

matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(14, 16)
# sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def create_hist3d(dir, classes, key, label, filename, bins, val_scale=1):
    paths = get_all_result_paths(dir)
    results = load(paths)

    x = []
    t = []

    for i, res in enumerate(results):
        v_list = []
        for c in classes:
            if isinstance(key, list):
                for k in key:
                    v_list.extend(res["evaluation"][c][k]["values"])
            else:
                v_list.extend(res["evaluation"][c][key]["values"])

        x.append(v_list)
        t.append(i)

    x_all = list(itertools.chain(x))
    hist, bin_edges = np.histogram(x_all, bins=bins)
    img_data = np.zeros((len(t), len(bin_edges) - 1))
    
    for i, x_i in enumerate(x):
        # [1:-1]
        hist_i, _ = np.histogram(x_i, bins=bin_edges, density=True)
        img_data[i, :] = hist_i
        
    plt.imshow(img_data.T, cmap="inferno", origin="lower")
    y_ticks = range(0, len(bin_edges) - 1, 5)
    plt.yticks(y_ticks, [f"{round(e)}" for e in bin_edges[1:][y_ticks]])
    x_ticks = range(0, len(x), 5)
    plt.xticks(x_ticks, [f"{round(e)}" for e in np.array(t)[x_ticks]])
    plt.xlabel("Evaluation step")
    plt.ylabel("Reward")
    plt.savefig(filename, bbox_inches="tight")

    # df = pd.DataFrame(dict(x=x, g=g))

    # # Initialize the FacetGrid object
    # pal = sns.color_palette("Oranges", 20)[5:-5]
    # g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # # Draw the densities in a few steps
    # g.map(sns.kdeplot, "x",
    #     bw_adjust=.5, clip_on=False,
    #     fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # # passing color=None to refline() uses the hue mapping
    # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # # Set the subplots to overlap
    # g.figure.subplots_adjust(hspace=-.85)

    # # Remove axes details that don't play well with overlap
    # g.set_titles("")
    # g.set(yticks=[], ylabel="")
    # g.despine(bottom=True, left=True)
    plt.clf()
    return bin_edges


def get_eval_line_agg(dir, classes_agg, key, val_scale=1, agg_type='mean'):
    paths = get_all_result_paths(dir)
    results = load(paths)

    all_y = []
    all_y_err = []
    all_t = None
    for c in classes_agg:
        y = []
        y_err = []
        t = []
        for i, res in enumerate(results):
            v_list = []
            if isinstance(key, list):
                for k in key:
                    v_list.extend(res["evaluation"][c][k]["values"])
            else:
                v_list.extend(res["evaluation"][c][key]["values"])

            v_list = np.array(v_list) * val_scale
            if agg_type == 'mean':
                agg_val = v_list.mean()
            elif agg_type == 'min':
                agg_val = v_list.min()
            else:
                raise NotImplementedError()
            y.append(agg_val)
            y_err.append(v_list.std())
            t.append(i)

        all_y.append(np.array(y))
        all_y_err.append(np.array(y_err))
        t = np.array(t) * 10
        t[1:] += 1
        if all_t is not None:
            assert (t == all_t).all()
        else:
            all_t = t

    return np.array(all_y), np.array(all_y_err), t


handles, labels = None, None

def create_eval_line(dir, classes, key, label, filename, val_scale=1, mark_diff=False, ylog=False, legend=True):
    paths = get_all_result_paths(dir)
    results = load(paths)

    all_x = []
    for c in classes:
        x = []
        x_err = []
        t = []
        for i, res in enumerate(results):
            v_list = []
            if isinstance(key, list):
                for k in key:
                    v_list.extend(res["evaluation"][c][k]["values"])
            else:
                v_list.extend(res["evaluation"][c][key]["values"])
            v_list = np.array(v_list) * val_scale
            x.append(v_list.mean())
            x_err.append(v_list.std())
            t.append(i)

        x = np.array(x)
        all_x.append(x)
        x_err = np.array(x_err)
        t = np.array(t) * 10
        t[1:] += 1
        plt.fill_between(t, x - x_err, x + x_err, alpha=0.5)
        

        plt.plot(t, x, label=c)
        
        if c == "low" and mark_diff:
            begin=5
            diffs = np.abs((x[begin+1:] - x[begin:-1]))
            biggest_diff_idx = diffs.argmax() + begin
            biggest_diff = diffs.max()
            print(f"Biggest diff of {biggest_diff} at {biggest_diff_idx}")        
            # plt.plot(t[biggest_diff_idx], x[biggest_diff_idx], marker=".", color='black')
            plt.plot(t[biggest_diff_idx + 1], x[biggest_diff_idx + 1], marker="*", color='black')
        
    # plt.plot(t, np.stack(all_x).mean(axis=0), color='k', label="all")

    plt.xlabel("Training iteration")
    plt.ylabel(label)
    if legend:
        plt.legend(ncol=3)
    if ylog:
        plt.yscale("log")
    plt.savefig(filename, bbox_inches="tight")
    global handles, labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.clf()
    
    
def create_eval_line_agents(dir, classes, keys, labels, ylabel, filename, val_scale=1):
    paths = get_all_result_paths(dir)
    results = load(paths)

    all_x = []
    for i, key in enumerate(keys):
        x = []
        x_err = []
        t = []
        for r_i, res in enumerate(results):
            v_list = []
            for c in classes:
                v_list.extend(res["evaluation"][c][key]["values"])

            v_list = np.array(v_list) * val_scale
            x.append(v_list.mean())
            x_err.append(v_list.std())
            t.append(r_i)

        x = np.array(x)
        all_x.append(x)
        x_err = np.array(x_err)
        t = np.array(t)
        plt.fill_between(t, x - x_err, x + x_err, alpha=0.5)
        print(len(labels), i)
        plt.plot(t, x, label=labels[i])

    plt.xlabel("Evaluation step")
    plt.ylabel(ylabel)
    plt.legend(ncol=3)
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def get_all_result_paths(run_dir: str):
    results_list = []
    for path in Path(run_dir).iterdir():
        match = re.search("^eval_results_(\d*)_it_\d*\.json$", path.name)
        if match is None:
            continue
        results_list.append(
            (int(match.groups()[0]), path)
        )
    sorted_results = sorted(results_list, key=lambda x: x[0])
    only_paths = list(map(lambda x: x[1], sorted_results))
    return only_paths


classes = all_classes.copy()
# classes = list(reversed(["low", "fluctuation", "normal", "high", "veryhigh"]))
# classes = ["low"]

if __name__ == "__main__":
    create_eval_line(best_ppo_agent_dir, classes, "reward", "Mean return", plots_dir / "val_lines.pdf", 1/4, legend=False)
    # create_eval_line(dir_worst, classes, "reward", "Mean return", out_dir / "val_lines_worst.pdf", 1/4)
    # create_eval_line_agents(dir, classes, [f"reward_{i}" for i in range(4)], [f"Agent {i}" for i in range(4)], "Reward", out_dir / "val_lines_agents.pdf")

    rebuffer_keys = [f"rebuffer_time_{i}" for i in range(4)]
    create_eval_line(best_ppo_agent_dir, classes, rebuffer_keys, "Mean rebuffer time [s]", plots_dir / "val_rebuffer.pdf", legend=False)
    # create_eval_line_agents(dir, classes, rebuffer_keys, [f"Agent {i}" for i in range(4)], "Mean rebuffer time [s]", out_dir / "val_rebuffer_agents.pdf")

    plt.clf()
    plt.legend(handles, labels, ncol=len(labels))
    plt.axis('off')
    plt.savefig(plots_dir / "val_legend.pdf", bbox_inches='tight')


# bins = create_eval_line(dir, classes, "reward", "Mean Reward", out_dir / "hist3d.pdf", 25, 1/4)
# for c in classes:
#     create_eval_line(dir, [c], "reward", "Mean Reward", out_dir / f"hist3d_{c}.pdf", bins, 1/4)
