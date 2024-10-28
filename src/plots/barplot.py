import matplotlib.pyplot as plt
import plots.matplotlib_settings as matplotlib_settings
import json
import numpy as np
from plots.constants import plots_dir, main_agent_label_dir_pairs
from pathlib import Path

matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(14, 16)

def load(path):
    with open(path, "r") as f:
        return json.load(f)


def create_barplot_per_class(label_path_pairs, classes, key, y_label, filename, val_scale=1, ylog=False, legend=True, legend_only=False, ylim=None):
    width = 0.8 / len(label_path_pairs)
    for approach_id, (label, path) in enumerate(label_path_pairs):
        y = []
        yerr = []
        x = []
        res = load(path)
        for i, c in enumerate(classes):
            x.append(i + (approach_id + 0.5) * width - len(label_path_pairs) * width / 2)
            if isinstance(key, list):
                v_list = []
                for k in key:
                    v_list.append(np.array(res["evaluation"][c][k]["values"]))
                vals = np.stack(v_list)
            else:
                vals = np.array(res["evaluation"][c][key]["values"])
            vals *= val_scale
            y.append(vals.mean())
            yerr.append(vals.std())

        plt.bar(x, y, yerr=yerr, width=width, capsize=5 * 4 / len(label_path_pairs), label=label)

    plt.xticks(np.arange(len(classes)), classes)
    plt.ylabel(y_label)
    if ylog:
        plt.yscale("log")
    plt.xlabel("Traffic class")
    if legend:
        plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    if legend_only:
        plt.legend(handles, labels, ncol=len(labels))
        plt.axis('off')
        plt.savefig(Path(filename).parent / (Path(filename).stem + "_legend" + Path(filename).suffix), bbox_inches='tight')
        plt.clf()


def create_boxplot_per_agent(label_path_pairs, classes, keys, y_label, filename, val_scale=1, figsize=None, col_start=0, x_labels=None, legend_only=False):
    assert len(keys) >= 2
    plt.figure(figsize=figsize)
    width = 0.8 / len(label_path_pairs)
    cols = plt.cm.tab10.colors
    handles = []
    labels = []
    for approach_id, (label, path) in enumerate(label_path_pairs):
        y = []
        x = []
        res = load(path)
        for i, k in enumerate(keys):
            x.append(i + (approach_id + 0.5) * width - len(label_path_pairs) * width / 2)
            v_list = []
            for c in classes:
                v_list.extend(np.array(res["evaluation"][c][k]["values"]))
            vals = np.array(v_list)
            vals *= val_scale
            y.append(vals)

        box = plt.boxplot(y, sym="", positions=x, patch_artist=True, widths=[width * 0.8] * len(x))
        handles.append(box["boxes"][0])
        labels.append(label)
        col = cols[approach_id + col_start]
        # coloring fromhttps://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box[item], color=col)
        plt.setp(box["medians"], color='k')
        plt.setp(box["boxes"], facecolor=col)
        plt.setp(box["fliers"], markeredgecolor=col)
        # plt.bar(x, y, yerr=yerr, width=width, capsize=5, label=label)

    if x_labels is None:
        plt.xticks(np.arange(len(keys)), np.arange(len(keys)))
    else:
        plt.xticks(np.arange(len(keys)), x_labels)
    plt.ylabel(y_label)
    plt.xlabel("Client type")
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    
    if legend_only:
        plt.legend(handles, labels, ncol=len(labels))
        plt.axis('off')
        plt.savefig(Path(filename).parent / (Path(filename).stem + "_legend" + Path(filename).suffix), bbox_inches='tight')
        plt.clf()

classes = ['low', 'normal', 'high', 'veryhigh', 'fluctuation']

create_barplot_per_class(main_agent_label_dir_pairs, classes, 'reward', 'Mean return', plots_dir / "bar_reward.pdf", 1 / 4, legend=False)
create_barplot_per_class(main_agent_label_dir_pairs, classes, [f'fairness_{i}' for i in range(4)], 'Mean fairness', plots_dir / "bar_fair.pdf", legend=False, legend_only=True)
create_barplot_per_class(main_agent_label_dir_pairs, classes, [f'qoe_{i}' for i in range(4)], 'Mean QoE', plots_dir / "bar_qoe.pdf", legend=False, ylim=(0, 1))
create_barplot_per_class(main_agent_label_dir_pairs, classes, [f'qoe_ema_corrected_{i}' for i in range(4)], 'QoE EMA', plots_dir / "bar_qoe_ema.pdf")
create_barplot_per_class(main_agent_label_dir_pairs, classes, [f'rebuffer_time_{i}' for i in range(4)], 'Rebuffer time [s]', plots_dir / "bar_rebuffer.pdf", ylog=True, legend=False)

boxplot_agents = [main_agent_label_dir_pairs[4], main_agent_label_dir_pairs[5]]
boxplot_figsize = (3.4, 4.0)
boxplot_colstart = 4
x_labels = ["Phone", "HDTV", "4KTV", "PCV"]
create_boxplot_per_agent(boxplot_agents, classes, [f'reward_{i}' for i in range(4)], 'Return', plots_dir / "box_agent_reward.pdf", figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels, legend_only=True)
create_boxplot_per_agent(boxplot_agents, classes, [f'qoe_{i}' for i in range(4)], 'QoE', plots_dir / "box_agent_qoe.pdf", figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels)
create_boxplot_per_agent(boxplot_agents, classes, [f'qoe_ema_corrected_{i}' for i in range(4)], 'QoE EMA', plots_dir / "box_agent_qoe_ema.pdf", figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels)
create_boxplot_per_agent(boxplot_agents, classes, [f'fairness_{i}' for i in range(4)], 'Fairness', plots_dir / "box_agent_fairness.pdf", figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels)
create_boxplot_per_agent(boxplot_agents, classes, [f'download_rate_{i}' for i in range(4)], 'Download rate [Mbps]', plots_dir / "box_agent_download_rate.pdf", val_scale=1 / 1_000_000, figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels)
create_boxplot_per_agent(boxplot_agents, classes, [f'bitrate_{i}' for i in range(4)], 'Bitrate', plots_dir / "box_agent_bitrate.pdf", figsize=boxplot_figsize, col_start=boxplot_colstart, x_labels=x_labels)
