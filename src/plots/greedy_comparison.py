import numpy as np
import matplotlib.pyplot as plt

from plots.util import load, agentize, aggregate_stat
from collections import defaultdict
import plots.matplotlib_settings as matplotlib_settings
from plots.constants import plots_dir

matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(14, 16)

base_dir = "results/ray"
minerva_k_dir_tuples = [
    (False, 1, "2024_05_29_12_20_33_eval-val-only-greedy-1"),
    (False, 2, "2024_05_29_12_21_40_eval-val-only-greedy-2"),
    (False, 4, "2024_05_29_12_22_47_eval-val-only-greedy-4"),
    (False, 8, "2024_05_29_12_23_54_eval-val-only-greedy-8"),
    (False, 16, "2024_05_29_12_25_00_eval-val-only-greedy-16"),
    (False, 32, "2024_05_29_12_26_06_eval-val-only-greedy-32"),
    (True, 1, "2024_05_29_12_27_14_eval-val-only-greedy-1-minerva"),
    (True, 2, "2024_05_29_12_29_22_eval-val-only-greedy-2-minerva"),
    (True, 4, "2024_05_29_12_31_30_eval-val-only-greedy-4-minerva"),
    (True, 8, "2024_05_29_12_33_38_eval-val-only-greedy-8-minerva"),
    (True, 16, "2024_05_29_12_35_43_eval-val-only-greedy-16-minerva"),
    (True, 32, "2024_05_29_12_37_44_eval-val-only-greedy-32-minerva"),
]
filename = "evalresults_best_test.json"
handles, labels = None, None
minerva_linestyle="--"

def plot_stat(key, yaxis_label, export_filename, skip_colors=5):
    global handles, labels
    stats = defaultdict(list)
    for (i, (minerva, k, dir)) in enumerate(minerva_k_dir_tuples):
        full_dir = base_dir + "/" + dir + "/" + filename
        results = load(full_dir)["evaluation"]
        mean, std = aggregate_stat(results, agentize(key), classes=None)
        if minerva:
            stats["minerva_mean"].append(mean)
            stats["minerva_std"].append(std)
            stats["minerva_k"].append(k)
        else:
            stats["mean"].append(mean)
            stats["std"].append(std)
            stats["k"].append(k)

    for k in stats:
        stats[k] = np.array(stats[k])

    plt.clf()

    for _ in range(skip_colors):
        plt.plot([], [])

    # plt.fill_between(stats["k"], stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.3)
    plt.plot(stats["k"], stats["mean"], marker=".", label="Greedy-$k$")

    # plt.fill_between(stats["minerva_k"], stats["minerva_mean"] - stats["minerva_std"], stats["minerva_mean"] + stats["minerva_std"], alpha=0.3)
    plt.plot(stats["minerva_k"], stats["minerva_mean"], marker=".", label="Greedy-$k$-Minerva", linestyle=minerva_linestyle)

    plt.xticks(stats["k"], stats["k"])
    plt.xlabel("Parameter k")
    plt.ylabel(yaxis_label)
    plt.savefig(export_filename, bbox_inches="tight")
    handles, labels = plt.gca().get_legend_handles_labels()


def plot_stat_per_classes(key, yaxis_label, export_filename, minerva: bool, classes: list):
    global handles, labels
    means = defaultdict(list)
    stds = defaultdict(list)
    ks = defaultdict(list)
    for (i, (m, k, dir)) in enumerate(minerva_k_dir_tuples):
        full_dir = base_dir + "/" + dir + "/" + filename
        results = load(full_dir)["evaluation"]
        for c in classes:
            mean, std = aggregate_stat(results, agentize(key), classes=c)
            if m != minerva:
                continue
            means[c].append(mean)
            stds[c].append(std)
            ks[c].append(k)

    plt.clf()

    for c in means:
        means[c] = np.array(means[c])
        stds[c] = np.array(stds[c])
        ks[c] = np.array(ks[c])

        linestyle = minerva_linestyle if minerva else "-"

        plt.fill_between(ks[c], means[c] - stds[c], means[c] + stds[c], alpha=0.3)
        plt.plot(ks[c], means[c], marker=".", label=c, linestyle=linestyle)
        # plt.fill_between(stats["minerva_k"], stats["minerva_mean"] - stats["minerva_std"], stats["minerva_mean"] + stats["minerva_std"], alpha=0.3)
        plt.xticks(ks[c], ks[c])

    plt.xlabel("Parameter k")
    plt.ylabel(yaxis_label)
    # plt.legend()
    plt.savefig(export_filename, bbox_inches="tight")
    handles, labels = plt.gca().get_legend_handles_labels()



plot_stat("qoe", "Mean QoE", plots_dir / "greedy_comparison_qoe.pdf")
plot_stat("fairness", "Mean fairness", plots_dir / "greedy_comparison_fairness.pdf")
plot_stat("reward", "Mean return", plots_dir / "greedy_comparison_return.pdf")

plt.clf()
plt.legend(handles, labels, ncol=len(labels))
plt.axis('off')
plt.savefig(plots_dir / "greedy_comparison_legend.pdf", bbox_inches='tight')
plt.clf()

classes = ['low', 'normal', 'high', 'veryhigh', 'fluctuation']
plot_stat_per_classes("reward", "Mean return", plots_dir / "greedy_comparison_return_per_class_minerva.pdf", True, classes)
plot_stat_per_classes("reward", "Mean return", plots_dir / "greedy_comparison_return_per_class.pdf", False, classes)
plot_stat_per_classes("qoe", "Mean QoE", plots_dir / "greedy_comparison_qoe_per_class_minerva.pdf", True, classes)
plot_stat_per_classes("qoe", "Mean QoE", plots_dir / "greedy_comparison_qoe_per_class.pdf", False, classes)

plt.clf()
plt.legend(handles, labels, ncol=len(labels))
plt.axis('off')
plt.savefig(plots_dir / "greedy_comparison_return_per_class_legend.pdf", bbox_inches='tight')
plt.clf()
