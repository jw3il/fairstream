import numpy as np
import matplotlib.pyplot as plt
import plots.matplotlib_settings as matplotlib_settings

from plots.lineplot import get_eval_line_agg
from plots.util import all_classes
from plots.constants import ppo_ablation_label_dirs_pairs, plots_dir

matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(16, 18, 20)

linestyles = {
    0: "-",
    1: "--",
    2: ":"
}


def get_style(i):
    if i in linestyles:
        return linestyles[i]
    else:
        return '-'


def plot_main():
    for (label, dirs) in ppo_ablation_label_dirs_pairs:
        p = plt.plot([], [], label=label)
        color = p[0].get_color()
        for i, d in enumerate(dirs):
            linestyle = get_style(i)
            print(d)
            y, y_err, t = get_eval_line_agg(d, all_classes, "reward", 1/4)
            plt.plot(t, y.mean(axis=0), color=color, linestyle=linestyle)

    plt.legend()
    plt.xlabel("Training iteration")
    plt.ylabel("Mean return")
    plt.savefig(plots_dir / "stability.pdf", bbox_inches="tight")
    plt.clf()


def plot_details():
    label, dirs = ppo_ablation_label_dirs_pairs[0]
    for c in ["fluctuation", "high"]:
        p = plt.plot([], [])
        color = p[0].get_color()
        for i, d in enumerate(dirs):
            linestyle = get_style(i)
            print(d)
            y, y_err, t = get_eval_line_agg(d, [c], "reward", 1/4, agg_type="min")
            # single class => take first element y[0]
            plt.plot(t, y[0], color=color, linestyle=linestyle, label=f"{label} {i}")

        plt.legend()
        plt.xlabel("Training iteration")
        plt.ylabel("Minimum mean return")
        plt.savefig(plots_dir / f"stability_min_{c}.pdf", bbox_inches="tight")
        plt.clf()


plot_main()
plot_details()
