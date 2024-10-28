from matplotlib.legend_handler import HandlerTuple
import numpy as np
import matplotlib.pyplot as plt

from plots.util import load, all_classes, get_all_values, agentize
from plots.constants import plots_dir
from collections import defaultdict
import plots.matplotlib_settings as matplotlib_settings
from plots.constants import ppo_ablation_coeff_path_dict

matplotlib_settings.init_plt()
plt.rcParams["axes.grid.axis"] = "both"
matplotlib_settings.set_matplotlib_font_size(16, 18)

keys = [0.25, 0.5, 0.75]
ps = defaultdict(list)
labels = {}

manual_marker_offsets = {
    (0.25, "low"): (0.03, -0.0004, "center", "left", False),
    (0.25, "high"): (-0.06, 0.003, "center", "right", True),
    (0.25, "veryhigh"): (0.05, 0.0015, "center", "left", True),
    (0.5, "normal"): (0.05, 0, "center", "left", True),
    (0.5, "high"): (0.04, -0.004, "top", "left", True),
    (0.5, "veryhigh"): (0, 0.005, "bottom", "center", True),
    (0.75, "high"): (-0.05, -0.001, "center", "right", True),
    (0.75, "veryhigh"): (0, -0.007, "top", "center", True),
    # (0.75, "fluctuation"): (0, -0.002, "top", "center", False),
    ("b", "fluctuation"): (0.05, -0.007, "top", "left", True),
    ("b", "normal"): (0.15, -0.007, "center", "left", True),
}

manual_marker_colors = {
    "b": '0.55'
}

markers = ["o", "s", "D", "^", "*"]
assert len(all_classes) == len(markers)
cols = []
for c in ppo_ablation_coeff_path_dict:
    if c in manual_marker_colors:
        cols.append(manual_marker_colors[c])
        continue

    p = plt.plot([], [])
    cols.append(p[0].get_color())

for k, c in enumerate(ppo_ablation_coeff_path_dict):
    path = ppo_ablation_coeff_path_dict[c]
    results = load(path)
    for i, cls in enumerate(all_classes):
        qoe = get_all_values(results, cls, agentize("qoe"))
        fairness = get_all_values(results, cls, agentize("fairness"))
        rebuffer = get_all_values(results, cls, agentize("rebuffer_time"))
        init = get_all_values(results, cls, agentize("init_time"))
        x = qoe.mean()
        y = fairness.mean()

        marker = markers[i]
        if marker == "*":
            markersize = 150
        else:
            markersize = 80
        p = plt.scatter(x, y, color=cols[k], marker=marker, s=markersize, linewidths=0)
        manual_offset = (0, 0.002, "bottom", "center")
        manual_offset_key = (c, cls)
        if manual_offset_key in manual_marker_offsets:
            manual_offset = manual_marker_offsets[manual_offset_key]
            draw_arrow = manual_offset[-1]
            if draw_arrow:
                plt.arrow(x + manual_offset[0], y + manual_offset[1], dx=-manual_offset[0], dy=-manual_offset[1], color=cols[k], width=0.0001, head_length=0, head_width=0)
        plt.text(x + manual_offset[0], y + manual_offset[1], f"${init.mean():.2f}, {rebuffer.mean():.2f}$", fontsize=16, verticalalignment=manual_offset[2], horizontalalignment=manual_offset[3], color=cols[k])

        ps[c].append(p)

    labels[c] = f"$\\alpha={c}$"

plt.xlabel("Mean QoE")
plt.ylabel("Mean fairness")
all_ps = [tuple(ps[k]) for k in keys]
all_labels = [labels[k] for k in keys]
#print(all_ps)
#print(all_labels)
plt.gca().legend(all_ps, all_labels, loc='lower center', ncol=len(keys), bbox_to_anchor=(0.5, -0.35),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, handlelength=3)
plt.savefig(plots_dir / "coeff.pdf", bbox_inches="tight")
