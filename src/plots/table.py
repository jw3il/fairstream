from plots.util import agentize, aggregate_stat, load
from plots.constants import main_agent_label_dir_pairs
from collections import defaultdict
import numpy as np

print(main_agent_label_dir_pairs)

stats = defaultdict(list)
stats_agents = ["reward", "qoe", "quality", "fairness", "rebuffer_time", "init_time", "quality_switch", "quality_diff", "t", "buffer"]
stats_static = []
stats_labels = {
    "reward": "Return",
    "qoe": "\\ac{qoe}",
    "quality": "Perceptual Quality",
    "fairness": "Fairness",
    "rebuffer_time": "Playback Stalling Time [s]",
    "init_time": "Initial Stalling Time [s]",
    "quality_switch": "Quality Switches",
    "quality_diff": "Quality Difference",
    "buffer": "Buffer level",
    "t": "Total Playback Time"
}
stats_bigger_is_better = {
    "reward": True,
    "qoe": True,
    "quality": True,
    "fairness": True,
    "rebuffer_time": False,
    "init_time": False,
    "quality_switch": False,
    "quality_diff": False,
    "buffer": True,
    "t": True
}
stats_table_order = ["reward", "qoe", "fairness", "quality", "init_time", "rebuffer_time", "quality_switch", "quality_diff", "buffer", "t"]

for label, path in main_agent_label_dir_pairs:
    results = load(path)["evaluation"]
    for key in stats_agents:
        mean, std = aggregate_stat(results, agentize(key))
        stats[key].append((mean, std))
    for key in stats_static:
        mean, std = aggregate_stat(results, key)
        stats[key].append((mean, std))

# header
print("\\toprule")
# print("\\textbf{Metric} & \\multicolumn{" + str(len(label_path_pairs)) + "}{c}{\\textbf{Agent}} \\\\")
print("\\textbf{Metric} / \\textbf{Agent} ", end='')
for label, path in main_agent_label_dir_pairs:
    print(f" & {label}", end='')
print("\\\\ \\midrule")

# rows
for key in stats_table_order:
    if stats_bigger_is_better[key]:
        marker = "$\\uparrow$"
    else:
        marker = "$\\downarrow$"
    print(f"{stats_labels[key]} {marker}", end='')
    means = list(map(lambda x: x[0], stats[key]))
    stats_key_best = max(means) if stats_bigger_is_better[key] else min(means)
    for mean, std in stats[key]:
        #if mean == stats_key_best:
        #    print(f" & $\\mathbf{{{mean:.2f}}} \\pm {std:.2f}$ ", end='')
        #else:
        # print(f" & ${mean:.2f} \\pm {std:.2f}$ ", end='')
        print(f" & ${mean:.2f}\\, {{\\scriptstyle \\pm\\, {std:.2f}}}$ ", end='')
    print("\\\\")
print("\\bottomrule")
