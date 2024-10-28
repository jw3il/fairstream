import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import plots.matplotlib_settings as mpls
from pathlib import Path

mpls.init_plt()
mpls.set_matplotlib_font_size(12, 14)

def linestyle(model):
    style = "-"
    if "phone" in model:
        style = ":"
    if "4k" in model:
        style = "--"
    return style

def plot_scores(data, filename, model_labels: dict = {}):
    plt.clf()

    resolution_colors = ["#e60049", "#0bb4ff", "#9b19f5", "#50e991", "#e6d800", "#ffa300"]
    
    for model in data["model"].unique():
        if model in model_labels:
            model_label = model_labels[model]
        else:
            model_label = model
        data_model = data[data["model"] == model]
        for i, resolution in enumerate(data_model["resolution"].unique()):
            data_model_res = data_model[data_model["resolution"] == resolution]
            plt.plot(data_model_res["bitrate"] / 1_000, data_model_res["vmaf"], label=f'{model_label} {resolution}p', color=resolution_colors[i], linestyle=linestyle(model), marker=".")
        
        # min_len = min([x.size for x in s])
        # scores = np.stack([x[:min_len] for x in s])
        # max_ix = np.argmax(scores, axis=0)
        # change_positions = np.where(np.diff(max_ix) != 0)[0] + 1 
        # print(change_positions)

        # for ix in change_positions:
        #     print(f"lines for {metric}")
        #     res = max_ix[ix] 
        #     plt.axvline(x=bitrates[ix], linestyle=linestyle(metric), color=colors[res], alpha=0.5 if linestyle(metric)=="-" else 1)
    
    plt.ylim([20,105])
    plt.xlim([0,21])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Quality (VMAF)')
    # plt.title('VMAF Scores for Different Resolutions and Bitrates')
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")


def get_best_data(data, normalize_vmaf=False):
    # first get best results for each model and bitrate
    best_data = data.sort_values("vmaf", ascending=False).groupby(["model", "target_bitrate"], as_index=False).first()

    # then filter out all rows for which one can achieve a better vmaf with a lower bitrate
    for i in reversed(range(len(best_data))):
        row = best_data.iloc[i]
        less_br_rows = best_data[(best_data["model"] == row["model"]) & (best_data["target_bitrate"] < row["target_bitrate"])]
        if less_br_rows["vmaf"].max() >= row["vmaf"]:
            best_data.drop(i, axis=0, inplace=True)

    if normalize_vmaf:
        for i, model in enumerate(best_data["model"].unique()):
            data_model = best_data[best_data["model"] == model]
            # assuming that the max() is the score for the reference and that 20 is the min possible quality
            best_data.loc[best_data["model"] == model, "vmaf"] = (data_model["vmaf"] - 20) / (data_model["vmaf"].max() - 20)

    return best_data

def plot_best_scores(data, filename, model_labels: dict = {}, normalize=False):
    plt.clf()
    best_data = get_best_data(data, normalize)

    # plot best bitrates
    for i, model in enumerate(best_data["model"].unique()):
        if model in model_labels:
            model_label = model_labels[model]
        else:
            model_label = model

        data_model = best_data[best_data["model"] == model]
        plt.plot(data_model["bitrate"] / 1_000, data_model["vmaf"], label=f'{model_label}', color=mpls.COLORS_LIST[i], linestyle=linestyle(model), marker=".")
        
        # min_len = min([x.size for x in s])
        # scores = np.stack([x[:min_len] for x in s])
        # max_ix = np.argmax(scores, axis=0)
        # change_positions = np.where(np.diff(max_ix) != 0)[0] + 1 
        # print(change_positions)

        # for ix in change_positions:
        #     print(f"lines for {metric}")
        #     res = max_ix[ix] 
        #     plt.axvline(x=bitrates[ix], linestyle=linestyle(metric), color=colors[res], alpha=0.5 if linestyle(metric)=="-" else 1)
    
    if normalize:
        plt.ylim([0.5, 1.05])
        plt.ylabel('Quality')
    else:
        plt.ylim([50,105])
        plt.ylabel('Quality (VMAF)')
    plt.xlim([0,21])
    plt.xlabel('Bitrate (Mbps)')
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")

    return best_data

def main():
    parser = argparse.ArgumentParser(description="Process vmaf scores from a csv file.")
    parser.add_argument("csv", help="csv file containing vmaf data")
    args = parser.parse_args()

    model_labels = {
        "vmaf_v0.6.1.json_phone": "Phone",
        "vmaf_v0.6.1.json": "HDTV",
        "vmaf_4k_v0.6.1.json": "4KTV"
    }

    df = pd.read_csv(args.csv)

    csv_filter = (
        ((df["reference_resolution"]==2160) & (df["model"] == "vmaf_4k_v0.6.1.json")) |
        ((df["reference_resolution"]==1080) & (df["model"] != "vmaf_4k_v0.6.1.json"))
    )
    # df = df[csv_filter]
    #plot_scores(df, "vmafs.pdf", model_labels)
    #plot_best_scores(df, "vmafs_best.pdf", model_labels)
    plot_best_scores(df, "vmafs_best_norm.pdf", model_labels, normalize=True)

    best = get_best_data(df)
    csv_in_path = Path(args.csv)
    best.to_csv(str(csv_in_path.parent / csv_in_path.stem) + "_processed_best.csv", index=False)


if __name__ == "__main__":
    main()
