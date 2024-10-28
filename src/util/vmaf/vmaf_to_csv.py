from dataclasses import dataclass
from typing import NamedTuple
from xml.dom.minidom import parse
import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def get_vmaf(filename, flag=""):
    """
    Extracts the mean VMAF score from an XML file.
    
    Args:
    - filename: Path to the XML file.
    - flag: Optional flag to specify the VMAF type.
    
    Returns:
    - Mean VMAF score as a float, or None if not found.
    """
    file = parse(filename)
    pooled_metrics = file.getElementsByTagName('pooled_metrics')[0]
    pooled_metric = pooled_metrics.getElementsByTagName('metric')

    for metric in pooled_metric:
        if metric.getAttribute('name') == f'{flag}vmaf':
            return float(metric.getAttribute('mean'))

    return None


def get_frame_vmaf_values(filename, flag=""):
    """
    Extracts frame-by-frame VMAF values from an XML file.

    Args:
    - filename: Path to the XML file.
    - flag: Optional flag to specify the VMAF type.

    Returns:
    - List of VMAF values for each frame.
    """
    file = parse(filename)
    frames = file.getElementsByTagName('frame')
    
    vmaf_values = []
    frame_num = -1
    for frame in frames:
        # make sure that frame order is correct in all files and that no frame is missing
        new_frame_num = int(frame.getAttribute('frameNum'))
        assert new_frame_num == frame_num + 1, f"Frame order is incorrect: {frame_num} -> {new_frame_num}"
        frame_num = new_frame_num

        vmaf = float(frame.getAttribute(f'{flag}vmaf'))
        vmaf_values.append(vmaf)
    
    return vmaf_values

def extract_filename_info(filename):
    """
    Extracts information from a filename using regular expressions.
    
    Args:
    - filename: The filename to be parsed.
    
    Returns:
    - A tuple of extracted information, or None if the pattern doesn't match.
    """
    pattern = re.compile(r'(\d+p)_(\d+)f_v(\d+)_vs_\d+p_\d+f_(\w+)')
    match = pattern.match(filename)

    return match.groups() if match else None

# from https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
def find_nth(string, substring, n):
    assert n >= 1
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)

@dataclass
class FileNameInfo:
    res: int
    dur: int
    br: int
    ref_res: int
    ref_dur: int
    ref_br: int
    model: str

    @staticmethod
    def from_filename(filename) -> "FileNameInfo":
        """
        Parses filename to extract resolution, bitrate, and metric.
        
        Args:
        - filename: The filename to be parsed.
        """
        splits = filename.split("_")
        res = int(splits[0][:-1])
        dur = int(splits[1][:-1])
        br = int(splits[2][:-4])
        ref_res = int(splits[4][:-1])
        ref_dur = int(splits[5][:-1])
        ref_br = int(splits[6][:-4])
        model = ".".join(filename[find_nth(filename, "_", 7)+1:].split(".")[0:-1])
        return FileNameInfo(res=res, dur=dur, br=br, ref_res=ref_res, ref_dur=ref_dur, ref_br=ref_br, model=model)

    def to_dataframe(self):
        return pd.DataFrame([[self.res, self.dur, self.br, self.ref_res, self.ref_dur, self.ref_br, self.model]], columns=["resolution", "duration", "bitrate", "reference_resolution", "reference_duration", "reference_birate", "model"])


def plot_vmafs(scores, infos, filter, filename):
    """
    Plots VMAF values for multiple videos.
    
    Args:
    - scores: List of lists containing VMAF scores for each video.
    - labels: Labels for each video.
    - filter: A function to filter the videos to be plotted.
    - flag: Optional flag to specify the VMAF type.
    """
    for video_scores, info in zip(scores, infos):
        if filter(info):
            mapping = lambda x: x.split("_")[0]
            colormap = {"240p": "tab:olive",
                        "360p": "tab:gray", 
                        "480p": "tab:brown",
                        "540p": "tab:purple",
                        "720p": "tab:green",
                        "1080p": "tab:blue",
                        "1440p": "tab:orange",
                        "2160p": "tab:red"}
            label = f"{info.res}p_{info.br}kbit/s ({info.model}) ref: {info.ref_res}p_{info.ref_br}kbit/s"
            plt.plot(range(len(video_scores)), video_scores, label=label, color=colormap[mapping(label)])
    plt.xlabel("frame")
    plt.ylabel("VMAF")
    plt.savefig(filename)

def main():
    """
    Main function to process XML files and generate a CSV with VMAF data.
    """
    parser = argparse.ArgumentParser(description="Process XML files and generate CSV with VMAF data.")
    parser.add_argument("xml_dir", help="Directory containing XML files")
    parser.add_argument("outname", help="Name of the output file")
    args = parser.parse_args()

    xml_dir = args.xml_dir
    df_rows = []
    frame_data = []
    frame_infos = []

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            filepath = os.path.join(xml_dir, filename)
            phoneflag = "phone" if "phone" in filename else ""
            print(f"Reading {filepath}")
            
            info = FileNameInfo.from_filename(filename)
            info_df = info.to_dataframe()

            vmaf = get_vmaf(filepath, phoneflag)
            info_df["vmaf"] = vmaf

            df_rows.append(info_df)
            frame_data.append(get_frame_vmaf_values(filepath, phoneflag))
            frame_infos.append(info)

    df = pd.concat(df_rows, ignore_index=True)

    # print(f"Creating plots for each model..")
    # for model in df["model"].unique():
    #     plot_vmafs(frame_data, frame_infos, lambda info: model == info.model, f"{model}.pdf")

    print(f"Saving results CSV..")
    df = df.sort_values(by=df.columns.values.tolist()).reset_index(drop=True)
    out_filename = f"{args.outname}.csv"
    df.to_csv(out_filename, index=False)
    print(f"Done. CSV file saved to {out_filename}")

if __name__ == "__main__":
    main()