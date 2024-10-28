# VMAF Pipeline Documentation

## Overview
This documentation outlines the steps for using the VMAF pipeline scripts. These scripts are designed to automate the process of encoding videos at various bitrates and resolutions, renaming the encoded files, and calculating their VMAF scores.

## Prerequisites

1. Install `menconder`

    ```
    sudo apt install mencoder
    ```

2. Install `ffmpeg` with `libvmaf` support (see https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md)

3. Install https://pypi.org/project/simple-gpu-scheduler/

## Usage

### Step 1: Video Encoding

1. **Download Videos/Images:** Obtain the source videos or images for processing. We have used the Big Buck Bunny PNG images from https://media.xiph.org/BBB/bbb3d/video/png/.
2. **List Images:** Create an `list.txt` file listing all images for a video. In the Big Buck Bunny png directory, this file already exists (see https://media.xiph.org/BBB/bbb3d/video/png/list.txt). You may modify it to only render specific scenes, e.g., for debugging.
3. **Configure Script:** Modify `generate_paper_videos.sh` to set desired resolutions and bitrates. This script will use the `encode_video.sh` helper script to encode videos.
4. **Execute Script:** Run `generate_paper_videos.sh` from the location of `index.txt`.

### Step 2: VMAF Calculation

This step depends on the videos creates in the first step and will compute the VMAFs for each pair of reference and degraded video.

1. **Configure Script:**  Adapt the paths provided in `generate_paper_vmafs.sh` according to your output directories and generated files.
2. **Execute Script:** Run `generate_paper_vmafs.sh` to generate the VMAFs.

### Step 3: VMAF to CSV

The script `vmaf_to_csv.py` can be used to aggregate the generated mean VMAF scores into a single CSV file as follows:

```
python src/util/vmaf/vmaf_to_csv.py your_vmaf_dir vmafs
```

This `vmafs.csv` file that was used for our paper is provided with this repository. 

> [!CAUTION] Note that we have manually added a column for the respective target bitrates, which are currently not automatically stored by our processing pipeline.

### Step 4: Process VMAFs

Finally, we filter the best VMAFs for each bitrate with the script `process_vmafs.py`. This uses the `target_bitrate` column to filter resolutions that lead to inferior results. The output contains the best VMAF scores for each model and target bitrate.

The corresponding output file is available as `vmafs_processed_best.csv` in this repository.
This file contains the final VMAF scores that are used in our paper.

## Additional Information
- The `encode_video.sh` script encodes videos using `mencoder` with specified bitrates and resolutions. It supports a range of bitrates and dynamically calculates the corresponding video height assuming a 16:9 aspect ratio.
- Post-encoding, videos are renamed to include their resolution, duration, and bitrate in the filename.
- The scripts assume the presence of necessary tools like `mencoder`, `ffprobe`, and the VMAF calculation scripts.
- It is important to ensure the paths and environment variables are correctly set for these tools.
