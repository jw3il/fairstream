#!/bin/bash

# number of parallel processes for encoding
# using simple-gpu-scheduler python package (note that GPU support was not tested)
ENCODE_GPUs="0 1"

# file input list for encode
mencode_input="mf://@list.txt"

dry_run=true
if $dry_run; then
    echo "PERFORMING DRY RUN (only prints the commands)"
fi


script_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="$(pwd)/$(date +"%Y_%m_%d_%H_%M_%S")"
mkdir -p "$output_dir"

rename_video() {
    output_file=$1

    # Extract height from the generated video
    height=$(get_height "${output_file}")
    # Extract duration from the generated video
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${output_file}" | cut -d. -f1)
    # Extract bitrate from the generated video and round it to whole kbps
    bitrate_kbps=$(round_bitrate $(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 "${output_file}"))

    # Generate the new filename
    new_name="$(dirname $output_file)/${height}p_${duration}s_${bitrate_kbps}kbps.avi"
    
    # Rename the file (and the corresponding log file)
    mv "${output_file}" "${new_name}"
    mv "${output_file}.log" "${new_name}.log"
    echo "${new_name}"
}

# Function to round the bitrate to the nearest whole kbps
round_bitrate() {
    echo "$1" | awk '{printf "%.0f\n", $1 / 1000}'
}

# Function to extract height from ffprobe
get_height() {
    ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$1"
}

encodeall() {
    # Check if two arguments are provided
    if [ "$#" -ne 2 ]; then
        echo "$#"
        echo "Usage: $0 widths bitrates"
        exit 1
    fi

    widths=($1)
    bitrates=($2)
    out_files=()
    comp_files=()

    echo "Widths ${widths[@]}"
    echo "Bitrates ${bitrates[@]}"

    echo "======================================"
    echo "Encoding videos.."

    # TODO: it would be better to add the comp_files directly below, but it does not seem
    # possible to modify the variable from a command substitution $(...)
    for width in "${widths[@]}"; do
        for bitrate in "${bitrates[@]}"; do
            height=$((width * 9 / 16))  # Assuming a 16:9 aspect ratio
            output_filename="${output_dir}/bbb_${height}p_${bitrate}kb.avi"
            comp_files+=("$output_filename")
        done
    done

    encode_commands=$(
        echo "(time bash ${script_dir}/encode_video.sh $mencode_input $ref_bitrate $ref_width $ref_filename) > ${ref_filename}.log 2>&1 "

        for width in "${widths[@]}"; do
            for bitrate in "${bitrates[@]}"; do
                height=$((width * 9 / 16))  # Assuming a 16:9 aspect ratio
                output_filename="${output_dir}/bbb_${height}p_${bitrate}kb.avi"
                echo "(time bash ${script_dir}/encode_video.sh $mencode_input $bitrate $width $output_filename) > ${output_filename}.log 2>&1 "
            done
        done
    )
    
    if $dry_run; then
        echo "$encode_commands"
        echo "(above commands would be scheduled on GPUs $ENCODE_GPUs)"
    else
        echo "$encode_commands" | simple_gpu_scheduler --gpus $ENCODE_GPUs
    fi

    # rename all videos according to their bitrates
    echo "======================================"
    echo "Renaming videos and logs.."
    if $dry_run; then
        echo "Note: this step does nothing in a dry run."
    fi
    if ! $dry_run; then
        ref_filename=$(rename_video $ref_filename)
    fi
    out_files+=("$ref_filename")

    for filename in "${comp_files[@]}"; do
        if ! $dry_run; then
            filename=$(rename_video $filename)
        fi
        out_files+=("$filename")
    done
}

widths="3840 2560 1920 1280"
video_bitrates="20000 10000 7500 5000 2500 1000 500"
encodeall "${widths[@]}" "${video_bitrates[@]}"
