#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 reference_video distorted_video model_path [-t]"
    exit 1
fi

reference_video="$1"
distorted_video="$2"
model_path="$3"
output_dir="$4"
use_transform=false
use_scaling=false  

# Check for the presence of the optional flag
while [ "$#" -gt 0 ]; do
    case "$5" in
        -t)
            use_transform=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Construct the additional parameters based on the flag
additional_params=""
model_suffix=$(basename ${model_path})
if $use_transform; then
    additional_params="\\:enable_transform=true\\:name=phonevmaf"
    model_suffix="${model_suffix}_phone"
    echo "Using mobile"
fi

resolution_distorted=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$distorted_video")
resolution_reference=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$reference_video")

if [ "$resolution_distorted" == "$resolution_reference" ]; then
    echo "Reference and distorted video have the same resolution $resolution_distorted"
    scaling_filter=""
else
    echo "Reference and distorted video have different resolutions:"
    echo "- Reference: $resolution_reference"
    echo "- Distorted: $resolution_distorted"
    echo "=> Applying bicubic scaling filter"
    scaling_filter=",scale=$resolution_reference:flags=bicubic"
fi

# Extract file names without extensions
reference_name=$(basename -- "$reference_video")
reference_name_no_ext="${reference_name%.*}"

distorted_name=$(basename -- "$distorted_video")
distorted_name_no_ext="${distorted_name%.*}"

# Combine names to create the output file name
output_file="${output_dir}/${distorted_name_no_ext}_vs_${reference_name_no_ext}_${model_suffix}_noscale.xml"

# Run FFmpeg for VMAF calculation
ffmpeg -i "$reference_video" \
       -i "$distorted_video" \
       -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
                [1:v]setpts=PTS-STARTPTS$scaling_filter[distorted]; \
                [distorted][reference]libvmaf=log_fmt=xml:model='path=${model_path}${additional_params}':log_path=/dev/stdout:n_threads=8" \
       -f null - > "$output_file";

echo "VMAF calculation complete. VMAF stats saved to: $output_file"
