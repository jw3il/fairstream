if [ "$#" -lt 4 ]; then
    echo "Usage: $0 mencoder_input bitrate width output_file"
    exit 1
fi

mencoder_input=$1
bitrate=$2
width=$3
output_file=$4

echo "Encode video with width ${width}: ${output_file}"
# use "veryslow" preset as "placebo" is "a waste of time" https://trac.ffmpeg.org/wiki/Encode/H.264#FAQ
mencoder "${mencoder_input}" -mf w=4000:h=4500:fps=60:type=png -ovc x264 \
    -x264encopts "profile=high:preset=veryslow:tune=animation:ref=4:bitrate=${bitrate}:threads=auto" \
    -vf "scale=${width}:-2,stereo3d=abl:ml,scale" -fps 60 -ofps 60 -o "${output_file}"
