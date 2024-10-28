ref_4k="~/png/bbb3d-png/2024_02_16_15_20_50/2160p_634s_20004kbps.avi"
ref_HD="~/png/bbb3d-png/2024_02_16_15_20_50/1080p_634s_20089kbps.avi"
distorted_video_dir="~/png/bbb3d-png/2024_02_16_15_20_50"

script_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="$(pwd)/vmaf_paper_$(date +"%Y_%m_%d_%H_%M_%S")"
mkdir -p "$output_dir"

echo "Calculating VMAFs.."
# HD VMAF
for file in "$distorted_video_dir"/*.avi; do
    file_base=$(basename $file)
    case $file_base in 1440*|2160*)
        # skip higher qualities
        continue
    esac
    echo $file
    (set -x; time bash "${script_dir}/calculate_vmaf.sh" "$ref_HD" "$file" "vmaf_v0.6.1.json" "$output_dir" -t) > "${output_dir}/$(basename $file)-hd-phone.log" 2>&1
    (set -x; time bash "${script_dir}/calculate_vmaf.sh" "$ref_HD" "$file" "vmaf_v0.6.1.json" "$output_dir") > "${output_dir}/$(basename $file)-hd.log" 2>&1
done

# 4K VMAF
for file in "$distorted_video_dir"/*.avi; do
    file_base=$(basename $file)
    echo $file
    (set -x; time bash "${script_dir}/calculate_vmaf.sh" "$ref_4k" "$file" "vmaf_4k_v0.6.1.json" "$output_dir") > "${output_dir}/$(basename $file)-4k.log" 2>&1
done