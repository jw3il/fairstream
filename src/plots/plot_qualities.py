import matplotlib.pyplot as plt
import plots.matplotlib_settings as matplotlib_settings
from quality import get_bitrates, get_qualities, VMAF_PHONE, VMAF_HD, VMAF_4K, QoE_POINT_CLOUD
from plots.constants import plots_dir

matplotlib_settings.init_plt()
matplotlib_settings.set_matplotlib_font_size(14, 16, 20)
plt.figure(figsize=(6.4, 4.8))

# horizontal line for qoe fairness
quality_fairness_value = get_qualities(VMAF_PHONE)[0]
# plt.axhline(y=quality_fairness_value, color='grey', linestyle='--')
# plt.annotate('Horizontal: Quality Fairness', color='grey', xy=(5.5, quality_fairness_value - 0.165))

# vertical line for bitrate fairness
bitrate_fairness_value = 2.5
# plt.axvline(x=bitrate_fairness_value, color='grey', linestyle='-', linewidth=0.5)
# plt.annotate('Vertical:\\ Bitrate Fairness', color='grey', xy=(bitrate_fairness_value + 0.25, 0.265))

# Plot lines for HD and 4K
plt.plot(get_bitrates(VMAF_PHONE), get_qualities(VMAF_PHONE), marker=".", linestyle=":", label="Phone")
plt.plot(get_bitrates(VMAF_HD), get_qualities(VMAF_HD), marker=".", label="HDTV")
plt.plot(get_bitrates(VMAF_4K), get_qualities(VMAF_4K), marker=".", linestyle="--", label="4KTV")
plt.plot(get_bitrates(QoE_POINT_CLOUD), get_qualities(QoE_POINT_CLOUD), marker=".", linestyle="-.", label="PCV")

min_total_bitrate = get_bitrates(VMAF_PHONE)[0] + get_bitrates(VMAF_HD)[0] + get_bitrates(VMAF_4K)[0] + get_bitrates(QoE_POINT_CLOUD)[0]
min_avg_quality = get_qualities(VMAF_PHONE)[0] + get_qualities(VMAF_HD)[0] + get_qualities(VMAF_4K)[0] + get_qualities(QoE_POINT_CLOUD)[0]
min_avg_quality /= 4
max_total_bitrate = get_bitrates(VMAF_PHONE)[-1] + get_bitrates(VMAF_HD)[-1] + get_bitrates(VMAF_4K)[-1] + get_bitrates(QoE_POINT_CLOUD)[-1]
max_avg_quality = get_qualities(VMAF_PHONE)[-1] + get_qualities(VMAF_HD)[-1] + get_qualities(VMAF_4K)[-1] + get_qualities(QoE_POINT_CLOUD)[-1]
max_avg_quality /= 4

print(f"Min: {min_total_bitrate:.2f} Mbit/s with quality {min_avg_quality:.2f}")
print(f"Max: {max_total_bitrate:.2f} Mbit/s with quality {max_avg_quality:.2f}")

plt.ylim(0, 1.05)
plt.xlim(0, 23)
plt.xlabel("Bitrate [Mbps]")
plt.ylabel("Perceptual quality", labelpad=10)
plt.legend()
plt.tight_layout()

plt.savefig(plots_dir / "media_quality.pdf", bbox_inches="tight")
