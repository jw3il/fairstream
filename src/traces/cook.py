from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOWrapper
import lzma
from pathlib import Path
import random
import tarfile
import time

import numpy as np
import datetime
import collections
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import plots.matplotlib_settings as matplotlib_settings


FILE_PATH = [
    'datasets/curr_webget-2022-jul.csv',
    'datasets/curr_webget-2022-aug.csv',
    'datasets/curr_webget-2022-nov.csv',
    'datasets/curr_webget-2022-dec.csv',
    'datasets/curr_webget-2023-jan.csv',
    'datasets/curr_webget-2023-feb.csv',
    # # 'datasets/curr_webget-2023-mar.csv',
    'datasets/curr_webget-2023-apr.csv',
    'datasets/curr_webget-2023-may.csv',
    'datasets/curr_webget-2023-jun.csv',
    'datasets/curr_webget-2023-jul.csv'
]
# unit_id,dtime,target,address,fetch_time,bytes_total,bytes_sec,bytes_sec_interval,warmup_time,warmup_bytes,sequence,
# threads,successes,failures
OUTPUT_DIR = "cooked_traces"
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)
MIN_NON_ZEROS = 0.95
TIME_INTERVAL = 1

BANDWITH_FACTOR = 3

NORMAL_BW = 10_000_000
HIGH_BW = 25_000_000
VERY_HIGH_BW = 50_000_000
FLUCTUATING_CV_THRESHOLD = 0.35
NUM_BW_CLASSES = 4
SAMPLES_PER_TRACE = 200
# everything with a mean bandwidth below
# this value gets filtered out (in the original data set)
LOW_BW_FILTER = 3_000_000 / BANDWITH_FACTOR

SEED = 42
# different dataset sizes for experiments
# determined by number of samples per class
DATASET_CLASS_SAMPLES = {
    "full": 10_000,
    "small": 1_000,
    "tiny": 100
}
MAX_SAMPLES_PER_CLASS = max(list(DATASET_CLASS_SAMPLES.values()))

# ratio of training data
TRAIN_RATIO = 0.9
# ratio of validation data
VAL_RATIO = 0.05

assert (
    0 < TRAIN_RATIO < 1
    and 0 <= VAL_RATIO < 1
    # remaining samples must exist, are used for testing
    and 0 < 1 - TRAIN_RATIO - VAL_RATIO < 1
    and int(MAX_SAMPLES_PER_CLASS * TRAIN_RATIO) > 0
)

# Full HD min. ~ 5 Mbps
# HD ready min. ~ 2,75 Mbps
# 4K min. ~ 7,7 Mbps
# Min res min. ~ 0.5 Mbps

def invalid_trace(data_points):
    arr = np.array(data_points)
    return (
        np.count_nonzero(arr <= 0) > 0
    )


@dataclass
class TraceStats:
    bandwidth_min: np.ndarray
    bandwidth_mean: np.ndarray
    bandwidth_max: np.ndarray
    bandwidth_std: np.ndarray

    @staticmethod
    def create_from(traces) -> "TraceStats":
        if isinstance(traces, dict):
            traces = [v for (_, v) in traces.items()]
        elif isinstance(traces, np.ndarray):
            traces = [traces]

        assert isinstance(traces, list)
        if len(traces) > 0:
            assert isinstance(traces[0], np.ndarray)

        bandwidth_min = np.zeros(len(traces))
        bandwidth_mean = np.zeros(len(traces))
        bandwidth_max = np.zeros(len(traces))
        bandwidth_std = np.zeros(len(traces))

        for i, v in enumerate(traces):
            v: np.ndarray
            bandwidth_min[i] = v.min()
            bandwidth_mean[i] = v.mean()
            bandwidth_max[i] = v.max()
            bandwidth_std[i] = v.std()

        return TraceStats(
            bandwidth_min=bandwidth_min,
            bandwidth_mean=bandwidth_mean,
            bandwidth_max=bandwidth_max,
            bandwidth_std=bandwidth_std
        )

    def __getitem__(self, key):
        return TraceStats(
            bandwidth_min=self.bandwidth_min[key],
            bandwidth_mean=self.bandwidth_mean[key],
            bandwidth_max=self.bandwidth_max[key],
            bandwidth_std=self.bandwidth_std[key]
        )
        
    def __len__(self):
        if self.bandwidth_mean is None:
            return 0
        return len(self.bandwidth_mean)

    def plot(self, file_prefix, bins=100):
        color = '0.55'
        plt.clf()
        plt.hist(self.bandwidth_mean / 1_000_000, bins=bins, color=color)
        plt.xlabel("Mean bandwidth [Mbps]")
        print("Max mean bandwidth: ", self.bandwidth_mean.max())
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_bw_mean.pdf", bbox_inches='tight')
        plt.clf()

        plt.hist(self.bandwidth_min / 1_000_000, bins=bins, color=color)
        plt.xlabel("Min bandwidth [Mbps]")
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_bw_min.pdf", bbox_inches='tight')
        plt.clf()

        plt.hist(self.bandwidth_std, bins=bins, color=color)
        plt.xlabel("Std bandwidth")
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_bw_std.pdf", bbox_inches='tight')
        plt.clf()

        all_cv = self.bandwidth_std / self.bandwidth_mean
        plt.hist(all_cv[all_cv <= 1], bins=bins, color=color)
        plt.xlabel("Coefficient of variation")
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_cv_1.pdf", bbox_inches='tight')
        plt.clf()
        print(f"Proportion of samples with cv <= 1 for {file_prefix}: {((all_cv <= 1).sum() / len(all_cv)):.4f}")
        print(f"Max CV: {all_cv.max()}")

        all_cv = self.bandwidth_std / self.bandwidth_mean
        plt.hist(all_cv[all_cv > 1], bins=bins, color=color)
        plt.xlabel("Coefficient of variation")
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_cv_2.pdf", bbox_inches='tight')
        plt.clf()
        
        all_cv = self.bandwidth_std / self.bandwidth_mean
        plt.hist(all_cv, bins=bins, color=color)
        plt.xlabel("Coefficient of variation")
        plt.ylabel("Number of traces")
        plt.savefig(f"{file_prefix}_cv.pdf", bbox_inches='tight')
        plt.clf()

        plt.scatter(self.bandwidth_mean / 1_000_000, all_cv, alpha=0.1, s=5, color=color)
        plt.xlabel("Mean bandwidth [Mbps]")
        plt.ylabel("Coefficient of variation")
        plt.savefig(f"{file_prefix}_scatter_bw_mean_cv.png", bbox_inches='tight')
        plt.clf()

        plt.scatter(self.bandwidth_mean / 1_000_000, self.bandwidth_std / 1_000_000, alpha=0.1, s=5, color=color)
        plt.xlabel("Mean bandwidth [Mbps]")
        plt.ylabel("Std bandwidth")
        plt.savefig(f"{file_prefix}_scatter_bw_mean_std.png", bbox_inches='tight')
        plt.clf()

    def get_mean_bw_p(self, n_bins, debug_plot_prefix):
        """
        Probability for each sample so that a selection with
        repetition results in each (binned) bandwidth being
        chosen with the same probability, i.e. having a uniform
        distribution over all available bandwidths/bins.

        :param n_bins: Number of bins
        :param debug_plot_prefix: Prefix for the debug plots
        """
        scaled_bw_mean = self.bandwidth_mean
        hist, bin_edges = np.histogram(scaled_bw_mean, n_bins)
        # outer edges are not part of the bins
        bin_indices = np.digitize(scaled_bw_mean, bin_edges[1:-1])

        # validate bin indices
        bin_counts = np.zeros(len(hist))
        for i in range(len(bin_indices)):
            bin_counts[bin_indices[i]] += 1
        assert (hist == bin_counts).all()

        plt.clf()
        plt.bar(np.arange(len(bin_counts)), bin_counts)
        plt.savefig(f"{debug_plot_prefix}_bins_count.pdf", bbox_inches='tight')

        # we have: counts c_i for each bin i
        # we want: p_i to get c_i * p_i = d, all samples from bin i
        #    should be chosen with equal probability p_i so that each
        #    bin gets chosen with the same probability d = 1 / |I|
        # => p_i = d / c_i = (1 / |I|) / c_i = 1 / (|I| * c_i)

        p_i = 1 / (len(hist) * hist)
        p = p_i[bin_indices]
        assert np.isclose(p.sum(), 1)

        plt.clf()
        plt.bar(np.arange(len(p_i)), p_i)
        plt.savefig(f"{debug_plot_prefix}_bins_p_i.pdf", bbox_inches='tight')

        return p


def create_all_bandwidth_values_histogram(traces, file_prefix, bins):
    plt.clf()
    all_values = np.array(list(traces.values())).flatten()
    print(f"Proportion of samples covered by VALUE bandwidth histogram: {(all_values <= 100_000_000).sum() / len(all_values)}")
    (n, bins, patches) = plt.hist(all_values[all_values <= 100_000_000] / 1_000_000, bins=bins)
    plt.xlabel("Bandwidth [Mbps]")
    plt.ylabel("Number of samples")
    plt.savefig(f"{file_prefix}_samples.pdf", bbox_inches='tight')
    plt.clf()
    
    return bins
    
    
def create_all_bandwidth_values_per_class_histogram(traces, classes_idx, file_prefix, bins, class_order):
    plt.clf()
    traces_values = np.array(list(traces.values()))
    for i, c in enumerate(class_order):
        class_traces = traces_values[classes_idx[c]].mean(axis=-1)
        plt.hist(class_traces / 1_000_000, bins=bins, label=c, alpha=0.7, zorder=10 - i)
    plt.xlabel("Bandwidth [Mbps]")
    plt.ylabel("Number of traces")
    plt.legend()
    plt.savefig(f"{file_prefix}_num_traces_per_class.pdf", bbox_inches='tight')
    plt.clf()


def select_traces_uniform_bw(stats: TraceStats, classes: dict, bins, samples_per_class):
    for class_name, indices in classes.items():
        assert len(indices) >= samples_per_class, f"Class '{class_name}' has not enough items: {len(indices)} < {samples_per_class}"

    scaled_bw_mean = stats.bandwidth_mean
    hist, bin_edges = np.histogram(scaled_bw_mean, bins)
    # outer edges are not part of the bins
    bin_indices = np.digitize(scaled_bw_mean, bin_edges[1:-1])

    # validate bin indices
    bin_counts = np.zeros(len(hist))
    for i in range(len(bin_indices)):
        bin_counts[bin_indices[i]] += 1
    assert (hist == bin_counts).all()

    selected_trace_indices = []
    for class_name, indices in classes.items():
        remaining_indices = np.array(indices)
        class_sample_indices = []
        # shuffle indices across datasets
        np.random.shuffle(remaining_indices)
        # UNIQUE, fill up one by one. But somehow random sampled...
        while len(class_sample_indices) < samples_per_class:
            # get unique bins and indices that correspond to one entry in each of these unique bins
            selected_bin_indices, remaining_indices_idx = np.unique(bin_indices[remaining_indices], return_index=True)
            # print(class_name, len(remaining_indices_idx))
            size_diff = samples_per_class - len(class_sample_indices)
            if len(remaining_indices_idx) > size_diff:
                # select random subset to get exact number of samples as requested
                remaining_indices_idx = np.random.choice(remaining_indices_idx, size=size_diff, replace=False)
            class_sample_indices += list(remaining_indices[remaining_indices_idx])
            remaining_indices = np.delete(remaining_indices, remaining_indices_idx)

        selected_trace_indices += class_sample_indices

    return selected_trace_indices, bin_edges


def parse_traces_from_curr(files):
    traces = {}
    total_unfiltered = 0
    filtered_invalid = 0
    filtered_low_bw = 0
    for file in files:
        bw_measurements = collections.defaultdict(list)
        stem = Path(file).stem
        print(f"Loading data from file '{file}' ({stem})")
        # load measurements from file
        with open(file, 'r') as f:
            for line in tqdm(f):
                parse = line.split(',')

                uid = parse[0]
                target = parse[2]
                try:
                    # bytes to bits
                    throughput = float(parse[6]) * 8
                except ValueError:
                    continue

                bw_measurements[(stem, uid, target)].append(throughput)

        print(f"Number of unique uid-target pairs: {len(bw_measurements)}")

        # filter measurements and store traces
        for k in bw_measurements:
            for t in range(0, len(bw_measurements[k]) - SAMPLES_PER_TRACE, SAMPLES_PER_TRACE): 
                total_unfiltered += 1

                bw = np.array(bw_measurements[k][t:t + SAMPLES_PER_TRACE])
                mean_bw = np.mean(bw)

                # ignore traces with 0-artifacts and too low bandwidth
                if invalid_trace(bw):
                    filtered_invalid += 1
                    continue

                if mean_bw < LOW_BW_FILTER:
                    filtered_low_bw += 1
                    continue

                # this trace gets considered, store it
                traces[(*k, t)] = bw

    return traces, {
        "total_unfiltered": total_unfiltered,
        "filtered_invalid": filtered_invalid,
        "filtered_low_bw": filtered_low_bw,
        "low_bw_filter": LOW_BW_FILTER
    }


def get_class(stats: TraceStats):
    if stats.bandwidth_std / stats.bandwidth_mean > FLUCTUATING_CV_THRESHOLD:
        return 'fluctuation'
    elif stats.bandwidth_mean > VERY_HIGH_BW:
        return 'veryhigh'
    elif HIGH_BW < stats.bandwidth_mean <= VERY_HIGH_BW:
        return 'high'
    elif NORMAL_BW < stats.bandwidth_mean <= HIGH_BW:
        return 'normal'

    assert stats.bandwidth_mean <= NORMAL_BW
    return 'low'


def get_classes_dict(stats: TraceStats):
    classes = collections.defaultdict(list)
    for i in range(len(stats)):
        bw_class = get_class(stats[i])
        classes[bw_class].append(i)
    return classes

def print_class_counts(classes, traces_info):
    print(f"- low bandwidth ({BANDWITH_FACTOR * traces_info['low_bw_filter'] / 1_000_000} <= x <= {NORMAL_BW / 1_000_000} Mbps): {len(classes['low'])}")
    print(f"- normal bandwidth ({NORMAL_BW / 1_000_000} <= x <= {HIGH_BW / 1_000_000} Mbps): {len(classes['normal'])}")
    print(f"- high bandwidth ({HIGH_BW / 1_000_000} <= x <= {VERY_HIGH_BW / 1_000_000} Mbps): {len(classes['high'])}")
    print(f"- very high bandwidth (x >= {VERY_HIGH_BW / 1_000_000} Mbps): {len(classes['veryhigh'])}")
    print(f"- fluctuating bandwidth (cv >= {FLUCTUATING_CV_THRESHOLD:.2f}): {len(classes['fluctuation'])}")


def plot_single_trace(trace, filename):
    plt.clf()
    plt.xlabel('Time (second)')
    plt.ylabel('Throughput (Mbps)')
    plt.plot(np.arange(len(trace)), trace)
    plt.savefig(filename, bbox_inches="tight")


def get_trace_filename(trace_key, trace_class: str) -> str:
    # key: (source dataset stem, source id, target url, idx)
    (stem, source, target_url, idx) = trace_key
    target_url = str.replace(target_url, "http://", "")
    target_url = str.replace(target_url, "https://", "")
    if target_url[-1] == "/":
        target_url = target_url[:-1]
    target_url = str.replace(target_url, "/", "-")
    target_url = str.replace(target_url, ".", "-")
    # idea for naming scheme: id, source dataset, source agent, target, time
    # target: replace http:// and https:// with "", then replace "/" and "." with "-"
    return f"{trace_class}_{stem}_{source}_{target_url}_{idx}.txt"


def write_trace_data(f, trace_value: list):
    for i_val in trace_value:
        f.write(f"{i_val}\n")


def save_traces_archive(traces_dict, filename):
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    stats = TraceStats.create_from(traces_dict)
    print(f"Saving {len(traces_dict)} traces in {filename}")
    with tarfile.open(filename, "w:gz") as t:
        for i, (k, v) in tqdm(enumerate(traces_dict.items()), total=len(traces_dict)):
            s = StringIO()
            write_trace_data(s, v)
            s.seek(0)
            s_bytes = s.read().encode("ascii")

            trace_name = get_trace_filename(k, get_class(stats[i]))
            info = tarfile.TarInfo(trace_name)
            info.size = len(s_bytes)
            info.mtime = time.mktime(time.gmtime())
            info.type = tarfile.AREGTYPE  # regular file type
            t.addfile(info, BytesIO(s_bytes))


def filter_traces_by_index(traces_dict: dict, idx: list):
    traces_list = list(traces_dict.items())
    return dict([traces_list[i] for i in idx])


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    
    plot_output_dir = Path("plots_traces")
    plot_output_dir.mkdir(exist_ok=True, parents=True)
    
    TMP_FILENAME = "traces.tmp.lz"
    TMP_FILENAME_SELECTED = "traces_selected.tmp.lz"
    if Path(TMP_FILENAME).exists():
        print(f"Loading traces from cache {TMP_FILENAME}.")
        with lzma.open(TMP_FILENAME, "rb") as f:
            traces, info = pickle.load(f)
    else:
        traces, info = parse_traces_from_curr(FILE_PATH)
        with lzma.open(TMP_FILENAME, "wb") as f:
            pickle.dump((traces, info), f)
        print(f"Dumped traces to {TMP_FILENAME}.")

    print(f"Loaded {len(traces)} traces")
    print(
        f"- Info: generated from {info['total_unfiltered']} traces, "
        f"filtered {info['filtered_invalid']} invalid, "
        f"{info['filtered_low_bw']} with mean bw below "
        f"{info['low_bw_filter'] / 1_000_000} Mbit/s"
    )

    # agg_stats = TraceStats.create_from(traces)
    for k, v in traces.items():
        traces[k] = v * 3

    all_traces_list = list(traces.items())
    agg_stats_scaled = TraceStats.create_from(traces)
    classes = get_classes_dict(agg_stats_scaled)

    matplotlib_settings.init_plt()
    matplotlib_settings.set_matplotlib_font_size(24, 26, 26)

    # agg_stats.plot("traces_original")
    agg_stats_scaled.plot(plot_output_dir / "traces")

    print(f"Classes of all {len(traces)} traces")
    print_class_counts(classes, info)

    filtered_trace_indices, selected_bin_edges = select_traces_uniform_bw(
        agg_stats_scaled, classes, 100, MAX_SAMPLES_PER_CLASS
    )
    filtered_traces = dict([all_traces_list[i] for i in filtered_trace_indices])

    filtered_stats = TraceStats.create_from(filtered_traces)
    filtered_stats.plot(plot_output_dir / "traces_filtered")

    print(f"Classes of {len(filtered_traces)} selected traces")
    filtered_classes = get_classes_dict(filtered_stats)
    print_class_counts(filtered_classes, info)

    for class_name in filtered_classes:
        random.shuffle(filtered_classes[class_name])
        assert len(filtered_classes[class_name]) == MAX_SAMPLES_PER_CLASS, f"Not enough samples for class {class_name}"

    # with lzma.open(TMP_FILENAME_SELECTED, "wb") as f:
    #     pickle.dump({
    #         "traces": filtered_traces,
    #         "classes": filtered_classes,
    #         "bin_edges": selected_bin_edges
    #     }, f)

    create_all_bandwidth_values_histogram(filtered_traces, plot_output_dir/ "traces_filtered", 100)
    create_all_bandwidth_values_per_class_histogram(filtered_traces, filtered_classes, plot_output_dir / "traces_filtered", selected_bin_edges / 1_000_000, class_order=["fluctuation", "low", "normal", "high", "veryhigh"])

    filtered_traces_list = list(filtered_traces.items())
    for c in filtered_classes:
        for i in range(0, min(len(filtered_classes[c]), 3)):
            plot_single_trace(filtered_traces_list[filtered_classes[c][i]][1], plot_output_dir / f"traces_filtered_ex_{c}_{i}.pdf")

    # select random samples for training, validation, and testing
    train_indices = collections.defaultdict(list)
    val_indices = collections.defaultdict(list)
    test_indices = collections.defaultdict(list)

    # shuffle all class indices and select samples
    for class_name in filtered_classes:
        li = filtered_classes[class_name]
        random.shuffle(li)

        for dataset_name, dataset_samples in DATASET_CLASS_SAMPLES.items():
            train_samples = int(TRAIN_RATIO * dataset_samples)
            val_samples = int(VAL_RATIO * dataset_samples)

            train_indices[dataset_name] += li[0:train_samples]
            val_indices[dataset_name] += li[train_samples:train_samples + val_samples]
            test_indices[dataset_name] += li[train_samples + val_samples:dataset_samples]

    # save datasets
    for kind, d in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        for dataset_name, trace_indices in d.items():
            traces = filter_traces_by_index(filtered_traces, trace_indices)
            filename = Path(OUTPUT_DIR) / f"traces_{dataset_name}_{kind}.tar.gz"
            save_traces_archive(traces, filename)


if __name__ == '__main__':
    main()
