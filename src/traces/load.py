from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import tarfile
from tqdm import tqdm
import argparse


def bw_type(name):
    """
    Extracts the (optional) bandwidth type from a given filename.
    Assumes the format where the type is included in the filename,
    appearing at the very beginning before an underscore.
    """
    name = str(name)
    splits = name.split("_")
    if len(name.split("_")) > 1:
        return splits[0]
    return ""


def read_bandwidths_from_file(f):
    lines = f.readlines()
    if len(lines) > 0:
        num_splits = len(lines[0].split())
        assert 1 <= num_splits <= 2
        contains_timestamps = num_splits == 2

    if contains_timestamps:
        bws = [float(line.split()[1]) for line in lines]
    else:
        bws = [float(line) for line in lines]

    return np.array(bws)


def load_traces(path, includes=None, excludes=None):
    """
    Loads bandwidth traces from a specified directory.
    Optionally filters files to include or exclude based on keywords in their filenames.

    Args:
    - path: The directory path from which to load the traces.
    - includes: List of substrings. Only filenames containing any of these substrings are included.
    - excludes: List of substrings. Filenames containing any of these substrings are excluded.

    Returns:
    - A tuple of (array of bandwidth values, list of filenames) that match the criteria.
    """
    all_bw = []
    loaded_files = []
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist! (abs: {path.absolute()})")

    if path.is_dir():
        tar = None
        filenames = list(path.iterdir())
    else:
        tar = tarfile.open(path)
        filenames = [info.name for info in tar.getmembers() if info.isfile()]

    print(f"Attempting to load up to {len(filenames)} trace files from {path}.")

    num_include_filtered = 0
    num_exclude_filtered = 0
    for filename in tqdm(filenames):
        if (
            includes is not None
            and not sum([include in str(filename) for include in includes])
        ):
            num_include_filtered += 1
            continue

        if (
            excludes is not None
            and sum([exclude in str(filename) for exclude in excludes])
        ):
            num_exclude_filtered += 1
            continue

        if tar is None:
            with open(filename, 'r') as f:
                bws = read_bandwidths_from_file(f)
        else:
            with tar.extractfile(filename) as f:
                bws = read_bandwidths_from_file(f)

        loaded_files.append(str(filename))
        all_bw.append(np.array(bws))

    print(f"Loaded {len(all_bw)} traces from {path} (filter: include {num_include_filtered}, exclude {num_exclude_filtered})")
    return np.array(all_bw, dtype=object), loaded_files


def train_eval_split(path, n_eval=10, seed=None, include=None, exclude=None):
    """
    Splits the loaded traces into training and evaluation sets.

    Args:
    - path: Directory path for loading traces.
    - n_eval: Number of traces to use for evaluation.
    - seed: Random seed for reproducibility.
    - include: Include filter for filenames.
    - exclude: Exclude filter for filenames.

    Returns:
    - Tuple of (training bandwidths, training filenames, evaluation bandwidths, evaluation filenames).
    """
    bws, names = load_traces(path, include, exclude)

    type_indicies = defaultdict(list)
    types = [bw_type(name) for name in names]

    for i, t in enumerate(types):
        type_indicies[t].append(i)

    eval_ixs = []
    rng = np.random.default_rng(seed)
    for ixs in type_indicies.values():
        eval_ixs.extend(rng.choice(ixs, n_eval))

    names = np.array(names)
    bws_eval = bws[eval_ixs]
    names_eval = list(names[eval_ixs])
    bws_train = np.delete(bws, eval_ixs, axis=0)
    names_train = list(np.delete(names, eval_ixs, axis=0))

    return bws_train, names_train, bws_eval, names_eval


def main():
    parser = argparse.ArgumentParser(description="Load given traces")
    parser.add_argument("path", type=str, help="Path to traces directory or .tar.gz archive")
    args = parser.parse_args()
    _, names = load_traces(args.path)
    train_types = [bw_type(name) for name in names]
    print("Traces:\t", Counter(train_types))


if __name__ == "__main__":
    main()
