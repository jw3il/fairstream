from pathlib import Path
from typing import Union
import json
import numpy as np

all_classes = ["fluctuation", "low", "normal", "high", "veryhigh"]


def load(path: Union[str, list[str]]):
    if isinstance(path, list):
        res = []
        for p in path:
            with open(p, "r") as f:
                res.append(json.load(f))
        return res

    with open(path, "r") as f:
        return json.load(f)


def aggregate_stat(results, key, classes=None):
    all_results = []
    for traffic_class in results:
        if classes is not None and traffic_class not in classes:
            continue
        if isinstance(key, list):
            for k in key:
                all_results.extend(results[traffic_class][k]["values"])
        else:
            all_results.extend(results[traffic_class][key]["values"])
    
    all_results = np.array(all_results)
    return all_results.mean(), all_results.std()


def get_all_values(res, classes, keys):
    v_list = []
    if not isinstance(classes, list):
        classes = [classes]
    if not isinstance(keys, list):
        keys = [keys]
    
    for k in keys:
        for c in classes:
            v_list.extend(res["evaluation"][c][k]["values"])

    return np.array(v_list)


def agentize(key):
    return [f"{key}_{i}" for i in range(4)]

