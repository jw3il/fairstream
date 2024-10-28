from enum import Enum
import numpy as np


def map_dict_vals(d, fun):
    return dict([(k, fun(v)) for (k, v) in d.items()])


def map_dict_keys(d, fun):
    return dict([(fun(k), v) for (k, v) in d.items()])

def dict_to_arrays(d: dict):
    items = list(d.items())
    sorted_items = sorted(items, key=lambda i: i[0])

    keys = []
    values = []
    for (k, v) in sorted_items:
        keys.append(k)
        values.append(v)

    return np.array(keys, dtype=float), np.array(values, dtype=float)

def get_bitrates(d: dict):
    return dict_to_arrays(d)[0]

def get_qualities(d: dict):
    return dict_to_arrays(d)[1]

class QualityNormType(Enum):
    # no normalization
    NONE = 0
    # 0 to max value
    MAX = 1
    # min to max value (assuming min and max references)
    MIN_MAX = 2
    # VMAF from 20 to max value (assuming max reference)
    VMAF_MIN_MAX = 3
    # ACR from 1 to max value (assuming max reference)
    ACR_MIN_MAX = 4


def normalize_quality(quality_dict: dict, norm_type: QualityNormType):
    if norm_type == QualityNormType.NONE:
        return quality_dict

    vals = np.array(list(quality_dict.values()))
    vals_max = vals.max()
    vals_min = vals.min()

    if norm_type == QualityNormType.MAX:
        return map_dict_vals(quality_dict, lambda v: v / vals_max)

    if norm_type == QualityNormType.MIN_MAX:
        return map_dict_vals(
            quality_dict,
            lambda v: (v - vals_min) / (vals_max - vals_min)
        )

    if norm_type == QualityNormType.VMAF_MIN_MAX:
        return map_dict_vals(
            quality_dict,
            lambda v: (v - 20) / (vals_max - 20)
        )
    
    if norm_type == QualityNormType.ACR_MIN_MAX:
        return map_dict_vals(
            quality_dict,
            lambda v: (v - 1) / (vals_max - 1)
        )
    
    raise ValueError(f"Unknown norm type {norm_type}")


def scale_bitrate(quality_dict: dict, bitrate_factor: float):
    return map_dict_keys(
        quality_dict,
        lambda bitrate: bitrate_factor * bitrate
    )


def to_bits(quality_dict: dict):
    return scale_bitrate(quality_dict, 1_000_000)


def to_megabits(quality_dict: dict):
    return scale_bitrate(quality_dict, 1 / 1_000_000)


FACTOR_Kb2Mb = 1 / 1_000
FACTOR_b2Mb = 1 / 1_000_000
vmaf_norm_type = QualityNormType.VMAF_MIN_MAX

# quality dictionaries

VMAF_PHONE = scale_bitrate(
    normalize_quality(
        {
            494: 87.093952,
            989: 95.936829,
            2484: 99.836164,
            4982: 99.996809,
            7490: 99.999772,
            10013: 99.999996,
            20089: 100.0,
        },
        vmaf_norm_type
    ),
    FACTOR_Kb2Mb
)

VMAF_HD = scale_bitrate(
    normalize_quality(
        {
            494: 69.654153,
            989: 82.672646,
            2484: 93.412436,
            4982: 96.578481,
            7490: 97.444711,
            10013: 97.828828,
            20089: 98.838255
        },
        vmaf_norm_type
    ),
    FACTOR_Kb2Mb
)

VMAF_4K = scale_bitrate(
    normalize_quality(
        {
            494: 62.477523,
            989: 75.190991,
            2508: 87.896831,
            4960: 93.925376,
            7466: 96.570844,
            9976: 97.790523,
            20004: 100.0,
        },
        vmaf_norm_type
    ),
    FACTOR_Kb2Mb
)

NPPD_HD = scale_bitrate(
    {
        449480: 0.2667,
        843768: 0.3667,
        1416688: 0.4667,
        2656696: 0.6667,
        4741120: 1.0,
        7498176: 1.0,
    },
    FACTOR_b2Mb
)

NPPD_4K = scale_bitrate(
    {
        449480: 0.1333,
        843768: 0.1833,
        1416688: 0.2333,
        2656696: 0.3333,
        4741120: 0.5,
        7498176: 1.0,
    },
    FACTOR_b2Mb
)


QoE_POINT_CLOUD = scale_bitrate(
    normalize_quality(
        {
            22.4980: 3.910131,
            9.6115: 3.736928,
            7.7770: 3.568627,
            4.8050: 3.116013,
            3.8880: 2.934641,
            # 3.2045: 2.630719,
            2.5925: 2.540850,
            # 1.8925: 1.604575,
            1.2630: 1.491830,
        },
        QualityNormType.ACR_MIN_MAX
    ),
    1
)