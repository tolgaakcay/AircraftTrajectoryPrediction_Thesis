"""Utils funtions for common codes"""
from datetime import datetime
from typing import Literal, TypeAlias

import pymap3d as pm
from scipy.interpolate import splprep, splev
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike


Minutes: TypeAlias = int
Knots: TypeAlias = float | None
Feet: TypeAlias = float | None

def get_individual_coord_list(coords: list[tuple[float, float, float]], index: int):
    return [coord[index] for coord in coords]

def geo_to_cart(lats: list[float], lons: list[float], alts: list[float]) -> list[tuple[float, float, float]]:
    FT_TO_M = 1. / 3.28084
    cartesian_coords = []
    for lat, lon, alt in zip(lats, lons, alts):
        cartesian_coords.append(pm.geodetic2ecef(lat, lon, alt * FT_TO_M))
    return cartesian_coords

def get_start_datetime(ms_unix_times: list[int]) -> datetime:
    """Return the first time of a trajectory as a datime"""
    return datetime.utcfromtimestamp(int(ms_unix_times[0]) / 1000)

def get_min(data: list[int | float]) -> int | float:
    """Get the min value of a list of values"""
    return min([val for val in data if val is not None])

def get_max(data: list[int | float]) -> int | float:
    """Get the max value of a list of values"""
    return max([val for val in data if val is not None])

def get_item(data: list[int], idx: int) -> int:
    """Get the a value of a list of values in a given index"""
    return data[idx]

def get_samples_times(times: list[int], time_step: Minutes = 1) -> list[int]:
    """Get a list of ms unix times and returns the values equally spaced in time"""
    start_time, end_time = times[0], times[-1]
    samples_times = np.arange(start_time, end_time, step=(time_step * 60 * 1000), dtype=int)
    return ((samples_times - start_time) / (end_time - start_time)).tolist()

def min_max_scaler(values: list[int | float]) -> list[int | float]:
    """Perform a MinMax scale transformation on a list of values"""
    min_value, max_value = min(values), max(values)
    return [(value - min_value) / max((max_value - min_value), 1e-9) for value in values]

def interpolation_sampling(
        principal_value: list[float | int],
        other_values: list[list[float | int]],
        times: list[int],
        clip_values: tuple[float, float] | None = None,
        relative_to: Literal["prev", "start", "end"] | None = None,
        min_num_points: int = 0
    ) -> list[float | int]:
    """Perform a bi-cubic multivariate interpolation
    and return the principal values sampled in a equally time step and scaled using min-max
    """
    if clip_values:
        filtered_vals_idxs = {
            idx: val
            for idx, val in enumerate(principal_value)
            if val is not None and val >= clip_values[0] and val <= clip_values[1]
        }
        values = list(filtered_vals_idxs.values())
        times = [time for idx, time in enumerate(times) if idx in filtered_vals_idxs]
        new_other_values = []
        for values in other_values:
            new_other_values.append(
                [val for idx, val in enumerate(values) if idx in filtered_vals_idxs]
            )
        other_values = new_other_values

    x = [principal_value, *other_values, times]
    try:
        tck, _ = splprep(x=x, s=0)
    except (ValueError, TypeError):
        tck = None

    if tck is None:
        return []
    
    u_samples = get_samples_times(times)
    samples = splev(u_samples, tck)[0]

    if len(samples) < min_num_points:
        return []

    if clip_values:
        samples = [min(max(val, clip_values[0]), clip_values[1]) for val in samples]
    if samples is not None:
        if relative_to == "prev":
            samples = np.array(samples)
            samples = [val - samples[max(0, i - 1)] for i, val in enumerate(samples)]
        elif relative_to == "start":
            samples = np.array(samples)
            relative_val = samples[0]
            samples = [val - relative_val for val in samples]
        elif relative_to == "end":
            samples = np.array(samples)
            relative_val = samples[-1]
            samples = [val - relative_val for val in samples]
            # relative_val = samples[-1] if relative_to == "end" else samples[0]
            # samples = [abs(val) for val in samples] # from 0 to max ever
        # return min_max_scaler(samples)
        return samples
    return []

def to_beaufort_category(wind: Knots) -> int:
    """Categorize wind data according to the Beaufort Wind Scale"""
    if wind is None:
        return 18
    return round((wind * (8 / 13)) ** (2/3))


def filter_partitions(destination: str, start: datetime, end: datetime) -> pl.Expr:
    """Generate a filter Expr to filter parquet data with polars"""
    days_in_between = pd.date_range(start, end, freq="1D")
    filter_date_range = [
        {"year": date.year, "month": date.month, "day": date.day} for date in days_in_between
    ]
    print(filter_date_range)
    return (pl.struct(["year", "month", "day"]).is_in(filter_date_range)) & (
        pl.col("destination").is_in([destination])
    )

def filter_path_partitions(
    partitions: dict[str, str], destinations: list[str] | str, start: str, end: str
) -> bool:
    """Generate the filter mask for partitions of a specified destination and time range"""
    start = start[0] if isinstance(start, tuple) else start
    end = end[0] if isinstance(end, tuple) else end
    destinations = destinations[0] if isinstance(destinations, tuple) else destinations

    start_datetime = datetime.strptime(start, "%Y-%m-%d").date()  # noqa: DTZ007
    end_datetime = datetime.strptime(end, "%Y-%m-%d").date()  # noqa: DTZ007

    if isinstance(destinations, str):
        destinations = [destinations]

    if ("destination" in partitions) and (partitions["destination"] in destinations):
        partition_date_str = (
            partitions["year"] + "-" + partitions["month"] + "-" + partitions["day"]
        )
        partition_datetime = datetime.strptime(  # noqa: DTZ007
            partition_date_str, "%Y-%m-%d"
        ).date()
        if start_datetime <= partition_datetime and partition_datetime <= end_datetime:
            return True
    return False

def normalize_vector(vec: ArrayLike) -> ArrayLike:
    """Normalize an numpy array"""
    norm = np.sqrt(np.sum(vec**2))
    if norm != 0:
        return vec / norm
    return vec

@np.vectorize
def tokenizer(row: str | list[str]) -> ArrayLike:
    """Convert sequences of categorical values into a numerical vector"""
    if isinstance(row, str):
        return np.array([float(ord(char)) for char in row], dtype=object)
    return np.array(
        [
            np.array([float(ord(char)) for char in value], dtype=float)
            if isinstance(value, str)
            else np.array([float(ord(char)) for char in bin(10).ljust(6, "0")[2:]], dtype=float)
            for value in row
        ],
        dtype=object,
    )


# @np.vectorize
# def hash_lat_lon(lats: float, lons: float) -> ArrayLike:
#     """Compute the geohash for a list of latitude,longitude pairs"""
#     return np.array([geohash.encode(lat, lon) for lat, lon in zip(lats, lons)], dtype=object)

def get_quantile_bins(v: ArrayLike, q_step: float = 0.1) -> list[float]:
    """Get list of quantile values for a distribution of float values v"""
    all_v = np.concatenate(v, axis=None).astype(float)
    all_not_nan_v = all_v[~np.isnan(all_v)]
    return np.quantile(all_not_nan_v, q=np.arange(0, 1.01, q_step))

def parse_stringfied_list(str_list: str) -> list[float]:
    """Get a str(list) and turn it into the original list"""
    str_list = str(str_list).strip("[").strip("]").split(",")
    return [float(s_v.strip()) for s_v in str_list]

@np.vectorize
def discretize(v: ArrayLike, bins: str) -> ArrayLike:
    """Discretize continuous data based on specified bin values"""
    # Work around to use this function as a np vectorize with two array like objects
    # with different shapes, so bins is transformed into string before
    bins = parse_stringfied_list(bins)
    v = np.array(v, dtype=float)

    d_v = np.digitize(v, bins, right=True).astype(float)
    d_v[np.isnan(v)] = np.nan
    return np.array(d_v, dtype=object)

def get_global_stats(v: ArrayLike) -> ArrayLike:
    """Get global stats from list of np arrays"""
    all_v = np.concatenate(v, axis=None).astype(float)
    all_not_nan_v = all_v[~np.isnan(all_v)]

    min_v = all_not_nan_v.min()
    max_v = all_not_nan_v.max()
    mean_v = all_not_nan_v.mean()
    std_v = all_not_nan_v.std()
    return [min_v, max_v, mean_v, std_v]

@np.vectorize
def scale_normalize(v: ArrayLike, stats: str) -> ArrayLike:
    """Scale and normalize values in np array based on global stats"""
    stats = parse_stringfied_list(stats)
    min_v, max_v, mean_v, std_v = stats
    v = v.astype(float)

    # MinMax scaling
    min_max_div = (max_v - min_v) + 0.01
    if min_max_div == 0.0:
        min_max_div = 1e-100
    scaled_v = (v - min_v) / min_max_div
    scaled_v[np.isnan(scaled_v)] = 0.0

    # Normalization
    if std_v == 0.0:
        std_v = 1e-100
    normalized_v = (scaled_v - mean_v) / std_v
    return np.array(normalized_v, dtype=object)
