import os

from typing import Literal
import pyproj
import h3
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timezone
from torchtext.vocab import vocab
from collections import Counter, OrderedDict

import numpy as np
from sklearn.preprocessing import QuantileTransformer

import numpy as np
from numpy.typing import ArrayLike
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.utils import filter_partitions
from src.data.schemas import (
    trajectories_schema,
    training_trajectories_schema,
    non_seq_numerical_schema,
    flight_categories_schema,
    temporal_categories_schema,
    sequential_weather_schema,  #Change to OG Code - for the sequential ERA5 weather features
)

MAX_ALTITUDE = 43100    # ft
MAX_SPEED = 600         # knots
MAX_TRACK = 360         # degrees
MAX_RATE = 3000         # ft./min
MAX_LAT = 90.           # degrees
MAX_LON = 180.          # degrees
MIN_LAT = -90.          # degrees
MIN_LON = -180.         # degrees

TRAJECTORY_COLUMNS = list(trajectories_schema.keys())
FLIGHT_CATEGORICAL_COLUMNS = list(flight_categories_schema.keys())
NON_SEQ_NUM_COLUMNS = list(non_seq_numerical_schema.keys())
TIME_CATEGORY_COLUMNS = list(temporal_categories_schema.keys())
TRAINING_COLUMNS = list(training_trajectories_schema.keys())
SEQUENTIAL_WEATHER_COLUMNS = list(sequential_weather_schema.keys())  #Change to OG Code - engineered ERA5 feature names, encoded but never predicted

TROPOPAUSE_HEIGHT_M = 11_000  #Change to OG Code - ISA deviation constants (Feature set section of CLAUDE.md)
ISA_SEA_LEVEL_TEMP_K = 288.15
ISA_LAPSE_RATE_K_PER_M = 0.0065
ISA_TROPOPAUSE_TEMP_K = 216.65

WEIRD_IDS_NOT_FILTERED = [
    'AFR83EE_057_2024_10_23',
    'AFR6136_050_2024_10_21',
    'AFR95LJ_015_2024_10_17',
    'AFR32ZB_025_2024_10_23',
    'AFR32VC_037_2024_10_22',
    'AFR24PR_044_2024_10_21',
    'AFR12LW_036_2024_10_23',
    'AFR24PR_046_2024_10_22',
    'AFR6136_038_2024_10_22',
    'AFR6136_050_2024_10_11',
    'AFR63SP_028_2024_10_27',
    'AFR67LK_018_2024_10_19',
    'AFR68KF_020_2024_10_20',
    'AFR71QP_037_2024_10_17',
    'AFR71QP_051_2024_10_7',
    'AFR71QP_053_2024_10_15',
    'AFR74VU_017_2024_10_17',
    'AFR86AM_012_2024_10_19',
    'AFR91HJ_035_2024_10_27',
    'AFR95LJ_035_2024_10_25',
    'AFR95LJ_052_2024_10_15',
    'AFR98RP_020_2024_10_27',
    'AIB03BE_019_2024_10_24',
    'BAW2DP_061_2024_10_29',
    'BAW2DP_061_2024_10_29',
    'BAW4DP_069_2024_10_25',
    'BAW4DP_080_2024_10_14',
    'BAW4DP_037_2024_10_26',
    'BAW4DP_028_2024_10_20',
    'BAW6DP_026_2024_10_19',
    'BEL29D_083_2024_10_22',
    'BEL8WD_053_2024_10_18',
    'BGA131N_037_2024_10_29',
    'CCM321L_057_2024_10_9',
    'DAH1076_001_2024_10_20',
    'DLH23U_063_2024_10_21', 
    'DLH42V_059_2024_10_4',
    'DLH42V_070_2024_10_8',
    'EFW40K_105_2024_10_14',
    'EJU43HC_075_2024_10_11',
    'EJU46JF_082_2024_10_21',
    'EJU4980_036_2024_10_20',
    'EJU4980_085_2024_10_9',
    'EJU69BL_065_2024_10_17',
    'EJU69EW_073_2024_10_22',
    'EJU69EW_086_2024_10_21',
    'EJU724X_066_2024_10_27',
    'EJU74ZM_077_2024_10_11',
    'EJU74ZM_078_2024_10_4',
    'EJU78CN_087_2024_10_9',
    'EJU963A_054_2024_10_3',
    'EJU97GL_084_2024_10_24',
    'EJU963A_054_2024_11_3',
    'EVX72EV_018_2024_10_15',
    'EZS87JT_074_2024_10_27',
    'EZY38ZH_073_2024_10_10',
    'OYO8_044_2024_10_17',
    'OYO10_033_2024_11_1',
    'RYR19PX_109_2024_10_7',
    'RYR1TE_087_2024_10_17',
    'N3117J_043_2024_10_19',
    'LRQ624D_125_2024_10_14',
    'LRQ611A_111_2024_10_9',
    'KLM60B_097_2024_10_11',
    'KLM60B_095_2024_10_23',
    'KLM47U_094_2024_10_8',
    'KLM1451_088_2024_11_5',
    'KLM37H_095_2024_10_21',
    'RYR2444_118_2024_10_21',
    'TJT33LX_037_2024_10_25',
    'TVF54SQ_041_2024_10_9',
    'VOE7347_007_2024_10_23',
    'VOE7347_012_2024_10_25',
    'TVF54SQ_041_2024_10_9',
    'THY9JL_102_2024_10_11',
    'VOE1AD_010_2024_10_29',
    'DLH87M_024_2024_10_19',
    'CCM321L_058_2024_10_1',
    'TAP492_090_2024_10_1',
    'CCM321L_058_2024_10_1',
]


def far_from_airport(dist_from_orig, dist_to_dest, threshold = 5):
    if dist_from_orig > threshold and dist_to_dest > threshold:
        return True
    return False

def latlon_to_h3(
        lat: float,
        lon: float,
        dist_orig: list[float],
        dist_dest: list[float],
        res: str = "5"
    ) -> str:
    if res == "multi":
        if (dist_orig > 100. and dist_dest > 100.):
            return int(h3.latlng_to_cell(lat, lon, res=5), 16) 
        elif (dist_orig > 50. and dist_dest > 50.):
            return int(h3.latlng_to_cell(lat, lon, res=6), 16)
        else:
            return int(h3.latlng_to_cell(lat, lon, res=7), 16)
    else:
        res = int(res)
    return int(h3.latlng_to_cell(lat, lon, res=res), 16)

def gps_to_ecef(lat, lon, alt):
    alt_in_m = alt / 0.3048
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt_in_m, radians=False)
    return x, y, z

def ecef_to_gps(x, y, z):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat,  alt_in_m = pyproj.transform(ecef, lla, x, y, z, radians=False)
    alt_in_ft = round(alt_in_m * 0.3048, 2)
    return lat, lon, alt_in_ft

def num_sampling_point(start_time, time_column, sampling_time):
    times = np.array(time_column)
    diff_times = times - start_time
    to_seconds = np.vectorize(lambda x: x.seconds)
    diff_times = to_seconds(diff_times)
    sampled_data = (diff_times % sampling_time)

    filter_data = list(filter(lambda x: x == 0, sampled_data))
    return len(filter_data)

class TrajectoryDataset(Dataset):
    """Custom dataset for trajectory data"""
    
    def __init__(
        self,
        destination: str,
        start: str,
        end: str,
        input_len: int,
        target_len: int,
        data_source: str,
        sampling_time: int,
        h3_resolution: Literal["5", "6", "7", "8", "9", "10", "11", "multi"]="5",
        training_columns: list[str] = TRAINING_COLUMNS,
        columns: list[str] = TRAJECTORY_COLUMNS,
        weather_data_path: str | None = None,  #Change to OG Code - path to a trajectory_weather_features_*.parquet; None means no weather (fully backward compatible)
        weather_columns: list[str] = SEQUENTIAL_WEATHER_COLUMNS,
    ) -> None:
        self.destination = destination
        self.start = start
        self.end = end
        self.data_source = data_source
        self.sampling_time = sampling_time
        self.h3_resolution = h3_resolution

        self.input_len = input_len
        self.target_len = target_len
        self.training_columns = training_columns

        self.columns = columns
        self.feature_columns = columns
        self.columns_idxs = {col: idx for idx, col in enumerate(self.columns)}

        #Change to OG Code - random_state pinned: QuantileTransformer subsamples internally using the
        #unseeded global numpy random state, so vocab sizes (and thus embedding/dense layer dimensions)
        #were silently different every process run, making every saved checkpoint unloadable after a
        #kernel restart. Confirmed empirically: two runs on identical data gave altitude vocab sizes of
        #1783 vs 1789. 42 matches the manual_seed(42) already used for the train/val/test split.
        self.scalers = [QuantileTransformer(output_distribution="normal", random_state=42) for _ in range(5)]

        #Change to OG Code - weather is off by default; when a path is given, weather_columns get their own
        #scalers (separate from the 5 fixed lat/lon/altitude/x/y ones above) and their own vocabs, and are
        #encoded into the model but never predicted (see CLAUDE.md "Weather integration design")
        self.use_weather = weather_data_path is not None
        self.weather_data_path = weather_data_path
        self.weather_columns = weather_columns if self.use_weather else []
        self.weather_scalers = [QuantileTransformer(output_distribution="normal", random_state=42) for _ in self.weather_columns]

        self.trajectory_features = None

        self._read_data()
        self._attach_weather_features()  #Change to OG Code - no-op when use_weather is False
        self._transform_data()


    def _reverse_scale_data(self, scaled_data, idx) -> None:
        scaled_data = scaled_data.astype(float) + self.mins[idx]
        return self.scalers[idx].inverse_transform(scaled_data.reshape(-1, 1))

    def _build_vocab(self, tokens_counter, specials=None):
        sorted_by_freq_tuples = sorted(tokens_counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        return vocab(ordered_dict, specials=specials)
    
    def _read_data(self) -> None:
        """Read raw data"""
        self.agg_map_list = [pl.col(col_name) for col_name in TRAJECTORY_COLUMNS]
  
        additional_columns = [
           "flight_id",
           "start_timestamp",
           "dist_from_orig",
           "dist_to_dest",
           "gps_altitude",
           "unix_timestamp",
           "x",
           "y",
           "z",
           "year",
           "month",
           "day"
        ] 
        if "destination" not in self.columns:
            additional_columns.append("destination")

        if "origin" not in self.columns:
            additional_columns.append("origin")
        
        if "timestamp" not in self.columns:
            additional_columns.append("origin")
        
        partitions_to_read = filter_partitions(
            destination=self.destination,
            start=datetime.strptime(self.start, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, tzinfo=timezone.utc
            ),
            end=datetime.strptime(self.end, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, tzinfo=timezone.utc
            ),
        )

        explode_columns = [
            "dist_from_orig",
            "dist_to_dest",
            "gps_altitude",
            "unix_timestamp",
            "x",
            "y",
            "z",
            *TRAJECTORY_COLUMNS
        ]

        self.data: pd.DataFrame = (
            (
                pl.scan_parquet(
                    source=self.data_source,
                    hive_schema={
                        "destination": pl.String,
                        "year": pl.Int64,
                        "month": pl.Int64,
                        "day": pl.Int64,
                    }, #Changed from OG Code - Int64 used instead of Int32
                    cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast'), #Addition to OG Code - Downcast the ns columns to match start_timestamp across all partitions
                )
                .select([*self.columns, *additional_columns])
                .filter(partitions_to_read)
                .filter(pl.col("destination") == self.destination) #Addition to OG Code - Additional filter added to stop other airport data from leaking in.
                .with_columns(
                    pl.struct(["start_timestamp", "timestamp"]).map_elements(
                        lambda row: num_sampling_point(
                            row["start_timestamp"],
                            row["timestamp"],
                            self.sampling_time
                        )
                    ).alias("num_sampled_points")
                )
                .filter(
                    (pl.col("origin") != pl.col("destination")) &
                    (~pl.col("flight_id").is_in(WEIRD_IDS_NOT_FILTERED)) &
                    (pl.col("num_sampled_points") >= self.input_len + self.target_len)
                )
                .unique()
                .explode(explode_columns)
                .sort(["flight_id", "timestamp"])
                .with_columns( 
                    pl.struct(["latitude", "longitude", "dist_from_orig", "dist_to_dest"]).map_elements(
                        lambda row: hex(latlon_to_h3(
                            row["latitude"],
                            row["longitude"],
                            row["dist_from_orig"],
                            row["dist_to_dest"],
                            self.h3_resolution
                        ))
                    ).alias("h3_cell"),
                    (
                        (pl.col("dist_from_orig") > 5) & (pl.col("dist_to_dest") > 5)
                    ).alias("far_from_airport"),
                )
                .filter(
                    pl.col("far_from_airport") &
                    (pl.col("latitude") >= MIN_LAT) &
                    (pl.col("latitude") <= MAX_LAT) &
                    (pl.col("longitude") >= MIN_LON) &
                    (pl.col("longitude") <= MAX_LON)
                )
                .with_columns(
                    pl.col("altitude").map_elements(lambda x: x * 0.3048).alias("altitude"),
                    pl.col("gps_altitude").map_elements(lambda x: x * 0.3048).alias("gps_altitude")
                )
                .select(
                    pl.col("flight_id"),
                    pl.col("timestamp").dt.replace_time_zone(None), #Change from OG code - Datetime deprecated, replacing it.
                    pl.col("latitude").cast(pl.Float32),
                    pl.col("longitude").cast(pl.Float32),
                    pl.col("altitude").cast(pl.Float32),
                    pl.col("gps_altitude").cast(pl.Float32),
                    pl.col("track").cast(pl.Float32),
                    pl.col("ground_speed").cast(pl.Float32),
                    pl.col("vertical_rate").cast(pl.Float32),
                    pl.col("h3_cell").cast(pl.String),
                    pl.col("x").cast(pl.Float32),
                    pl.col("y").cast(pl.Float32),
                    pl.col("z").cast(pl.Float32),
                    pl.col("dist_from_orig"),
                    pl.col("dist_to_dest"),
                    pl.col("start_timestamp"),
                    pl.col("unix_timestamp"),
                    pl.col("origin"),
                    number_points=pl.col("num_sampled_points").cast(pl.Int64),
                    diff_time=((pl.col("timestamp") - pl.col("start_timestamp")).dt.total_seconds().cast(pl.Int64)), #Change from OG code - Forced diff_time to integer seconds.
                    year=pl.col("timestamp").dt.year().cast(pl.Int16),
                    month=pl.col("timestamp").dt.month().cast(pl.Int8),
                    day=pl.col("timestamp").dt.day().cast(pl.Int8),
                )
                .filter((pl.col("diff_time").cast(pl.Int64) % self.sampling_time) == 0) #Change from OG Code
                .filter((pl.col("diff_time") % self.sampling_time) == 0)
            )
            .collect()
        )

    def _attach_weather_features(self) -> None:
        """Change to OG Code - left-join ERA5 weather (u,v,t,gps_altitude_m already exact-altitude
        interpolated per trajectory point, see h3_alt_layer_generation.ipynb) onto self.data by
        (flight_id, timestamp), then derive the engineered feature set from CLAUDE.md's "Feature set"
        section: along-track wind and ISA-deviation temperature, each as a trailing rolling
        mean/std over the input window plus a step-to-step time gradient. Weather is a known
        covariate for the model (see "Weather integration design"), never a prediction target."""
        if not self.use_weather:
            return

        weather = pl.read_parquet(self.weather_data_path).select(
            pl.col("flight_id"),
            pl.col("timestamp").cast(pl.Datetime("us")),  #Change to OG Code - explicit cast, known datetime-precision hazard
            pl.col("u"), pl.col("v"), pl.col("t"),
        )
        self.data = self.data.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

        n_before = self.data.height
        self.data = self.data.join(weather, on=["flight_id", "timestamp"], how="left")
        n_unmatched = self.data.filter(pl.col("u").is_null()).height
        if n_unmatched:
            raise ValueError(
                f"{n_unmatched} of {n_before} trajectory points have no matching weather row "
                f"(joined on flight_id, timestamp against {self.weather_data_path}). "
                f"start/end here must be within the weather file's own date range."
            )

        track_rad = (pl.col("track") * np.pi / 180.0)
        along_track = pl.col("u") * track_rad.sin() + pl.col("v") * track_rad.cos()
        wind_magnitude = (pl.col("u") ** 2 + pl.col("v") ** 2).sqrt()
        t_isa = (
            pl.when(pl.col("gps_altitude") <= TROPOPAUSE_HEIGHT_M)
            .then(ISA_SEA_LEVEL_TEMP_K - ISA_LAPSE_RATE_K_PER_M * pl.col("gps_altitude"))
            .otherwise(ISA_TROPOPAUSE_TEMP_K)
        )
        isa_dev = pl.col("t") - t_isa

        #Change to OG Code - display-only weather columns (wind speed/direction, temperature in C),
        #not part of training_columns/weather_columns so they don't affect the model at all. Kept
        #around after u/v/t are dropped, purely so map hover_data (experiments_weather.ipynb) can
        #show human-readable weather instead of nothing. Wind direction is meteorological convention
        #(compass bearing the wind blows FROM, not toward), verified against known cases.
        wind_direction = ((pl.arctan2(pl.col("u"), pl.col("v")).degrees() + 180) % 360)

        self.data = (
            self.data
            .sort(["flight_id", "timestamp"])
            .with_columns(along_track.alias("_along_track"), wind_magnitude.alias("_wind_mag"), isa_dev.alias("_isa_dev"))
            .with_columns(
                pl.col("_along_track").rolling_mean(window_size=self.input_len, min_periods=1).over("flight_id").alias("along_track_mean"),
                pl.col("_isa_dev").rolling_mean(window_size=self.input_len, min_periods=1).over("flight_id").alias("isa_dev_mean"),
                pl.col("_isa_dev").rolling_std(window_size=self.input_len, min_periods=1).over("flight_id").fill_null(0.0).alias("isa_dev_std"),
                (pl.col("_along_track").diff().over("flight_id") / pl.col("diff_time").diff().over("flight_id")).fill_null(0.0).fill_nan(0.0).alias("along_track_grad"),
                (pl.col("_wind_mag").diff().over("flight_id") / pl.col("diff_time").diff().over("flight_id")).fill_null(0.0).fill_nan(0.0).alias("wind_mag_grad"),
                (pl.col("_isa_dev").diff().over("flight_id") / pl.col("diff_time").diff().over("flight_id")).fill_null(0.0).fill_nan(0.0).alias("isa_dev_grad"),
                pl.col("_wind_mag").alias("wind_speed_ms"),
                wind_direction.alias("wind_direction_deg"),
                (pl.col("t") - 273.15).alias("temperature_C"),
            )
            .drop("_along_track", "_wind_mag", "_isa_dev", "u", "v", "t")
        )

    def _transform_data(self):
        transformed_data = (
            self.data.select(
                pl.col("latitude"),
                pl.col("longitude"),
                pl.col("altitude"),
                pl.col("x"),
                pl.col("y"),
                pl.col("flight_id"),
                pl.col("timestamp"),
                pl.col("h3_cell").map_elements(lambda x: str(x) ),
                pl.col("diff_time").map_elements(lambda x: str(x)),
                *[pl.col(c) for c in self.weather_columns],  #Change to OG Code - carry the engineered weather columns through
                x_raw=pl.col("x"),
                y_raw=pl.col("y"),
                z_raw=pl.col("z"),
                latitude_raw=pl.col("latitude"),
                longitude_raw=pl.col("longitude"),
                gps_altitude_raw=pl.col("gps_altitude")
            )
        ).to_pandas()


        numeric_cols = ["latitude", "longitude", "altitude", "x", "y"]
        passthrough_cols = ["flight_id", "timestamp", "h3_cell", "diff_time", "x_raw", "y_raw", "z_raw", "latitude_raw", "longitude_raw", "gps_altitude_raw"]

        scaled_data = transformed_data.copy()
        for col, scaler in zip(numeric_cols, self.scalers):
            scaled_data[col] = scaler.fit_transform(scaled_data[[col]])
        for col, scaler in zip(self.weather_columns, self.weather_scalers):  #Change to OG Code - weather gets its own scaler set, same fit_transform pattern
            scaled_data[col] = scaler.fit_transform(scaled_data[[col]])

        scaled_data = scaled_data[numeric_cols + self.weather_columns + passthrough_cols].sort_values(["flight_id", "timestamp", "diff_time"])
        self.transformed_data = scaled_data.drop_duplicates(subset=["latitude", "longitude", "altitude", "x", "y", "flight_id", "timestamp", "h3_cell"], keep="last")

        self.mins = self.transformed_data[["latitude", "longitude", "altitude", "x", "y"]].min().to_numpy().astype(np.float32)
        self.transformed_data["latitude"] = self.transformed_data["latitude"].apply(lambda x: str(round(x - self.mins[0], 3)))
        self.transformed_data["longitude"] = self.transformed_data["longitude"].apply(lambda x: str(round(x - self.mins[1], 3)))
        self.transformed_data["altitude"] = self.transformed_data["altitude"].apply(lambda x: str(round(x - self.mins[2], 3)))
        self.transformed_data["x"] = self.transformed_data["x"].apply(lambda x: str(round(x - self.mins[3], 3)))
        self.transformed_data["y"] = self.transformed_data["y"].apply(lambda x: str(round(x - self.mins[4], 3)))

        #Change to OG Code - same "subtract min, round, stringify" binning as the block above, looped over the weather columns
        self.weather_mins = self.transformed_data[self.weather_columns].min().to_numpy().astype(np.float32) if self.weather_columns else np.array([])
        for i, col in enumerate(self.weather_columns):
            self.transformed_data[col] = self.transformed_data[col].apply(lambda x, i=i: str(round(x - self.weather_mins[i], 3)))

        self.vocabs = {
            feature_name: self._build_vocab(Counter(self.transformed_data[feature_name].to_list()), specials=["START", "END", "PAD"])
            for feature_name in ["latitude", "longitude", "altitude", "x", "y", "diff_time", "h3_cell", *self.weather_columns]
        }

        self.transformed_data["diff_time"] = self.transformed_data["diff_time"].apply(lambda x: self.vocabs["diff_time"][x])
        self.transformed_data["latitude"] = self.transformed_data["latitude"].apply(lambda x: self.vocabs["latitude"][x])
        self.transformed_data["longitude"] = self.transformed_data["longitude"].apply(lambda x: self.vocabs["longitude"][x])
        self.transformed_data["altitude"] = self.transformed_data["altitude"].apply(lambda x: self.vocabs["altitude"][x])
        self.transformed_data["x"] = self.transformed_data["x"].apply(lambda x: self.vocabs["x"][x])
        for col in self.weather_columns:  #Change to OG Code - vocab lookup for weather, same pattern as the columns above
            self.transformed_data[col] = self.transformed_data[col].apply(lambda x, col=col: self.vocabs[col][x])
        self.transformed_data["y"] = self.transformed_data["y"].apply(lambda x: self.vocabs["y"][x])
        self.transformed_data["h3_cell"] = self.transformed_data["h3_cell"].apply(lambda x: self.vocabs["h3_cell"][x])

        
        self.trajectory_features = (
            pl.from_pandas(self.transformed_data)
            .sort(["flight_id", "timestamp"])
            .group_by("flight_id", maintain_order=True)
            .agg([*self.training_columns, "x_raw", "y_raw", "z_raw", "latitude_raw", "longitude_raw", "gps_altitude_raw", *self.weather_columns])  #Change to OG Code - weather columns appended last, see __getitem__ offset
        ).to_numpy()

    def __len__(self) -> int:
        """Get length of dataset"""
        return len(self.trajectory_features)
    
    def __getitem__(self, index: int) -> tuple[ArrayLike, ArrayLike]:
        """Get one item on specified index"""
        trajectory_data = self.trajectory_features[index] 

        raw_data = {
            (f"{col_name}" if "_raw" in col_name else f"{col_name}_raw"): 
            trajectory_data[idx+len(self.training_columns)+1][self.input_len:self.input_len + self.target_len]
            for idx, col_name in enumerate([
                "x_raw", "y_raw", "z_raw", "latitude_raw", "longitude_raw", "gps_altitude_raw"
            ])
        }
        input_data = {
            f"{col_name}_in": trajectory_data[idx+1][:self.input_len]
            for idx, col_name in enumerate(self.training_columns)
        }
        output_data = {
            f"{col_name}_out": np.concatenate(
                [
                    [self.vocabs[col_name]["START"]],
                    trajectory_data[idx+1][self.input_len:self.input_len + self.target_len],
                    [self.vocabs[col_name]["END"]]
                ]
            )
            for idx, col_name in enumerate(self.training_columns)
        }

        #Change to OG Code - weather encoder input (matches x_i: plain input_len slice, no START/END) and
        #weather decoder input (matches the target window, but no START/END either - weather is never
        #autoregressively generated, it's fully known upfront, see the "Plumbing plan" discussion). Keyed
        #with "_wxenc"/"_wxdec" rather than "_in"/"_out" so pad_collate (dataloader.py) routes them to
        #their own tensors instead of getting silently mixed into the predicted-feature tensors.
        weather_offset = len(self.training_columns) + 6 + 1
        weather_encoder_data = {
            f"{col_name}_wxenc": trajectory_data[idx + weather_offset][:self.input_len]
            for idx, col_name in enumerate(self.weather_columns)
        }
        weather_decoder_data = {
            f"{col_name}_wxdec": trajectory_data[idx + weather_offset][self.input_len:self.input_len + self.target_len]
            for idx, col_name in enumerate(self.weather_columns)
        }

        item_data = dict(input_data, **output_data)
        item_data.update(raw_data)
        item_data.update(weather_encoder_data)
        item_data.update(weather_decoder_data)
        item_data["flight_id"] = trajectory_data[0]
        return item_data