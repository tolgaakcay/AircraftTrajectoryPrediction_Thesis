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
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn_pandas import DataFrameMapper

import numpy as np
from numpy.typing import ArrayLike
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.original.utils_soares import filter_partitions
from src.data.original.schemas_soares import (
    trajectories_schema,
    training_trajectories_schema,
    non_seq_numerical_schema,
    flight_categories_schema,
    temporal_categories_schema,
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
        
        self.scalers = [QuantileTransformer(output_distribution="normal") for _ in range(5)]

        self.trajectory_features = None

        self._read_data()
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
                        "year": pl.Int32,
                        "month": pl.Int32,
                        "day": pl.Int32,
                    },
                )
                .select([*self.columns, *additional_columns])
                .filter(partitions_to_read)
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
                    pl.struct(["dist_from_orig","dist_to_dest"]).map_elements(
                        lambda x: far_from_airport(x["dist_from_orig"], x["dist_to_dest"])
                    ).alias("far_from_airport"),
                )
                .filter(
                    (pl.col("far_from_airport") == True) &
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
                    pl.col("timestamp").dt.datetime(),
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
                    diff_time=(pl.col("timestamp") - pl.col("start_timestamp")).dt.total_seconds(),
                    year=pl.col("timestamp").dt.year().cast(pl.Int16),
                    month=pl.col("timestamp").dt.month().cast(pl.Int8),
                    day=pl.col("timestamp").dt.day().cast(pl.Int8),
                )
                .fill_null(strategy="forward")
                .filter((pl.col("diff_time") % self.sampling_time) == 0)
            )
            .collect()
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
                x_raw=pl.col("x"),
                y_raw=pl.col("y"),
                z_raw=pl.col("z"),
                latitude_raw=pl.col("latitude"),
                longitude_raw=pl.col("longitude"),
                gps_altitude_raw=pl.col("gps_altitude")
            )
        ).to_pandas()
        
        
        mapper = DataFrameMapper([
            (["latitude"], self.scalers[0]),
            (["longitude"], self.scalers[1]),
            (["altitude"], self.scalers[2]),
            (["x"], self.scalers[3]),
            (["y"], self.scalers[4]),
            (["flight_id", "timestamp", "h3_cell", "diff_time", "x_raw", "y_raw", "z_raw", "latitude_raw", "longitude_raw", "gps_altitude_raw"], FunctionTransformer(lambda x: x))
        ])
        scaled_features = mapper.fit_transform(transformed_data.copy())
        scaled_data = pd.DataFrame(scaled_features, index=transformed_data.index, columns=transformed_data.columns).sort_values(["flight_id", "timestamp", "diff_time"])
        self.transformed_data = scaled_data.drop_duplicates(subset=["latitude", "longitude", "altitude", "x", "y", "flight_id", "timestamp", "h3_cell"], keep="last")

        self.mins = self.transformed_data[["latitude", "longitude", "altitude", "x", "y"]].min().to_numpy().astype(np.float32)
        self.transformed_data["latitude"] = self.transformed_data["latitude"].apply(lambda x: str(round(x - self.mins[0], 3)))
        self.transformed_data["longitude"] = self.transformed_data["longitude"].apply(lambda x: str(round(x - self.mins[1], 3)))
        self.transformed_data["altitude"] = self.transformed_data["altitude"].apply(lambda x: str(round(x - self.mins[2], 3)))
        self.transformed_data["x"] = self.transformed_data["x"].apply(lambda x: str(round(x - self.mins[3], 3)))
        self.transformed_data["y"] = self.transformed_data["y"].apply(lambda x: str(round(x - self.mins[4], 3)))
        
        
        
        self.vocabs = {
            feature_name: self._build_vocab(Counter(self.transformed_data[feature_name].to_list()), specials=["START", "END", "PAD"]) 
            for feature_name in ["latitude", "longitude", "altitude", "x", "y", "diff_time", "h3_cell"]
        }

        self.transformed_data["diff_time"] = self.transformed_data["diff_time"].apply(lambda x: self.vocabs["diff_time"][x])
        self.transformed_data["latitude"] = self.transformed_data["latitude"].apply(lambda x: self.vocabs["latitude"][x])
        self.transformed_data["longitude"] = self.transformed_data["longitude"].apply(lambda x: self.vocabs["longitude"][x])
        self.transformed_data["altitude"] = self.transformed_data["altitude"].apply(lambda x: self.vocabs["altitude"][x])
        self.transformed_data["x"] = self.transformed_data["x"].apply(lambda x: self.vocabs["x"][x])
        self.transformed_data["y"] = self.transformed_data["y"].apply(lambda x: self.vocabs["y"][x])
        self.transformed_data["h3_cell"] = self.transformed_data["h3_cell"].apply(lambda x: self.vocabs["h3_cell"][x])

        
        self.trajectory_features = (
            pl.from_pandas(self.transformed_data)
            .sort(["flight_id", "timestamp"])
            .group_by("flight_id", maintain_order=True)
            .agg([*self.training_columns, "x_raw", "y_raw", "z_raw", "latitude_raw", "longitude_raw", "gps_altitude_raw"])
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

        item_data = dict(input_data, **output_data)
        item_data.update(raw_data)
        item_data["flight_id"] = trajectory_data[0]
        return item_data