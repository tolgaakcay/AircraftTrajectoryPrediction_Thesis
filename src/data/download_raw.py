# %%
# This Python file has been duplicated from the original "download_raw.py" by Natalia Souares
# Its purpose is to increase the 36-day data period from the original file to 120-days and observe the difference in accuracy of complexity increase in data.
# Project done by Tolga Akcay for the Chair of Big Geospatial Data at the Technical University of Munich in 2026.

from __future__ import annotations
from pathlib import Path

import os
import argparse
import configparser
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime, timedelta

import pyproj
import pandas as pd
import numpy as np
import configparser
from sklearn.metrics.pairwise import haversine_distances
from math import radians, degrees

import traffic
from traffic.data import opensky, aircraft, airports
from traffic.core import Flight, Traffic
from openmeteopy import OpenMeteo
from openmeteopy.hourly import HourlyHistorical
from openmeteopy.daily import DailyHistorical
from openmeteopy.options import HistoricalOptions
from openmeteopy.utils.constants import *

from schemas import (
    weather_schema,
    adsb_schema
)

# %%
config = configparser.ConfigParser()
config.read("../secrets.conf")

traffic.config["opensky"] = config["opensky"]
opensky.rest_client.username = config["opensky"]["username"] 
opensky.rest_client.password = config["opensky"]["password"]
os.environ["trino_username"] = config["opensky"]["username"]
os.environ["trino_password"] = config["opensky"]["password"]

# %%
DESTINATION = "EDJA"  #Change from Original Code 
START_DATE = datetime(year=2024, month=10, day=2) #Change from Original Code  
END_DATE = datetime(year=2024, month=10, day=6) #Change from Original Code 

ADSB_SCHEMA = list(adsb_schema.keys())
WEATHER_COLUMNS = list(weather_schema.keys())

GRID_RESOLUTION = 5
MIN_CRUISE_ALTITUDE = 28000 #ft
MIN_NOT_TAXIING_SPEED = 110 #kt

MAX_LAT = 90.
MAX_LON = 180.
MAX_ALTITUDE = 43100 #ft
MAX_SPEED = 600 #knots
MAX_TRACK = 360 #degrees
MAX_RATE = 3000 #ft./min

MIN_LAT = -90.
MIN_LON = -180.
MIN_ALTITUDE = 0.0
MIN_RATE = -MAX_RATE
MIN_SPEED = 0.0
MIN_TRACK = 0.0


parser = argparse.ArgumentParser(
    prog='Spacetimeformer train',
    description='Training of spacetimeformer model on the flight trajectory data',
)
parser.add_argument(
    "-a", "--airport",
    action="store",
    help="Flights destination airport ICAO code",
    default="LFBO"
)
parser.add_argument(
    "-s", "--startDate",
    action="store",
    help="Start date to collect data from",
    default="2023-1-1"
)
parser.add_argument(
    "-n", "--numDates",
    action="store",
    help="Total number of days to collect",
    default=365
)

COLUMNS_AGGREGATION = {
    "timestamp": list,
    "unix_timestamp": list,
    "start_timestamp": "first",
    "end_timestamp": "last",
    "icao24": "last",
    "latitude": list,
    "longitude": list,
    "x": list,
    "y": list,
    "z": list,
    "altitude": list,
    "gps_altitude": list,
    "track": list,
    "ground_speed": list,
    "vertical_rate": list,
    "callsign": "last",
    "origin": "last",
    "destination": "last",
    "time_diff": "mean",
    "dist_from_orig": list,
    "dist_to_dest": list,
}

EARTH_RADIUS_M = 6378000

def download_check(airport: str, day: datetime, raw_path: str = "./raw") -> bool:
    partition_dir = (
        Path(raw_path)
        / f"destination={airport}"
        / f"year={day.year}"
        / f"month={day.month}"
        / f"day={day.day}"
    )
    return partition_dir.is_dir() and any(partition_dir.glob("*.parquet"))

# %%
def limit_values(value, min_threshold, max_threshold):
    if value < min_threshold or value > max_threshold:
        return None
    return value

def deg_to_rad(deg: float | list[float]) -> float | list[float]:
    if isinstance(deg, list):
        rad = [radians(deg_coord) for deg_coord in deg]
    else:
        rad = radians(deg)
    return rad

def rad_to_deg(rad: float | list[float]) -> float | list[float]:
    if isinstance(rad, list):
        deg = [degrees(rad_coord) for rad_coord in rad]
    else:
        deg = degrees(rad)
    return deg

def get_geo_distance(
        origin: tuple[float, float],
        destination: tuple[float, float],
        unit: Literal["km", "m"] = "km",
    ) -> float:

    origin_in_radians = [radians(deg_coord) for deg_coord in origin]
    destination_in_radians = [radians(deg_coord) for deg_coord in destination]
    result = haversine_distances([origin_in_radians, destination_in_radians])
    result = result[np.where(result != 0.)][0]
    if unit == "km":
        return result * EARTH_RADIUS_M / 1000
    return result * EARTH_RADIUS_M

def gps_to_ecef(lat, lon, alt):
    alt_in_m = alt * 0.3048
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt_in_m, radians=False)
    return x, y, z

def compute_xyz(row):
    return gps_to_ecef(row["latitude"], row["longitude"], row["gps_altitude"])

def far_from_airport(dist_from_orig, dist_to_dest, threshold = 5):
    if dist_from_orig > threshold and dist_to_dest > threshold:
        return True
    return False

def resample_times(traffic_data: Traffic, rule: str | int = "1min"):
    dfs = []
    for flight in traffic_data:
        original_df = flight.data

        original_df["latitude"] = original_df["latitude"].map(lambda lat: limit_values(lat, MIN_LAT, MAX_LAT))
        original_df["latitude"] = original_df["latitude"].map(lambda lon: limit_values(lon, MIN_LON, MAX_LON))
        original_df["altitude"] = original_df["altitude"].map(lambda alt: limit_values(alt, MIN_ALTITUDE, MAX_ALTITUDE))
        original_df["track"] = original_df["track"].map(lambda track: limit_values(track, MIN_TRACK, MAX_TRACK))
        original_df["groundspeed"] = original_df["groundspeed"].map(lambda g_speed: limit_values(g_speed, MIN_SPEED, MAX_SPEED))
        original_df["vertical_rate"] = original_df["vertical_rate"].map(lambda v_rate: limit_values(v_rate, MIN_RATE, MAX_RATE))

        new_times = pd.date_range(
            start=original_df["timestamp"].min(),
            end=original_df["timestamp"].max(),
            freq=rule
        )
        resampled_times = pd.DataFrame(
            data=zip(np.full(new_times.shape, flight.flight_id), new_times),
            columns=["flight_id", "timestamp"]
        )
        
        resampled_times["datetime"] = pd.to_datetime(resampled_times["timestamp"], unit="ms").dt.tz_localize(None)
        resampled_times["timestamp"] = (resampled_times["timestamp"].astype("int") / 1e6).astype("str")
        original_df["timestamp"] = (original_df["timestamp"].astype("int") / 1e6).astype("str")
        resampled_flights = pd.merge(
            left=resampled_times, 
            right=original_df,
            how="left",
            on=["flight_id", "timestamp"],
        )

        interpolated_flights_numerical = resampled_flights[[
            "datetime",
            "latitude",
            "longitude",
            "altitude",
            "gps_altitude",
            "groundspeed",
            "track",
            "vertical_rate"
        ]].set_index("datetime")
        interpolated_flights_numerical.interpolate(
            method='cubicspline',
            limit_area='inside',
            limit_direction='both',
            inplace=True,
        )
        interpolated_flights_numerical["flight_id"] = flight.flight_id


        interpolated_flights_categorical = resampled_flights[[
            "datetime",
            "icao24",
            "callsign",
            "origin",
            "destination",
        ]].set_index("datetime")
        interpolated_flights_categorical.ffill(
            limit_area='inside',
            inplace=True,
        )
        interpolated_flights_categorical["flight_id"] = flight.flight_id

        resampled_interpolated_flights = pd.merge(
            left=interpolated_flights_numerical, 
            right=interpolated_flights_categorical,
            how="left",
            on=["flight_id", "datetime"],
        ).reset_index()
        dfs.append(resampled_interpolated_flights)

    return pd.concat(dfs)

def check_go_arounds(flight: Flight) -> bool:
    try: 
        has_go_around = flight.go_around(DESTINATION).has()
    except (TypeError, ValueError) as e:
        print(f"go_around check failed for {flight.flight_id}: {e}")
        has_go_around = False
    return has_go_around

def download_adsb_history(
    airport: str,
    start: datetime,
    end: datetime
) -> Traffic:
    """Get historical ADS-B data of complete flights from the OpenSky Network history"""
    adsb_history = opensky.history(  # type: ignore
        start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        arrival_airport=airport,
        selected_columns=tuple(ADSB_SCHEMA),
    )
    return (adsb_history
            .filter()
            .clean_invalid()
            .assign_id()
            .rename(columns={
                "StateVectorsData4.time": "timestamp",
                "estdepartureairport": "origin",
                "estarrivalairport": "destination",
                "geoaltitude": "gps_altitude",
            })
    ).eval()
    
def group_adsb_per_flight(adsb_df: pd.DataFrame) -> pd.DataFrame:
    """Groups flights trajectories"""
    adsb_df["timestamp"] =  pd.to_datetime(adsb_df["timestamp"], unit="ms").dt.tz_localize(None)
    adsb_df["start_timestamp"] = adsb_df["timestamp"]
    adsb_df["end_timestamp"] = adsb_df["timestamp"]
    adsb_df["unix_timestamp"] = (adsb_df["timestamp"].astype("int") / 1e6).astype("int")
    
    grouped_flights = (
        adsb_df
        .groupby("flight_id").agg(COLUMNS_AGGREGATION)
    )
    grouped_flights["flight_id"] = grouped_flights.index
    return grouped_flights

def download_opensky_data(
    airport: str,
    start: datetime,
    end: datetime,
    sampling_interval="30s",
) -> pd.DataFrame:
    opensky_data = download_adsb_history(
        airport,
        start,
        end
    )
    
    adsb_df = (
        resample_times(opensky_data, rule=sampling_interval)
        .dropna()
        .rename(columns={
            "groundspeed": "ground_speed",
            "datetime": "timestamp"
        })
    )
    adsb_df = (
        adsb_df
        .assign(time_diff=adsb_df["timestamp"].diff().dt.seconds.fillna(0))
    )
    adsb_df["x"], adsb_df["y"], adsb_df["z"] = zip(
        *adsb_df[["latitude", "longitude", "gps_altitude"]].apply(compute_xyz, axis=1)
    )

    filtered_adsb_df = adsb_df[
        (adsb_df.ground_speed >= MIN_NOT_TAXIING_SPEED) &
        (adsb_df.origin != adsb_df.destination)
    ]

    aligned_opensky_data = opensky_data.all(f'aligned_on_ils("{airport}")').eval()
    rnwy_df = (
        aligned_opensky_data
        .rename(columns={"ILS": "landing_runway", "bearing": "landing_bearing"})
        .data[["flight_id", "landing_runway", "landing_bearing"]]
        .groupby("flight_id")
        .agg("last")
    )
    go_around_df = pd.DataFrame(
        [check_go_arounds(flight) for flight in aligned_opensky_data],
        index=[flight.flight_id for flight in aligned_opensky_data],
        columns=["has_go_around"]
    )
    enhanced_adsb_df = pd.concat([rnwy_df, go_around_df], axis=1)
    
    historical_adsb_df = filtered_adsb_df.join(
        enhanced_adsb_df,
        how="left",
        on="flight_id"
    )
    return historical_adsb_df.loc[historical_adsb_df["has_go_around"] == False]

def get_weather_info(
    latitude: float,
    longitude: float,
    elevation: float,
    time: datetime,
) -> pd.DataFrame:
    current_time = time.replace(minute=0, second=0, microsecond=0)
    next_time = current_time + timedelta(hours=1)
    
    if abs(current_time - time) < abs(next_time - time):
        query_date_time = current_time.strftime("%Y-%m-%dT%H:%M")
    else:
        query_date_time = next_time.strftime("%Y-%m-%dT%H:%M")
    query_date = time.strftime("%Y-%m-%d")

    hourly = HourlyHistorical()
    daily = DailyHistorical()
    try:
        options = HistoricalOptions(
            latitude=latitude,
            longitude=longitude, 
            elevation=elevation,
            start_date=query_date,
            end_date=query_date
        )
        
        mgr = OpenMeteo(options, hourly.all(), daily.all())
        meteo_all = pd.concat(mgr.get_pandas())

        meteo = meteo_all[
            (meteo_all.index == query_date_time) |
            (meteo_all.index == query_date)
        ][WEATHER_COLUMNS].iloc[0]
    except:
        meteo = pd.DataFrame(np.nan, index=[0], columns=WEATHER_COLUMNS).iloc[0]
    return meteo

# %%
aircrafts_df = aircraft.data[["icao24", "registration", "model", "operator"]]
airports_df = (
    airports.data[["icao", "latitude", "longitude", "altitude", "country", "municipality", "type"]]
    .rename(columns={
        "latitude": "airport_latitude",
        "longitude": "airport_longitude",
        "altitude": "airport_altitude",
    })
)

start_date = START_DATE
end_date = END_DATE
num_days = (end_date - start_date).days
date_list = [
    start_date + timedelta(days=x)
    for x in range(num_days)
]

sampling_interval = "5s"
print("Sampling Interval =", sampling_interval)
if "s" in sampling_interval:
    sampling_interval_secs = int(sampling_interval.split("s")[0])
else:
    sampling_interval_secs = int(sampling_interval.split("min")[0]) * 60
for i, start in enumerate(date_list):
    next_id = min(i + 1, len(date_list) - 1)
    end = date_list[next_id]
    if end == start:
        end = start + timedelta(days=1)

    if download_check(DESTINATION, start):
        print(f"SKIP  {start.date()} (already downloaded)")
        continue
    print(f"FETCH {start.date()} → {end.date()}")

    try:
        opensky_data = download_opensky_data(
            airport=DESTINATION,
            start=start,
            end=end,
            sampling_interval=sampling_interval,
        )

        trajectories = opensky_data.merge(
            airports_df,
            how="inner",
            left_on="origin",
            right_on="icao",
        ).merge(
            airports_df,
            how="inner",
            left_on="destination",
            right_on="icao",
            suffixes=("_orig", "_dest")
        ).merge(
            aircrafts_df,
            how="inner",
            on="icao24"
        )

        trajectories = trajectories.assign(
            dist_from_orig=trajectories.apply(
                lambda row: get_geo_distance(
                    (row["latitude"], row["longitude"]),
                    (row["airport_latitude_orig"], row["airport_longitude_orig"])
                ),
                axis=1
            ),
            dist_to_dest=trajectories.apply(
                lambda row: get_geo_distance(
                    (row["latitude"], row["longitude"]),
                    (row["airport_latitude_dest"], row["airport_longitude_dest"])
                ),
                axis=1
            ),
        )
        trajectories["far_from_airport"] = trajectories.apply(
            lambda row: far_from_airport(
                row["dist_from_orig"],
                row["dist_to_dest"]
            ),
            axis=1
        )
        filtered_trajectories = trajectories.loc[trajectories["far_from_airport"] == True]

        flights_data = group_adsb_per_flight(filtered_trajectories)
        flights_data = flights_data.assign(
            year=flights_data.apply(lambda row: row["start_timestamp"].year, axis=1),
            month=flights_data.apply(lambda row: row["start_timestamp"].month, axis=1),
            day=flights_data.apply(lambda row: row["start_timestamp"].day, axis=1),
            number_points=flights_data.apply(lambda row: len(row["latitude"]), axis=1),
        )
        flights_data["flight_id"] = (
            flights_data["flight_id"]
            + "_"
            + flights_data["year"].astype("str")
            + "_"
            + flights_data["month"].astype("str")
            + "_"
            + flights_data["day"].astype("str")
        )
        filtered_flights_data = flights_data.loc[
            (flights_data["time_diff"] <= sampling_interval_secs) &
            (flights_data["number_points"] >= 30)
        ]
        
        filtered_flights_data.to_parquet(
            path=f"./raw/",
            partition_cols=["destination", "year", "month", "day"],
            index=False,
        )
    except AttributeError as e:
        import traceback
        print(f"SKIPPED {start.date()}: {type(e).__name__}: {e}") #Addition to OG Code - Skip failed days, then print the type of error experienced.
        traceback.print_exc()
        continue

# %%
 