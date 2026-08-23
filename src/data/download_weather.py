from __future__ import annotations
from pathlib import Path
import cdsapi
import polars as pl
from datetime import datetime, timedelta
import configparser

AIRPORT = "LFBO"
 
CDS_CONFIG_PATH = "../secrets.conf"
 
 
def load_cds_client() -> cdsapi.Client:
    """Read CDS credentials from the project config file (same pattern as the
    opensky credentials in secrets.conf) and build a client, instead of
    relying on a ~/.cdsapirc file in the home directory."""
    config = configparser.ConfigParser()
    read_ok = config.read(CDS_CONFIG_PATH)
    if not read_ok:
        raise FileNotFoundError(
            f"Could not read config file at {CDS_CONFIG_PATH}. "
            f"Expected a [cds] section with 'url' and 'key' entries."
        )
    if "cds" not in config:
        raise KeyError(
            f"No [cds] section found in {CDS_CONFIG_PATH}. "
            f"Add:\n\n[cds]\nurl = https://cds.climate.copernicus.eu/api\nkey = YOUR_TOKEN\n"
        )
    return cdsapi.Client(
        url=config["cds"]["url"],
        key=config["cds"]["key"],
    )
 
DATA_SOURCE = Path("/home/tolgaakcay/projects/AircraftTrajectoryPrediction_Thesis/src/data/raw")
BBOX_MARGIN_DEG = 2.0
GRID_STEP_DEG = 0.25
 
def compute_area_from_trajectories(
    data_source: Path,
    airport: str,
    margin_deg: float = BBOX_MARGIN_DEG,
) -> list[float]:
    """Derive the ERA5 bounding box from the actual latitude/longitude extent
    of the trajectory data for this airport, so the weather grid covers the
    whole flight area rather than just the airport location."""
    lf = pl.scan_parquet(
        source=data_source,
        hive_schema={
            "destination": pl.String,
            "year": pl.Int64,
            "month": pl.Int64,
            "day": pl.Int64,
        },
        cast_options=pl.ScanCastOptions(datetime_cast="nanosecond-downcast"),
    ).filter(pl.col("destination") == airport)
 
    bounds = (
        lf.select(
            pl.col("latitude").explode().min().alias("min_lat"),
            pl.col("latitude").explode().max().alias("max_lat"),
            pl.col("longitude").explode().min().alias("min_lon"),
            pl.col("longitude").explode().max().alias("max_lon"),
        )
        .collect()
        .row(0, named=True)
    )
 
    north = bounds["max_lat"] + margin_deg
    south = bounds["min_lat"] - margin_deg
    west = bounds["min_lon"] - margin_deg
    east = bounds["max_lon"] + margin_deg
 
    # snap outward to the ERA5 grid so no edge trajectory points fall outside
    import math
    north = math.ceil(north / GRID_STEP_DEG) * GRID_STEP_DEG
    south = math.floor(south / GRID_STEP_DEG) * GRID_STEP_DEG
    west = math.floor(west / GRID_STEP_DEG) * GRID_STEP_DEG
    east = math.ceil(east / GRID_STEP_DEG) * GRID_STEP_DEG
 
    return [round(north, 2), round(west, 2), round(south, 2), round(east, 2)]
 
START_DATE = datetime(year=2024, month=10, day=16)
END_DATE = datetime(year=2024, month=10, day=30)

num_days = (END_DATE - START_DATE).days
date_list = [
    START_DATE + timedelta(days=x)
    for x in range(num_days)
]

TIMES = [f"{h:02d}:00" for h in range(24)]

PRESSURE_LEVELS = [
    "150", "175", "200", "225", "250", "300", "350", "400", "450","500", "550", "600", "650", "700","750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000"
]

VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "geopotential",
    "specific_humidity",
]

OUTPUT_DIR = Path("/home/tolgaakcay/projects/AircraftTrajectoryPrediction_Thesis/src/data/weather")  #Change to OG Code - folder renamed from weather_pressure


def output_path_for_day(day: datetime) -> Path:
    """Change to OG Code - one file per day, partitioned year=/month=/day= like download_raw.py's
    raw trajectory output, instead of one file spanning the whole requested range. Creates the
    partition folders if they don't exist yet."""
    day_dir = OUTPUT_DIR / f"year={day.year}" / f"month={day.month}" / f"day={day.day}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"era5_{AIRPORT}_{day.strftime('%Y-%m-%d')}.grib"


def main() -> None:
    client = load_cds_client()

    area = compute_area_from_trajectories(DATA_SOURCE, AIRPORT)
    print(f"Bounding box [N, W, S, E] from trajectory extent: {area}")
    print(f"Pressure levels: {PRESSURE_LEVELS}")
    print(f"{len(date_list)} day(s) to download: {[d.strftime('%Y-%m-%d') for d in date_list]}")
    print("This may take several minutes to hours per day depending on CDS queue.")

    failed_days = []
    for day in date_list:
        output_file = output_path_for_day(day)
        request = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(day.year)],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            "time": TIMES,
            "pressure_level": PRESSURE_LEVELS,
            "area": area,
            "data_format": "grib",
            "download_format": "unarchived",
        }

        print(f"\nRequesting ERA5 pressure-level data for {AIRPORT}, {day.strftime('%Y-%m-%d')}")
        print(f"Target: {output_file}")

        try:
            client.retrieve("reanalysis-era5-pressure-levels", request).download(str(output_file))
            print(f"Done. Saved to {output_file}")
        except Exception as e:
            print(f"FAILED for {day.strftime('%Y-%m-%d')}: {e!r}")
            failed_days.append(day.strftime('%Y-%m-%d'))

    if failed_days:
        print(f"\n{len(failed_days)} of {len(date_list)} day(s) failed: {failed_days}")
    else:
        print(f"\nAll {len(date_list)} day(s) downloaded successfully.")


if __name__ == "__main__":
    main()
