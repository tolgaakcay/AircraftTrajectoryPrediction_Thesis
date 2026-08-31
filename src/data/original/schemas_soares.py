from datetime import datetime


training_trajectories_schema = {
    "h3_cell": float,
    "altitude": float,
    "diff_time": int,
}

training_trajectories_with_latlon_schema = {
    "latitude": float,
    "longitude": float,
    "altitude": float,
    "diff_time": int,
}

training_trajectories_with_xy_schema = {
    "x": float,
    "y": float,
    "altitude": float,
    "diff_time": int,
}

trajectories_schema = {
    "latitude": float,
    "longitude": float,
    "altitude": float,
    "track": float,
    "ground_speed": float,
    "vertical_rate": float,
    "timestamp": int,
}

non_seq_numerical_schema = {
    "airport_latitude_orig": float,
    "airport_longitude_orig": float,
    "airport_altitude_orig": float,
    "airport_latitude_dest": float,
    "airport_longitude_dest": float,
    "airport_altitude_dest": float,
    "winddirection_10m_dest": float,
    "windgusts_10m_dest": float,
    "windspeed_10m_dest": float,
    "windspeed_100m_dest": float,
    "winddirection_100m_dest": float,
    "cloudcover_high_dest": float,
    "cloudcover_mid_dest": float,
    "cloudcover_low_dest": float,
    "apparent_temperature_dest": float,
    "precipitation_dest": float,
    "snowfall_dest": float,
    "winddirection_10m_orig": float,
    "windgusts_10m_orig": float,
    "windspeed_10m_orig": float,
    "windspeed_100m_orig": float,
    "winddirection_100m_orig": float,
    "cloudcover_high_orig": float,
    "cloudcover_mid_orig": float,
    "cloudcover_low_orig": float,
    "apparent_temperature_orig": float,
    "precipitation_orig": float,
    "snowfall_orig": float,
}

flight_categories_schema = {
    "callsign": str,
    "model": str,
    "operator": str,
    "origin": str,
    "country_orig": str,
    "municipality_orig": str,
    "type_orig": str,
    "destination": str,
    "country_dest": str,
    "municipality_dest": str,
    "type_dest": str,
}

temporal_categories_schema = {
    "year": int,
    "month": int,
    "day": int,
    "hour": int,
    "minute": int,
    "second": int,
    "weekday": int,
}

weather_schema = {
    "winddirection_10m": float,
    "windgusts_10m": float,
    "windspeed_10m": float,
    "windspeed_100m": float,
    "winddirection_100m": float,
    "cloudcover_high": float,
    "cloudcover_mid": float,
    "cloudcover_low": float,
    "apparent_temperature": float,
    "precipitation": float,
    "snowfall": float
}

adsb_schema = {
    "StateVectorsData4.time": datetime,
    "icao24": str,
    "lat": float,
    "lon": float,
    "velocity": float,
    "heading": int,
    "vertrate": float,
    "callsign": str,
    "baroaltitude": float,
    "geoaltitude": float,
    # "hour": datetime,
    "FlightsData4.estdepartureairport": str,
    "FlightsData4.estarrivalairport": str,
}