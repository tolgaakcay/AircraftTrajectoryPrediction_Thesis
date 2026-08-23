from traffic.data import opensky
df = opensky.history(
    "2024-08-01T00:00:00Z",
    "2024-08-02T00:00:00Z",
    arrival_airport="LFBO",
)
print(type(df), None if df is None else len(df.data))