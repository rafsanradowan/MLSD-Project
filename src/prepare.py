"""Stage 1 - prepare.

Reads the 12 raw station CSVs and writes ONE clean hourly table with the target.

Important design points (each one prevents a real bug):
  * every station gets a COMPLETE hourly timeline, so "24 rows later" always means "24 hours later";
  * the target is built from the REAL (unfilled) PM2.5 readings;
  * rows are NOT dropped here - rolling windows in the next stage need an unbroken timeline.
"""
import pandas as pd

from common import CLEAN, RAW_DIR, TARGET, VALUE_COLUMNS, load_params, save_parquet

RENAME = {"PM2.5": "pm25", "PM10": "pm10", "SO2": "so2", "NO2": "no2", "CO": "co", "O3": "o3",
          "TEMP": "temp", "PRES": "pres", "DEWP": "dewp", "RAIN": "rain", "WSPM": "wspm"}
NEEDED_COLUMNS = {"year", "month", "day", "hour", "station", *RENAME}


def read_raw_files(raw_dir):
    """Read every CSV under data/raw that looks like a station file; skip the others."""
    frames = []
    for path in sorted(raw_dir.rglob("*.csv")):
        df = pd.read_csv(path)
        if NEEDED_COLUMNS.issubset(df.columns):
            frames.append(df)
        else:
            print(f"Skipping {path.name} (not a station file)")
    if not frames:
        raise SystemExit("No station CSV files found in data/raw. Check Step 2 of the guide.")
    return pd.concat(frames, ignore_index=True)


def make_clean_table(raw, horizon_hours, threshold, max_ffill_hours):
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.rename(columns=RENAME)
    for col in VALUE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["station", "timestamp"] + VALUE_COLUMNS]
    df = df.drop_duplicates(subset=["station", "timestamp"])

    # 1) complete hourly timeline per station (does nothing if the data is already complete)
    parts = []
    for name, g in df.groupby("station", sort=True):
        g = g.set_index("timestamp").sort_index()
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_range)
        g["station"] = name
        g.index.name = "timestamp"
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)

    # 2) target from the REAL pm25 values: will pm25 be >= threshold `horizon_hours` later?
    future_pm25 = df.groupby("station")["pm25"].shift(-horizon_hours)
    target = (future_pm25 >= threshold).astype("float64")
    df[TARGET] = target.where(future_pm25.notna())          # NaN when the future value is unknown

    # 3) only now fill short gaps in the input columns (never used to build the target)
    df[VALUE_COLUMNS] = df.groupby("station")[VALUE_COLUMNS].ffill(limit=max_ffill_hours)

    return df[["station", "timestamp"] + VALUE_COLUMNS + [TARGET]]


def main():
    p = load_params()["data"]
    raw = read_raw_files(RAW_DIR)
    clean = make_clean_table(raw, p["horizon_hours"], p["pm25_threshold"], p["max_ffill_hours"])
    save_parquet(clean, CLEAN)

    labelled = clean[TARGET].notna()
    print(f"rows: {len(clean)}   stations: {clean['station'].nunique()}")
    print(f"time range: {clean['timestamp'].min()}  ->  {clean['timestamp'].max()}")
    print(f"rows with a known target: {int(labelled.sum())}   alert rate: {clean.loc[labelled, TARGET].mean():.3f}")
    print(f"rows with missing pm25 after filling: {int(clean['pm25'].isna().sum())}")
    print(clean.groupby("station").size().to_string())


if __name__ == "__main__":
    main()
