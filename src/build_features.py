"""Stage 2 - build_features.

Turns the clean table into:
  * station_pollution.parquet and station_weather.parquet  (the "feature tables" Feast will serve)
  * entity_df.parquet   (which station/hour rows we train on + the answer)

A feature at hour t uses ONLY data up to and including hour t (no peeking into the future).
"""
import numpy as np

from common import (CLEAN, ENTITY, POLLUTION, POLLUTION_FEATURES, TARGET, WEATHER,
                    WEATHER_FEATURES, load_params, read_parquet, save_parquet)


def add_features(df, short_window, long_window):
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    def rolling(col, window, how):
        # rolling window computed separately for each station, aligned back to df's rows
        r = df.groupby("station")[col].rolling(window, min_periods=1)
        return getattr(r, how)().reset_index(level=0, drop=True)

    out = df.copy()
    out["pm25_mean_short"] = rolling("pm25", short_window, "mean")
    out["pm25_mean_long"] = rolling("pm25", long_window, "mean")
    out["pm25_max_long"] = rolling("pm25", long_window, "max")
    out["pm25_change_3h"] = df["pm25"] - df.groupby("station")["pm25"].shift(3)
    out["wspm_mean_short"] = rolling("wspm", short_window, "mean")
    out["wspm_mean_long"] = rolling("wspm", long_window, "mean")
    out["rain_sum_long"] = rolling("rain", long_window, "sum")
    out["pres_change_3h"] = df["pres"] - df.groupby("station")["pres"].shift(3)

    hour = df["timestamp"].dt.hour
    month = df["timestamp"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Feast needs a timezone-aware timestamp column called event_timestamp.
    # The data is Beijing local time; we just label it UTC so every file uses the same convention.
    out["event_timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    return out


def main():
    short_window, long_window = load_params()["features"]["windows_hours"]
    out = add_features(read_parquet(CLEAN), short_window, long_window)

    pollution = out[["station", "event_timestamp"] + POLLUTION_FEATURES].astype(
        {c: "float64" for c in POLLUTION_FEATURES})
    weather = out[["station", "event_timestamp"] + WEATHER_FEATURES].astype(
        {c: "float64" for c in WEATHER_FEATURES})

    # training rows: we must know the answer, and the current pm25 reading must exist
    usable = out[TARGET].notna() & out["pm25"].notna()
    entity = out.loc[usable, ["station", "event_timestamp", TARGET]].copy()
    entity[TARGET] = entity[TARGET].astype("int64")
    entity = entity.sort_values(["event_timestamp", "station"]).reset_index(drop=True)

    save_parquet(pollution, POLLUTION)
    save_parquet(weather, WEATHER)
    save_parquet(entity, ENTITY)
    print(f"feature rows: {len(pollution)}   training rows (entity_df): {len(entity)}")


if __name__ == "__main__":
    main()
