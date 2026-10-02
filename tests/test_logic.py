"""Small safety tests for the trickiest logic. Run with:  pytest -q"""
import numpy as np
import pandas as pd

from build_features import add_features
from build_training_set import build_dataset
from common import ALL_FEATURES, POLLUTION_FEATURES, TARGET, VALUE_COLUMNS, WEATHER_FEATURES
from prepare import make_clean_table
from split import split_by_time

RAW_NAMES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def toy_raw(pm25_by_station, hours=60, drop_rows=()):
    """Build a tiny table that looks like the raw CSVs."""
    frames = []
    for station, level in pm25_by_station.items():
        t = pd.date_range("2013-03-01", periods=hours, freq="h")
        df = pd.DataFrame({"year": t.year, "month": t.month, "day": t.day, "hour": t.hour})
        for name in RAW_NAMES:
            df[name] = 1.0
        df["PM2.5"] = float(level)
        df["station"] = station
        if station == "A":
            df = df.drop(index=list(drop_rows))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_target_is_shifted_inside_each_station():
    clean = make_clean_table(toy_raw({"A": 100, "B": 10}), horizon_hours=24, threshold=75, max_ffill_hours=3)
    for station, expected in (("A", 1.0), ("B", 0.0)):
        t = clean[clean["station"] == station][TARGET].to_numpy()
        assert (t[:36] == expected).all()          # answer known and correct
        assert np.isnan(t[36:]).all()              # last 24 hours: future unknown (never taken from the other station)


def test_missing_hours_are_restored_to_a_complete_timeline():
    clean = make_clean_table(toy_raw({"A": 100}, drop_rows=[10, 11, 12]), 24, 75, 3)
    assert len(clean) == 60
    assert (clean["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_features_never_use_future_data():
    rng = np.random.default_rng(0)
    t = pd.date_range("2013-03-01", periods=100, freq="h")
    frames = []
    for station in ("A", "B"):
        df = pd.DataFrame(rng.random((100, len(VALUE_COLUMNS))) * 100, columns=VALUE_COLUMNS)
        df["station"], df["timestamp"], df[TARGET] = station, t, 0.0
        frames.append(df)
    base = pd.concat(frames, ignore_index=True)
    changed = base.copy()
    future = changed["timestamp"] >= t[50]
    changed.loc[future, VALUE_COLUMNS] = 999.0            # rewrite the future
    a = add_features(base, 6, 24).set_index(["station", "timestamp"])
    b = add_features(changed, 6, 24).set_index(["station", "timestamp"])
    past = a.index.get_level_values("timestamp") < t[50]
    np.testing.assert_allclose(a.loc[past, ALL_FEATURES].to_numpy(dtype=float),
                               b.loc[past, ALL_FEATURES].to_numpy(dtype=float), equal_nan=True)


def test_split_keeps_a_gap_between_train_and_test():
    t = pd.date_range("2016-01-01", periods=24 * 20, freq="h", tz="UTC")
    ds = pd.DataFrame({"station": "A", "event_timestamp": t, TARGET: 0})
    train, test = split_by_time(ds, "2016-01-10", horizon_hours=24)
    cutoff = pd.Timestamp("2016-01-10", tz="UTC")
    assert train["event_timestamp"].max() + pd.Timedelta(hours=24) <= cutoff
    assert test["event_timestamp"].min() >= cutoff
    assert train["event_timestamp"].is_monotonic_increasing


def test_training_set_merge_keeps_every_row_and_all_features():
    t = pd.date_range("2016-01-01", periods=5, freq="h", tz="UTC")
    entity = pd.DataFrame({"station": "A", "event_timestamp": t, TARGET: [0, 1, 0, 1, 0]})
    pollution = entity[["station", "event_timestamp"]].assign(**{c: 1.0 for c in POLLUTION_FEATURES})
    weather = entity[["station", "event_timestamp"]].assign(**{c: 2.0 for c in WEATHER_FEATURES})
    ds = build_dataset(entity, pollution, weather)
    assert ds.shape == (5, 3 + len(ALL_FEATURES))
    assert not ds[ALL_FEATURES].isna().any().any()
