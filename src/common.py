"""Shared paths, column lists and small helpers used by every stage.

Every stage script does `from common import ...`, so keep this file small and simple.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# ---------- paths (these names are the "contract" between stages) ----------
RAW_DIR = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "interim" / "clean.parquet"

FEATURE_REPO = ROOT / "feature_repo"
POLLUTION = FEATURE_REPO / "data" / "station_pollution.parquet"
WEATHER = FEATURE_REPO / "data" / "station_weather.parquet"

ENTITY = ROOT / "data" / "processed" / "entity_df.parquet"
DATASET = ROOT / "data" / "processed" / "dataset.parquet"
TRAIN = ROOT / "data" / "processed" / "train.parquet"
TEST = ROOT / "data" / "processed" / "test.parquet"

FAST_MODEL = ROOT / "models" / "fast.joblib"
SLOW_MODEL = ROOT / "models" / "slow.joblib"
FAST_METRICS = ROOT / "metrics" / "train_fast.json"
SLOW_METRICS = ROOT / "metrics" / "train_slow.json"
EVALUATION = ROOT / "reports" / "evaluation.json"

# ---------- columns ----------
VALUE_COLUMNS = ["pm25", "pm10", "so2", "no2", "co", "o3",
                 "temp", "pres", "dewp", "rain", "wspm"]
TARGET = "polluted_in_24h"
KEYS = ["station", "event_timestamp"]

POLLUTION_FEATURES = ["pm25", "pm10", "so2", "no2", "co", "o3",
                      "pm25_mean_short", "pm25_mean_long", "pm25_max_long", "pm25_change_3h"]
WEATHER_FEATURES = ["temp", "pres", "dewp", "rain", "wspm",
                    "wspm_mean_short", "wspm_mean_long", "rain_sum_long", "pres_change_3h",
                    "hour_sin", "hour_cos", "month_sin", "month_cos"]
ALL_FEATURES = POLLUTION_FEATURES + WEATHER_FEATURES


# ---------- helpers ----------
def load_params():
    """Read params.yaml and return it as a dict."""
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_parquet(df, path):
    """Write a DataFrame to parquet (creating folders). Text columns are stored as plain strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(object)
    out.to_parquet(path, index=False)


def read_parquet(path, columns=None):
    return pd.read_parquet(path, columns=columns)


def to_jsonable(value):
    """Turn numpy numbers/arrays into plain Python so json can save them."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sort_by_time(df):
    """Time order matters: TimeSeriesSplit assumes rows go from oldest to newest."""
    return df.sort_values(["event_timestamp", "station"]).reset_index(drop=True)


def print_table(headers, rows):
    """Print a simple text table."""
    rows = [[str(c) for c in row] for row in rows]
    widths = [max([len(str(h))] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def fmt(value, digits=4):
    """Format a number for tables ('-' when missing)."""
    return "-" if value is None else f"{value:.{digits}f}"
