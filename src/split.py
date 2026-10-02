"""Stage 4 - split.

Time-based split (never random!):
  train = hours BEFORE (cutoff - horizon)   <- the gap stops training answers from "seeing" the test period
  test  = hours FROM the cutoff on
"""
import pandas as pd

from common import DATASET, TARGET, TEST, TRAIN, load_params, read_parquet, save_parquet, sort_by_time


def split_by_time(ds, cutoff_date, horizon_hours):
    cutoff = pd.Timestamp(str(cutoff_date), tz="UTC")
    gap = pd.Timedelta(hours=horizon_hours)
    ds = sort_by_time(ds)
    train = ds[ds["event_timestamp"] < cutoff - gap].reset_index(drop=True)
    test = ds[ds["event_timestamp"] >= cutoff].reset_index(drop=True)
    return train, test


def main():
    params = load_params()
    train, test = split_by_time(read_parquet(DATASET), params["split"]["cutoff_date"],
                                params["data"]["horizon_hours"])
    if train.empty or test.empty:
        raise SystemExit("Train or test is empty - check split.cutoff_date in params.yaml")
    save_parquet(train, TRAIN)
    save_parquet(test, TEST)
    print(f"train rows: {len(train)}   alert rate: {train[TARGET].mean():.3f}")
    print(f"test rows:  {len(test)}   alert rate: {test[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
