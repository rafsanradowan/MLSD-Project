"""Feast stage: register the feature definitions, then load the latest values into the online store.

  apply        = write the definitions into Feast's registry
  materialize  = copy feature values from the offline files (history) into the online store (latest)
We pass the start and end explicitly, because this dataset is old (2013-2017).
"""
import sys
from datetime import timedelta

from feast import FeatureStore

from common import FEATURE_REPO, POLLUTION, read_parquet

sys.path.insert(0, str(FEATURE_REPO))
from features import alert_model_v1, station, station_pollution, station_weather  # noqa: E402


def main():
    (FEATURE_REPO / "data").mkdir(parents=True, exist_ok=True)
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    store.apply([station, station_pollution, station_weather, alert_model_v1])

    times = read_parquet(POLLUTION, columns=["event_timestamp"])["event_timestamp"]
    start = times.min().to_pydatetime()
    end = times.max().to_pydatetime() + timedelta(hours=1)
    store.materialize(start_date=start, end_date=end)
    print(f"Feast applied and materialized {start} -> {end}")


if __name__ == "__main__":
    main()
