"""Stage 3 (Feast version) - build_training_set.

Same job as build_training_set.py, but the features come from the Feast OFFLINE store using a
point-in-time join: for every (station, hour) row Feast returns the feature values valid AT that hour.
The output file is identical to the pandas version (checked by src/compare_datasets.py).
"""
from feast import FeatureStore

from common import ALL_FEATURES, DATASET, ENTITY, FEATURE_REPO, KEYS, TARGET, read_parquet, save_parquet, sort_by_time

SERVICE_NAME = "alert_model_v1"


def main():
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    entity_df = read_parquet(ENTITY)              # station, event_timestamp, polluted_in_24h
    ds = store.get_historical_features(
        entity_df=entity_df, features=store.get_feature_service(SERVICE_NAME)).to_df()

    missing = [c for c in ALL_FEATURES if c not in ds.columns]
    if missing:
        raise SystemExit(f"Feast did not return these features: {missing}. Columns: {list(ds.columns)}")
    ds = ds[KEYS + [TARGET] + ALL_FEATURES].copy()
    ds[TARGET] = ds[TARGET].astype("int64")
    ds = sort_by_time(ds)
    save_parquet(ds, DATASET)
    print("dataset shape:", ds.shape)
    print(ds.dtypes.to_string())


if __name__ == "__main__":
    main()
