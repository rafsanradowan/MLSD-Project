"""Stage 3 - build_training_set (plain pandas version).

Joins the answers (entity_df) with the two feature tables.
Step 4 of the guide replaces this stage with a Feast version that produces the SAME file.
"""
from common import (ALL_FEATURES, DATASET, ENTITY, KEYS, POLLUTION, TARGET, WEATHER,
                    read_parquet, save_parquet, sort_by_time)


def build_dataset(entity, pollution, weather):
    ds = entity.merge(pollution, on=KEYS, how="left").merge(weather, on=KEYS, how="left")
    ds = ds[KEYS + [TARGET] + ALL_FEATURES]
    return sort_by_time(ds)


def main():
    ds = build_dataset(read_parquet(ENTITY), read_parquet(POLLUTION), read_parquet(WEATHER))
    save_parquet(ds, DATASET)
    print("dataset shape:", ds.shape)
    print(ds.dtypes.to_string())


if __name__ == "__main__":
    main()
