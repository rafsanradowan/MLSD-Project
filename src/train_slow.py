"""Stage 5b - train_slow: LightGBM with a random hyper-parameter search (the 5-10 minute model).

LightGBM = many small decision trees built one after another (gradient boosting).
The run time comes from the search: n_iter candidate settings x cv_folds validation folds.
Change slow_model.n_iter / cv_folds / max_estimators in params.yaml to make it faster or slower.
"""
import time

import joblib
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from common import (ALL_FEATURES, SLOW_METRICS, SLOW_MODEL, TARGET, TRAIN, load_params,
                    read_parquet, save_json, sort_by_time)


def search_space(max_estimators):
    return {
        "num_leaves": randint(15, 128),
        "max_depth": randint(3, 13),
        "learning_rate": loguniform(0.02, 0.2),
        "n_estimators": randint(100, max(101, max_estimators + 1)),
        "min_child_samples": randint(20, 200),
        "subsample": uniform(0.6, 0.4),          # 0.6 .. 1.0
        "colsample_bytree": uniform(0.6, 0.4),   # 0.6 .. 1.0
        "reg_alpha": loguniform(1e-3, 10),
        "reg_lambda": loguniform(1e-3, 10),
    }


def train_model(train, n_iter, cv_folds, max_estimators, seed):
    train = sort_by_time(train)                 # oldest -> newest, required by TimeSeriesSplit
    base = LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1,
                          deterministic=True, force_row_wise=True)
    search = RandomizedSearchCV(base, search_space(max_estimators), n_iter=n_iter,
                                cv=TimeSeriesSplit(n_splits=cv_folds), scoring="roc_auc",
                                random_state=seed, n_jobs=1)
    search.fit(train[ALL_FEATURES], train[TARGET])    # NaN values are fine for LightGBM
    return search


def main():
    params = load_params()
    p = params["slow_model"]
    train = read_parquet(TRAIN)
    start = time.perf_counter()
    search = train_model(train, p["n_iter"], p["cv_folds"], p["max_estimators"], params["seed"])
    seconds = time.perf_counter() - start

    SLOW_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": search.best_estimator_, "features": ALL_FEATURES}, SLOW_MODEL)
    save_json({"train_seconds": round(seconds, 1), "cv_score": search.best_score_,
               "best_params": search.best_params_}, SLOW_METRICS)
    print(f"slow model trained in {seconds:.1f}s ({seconds / 60:.1f} min)  cv roc_auc={search.best_score_:.4f}")


if __name__ == "__main__":
    main()
