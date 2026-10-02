"""Stage 5a - train_fast: Logistic Regression (the quick, easy-to-explain model, ~1 minute).

Pipeline = fill missing values -> scale numbers -> logistic regression.
We try a few values of C (regularisation strength) with time-ordered cross-validation.
"""
import time

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (ALL_FEATURES, FAST_METRICS, FAST_MODEL, TARGET, TRAIN, load_params,
                    read_parquet, save_json, sort_by_time)


def train_model(train, c_grid, cv_folds, class_weight):
    train = sort_by_time(train)                 # oldest -> newest, required by TimeSeriesSplit
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])
    search = GridSearchCV(pipe, {"clf__C": c_grid}, cv=TimeSeriesSplit(n_splits=cv_folds),
                          scoring="roc_auc", n_jobs=1)
    search.fit(train[ALL_FEATURES], train[TARGET])
    return search


def main():
    p = load_params()["fast_model"]
    train = read_parquet(TRAIN)
    start = time.perf_counter()
    search = train_model(train, p["c_grid"], p["cv_folds"], p["class_weight"])
    seconds = time.perf_counter() - start

    FAST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": search.best_estimator_, "features": ALL_FEATURES}, FAST_MODEL)
    save_json({"train_seconds": round(seconds, 1), "cv_score": search.best_score_,
               "best_params": search.best_params_}, FAST_METRICS)
    print(f"fast model trained in {seconds:.1f}s  cv roc_auc={search.best_score_:.4f}  best={search.best_params_}")


if __name__ == "__main__":
    main()
