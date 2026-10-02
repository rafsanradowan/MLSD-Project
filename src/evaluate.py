"""Stage 6 - evaluate.

Scores two simple baselines and both models on the TEST set (the last year, never seen in training).
  majority    - always predicts the most common answer seen in training
  persistence - predicts "polluted in 24h" if PM2.5 is >= the threshold right now
"""
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, log_loss,
                             roc_auc_score)

from common import (EVALUATION, FAST_MODEL, SLOW_MODEL, TARGET, TEST, TRAIN, fmt, load_params,
                    print_table, read_parquet, save_json)


def model_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
            "log_loss": log_loss(y_true, proba, labels=[0, 1]),
            "f1": f1_score(y_true, pred, zero_division=0),
            "accuracy": accuracy_score(y_true, pred)}


def baseline_metrics(train, test, pm25_threshold):
    y = test[TARGET]
    majority = int(train[TARGET].mean() >= 0.5)
    majority_pred = np.full(len(test), majority)
    persistence_pred = (test["pm25"] >= pm25_threshold).astype(int)
    return {
        "majority": {"f1": f1_score(y, majority_pred, zero_division=0),
                     "accuracy": accuracy_score(y, majority_pred)},
        "persistence": {"roc_auc": roc_auc_score(y, test["pm25"]),     # higher pm25 = higher score
                        "pr_auc": average_precision_score(y, test["pm25"]),
                        "f1": f1_score(y, persistence_pred, zero_division=0),
                        "accuracy": accuracy_score(y, persistence_pred)},
    }


def main():
    params = load_params()
    train, test = read_parquet(TRAIN), read_parquet(TEST)
    y = test[TARGET]

    results = baseline_metrics(train, test, params["data"]["pm25_threshold"])
    for name, path in (("fast", FAST_MODEL), ("slow", SLOW_MODEL)):
        if not path.exists():
            print(f"({name} model not found - skipped)")
            continue
        bundle = joblib.load(path)
        proba = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
        results[name] = model_metrics(y, proba, params["evaluate"]["threshold"])

    save_json(results, EVALUATION)
    columns = ["roc_auc", "pr_auc", "log_loss", "f1", "accuracy"]
    print_table(["model"] + columns,
                [[name] + [fmt(m.get(c)) for c in columns] for name, m in results.items()])


if __name__ == "__main__":
    main()
