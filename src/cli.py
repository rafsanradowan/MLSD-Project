"""Terminal commands for the project (no UI).

  python src/cli.py info
  python src/cli.py compare
  python src/cli.py backtest --station Dongsi --hours 24
  python src/cli.py predict --station Dongsi          (needs Feast set up: Step 4 of the guide)
  python src/cli.py feast-demo --station Dongsi       (needs Feast set up)
"""
import argparse

import joblib
import pandas as pd

from common import (ALL_FEATURES, CLEAN, EVALUATION, FAST_METRICS, FAST_MODEL, FEATURE_REPO,
                    POLLUTION, SLOW_METRICS, SLOW_MODEL, TARGET, TEST, fmt, load_json,
                    print_table, read_parquet)

SERVICE_NAME = "alert_model_v1"


def load_models():
    models = {}
    for name, path in (("fast", FAST_MODEL), ("slow", SLOW_MODEL)):
        if path.exists():
            models[name] = joblib.load(path)
    if not models:
        raise SystemExit("No trained models found. Run `dvc repro` first.")
    return models


def check_station(station, available):
    if station not in available:
        raise SystemExit(f"Unknown station '{station}'. Choose one of: {', '.join(sorted(available))}")


def get_feature_store():
    try:
        from feast import FeatureStore
    except ImportError:
        raise SystemExit("Feast is not installed. Do Step 4 of the guide first (pip install feast).")
    return FeatureStore(repo_path=str(FEATURE_REPO))


# ---------------------------------------------------------------- commands
def cmd_info(args):
    df = read_parquet(CLEAN)
    known = df[TARGET].notna()
    print(f"stations ({df['station'].nunique()}): {', '.join(sorted(df['station'].unique()))}")
    print(f"time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"rows: {len(df)}   rows with a known answer: {int(known.sum())}")
    print(f"share of 'polluted in 24h' (alert rate): {df.loc[known, TARGET].mean():.3f}")


def cmd_compare(args):
    evaluation = load_json(EVALUATION)
    train_seconds = {}
    for name, path in (("fast", FAST_METRICS), ("slow", SLOW_METRICS)):
        if path.exists():
            train_seconds[name] = load_json(path).get("train_seconds")
    columns = ["roc_auc", "pr_auc", "log_loss", "f1", "accuracy"]
    rows = [[name] + [fmt(m.get(c)) for c in columns] + [fmt(train_seconds.get(name), 1)]
            for name, m in evaluation.items()]
    print_table(["model"] + columns + ["train_seconds"], rows)


def cmd_backtest(args):
    test = read_parquet(TEST)
    check_station(args.station, set(test["station"]))
    sub = test[test["station"] == args.station].sort_values("event_timestamp").tail(args.hours)
    models = load_models()
    probs = {n: b["model"].predict_proba(sub[b["features"]])[:, 1] for n, b in models.items()}
    rows = []
    for i, (_, r) in enumerate(sub.iterrows()):
        rows.append([str(r["event_timestamp"])[:16], fmt(r["pm25"], 1)]
                    + [fmt(probs[n][i], 3) for n in models] + [int(r[TARGET])])
    print(f"Last {len(sub)} test hours for {args.station} (probability that PM2.5 will be high 24h later)")
    print_table(["hour", "pm25_now"] + [f"p_{n}" for n in models] + ["actual"], rows)


def cmd_predict(args):
    store = get_feature_store()
    models = load_models()
    response = store.get_online_features(
        features=store.get_feature_service(SERVICE_NAME),
        entity_rows=[{"station": args.station}]).to_dict()
    row = {k: v[0] for k, v in response.items()}
    if row.get("pm25") is None:
        known = sorted(read_parquet(CLEAN, columns=["station"])["station"].unique())
        raise SystemExit(f"No online features for '{args.station}'. Stations: {', '.join(known)}")
    print(f"Latest stored readings for {args.station}: pm25 = {row['pm25']:.1f}")
    for name, bundle in models.items():
        x = pd.DataFrame([row])[bundle["features"]].astype(float)
        p = bundle["model"].predict_proba(x)[0, 1]
        print(f"  {name} model: probability of a PM2.5 alert 24 hours after that hour = {p:.3f}")


def cmd_feast_demo(args):
    store = get_feature_store()
    service = store.get_feature_service(SERVICE_NAME)
    times = read_parquet(POLLUTION, columns=["station", "event_timestamp"])
    check_station(args.station, set(times["station"]))
    last = times.loc[times["station"] == args.station, "event_timestamp"].max()
    entity_df = pd.DataFrame({"station": [args.station], "event_timestamp": [last]})
    offline = store.get_historical_features(entity_df=entity_df, features=service).to_df().iloc[0]
    online = store.get_online_features(features=service,
                                       entity_rows=[{"station": args.station}]).to_dict()
    print(f"Offline store (history, used for training) vs online store (latest values, used for serving)")
    print(f"Offline lookup 'as of' {last}")
    rows = [[f, fmt(offline.get(f), 3), fmt(online[f][0], 3)] for f in ALL_FEATURES]
    print_table(["feature", "offline", "online"], rows)


def main():
    parser = argparse.ArgumentParser(description="airalert terminal commands")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="dataset summary").set_defaults(func=cmd_info)
    sub.add_parser("compare", help="baselines vs models").set_defaults(func=cmd_compare)
    for name, func, help_text in (("backtest", cmd_backtest, "predictions vs reality (test set)"),
                                  ("predict", cmd_predict, "latest prediction using the Feast online store"),
                                  ("feast-demo", cmd_feast_demo, "offline vs online features")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--station", required=True)
        if name == "backtest":
            p.add_argument("--hours", type=int, default=24)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
