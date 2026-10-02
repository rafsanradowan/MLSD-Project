# airalert: 24-hour-ahead PM2.5 alert for Beijing (DVC + Feast project)

> Course project for DS-4491 Machine Learning Systems Design.
> **TODO before submitting:** replace every `<<...>>` placeholder below with your own text/results.

## 1. Problem and dataset
**Problem.** Given the current pollution and weather readings (plus recent history) at a monitoring station, predict whether PM2.5 will be **at least 75 µg/m³ 24 hours from now** (binary classification, with probabilities).

**Dataset.** UCI *Beijing Multi-Site Air Quality* (id 501): hourly data from 12 monitoring stations, 1 Mar 2013 – 28 Feb 2017 (35,064 hours × 12 stations = 420,768 rows). Licence CC BY 4.0.
Citation: Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G
The wind-direction column (`wd`) is not used (a text category); this is a known limitation, since wind direction affects pollution transport. <<add your own comment>>

## 2. ML models
- **Baselines:** majority class, and *persistence* ("polluted in 24 h if PM2.5 is already ≥ 75 now").
- **Fast model (~1 min):** Logistic Regression (imputer → scaler → regression), tuned over `C` with time-ordered cross-validation.
- **Slow model (~5–10 min):** LightGBM with a random hyper-parameter search (`RandomizedSearchCV`). The run time comes from the search budget, set in `params.yaml`.
- **Evaluation:** time-based split. Train before 2016-03-01 (minus a 24 h gap), test = the final year. Metrics: ROC-AUC, PR-AUC, log-loss, F1, accuracy.

## 3. Project structure
```
params.yaml  dvc.yaml  dvc.lock  requirements.txt
data/raw.dvc                 # raw CSVs tracked by DVC (not in Git)
src/                         # one script per pipeline stage + cli.py + common.py
feature_repo/                # Feast: feature_store.yaml, features.py
tests/                       # small safety tests
docs/VIVA_NOTES.md
```

## 4. DVC pipeline
<<paste the output of `dvc dag` here>>

| Stage | What it does | Main outputs |
|---|---|---|
| prepare | read 12 CSVs, complete hourly timeline, build target | `data/interim/clean.parquet` |
| build_features | rolling features, answer table | `feature_repo/data/*.parquet`, `entity_df.parquet` |
| feast_apply | register features in Feast, fill online store | `registry.db`, `online_store.db` |
| build_training_set | point-in-time join (Feast offline store) | `dataset.parquet` |
| split | time-based train/test split | `train.parquet`, `test.parquet` |
| train_fast | Logistic Regression | `models/fast.joblib` |
| train_slow | LightGBM + random search | `models/slow.joblib` |
| evaluate | baselines + both models on the test year | `reports/evaluation.json` |

**What reruns when something changes**

| Change | Stages that rerun |
|---|---|
| anything in `data/raw` | everything |
| `data.pm25_threshold`, `data.horizon_hours`, `data.max_ffill_hours` | `prepare` and everything after it |
| `features.windows_hours` | `build_features` and after |
| `split.cutoff_date` | `split`, both trainers, `evaluate` |
| `slow_model.*` | `train_slow`, `evaluate` |
| `fast_model.*` | `train_fast`, `evaluate` |
| code of one stage | that stage and what depends on it |

Remote storage: a local folder remote (absolute path), see "How to run".

## 5. Feast
Entity `station`; two feature views (`station_pollution`, `station_weather`); one feature service `alert_model_v1`.
- **Offline store** (parquet history) → used for training with a *point-in-time join*.
- **Online store** (SQLite, latest values) → used by `python src/cli.py predict`.
- `feast_apply` registers the definitions and calls `materialize` with explicit start/end dates (the data is old).
<<add 2-3 sentences in your own words on why a feature store helps>>

## 6. How to run
```
git clone <your-repo-url> && cd airalert
python -m venv .venv && <activate it>
pip install -r requirements.txt
dvc pull          # downloads the data/models from the DVC remote (needs the remote path to exist)
dvc repro         # rebuilds only what is out of date
python src/cli.py compare
python src/cli.py backtest --station Dongsi --hours 24
python src/cli.py predict --station Dongsi
```
To start from scratch: download the dataset from UCI, put the 12 station CSVs in `data/raw/`, run `dvc add data/raw`, then `dvc repro`.

## 7. Results
<<paste the table printed by `python src/cli.py compare`>>

<<paste a short `python src/cli.py backtest ...` output>>

**Limitations.** <<e.g. one city and four years of data (2013–2017), wind direction unused, one threshold (75 µg/m³), models not retrained over time>>
