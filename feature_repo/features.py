"""Feast feature definitions (the "menu" of features).

  Entity         = what a feature belongs to (here: a monitoring station)
  FeatureView    = one table of features with a timestamp column
  FeatureService = a named bundle of feature views; training AND serving both ask for this bundle,
                   so they can never use different features.
"""
import sys
from datetime import timedelta
from pathlib import Path

import yaml
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import POLLUTION_FEATURES, WEATHER_FEATURES  # noqa: E402

with open(ROOT / "params.yaml", encoding="utf-8") as f:
    TTL = timedelta(days=yaml.safe_load(f)["feast"]["ttl_days"])

station = Entity(name="station", join_keys=["station"])

pollution_source = FileSource(path="data/station_pollution.parquet",
                              timestamp_field="event_timestamp")
weather_source = FileSource(path="data/station_weather.parquet",
                            timestamp_field="event_timestamp")

station_pollution = FeatureView(
    name="station_pollution", entities=[station], ttl=TTL,
    schema=[Field(name=n, dtype=Float64) for n in POLLUTION_FEATURES],
    source=pollution_source)

station_weather = FeatureView(
    name="station_weather", entities=[station], ttl=TTL,
    schema=[Field(name=n, dtype=Float64) for n in WEATHER_FEATURES],
    source=weather_source)

alert_model_v1 = FeatureService(name="alert_model_v1",
                                features=[station_pollution, station_weather])
