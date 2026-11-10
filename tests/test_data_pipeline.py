"""Source authenticity, split isolation and reusable preprocessing boundaries."""

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from geographiclib.geodesic import Geodesic

from src.astronomy import add_astronomy
from src.data_pipeline import (
    clean_weather,
    match_weather,
    select_in_window,
    split,
    temporal_split,
)
from src.features import CATEGORICAL_INPUTS, NUMERIC_INPUTS, SCHEMA, build_pipeline
from src.validation import validate_frame


def weather_row(provider="NOAA_GHCNh", time="2024-01-01T12:00:00Z", number=1):
    record = {
        "provider": provider,
        "station_id": "TEST",
        "time_utc": time,
        "latitude": 51.5,
        "longitude": -0.1,
        "native_file_id": "N_TEST_2024",
        "native_row": number,
        "wspd_unit": "m/s",
        "prcp_period_minutes": 60,
        "rhum_measurement_code": "",
        "prcp_measurement_code": "",
        "prcp_report_type": "FM12",
        "dewpoint": 5,
        "dewpoint_flag": "1",
        "precip_raine": "",
        "precip_raine_flag": "",
        "precip_tipping": "",
        "precip_tipping_flag": "",
        "station_pres": 1000,
        "station_pres_source": "220",
        "station_pres_flag": "1",
    }
    for name, value in {
        "temp": 10,
        "rhum": 70,
        "prcp": 1,
        "wdir": 180,
        "wspd": 2,
        "pres": 1010,
    }.items():
        record[name] = value
        record[name + "_source"] = "220"
        record[name + "_flag"] = "1"
    if provider == "COSMOS_UK":
        for name in ["temp", "rhum", "prcp", "wdir", "wspd", "pres"]:
            record[name + "_flag"] = ""
    return record


def load_weather(rows, tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return clean_weather(path, SCHEMA)[0]


def model_frame():
    frame = pd.DataFrame(
        {
            "collision_datetime_utc": pd.to_datetime(
                ["2024-01-01T12:00Z", "2024-01-02T12:00Z", "2024-01-03T12:00Z"]
            )
        }
    )
    for column in NUMERIC_INPUTS:
        frame[column] = [1.0, 2.0, 3.0]
    frame["latitude"] = 51.5
    frame["longitude"] = -0.1
    frame["speed_limit"] = 30.0
    frame["number_of_vehicles"] = 2.0
    frame["pres"] = 1010.0
    frame["rhum"] = 70.0
    frame["wdir"] = 180.0
    for column in CATEGORICAL_INPUTS:
        frame[column] = pd.Series(
            [next(iter(SCHEMA["categorical_features"][column]["categories"]))] * 3,
            dtype=object,
        )
    return frame


def test_noaa_humidity_derived_requires_measured_input_qc(tmp_path):
    good = weather_row()
    good.update(rhum_flag="", rhum_measurement_code="D")
    bad = weather_row(number=2)
    bad.update(rhum_flag="", rhum_measurement_code="D", dewpoint_flag="2")
    weather = load_weather([good, bad], tmp_path)
    assert weather.loc[0, "rhum"] == 70 and weather.loc[0, "rhum_qc_verified"]
    assert pd.isna(weather.loc[1, "rhum"])
    assert weather.loc[0, "wspd"] == 7.2


def test_forecast_is_not_accepted_as_meteostat_observation(tmp_path):
    row = weather_row(provider="Meteostat")
    row.update(temp_source="metno_forecast", rhum_source="metar", wspd_unit="km/h")
    weather = load_weather([row], tmp_path)
    assert pd.isna(weather.temp.iloc[0])
    assert weather.rhum.iloc[0] == 70
    assert not weather.rhum_qc_verified.iloc[0]


def test_noaa_running_metar_rain_is_not_hourly_synop(tmp_path):
    row = weather_row()
    row["prcp_report_type"] = "FM15"
    weather = load_weather([row], tmp_path)
    assert pd.isna(weather.prcp.iloc[0])


def test_cosmos_rain_needs_adjacent_unflagged_same_gauge(tmp_path):
    rows = [
        weather_row("COSMOS_UK", f"2024-01-01T{hour}:00Z", i + 1)
        for i, hour in enumerate(["12:00", "12:30", "14:00"])
    ]
    rows[0]["prcp"] = 1
    rows[1]["prcp"] = 2
    rows[2]["prcp"] = 3
    weather = load_weather(rows, tmp_path)
    assert pd.isna(weather.prcp.iloc[0])
    assert weather.prcp.iloc[1] == 3 and weather.prcp_derived.iloc[1]
    assert pd.isna(weather.prcp.iloc[2])
    assert weather.pres.isna().all()


def test_quality_precedes_nearest_time_and_boundary_is_inclusive():
    minute = 60_000_000_000
    times = np.array([-60, 0, 60], dtype=np.int64) * minute
    selected = select_in_window(
        times,
        np.array([0], dtype=np.int64),
        np.array([6, 5, 6]),
        np.array([6, 5, 6]),
        np.zeros(3),
        60 * minute,
    )
    assert selected.tolist() == [0]
    selected = select_in_window(
        times,
        np.array([121 * minute], dtype=np.int64),
        np.ones(3),
        np.ones(3),
        np.zeros(3),
        60 * minute,
    )
    assert selected.tolist() == [-1]


def test_wgs84_distance_40km_and_plus_60min_limits(tmp_path):
    weather = load_weather([weather_row()], tmp_path)
    locations = [
        Geodesic.WGS84.Direct(51.5, -0.1, 0, distance)
        for distance in [39999.99, 40000.01]
    ]
    accidents = pd.DataFrame(
        {
            "latitude": [p["lat2"] for p in locations],
            "longitude": [p["lon2"] for p in locations],
            "collision_datetime_utc": pd.to_datetime(["2024-01-01T11:00Z"] * 2),
            "is_severe": [0.0, 1.0],
            "time_issue": ["", ""],
        }
    )
    result, report = match_weather(accidents, weather, SCHEMA, progress=False)
    assert report["any_observed_weather"] == 1
    assert result.loc[0, "weather_time_offset_minutes"] == 60
    assert pd.isna(result.loc[1, "weather_id"])


def test_temporal_splits_are_reproducible_with_a_purged_boundary():
    frame = pd.DataFrame(
        {
            "collision_datetime_utc": pd.date_range(
                "2024-01-01", periods=1600, freq="2h", tz="UTC"
            )
        }
    )
    first, boundaries = temporal_split(frame, SCHEMA["split"])
    second, _ = temporal_split(frame, SCHEMA["split"])
    assert first.equals(second) and first.eq("purged").any()
    end = frame.loc[first.eq("train"), "collision_datetime_utc"].max()
    start = frame.loc[first.eq("val"), "collision_datetime_utc"].min()
    assert start - end > pd.Timedelta(hours=2)
    assert len(boundaries) == 2


def test_preprocessing_uses_only_train_statistics_and_discards_targets():
    train = model_frame()
    train["temp"] = [1.0, 3.0, np.nan]
    train["is_severe"] = [0, 1, 0]
    train["enhanced_severity_collision"] = [1, 2, 3]
    pipeline = build_pipeline()
    matrix = pipeline.fit_transform(train)
    imputer = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    position = NUMERIC_INPUTS.index("temp")
    assert imputer.statistics_[position] == 2
    validation = model_frame()
    validation["temp"] = 1000.0
    validation["road_type"] = "unseen"
    assert pipeline.transform(validation).shape[1] == matrix.shape[1]
    assert imputer.statistics_[position] == 2
    altered = train.copy()
    altered["is_severe"] = 1
    altered["enhanced_severity_collision"] = 99
    actual = pipeline.transform(altered)
    actual = actual.toarray() if hasattr(actual, "toarray") else np.asarray(actual)
    expected = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    assert np.allclose(actual, expected)
    assert not any(
        "severity" in n or "is_severe" in n for n in pipeline.get_feature_names_out()
    )


def test_pandera_rejects_invalid_coordinates():
    frame = model_frame()
    frame["collision_index"] = ["A", "B", "C"]
    frame["is_severe"] = 0.0
    frame.loc[0, "latitude"] = 100.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_frame(frame, modeling=True)


def test_astronomy_uses_collision_coordinates_and_leaves_absent_twilight():
    frame = pd.DataFrame(
        {
            "latitude": [51.5, 60.0],
            "longitude": [-0.1, -1.0],
            "collision_datetime_utc": pd.to_datetime(["2024-06-21T12:00Z"] * 2),
        }
    )
    calculated = add_astronomy(frame)
    assert calculated.loc[0, "sun_elevation_deg"] > 50
    assert (
        calculated.loc[0, "civil_dawn_utc"]
        < calculated.loc[0, "sunrise_utc"]
        < calculated.loc[0, "sunset_utc"]
    )
    assert calculated.loc[1, "astronomical_twilight_absent"]
    assert pd.isna(calculated.loc[1, "astronomical_dawn_utc"])


def test_csv_roundtrip_preserves_numeric_looking_collision_ids(tmp_path):
    frame = pd.concat([model_frame()] * 170, ignore_index=True).iloc[:500].copy()
    frame["collision_index"] = [str(2024123456000 + i) for i in range(len(frame))]
    frame["collision_datetime_utc"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="6h", tz="UTC"
    )
    frame["is_severe"] = 0.0
    frame["weather_quality_tier"] = "qc_core6"
    frame["exclusion_reason"] = ""
    frame["time_issue"] = ""
    frame["collision_year"] = 2024
    frame["weather_id"] = [f"weather_{i}" for i in range(len(frame))]
    frame["weather_provider"] = "NOAA_GHCNh"
    frame["weather_distance_km"] = 1.0
    frame["weather_time_offset_minutes"] = 0.0
    (tmp_path / "processed").mkdir()
    frame.to_csv(tmp_path / "processed/collisions_clean.csv", index=False)
    result = split(tmp_path)
    assert result["accident_ids_disjoint"]
    actual = pd.read_csv(tmp_path / "splits/train.csv", dtype={"collision_index": str})
    assert actual.collision_index.isin(frame.collision_index).all()
