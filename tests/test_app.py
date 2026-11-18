"""D 负责:预测接口的契约、安全底线与模型接入测试。

安全底线在这里指接口层不能把问题输入悄悄分成 3 级(常规):NaN / 越界 / 非法
时间一律 422;优先级只由模型的概率与它自己的两条界线决定。模型本身的漏判率
由 B 的训练验收(``src.triage.acceptance``)负责,不在这里用占位模型冒充。

运行:
    python -m pytest tests/test_app.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.frontend import filled_records
from app.main import app
from app.model_loader import ModelService, build_frame, load_model
from app.schemas import PredictRequest
from src.features import INPUT_COLUMNS, SCHEMA, build_pipeline
from src.triage import TriageClassifier

SCENE = {
    "collision_datetime_utc": "2025-01-15T08:30:00+00:00",
    "latitude": 51.5,
    "longitude": -0.1,
}
HIGH_RISK = {
    **SCENE,
    "speed_limit": 70,
    "number_of_vehicles": 8,
    "temp": -10.0,
    "wspd": 40.0,
    "road_surface_conditions": "3",
}
MINOR = {
    **SCENE,
    "speed_limit": 20,
    "number_of_vehicles": 1,
    "temp": 18.0,
    "wspd": 5.0,
    "road_surface_conditions": "1",
}
VEHICLES = [
    {"vehicle_type": "9", "age_of_driver": 40},
    {"vehicle_type": 1, "sex_of_driver": "1", "age_of_driver": 20},
]
CASUALTIES = [{"casualty_class": "3", "age_of_casualty": 70}]
REGISTERED_METRICS = {
    "test_undertriage_life": 0.038,
    "test_undertriage_fatal": 0.003,
    "test_p3_share": 0.258,
    "test_p1_life_lift": 3.5,
}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "request_log.jsonl"
    monkeypatch.setenv("REQUEST_LOG_PATH", str(path))
    return path


@pytest.fixture
def client(log_path, monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def triage_classifier():
    """A real ``TriageClassifier`` (A's pipeline + an estimator + cut-offs), fitted small."""
    rows = [
        build_frame(
            PredictRequest(
                collision_datetime_utc=f"2025-01-{1 + i % 28:02d}T{i % 24:02d}:30:00Z",
                latitude=50 + i % 10,
                longitude=-3 + i % 5,
                speed_limit=[20, 30, 40, 50, 60, 70][i % 6],
                number_of_vehicles=1 + i % 4,
                temp=float(i % 20),
                road_surface_conditions=str(1 + i % 5),
                vehicles=[{"vehicle_type": "9", "age_of_driver": 20 + i}],
                casualties=[{"casualty_class": str(1 + i % 3), "age_of_casualty": i}],
            )
        )
        for i in range(60)
    ]
    frame = pd.concat(rows, ignore_index=True)
    target = (frame["speed_limit"] >= 50).astype(int)
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(frame, target)
    return TriageClassifier(pipeline, (0.3, 0.8))


def predict(client: TestClient, payload: dict) -> dict:
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 输入校验:问题输入必须被拒绝,而不是落成常规级
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        (
            '{"collision_datetime_utc":"2025-01-15T08:30:00Z","latitude":51.5,'
            '"longitude":-0.1,"speed_limit":NaN}'
        ),
        json.dumps({**SCENE, "temp": "nan"}),
        json.dumps({**SCENE, "wspd": "Infinity"}),
    ],
    ids=["nan-literal", "nan-string", "infinity"],
)
def test_non_finite_numbers_are_rejected(client, body):
    response = client.post(
        "/predict", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["type"] == "finite_number"


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"collision_datetime_utc": "garbage"}, "collision_datetime_utc"),
        ({"collision_datetime_utc": "2025-01-15T08:30:00"}, "collision_datetime_utc"),
        ({"collision_datetime_utc": None}, "collision_datetime_utc"),
        ({"latitude": None}, "latitude"),
        ({"latitude": 999}, "latitude"),
        ({"longitude": -40}, "longitude"),
        ({"speed_limit": 35}, "speed_limit"),
        ({"number_of_vehicles": 0}, "number_of_vehicles"),
        ({"rhum": 140}, "rhum"),
        ({"road_type": "99"}, "road_type"),
        ({"veh_vehicle_type_1": 1}, "veh_vehicle_type_1"),
        ({"vehicles": [{"age_of_driver": -5}]}, "age_of_driver"),
        ({"casualties": [{"casualty_severity": "1"}]}, "casualty_severity"),
    ],
)
def test_invalid_requests_are_rejected(client, change, field):
    response = client.post("/predict", json={**SCENE, **change})
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"][-1] == field


def test_time_and_place_are_required(client):
    for name in SCENE:
        payload = {key: value for key, value in SCENE.items() if key != name}
        assert client.post("/predict", json=payload).status_code == 422


def test_source_missing_codes_and_blanks_mean_unknown():
    request = PredictRequest(
        **SCENE, road_type="9", light_conditions="", junction_detail=0
    )
    assert request.road_type is None
    assert request.light_conditions is None
    assert request.junction_detail == "0"


# ---------------------------------------------------------------------------
# 请求 → 模型输入:与训练用的列完全一致
# ---------------------------------------------------------------------------
def test_request_contract_covers_every_schema_input():
    fields = set(PredictRequest.model_fields)
    collision_inputs = {
        SCHEMA["datetime_column"],
        *SCHEMA["numeric_features"],
        *SCHEMA["categorical_features"],
    }
    assert fields == collision_inputs | {"vehicles", "casualties"}


def test_build_frame_matches_training_columns_and_aggregates_participants():
    request = PredictRequest(**MINOR, vehicles=VEHICLES, casualties=CASUALTIES)
    frame = build_frame(request)
    assert list(frame.columns) == INPUT_COLUMNS
    assert len(frame) == 1
    row = frame.iloc[0]
    assert str(frame["collision_datetime_utc"].dt.tz) == "UTC"
    assert row["veh_vehicle_type_9"] == 1
    assert row["veh_vehicle_type_1"] == 1
    assert row["veh_age_of_driver_mean"] == 30
    assert row["veh_age_of_driver_max"] == 40
    # 没填的编码按 DfT 的 -1(未知)计数,与训练数据一致。
    assert row["veh_sex_of_driver_-1"] == 1
    assert row["veh_sex_of_driver_1"] == 1
    assert row["cas_casualty_class_3"] == 1
    assert row["cas_age_of_casualty_min"] == 70


def test_absent_participants_stay_missing_not_zero():
    frame = build_frame(PredictRequest(**MINOR, vehicles=[], casualties=None))
    participants = [c for c in INPUT_COLUMNS if c.startswith(("veh_", "cas_"))]
    assert len(participants) == 205
    assert frame[participants].isna().all(axis=None)


def test_missing_info_reports_absent_groups(client):
    assert predict(client, SCENE)["missing_info"] == [
        "weather",
        "road",
        "vehicle",
        "casualty",
    ]
    complete = {**MINOR, "vehicles": VEHICLES, "casualties": CASUALTIES}
    assert predict(client, complete)["missing_info"] == []


# ---------------------------------------------------------------------------
# 占位模型:明确标记,不冒充评估结果
# ---------------------------------------------------------------------------
def test_placeholder_model_is_labelled_and_not_evaluated(client):
    info = client.get("/model-info").json()
    assert info["model_version"] == "dummy-v1"
    assert info["is_placeholder"] is True
    assert info["risk_metrics"] == {
        "undertriage_life": None,
        "undertriage_fatal": None,
        "routine_share": None,
        "immediate_lift": None,
        "status": "not_evaluated",
    }
    limits = SCHEMA["triage"]["acceptance"]
    assert (
        info["acceptance_limits"]["max_undertriage_life"]
        == limits["max_undertriage_life"]
    )


def test_placeholder_ranks_danger_signals(client):
    assert predict(client, HIGH_RISK)["priority"] == 1
    assert predict(client, MINOR)["priority"] == 3


def test_priority_always_follows_probability_and_cutoffs(client):
    for speed_limit in SCHEMA["numeric_features"]["speed_limit"]["allowed"]:
        for vehicles in (1, 2, 4, 8):
            for temp in (-10.0, 8.0, 20.0):
                body = predict(
                    client,
                    {
                        **SCENE,
                        "speed_limit": speed_limit,
                        "number_of_vehicles": vehicles,
                        "temp": temp,
                    },
                )
                low = body["cutoffs"]["routine_urgent"]
                high = body["cutoffs"]["urgent_immediate"]
                expected = (
                    3 - (body["probability"] >= low) - (body["probability"] >= high)
                )
                assert body["priority"] == expected, body


# ---------------------------------------------------------------------------
# 真实模型:服务委托给 TriageClassifier,而不是自带一套分级逻辑
# ---------------------------------------------------------------------------
def test_service_delegates_to_the_loaded_classifier(client, triage_classifier):
    app.state.service = ModelService(triage_classifier, "road-severity-model v7")
    payload = {**HIGH_RISK, "vehicles": VEHICLES, "casualties": CASUALTIES}
    body = predict(client, payload)
    frame = build_frame(PredictRequest(**payload))
    assert body["model_version"] == "road-severity-model v7"
    assert body["cutoffs"] == {"routine_urgent": 0.3, "urgent_immediate": 0.8}
    assert body["priority"] == triage_classifier.predict(frame)[0]
    assert body["probability"] == pytest.approx(
        triage_classifier.predict_proba(frame)[0, 1]
    )
    assert client.get("/model-info").json()["is_placeholder"] is False


def test_partial_information_reaches_the_real_pipeline(client, triage_classifier):
    app.state.service = ModelService(triage_classifier, "road-severity-model v7")
    body = predict(client, SCENE)
    assert body["priority"] in (1, 2, 3)
    assert 0.0 <= body["probability"] <= 1.0


def test_non_finite_probability_is_an_error_not_routine(triage_classifier):
    class Broken:
        cutoffs = (0.3, 0.8)

        def predict_proba(self, X):
            return np.array([[np.nan, np.nan]])

    frame = build_frame(PredictRequest(**SCENE))
    with pytest.raises(ValueError, match="非有限"):
        ModelService(Broken(), "broken").predict(frame)


def test_load_model_from_mlflow_registry(tmp_path, triage_classifier):
    """Same save/registration calls as ``src.train``; metrics come from the final run."""
    import mlflow
    import skops.io

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment(
            experiment_id=mlflow.create_experiment(
                "api-test", artifact_location=(tmp_path / "artifacts").as_uri()
            )
        )
        with mlflow.start_run(run_name="final-lr"):
            mlflow.log_metrics(REGISTERED_METRICS)
            info = mlflow.sklearn.log_model(
                triage_classifier,
                name="model",
                skops_trusted_types=skops.io.get_untrusted_types(
                    data=skops.io.dumps(triage_classifier)
                ),
                registered_model_name="road-severity-model",
            )
        mlflow.MlflowClient().set_registered_model_alias(
            "road-severity-model", "production", info.registered_model_version
        )

        service = load_model("models:/road-severity-model@production")
    finally:
        mlflow.set_tracking_uri(previous)

    assert isinstance(service.model, TriageClassifier)
    assert service.is_placeholder is False
    assert service.model_version == "road-severity-model v1"
    assert tuple(service.model.cutoffs) == (0.3, 0.8)
    assert service.risk_metrics.status == "pass"
    assert service.risk_metrics.undertriage_life == pytest.approx(0.038)
    assert service.risk_metrics.routine_share == pytest.approx(0.258)
    frame = build_frame(PredictRequest(**HIGH_RISK))
    priority, probability = service.predict(frame)
    assert priority == triage_classifier.predict(frame)[0]
    assert probability == pytest.approx(triage_classifier.predict_proba(frame)[0, 1])


# ---------------------------------------------------------------------------
# 请求日志与其余契约
# ---------------------------------------------------------------------------
def test_request_log_supports_audit(client, log_path):
    payload = {**HIGH_RISK, "vehicles": VEHICLES}
    body = predict(client, payload)
    predict(client, MINOR)
    lines = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    record = lines[0]
    assert record["request_id"] == body["request_id"] != lines[1]["request_id"]
    assert record["model_version"] == "dummy-v1"
    assert record["priority"] == body["priority"]
    assert record["cutoffs"] == body["cutoffs"]
    assert record["inputs"]["speed_limit"] == 70
    assert record["inputs"]["vehicles"][0] == {
        "vehicle_type": "9",
        "age_of_driver": 40.0,
    }
    assert record["timestamp_utc"].endswith("+00:00")


def test_log_failure_does_not_lose_the_prediction(client, tmp_path, monkeypatch):
    blocked = tmp_path / "is-a-directory"
    blocked.mkdir()
    monkeypatch.setenv("REQUEST_LOG_PATH", str(blocked))
    assert predict(client, HIGH_RISK)["priority"] == 1


def test_predict_contract(client):
    body = predict(client, MINOR)
    assert set(body) == {
        "request_id",
        "priority",
        "probability",
        "cutoffs",
        "missing_info",
        "model_version",
        "latency_ms",
    }
    assert body["latency_ms"] >= 0


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_frontend_keeps_only_filled_participant_cells():
    table = pd.DataFrame(
        {
            "vehicle_type": ["9", None, None],
            "age_of_driver": [40.0, np.nan, 25.0],
            "sex_of_driver": [None, None, "1"],
        }
    )
    assert filled_records(table) == [
        {"vehicle_type": "9", "age_of_driver": 40.0},
        {"age_of_driver": 25.0, "sex_of_driver": "1"},
    ]
