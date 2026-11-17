"""API tests: every model call is mocked, so CI needs no registry and no network."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    InvalidInputError,
    ModelTimeoutError,
    ModelUnavailableError,
    UpstreamRateLimitError,
)
from app.main import app
from app.schemas.prediction import (
    FeatureContribution,
    PredictRequest,
    PredictResult,
    UsageStats,
)
from app.services.prediction_service import PredictionService, get_prediction_service

BASE_REQUEST = {
    "collision_datetime_utc": "2025-11-01T08:30:00Z",
    "latitude": 51.5072,
    "longitude": -0.1276,
    "speed_limit": 30,
    "number_of_vehicles": 2,
    "road_type": 6,
    "light_conditions": 1,
    "urban_or_rural_area": 1,
}


class StubExplanation:
    """Mirrors src.explain.SingleExplanation without loading shap or MLflow."""

    probability = 0.7123
    priority = 1
    base_value = -1.2044
    contributions = [
        FeatureContribution(feature="speed_limit", value="30", impact=0.4188),
        FeatureContribution(feature="light_conditions", value="1", impact=0.1063),
    ]
    other_impact = 0.0731
    cutoffs = (0.201, 0.664)


class FakeService:
    async def predict(self, request: PredictRequest) -> PredictResult:
        return PredictResult(
            probability=0.7123,
            priority=1,
            priority_label="立即",
            base_value=-1.2044,
            contributions=StubExplanation.contributions,
            other_impact=0.0731,
            cutoffs=[0.201, 0.664],
            model_uri="models:/road-severity-model@production",
            usage=UsageStats(
                input_fields=6,
                participant_rows=0,
                explained_features=2,
                elapsed_ms=4,
            ),
        )


class FailingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def predict(self, request: PredictRequest) -> PredictResult:
        raise self.error


@pytest.fixture
def client():
    app.dependency_overrides[get_prediction_service] = lambda: FakeService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _client_for(error: Exception) -> TestClient:
    app.dependency_overrides[get_prediction_service] = lambda: FailingService(error)
    return TestClient(app)


def test_health_still_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_contract(client):
    response = client.post("/api/v1/predict", json=BASE_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert body["priority"] in (1, 2, 3)
    assert body["priority_label"] == "立即"
    assert body["cutoffs"] == [0.201, 0.664]
    assert body["contributions"][0]["feature"] == "speed_limit"
    assert body["usage"]["explained_features"] == 2


def test_predict_sets_trace_header(client):
    response = client.post("/api/v1/predict", json=BASE_REQUEST)
    assert response.headers["X-Trace-ID"]
    assert (body_trace := response.json()["trace_id"])
    assert body_trace == response.headers["X-Trace-ID"]


def test_missing_required_field_is_400(client):
    payload = {key: value for key, value in BASE_REQUEST.items() if key != "latitude"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_out_of_range_latitude_is_400(client):
    payload = {**BASE_REQUEST, "latitude": 10.0}
    assert client.post("/api/v1/predict", json=payload).status_code == 400


def test_unknown_speed_limit_is_400(client):
    payload = {**BASE_REQUEST, "speed_limit": 55}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_unknown_field_is_400(client):
    payload = {**BASE_REQUEST, "collision_severity": 1}
    assert client.post("/api/v1/predict", json=payload).status_code == 400


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (InvalidInputError("bad row"), 400),
        (UpstreamRateLimitError("busy"), 429),
        (ModelUnavailableError("no model"), 503),
        (ModelTimeoutError("too slow"), 504),
    ],
)
def test_business_errors_map_to_status(error, status):
    try:
        response = _client_for(error).post("/api/v1/predict", json=BASE_REQUEST)
        assert response.status_code == status
        assert response.json()["detail"]["error"] == error.code
    finally:
        app.dependency_overrides.clear()


def test_service_aggregates_participants_without_registry():
    captured = {}

    def stub_explain(frame, top_n):
        captured["frame"] = frame
        captured["top_n"] = top_n
        return StubExplanation()

    service = PredictionService(explain_fn=stub_explain)
    request = PredictRequest(
        **BASE_REQUEST,
        vehicles=[
            {"vehicle_type": "9", "age_of_driver": 41},
            {"vehicle_type": "9", "age_of_driver": 25},
        ],
        casualties=[{"casualty_class": "3", "age_of_casualty": 33}],
    )
    result = asyncio.run(service.predict(request))

    frame = captured["frame"]
    assert len(frame) == 1
    assert frame["veh_vehicle_type_9"].iloc[0] == 2
    assert frame["veh_age_of_driver_mean"].iloc[0] == 33
    assert frame["cas_casualty_class_3"].iloc[0] == 1
    assert captured["top_n"] == 8
    assert result.usage.participant_rows == 3
    assert result.priority_label == "立即"


def test_service_maps_value_error_to_400():
    def broken(frame, top_n):
        raise ValueError("Prediction needs time and place")

    service = PredictionService(explain_fn=broken)
    with pytest.raises(InvalidInputError):
        asyncio.run(service.predict(PredictRequest(**BASE_REQUEST)))


def test_service_rejects_too_many_participants():
    service = PredictionService(explain_fn=lambda frame, top_n: StubExplanation())
    request = PredictRequest(
        **BASE_REQUEST,
        vehicles=[{"vehicle_type": "9"}] * 21,
    )
    with pytest.raises(InvalidInputError):
        asyncio.run(service.predict(request))


def test_service_maps_unexpected_error_to_503():
    def broken(frame, top_n):
        raise RuntimeError("registry exploded")

    service = PredictionService(explain_fn=broken)
    with pytest.raises(ModelUnavailableError):
        asyncio.run(service.predict(PredictRequest(**BASE_REQUEST)))
