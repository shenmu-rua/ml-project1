"""D 负责:FastAPI 预测接口的安全底线测试。

安全底线:已知危及生命的事故绝不能落为 3 级(常规),否则抛出致命断言
(等价于业务拦截)。Dummy 模型按危险信号(高速、多车、低温、恶劣路面)
升高严重度得分,因此危险输入会稳定落为 1 或 2 级。

运行:
    pytest tests/test_app.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# 测试特征集
# ---------------------------------------------------------------------------
LIFE_THREATENING_FEATURES: dict = {
    "collision_datetime_utc": "2025-01-15T08:30:00+00:00",
    "longitude": 0.0,
    "latitude": 51.5,
    "speed_limit": 70.0,          # 高速
    "number_of_vehicles": 8.0,    # 多车
    "temp": -10.0,                # 低温
    "wspd": 40.0,                 # 大风
    "prcp": 0.0,
    "road_surface_conditions": "3",  # 冰雪路面
    "weather_conditions": "3",       # 降雪
    "urban_or_rural_area": "2",      # 乡村
}

MINOR_ACCIDENT_FEATURES: dict = {
    "collision_datetime_utc": "2025-01-15T08:30:00+00:00",
    "longitude": 0.0,
    "latitude": 51.5,
    "speed_limit": 20.0,
    "number_of_vehicles": 1.0,
    "temp": 18.0,
    "wspd": 5.0,
    "prcp": 0.0,
    "road_surface_conditions": "1",  # 干燥
    "weather_conditions": "1",       # 晴朗
    "urban_or_rural_area": "1",      # 城区
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _post_predict(client: TestClient, features: dict) -> dict:
    resp = client.post("/predict", json=features)
    assert resp.status_code == 200, f"/predict 失败: {resp.status_code} {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# 安全底线 1:危及生命事故绝不能落为 3 级
# ---------------------------------------------------------------------------
def test_life_threatening_never_routine(client: TestClient) -> None:
    """已知危及生命的输入,断言 priority != 3(致命断言)。"""
    body = _post_predict(client, LIFE_THREATENING_FEATURES)

    assert body["priority"] != 3, (
        f"【致命拦截】危及生命事故被分诊为 3 级(常规)! "
        f"probability={body['probability']} —— 漏判,已触发业务拦截!"
    )
    # 应立即级或紧急级,且危及生命标记为真概率高
    assert body["priority"] in (1, 2)
    assert body["risk_metrics"]["status"] == "pass"


# ---------------------------------------------------------------------------
# 安全底线 2:轻微事故应稳定落为 3 级
# ---------------------------------------------------------------------------
def test_minor_accident_routine(client: TestClient) -> None:
    body = _post_predict(client, MINOR_ACCIDENT_FEATURES)
    assert body["priority"] == 3, (
        f"轻微事故被误分为 {body['priority']} 级,probability={body['probability']}"
    )
    assert body["is_life_threatening"] is False


# ---------------------------------------------------------------------------
# 可观测性:/model-info 返回验收指标
# ---------------------------------------------------------------------------
def test_model_info_metrics(client: TestClient) -> None:
    resp = client.get("/model-info")
    assert resp.status_code == 200, resp.text
    info = resp.json()
    rm = info["risk_metrics"]
    assert rm["undertriage_life"] == pytest.approx(0.038)
    assert rm["undertriage_fatal"] == pytest.approx(0.003)
    assert rm["routine_share"] == pytest.approx(0.258)
    assert rm["immediate_lift"] == pytest.approx(3.5)
    assert rm["status"] == "pass"
    assert info["model_version"]
    assert "collision_datetime_utc" in info["input_columns"]


# ---------------------------------------------------------------------------
# 契约完整性
# ---------------------------------------------------------------------------
def test_predict_contract(client: TestClient) -> None:
    body = _post_predict(client, MINOR_ACCIDENT_FEATURES)
    assert body["priority"] in (1, 2, 3)
    probs = body["probabilities"]
    assert set(probs) == {"level_1", "level_2", "level_3"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-2)
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["is_life_threatening"], bool)
    assert body["latency_ms"] >= 0


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}