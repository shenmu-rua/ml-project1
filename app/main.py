"""D 负责:FastAPI 预测服务。

* 先用 Dummy 模型跑通(project_plan 第 55 行),接真实模型只改 :func:`app.model_loader.load_model`。
* ``POST /predict``  输入一次事故特征,输出分诊优先级 1/2/3 与各等级概率。
* ``GET /model-info`` 返回模型版本与验收指标(前端信任表格用)。
* 每次预测写入请求日志(:func:`log_request`),便于审计"生命底线"。

运行(仓库根目录):
    uvicorn app.main:app --reload --port 8000
    # 文档: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI

from app.model_loader import DUMMY_RISK_METRICS, ModelService, load_model
from app.schemas import (
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RiskMetrics,
)

# ---------------------------------------------------------------------------
# 应用与模型
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Road Collision Severity Triage API",
    description="英国道路事故严重程度三级分诊预测接口(1 立即 / 2 紧急 / 3 常规)。",
    version="0.2.0",
)

model: ModelService = load_model()

# 验收指标(占位;真实值接 C 评估流水线后替换为从 MLflow/文件读取)
_RISK_METRICS = RiskMetrics(**DUMMY_RISK_METRICS)

# 请求日志
LOG_PATH = Path(__file__).resolve().parents[1] / "request_log.jsonl"
_logger = logging.getLogger("app.predict")


def log_request(payload: dict) -> None:
    """追加一条预测请求日志(追加写,不阻塞主流程)。"""
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        _logger.warning("无法写入请求日志 %s", LOG_PATH)


def _features_dict(req: PredictRequest) -> dict:
    """把 Pydantic 请求转成普通 dict,dtype None 保留为 None。"""
    return req.model_dump()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["info"],
    summary="模型信息与验收指标",
)
def model_info() -> ModelInfoResponse:
    """返回模型版本与测试集验收指标,供前端信任表格展示。"""
    return ModelInfoResponse(
        model_version=model.model_version,
        risk_metrics=_RISK_METRICS,
        input_columns=sorted(_features_dict(PredictRequest(collision_datetime_utc="1970-01-01T00:00:00Z")).keys()),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="事故分级分诊预测",
)
def predict(req: PredictRequest) -> PredictResponse:
    """输入一次事故特征,输出分诊优先级与各等级概率。"""
    started = time.perf_counter()

    features = _features_dict(req)
    priority, probability, probabilities = model.predict_with_probs(features)

    # 危及生命判定:命中最高的"立即"级即视为需要立即处置(保守兜底)
    is_life_threatening = priority == 1

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    response = PredictResponse(
        priority=priority,
        probability=probability,
        probabilities=probabilities,
        is_life_threatening=is_life_threatening,
        risk_metrics=_RISK_METRICS,
        model_version=model.model_version,
        latency_ms=latency_ms,
    )

    log_request(
        {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "priority": priority,
            "probability": probability,
            "is_life_threatening": is_life_threatening,
            "latency_ms": latency_ms,
        }
    )
    return response