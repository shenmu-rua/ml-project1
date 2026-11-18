"""D 负责:FastAPI 预测服务。

* ``POST /predict``   输入一次事故的现场信息,输出派单优先级 1/2/3、严重度概率与分级界线。
* ``GET /model-info`` 返回模型版本、分级界线与测试集验收情况(前端信任表格用)。
* 每次预测追加一行请求日志(输入、结果、模型版本、请求标识),用于事后审计漏判。

模型由环境变量 ``MODEL_URI`` 指定,未设置时使用占位模型,见 :mod:`app.model_loader`。

运行(仓库根目录):
    uvicorn app.main:app --reload --port 8000
    # 文档: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model_loader import ModelService, build_frame, load_model, missing_info
from app.schemas import ModelInfoResponse, PredictRequest, PredictResponse
from src.explain import SingleExplanation, explain_single
from src.features import SCHEMA

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "request_log.jsonl"
ACCEPTANCE_LIMITS = {
    key: value
    for key, value in SCHEMA["triage"]["acceptance"].items()
    if isinstance(value, (int, float))
}
_logger = logging.getLogger("app.predict")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = load_model()
    yield


app = FastAPI(
    title="Road Collision Severity Triage API",
    description="英国道路事故严重程度三级派单优先级预测接口(1 立即 / 2 紧急 / 3 常规)。",
    version="0.3.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 without echoing the input: a rejected NaN is not valid JSON to send back."""
    detail = [
        {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def get_service(request: Request) -> ModelService:
    return request.app.state.service


Service = Annotated[ModelService, Depends(get_service)]


def log_request(record: dict) -> None:
    """同步追加一行 JSON;写入失败只告警,不影响已经算出的分级结果。"""
    path = Path(os.getenv("REQUEST_LOG_PATH") or DEFAULT_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        _logger.warning("无法写入请求日志 %s", path, exc_info=True)


@app.get("/health", tags=["ops"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["info"],
    summary="模型版本、分级界线与验收情况",
)
def model_info(service: Service) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=service.model_version,
        is_placeholder=service.is_placeholder,
        cutoffs=service.cutoffs,
        risk_metrics=service.risk_metrics,
        acceptance_limits=ACCEPTANCE_LIMITS,
        loaded_at=service.loaded_at,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_unset=True,
    tags=["inference"],
    summary="事故派单优先级预测",
)
def predict(req: PredictRequest, service: Service, explain: bool = False) -> PredictResponse:
    started = time.perf_counter()
    frame = build_frame(req)
    priority, probability = service.predict(frame)
    explanation: SingleExplanation | None = None
    if explain and not service.is_placeholder:
        try:
            candidate = explain_single(frame, model=service.model)
            if candidate.priority == priority and math.isclose(
                candidate.probability, probability, rel_tol=1e-6, abs_tol=1e-8
            ):
                explanation = candidate
            else:
                _logger.error("SHAP 与本次预测不一致，模型 %s", service.model_version)
        except Exception:
            _logger.exception("单样本 SHAP 计算失败，模型 %s", service.model_version)
    response = PredictResponse(
        request_id=uuid.uuid4().hex,
        priority=priority,
        probability=probability,
        cutoffs=service.cutoffs,
        missing_info=missing_info(frame),
        model_version=service.model_version,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    if explain:
        response.explanation = explanation
    log_request(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **response.model_dump(exclude={"explanation"}),
            "inputs": req.model_dump(mode="json", exclude_none=True),
        }
    )
    return response
