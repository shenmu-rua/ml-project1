"""D 负责:模型加载与统一推理入口。

* API 只通过 :class:`ModelService` 使用模型。它委托给一个与
  ``src.triage.TriageClassifier`` 同接口的对象:``predict_proba(X)`` 返回
  ``(n, 2)`` 概率,``cutoffs`` 是(常规/紧急, 紧急/立即)两条界线。
* 环境变量 ``MODEL_URI`` 指向 MLflow 模型(``models:/road-severity-model@production``)
  时加载真实模型,验收指标取自该模型版本对应的 final run;加载失败直接报错,
  不会悄悄退回占位模型。
* ``MODEL_URI`` 未设置时使用 :class:`DummyTriageModel` 跑通联调,并明确标记为
  占位、未评估。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.schemas import Cutoffs, RiskMetrics
from src.features import (
    CATEGORICAL_INPUTS,
    INFO_GROUPS,
    NUMERIC_INPUTS,
    SCHEMA,
    complete_inputs,
)
from src.participants import aggregate
from src.triage import acceptance, priorities

__all__ = ["DummyTriageModel", "ModelService", "build_frame", "load_model"]

_logger = logging.getLogger("app.model")

DATETIME_COLUMN = SCHEMA["datetime_column"]
PARTICIPANT_FIELDS = {"vehicle": "vehicles", "casualty": "casualties"}
# DfT 原始表用 -1 表示"未知";现场没填的编码按同样方式计数,与训练数据一致。
UNKNOWN_CODE = "-1"
NOT_EVALUATED = RiskMetrics(
    undertriage_life=None,
    undertriage_fatal=None,
    routine_share=None,
    immediate_lift=None,
    status="not_evaluated",
)
_REGISTRY_URI = re.compile(
    r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>[^/]+)|/(?P<version>\d+))$"
)


def _aggregate(table: str, records: list) -> pd.DataFrame:
    counted = list(SCHEMA["participants"][table]["counts"])
    rows = pd.DataFrame([record.model_dump() for record in records])
    rows[counted] = rows[counted].astype(object).fillna(UNKNOWN_CODE)
    rows["collision_index"] = "request"
    return aggregate(rows, table, SCHEMA).reset_index(drop=True)


def build_frame(request) -> pd.DataFrame:
    """One model input row from a validated request, in the training column order."""
    collision = request.model_dump(exclude=set(PARTICIPANT_FIELDS.values()))
    parts = [pd.DataFrame([collision])]
    for table, attribute in PARTICIPANT_FIELDS.items():
        records = getattr(request, attribute)
        if records:
            parts.append(_aggregate(table, records))
    frame = complete_inputs(pd.concat(parts, axis=1))
    frame[DATETIME_COLUMN] = pd.to_datetime(frame[DATETIME_COLUMN], utc=True)
    frame[NUMERIC_INPUTS] = frame[NUMERIC_INPUTS].astype(float)
    frame[CATEGORICAL_INPUTS] = frame[CATEGORICAL_INPUTS].astype(object)
    return frame


def missing_info(frame: pd.DataFrame) -> list[str]:
    """Information groups (see ``src.features.INFO_GROUPS``) with no value at all."""
    return [
        group
        for group, columns in INFO_GROUPS.items()
        if frame[columns].isna().all(axis=None)
    ]


class DummyTriageModel:
    """启发式占位模型,接口与 ``src.triage.TriageClassifier`` 相同。

    得分随危险信号(高限速、多车、低温、大风、湿滑路面)升高,只用于在真实
    模型就绪前跑通 API、前端与容器,没有任何预测效力。
    """

    cutoffs = (0.40, 0.70)
    _HAZARDOUS_SURFACES = ("2", "3", "4", "5", "6", "7")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        speed_limit = X["speed_limit"].fillna(30.0).clip(upper=70.0)
        vehicles = X["number_of_vehicles"].fillna(1.0).clip(upper=8.0)
        cold = (12.0 - X["temp"].fillna(12.0)).clip(lower=0.0)
        wind = X["wspd"].fillna(10.0).clip(upper=40.0)
        hazardous = X["road_surface_conditions"].isin(self._HAZARDOUS_SURFACES)
        score = (
            0.10
            + speed_limit / 70.0 * 0.35
            + vehicles / 8.0 * 0.20
            + cold / 40.0 * 0.10
            + wind / 40.0 * 0.10
            + hazardous * 0.15
        )
        positive = score.clip(0.0, 1.0).to_numpy(dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return priorities(self.predict_proba(X)[:, 1], self.cutoffs)


@dataclass(frozen=True)
class ModelService:
    """统一的推理入口:一行模型输入 → 严重度概率与派单优先级。"""

    model: Any
    model_version: str
    risk_metrics: RiskMetrics = field(default_factory=lambda: NOT_EVALUATED)
    is_placeholder: bool = False
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cutoffs(self) -> Cutoffs:
        low, high = self.model.cutoffs
        return Cutoffs(routine_urgent=low, urgent_immediate=high)

    def predict(self, frame: pd.DataFrame) -> tuple[int, float]:
        """(priority, 严重度概率);优先级与 ``TriageClassifier.predict`` 同一算法。"""
        probability = float(self.model.predict_proba(frame)[0, 1])
        if not np.isfinite(probability):
            # 不能让算不出的概率在界线比较中落成"常规"。
            raise ValueError("模型返回了非有限的概率,拒绝给出分级")
        priority = priorities(np.array([probability]), self.model.cutoffs)[0]
        return int(priority), probability


def _risk_metrics(metrics: dict[str, float]) -> RiskMetrics:
    try:
        checks = acceptance(metrics)
    except KeyError:
        return NOT_EVALUATED
    return RiskMetrics(
        undertriage_life=metrics["test_undertriage_life"],
        undertriage_fatal=metrics["test_undertriage_fatal"],
        routine_share=metrics["test_p3_share"],
        immediate_lift=metrics["test_p1_life_lift"],
        status="pass" if all(checks.values()) else "fail",
    )


def _registry_details(uri: str) -> tuple[str, RiskMetrics]:
    """Version label and test metrics of a registered model version's final run."""
    import mlflow

    match = _REGISTRY_URI.match(uri)
    if match is None:
        return uri, NOT_EVALUATED
    client, name = mlflow.MlflowClient(), match["name"]
    if match["alias"]:
        entry = client.get_model_version_by_alias(name, match["alias"])
    else:
        entry = client.get_model_version(name, match["version"])
    metrics = client.get_run(entry.run_id).data.metrics if entry.run_id else {}
    return f"{name} v{entry.version}", _risk_metrics(metrics)


def load_model(uri: str | None = None) -> ModelService:
    """Load the model named by ``uri`` (default: ``MODEL_URI``), else the placeholder."""
    uri = os.getenv("MODEL_URI", "") if uri is None else uri
    if not uri:
        _logger.warning(
            "MODEL_URI 未设置:使用占位模型 dummy-v1,分级结果不可用于实际派单"
        )
        return ModelService(DummyTriageModel(), "dummy-v1", is_placeholder=True)

    import mlflow

    # 需在仓库根目录运行:反序列化 TriageClassifier 要能导入 src 包。
    model = mlflow.sklearn.load_model(uri)
    version, risk_metrics = _registry_details(uri)
    _logger.info("已加载模型 %s (%s),验收状态 %s", version, uri, risk_metrics.status)
    return ModelService(model, version, risk_metrics)
