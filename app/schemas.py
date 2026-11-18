"""D 负责:FastAPI 预测接口的请求/响应契约(Pydantic v2)。

请求契约由 ``schema.json`` 生成,不在这里重复抄写字段:

* 时间与地点必填(特征工程的太阳高度角无法补缺),时间必须带时区。
* 其余数值、类别特征可空;数值范围、类别取值都按 ``schema.json`` 校验,
  NaN / Infinity 一律拒绝,避免脏输入被模型静默当成"常规"。
* 车辆与伤亡人员以现场记录列表传入,由 ``src.participants.aggregate`` 聚合成
  模型的 ``veh_*`` / ``cas_*`` 输入列(见 :mod:`app.model_loader`)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    create_model,
)

from src.explain import SingleExplanation
from src.features import REQUIRED_INPUTS, SCHEMA

__all__ = [
    "CasualtyRecord",
    "Cutoffs",
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "RiskMetrics",
    "VehicleRecord",
]

_REQUEST_CONFIG = ConfigDict(
    extra="forbid", allow_inf_nan=False, coerce_numbers_to_str=True
)


def _blank_to_none(value: Any) -> Any:
    return None if isinstance(value, str) and not value.strip() else value


def _one_of(allowed: list, label: str):
    def check(value):
        if value is not None and value not in allowed:
            raise ValueError(f"{label} 必须是 {allowed} 之一")
        return value

    return check


def _numeric_field(name: str, spec: dict) -> tuple:
    bounds = {
        option: spec[key]
        for key, option in (("min", "ge"), ("max", "le"))
        if key in spec
    }
    annotation: Any = float
    if "allowed" in spec:
        annotation = Annotated[float, AfterValidator(_one_of(spec["allowed"], name))]
    if name in REQUIRED_INPUTS:
        return annotation, Field(description=spec["unit"], **bounds)
    return annotation | None, Field(default=None, description=spec["unit"], **bounds)


def _categorical_field(name: str, spec: dict) -> tuple:
    codes = list(spec["categories"])
    missing = set(spec["missing_codes"])

    def normalise(value: str | None) -> str | None:
        # 源数据里表示"未知"的编码与缺省等价,统一成 None。
        return None if value is None or value in missing else value

    annotation = Annotated[
        str | None,
        BeforeValidator(_blank_to_none),
        AfterValidator(normalise),
        AfterValidator(_one_of(codes, name)),
    ]
    labels = "; ".join(f"{code}={label}" for code, label in spec["categories"].items())
    return annotation, Field(default=None, description=labels)


def _record_model(name: str, table: str, doc: str) -> type[BaseModel]:
    """One on-scene vehicle or casualty row, with the columns the model aggregates."""
    spec = SCHEMA["participants"][table]
    fields: dict[str, Any] = {
        column: (
            Annotated[str | None, BeforeValidator(_blank_to_none)],
            Field(default=None, description="DfT 编码;缺省按未知(-1)计"),
        )
        for column in spec["counts"]
    }
    fields.update(
        {column: (float | None, Field(default=None, ge=0)) for column in spec["stats"]}
    )
    return create_model(name, __config__=_REQUEST_CONFIG, __doc__=doc, **fields)


VehicleRecord = _record_model(
    "VehicleRecord", "vehicle", "一辆涉事车辆(字段名同 DfT 车辆表)。"
)
CasualtyRecord = _record_model(
    "CasualtyRecord", "casualty", "一名伤亡人员(字段名同 DfT 伤亡表,不含伤情结论)。"
)

PredictRequest = create_model(
    "PredictRequest",
    __config__=_REQUEST_CONFIG,
    __doc__=(
        "``POST /predict`` 请求体:一次事故的现场信息。\n\n"
        "时间与经纬度必填,其余信息可以部分缺失;``vehicles`` / ``casualties`` "
        "缺省或为空列表表示该类信息尚未掌握。"
    ),
    **{
        SCHEMA["datetime_column"]: (
            AwareDatetime,
            Field(
                description="事故时间,ISO 8601 且带时区,例如 2025-01-15T08:30:00+00:00"
            ),
        )
    },
    **{
        name: _numeric_field(name, spec)
        for name, spec in SCHEMA["numeric_features"].items()
    },
    **{
        name: _categorical_field(name, spec)
        for name, spec in SCHEMA["categorical_features"].items()
    },
    vehicles=(
        list[VehicleRecord] | None,
        Field(default=None, description="涉事车辆列表"),
    ),
    casualties=(
        list[CasualtyRecord] | None,
        Field(default=None, description="伤亡人员列表"),
    ),
)


class Cutoffs(BaseModel):
    """严重度概率上的两条分级界线(由验证集确定,随模型一起保存)。"""

    routine_urgent: float = Field(
        description="概率 ≥ 此值为 2 级(紧急),否则 3 级(常规)"
    )
    urgent_immediate: float = Field(description="概率 ≥ 此值为 1 级(立即)")


class RiskMetrics(BaseModel):
    """当前模型在测试集(完整信息)上的四项验收指标。

    ``status`` 为 ``not_evaluated`` 时各指标为空:占位模型没有做过测试集评估。
    """

    undertriage_life: float | None = Field(description="危及生命事故被分到常规级的比例")
    undertriage_fatal: float | None = Field(description="致死事故被分到常规级的比例")
    routine_share: float | None = Field(description="常规级(3)占比")
    immediate_lift: float | None = Field(
        description="立即级(1)危及生命比例相对整体的倍数"
    )
    status: Literal["pass", "fail", "not_evaluated"] = Field(
        description="四项验收是否全部达标"
    )


class PredictResponse(BaseModel):
    """``POST /predict`` 响应体。"""

    request_id: str = Field(description="请求标识,与请求日志中的记录一一对应")
    priority: Literal[1, 2, 3] = Field(
        description="派单优先级:1 立即 / 2 紧急 / 3 常规"
    )
    probability: float = Field(description="严重度概率(致死或重伤)")
    cutoffs: Cutoffs
    missing_info: list[str] = Field(
        description="本次请求未提供的信息组(weather/road/vehicle/casualty);"
        "验收指标只针对信息完整的事故"
    )
    model_version: str = Field(description="模型版本")
    latency_ms: float = Field(description="推理耗时(毫秒)")
    explanation: SingleExplanation | None = Field(
        default=None, description="请求 explain=true 时返回单样本 SHAP 解释，贡献单位为 log-odds"
    )


class ModelInfoResponse(BaseModel):
    """``GET /model-info`` 响应:模型版本、分级界线与验收情况。"""

    model_version: str
    is_placeholder: bool = Field(description="是否为未接入真实模型的占位实现")
    cutoffs: Cutoffs
    risk_metrics: RiskMetrics
    acceptance_limits: dict[str, float] = Field(
        description="schema.json 中 triage.acceptance 的验收门槛"
    )
    loaded_at: datetime
