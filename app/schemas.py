"""D 负责：FastAPI 预测接口的请求/响应契约(Pydantic v2)。

契约对齐 ``schema.json``:
* 输入列 = 时间列 ``collision_datetime_utc`` + 10 个数值特征 + 11 个类别特征。
* 输出 = 分诊优先级 priority(1 立即 / 2 紧急 / 3 常规)+ 严重度概率
  + 各级概率 + 是否危及生命 + 模型评估指标。

说明
====
* 类别特征在 schema.json 里都是字符串码(如 "1"/"2"),这里统一用 ``str | None``
  接收,前端下拉框维护合法取值;后端不做硬枚举校验,交给 A 的 pandera
  pipeline 在真实模型侧把关。
* 数值特征允许缺失(None),Dummy 模型会忽略缺省值;接真实模型后由
  ``src.features.complete_inputs`` 兼容部分信息缺失。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "RiskMetrics",
]


class PredictRequest(BaseModel):
    """``POST /predict`` 请求体:一次事故的特征(部分字段可为空)。

    字段名与 schema.json 完全一致。datetime 用 ISO 8601 UTC 字符串。
    """

    model_config = ConfigDict(extra="forbid")

    # ---- 必填:时间与地点(特征工程需要) ----
    collision_datetime_utc: str = Field(
        description="事故时间,ISO 8601,例如 2025-01-15T08:30:00+00:00"
    )
    longitude: float | None = Field(default=None, description="经度(度,-9~3)")
    latitude: float | None = Field(default=None, description="纬度(度,49~62)")

    # ---- 数值特征(可空) ----
    speed_limit: float | None = Field(default=None, description="限速(mph)")
    number_of_vehicles: float | None = Field(default=None, description="涉事车辆数")
    temp: float | None = Field(default=None, description="气温(degC)")
    rhum: float | None = Field(default=None, description="相对湿度(%)")
    prcp: float | None = Field(default=None, description="降水量(mm/小时)")
    wdir: float | None = Field(default=None, description="风向(度)")
    wspd: float | None = Field(default=None, description="风速(km/h)")
    pres: float | None = Field(default=None, description="气压(hPa)")

    # ---- 类别特征(字符串码,可空) ----
    road_type: str | None = Field(default=None, description="道路类型")
    junction_detail: str | None = Field(default=None, description="路口详情")
    junction_control: str | None = Field(default=None, description="路口控制")
    pedestrian_crossing: str | None = Field(default=None, description="人行横道设施")
    weather_conditions: str | None = Field(default=None, description="天气条件")
    road_surface_conditions: str | None = Field(default=None, description="路面状况")
    urban_or_rural_area: str | None = Field(default=None, description="城区/乡村")
    first_road_class: str | None = Field(default=None, description="一级道路等级")
    second_road_class: str | None = Field(default=None, description="二级道路等级")
    light_conditions: str | None = Field(default=None, description="光照条件")

    # ---- 参与者聚合特征(可空,B 用 participants.py 生成,前端可用缺省) ----
    vehicle_count: float | None = Field(default=None, description="车辆聚合特征数(占位)")
    casualty_count: float | None = Field(default=None, description="伤亡聚合特征数(占位)")


class RiskMetrics(BaseModel):
    """模型在测试集上的验收指标(来自 schema.json 的 triage.acceptance)。

    供前端信任表格与答辩展示;数值由评估流水线写入,随响应回传。
    """

    undertriage_life: float = Field(description="危及生命事故被分到常规级的比例(漏判率)")
    undertriage_fatal: float = Field(description="致死事故被分到常规级的比例(致死漏判率)")
    routine_share: float = Field(description="常规级(3)占比")
    immediate_lift: float = Field(description="立即级(1)危及生命比例相对整体的提纯倍数")
    status: Literal["pass", "fail"] = Field(description="四项验收是否全部达标")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "undertriage_life": 0.038,
                "undertriage_fatal": 0.003,
                "routine_share": 0.258,
                "immediate_lift": 3.5,
                "status": "pass",
            }
        }
    )


class PredictResponse(BaseModel):
    """``POST /predict`` 响应体。"""

    priority: Literal[1, 2, 3] = Field(description="分诊优先级:1 立即 / 2 紧急 / 3 常规")
    probability: float = Field(description="严重度概率(模型 predict_proba 的正类概率)")
    probabilities: dict[str, float] = Field(
        description="三级分诊的相对概率(归一化后的展示值)"
    )
    is_life_threatening: bool = Field(description="是否判定为危及生命")
    risk_metrics: RiskMetrics = Field(description="当前模型的测试集验收指标")
    model_version: str = Field(description="模型版本/别名")
    latency_ms: float = Field(description="推理耗时(毫秒)")


class ModelInfoResponse(BaseModel):
    """``GET /model-info`` 响应:模型信息与验收指标(前端信任表格用)。"""

    model_version: str = Field(description="模型版本/别名")
    risk_metrics: RiskMetrics = Field(description="测试集验收指标")
    input_columns: list[str] = Field(description="模型输入列名(schema.json 契约)")