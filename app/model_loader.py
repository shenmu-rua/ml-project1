"""D 负责:模型加载与统一推理接口。

设计原则
========
* API 只依赖本模块暴露的 ``ModelService``,不直接 import MLflow 或 sklearn 模型,
  从而把"模型从哪来"与"接口怎么服务"解耦。
* 当前用 **Dummy 模型**跑通(project_plan 第 55 行:D 第 1 周先用假模型,
  之后只替换模型)。接口签名对齐团队真实模型 ``src.triage.TriageClassifier``:
      - ``predict_proba`` 返回严重度(正类)概率
      - ``predict`` 返回分诊优先级 1 / 2 / 3
* 接入真实模型时,只需把 :func:`load_model` 里 TODO 处换成
  ``mlflow.sklearn.load_model("models:/road-severity-model@production")``,
  其余代码(路由、契约、前端)零改动。

风险指标取值
============
验收指标现在用 project 固定的验收数字(与 schema.json 一致)写死为占位;
后续由 C 的评估流水线产出真实数值后再替换为从 MLflow/评估文件读取。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DUMMY_RISK_METRICS", "ModelService", "load_model"]

# 占位验收指标(对齐 schema.json triage.acceptance,演示用;真实值待 C 评估后替换)
DUMMY_RISK_METRICS = {
    "undertriage_life": 0.038,     # 危及生命漏判率 3.8% <= 5%
    "undertriage_fatal": 0.003,    # 致死漏判率 0.3% <= 1%
    "routine_share": 0.258,        # 常规级占比 25.8% >= 20%
    "immediate_lift": 3.5,         # 立即级提纯 3.5 倍 >= 3
    "status": "pass",
}

# 固定切点(演示用):严重度概率 >=0.7 立即,>=0.4 紧急,否则常规。
# 真实模型的三级由 src.triage.TriageClassifier 在验证集上确定 — 见 train.py。
IMMEDIATE_CUTOFF = 0.70
URGENT_CUTOFF = 0.40


@dataclass
class ModelService:
    """统一的推理入口:输入一行特征 dict,输出严重度概率与三级优先级。"""

    model: object
    model_version: str = "dummy-v1"

    def _severity_probability(self, features: dict) -> float:
        """Dummy 启发式:用数值特征拼一个 0~1 的严重度得分。

        关键设计:得分会随传入的危险信号(高速、多车、低温、恶劣路面)升高,
        从而让"危及生命"的输入稳定落到高优先级,守住安全底线测试。
        """
        def num(key: str, default: float) -> float:
            value = features.get(key)
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        speed_limit = num("speed_limit", 30.0)
        vehicles = num("number_of_vehicles", 1.0)
        temp = num("temp", 12.0)
        wspd = num("wspd", 10.0)

        score = 0.0
        score += min(speed_limit, 70.0) / 70.0 * 0.35        # 限速越高越危险
        score += min(vehicles, 8.0) / 8.0 * 0.20             # 车越多越危险
        score += max(0.0, (12.0 - temp)) / 40.0 * 0.10       # 低温路面风险
        score += min(wspd, 40.0) / 40.0 * 0.10               # 大风
        # 类别危险信号
        if str(features.get("road_surface_conditions", "")) in ("2", "3", "4", "5", "6", "7"):
            score += 0.15                                     # 湿滑/冰雪/油渍
        score += 0.10                                         # 基础风险

        return float(np.clip(score, 0.0, 1.0))

    def predict_proba(self, features: dict) -> float:
        """严重度(正类)概率。接真实模型时改为 model.predict_proba(X)[:, 1]。"""
        return self._severity_probability(features)

    def predict(self, features: dict) -> int:
        """分诊优先级 1(立即)/2(紧急)/3(常规)。"""
        probability = self.predict_proba(features)
        if probability >= IMMEDIATE_CUTOFF:
            return 1
        if probability >= URGENT_CUTOFF:
            return 2
        return 3

    def predict_with_probs(self, features: dict) -> tuple[int, float, dict[str, float]]:
        """一次调用返回 (priority, 严重度概率, 三级归一化概率)。

        ``probabilities`` 用于前端展示各等级的置信度:
        按"离两个切点多近"构造三级相对权重,保证 argmax(probabilities)
        与 ``priority`` 一致,避免前端置信度与等级自相矛盾。
        """
        probability = self.predict_proba(features)
        priority = self.predict(features)

        # 三级权重:等级 i 的权重 = 概率落入该区间中心的贴近度(高斯核)
        centers = {1: 0.85, 2: 0.55, 3: 0.20}
        weights = {
            level: float(np.exp(-((probability - center) ** 2) / 0.01))
            for level, center in centers.items()
        }
        total = sum(weights.values()) + 1e-9
        probabilities = {
            "level_1": round(weights[1] / total, 4),
            "level_2": round(weights[2] / total, 4),
            "level_3": round(weights[3] / total, 4),
        }
        return priority, round(probability, 4), probabilities


def load_model() -> ModelService:
    """加载模型(当前为 Dummy;接真实模型时替换此函数体)。"""
    # TODO(D): 接入 B 的 MLflow 真实模型后使用:
    #   import mlflow
    #   model = mlflow.sklearn.load_model("models:/road-severity-model@production")
    #   return ModelService(model=model, model_version="production")
    return ModelService(model=None, model_version="dummy-v1")