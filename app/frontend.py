"""D 负责:Streamlit 前端 —— 事故分级分诊交互界面。

设计
====
1. 醒目分级:1 级大红警报 / 2 级橙色 / 3 级绿色横幅。
2. 置信度:三级概率百分比 + 进度条。
3. SHAP 瀑布图:调用 C 的 ``src.explain`` —— 目前 ``src/explain.py`` 为空占位,
   故用 mock 数据渲染瀑布图,并预留真实调用点(TODO)。
4. 底部信任表格:展示模型测试集验收指标,增强使用者信任。

运行:
    streamlit run app/frontend.py

API 地址:
    优先读环境变量 API_URL(默认本地 http://127.0.0.1:8000/predict),
    Docker 内通过 compose 注入 http://api:8000/predict。
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
MODEL_INFO_URL = API_URL.rsplit("/predict", 1)[0] + "/model-info"

LEVEL_META: dict[int, dict[str, str]] = {
    1: {"label": "立即处置 (Immediate)", "color": "#dc2626", "bg": "#fef2f2", "icon": "🚨", "emoji": "🔴"},
    2: {"label": "紧急处置 (Urgent)", "color": "#ea580c", "bg": "#fff7ed", "icon": "⚠️", "emoji": "🟠"},
    3: {"label": "常规处置 (Routine)", "color": "#16a34a", "bg": "#f0fdf4", "icon": "✅", "emoji": "🟢"},
}

# 类别特征的合法下拉取值(来自 schema.json,前端只展示常用项)
CATEGORY_OPTIONS: dict[str, list[str]] = {
    "road_type": ["", "1", "2", "3", "6", "7", "12"],
    "road_surface_conditions": ["", "1", "2", "3", "4", "5", "6", "7"],
    "weather_conditions": ["", "1", "2", "3", "4", "5", "6", "7", "8"],
    "urban_or_rural_area": ["", "1", "2"],
    "light_conditions": ["", "1", "4", "5", "6", "7"],
    "junction_control": ["", "0", "1", "2", "3", "4"],
}


# ---------------------------------------------------------------------------
# 后端调用
# ---------------------------------------------------------------------------
def call_predict(features: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(API_URL, json=features, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def call_model_info() -> dict[str, Any]:
    resp = requests.get(MODEL_INFO_URL, timeout=5)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# SHAP 瀑布图(预留 C 的 src.explain 接入)
# ---------------------------------------------------------------------------
def shap_waterfall(features: dict[str, Any]) -> None:
    """渲染单样本 SHAP 瀑布图。

    TODO(D): 待 C 实现 ``src.explain`` 后,替换为真实调用,例如::
        from src.explain import explain
        sv = explain(model, pd.DataFrame([features]))   # 返回 shap.Explanation
        shap.plots.waterfall(sv, max_display=10, show=False)
        st.pyplot(plt.gcf())

    当前为 mock:按特征绝对贡献生成确定性瀑布,仅供演示布局。
    """
    try:
        import shap  # noqa: F401

        st.warning("SHAP 评估模块(C)尚未接入,当前为 mock 瀑布图。")
    except ImportError:
        pass

    numeric = {
        k: (float(v) if v is not None else 0.0)
        for k, v in features.items()
        if k not in ("collision_datetime_utc",)
        and isinstance(v, (int, float))
    }
    ordered = sorted(numeric.items(), key=lambda kv: -abs(kv[1]))[:8]
    if not ordered:
        st.info("无可解释的数值特征。")
        return

    names = [k for k, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(names))))
    colors = ["#dc2626" if v >= 0 else "#2563eb" for v in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("SHAP value (对严重度概率的贡献)")
    ax.set_title("单样本解释 · 瀑布图 (mock,待 C 接入)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 醒目横幅
# ---------------------------------------------------------------------------
def banner(priority: int, is_life_threatening: bool) -> None:
    meta = LEVEL_META[priority]
    alert = "⚠ 危及生命" if is_life_threatening else ""
    st.markdown(
        f"""
        <div style="background:{meta['bg']};border:3px solid {meta['color']};
                    border-radius:14px;padding:24px 20px;text-align:center;">
            <div style="font-size:56px;">{meta['icon']}</div>
            <div style="font-size:30px;font-weight:800;color:{meta['color']};">
                {meta['label']}
            </div>
            <div style="font-size:18px;color:#334155;margin-top:6px;">
                分诊优先级 <b style="color:{meta['color']};">{priority}</b>
                {' · ' + alert if alert else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence(probabilities: dict[str, float]) -> None:
    cols = st.columns(3)
    items = [
        ("1 级 · 立即", probabilities.get("level_1", 0.0), LEVEL_META[1]),
        ("2 级 · 紧急", probabilities.get("level_2", 0.0), LEVEL_META[2]),
        ("3 级 · 常规", probabilities.get("level_3", 0.0), LEVEL_META[3]),
    ]
    for col, (name, prob, meta) in zip(cols, items):
        with col:
            st.metric(name, f"{prob:.1%}")
            st.progress(min(1.0, prob))


def trust_table(model_info: dict[str, Any]) -> None:
    rm = model_info.get("risk_metrics", {})
    rows = [
        ("危及生命事故漏判率", f"{rm.get('undertriage_life', 0):.1%}", "≤ 5%", "✅" if rm.get("undertriage_life", 1) <= 0.05 else "❌"),
        ("致死事故漏判率", f"{rm.get('undertriage_fatal', 0):.1%}", "≤ 1%", "✅" if rm.get("undertriage_fatal", 1) <= 0.01 else "❌"),
        ("常规级(3)占比", f"{rm.get('routine_share', 0):.1%}", "≥ 20%", "✅" if rm.get("routine_share", 0) >= 0.20 else "❌"),
        ("立即级提纯倍数", f"{rm.get('immediate_lift', 0):.1f}×", "≥ 3×", "✅" if rm.get("immediate_lift", 0) >= 3.0 else "❌"),
    ]
    df = pd.DataFrame(rows, columns=["验收指标", "测试集结果", "验收标准", "是否达标"])
    st.markdown("#### 🛡️ 本模型在测试集上的表现")
    st.table(df)
    st.caption(f"模型版本 {model_info.get('model_version', 'N/A')} · 验收状态:"
               f"{'✅ 全部通过' if rm.get('status') == 'pass' else '❌ 存在超标项'}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="事故分级分诊系统", page_icon="🚑", layout="centered")
    st.title("🚑 道路事故分级分诊系统")
    st.caption("输入事故特征,系统分诊为 1(立即)/2(紧急)/3(常规)三级急救优先级。")

    with st.sidebar:
        st.header("事故特征输入")
        dt = st.text_input("事故时间 (UTC)", value="2025-01-15T08:30:00+00:00")
        longitude = st.number_input("经度 longitude", value=0.0, step=0.1)
        latitude = st.number_input("纬度 latitude", value=51.5, step=0.1)
        speed_limit = st.selectbox("限速 speed_limit (mph)", [20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        number_of_vehicles = st.number_input("涉事车辆数", value=1.0, min_value=1.0, step=1.0)
        temp = st.number_input("气温 temp (℃)", value=12.0, step=1.0)
        wspd = st.number_input("风速 wspd (km/h)", value=10.0, step=1.0)
        prcp = st.number_input("降水 prcp (mm)", value=0.0, step=0.5)
        road_surface = st.selectbox("路面状况", CATEGORY_OPTIONS["road_surface_conditions"])
        weather = st.selectbox("天气条件", CATEGORY_OPTIONS["weather_conditions"])
        urban = st.selectbox("城区/乡村", CATEGORY_OPTIONS["urban_or_rural_area"])

        predict_btn = st.button("🔍 开始分诊", type="primary", use_container_width=True)

    if not predict_btn:
        st.info("在左侧填写特征后点击「开始分诊」。")
        try:
            trust_table(call_model_info())
        except requests.RequestException:
            st.error("无法连接后端 /model-info,请先启动 FastAPI。")
        return

    features = {
        "collision_datetime_utc": dt,
        "longitude": longitude,
        "latitude": latitude,
        "speed_limit": speed_limit,
        "number_of_vehicles": number_of_vehicles,
        "temp": temp,
        "wspd": wspd,
        "prcp": prcp,
        "road_surface_conditions": road_surface or None,
        "weather_conditions": weather or None,
        "urban_or_rural_area": urban or None,
    }

    with st.spinner("模型推理中…"):
        try:
            result = call_predict(features)
        except requests.RequestException as exc:
            st.error(f"调用后端失败:{exc}")
            st.stop()

    priority = int(result["priority"])
    banner(priority, bool(result["is_life_threatening"]))
    st.markdown("#### 置信度")
    confidence(result["probabilities"])

    st.markdown("##### 严重度概率")
    st.metric("严重度(正类)概率", f"{result['probability']:.1%}")

    st.markdown("#### 🔍 模型解读 (SHAP)")
    shap_waterfall(features)

    st.divider()
    try:
        trust_table(call_model_info())
    except requests.RequestException:
        st.warning("评估指标展示不可用(后端未启动)。")


if __name__ == "__main__":
    main()