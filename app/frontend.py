"""D 负责:Streamlit 前端 —— 事故三级派单优先级交互界面。

* 输入项与下拉取值由 ``schema.json`` 生成,与 API 的校验同源。
* 结果:醒目的 1 / 2 / 3 级横幅 + 严重度概率与模型的两条分级界线。
* 底部信任表格:当前模型在测试集上的验收指标,门槛与数值都来自 ``/model-info``;
  占位模型没有评估结果时如实说明。
* 单样本 SHAP 解释待 C 的 ``src.explain`` 就绪后接入,目前只给出说明,不画示意图。

运行:
    streamlit run app/frontend.py

环境变量 ``API_URL`` 是后端根地址,默认 http://127.0.0.1:8000;
docker-compose 内为 http://api:8000。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema.json").read_text(encoding="utf-8")
)
API_URL = (
    os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/").removesuffix("/predict")
)

LEVEL_META: dict[int, dict[str, str]] = {
    1: {
        "label": "立即处置 (Immediate)",
        "color": "#dc2626",
        "bg": "#fef2f2",
        "icon": "🚨",
    },
    2: {"label": "紧急处置 (Urgent)", "color": "#ea580c", "bg": "#fff7ed", "icon": "⚠️"},
    3: {
        "label": "常规处置 (Routine)",
        "color": "#16a34a",
        "bg": "#f0fdf4",
        "icon": "✅",
    },
}
CATEGORY_LABELS = {
    "road_type": "道路类型",
    "junction_detail": "路口详情",
    "junction_control": "路口控制",
    "pedestrian_crossing": "人行横道设施",
    "weather_conditions": "天气条件",
    "road_surface_conditions": "路面状况",
    "urban_or_rural_area": "城区/乡村",
    "first_road_class": "一级道路等级",
    "second_road_class": "二级道路等级",
    "light_conditions": "光照条件",
}
WEATHER_INPUTS = {
    "temp": ("气温 (℃)", 12.0),
    "rhum": ("相对湿度 (%)", 80.0),
    "prcp": ("降水 (mm/小时)", 0.0),
    "wdir": ("风向 (度)", 220.0),
    "wspd": ("风速 (km/h)", 15.0),
    "pres": ("海平面气压 (hPa)", 1013.0),
}
# 现场最容易掌握的参与者字段;其余字段缺省,由后端按"未知"计。
PARTICIPANT_COLUMNS = {
    "vehicle": [
        "vehicle_type",
        "vehicle_manoeuvre",
        "first_point_of_impact",
        "skidding_and_overturning",
        "sex_of_driver",
        "age_of_driver",
        "engine_capacity_cc",
    ],
    "casualty": [
        "casualty_class",
        "casualty_type",
        "sex_of_casualty",
        "age_of_casualty",
        "pedestrian_location",
        "pedestrian_movement",
    ],
}
INFO_GROUP_LABELS = {
    "weather": "天气观测",
    "road": "道路与环境",
    "vehicle": "车辆",
    "casualty": "伤亡人员",
}
# (验收项, 指标键, 门槛键, 方向, 数值格式)
ACCEPTANCE_ROWS = [
    ("危及生命事故漏判率", "undertriage_life", "max_undertriage_life", "≤", "{:.1%}"),
    ("致死事故漏判率", "undertriage_fatal", "max_undertriage_fatal", "≤", "{:.1%}"),
    ("常规级(3)占比", "routine_share", "min_routine_share", "≥", "{:.1%}"),
    ("立即级提纯倍数", "immediate_lift", "min_immediate_life_lift", "≥", "{:.1f}×"),
]


# ---------------------------------------------------------------------------
# 后端调用
# ---------------------------------------------------------------------------
def call_predict(features: dict[str, Any]) -> requests.Response:
    return requests.post(f"{API_URL}/predict", json=features, timeout=10)


@st.cache_data(ttl=300, show_spinner=False)
def call_model_info() -> dict[str, Any]:
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
def category_select(name: str) -> str | None:
    categories = SCHEMA["categorical_features"][name]["categories"]
    return st.selectbox(
        CATEGORY_LABELS[name],
        [None, *categories],
        format_func=lambda code: (
            "未知" if code is None else f"{code} · {categories[code]}"
        ),
    )


def participant_editor(table: str, title: str) -> list[dict[str, Any]]:
    """可增删行的表格;返回非空行,每行只保留填了的字段。"""
    spec = SCHEMA["participants"][table]
    columns = PARTICIPANT_COLUMNS[table]
    config = {
        column: (
            st.column_config.NumberColumn(min_value=0)
            if column in spec["stats"]
            else st.column_config.SelectboxColumn(options=spec["counts"][column])
        )
        for column in columns
    }
    with st.expander(title):
        st.caption(
            "每行一条记录,取值为 DfT 编码(-1 表示未知);不填则按该类信息未掌握处理。"
        )
        edited = st.data_editor(
            pd.DataFrame(columns=columns),
            column_config=config,
            num_rows="dynamic",
            hide_index=True,
            key=f"{table}_editor",
        )
    return filled_records(edited)


def filled_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Non-empty rows of an edited table, each keeping only the cells that were filled."""
    records = [
        {key: value for key, value in row.items() if pd.notna(value)}
        for row in table.to_dict("records")
    ]
    return [record for record in records if record]


def sidebar_inputs() -> dict[str, Any]:
    numeric = SCHEMA["numeric_features"]
    features: dict[str, Any] = {}
    st.header("事故信息")
    st.subheader("时间与地点(必填)")
    features["collision_datetime_utc"] = st.text_input(
        "事故时间 (ISO 8601,带时区)", value="2025-01-15T08:30:00+00:00"
    )
    for name, label, default in (
        ("latitude", "纬度", 51.5),
        ("longitude", "经度", -0.1),
    ):
        features[name] = st.number_input(
            label,
            value=default,
            min_value=float(numeric[name]["min"]),
            max_value=float(numeric[name]["max"]),
            step=0.1,
        )

    st.subheader("道路与环境")
    features["speed_limit"] = st.selectbox(
        "限速 (mph)",
        [None, *numeric["speed_limit"]["allowed"]],
        index=2,
        format_func=lambda value: "未知" if value is None else str(value),
    )
    features["number_of_vehicles"] = st.number_input(
        "涉事车辆数", value=None, min_value=1, step=1, placeholder="未知"
    )
    for name in SCHEMA["categorical_features"]:
        features[name] = category_select(name)

    st.subheader("天气观测")
    if st.checkbox("已掌握事故前后一小时内的天气观测"):
        for name, (label, default) in WEATHER_INPUTS.items():
            features[name] = st.number_input(
                label,
                value=default,
                min_value=float(numeric[name]["min"])
                if "min" in numeric[name]
                else None,
                max_value=float(numeric[name]["max"])
                if "max" in numeric[name]
                else None,
            )
    return {key: value for key, value in features.items() if value is not None}


# ---------------------------------------------------------------------------
# 结果展示
# ---------------------------------------------------------------------------
def banner(priority: int) -> None:
    meta = LEVEL_META[priority]
    st.markdown(
        f"""
        <div style="background:{meta["bg"]};border:3px solid {meta["color"]};
                    border-radius:14px;padding:24px 20px;text-align:center;">
            <div style="font-size:56px;">{meta["icon"]}</div>
            <div style="font-size:30px;font-weight:800;color:{meta["color"]};">
                {meta["label"]}
            </div>
            <div style="font-size:18px;color:#334155;margin-top:6px;">
                派单优先级 <b style="color:{meta["color"]};">{priority}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_against_cutoffs(result: dict[str, Any]) -> None:
    cutoffs = result["cutoffs"]
    st.markdown("#### 严重度概率与分级界线")
    columns = st.columns(3)
    columns[0].metric("严重度概率(致死或重伤)", f"{result['probability']:.1%}")
    columns[1].metric("≥ 此值为 2 级(紧急)", f"{cutoffs['routine_urgent']:.1%}")
    columns[2].metric("≥ 此值为 1 级(立即)", f"{cutoffs['urgent_immediate']:.1%}")
    st.progress(min(1.0, max(0.0, result["probability"])))
    st.caption(
        "优先级由概率与两条界线比较得出;界线在验证集上确定,随模型一起保存。"
        f" 请求标识 {result['request_id']}"
    )


def missing_info_notice(missing: list[str]) -> None:
    if missing:
        names = "、".join(INFO_GROUP_LABELS.get(group, group) for group in missing)
        st.warning(
            f"本次未提供:{names}。信息越少,模型越倾向给出 2 级,分流作用减弱;"
            "下方验收指标只针对信息完整的事故。"
        )


def trust_table(info: dict[str, Any]) -> None:
    metrics, limits = info["risk_metrics"], info["acceptance_limits"]
    st.markdown("#### 🛡️ 当前模型在测试集上的表现")
    if info["is_placeholder"]:
        st.warning(
            f"当前为占位模型 {info['model_version']},没有在测试集上评估过,"
            "分级结果仅用于联调演示,不能用于实际派单。"
        )
        return
    if metrics["status"] == "not_evaluated":
        st.warning(f"模型 {info['model_version']} 没有可用的测试集验收记录。")
        return
    rows = []
    for label, key, limit_key, direction, fmt in ACCEPTANCE_ROWS:
        value, limit = metrics[key], limits[limit_key]
        passed = value <= limit if direction == "≤" else value >= limit
        rows.append(
            (
                label,
                fmt.format(value),
                f"{direction} {fmt.format(limit)}",
                "✅" if passed else "❌",
            )
        )
    st.table(
        pd.DataFrame(rows, columns=["验收指标", "测试集结果", "验收标准", "是否达标"])
    )
    st.caption(
        f"模型版本 {info['model_version']} · 验收状态:"
        f"{'✅ 全部通过' if metrics['status'] == 'pass' else '❌ 存在未达标项'}"
    )


def show_model_info() -> None:
    try:
        trust_table(call_model_info())
    except requests.RequestException:
        st.error(f"无法连接后端 {API_URL}/model-info,请先启动 FastAPI。")


def show_request_error(response: requests.Response) -> None:
    if response.status_code == 422:
        st.error("输入未通过校验:")
        for error in response.json()["detail"]:
            field = ".".join(str(part) for part in error["loc"][1:])
            st.write(f"- `{field}`:{error['msg']}")
    else:
        st.error(f"后端返回 {response.status_code}:{response.text[:300]}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="事故派单优先级系统", page_icon="🚑", layout="centered"
    )
    st.title("🚑 道路事故派单优先级系统")
    st.caption("输入事故现场信息,系统给出 1(立即)/2(紧急)/3(常规)三级急救派单优先级。")

    with st.sidebar:
        features = sidebar_inputs()
    vehicles = participant_editor("vehicle", "涉事车辆(可选)")
    casualties = participant_editor("casualty", "伤亡人员(可选)")
    if vehicles:
        features["vehicles"] = vehicles
    if casualties:
        features["casualties"] = casualties

    if not st.button("🔍 开始分诊", type="primary", use_container_width=True):
        st.info("在左侧填写事故信息后点击「开始分诊」。")
        show_model_info()
        return

    with st.spinner("模型推理中…"):
        try:
            response = call_predict(features)
        except requests.RequestException as exc:
            st.error(f"调用后端失败:{exc}")
            return
    if response.status_code != 200:
        show_request_error(response)
        return

    result = response.json()
    banner(int(result["priority"]))
    missing_info_notice(result["missing_info"])
    probability_against_cutoffs(result)

    st.markdown("#### 🔍 模型解读 (SHAP)")
    st.info("单样本 SHAP 瀑布图待评估模块 `src.explain`(C)就绪后接入,目前暂不提供。")

    st.divider()
    show_model_info()


if __name__ == "__main__":
    main()
