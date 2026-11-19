"""Streamlit workbench for the road collision triage API."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SCHEMA = json.loads((APP_DIR.parent / "schema.json").read_text(encoding="utf-8"))
CHOICES = json.loads((APP_DIR / "frontend_choices.json").read_text(encoding="utf-8"))
ZH = json.loads((APP_DIR / "frontend_zh.json").read_text(encoding="utf-8"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/").removesuffix("/predict")
API_DOCS_URL = os.getenv("API_DOCS_URL") or f"{API_URL}/docs"

LEVELS = {1: "Immediate", 2: "Urgent", 3: "Routine"}
ZH_LEVELS = {1: "立即", 2: "紧急", 3: "常规"}
NUMERIC_LABELS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "speed_limit": "Speed limit",
    "number_of_vehicles": "Vehicle count",
    "temp": "Temperature (°C)",
    "rhum": "Relative humidity (%)",
    "prcp": "Hourly rainfall (mm)",
    "wdir": "Wind direction (°)",
    "wspd": "Wind speed (km/h)",
    "pres": "Sea-level pressure (hPa)",
}
ROAD_LABELS = {
    "road_type": "Road type",
    "junction_detail": "Junction type",
    "junction_control": "Junction control",
    "pedestrian_crossing": "Pedestrian crossing",
    "weather_conditions": "Weather at scene",
    "road_surface_conditions": "Road surface",
    "urban_or_rural_area": "Area type",
    "first_road_class": "Primary road class",
    "second_road_class": "Intersecting road class",
    "light_conditions": "Lighting conditions",
}
ROAD_SELECT_FIELDS = ("speed_limit", *ROAD_LABELS)
WEATHER_FIELDS = ("temp", "rhum", "prcp", "wdir", "wspd", "pres")
PARTICIPANT_LABELS = {
    "vehicle": {
        "vehicle_type": "Vehicle type",
        "vehicle_manoeuvre": "Manoeuvre",
        "first_point_of_impact": "First point of impact",
        "skidding_and_overturning": "Skidding or overturning",
        "sex_of_driver": "Driver sex",
        "age_of_driver": "Driver age",
        "engine_capacity_cc": "Engine capacity (cc)",
        "towing_and_articulation": "Towing or articulation",
        "propulsion_code": "Propulsion",
        "journey_purpose_of_driver": "Journey purpose",
        "escooter_flag": "E-scooter",
        "vehicle_left_hand_drive": "Left-hand drive",
        "junction_location": "Position at junction",
        "hit_object_in_carriageway": "Object struck on road",
        "vehicle_leaving_carriageway": "Departure from road",
        "hit_object_off_carriageway": "Object struck off road",
        "vehicle_location_restricted_lane": "Lane or road position",
        "driver_imd_decile": "Driver IMD decile",
    },
    "casualty": {
        "casualty_class": "Role",
        "casualty_type": "Road user type",
        "sex_of_casualty": "Sex",
        "age_of_casualty": "Age",
        "pedestrian_location": "Pedestrian location",
        "pedestrian_movement": "Pedestrian movement",
        "car_passenger": "Car passenger position",
        "bus_or_coach_passenger": "Bus or coach passenger",
        "casualty_imd_decile": "Casualty IMD decile",
    },
}
FIELD_LABELS = {
    "collision_datetime_utc": "Collision time",
    "vehicles": "Vehicles",
    "casualties": "Casualties",
    **NUMERIC_LABELS,
    **ROAD_LABELS,
    **PARTICIPANT_LABELS["vehicle"],
    **PARTICIPANT_LABELS["casualty"],
}
INFO_GROUP_LABELS = {
    "weather": ("weather observations", "天气观测"),
    "road": ("road conditions", "道路环境"),
    "vehicle": ("vehicle records", "车辆记录"),
    "casualty": ("casualty records", "伤亡人员记录"),
}
ACCEPTANCE_ROWS = (
    (("Life-threatening cases classed as routine", "危及生命事故进入三级"),
     "undertriage_life", "max_undertriage_life", "≤", "{:.2%}"),
    (("Fatal cases classed as routine", "致死事故进入三级"),
     "undertriage_fatal", "max_undertriage_fatal", "≤", "{:.2%}"),
    (("Share classed as routine", "三级占全部事故"),
     "routine_share", "min_routine_share", "≥", "{:.2%}"),
    (("Life-threatening case lift in Priority 1", "一级危及生命事故富集倍数"),
     "immediate_lift", "min_immediate_life_lift", "≥", "{:.2f}×"),
)


def tr(english: str, chinese: str) -> str:
    return chinese if st.session_state.get("language") == "zh" else english


def field_label(name: str) -> str:
    return tr(FIELD_LABELS.get(name, name), ZH["fields"].get(name, name))


@st.cache_data(ttl=60, show_spinner=False)
def call_model_info(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url}/model-info", timeout=(3, 5))
    response.raise_for_status()
    return response.json()


def backend_info() -> dict[str, Any] | None:
    try:
        return call_model_info(API_URL)
    except (requests.RequestException, ValueError):
        return None


def input_number(name: str, *, precision: int = 1, integer: bool = False) -> float | int | None:
    spec = SCHEMA["numeric_features"][name]
    convert = int if integer else float
    bounds = {f"{side}_value": convert(spec[side]) for side in ("min", "max") if side in spec}
    return st.number_input(
        field_label(name),
        value=None,
        step=1 if integer else 10 ** -precision,
        format="%d" if integer else f"%.{precision}f",
        key=name,
        placeholder=tr("Required", "必填") if name in {"latitude", "longitude"} else tr("Unknown", "未知"),
        **bounds,
    )


def choice_key(name: str, language: str) -> str:
    return f"{name}_zh" if language == "zh" else name


def select_road_value(name: str, labels: dict) -> Any:
    language = tr("en", "zh")
    key = choice_key(name, language)
    if key != name and key not in st.session_state and name in st.session_state:
        st.session_state[key] = st.session_state[name]
    unknown = tr("Unknown", "未知")
    return st.selectbox(
        field_label(name),
        [None, *labels],
        format_func=lambda code: unknown if code is None else labels[code],
        key=key,
        placeholder=tr("Choose an option", "请选择"),
    )


def input_category(name: str) -> str | None:
    options = SCHEMA["categorical_features"][name]["categories"]
    labels = ZH["road"][name] if st.session_state.get("language") == "zh" else options
    return select_road_value(name, labels)


def participant_editor(table: str) -> list[dict[str, Any]]:
    spec = SCHEMA["participants"][table]
    labels = PARTICIPANT_LABELS[table]
    choices = ZH[table] if st.session_state.get("language") == "zh" else CHOICES[table]
    columns = {}
    for name in labels:
        label = field_label(name)
        if name in spec["stats"]:
            columns[name] = st.column_config.NumberColumn(label, min_value=0, step=1, format="%d")
        else:
            columns[name] = st.column_config.SelectboxColumn(
                label,
                options=spec["counts"][name],
                format_func=choices[name].__getitem__,
                width="medium",
            )

    rows = st.session_state.get("participant_rows", {}).get(table, [])
    dtypes = {name: "float64" if name in spec["stats"] else "string" for name in labels}
    edited = st.data_editor(
        pd.DataFrame.from_records(rows, columns=list(labels)).astype(dtypes),
        column_config=columns,
        num_rows="dynamic",
        hide_index=True,
        key=f"{table}_editor_{st.session_state.get('editor_revision', 0)}",
        width="stretch",
        height=160,
    )
    return filled_records(edited)


def filled_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Keep populated participant rows and omit empty cells from the API payload."""
    records = [
        {name: value for name, value in row.items() if pd.notna(value)}
        for row in table.to_dict("records")
    ]
    return [record for record in records if record]


def fill_example() -> None:
    example = CHOICES["example"].copy()
    event = datetime.fromisoformat(example.pop("collision_datetime_utc"))
    language = tr("en", "zh")
    for name in ROAD_SELECT_FIELDS:
        example[choice_key(name, language)] = example.pop(name)
    st.session_state.update(example, event_date=event.date(), event_time=event.strftime("%H:%M"))
    st.session_state["editor_revision"] = st.session_state.get("editor_revision", 0) + 1
    st.session_state.pop("participant_rows", None)
    st.session_state.pop("prediction", None)


def switch_language(fields: dict[str, Any]) -> None:
    next_language = tr("zh", "en")
    # Select boxes keep the rendered selection in the browser. Give each
    # locale its own widget and carry over the original values, not the labels.
    for name in ROAD_SELECT_FIELDS:
        st.session_state[choice_key(name, next_language)] = fields[name]
    # The editor's identity depends on translated column metadata. Seed its new
    # instance with the submitted rows from the current language.
    st.session_state["participant_rows"] = {
        "vehicle": fields["vehicles"] or [],
        "casualty": fields["casualties"] or [],
    }
    st.session_state["editor_revision"] = st.session_state.get("editor_revision", 0) + 1
    st.session_state["language"] = next_language
    st.rerun()


def intake_fields() -> dict[str, Any]:
    date_col, time_col = st.columns(2)
    with date_col:
        event_date = st.date_input(
            tr("Date", "事故日期"), value=None, format="YYYY-MM-DD", key="event_date",
        )
    with time_col:
        event_time = st.text_input(
            tr("Time (UTC)", "事故时间（UTC）"), placeholder=tr("e.g. 08:30", "例如 08:30"),
            max_chars=5, key="event_time",
        )
    north, east = st.columns(2)
    with north:
        latitude = input_number("latitude", precision=6)
    with east:
        longitude = input_number("longitude", precision=6)

    values = {"latitude": latitude, "longitude": longitude}
    st.markdown(f"#### {tr('Optional scene details', '选填现场信息')}")
    with st.expander(tr("Road & environment", "道路与环境"), icon=":material/add_road:"):
        left, right = st.columns(2)
        with left:
            limits = SCHEMA["numeric_features"]["speed_limit"]["allowed"]
            unit = tr("mph", "英里/小时")
            values["speed_limit"] = select_road_value("speed_limit", {limit: f"{limit} {unit}" for limit in limits})
        with right:
            values["number_of_vehicles"] = input_number("number_of_vehicles", integer=True)
        for index, name in enumerate(ROAD_LABELS):
            with (left, right)[index % 2]:
                values[name] = input_category(name)

    with st.expander(tr("Weather observations", "天气观测"), icon=":material/cloud:"):
        st.caption(tr("Use observations from within one hour of the collision.", "使用事故前后一小时内的观测值。"))
        left, right = st.columns(2)
        for index, name in enumerate(WEATHER_FIELDS):
            with (left, right)[index % 2]:
                values[name] = input_number(name)

    with st.expander(tr("Vehicles", "涉事车辆"), icon=":material/directions_car:"):
        st.caption(tr("Add one row per vehicle. Scroll horizontally for more fields.", "每辆车填写一行，横向滚动查看其余字段。"))
        values["vehicles"] = participant_editor("vehicle") or None

    with st.expander(tr("Casualties", "伤亡人员"), icon=":material/group:"):
        st.caption(tr("Add one row per person. Scroll horizontally for more fields.", "每人填写一行，横向滚动查看其余字段。"))
        values["casualties"] = participant_editor("casualty") or None

    return {"event_date": event_date, "event_time": event_time, **values}


def build_payload(fields: dict[str, Any]) -> dict[str, Any]:
    required = {
        "event_date": tr("date", "事故日期"),
        "event_time": tr("UTC time", "UTC 时间"),
        "latitude": tr("latitude", "纬度"),
        "longitude": tr("longitude", "经度"),
    }
    missing = [label for key, label in required.items() if fields[key] in (None, "")]
    if missing:
        raise ValueError(tr(f"Required fields: {', '.join(missing)}.", f"请填写{'、'.join(missing)}。"))
    try:
        clock = datetime.strptime(fields["event_time"], "%H:%M").time()
    except ValueError:
        raise ValueError(tr("Enter a valid UTC time, for example 08:30.", "请输入有效的 UTC 时间，例如 08:30。")) from None

    return {
        "collision_datetime_utc": datetime.combine(fields["event_date"], clock, UTC).isoformat(),
        **{key: value for key, value in fields.items()
           if key not in ("event_date", "event_time") and value is not None},
    }


def service_error(response: requests.Response, body: Any) -> str:
    if response.status_code == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        messages = []
        for error in body["detail"]:
            field = " · ".join(
                str(part + 1) if isinstance(part, int) else field_label(str(part))
                for part in error["loc"][1:]
            )
            messages.append(f"{field} · {error['msg']}" if field else error["msg"])
        return "\n\n".join(messages)
    return tr(
        f"The prediction API returned HTTP {response.status_code}. Check the service and retry.",
        f"预测 API 返回 HTTP {response.status_code}。请检查服务状态后重试。",
    )


def submit_prediction(fields: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = build_payload(fields)
    except ValueError as error:
        return None, str(error)

    try:
        response = requests.post(
            f"{API_URL}/predict", params={"explain": "true"}, json=payload, timeout=(5, 60)
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        if not response.ok:
            return None, service_error(response, body)
        if not isinstance(body, dict) or body.get("priority") not in LEVELS:
            return None, tr("The prediction API returned an unrecognised result.", "预测 API 返回了无法识别的结果。")
        return body, None
    except requests.Timeout:
        return None, tr("The prediction timed out. Please retry.", "预测超时，请重试。")
    except requests.RequestException:
        return None, tr("Could not reach the prediction API.", "无法连接预测 API。")


def page_header(info: dict[str, Any] | None) -> None:
    connected = info is not None
    status = tr("API ONLINE", "API 在线") if connected else tr("API OFFLINE", "API 离线")
    version = info["model_version"] if connected else tr("Model unavailable", "模型暂不可用")
    docs = escape(API_DOCS_URL, quote=True)
    st.html(f"""
        <div class="workbench-bar">
          <span class="workbench-brand"><i aria-hidden="true"></i> STATS19 / TRIAGE</span>
          <div class="workbench-tools">
            <span class="workbench-status {'is-online' if connected else 'is-offline'}">{status}</span>
            <a href="{docs}" target="_blank" rel="noopener noreferrer">{tr('API Docs', 'API 文档')} <span aria-hidden="true">↗</span></a>
          </div>
        </div>
        <header class="workbench-header">
          <div class="workbench-copy">
            <p class="workbench-region">{tr('GREAT BRITAIN', '大不列颠道路')}</p>
            <h1>{tr('Collision <em>Triage</em>', '道路事故<em>分诊</em>')}</h1>
            <p class="workbench-subtitle">{tr(
                'Assess a road collision and see what drives its dispatch priority.',
                '评估道路事故，查看派单优先级和影响结果的因素。'
            )}</p>
            <span class="workbench-version">{escape(version)}</span>
          </div>
          <div class="junction-plot" aria-hidden="true">
            <span class="junction-spectrum"></span>
            <span class="junction-facets"><i></i><i></i><i></i><i></i></span>
            <span class="junction-road road-north"></span>
            <span class="junction-road road-east"></span>
            <span class="junction-road road-ramp"></span>
            <span class="junction-crosshair"><i></i></span>
            <span class="junction-dot plot-dot-a"></span>
            <span class="junction-dot plot-dot-b"></span>
            <span class="junction-scale">+ &nbsp; · &nbsp; · &nbsp; · &nbsp; +</span>
          </div>
        </header>
    """)


def score_scale(result: dict[str, Any]) -> None:
    score = result["probability"]
    lower = result["cutoffs"]["routine_urgent"]
    upper = result["cutoffs"]["urgent_immediate"]
    aria_label = tr(
        f"Score {score:.1%}, urgent threshold {lower:.1%}, immediate threshold {upper:.1%}",
        f"分数 {score:.1%}，紧急界线 {lower:.1%}，立即处置界线 {upper:.1%}",
    )
    st.html(f"""
        <div class="score-block">
          <div class="score-heading"><span>{tr('Severity score', '严重度分数')}</span><strong>{score:.1%}</strong></div>
          <div class="score-track" role="img" aria-label="{aria_label}" style="--lower:{lower:.3%};--upper:{upper:.3%}">
            <span class="score-marker" style="left:{min(100, max(0, score * 100)):.3f}%"></span>
          </div>
          <div class="score-ticks"><span>0%</span><span>100%</span></div>
          <div class="score-cutoffs">
            <span>{tr('Urgent threshold', '二级界线')} <b>{lower:.1%}</b></span>
            <span>{tr('Immediate threshold', '一级界线')} <b>{upper:.1%}</b></span>
          </div>
        </div>
    """)


def show_explanation(result: dict[str, Any]) -> None:
    with st.expander(
        tr("Prediction drivers · SHAP", "分诊影响因素 · SHAP"),
        expanded=True,
        key=f"prediction_drivers_{result['request_id']}",
        icon=":material/bar_chart:",
    ):
        explanation = result.get("explanation")
        if explanation is None:
            st.caption(tr("Feature contributions are unavailable for this prediction.", "此预测暂无特征贡献。"))
            return

        base = explanation["base_value"]
        position = base
        rows = []
        raises = tr("Raises score", "推高")
        lowers = tr("Lowers score", "降低")
        items = [*explanation["contributions"], {
            "feature": "other_features", "value": "—", "impact": explanation["other_impact"],
        }]
        for order, item in enumerate(items):
            field = item["feature"]
            impact = item["impact"]
            rows.append({
                "Order": order, "Feature": (
                    tr("Other features", "其余特征") if field == "other_features" else field_label(field)
                ), "Model field": field,
                "Input": item["value"], "Contribution": impact,
                "Start": position, "End": position + impact,
                "Direction": raises if impact >= 0 else lowers,
            })
            position += impact

        st.caption(tr(
            f"Baseline {base:+.3f} → prediction {position:+.3f} log-odds.",
            f"基准 {base:+.3f} → 预测 {position:+.3f} log-odds。",
        ))
        st.vega_lite_chart(
            pd.DataFrame(rows),
            {
                "description": tr("SHAP waterfall for this prediction", "本次预测的 SHAP 瀑布图"),
                "height": {"step": 24},
                "mark": {"type": "bar", "size": 12, "cornerRadius": 3},
                "encoding": {
                    "y": {"field": "Feature", "type": "nominal", "title": None,
                          "sort": {"field": "Order", "op": "min"}, "axis": {"labelLimit": 128}},
                    "x": {"field": "Start", "type": "quantitative",
                          "title": tr("Cumulative log-odds", "累计 log-odds"),
                          "scale": {"zero": False}},
                    "x2": {"field": "End"},
                    "color": {"field": "Direction", "type": "nominal", "title": None,
                              "scale": {"domain": [raises, lowers], "range": ["#ff8199", "#7db8ff"]},
                              "legend": {"orient": "bottom"}},
                    "tooltip": [
                        {"field": "Model field", "type": "nominal", "title": tr("Model field", "模型字段")},
                        {"field": "Input", "type": "nominal", "title": tr("Input", "输入")},
                        {"field": "Contribution", "type": "quantitative",
                         "title": tr("Contribution", "贡献"), "format": "+.4f"},
                    ],
                },
            },
            width="stretch",
        )


def show_result(result: dict[str, Any], info: dict[str, Any] | None) -> None:
    priority = result["priority"]
    st.html(f"""
        <div class="result-banner priority-{priority}">
          <span class="priority-number" aria-hidden="true">{priority}</span>
          <div><span class="result-label">{tr(f'Dispatch priority {priority}', f'派单优先级 {priority} 级')}</span>
          <strong>{tr(LEVELS[priority], ZH_LEVELS[priority])}</strong></div>
        </div>
    """)
    score_scale(result)
    missing = result["missing_info"]
    if missing:
        names = tr(", ", "、").join(tr(*INFO_GROUP_LABELS.get(group, (group, group))) for group in missing)
        st.warning(
            tr(f"Missing {names}. Incomplete scene details can reduce triage accuracy.",
               f"缺少{names}。现场信息不完整可能降低分诊准确性。"),
            icon=":material/warning:",
        )
    if ((info and info["is_placeholder"] and result["model_version"] == info["model_version"])
            or result["model_version"] == "dummy-v1"):
        st.warning(tr("Use demo model results for integration testing only.", "演示模型结果仅用于接口联调。"))
    show_explanation(result)
    with st.expander(
        tr("Request details", "请求详情"),
        expanded=True,
        key=f"request_details_{result['request_id']}",
        icon=":material/receipt_long:",
    ):
        st.text(f"{tr('Model version', '模型版本')}  {result['model_version']}")
        st.text(f"{tr('Inference time', '推理耗时')}  {result['latency_ms']:g} ms")
        st.text(f"{tr('Request ID', '请求标识')}  {result['request_id']}")


def show_empty_result() -> None:
    st.html(f"""
        <div class="result-empty">
          <div class="result-empty-mark" aria-hidden="true"><i></i></div>
          <strong>{tr('Awaiting assessment', '等待分诊')}</strong>
          <p>{tr('Enter the collision time and coordinates to get a dispatch priority.', '填写事故时间和坐标后即可提交分诊。')}</p>
        </div>
        <div class="priority-reference" aria-label="{tr('Dispatch priorities', '派单优先级')}">
          <span><i class="level-1"></i>1 · {tr(LEVELS[1], ZH_LEVELS[1])}</span>
          <span><i class="level-2"></i>2 · {tr(LEVELS[2], ZH_LEVELS[2])}</span>
          <span><i class="level-3"></i>3 · {tr(LEVELS[3], ZH_LEVELS[3])}</span>
        </div>
    """)


def show_model_info(info: dict[str, Any] | None) -> None:
    with st.expander(tr("Model validation & scope", "模型验收与适用范围"), icon=":material/fact_check:"):
        if info is None:
            st.caption(tr("Model validation data is currently unavailable.", "暂时无法获取模型验收记录。"))
            return
        cutoffs = info["cutoffs"]
        st.caption(
            f"{info['model_version']} · {tr('Urgent threshold', '二级界线')} {cutoffs['routine_urgent']:.1%} "
            f"· {tr('Immediate threshold', '一级界线')} {cutoffs['urgent_immediate']:.1%}"
        )
        metrics, limits = info["risk_metrics"], info["acceptance_limits"]
        if info["is_placeholder"] or metrics["status"] == "not_evaluated":
            st.caption(tr(f"No test-set validation data for {info['model_version']}.",
                          f"{info['model_version']} 暂无测试集验收记录。"))
            return
        rows = []
        for label, key, limit_key, direction, fmt in ACCEPTANCE_ROWS:
            value, limit = metrics[key], limits[limit_key]
            passed = value <= limit if direction == "≤" else value >= limit
            rows.append((tr(*label), fmt.format(value), f"{direction} {fmt.format(limit)}",
                         tr("Pass", "达标") if passed else tr("Fail", "未达标")))
        st.dataframe(
            pd.DataFrame(rows, columns=[
                tr("Metric", "指标"), tr("Test set", "测试集"),
                tr("Requirement", "要求"), tr("Result", "结果"),
            ]),
            hide_index=True,
            width="stretch",
        )
        st.caption(tr("Validation covers complete scene records. Dispatchers retain the final decision.",
                      "验收基于现场信息完整的记录。最终派单仍由调度员决定。"))


def main() -> None:
    st.set_page_config(page_title=tr("Collision Triage", "道路事故分诊"),
                       page_icon=":material/monitor_heart:", layout="wide")
    stylesheet = (APP_DIR / "frontend.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
    info = backend_info()
    page_header(info)

    intake, output = st.columns([1.15, 1], gap="medium")
    with intake, st.container(key="intake", border=True):
        with st.form("triage", border=False, enter_to_submit=False):
            with st.container(key="intake_heading", horizontal=True,
                              horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown(f"## {tr('Collision details', '事故信息')}", width="content")
                with st.container(horizontal=True, horizontal_alignment="right", width="content"):
                    st.form_submit_button(
                        tr("Load example", "载入示例"), key="load_example",
                        on_click=fill_example, icon=":material/edit_note:",
                        help=tr("Replace the form with a cleaned dataset record.", "用清洗后的事故记录填充表单。"),
                    )
                    change_language = st.form_submit_button(
                        tr("中文", "English"), key="switch_language", icon=":material/language:",
                        help=tr("Switch to Chinese", "切换至英文"),
                    )
            st.caption(tr("Date, UTC time and coordinates are required.", "事故日期、UTC 时间和经纬度必填。"))
            fields = intake_fields()
            with st.container(key="assess_action"):
                submitted = st.form_submit_button(
                    tr("Assess collision", "生成分诊结果"), key="run_assessment",
                    type="primary", icon=":material/arrow_forward:", width="stretch",
                )
        if change_language:
            switch_language(fields)
        if submitted:
            st.session_state.pop("prediction", None)
            with st.spinner(tr("Assessing collision…", "正在生成分诊结果…")):
                prediction, error = submit_prediction(fields)
            if error:
                st.error(error)
            else:
                st.session_state.prediction = prediction

    with output, st.container(key="output", border=True):
        st.markdown(f"## {tr('Dispatch assessment', '分诊结果')}")
        result = st.session_state.get("prediction")
        if result:
            show_result(result, info)
        else:
            show_empty_result()

    with st.container(key="model_details", border=True):
        show_model_info(info)


if __name__ == "__main__":
    main()
