"""Build the Chinese analysis report directly from Python pipeline outputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(output=None, report_date="2026-11-10"):
    data = ROOT / "data"
    quality = load(data / "quality_report.json")
    splitting = load(data / "split_report.json")
    preprocessing = load(data / "preprocessing_report.json")
    insights = load(data / "eda_figures/insights.json")
    distances, offsets = [], []
    severe, providers = 0, Counter()
    with (data / "processed/collisions_clean.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if row["model_eligible"] != "True":
                continue
            distances.append(float(row["weather_distance_km"]))
            offsets.append(abs(float(row["weather_time_offset_minutes"])))
            severe += int(float(row["is_severe"]))
            providers[row["weather_provider"]] += 1
    distances.sort()
    matched = quality["matching"]
    total = quality["collisions"]["raw_rows"]
    eligible = matched["model_eligible"]
    used_rows = sum(item["rows"] for item in splitting["splits"].values())
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.85)
    for name, size in [
        ("Normal", 11),
        ("Title", 23),
        ("Subtitle", 11),
        ("Heading 1", 16),
        ("Heading 2", 12.5),
        ("Caption", 10),
    ]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.underline = False
        if name in ("Subtitle", "Caption"):
            style.font.bold = False
            style.font.italic = False
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.15
    doc.styles["Normal"].paragraph_format.widow_control = True
    for name in ["Heading 1", "Heading 2"]:
        doc.styles[name].paragraph_format.keep_with_next = True
    header = section.header.paragraphs[0]
    header.text = "英国道路事故数据分析"
    header.style = doc.styles["Normal"]
    for run in header.runs:
        run.font.size = Pt(9)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("第 ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    footer.add_run(" 页")

    def paragraph(text, style=None):
        p = doc.add_paragraph(text, style=style)
        return p

    def heading(text, level=1):
        return doc.add_heading(text, level=level)

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, (title, width) in enumerate(zip(headers, widths)):
            t.columns[i].width = Inches(width)
            t.rows[0].cells[i].text = str(title)
        for row in rows:
            cells = t.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = str(value)
        for row_index, row in enumerate(t.rows):
            for column_index, cell in enumerate(row.cells):
                cell.width = Inches(widths[column_index])
                cell.vertical_alignment = 1
                properties = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                for edge in ["top", "left", "bottom", "right"]:
                    element = OxmlElement("w:" + edge)
                    element.set(qn("w:val"), "single")
                    element.set(qn("w:sz"), "4")
                    element.set(qn("w:color"), "D9D9D9")
                    borders.append(element)
                properties.append(borders)
                margins = OxmlElement("w:tcMar")
                for edge, amount in [
                    ("top", 90),
                    ("bottom", 90),
                    ("left", 100),
                    ("right", 100),
                ]:
                    element = OxmlElement("w:" + edge)
                    element.set(qn("w:w"), str(amount))
                    element.set(qn("w:type"), "dxa")
                    margins.append(element)
                properties.append(margins)
                if row_index == 0 or row_index % 2 == 0:
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:fill"), "E8EFF4" if row_index == 0 else "F7F9FB")
                    properties.append(shading)
                for p in cell.paragraphs:
                    p.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT
                        if column_index == 0
                        else WD_ALIGN_PARAGRAPH.CENTER
                    )
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.12
                    for run in p.runs:
                        run.font.size = Pt(10)
                        run.bold = row_index == 0
        repeat = OxmlElement("w:tblHeader")
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        paragraph("").paragraph_format.space_after = Pt(0)
        return t

    paragraph("2024 至 2025 年英国道路事故数据分析报告", "Title")
    display_date = report_date.replace("-", " ")
    paragraph(
        f"任务 A 数据获取 清洗 特征工程与探索分析    报告日期 {display_date}",
        "Subtitle",
    )
    paragraph(
        f"本报告使用英国交通部的 {total:,} 条伤人事故记录，以及按事故所需站点小时块精简后的 369,383 条天气记录。六项天气均有观测来源的事故为 {matched['observed_core6_complete']:,} 条，其中 {eligible:,} 条能够核对原值质控或实测推导输入质控，作为主建模候选样本。原始文件保持不变，清洗、特征、校验、划分和图表均由 Python 生成。"
    )
    heading("一 数据范围与预测目标")
    paragraph(
        "预测目标为已记录伤人事故是否造成致死或重伤。原始 collision_severity 编码 1 或 2 映射为 is_severe 等于 1，编码 3 映射为 0。数据不含未发生事故的道路时段，因此本模块不估计事故是否发生，也不把严重事故比例解释为道路事故发生概率。"
    )
    table(
        ["数据层次", "数量", "说明"],
        [
            ["2024 年事故原件", "100,927", "44 列"],
            ["2025 年事故原件", "101,525", "44 列"],
            [
                "六项观测天气齐全",
                f"{matched['observed_core6_complete']:,}",
                f"占全部事故 {matched['observed_core6_complete'] / total:.2%}",
            ],
            ["主建模候选", f"{eligible:,}", f"占全部事故 {eligible / total:.2%}"],
            [
                "边界空档后实际划分",
                f"{used_rows:,}",
                f"另有 {splitting['boundary_gap_rows']:,} 条边界记录暂不使用",
            ],
        ],
        [2.35, 1.15, 3.2],
    )
    paragraph(
        f"全体原始事故中，致死事故 2,955 条、重伤事故 48,758 条、轻伤事故 150,739 条，致死或重伤合计比例为 {51713 / total:.2%}。主建模候选中该比例为 {severe / eligible:.2%}，两者的差异说明天气完整性筛选改变了样本构成，不能默认子样本代表全部英国事故。"
    )
    table(
        ["天气来源", "精简后记录", "主建模候选实际采用"],
        [
            ["NOAA GHCNh", "336,585", f"{providers.get('NOAA_GHCNh', 0):,} 次事故"],
            ["COSMOS UK", "19,204", f"{providers.get('COSMOS_UK', 0):,} 次事故"],
            ["Meteostat", "13,594", f"{providers.get('Meteostat', 0):,} 次事故"],
        ],
        [2.2, 1.5, 3.0],
    )
    paragraph(
        "天气行数是测站观测数量，同一条观测可对应多次事故。不同来源之间也可能覆盖相同地点和时刻，不能把天气行数当作独立事故样本数。"
    )

    doc.add_page_break()
    heading("二 清洗规则与质量控制")
    paragraph(
        "事故编号按文本读取，先核对完全重复行与冲突编号。当前事故编号无重复。未知分类码和超出字段约定的数值转为空值，缺失单元格不直接导致整行删除。事故原件始终留存，不能将缺失码负一解释为真实的速度或天气数值。非交叉口的控制类型按官方不适用编码处理。"
    )
    unknown = quality["collisions"]["invalid_or_unknown_cells_set_missing"]
    top_unknown = sorted(unknown.items(), key=lambda item: item[1], reverse=True)[:6]
    labels = {
        "junction_detail": "交叉口类型",
        "junction_control": "交叉口控制",
        "road_type": "道路类型",
        "weather_conditions": "事故记录天气",
        "road_surface_conditions": "路面状态",
        "pedestrian_crossing": "行人过街设施",
        "second_road_class": "第二道路等级",
        "light_conditions": "现场照明",
        "urban_or_rural_area": "城乡类别",
    }
    table(
        ["字段", "处理为空的单元格", "后续处理"],
        [[labels.get(k, k), f"{v:,}", "训练集拟合缺失处理"] for k, v in top_unknown],
        [2.4, 1.65, 2.65],
    )
    paragraph(
        "事故本地时间使用 Europe/London 时区换算，保留英国夏令时规则。18 条重复夏令时小时不能唯一确定 UTC，另有 2 条事故缺失坐标；不猜测时间或位置。它们保留在清洗主表和排除清单中，未进入主模型。"
    )
    paragraph(
        "天气逐字段核对来源和标记。Meteostat 只接受明确的观测提供方，不接受模型来源值；其小时文件不暴露逐值质控，故不进入本版严格质控主样本。NOAA 使用合格原值质控；原库推导的相对湿度另核对气温和露点输入质控。降水按测量码和报告类型判断，不能将 METAR 的运行累计直接当作合格一小时降水。COSMOS 排除补值和估算标记，只累计同一雨量计连续两段合格半小时输入。站点气压保持独立，不充当海平面气压。"
    )
    reasons = matched["exclusion_reasons"]
    table(
        ["未进入主建模候选的原因", "事故数"],
        [
            ["六项观测天气仍不完整", f"{reasons.get('core_weather_incomplete', 0):,}"],
            [
                "六项齐全但原逐值质控未暴露",
                f"{reasons.get('weather_qc_not_exposed', 0):,}",
            ],
            ["本地时间无法唯一换算", f"{reasons.get('unresolved_local_time', 0):,}"],
            ["坐标缺失或不可用", f"{reasons.get('invalid_coordinates', 0):,}"],
        ],
        [4.8, 1.9],
    )
    paragraph(
        f"时空匹配上限为 40 公里和前后各 60 分钟，每个事故的核心天气使用同一站点同一条记录。主样本测站距离中位数为 {median(distances):.2f} 公里，90% 分位数为 {distances[int((len(distances) - 1) * 0.9)]:.2f} 公里；绝对时间差中位数为 {median(offsets):.1f} 分钟。观测值真实且可追溯，并不代表它完全等于事故现场的微天气。"
    )

    doc.add_page_break()
    heading("三 时间划分与特征接口")
    paragraph(
        "按完整 UTC 日期进行时间顺序划分，目标比例约为 70%、15%、15%。训练结束与验证开始、验证结束与测试开始之间各留前后 60 分钟空档。该间隔防止事故前后天气窗口导致同一条测站记录跨集合复用。实际样本比例随整日边界略有变化，未使用随机打乱来消除未来时间顺序。"
    )
    table(
        ["集合", "样本数", "致死或重伤", "开始日期", "结束日期"],
        [
            [
                name,
                f"{item['rows']:,}",
                f"{item['positive_rate']:.2%}",
                item["start_utc"][:10],
                item["end_utc"][:10],
            ]
            for name, item in splitting["splits"].items()
        ],
        [0.75, 1.0, 1.25, 1.85, 1.85],
    )
    paragraph(
        f"两条 UTC 分界线为 {splitting['boundaries_utc'][0][:10]} 和 {splitting['boundaries_utc'][1][:10]}，边界空档保留 {splitting['boundary_gap_rows']:,} 条记录供审计。本次核对事故编号和天气原始记录编号均未跨集合重叠。验证集用于调参和阈值选择，测试集留作最终一次评估；下文严重程度探索只使用训练集。"
    )
    table(
        ["特征组", "设计"],
        [
            [
                "道路和场景",
                "坐标、限速 mph、车辆数、道路类别、交叉口、过街设施、照明、事故记录天气及路面类别",
            ],
            ["测站天气", "气温、湿度、一小时降水、风向、风速 km/h、海平面气压 hPa"],
            [
                "时间与天文",
                "小时、星期和月份的正余弦编码，周末标记，太阳几何高度与曙暮光阶段",
            ],
            ["其他派生", "是否降水，按气象来向定义计算的东西和南北风分量"],
        ],
        [1.5, 5.2],
    )
    paragraph(
        "详细日出日落与三种曙暮光时刻由 Astral 3.2 在每次事故的经纬度和日期计算。地平线按标准无遮挡情形、观测高度按零米；高纬夏季太阳未穿过对应角度时保持空事件。太阳角度属于计算特征，不能冒充气象站实测。"
    )
    paragraph(
        f"src/features.py 提供 load_splits 和 build_pipeline 两个统一接口。Pipeline 内先生成特征，再由 ColumnTransformer 对数值列做训练集中位数填补和标准化，对分类列填充 unknown 并独热编码。验证和测试仅 transform，未知类别不会报错。本次编码后共 {preprocessing['encoded_feature_count']:,} 个特征。训练统计量在验证和测试变换后保持不变，模型矩阵全部有限。"
    )
    paragraph(
        "模型输入采用允许名单，排除严重程度及调整字段、伤亡人数、事故编号、警察到场情况和天气匹配质量字段。B 同学只需传入分类器并拟合训练集；C、D 复用保存的完整 Pipeline，避免各自复制预处理。任何交叉验证或重采样也应限制在训练集及其相应折内。"
    )

    hourly = [item for item in insights["hourly"] if item["size"] >= 200]
    hour_peak = max(hourly, key=lambda item: item["mean"])
    speed = [item for item in insights["speed_limit"] if item["size"] >= 200]
    speed_peak = max(speed, key=lambda item: item["mean"])
    missing = max(insights["missingness"].items(), key=lambda item: item[1])
    dry = next(item for item in insights["rainfall"] if str(item["rain_group"]) == "0")
    phases = {item["solar_phase"]: item for item in insights["solar_phase"]}
    chart_comments = [
        f"训练集包含 {insights['rows']:,} 条事故，致死或重伤比例为 {insights['positive_rate']:.2%}。类别不平衡需要在 B 的训练评估中处理，不能用只预测轻伤的总体准确率代替对严重类别的识别能力。",
        "事故数量与严重比例分开绘制，避免混淆频次和构成。训练集末月可能不是完整月份，月数量不能直接解释为道路暴露量或风险趋势。",
        f"在至少 200 条样本的小时组中，当地 {int(hour_peak['hour'])} 时的严重比例最高，为 {hour_peak['mean']:.2%}，对应 {int(hour_peak['size']):,} 条记录。夜间样本数量和区间宽度应一起判断，不作因果解释。",
        f"在至少 200 条样本的限速组中，{int(speed_peak['speed_limit'])} mph 组的严重比例最高，为 {speed_peak['mean']:.2%}。该关系同时可能受道路类型、交通参与者和测站覆盖影响，不能视为仅由限速造成。",
        f"填补前缺失比例最高的模型字段为 {missing[0]}，占 {missing[1]:.2%}。道路分类保留未知缺失，由训练集拟合处理；主建模样本六项天气完整，因此没有为天气列虚构测量值。",
        f"训练集无降水组的严重比例为 {dry['mean']:.2%}，样本数 {int(dry['size']):,}。降水分组的区间和样本量需一并阅读；箱线图只为可读性不显示离群点，没有按图形外观删值。",
        f"按太阳高度计算，白天组严重比例为 {phases['day']['mean']:.2%}，夜间组为 {phases['night']['mean']:.2%}。该分组使用事故坐标和时刻，能区分固定钟点法忽略的季节与纬度差异。",
    ]
    titles = [
        "训练集严重程度分布",
        "逐月事故数量与严重比例",
        "事故小时与严重比例",
        "限速分组与严重比例",
        "训练输入缺失情况",
        "气温降水与严重程度",
        "太阳高度与曙暮光阶段",
    ]
    for i, (figure, title, comment) in enumerate(
        zip(insights["figures"], titles, chart_comments)
    ):
        if i % 2 == 0:
            doc.add_page_break()
        heading(("四 探索性分析 " if i == 0 else "") + title, level=1 if i == 0 else 2)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(
            str(data / "eda_figures" / figure["file"]), width=Inches(6.0)
        )
        paragraph(f"图 {i + 1}  {figure['title']}。{comment}", style="Caption")
    heading("五 版本管理与交付验证", level=2)
    paragraph(
        "DVC 管道包含清洗、固定划分、训练集 EDA 和预处理契约检查，依赖原始 CSV、schema 与 Python 源文件，锁文件保存输出哈希。本机已配置本地备份；这不是团队共享云端。其他同学可从仓库原始 CSV 运行 python -m src.manage_data build 重建结果，读取固定分组后复用同一 Python 接口。"
    )
    paragraph(
        "已执行原件校验、pandera 主表与候选数据校验、时空边界与来源质控测试、时间划分复现和防泄漏测试。天气原值来自下载文件；原库湿度推导与本项目基于连续半小时观测的雨量累计均有明确规则。清洗没有添加预报、网格估计或人工编造天气。"
    )
    paragraph(
        "分析范围仍受测站与记录制度限制。前后各一小时窗口允许使用事故之后的观测，所以结论属于回顾性研究，不能直接宣传为实时预测。分组比例图的 Wilson 区间采用二项近似，没有校正同测站小时内的相关性，仅用于描述。特征相关关系不构成因果证据。"
    )
    heading("数据与方法来源", level=2)
    for text in [
        "英国交通部 Road safety open data  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data",
        "Meteostat 小时数据  https://dev.meteostat.net/data/timeseries/hourly",
        "NOAA GHCNh  https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly",
        "UKCEH COSMOS UK API  https://cosmos-api.ceh.ac.uk/",
        "Astral 3.2  https://sffjunkie.github.io/astral/package.html",
        "scikit learn 数据泄漏说明  https://scikit-learn.org/stable/common_pitfalls.html",
        "pandera  https://pandera.readthedocs.io/en/stable/dataframe_schemas.html",
        "DVC  https://doc.dvc.org/user-guide",
    ]:
        p = paragraph(text)
        for run in p.runs:
            run.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(3)
    output = Path(output) if output else ROOT / "artifacts/data_analysis_report.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "2024至2025年英国道路事故数据分析报告"
    doc.core_properties.subject = "任务A 数据清洗与特征工程"
    doc.core_properties.author = ""
    for xml_root in (doc.styles.element, doc.element):
        for border in list(xml_root.xpath(".//w:pBdr")):
            border.getparent().remove(border)
    doc.save(output)
    print(f"REPORT_CREATED {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-date", default="2026-11-10")
    arguments = parser.parse_args()
    build_report(arguments.output, arguments.report_date)
