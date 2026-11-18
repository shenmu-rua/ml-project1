# ML-project1：英国车祸数据分析与预测

四人小组机器学习项目：数据 → 分析 → 建模 → 评估 → 上线（Web 预测）。
完整分工、架构与仓库结构见 [docs/project_plan.md](docs/project_plan.md)。

## 模块与负责人

| 成员 | 模块 | 目录 / 文件 |
|---|---|---|
| A | 数据与特征 | `data/`, `src/data_pipeline.py`, `src/features.py`, `notebooks/eda.ipynb`, `schema.json` |
| B | 训练与实验追踪 | `src/train.py`, `src/triage.py`, `src/database.py`, `src/participants.py` |
| C | 评估与解释 | `src/evaluate.py`, `src/explain.py` |
| D | 部署与工程化 | `app/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/` |

## 快速开始

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate        # Windows
uv pip install -r requirements.txt
pytest
```

## 数据与训练

仓库保留源码、测试、配置和 Markdown 文档。`data/`、`artifacts/`、数据库、模型和训练输出均由 `.gitignore` 排除，
本地文件保留；`schema.json`、`dvc.yaml` 和 `dvc.lock` 作为工程配置继续入库。Notebook 提交代码与说明，提交前清除运行输出。
运行管线前，请在 `data/raw/` 放置 2024、2025 年的事故、车辆、伤亡三类 CSV，以及 `weather_2024_2025.csv` 天气观测表。

```bash
python -m src.data_pipeline all   # A：清洗、天气匹配、时间切分（或 dvc repro）
python -m src.database            # 合并全部 CSV 为 data/road_collisions.sqlite
python -m src.train               # B：基线 + Optuna 调参 + MLflow 追踪与注册
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

原始数据在 `data/raw/`：事故表、车辆表、伤亡表（DfT Road Safety Open Data，2024–2025，
`https://data.dft.gov.uk/road-accidents-safety-data/`）与天气观测表。车辆表和伤亡表由
`src/participants.py` 按 `schema.json` 的 `participants` 定义聚合到事故级；伤情结论字段不进入模型。

训练产物通过 `models:/road-severity-model@production` 加载（需在仓库根目录运行，使 `src` 可被导入）。
它用于辅助急救调度分级派单：`predict` 返回优先级（1 立即、2 紧急、3 常规），`predict_proba` 返回严重程度概率，
`model.pipeline` 是包含预处理的完整 Pipeline。分级界线与验收标准见 `schema.json` 的 `triage` 段和
[docs/report_B_modeling.md](docs/report_B_modeling.md)。

## 模型训练结果（LightGBM v5）

注册模型 `road-severity-model` v5 预测英国已记录的伤人事故中是否发生致死或重伤。数据按事故时间切分，训练集包含 115,535 起事故，验证集用于确定派单分级界线。

| 数据集 | 事故数 | PR-AUC | ROC-AUC | 精确率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 验证集（2025-06-15 至 2025-09-24） | 24,636 | 0.5297 | 0.7459 | 0.4599 | 0.6130 | 0.5255 |
| 测试集（2025-09-25 至 2025-12-31） | 24,679 | 0.5060 | 0.7415 | 0.4502 | 0.5829 | 0.5080 |

精确率、召回率和 F1 使用 0.5 的二分类阈值。实际派单使用两条独立界线，将事故分为一级（立即）、二级（紧急）和三级（常规）。在现场信息完整的测试集上，四项分级指标均达到项目验收要求。

| 分级指标 | 验收要求 | 测试集结果 |
| --- | ---: | ---: |
| 危及生命事故中被分到三级的比例 | ≤ 5% | 3.79% |
| 致死事故中被分到三级的比例 | ≤ 1% | 0.27%（1/368） |
| 全部事故中被分到三级的比例 | ≥ 20% | 25.82% |
| 一级中危及生命事故的比例相对总体的倍数 | ≥ 3 倍 | 3.52 倍 |

危及生命指标仅统计有细分伤情记录的警区。分级用于辅助调度员排序。单独缺少伤亡人员信息时，致死事故被分到三级的比例升至 1.09%，超过项目的 1% 上限。详见[建模报告](docs/report_B_modeling.md)、[评估报告](docs/report_C_evaluation.md)和[模型卡](docs/model_card.md)。

## 预测服务与前端（D）

网页每次分诊都会请求 `POST /predict?explain=true`，用当前模型的 SHAP 贡献绘制单样本瀑布图。横轴使用 log-odds。
“查看 API 文档”打开 Swagger UI。反向代理部署时，将前端的 `API_DOCS_URL` 设置为浏览器可访问的文档地址，例如 `https://ml.shenmu.xyz/api/docs`。

```bash
uvicorn app.main:app --port 8000        # API，文档在 http://127.0.0.1:8000/docs
streamlit run app/frontend.py           # 前端，默认访问 http://127.0.0.1:8000
```

- `POST /predict`：时间和经纬度必填，其余信息可以部分缺失；车辆与伤亡人员以现场记录列表
  （`vehicles` / `casualties`，字段名同 DfT 原始表）传入，服务端用 `src.participants.aggregate`
  聚合成模型输入。返回派单优先级、严重度概率、模型的两条分级界线，以及本次缺失的信息组。
  请求字段、取值范围和类别编码都由 `schema.json` 生成，NaN、越界值和不带时区的时间返回 422。
- `GET /model-info`：模型版本、分级界线，以及该模型版本在测试集上的四项验收指标。
- 模型由环境变量 `MODEL_URI` 指定（如 `models:/road-severity-model@production`，配合
  `MLFLOW_TRACKING_URI`）。未设置时使用启发式占位模型 `dummy-v1`，接口会标明它是占位、未评估，
  不能用于实际派单；设置后加载失败会直接报错，不会退回占位模型。
- 每次预测在 `logs/request_log.jsonl`（可用 `REQUEST_LOG_PATH` 修改）追加一行，包含请求标识、
  输入、结果与模型版本。

容器编排（API + 前端 + MLflow）：

```bash
docker compose up --build               # 默认使用占位模型
```

接入真实模型：把模型训练并注册到编排里的 MLflow，再带上 `MODEL_URI` 重启 API。

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train
```

```bash
MODEL_URI=models:/road-severity-model@production docker compose up -d api
```

## 协作规范

- 分支：`feat/a-xxx`、`feat/b-xxx`、`feat/c-xxx`、`feat/d-xxx`
- 提交：`feat:` / `fix:` / `docs:` / `test:`
- `main` 受保护，PR 需 1 人审核且 CI 通过
- 互审：A→B→C→D→A
