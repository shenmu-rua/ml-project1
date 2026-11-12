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

## 协作规范

- 分支：`feat/a-xxx`、`feat/b-xxx`、`feat/c-xxx`、`feat/d-xxx`
- 提交：`feat:` / `fix:` / `docs:` / `test:`
- `main` 受保护，PR 需 1 人审核且 CI 通过
- 互审：A→B→C→D→A
