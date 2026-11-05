# Churn MLOps：客户流失预测

四人小组机器学习项目：数据 → 分析 → 建模 → 评估 → 上线（Web 预测）。
完整分工与架构见 [docs/project_plan.md](docs/project_plan.md)。

## 模块与负责人

| 成员 | 模块 | 目录 / 文件 |
|---|---|---|
| A | 数据与特征 | `data/`, `src/features.py`, `notebooks/eda.ipynb`, `schema.json` |
| B | 训练与实验追踪 | `src/train.py` |
| C | 评估与解释 | `src/evaluate.py`, `src/explain.py` |
| D | 部署与工程化 | `app/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/` |

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
```

## 协作规范

- 分支：`feat/a-xxx`、`feat/b-xxx`、`feat/c-xxx`、`feat/d-xxx`
- 提交：`feat:` / `fix:` / `docs:` / `test:`
- `main` 受保护，PR 需 1 人审核且 CI 通过
- 互审：A→B→C→D→A
