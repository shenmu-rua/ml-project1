# 机器学习小组项目：分工与架构方案

> 题目建议：**Telco 客户流失预测**（二分类，表格数据，约 7000 行）。
> 目标：完成"数据 → 分析 → 建模 → 评估 → 上线（Web 端预测）"完整流程，并在 MLOps 上做规范（MLflow、SHAP 等）。
> 分析部分从简，重点放在工程规范。

---

## 一、项目概览

| 项目 | 说明 |
|---|---|
| 任务类型 | 二分类（客户是否流失） |
| 数据集 | Kaggle: Telco Customer Churn（可替换，只要是 10~20 个字段的表格数据，分工不变） |
| 核心技术栈 | pandas / scikit-learn / XGBoost / LightGBM / Optuna / MLflow / SHAP / Evidently / FastAPI / Streamlit / Docker / GitHub Actions / DVC / pytest |
| 最终成果 | 可运行的 Web 预测系统 + MLflow 实验记录 + 模型解释 + 报告 + 答辩 |

---

## 二、分工（每人 25 分，共 100 分）

每人 = **15 分核心模块 + 10 分共同任务**，工作量对等。

### 2.1 核心模块（各 15 分）

| 成员 | 模块 | 核心任务 | 交付物 |
|---|---|---|---|
| **A：数据** | 数据与特征 | 数据获取与清洗；简单 EDA（5~8 张图）；特征工程；sklearn `Pipeline/ColumnTransformer`；数据校验（pandera）；DVC 数据版本管理；固定 train/val/test 划分 | `data/`、`src/features.py`、`eda.ipynb`、`schema.json` |
| **B：建模** | 训练与实验追踪 | LR / RF / XGBoost / LightGBM 基线；Optuna 调参；**MLflow 实验追踪**（参数、指标、产物）；**Model Registry** 注册并设置 `production` 别名 | `src/train.py`、MLflow 实验记录、注册模型 |
| **C：评估** | 评估与解释 | AUC / PR / F1 / 校准曲线；交叉验证；阈值选择；错误分析与分群表现；**SHAP** 全局 + 单样本解释；Evidently 漂移检测；Model Card | `src/evaluate.py`、`src/explain.py`、评估报告、Model Card |
| **D：上线** | 部署与工程化 | **FastAPI** 预测接口；Web 前端（Streamlit 或 HTML 表单，含 SHAP 单样本瀑布图）；Docker + docker-compose（API + MLflow）；GitHub Actions CI（lint + pytest）；请求日志 | `app/`、`Dockerfile`、`docker-compose.yml`、`.github/workflows/`、`README.md` |

### 2.2 共同任务（每人各 10 分）

| 项目 | 分值 | 要求 |
|---|---|---|
| 自己模块的单元测试 | 3 | pytest，覆盖核心函数 |
| 交叉 Code Review | 2 | A→B→C→D→A 环形互审，每人审一个 PR |
| 报告章节 | 3 | 每人撰写自己模块对应的章节 |
| 答辩 PPT 与讲解 | 2 | 每人负责自己模块的 PPT 页，讲 3~4 分钟 |

---

## 三、如何并行推进（互不等待）

### 3.1 第 0 天：先开会敲定四份接口契约

1. **特征 schema**：`schema.json`（字段名、类型、取值范围）。A 起草，所有人确认。
2. **模型产物格式**：用 `mlflow.sklearn` 保存**包含预处理的完整 Pipeline**，通过 `models:/churn-model@production` 加载。D 部署时不依赖 A 的代码细节。
3. **API 格式**：D 先定好请求/响应 JSON，例如 `POST /predict` → `{"prob": 0.73, "label": 1}`。
4. **仓库与分支规范**：`main` 受保护，PR 合并需 1 人审核。

### 3.2 各人第 1 天即可开工的依据

- **D**：用 `DummyClassifier`（符合 schema）做假模型，先跑通 API、Web、Docker，之后只替换模型。
- **B**：先用 A 提供的原始数据样例直接跑基线，不必等预处理完成。
- **C**：先用 B 的基线结果写评估与 SHAP 脚本，模型更新后重跑即可。
- **A**：优先交付 Pipeline v1 与 schema，再迭代特征。

---

## 四、时间线（以 3 周为例）

| 阶段 | A 数据 | B 建模 | C 评估 | D 上线 |
|---|---|---|---|---|
| **第 1 周** | 清洗、EDA、Pipeline v1、schema | 基线模型、接入 MLflow | 评估脚本框架（用基线跑通） | 假模型 + FastAPI + 前端 + Docker |
| **第 2 周** | 特征迭代、数据校验、DVC | Optuna 调参、模型注册 | 阈值、校准、SHAP、错误分析 | 接入真实模型、CI、日志 |
| **第 3 周** | 数据部分文档 | 最终模型冻结 | 漂移检测、Model Card | 联调、录制 Demo、README |

> 最后 2~3 天：联调、写报告、做 PPT、彩排。**不再加新功能。**

---

## 五、系统架构

```
 原始数据 ──► [A] 清洗 / 校验 / 特征 Pipeline ──► train/val/test (DVC)
                                                        │
                                                        ▼
                                  [B] 训练 + Optuna ──► MLflow Tracking
                                                        │
                                                        ▼
                                          MLflow Model Registry
                                   models:/churn-model@production
                                        │                    │
                                        ▼                    ▼
                          [C] 评估 / SHAP / 漂移      [D] FastAPI 服务
                              Model Card                    │
                                                            ▼
                                                  Web 前端（填表 → 预测 + SHAP）
```

### 推荐仓库结构

```
churn-mlops/
├── data/                       # A（DVC 管理）
├── src/
│   ├── features.py             # A
│   ├── train.py                # B
│   ├── evaluate.py             # C
│   └── explain.py              # C
├── app/                        # D（FastAPI + 前端）
├── tests/                      # 每人写自己的
├── docs/                       # 报告章节，每人一份
├── docker-compose.yml          # D（API + MLflow）
├── requirements.txt
└── .github/workflows/ci.yml    # D
```

---

## 六、协作规范

- **分支命名**：`feat/a-eda`、`feat/b-optuna`、`feat/c-shap`、`feat/d-api`
- **提交信息**：`feat: ...` / `fix: ...` / `docs: ...` / `test: ...`
- **PR 要求**：至少 1 人审核；CI 通过后合并
- **互审顺序**：A 审 B，B 审 C，C 审 D，D 审 A
- **每周同步**：固定 1 次短会（15 分钟），对齐接口与进度

---

## 七、验收清单（证明均分 25%）

- [ ] 每人至少 **8 次有意义的 commit**，并各有 **1 个合并的 PR**
- [ ] 每人至少 **1 次 Code Review 记录**
- [ ] 每人提交 **1 份报告章节**（`docs/`）
- [ ] 每人完成 **自己模块的 pytest 测试**
- [ ] 答辩中每人讲 **相同时长**

---

## 八、最终交付物汇总

1. 可运行的 Web 预测系统（`docker-compose up` 一键启动）
2. MLflow 实验记录与已注册模型
3. SHAP 解释（全局 + 单样本）与 Model Card
4. 完整仓库（含 CI、测试、README）
5. 项目报告（四人章节合并）
6. 答辩 PPT 与 Demo 录屏
