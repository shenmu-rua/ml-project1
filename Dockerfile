# ============================================================
# D 负责：FastAPI 预测服务镜像（compose 里的 MLflow 服务也用它构建）
# 构建：docker build -t ml-project1-api .
# 运行：docker run -p 8000:8000 ml-project1-api
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# libgomp1：LightGBM / XGBoost 运行时需要的 OpenMP 库，slim 镜像不带。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖清单，利用 Docker 层缓存。
# 这里安装完整依赖：反序列化注册表里的真实模型需要 scikit-learn、LightGBM/XGBoost、
# skops、mlflow 以及 src 包用到的全部库。
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY schema.json ./schema.json
COPY src/ ./src/
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
