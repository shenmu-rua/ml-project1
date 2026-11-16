# ============================================================
# D 负责：FastAPI 预测服务镜像
# 构建：docker build -f Dockerfile -t ml-project1-api .
# 运行：docker run -p 8000:8000 ml-project1-api
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先拷贝依赖清单，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码（app 包 + src 包，src 是模型/特征/评估依赖）
COPY app/ ./app/
COPY src/ ./src/
COPY schema.json ./schema.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]