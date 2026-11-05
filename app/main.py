"""D 负责：FastAPI 预测服务（先用假模型跑通，再接入 MLflow production 模型）。"""
from fastapi import FastAPI

app = FastAPI(title="Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}
