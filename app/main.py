"""D 负责：FastAPI 预测服务（先用假模型跑通，再接入 MLflow production 模型）。"""

from fastapi import FastAPI

from app.api.endpoints import register_error_handlers, router

app = FastAPI(title="Road Collision Severity Triage API")
register_error_handlers(app)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
