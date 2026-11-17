"""Routing layer: validate, delegate, translate business errors to HTTP."""

import uuid

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import ServiceError
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter(prefix="/api/v1", tags=["prediction"])

ERROR_RESPONSES = {
    400: {"description": "Input violates schema.json or an operational limit"},
    429: {"description": "Upstream refused the request after retries"},
    503: {"description": "Production model unavailable in the registry"},
    504: {"description": "Inference exceeded the configured deadline"},
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score one collision and explain the score",
    responses=ERROR_RESPONSES,
)
async def predict(
    request: PredictRequest,
    response: Response,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    """Return P(severe), the dispatch tier and aggregated SHAP contributions."""
    trace_id = uuid.uuid4().hex
    try:
        result = await service.predict(request)
    except ServiceError as exc:
        raise _http_error(exc, trace_id) from exc
    response.headers["X-Trace-ID"] = trace_id
    return PredictResponse(trace_id=trace_id, **result.model_dump())


def _http_error(exc: ServiceError, trace_id: str):
    from fastapi import HTTPException

    exc.trace_id = trace_id
    return HTTPException(status_code=exc.status_code, detail=exc.detail())


def register_error_handlers(app: FastAPI) -> None:
    """Map Pydantic failures to 400 so C and D share one error contract."""

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "message": "Request failed schema validation.",
                "detail": jsonable_encoder(exc.errors()),
            },
        )
