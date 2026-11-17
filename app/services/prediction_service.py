"""Core prediction logic; the router never touches pandas or MLflow directly."""

import asyncio
import time
from typing import Callable

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvalidInputError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from app.schemas.prediction import (
    FeatureContribution,
    PredictRequest,
    PredictResult,
    UsageStats,
)

PRIORITY_LABELS = {1: "立即", 2: "紧急", 3: "常规"}


class PredictionService:
    """Turn one validated request into a probability, a tier and SHAP impacts.

    ``explain_fn`` is injected by the tests so the whole suite runs without
    MLflow, without the registry and without a fitted model on disk.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        explain_fn: Callable | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._explain_fn = explain_fn

    async def predict(self, request: PredictRequest) -> PredictResult:
        started = time.perf_counter()
        participant_rows = len(request.vehicles) + len(request.casualties)
        limit = self.settings.max_participants
        if len(request.vehicles) > limit or len(request.casualties) > limit:
            raise InvalidInputError(
                f"At most {limit} vehicles and {limit} casualties per collision."
            )
        try:
            explanation = await asyncio.wait_for(
                asyncio.to_thread(self._explain, request),
                timeout=self.settings.timeout_seconds,
            )
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"Inference exceeded {self.settings.timeout_seconds:g}s."
            ) from exc

        return PredictResult(
            probability=float(explanation.probability),
            priority=int(explanation.priority),
            priority_label=PRIORITY_LABELS.get(int(explanation.priority), "未知"),
            base_value=float(explanation.base_value),
            contributions=[
                FeatureContribution(
                    feature=item.feature,
                    value=item.value,
                    impact=float(item.impact),
                )
                for item in explanation.contributions
            ],
            other_impact=float(explanation.other_impact),
            cutoffs=[float(value) for value in explanation.cutoffs],
            model_uri=self.settings.model_uri,
            usage=UsageStats(
                input_fields=self._supplied_fields(request),
                participant_rows=participant_rows,
                explained_features=len(explanation.contributions),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    def _explain(self, request: PredictRequest):
        """Runs in a worker thread: inference is CPU-bound, not I/O-bound."""
        frame = self._build_frame(request)
        explain_fn = self._explain_fn or self._default_explain
        try:
            return explain_fn(frame, self.settings.top_features)
        except InvalidInputError:
            raise
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        except Exception as exc:  # registry down, artefact corrupt, shap missing
            raise ModelUnavailableError(
                f"Cannot score with {self.settings.model_uri}: {exc}"
            ) from exc

    @staticmethod
    def _default_explain(frame, top_n: int):
        from src.explain import explain_single

        return explain_single(frame, top_n=top_n)

    @staticmethod
    def _build_frame(request: PredictRequest):
        """Raw inputs -> one-row frame with the same columns as training."""
        import pandas as pd

        core = request.model_dump(exclude={"vehicles", "casualties"})
        frame = pd.DataFrame([core])
        frame["collision_datetime_utc"] = pd.to_datetime(
            frame["collision_datetime_utc"], utc=True
        )
        if request.vehicles:
            frame = pd.concat(
                [frame, _participant_frame(request.vehicles, "vehicle")], axis=1
            )
        if request.casualties:
            frame = pd.concat(
                [frame, _participant_frame(request.casualties, "casualty")], axis=1
            )
        return frame

    @staticmethod
    def _supplied_fields(request: PredictRequest) -> int:
        core = request.model_dump(exclude={"vehicles", "casualties"})
        return sum(value is not None for value in core.values())


def _participant_frame(rows, table: str):
    """Aggregate on-scene rows with A's contract instead of re-implementing it."""
    import pandas as pd

    from src.participants import aggregate

    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame["collision_index"] = "api"
    return aggregate(frame, table).reset_index(drop=True)


_SERVICE: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """FastAPI dependency; tests override this to inject a fake service."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PredictionService()
    return _SERVICE
