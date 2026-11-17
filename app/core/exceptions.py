"""Business exceptions; the endpoint layer turns them into HTTP status codes."""


class ServiceError(Exception):
    """Base class carrying the HTTP status the router should return."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id

    def detail(self) -> dict[str, str]:
        payload = {"error": self.code, "message": self.message}
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        return payload


class InvalidInputError(ServiceError):
    """Violates the schema contract or the operational safety limits."""

    status_code = 400
    code = "invalid_input"


class ContentRejectedError(ServiceError):
    """Input passed parsing but was rejected before inference."""

    status_code = 400
    code = "content_rejected"


class UpstreamRateLimitError(ServiceError):
    """Model registry or upstream service kept refusing after retries."""

    status_code = 429
    code = "upstream_rate_limited"


class ModelUnavailableError(ServiceError):
    """The production model could not be loaded from the registry."""

    status_code = 503
    code = "model_unavailable"


class ModelTimeoutError(ServiceError):
    """Inference exceeded the configured deadline."""

    status_code = 504
    code = "model_timeout"
