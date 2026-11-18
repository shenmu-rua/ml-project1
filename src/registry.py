"""Resolve the registry alias once per audit and load that immutable version."""

import os

from src.features import ROOT

MODEL_NAME = "road-severity-model"


def load_registered_model(version=None):
    """Return the fitted classifier and the registry entry used to load it."""
    import mlflow

    tracking = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking)
    client = mlflow.MlflowClient()
    entry = (
        client.get_model_version(MODEL_NAME, str(version))
        if version is not None
        else client.get_model_version_by_alias(MODEL_NAME, "production")
    )
    uri = f"models:/{entry.name}/{entry.version}"
    return mlflow.sklearn.load_model(uri), entry
