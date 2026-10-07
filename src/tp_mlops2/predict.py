import os
import pickle
import tempfile

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from tp_mlops2.features import FEATURE_COLUMNS

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "random_forest_demanda"
MODEL_ALIAS = "production"


def load_model():
    """Carga el modelo en produccion y sus metricas desde el MLflow Model Registry."""
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    return model, FEATURE_COLUMNS, metrics


def load_model_lightweight(s3_client, bucket: str = "mlflow-artifacts"):
    """Igual que load_model(), pero sin el SDK de mlflow: resuelve el modelo en
    produccion via REST API y descarga el artefacto pickled directo de MinIO/S3.

    Pensada para entornos (como Airflow) donde instalar el paquete mlflow
    completo generaria conflictos de dependencias con otros paquetes fijados.
    """
    base_url = MLFLOW_TRACKING_URI.rstrip("/")

    alias_resp = requests.get(
        f"{base_url}/api/2.0/mlflow/registered-models/alias",
        params={"name": REGISTERED_MODEL_NAME, "alias": MODEL_ALIAS},
        timeout=30,
    )
    alias_resp.raise_for_status()
    model_version = alias_resp.json()["model_version"]
    model_id = model_version["source"].removeprefix("models:/")
    run_id = model_version["run_id"]

    run_resp = requests.get(
        f"{base_url}/api/2.0/mlflow/runs/get",
        params={"run_id": run_id},
        timeout=30,
    )
    run_resp.raise_for_status()
    run = run_resp.json()["run"]
    experiment_id = run["info"]["experiment_id"]
    metrics_list = run["data"].get("metrics", [])
    metrics = {m["key"]: m["value"] for m in metrics_list}

    artifacts_prefix = f"{experiment_id}/models/{model_id}/artifacts"

    with tempfile.TemporaryDirectory() as tmp_dir:
        mlmodel_path = f"{tmp_dir}/MLmodel"
        s3_client.download_file(bucket, f"{artifacts_prefix}/MLmodel", mlmodel_path)
        with open(mlmodel_path, "r", encoding="utf-8") as f:
            mlmodel = yaml.safe_load(f)
        model_filename = mlmodel["flavors"]["sklearn"]["pickled_model"]

        model_path = f"{tmp_dir}/{model_filename}"
        s3_client.download_file(bucket, f"{artifacts_prefix}/{model_filename}", model_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model, FEATURE_COLUMNS, metrics


def predict(model, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Predice la demanda para cada fila de df, usando el orden de features del modelo."""
    X = df[feature_columns]
    y_pred = model.predict(X)

    return pd.Series(y_pred, index=df.index, name="predicted_load")
