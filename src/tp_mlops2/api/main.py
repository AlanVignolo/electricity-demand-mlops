import pandas as pd
from fastapi import Depends, FastAPI
from mlflow.tracking import MlflowClient

from tp_mlops2.api.schemas import ModelInfo, PredictionRequest, PredictionResponse
from tp_mlops2.api.security import auditar, rate_limit, requiere_rol
from tp_mlops2.features import add_cyclical_features
from tp_mlops2.predict import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    load_model,
    predict,
)

app = FastAPI(title="Demanda Electrica España API", version="1.0")

model, feature_cols, metrics = load_model()

_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
_model_version = _client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
MODEL_VERSION = f"{REGISTERED_MODEL_NAME} v{_model_version.version}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        model_name="random_forest",
        model_version=MODEL_VERSION,
        feature_cols=feature_cols,
        mae_mw=metrics["mae_mw"],
        mape=metrics["mape"],
    )


@app.get("/v1/model/metrics")
def model_metrics(usuario: dict = Depends(requiere_rol("admin", "analista"))) -> dict:
    auditar(usuario["sub"], "consulta_metricas", "acceso a /v1/model/metrics")
    return {"mae_mw": metrics["mae_mw"], "mape": metrics["mape"]}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_load(
    request: PredictionRequest, cliente: str = Depends(rate_limit)
) -> PredictionResponse:
    row = pd.DataFrame(
        [{
            "hour": request.timestamp.hour,
            "dow": request.timestamp.weekday(),
            "month": request.timestamp.month,
            "is_weekend": int(request.timestamp.weekday() >= 5),
            "temp_Madrid": request.temp_madrid,
            "temp_Barcelona": request.temp_barcelona,
            "temp_Valencia": request.temp_valencia,
            "temp_Seville": request.temp_seville,
            "temp_Bilbao": request.temp_bilbao,
            "load_lag_24h": request.load_24h_ago,
            "load_lag_168h": request.load_168h_ago,
        }],
        index=[request.timestamp],
    )
    row = add_cyclical_features(row)

    y_pred = predict(model, row, feature_cols)
    auditar(cliente, "prediccion", f"{y_pred.iloc[0]:.1f} MW")

    return PredictionResponse(
        predicted_load_mw=float(y_pred.iloc[0]),
        timestamp=request.timestamp,
        model_version=MODEL_VERSION,
    )
