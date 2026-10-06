import os
from pathlib import Path

import boto3
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BUCKET = "datalake"
S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
MODEL_VERSION_PREFIX = "models/v1"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def load_model_from_lake():
    """Descarga el modelo y la lista de features
    directo del Data Lake (boto3 + joblib, sin MLflow)."""
    s3 = get_s3_client()

    tmp_dir = Path("/tmp") if Path("/tmp").exists() else Path(".")
    model_path = tmp_dir / "lake_model.joblib"
    features_path = tmp_dir / "lake_feature_cols.joblib"

    s3.download_file(BUCKET, f"{MODEL_VERSION_PREFIX}/model.joblib", str(model_path))
    s3.download_file(BUCKET, f"{MODEL_VERSION_PREFIX}/feature_cols.joblib", str(features_path))

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    model_path.unlink()
    features_path.unlink()

    return model, feature_cols


def predict_from_lake(df: pd.DataFrame) -> pd.Series:
    """Carga el modelo desde el lake y predice sobre df.
    df debe tener las columnas listadas en feature_cols.
    """

    model, feature_cols = load_model_from_lake()
    X = df[feature_cols]
    y_pred = model.predict(X)
    return pd.Series(y_pred, index=df.index, name="predicted_load")


if __name__ == "__main__":
    import pandas as pd

    from tp_mlops2.features import add_cyclical_features

    row = pd.DataFrame(
        [{
            "hour": 14, "dow": 4, "month": 6, "is_weekend": 0,
            "temp_Madrid": 28.0, "temp_Barcelona": 25.0, "temp_Valencia": 27.0,
            "temp_Seville": 32.0, "temp_Bilbao": 20.0,
            "load_lag_24h": 30000.0, "load_lag_168h": 29500.0,
        }],
        index=[pd.Timestamp("2018-06-15 14:00:00")],
    )
    row = add_cyclical_features(row)

    pred = predict_from_lake(row)
    print(f"Predicción (modelo cargado desde el Data Lake): {pred.iloc[0]:.1f} MW")
