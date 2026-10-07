import os
from pathlib import Path

import boto3
import joblib
from dotenv import load_dotenv

from tp_mlops2.predict import load_model_lightweight

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


def upload_raw_zone(s3):
    raw_dir = Path("data/raw")
    for filename in ["energy_dataset.csv", "weather_features.csv"]:
        local_path = raw_dir / filename
        key = f"raw/{filename}"
        s3.upload_file(str(local_path), BUCKET, key)
        print(f"Subido: s3://{BUCKET}/{key}")


def upload_curated_zone(s3):
    local_path = Path("data/processed/energy_weather_clean.parquet")
    key = "curated/energy_weather_clean.parquet"
    s3.upload_file(str(local_path), BUCKET, key)
    print(f"Subido: s3://{BUCKET}/{key}")


def upload_models_zone(s3):
    model, feature_cols, metrics = load_model_lightweight(s3)

    local_tmp = Path("model_tmp.joblib")
    joblib.dump(model, local_tmp)

    key = f"{MODEL_VERSION_PREFIX}/model.joblib"
    s3.upload_file(str(local_tmp), BUCKET, key)
    local_tmp.unlink()

    joblib.dump(feature_cols, Path("feature_cols_tmp.joblib"))
    s3.upload_file("feature_cols_tmp.joblib", BUCKET, f"{MODEL_VERSION_PREFIX}/feature_cols.joblib")
    Path("feature_cols_tmp.joblib").unlink()

    print(f"Subido: s3://{BUCKET}/{key}")
    print(f"Subido: s3://{BUCKET}/{MODEL_VERSION_PREFIX}/feature_cols.joblib")
    print(f"Métricas del modelo (referencia): {metrics}")


if __name__ == "__main__":
    s3 = get_s3_client()
    upload_raw_zone(s3)
    upload_curated_zone(s3)
    upload_models_zone(s3)
    print("\nData Lake poblado con las 3 zonas: raw/, curated/, models/v1/")
