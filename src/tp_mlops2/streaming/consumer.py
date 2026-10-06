import json
import statistics
import time

import pandas as pd
from kafka import KafkaConsumer
from mlflow.tracking import MlflowClient

from tp_mlops2.features import add_cyclical_features
from tp_mlops2.predict import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    load_model,
    predict,
)

TOPIC = "demanda-eventos"
BOOTSTRAP_SERVERS = "localhost:9092"
WINDOW_SIZE = 20
DRIFT_THRESHOLD_STD = 1.0


def get_reference_stats():
    """Get reference statistics (mean and standard deviation) from the MLflow model run.

    Returns:
        tuple: A tuple containing the mean, standard deviation, and a description string.
               If the metrics are not available, returns (None, None, None).
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_version = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=MODEL_ALIAS)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    if "mean_pred_mw" in metrics and "std_pred_mw" in metrics:
        return metrics["mean_pred_mw"], metrics["std_pred_mw"], "MLflow (test 2018)"
    return None, None, None


def build_row(event):
    timestamp = pd.Timestamp(event["timestamp"])
    row = pd.DataFrame(
        [{
            "hour": event["hour"],
            "dow": event["dow"],
            "month": event["month"],
            "is_weekend": event["is_weekend"],
            "temp_Madrid": event["temp_Madrid"],
            "temp_Barcelona": event["temp_Barcelona"],
            "temp_Valencia": event["temp_Valencia"],
            "temp_Seville": event["temp_Seville"],
            "temp_Bilbao": event["temp_Bilbao"],
            "load_lag_24h": event["load_lag_24h"],
            "load_lag_168h": event["load_lag_168h"],
        }],
        index=[timestamp],
    )
    return add_cyclical_features(row)


def main():
    model, feature_cols, _metrics = load_model()
    ref_mean, ref_std, ref_source = get_reference_stats()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )

    window_preds = []
    window_latencies = []
    window_num = 0

    print(f"Referencia de drift: {ref_source or 'primera ventana del stream'}\n")

    for message in consumer:
        start = time.perf_counter()
        row = build_row(message.value)
        y_pred = predict(model, row, feature_cols)
        latency_ms = (time.perf_counter() - start) * 1000

        window_preds.append(float(y_pred.iloc[0]))
        window_latencies.append(latency_ms)

        if len(window_preds) == WINDOW_SIZE:
            window_num += 1

            if ref_mean is None:
                ref_mean = statistics.mean(window_preds)
                ref_std = statistics.stdev(window_preds) if len(window_preds) > 1 else 1.0
                ref_source = f"primera ventana (#{window_num})"
                print(f"[Ventana {window_num}] referencia fijada: media={ref_mean:.1f} MW")

            window_mean = statistics.mean(window_preds)
            drift_z = (window_mean - ref_mean) / ref_std if ref_std else 0.0

            p95_latency = sorted(window_latencies)[int(len(window_latencies) * 0.95) - 1]
            elapsed = sum(window_latencies) / 1000
            throughput = WINDOW_SIZE / elapsed if elapsed > 0 else float("inf")

            print(
                f"[Ventana {window_num}] "
                f"throughput={throughput:.1f} ev/s | "
                f"p95={p95_latency:.2f} ms | "
                f"media_pred={window_mean:.1f} MW | "
                f"drift_z={drift_z:+.2f}"
            )

            if abs(drift_z) > DRIFT_THRESHOLD_STD:
                print(
                    f"  ALERTA: drift de {drift_z:+.2f} desvíos "
                    f"respecto a la referencia ({ref_source})"
                )


            window_preds = []
            window_latencies = []


if __name__ == "__main__":
    main()
