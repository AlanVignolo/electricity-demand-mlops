import statistics
import time
from pathlib import Path

from tp_mlops2.data import clean_and_merge, load_raw
from tp_mlops2.features import add_cyclical_features
from tp_mlops2.predict import load_model, predict

N_ROWS = 500


def build_sample():
    energy, weather = load_raw(Path("data"))
    df = clean_and_merge(energy, weather)
    df2018 = df[df.index.year == 2018].copy()
    df2018["load_lag_24h"] = df2018["total load actual"]
    df2018["load_lag_168h"] = df2018["total load actual"]
    df2018 = add_cyclical_features(df2018)
    return df2018.head(N_ROWS)


def run_batch(model, feature_cols, df):
    start = time.perf_counter()
    predict(model, df, feature_cols)
    elapsed = time.perf_counter() - start

    throughput = len(df) / elapsed
    latency_per_item_ms = (elapsed / len(df)) * 1000
    return elapsed, throughput, latency_per_item_ms


def run_streaming_simulated(model, feature_cols, df):
    latencies = []
    start_total = time.perf_counter()

    for idx in range(len(df)):
        row = df.iloc[[idx]]
        start = time.perf_counter()
        predict(model, row, feature_cols)
        latencies.append((time.perf_counter() - start) * 1000)

    elapsed_total = time.perf_counter() - start_total
    throughput = len(df) / elapsed_total
    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    return elapsed_total, throughput, statistics.mean(latencies), p95


if __name__ == "__main__":
    print(f"Comparando batch vs streaming sobre {N_ROWS} filas de test 2018...\n")

    model, feature_cols, _metrics = load_model()
    df = build_sample()

    batch_elapsed, batch_throughput, batch_latency = run_batch(model, feature_cols, df)
    print(
        f"BATCH:      {batch_elapsed:.3f}s total | "
        f"{batch_throughput:.0f} filas/s | "
        f"{batch_latency:.3f} ms/fila (amortizado)"
    )

    stream_elapsed, stream_throughput, stream_mean_latency, stream_p95 = run_streaming_simulated(
        model, feature_cols, df
    )
    print(
        f"STREAMING:  {stream_elapsed:.3f}s total | "
        f"{stream_throughput:.0f} eventos/s | "
        f"{stream_mean_latency:.3f} ms/evento (media) | "
        f"{stream_p95:.3f} ms/evento (p95)"
    )

    print(f"\nBatch es {stream_elapsed / batch_elapsed:.1f}x más rápido en throughput total.")
