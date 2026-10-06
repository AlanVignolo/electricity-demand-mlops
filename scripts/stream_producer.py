import json
import time
from pathlib import Path

from kafka import KafkaProducer

from tp_mlops2.data import clean_and_merge, load_raw

TOPIC = "demanda-eventos"
BOOTSTRAP_SERVERS = "localhost:9092"
DELAY_SECONDS = 0.2


def build_stream_data():
    energy, weather = load_raw(Path("data"))
    df = clean_and_merge(energy, weather)
    return df[df.index.year == 2018]

def main():
    df = build_stream_data()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Emitiendo {len(df)} eventos al topic '{TOPIC}'...")
    for timestamp, row in df.iterrows():
        event = {
            "timestamp": timestamp.isoformat(),
            "hour": int(row["hour"]),
            "dow": int(row["dow"]),
            "month": int(row["month"]),
            "is_weekend": int(row["is_weekend"]),
            "temp_Madrid": float(row["temp_Madrid"]),
            "temp_Barcelona": float(row["temp_Barcelona"]),
            "temp_Valencia": float(row["temp_Valencia"]),
            "temp_Seville": float(row["temp_Seville"]),
            "temp_Bilbao": float(row["temp_Bilbao"]),
            "load_lag_24h": float(row["total load actual"]),
            "load_lag_168h": float(row["total load actual"]),
        }
        producer.send(TOPIC, value=event)
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print("Listo.")

if __name__ == "__main__":
    main()
