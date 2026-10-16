import json
import time
from pathlib import Path

from kafka import KafkaProducer

from tp_mlops2.data import clean_and_merge, load_raw
from tp_mlops2.streaming.security import firmar

TOPIC = "demanda-eventos"
BOOTSTRAP_SERVERS = "localhost:9092"
DELAY_SECONDS = 0.2
FIXED_HOT_TEMP = 45.0
# La demanda real de 2018 va de ~18.179 a ~40.693 MW (media ~29.065). Un lag
# de carga muy por fuera de ese rango, pero dentro de lo que el esquema de
# streaming acepta (0 a 60.000), fuerza al modelo a extrapolar con fuerza,
# porque los lags de carga pesan mucho mas en la prediccion que la temperatura.
FIXED_LOW_LOAD = 0.0

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

    print(
        f"Emitiendo {len(df)} eventos al topic '{TOPIC}' con temperaturas fijas "
        f"de {FIXED_HOT_TEMP}°C y lags de carga fijos en {FIXED_LOW_LOAD:.0f} MW "
        "(muy por debajo del rango real, ~18.179-40.693 MW)..."
    )

    for timestamp, row in df.iterrows():
        event = {
            "timestamp": timestamp.isoformat(),
            "hour": int(row["hour"]),
            "dow": int(row["dow"]),
            "month": int(row["month"]),
            "is_weekend": int(row["is_weekend"]),
            "temp_Madrid": FIXED_HOT_TEMP,
            "temp_Barcelona": FIXED_HOT_TEMP,
            "temp_Valencia": FIXED_HOT_TEMP,
            "temp_Seville": FIXED_HOT_TEMP,
            "temp_Bilbao": FIXED_HOT_TEMP,
            "load_lag_24h": FIXED_LOW_LOAD,
            "load_lag_168h": FIXED_LOW_LOAD,
        }
        envelope = {"payload": event, "firma": firmar(event)}
        producer.send(TOPIC, value=envelope)
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print("Listo.")


if __name__ == "__main__":
    main()
