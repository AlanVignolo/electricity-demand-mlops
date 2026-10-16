# TP MLOps II — demanda eléctrica España

TP integrador de Operaciones de Aprendizaje de Máquina II (CEIA/FIUBA). Predice la demanda eléctrica horaria de España (`total load actual`) con el dataset de Kaggle [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather): consumo, generación, precios y clima de las 5 ciudades más grandes del país, 2015-2018.

Un mismo modelo (Random Forest) servido por REST, GraphQL y gRPC, integrado a streaming y a aprendizaje federado, con Data Lake propio, orquestación con Airflow, seguridad real por capa (SAIF), y todo corriendo en Docker Compose como un solo sistema integrado.

## Levantar con Docker

Copiar `.env.example` a `.env` y completar las credenciales.

```powershell
docker compose up -d --build
```

Levanta 13 contenedores: `postgres` (MLflow), `postgres-airflow` (backend de Airflow, separado), `minio` (S3-compatible), `mlflow` (tracking + registry, puerto 5000), `neo4j` (linaje, 7474/7687), `api` (REST, 8000), `graphql-api` (8001), `grpc-api` (50051), `redpanda` (broker de streaming, 9092), `streaming-consumer` (consumidor real, siempre escuchando), `airflow-init` (setup, corre una vez y termina), `airflow-webserver` (UI, 8080) y `airflow-scheduler`.

Primera vez, a mano:
1. Crear el bucket `mlflow-artifacts` en MinIO (`localhost:9001`).
2. Entrenar un modelo (`uv run python -m tp_mlops2.train`) y, en MLflow (`localhost:5000`), ponerle el alias `production`.
3. Sembrar el grafo de linaje: `uv run python scripts/seed_neo4j.py`.

## Setup local sin Docker

Necesita [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

`data/raw/` no está en el repo — bajar el dataset de Kaggle y poner `energy_dataset.csv` y `weather_features.csv` ahí antes de entrenar.

## Servicios

| Servicio | Cómo correrlo local | Qué hace |
|---|---|---|
| **Entrenamiento** | `uv run python -m tp_mlops2.train` | Tuning (`RandomizedSearchCV`, Random Forest, `TimeSeriesSplit`), loguea a MLflow. MAE ≈ 1736.8 MW / MAPE ≈ 6% en test 2018. |
| **REST** | `uv run uvicorn tp_mlops2.api.main:app --reload` | `localhost:8000/docs`. `POST /v1/predict` (API key), `GET /v1/model/metrics` (JWT por rol). |
| **GraphQL** | `uv run uvicorn tp_mlops2.graphql_api.main:app --reload --port 8001` | `localhost:8001/graphql`. Campo `lineage` requiere JWT. |
| **gRPC** | `uv run python -m tp_mlops2.grpc_api.server` | Puerto 50051. Interceptor JWT. Cliente de prueba: `uv run python -m tp_mlops2.grpc_api.client`. |
| **Streaming** | ya corre como servicio Docker (`streaming-consumer`) | Productor manual: `uv run python scripts/stream_producer.py`. Firma HMAC, ventanas de 20 eventos, alerta de drift. |
| **Data Lake** | `uv run python scripts/upload_to_datalake.py` | Publica raw/curated/models en bucket propio de MinIO, con checksum. |
| **Linaje** | `uv run python scripts/seed_neo4j.py` | Arma el grafo en Neo4j desde el modelo activo. |
| **Orquestación** | DAG `train_demanda_electrica`, `@weekly` | `localhost:8080` (admin/admin). Reentrena y publica al Data Lake; promover a `production` sigue siendo manual. |

## Decisiones clave (resumen)

- **Sin leakage**: se sacó `total load forecast` (el pronóstico del propio TSO) como feature — se llevaba el 99% de la importancia y falseaba el error.
- **Random Forest** ganó la comparación con el presupuesto de tuning usado (no es un veredicto absoluto contra boosting).
- **Airflow y MLflow no comparten entorno de Python** (conflicto de versión de `cryptography`): el DAG lanza el entrenamiento en un venv aislado (`/opt/airflow/venv-ml`) vía `subprocess`.
- **Seguridad (SAIF) aplicada a 5 capas**, integrada en `src/tp_mlops2/` y activa en los contenedores — no solo demostrada en notebook: API key + rate limit + JWT por rol en REST, interceptor JWT en gRPC, límite de profundidad + introspección deshabilitada en GraphQL, firma HMAC + cuarentena en streaming, checksum en Data Lake. Un mismo JWT (`tp_mlops2.auth`) autentica en las tres APIs. Notebooks `07` a `12` prueban esos controles contra el servicio real.
- **Federado aparte**: FedAvg necesita un modelo paramétrico; Random Forest no lo es. Corre sobre el dataset `digits` (notebook `06`), no sobre el modelo de demanda.

Detalle completo de cada decisión, con resultados medidos: `docs/DECISIONES.md` (no versionado, ver `.gitignore`).

## Estructura

```
tp_mlops2/
├── dags/train_demanda_electrica.py   # DAG: train_model >> upload_to_datalake
├── scripts/                          # seed_neo4j, benchmark_latency, stream_producer*, upload_to_datalake
├── scoring.proto                     # contrato gRPC
├── src/tp_mlops2/
│   ├── data.py / features.py / train.py / predict.py / auth.py
│   ├── api/            # REST + security.py
│   ├── graphql_api/    # GraphQL
│   ├── grpc_api/       # gRPC (server, client, stubs)
│   └── streaming/      # consumer.py + security.py
├── notebooks/           # EDA (01-06) + seguridad por capa (07-12)
├── tests/test_cliente.py
├── docker-compose.yml
└── Dockerfile.*          # api, mlflow, graphql, grpc, airflow, streaming-consumer
```

## Tests

```powershell
uv run pytest tests/test_cliente.py -v
```

Con REST y GraphQL corriendo. CI (GitHub Actions) corre lint (Ruff) y chequea imports en cada push a `main`.
