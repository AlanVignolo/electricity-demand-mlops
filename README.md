# TP MLOps II — demanda eléctrica España

TP integrador de Operaciones de Aprendizaje de Máquina II (CEIA/FIUBA). La idea es predecir la demanda eléctrica horaria de España (`total load actual`) con el dataset de Kaggle [energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather): consumo, generación, precios y clima de las 5 ciudades más grandes del país, 2015-2018.

Voy por el nivel contenedores del TP: todo corre en Docker Compose (Postgres, MinIO, MLflow, Neo4j, y dos APIs — REST y GraphQL). Lo fui armando de a poco, mini-TP por mini-TP, según el orden del curso.

## Setup local (sin Docker)

Necesita [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv venv .venv --python 3.12
uv sync
```

Para los notebooks, registro el kernel:

```powershell
uv run python -m ipykernel install --user --name=tp-mlops2 --display-name "Python (TP MLOps II)"
```

`data/raw/` y `data/processed/` no están en el repo, pesan bastante. Hay que bajar el dataset de Kaggle y poner `energy_dataset.csv` y `weather_features.csv` en `data/raw/` antes de entrenar.

## Levantar con Docker

Copiar `.env.example` a `.env` y completar las credenciales.

```powershell
docker compose up -d --build
```

Levanta 9 contenedores: `postgres` (guarda experimentos/métricas de MLflow), `minio` (guarda los modelos serializados, S3-compatible), `mlflow` (tracking + registry, puerto 5000), `neo4j` (grafo de linaje, 7474 consola / 7687 driver), `api` (REST, 8000), `graphql-api` (8001), `grpc-api` (50051) y `redpanda` (broker de streaming, 9092).

La primera vez hay que hacer 3 cosas a mano:
1. Crear el bucket `mlflow-artifacts` en MinIO (`localhost:9001`).
2. Entrenar un modelo y, en la UI de MLflow (`localhost:5000`), ponerle el alias `production` a la versión que quiero servir.
3. Sembrar el grafo de linaje: `uv run python scripts/seed_neo4j.py`.

## Entrenar

```powershell
uv run python -m tp_mlops2.train
```

Tuning con `RandomizedSearchCV` sobre un Random Forest (`TimeSeriesSplit`, 5 folds), evaluado contra 2018 como test. Loguea todo a MLflow — hiperparámetros, métricas y el modelo, que queda registrado como nueva versión. Ya no guarda nada en `.joblib` local, todo vive en MLflow/MinIO.

Ahora mismo da MAE ≈ 1736.8 MW / MAPE ≈ 6% contra 2018, bastante cerca del forecast oficial del operador de red español.

Hay un modo rápido para no esperar el tuning completo (útil para chequear que el pipeline no se rompió):

```powershell
$env:TP_MLOPS2_QUICK_TRAIN="1"
uv run python -m tp_mlops2.train
```

## API REST

```powershell
uv run uvicorn tp_mlops2.api.main:app --reload
```

Swagger en `localhost:8000/docs`. `GET /health`, `GET /model/info` (metadata + métricas del modelo activo), `POST /v1/predict` (timestamp + temperatura de las 5 ciudades + demanda de hace 24h y 168h, devuelve la predicción).

```json
{
  "timestamp": "2018-06-15T14:00:00",
  "temp_madrid": 28.0,
  "temp_barcelona": 25.0,
  "temp_valencia": 27.0,
  "temp_seville": 32.0,
  "temp_bilbao": 20.0,
  "load_24h_ago": 30000,
  "load_168h_ago": 29500
}
```

Si mando algo mal tipado tira 422 (validación de Pydantic).

## API GraphQL

```powershell
uv run uvicorn tp_mlops2.graphql_api.main:app --reload --port 8001
```

GraphiQL en `localhost:8001/graphql`.

```graphql
{
  model(name: "random_forest_demanda") {
    name
    version
    metrics { maeMw mape rmseMw }
    lineage { name kind }
  }
}
```

`lineage` consulta Neo4j al momento y trae la cadena completa: dataset crudo → dataset limpio → features del modelo → experimento → modelo.

**REST vs GraphQL con el mismo dato:** `/model/info` en REST siempre trae todos los campos aunque me interese uno solo; en GraphQL pido justo lo que necesito. Para traer el linaje además de las métricas, en REST tendría que pegarle a un segundo endpoint; en GraphQL lo pido anidado en la misma query. REST usa el código HTTP para errores; GraphQL casi siempre devuelve 200 y el error viaja adentro del JSON (`"errors"`), así que hay que mirar el body. A cambio GraphQL exige que el cliente sepa armar la query — para algo tan chico como esto la ventaja no es enorme, se nota más con muchas entidades relacionadas.

## API gRPC

```powershell
uv run python -m tp_mlops2.grpc_api.server
```

Queda escuchando en el puerto 50051, con el modelo cargado una sola vez al arrancar. El contrato está en `scoring.proto`: `Predict` (unary, una request → una response) y `PredictBatch` (server-streaming, mando varios items en un solo request y recibo las predicciones de a una a medida que están listas).

```powershell
uv run python -m tp_mlops2.grpc_api.client
```

Prueba ambos métodos contra el servidor.

**Latencia gRPC vs REST**, 50 requests a cada uno (`scripts/benchmark_latency.py`), misma predicción en los dos casos:

```
REST: media 212.20 ms | mediana 95.40 ms | p95 127.14 ms
gRPC: media  71.11 ms | mediana 72.22 ms | p95  93.49 ms
```

gRPC dio bastante más rápido y más estable (la media de REST se aleja mucho de su mediana, señal de que hubo algunas requests lentas sueltas; gRPC se mantuvo parejo). Tiene sentido con lo que vimos en la teoría: gRPC usa HTTP/2 (multiplexado, binario) y Protobuf en vez de JSON sobre HTTP/1.1, así que paga menos overhead de serialización y de conexión por request. La contra es que perdés la inspección fácil que tenés con REST (no puedo pegarle con curl o abrir `/docs` en el navegador) y hay que generar y mantener los stubs cuando cambia el contrato.

```powershell
uv run python scripts/benchmark_latency.py
```

## Streaming (Kafka/Redpanda)

Simula un flujo de eventos y puntúa el modelo online, en vez de esperar un pedido puntual.

```powershell
uv run python -m tp_mlops2.streaming.consumer
```

Se suscribe al topic `demanda-eventos`, carga el modelo una sola vez (igual que las otras APIs) y va prediciendo evento por evento. Agrupa en ventanas de 20 eventos y por cada ventana completa calcula throughput, p95 de latencia, y un indicador de drift.

```powershell
uv run python scripts/stream_producer.py
```

Reconstruye el dataset limpio, toma el año 2018 (el mismo test set de siempre) y lo manda al topic de a un evento por vez, con un pequeño delay simulando que llega en tiempo real.

**Drift:** la referencia es la media y el desvío de las predicciones logueadas en el run de MLflow activo (`mean_pred_mw`/`std_pred_mw`, agregadas a `train.py`); si el modelo activo no las tiene logueadas, usa la primera ventana del stream como referencia. Por ventana calcula un z-score: cuántos desvíos estándar se corre la media de esa ventana respecto a la referencia. Si supera el umbral (1.0), tira una alerta.

Para ver la alerta disparar de verdad armé un segundo productor con un escenario exagerado — temperatura fija en 45°C en las 5 ciudades (`scripts/stream_producer_drift.py`), simulando una ola de calor extrema:

```powershell
uv run python scripts/stream_producer_drift.py
```

Acá encontré algo que no esperaba: el drift **no aparece parejo** a lo largo del stream. En las primeras ventanas (meses más fríos del año en el dataset), 45°C es un salto tan grande respecto a lo que el modelo vio en esa época que casi no reacciona — el Random Forest no extrapola bien fuera de la distribución que vio al entrenar. Recién en las ventanas que caen en meses donde temperaturas altas sí son algo que el modelo conoce, la predicción se dispara y ahí sí salta la alerta. Probé esto de forma aislada (misma fila, con y sin el salto de temperatura): en una muestra de enero el efecto es chico, unos +368 MW de diferencia promedio, contra +2600 MW en una fila de junio. Tiene sentido con lo que ya habíamos visto en el EDA: la demanda tiene una relación en U con la temperatura, no lineal, y el modelo aprendió esa forma solo donde tuvo datos para aprenderla.

**Batch vs streaming**, mismas 500 filas de test 2018, prediciendo todas de una (`model.predict()` vectorizado) contra prediciendo de a una simulando el consumidor (`scripts/compare_batch_streaming.py`):

```
BATCH:      0.104s total | 4799 filas/s
STREAMING:  29.321s total | 17 eventos/s (p95 72.8 ms)
```

Batch es como 280 veces más rápido en throughput total — tiene sentido, scikit-learn vectoriza la predicción de todo el lote en una sola llamada en vez de pagar el overhead de 500 llamadas sueltas. Pero esa no es la comparación justa para decidir cuál usar: en batch no tenés ninguna predicción hasta que termina todo el lote, en streaming tenés cada predicción lista casi al instante de que llega su evento. Si lo que importa es reaccionar rápido a un evento individual (alguien quiere saber la demanda prevista *ahora*, no al final del día), streaming gana aunque su throughput total sea mucho menor.

```powershell
uv run python scripts/compare_batch_streaming.py
```

## Aprendizaje Federado

Esto es aparte del resto del proyecto — no usa el modelo de demanda eléctrica (FedAvg promedia pesos de un modelo paramétrico, y un Random Forest no se presta a eso sin cambiar de modelo), así que lo armé con el dataset `digits` de scikit-learn que permite el enunciado, siguiendo el tutorial de la cátedra.

Notebook: `notebooks/06_federated_learning.ipynb`. Un clasificador softmax (W, b) entrenado con SGD manual en numpy, repartiendo el dataset entre 5 "clientes" que nunca comparten sus datos entre sí — solo mandan sus pesos entrenados localmente al servidor, que los promedia (FedAvg).

Resultados:

```
Centralizado:        0.967
Federado (IID):       0.973
Federado (non-IID):   0.769
```

Con los datos repartidos de forma pareja entre clientes (IID), el federado prácticamente empata al centralizado — no perdés nada por no mover los datos. Repartiendo distinto, de forma que cada cliente solo ve 3 de las 10 clases de dígitos (non-IID, heterogeneidad fuerte), la accuracy cae bastante: cada cliente optimiza para lo poco que ve, y promediar esos pesos tan distintos no da un buen modelo global.

También medí el trade-off privacidad vs performance con DP-FedAvg (cada cliente recorta su actualización a una norma máxima y le suma ruido gaussiano antes de mandarla):

```
noise_std=0.00 -> 0.956
noise_std=0.01 -> 0.969
noise_std=0.05 -> 0.960
noise_std=0.10 -> 0.947
```

Con poco ruido casi no se nota (incluso subió un poco, puede estar actuando como regularizador), pero a partir de ahí cae de forma consistente a medida que subo el ruido — exactamente el trade-off que se espera: más privacidad, peor performance.

## Data Lake (MinIO)

Un bucket separado (`datalake`, distinto del `mlflow-artifacts` que usa MLflow internamente) con tres zonas, siguiendo el patrón raw → curated → models del Data Lake.

```powershell
uv run python scripts/upload_to_datalake.py
```

Sube `energy_dataset.csv` y `weather_features.csv` a `raw/`, el parquet limpio a `curated/`, y una copia versionada del modelo activo (bajada de MLflow y resubida como joblib puro) a `models/v1/`.

```python
from tp_mlops2.datalake import predict_from_lake
```

`src/tp_mlops2/datalake.py` carga el modelo **directo del lake con boto3**, sin pasar por MLflow — un camino de carga completamente aparte del que usan las tres APIs. Lo probé con el mismo caso de siempre y dio la misma predicción (30694.1 MW), así que ambos caminos son consistentes.

```powershell
uv run python -m tp_mlops2.datalake
```

**¿Por qué servir desde el lake en vez de empaquetar el modelo dentro de la imagen Docker?** Lo que gano: no tengo que rebuildear la imagen cada vez que reentreno — subo el artefacto nuevo al lake y el contenedor lo trae al reiniciar, sin tocar el `Dockerfile`. La imagen queda más genérica, no lleva pegado un modelo específico de 32MB. El lake además retiene el historial de versiones por su cuenta, separado de qué imagen está corriendo en cada momento, así que puedo auditar qué modelo estuvo activo en una fecha sin rastrear tags de Docker. Y separa responsabilidades: quien entrena puede publicar un modelo nuevo sin coordinar un deploy de infraestructura.

Lo que pierdo: el arranque es más lento (el contenedor tiene que bajar el modelo al iniciar — en este proyecto, unos 30 segundos la primera vez) y la API pasa a depender de que MinIO esté arriba en ese momento; un modelo horneado en la imagen no tiene esa dependencia externa al arrancar.

## Linaje (Neo4j)

`scripts/seed_neo4j.py` lee el modelo activo de MLflow y arma el grafo: dataset crudo, dataset limpio, una feature por cada columna que usa el modelo, el experimento y el modelo, todo conectado. Se puede correr de nuevo sin duplicar nada.

```powershell
uv run python scripts/seed_neo4j.py
```

## Estructura

```
tp_mlops2/
├── data/                        # raw/ y processed/, no versionados
├── notebooks/                    # EDA, no es código de producción
├── scripts/
│   ├── seed_neo4j.py
│   ├── benchmark_latency.py      # gRPC vs REST
│   ├── stream_producer.py        # productor normal
│   ├── stream_producer_drift.py  # productor con temperatura fija, para probar la alerta
│   └── compare_batch_streaming.py
├── scoring.proto                 # contrato gRPC
├── src/tp_mlops2/
│   ├── data.py                   # limpieza
│   ├── features.py               # cíclicas + lags
│   ├── train.py                  # tuning + logging a MLflow
│   ├── predict.py                # carga el modelo desde el registry
│   ├── api/                       # REST
│   ├── graphql_api/               # GraphQL
│   ├── grpc_api/                  # gRPC (server, client, stubs generados)
│   └── streaming/consumer.py      # consumidor Kafka/Redpanda
├── tests/test_cliente.py
├── docker-compose.yml
└── Dockerfile.api / Dockerfile.mlflow / Dockerfile.graphql / Dockerfile.grpc
```

El pipeline (`data.py` → `features.py` → `train.py`/`predict.py`) no depende de los notebooks para nada — esos quedaron solo para el análisis exploratorio.

## Tests

Con la API REST y la GraphQL corriendo:

```powershell
uv run pytest tests/test_cliente.py -v
```

Prueba un caso válido y uno inválido contra REST, y una query contra GraphQL.

## CI

En cada push a `main`, GitHub Actions corre el lint (Ruff) y chequea que los módulos del pipeline importen bien. No entrena ni levanta las APIs porque eso requiere Postgres/MinIO/MLflow corriendo, y todavía no vale la pena levantar todo ese stack solo para el CI.
