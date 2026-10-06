# Decisiones técnicas del TP

Este documento explica la teoría detrás de cada parte del proyecto, por qué se tomó cada decisión y cómo encajan las piezas entre sí. Pensado para repasar antes de una devolución oral o para justificar el diseño ante quien lo evalúe.

## Índice

1. [El problema y el dataset](#1-el-problema-y-el-dataset)
2. [Modelado](#2-modelado)
3. [De notebooks a código reproducible](#3-de-notebooks-a-código-reproducible)
4. [MLflow: tracking y model registry](#4-mlflow-tracking-y-model-registry)
5. [Docker y Docker Compose](#5-docker-y-docker-compose)
6. [Mini-TP 1: API REST](#6-mini-tp-1-api-rest)
7. [Mini-TP 2: API GraphQL y linaje con Neo4j](#7-mini-tp-2-api-graphql-y-linaje-con-neo4j)
8. [Mini-TP 3: API gRPC](#8-mini-tp-3-api-grpc)
9. [Mini-TP 4: streaming con Kafka/Redpanda](#9-mini-tp-4-streaming-con-kafkaredpanda)
10. [Mini-TP 5: aprendizaje federado](#10-mini-tp-5-aprendizaje-federado)
11. [Mini-TP 6: Data Lake](#11-mini-tp-6-data-lake)
12. [CI/CD](#12-cicd)
13. [Decisiones de alcance y qué quedó afuera](#13-decisiones-de-alcance-y-qué-quedó-afuera)

---

## 1. El problema y el dataset

El objetivo es predecir la demanda eléctrica horaria de España (`total load actual`) usando el dataset de Kaggle [ENTSO-E — energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather): cuatro años (2015-2018) de datos horarios de consumo, generación por tipo de fuente, precios del mercado eléctrico, y clima real de las 5 ciudades más grandes del país (Madrid, Barcelona, Valencia, Sevilla, Bilbao).

**Por qué este dataset y no otro más trillado (PJM Hourly Energy, muy usado en tutoriales de Kaggle) o más simple (Hourly Load India):** este trae clima real por ciudad (no solo temperatura mensual agregada), lo que permite ingeniería de features rica y justifica estacionalidad multi-nivel (hora del día, día de semana, temporada). Además trae un dato poco común: `total load forecast`, que es el pronóstico oficial que ya hace el operador de la red eléctrica española (el TSO — Transmission System Operator). Eso da un benchmark de industria real contra el cual comparar el modelo propio, en vez de compararlo solo contra un baseline ingenuo.

Este último punto termina siendo importante para una decisión posterior (ver sección 2, "el hallazgo del data leakage").

## 2. Modelado

El modelado se hizo en notebooks (`notebooks/01` a `05`), que es donde corresponde para trabajo exploratorio — no se pretende que ese código sea reproducible en producción, solo que documente el razonamiento y las pruebas.

### 2.1. Limpieza

Los datos crudos tienen: dos columnas 100% vacías (se dropean directamente), algunos nulos puntuales en `energy` (se interpolan con `method='time'`, que interpola respetando el espaciado real entre timestamps en vez de asumir intervalos regulares — importante en series temporales), y unos 3000 duplicados en el dataset de clima (mismo timestamp y ciudad, pero un evento climático categórico distinto con los mismos valores numéricos — se quedan con el primero).

El dataset de clima viene en formato "largo" (una fila por ciudad y hora) y se pivotea a "ancho" (una columna de temperatura por ciudad) para poder unirlo al dataset de energía por timestamp.

### 2.2. El hallazgo del data leakage

Este es el punto más importante de todo el proceso de modelado. En una primera pasada, se entrenó un modelo usando `total load forecast` (el pronóstico oficial del TSO) como una feature más. El resultado fue excelente — sospechosamente excelente. Al mirar la importancia de features de un Random Forest, esa única columna concentraba el **99.24%** de la importancia total.

Esto es **data leakage**: el modelo no estaba aprendiendo a predecir demanda a partir de sus verdaderos drivers (temperatura, hora, día de la semana), estaba aprendiendo a copiar al operador de red, que ya resolvió el problema por su cuenta con información y métodos que este TP no tiene acceso a replicar (el TSO usa datos que no están en este dataset). Un modelo así sería inútil en un escenario real de "quiero predecir demanda sin depender de que otro ya la haya predicho".

**Decisión:** sacar `total load forecast` de las features y tratar el problema como forecasting genuino: predecir la demanda a partir de clima, calendario, y el historial de demanda misma (lags). El benchmark contra el TSO se mantiene, pero como punto de comparación externo, no como insumo del modelo.

### 2.3. Feature engineering

- **Variables de calendario:** hora, día de la semana, mes, si es fin de semana.
- **Codificación cíclica (sin/cos):** la hora 23 y la hora 0 son consecutivas, pero si se usa el número crudo (23 vs 0) el modelo las ve como opuestas. La transformación `sin(2π·hora/24)`, `cos(2π·hora/24)` las proyecta en un círculo, donde la distancia entre 23 y 0 es la misma que entre cualquier par de horas consecutivas. Se aplica a hora (período 24), día de la semana (período 7) y mes (período 12).
- **Lags:** demanda de hace 24 horas y de hace 168 horas (una semana). Estos capturan patrones de autocorrelación fuertes en series de demanda eléctrica (el patrón de "ayer a esta hora" y "la semana pasada este mismo día a esta hora" son señales muy fuertes) sin necesitar modelos de series temporales más complejos.

### 2.4. Selección y tuning del modelo

Se probaron Regresión Lineal, Gradient Boosting y Random Forest. Ganó **Random Forest**, y no era un resultado obvio de antemano — normalmente el boosting (Gradient Boosting, LightGBM) tiende a superar a Random Forest cuando está bien tuneado, pero en este dataset (~26.000 filas de entrenamiento) Random Forest resultó más robusto, probablemente porque el boosting es más sensible a una búsqueda de hiperparámetros no exhaustiva.

El tuning se hizo con `RandomizedSearchCV` (20 combinaciones aleatorias del espacio de hiperparámetros, en vez de una grilla exhaustiva — más barato computacionalmente con una pérdida de calidad marginal) combinado con `TimeSeriesSplit` de 5 folds.

**Por qué `TimeSeriesSplit` y no un k-fold común:** en un k-fold aleatorio, un fold de validación podría contener datos anteriores a los del fold de entrenamiento — el modelo "vería el futuro" al validar contra el pasado. `TimeSeriesSplit` respeta el orden temporal: cada fold de validación es siempre posterior a los datos con los que se entrenó ese fold. Es la manera correcta de hacer cross-validation en series temporales.

El split final train/test también es temporal: entrenamiento con 2015-2017, test con 2018 completo (para cubrir las 4 estaciones del año en el conjunto de evaluación).

**Resultado final:** Random Forest con `n_estimators=300, max_depth=12, min_samples_leaf=10, max_features=0.8` — MAE 1736.8 MW, MAPE 6.00% contra el test 2018. El benchmark del TSO da MAE ~270 MW cuando se le permite usar su propio forecast como feature (que es lo que se descartó por leakage) — sin ese atajo, el problema es más difícil, y este resultado es razonable para un modelo entrenado con datos públicos y sin acceso a la información privilegiada que tiene el operador de la red.

## 3. De notebooks a código reproducible

Una vez cerrado el modelado, todo el pipeline se reescribió como código Python en `src/tp_mlops2/`, separado en cuatro módulos con responsabilidades bien definidas:

- **`data.py`** — carga los CSV crudos y aplica la limpieza (sección 2.1).
- **`features.py`** — aplica las transformaciones de feature engineering (sección 2.3). Expone `FEATURE_COLUMNS` como una constante: la lista ordenada de las 14 features finales. Esto es importante — es la **única fuente de verdad** sobre qué features usa el modelo y en qué orden. Tanto el entrenamiento como cada API que sirve el modelo importan esta misma constante, en vez de que cada uno mantenga su propia copia de la lista (que se podría desincronizar).
- **`train.py`** — orquesta todo el pipeline (`data.py` → `features.py`), corre el tuning, y loguea el resultado a MLflow (sección 4).
- **`predict.py`** — carga el modelo desde el MLflow Model Registry y expone una función `predict()` que cualquier API puede reusar.

**Por qué separar en estos cuatro módulos y no dejarlo todo en un script:** cada API (REST, GraphQL, gRPC) necesita cargar el modelo y hacer predicciones, pero ninguna necesita reentrenar. Al separar `predict.py` de `train.py`, las APIs importan solo lo que necesitan sin arrastrar dependencias de tuning (`RandomizedSearchCV`, etc.) ni el riesgo de disparar un entrenamiento por accidente.

**Por qué el entrenamiento también se movió a código (y no se dejó como una celda de notebook):** un notebook es apropiado para explorar y decidir qué modelo usar, pero no para *producir* el artefacto que se sirve en producción de forma repetible. Si el proceso de entrenar vive en una celda de Jupyter, no hay garantía de que correrlo dos veces dé el mismo resultado (fácil ensuciarse con el orden de ejecución de celdas, variables reusadas de una celda vieja, etc. — de hecho pasó una vez durante el desarrollo: una comparación de modelos usaba por error un dataset de una celda anterior). Un script de Python (`train.py`) ejecutado de punta a punta es determinístico y auditable.

## 4. MLflow: tracking y model registry

### 4.1. El problema que resuelve

Sin una herramienta de este tipo, "el modelo entrenado" es un archivo (`.joblib`, `.pkl`) que vive en algún disco, sin metadata asociada: no queda registro de con qué hiperparámetros se entrenó, con qué métricas de test, en qué fecha, ni si esa fue la mejor versión de las que se probaron. Si se reentrena, hay que decidir manualmente si sobrescribir el archivo o mantener versiones con nombres tipo `model_v2_final_ahora_si.joblib`.

MLflow separa dos responsabilidades:

- **Tracking**: cada corrida de entrenamiento (un "run") queda registrada con sus hiperparámetros, sus métricas, y los artefactos que produjo (en este caso, el modelo serializado). Se puede comparar runs entre sí en una UI.
- **Model Registry**: un catálogo de modelos con nombre y versión, donde cada versión apunta a un run específico del tracking. Permite decidir, independientemente de "cuándo se entrenó", qué versión es la que efectivamente se sirve en producción.

### 4.2. Arquitectura elegida

MLflow server necesita dos tipos de almacenamiento:

- **Backend store**: guarda la metadata (qué runs existen, qué parámetros y métricas tiene cada uno, el catálogo del registry). Se usó **PostgreSQL** en vez de la alternativa más simple (SQLite en un archivo), porque SQLite no funciona bien cuando varios procesos necesitan escribir/leer concurrentemente a través de contenedores distintos, y porque Postgres es parte de lo que pide la cátedra para el nivel "contenedores" de todos modos.
- **Artifact store**: guarda los archivos binarios pesados (el modelo serializado en sí). Se usó **MinIO**, un servidor de almacenamiento de objetos que implementa el mismo protocolo que Amazon S3. La ventaja de que sea "S3-compatible" es que MLflow (y cualquier librería que use `boto3`, el cliente oficial de AWS) puede hablar con MinIO sin saber que no es Amazon real — solo hace falta apuntarlo a otra URL (`MLFLOW_S3_ENDPOINT_URL`) y darle credenciales locales.

### 4.3. Alias en vez de Stages

Versiones antiguas de MLflow usaban "Stages" fijos (Staging, Production, Archived) para marcar qué versión de un modelo está activa. Versiones más recientes (la usada en este proyecto, 3.16) reemplazaron eso por **Aliases**: etiquetas de texto libre que se pueden asignar a cualquier versión (`production`, `champion`, `shadow-test`, lo que haga falta). Se usó el alias `production` sobre la versión que se decide servir. Esto permite que `predict.py` siempre cargue `models:/random_forest_demanda@production` sin necesidad de saber el número de versión — cuando se entrena y se decide promover una versión nueva, alcanza con reasignar el alias en la UI, sin tocar código en ninguna de las tres APIs.

### 4.4. Qué reemplaza en la práctica

Antes de integrar MLflow, `train.py` guardaba tres archivos sueltos: el modelo (`.joblib`), la lista de features (`.joblib`) y las métricas (`.joblib`). Con MLflow, ninguno de esos tres archivos existe más: el modelo se sube al artifact store (MinIO), las métricas quedan logueadas en el run, y la lista de features se resuelve directamente desde la constante `FEATURE_COLUMNS` de `features.py` (que ya es la fuente de verdad, así que no hacía falta duplicarla en un archivo aparte).

## 5. Docker y Docker Compose

### 5.1. Por qué contenedores

Un contenedor empaqueta una aplicación junto con exactamente las dependencias que necesita (versión de Python, librerías del sistema, etc.), aislado del resto de la máquina. Esto resuelve el clásico "en mi máquina funciona": cualquiera que tenga Docker puede levantar el mismo entorno exacto sin instalar nada más que Docker mismo. Para un TP que se entrega como repositorio, esto es lo que garantiza que el evaluador pueda correrlo sin pelearse con versiones de Python o dependencias del sistema.

### 5.2. Diferencia entre un Dockerfile y Docker Compose

Un **Dockerfile** es la receta para construir **una** imagen: una secuencia de pasos (partir de una base, instalar paquetes, copiar código) que termina en un artefacto reutilizable. Un **docker-compose.yml** no construye nada por sí mismo — orquesta **varios contenedores** a la vez, cada uno corriendo a partir de una imagen (una imagen ya publicada de terceros, como `postgres:16`, o una imagen propia construida a partir de un Dockerfile). El proyecto tiene 4 Dockerfiles propios (`Dockerfile.mlflow`, `Dockerfile.api`, `Dockerfile.graphql`, `Dockerfile.grpc`) porque cada servicio necesita un conjunto distinto de dependencias y arranca un proceso distinto, y un solo `docker-compose.yml` que orquesta los 7 servicios juntos (Postgres, MinIO, MLflow, Neo4j, y las tres APIs).

### 5.3. Redes internas y nombres de servicio

Dentro de la red que Docker Compose crea automáticamente, cada servicio es alcanzable por los demás usando su **nombre de servicio** como si fuera un hostname (por ejemplo, `http://mlflow:5000` desde el contenedor de la API REST) — Docker resuelve ese nombre a la IP interna correcta. Esto es distinto de cómo se accede desde la máquina host (`localhost:5000`), y fue la causa de más de un error durante el desarrollo: cualquier URL hardcodeada con `localhost` funciona cuando el proceso corre en la máquina del desarrollador, pero falla apenas ese mismo proceso se mueve a un contenedor, porque `localhost` dentro de un contenedor se refiere al contenedor mismo, no a sus vecinos. La solución general fue leer esas URLs desde variables de entorno con un valor por defecto de `localhost` (para seguir funcionando en desarrollo local sin Docker) y que `docker-compose.yml` las sobrescriba con el nombre del servicio correspondiente cuando corre containerizado.

### 5.4. Healthchecks y orden de arranque

Un contenedor puede reportarse como "iniciado" (`Started`) mucho antes de estar realmente listo para atender pedidos (por ejemplo, Postgres tarda unos segundos en aceptar conexiones después de arrancar el proceso). Un `healthcheck` define un comando que Docker corre periódicamente para decidir si el contenedor está realmente sano, y `depends_on: condition: service_healthy` hace que un servicio espere a que otro esté saludable (no solo iniciado) antes de arrancar. Se usó en Postgres y MinIO, de los cuales depende MLflow — sin esto, MLflow podía arrancar e intentar conectarse a una base de datos que todavía no aceptaba conexiones, y fallar.

### 5.5. Un problema real que apareció: protección contra DNS rebinding

Al conectar la API (corriendo en su propio contenedor) contra MLflow usando el nombre de servicio (`http://mlflow:5000`), MLflow rechazaba la conexión con `403 Invalid Host header - possible DNS rebinding attack detected`. Esta es una protección de seguridad de versiones recientes del servidor de MLflow: por defecto, solo confía en pedidos cuyo header `Host` sea `localhost` o `127.0.0.1`, para evitar un tipo de ataque donde un sitio malicioso hace que el navegador de una víctima le pegue a un servicio interno creyendo que es un dominio externo. Como en este caso `mlflow` es un nombre de host legítimo (el propio servicio Docker), la solución fue declarar explícitamente qué hosts son confiables con la variable `MLFLOW_SERVER_ALLOWED_HOSTS`, incluyendo tanto el nombre sin puerto como con puerto (`mlflow`, `mlflow:5000`) — el chequeo compara el header completo, así que faltaba la variante con puerto explícito en un primer intento.

## 6. Mini-TP 1: API REST

### 6.1. Por qué FastAPI

FastAPI genera automáticamente documentación interactiva (Swagger, en `/docs`) a partir del código, y usa Pydantic para declarar los contratos de entrada/salida — los tipos de cada campo se validan solos, sin escribir a mano los `if isinstance(...)`. Un pedido con un campo del tipo incorrecto es rechazado automáticamente con `422 Unprocessable Entity`.

### 6.2. Diseño del endpoint de predicción

Se decidió que `/v1/predict` reciba **datos crudos** (timestamp, temperaturas de las 5 ciudades, demanda real de hace 24h y 168h) en vez de las features ya calculadas (los sin/cos, etc.). Esto es más realista: un cliente real de esta API no tiene por qué saber que el modelo usa codificación cíclica internamente — le importa la demanda futura dado un contexto, no los detalles de implementación del feature engineering. La API reconstruye las features internamente reusando la misma función `add_cyclical_features` de `features.py` que usa el entrenamiento, evitando que la lógica de transformación viva duplicada (y potencialmente desincronizada) en dos lugares.

### 6.3. Versionado y respuesta con metadata

El endpoint es `/v1/predict` (con el prefijo de versión) en vez de `/predict` a secas, y la respuesta incluye `model_version` — ambos son parte del contrato pedido: un cliente de la API no solo necesita el número, necesita saber con qué versión del modelo se generó, algo relevante si conviven varias versiones o si hace falta reproducir un resultado más adelante.

## 7. Mini-TP 2: API GraphQL y linaje con Neo4j

### 7.1. Qué resuelve GraphQL que REST no

Con REST, cada endpoint devuelve una forma fija de datos. Si un cliente solo necesita un campo, igual recibe todos los que el endpoint define (**over-fetching**). Si necesita datos relacionados que viven en otro endpoint, tiene que hacer una segunda llamada (o el servidor tiene que anticipar esa necesidad y anidar todo de antemano, infle o no el caso de uso). GraphQL invierte el control: el cliente escribe una consulta que especifica exactamente qué campos quiere, incluyendo estructuras anidadas, y el servidor resuelve solo eso.

En este proyecto, la query principal permite pedir `model { name version metrics { mape } }` sin traer el resto de las métricas si no hacen falta, y permite pedir `lineage { name kind }` anidado en la misma consulta — sin ese campo, el resolver de linaje ni siquiera se ejecuta (no hay costo de ir a buscar el grafo en Neo4j si nadie lo pidió).

### 7.2. Grafos de propiedades y por qué Neo4j para linaje

Una base de datos relacional (SQL) modela bien datos tabulares, pero consultar relaciones de varios saltos (A se relaciona con B, que se relaciona con C, que se relaciona con D) requiere encadenar `JOIN`s, que se vuelven costosos y difíciles de leer a medida que la cadena crece. Una base de datos de grafos como Neo4j modela nodos y relaciones como ciudadanos de primera clase, y el lenguaje de consulta (**Cypher**) está pensado para expresar "caminos" de forma directa: `MATCH (a)-[:RELACION*]->(b)` recorre cualquier cantidad de saltos con una sintaxis compacta.

**Linaje de datos** en MLOps es justo un problema de este tipo: trazar de dónde viene un modelo — qué dataset crudo, qué transformaciones, qué experimento, qué features usó — es naturalmente un grafo (`Dataset → Feature → Experiment → Model`), no una tabla. Sirve para trazabilidad (si el modelo en producción falla, poder reconstruir con qué se entrenó) y para análisis de impacto (si un dataset cambia, saber qué modelos habría que reentrenar).

### 7.3. Cómo se construyó el grafo en este proyecto

`scripts/seed_neo4j.py` lee el modelo activo real desde MLflow (mismo alias `production` que usa la API REST) y construye: dos nodos `Dataset` (el crudo y el limpio, conectados por `DERIVES`), un nodo `Feature` por cada una de las 14 columnas que usa el modelo (conectadas al dataset limpio por `HAS_FEATURE`, y a un nodo `Experiment` por `USED_IN`), y el nodo `Model` final, conectado al experimento por `PRODUCES`. El resolver `lineage` de la API GraphQL corre una consulta Cypher con un patrón de "cualquier cantidad de saltos" (`[:DERIVES|HAS_FEATURE|USED_IN|PRODUCES*]->`) para traer toda la cadena de una vez, sin necesidad de escribir un `MATCH` por cada tipo de relación.

El script es **idempotente** (usa `MERGE` en vez de `CREATE`, y borra todo al principio) — se puede correr las veces que haga falta sin ir duplicando nodos.

### 7.4. Diferencias prácticas observadas entre REST y GraphQL

Documentadas en el README, resumidas acá: GraphQL responde casi siempre con `200 OK` incluso cuando hay errores de lógica (los errores viajan dentro del JSON, en una clave `errors`, no en el código HTTP) — esto es una diferencia de diseño real que hay que tener en cuenta al programar un cliente. A cambio de la flexibilidad, GraphQL exige que el cliente sepa construir una consulta válida, mientras que REST es más directo de invocar con una sola URL.

## 8. Mini-TP 3: API gRPC

### 8.1. Qué problema resuelve gRPC

REST (HTTP/1.1 + JSON de texto) y GraphQL comparten una capa de transporte relativamente pesada para comunicación **interna** entre microservicios de baja latencia: cada request abre su propia conexión (o reusa una con overhead de texto), y JSON es un formato de texto que hay que parsear. gRPC usa **HTTP/2** como transporte (multiplexado — varias llamadas simultáneas sobre una sola conexión TCP, binario en vez de texto plano) y **Protocol Buffers (Protobuf)** como formato de serialización — un formato binario compacto, definido por un contrato estricto (el archivo `.proto`), en vez del JSON sin esquema fijo de REST.

### 8.2. El contrato: `scoring.proto`

El archivo `.proto` define los mensajes (`PredictRequest`, `PredictResponse`, `BatchPredictRequest`) y el servicio (`Scoring`, con los métodos `Predict` y `PredictBatch`). Es un contrato **tipado y versionado por posición**: cada campo de un mensaje tiene un tipo explícito y un número de posición fijo (usado en la codificación binaria), que en un proyecto real no se debe reasignar una vez publicado, porque rompería la compatibilidad con clientes existentes.

De este archivo se generan automáticamente dos módulos Python (`scoring_pb2.py` con las clases de los mensajes, `scoring_pb2_grpc.py` con el cliente y la clase base del servidor) usando el compilador `protoc` — estos archivos generados no se editan a mano, se regeneran cada vez que cambia el `.proto`.

### 8.3. Los cuatro tipos de RPC (y cuáles se usaron)

gRPC soporta cuatro patrones de comunicación: unary (una request, una response — como una llamada de función normal), server-streaming (una request, varias responses en secuencia), client-streaming (varias requests, una response), y bidireccional (ambos lados mandan streams de forma independiente). Este proyecto implementa los dos primeros:

- **`Predict` (unary):** el caso equivalente a `/v1/predict` de REST — un timestamp y contexto climático, una predicción.
- **`PredictBatch` (server-streaming):** el cliente manda una lista de requests en un solo mensaje, y el servidor responde con las predicciones de a una, a medida que están listas, en vez de esperar a calcular todo el lote antes de mandar nada. En el código, la diferencia entre ambos es literal: el método unary usa `return` una vez, el método streaming usa `yield` repetidamente (convirtiéndolo en un generador de Python).

### 8.4. El servidor carga el modelo una sola vez

Requisito explícito del enunciado, y una práctica real importante: cargar un modelo de varios megabytes desde el Model Registry en cada request sería absurdamente costoso. El servidor gRPC lo carga una vez en el constructor (`__init__`) del servicio, cuando el proceso arranca, y lo reusa en memoria para todas las llamadas subsiguientes mientras el proceso viva — el mismo patrón que ya se usaba en las APIs REST y GraphQL (cargar el modelo a nivel de módulo, no dentro de cada handler).

### 8.5. Comparación de latencia (el resultado medido)

Se midieron 50 llamadas a cada protocolo contra el mismo caso de predicción, usando el mismo canal/conexión reusada en ambos casos (para no medir el costo de abrir la conexión, que no es representativo del costo *por predicción*):

```
REST: media 212.20 ms | mediana 95.40 ms | p95 127.14 ms
gRPC: media  71.11 ms | mediana 72.22 ms | p95  93.49 ms
```

gRPC resultó considerablemente más rápido y, sobre todo, más **consistente**: la media de REST está muy por encima de su mediana, lo cual sugiere que algunas llamadas particulares fueron bastante más lentas que el resto (colas de latencia), mientras que en gRPC la media y la mediana están cerca, señal de un comportamiento parejo en todo el benchmark. Esto es consistente con la teoría: menos overhead de serialización (binario vs JSON) y de conexión (HTTP/2 multiplexado vs HTTP/1.1) por request.

La contrapartida, no capturada en el número pero relevante para la decisión de cuándo usar cada uno: gRPC pierde la inspeccionabilidad fácil de REST (no se puede probar con `curl` a mano ni explorar con un navegador, como sí se puede con Swagger o GraphiQL), y requiere generar y mantener stubs cada vez que cambia el contrato.

## 9. Mini-TP 4: streaming con Kafka/Redpanda

### 9.1. Qué problema resuelve streaming que las tres APIs anteriores no

REST, GraphQL y gRPC comparten un mismo paradigma: el cliente pide, el servidor responde — predicción bajo demanda. Eso funciona bien cuando alguien necesita una predicción puntual, pero no modela un escenario real de un operador de red: lecturas de demanda y clima llegando de forma continua, sin que nadie las "pida" una por una. Streaming invierte el control — los datos empujan, el sistema reacciona a medida que llegan, sin esperar a que se acumule un lote ni a que alguien haga una consulta.

### 9.2. Por qué Kafka (acá, Redpanda) y no otra cosa

Un sistema de mensajería simple (tipo MQTT) resuelve "entregar un mensaje a quien esté escuchando ahora", pero no retiene el historial — si el consumidor no está conectado en el momento exacto, lo pierde. Kafka modela el flujo como un **log distribuido**: cada evento se agrega al final de una partición con un número de posición (offset), y los consumidores leen desde la posición que ellos controlan — pueden releer desde el principio, reiniciarse sin perder nada, o tener varios consumidores independientes leyendo el mismo flujo completo a su propio ritmo. Esa persistencia y reproducibilidad es la base de muchos pipelines de datos reales, y es lo que distingue a Kafka de un simple pub/sub.

Se usó **Redpanda** en vez de un Kafka real: habla exactamente el mismo protocolo de red (un cliente Python con la librería `kafka-python` no distingue uno de otro), pero es un solo binario en C++, sin necesidad de Zookeeper ni de la JVM — mucho más liviano para correr en una laptop de desarrollo, que es justo lo que recomienda la cátedra para esta sesión.

### 9.3. Arquitectura: productor, broker, consumidor

- **Productor** (`scripts/stream_producer.py`): reconstruye el dataset limpio con el mismo código (`data.py`) que usa el entrenamiento, toma el año 2018 (el mismo conjunto de test que se usa para evaluar el modelo — así se puede comparar streaming contra batch sobre exactamente los mismos datos) y manda cada fila como un evento al topic `demanda-eventos`, con un pequeño delay entre cada una para simular que llegan en tiempo real.
- **Broker** (Redpanda): recibe y retiene los eventos.
- **Consumidor** (`src/tp_mlops2/streaming/consumer.py`): se suscribe al topic, carga el modelo una sola vez al arrancar (mismo patrón que las tres APIs anteriores — nunca recargar el modelo por evento), y predice cada evento a medida que llega.

Una simplificación consciente en el productor: los lags de demanda (`load_lag_24h`, `load_lag_168h`) que en el pipeline de entrenamiento se calculan con un `.shift()` sobre toda la serie histórica, acá se aproximan tomando la demanda real de la fila actual. Reconstruir el lag verdadero dentro de un evento de streaming requeriría consultar un almacén de features en tiempo real (un "feature store" online) con el historial reciente — una pieza de infraestructura más, fuera del alcance razonable de este mini-TP.

### 9.4. Métricas por ventana: throughput, p95, drift

En vez de reportar una métrica por cada evento individual (ruidoso, difícil de leer), el consumidor agrupa en **ventanas** de 20 eventos (ventana por cantidad fija de eventos, no por tiempo — más simple de implementar y de razonar que una ventana tumbling por reloj) y por cada ventana completa calcula:

- **Throughput**: eventos procesados por segundo en esa ventana.
- **p95 de latencia**: el tiempo de predicción (desde que llega el evento hasta que hay una predicción) por debajo del cual cae el 95% de los eventos de la ventana — igual que se usó en el benchmark de latencia del mini-TP 3, pero calculado en vivo, ventana por ventana.
- **Drift**: un z-score — cuántos desvíos estándar se aleja la media de las predicciones de esa ventana respecto de una referencia.

### 9.5. La referencia de drift y por qué tiene un mecanismo de respaldo

Para calcular el z-score hace falta una media y un desvío de referencia contra los cuales comparar. Se usó un esquema en dos niveles: si el modelo activo en el registry tiene logueadas `mean_pred_mw`/`std_pred_mw` (agregadas a `train.py` junto con las métricas de evaluación habituales — son la media y el desvío de las predicciones sobre el test 2018 completo), esa es la referencia. Si el modelo activo es uno viejo que no las tiene (por ejemplo, antes de este cambio), el consumidor cae a un respaldo: usa la **primera ventana completa del stream** como su propia referencia, fijándola una sola vez. Esto evita que el mecanismo de drift dependa de un solo camino — sirve tanto con un modelo bien instrumentado como con uno que no lo está, sin romperse.

### 9.6. El hallazgo real al simular drift (y por qué no se forzó el resultado)

Para probar que la alerta de verdad dispara, se armó un segundo productor (`scripts/stream_producer_drift.py`) con un escenario exagerado: temperatura fija en 45°C en las 5 ciudades, simulando una ola de calor extrema. El primer intento, con un *shift relativo* de +15°C y luego +30°C sobre la temperatura real, dio un drift casi imperceptible (z-score de ventana entre 0.4 y 1.0, muy por debajo del umbral inicial de 2.0) — y la primera hipótesis fue que había un bug en el pipeline.

Antes de tocar el umbral a ciegas, se aisló el efecto con una prueba directa: la misma fila de datos, predicha con y sin el salto de temperatura, manteniendo todo lo demás igual. Confirmó que el modelo sí reacciona a la temperatura (+2611 MW en una fila de junio con +15°C) — no había ningún bug. La explicación real: los **lags de demanda dominan la predicción mucho más que la temperatura**, y además el tamaño del efecto de un valor de temperatura extremo varía fuertemente según qué tan lejos esté esa combinación (mes del año + temperatura) de lo que el modelo vio en entrenamiento. Repitiendo la misma prueba aislada sobre 100 filas de **enero** (donde 45°C es un salto brutal, totalmente fuera de la distribución de entrenamiento para esa época), la diferencia promedio fue de apenas +368 MW — el Random Forest, lejos de la región de datos que conoce, simplemente no extrapola con fuerza.

Con el topic reiniciado y el productor de temperatura fija corriendo desde el principio del año 2018, el patrón se vio con claridad: las primeras ventanas (meses fríos, 45°C muy fuera de rango) tuvieron drift bajo, sin alerta; recién en ventanas correspondientes a meses donde temperaturas altas sí forman parte de lo que el modelo conoce, el drift se disparó con fuerza (z-score +1.48 y +1.10 en dos ventanas consecutivas), activando la alerta. Se bajó el umbral de 2.0 a 1.0 (justificado: el desvío de referencia, `std_pred_mw≈3846`, ya absorbe la variación natural de un año entero, así que 2.0 desvíos es un umbral demasiado exigente para detectar un evento real) y se documentó el hallazgo tal como salió, en vez de forzar el escenario hasta que diera un número prolijo.

Esto termina siendo, en los hechos, una demostración más honesta de *qué es* el drift y de una limitación real de los modelos basados en árboles: no extrapolan bien fuera del rango de datos que vieron entrenar, y el grado de "sorpresa" del modelo frente a un input atípico depende del contexto (acá, la época del año), no es una propiedad fija del input en sí.

### 9.7. Streaming vs batch

Se midió predecir las mismas 500 filas de test 2018 de dos formas (`scripts/compare_batch_streaming.py`): todas de una sola vez con `model.predict()` sobre el DataFrame completo (batch, vectorizado) contra una por una en un loop, simulando el patrón del consumidor (streaming):

```
BATCH:      0.104s total | 4799 filas/s
STREAMING:  29.321s total |   17 eventos/s (p95 72.8 ms)
```

Batch resultó **~280 veces más rápido en throughput total**. La razón es estructural: scikit-learn vectoriza internamente la predicción sobre una matriz completa (usa numpy/C por debajo para todas las filas a la vez), mientras que predecir de a una paga el overhead de la llamada 500 veces en vez de una.

Pero ese número no responde la pregunta de cuál conviene — responde solo "cuál mueve más volumen en menos tiempo total". La pregunta que importa en un sistema real es otra: ¿cuánto tardo en tener *una* predicción lista desde que *ese* dato puntual está disponible? En batch, ninguna predicción está lista hasta que el lote completo terminó de procesarse — si el lote se arma una vez por hora, la predicción más temprana del lote espera hasta una hora para estar disponible. En streaming, cada predicción está lista casi al instante de que llega su evento (en este caso, en el orden de los 50-70ms medidos). Streaming sacrifica throughput agregado a cambio de latencia de respuesta por evento individual — la elección correcta depende de si el caso de uso necesita "la predicción de este evento, ya" o "las predicciones de este lote, en algún momento cercano".

## 10. Mini-TP 5: aprendizaje federado

### 10.1. Por qué este mini-TP no usa el modelo de demanda eléctrica

Aprendizaje federado (FedAvg) funciona promediando los **parámetros** de un modelo paramétrico — pesos de una red neuronal, de una regresión, de un clasificador lineal. Un Random Forest no tiene un conjunto fijo de parámetros que se puedan promediar de esa forma (cada árbol es una estructura discreta distinta, entrenada con splits propios) — "promediar" dos Random Forest no tiene un equivalente directo y natural al promedio ponderado de FedAvg. Adaptar el proyecto de demanda eléctrica a este paradigma hubiese requerido cambiar de familia de modelo (por ejemplo, a una red neuronal), lo cual es una reescritura de fondo del pipeline, no una extensión — dado el apuro de tiempo de la entrega, se priorizó seguir el enunciado con el dataset alternativo que permite explícitamente (`digits`, de scikit-learn), siguiendo el notebook-tutorial de la cátedra.

### 10.2. Qué es FedAvg y por qué "solo se mandan parámetros"

La idea central del aprendizaje federado: en vez de centralizar los datos de todos los participantes en un solo lugar para entrenar (lo habitual), el modelo **viaja** a donde están los datos. Cada cliente entrena localmente con sus propios datos, que nunca salen de su posesión, y solo manda al servidor los **parámetros resultantes** de ese entrenamiento local — nunca los datos en sí. El servidor agrega esos parámetros (en el caso más simple, FedAvg, con un promedio ponderado por cuántos datos tiene cada cliente: `w_global = Σ (n_k/n) w_k`) y distribuye el modelo global actualizado de vuelta a los clientes para la próxima ronda.

Esto es relevante cuando centralizar el dato no es posible o no conviene: regulaciones de privacidad (datos médicos, financieros), latencia/conectividad intermitente (dispositivos móviles), o simplemente porque el volumen de datos distribuidos es demasiado grande para mover.

### 10.3. Implementación: softmax desde cero en numpy

Siguiendo el tutorial de la cátedra (`clase5/Practica/federated_tutorial.ipynb`), se implementó un clasificador softmax (regresión logística multiclase) completamente desde cero en numpy — sin usar `scikit-learn` para el modelo en sí — porque sus parámetros son exactamente una matriz de pesos `W` y un vector de sesgos `b`, el caso más simple y transparente para mostrar qué es lo que efectivamente "viaja" entre cliente y servidor en FedAvg: esos dos arrays, nada más.

- `entrenar_local(W_glob, b_glob, Xc, yc, ...)`: parte del modelo global recibido, entrena unas pocas épocas de SGD solo con los datos del cliente, devuelve los pesos actualizados y cuántos datos usó.
- `fedavg(actualizaciones)`: en el servidor, promedia los `(W, b)` de los clientes que participaron en la ronda, ponderado por `n_k` (cantidad de datos de cada cliente) — así un cliente con más datos influye proporcionalmente más en el modelo global, en vez de que todos pesen igual sin importar cuánta señal aportaron.
- `federado(clientes, R, frac, ...)`: orquesta `R` rondas de comunicación; en cada ronda selecciona una fracción (`frac`) de clientes al azar (simula que no todos los clientes están siempre disponibles — un escenario realista en dispositivos móviles, por ejemplo) y aplica el ciclo entrenar local → agregar en servidor.

### 10.4. Resultado 1: IID vs centralizado

Con el dataset repartido de forma homogénea entre 5 clientes (**IID** — cada cliente tiene una muestra representativa de las 10 clases, en proporciones similares), el resultado federado (accuracy **0.973**) prácticamente empató al centralizado (**0.967**, entrenado con todos los datos juntos en un solo lugar). Esto confirma el resultado teórico central de FedAvg: cuando los datos están distribuidos de forma pareja, no mover los datos no tiene casi ningún costo en calidad del modelo.

### 10.5. Resultado 2: el costo de non-IID

Se repitió el experimento con una partición deliberadamente heterogénea: cada cliente ve datos de solo **3 de las 10 clases** de dígitos (en vez de las 10 mezcladas). La accuracy federada cayó a **0.769**, una diferencia considerable respecto al 0.973 anterior. La explicación: cada cliente entrena un modelo local que se especializa en las pocas clases que ve, optimizando una función de pérdida distinta a la del problema global — cuando el servidor promedia esos pesos tan dispares entre sí, el resultado es un compromiso que no representa bien a ninguno de los clientes individuales. Este es justamente el desafío de "datos non-IID" que menciona la teoría del curso como uno de los puntos más difíciles de resolver en sistemas federados reales (donde, además, casi nunca los datos están limpiamente distribuidos entre participantes).

### 10.6. Resultado 3: el trade-off privacidad vs performance (DP-FedAvg)

Se implementó una versión con privacidad diferencial simple: antes de mandar su actualización de pesos al servidor, cada cliente la **recorta** a una norma L2 máxima fija (acota cuánto puede influir un solo cliente en la actualización agregada — limita la "sensibilidad" de la contribución) y le agrega **ruido gaussiano** calibrado por un parámetro `noise_std`. Cuanto más ruido, más difícil es para un observador externo (o un servidor malicioso) inferir información precisa sobre los datos de un cliente individual a partir de su actualización — pero también más se degrada la señal útil que llega al modelo.

```
noise_std=0.00 -> accuracy 0.956
noise_std=0.01 -> accuracy 0.969
noise_std=0.05 -> accuracy 0.960
noise_std=0.10 -> accuracy 0.947
```

Con ruido bajo (0.01) el efecto fue casi nulo — de hecho levemente mejor que sin ruido en absoluto, posiblemente porque un ruido chico actúa parecido a una regularización, atenuando el sobreajuste de los clientes individuales. A partir de ahí, la accuracy cae de forma consistente al subir el ruido — el trade-off medible que pide el enunciado: ganar privacidad (más ruido, menos información filtrada por cliente) cuesta calidad del modelo global, y la relación no es necesariamente lineal desde el principio (el primer tramo de ruido es casi gratis).

## 11. Mini-TP 6: Data Lake

### 11.1. Data Lake vs Data Warehouse, y por qué importa para ML

Un Data Warehouse tradicional impone un esquema **antes** de escribir el dato (schema-on-write): hay que transformarlo y limpiarlo para que encaje en tablas predefinidas antes de que exista en el sistema. Un Data Lake invierte el orden — guarda el dato **crudo**, tal como llega, en cualquier formato, y el esquema se aplica recién al leerlo (schema-on-read). Para Machine Learning esto importa especialmente: el dato crudo sin transformar es valioso en sí mismo (un feature que hoy parece irrelevante puede ser útil en un modelo futuro), y forzar un esquema rígido de antemano puede destruir información antes de saber si hacía falta.

El riesgo de este enfoque es terminar con un **data swamp** — un lake sin organización, donde nadie sabe qué hay ni en qué estado está. El antídoto son **zonas**: `raw` (el dato tal cual llegó, sin tocar), `staged`/`curated` (ya limpio, transformado, listo para consumir), y en este proyecto se agregó una tercera zona, `models`, para los artefactos de modelos versionados — una extensión natural del mismo patrón de zonas aplicado no a datos sino a artefactos de ML.

### 11.2. Por qué un bucket separado del de MLflow

El proyecto ya tenía un bucket (`mlflow-artifacts`) que usa MLflow internamente como artifact store — ahí es donde `mlflow.sklearn.log_model()` sube el modelo cuando se llama desde `train.py` (ver sección 4). Para este mini-TP se creó un bucket **nuevo y separado** (`datalake`), con sus propias zonas gestionadas directamente con `boto3`, en vez de reusar el bucket de MLflow. La razón: son responsabilidades distintas — `mlflow-artifacts` es un detalle de implementación interno de MLflow (su estructura de carpetas la define MLflow, no se gestiona a mano), mientras que `datalake` es la infraestructura de datos del proyecto en sí, con una estructura de zonas pensada y controlada explícitamente, independiente de qué herramienta de tracking se use.

### 11.3. El mismo `boto3`, dos endpoints distintos

`scripts/upload_to_datalake.py` y `src/tp_mlops2/datalake.py` usan `boto3.client("s3", endpoint_url=...)` de forma directa — el mismo patrón que ya se explicó en la sección 4.2 para MLflow, pero esta vez el código del proyecto lo controla explícitamente, no una librería de terceros por detrás. El punto central, que menciona la teoría del curso: este mismo código correría sin cambios contra Amazon S3 real — alcanza con cambiar el `endpoint_url` (y las credenciales) de MinIO local a un bucket de AWS. No hay nada en la lógica de subida/descarga atado a que el backend sea MinIO.

### 11.4. Dos caminos independientes para cargar el modelo

El proyecto ahora tiene dos formas distintas de llegar al mismo modelo:

- **Vía MLflow** (`predict.load_model()`, usado por las tres APIs): resuelve el alias `production` en el Model Registry, que internamente sabe en qué artifact store (MinIO) y bajo qué ruta están los archivos, y los descarga a través de la API de MLflow.
- **Vía Data Lake directo** (`datalake.load_model_from_lake()`): descarga `models/v1/model.joblib` del bucket `datalake` con `boto3` puro, sin que MLflow intervenga en absoluto.

Ambos caminos, probados con el mismo caso de entrada, dieron exactamente la misma predicción (30694.1 MW) — confirma que son consistentes entre sí, aunque recorran infraestructura distinta para llegar al mismo resultado. El modelo que vive en `datalake/models/v1/` es una **copia** del que está en el registry de MLflow en el momento de la subida (`scripts/upload_to_datalake.py` lo baja de MLflow una vez y lo resube plano) — no se actualiza sola si se reentrena y se promueve una versión nueva; haría falta correr el script de nuevo para sincronizarla. Esa es una limitación real de tener dos fuentes de verdad en paralelo, aceptable para los fines de este mini-TP pero algo a resolver (automatizar la sincronización, o elegir un solo camino) en un sistema de producción real.

### 11.5. La reflexión: ¿lake o imagen horneada?

El enunciado pide justificar por qué servir el modelo desde el lake en vez de empaquetarlo dentro de la imagen Docker (es decir, un `COPY model.joblib` directo en el `Dockerfile`, el enfoque más simple posible).

**A favor del lake:**
- **No hace falta un rebuild de la imagen para desplegar un modelo nuevo.** Con el modelo horneado, cada reentrenamiento implicaría generar una imagen nueva y volver a desplegarla — acoplando el ciclo de vida del modelo al ciclo de vida del código de la aplicación, que son cosas que cambian a ritmos distintos (el modelo se reentrena cuando hay datos nuevos o el rendimiento decae; el código de la API cambia cuando se agrega una funcionalidad). Separarlos permite actualizar uno sin tocar el otro.
- **La imagen queda genérica**, no atada a una versión específica de un artefacto pesado — la misma imagen puede servir cualquier versión del modelo según qué esté publicado en el lake/registry en ese momento.
- **El historial de versiones queda en un lugar separado de las imágenes Docker**, facilitando auditar qué modelo estuvo activo en qué momento sin tener que rastrear tags o digests de imágenes.
- **Separación de responsabilidades**: quien entrena y publica un modelo no necesita coordinar un despliegue de infraestructura para que esa versión entre en producción.

**En contra:**
- El arranque del contenedor es más lento (hay que descargar el modelo desde la red al iniciar — en este proyecto, del orden de 30 segundos la primera vez que se visto en los logs de los contenedores).
- Agrega una dependencia externa en tiempo de arranque: si el almacenamiento (MinIO/S3) no está disponible en ese momento, el servicio no puede levantar — un modelo horneado en la imagen no tiene ese punto de falla adicional.

La elección correcta depende del contexto: para un modelo que cambia con frecuencia y un equipo que separa el rol de quien entrena del de quien despliega, el lake es casi siempre la opción correcta pese al costo de arranque. Para un modelo que prácticamente no cambia, o un entorno donde minimizar dependencias externas es crítico (edge computing, por ejemplo), hornear el modelo puede ser la decisión más simple y robusta.

## 12. CI/CD

El workflow de GitHub Actions (`.github/workflows/ci.yml`) corre en cada push a `main`: instala dependencias con `uv`, corre el linter (`ruff check .`), y verifica que los módulos centrales del pipeline (`data`, `features`, `train`) se puedan importar sin errores.

**Por qué no entrena ni levanta las APIs en el CI:** en una etapa temprana del proyecto, cuando el modelo se guardaba como archivos `.joblib` sueltos, el CI sí entrenaba un modelo en modo rápido y probaba la API end-to-end contra un servidor real. Al integrar MLflow, esto dejó de ser viable sin más trabajo: entrenar y servir el modelo ahora requieren Postgres, MinIO y MLflow corriendo, que el runner de GitHub Actions no tiene por defecto. Levantar todo ese stack dentro del CI (con sus healthchecks, creación de bucket, etc.) es una inversión de infraestructura considerable para el beneficio que aporta en esta etapa del proyecto — quedó documentado como decisión consciente, no como una limitación no notada. El CI actual es honesto sobre lo que garantiza: código bien formateado y sin errores triviales de import, no una corrida end-to-end del sistema completo.

## 13. Decisiones de alcance y qué quedó afuera

- **Orden de trabajo — infraestructura containerizada desde el principio, no al final:** se decidió explícitamente ir sumando cada pieza (MLflow, luego Neo4j, luego cada API, luego Redpanda) directamente como servicios de Docker Compose a medida que se armaban, en vez de resolver todo en local primero y containerizar recién al final. La ventaja de containerizar temprano es que el entorno final se prueba de entrada, en vez de descubrir problemas de containerización (como los de red interna descriptos en 5.3) recién al empaquetar un sistema ya grande.
- **Adelantar tecnologías de clases posteriores:** Docker, MLflow, streaming, y el nivel de contenedores en general corresponden a clases más avanzadas del curso (MLflow se presenta formalmente en la Clase 2, Docker se profundiza en la Sesión 5, streaming es la Sesión 4) — se decidió avanzar con ellas de todos modos, conscientes de que un enfoque distinto presentado más adelante en el curso podría requerir ajustes.
- **Airflow no se implementó todavía.** Es el orquestador que falta para completar el conjunto de herramientas que pide `CriteriosAprobacion.md` para el nivel contenedores (que exige explícitamente "un servicio de orquestación y algún servicio de ciclo de vida de modelos" — MLflow ya cubre lo segundo). Queda como próximo paso natural: un DAG que dispare `train.py` según un cronograma o disparador, en vez de correrlo a mano.
- **El CI no cubre el sistema completo**, como se explica en la sección 12 — es una limitación conocida y documentada, no un descuido.
- **El Mini-TP 5 (federado) es un sistema aparte, no integrado al resto del proyecto.** Usa el dataset `digits`, no el de demanda eléctrica, por la incompatibilidad entre FedAvg (promedia parámetros de un modelo paramétrico) y Random Forest (no tiene parámetros promediables de esa forma) explicada en 10.1. Es una decisión consciente, priorizando entregar un resultado correcto sobre el dataset alternativo que el enunciado permite, en vez de forzar una adaptación de fondo del pipeline de demanda bajo presión de tiempo.
- **El consumidor y los productores de streaming no están containerizados.** A diferencia de las tres APIs (procesos de servidor persistentes, con un ciclo de vida claro de "levantar y dejar corriendo"), el consumidor de Kafka es un proceso de larga vida pero los productores son scripts de una sola pasada — no tienen el mismo encaje natural en un `docker-compose.yml` pensado para servicios. Quedaron como scripts que se corren a mano con `uv run`, contra el broker Redpanda que sí está containerizado.
- **La copia del modelo en el Data Lake no se sincroniza sola con el registry de MLflow**, como se explica en 11.4 — si se reentrena y se promueve una versión nueva a `production`, hay que volver a correr `scripts/upload_to_datalake.py` a mano para que la copia del lake quede al día. Es una limitación real de tener dos fuentes de verdad en paralelo, aceptada conscientemente para el alcance de este mini-TP.
- **El Hito TP #1 grupal** (equipo, roles, diagrama de arquitectura — mencionado en el enunciado de la Sesión 4) y **el Hito TP #2** (checkpoint del integrador, mencionado en la Sesión 6) están fuera del alcance de este documento, que cubre el trabajo individual sobre el modelo de demanda eléctrica.
