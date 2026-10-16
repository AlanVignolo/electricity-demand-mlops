import os

import strawberry
from dotenv import load_dotenv
from graphql import NoSchemaIntrospectionCustomRule
from mlflow.tracking import MlflowClient
from neo4j import GraphDatabase
from strawberry.extensions import AddValidationRules, QueryDepthLimiter

from tp_mlops2.predict import MLFLOW_TRACKING_URI, MODEL_ALIAS, REGISTERED_MODEL_NAME

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

@strawberry.type
class Metrics:
    mae_mw: float
    mape: float
    rmse_mw: float


@strawberry.type
class LineageNode:
    name: str
    kind: str


@strawberry.type
class Model:
    name: str
    version: str
    metrics: Metrics

    def lineage(self) -> list[LineageNode]:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(
                """
                MATCH (m:Model {name: $model_name})
                OPTIONAL MATCH path = (n)-[:DERIVES|HAS_FEATURE|USED_IN|PRODUCES*]->(m)
                UNWIND nodes(path) AS node
                RETURN DISTINCT node.name AS name, labels(node)[0] AS kind
                """,
                model_name=REGISTERED_MODEL_NAME,
            )
            nodes = [LineageNode(name=r["name"], kind=r["kind"]) for r in result if r["name"]]
        driver.close()
        return nodes


def get_model() -> Model:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(model_version.run_id)
    m = run.data.metrics

    return Model(
        name=REGISTERED_MODEL_NAME,
        version=str(model_version.version),
        metrics=Metrics(
            mae_mw=m["mae_mw"],
            mape=m["mape"],
            rmse_mw=m["rmse_mw"],
        ),
    )


@strawberry.type
class Query:
    @strawberry.field
    def model(self, name: str) -> Model:
        return get_model()

    @strawberry.field
    def lineage(self, info: strawberry.Info) -> list[LineageNode]:
        # autorizacion por campo: el linaje completo (consulta a Neo4j) es mas
        # costoso y mas sensible que las metricas, solo para roles autorizados
        rol = info.context.get("rol")
        if rol not in ("analista", "admin"):
            raise Exception("no autorizado: el campo lineage requiere rol analista o admin")
        return get_model().lineage()


schema = strawberry.Schema(
    query=Query,
    extensions=[
        lambda: AddValidationRules([NoSchemaIntrospectionCustomRule]),
        lambda: QueryDepthLimiter(max_depth=4),
    ],
)
