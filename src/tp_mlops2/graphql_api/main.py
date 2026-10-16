import jwt
from fastapi import FastAPI, Request
from strawberry.fastapi import GraphQLRouter

from tp_mlops2.auth import decodificar_token
from tp_mlops2.graphql_api.schema import schema

app = FastAPI(title="Demanda Electrica España - GraphQL API")


async def get_context(request: Request) -> dict:
    """Decodifica el JWT del header Authorization (si viene) y expone el rol
    en el contexto, para que Query.lineage pueda autorizar por campo."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return {}
    token = auth_header.split(" ", 1)[1]
    try:
        payload = decodificar_token(token)
    except jwt.PyJWTError:
        return {}
    return {"rol": payload.get("rol")}


graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

