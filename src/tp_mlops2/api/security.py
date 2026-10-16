import datetime
import os
import time

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from tp_mlops2.auth import decodificar_token

load_dotenv()

API_KEY = os.environ["MODELO_API_KEY"]
LIMITE_POR_MINUTO = 60

AUDITORIA: list[dict] = []


def auditar(usuario: str, accion: str, detalle: str) -> None:
    AUDITORIA.append(
        {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
            "usuario": usuario,
            "accion": accion,
            "detalle": detalle,
        }
    )


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def auth(k: str = Depends(_api_key_header)) -> str:
    if k != API_KEY:
        auditar("desconocido", "auth_fallida", "API key invalida o ausente")
        raise HTTPException(401, "no autorizado")
    return "cliente"


_hits: dict[str, list[float]] = {}


def rate_limit(cliente: str = Depends(auth)) -> str:
    now = time.time()
    ventana = [t for t in _hits.get(cliente, []) if now - t < 60]
    if len(ventana) >= LIMITE_POR_MINUTO:
        auditar(cliente, "rate_limit_excedido", f"{len(ventana)} requests en 60s")
        raise HTTPException(429, "demasiadas requests, esperá un minuto")
    ventana.append(now)
    _hits[cliente] = ventana
    return cliente


_bearer_scheme = HTTPBearer(auto_error=False)


def requiere_rol(*roles_permitidos: str):
    def dependencia(cred: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> dict:
        if cred is None:
            auditar("desconocido", "auth_fallida", "sin token Bearer")
            raise HTTPException(401, "falta token Bearer")
        try:
            payload = decodificar_token(cred.credentials)
        except jwt.PyJWTError as e:
            auditar("desconocido", "auth_fallida", f"token invalido: {e}")
            raise HTTPException(401, "token invalido o vencido")
        if payload.get("rol") not in roles_permitidos:
            auditar(
                payload.get("sub", "?"),
                "autorizacion_fallida",
                f"rol {payload.get('rol')} insuficiente",
            )
            raise HTTPException(403, "rol insuficiente")
        return payload

    return dependencia
