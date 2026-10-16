import hashlib
import hmac
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

SECRETO_PRODUCTOR = os.environ["STREAM_HMAC_SECRET"].encode()
UMBRAL_ALERTA_DLQ = 2  # eventos en cuarentena por ventana para disparar alerta


def firmar(evento: dict) -> str:
    cuerpo = json.dumps(evento, sort_keys=True).encode()
    return hmac.new(SECRETO_PRODUCTOR, cuerpo, hashlib.sha256).hexdigest()


class EventoDemandaValido(BaseModel):
    """Esquema real de un evento de demanda, con los rangos fisicamente
    plausibles observados en notebooks/01_dataset_EDA.ipynb."""

    timestamp: str
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    temp_Madrid: float = Field(ge=-15, le=50)
    temp_Barcelona: float = Field(ge=-15, le=50)
    temp_Valencia: float = Field(ge=-15, le=50)
    temp_Seville: float = Field(ge=-15, le=50)
    temp_Bilbao: float = Field(ge=-15, le=50)
    load_lag_24h: float = Field(ge=0, le=60000)
    load_lag_168h: float = Field(ge=0, le=60000)


DLQ: list[dict] = []
AUDITORIA_PROCEDENCIA: list[dict] = []


def consumir_validado(evento: dict, firma: str) -> dict | None:
    """Verifica firma HMAC y esquema/rangos. Si falla cualquiera, el evento
    va a la DLQ y devuelve None; si pasa, devuelve el evento validado."""
    if not hmac.compare_digest(firma, firmar(evento)):
        DLQ.append({"evento": evento, "motivo": "firma HMAC invalida"})
        return None

    try:
        validado = EventoDemandaValido(**evento)
    except ValidationError as e:
        DLQ.append({"evento": evento, "motivo": f"esquema invalido: {e.errors()[0]['msg']}"})
        return None

    AUDITORIA_PROCEDENCIA.append(
        {"timestamp": validado.timestamp, "firma": firma[:12] + "...", "hash_cuerpo": firma}
    )
    return validado.model_dump()


def chequear_alerta_dlq(dlq_ventana: list[dict]) -> bool:
    return len(dlq_ventana) >= UMBRAL_ALERTA_DLQ
