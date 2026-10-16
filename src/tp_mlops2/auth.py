import datetime
import os

import jwt
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.environ["MODELO_JWT_SECRET"]


def crear_token(usuario: str, rol: str, minutos: int = 15) -> str:
    payload = {
        "sub": usuario,
        "rol": rol,
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=minutos),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decodificar_token(token: str) -> dict:
    """Lanza jwt.PyJWTError si el token es invalido o esta vencido."""
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
