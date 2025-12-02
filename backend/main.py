"""
Backend API para o sistema de sala de aula com atenção e IA.
"""
import logging
from pathlib import Path

import socketio
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

from api import (
    auth_routes,
    dashboard_routes,
    prediction_routes,
    root as root_routes,
    session_routes,
    transcription_routes,
)
from core.config import get_allowed_origins, get_model_config
from database import init_db
from realtime.socket_handlers import create_socket_server
from services.attention_model import load_attention_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attention-api")

MODEL_CONFIG = get_model_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path(__file__).resolve().parent / "model.pth"
MODEL, TRANSFORM = load_attention_model(MODEL_CONFIG, WEIGHTS_PATH, DEVICE)

app = FastAPI(
    title="Detector de Atenção - API",
    description="API para detecção de atenção de alunos com autenticação e dashboard",
    version="2.0.0",
)

allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin"],
    expose_headers=["Content-Length", "Content-Type"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

app.include_router(root_routes.router)
app.include_router(auth_routes.router)
app.include_router(session_routes.router)
app.include_router(dashboard_routes.router)
app.include_router(prediction_routes.router)
app.include_router(transcription_routes.router)

try:
    from rotas_ia import router as roteador_ia_v2

    app.include_router(roteador_ia_v2)
    logger.info("Assistente de IA v2 (LangChain) carregado com sucesso")
except ImportError as exc:
    logger.warning("Assistente de IA v2 não disponível: %s", exc)
    logger.warning("Instale as dependências do LangChain para habilitar IA v2")


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Banco de dados inicializado")


app.state.model = MODEL
app.state.transform = TRANSFORM
app.state.model_config = MODEL_CONFIG
app.state.device = DEVICE
app.state.request_count = 0

sio = create_socket_server(allowed_origins)
app.state.sio = sio
socket_app = socketio.ASGIApp(sio, app)


if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)

