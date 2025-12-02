import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("attention-api.config")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


@lru_cache()
def get_model_config() -> Dict[str, Any]:
    """
    Carrega configurações do modelo (apenas uma vez).
    Valida que o arquivo existe e contém dados válidos.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {CONFIG_PATH}\n"
            "Certifique-se de que o arquivo config.json existe no diretório backend/"
        )

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        if not isinstance(config, dict):
            raise ValueError("config.json deve conter um objeto JSON")

        logger.info("Configuração do modelo carregada com sucesso")
        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar config.json: {e}")


def get_allowed_origins() -> List[str]:
    """
    Lê origens permitidas da variável de ambiente ALLOWED_ORIGINS.
    Retorna lista de origens validadas (URLs bem formatadas).
    """
    origins_env = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )

    origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]

    # Validação básica de URL
    validated_origins = []
    for origin in origins:
        if not origin.startswith(("http://", "https://")):
            logger.warning(
                "Origem CORS ignorada (deve começar com http:// ou https://): %s",
                origin
            )
            continue
        validated_origins.append(origin)

    if not validated_origins:
        logger.warning(
            "Nenhuma origem CORS válida configurada! "
            "Usando padrões de desenvolvimento."
        )
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    logger.info("Origens CORS permitidas: %s", ", ".join(validated_origins))
    return validated_origins


def get_environment() -> str:
    """
    Retorna o ambiente de execução: development, staging ou production.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    valid_environments = ["development", "staging", "production"]

    if env not in valid_environments:
        logger.warning(
            "ENVIRONMENT inválido '%s'. Usando 'development'. "
            "Valores válidos: %s",
            env,
            ", ".join(valid_environments)
        )
        return "development"

    return env


def is_production() -> bool:
    """Retorna True se estiver em ambiente de produção."""
    return get_environment() == "production"

