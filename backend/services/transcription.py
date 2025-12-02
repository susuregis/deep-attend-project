import logging
from functools import lru_cache

import whisper

logger = logging.getLogger("attention-api.transcription")


@lru_cache()
def get_whisper_model(model_size: str = "base"):
    logger.info("Carregando modelo Whisper (%s)...", model_size)
    return whisper.load_model(model_size)

