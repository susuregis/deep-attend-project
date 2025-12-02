import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database import ClassSession, Transcript, get_db
from services.transcription import get_whisper_model

router = APIRouter(prefix="/transcribe", tags=["Transcrição"])


@router.post("")
async def transcribe_audio(
    file: UploadFile = File(...),
    room_code: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    temp_input_path = None
    temp_wav_path = None

    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser de áudio")

        contents = await file.read()
        ext_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
        }
        file_ext = ext_map.get(file.content_type, ".webm")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        try:
            from pydub import AudioSegment

            if file_ext == ".webm":
                audio = AudioSegment.from_file(temp_input_path, format="webm")
            elif file_ext == ".ogg":
                audio = AudioSegment.from_ogg(temp_input_path)
            elif file_ext in [".mp3"]:
                audio = AudioSegment.from_mp3(temp_input_path)
            else:
                audio = AudioSegment.from_file(temp_input_path)

            temp_wav_path = temp_input_path.replace(file_ext, ".wav")
            audio.export(temp_wav_path, format="wav")
        except ImportError:
            temp_wav_path = temp_input_path
        except Exception:
            temp_wav_path = temp_input_path

        model = get_whisper_model()
        result = model.transcribe(temp_wav_path, language="pt", fp16=False)
        transcribed_text = result["text"].strip()

        if room_code and user_id and transcribed_text:
            try:
                user_id_int = int(user_id)
                session = (
                    db.query(ClassSession).filter(ClassSession.room_code == room_code).first()
                )
                if session:
                    transcript = Transcript(
                        session_id=session.id,
                        user_id=user_id_int,
                        content=transcribed_text,
                    )
                    db.add(transcript)
                    db.commit()
            except (ValueError, TypeError):
                pass

        return JSONResponse(
            content={
                "success": True,
                "text": transcribed_text,
                "language": result.get("language", "pt"),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            },
        )
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if (
            temp_wav_path
            and temp_wav_path != temp_input_path
            and os.path.exists(temp_wav_path)
        ):
            os.unlink(temp_wav_path)

