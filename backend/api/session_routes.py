import random
import string
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from auth import get_current_teacher, get_current_user
from database import ClassSession, SessionParticipant, User, get_db

router = APIRouter(prefix="/sessions", tags=["Sessões"])


def _generate_room_code(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


@router.post("/create")
async def create_session(
    name: str = Query(..., description="Nome da sessão/aula"),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    room_code = _generate_room_code()
    while db.query(ClassSession).filter(ClassSession.room_code == room_code).first():
        room_code = _generate_room_code()

    session = ClassSession(
        room_code=room_code,
        name=name,
        teacher_id=current_user.id,
        is_active=True,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "success": True,
        "session": {
            "id": session.id,
            "room_code": session.room_code,
            "name": session.name,
            "started_at": session.started_at.isoformat(),
        },
    }


@router.get("/active")
async def get_active_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role in ["teacher", "admin"]:
        sessions = db.query(ClassSession).filter(
            ClassSession.teacher_id == current_user.id,
            ClassSession.is_active.is_(True),
        ).all()
    else:
        participations = db.query(SessionParticipant).filter(
            SessionParticipant.user_id == current_user.id,
            SessionParticipant.left_at.is_(None),
        ).all()
        session_ids = [p.session_id for p in participations] or [0]
        sessions = db.query(ClassSession).filter(
            ClassSession.id.in_(session_ids),
            ClassSession.is_active.is_(True),
        ).all()

    from realtime.socket_handlers import rooms  # lazy import

    return {
        "sessions": [
            {
                "id": s.id,
                "room_code": s.room_code,
                "name": s.name,
                "started_at": s.started_at.isoformat(),
                "participant_count": len(rooms.get(s.room_code, {})),
            }
            for s in sessions
        ]
    }


@router.post("/{room_code}/end")
async def end_session(
    room_code: str,
    request: Request,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ClassSession)
        .filter(
            ClassSession.room_code == room_code,
            ClassSession.teacher_id == current_user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    session.is_active = False
    session.ended_at = datetime.utcnow()
    db.query(SessionParticipant).filter(
        SessionParticipant.session_id == session.id,
        SessionParticipant.left_at.is_(None),
    ).update({"left_at": datetime.utcnow()})
    db.commit()

    sio = getattr(request.app.state, "sio")
    await sio.emit("session-ended", {"room_code": room_code}, room=room_code)
    return {"success": True, "message": "Sessão encerrada"}

