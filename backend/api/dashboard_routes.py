from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_teacher
from database import AttentionMetric, ClassSession, User, get_db
from realtime.socket_handlers import room_attention_data, rooms

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/session/{room_code}")
async def get_session_dashboard(
    room_code: str,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    session = db.query(ClassSession).filter(ClassSession.room_code == room_code).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    attention_data = room_attention_data.get(room_code, {})
    room_users = rooms.get(room_code, {})

    participants_info = []
    for user_id, user_info in room_users.items():
        user_attention = attention_data.get(user_id, {})
        participants_info.append(
            {
                "id": user_id,
                "name": user_info.get("name", "Anônimo"),
                "is_attentive": user_attention.get("is_attentive"),
                "confidence": user_attention.get("confidence", 0),
                "prob_attentive": user_attention.get("prob_attentive", 0),
                "prob_inattentive": user_attention.get("prob_inattentive", 0),
                "last_update": user_attention.get("timestamp"),
            }
        )

    total = len(participants_info)
    attentive_count = sum(
        1 for p in participants_info if p["is_attentive"] is True
    )
    inattentive_count = sum(
        1 for p in participants_info if p["is_attentive"] is False
    )

    return {
        "session": {
            "id": session.id,
            "name": session.name,
            "room_code": session.room_code,
            "started_at": session.started_at.isoformat(),
        },
        "stats": {
            "total_participants": total,
            "attentive": attentive_count,
            "inattentive": inattentive_count,
            "attention_rate": round((attentive_count / total * 100) if total else 0, 1),
        },
        "participants": participants_info,
    }


@router.get("/session/{room_code}/history")
async def get_session_history(
    room_code: str,
    minutes: int = Query(30, description="Minutes of history to retrieve"),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    session = db.query(ClassSession).filter(ClassSession.room_code == room_code).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    since = datetime.utcnow() - timedelta(minutes=minutes)
    metrics = (
        db.query(
            func.strftime("%Y-%m-%d %H:%M", AttentionMetric.timestamp).label("minute"),
            func.avg(AttentionMetric.prob_attentive).label("avg_attentive"),
            func.count(AttentionMetric.id).label("count"),
        )
        .filter(
            AttentionMetric.session_id == session.id,
            AttentionMetric.timestamp >= since,
        )
        .group_by(func.strftime("%Y-%m-%d %H:%M", AttentionMetric.timestamp))
        .order_by("minute")
        .all()
    )

    return {
        "history": [
            {
                "minute": m.minute,
                "avg_attention": round(float(m.avg_attentive) * 100, 1),
                "data_points": m.count,
            }
            for m in metrics
        ]
    }


@router.get("/user/{user_id}/stats")
async def get_user_stats(
    user_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    total_metrics = (
        db.query(AttentionMetric).filter(AttentionMetric.user_id == user_id).count()
    )
    attentive_metrics = (
        db.query(AttentionMetric)
            .filter(
                AttentionMetric.user_id == user_id,
                AttentionMetric.is_attentive.is_(True),
            )
            .count()
    )
    avg_confidence = (
        db.query(func.avg(AttentionMetric.confidence))
        .filter(AttentionMetric.user_id == user_id)
        .scalar()
        or 0
    )

    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
        },
        "stats": {
            "total_measurements": total_metrics,
            "attentive_count": attentive_metrics,
            "attention_rate": round(
                (attentive_metrics / total_metrics * 100) if total_metrics else 0, 1
            ),
            "avg_confidence": round(float(avg_confidence) * 100, 1),
        },
    }

