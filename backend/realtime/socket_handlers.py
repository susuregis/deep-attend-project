import logging
from datetime import datetime
from typing import Dict

import socketio

from database import ClassSession, SessionParticipant, SessionLocal

logger = logging.getLogger("attention-api.realtime")

rooms: Dict[str, Dict[str, Dict]] = {}
room_attention_data: Dict[str, Dict[str, Dict]] = {}
sio: socketio.AsyncServer


def create_socket_server(allowed_origins):
    global sio
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=allowed_origins,
        logger=True,
        engineio_logger=False,
    )
    _register_handlers()
    return sio


def _register_handlers():
    @sio.event
    async def connect(sid, environ):
        logger.info("Cliente conectado: %s", sid)

    @sio.event
    async def disconnect(sid):
        logger.info("Cliente desconectado: %s", sid)
        for room_code in list(rooms.keys()):
            for user_id in list(rooms[room_code].keys()):
                if rooms[room_code][user_id].get("sid") == sid:
                    del rooms[room_code][user_id]
                    if (
                        room_code in room_attention_data
                        and user_id in room_attention_data[room_code]
                    ):
                        del room_attention_data[room_code][user_id]
                    await sio.emit(
                        "user-left", {"id": user_id}, room=room_code, skip_sid=sid
                    )
                    await _broadcast_dashboard_update(room_code)
                    logger.info("User %s left room %s", user_id, room_code)
            if not rooms[room_code]:
                del rooms[room_code]
                room_attention_data.pop(room_code, None)

    @sio.event
    async def join_room(sid, data):
        room = data.get("room", "default")
        name = data.get("name", "Anonymous")
        user_db_id = data.get("user_id")
        user_id = sid

        logger.info("Usuário %s (%s) entrando na sala %s", name, user_id, room)

        await sio.enter_room(sid, room)
        rooms.setdefault(room, {})
        room_attention_data.setdefault(room, {})

        existing_users = [
            {"id": uid, "name": udata.get("name", "Anonymous")}
            for uid, udata in rooms[room].items()
        ]

        rooms[room][user_id] = {"name": name, "sid": sid, "user_db_id": user_db_id}

        if user_db_id:
            db = SessionLocal()
            try:
                session = (
                    db.query(ClassSession)
                    .filter(ClassSession.room_code == room, ClassSession.is_active.is_(True))
                    .first()
                )
                if session:
                    existing_participant = (
                        db.query(SessionParticipant)
                        .filter(
                            SessionParticipant.session_id == session.id,
                            SessionParticipant.user_id == user_db_id,
                        )
                        .first()
                    )
                    if not existing_participant:
                        participant = SessionParticipant(
                            session_id=session.id,
                            user_id=user_db_id,
                        )
                        db.add(participant)
                        db.commit()
                        logger.info(
                            "Adicionado usuário %s como participante da sessão %s",
                            user_db_id,
                            session.id,
                        )
            except Exception as exc:
                logger.error("Erro ao adicionar participante: %s", exc)
            finally:
                db.close()

        await sio.emit("existing-users", {"users": existing_users}, room=sid)
        await sio.emit(
            "user-joined", {"id": user_id, "name": name}, room=room, skip_sid=sid
        )
        await _broadcast_dashboard_update(room)
        logger.info("Sala %s agora tem %d usuários", room, len(rooms[room]))

    @sio.event
    async def attention_update(sid, data):
        room_code = data.get("room")
        user_name = data.get("name", "Anônimo")

        if room_code and room_code in room_attention_data:
            room_attention_data[room_code][sid] = {
                "is_attentive": data.get("is_attentive"),
                "confidence": data.get("confidence", 0),
                "prob_attentive": data.get("prob_attentive", 0),
                "prob_inattentive": data.get("prob_inattentive", 0),
                "timestamp": datetime.utcnow().isoformat(),
            }

            await sio.emit(
                "student_attention",
                {
                    "odId": sid,
                    "name": user_name,
                    "is_attentive": data.get("is_attentive"),
                    "prob_attentive": data.get("prob_attentive", 0),
                    "prob_inattentive": data.get("prob_inattentive", 0),
                },
                room=room_code,
                skip_sid=sid,
            )
            await _broadcast_dashboard_update(room_code)

    @sio.event
    async def chat_message(sid, data):
        logger.info("Chat message from %s: %s", sid, data)
        room = data.get("room")
        if room and room in rooms:
            await sio.emit("chat_message", data, room=room, skip_sid=sid)
            return

        for room_code, users in rooms.items():
            for user_id, user_info in users.items():
                if user_info.get("sid") == sid:
                    await sio.emit("chat_message", data, room=room_code, skip_sid=sid)
                    return

    @sio.event
    async def offer(sid, data):
        target = data.get("target")
        offer = data.get("offer")
        sender_name = "Anônimo"
        for room_code, users in rooms.items():
            if sid in users:
                sender_name = users[sid].get("name", "Anônimo")
                break
        await sio.emit("offer", {"from": sid, "offer": offer, "name": sender_name}, to=target)

    @sio.event
    async def answer(sid, data):
        target = data.get("target")
        answer = data.get("answer")
        await sio.emit("answer", {"from": sid, "answer": answer}, to=target)

    @sio.event
    async def ice_candidate(sid, data):
        target = data.get("target")
        candidate = data.get("candidate")
        await sio.emit("ice-candidate", {"from": sid, "candidate": candidate}, to=target)

    @sio.event
    async def end_session(sid, data):
        room = data.get("room")
        if room:
            await sio.emit("session-ended", {"room_code": room}, room=room)

    @sio.event
    async def transcript_update(sid, data):
        room = data.get("room")
        payload = {
            "odId": sid,
            "speaker": data.get("speaker"),
            "text": data.get("text"),
            "timestamp": data.get("timestamp"),
        }
        if room and room in rooms:
            await sio.emit("transcript_update", payload, room=room, skip_sid=sid)
            return

        for room_code, users in rooms.items():
            for user_id, user_info in users.items():
                if user_info.get("sid") == sid:
                    await sio.emit("transcript_update", payload, room=room_code, skip_sid=sid)
                    return


async def _broadcast_dashboard_update(room_code: str):
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
            }
        )

    total = len(participants_info)
    attentive = sum(1 for p in participants_info if p["is_attentive"] is True)

    await sio.emit(
        "dashboard-update",
        {
            "room_code": room_code,
            "stats": {
                "total": total,
                "attentive": attentive,
                "inattentive": total - attentive,
                "attention_rate": round((attentive / total * 100) if total else 0, 1),
            },
            "participants": participants_info,
        },
        room=room_code,
    )

