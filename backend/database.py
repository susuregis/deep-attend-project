"""
Database module using SQLite with SQLAlchemy
Stores users, sessions, and attention metrics
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attention_app.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="student")  # "student", "teacher", "admin"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions_created = relationship("ClassSession", back_populates="teacher", foreign_keys="ClassSession.teacher_id")
    attention_metrics = relationship("AttentionMetric", back_populates="user")
    session_participations = relationship("SessionParticipant", back_populates="user")


class ClassSession(Base):
    """Modelo de sessão/aula"""
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True, index=True)
    room_code = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    teacher = relationship("User", back_populates="sessions_created", foreign_keys=[teacher_id])
    participants = relationship("SessionParticipant", back_populates="session")
    attention_metrics = relationship("AttentionMetric", back_populates="session")


class SessionParticipant(Base):
    """Rastreia usuários participando de sessões"""
    __tablename__ = "session_participants"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    left_at = Column(DateTime, nullable=True)

    # Relationships
    session = relationship("ClassSession", back_populates="participants")
    user = relationship("User", back_populates="session_participations")


class AttentionMetric(Base):
    """Armazena métricas de atenção para análise"""
    __tablename__ = "attention_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_attentive = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    prob_attentive = Column(Float, nullable=False)
    prob_inattentive = Column(Float, nullable=False)

    # Relationships
    user = relationship("User", back_populates="attention_metrics")
    session = relationship("ClassSession", back_populates="attention_metrics")


class ChatMessage(Base):
    """Armazena mensagens de chat para sessões"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Transcript(Base):
    """Armazena transcrições para sessões"""
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AIContext(Base):
    """Armazena contexto/materiais de IA para cada sessão"""
    __tablename__ = "ai_contexts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False, unique=True)
    context_text = Column(Text, nullable=False)  # Materiais do professor
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIConversation(Base):
    """Armazena histórico de conversas de IA para estudantes"""
    __tablename__ = "ai_conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False)  
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Criar todas as tabelas"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependência para obter sessão do banco de dados"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
