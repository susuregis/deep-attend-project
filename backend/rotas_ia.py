"""
Rotas da API para Assistente de IA com suporte ao LangChain
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import tempfile
import os
import logging

# Limite de tamanho de arquivo: 10MB (para evitar segfault com PDFs grandes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB em bytes

from database import get_db, User, ClassSession, SessionParticipant
from auth import get_current_user, get_current_teacher

# Importa funções do assistente de IA
from assistente_ia import (
    salvar_contexto_ia_langchain,
    obter_contexto_ia,
    perguntar_assistente_ia_langchain,
    obter_historico_conversa,
    limpar_historico_conversa,
    LANGCHAIN_DISPONIVEL
)

router = APIRouter(prefix="/ai/v2", tags=["Assistente de IA V2"])
logger = logging.getLogger(__name__)

logger.info("Rotas do Assistente de IA carregadas com LangChain!")


@router.post("/context/{room_code}")
async def definir_contexto_ia_v2(
    room_code: str,
    files: List[UploadFile] = File(default=[]),
    context: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db)
):
    """
    Upload de contexto de IA com arquivos e/ou texto (apenas professores)
    Suporta: PDF, DOCX, XLSX, PPTX, TXT, MD
    """
    # Verifica se a sessão existe (ativa ou inativa)
    sessao = db.query(ClassSession).filter(
        ClassSession.room_code == room_code
    ).first()

    # Se a sessão não existe, cria automaticamente
    if not sessao:
        sessao = ClassSession(
            room_code=room_code,
            name=f"Aula {room_code}",
            teacher_id=current_user.id,
            is_active=True
        )
        db.add(sessao)
        db.commit()
        db.refresh(sessao)
    else:
        # Se a sessão existe mas está inativa, reativa ela
        if not sessao.is_active:
            sessao.is_active = True
            db.commit()
            db.refresh(sessao)

    # Verifica se o professor é dono da sessão
    if sessao.teacher_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Você não tem permissão para modificar esta sessão"
        )

    if not files and not context:
        raise HTTPException(
            status_code=400,
            detail="Forneça pelo menos um arquivo ou texto"
        )

    # Processa arquivos enviados
    arquivos_temp = []
    todo_texto = ""

    try:
        # Adiciona conteúdo de texto manual primeiro
        if context and context.strip():
            todo_texto = f"=== Texto Fornecido ===\n{context}\n\n"

        # Processa arquivos enviados
        for file in files:
            # Lê conteúdo do arquivo
            conteudo_bytes = await file.read()

            # Valida tamanho do arquivo
            tamanho_mb = len(conteudo_bytes) / (1024 * 1024)
            if len(conteudo_bytes) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Arquivo '{file.filename}' muito grande ({tamanho_mb:.1f}MB). Máximo permitido: 10MB"
                )

            # Salva em arquivo temporário
            sufixo = f".{file.filename.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
                tmp.write(conteudo_bytes)
                arquivos_temp.append((tmp.name, file.filename))

            logger.info(f"Arquivo recebido: {file.filename} ({tamanho_mb:.2f}MB)")

        # Usa LangChain se disponível
        if LANGCHAIN_DISPONIVEL:
            logger.info("Usando LangChain com FAISS vectorstore")
            logger.info(f"Processando {len(arquivos_temp)} arquivo(s)...")

            # Processamento pode demorar - executar em thread separada
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()

            contexto_ia = await loop.run_in_executor(
                executor,
                salvar_contexto_ia_langchain,
                db,
                sessao.id,
                context if context and context.strip() else None,
                arquivos_temp if arquivos_temp else None
            )
        else:
            raise ImportError("LangChain não está disponível no sistema")

        return {
            "success": True,
            "message": "Materiais processados com sucesso",
            "files_processed": len(arquivos_temp),
            "context_id": contexto_ia.id,
            "updated_at": contexto_ia.updated_at.isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao processar arquivos: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivos: {str(e)}")
    finally:
        # Limpeza de arquivos temporários
        for caminho_temp, _ in arquivos_temp:
            try:
                os.unlink(caminho_temp)
            except OSError as e:
                logger.warning(f"Erro ao deletar arquivo temporário {caminho_temp}: {e}")


@router.get("/context/{room_code}")
async def verificar_contexto_ia_v2(
    room_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verifica se existe contexto de IA"""
    sessao = db.query(ClassSession).filter(
        ClassSession.room_code == room_code,
        ClassSession.is_active == True
    ).first()

    if not sessao:
        # Retorna false se a sessão não existe ainda
        return {
            "has_context": False,
            "context_summary": None,
            "updated_at": None
        }

    # Qualquer pessoa em uma sessão ativa pode verificar se o contexto existe
    # Sem verificação de permissão estrita aqui - eles apenas veem se materiais estão disponíveis
    contexto_ia = obter_contexto_ia(db, sessao.id)

    return {
        "has_context": contexto_ia is not None,
        "context_summary": contexto_ia.context_text if contexto_ia else None,
        "updated_at": contexto_ia.updated_at.isoformat() if contexto_ia else None
    }


@router.post("/ask")
async def perguntar_ia_v2(
    room_code: str = Form(...),
    question: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Faz pergunta ao assistente de IA"""
    print(f"[DEBUG] /ai/v2/ask - room_code: {room_code}, question: {question[:50]}...")

    sessao = db.query(ClassSession).filter(
        ClassSession.room_code == room_code,
        ClassSession.is_active == True
    ).first()

    if not sessao:
        print(f"[DEBUG] Sessão não encontrada: {room_code}")
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    # Verifica permissão
    eh_professor = sessao.teacher_id == current_user.id
    eh_participante = db.query(SessionParticipant).filter(
        SessionParticipant.session_id == sessao.id,
        SessionParticipant.user_id == current_user.id
    ).first() is not None

    print(f"[DEBUG] eh_professor: {eh_professor}, eh_participante: {eh_participante}")

    if not (eh_professor or eh_participante):
        print(f"[DEBUG] Acesso negado para user_id: {current_user.id}")
        raise HTTPException(status_code=403, detail="Acesso negado")

    # Pergunta à IA
    print(f"[DEBUG] Chamando perguntar_assistente_ia_langchain...")
    resultado = perguntar_assistente_ia_langchain(db, sessao.id, current_user.id, question)
    print(f"[DEBUG] Resultado: success={resultado.get('success')}, error={resultado.get('error')}")

    if not resultado.get("success"):
        raise HTTPException(
            status_code=400,
            detail=resultado.get("error", "Erro ao processar pergunta")
        )

    return resultado


@router.get("/history/{room_code}")
async def obter_historico_ia_v2(
    room_code: str,
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtém histórico de conversa"""
    sessao = db.query(ClassSession).filter(
        ClassSession.room_code == room_code
    ).first()

    if not sessao:
        # Retorna histórico vazio se a sessão não existe ainda
        return {
            "room_code": room_code,
            "user_id": current_user.id,
            "conversation": []
        }

    historico = obter_historico_conversa(db, sessao.id, current_user.id, limit)

    return {
        "room_code": room_code,
        "user_id": current_user.id,
        "conversation": [
            {
                "role": conv.role,
                "message": conv.message,
                "timestamp": conv.timestamp.isoformat()
            }
            for conv in reversed(historico)
        ]
    }


@router.delete("/history/{room_code}")
async def deletar_historico_ia_v2(
    room_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Limpa histórico de conversa"""
    sessao = db.query(ClassSession).filter(
        ClassSession.room_code == room_code
    ).first()

    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    sucesso = limpar_historico_conversa(db, sessao.id, current_user.id)

    if sucesso:
        return {"success": True, "message": "Histórico limpo com sucesso"}
    else:
        raise HTTPException(status_code=500, detail="Erro ao limpar histórico")
