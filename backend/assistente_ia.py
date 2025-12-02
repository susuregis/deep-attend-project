"""
Assistente de IA com LangChain usando FAISS
Usa UnstructuredFileLoader para extração universal de arquivos
"""
import os
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar logger
logger = logging.getLogger(__name__)

try:
    # Importações do LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document

    # Carregador universal de documentos
    from langchain_community.document_loaders import UnstructuredFileLoader

    LANGCHAIN_DISPONIVEL = True
except ImportError as e:
    logger.warning(f"LangChain não disponível: {e}")
    LANGCHAIN_DISPONIVEL = False

from database import AIContext, AIConversation

# Chave API do Groq 
CHAVE_API_GROQ = os.getenv("GROQ_API_KEY")
if not CHAVE_API_GROQ:
    logger.warning("GROQ_API_KEY não configurada. Assistente de IA não funcionará.")
    logger.warning("Configure a variável de ambiente GROQ_API_KEY para habilitar o assistente.")

# Diretório para armazenar vectorstores
DIRETORIO_VECTORSTORE = "./faiss_vectorstores"
os.makedirs(DIRETORIO_VECTORSTORE, exist_ok=True)


# Configuração do text splitter
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configuração do LLM Groq
MODELO_LLM = "llama-3.3-70b-versatile"
TEMPERATURA_LLM = 0.7
MAX_TOKENS_RESPOSTA = 1024

# Configuração do retriever
NUM_DOCUMENTOS_RELEVANTES = 4
LIMITE_HISTORICO_CHAT = 10
MAX_FONTES_EXIBIDAS = 3

# Modelo de embeddings
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cache do modelo de embeddings 
_modelo_embeddings = None


def obter_embeddings():
    """
    Retorna modelo de embeddings usando padrão singleton.

    O modelo é carregado uma única vez e reutilizado em todas as chamadas
    subsequentes para economizar memória e tempo de inicialização.

    Returns:
        HuggingFaceEmbeddings: Modelo de embeddings multilíngue

    Note:
        Usa o modelo 'paraphrase-multilingual-MiniLM-L12-v2' otimizado
        para português e outros idiomas.
    """
    global _modelo_embeddings
    if _modelo_embeddings is None:
        logger.info("Carregando modelo de embeddings...")
        _modelo_embeddings = HuggingFaceEmbeddings(
            model_name=MODELO_EMBEDDINGS,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Modelo de embeddings carregado!")
    return _modelo_embeddings


def carregar_documento_arquivo(caminho_arquivo: str, nome_arquivo: str) -> List[Document]:
    """
    Carrega documento usando UnstructuredFileLoader (detecção automática de tipo).

    Utiliza o carregador universal do LangChain que detecta automaticamente
    o tipo de arquivo e extrai seu conteúdo textual.

    Args:
        caminho_arquivo: Caminho completo do arquivo no sistema
        nome_arquivo: Nome original do arquivo (para metadados)

    Returns:
        Lista de Document objects do LangChain com conteúdo e metadados

    Note:
        Tipos suportados: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XML,
        CSV, RTF, ODT, e muitos outros formatos.
    """
    try:
        extensao = nome_arquivo.lower().split('.')[-1]

        # Usa carregadores específicos (mais estáveis que UnstructuredFileLoader)
        if extensao == 'pdf':
            from langchain_community.document_loaders import PyPDFLoader
            carregador = PyPDFLoader(caminho_arquivo)
        elif extensao in ['docx', 'doc']:
            from langchain_community.document_loaders import Docx2txtLoader
            carregador = Docx2txtLoader(caminho_arquivo)
        elif extensao in ['txt', 'md']:
            from langchain_community.document_loaders import TextLoader
            carregador = TextLoader(caminho_arquivo, encoding='utf-8')
        else:
            # Para outros tipos não suportados
            raise ValueError(f"Tipo de arquivo não suportado: {extensao}. Use PDF, DOCX ou TXT")

        documentos = carregador.load()

        # Adiciona metadados
        for doc in documentos:
            doc.metadata["filename"] = nome_arquivo
            doc.metadata["source"] = nome_arquivo

        logger.info(f"{nome_arquivo}: {len(documentos)} documento(s) carregado(s)")
        return documentos

    except Exception as e:
        logger.error(f"Erro ao carregar {nome_arquivo}: {e}")
        # Retorna documento vazio com mensagem de erro
        return [Document(
            page_content=f"[Erro ao processar {nome_arquivo}: {str(e)}]",
            metadata={"source": nome_arquivo, "error": str(e)}
        )]


def criar_vectorstore(
    id_sessao: int,
    arquivos: Optional[List[tuple]] = None,
    conteudo_texto: Optional[str] = None
) -> FAISS:
    """
    Cria vectorstore FAISS a partir de arquivos e/ou texto

    Args:
        id_sessao: ID da sessão
        arquivos: Lista de tuplas (caminho_arquivo, nome_arquivo)
        conteudo_texto: Texto adicional

    Returns:
        FAISS vectorstore
    """
    if not LANGCHAIN_DISPONIVEL:
        raise ImportError("LangChain não está instalado")

    todos_documentos = []

    # Adiciona texto manual
    if conteudo_texto and conteudo_texto.strip():
        doc = Document(
            page_content=conteudo_texto,
            metadata={"source": "texto_manual", "filename": "Texto fornecido"}
        )
        todos_documentos.append(doc)

    # Processa arquivos com UnstructuredFileLoader
    if arquivos:
        for caminho_arquivo, nome_arquivo in arquivos:
            try:
                logger.info(f"Processando {nome_arquivo}...")
                documentos = carregar_documento_arquivo(caminho_arquivo, nome_arquivo)
                todos_documentos.extend(documentos)
            except Exception as e:
                logger.error(f"Erro ao processar {nome_arquivo}: {e}")
                continue

    if not todos_documentos:
        raise ValueError("Nenhum documento foi processado")

    logger.info(f"Total de documentos: {len(todos_documentos)}")

    # Divide em chunks
    divisor_texto = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    pedacos = divisor_texto.split_documents(todos_documentos)
    logger.info(f"Documentos divididos em {len(pedacos)} chunks")

    # Cria vectorstore FAISS
    embeddings = obter_embeddings()
    vectorstore = FAISS.from_documents(pedacos, embeddings)

    # Salva no disco
    caminho_vectorstore = os.path.join(DIRETORIO_VECTORSTORE, f"session_{id_sessao}")
    vectorstore.save_local(caminho_vectorstore)
    logger.info(f"Vectorstore salvo em {caminho_vectorstore}")

    return vectorstore


def carregar_vectorstore(id_sessao: int) -> Optional[FAISS]:
    """Carrega vectorstore do disco"""
    if not LANGCHAIN_DISPONIVEL:
        return None

    caminho_vectorstore = os.path.join(DIRETORIO_VECTORSTORE, f"session_{id_sessao}")

    if not os.path.exists(caminho_vectorstore):
        return None

    try:
        embeddings = obter_embeddings()
        vectorstore = FAISS.load_local(
            caminho_vectorstore,
            embeddings,
            allow_dangerous_deserialization=True  # Necessário para FAISS
        )
        logger.info(f"Vectorstore carregado de {caminho_vectorstore}")
        return vectorstore
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore: {e}")
        return None


def salvar_contexto_ia_langchain(
    db: Session,
    id_sessao: int,
    conteudo_texto: Optional[str] = None,
    arquivos: Optional[List[tuple]] = None
) -> AIContext:
    """
    Salva contexto de IA com LangChain

    Args:
        db: Sessão do banco
        id_sessao: ID da sessão
        conteudo_texto: Texto manual
        arquivos: Lista de arquivos (caminho_arquivo, nome_arquivo)

    Returns:
        Objeto AIContext
    """
    # Cria vectorstore
    vectorstore = criar_vectorstore(id_sessao, arquivos, conteudo_texto)

    # Cria resumo para o banco
    partes_resumo = []
    if conteudo_texto:
        partes_resumo.append(f"Texto: {len(conteudo_texto)} caracteres")
    if arquivos:
        partes_resumo.append(f"Arquivos: {', '.join([f[1] for f in arquivos])}")

    resumo = " | ".join(partes_resumo) if partes_resumo else "Materiais processados"

    # Salva/atualiza no banco
    contexto_existente = db.query(AIContext).filter(
        AIContext.session_id == id_sessao
    ).first()

    if contexto_existente:
        contexto_existente.context_text = resumo
        contexto_existente.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(contexto_existente)
        return contexto_existente
    else:
        novo_contexto = AIContext(
            session_id=id_sessao,
            context_text=resumo
        )
        db.add(novo_contexto)
        db.commit()
        db.refresh(novo_contexto)
        return novo_contexto


def obter_contexto_ia(db: Session, id_sessao: int) -> Optional[AIContext]:
    """Busca contexto de IA"""
    return db.query(AIContext).filter(
        AIContext.session_id == id_sessao
    ).first()


def salvar_mensagem_conversa(
    db: Session,
    id_sessao: int,
    id_usuario: int,
    papel: str,
    mensagem: str
) -> AIConversation:
    conversa = AIConversation(
        session_id=id_sessao,
        user_id=id_usuario,
        role=papel,
        message=mensagem
    )
    db.add(conversa)
    db.commit()
    db.refresh(conversa)
    return conversa


def obter_historico_conversa(
    db: Session,
    id_sessao: int,
    id_usuario: int,
    limite: int = 50
) -> List[AIConversation]:
    """Busca histórico de conversas"""
    return db.query(AIConversation).filter(
        AIConversation.session_id == id_sessao,
        AIConversation.user_id == id_usuario
    ).order_by(AIConversation.timestamp.desc()).limit(limite).all()


def perguntar_assistente_ia_langchain(
    db: Session,
    id_sessao: int,
    id_usuario: int,
    pergunta: str
) -> Dict:
    """
    Pergunta para o assistente de IA com LangChain

    Args:
        db: Sessão do banco
        id_sessao: ID da sessão
        id_usuario: ID do usuário
        pergunta: Pergunta

    Returns:
        Dict com resposta
    """
    if not LANGCHAIN_DISPONIVEL:
        return {
            "success": False,
            "error": "LangChain não está disponível no sistema"
        }

    # Verifica se tem contexto
    contexto_ia = obter_contexto_ia(db, id_sessao)
    if not contexto_ia:
        return {
            "success": False,
            "error": "Nenhum material foi fornecido pelo professor para esta aula."
        }

    # Carrega vectorstore
    vectorstore = carregar_vectorstore(id_sessao)
    if not vectorstore:
        return {
            "success": False,
            "error": "Erro ao carregar materiais. Por favor, peça ao professor para reenviar."
        }

    try:
        # Inicializa LLM Groq
        llm = ChatGroq(
            temperature=TEMPERATURA_LLM,
            model_name=MODELO_LLM,
            groq_api_key=CHAVE_API_GROQ,
            max_tokens=MAX_TOKENS_RESPOSTA
        )

        # Busca histórico de conversas
        historico = obter_historico_conversa(db, id_sessao, id_usuario, limite=LIMITE_HISTORICO_CHAT)
        historico_chat = []
        for conv in reversed(historico):
            if conv.role == "user":
                historico_chat.append((conv.message, ""))
            elif historico_chat and conv.role == "assistant":
                # Completa o último par
                historico_chat[-1] = (historico_chat[-1][0], conv.message)

        # Cria chain conversacional
        cadeia_qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": NUM_DOCUMENTOS_RELEVANTES}),
            return_source_documents=True,
            verbose=False
        )

        # Faz pergunta
        resultado = cadeia_qa({
            "question": pergunta,
            "chat_history": historico_chat
        })

        resposta = resultado['answer']

        # Extrai fontes
        fontes = []
        if resultado.get('source_documents'):
            fontes_vistas = set()
            for doc in resultado['source_documents'][:MAX_FONTES_EXIBIDAS]:
                fonte = doc.metadata.get('filename', 'Material da aula')
                if fonte not in fontes_vistas:
                    fontes.append(fonte)
                    fontes_vistas.add(fonte)

        # Salva conversa
        salvar_mensagem_conversa(db, id_sessao, id_usuario, "user", pergunta)
        salvar_mensagem_conversa(db, id_sessao, id_usuario, "assistant", resposta)

        return {
            "success": True,
            "question": pergunta,
            "answer": resposta,
            "sources": fontes,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Erro no LangChain: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Erro ao processar pergunta: {str(e)}"
        }


def limpar_historico_conversa(db: Session, id_sessao: int, id_usuario: int) -> bool:
    """Limpa histórico de conversas"""
    try:
        db.query(AIConversation).filter(
            AIConversation.session_id == id_sessao,
            AIConversation.user_id == id_usuario
        ).delete()
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Erro ao limpar histórico: {e}")
        return False
