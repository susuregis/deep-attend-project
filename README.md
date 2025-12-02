### Sistema de Sala de Aula Virtual com IA

Sistema completo de videoconferência educacional com detecção de atenção e assistente de IA inteligente.

#### Treinamento do modelo de visão computacional com deep learning.

#### Técnicas ultilizadas para o processamento das imagens:

- Redimensionamento de Imagens: Ajusta todas as imagens para o mesmo tamanho.
- Normalização : Ajusta os valores dos pixels para média 0 e desvio padrão 1

- *Data Augmentation (Aumento de Dados)* : 
  - Random Horizontal Flip - Espelha a imagem horizontalmente com 50% de probabilidade
  - Color Jitter (Variação de Cor) - Varia aleatoriamente brilho e contraste
  - Random Rotation - Rotaciona a imagem aleatoriamente até 10 graus
  -  Conversão para Tensor - Converte imagem PIL/NumPy para tensor PyTorch


-  *Mapeamento de Classes (Class Remapping)* :  Reduz múltiplas classes originais para 2 classes. 
   - Antes : Classes: [Frontal, Left, Right, Up, Down] (5 classes)
   - Depois : Classes: [Atento, Desatento] (2 classes)

- Carregamento de Imagens com OpenCV: Lê imagem e converte de BGR para RGB

#### Técnicas para o treinamento 


- Transfer Learning (Aprendizado por Transferência) :  Usar modelo pré-treinado no ImageNet paar iniciar o treinamento.

- Mixed Precision Training : Usa FP16 (16 bits) em vez de FP32 (32 bits) quando possível

- Batch Training : Processa múltiplas imagens simultaneamente

- Early Stopping (Implícito) : Salva apenas o modelo com melhor desempenho na validação

- Gradient Accumulation : Acumula gradientes antes de atualizar pesos

- Pin Memory :  Mantém dados na memória paginada para transferência mais rápida à GPU

#### Técnicas de regularização 

- Dropout :  Durante treinamento, desliga aleatoriamente 30% dos neurônios

- Weight Decay : Penaliza pesos muito grandes

- Batch Normalization  :  Normaliza ativações entre camadas

- Class Weighting : Ajusta importância de cada classe.

#### Técnicas de avaliação 


- Cross-Validation Split (via Train/Val/Test Split) : Train : 70%, Val : 15% e Test : 15%

- Test-Time Augmentation (TTA) : Testa múltiplas versões de cada imagem

#### Métricas ultilizadas :
- Acurácia     
- Precisão     
- Recall       
- F1-Score 

Conjunto        Acurácia     Precisão     Recall       F1-Score    
---------------------------------------------------------------
Treino               71.68%      74.40%      71.68%      72.47%
Validação            60.66%      64.97%      60.66%      61.81%
Teste                71.94%      73.34%      71.94%      72.27%


#### Funcionalidades do sistema

#### Videoconferência
- Vídeo e áudio em tempo real usando WebRTC
- Salas virtuais com códigos únicos
- Chat em tempo real
- Transcrição automática de fala

#### Detecção de Atenção (Alunos)
- Análise automática de atenção através da câmera
- Classificação: Atento / Desatento
- Probabilidades em tempo real
- Alertas automáticos para o professor

#### Dashboard do Professor
- Visualização em tempo real da atenção dos alunos
- Alertas quando alunos ficam desatentos
- Estatísticas consolidadas da turma
- Métricas de engajamento

#### Assistente de IA 
- **Interface flutuante moderna** (botão roxo no canto inferior direito)
- **Upload de arquivos**: PDFs, Word, Excel, PowerPoint, Markdown, TXT
- **Sistema RAG (Retrieval Augmented Generation)** com LangChain
- **ChromaDB** para busca semântica em documentos
- **Histórico de conversas** salvo por aluno
- **Contexto independente** por sala de aula
- **Respostas contextuais** baseadas nos materiais do professor

#### Requisitos

#### Backend
- Python 3.8+
- PyTorch
- FastAPI
- SQLite
- Groq API Key
- LangChain e dependências 

#### Frontend
- Node.js 14+
- React
- TypeScript

#### Instalação Rápida

#### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```


#### 2. Frontend

```bash
cd frontend
npm install
```

#### Executando o Sistema

#### 1. Inicie o Backend

```bash
cd backend
python main.py
```

O servidor estará disponível em: http://localhost:8000


#### 2. Inicie o Frontend

```bash
cd frontend
npm start
```

O aplicativo estará disponível em: http://localhost:3000

#### Como Usar

#### Professor

1. **Criar Conta**
   - Registre-se com role "teacher"
   - Faça login

2. **Criar Sala**
   - Clique em "Criar Nova Sala"
   - Compartilhe o código com os alunos

3. **Adicionar Materiais da Aula (IA)**
   - Clique no botão roxo flutuante (canto inferior direito)
   - Clique em "Adicionar Materiais"
   - Faça upload de PDFs, Word, PowerPoint, etc.
   - OU cole texto/material diretamente
   - Clique em "Salvar Materiais"
   - Os alunos serão notificados automaticamente

4. **Monitorar Atenção**
   - Veja em tempo real quem está atento/desatento
   - Receba alertas quando alunos ficarem desatentos
   - Visualize estatísticas da turma

5. **Encerrar Aula**
   - Clique em "Encerrar Aula"
   - Todos os alunos serão desconectados

#### Aluno

1. **Criar Conta**
   - Registre-se com role "student"
   - Faça login

2. **Entrar na Sala**
   - Digite o código fornecido pelo professor
   - Digite seu nome
   - Clique em "Entrar na Sala"

3. **Usar o Assistente de IA**
   - Clique no botão roxo flutuante (canto inferior direito)
   - Digite sua pergunta sobre o material da aula
   - A IA responderá baseada nos materiais do professor
   - Seu histórico é salvo automaticamente

4. **Durante a Aula**
   - Sua câmera deve estar ligada (obrigatório)
   - O sistema detecta automaticamente sua atenção
   - Você pode ver seu próprio status (Atento/Desatento)
   - Use o chat para se comunicar
   - Use o microfone para participar (com transcrição automática)

#### Estrutura do Projeto

```
deploy/
├── backend/
│   ├── main.py                 # Servidor FastAPI principal
│   ├── auth.py                 # Autenticação JWT
│   ├── database.py             # Modelos SQLAlchemy
│   ├── ai_assistant.py         # IA v1 (Groq direto)
│   ├── ai_assistant_v2.py      # IA v2 (LangChain + RAG) ✨ NOVO
│   ├── ai_routes.py            # Endpoints
│   ├── config.json             # Configuração do modelo
│   ├── model.pth               # Modelo de detecção de atenção
│   └── requirements.txt        # Dependências Python
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AIAssistant.tsx      # IA v (tab na sidebar)
│   │   │   ├── AIAssistantV2.tsx    # IA v (floating chat) 
│   │   │   └── AIAssistantV2.css    # Estilos modernos 
│   │   ├── context/
│   │   │   └── AuthContext.tsx      # Context de autenticação
│   │   ├── App.tsx             # Componente principal
│   │   ├── VideoRoom.tsx       # Sala de videoconferência
│   │   ├── Login.tsx           # Página de login
│   │   └── socket.ts           # Cliente Socket.IO
│   └── public/
│       └── models/             # Modelos do face-api.js
│
├── README.md                   # Este arquivo


```

#### Componentes React Principais

#### VideoRoom
Sala principal de videoconferência com:
- Grid de vídeos dos participantes
- Controles de áudio/vídeo
- Chat em tempo real
- Transcrição automática
- Dashboard de atenção (professor)
- **AI Assistant v2 flutuante** 

#### AIAssistantV2 
Assistente de IA com interface flutuante:
- Botão roxo no canto inferior direito
- Janela expansível de chat
- Upload de múltiplos arquivos
- Respostas contextuais com fontes
- Histórico de conversas

#### AuthContext
Context React para gerenciamento de autenticação:
- Login/Logout
- Registro
- Validação de token
- Informações do usuário

#### Segurança

- Autenticação JWT com Bearer tokens
- Senhas hashadas com bcrypt
- CORS configurado
- Validação de permissões (professor vs aluno)
- Tokens expiram após 24 horas

#### Banco de Dados

SQLite com as seguintes tabelas:

- **users** - Usuários do sistema
- **class_sessions** - Salas de aula
- **session_participants** - Participantes das salas
- **attention_metrics** - Métricas de atenção
- **chat_messages** - Mensagens de chat
- **transcripts** - Transcrições de áudio
- **ai_contexts** - Contextos/materiais da IA
- **ai_conversations** - Histórico de conversas com IA



#### Tecnologias Utilizadas

#### Backend
- FastAPI - Framework web assíncrono
- Socket.IO - Comunicação em tempo real
- PyTorch - Deep learning para detecção de atenção
- SQLAlchemy - ORM para banco de dados
- Groq API - LLM para assistente de IA
- **LangChain** - Framework para aplicações LLM 
- **ChromaDB** - Banco de dados vetorial 
- **HuggingFace** - Embeddings multilíngues 
- Whisper - Transcrição de áudio
- JWT - Autenticação
- Bcrypt - Hash de senhas

#### Frontend
- React - Framework UI
- TypeScript - Tipagem estática
- Socket.IO Client - Comunicação em tempo real
- SimplePeer - WebRTC wrapper
- face-api.js - Detecção facial
- Web Speech API - Reconhecimento de voz




