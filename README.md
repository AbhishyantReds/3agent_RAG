# 🏢 DROMA Electronics Multilingual Support Chatbot

> **Production-ready RAG chatbot with native Hinglish/Telgish support, 5-language AI customer assistant**

---

## ✨ Features

✅ **5 Languages Supported**: English, Hindi, Telugu, Gujarati, Tamil
✅ **Hinglish/Telgish Compatible**: "Return policy kya hai?" works perfectly
✅ **Mixed-Script Support**: Detects and normalizes Roman-script inputs (e.g., "polcy" → "policy")
✅ **Instant Setup**: No ML expertise needed—production-ready in 5 minutes
✅ **Local Embeddings**: CPU-only option (free, no API calls)
✅ **Fast Retrieval**: FAISS vector search on local disk
✅ **Policy-Grounded**: Only answers from your knowledge base
✅ **Chat UI**: Beautiful Gradio web interface

---

## 📊 What It Can Do

### Example Conversations

**English:**
```
Q: Is laptop screen damage covered under warranty?
A: No, laptop screen damage is not covered under warranty.
   Physical damage is explicitly excluded from our 1-year limited warranty.
```

**Hindi:**
```
Q: DROMA लैपटॉप की वारंटी कितने साल की है?
A: DROMA लैपटॉप की वारंटी 1 साल की है। यह manufacturer की warranty है।
```

**Telugu:**
```
Q: Laptop screen damage warranty lo cover avutunda?
A: No, laptop screen damage warranty lo cover avutalu.
   Physical damage exclusion lo reside avutundi.
```

**Hinglish (Mixed Roman + Hindi):**
```
Q: Return policy kya hai? Kitne din ka hai?
A: Return policy 7 to 10 business days ka hai.
   Approved refunds within this time automatically credited honge.
```

**Telgish (Mixed Roman + Telugu):**
```
Q: Return polcy eni rojulu?
A: Return policy 7 to 10 rojulu. Approved refunds credit avutay.
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose | Cost |
|-----------|-----------|---------|------|
| **Frontend** | Gradio | Web UI for chat | Free |
| **Backend** | Python 3.11 | Processing pipeline | Free |
| **Vectorization** | sentence-transformers (all-MiniLM-L6-v2) | Local embeddings | Free |
| **Vector DB** | FAISS | Similarity search | Free |
| **LLM** | OpenAI (gpt-4o-mini) | High quality AI | ~$0.15/1M tokens |
| **Language Detection** | langdetect | Detect input language | Free |
| **Text Splitting** | langchain | Chunk long policies | Free |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Prerequisites
- Python 3.11+ installed

### Step 2: Clone & Setup
```bash
cd C:\Users\abhir\3agent_RAG

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 3: Configure API Keys
Create `.env` file in project root:

```
OPENAI_API_KEY=sk_your_openai_key_here
```

### Step 4: Build Vector Store
```bash
.venv\Scripts\python.exe ingest.py
```
Output: `Indexed 30 chunks from 5 files using local embeddings`

### Step 5: Launch Chatbot
```bash
.venv\Scripts\python.exe app.py
```
Opens: `http://127.0.0.1:7860`

### Step 6: Test!
Try these:
- `Is laptop screen damage covered?`
- `लैपटॉप की वारंटी कितने साल की है?`
- `Return policy kya hai?`

---

## 📂 Project Structure

```
3agent_RAG/
├── app.py                    ← Main chatbot (Gradio UI + RAG pipeline)
├── ingest.py                 ← Vector store builder (run once)
├── requirements.txt          ← Python dependencies
├── .env                      ← API keys (create this)
├── README.md                 ← This file
│
├── data/                     ← Knowledge base (your policies)
│   ├── laptops.txt          ← Laptop warranty, repair timelines
│   ├── phones.txt           ← Phone warranty policies
│   ├── tvs.txt              ← TV coverage, DOA policy
│   ├── general.txt          ← Support hours, refund process
│   └── accessories.txt       ← Accessory warranty
│
├── vector_store/            ← Generated after ingest.py (don't edit)
│   ├── index.faiss          ← FAISS vector index (binary)
│   ├── chunks.pkl           ← Text chunks (serialized)
│   └── meta.pkl             ← Embedding metadata
│
└── .venv/                   ← Virtual environment (created by you)
```

---

## 📋 File Descriptions

### `app.py` (344 lines)
**The chatbot engine.** Runs the full RAG pipeline:
1. Detects input language style (including Hinglish/Telgish)
2. Normalizes queries to English for embedding
3. Generates embedding
4. Searches FAISS index
5. Calls LLM with context
6. Ensures response in user's original language/style
7. Displays in Gradio UI

**Key Functions:**
- `answer(query, history)` → RAG pipeline
- `detect_style(query)` → Language style detection
- `normalize_to_english(query)` → Query normalization
- `retrieve_context(english_query)` → FAISS search

### `ingest.py` (61 lines)
**One-time setup script.** Builds the vector store:
1. Reads all `.txt` files from `data/`
2. Splits into 800-char chunks (100 overlap)
3. Generates embeddings (OpenAI or local)
4. Creates FAISS IndexFlatL2
5. Saves to `vector_store/`

**Run once after updating `data/`:**
```bash
.venv\Scripts\python.exe ingest.py
```

### `data/*.txt` Files
**Your knowledge base.** Plain text files with policies:
- `laptops.txt`: Warranty (1yr), screen/liquid damage exclusions
- `phones.txt`: 12-month limited warranty, DOA handling
- `tvs.txt`: 2yr panel + 1yr accessories
- `general.txt`: Support hours, refund timelines, pickup policy
- `accessories.txt`: 6-month warranty, voltage exclusions

**Format:** Simple plain text (no markup needed). Chunks automatically created.

### `requirements.txt`
Pinned versions for compatibility:
```
gradio==6.9.0               ← Web UI
openai                      ← OpenAI API client
faiss-cpu==1.7.4            ← Vector search
langchain-text-splitters    ← Text chunking
deep_translator             ← Google Translate wrapper
langdetect                  ← Language detection
python-dotenv               ← .env file support
numpy==1.26.4               ← Numerical computing
sentence-transformers       ← Local embedding model
huggingface_hub==0.25.2     ← Model download utility
```

---

## 🎯 How It Works (Full RAG Pipeline)

### 7-Stage Processing

```
INPUT: Customer Question
   │ (any language, any script)
   │
   ├─ STAGE 1: Language Style Detection
   │  └─ Detects: english / hindi / telugu / gujarati / tamil / hinglish / telgish
   │
   ├─ STAGE 2: Normalize to English
   │  └─ Uses LLM to convert any style to clean English for embedding
   │
   ├─ STAGE 3: Generate Embedding
   │  └─ Convert to 1536-dim vector (OpenAI) or 384-dim (local)
   │
   ├─ STAGE 4: Search Vector Store
   │  └─ FAISS finds top-3 similar policy chunks
   │
   ├─ STAGE 5: Build Context
   │  └─ Combine chunks into prompt
   │
   ├─ STAGE 6: Call LLM
   │  └─ Get policy-grounded answer
   │
   └─ STAGE 7: Preserve Language Style
      └─ Ensures answer matches user's original language/style

OUTPUT: Policy-Based Answer
```

---

## 🌍 Language Support Matrix

| Language | Code | Detection | Examples |
|----------|------|-----------|----------|
| **English** | en | ✅ Perfect | "Is screen damage covered?" |
| **Hindi** | hi | ✅ Perfect | "लैपटॉप की वारंटी?" |
| **Telugu** | te | ✅ Perfect | "Warranty lo cover avutunda?" |
| **Gujarati** | gu | ✅ Perfect | "વોરંટી કેટલા વર્ષની?" |
| **Tamil** | ta | ✅ Perfect | "வாரண்டி எத்தனை வருடம்?" |
| **Hinglish** | auto→en | ✅ Smart normalize | "Return policy kya hai?" |
| **Telgish** | auto→en | ✅ Smart normalize | "Return polcy eni rojulu?" |

**Hinglish/Telgish Examples That Work:**
```
"Return policy kya hai?"              ✅ Works
"Return polcy eni rojulu?"            ✅ Works
"Laptop screen damage warranty lo cover avutunda?" ✅ Works
"refund entha rojulu?"                ✅ Works
"DROMA laptop ki warranty kitni hai?" ✅ Works
```

---

## 🏗️ Architecture Flow

### System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE SYSTEM ARCHITECTURE                        │
└──────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER (User Facing)                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  🎨 Gradio Web Interface (Python library)                             │
│  ├─ URL: http://127.0.0.1:7860                                        │
│  ├─ Technology: React-based (JavaScript frontend)                     │
│  ├─ Backend: FastAPI (auto-generated by Gradio)                       │
│  │                                                                    │
│  │ Components:                                                        │
│  │  ├─ Chat Display Area                                              │
│  │  │  ├─ User messages (left, blue background)                       │
│  │  │  └─ Bot responses (right, orange background)                    │
│  │  │                                                                 │
│  │  ├─ Input Textbox                                                  │
│  │  │  ├─ Placeholder: "Ask your DROMA question here..."              │
│  │  │  ├─ 2-4 lines dynamic height                                    │
│  │  │  └─ Auto-focus on page load                                     │
│  │  │                                                                 │
│  │  ├─ Example Prompts (clickable buttons)                             │
│  │  │  ├─ "Is laptop screen damage covered?"  (English)               │
│  │  │  ├─ "लैपटॉप की वारंटी?" (Hindi)                                 │
│  │  │  ├─ "Return policy kya hai?" (Hinglish)                         │
│  │  │  ├─ "વોરંટી કેટલા વર્ષની?" (Gujarati)                            │
│  │  │  └─ "வாரண்டி எத்தனை வருடம்?" (Tamil)                            │
│  │  │                                                                 │
│  │  └─ Send Button (Submit on Enter or Click)                          │
│  │                                                                    │
│  └─ Session State (Browser Local Storage)                             │
│     ├─ Chat history (user ↔ bot messages)                             │
│     └─ Clears on page refresh                                         │
│                                                                        │
│                    COMMUNICATION LAYER (HTTP/WebSocket)                │
│  ═════════════════════════════════════════════════════════════════    │
│  POST http://127.0.0.1:7860/api/predict/  (JSON)                     │
│  {                                                                     │
│    "data": ["User query here", [[previous_messages]]]                 │
│  }                                                                     │
│  ═════════════════════════════════════════════════════════════════    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              │ JSON Request
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    BACKEND LAYER (Processing Logic)                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ 🔧 app.py (Python - Run on your machine or server)                    │
│                                                                        │
│ ┌─ STEP 1: INPUT HANDLING                                             │
│ │  ├─ Function: answer(query, history)                                │
│ │  ├─ Input: {query: str, history: list}                              │
│ │  ├─ Validate: Not empty, not too long                               │
│ │  └─ Output: Cleaned query string                                    │
│ │                                                                     │
│ ├─ STEP 2: LANGUAGE STYLE DETECTION                                   │
│ │  ├─ Function: detect_style()                                        │
│ │  ├─ Input: Raw user query                                           │
│ │  ├─ Tool: OpenAI LLM with STYLE_DETECT_PROMPT                       │
│ │  └─ Output: Language style (english/hindi/telugu/gujarati/tamil/hinglish/telgish) │
│ │                                                                     │
│ ├─ STEP 3: GREETING HANDLING                                          │
│ │  ├─ Function: is_greeting()                                         │
│ │  ├─ Input: Cleaned query                                            │
│ │  └─ Output: Match to predefined greeting words                      │
│ │                                                                     │
│ ├─ STEP 4: QUERY NORMALIZATION                                        │
│ │  ├─ Function: normalize_to_english()                                │
│ │  ├─ Input: Original query in any language/style                     │
│ │  ├─ Tool: OpenAI LLM with NORMALIZE_PROMPT                          │
│ │  └─ Output: Clean English query for embedding                       │
│ │                                                                     │
│ ├─ STEP 5: EMBEDDING GENERATION                                       │
│ │  ├─ Function: get_embedding()                                       │
│ │  ├─ Input: Clean English query                                      │
│ │  │                                                                 │
│ │  ├─ Option A: Local Embeddings (CPU)                                │
│ │  │  ├─ Tool: sentence-transformers                                  │
│ │  │  ├─ Model: all-MiniLM-L6-v2                                      │
│ │  │  ├─ Output: 384-dimensional vector                               │
│ │  │  └─ Latency: ~50-100ms (first run: ~500ms model load)            │
│ │  │                                                                 │
│ │  └─ Option B: OpenAI Embeddings (Cloud)                             │
│ │     ├─ Tool: OpenAI API                                             │
│ │     ├─ Model: text-embedding-3-small                                │
│ │     ├─ Output: 1536-dimensional vector                              │
│ │     └─ Latency: ~300-600ms                                          │
│ │                                                                     │
│ ├─ STEP 6: VECTOR SIMILARITY SEARCH                                   │
│ │  ├─ Function: retrieve_context()                                    │
│ │  ├─ Tool: FAISS (IndexFlatL2)                                       │
│ │  ├─ Database: vector_store/index.faiss                              │
│ │  ├─ Algorithm: L2 Euclidean distance                                │
│ │  ├─ Search: Top-3 most similar chunks                               │
│ │  └─ Latency: ~5ms (very fast, local disk)                           │
│ │                                                                     │
│ ├─ STEP 7: CONTEXT ASSEMBLY                                           │
│ │  ├─ Load: Top-3 chunks from vector_store/chunks.pkl                 │
│ │  ├─ Format: Combine into single context string                      │
│ │  ├─ Max size: ~2000 chars (fits in LLM context)                     │
│ │  └─ Memory: ~10-50KB per request                                    │
│ │                                                                     │
│ ├─ STEP 8: LANGUAGE-AWARE PROMPT CONSTRUCTION                         │
│ │  ├─ Function: build_style_instruction()                             │
│ │  ├─ Input: Detected language style                                  │
│ │  ├─ Process: Generate specific instruction to preserve language/style │
│ │  └─ Output: Language instruction for LLM                            │
│ │                                                                     │
│ ├─ STEP 9: LLM INFERENCE                                              │
│ │  ├─ Tool: OpenAI API                                                │
│ │  ├─ Endpoint: https://api.openai.com/v1/chat/completions            │
│ │  ├─ Model: gpt-4o-mini                                              │
│ │  ├─ Latency: 1-3s                                                   │
│ │  ├─ Cost: ~$0.15 per 1M tokens                                      │
│ │  │                                                                  │
│ │  ├─ Input Components:                                               │
│ │  │  ├─ SYSTEM_PROMPT (rules, context instructions)                  │
│ │  │  ├─ Context (retrieved policy chunks)                            │
│ │  │  ├─ User question (original + normalized)                        │
│ │  │  └─ Language style instruction (preserves response style)        │
│ │  │                                                                  │
│ │  └─ Output: Answer in English (grounded in context only)            │
│ │                                                                     │
│ └─ STEP 10: RESPONSE FORMATTING                                       │
│    ├─ Function: answer() final processing                             │
│    ├─ Process: Check if LLM returned fallback message                 │
│    ├─ Replace with language-appropriate fallback if needed            │
│    └─ Return: Final answer                                            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              │ JSON Response
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  RESPONSE LAYER (Back to Frontend)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ HTTP/JSON Response:                                                    │
│ {                                                                      │
│   "data": ["Answer text in customer's language"]                      │
│ }                                                                      │
│                                                                        │
│ Frontend renders:                                                      │
│  • Answer appended to chat history                                     │
│  • Styled as bot message                                              │
│  • Text wrapping enabled                                              │
│  • Auto-scroll to latest message                                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Frontend to Backend Data Flow

```
CUSTOMER INTERACTION FLOW
═════════════════════════

1. Customer types: "Return policy kya hai?"
   └─ In textbox, UI captures keystroke

2. Click "Submit" or Press Enter
   └─ Gradio captures event

3. Frontend sends JSON:
   ┌────────────────────────────────────────┐
   │ POST /api/predict/                     │
   ├────────────────────────────────────────┤
   │ {                                      │
   │   "data": [                            │
   │     "Return policy kya hai?",  ← Query │
   │     [[                                 │
   │       ["Previous user Q", "Answer"],   │
   │       ["Another Q", "Answer"]          │
   │     ]]  ← Chat history                 │
   │   ]                                    │
   │ }                                      │
   └────────────────────────────────────────┘

4. Backend processes (10 steps, ~2-3 seconds total):
   ├─ Detects: Hinglish (hinglish)
   ├─ Normalizes: "Return policy kya hai?" → "What is the return policy?"
   ├─ Embeds: Converts to 1536-dim vector
   ├─ Searches: Finds "Refunds are processed only after..." chunk
   ├─ Builds context: "Refunds are processed... Approved refunds credited..."
   ├─ Calls LLM: Gets "Approved refunds within 7-10 business days"
   └─ Returns: JSON with answer

5. Frontend receives JSON:
   ┌────────────────────────────────────────┐
   │ {                                      │
   │   "data": [                            │
   │     "7 to 10 business days ka refund   │
   │      hota hai. Approved refunds        │
   │      automatically credited honge."    │
   │   ]                                    │
   │ }                                      │
   └────────────────────────────────────────┘

6. Frontend renders:
   ├─ Updates chat message history
   ├─ Appends bot response
   ├─ Scrolls to bottom
   └─ Waits for next input

7. Customer sees: "7 to 10 business days ka refund hota hai..."
   └─ Cycle repeats
```

### Component Dependency Map

```
FRONTEND DEPENDENCIES
──────────────────────────────
gradio (6.9.0)
  ├─ React (frontend framework)
  ├─ FastAPI (HTTP server - auto-generated)
  ├─ Pydantic (request validation)
  └─ WebSocket support (real-time updates)

BACKEND DEPENDENCIES
──────────────────────────────
Python 3.11 Core:
  ├─ os, pickle, re (stdlib)
  └─ typing (stdlib)

Vector Search:
  ├─ faiss-cpu (1.7.4) - Local FAISS index
  ├─ numpy (1.26.4) - Numerical arrays
  ├─ sentence-transformers (latest) - Local embeddings
  └─ huggingface_hub (0.25.2) - Model downloads

Language Processing:
  ├─ langdetect - Detect language
  ├─ langchain-text-splitters - Text chunking (ingest only)
  └─ openai - OpenAI API client

API Clients:
  └─ python-dotenv - .env file support

CLOUD SERVICES (External APIs)
──────────────────────────────
1. OpenAI API
   ├─ Key: OPENAI_API_KEY (from https://platform.openai.com)
   ├─ Models:
   │  ├─ text-embedding-3-small (embeddings)
   │  └─ gpt-4o-mini (LLM)
   └─ Cost: ~$0.02/1M embed tokens, ~$0.15/1M tokens

LOCAL STORAGE
──────────────────────────────
vector_store/ (Generated by ingest.py)
  ├─ index.faiss - FAISS binary index
  ├─ chunks.pkl - Serialized text chunks
  └─ meta.pkl - Embedding metadata

data/ (Your knowledge base)
  ├─ laptops.txt
  ├─ phones.txt
  ├─ tvs.txt
  ├─ general.txt
  └─ accessories.txt

.env (Your API keys)
  ├─ OPENAI_API_KEY (required)
```

---

## 🔧 Configuration

### API Key Setup

**OpenAI Required: ~$0.15 per 1M tokens**
```
OPENAI_API_KEY=sk_...
```

### Model Selection

Edit `app.py` to change LLM or embedding model:

```python
# Line 16-19: Change these
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"         # or other sentence-transformers model
OPENAI_LLM_MODEL = "gpt-4o-mini"               # or "gpt-4", "gpt-3.5-turbo"
```

### Retrieval Tuning

```python
# Line 16: Number of chunks to retrieve
TOP_K = 3  # Increase for more context, higher cost
```

---

## 🧪 Testing & Troubleshooting

### Test All Languages

```bash
# After running app.py
# Go to http://127.0.0.1:7860 and try:

English:
  "Is laptop warranty 1 year?"

Hindi:
  "क्या लैपटॉप की वारंटी 1 साल की है?"

Telugu:
  "Laptop warranty 1 year ka undi?"

Gujarati:
  "લેપટોપ વોરંટી 1 વર્ષની છે?"

Tamil:
  "லேப்டாப் வாரண்டி 1 வருடம்?"

Hinglish:
  "Return policy kya hai kitne din ka hai?"

Telgish:
  "Return policy eni rojulu warranty?"
```

### Common Issues & Fixes

**Issue: "ModuleNotFoundError: No module named 'faiss'"**
```bash
# Solution: Use venv Python
.venv\Scripts\python.exe app.py  # ✅ Correct
python app.py                    # ❌ Wrong
```

**Issue: "vector_store missing. Run: python ingest.py"**
```bash
# Solution: Build vector store first
.venv\Scripts\python.exe ingest.py
```

```bash
# Solution: Create .env with at least one key
```

**Issue: "Response takes 30+ seconds"**
```bash
# First response is slow (model download)
# Subsequent: 2-5 seconds normal
# If persistent, check:
# 1. Internet speed (for translations)
# 3. Use local embeddings in app.py
```

**Issue: "Hinglish query returns wrong answer"**
```bash
# Solution: Model knows 30+ Hinglish/Telgish words
# To add more, edit MIXED_MAP in app.py (line 26-38)
MIXED_MAP = {
    "your_typo": "correct_term",
    ...
}
```

---

## 📈 Architecture Deep Dive

### Embedding Models

**Option 1: Local (sentence-transformers)**
- Model: `all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: Fast (~100ms per query)
- Cost: FREE
- Hardware: CPU only
- Use when: Budget is critical

**Option 2: OpenAI**
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Speed: Slower (~500ms per query)
- Cost: $0.02 per 1M tokens
- Hardware: Cloud (OpenAI servers)
- Use when: Quality > speed

### LLM Models

**OpenAI (High Quality)**
- Model: `gpt-4o-mini`
- Quality: Excellent, nuanced answers
- Speed: Standard
- Cost: ~$0.15 per 1M input tokens
- Latency: 2-5 seconds typical

### System Temperature

`temperature=0.2` (Low = Factual)
- Keeps responses grounded to context
- Prevents hallucination
- Good for support policies

If you want more creative responses, change to `temperature=0.7`.

---

## 🚢 Production Deployment

### Docker (Optional)
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

### Hugging Face Spaces
1. Push repo to GitHub
2. Create Hugging Face Space
4. Deploy!

### AWS/Azure/GCP
1. Use cloud VM (t3.medium or equivalent)
2. Set up `.env` with API keys
3. Run `python app.py`
4. Expose port 7860 via firewall

---

## 📊 Performance Metrics

### Typical Response Time Breakdown (in milliseconds)

```
├─ Language style detection: 500ms   ← LLM call
├─ Query normalization       : 500ms   ← LLM call
├─ Embedding generation      : 300ms   ← OpenAI embedding
├─ FAISS search              : 5ms
├─ Context assembly          : 3ms
├─ LLM inference             : 1500ms  ← LLM call
└─ Total                    : ~2800ms (2.8 seconds)
```

---

## 🔐 Security & Privacy

- **Vector store**: Local disk, encrypted by OS
- **Chunks**: Text only, no PII handling
- **API keys**: Stored in `.env` (gitignore it!)
- **Data**: Queries NOT logged (unless you code it)

**.gitignore essentials:**
```
.env
.venv/
vector_store/
*.pyc
__pycache__/
```

---

## 🎓 How To Customize

### Add Your Company Policies

1. Create `data/your_topic.txt`
2. Write plain text policies
3. Run: `.venv\Scripts\python.exe ingest.py`
4. Restart app: `.venv\Scripts\python.exe app.py`

### Add New Languages

Edit `app.py` line 20:
```python
SUPPORTED_LANGS = {"en", "hi", "te", "gu", "ta", "kn", "ml", "mr"}  # Add more codes
```

### Add Hinglish/Telgish Words

Edit `app.py` `NORMALIZE_PROMPT` examples and `STYLE_DETECT_PROMPT` rules to add more patterns.

### Change Greeting/System Prompt

Edit `app.py` line 23-40, `SYSTEM_PROMPT`:
```python
SYSTEM_PROMPT = """Your custom system instructions here."""
```

---

## ❓ FAQ

**Q: Do I need GPU?**
No. CPU-only is fine. First query may load embedding model (~500ms), then ~2-3s per query.

**Q: What if API key expires?**
Update `.env` and restart app. No rebuild needed.

**Q: Can I deploy on free tier?**
Yes! Hugging Face Spaces or Replit free tier work great.

**Q: How to add more languages?**
Edit `SUPPORTED_LANGS` in `app.py` and test with sample queries.

**Q: Is my data stored?**
No. Queries + answers live in browser session only (unless you code logging).

**Q: Can I fine-tune the LLM?**
Not in this version (RAG-only). Use OpenAI fine-tuning separately.

---

## 📚 Additional Resources

- [Gradio Docs](https://gradio.app/docs)
- [FAISS Guide](https://github.com/facebookresearch/faiss)
- [sentence-transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [OpenAI API Docs](https://platform.openai.com/docs)

---

## 🤝 Contributing

To improve this chatbot:

1. **Add policies**: Update `data/*.txt`
2. **Test languages**: Try all 5 languages + Hinglish/Telgish
3. **Report issues**: Note latency, accuracy, language issues
4. **Suggest models**: Recommend better embeddings or LLMs

---

## 📝 License

MIT. Use freely for any purpose.

---

## 🎉 Next Steps

1. ✅ Clone project
2. ✅ Install dependencies: `pip install -r requirements.txt`
4. ✅ Build vector store: `python ingest.py`
5. ✅ Launch: `python app.py`
6. ✅ Visit: http://127.0.0.1:7860
7. ✅ Test in all languages!

**Enjoy your production-ready multilingual RAG chatbot!** 🚀

---

## 👤 Author

**Abhishyant Reddy**

📧 For questions, suggestions, or improvements: [abhireds22@gmail.com](mailto:abhireds22@gmail.com)

---

*Made with ❤️ for multilingual customer support automation.*