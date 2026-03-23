"""DROMA ELECTRONICS - MULTILINGUAL SUPPORT CHATBOT
Provides customer support using AI and vector search.
Answers questions about warranties, refunds, and policies in multiple languages.
"""

import os  # Operating system functions
import pickle  # Save/load Python objects
import re  # Regular expressions for text processing
import subprocess  # Run ingest.py if needed
import sys  # System utilities

import faiss  # Facebook vector search (find similar documents)
import gradio as gr  # Web UI framework
import numpy as np  # Numerical arrays for vectors
from deep_translator import GoogleTranslator  # Translate between languages
from dotenv import load_dotenv  # Load API keys from .env file
from langdetect import detect  # Detect what language text is in
from openai import OpenAI  # OpenAI API for embeddings & LLM

# ============ CONFIGURATION ============
TOP_K = 3  # Return top 3 most similar documents  # Return top 3 most similar documents
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # Model to convert text to vectors
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"  # Fallback local embedding model
OPENAI_LLM_MODEL = "gpt-4o-mini"  # Language model for generating responses
SUPPORTED_LANGS = {"en", "hi", "te", "gu", "ta"}  # English, Hindi, Telugu, Gujarati, Tamil
STORE_DIR = "vector_store"  # Directory with indexed documents

# System instructions for the LLM - controls how it behaves
SYSTEM_PROMPT = """You are a customer support assistant for DROMA Electronics.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: \"I don't have information on that.\"
Be concise and helpful.
Always reply strictly in the same language as the user's question."""
# Message to show when we don't have the information
FALLBACK_EN = (
    "I'm here to help. I couldn't find that in DROMA policy docs yet. "
    "Please ask about warranties, returns/refunds, service timelines, or accessories."
)

# Words that indicate a user is greeting us (instead of asking a real question)
GREETING_WORDS = {
    "hi",
    "hello",
    "hey",
    "hola",
    "namaste",
    "namaskaram",
    "vanakkam",
    "hii",
    "hlo",
}

# Dictionary to fix common spelling mistakes and convert to English
MIXED_MAP = {
    "polciy": "policy",
    "polcy": "policy",
    "warrenty": "warranty",
    "kya hai": "what is",
    "kitna": "how much",
    "kitne": "how many",
    "rojulu": "days",
    "eni": "how many",
    "entha": "how many",
    "divas": "days",
    "naal": "days",
    "niyam": "policy",
}

# Load API keys from .env file
load_dotenv()
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

index_path = os.path.join(STORE_DIR, "index.faiss")
chunks_path = os.path.join(STORE_DIR, "chunks.pkl")
meta_path = os.path.join(STORE_DIR, "meta.pkl")


def ensure_vector_store() -> None:
    if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(meta_path):
        return
    print("vector_store missing. Running ingest.py...")  # Auto-build vector store
    cmd = [sys.executable, "ingest.py"]  # Run ingest.py to build it
    result = subprocess.run(cmd, capture_output=True, text=True)  # Execute command
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"Failed to build vector_store: {detail}")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path) or not os.path.exists(meta_path):
        raise SystemExit("ingest.py finished but vector_store files were not created")


# Auto-create vector store if it doesn't exist
ensure_vector_store()

# Load the FAISS index for fast similarity search
index = faiss.read_index(index_path)
# Load document chunks (the actual text pieces)
with open(chunks_path, "rb") as f:
    CHUNKS = pickle.load(f)
# Load metadata about how embeddings were created
with open(meta_path, "rb") as f:
    META = pickle.load(f)

# Extract embedding configuration
EMBED_MODE = META.get("embed_mode", "openai")  # "openai" or "local"
EMBED_MODEL = META.get("embed_model", OPENAI_EMBED_MODEL)  # Which model was used
EMBED_DIM = int(META.get("embed_dim", index.d))  # Vector size (384 or similar)
if EMBED_DIM != index.d:
    raise SystemExit(f"Index/meta dim mismatch: {index.d} vs {EMBED_DIM}")

if EMBED_MODE == "openai" and not openai_key:
    raise SystemExit("OPENAI_API_KEY required for this vector_store. Re-run ingest.py or add key.")
if not openai_key:
    raise SystemExit("Set OPENAI_API_KEY in .env")

# API clients
embed_client = OpenAI(api_key=openai_key)  # For creating embeddings
llm_client, LLM_MODEL = OpenAI(api_key=openai_key), OPENAI_LLM_MODEL  # For generating responses

# If using local embeddings, load the model
if EMBED_MODE == "local":
    from sentence_transformers import SentenceTransformer
    local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)


def translate_text(text: str, src: str, tgt: str) -> str:
    """Translate text from source language to target language."""
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception:
        return text


def normalize_mixed_query(text: str) -> str:
    q = text.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    for bad, good in MIXED_MAP.items():
        q = q.replace(bad, good)
    return q


def is_mixed_style_query(text: str) -> bool:
    q = text.lower()
    hints = ["kya", "hai", "kitna", "kitne", "rojulu", "eni", "entha", "avutunda", "naa", "niyam"]
    return any(h in q for h in hints)


def lang_name(lang_code: str) -> str:
    names = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu",
        "gu": "Gujarati",
        "ta": "Tamil",
    }
    return names.get(lang_code, "English")


def detect_supported_lang(text: str) -> str:
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    return lang if lang in SUPPORTED_LANGS else "en"


def is_greeting(text: str) -> bool:
    q = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    if not q:
        return False
    return q in GREETING_WORDS or len(q.split()) <= 2 and any(w in GREETING_WORDS for w in q.split())


def greeting_reply(src_lang: str) -> str:
    replies = {
        "en": "Hi! I can help with DROMA warranties, refunds, repairs, and accessories.",
        "hi": "नमस्ते! मैं DROMA की वारंटी, रिफंड, रिपेयर और एक्सेसरीज़ में मदद कर सकता हूँ।",
        "te": "హాయ్! DROMA వారంటీ, రిఫండ్, రిపేర్, యాక్సెసరీస్ గురించి సహాయం చేస్తాను.",
        "gu": "નમસ્તે! હું DROMA ની વોરંટી, રિફંડ, રિપેર અને એક્સેસરીઝ વિશે મદદ કરી શકું છું.",
        "ta": "வணக்கம்! DROMA வாரண்டி, ரிஃபண்ட், ரிப்பேர் மற்றும் ஆக்சஸரீஸ் பற்றி உதவ முடியும்.",
    }
    return replies.get(src_lang, replies["en"])


def answer(query: str, history: list) -> str:
    if not query or not query.strip():
        return "Please enter a question."
    src_lang = detect_supported_lang(query)
    if is_greeting(query):
        return greeting_reply(src_lang)
    try:
        # Prepare query for document search
        if src_lang == "en":
            # For English: auto-translate, normalize, combine variants
            auto_en = translate_text(query, "auto", "en")
            normalized = normalize_mixed_query(query)
            parts = [query]
            if auto_en and auto_en.strip().lower() != query.strip().lower():
                parts.append(auto_en.strip())
            if normalized and normalized != query.strip().lower():
                parts.append(normalized)
            english_query = "\n".join(dict.fromkeys(parts))
        else:
            # For non-English: translate to English, then normalize
            english_query = translate_text(query, src_lang, "en")
            normalized = normalize_mixed_query(english_query)
            if normalized and normalized != english_query.strip().lower():
                english_query = f"{english_query}\n{normalized}"
        # Convert query to vector (embedding)
        if EMBED_MODE == "openai":
            qvec = embed_client.embeddings.create(model=EMBED_MODEL, input=[english_query]).data[0].embedding
        else:
            qvec = local_embedder.encode([english_query], normalize_embeddings=True)[0]
        
        # Search FAISS index for similar documents
        vector = np.array([qvec], dtype=np.float32)
        if vector.shape[1] != EMBED_DIM:
            return "Embedding error. Please try again."
        _, ids = index.search(vector, TOP_K)  # Find TOP_K closest matches
        # Extract relevant document chunks
        # Build LLM prompt with context and language instructions
        style_hint = ""
        if src_lang == "en" and is_mixed_style_query(query):
            # If mixed language (Hinglish/Telgish), tell LLM to match style
            style_hint = "\nReply in concise mixed Roman-script style similar to the user's wording."
        
        # Strict language enforcement
        strict_lang_hint = f"\nReply strictly in {lang_name(src_lang)}. Do not switch languages."
        # Call OpenAI LLM
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},  # Behavior instructions
                {"role": "user", "content": user_prompt},  # The actual question
            ],
            temperature=0.2,  # Lower = more consistent, less creative
        )
        
        # Extract the response
        answer_en = resp.choices[0].message.content.strip()
        # If LLM says it doesn't have info, use friendlier fallback
        if answer_en.lower() == "i don't have information on that.":
            answer_en = FALLBACK_EN
        
        # Translate response back to user's language if needed
        if src_lang == "en":
            return translate_text(answer_en, "auto", "en")
        return translate_text(answer_en, "auto", src_lang)  # Translate to user's language
    
    except Exception:
        # If anything goes wrong, return helpful error message
        return "I had a temporary issue processing that. Please try again."


# ============ GRADIO WEB UI ============
# Create the chat interface that users interact with

demo = gr.ChatInterface(
    fn=answer,  # Function to call when user sends a message
    title="DROMA Electronics Support Assistant",  # Page title
    description=(
        "Multilingual policy helper for warranty, refunds, service timelines, and accessories. "
        "Ask in English, Hindi, Telugu, Gujarati, or Tamil."
    ),  # Info text
    examples=[  # Example questions to show users
        "Is laptop screen damage covered under warranty?",
        "DROMA लैपटॉप की वारंटी कितने साल की है?",
        "Laptop screen damage warranty lo cover avutunda?",
        "DROMA લેપટોપની વોરંટી કેટલા વર્ષની છે?",
        "DROMA லேப்டாப் வாரண்டி எத்தனை வருடம்?",
        "How long does refund processing take?",
    ],
    textbox=gr.Textbox(
        placeholder="Ask your DROMA question here...",  # Input hint
        lines=2,
        max_lines=4,
        show_label=False,
    ),  # Input text box
)

# Run the web app
if __name__ == "__main__":
    demo.launch(share=False)  # Start server (share=False means local only)
