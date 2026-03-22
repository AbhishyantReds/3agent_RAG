import os
import pickle
import re
import subprocess
import sys

import faiss
import gradio as gr
import numpy as np
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

TOP_K = 3
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_LLM_MODEL = "gpt-4o-mini"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
SUPPORTED_LANGS = {"en", "hi", "te", "gu", "ta"}
STORE_DIR = "vector_store"
SYSTEM_PROMPT = """You are a customer support assistant for DROMA Electronics.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: \"I don't have information on that.\"
Be concise and helpful."""
FALLBACK_EN = (
    "I'm here to help. I couldn't find that in DROMA policy docs yet. "
    "Please ask about warranties, returns/refunds, service timelines, or accessories."
)
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

load_dotenv()
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
groq_base = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").strip()

index_path = os.path.join(STORE_DIR, "index.faiss")
chunks_path = os.path.join(STORE_DIR, "chunks.pkl")
meta_path = os.path.join(STORE_DIR, "meta.pkl")


def ensure_vector_store() -> None:
    if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(meta_path):
        return
    print("vector_store missing. Running ingest.py...")
    cmd = [sys.executable, "ingest.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"Failed to build vector_store: {detail}")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path) or not os.path.exists(meta_path):
        raise SystemExit("ingest.py finished but vector_store files were not created")


ensure_vector_store()

index = faiss.read_index(index_path)
with open(chunks_path, "rb") as f:
    CHUNKS = pickle.load(f)
with open(meta_path, "rb") as f:
    META = pickle.load(f)

EMBED_MODE = META.get("embed_mode", "openai")
EMBED_MODEL = META.get("embed_model", OPENAI_EMBED_MODEL)
EMBED_DIM = int(META.get("embed_dim", index.d))
if EMBED_DIM != index.d:
    raise SystemExit(f"Index/meta dim mismatch: {index.d} vs {EMBED_DIM}")

if EMBED_MODE == "openai" and not openai_key:
    raise SystemExit("OPENAI_API_KEY required for this vector_store. Re-run ingest.py or add key.")
if not openai_key and not groq_key:
    raise SystemExit("Set OPENAI_API_KEY or GROQ_API_KEY in .env")

embed_client = OpenAI(api_key=openai_key) if openai_key else None
if openai_key:
    llm_client, LLM_MODEL = OpenAI(api_key=openai_key), OPENAI_LLM_MODEL
else:
    llm_client, LLM_MODEL = OpenAI(api_key=groq_key, base_url=groq_base), GROQ_LLM_MODEL

if EMBED_MODE == "local":
    from sentence_transformers import SentenceTransformer

    local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)


def translate_text(text: str, src: str, tgt: str) -> str:
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
    hints = ["kya", "hai", "kitna", "rojulu", "eni", "entha", "lo", "policy", "warranty"]
    return any(h in q for h in hints)


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
        if src_lang == "en":
            auto_en = translate_text(query, "auto", "en")
            normalized = normalize_mixed_query(query)
            parts = [query]
            if auto_en and auto_en.strip().lower() != query.strip().lower():
                parts.append(auto_en.strip())
            if normalized and normalized != query.strip().lower():
                parts.append(normalized)
            english_query = "\n".join(dict.fromkeys(parts))
        else:
            english_query = translate_text(query, src_lang, "en")
            normalized = normalize_mixed_query(english_query)
            if normalized and normalized != english_query.strip().lower():
                english_query = f"{english_query}\n{normalized}"
        if EMBED_MODE == "openai":
            qvec = embed_client.embeddings.create(model=EMBED_MODEL, input=[english_query]).data[0].embedding
        else:
            qvec = local_embedder.encode([english_query], normalize_embeddings=True)[0]
        vector = np.array([qvec], dtype=np.float32)
        if vector.shape[1] != EMBED_DIM:
            return "Embedding error. Please try again."
        _, ids = index.search(vector, TOP_K)
        context = "\n\n".join(CHUNKS[int(i)] for i in ids[0] if 0 <= int(i) < len(CHUNKS))
        style_hint = ""
        if is_mixed_style_query(query):
            style_hint = "\nReply in concise mixed Roman-script style similar to the user's wording."
        user_prompt = f"Context:\n{context}\n\nQuestion: {english_query}{style_hint}"
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer_en = resp.choices[0].message.content.strip()
        if answer_en.lower() == "i don't have information on that.":
            answer_en = FALLBACK_EN
        return answer_en if src_lang == "en" else translate_text(answer_en, "en", src_lang)
    except Exception:
        return "I had a temporary issue processing that. Please try again."


demo = gr.ChatInterface(
    fn=answer,
    title="DROMA Electronics Support Assistant",
    description=(
        "Multilingual policy helper for warranty, refunds, service timelines, and accessories. "
        "Ask in English, Hindi, Telugu, Gujarati, or Tamil."
    ),
    examples=[
        "Is laptop screen damage covered under warranty?",
        "DROMA लैपटॉप की वारंटी कितने साल की है?",
        "Laptop screen damage warranty lo cover avutunda?",
        "DROMA લેપટોપની વોરંટી કેટલા વર્ષની છે?",
        "DROMA லேப்டாப் வாரண்டி எத்தனை வருடம்?",
        "How long does refund processing take?",
    ],
    textbox=gr.Textbox(
        placeholder="Ask your DROMA question here...",
        lines=2,
        max_lines=4,
        show_label=False,
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
