import os
import pickle
import re
import time
from functools import lru_cache
from typing import Tuple

import faiss
import gradio as gr
import numpy as np
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

TOP_K = 2
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_LLM_MODEL = "gpt-4o-mini"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
SUPPORTED_LANGS = {"en", "hi", "te"}
STORE_DIR = "vector_store"
SYSTEM_PROMPT = """You are a customer support assistant for DROMA Electronics.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: \"I don't have information on that.\"
Be concise and helpful."""
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Merriweather:wght@400;700&display=swap');

:root {
    --droma-bg-1: #fdf8ef;
    --droma-bg-2: #e8f4ef;
    --droma-card: #ffffff;
    --droma-accent: #e56a14;
    --droma-accent-soft: #ffefe1;
    --droma-text: #1f2937;
}

body {
    background: radial-gradient(circle at 10% 10%, #ffe7d1 0%, transparent 35%),
                            radial-gradient(circle at 90% 20%, #d7f3e7 0%, transparent 35%),
                            linear-gradient(135deg, var(--droma-bg-1), var(--droma-bg-2));
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    max-width: 980px !important;
    margin: 20px auto !important;
}

#component-0 {
    border-radius: 20px !important;
    background: var(--droma-card) !important;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #f0e8dd !important;
}

h1, h2, h3 {
    font-family: 'Merriweather', serif !important;
    color: var(--droma-text) !important;
}

.prose p {
    color: #374151 !important;
}

.message.user {
    background: #eef7ff !important;
    border: 1px solid #d7ebff !important;
}

.message.bot {
    background: #fff8f1 !important;
    border: 1px solid #ffe7d1 !important;
}

button.primary {
    background: var(--droma-accent) !important;
    border: none !important;
    color: white !important;
}

button.primary:hover {
    filter: brightness(0.94);
}

footer {
    display: none !important;
}
"""

load_dotenv()
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
groq_base = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").strip()

index_path = os.path.join(STORE_DIR, "index.faiss")
chunks_path = os.path.join(STORE_DIR, "chunks.pkl")
meta_path = os.path.join(STORE_DIR, "meta.pkl")
if not os.path.exists(index_path) or not os.path.exists(chunks_path) or not os.path.exists(meta_path):
    raise SystemExit("vector_store missing. Run: python ingest.py")

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

STOPWORDS = {"the", "and", "for", "with", "under", "from", "that", "this", "what", "is", "are"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
LANG_NAMES = {"en": "🇬🇧 English", "hi": "🇮🇳 हिंदी", "te": "🇮🇳 తెలుగు"}

# Global state
QUERY_CACHE = {}
SELECTED_LANG = "en"
local_embedder = None  # Lazy-loaded on first use


def get_local_embedder():
    """Lazy-load sentence transformer on first use to speed up startup."""
    global local_embedder
    if local_embedder is None and EMBED_MODE == "local":
        from sentence_transformers import SentenceTransformer
        local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
    return local_embedder


@lru_cache(maxsize=256)
def tokenize(text: str) -> frozenset:
    return frozenset({w for w in TOKEN_PATTERN.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS})


CHUNK_TOKENS = [tokenize(c) for c in CHUNKS]


@lru_cache(maxsize=512)
def translate_text(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception:
        return text


def retrieve_context_ids(query_text: str, vector_ids: np.ndarray) -> list:
    ids = [int(i) for i in vector_ids[0] if 0 <= int(i) < len(CHUNKS)]
    q_tokens = tokenize(query_text)
    if not q_tokens:
        return ids
    best_idx, best_score = -1, 0
    for i, c_tokens in enumerate(CHUNK_TOKENS):
        overlap = len(q_tokens.intersection(c_tokens))
        if overlap > best_score:
            best_idx, best_score = i, overlap
    if best_idx >= 0 and best_score >= 2 and best_idx not in ids:
        ids = [best_idx] + ids
    return ids[: max(TOP_K, 3)]


def answer(query: str, history: list, lang_choice: str = None) -> str:
    global SELECTED_LANG, QUERY_CACHE
    
    if not query or not query.strip():
        return "Please enter a question."
    
    query_lower = query.strip().lower()
    cache_key = (query_lower, lang_choice or "auto")
    
    # Check query cache first
    if cache_key in QUERY_CACHE:
        cached_answer, _ = QUERY_CACHE[cache_key]
        return cached_answer
    
    try:
        # Language detection
        if lang_choice and lang_choice != "auto":
            src_lang = lang_choice
        else:
            try:
                src_lang = detect(query)
            except Exception:
                src_lang = "en"
        
        SELECTED_LANG = src_lang
        
        if src_lang not in SUPPORTED_LANGS:
            return "Supported languages: English, Hindi, Telugu only."
        
        # Translate to English if needed
        english_query = query if src_lang == "en" else translate_text(query, src_lang, "en")
        
        # Optimize: only embed first 200 chars for speed
        embed_text = english_query.split("\n")[0][:200]
        
        # Get embedding
        if EMBED_MODE == "openai":
            qvec = embed_client.embeddings.create(model=EMBED_MODEL, input=[embed_text]).data[0].embedding
        else:
            embedder = get_local_embedder()
            qvec = embedder.encode([embed_text], normalize_embeddings=True)[0]
        
        vector = np.array([qvec], dtype=np.float32)
        if vector.shape[1] != EMBED_DIM:
            return "Embedding error. Please try again."
        
        # Retrieve top candidates
        _, ids = index.search(vector, TOP_K + 1)
        context_ids = retrieve_context_ids(english_query, ids)
        context = "\n\n".join(CHUNKS[i] for i in context_ids[:TOP_K])
        
        # Generate answer
        user_prompt = f"Context:\n{context}\n\nQuestion: {english_query}"
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=15,
        )
        answer_en = resp.choices[0].message.content.strip()
        final_answer = answer_en if src_lang == "en" else translate_text(answer_en, "en", src_lang)
        
        # Cache result
        QUERY_CACHE[cache_key] = (final_answer, src_lang)
        if len(QUERY_CACHE) > 1024:
            QUERY_CACHE.pop(next(iter(QUERY_CACHE)))
        
        return final_answer
    except Exception as e:
        return f"Temporary issue. Please try again."


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 DROMA Electronics Support Assistant")
    gr.Markdown("Multilingual policy helper • Warranty • Refunds • Service timelines • Accessories")
    
    with gr.Group():
        gr.Markdown("**Select Language or Auto-Detect:**")
        with gr.Row():
            lang_en = gr.Button("🇬🇧 English", variant="secondary")
            lang_hi = gr.Button("🇮🇳 हिंदी", variant="secondary")
            lang_te = gr.Button("🇮🇳 తెలుగు", variant="secondary")
            lang_auto = gr.Button("🔄 Auto-Detect", variant="primary")
    
    selected_lang_state = gr.State("auto")
    lang_display = gr.Textbox(value="Language: Auto-Detect", interactive=False, label="Current Selection")
    
    def set_lang(lang):
        return lang, LANG_NAMES.get(lang, "Auto-Detect")
    
    lang_en.click(set_lang, inputs=gr.State("en"), outputs=[selected_lang_state, lang_display])
    lang_hi.click(set_lang, inputs=gr.State("hi"), outputs=[selected_lang_state, lang_display])
    lang_te.click(set_lang, inputs=gr.State("te"), outputs=[selected_lang_state, lang_display])
    lang_auto.click(set_lang, inputs=gr.State("auto"), outputs=[selected_lang_state, lang_display])
    
    with gr.Group():
        chatbot = gr.Chatbot(label="Conversation", height=400)
        msg_input = gr.Textbox(
            placeholder="Ask your DROMA question here... (auto-translates if needed)",
            lines=2,
            max_lines=4,
            label="Your Question",
            show_label=True
        )
        send_btn = gr.Button("Send Message", variant="primary")
    
    load_status = gr.Textbox(value="Ready ✓", interactive=False, label="Status")
    
    def process_message(message, chat_history, selected_lang):
        chat_history = chat_history or []
        load_status.value = "Processing... ⏳"
        
        try:
            response = answer(message, chat_history, selected_lang if selected_lang != "auto" else None)
            chat_history.append((message, response))
            load_status.value = f"Done ✓ | Cache: {len(QUERY_CACHE)} queries | {EMBED_MODE.upper()}"
            return chat_history, "", load_status.value
        except Exception as e:
            load_status.value = f"Error ✗ | {str(e)[:30]}"
            return chat_history, message, load_status.value
    
    send_btn.click(
        process_message,
        inputs=[msg_input, chatbot, selected_lang_state],
        outputs=[chatbot, msg_input, load_status]
    )
    msg_input.submit(
        process_message,
        inputs=[msg_input, chatbot, selected_lang_state],
        outputs=[chatbot, msg_input, load_status]
    )
    
    gr.Examples(
        examples=[
            "Is laptop screen damage covered under warranty?",
            "DROMA लैपटॉप की वारंटी कितने साल की है?",
            "Laptop screen damage warranty lo cover avutunda?",
            "How long does refund processing take?",
        ],
        inputs=msg_input
    )

if __name__ == "__main__":
    demo.launch(share=False)
