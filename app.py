"""DROMA ELECTRONICS - MULTILINGUAL SUPPORT CHATBOT
Pipeline:
  User query (any language/style)
    → LLM detects style (hindi / telgish / hinglish / etc.)
    → LLM normalizes query to clean English
    → Embed English query → FAISS search → retrieve top-K chunks
    → LLM generates answer grounded in chunks
    → Answer returned in user's original style
"""

import os
import pickle
import re
import subprocess
import sys

import faiss
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_K              = 3                        # How many policy chunks to retrieve from FAISS
OPENAI_EMBED_MODEL = "text-embedding-3-small" # OpenAI model to convert text → vector
LOCAL_EMBED_MODEL  = "all-MiniLM-L6-v2"       # Local fallback embedding model (no API needed)
OPENAI_LLM_MODEL   = "gpt-4o-mini"            # LLM used for style detection, normalization, answering
STORE_DIR          = "vector_store"           # Folder containing FAISS index + chunks

# ── System prompt: controls LLM answer behaviour ──────────────────────────────
# This is sent as the "system" role in every LLM call for generating answers.
# Key rules: only use provided context, never guess, reply in user's exact style.
SYSTEM_PROMPT = """You are a customer support assistant for DROMA Electronics, which sells laptops, phones, TVs, and accessories.

STRICT RULES:
1. Answer ONLY using the context provided. Do not use outside knowledge.
2. If the context does not contain the answer, respond exactly with: "I don't have information on that. Please contact DROMA support."
3. Never guess, infer, or assume warranty terms, timelines, or prices not explicitly stated in the context.
4. Keep answers under 4 sentences. Be direct and specific.
5. If the question has multiple parts, address each part.
6. Do not repeat the question back to the user.
7. CRITICAL: Reply in the EXACT same language style as the user's question.
   - If user asked in Hindi (Devanagari script like की, है) → reply in Hindi using Devanagari script
   - If user asked in Telugu (Telugu script like వారంటీ) → reply in Telugu using Telugu script
   - If user asked in Gujarati (Gujarati script like વોરંટી) → reply in Gujarati using Gujarati script
   - If user asked in Tamil (Tamil script like வாரண்டி) → reply in Tamil using Tamil script
   - If user asked in Hinglish (Roman script mixing Hindi + English, e.g. "kya hai") → reply in Hinglish Roman script, NO Devanagari
   - If user asked in Telgish (Roman script mixing Telugu + English, e.g. "ela cheyali") → reply in Telgish Roman script, NO Telugu script
   - If user asked in English → reply in English
   Never switch to a different language or script than the user used."""

# ── Normalize prompt: LLM converts any input to clean English for embedding ───
# This is critical — FAISS was indexed in English, so we must search in English.
# The LLM handles typos, mixed scripts, informal phrasing — far better than regex.
NORMALIZE_PROMPT = """You are a query understanding assistant.

Your job:
1. Read the user's question — it may be in English, Hindi, Telugu, Gujarati, Tamil,
   Hinglish (Hindi+English in Roman script), or Telgish (Telugu+English in Roman script),
   with possible typos or informal spelling.
2. Output ONLY a clean English version of the question that captures the full intent.
3. Do not add any explanation, preamble, or punctuation beyond the question itself.

Examples:
- "na laptop return ela cheyali"         → "How do I return my laptop?"
- "Return polcy eni rojulu?"             → "How many days is the return policy?"
- "warranty kya hai laptop ka?"          → "What is the warranty on laptops?"
- "screen damage cover avutunda?"        → "Is screen damage covered under warranty?"
- "refund entha rojulu padadam?"         → "How many days does a refund take?"
- "accessories ki warranty kitni hai?"   → "What is the warranty for accessories?"
- "phone lo DOA ante enti?"              → "What is DOA policy for phones?"
- "TV panel warranty enti?"              → "What is the warranty on TV panels?"

Output ONLY the English question."""

# ── Style detect prompt: LLM classifies input language style ──────────────────
# Script rules come FIRST — if Devanagari is present, it's "hindi" not "hinglish".
# This was the root cause of Hindi questions being answered in Hinglish.
STYLE_DETECT_PROMPT = """Classify the language style of the following user message.
Reply with EXACTLY one of these labels and nothing else:
- english
- hindi
- telugu
- gujarati
- tamil
- hinglish
- telgish

SCRIPT RULES (check these first, they override everything):
- If the message contains Devanagari script (like की, है, वारंटी, लैपटॉप) → label is "hindi"
- If the message contains Telugu script (like వారంటీ, రిఫండ్, లాప్‌టాప్) → label is "telugu"
- If the message contains Gujarati script (like વોરંટી, લેપટોપ, રિફંડ) → label is "gujarati"
- If the message contains Tamil script (like வாரண்டி, லேப்டாப், ரிஃபண்ட்) → label is "tamil"
- If the message is ONLY Roman script with Hindi words (kya, hai, kitne, nahi, mein) → "hinglish"
- If the message is ONLY Roman script with Telugu words (ela, enti, undi, rojulu, cheyali) → "telgish"
- If the message is ONLY Roman script English → "english"

Message: {query}
Label:"""

# ── Fallback messages when answer is not in policy docs ───────────────────────
FALLBACK_EN = (
    "I'm here to help. I couldn't find that in DROMA policy docs yet. "
    "Please ask about warranties, returns/refunds, service timelines, or accessories."
)

FALLBACK_BY_STYLE = {
    "english":  FALLBACK_EN,
    "hindi":    "मुझे इस बारे में जानकारी नहीं है। कृपया DROMA सपोर्ट से संपर्क करें।",
    "telugu":   "ఈ విషయంలో నాకు సమాచారం లేదు. దయచేసి DROMA సపోర్ట్‌ని సంప్రదించండి.",
    "gujarati": "મારી પાસે આ વિશે માહિતી નથી. કૃપા કરીને DROMA સપોર્ટનો સંપર્ક કરો.",
    "tamil":    "இது பற்றி என்னிடம் தகவல் இல்லை. DROMA ஆதரவை தொடர்பு கொள்ளவும்.",
    "hinglish": "Is baare mein mujhe information nahi hai. Please DROMA support se contact karein.",
    "telgish":  "Ee vishayam gurinchi naaku information ledu. Please DROMA support ni contact cheyyandi.",
}

# ── Greeting detection + replies ──────────────────────────────────────────────
# Short greetings are intercepted early — no need to hit FAISS for "hello"
GREETING_WORDS = {
    "hi", "hello", "hey", "hola", "namaste", "namaskaram",
    "vanakkam", "hii", "hlo", "hai", "sup", "yo",
}

GREETING_REPLIES = {
    "english":  "Hi! I can help with DROMA warranties, refunds, repairs, and accessories.",
    "hindi":    "नमस्ते! मैं DROMA की वारंटी, रिफंड, रिपेयर और एक्सेसरीज़ में मदद कर सकता हूँ।",
    "telugu":   "హాయ్! DROMA వారంటీ, రిఫండ్, రిపేర్, యాక్సెసరీస్ గురించి సహాయం చేస్తాను.",
    "gujarati": "નમસ્તે! હું DROMA ની વોરંટી, રિફંડ, રિપેર અને એક્સેસરીઝ વિશે મદદ કરી શકું છું.",
    "tamil":    "வணக்கம்! DROMA வாரண்டி, ரிஃபண்ட், ரிப்பேர் மற்றும் ஆக்சஸரீஸ் பற்றி உதவ முடியும்.",
    "hinglish": "Hey! DROMA warranties, refunds, repairs aur accessories ke baare mein help kar sakta hoon.",
    "telgish":  "Hey! DROMA warranties, refunds, repairs, accessories gurinchi help cheyagalanu.",
}

# ── Load API key + vector store paths ─────────────────────────────────────────
load_dotenv()
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

index_path  = os.path.join(STORE_DIR, "index.faiss")
chunks_path = os.path.join(STORE_DIR, "chunks.pkl")
meta_path   = os.path.join(STORE_DIR, "meta.pkl")


def ensure_vector_store() -> None:
    """Auto-run ingest.py if the vector store doesn't exist yet."""
    if (
        os.path.exists(index_path)
        and os.path.exists(chunks_path)
        and os.path.exists(meta_path)
    ):
        return
    print("vector_store missing. Running ingest.py...")
    result = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"Failed to build vector_store: {detail}")
    if not (
        os.path.exists(index_path)
        and os.path.exists(chunks_path)
        and os.path.exists(meta_path)
    ):
        raise SystemExit("ingest.py finished but vector_store files were not created.")


# Run at startup — ensures FAISS index is ready before first query
ensure_vector_store()

# Load FAISS index (the searchable vector database of policy chunks)
index = faiss.read_index(index_path)

# Load the actual text chunks (what gets shown to the LLM as context)
with open(chunks_path, "rb") as f:
    CHUNKS = pickle.load(f)

# Load metadata: which embedding model was used + vector dimensions
with open(meta_path, "rb") as f:
    META = pickle.load(f)

EMBED_MODE  = META.get("embed_mode", "openai")        # "openai" or "local"
EMBED_MODEL = META.get("embed_model", OPENAI_EMBED_MODEL)
EMBED_DIM   = int(META.get("embed_dim", index.d))     # Must match index dimensions

# Sanity checks before starting
if EMBED_DIM != index.d:
    raise SystemExit(f"Index/meta dim mismatch: {index.d} vs {EMBED_DIM}")
if EMBED_MODE == "openai" and not openai_key:
    raise SystemExit("OPENAI_API_KEY required for this vector_store. Re-run ingest.py or add key.")
if not openai_key:
    raise SystemExit("Set OPENAI_API_KEY in .env")

# Two separate OpenAI clients: one for embeddings, one for LLM calls
embed_client = OpenAI(api_key=openai_key)
llm_client   = OpenAI(api_key=openai_key)

# Only load local embedding model if ingest.py used it
if EMBED_MODE == "local":
    from sentence_transformers import SentenceTransformer
    local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)


# ── Helper functions ──────────────────────────────────────────────────────────

def detect_style(query: str) -> str:
    """
    Use LLM to classify the user's language style.
    Returns one of: english / hindi / telugu / gujarati / tamil / hinglish / telgish
    Falls back to 'english' on any error.
    Script-based rules in the prompt prevent Hindi from being misclassified as Hinglish.
    """
    try:
        resp = llm_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "user", "content": STYLE_DETECT_PROMPT.format(query=query)}
            ],
            temperature=0,   # Deterministic — always same output for same input
            max_tokens=10,   # Only needs one word back
        )
        label = resp.choices[0].message.content.strip().lower()
        valid = {"english", "hindi", "telugu", "gujarati", "tamil", "hinglish", "telgish"}
        return label if label in valid else "english"
    except Exception:
        return "english"


def normalize_to_english(query: str) -> str:
    """
    Use LLM to convert any language/style/typo query into clean English.
    This English version is used ONLY for embedding + FAISS search — not shown to user.
    Example: "na laptop return ela cheyali" → "How do I return my laptop?"
    """
    try:
        resp = llm_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": NORMALIZE_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return query  # If normalization fails, use raw query (better than crashing)


def get_embedding(text: str) -> np.ndarray:
    """Convert text to a float32 vector using either OpenAI or local model."""
    if EMBED_MODE == "openai":
        vec = embed_client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    else:
        vec = local_embedder.encode([text], normalize_embeddings=True)[0]
    return np.array([vec], dtype=np.float32)


def retrieve_context(english_query: str) -> str:
    """
    Embed the English query and search FAISS for the top-K most similar policy chunks.
    Returns them joined as a single context string for the LLM.
    """
    vector = get_embedding(english_query)
    if vector.shape[1] != EMBED_DIM:
        raise ValueError("Embedding dimension mismatch.")
    _, ids = index.search(vector, TOP_K)
    chunks = [CHUNKS[int(i)] for i in ids[0] if 0 <= int(i) < len(CHUNKS)]
    return "\n\n".join(chunks)


def is_greeting(text: str) -> bool:
    """Return True if the message is just a greeting (no real question to answer)."""
    q = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    if not q:
        return False
    words = q.split()
    return q in GREETING_WORDS or (len(words) <= 2 and any(w in GREETING_WORDS for w in words))


def build_style_instruction(style: str) -> str:
    """
    Build a precise style instruction injected into the user prompt.
    For Hinglish/Telgish: explicitly shows example style + bans native script.
    For native scripts: enforces correct script.
    This is separate from SYSTEM_PROMPT to keep instructions close to the question.
    """
    base = f"\n\nCRITICAL INSTRUCTION: The user wrote in {style} style. Reply ONLY in {style} style.\n"

    if style == "hinglish":
        return base + (
            "Hinglish = Roman script only, naturally mixing Hindi + English words. "
            "Example: 'Aapka refund 7-10 business days mein process hoga. "
            "Physical damage warranty mein cover nahi hota.' "
            "DO NOT use Devanagari script at all."
        )
    elif style == "telgish":
        return base + (
            "Telgish = Roman script only, naturally mixing Telugu + English words. "
            "Example: 'Mee refund 7-10 rojullo process avutundi. "
            "Physical damage warranty lo cover avvadu.' "
            "DO NOT use Telugu script at all."
        )
    elif style == "hindi":
        return base + "Reply fully in Hindi using Devanagari script only."
    elif style == "telugu":
        return base + "Reply fully in Telugu using Telugu script only."
    elif style == "gujarati":
        return base + "Reply fully in Gujarati using Gujarati script only."
    elif style == "tamil":
        return base + "Reply fully in Tamil using Tamil script only."
    else:
        return base + "Reply in English only."


# ── Main answer pipeline ──────────────────────────────────────────────────────

def answer(query: str, history: list) -> str:
    """
    Full RAG pipeline:
    1. Detect style  →  2. Greetings check  →  3. Normalize to English
    →  4. FAISS retrieval  →  5. LLM answer in user's style
    """
    if not query or not query.strip():
        return "Please enter a question."

    query = query.strip()

    # Step 1: Ask LLM to classify language style (hindi / hinglish / telgish / etc.)
    style = detect_style(query)

    # Step 2: Short-circuit for greetings — no need to search policy docs
    if is_greeting(query):
        return GREETING_REPLIES.get(style, GREETING_REPLIES["english"])

    try:
        # Step 3: Convert query to clean English for FAISS search
        # (FAISS was indexed in English, so searching in Telugu/Hinglish gives bad results)
        english_query = normalize_to_english(query)

        # Step 4: Retrieve top-K relevant policy chunks from vector store
        context = retrieve_context(english_query)

        # Step 5: Build reply instruction based on detected style
        style_instruction = build_style_instruction(style)

        # Combine context + original question + style instruction into one prompt
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"User's original question: {query}\n"
            f"(Normalized for retrieval: {english_query})"
            f"{style_instruction}"
        )

        # Step 6: Call LLM to generate a grounded, style-matched answer
        resp = llm_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,  # Low = factual and consistent; increase for more conversational tone
        )

        result = resp.choices[0].message.content.strip()

        # Step 7: If LLM couldn't find info, return a language-appropriate fallback
        if "i don't have information on that" in result.lower():
            return FALLBACK_BY_STYLE.get(style, FALLBACK_EN)

        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        return "I had a temporary issue processing that. Please try again."


# ── Gradio web UI ─────────────────────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=answer,
    title="DROMA Electronics Support Assistant",
    description=(
        "Multilingual policy helper for warranty, refunds, service timelines, and accessories. "
        "Ask in English, Hindi, Telugu, Gujarati, Tamil, Hinglish, or Telgish."
    ),
    examples=[
        "Is laptop screen damage covered under warranty?",
        "DROMA लैपटॉप की वारंटी कितने साल की है?",   # Hindi (Devanagari)
        "na laptop return ela cheyali",                 # Telgish
        "Return policy kya hai kitne din ka?",          # Hinglish
        "screen damage cover avutunda warranty lo?",    # Telgish
        "refund kitne din mein milega?",                # Hinglish
        "DROMA લેપટોપની વોરંટી કેટલા વર્ષની છે?",      # Gujarati
        "DROMA லேப்டாப் வாரண்டி எத்தனை வருடம்?",       # Tamil
    ],
    textbox=gr.Textbox(
        placeholder="Ask your DROMA question here... (any language or style)",
        lines=2,
        max_lines=4,
        show_label=False,
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)