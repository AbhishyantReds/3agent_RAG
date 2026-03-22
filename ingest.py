import glob
import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_GLOB = "data/*.txt"
STORE_DIR = "vector_store"

load_dotenv()
api_key = (os.getenv("OPENAI_API_KEY") or "").strip()

files = glob.glob(DATA_GLOB)
if not files:
    raise SystemExit("No .txt files found in data/")

texts = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        texts.append(f.read().strip())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = [c for t in texts for c in splitter.split_text(t) if c.strip()]
if not chunks:
    raise SystemExit("No chunks created. Check your data files.")

client = OpenAI(api_key=api_key)
if api_key:
    emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=chunks)
    vectors = np.array([d.embedding for d in emb.data], dtype=np.float32)
    embed_mode, embed_model = "openai", OPENAI_EMBED_MODEL
else:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    vectors = np.array(model.encode(chunks, normalize_embeddings=True), dtype=np.float32)
    embed_mode, embed_model = "local", LOCAL_EMBED_MODEL

embed_dim = int(vectors.shape[1])

os.makedirs(STORE_DIR, exist_ok=True)
index = faiss.IndexFlatL2(embed_dim)
index.add(vectors)
faiss.write_index(index, os.path.join(STORE_DIR, "index.faiss"))
with open(os.path.join(STORE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)
with open(os.path.join(STORE_DIR, "meta.pkl"), "wb") as f:
    pickle.dump({"embed_mode": embed_mode, "embed_model": embed_model, "embed_dim": embed_dim}, f)

print(f"Indexed {len(chunks)} chunks from {len(files)} files using {embed_mode} embeddings")
