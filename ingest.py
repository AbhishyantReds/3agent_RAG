"""
VECTOR STORE INGESTION SCRIPT
This script processes documents and builds a vector database (FAISS index).
Steps:
1. Load policy documents from data/ folder
2. Split documents into chunks
3. Convert chunks to embedding vectors
4. Build FAISS index for fast similarity search
5. Save index, chunks, and metadata for use by app.py

Run this once initially, or re-run if you add new documents to data/
"""

import glob
import os
import pickle

import faiss  # Facebook's vector search library
import numpy as np  # Numerical computing for vectors
from dotenv import load_dotenv  # Load API keys from .env
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks
from openai import OpenAI  # Use OpenAI for embeddings

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Parameters for chunking and embedding
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 800  # Size of each text chunk in characters
CHUNK_OVERLAP = 100  # Overlap between chunks so we don't lose context at boundaries
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # Model to convert text to vectors (OpenAI)
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"  # Fallback local model if OpenAI isn't available
DATA_GLOB = "data/*.txt"  # Wildcard pattern to find all text files in data/ folder
STORE_DIR = "vector_store"  # Directory where index will be saved

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Load API key from .env file
load_dotenv()
api_key = (os.getenv("OPENAI_API_KEY") or "").strip()

# Find all text files in data/ folder
files = glob.glob(DATA_GLOB)
if not files:
    raise SystemExit("No .txt files found in data/")

# Read the content of each file
texts = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        texts.append(f.read().strip())

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: CHUNK DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Split documents into smaller chunks
# RecursiveCharacterTextSplitter tries to keep sentences together
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = [c for t in texts for c in splitter.split_text(t) if c.strip()]
if not chunks:
    raise SystemExit("No chunks created. Check your data files.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CONVERT CHUNKS TO EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

client = OpenAI(api_key=api_key)
if api_key:
    # Use OpenAI's embedding API to convert text to vectors
    print(f"Creating embeddings for {len(chunks)} chunks using OpenAI...")
    emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=chunks)
    vectors = np.array([d.embedding for d in emb.data], dtype=np.float32)
    embed_mode, embed_model = "openai", OPENAI_EMBED_MODEL
else:
    # If no API key, fall back to local embedding model
    print(f"Creating embeddings for {len(chunks)} chunks using local model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    vectors = np.array(model.encode(chunks, normalize_embeddings=True), dtype=np.float32)
    embed_mode, embed_model = "local", LOCAL_EMBED_MODEL

# Get the dimension (length) of each embedding vector
embed_dim = int(vectors.shape[1])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD FAISS INDEX
# ═══════════════════════════════════════════════════════════════════════════════

# Create FAISS index directory if it doesn't exist
os.makedirs(STORE_DIR, exist_ok=True)

# Create FAISS index with L2 distance (Euclidean distance for vector similarity)
index = faiss.IndexFlatL2(embed_dim)
# Add all embedding vectors to the index
index.add(vectors)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE INDEX, CHUNKS, AND METADATA
# ═══════════════════════════════════════════════════════════════════════════════

# Save the FAISS index to disk
faiss.write_index(index, os.path.join(STORE_DIR, "index.faiss"))

# Save the actual text chunks (so we can retrieve them later)
with open(os.path.join(STORE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

# Save metadata about how embeddings were created (in case we need to recreate them)
with open(os.path.join(STORE_DIR, "meta.pkl"), "wb") as f:
    pickle.dump({"embed_mode": embed_mode, "embed_model": embed_model, "embed_dim": embed_dim}, f)

# Print summary
print(f"✓ Indexed {len(chunks)} chunks from {len(files)} files using {embed_mode} embeddings")
