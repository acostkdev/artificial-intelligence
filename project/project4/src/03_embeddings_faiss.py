import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks/chunks.jsonl"
INDEX_FILE = "models/faiss_index.bin"
META_FILE = "models/chunks_meta.pkl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
print(f"Generating embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (normalized)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"Index saved: {index.ntotal} vectors, dim={dim}")
print(f"Metadata saved: {len(chunks)} chunks")
