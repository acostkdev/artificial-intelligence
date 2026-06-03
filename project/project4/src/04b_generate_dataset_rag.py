import json
import pickle
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "models/faiss_index.bin"
META_FILE = "models/chunks_meta.pkl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INPUT_DATASET = "dataset/tutor_dataset.jsonl"
OUTPUT_DATASET = "dataset/tutor_dataset_rag.jsonl"

SYSTEM_PROMPT = (
    "Eres un tutor académico especializado en seguridad pública en México. "
    "Responde solo con información de los documentos. No inventes nada. "
    "Si no hay datos suficientes, dilo claramente."
)

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading chunk metadata...")
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)
print(f"Index: {index.ntotal} vectors, {len(chunks)} chunks")

print("Loading existing dataset...")
examples = []
with open(INPUT_DATASET, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))
print(f"Dataset: {len(examples)} examples")

def retrieve(q, k=3):
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        c = chunks[idx]
        results.append({
            "score": round(float(score), 3),
            "text": c["text"].strip()[:300],
            "doc_name": c.get("doc_name", c.get("doc_id", "?")),
            "chunk_id": c.get("chunk_id", "?"),
        })
    return results

def build_instruction(question, context):
    ctx_lines = []
    for r in context:
        ctx_lines.append(f"[{r['doc_name']} / {r['chunk_id']}]\n{r['text']}")
    ctx_str = "\n\n".join(ctx_lines)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Documentos:\n{ctx_str}\n\n"
        f"Pregunta: {question}"
    )

new_examples = []
for ex in examples:
    question = ex["instruction"]
    ctx = retrieve(question)
    instruction = build_instruction(question, ctx)
    new_examples.append({
        "instruction": instruction,
        "output": ex["output"],
        "level": ex["level"],
    })

os.makedirs("dataset", exist_ok=True)
with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
    for ex in new_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nDataset RAG guardado: {len(new_examples)} ejemplos en {OUTPUT_DATASET}")
print("Ejemplo:")
print(f"  Instruction:\n{new_examples[0]['instruction'][:200]}...")
print(f"  Output:\n{new_examples[0]['output'][:100]}...")
