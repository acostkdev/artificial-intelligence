import os
import json

RAW_DIR = "corpus_raw"
CHUNK_DIR = "chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

docs_map = {
    "ONC-HOMICIDIO.txt": "Documento 1: ONC - Homicidio: Una mirada a la violencia en México",
    "ENVIPE_2025.txt": "Documento 2: INEGI - ENVIPE 2025",
    "ONC_Anual_2025.txt": "Documento 3: ONC - Reporte Anual de Seguridad 2025",
}

def recursive_chunk(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # try to cut at newline or period
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut < start + size // 2:
            cut = text.rfind(". ", start, end)
        if cut == -1 or cut < start + size // 2:
            cut = end
        else:
            cut = cut + 1  # include the newline or period
        chunks.append(text[start:cut])
        start = cut - overlap
    return chunks

all_chunks = []
for fname, label in docs_map.items():
    path = os.path.join(RAW_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    doc_id = label.split(":")[0].strip()
    chunks = recursive_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 50:
            continue
        all_chunks.append({
            "doc_id": doc_id,
            "doc_name": label,
            "chunk_id": f"{doc_id}_chunk_{i:04d}",
            "text": chunk_text,
        })

outpath = os.path.join(CHUNK_DIR, "chunks.jsonl")
with open(outpath, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Total chunks: {len(all_chunks)}")
for label in docs_map.values():
    doc_id = label.split(":")[0].strip()
    count = sum(1 for c in all_chunks if c["doc_id"] == doc_id)
    print(f"  {label}: {count} chunks")
