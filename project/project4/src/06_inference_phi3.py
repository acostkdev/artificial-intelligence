import json
import pickle
import os

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "models/lora_adapter_phi3_rag"
INDEX_FILE = "models/faiss_index.bin"
META_FILE = "models/chunks_meta.pkl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cuda"

QUESTIONS = {
    "Q1": "¿Cuáles son las tres entidades federativas con mayor índice de homicidios dolosos según los datos más recientes incluidos en el corpus?",
    "Q2": "¿Qué organizaciones, cárteles o grupos delictivos se mencionan con mayor frecuencia operando en la región de Tierra Caliente?",
    "Q3": "¿Cuáles son las cifras oficiales reportadas sobre el desplazamiento forzado interno a causa de la violencia durante el último sexenio documentado?",
    "Q4": "Según los documentos, ¿cuáles son las principales causas socioeconómicas que los autores asocian directamente al incremento de la violencia urbana?",
    "Q5": "Contrasta las estrategias de seguridad pública mencionadas en el corpus. ¿Qué diferencias de enfoque existen entre la militarización y las políticas de prevención social?",
    "Q6": "¿Cómo ha evolucionado la tasa de delitos de extorsión (cobro de piso) a nivel nacional y qué sectores económicos se reportan como los más afectados?",
    "Q7": "¿Existe alguna diferencia significativa documentada en los tipos de violencia que experimentan las zonas rurales en comparación con las zonas metropolitanas?",
    "Q8": "Con base en las posturas de las ONGs y las fuentes gubernamentales presentes en los textos, ¿cuáles son las principales contradicciones o discrepancias en el registro de víctimas?",
    "Q9": "¿Qué impacto específico tiene la violencia documentada sobre la tasa de deserción escolar en las zonas de alto conflicto?",
    "Q10": "A partir de las conclusiones de los autores en el corpus, ¿qué vacíos de información, subregistros o falta de datos fiables se identifican como el principal obstáculo para medir la violencia real en el país?",
}

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# move embedder to CPU to save VRAM
embedder.to("cpu")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading chunk metadata...")
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Index: {index.ntotal} vectors, {len(chunks)} chunks")

print("Loading Phi-3 in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float32,
)
model.eval()

if os.path.exists(ADAPTER_PATH):
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("LoRA adapters loaded")
else:
    print("No adapters found — using base model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
            "text": c["text"].strip(),
            "doc_id": c.get("doc_id", "?"),
            "doc_name": c.get("doc_name", c.get("doc_id", "?")),
            "chunk_id": c.get("chunk_id", "?"),
        })
    return results

def build_prompt(question, context):
    ctx_lines = []
    for r in context:
        text = r["text"][:300]
        ctx_lines.append(f"[{r['doc_name']} / {r['chunk_id']}]\n{text}")
    ctx_str = "\n\n".join(ctx_lines)

    return (
        "<|user|>\n"
        "Eres un tutor académico especializado en seguridad pública en México. "
        "Responde solo con información de los documentos incluidos abajo. "
        "NO inventes cifras, nombres de instituciones, siglas, acrónimos, "
        "fechas ni citas bibliográficas que no aparezcan explícitamente en "
        "los Documentos. Si los Documentos no contienen la respuesta exacta, "
        "dilo claramente: 'El corpus disponible no contiene información "
        "específica sobre... No es posible ofrecer una respuesta "
        "fundamentada con los documentos disponibles.'\n\n"
        "Documentos:\n"
        f"{ctx_str}\n\n"
        f"Pregunta: {question}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

def generate(question, context):
    prompt = build_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer

os.makedirs("results", exist_ok=True)
results = {}
for qid, question in QUESTIONS.items():
    print(f"\n{'='*70}")
    print(f"  {qid}: {question[:80]}...")

    ctx = retrieve(question)
    print(f"  Retrieved {len(ctx)} chunks (top score: {ctx[0]['score'] if ctx else 'N/A'})")

    answer = generate(question, ctx)

    results[qid] = {
        "question": question,
        "answer": answer,
        "retrieved_chunks": ctx,
        "retrieval_top_score": ctx[0]["score"] if ctx else None,
    }

    print(f"\n  Answer:\n{answer}\n")

    # save incrementally so timeout doesn't lose everything
    with open("results/evaluation_phi3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [saved {len(results)}/{len(QUESTIONS)}]")

print("\n" + "="*70)
print("  All results saved to results/evaluation_phi3.json")
print("="*70)
