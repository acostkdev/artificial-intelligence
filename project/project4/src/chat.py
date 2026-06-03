import json, pickle, faiss, numpy as np, torch, os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index(os.path.join(BASE, "models", "faiss_index.bin"))
chunks = pickle.load(open(os.path.join(BASE, "models", "chunks_meta.pkl"), "rb"))

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(model, os.path.join(BASE, "models/lora_adapter_phi3_rag"))
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

print("Chat listo. Escribe 'salir' para terminar.\n")
while True:
    q = input(">>> ")
    if q.lower() in ("salir", "exit", "quit"):
        break
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), 3)
    ctx = "\n\n".join(
        f"[{chunks[i]['doc_name']}]\n{chunks[i]['text'][:300]}"
        for i in indices[0]
    )
    prompt = (
        "<|user|>\nEres un tutor académico especializado en seguridad pública en México. "
        "Responde solo con información de los documentos. No inventes nada. "
        "Si no hay datos suficientes, dilo claramente.\n\n"
        f"Documentos:\n{ctx}\n\n"
        f"Pregunta: {q}\n<|end|>\n<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=300, do_sample=False,
        repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"\n{answer}\n")
