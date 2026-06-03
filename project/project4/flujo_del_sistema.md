# Flujo del Sistema — Tutor Analítico Híbrido (RAG + LoRA)

## Arquitectura general

```
PDFs (3 documentos)
    │
    ▼
┌──────────────────────────────┐
│  01_extract_text.py          │  Extrae texto + tablas de PDFs
│  PyMuPDF (fitz)              │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  02_chunking.py              │  Divide texto en fragmentos de ~700 chars
│  Chunking recursivo          │  con solapamiento de 100 chars
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  03_embeddings_faiss.py      │  Genera vectores (384 dim) con
│  sentence-transformers       │  sentence-transformers e índice FAISS
│  + FAISS                     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  04_generate_dataset.py      │  83 ejemplos instruction→output
│  + 04b_generate_dataset_rag  │  en 3 niveles de dificultad
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  05_finetune_phi3.py         │  LoRA fine-tuning sobre Phi-3-mini
│  transformers + PEFT + TRL   │  Solo entrena 0.62% de parámetros
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  06_inference_phi3.py        │  Pipeline completo:
│  RAG + modelo fine-tuned     │  1) FAISS busca chunks relevantes
│                              │  2) Modelo genera respuesta
└──────────────────────────────┘
```

---

## 01_extract_text.py — Extracción de texto de PDFs

**Biblioteca**: `fitz` (PyMuPDF)

**Qué hace**: Abre cada PDF, extrae el texto de todas las páginas usando `page.get_text()`, y también extrae tablas con `page.find_tables()` para preservar estructura tabular. Concatena todo separado por `--- PAGE BREAK ---`.

```python
# Extracción de tablas con coordenadas
ft = page.find_tables()
for t in ft.tables:
    tables_text += table_to_text(t)
```

**Partes importantes**:
- `table_to_text()`: Convierte cada tabla PDF a formato texto con tuberías (`|`) separando columnas, etiquetado con `[TABLA]...[/TABLA]`
- `page.get_text()`: Extrae texto plano preservando orden de lectura pero pierde estructura de tablas
- `find_tables()`: PyMuPDF detecta tablas automáticamente y extrae celdas con coordenadas

**Salida**: `corpus_raw/{nombre}.txt` — un archivo por PDF con todo el texto extraído.

**Limitación**: PyMuPDF no siempre reconstruye tablas complejas correctamente. Tablas con celdas fusionadas, bordes irregulares o sin bordes explícitos pueden extraerse mal.

---

## 02_chunking.py — Fragmentación del texto

**Divide el texto en chunks** de ~700 caracteres con solapamiento de 100 caracteres.

```python
def recursive_chunk(text, size=700, overlap=100):
```

**Algoritmo**:
1. Avanza `size` caracteres desde el inicio
2. Busca un corte natural: primero salto de línea (`\n`), luego punto seguido (`. `)
3. Si no encuentra corte natural en la primera mitad del chunk, corta forzado en `end`
4. Retrocede `overlap` caracteres para el siguiente chunk (solapamiento)

```
Ejemplo de chunking:
    [Chunk 1: 700 chars]────┐
                            ├── 100 chars de solapamiento
    ┌───────────────────────┘
    [Chunk 2: 700 chars]────┐
                            ├── 100 chars de solapamiento
    ┌───────────────────────┘
    [Chunk 3: ...]
```

**Cada chunk incluye metadatos**:
```json
{
  "doc_id": "Documento 1",
  "doc_name": "Documento 1: ONC - Homicidio...",
  "chunk_id": "Documento 1_chunk_1166",
  "text": "...contenido del fragmento..."
}
```

**Filtro**: Se descartan chunks con menos de 50 caracteres (páginas en blanco, encabezados sueltos).

**Salida**: `chunks/chunks.jsonl` — ~1322 chunks en formato JSON Lines.

---

## 03_embeddings_faiss.py — Vectorización e índice

**Modelo de embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
- 384 dimensiones por vector
- Multilingüe (español incluido)
- Normaliza vectores automáticamente (`normalize_embeddings=True`)

**Índice FAISS**: `IndexFlatIP` (Inner Product)
- Búsqueda exacta por fuerza bruta (compara contra todos los vectores)
- Con vectores normalizados, el producto punto equivale a similitud coseno

```python
embeddings = model.encode(texts, normalize_embeddings=True)
# embeddings.shape → (1322, 384)

index = faiss.IndexFlatIP(384)  # Producto punto = coseno (vectores normalizados)
index.add(embeddings)           # Añade todos los vectores
```

**Salida dual**:
- `models/faiss_index.bin`: Índice binario FAISS (los 1322 vectores)
- `models/chunks_meta.pkl`: Metadatos de los chunks (pickle) — necesario para recuperar texto original desde los índices

**Búsqueda en inferencia**: Dada una pregunta, se genera su embedding y FAISS devuelve los índices de los k chunks más cercanos con sus scores de similitud.

---

## 04_generate_dataset.py — Dataset de fine-tuning

Genera 83 pares **instruction → output** en 3 niveles de dificultad.

### Distribución
| Nivel | Tipo | Cantidad | Comportamiento |
|-------|------|----------|----------------|
| 1 | Extracción directa (factoid) | 29 | Respuesta literal con cita: `[Documento N, Pág M]` |
| 2 | Síntesis | 21 | Respuesta estructurada + pregunta de reflexión al final |
| 3 | Razonamiento analítico | 33 | Método socrático (preguntas guía antes de responder) + manejo de incertidumbre |

### Estructura de cada ejemplo
```json
{
  "instruction": "¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?",
  "output": "Para abordar esta pregunta, los datos del ONC indican que... [Documento 1, Pág 29]",
  "level": 1
}
```

### Helpers de estilo académico
```python
contextos = [
    "Para abordar esta pregunta,", "Cabe señalar que,",
    "En primer lugar, conviene precisar que,", ...
]
conectores = [
    "No obstante,", "Por otra parte,", "En consecuencia,", ...
]
cierres = [
    "En conclusión,", "Por lo tanto,", "Así pues,", ...
]
```

Estos se concatenan aleatoriamente para variar el fraseo de las respuestas.

### Comportamientos enseñados
- **Citar fuentes**: `[Documento 1, Pág 29]` al final de cada respuesta
- **Método socrático** (nivel 3): Comienza con preguntas como _"Antes de responder, permítame preguntar: ¿considera usted que...?"_
- **Manejo de incertidumbre**: _"El corpus disponible no contiene información específica sobre... No es posible ofrecer una respuesta fundamentada."_
- **Tono académico**: Conectores lógicos, estructura predecible

**Salida**: `dataset/tutor_dataset.jsonl` (83 líneas JSON)

---

## 04b_generate_dataset_rag.py — Aumento del dataset con RAG

Toma los 83 ejemplos base y **pre-recupera chunks reales** de FAISS para cada pregunta, incrustando ese contexto en la instrucción.

### Técnica: RALT / RA-DIT
**R**etrieval-**A**ugmented **L**anguage **T**raining — también conocida como RA-DIT.

La idea: durante el entrenamiento el modelo ve instrucciones que YA contienen chunks de documentos (exactamente como en inferencia). Así aprende a **usar el contexto** en lugar de memorizar respuestas.

### Transformación
**Antes** (instruction original):
```
¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?
```

**Después** (instruction RAG):
```
Eres un tutor académico especializado en seguridad pública en México.
Responde solo con información de los documentos. No inventes nada.
Si no hay datos suficientes, dilo claramente.

Documentos:
[Documento 1: ONC - Homicidio / chunk_1166]
texto del chunk...

[Documento 2: INEGI - ENVIPE 2025 / chunk_0044]
texto del chunk...

Pregunta: ¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?
```

```python
def build_instruction(question, context):
    ctx_str = "\n\n".join([f"[{r['doc_name']} / {r['chunk_id']}]\n{r['text']}" for r in context])
    return f"{SYSTEM_PROMPT}\n\nDocumentos:\n{ctx_str}\n\nPregunta: {question}"
```

**Salida**: `dataset/tutor_dataset_rag.jsonl` — 83 ejemplos con contexto incrustado. Este es el dataset que se usa para fine-tuning.

---

## 05_finetune_phi3.py — LoRA Fine-tuning

### Modelo base
- **Nombre**: `microsoft/Phi-3-mini-4k-instruct`
- **Parámetros**: 3.8B (3,800 millones)
- **Ventana de contexto**: 4,096 tokens

### Cuantización 4-bit (NF4)
Para que el modelo quepa en una GPU GTX 1050 (4GB VRAM):
- FP32: ~14 GB → imposible
- FP16: ~7.5 GB → no cabe
- **NF4**: ~2.5 GB → cabe con margen

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",     # Normal Float 4 — mejor calidad que cuantización lineal
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Double quantization — ahorra ~0.5GB extra
)
```

### LoRA (Low-Rank Adaptation)
No entrena todos los parámetros. Solo entrena **matrices pequeñas** insertadas en las capas de atención.

```python
lora_config = LoraConfig(
    r=8,                              # Rango de las matrices LoRA
    lora_alpha=16,                    # Factor de escala
    target_modules=[                   # Capas donde insertar LoRA
        "qkv_proj", "o_proj",          # Atención (self-attention)
        "gate_up_proj", "down_proj",   # Feed-forward
    ],
    lora_dropout=0.1,
)
```

| Parámetros totales | Entrenables (LoRA) | % |
|---|---|---|
| 2,021,723,136 | 12,582,912 | 0.62% |

### Formato de entrenamiento
Cada ejemplo se formatea con el chat template de Phi-3:
```
<|user|>\n{instruction (con RAG context)}<|end|>\n
<|assistant|>\n{output}<|end|>\n<|endoftext|>
```

### Configuración de entrenamiento
```python
training_args = SFTConfig(
    per_device_train_batch_size=1,     # Solo cabe 1 ejemplo a la vez
    gradient_accumulation_steps=4,     # Simula batch_size=4
    gradient_checkpointing=True,       # Ahorra VRAM (recalcula activaciones)
    num_train_epochs=3,                # 3 épocas sobre 83 ejemplos
    learning_rate=2e-4,                # Tasa típica para LoRA
    max_length=768,                    # Límite de tokens por secuencia
)
```

**Progreso del entrenamiento** (última corrida, 83 ejemplos):
```
Step  5/63: loss = 1.682
Step 10/63: loss = 1.431
Step 20/63: loss = 1.379
Step 30/63: loss = 1.231
Step 40/63: loss = 1.144
Step 50/63: loss = 1.057
Step 60/63: loss = 0.958
```
La pérdida baja consistentemente: el modelo está aprendiendo. El valor final de 0.958 es el mejor obtenido.

**Salida**: `models/lora_adapter_phi3_rag/` — adaptador LoRA (~25MB) + tokenizer

---

## 06_inference_phi3.py — Pipeline RAG + Generación

### Flujo completo por cada pregunta

```
Pregunta del usuario
    │
    ▼
┌──────────────────────┐
│ 1. Embedding         │  SentenceTransformer codifica la pregunta
│    de la pregunta    │  → vector 384-dim (normalizado)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Búsqueda FAISS    │  IndexFlatIP busca top-k vectores
│    (k=3)             │  → scores de similitud coseno
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Recuperar chunks  │  Con los índices devueltos por FAISS,
│    de metadata       │  se obtienen los textos originales
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Construir prompt  │  System prompt + chunks + pregunta
│                      │  en formato Phi-3 (<|user|>...<|end|>)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. Generar respuesta │  Phi-3-mini + LoRA generan
│    (autoregresivo)   │  hasta 300 tokens nuevos
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Guardar resultado │  Se guarda incrementalmente en
│                      │  results/evaluation_phi3.json
└──────────────────────┘
```

### 1. Recuperación (FAISS)
```python
def retrieve(q, k=3):
    q_emb = embedder.encode([q], normalize_embeddings=True)  # (1, 384)
    scores, indices = index.search(q_emb.astype(np.float32), k)  # top-3
    # scores: [0.747, 0.746, 0.746] — similitud coseno
    # indices: [1166, 948, 818] — índices en chunks_meta.pkl
    return [chunks[idx] for idx in indices[0]]
```

### 2. Construcción del prompt
El prompt replica EXACTAMENTE el formato visto durante el fine-tuning:
```
<|user|>
Eres un tutor académico especializado en seguridad pública en México.
Responde solo con información de los documentos. No inventes nada.
Si no hay datos suficientes, dilo claramente.

Documentos:
[Documento 1 / chunk_1166]
texto del chunk 1...

[Documento 1 / chunk_0948]
texto del chunk 2...

Pregunta: ¿Cuáles son las tres entidades federativas...?
<|end|>
<|assistant|>
```

### 3. Generación
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=300,        # Límite de tokens generados
    do_sample=False,           # Modo determinista (greedy decoding)
    repetition_penalty=1.1,    # Penaliza repeticiones
)
```

### 4. Post-procesamiento
Se decodifican solo los tokens generados (no el prompt de entrada):
```python
input_len = inputs.input_ids.shape[1]
generated_ids = outputs[0][input_len:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
```

### Parámetros de generación
| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `max_new_tokens=300` | 300 | Genera hasta 300 tokens nuevos (~225 palabras) |
| `do_sample=False` | False | Greedy decoding — siempre la misma respuesta |
| `repetition_penalty=1.1` | 1.1 | Reduce repetición de frases |
| `pad_token_id=eos_token_id` | — | Usa token de fin como padding |
| `max_length=768` | 768 | Trunca el prompt si excede (input + output) |

### Evaluación
Las respuestas se guardan incrementalmente en `results/evaluation_phi3.json` tras cada pregunta, así que si el proceso se interrumpe no se pierde todo:
```json
{
  "Q1": {
    "question": "¿Cuáles son las tres entidades...?",
    "answer": "Para abordar esta pregunta...",
    "retrieved_chunks": [
      {"score": 0.747, "text": "...", "doc_name": "Documento 1...", "chunk_id": "Documento 1_chunk_1166"}
    ],
    "retrieval_top_score": 0.747
  }
}
```

---

## Dependencias del sistema

| Paquete | Uso |
|---------|-----|
| `PyMuPDF` (fitz) | Extracción de PDFs |
| `sentence-transformers` | Embeddings multilingües (384 dim) |
| `faiss-cpu` | Índice vectorial y búsqueda |
| `transformers` + `torch` | Modelo Phi-3-mini y generación |
| `peft` | LoRA (adaptadores eficientes) |
| `trl` | SFTTrainer (fine-tuning supervisado) |
| `bitsandbytes` | Cuantización 4-bit NF4 |
| `datasets` | Carga del dataset |
| `numpy` | Operaciones numéricas |

## Requisitos de hardware

| Componente | Especificación |
|------------|---------------|
| GPU | GTX 1050 o superior (4GB VRAM mínimo) |
| RAM | 16 GB |
| Disco | ~15 GB para modelos cacheados (Phi-3) |

## Limitaciones conocidas

1. **Pérdida de datos tabulares**: PyMuPDF no siempre preserva tablas correctamente
2. **Alucinación residual**: Con 83 ejemplos y entrenamiento anti-invención, el modelo ya no inventa fuentes/siglas, pero aún inventa cifras en preguntas que requieren datos tabulares (Q1, Q3)
3. **Ventana de contexto limitada**: `max_length=768` restringe la cantidad de chunks por pregunta
4. **Velocidad**: ~8-12 tokens/segundo en GTX 1050 (cada pregunta toma ~3-5 minutos)
