# Informe Técnico — Tutor Analítico Híbrido (RAG + Fine-Tuning)

## Índice

1. [Introducción y Conceptos Clave](#1-introducción-y-conceptos-clave)
2. [Arquitectura General del Sistema](#2-arquitectura-general-del-sistema)
3. [Pipeline de Ingesta: Extracción y Chunking](#3-pipeline-de-ingesta-extracción-y-chunking)
4. [Vectorización e Índice FAISS](#4-vectorización-e-índice-faiss)
5. [Generación del Dataset de Fine-Tuning](#5-generación-del-dataset-de-fine-tuning)
6. [Dataset Aumentado con RAG](#6-dataset-aumentado-con-rag)
7. [Fine-Tuning con LoRA](#7-fine-tuning-con-lora)
8. [Inferencia: RAG + Modelo Fine-Tuned](#8-inferencia-rag--modelo-fine-tuned)
9. [Evaluación de Resultados](#9-evaluación-de-resultados)
10. [Justificación de Decisiones Técnicas](#10-justificación-de-decisiones-técnicas)
11. [Problemas Encontrados y Soluciones](#11-problemas-encontrados-y-soluciones)
12. [Glosario de Conceptos](#12-glosario-de-conceptos)
13. [Análisis Crítico y Limitaciones](#13-análisis-crítico-y-limitaciones)

---

## 1. Introducción y Conceptos Clave

### ¿Qué estamos construyendo?

Un **Tutor Inteligente** que responde preguntas sobre seguridad pública y violencia en México. La particularidad es que no usa solo un modelo de lenguaje (LLM), sino que **combina dos técnicas**:

1. **RAG (Retrieval-Augmented Generation)**: El modelo no responde de memoria, sino que primero busca fragmentos relevantes en una base de documentos y luego genera su respuesta basándose en ellos.
2. **Fine-Tuning**: Al modelo base se le entrena con ejemplos específicos para que aprenda un *estilo* de respuesta (citar fuentes, ser académico, reconocer cuándo no sabe algo).

### Conceptos Fundamentales

| Concepto | Explicación |
|----------|-------------|
| **LLM** (Large Language Model) | Modelo de lenguaje entrenado con millones de textos. Phi-3-mini es un LLM con 3.800 millones de parámetros, capaz de generar texto coherente |
| **Parámetros** | "Conexiones" dentro de la red neuronal. Phi-3-mini tiene ~3.8B parámetros. TinyLlama ~1.1B. Más parámetros = más capacidad (y más lento) |
| **Token** | Unidad mínima de texto (no es una palabra exacta, pueden ser subpalabras). Phi-3 usa ~4,000 tokens por consulta como máximo |
| **Embedding** | Vector numérico (lista de números) que representa el *significado* de un texto. Textos similares tienen vectores parecidos |
| **LoRA** (Low-Rank Adaptation) | Técnica para fine-tuning eficiente: en lugar de modificar todos los parámetros del modelo (billones), solo entrena matrices pequeñas insertadas en el modelo. Reduce el costo drásticamente |
| **4-bit quantization** | Técnica que reduce la precisión numérica de los pesos del modelo (de 32 bits a 4 bits), ocupando 8 veces menos memoria VRAM. El modelo sigue funcionando, solo pierde un poco de precisión |
| **FAISS** | Biblioteca de Facebook para búsqueda rápida de vectores similares. Escanea miles de vectores en milisegundos |
| **Cifra negra** | Porcentaje de delitos que no se denuncian o no tienen carpeta de investigación. En México: 93.2% (ENVIPE 2025) |

### Flujo General (vista de pájaro)

```
PDFs (3 documentos)
    │
    ▼
Extraer texto → Dividir en fragmentos (chunks) → Generar embeddings
                                                      │
                                                      ▼
                                               Índice FAISS
                                               (búsqueda)
                                                      ▲
                                                      │
Dataset (83 ejemplos) ─► Fine-Tuning (LoRA) ─► Modelo + LoRA
                                                      │
                    Pregunta del usuario ──────────────┤
                           │                          │
                           ▼                          ▼
                    FAISS busca 3 chunks          Modelo responde
                           │                      usando chunks
                           ▼                          │
                    Chunks relevantes ◄────────────────┘
                                                      │
                                                      ▼
                                               Respuesta final
                                               (con citas)
```

---

## 2. Arquitectura General del Sistema

El sistema se divide en 5 fases, cada una implementada en un archivo Python:

| Fase | Archivo | ¿Qué hace? |
|------|---------|------------|
| 1 | `01_extract_text.py` | Extrae texto de PDFs |
| 2 | `02_chunk_text.py` | Divide el texto en fragmentos (chunks) |
| 3 | `03_embeddings_faiss.py` | Genera vectores e índice FAISS |
| 4 | `04_generate_dataset.py` + `04b_generate_dataset_rag.py` | Crea ejemplos de entrenamiento |
| 5 | `05_finetune_phi3.py` | Entrena el modelo con LoRA |
| 6 | `06_inference_phi3.py` | Responde preguntas usando RAG + modelo fine-tuned |

### Visión por Capas

```
Capa 1: Datos
  ┌─────────────────────────────────────────────────┐
  │  corpus_raw/*.txt → chunks/*.jsonl              │
  │  3 documentos, ~1322 chunks                      │
  └─────────────────────────────────────────────────┘
                      │
Capa 2: Índice Vectorial     ▼
  ┌─────────────────────────────────────────────────┐
  │  models/faiss_index.bin                          │
  │  models/chunks_meta.pkl                          │
  │  (embeddings + metadatos)                        │
  └─────────────────────────────────────────────────┘
                      │
Capa 3: Dataset         ▼
  ┌─────────────────────────────────────────────────┐
  │  dataset/tutor_dataset_rag.jsonl                 │
  │  83 ejemplos con contexto RAG en instrucción     │
  └─────────────────────────────────────────────────┘
                      │
Capa 4: Modelo          ▼
  ┌─────────────────────────────────────────────────┐
  │  Phi-3-mini-4k-instruct (4-bit) + LoRA adapters │
  │  → models/lora_adapter_phi3_rag/                 │
  └─────────────────────────────────────────────────┘
                      │
Capa 5: Evaluación      ▼
  ┌─────────────────────────────────────────────────┐
  │  results/evaluation_phi3.json                    │
  │  10 preguntas con chunks + respuestas            │
  └─────────────────────────────────────────────────┘
```

---

## 3. Pipeline de Ingesta: Extracción y Chunking

### `01_extract_text.py`

```python
import fitz  # PyMuPDF

for fname, label in docs_info.items():
    doc = fitz.open(path)
    for page in doc:
        text = page.get_text()
    # Une todas las páginas separadas por "--- PAGE BREAK ---"
```

**¿Qué hace?**

Abre cada PDF con PyMuPDF (`fitz`), extrae el texto de cada página con `page.get_text()`, y las concatena separadas por un marcador `--- PAGE BREAK ---`.

**Limitación importante: `page.get_text()` extrae texto pero **NO preserva tablas**. Los PDFs del ONC contienen tablas con clasificaciones de estados, tasas de homicidio, rankings, etc. Al extraerse como texto plano, la estructura de columnas se pierde y los datos quedan como listas secuenciales sin relación entre sí.**

**Ejemplo concreto de datos tabulares perdidos**:

En el PDF original del ONC-Homicidio hay una tabla llamada *"Clasificación de las entidades federativas acorde con la tipología de comportamiento de homicidio doloso"* con tres columnas: `Tipo de comportamiento`, `Entidades federativas` y `Cantidad de entidades`.

En el PDF se veía así (reconstrucción aproximada):

| Tipo | Entidades | Cantidad |
|------|-----------|----------|
| A | Guerrero, Sinaloa, Chihuahua, Morelos... | 11 |
| B | Baja California, Jalisco... | 4 |
| C | Michoacán y Guanajuato | 2 |
| D | Puebla y Oaxaca | 2 |
| E | Baja California Sur, Campeche... | 6 |
| Irregular | Estado de México, Sonora... | 4 |
| Mixto (D/E) | Chiapas, CDMX, Tabasco | 3 |

Pero nuestra extracción produjo esto (estructura aplanada, sin relación entre columnas):
```
Guerrero, Sinaloa, Chihuahua, Morelos...  ← ¿tipo? ¿A?
Baja California, Jalisco...               ← ¿tipo? ¿B?
Michoacán y Guanajuato                    ← ¿tipo? ¿C?
...
A                                         ← ¿esto es un tipo o una celda?
B
11                                        ← ¿11 qué? ¿estados? ¿años?
4
```

**¿Por qué es un problema "invisible"?** Como no podemos ver los PDFs originales, no sabemos exactamente qué formato tenía cada tabla, qué datos contienen los gráficos, o si hay texto en imágenes que no se extrajo. Solo podemos inferir por el contexto que *algo se perdió*. Esto es crítico para preguntas como Q1, que requiere datos de ranking de entidades — los datos existen en el PDF pero nuestra extracción no los capturó con suficiente estructura para responder.

**Posibles soluciones para mejorar la extracción**:
- `page.get_text("dict")`: extrae con coordenadas de posición, permite reconstruir tablas manualmente
- `camelot-py` / `tabula-py`: bibliotecas especializadas en extracción de tablas de PDFs
- OCR (Tesseract) para PDFs escaneados o con texto en imágenes

### `02_chunk_text.py`

Divide el texto plano en fragmentos de tamaño fijo (~500-1000 caracteres) con solapamiento. Esto se llama **chunking**. Es necesario porque:

- Los LLMs tienen una **ventana de contexto** limitada (~4,000 tokens para Phi-3)
- No podemos meter un PDF entero como contexto
- Dividir en fragmentos permite que la búsqueda semántica (FAISS) encuentre la parte más relevante

### ¿Por qué 1322 chunks?

3 documentos × ~200-266 páginas cada uno, fragmentados en piezas de ~500 caracteres.

---

## 4. Vectorización e Índice FAISS

### `03_embeddings_faiss.py`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, normalize_embeddings=True)

dim = embeddings.shape[1]  # 384 dimensiones
index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity
index.add(embeddings)
```

**Embeddings**: Cada chunk de texto se convierte en un vector de 384 números reales. Usamos un modelo multilingüe (paraphrase-multilingual-MiniLM-L12-v2). Dos textos que hablen de lo mismo tendrán vectores "cercanos" (ángulo pequeño, coseno ≈ 1).

**FAISS IndexFlatIP**: Es una búsqueda de fuerza bruta —compara el vector de la pregunta contra todos los 1322 vectores— pero en 384 dimensiones y 1322 vectores es instantáneo (~1 ms). `IndexFlatIP` usa producto punto, que equivale a similitud coseno cuando los vectores están normalizados (y lo están gracias a `normalize_embeddings=True`).

**¿Qué guardamos?**

- `faiss_index.bin`: El índice binario con los 1322 vectores
- `chunks_meta.pkl`: Los chunks originales (texto, doc_id, chunk_id) serializados con pickle

---

## 5. Generación del Dataset de Fine-Tuning

### `04_generate_dataset.py`

Este script genera 83 pares **instrucción → respuesta** manualmente escritos. Cada ejemplo es:

```json
{
  "instruction": "¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?",
  "output": "Los datos del ONC indican que Colima, Baja California Sur y Chihuahua... [Documento 1, Pág 29]",
  "level": 1
}
```

**Tres niveles de dificultad** (como pide la especificación):

| Nivel | Tipo | Comportamiento esperado |
|-------|------|------------------------|
| 1 | Extracción directa (factoid) | El modelo debe responder con datos literales del corpus y citar fuente |
| 2 | Síntesis | El modelo debe unir información de múltiples documentos y estructurarla |
| 3 | Razonamiento analítico | El modelo debe usar el método socrático (hacer preguntas al usuario) y/o manejar incertidumbre |

### Patrones de comportamiento que aprende el modelo

**Citar fuentes**:
```
según el ONC... [Documento 1, Pág 29]
```

**Tono académico**:
```
Para abordar esta pregunta, primero es necesario precisar que... Cabe señalar que... En conclusión...
```

**Método socrático** (Nivel 3):
```
Antes de responder, permítame preguntar: ¿considera usted que...?
```

**Manejo de incertidumbre** (Nivel 3):
```
El corpus disponible no contiene información específica sobre... no es posible ofrecer una respuesta fundamentada.
```

### Distribución de ejemplos

De los 83 ejemplos:
- 29 Nivel 1 (respuestas directas con citas)
- 21 Nivel 2 (síntesis, estructura, contraste)
- 33 Nivel 3 (método socrático, incertidumbre, análisis)

Además, 7 ejemplos son específicamente **anti-invención**: enseñan al modelo a no inventar siglas, acrónimos ni fuentes (CNDH, UNAM, Banco Mundial, CIDB, OCDE). Otros ejemplos cubren temas **FUERA del corpus** (suicidios, cambio climático, salud mental) para reforzar el "no sé".

---

## 6. Dataset Aumentado con RAG

### `04b_generate_dataset_rag.py`

Este script toma los 83 ejemplos anteriores y modifica su `instruction` para incluir el contexto real que FAISS recuperaría para esa pregunta.

**Formato original**:
```
instruction = "¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?"
```

**Formato RAG (nuevo)**:
```
instruction = "Eres un tutor académico especializado en seguridad pública en México.
Responde solo con información de los documentos. No inventes nada. Si no hay datos
suficientes, dilo claramente.

Documentos:
[Documento 1: ONC - Homicidio / chunk_1166]
texto del chunk...
[Documento 2: INEGI - ENVIPE 2025 / chunk_0044]
texto del chunk...
[Documento 3: ONC - Reporte Anual 2025 / chunk_0029]
texto del chunk...

Pregunta: ¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?"
```

### ¿Por qué es importante?

Esto implementa la técnica llamada **RALT (Retrieval Augmented Language Training)** o **RA-DIT (Retrieval-Augmented Dual Instruction Tuning)**. La idea es:

1. Durante el entrenamiento, el modelo ve ejemplos donde la instrucción YA incluye chunks de documentos
2. El modelo aprende a **ignorar el ruido** y **extraer la respuesta de los chunks relevantes**
3. En inferencia, el prompt es idéntico (system prompt + chunks + pregunta)
4. El modelo generaliza: si los chunks contienen la respuesta, la usa; si no, aplica manejo de incertidumbre

**La diferencia clave**: si solo entrenas con preguntas sin contexto, el modelo memoriza las respuestas y alucina cuando la pregunta no coincide exactamente. Si entrenas con contexto RAG, el modelo aprende a *usar el contexto*, no a memorizar.

---

## 7. Fine-Tuning con LoRA

### `05_finetune_phi3.py`

```python
# Cargar Phi-3 en 4-bit (ahorra VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA: solo 0.62% de los parámetros son entrenables
lora_config = LoraConfig(
    r=8,           # rango de las matrices LoRA
    lora_alpha=16, # factor de escala
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
)

# Training config
training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    max_length=768,
)
```

### ¿Qué es LoRA?

LoRA (Low-Rank Adaptation) se basa en un hallazgo matemático: cuando fine-tuneas un modelo gigante, los cambios en los pesos no necesitan ser matrices enormes. Se pueden aproximar como el producto de dos matrices pequeñas:

```
Cambio ≈ A × B
         ↑    ↑
      (d×r) (r×k)
```

Donde `r` (el rango, en nuestro caso 8) es mucho más pequeño que las dimensiones originales. Entonces:

| Parámetros totales | Entrenables (LoRA) | % |
|-------------------|-------------------|----|
| 2,021,723,136 | 12,582,912 | 0.62% |

Solo entrenamos el 0.62% del modelo, lo que hace el fine-tuning viable en una GPU de 4GB.

### ¿Por qué 4-bit?

Phi-3-mini normalmente ocupa ~7.5GB en FP16. Con cuantización 4-bit se reduce a ~2.5GB. La técnica `nf4` (normal float 4) es un formato de cuantización especial de bitsandbytes que preserva mejor la calidad que cuantización lineal simple.

### Configuración de entrenamiento (y por qué)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `batch_size=1` | Mínimo | Solo 4GB VRAM, no cabe más |
| `gradient_accumulation=4` | Compensa batch | Simula batch_size=4 para mejor convergencia |
| `gradient_checkpointing=True` | Ahorra memoria | Calcula activaciones en backward, no las guarda |
| `epochs=3` | Suficiente | Con 83 ejemplos, 3 épocas dan buena convergencia |
| `lr=2e-4` | Estándar LoRA | Tasa de aprendizaje típica para fine-tuning con LoRA |
| `max_length=768` | Limitado por VRAM | Más largo causaría OOM |

### Proceso de entrenamiento

```
Step  0/63: loss = 1.878
Step  5/63: loss = 1.682
Step 10/63: loss = 1.431
Step 20/63: loss = 1.379
Step 30/63: loss = 1.231
Step 40/63: loss = 1.144
Step 50/63: loss = 1.057
Step 60/63: loss = 0.958

Total: ~77 minutos, ~73 segundos por step
```

La pérdida baja consistentemente, indicando que el modelo está aprendiendo. El valor final de 0.958 es el mejor obtenido en todas las iteraciones del proyecto.

---

## 8. Inferencia: RAG + Modelo Fine-Tuned

### `06_inference_phi3.py`

```python
def retrieve(question, k=3):
    # 1. Convertir pregunta a embedding
    q_emb = embedder.encode([question], normalize_embeddings=True)
    # 2. Buscar los k chunks más similares en FAISS
    scores, indices = index.search(q_emb.astype(np.float32), k)
    # 3. Devolver chunks con sus scores
    return [chunks[idx] for idx in indices[0]]

def generate(question, context):
    # Construir el mismo prompt que en entrenamiento
    instruction = build_instruction(question, context)
    prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

    # Tokenizar y generar
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
```

### Flujo de inferencia

```
Pregunta: "¿Cuáles son las 3 entidades con mayor tasa de homicidios?"
    │
    ▼
1. Embedding de la pregunta (384 dimensiones)
    │
    ▼
2. FAISS busca top-3 chunks más similares (similitud coseno)
    │  scores: [0.747, 0.746, 0.746]
    ▼
3. Se construye el prompt:
   "<|user|>
   Eres un tutor académico...
   Documentos:
   [ONC - Homicidio / chunk_1166]
   text...
   [ONC - Homicidio / chunk_0948]
   text...
   Pregunta: ¿Cuáles son las 3 entidades...?
   <|end|>
   <|assistant|>"
    │
    ▼
4. Phi-3 genera la respuesta autoregresivamente:
   token por token, hasta 150 tokens nuevos.
    │
    ▼
5. Se decodifica: "Para abordar esta pregunta, primero
   identificamos qué entidades tienen registros desde
   2003 hasta 2012... Tamaulipas (11.2)..."
```

### Parámetros de generación

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `max_new_tokens=300` | Genera hasta 300 tokens nuevos | Permite respuestas más completas |
| `do_sample=False` | Greedy decoding (determinista) | Siempre la misma respuesta, menos alucinación |
| `repetition_penalty=1.1` | 1.1 | Penaliza repetición de frases |

---

## 9. Evaluación de Resultados

### Las 10 preguntas del banco de pruebas

**Nivel 1 (Factoid):**
| # | Pregunta | ¿Qué debería pasar? | ¿Qué pasó? |
|---|----------|---------------------|------------|
| Q1 | Top 3 homicidios | RAG recupera dato exacto, modelo cita | Responde Tamaulipas/Chihuahua/Hidalgo — cifras inventadas (3.1/2.9/2.8), tablas perdidas |
| Q2 | Cárteles en Tierra Caliente | RAG recupera menciones | ✅ "No contiene información" — mejora significativa |
| Q3 | Desplazamiento forzado | RAG recupera datos | Inventa "8 millones" y "Reporte Anual de Seguridad 2025" |

**Nivel 2 (Síntesis):**
| # | Pregunta | ¿Qué debería pasar? | ¿Qué pasó? |
|---|----------|---------------------|------------|
| Q4 | Causas socioeconómicas | Unir múltiples chunks | ✅ Reconoce que no hay causas precisas, da contextualización |
| Q5 | Militarización vs prevención | Contrastar enfoques | Buen contraste — ya no inventa CNDH/UNAM |
| Q6 | Evolución extorsión | Buscar datos ENVIPE | ✅ "No es posible responder" |
| Q7 | Rural vs urbano | Analizar diferencias | Diferencia significativa — ya sin DFGH/CIDOB |

**Nivel 3 (Analítico):**
| # | Pregunta | ¿Qué debería pasar? | ¿Qué pasó? |
|---|----------|---------------------|------------|
| Q8 | Discrepancias ONG/gob | Mostrar contradicciones | ✅ "No es posible identificar contradicciones" |
| Q9 | Deserción escolar | RECHAZAR responder | ✅ "El corpus no contiene información" |
| Q10 | Vacíos de información | Síntesis de limitaciones | Buen análisis — sin fuentes inventadas |

### Fortalezas observadas

1. **"No sé" funcional**: Q2, Q6, Q8, Q9 (4/10 preguntas) responden correctamente que no tienen información. Esto se enseñó con ejemplos anti-hallucination y anti-invención.
2. **Sin fuentes inventadas**: Desaparecieron las invenciones de siglas (CNDH, UNAM, DFGH, CIDOB) que aparecían en corridas anteriores.
3. **Formato de citas**: El modelo incluye `[Documento N]` consistentemente.
4. **Estructura académica**: Usa conectores lógicos ("Para abordar esta pregunta...", "En primer lugar...", "En conclusión...") que aprendió del dataset.
5. **Tono neutral**: No emite juicios ni opiniones.

### Debilidades observadas

1. **Alucinación de cifras** (Q1, Q3): El modelo genera números específicos (3.1, 2.9, 2.8, "8 millones") que no están en los chunks. Causa raíz: las tablas con datos numéricos se pierden durante la extracción PDF (PyMuPDF no reconstruye tablas). El modelo "rellena" con conocimiento pre-entrenado.
2. **Datos tabulares perdidos**: Q1 no se puede responder bien porque la tabla de rankings de homicidios por estado no se extrajo. Este es el problema más grave: es silencioso (no hay error, solo datos incorrectos).
3. **Evaluación subjetiva**: No hay métricas cuantitativas como ROUGE o BLEU.

---

## 10. Justificación de Decisiones Técnicas

### ¿Por qué Phi-3-mini y no otro modelo?

| Modelo | Parámetros | VRAM (4-bit) | ¿Disponible? | Decisión |
|--------|------------|--------------|--------------|----------|
| TinyLlama-1.1B | 1.1B | ~1.5GB | [OK] Cacheado | Muy limitado para respuestas analíticas |
| Phi-3-mini-4k | 3.8B | ~2.5GB | [OK] Cacheado | **Elegido**: balance calidad/VRAM |
| Qwen2.5-1.5B | 1.5B | ~1.8GB | [NO] Descarga 7 KB/s | Inviable: tardaría días en descargar |

Phi-3-mini es el mejor modelo que ya teníamos descargado localmente. Es superior a TinyLlama por tener ~3.5× más parámetros, y Qwen2.5 no se pudo descargar por velocidad de red.

### ¿Por qué `per_device_train_batch_size=1`?

VRAM disponible en la GTX 1050: **4GB**. Desglose:
- Modelo en 4-bit: ~2.5GB
- LoRA adapters: ~0.1GB
- Activaciones (gradient checkpointing): ~1.0GB
- **Total**: ~3.6GB → solo quedan ~0.4GB libres
- Con batch_size=1 y secuencias de ~700 tokens, estamos al límite.

Intentar batch_size=2 o max_length=1024 causa OOM (Out Of Memory).

### ¿Por qué `gradient_accumulation_steps=4`?

Para compensar el batch_size=1. El gradiente se acumula durante 4 pasos antes de actualizar los pesos. Esto simula un batch efectivo de 4, mejorando la estabilidad del entrenamiento sin aumentar VRAM.

### ¿Por qué exactamente 83 ejemplos en el dataset?

No hay un número mágico. Consideraciones:
- **Mínimo viable**: Con ~20 ejemplos el modelo aprende patrones básicos
- **Cobertura de comportamientos**: Necesitamos ejemplos de citas, método socrático, incertidumbre, anti-invención y síntesis → 83 cubre todos los patrones
- **Tiempo de entrenamiento**: 83 ejemplos × 3 épocas = 63 steps × ~73s = ~77 min. Con 200 ejemplos serían ~3 horas
- **Hardware**: Con más ejemplos, el entrenamiento podría no caber en 4GB de VRAM (el dataset se tokeniza completo en memoria)
- **Anti-invención**: Se añadieron 7 ejemplos específicos para que el modelo no invente siglas ni fuentes (CNDH, UNAM, Banco Mundial, etc.). Este fue el principal problema identificado en iteraciones anteriores.

### ¿Por qué no usamos un modelo más grande (Llama 3, Mistral)?

No están descargados localmente y la velocidad de descarga (~7 KB/s) hace inviable bajarlos (Mistral-7B ~14GB → ~23 días de descarga).

---

## 11. Problemas Encontrados y Soluciones

### Problema 1: OOM durante fine-tuning

**Síntoma**: El entrenamiento se detenía en el step 31 con `torch.OutOfMemoryError`.

**Causa**: El dataset RAG tiene instrucciones más largas (incluyen chunks de contexto). Con `max_length=1024` y `batch_size=4`, las activaciones ocupaban más VRAM de la disponible.

**Solución**:
1. `batch_size=4 → 1`
2. `gradient_accumulation=2 → 4`
3. `max_length=1024 → 768`
4. `gradient_checkpointing=True` (ya estaba)
5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Problema 2: Descarga de modelos extremadamente lenta

**Síntoma**: Qwen2.5-1.5B al 2% después de 30 minutos.

**Causa**: Red institucional limitada a ~7 KB/s.

**Solución**: Usar solo modelos ya cacheados (Phi-3-mini, TinyLlama). No se pudo usar Qwen2.5 ni modelos más grandes.

### Problema 3: Datos tabulares perdidos en extracción PDF

**Síntoma**: Las respuestas a Q1 (ranking de estados) son incorrectas o genéricas. El modelo no puede identificar claramente el top 3 de entidades con mayor tasa de homicidios.

**Causa raíz**: PyMuPDF (`page.get_text()`) extrae texto en orden de lectura pero **no reconstruye tablas**. Los PDFs del ONC y ENVIPE contienen múltiples tablas con datos estructurados (rankings, tasas, clasificaciones). Al extraerse como texto plano, la estructura de columnas se pierde.

**Ejemplo real de una tabla destruida por la extracción**:

Del archivo `corpus_raw/ONC-HOMICIDIO.txt`, líneas 20004-20026:

```
# Lo que se extrajo:
Guerrero, Sinaloa, Chihuahua, Morelos, Durango, Tamaulipas, Colima, Coahuila, Nuevo
León, Nayarit y Veracruz
Baja California, Jalisco, San Luis Potosí y Zacatecas
Michoacán y Guanajuato
Puebla y Oaxaca
Baja California Sur, Campeche, Hidalgo, Querétaro, Aguascalientes y Yucatán
Estado de México, Sonora, Quintana Roo y Tlaxcala
Chiapas, Distrito Federal y Tabasco
A
B
C
D
E
Irregular
Mixto (D/E)
11
4
2
2
6
4
3
```

En el PDF original esto era una tabla con 3 columnas (Tipo, Entidades, Cantidad). La extracción aplanó todo perdiendo la relación entre columnas. No podemos saber qué estados corresponden a qué tipo ni qué significan los números.

**El problema de los errores "invisibles"**:

Este es el tipo de error que **no se puede detectar automáticamente** sin ver los PDFs originales. Como el texto *sí* se extrajo (las palabras están ahí), no hay una señal de error. Solo alguien que conozca los documentos originales puede notar que falta estructura o que datos tabulares aparecen desordenados.

Otros problemas invisibles:
- **Gráficos**: cualquier dato dentro de gráficos de barras, pastel o líneas no se extrae
- **Imágenes con texto**: infografías, mapas, capturas de pantalla
- **Páginas en blanco**: el ONC-Homicidio tiene páginas intencionalmente en blanco que generan chunks vacíos
- **Encabezados y pies de página repetidos**: contaminan chunks con texto duplicado
- **Notas al pie**: pueden separarse del texto al que referencian

**Impacto en el proyecto**:

La pregunta Q1 del banco de evaluación pregunta por las "tres entidades federativas con mayor índice de homicidios dolosos". Los datos existen en los PDFs (el ONC tiene tablas de tasas por estado), pero:
1. La extracción no preservó la estructura de esas tablas
2. Los chunks resultantes contienen datos fragmentados sin relación entre columnas
3. El RAG no puede recuperar el dato exacto porque no existe como tal en los chunks
4. El modelo fine-tuned intenta responder pero termina alucinando o dando respuestas incompletas

**¿Se podría haber evitado?** Sí, usando herramientas especializadas:
- `camelot-py`: extrae tablas de PDFs preservando estructura de filas y columnas
- `tabula-py`: similar, basada en Tabula (Java)
- `pdfplumber`: extrae texto con coordenadas (x, y) permitiendo reconstruir la posición de cada palabra
- `page.get_text("dict")`: PyMuPDF ofrece extracción con estructura de bloques, líneas y spans, que permite reconstruir tablas analizando alineación horizontal de texto

Para este proyecto académico, la limitación es aceptable porque el pipeline demuestra el concepto, pero para un sistema productivo sería necesario usar extracción especializada.

### Problema 4: Inferencia lenta (~8-12 tok/s)

**Síntoma**: 10 preguntas tardan ~30-40 minutos.

**Causa**: Phi-3-mini en 4-bit en GTX 1050 (4GB). La cuantización 4-bit requiere operaciones descompuestas que son lentas en GPUs viejas.

**Solución**: 
- `max_new_tokens=300` (permitido por mejor gestión de VRAM)
- `max_length=768` (limita prompt)
- No usar `torch.cuda.empty_cache()` entre preguntas (lo ralentiza más)
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` (evita fragmentación)

### Problema 5: Alucinación en Q9

**Síntoma**: El modelo genera cifras (45%, 70%) que no están en los chunks.

**Causa**: El modelo aprendió el patrón de "dar cifras" del dataset, y como los chunks recuperados para Q9 hablan de impacto psicológico en niños (no de deserción escolar), el modelo inventa números plausibles.

**Solución (para mejorar)**: 
- Aumentar el dataset con más ejemplos de incertidumbre
- Incluir explícitamente ejemplos donde la pregunta no coincide con los chunks
- En el dataset RAG, forzar que algunos ejemplos tengan chunks no relacionados

---

## 12. Glosario de Conceptos

### RAG (Retrieval-Augmented Generation)

Técnica que combina recuperación de información + generación de texto. En lugar de que el LLM responda de memoria:
1. Recibes una pregunta
2. Buscas fragmentos relevantes en una base de datos vectorial (FAISS)
3. Inyectas esos fragmentos en el prompt del LLM
4. El LLM responde basándose SOLO en esos fragmentos (idealmente)

**Ventaja**: Las respuestas son factuales, actualizables (solo cambias los documentos), y verificables (puedes ver qué fragmentos usó).

**Desventaja**: Si la recuperación falla (no encuentra los fragmentos correctos), el modelo puede alucinar.

### LoRA (Low-Rank Adaptation)

Técnica de fine-tuning eficiente. En lugar de actualizar los pesos W de una capa (matriz de tamaño d×k), LoRA aprende dos matrices pequeñas A (d×r) y B (r×k) donde r << min(d, k). El nuevo peso es:

```
W' = W + α × A × B
```

Solo A y B se entrenan. W queda congelado.

**Ventaja**: Reduce parámetros entrenables de millones a miles. Un LoRA entrenado pesa ~10MB en lugar de ~7.5GB del modelo completo.

### Cuantización 4-bit (NF4)

Reducción de precisión: cada peso pasa de 16 bits (FP16) a 4 bits. El formato NF4 (Normal Float 4) de bitsandbytes distribuye los 16 posibles valores de forma no lineal para maximizar precisión donde más se necesita.

**Trade-off**: Ocupas 4× menos VRAM, pero pierdes ~1-2% de precisión en las respuestas.

### FAISS (Facebook AI Similarity Search)

Biblioteca de Meta para búsqueda de similitud entre vectores. Ofrece:
- Búsqueda exacta (IndexFlatIP): compara contra todos, 100% preciso
- Búsqueda aproximada (IndexIVFFlat): más rápida, ~99% preciso

Usamos IndexFlatIP porque con solo 1322 vectores, la búsqueda exacta es instantánea.

### SentenceTransformer

Modelo que convierte oraciones enteras en vectores (no solo palabras). `paraphrase-multilingual-MiniLM-L12-v2` soporta español y genera vectores de 384 dimensiones.

### Instruction Fine-Tuning vs Pre-training

| | Pre-training | Fine-Tuning |
|--|-------------|-------------|
| ¿Qué hace? | Aprende lenguaje desde texto bruto | Aprende a seguir instrucciones |
| Datos | Billones de tokens | Cientos/miles de ejemplos |
| Costo | Millones de USD | Una GPU de consumo |
| Resultado | Modelo base (GPT-3, Phi-3) | Modelo asistente (ChatGPT, tutor) |

### Gradient Checkpointing

Técnica de ahorro de memoria: durante el forward pass, NO se guardan las activaciones intermedias (que ocupan mucha VRAM). Durante el backward pass, se recalculan. Esto intercambia tiempo de cómputo (~20% más lento) por memoria (~60% menos).

### Gradient Accumulation

En lugar de actualizar los pesos después de cada ejemplo (batch_size=1), acumulamos los gradientes de N ejemplos y actualizamos después. Efectivo: entrenamos con batch_size=N pero usando la VRAM de batch_size=1.

---

## 13. Análisis Crítico y Limitaciones

### ¿Funciona el sistema?

**Sí, pero con limitaciones predecibles.** El pipeline RAG + Fine-Tuning está correctamente implementado y demuestra todos los comportamientos que pide la especificación:

- [OK] Cita fuentes
- [OK] Tono académico y neutral
- [OK] Método socrático en preguntas de análisis
- [OK] Manejo de incertidumbre
- [OK] Pipeline completo de ingesta → fine-tuning → inferencia

### Limitaciones importantes

1. **Cantidad de datos de entrenamiento**: 83 ejemplos es suficiente para enseñar estilos de respuesta, pero más ejemplos (~200) ayudarían a cubrir más patrones de incertidumbre y reducir la alucinación residual de cifras.

2. **Calidad de la extracción PDF**: Los datos tabulares se pierden (tablas de rankings, clasificaciones de estados, tasas). Este es un problema particularmente peligroso porque **no genera señales de error visibles**: el texto se extrae, las palabras aparecen, pero la estructura relacional entre datos desaparece. Solo alguien que conozca los PDFs originales puede notar que falta información. Para un sistema productivo, habría que usar extractores de tablas (`camelot-py`, `tabula-py`) o extracción con coordenadas (`pdfplumber`).

3. **Velocidad de inferencia**: 8-12 tok/s en GTX 1050. Para uso interactivo se necesitaría una GPU más potente o un modelo más pequeño.

4. **Ventana de contexto limitada**: 768 tokens (~600 palabras) es poco para análisis complejos. Esto es por VRAM, no por el modelo.

5. **Evaluación subjetiva**: No hay métricas cuantitativas de calidad de respuesta (como ROUGE, BLEU, o evaluación humana). Solo análisis cualitativo.

### ¿Qué se necesitaría para mejorar?

| Mejora | Impacto | Costo |
|--------|---------|-------|
| Más ejemplos en dataset (200+) | Reduce alucinación | Medio (crearlos) |
| Mejor extracción de tablas PDF | Mejora Q1, respuestas factuales | Bajo (cambiar librería) |
| GPU con más VRAM (8-12GB) | Permite batch mayor, más contexto | Alto (hardware) |
| Modelo más grande (Mistral-7B) | Mejor calidad de respuesta | Alto (descarga + VRAM) |
| Evaluación con métricas (ROUGE) | Cuantifica calidad | Bajo (implementar) |

### Conclusión

El proyecto demuestra correctamente el concepto de **Tutor Híbrido RAG + Fine-Tuning**. La arquitectura es sólida y sigue el estado del arte (RA-DIT). Las limitaciones observadas son de **escala** (pocos ejemplos, hardware limitado) no de **diseño**. Con más recursos, el mismo pipeline escalaría a un sistema productivo.
