# Guía de Demostración y Preguntas Frecuentes

## Tutor Analítico Híbrido (RAG + Fine-Tuning con LoRA)

---

## 1. Cómo mostrar el programa al profesor

### Resumen de 30 segundos

> "Construimos un tutor inteligente que responde preguntas sobre seguridad pública en México. El sistema **no solo usa un LLM**: primero busca fragmentos relevantes en 3 documentos PDF usando búsqueda semántica (FAISS) y luego genera una respuesta con el modelo fine-tuned (Phi-3-mini + LoRA). El fine-tuning le enseñó un estilo específico: citar fuentes, manejar incertidumbre, usar método socrático."

### Demostración paso a paso (5-7 min)

| Paso | Duración | Qué hacer |
|------|----------|-----------|
| **1. Mostrar la estructura del proyecto** | 30 s | `tree -L 2 project4/` — señalar `corpus/`, `src/`, `models/`, `results/` |
| **2. Mostrar los PDFs fuente** | 30 s | Abrir `corpus/` — 3 PDFs sobre violencia en México (ONC Homicidio, ENVIPE 2025, ONC Anual) |
| **3. Ejecutar inferencia (el clímax)** | 2 min | `python src/06_inference_phi3.py` — responde 10 preguntas en vivo. Mostrar que busca chunks en FAISS y genera respuestas con citas |
| **4. Mostrar un resultado concreto** | 1 min | Abrir `results/evaluation_phi3.json` y mostrar que cada respuesta incluye: pregunta, chunks recuperados con score, respuesta generada |
| **5. Probar una pregunta manual (opcional)** | 1 min | Modificar el script o el JSON de preguntas con una pregunta nueva y mostrar que busca, recupera y responde |
| **6. Fine-tuning (opcional, solo mencionar)** | 30 s | Si el profesor pregunta: "entrenamos 83 ejemplos con LoRA en ~77 minutos. Solo entrenamos el 0.62% de los parámetros del modelo." |

### Puntos clave para mencionar durante la demo

- **RAG**: el modelo no responde de memoria, busca en los documentos reales. Esto evita alucinaciones.
- **Fine-tuning**: le enseñamos a citar fuentes (`[Documento 1, Pág 29]`), decir "no sé" cuando falta info, y usar método socrático.
- **LoRA**: técnica eficiente que permite fine-tuning en una GPU de 4GB (GTX 1050).
- **FAISS**: búsqueda de similitud coseno entre vectores de 384 dimensiones (modelo multilingüe).

---

## 2. Cómo probar cada funcionalidad

### 2.1 Probar la extracción de texto (01_extract_text.py)

```bash
source .venv/bin/activate
python src/01_extract_text.py
```

**Qué esperar**: Debe generar 3 archivos `.txt` en `corpus_raw/`. Verificar:

```bash
wc -l corpus_raw/*.txt
```

**Lo que prueba**: que PyMuPDF (`fitz`) extrae correctamente el texto de los PDFs. Si hay PDFs escaneados sin texto, esta etapa fallaría (no es el caso, son PDFs con texto seleccionable).

### 2.2 Probar el chunking (02_chunking.py)

```bash
python src/02_chunking.py
```

**Qué esperar**: Debe generar `chunks/chunks.jsonl` con ~1322 líneas. Verificar:

```bash
wc -l chunks/chunks.jsonl
python -c "import json; c=[json.loads(l) for l in open('chunks/chunks.jsonl')]; print(f'{len(c)} chunks, ejemplo: {c[0][\"text\"][:80]}...')"
```

**Lo que prueba**: que el texto se divide correctamente en fragmentos de ~700 caracteres con solapamiento de 100.

### 2.3 Probar embeddings e índice FAISS (03_embeddings_faiss.py)

```bash
python src/03_embeddings_faiss.py
```

**Qué esperar**: Debe generar `models/faiss_index.bin` (índice) y `models/chunks_meta.pkl` (metadatos). Mensaje: "Index saved: 1322 vectors, dim=384".

**Lo que prueba**: que `sentence-transformers` genera embeddings de 384 dimensiones y FAISS los indexa correctamente.

### 2.4 Probar la generación del dataset (04_generate_dataset.py)

```bash
python src/04_generate_dataset.py
```

**Qué esperar**: Genera `dataset/tutor_dataset.jsonl` con 83 ejemplos. Mensaje con distribución de niveles.

**Lo que prueba**: que se generan correctamente los 83 pares instrucción → respuesta con los 3 niveles de dificultad.

### 2.5 Probar el dataset aumentado con RAG (04b_generate_dataset_rag.py)

```bash
python src/04b_generate_dataset_rag.py
```

**Qué esperar**: Genera `dataset/tutor_dataset_rag.jsonl`. Cada instrucción ahora incluye contexto real recuperado de FAISS.

**Lo que prueba**: que la recuperación RAG funciona correctamente (técnica RALT / RA-DIT).

### 2.6 Probar el fine-tuning (05_finetune_phi3.py)

```bash
python src/05_finetune_phi3.py
```

**Qué esperar** (~77 min en GTX 1050 con 83 ejemplos):
- Carga Phi-3-mini en 4-bit (~2 GB VRAM)
- Entrenamiento: 3 epochs, batch 1, grad accum 4
- Guarda adaptadores LoRA en `models/lora_adapter_phi3_rag/`

**Verificar**:

```bash
ls models/lora_adapter_phi3_rag/
# debería ver: adapter_config.json, adapter_model.safetensors, tokenizer*, etc.
```

**Lo que prueba**: que LoRA se aplica correctamente sobre Phi-3 (targets: qkv_proj, o_proj, gate_up_proj, down_proj).

### 2.7 Probar la inferencia completa (06_inference_phi3.py)

```bash
python src/06_inference_phi3.py
```

**Qué esperar**: Responde 10 preguntas una por una. Cada una:
1. Recupera 3 chunks de FAISS (top-3 por similitud coseno)
2. Prepara el prompt con contexto + pregunta
3. Genera respuesta con el modelo fine-tuned

**Verificar resultados**:

```bash
python -c "import json; r=json.load(open('results/evaluation_phi3.json')); [print(f\"{q}: {a['answer'][:100]}...\\n\") for q,a in r.items()]"
```

**Lo que prueba**: el pipeline completo (RAG + modelo fine-tuned).

---

## 3. Preguntas frecuentes / cuestionamientos frecuentes

### 3.1 ¿Por qué combinaron RAG con fine-tuning? ¿No basta con uno solo?

**Respuesta corta**: Ambos resuelven problemas distintos y se complementan.

- **RAG**: evita que el modelo invente (alucine), porque lo obliga a basarse en documentos reales. Sin embargo, el modelo base no sabe *cómo* presentar la información (no cita fuentes, no estructura respuestas académicamente).
- **Fine-tuning**: le enseña el *estilo* de respuesta: citar fuentes, usar método socrático, manejar incertidumbre. Pero el fine-tuning por sí solo no puede añadir conocimiento nuevo —solo cambia el comportamiento— y el modelo igual podría alucinar si no se le da contexto.

**Juntos**: RAG le da los datos correctos y el fine-tuning le dice cómo usarlos.

**Referencia en apuntes**: El concepto de sistemas que actúan racionalmente se explica en la sección de **Categorías de la IA** (`apuntes_ia.md`, líneas 42-56).

### 3.2 ¿Por qué usaron LoRA en lugar de fine-tuning completo?

El fine-tuning completo de Phi-3-mini (3.8B parámetros) requiere ~60 GB de VRAM. Con una GPU GTX 1050 de 4GB es imposible.

LoRA (Low-Rank Adaptation, `apuntes_ia.md`, línea 38 del informe técnico) solo entrena matrices pequeñas insertadas en el modelo: ~12.5 millones de parámetros (0.62% del total). Se puede entrenar en cualquier GPU con 4GB+.

**Analogía**: Es como remodelar una casa sin volver a construir los cimientos —cambias la decoración (LoRA) manteniendo la estructura.

### 3.3 ¿Por qué usaron un modelo de 384 dimensiones para embeddings?

El modelo `paraphrase-multilingual-MiniLM-L12-v2` genera vectores de 384 dimensiones. Fue elegido por tres razones:

1. **Multilingüe**: funciona bien en español
2. **Ligero**: se ejecuta en CPU sin problema
3. **Suficiente**: 384 dimensiones capturan bien la similitud semántica para ~1300 chunks

Modelos más grandes (como `text-embedding-3-large` con 3076 dimensiones) dan mejor calidad pero son más lentos y no caben en una GPU de 4GB.

**Referencia en apuntes**: El concepto de representación vectorial y espacio de búsqueda se relaciona con el **espacio de estados** (`apuntes_ia.md`, líneas 892-930).

### 3.4 ¿Por qué usaron FAISS y no otra base de datos vectorial?

FAISS (Facebook AI Similarity Search) es una biblioteca ligera que no requiere servidor, se instala con pip, y busca en 1322 vectores en ~1 ms. Para un proyecto académico con ~1300 chunks es la opción más práctica.

Alternativas como Pinecone, Weaviate o ChromaDB añaden complejidad innecesaria para este volumen de datos.

### 3.5 ¿Por qué entrenaron 83 ejemplos y no más?

83 ejemplos bien diseñados son suficientes para enseñar un *estilo* de respuesta mediante LoRA. No se busca que el modelo memorice datos (para eso está RAG), sino que aprenda *cómo* responder.

Los 83 ejemplos cubren:
- 29 de extracción directa (nivel 1)
- 21 de síntesis (nivel 2)
- 33 de razonamiento analítico / incertidumbre (nivel 3)

De estos, 7 son ejemplos específicos **anti-invención** que enseñan al modelo a no inventar siglas ni fuentes (CNDH, UNAM, Banco Mundial, etc.). También incluyen preguntas FUERA del corpus (suicidios, cambio climático, salud mental) para que el modelo aprenda a decir "no sé".

### 3.6 ¿Qué pasa si el profesor pregunta algo que no está en los PDFs?

El modelo **debe** responder que no tiene información suficiente. Esto se enseñó explícitamente en el fine-tuning con ejemplos como:

> "El corpus disponible no contiene información específica sobre [tema]. No es posible ofrecer una respuesta fundamentada con los documentos disponibles."

Esto se alinea con el manejo de incertidumbre de la `style_guide.md`: "El corpus disponible no contiene información específica sobre... No es posible ofrecer una respuesta fundamentada".

**Referencia en apuntes**: Este comportamiento se relaciona con el **razonamiento lógico** (`apuntes_ia.md`, líneas 152-194): un sistema que no puede demostrar una proposición (por falta de axiomas/documentos) no debe afirmarla.

### 3.7 ¿El modelo realmente entiende lo que responde o solo repite?

El modelo no "entiende" en sentido humano —es un sistema estadístico que predice la siguiente palabra más probable. Sin embargo, el diseño híbrido (RAG + fine-tuning) mitiga las limitaciones:

- Con RAG, no puede inventar datos, porque el prompt contiene los chunks reales.
- El fine-tuning le enseña a no divagar y a reconocer límites.

**Referencia en apuntes**: Esto conecta directamente con el **modelo cognoscitivo** (`apuntes_ia.md`, líneas 257-330) —la mente como procesador de información, similar a un ordenador: recibe datos, los procesa y produce respuestas. Y con las **teorías de la inteligencia** (`apuntes_ia.md`, líneas 90-150)—el conductismo diría que el modelo solo da respuestas aprendidas; Gardner diría que carece de inteligencia lingüística real.

### 3.8 ¿Por qué no usaron un modelo más grande como Llama 3 o GPT-4?

Recursos limitados: GPU GTX 1050 con 4GB de VRAM. Phi-3-mini (3.8B parámetros) en 4-bit quantization ocupa ~2.5 GB y cabe. Modelos como Llama 3 (8B) en 4-bit ocupan ~6 GB —no caben.

Además, el proyecto busca demostrar que es posible construir un sistema funcional con hardware modesto, no competir con servidores empresariales.

### 3.9 ¿Por qué los chunks tienen ~700 caracteres?

700 caracteres (~175 tokens) es un balance entre:
- **Muy pequeños (<200 chars)**: no contienen suficiente información para responder
- **Muy grandes (>2000 chars)**: diluyen la relevancia semántica y desperdician contexto

El solapamiento de 100 caracteres asegura que ningún concepto quede partido entre dos chunks.

**Referencia en apuntes**: La idea de dividir un problema grande en partes más pequeñas se relaciona con la **búsqueda heurística** (`apuntes_ia.md`, líneas 405-450): se usan heurísticas (cortar en nueva línea o punto) para acelerar el proceso de chunking.

### 3.10 ¿Cómo sabemos que el fine-tuning funcionó y no es el modelo base?

Se puede demostrar de dos maneras:

1. **Comparar respuestas**: ejecutar `06_inference_phi3.py` sin cargar los adaptadores LoRA (comentar las líneas 64-66 del script) y ver que las respuestas son más genéricas y no usan el formato académico.
2. **Evidencia del entrenamiento**: el log del fine-tuning muestra que la loss disminuyó durante las 3 épocas.

En `05_finetune_phi3.py`, línea 55-56, se imprime:
```
Trainable: 12,582,912 / 2,021,723,136 (0.62%)
```

Esto confirma que los adaptores LoRA se insertaron y entrenaron correctamente.

### 3.11 ¿Qué es un "token" y por qué importa?

Un token es la unidad mínima de texto que procesa el modelo. No es una palabra exacta: "violencia" podría ser 1 o 2 tokens dependiendo del tokenizador. Phi-3-mini tiene un límite de ~4000 tokens por consulta. El proyecto usa `max_length=768` para el prompt completo y `max_new_tokens=300` para la respuesta.

Esto significa que la pregunta + chunks + instrucciones no debe exceder 768 tokens (~3000 caracteres). Si los 3 chunks suman más, se truncan.

**Referencia en apuntes**: Se relaciona con la **búsqueda en espacios de estados** (`apuntes_ia.md`, líneas 892-930): el espacio de tokens posible es enorme (~50k tokens en el vocabulario), y el modelo "navega" este espacio para generar la respuesta más probable.

### 3.12 ¿Por qué el modelo a veces responde "Tamaulipas" cuando la respuesta correcta es "Colima"?

En el resultado de evaluación Q1, el modelo responde "Tamaulipas" en lugar de las entidades correctas (Colima, Baja California Sur, Chihuahua). Esto ocurre porque la tabla de clasificación de entidades en el PDF original se perdió durante la extracción de texto (PyMuPDF no preserva tablas). Los chunks recuperados tienen scores de similitud similares (~0.74) pero no contienen los datos precisos.

**Limitación conocida**: las tablas del PDF original no se extraen correctamente. En `02_chunking.py`, el chunk de la tabla aparece como texto plano desestructurado. Esto se documenta en `informe_tecnico.md` sección 3.

### 3.13 ¿Qué diferencia hay entre IndexFlatIP (producto punto) y similitud coseno?

Son equivalentes cuando los vectores están normalizados (magnitud = 1). `faiss.IndexFlatIP` calcula el producto punto. `normalize_embeddings=True` en el embedding hace que todos los vectores tengan magnitud 1, por lo que:

```
coseno(a, b) = a·b / (|a|·|b|) = a·b / (1·1) = a·b
```

Usar producto punto directamente es más rápido computacionalmente.

### 3.14 ¿El sistema puede usarse con otros temas?

Sí, el pipeline es genérico. Para cambiar el dominio solo hay que:

1. Reemplazar los PDFs en `corpus/`
2. Ejecutar los 6 scripts en orden
3. (Opcional) Generar nuevos ejemplos de fine-tuning

Lo único específico del dominio es el dataset de fine-tuning (83 ejemplos) y los prompts en `06_inference_phi3.py`.

### 3.15 ¿Por qué la respuesta del nivel 3 usa método socrático?

El método socrático (devolver preguntas antes de responder) se especificó en la `style_guide.md` para el nivel 3 de dificultad. La idea es que el tutor no solo dé respuestas, sino que guíe al estudiante a reflexionar antes de recibir la información.

Ejemplo del dataset (`04_generate_dataset.py`, línea 134):
> "Antes de responder, permítame plantearle una reflexión: ¿a qué tipo de violencia se refiere específicamente?"

**Referencia en apuntes**: El método socrático se relaciona con el proceso de **razonamiento según la lógica** (`apuntes_ia.md`, líneas 152-194): usar preguntas para llevar al interlocutor a descubrir la verdad por sí mismo, similar a una demostración por deducción.

---

## 4. Glosario rápido

| Término | Definición |
|---------|------------|
| **RAG** | Retrieval-Augmented Generation — buscar documentos antes de responder |
| **LoRA** | Low-Rank Adaptation — fine-tuning eficiente con matrices pequeñas |
| **FAISS** | Biblioteca de búsqueda vectorial de Facebook |
| **Chunk** | Fragmento de texto (~700 caracteres) |
| **Embedding** | Vector numérico que representa el significado de un texto |
| **Cifra negra** | 93.2% de delitos no denunciados en México (ENVIPE 2025) |
| **SFT** | Supervised Fine-Tuning — entrenamiento supervisado |
| **4-bit quantization** | Técnica para reducir el tamaño del modelo (~75% menos memoria) |

---

## 5. Checklist rápida para la demostración

- [ ] `source .venv/bin/activate`
- [ ] Mostrar estructura del proyecto (`tree -L 2`)
- [ ] Mostrar los PDFs fuente
- [ ] Ejecutar `python src/06_inference_phi3.py` (respuestas en vivo)
- [ ] Abrir `results/evaluation_phi3.json` y mostrar chunks + respuestas
- [ ] Si preguntan por fine-tuning: mencionar LoRA, 0.62% parámetros, ~77 min (83 ejemplos)

### Si el profesor pide cambiar algo en vivo

- **Cambiar pregunta**: editar el dict `QUESTIONS` en `src/06_inference_phi3.py` (líneas 21-32) y re-ejecutar
- **Cambiar número de chunks**: modificar `k=3` en `retrieve()` (línea 74)
- **Cambiar temperatura/generación**: modificar `do_sample` o `temperature` en `generate()` (líneas 115-122)

---

*Documento generado como guía de demostración para el proyecto "Tutor Analítico Híbrido (RAG + Fine-Tuning)". Basado en `apuntes_ia.md` (apuntes de clase) y `informe_tecnico.md` (documentación técnica del proyecto).*
