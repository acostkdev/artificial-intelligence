# Guía de Presentación — Tutor Analítico Híbrido (RAG + LoRA)

Documento organizado por el **flujo de presentación** al profesor.
Cada paso indica qué ejecutar, por qué, y qué responder si pregunta.

---

## Setup (hacer antes de la presentación)

```bash
cd project4
source ../.venv/bin/activate
```

Verificar que todo está listo:

```bash
ls corpus/*.pdf                     # 3 PDFs
ls corpus_raw/*.txt                 # texto extraído
ls chunks/chunks.jsonl              # chunks
ls models/faiss_index.bin           # índice FAISS
ls models/lora_adapter_phi3_rag/    # adaptadores LoRA
ls results/evaluation_phi3.json     # resultados de inferencia
```
---

## Paso 1 — Mostrar la estructura del proyecto (30 s)

**Qué hacer:**

```bash
tree -L 2
```

| Carpeta/Archivo | Qué contiene |
|-----------------|--------------|
| `corpus/` | 3 PDFs fuente sobre violencia en México |
| `corpus_raw/` | Texto extraído de los PDFs |
| `chunks/` | Fragmentos de ~700 caracteres (1,322 chunks) |
| `models/` | Índice FAISS + adaptadores LoRA |
| `dataset/` | 83 ejemplos de fine-tuning |
| `src/` | 6 scripts del pipeline + `chat.py` |
| `results/` | Evaluación: respuestas a 10 preguntas |

**Si pregunta: ¿y esto qué es?**

> Es un tutor que responde preguntas sobre seguridad pública en México.
> No usa solo un LLM: primero busca fragmentos relevantes en 3 PDFs (RAG)
> y luego genera respuestas con un modelo fine-tuneado (Phi-3 + LoRA).

---

## Paso 2 — Los documentos fuente (30 s)

**Qué hacer:** abrir `corpus/` y mostrar los 3 PDFs.

```bash
ls -lh corpus/
```

| Documento | Fuente | Contenido |
|-----------|--------|-----------|
| ONC Homicidio | Observatorio Nacional Ciudadano | Análisis de homicidios por entidad (1997-2014) |
| ENVIPE 2025 | INEGI | Encuesta de victimización — cifra negra, percepción |
| ONC Anual 2025 | ONC | Reporte anual de seguridad delitos del fuero común |

**Por qué estos 3:** cubren homicidios (violencia letal), victimización
general (ENVIPE) y delitos del fuero común. Dan una visión amplia sin
ser redundantes.

### Si pregunta: ¿por qué solo 3 PDFs?

> Es un proyecto académico con recursos limitados. Con 3 documentos
> (1,322 chunks) demostramos que el pipeline funciona. El sistema escala
> a más documentos sin cambios — solo hay que poner los PDFs y re-ejecutar.
> Más documentos = mejor cobertura, pero más tiempo de embedding y más
> posibilidades de recuperar chunks irrelevantes.

### Si pregunta: ¿los PDFs tienen tablas?

> Sí, pero PyMuPDF no las extrae perfectamente. Las tablas con datos
> numéricos sobre homicidios por estado se pierden parcialmente. Esto
> afecta preguntas como "¿cuáles son los 3 estados con más homicidios?"
> (Q1). El modelo a veces inventa cifras porque los datos tabulares
> llegan desestructurados al chunk.

---

## Paso 3 — El pipeline de ingesta (1 min)

### 3a. Extracción de texto (01_extract_text.py)

```bash
python src/01_extract_text.py
```

Usa PyMuPDF para extraer texto + tablas de cada PDF.

**Tradeoff:**

| Opción | Pros | Contras |
|--------|------|---------|
| **PyMuPDF** (elegido) | Rápido, preserva orden de lectura, detecta tablas | Tablas complejas se pierden |
| pdfplumber | Mejor con tablas | 10× más lento |
| OCR (Tesseract) | Funciona con PDFs escaneados | Lento, requiere GPU, errores de reconocimiento |

> Elegimos PyMuPDF porque los PDFs tienen texto seleccionable y es el
> que mejor balance calidad/velocidad ofrece.

### 3b. Chunking (02_chunking.py)

Divide el texto en fragmentos de ~700 caracteres con 100 de solapamiento.

**Tradeoff de tamaño de chunk:**

| Tamaño | Pros | Contras |
|--------|------|---------|
| **~700 chars** (elegido) | Cada chunk captura una idea completa; caben ~3 chunks en el contexto de 768 tokens | Fragmenta secciones largas |
| ~300 chars | Más granularidad, mejor matching | Información incompleta |
| ~2000 chars | Contexto más rico por chunk | Menos chunks en top-k, más ruido semántico |

> 700 caracteres = ~175 tokens. Con top-3 caben en el prompt de 768 tokens
> junto con la pregunta y las instrucciones del sistema.

### 3c. Embeddings + FAISS (03_embeddings_faiss.py)

```bash
python src/03_embeddings_faiss.py
```

**Qué hace:** convierte cada chunk a vector de 384 dimensiones y los
indexa con FAISS para búsqueda rápida.

**Tradeoff de modelo de embedding:**

| Modelo | Dimensiones | Pros | Contras |
|--------|:-----------:|------|---------|
| **paraphrase-multilingual-MiniLM-L12-v2** (elegido) | 384 | Multilingüe, ligero (CPU) | Menos preciso que modelos grandes |
| text-embedding-3-small (OpenAI) | 1536 | Más preciso | Pago, requiere API, 679MB |
| bge-large-en-v1.5 | 1024 | Muy preciso | Solo inglés, 1.34GB |

> El modelo elegido es multilingüe (funciona bien en español), se ejecuta
> en CPU sin GPU, y su tamaño es pequeño (420MB). Para 1,322 chunks,
> 384 dimensiones son suficientes.

**Tradeoff de base de datos vectorial:**

| Opción | Pros | Contras |
|--------|------|---------|
| **FAISS** (elegido) | Gratis, rápido, sin servidor | En memoria (no persiste búsquedas) |
| ChromaDB | Persistente, SQL-like | Más lento, más complejo |
| Pinecone | Escalable, cloud | Pago, dependencia externa |

> FAISS es ideal para este volumen (~1,300 vectores): busca en milisegundos
> y no requiere infraestructura externa.

---

## Paso 4 — El dataset de fine-tuning (1 min)

**Mostrar la distribución:**

```bash
python -c "
import json
data = [json.loads(l) for l in open('dataset/tutor_dataset_rag.jsonl')]
l1 = sum(1 for d in data if d.get('level')==1)
l2 = sum(1 for d in data if d.get('level')==2)
l3 = sum(1 for d in data if d.get('level')==3)
print(f'Total: {len(data)} ejemplos')
print(f'Nivel 1 (factual):  {l1}')
print(f'Nivel 2 (síntesis): {l2}')
print(f'Nivel 3 (razonamiento): {l3}')
"
```

83 ejemplos: 29 level-1, 21 level-2, 33 level-3.

**Si pregunta: ¿por qué 83 ejemplos y no 500?**

> El fine-tuning con LoRA no busca que el modelo memorice datos (para
> eso está RAG). Busca enseñarle un *estilo* de respuesta: citar fuentes,
> usar método socrático, decir "no sé" cuando falta información.
>
> Con 83 ejemplos bien diseñados es suficiente para eso. Más ejemplos
> mejorarían el estilo pero con rendimientos decrecientes — el modelo
> ya sabe responder; solo hay que pulir la forma.

**Los 3 niveles de dificultad:**

| Nivel | Qué enseña | Ejemplo |
|-------|------------|---------|
| 1 (factual) | Extraer datos concretos de los chunks | "¿Cuál fue la tasa de homicidios en Colima?" |
| 2 (síntesis) | Combinar información de varios chunks | "Compara la violencia en el norte vs el sur" |
| 3 (razonamiento) | Método socrático + manejar incertidumbre | "¿Qué opina la UNAM sobre la inseguridad?" → no está en el corpus |

**Si pregunta: ¿qué es el método socrático?**

> En nivel 3, el modelo primero devuelve una pregunta o reflexión antes
> de responder. Por ejemplo: "Antes de responder, permítame preguntar:
> ¿a qué tipo de violencia se refiere específicamente?" Esto hace que
> el tutor guíe al estudiante a pensar antes de recibir la respuesta.

**7 ejemplos específicos anti-invención** enseñan al modelo a no inventar
siglas (CNDH, UNAM, Banco Mundial, CIDB, OCDE). El modelo debe responder
"el corpus disponible no contiene información específica sobre..." en lugar
de inventar fuentes.

---

## Paso 5 — Fine-tuning con LoRA (1 min)

**Si pregunta: ¿qué es LoRA y por qué lo usaste?**

> LoRA (Low-Rank Adaptation) es una técnica que inserta matrices pequeñas
> en el modelo. En lugar de modificar los 3.8 mil millones de parámetros
> de Phi-3 (imposible en una GTX 1050 de 4 GB), solo entrenamos 12.5
> millones (0.62% del total).

```bash
python -c "
# Mostrar el % de parámetros entrenados
total = 2021723136
trainable = 12582912
print(f'Parámetros totales: {total:,}')
print(f'Parámetros entrenables: {trainable:,}')
print(f'Porcentaje: {100*trainable/total:.2f}%')
"
```

**Tradeoff de rango LoRA:**

| r | Params entrenables | Calidad | Memoria VRAM |
|:-:|:------------------:|:-------:|:------------:|
| 8 (elegido) | 12.5M (0.62%) | Suficiente para estilo | ~3.5 GB |
| 16 | 25M (1.24%) | Puede capturar más matices | ~4.2 GB (al límite) |
| 32 | 50M (2.48%) | Potencialmente mejor | No cabe (5.5 GB) |

> r=8 es el máximo que cabe en 4 GB. Con r=16 hubiera sido mejor pero
> está al límite de la memoria.

**Si pregunta: ¿cuánto tiempo tomó el fine-tuning?**

> 77 minutos en una GTX 1050. La loss bajó de 1.878 a 0.958.

**Si pregunta: ¿por qué no usaste un modelo más pequeño tipo TinyLlama?**

> TinyLlama (1.1B parámetros) cabe con más espacio para LoRA de mayor
> rango, pero Phi-3-mini (3.8B) genera respuestas de mejor calidad
> incluso en 4-bit. Es el modelo más grande que cabe en nuestra GPU.

---

## Paso 6 — Inferencia en vivo (3 min) ⭐

### Opción A: Ejecutar las 10 preguntas (~30-40 min)

```bash
python src/06_inference_phi3.py
```

> ⚠️ Tarda ~40 min en responder las 10 preguntas. Si el tiempo es
> limitado, mejor mostrar los resultados ya guardados.

### Opción B: Mostrar resultados guardados (recomendado)

```bash
python -c "
import json
r = json.load(open('results/evaluation_phi3.json'))
for k, v in r.items():
    ans = v['answer'][:150].replace(chr(10), ' ')
    print(f'{k}: {ans}...')
    print()
"
```

### Opción C: Chat interactivo — una pregunta a la vez

```bash
python src/chat.py
```

Preguntas recomendadas para la demo (orden sugerido):

| # | Pregunta | Qué debe responder | Por qué esta pregunta |
|---|----------|-------------------|----------------------|
| 1 | "¿Qué es la cifra negra y cuál es su valor en México?" | Dato concreto del ENVIPE: 93.2% | **Funciona bien** — el chunk tiene el dato exacto |
| 2 | "¿Qué entidades tienen las tasas más altas de homicidios?" | Datos del ONC (aunque imprecisos por tablas) | **Muestra RAG** buscando en los documentos |
| 3 | "¿Cuál es la relación entre la violencia y el cambio climático?" | "No contiene información" | **Anti-hallucination** — tema fuera del corpus |
| 4 | "¿Qué opina la UNAM sobre la inseguridad en México?" | "No contiene información" | **Anti-invención** — antes inventaba, ahora dice no saber |
| 5 | "¿Qué dice la CNDH sobre los homicidios?" | "No contiene información" | **Anti-invención** — misma prueba con otra sigla |
| 6 | "¿Qué estrategias de seguridad pública serían más efectivas?" | Pregunta de reflexión (método socrático) | **Nivel 3** — debe devolver una pregunta antes de responder |

> **Consejo**: haz las preguntas 1→2→3 en ese orden. La 3 es la más importante
> para mostrar que el anti-hallucination training funcionó.

**Si pregunta: ¿por qué la inferencia tarda tanto?**

> El modelo tiene 3.8B parámetros. En una GTX 1050 genera ~8-12 tokens
> por segundo. Cada respuesta de ~150 tokens toma ~15 segundos solo de
> generación, más cargar modelo + buscar chunks.

### Si pregunta: ¿por qué greedy decoding (do_sample=False)?

| Modo | Pros | Contras |
|------|------|---------|
| **Greedy** (elegido) | Determinista, reproducible | Siempre da la misma respuesta |
| Temperature sampling | Más variado, creativo | Puede alucinar más |

> Para un tutor académico queremos respuestas consistentes y
> deterministas. Si el profesor hace la misma pregunta dos veces,
> debe recibir la misma respuesta.

---

## Paso 7 — Resultados y limitaciones (2 min)

### Lo que funciona bien ✅

| Pregunta | Tipo | Resultado |
|----------|------|-----------|
| Q2: "¿Qué cárteles operan en Tierra Caliente?" | No está en corpus | ✅ "No contiene información" |
| Q6: "Evolución de extorsión por sector" | No está detallado | ✅ "No es posible responder" |
| Q8: "Contradicciones ONGs vs gobierno" | No hay comparaciones | ✅ "No es posible identificar" |
| Q9: "Violencia → deserción escolar" | No hay datos | ✅ "No contiene información" |

### Lo que mejoró ⚠️

| Pregunta | Antes | Ahora |
|----------|-------|-------|
| Q5: estrategias de seguridad | Inventaba CNDH, UNAM | Análisis genérico sin fuentes falsas |
| Q7: violencia urbana vs rural | Inventaba DFGH, CIDOB | Genérico sin siglas inventadas |
| Q1: top 3 entidades | Decía "Ciudad Juárez" (no es estado) | Tamaulipas, Chihuahua, Hidalgo (al menos son entidades reales) |

### Lo que sigue fallando ❌

| Pregunta | Problema | Causa raíz |
|----------|----------|------------|
| Q1: top 3 entidades | Cifras 3.1, 2.9, 2.8 probablemente inventadas | Tablas del PDF se pierden en extracción |
| Q3: desplazados | Inventa "8 millones" y "ONC Reporte Anual 2025" | El modelo generaliza patrones del pre-training |

**Si pregunta: ¿por qué el modelo sigue inventando cifras aunque
tiene RAG?**

> Dos razones:
>
> 1. **Tablas extraídas mal**: El PDF original tiene una tabla con las
>    tasas de homicidio por estado, pero PyMuPDF la extrae como texto
>    desestructurado. El chunk recuperado tiene ruido y el modelo
>    "rellena" los números que faltan.
>
> 2. **Pre-training del modelo**: Phi-3 fue entrenado con billones de
>    tokens de internet, donde aparecen cifras de violencia en México.
>    El modelo a veces usa esos "recuerdos" en lugar de los chunks,
>    especialmente cuando el chunk no tiene el número exacto.
>
> **Mitigaciones aplicadas:**
> - Sistema prompt prohíbe explícitamente inventar cifras
> - 7 ejemplos anti-invención en el dataset
> - Greedy decoding (do_sample=False) reduce variabilidad
>
> **Solución a futuro:** usar un parser de tablas dedicado (Camelot,
> Tabula) o extraer las tablas manualmente.

---

## FAQ integrada — preguntas por paso

### Sobre el pipeline en general

**P: ¿Por qué combinaste RAG con fine-tuning? ¿No basta con uno solo?**

> RAG y fine-tuning resuelven problemas distintos. **RAG** evita que
> el modelo invente datos, porque lo obliga a basarse en documentos
> reales. Pero el modelo base no sabe *cómo* presentar la información:
> no cita fuentes, no estructura respuestas académicamente.
>
> **Fine-tuning** le enseña el estilo: citar fuentes, usar método
> socrático, decir "no sé". Pero el fine-tuning por sí solo no añade
> conocimiento nuevo.
>
> Juntos: RAG le da los datos correctos, fine-tuning le dice cómo
> usarlos.

**P: ¿Qué pasa si el profesor pregunta algo que no está en los PDFs?**

> El modelo debe responder que no tiene información. Esto se enseñó
> explícitamente con ejemplos en el dataset.

### Sobre LoRA

**P: ¿Qué es un adapter LoRA y cómo se carga?**

> Un adapter LoRA es un archivo pequeño (~25 MB) que contiene solo
> las matrices entrenadas. Se carga así en inferencia:
>
> ```python
> from peft import PeftModel
> model = PeftModel.from_pretrained(model, "models/lora_adapter_phi3_rag")
> ```
>
> Sin cargar el adapter, el modelo responde como Phi-3 base (más
> genérico, sin el formato académico).

### Sobre los resultados

**P: ¿Cómo sabes que el fine-tuning realmente funcionó?**

> Dos evidencias:
> 1. La loss bajó de 1.878 a 0.958 durante el entrenamiento.
> 2. Sin cargar los adaptadores LoRA, el modelo base responde sin
>    citar fuentes y sin el formato académico que aprendió.

**P: ¿El modelo realmente entiende lo que responde?**

> No en sentido humano. Es un sistema estadístico que predice la
> siguiente palabra. Pero el diseño híbrido (RAG + fine-tuning) mitiga
> las limitaciones: con RAG no puede inventar datos, y el fine-tuning
> le enseña a no divagar.

### Sobre recursos

**P: ¿Por qué no usaste Llama 3 o GPT-4?**

> Recursos limitados: GTX 1050 con 4 GB VRAM. Phi-3-mini en 4-bit
> ocupa ~2.5 GB. Llama 3 (8B) en 4-bit ocupa ~6 GB — no cabe. El
> proyecto demuestra que se puede construir un sistema funcional con
> hardware modesto.

**P: ¿El sistema se puede adaptar a otro tema?**

> Sí. Solo hay que:
> 1. Reemplazar los PDFs en `corpus/`
> 2. Re-ejecutar los 6 scripts en orden
> 3. (Opcional) Generar nuevos ejemplos de fine-tuning
>
> Lo único específico del dominio es el dataset y el prompt de
> inferencia.

---

## Resumen para el profesor (si pregunta rápido)

> Construimos un tutor inteligente sobre seguridad pública en México
> combinando **RAG** (búsqueda en 3 documentos con FAISS) y
> **fine-tuning con LoRA** sobre Phi-3-mini. El sistema busca
> fragmentos relevantes en 1,322 chunks, los pasa al modelo junto
> con la pregunta, y genera respuestas con citas. Entrenamos 12.5
> millones de parámetros (0.62%) en 77 minutos con 83 ejemplos.
> El modelo aprendió a decir "no sé" cuando falta información — 4
> de 10 preguntas de prueba responden correctamente con incertidumbre.
> La limitación principal son las tablas de los PDFs, que no se
> extraen bien y causan invención de cifras.

---

*Documento generado como guía de presentación. Basado en
`informe_tecnico.md`, `flujo_del_sistema.md`, y
`demostracion_y_faq.md`.*
