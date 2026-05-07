# Reporte: RAG Local con Python + Ollama + FAISS

## Objetivo

Construir un sistema RAG (Retrieval Augmented Generation) completamente local
que permita hacer preguntas sobre documentos de texto usando embeddings y un
modelo de lenguaje ejecutandose en nuestra computadora. La idea es no depender
de APIs externas como OpenAI y mantener el control de los datos.

## Que hicimos

Implementamos un pipeline clasico de RAG en un solo script, `rag_local_ollama.py`,
que consta de varias etapas que describimos a continuacion.

### 1. Ingesta de documentos

El script recibe una carpeta con archivos `.txt` y `.md` y lee todo su contenido.
Usamos `os.listdir()` para enumerar los archivos y filtramos por extension. Como
se trata de archivos de texto planos, no necesitamos nada mas que `open()` con
codificacion UTF-8.

### 2. Chunking

Dividimos cada documento en fragmentos (chunks) de tamano fijo con solapamiento.
Elegimos 1500 caracteres por chunk con 250 de solape porque es un tamano comun
para modelos pequeños como llama3.2:3b, que tiene una ventana de contexto de
4096 tokens y no podemos saturarla. El solapamiento ayuda a que informacion
importante no se parta justo en el borde de dos chunks.

Esto es mejorable: podriamos usar chunking recursivo por parrafos u oraciones,
pero para este ejercicio el chunking por caracteres es suficiente.

### 3. Embeddings

Usamos el modelo `nomic-embed-text` de Ollama para convertir cada chunk en un
vector numerico (embedding). La API de Ollama expone un endpoint `/api/embeddings`
que recibe el texto y devuelve el vector.

Llamamos a la API una vez por chunk con `requests.post()`. Esto es ineficiente
para muchos chunks porque no hay procesamiento por lotes, pero es funcional y
simple. Una mejora clara seria enviar los textos en batch cuando el modelo
lo soporte.

### 4. Vector store con FAISS

FAISS es una libreria de Facebook para busqueda de similitud en espacios
vectoriales. Usamos `IndexFlatIP` (Inner Product) junto con normalizacion L2
para obtener similitud coseno. La razon: la similitud coseno entre dos vectores
normalizados a norma 1 es exactamente el producto punto. Entonces:

1. Normalizamos todos los embeddings con `faiss.normalize_L2()`
2. Los agregamos a un `IndexFlatIP`
3. Para buscar, normalizamos el embedding de la pregunta de la misma forma
4. FAISS devuelve los indices con mayor producto punto (mayor similitud)

El indice se construye desde cero cada vez que ejecutamos el script. No hay
persistencia a disco, lo cual es otra area de mejora si trabajaramos con
muchos documentos.

### 5. Retrieval

Extraemos los top-k chunks mas similares (default 4). Mostramos cuales son
para que el usuario pueda verificar de donde viene la informacion. Esto es
importante en RAG porque permite auditar las fuentes.

### 6. Generacion

Construimos un prompt que combina el contexto recuperado con la pregunta del
usuario y se lo enviamos al modelo `llama3.2:3b` via el endpoint `/api/generate`
de Ollama.

Usamos un `system_prompt` que instruye al modelo a:
- Responder solo con la informacion del contexto
- Decir "no se" si no encuentra la respuesta
- Usar un tono didactico y claro

La temperatura la dejamos en 0.1 para que las respuestas sean deterministicas
y basadas en evidencia. Si quisieramos respuestas mas creativas podriamos
subirla hasta 0.3 como maximo, pero en un RAG queremos fidelidad sobre
creatividad.

## CLI con argparse

El script recibe parametros via linea de comandos:

```
python rag_local_ollama.py --docs ./docs --question "Explica el presente perfecto"
```

Los parametros disponibles son:

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--docs` | (requerido) | Carpeta con documentos .txt y .md |
| `--question` | (requerido) | Pregunta a responder |
| `--chunk-size` | 1500 | Caracteres por chunk |
| `--overlap` | 250 | Solapamiento entre chunks |
| `--top-k` | 4 | Chunks a recuperar |
| `--temperature` | 0.1 | Temperatura del modelo |

## Dependencias

Necesitamos tener instalado Python con las siguientes librerias:

```
pip install numpy requests faiss-cpu
```

Ademas, requiere tener Ollama corriendo en segundo plano con los modelos
`nomic-embed-text` y `llama3.2:3b` descargados:

```
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Notas

- El script asume que Ollama corre en `http://localhost:11434`. Si usamos otro
host o puerto, hay que cambiar la variable `URL_OLLAMA`.
- No implementamos cache de embeddings, asi que cada ejecucion regenera todo
el indice. Para un proyecto mas realista guardariamos los embeddings en disco.
- El chunking por caracteres puede cortar palabras o parrafos por la mitad.
Para documentos con estructura (como markdown) seria mejor usar chunking
recursivo que respete los encabezados.
- Usar la API de Ollama via HTTP es lento comparado con cargar el modelo
directamente en Python con `llama-cpp-python` o `sentence-transformers`,
pero tiene la ventaja de la simplicidad.
