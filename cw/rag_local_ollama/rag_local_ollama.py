#!/usr/bin/env python3
"""
RAG Local con Python + Ollama + FAISS

Ingesta documentos .txt/.md, genera chunks, embeddings via Ollama,
indexa con FAISS y responde preguntas con contexto.

Uso:
    python rag_local_ollama.py --docs ./docs --question "Explica X"
"""

import os
import sys
import argparse
import numpy as np
import requests
import faiss
import json

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CHARS_POR_CHUNK = 1500
SOLAPAMIENTO = 250
TOP_K = 4
MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_GENERACION = "llama3.2:3b"
URL_OLLAMA = "http://localhost:11434/api"

TEMPERATURA = 0.1


def cargar_documentos(ruta_carpeta):
    """Lee todos los .txt y .md de la carpeta."""
    documentos = []
    if not os.path.isdir(ruta_carpeta):
        print(f"Error: la carpeta '{ruta_carpeta}' no existe.")
        sys.exit(1)
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith((".txt", ".md")):
            ruta = os.path.join(ruta_carpeta, archivo)
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
            documentos.append({"archivo": archivo, "contenido": contenido})
    if not documentos:
        print(f"No se encontraron archivos .txt o .md en '{ruta_carpeta}'.")
        sys.exit(1)
    return documentos


def chunkear(texto, tamano=CHARS_POR_CHUNK, solape=SOLAPAMIENTO):
    """Divide texto en chunks con solapamiento."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        chunks.append(texto[inicio:fin])
        inicio += tamano - solape
    return chunks


def obtener_embedding(texto):
    """Llama a la API de Ollama para obtener el embedding."""
    respuesta = requests.post(
        f"{URL_OLLAMA}/embeddings",
        json={"model": MODELO_EMBEDDINGS, "prompt": texto},
    )
    respuesta.raise_for_status()
    return np.array(respuesta.json()["embedding"], dtype=np.float32)


def generar_respuesta(prompt, contexto, temperatura=TEMPERATURA):
    """Genera respuesta usando Ollama con el contexto recuperado."""
    system_prompt = (
        "Eres un asistente didactico que responde preguntas basandose exclusivamente "
        "en el contexto proporcionado. Si la respuesta no se encuentra en el contexto, "
        "di honestamente que no tienes esa informacion. Usa un tono claro y educativo."
    )

    prompt_completo = (
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {prompt}\n\n"
        f"Respuesta:"
    )

    payload = {
        "model": MODELO_GENERACION,
        "system": system_prompt,
        "prompt": prompt_completo,
        "options": {
            "temperature": temperatura,
        },
        "stream": False,
    }

    respuesta = requests.post(f"{URL_OLLAMA}/generate", json=payload)
    respuesta.raise_for_status()
    return respuesta.json()["response"]


def main():
    parser = argparse.ArgumentParser(
        description="RAG Local: responde preguntas sobre documentos usando Ollama + FAISS"
    )
    parser.add_argument(
        "--docs",
        required=True,
        help="Ruta a la carpeta con documentos .txt y .md",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Pregunta que quieres responder",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHARS_POR_CHUNK,
        help=f"Tamano de cada chunk (default: {CHARS_POR_CHUNK})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=SOLAPAMIENTO,
        help=f"Solapamiento entre chunks (default: {SOLAPAMIENTO})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Numero de chunks a recuperar (default: {TOP_K})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURA,
        help=f"Temperatura para generacion (default: {TEMPERATURA})",
    )

    args = parser.parse_args()
    tamano_chunk = args.chunk_size
    solape = args.overlap
    top_k = args.top_k
    temperatura = args.temperature

    print("Cargando documentos...")
    documentos = cargar_documentos(args.docs)

    print("Generando chunks...")
    todos_chunks = []
    metadatos = []
    for doc in documentos:
        chunks = chunkear(doc["contenido"], tamano_chunk, solape)
        for i, chunk in enumerate(chunks):
            todos_chunks.append(chunk)
            metadatos.append({"archivo": doc["archivo"], "chunk_id": i})

    print(f"  Total chunks: {len(todos_chunks)}")
    print("Generando embeddings via Ollama (esto puede tardar)...")

    embeddings_lista = []
    for i, chunk in enumerate(todos_chunks):
        emb = obtener_embedding(chunk)
        embeddings_lista.append(emb)
        if (i + 1) % 5 == 0:
            print(f"  Embeddings procesados: {i + 1}/{len(todos_chunks)}")

    embeddings = np.array(embeddings_lista, dtype=np.float32)

    # Normalizar L2 para similitud coseno con IndexFlatIP
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatIP(dimension)
    indice.add(embeddings)

    print(f"\nIndice FAISS construido con {indice.ntotal} vectores.")

    print("Generando embedding de la pregunta...")
    emb_pregunta = obtener_embedding(args.question)
    emb_pregunta = emb_pregunta.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(emb_pregunta)

    distancias, indices_recuperados = indice.search(emb_pregunta, top_k)

    print(f"\nChunks recuperados (top {top_k}):")
    contexto = ""
    for rank, idx in enumerate(indices_recuperados[0]):
        chunk = todos_chunks[idx]
        meta = metadatos[idx]
        contexto += f"\n--- Chunk {rank + 1} (archivo: {meta['archivo']}, chunk #{meta['chunk_id']}) ---\n"
        contexto += chunk.strip() + "\n"
        print(f"  {rank + 1}. {meta['archivo']} - chunk #{meta['chunk_id']} (distancia: {distancias[0][rank]:.4f})")

    print("\nGenerando respuesta...")
    respuesta = generar_respuesta(args.question, contexto, temperatura)

    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(respuesta)
    print("=" * 60)


if __name__ == "__main__":
    main()
