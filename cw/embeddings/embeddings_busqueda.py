import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

corpus = [
    "La inteligencia artificial transforma la industria moderna",
    "El aprendizaje profundo utiliza redes neuronales con muchas capas",
    "Los modelos de lenguaje generan texto de forma automatica",
    "El procesamiento de lenguaje natural permite entender el significado",
    "Las redes convolucionales son ideales para el analisis de imagenes",
    "Los transformers revolucionaron el campo del NLP en los ultimos anos",
    "El aprendizaje reforzado entrena agentes mediante recompensas",
    "Los sistemas de recomendacion usan filtrado colaborativo y basado en contenido",
    "La etica en IA aborda sesgos, privacidad y transparencia",
    "El fine-tuning adapta modelos pre-entrenados a tareas especificas"
]

modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = modelo.encode(corpus)

print(f"Dimension de cada embedding: {embeddings.shape[1]}")
print(f"Total de frases: {len(corpus)}")
print()

matriz_sim = cosine_similarity(embeddings)

print("Matriz de similitud coseno entre frases:")
print("-" * 50)
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        print(f"  [{i}] vs [{j}]: {matriz_sim[i][j]:.4f}  ->  {corpus[i][:40]}... / {corpus[j][:40]}...")
print()

def busqueda_semantica(query, embeddings, corpus, top_k=3):
    emb_query = modelo.encode([query])
    similitudes = cosine_similarity(emb_query, embeddings)[0]
    indices = np.argsort(similitudes)[::-1][:top_k]
    print(f"  Query: '{query}'")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. [{idx}] (sim={similitudes[idx]:.4f}) {corpus[idx]}")
    print()
    return indices

print("Busqueda semantica (top-3):")
print("=" * 50)
busqueda_semantica("redes neuronales y aprendizaje", embeddings, corpus)
busqueda_semantica("como las maquinas entienden el lenguaje", embeddings, corpus)
busqueda_semantica("recomendaciones personalizadas para usuarios", embeddings, corpus)

tsne = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(emb_2d):
    plt.scatter(x, y, c="steelblue", s=120, edgecolors="black", linewidths=0.5)
    plt.text(x + 0.3, y + 0.3, str(i), fontsize=9, fontweight="bold")

plt.title("Embeddings de frases sobre IA (t-SNE 2D)", fontsize=14)
plt.xlabel("Componente t-SNE 1")
plt.ylabel("Componente t-SNE 2")
plt.tight_layout()
plt.savefig("embeddings_tsne.png", dpi=150)
plt.show()

print("Grafico guardado como embeddings_tsne.png")
