# Embeddings y Busqueda Semantica — Reporte

## Que hicimos

Exploramos embeddings de texto usando sentence-transformers con el
modelo all-MiniLM-L6-v2 sobre un corpus de 10 frases en espanol
relacionadas con inteligencia artificial. Calculamos similitud coseno
entre todas las frases, probamos busqueda semantica con consultas
arbitrarias y visualizamos los embeddings en 2D con t-SNE.

## Proceso

Cargamos el corpus (10 frases sobre temas variados de IA: aprendizaje
profundo, NLP, transformers, sistemas de recomendacion, etica, etc.).
Con SentenceTransformer generamos vectores de 384 dimensiones para cada
frase.

Con cosine_similarity de sklearn calculamos la matriz de similitud
entre todos los pares. Las frases mas cercanas semanticamente (por
ejemplo, las que hablan de redes neuronales y aprendizaje profundo)
dieron valores altos de coseno (~0.6-0.7), mientras que pares no
relacionados (como etica vs. convolucionales) dieron valores bajos
(~0.1-0.2).

Para la busqueda semantica codificamos la consulta con el mismo modelo
y calculamos su similitud contra todos los embeddings del corpus. El
top-3 recupera las frases mas cercanas en el espacio vectorial.
Probamos con varias consultas y el orden de resultados fue coherente:
"redes neuronales y aprendizaje" devolvio las frases sobre aprendizaje
profundo y modelos de lenguaje; "recomendaciones personalizadas para
usuarios" recupero la frase sobre sistemas de recomendacion.

Aplicamos t-SNE para reducir los embeddings de 384D a 2D con
perplexity=5 y random_state=42. El grafico resultante muestra
agrupaciones esperadas: las frases sobre aprendizaje automatico
(profundo, reforzado, fine-tuning) aparecen cerca entre si, mientras
que etica y transformers quedan mas separadas.

## Problemas y aspectos mejorables

- No normalizamos los embeddings antes de calcular similitud. Aunque
  cosine_similarity normaliza internamente los vectores, es buena
  practica hacerlo explicitamente.
- t-SNE usamos con parametros fijos (perplexity=5, random_state=42).
  La interpretacion de las distancias en 2D no es exacta; cambios en
  perplexity mueven los clusters.
- El modelo all-MiniLM-L6-v2 esta entrenado en ingles principalmente.
  Para frases en espanol los embeddings funcionan (porque el modelo
  vio algo de multilenguaje) pero un modelo especifico para espanol
  daria mejores resultados.
- El corpus es muy pequeno (10 frases). t-SNE con pocos puntos tiende
  a sobreinterpretar distancias.
- La funcion busqueda_semantica recibe top_k fijo. Podria parametrizarse
  y tambien mostrar umbrales de confianza.
- No guardamos los embeddings en disco. Para un sistema real
  convendria usar FAISS o Annoy en lugar de calcular sobre la marcha.
