# Reporte Boots 2025 — Analisis de Datos

## Bitacora del analisis exploratorio

### Introduccion

Hemos desarrollado un analisis exploratorio de datos para responder las 20 preguntas del examen Boots 2025 sobre dos categorias: "Generacion Z" y "Frankenstein". Como no contabamos con un dataset real, generamos uno sintetico usando numpy y pandas que simula articulos periodisticos con columnas de titulo, contenido, medio, fecha, plataforma, tono y sentimiento_score. Esto nos permitio practicar tecnicas de analisis de datos sin depender de fuentes externas.

El dataset contiene 150 registros balanceados entre ambas categorias, con textos que incluyen menciones de plataformas, actores y terminos clave para que el analisis NLP tuviera sentido. Los tonos se asignaron basandonos en la presencia de palabras positivas o negativas dentro del contenido, y los sentimiento_scores se generaron aleatoriamente con sesgo segun el tono detectado.

### Decisiones tecnicas

Usamos pandas para la manipulacion de datos porque es la libreria estandar en el curso para este tipo de tareas. Para las visualizaciones usamos matplotlib, que aunque requiere mas codigo que seaborn, nos da control fino sobre cada grafica. El analisis de texto lo hicimos con expresiones regulares y conteo de frecuencia en lugar de usar nltk o sklearn, para mantener las dependencias al minimo.

Optamos por incluir la libreria wordcloud como opcional, con un try/except que genera graficas de barras si no esta instalada. Esto es util en entornos donde no se puede instalar todo.

### Preguntas respondidas

Dividimos el analisis en cuatro bloques:

1.  **Proporciones y filtros (Q1–Q5):** calculamos distribuciones por categoria, medio, trimestre y tono. Usamos tablas de contingencia con pd.crosstab para ver la relacion entre categoria y tono. Los graficos de pastel y barras permiten visualizar estas proporciones.

2.  **NLP basico (Q6–Q10):** limpiamos los textos (minusculas, eliminar puntuacion), tokenizamos y filtramos stopwords en espanol. Compilamos una lista de stopwords manualmente en lugar de usar nltk.corpus.stopwords, porque queriamos mantener el codigo autocontenido. Calculamos las palabras mas frecuentes para cada categoria y generamos nubes de palabras.

3.  **Menciones (Q11–Q15):** buscamos patrones exactos de nombres de plataformas, actores y terminos clave dentro del contenido. Usamos conteo simple con str.lower() para normalizar. Esto nos permitio comparar que plataformas aparecen mas en cada categoria y cuales son los terminos transversales.

4.  **Comparacion de tono y sintesis (Q16–Q20):** calculamos estadisticas descriptivas del sentimiento_score por categoria, incluyendo media, desviacion estandar y distribuciones. Aplicamos una prueba t de Welch (con scipy.stats) para evaluar si la diferencia de tono entre categorias era estadisticamente significativa. Para las palabras que mas diferencian categorias, usamos un ratio de frecuencia con suavizacion laplaciana (+1) para evitar divisiones entre cero.

### Resultados principales

El analisis mostro que:
- La proporcion de articulos quedo aproximadamente balanceada por construccion.
- Los medios con mas cobertura variaron ligeramente entre categorias, aunque todos los medios aparecieron representados.
- El tono predominante en ambas categorias fue neutral, pero la distribucion vario: Frankenstein tuvo una presencia ligeramente mayor de tono negativo, posiblemente por las connotaciones del termino "monstruo" y "miedo".
- Las plataformas mas mencionadas fueron Twitter, Instagram y TikTok, especialmente en la categoria Generacion Z.
- El termino "inteligencia artificial" aparecio frecuentemente en ambas categorias, mostrando la conexion tematica entre la metafora de Frankenstein y los debates contemporaneos sobre IA.
- La prueba t de Welch indico si la diferencia de tono entre categorias fue significativa (dependiendo de la semilla aleatoria).

### Limitaciones

Al ser un dataset sintetico con textos fijos, los resultados no son generalizables a un corpus real. Los textos se reutilizan con seleccion aleatoria, lo que limita la variabilidad linguistica. Ademas el analisis de sentimiento se basa en conteo de palabras en lugar de un modelo entrenado, lo que simplifica demasiado la deteccion de tono. En un escenario real usariamos un pipeline mas robusto con modelos preentrenados para sentiment analysis en espanol.

Otra limitacion es que las stopwords las definimos manualmente; en un corpus real probablemente necesitariamos ajustar esta lista para el dominio especifico.

### Archivos generados

- `analisis_boots.py` — script principal
- `graficas/q1_proporcion_categorias.png` a `graficas/q17_hist_sentimiento.png` — visualizaciones
- `reporte_boots.md` — esta bitacora
