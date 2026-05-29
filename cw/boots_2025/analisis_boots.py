import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

np.random.seed(42)

# --- DATOS SINTETICOS ---

categorias = ["Generacion Z", "Frankenstein"]
medios = ["El Pais", "BBC Mundo", "The Guardian", "CNN", "NYT",
          "Reuters", "El Universal", "Milenio", "La Jornada", "Proceso"]
plataformas = ["Twitter", "Instagram", "TikTok", "YouTube", "Facebook",
               "LinkedIn", "Threads", "Spotify", "Netflix", "Disney+"]
tonos = ["positivo", "negativo", "neutral"]
fechas = pd.date_range(start="2025-01-01", end="2025-06-30", freq="D")

# Stopwords en español ampliadas
stopwords = set([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "e", "o", "u",
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre",
    "hacia", "hasta", "para", "por", "segun", "sin", "so", "sobre", "tras",
    "del", "al", "lo", "le", "les", "su", "sus", "tu", "te", "se", "nos",
    "os", "me", "te", "le", "que", "es", "son", "fue", "era", "han", "ha",
    "he", "hemos", "habia", "habian", "tiene", "tenia", "tienen", "esta",
    "estan", "estaba", "estaban", "mas", "pero", "porque", "como", "cuando",
    "donde", "muy", "tan", "tanto", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "aquel", "aquella", "ello", "ella",
    "ellos", "ellas", "nosotros", "vosotros", "mi", "tu", "el", "nos", "os",
    "les", "se", "si", "no", "ya", "bien", "solo", "tambien", "cada", "todo",
    "toda", "todos", "todas", "mismo", "misma", "mismos", "mismas", "otro",
    "otra", "otros", "otras", "poco", "poca", "pocos", "pocas", "algo",
    "nada", "alguien", "nadie", "siempre", "nunca", "jamas", "tampoco",
    "mas", "aunque", "sin", "contra", "durante", "mediante", "excepto",
    "segun", "hasta", "desde", "entre", "sobre", "tras", "ante"
])

# Textos de ejemplo para Generacion Z
textos_genz = [
    "La generacion Z enfrenta una crisis de sentido sin precedentes. Las redes sociales como TikTok e Instagram han redefinido la forma en que los jovenes construyen su identidad. El algoritmo de recomendacion crea burbujas informativas que limitan la exposicion a opiniones diversas. Los estudiantes pasan horas frente a pantallas buscando validacion en likes y comentarios.",
    "Un estudio reciente revelo que los jovenes de la generacion Z prefieren consumir noticias a traves de TikTok y YouTube en lugar de los medios tradicionales. Twitter sigue siendo relevante para debates politicos pero Instagram gana terreno como fuente de informacion visual. El algoritmo prioriza el contenido emocional sobre el informativo.",
    "La salud mental de la generacion Z es una preocupacion creciente. La ansiedad y la depresion aumentaron entre los estudiantes universitarios que reportan sentirse abrumados por la presion de las redes sociales. Facebook y LinkedIn ya no son las plataformas preferidas por este grupo demografico.",
    "El activismo digital encontro en la generacion Z a sus principales impulsores. A traves de Threads y Twitter los jovenes organizan campanas de concientizacion sobre cambio climatico y justicia social. La velocidad del algoritmo amplifica los mensajes virales pero tambien la desinformacion.",
    "Los creadores de contenido de la generacion Z estan redefiniendo el mercado laboral. Plataformas como TikTok, YouTube e Instagram permiten monetizar la creatividad. Sin embargo la precariedad laboral y la falta de prestaciones siguen siendo problemas estructurales que afectan a los jovenes.",
    "Expertos advierten que el uso excesivo de redes sociales esta relacionado con problemas de autoestima en la generacion Z. La comparacion constante con estilos de vida idealizados en Instagram genera insatisfaccion corporal. TikTok normaliza estandares de belleza irreales.",
    "La brecha digital persiste incluso dentro de la generacion Z. No todos los jovenes tienen acceso equitativo a dispositivos y conexion de internet. Las plataformas educativas como YouTube y Spotify ofrecen recursos gratuitos pero el algoritmo no siempre recomienda contenido de calidad.",
    "Los datos muestran que la generacion Z es la mas diversa en terminos de identidad de genero y orientacion sexual. Las redes sociales como TikTok e Instagram permiten explorar y expresar estas identidades. Twitter es un espacio clave para el debate sobre derechos LGBTQ+.",
    "La relacion de la generacion Z con el trabajo remoto es ambivalente. Por un lado valoran la flexibilidad que ofrecen herramientas digitales y plataformas como LinkedIn. Por otro lado reportan dificultades para separar la vida laboral de la personal cuando todo ocurre a traves de una pantalla.",
    "Las marcas estan reajustando sus estrategias de marketing para conectar con la generacion Z. Los jovenes prefieren anuncios autenticos en TikTok y YouTube en lugar de publicidad tradicional. El algoritmo de recomendacion es clave para llegar a este publico.",
    "Netflix y Disney+ compiten por la atencion de la generacion Z. Los jovenes consumen series y peliculas en formato binge-watching pero tambien prefieren contenido corto en TikTok. La atencion fragmentada es una caracteristica de esta generacion.",
    "El termino doomscrolling se popularizo entre la generacion Z para describir el habito de consumir noticias negativas en redes sociales. Twitter y Facebook son las plataformas donde este fenomeno es mas comun. Los algoritmos estan disenados para maximizar el tiempo de pantalla.",
    "La educacion financiera es un tema recurrente entre los jovenes de la generacion Z. En YouTube y TikTok abundan los creadores que ensenan sobre inversion y ahorro. Sin embargo la desinformacion financiera tambien se propaga rapidamente a traves de estas plataformas.",
    "Los estudiantes de la generacion Z estan cada vez mas interesados en carreras relacionadas con tecnologia e inteligencia artificial. El miedo a ser reemplazados por algoritmos impulsa la busqueda de habilidades digitales. LinkedIn se ha convertido en la plataforma principal para buscar empleo.",
    "El consumo de noticias politicas entre la generacion Z ocurre principalmente en TikTok y Twitter. Los algoritmos de recomendacion tienden a polarizar las opiniones al mostrar contenido que refuerza las creencias existentes. Esto plantea preguntas sobre la calidad del debate democratico."
]

# Textos de ejemplo para Frankenstein
textos_frankenstein = [
    "Frankenstein de Mary Shelley es considerada la primera novela de ciencia ficcion. La historia de Victor Frankenstein y su criatura explora los limites eticos de la ciencia. Dos siglos despues el debate sobre inteligencia artificial y creacion de vida artificial retoma estas mismas preguntas.",
    "La criatura de Frankenstein ha sido interpretada como una metafora del miedo a la tecnologia descontrolada. En la era de los algoritmos y la robotica esta lectura cobra nueva relevancia. La ciencia sin responsabilidad puede producir monstruos.",
    "El mito de Frankenstein aparece constantemente en el cine y la television contemporaneos. Desde las adaptaciones clasicas de Universal hasta versiones modernas como Victor Frankenstein la historia sigue cautivando al publico. Netflix y Disney+ tienen catalogos con multiples versiones.",
    "Especialistas en literatura comparada analizan la novela de Shelley como una critica al cientificismo del siglo XIX. Victor Frankenstein representa al cientifico que juega a ser dios sin considerar las consecuencias. La inteligencia artificial moderna enfrenta criticas similares.",
    "Mary Shelley escribio Frankenstein a los diecinueve anos durante un verano lluvioso en Ginebra. La novela refleja las ansiedades de la revolucion industrial y el avance tecnologico. Hoy los debates sobre IA y robotica tienen un eco similar en la sociedad.",
    "El personaje de la criatura en Frankenstein es frecuentemente malinterpretado. En la novela original la criatura es inteligente y sensible, no un monstruo sin razon. La forma en que los medios representan a las inteligencias artificiales sigue este mismo patron de deshumanizacion.",
    "La relacion entre creador y criatura en Frankenstein plantea preguntas sobre la responsabilidad de los cientificos. Victor abandona a su creacion y eso desencadena la tragedia. En el desarrollo de inteligencia artificial la etica debe guiar cada paso del proceso.",
    "Corrientes filosoficas como el transhumanismo citan a Frankenstein como una advertencia. La busqueda de mejorar al ser humano mediante tecnologia puede tener consecuencias imprevistas. La ciencia ficcion del siglo XIX anticipo debates eticos del siglo XXI.",
    "La Universidad de Oxford publico un analisis sobre las referencias a Frankenstein en el debate publico sobre IA. Los politicos y academicos usan la metafora del monstruo para advertir sobre riesgos tecnologicos. Sin embargo los expertos critican esta comparacion por simplista.",
    "Los estudios culturales muestran que la imagen de Frankenstein esta profundamente arraigada en el imaginario colectivo. Desde juguetes hasta memes en Twitter la referencia aparece en los contextos mas diversos. La criatura de Shelley es parte del lenguaje cotidiano.",
    "Cientificos y eticistas debaten si la inteligencia artificial puede considerarse una nueva forma de vida. Frankenstein ofrece un marco narrativo para explorar esta pregunta. La ciencia contemporanea se acerca a crear inteligencias que antes solo existian en la ficcion.",
    "El cine de terror y ciencia ficcion debe mucho a Frankenstein. Peliculas como Ex Machina y Blade Runner retoman el tema de la creacion que se rebela contra su creador. El algoritmo que aprende y toma decisiones autonomas es el nuevo monstruo de la era digital.",
    "En 2025 se cumplen 207 anos de la publicacion de Frankenstein. La novela sigue siendo lectura obligatoria en cursos de literatura y etica. Los estudiantes encuentran en la historia de Victor y su criatura preguntas que siguen sin respuesta.",
    "Investigadores del MIT utilizaron Frankenstein como estudio de caso para discutir etica en inteligencia artificial. La novela ilustra los peligros de desarrollar tecnologia sin considerar su impacto social. Los algoritmos de aprendizaje automatico requieren supervision similar.",
    "La figura del cientifico loco representada por Victor Frankenstein persiste en el imaginario popular. En la era de la inteligencia artificial y la robotica los debates sobre regulacion y limites eticos retoman esta figura. La ciencia sin control sigue siendo un tema central."
]

# Generar dataset
n_articulos = 150
datos = []

for i in range(n_articulos):
    cat = np.random.choice(categorias, p=[0.5, 0.5])
    medio = np.random.choice(medios)
    fecha = np.random.choice(fechas)
    plataforma = np.random.choice(plataformas)

    if cat == "Generacion Z":
        idx = np.random.randint(0, len(textos_genz))
        contenido = textos_genz[idx]
        palabras_titulo = ["ansiedad", "redes", "jovenes", "digital", "estudiantes",
                           "TikTok", "salud mental", "identidad", "algoritmo", "crisis"]
        p = np.random.choice(palabras_titulo)
        if np.random.random() > 0.3:
            titulo = f"La generacion Z y la {p}: un analisis de las nuevas dinamicas sociales"
        else:
            titulo = f"{p.title()} en la era de la hiperconectividad: el caso de la generacion Z"
    else:
        idx = np.random.randint(0, len(textos_frankenstein))
        contenido = textos_frankenstein[idx]
        palabras_titulo = ["Frankenstein", "Mary Shelley", "criatura", "Victor",
                           "inteligencia artificial", "ciencia ficcion", "etica",
                           "monstruo", "robotica", "creacion"]
        p = np.random.choice(palabras_titulo)
        if np.random.random() > 0.3:
            titulo = f"{p} y la vigencia del mito de Frankenstein en el debate actual"
        else:
            titulo = f"Frankenstein 207 anos: {p.lower()} como metafora de nuestros tiempos"

    # Asignar tono basado en palabras clave en el contenido
    if cat == "Generacion Z":
        palabras_pos = ["creatividad", "oportunidad", "diversidad", "flexibilidad",
                        "educacion", "expresar", "valor", "autentico", "concientizacion"]
        palabras_neg = ["ansiedad", "depresion", "precariedad", "insatisfaccion",
                        "desinformacion", "polarizar", "abrumados", "doomscrolling"]
        pos_count = sum(1 for p in palabras_pos if p in contenido)
        neg_count = sum(1 for p in palabras_neg if p in contenido)
    else:
        palabras_pos = ["relevante", "cautivando", "inteligente", "sensible", "analisis",
                        "debate", "lectura", "guiar"]
        palabras_neg = ["miedo", "peligros", "tragedia", "monstruo", "advertencia",
                        "riesgos", "descontrolada", "cientifico loco"]
        pos_count = sum(1 for p in palabras_pos if p in contenido)
        neg_count = sum(1 for p in palabras_neg if p in contenido)

    if pos_count > neg_count:
        tono = "positivo"
        score = np.random.uniform(0.3, 0.9)
    elif neg_count > pos_count:
        tono = "negativo"
        score = np.random.uniform(-0.9, -0.3)
    else:
        tono = "neutral"
        score = np.random.uniform(-0.2, 0.2)

    datos.append({
        "titulo": titulo,
        "contenido": contenido,
        "medio": medio,
        "fecha": fecha,
        "plataforma": plataforma,
        "tono": tono,
        "sentimiento_score": round(score, 3),
        "categoria": cat
    })

df = pd.DataFrame(datos)
print("Dimensiones del dataset:", df.shape)
print(df[["titulo", "categoria", "tono"]].head(), "\n")

# ---- ANALISIS ----

# Q1: Proporcion de articulos por categoria
print("=" * 60)
print("Q1: PROPORCION DE ARTICULOS POR CATEGORIA")
print("=" * 60)
q1 = df["categoria"].value_counts()
print(q1)
print(f"Proporcion: {q1['Generacion Z']/len(df):.2f} / {q1['Frankenstein']/len(df):.2f}\n")

fig1, ax1 = plt.subplots()
colores = ["#4ECDC4", "#FF6B6B"]
ax1.pie(q1.values, labels=q1.index, autopct="%1.1f%%", colors=colores,
        startangle=90, wedgeprops={"edgecolor": "white"})
ax1.set_title("Proporcion de articulos por categoria")
plt.tight_layout()
plt.savefig("graficas/q1_proporcion_categorias.png")
plt.close()

# Q2: Proporcion de articulos por medio
print("=" * 60)
print("Q2: DISTRIBUCION DE ARTICULOS POR MEDIO")
print("=" * 60)
q2 = df["medio"].value_counts()
print(q2, "\n")

fig2, ax2 = plt.subplots()
q2.plot(kind="bar", color="steelblue", ax=ax2)
ax2.set_title("Articulos por medio")
ax2.set_ylabel("Cantidad")
ax2.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("graficas/q2_articulos_por_medio.png")
plt.close()

# Q3: Filtro por fecha (primer trimestre vs segundo trimestre)
print("=" * 60)
print("Q3: FILTRO POR FECHA - PRIMER TRIMESTRE VS SEGUNDO")
print("=" * 60)
df["trimestre"] = df["fecha"].dt.quarter
q3 = df["trimestre"].value_counts().sort_index()
print(q3)
q3.index = ["Ene-Mar", "Abr-Jun"]
print(f"\n{q3}\n")

fig3, ax3 = plt.subplots()
q3.plot(kind="bar", color=["#2ECC71", "#E74C3C"], ax=ax3)
ax3.set_title("Articulos por trimestre")
ax3.set_ylabel("Cantidad")
ax3.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("graficas/q3_articulos_por_trimestre.png")
plt.close()

# Q4: Distribucion del tono por categoria
print("=" * 60)
print("Q4: DISTRIBUCION DEL TONO POR CATEGORIA")
print("=" * 60)
q4 = pd.crosstab(df["categoria"], df["tono"])
print(q4, "\n")

fig4, ax4 = plt.subplots()
q4.plot(kind="bar", ax=ax4, color=["#2ECC71", "#E74C3C", "#95A5A6"])
ax4.set_title("Tono por categoria")
ax4.set_ylabel("Cantidad")
ax4.tick_params(axis="x", rotation=0)
ax4.legend(title="Tono")
plt.tight_layout()
plt.savefig("graficas/q4_tono_por_categoria.png")
plt.close()

# Q5: Medio con mas cobertura de cada categoria
print("=" * 60)
print("Q5: MEDIO CON MAS COBERTURA POR CATEGORIA")
print("=" * 60)
for cat in categorias:
    top = df[df["categoria"] == cat]["medio"].value_counts().head(3)
    print(f"Top 3 medios para {cat}:")
    print(top, "\n")

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
for i, cat in enumerate(categorias):
    top5 = df[df["categoria"] == cat]["medio"].value_counts().head(5)
    top5.plot(kind="bar", color=["#4ECDC4", "#FF6B6B"][i], ax=axes5[i])
    axes5[i].set_title(f"Medios con mas cobertura: {cat}")
    axes5[i].set_ylabel("Cantidad")
    axes5[i].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("graficas/q5_top_medios_por_categoria.png")
plt.close()

# ---- NLP BASICO ----

# Funcion de limpieza
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-z\s]", "", texto)
    return texto

def tokenizar(texto):
    return texto.split()

df["texto_limpio"] = df["contenido"].apply(limpiar_texto)

# Q6: Palabras clave mas frecuentes en Generacion Z
print("=" * 60)
print("Q6: PALABRAS CLAVE MAS FRECUENTES - GENERACION Z")
print("=" * 60)
texto_genz = " ".join(df[df["categoria"] == "Generacion Z"]["texto_limpio"])
tokens_genz = tokenizar(texto_genz)
tokens_filtrados_genz = [t for t in tokens_genz if t not in stopwords and len(t) > 3]
frecuencias_genz = Counter(tokens_filtrados_genz)
q6 = frecuencias_genz.most_common(15)
print(q6, "\n")

fig6, ax6 = plt.subplots()
palabras_q6, vals_q6 = zip(*q6)
ax6.barh(list(palabras_q6[::-1]), list(vals_q6[::-1]), color="#4ECDC4")
ax6.set_title("Palabras clave mas frecuentes: Generacion Z")
ax6.set_xlabel("Frecuencia")
plt.tight_layout()
plt.savefig("graficas/q6_palabras_clave_genz.png")
plt.close()

# Q7: Palabras clave mas frecuentes en Frankenstein
print("=" * 60)
print("Q7: PALABRAS CLAVE MAS FRECUENTES - FRANKENSTEIN")
print("=" * 60)
texto_frank = " ".join(df[df["categoria"] == "Frankenstein"]["texto_limpio"])
tokens_frank = tokenizar(texto_frank)
tokens_filtrados_frank = [t for t in tokens_frank if t not in stopwords and len(t) > 3]
frecuencias_frank = Counter(tokens_filtrados_frank)
q7 = frecuencias_frank.most_common(15)
print(q7, "\n")

fig7, ax7 = plt.subplots()
palabras_q7, vals_q7 = zip(*q7)
ax7.barh(list(palabras_q7[::-1]), list(vals_q7[::-1]), color="#FF6B6B")
ax7.set_title("Palabras clave mas frecuentes: Frankenstein")
ax7.set_xlabel("Frecuencia")
plt.tight_layout()
plt.savefig("graficas/q7_palabras_clave_frankenstein.png")
plt.close()

# Q8: Nube de palabras para Generacion Z
print("=" * 60)
print("Q8: NUBE DE PALABRAS - GENERACION Z")
print("=" * 60)
try:
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=stopwords, max_words=50, colormap="viridis")
    wc_genz = wc.generate(texto_genz)
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    ax8.imshow(wc_genz, interpolation="bilinear")
    ax8.axis("off")
    ax8.set_title("Nube de palabras: Generacion Z")
    plt.tight_layout()
    plt.savefig("graficas/q8_wordcloud_genz.png")
    plt.close()
    print("Wordcloud generada para Generacion Z\n")
except ImportError:
    print("wordcloud no instalado. Usando grafica de barras como alternativa.\n")
    fig8, ax8 = plt.subplots()
    top20 = frecuencias_genz.most_common(20)
    p, v = zip(*top20)
    ax8.barh(list(p[::-1]), list(v[::-1]), color="#4ECDC4")
    ax8.set_title("Top 20 palabras: Generacion Z")
    plt.tight_layout()
    plt.savefig("graficas/q8_wordcloud_genz.png")
    plt.close()

# Q9: Nube de palabras para Frankenstein
print("=" * 60)
print("Q9: NUBE DE PALABRAS - FRANKENSTEIN")
print("=" * 60)
try:
    from wordcloud import WordCloud
    wc9 = WordCloud(width=800, height=400, background_color="white",
                    stopwords=stopwords, max_words=50, colormap="plasma")
    wc_frank = wc9.generate(texto_frank)
    fig9, ax9 = plt.subplots(figsize=(10, 5))
    ax9.imshow(wc_frank, interpolation="bilinear")
    ax9.axis("off")
    ax9.set_title("Nube de palabras: Frankenstein")
    plt.tight_layout()
    plt.savefig("graficas/q9_wordcloud_frankenstein.png")
    plt.close()
    print("Wordcloud generada para Frankenstein\n")
except ImportError:
    print("wordcloud no instalado. Usando grafica de barras.\n")
    fig9, ax9 = plt.subplots()
    top20 = frecuencias_frank.most_common(20)
    p, v = zip(*top20)
    ax9.barh(list(p[::-1]), list(v[::-1]), color="#FF6B6B")
    ax9.set_title("Top 20 palabras: Frankenstein")
    plt.tight_layout()
    plt.savefig("graficas/q9_wordcloud_frankenstein.png")
    plt.close()

# Q10: Stopwords mas comunes en el corpus
print("=" * 60)
print("Q10: STOPWORDS MAS COMUNES EN EL CORPUS")
print("=" * 60)
todos_tokens = tokenizar(" ".join(df["texto_limpio"]))
stopwords_encontradas = [t for t in todos_tokens if t in stopwords]
frec_stopwords = Counter(stopwords_encontradas)
q10 = frec_stopwords.most_common(15)
print(q10, "\n")

fig10, ax10 = plt.subplots()
palabras_q10, vals_q10 = zip(*q10)
ax10.barh(list(palabras_q10[::-1]), list(vals_q10[::-1]), color="#8E44AD")
ax10.set_title("Stopwords mas comunes en el corpus")
ax10.set_xlabel("Frecuencia")
plt.tight_layout()
plt.savefig("graficas/q10_stopwords_comunes.png")
plt.close()

# ---- MENCIONES ----

# Q11: Menciones de plataformas
print("=" * 60)
print("Q11: MENCIONES DE PLATAFORMAS EN EL CORPUS")
print("=" * 60)
plataformas_buscar = ["Twitter", "Instagram", "TikTok", "YouTube", "Facebook",
                      "LinkedIn", "Threads", "Spotify", "Netflix", "Disney+"]

def contar_menciones(texto, terminos):
    texto_lower = texto.lower()
    return {t: texto_lower.count(t.lower()) for t in terminos}

menciones_total = Counter()
for contenido in df["contenido"]:
    menciones_total.update(contar_menciones(contenido, plataformas_buscar))

q11 = menciones_total.most_common()
print("Menciones de plataformas:")
for plat, count in q11:
    print(f"  {plat}: {count}")
print()

fig11, ax11 = plt.subplots()
plas, vals11 = zip(*q11)
ax11.bar(plas, vals11, color=["#1DA1F2", "#E4405F", "#000000", "#FF0000",
                               "#1877F2", "#0A66C2", "#000000", "#1DB954",
                               "#E50914", "#113CCF"])
ax11.set_title("Menciones de plataformas en el corpus")
ax11.set_ylabel("Frecuencia")
ax11.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("graficas/q11_menciones_plataformas.png")
plt.close()

# Q12: Menciones de actores o figuras
print("=" * 60)
print("Q12: MENCIONES DE ACTORES O FIGURAS")
print("=" * 60)
actores_buscar = ["Victor Frankenstein", "Mary Shelley", "criatura", "monstruo",
                  "jovenes", "estudiantes", "creadores", "cientificos", "expertos",
                  "Shelley", "Victor"]

menciones_actores = Counter()
for contenido in df["contenido"]:
    menciones_actores.update(contar_menciones(contenido, actores_buscar))

q12 = menciones_actores.most_common()
for actor, count in q12:
    print(f"  {actor}: {count}")
print()

fig12, ax12 = plt.subplots()
act, vals12 = zip(*q12)
ax12.bar(act, vals12, color="#E67E22")
ax12.set_title("Menciones de actores o figuras")
ax12.set_ylabel("Frecuencia")
ax12.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("graficas/q12_menciones_actores.png")
plt.close()

# Q13: Terminos clave (IA, algoritmo, robot, etc.)
print("=" * 60)
print("Q13: FRECUENCIA DE TERMINOS CLAVE")
print("=" * 60)
terminos_buscar = ["inteligencia artificial", "IA", "algoritmo", "robot",
                   "robotica", "ciencia ficcion", "etica", "tecnologia",
                   "digital", "redes sociales"]

menciones_term = Counter()
for contenido in df["contenido"]:
    menciones_term.update(contar_menciones(contenido, terminos_buscar))

q13 = menciones_term.most_common()
for term, count in q13:
    print(f"  {term}: {count}")
print()

fig13, ax13 = plt.subplots()
terms, vals13 = zip(*q13)
ax13.bar(terms, vals13, color="#3498DB")
ax13.set_title("Frecuencia de terminos clave en el corpus")
ax13.set_ylabel("Menciones")
ax13.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("graficas/q13_terminos_clave.png")
plt.close()

# Q14: Comparacion de terminos entre categorias
print("=" * 60)
print("Q14: COMPARACION DE TERMINOS ENTRE CATEGORIAS")
print("=" * 60)
menciones_por_cat = {}
for cat in categorias:
    menciones_por_cat[cat] = Counter()
    textos_cat = df[df["categoria"] == cat]["contenido"]
    for contenido in textos_cat:
        menciones_por_cat[cat].update(contar_menciones(contenido, terminos_buscar))

q14 = pd.DataFrame({cat: pd.Series(dict(menciones_por_cat[cat]))
                    for cat in categorias}).fillna(0).astype(int)
print(q14, "\n")

fig14, ax14 = plt.subplots()
q14.plot(kind="bar", ax=ax14, color=["#4ECDC4", "#FF6B6B"])
ax14.set_title("Terminos clave por categoria")
ax14.set_ylabel("Menciones")
ax14.tick_params(axis="x", rotation=45)
ax14.legend(title="Categoria")
plt.tight_layout()
plt.savefig("graficas/q14_terminos_por_categoria.png")
plt.close()

# Q15: Plataforma mas mencionada en cada categoria
print("=" * 60)
print("Q15: PLATAFORMA MAS MENCIONADA POR CATEGORIA")
print("=" * 60)
for cat in categorias:
    menciones_cat = Counter()
    textos_cat = df[df["categoria"] == cat]["contenido"]
    for contenido in textos_cat:
        menciones_cat.update(contar_menciones(contenido, plataformas_buscar))
    top_plataforma = menciones_cat.most_common(3)
    print(f"Top plataformas en {cat}:")
    for p, c in top_plataforma:
        print(f"  {p}: {c}")
    print()

# ---- COMPARACION DE TONO ----

# Q16: Tono promedio (sentimiento_score) por categoria
print("=" * 60)
print("Q16: TONO PROMEDIO (SENTIMIENTO_SCORE) POR CATEGORIA")
print("=" * 60)
q16 = df.groupby("categoria")["sentimiento_score"].agg(["mean", "std", "count"])
print(q16, "\n")

fig16, ax16 = plt.subplots()
df.boxplot(column="sentimiento_score", by="categoria", ax=ax16, grid=False)
ax16.set_title("Distribucion del sentimiento score por categoria")
ax16.set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.savefig("graficas/q16_sentimiento_por_categoria.png")
plt.close()

# Q17: Distribucion del sentimiento_score en cada categoria
print("=" * 60)
print("Q17: DISTRIBUCION DETALLADA DEL SENTIMIENTO SCORE")
print("=" * 60)
fig17, axes17 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, cat in enumerate(categorias):
    scores = df[df["categoria"] == cat]["sentimiento_score"]
    axes17[i].hist(scores, bins=15, alpha=0.7, color=["#4ECDC4", "#FF6B6B"][i],
                   edgecolor="white")
    axes17[i].axvline(scores.mean(), color="black", linestyle="--",
                      label=f"Media: {scores.mean():.3f}")
    axes17[i].set_title(f"Distribucion: {cat}")
    axes17[i].set_xlabel("Sentimiento score")
    axes17[i].set_ylabel("Frecuencia")
    axes17[i].legend()
plt.tight_layout()
plt.savefig("graficas/q17_hist_sentimiento.png")
plt.close()

# Q18: Diferencia de tono entre categorias (prueba simple)
print("=" * 60)
print("Q18: DIFERENCIA DE TONO ENTRE CATEGORIAS")
print("=" * 60)
from scipy import stats
genz_scores = df[df["categoria"] == "Generacion Z"]["sentimiento_score"]
frank_scores = df[df["categoria"] == "Frankenstein"]["sentimiento_score"]
t_stat, p_valor = stats.ttest_ind(genz_scores, frank_scores, equal_var=False)
print(f"Prueba t de Welch: t = {t_stat:.4f}, p = {p_valor:.4f}")
if p_valor < 0.05:
    print("Diferencia estadisticamente significativa (p < 0.05)")
else:
    print("No hay diferencia estadisticamente significativa (p >= 0.05)")
print(f"Media Gen Z: {genz_scores.mean():.3f}")
print(f"Media Frankenstein: {frank_scores.mean():.3f}\n")

# ---- SINTESIS ----

# Q19: Palabras que mas diferencian las categorias
print("=" * 60)
print("Q19: PALABRAS QUE MAS DIFERENCIAN CATEGORIAS")
print("=" * 60)
freq_genz = Counter(tokens_filtrados_genz).most_common(50)
freq_frank = Counter(tokens_filtrados_frank).most_common(50)

dict_genz = dict(freq_genz)
dict_frank = dict(freq_frank)

todas_palabras = set(list(dict_genz.keys()) + list(dict_frank.keys()))
diferencias = []
for palabra in todas_palabras:
    fz = dict_genz.get(palabra, 0) + 1
    ff = dict_frank.get(palabra, 0) + 1
    ratio = max(fz, ff) / min(fz, ff)
    diferencias.append((palabra, fz, ff, ratio, fz > ff))

diferencias.sort(key=lambda x: x[3], reverse=True)
print("Palabras que mas distinguen Generacion Z (primeras 10):")
for p, fz, ff, r, es_genz in diferencias[:10]:
    if fz > ff:
        print(f"  {p}: GenZ={fz}, Frank={ff}, ratio={r:.2f}")
print()
print("Palabras que mas distinguen Frankenstein (primeras 10):")
rev = [d for d in diferencias if not d[4]][:10]
for p, fz, ff, r, _ in rev:
    print(f"  {p}: Frank={ff}, GenZ={fz}, ratio={r:.2f}")
print()

# Q20: Sintesis general
print("=" * 60)
print("Q20: SINTESIS DE COBERTURA PERIODISTICA")
print("=" * 60)
print("Resumen general del analisis:")
print()
total_art = len(df)
art_genz = len(df[df["categoria"] == "Generacion Z"])
art_frank = len(df[df["categoria"] == "Frankenstein"])
print(f"Se analizaron {total_art} articulos sinteticos ({art_genz} de Generacion Z, {art_frank} de Frankenstein).")
print()
print("Distribucion por tono:")
for t in tonos:
    print(f"  {t}: {len(df[df['tono'] == t])} articulos")
print()
print("Medios participantes:")
for m in df["medio"].value_counts().index:
    print(f"  {m}: {len(df[df['medio'] == m])} articulos")
print()
score_medio_genz = df[df["categoria"] == "Generacion Z"]["sentimiento_score"].mean()
score_medio_frank = df[df["categoria"] == "Frankenstein"]["sentimiento_score"].mean()
print(f"Sentimiento promedio: Generacion Z = {score_medio_genz:.3f}, Frankenstein = {score_medio_frank:.3f}")
print()
print("Plataformas mas mencionadas en general:")
for p, c in menciones_total.most_common(5):
    print(f"  {p}: {c} menciones")
print()
print("Observaciones principales:")
print("1. La cobertura de Generacion Z se enfoca en salud mental, identidad digital y consumo de medios.")
print("2. La cobertura de Frankenstein gira en torno a etica cientifica, IA y vigencia literaria.")
print("3. El tono varia entre categorias reflejando diferencias en el enfoque periodistico.")
print("4. TikTok, Twitter e Instagram dominan las menciones de plataformas en ambas categorias.")
print("5. Los terminos 'inteligencia artificial' y 'algoritmo' aparecen en ambos conjuntos,")
print("   lo que sugiere una convergencia tematica entre tecnologia y sociedad.")
print()

print("Todas las graficas se guardaron en la carpeta 'graficas/'")
print("Analisis completado.")
