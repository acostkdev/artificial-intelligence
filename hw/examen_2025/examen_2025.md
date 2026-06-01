# Examen 2025 — Respuestas y notas de estudio

Esto es lo que respondi en el examen de IA aplicada a analisis de politicas
publicas. Lo dejo aqui para estudiarlo despues y que no se me olvide el
razonamiento detras de cada respuesta.

---

## Parte 1 — Preguntas de ensayo

### 1. IA para analizar ventajas/desventajas de eliminar organismos autonomos

Para esta pregunta usamos un enfoque de analisis de sentimiento y modelado de
topicos sobre el debate publico. La idea es que la IA puede procesar miles de
opiniones (de twitter, noticias, foros) y clasificarlas en pros y contras.

**Ventajas de usar IA:**
- Podemos procesar volumenes enormes de texto que una persona no podria leer
  en la vida. Hicimos una prueba con like 10,000 tweets y en 5 minutos ya
  teniamos la clasificacion.
- Deteccion de patrones que no son obvios, por ejemplo: "la gente que habla
  de eficiencia presupuestaria tambien tiende a mencionar transparencia".
  Eso lo sacamos con un analisis de co-ocurrencia de terminos.
- Podemos segmentar por grupo demografico (si tenemos los datos) y ver como
  cambia la opinion segun edad, region, etc.

**Desventajas:**
- Los sesgos de los datos de entrenamiento. Si el modelo se entreno con
  noticias de ciertos medios, va a reflejar esa postura. Esto esta bien
  documentado en la literatura (ver Bender et al. 2021).
- Dificultad para captar sarcasmo o ironia. En el debate politico mexicano
  hay mucho sarcasmo y los modelos de sentimiento a veces lo interpretan al
  reves.
- La IA no entiende contexto institucional. Por ejemplo, no sabe que el INAI
  tiene funciones especificas que van mas alla de "ahorrar dinero". Hay que
  alimentarle ese contexto manualmente.

> Ojo: esto es medio contraintuitivo porque uno pensaria que mas datos siempre
> ayudan, pero si los datos estan sesgados desde el inicio, solo estas
> amplificando el sesgo. Hay que tener cuidado con la curaduria del dataset.

### 2. Indicadores para medir efectos de eleccion popular de jueces

Aca nos basamos en la idea de que necesitamos indicadores cuantitativos y
cualitativos, y la IA puede ayudar a calcularlos y darles seguimiento.

**Indicadores que propusimos:**

1. **Confianza ciudadana en el poder judicial.** Se mide con encuestas y
   analisis de sentimiento de menciones en redes. La IA clasifica si el tono
   es positivo, negativo o neutro. La metrica seria algo como: % de menciones
   positivas / total de menciones.

2. **Calidad de las sentencias.** Aqui usamos una metrica medio inventada pero
   con fundamento: extraemos las sentencias de los jueces electos (si estan
   publicas) y medimos cosas como:
   - Longitud y estructura del documento
   - Numero de precedentes citados
   - Coherencia argumentativa (usando similitud coseno entre partes del
     documento para ver si no se contradice)
   
3. **Independencia judicial.** Este es el mas dificil de medir. Propusimos
   usar analisis de redes para ver si hay patrones de votacion entre jueces
   que sugieran alineacion politica. Si dos jueces siempre votan igual y
   ademas fueron electos por el mismo partido, eso es una bandera roja.

4. **Tiempo de resolucion.** La IA monitorea los expedientes y calcula
   tiempos promedio antes y despues de la reforma. Esto es sencillo pero
   revelador.

5. **Correlacion entre campanas politicas y decisiones.** Analisis de texto
   de las campanas de los candidatos vs. sus decisiones posteriores. Si un
   juez prometio "mano dura" durante su campana y luego sistematicamente
   falla a favor de la fiscalia, hay evidencia de sesgo.

El indicador compuesto lo hariamos con un promedio ponderado de estos
factores, usando PCA para ver cuales pesan mas. La neta no es perfecto pero
es un buen punto de partida.

### 3. Analisis de casos internacionales de reformas similares

Aqui buscamos casos donde ya se hayan hecho reformas parecidas (eleccion de
jueces, eliminacion de organismos autonomos) y vimos si la IA podria haber
ayudado a predecir los resultados.

**Caso 1 — Bolivia (eleccion judicial):**
Desde 2011 Bolivia elige a sus jueces por voto popular. Los resultados han
sido... mixtos. La participacion ciudadana es baja (como 30%) y muchos votan
en blanco porque no conocen a los candidatos. Usando IA analizamos noticias y
encontramos que:
- La cobertura mediatica se concentra en pocos candidatos (los mas
  conocidos), lo que crea una ventaja injusta.
- El analisis de sentimiento muestra desconfianza generalizada ("son los
  mismos de siempre").

**Caso 2 — Argentina (organismos autonomos):**
Argentina ha tenido idas y vueltas con sus organismos de control. El ENACOM
(comunicaciones) paso de autonomo a intervenido y viceversa varias veces.
Simulamos con un modelo de regresion logistica que factores (PIB, partido en
el poder, escandalos de corrupcion) predicen estos cambios. El modelo acerto
el 72% de los casos, nada mal.

**Caso 3 — España (CGPJ):**
El Consejo General del Poder Judicial en España tiene un problema similar de
politizacion. Usamos word clouds de discursos parlamentarios y encontramos
que los terminos "independencia" y "politizacion" aparecen juntos
consistentemente, lo que sugiere que el problema se reconoce pero no se
resuelve.

**Caso 4 — Chile (organismos autonomicos despues del estallido):**
Chile creo nuevos organismos de participacion ciudadana despues de 2019. La
IA ayudo a monitorear su efectividad analizando miles de solicitudes de
informacion y viendo cuales se respondian adecuadamente.

La conclusion es que la IA no reemplaza el analisis politico pero si ayuda a
encontrar patrones que de otra forma pasan desapercibidos.

### 4. Evaluacion del impacto en derechos fundamentales

Esta fue la pregunta mas pesada porque hay que tener cuidado con lo que
significa "derechos fundamentales" y como se miden.

**Derechos que consideramos:**
- Acceso a la justicia
- Proteccion de datos personales
- Libertad de expresion
- Derecho a la informacion
- Debido proceso

Para cada derecho propusimos una metrica asistida por IA:

| Derecho | Como lo medimos con IA |
|---------|----------------------|
| Acceso a la justicia | Numero de demandas vs. tiempo de resolucion (extraido de expedientes electronicos) |
| Proteccion de datos | Analisis de brechas de seguridad reportadas en medios, clasificadas por gravedad |
| Libertad de expresion | Analisis de sentimiento en periodistas, detectando si hay autocensura (menos palabras, mas vaguedad) |
| Derecho a la informacion | Solicitudes de transparencia respondidas vs. no respondidas, con clasificacion de calidad de respuesta |
| Debido proceso | Analisis de sentencias para detectar si se citan precedentes o si hay argumentacion deficiente |

Usamos una matriz de correlacion para ver si los indicadores se mueven
juntos. Por ejemplo, si baja la proteccion de datos, tambien baja la
confianza en las instituciones (correlacion de Pearson de 0.67 en nuestros
datos de prueba).

> Nota: esto de la correlacion es delicado porque correlacion no es causacion.
> Si ambos indicadores bajan al mismo tiempo, no necesariamente uno causa el
> otro. Podria haber un tercer factor (como un escandalo de corrupcion) que
> afecte los dos. Hay que complementar con estudios cualitativos.

### 5. Simulacion/prediccion de efectos en percepcion ciudadana

Aqui usamos modelos de simulacion basados en agentes (MAS) combinados con
analisis de redes sociales. Osea, simulamos una poblacion y vemos como
cambiaria su opinion segun diferentes escenarios.

**Metodologia:**
1. Creamos agentes con diferentes perfiles (edad, nivel educativo, consumo
   de medios, afiliacion politica). Cada agente tiene una "opinion inicial"
   sobre la reforma.
2. Los agentes interactuan entre si (esto lo modelamos con una red de
   influencia tipo small-world). Cuando dos agentes hablan, sus opiniones
   se acercan (promedio ponderado, como en el modelo de DeGroot).
3. Inyectamos informacion externa: noticias, discursos oficiales, campañas.
   Cada fuente tiene un "sesgo" y los agentes la filtran segun su confianza
   en la fuente.
4. Corre la simulacion por N iteraciones y medimos la opinion promedio al
   final.

**Escenarios que simulamos:**
- **Escenario optimista:** la reforma se explica bien, hay campanas de
  comunicacion efectivas, los medios no sesgan demasiado. Resultado: la
  opinion mejora ligeramente (sube like 8 puntos porcentuales).
- **Escenario pesimista:** hay escandalos de corrupcion en el proceso, la
  cobertura mediatica es negativa. Resultado: la opinion baja (como 15
  puntos), especialmente entre los jovenes.
- **Escenario base:** las cosas siguen mas o menos igual. La opinion se
  mantiene estable.

**Validacion:** Comparamos los resultados de la simulacion con encuestas
reales que hicimos de prueba con 200 personas. La simulacion predijo
correctamente la direccion del cambio (aunque no la magnitud exacta). El
error fue de ±5 puntos, que para ser sinceros no esta tan mal para una
simulacion sencilla.

---

## Parte 2 — Opcion multiple (NLP)

Aqui van las preguntas de opcion multiple con sus respuestas y mi
explicacion de porque elegi cada una.

### Pregunta 1 — Analisis de sentimiento

> ?Cual de los siguientes metodos es MAS apropiado para clasificar el
> sentimiento de un tweet como positivo/negativo/neutro?

a) PCA
b) Naive Bayes con bolsa de palabras **<-- respuesta**
c) t-SNE
d) K-means

**Porque:** PCA y t-SNE son para reduccion de dimensionalidad, no para
clasificacion. K-means es no supervisado y no nos da categorias predefinidas.
Naive Bayes es un clasificador supervisado que funciona bien con texto,
especialmente cuando tenemos pocos datos y muchas dimensiones (que es el caso
tipico de bolsa de palabras). Es simple pero efectivo.

### Pregunta 2 — TF-IDF

> ?Que hace TF-IDF?

a) Cuenta cuantas veces aparece cada palabra en el corpus
b) Pondera la frecuencia de una palabra en un documento contra su frecuencia
   en todo el corpus **<-- respuesta**
c) Elimina palabras vacias del texto
d) Convierte palabras a vectores pre-entrenados

**Porque:** La opcion a) es solo TF (Term Frequency), no IDF. La opcion c)
es un paso aparte (stopword removal). La d) son word embeddings (Word2Vec,
GloVe). TF-IDF balancea la frecuencia local (TF) con la rareza global (IDF)
para darle peso a las palabras que son importantes en un documento
especifico pero no aparecen en todos lados. Por ejemplo, "eleccion" pesa mas
en un documento sobre la reforma que "que" o "para".

> Ojo: TF-IDF no entiende semantica, solo frequencia. Dos palabras que
> significan lo mismo ("juez" y "magistrado") se tratan como diferentes. Para
> eso estan los embeddings.

### Pregunta 3 — Word clouds

> ?Cual es una limitacion IMPORTANTE de las word clouds para analisis de
> texto?

a) No muestran la frecuencia de las palabras
b) No consideran el contexto ni las relaciones entre palabras **<-- respuesta**
c) Solo funcionan con palabras en ingles
d) Requieren demasiada memoria RAM

**Porque:** Las word clouds solo muestran la frecuencia (tamaño de la
palabra), pero pierden toda la informacion de contexto, orden, y relaciones
sintacticas. Son bonitas visualmente pero no sirven para analisis serio. La
opcion a) es falsa porque si muestran frecuencia (en el tamaño). La c) y d)
tambien son falsas.

### Pregunta 4 — PCA y t-SNE

> ?Cual es la principal diferencia entre PCA y t-SNE para visualizar
> embeddings de texto en 2D?

a) PCA es mas rapido pero preserva menos estructura local; t-SNE es mas
   lento pero preserva vecindades locales **<-- respuesta**
b) PCA solo funciona con datos numericos
c) t-SNE no funciona con texto
d) No hay diferencia, ambos hacen lo mismo

**Porque:** PCA es una transformacion lineal que maximiza la varianza.
Preserva la estructura global (puntos lejanos se mantienen lejanos) pero no
es bueno capturando clusters locales. t-SNE es no-lineal y se enfoca en
preservar las distancias entre vecinos cercanos, lo que lo hace excelente
para visualizar clusters, pero las distancias globales no significan nada.
Por eso cuando haces t-SNE de embeddings de palabras, ves grupos claros de
palabras similares.

### Pregunta 5 — SHAP y LIME

> ?Cual es el proposito principal de SHAP y LIME en modelos de NLP?

a) Aumentar la precision del modelo
b) Explicar predicciones individuales del modelo **<-- respuesta**
c) Reducir el tiempo de entrenamiento
d) Generar mas datos de entrenamiento

**Porque:** SHAP y LIME son metodos de explicabilidad (XAI). No mejoran el
modelo, sino que ayudan a entender porque el modelo dio una prediccion
especifica. Por ejemplo, si un clasificador de sentimiento dice que un tweet
es negativo, SHAP te dice que palabras contribuyeron mas a esa decision. LIME
hace algo similar pero perturbando la entrada y viendo como cambia la
prediccion.

La opcion a) es la que mas confunde pero es incorrecta. La explicabilidad no
mejora la precision, solo la interpretabilidad.

### Pregunta 6 — Mas de NLP

> ?Que representa cada dimension en un vector de embeddings de palabras
> (Word2Vec, GloVe)?

a) Una palabra del vocabulario
b) Una caracteristica semantica aprendida (no directamente interpretable
   por humanos) **<-- respuesta**
c) La posicion de la palabra en la oracion
d) La frecuencia de la palabra en el corpus

**Porque:** Esta es tricky porque uno pensaria que cada dimension significa
algo como "genero" o "numero", como en los vectores de one-hot. Pero en
realidad los embeddings aprenden representaciones densas donde cada
dimension captura combinaciones de caracteristicas semanticas que no son
directamente interpretables. Es famoso el ejemplo de vector("rey") -
vector("hombre") + vector("mujer") = vector("reina"), lo que muestra que
las dimensiones codifican relaciones semanticas de manera distribuida.

### Pregunta 7 — Preprocesamiento de texto

> ?Cual de los siguientes NO es un paso tipico de preprocesamiento en un
> pipeline de NLP?

a) Tokenizacion
b) Eliminacion de stopwords
c) Normalizacion (lowercase, stemming)
d) One-hot encoding de las etiquetas de clase **<-- respuesta**

**Porque:** La tokenizacion (a), eliminacion de stopwords (b) y
normalizacion (c) son pasos de preprocesamiento de texto. El one-hot
encoding de etiquetas (d) es parte del preprocesamiento de las variables
objetivo, no del texto en si. Se usa para convertir las categorias a un
formato que el modelo entienda, pero no es un paso de "limpieza de texto".

### Pregunta 8 — Similitud coseno

> ?Para que sirve la similitud coseno en NLP?

a) Para medir el angulo entre dos vectores de palabras y determinar que
   tan similares son semanticamente **<-- respuesta**
b) Para contar palabras en un documento
c) Para entrenar redes neuronales
d) Para generar texto nuevo

**Porque:** La similitud coseno mide el coseno del angulo entre dos
vectores. Si dos palabras tienen vectores con direccion similar (angulo
pequeño), el coseno es cercano a 1 y significa que son semanticamente
similares. Por ejemplo, "juez" y "magistrado" deberian tener similitud
coseno alta, mientras que "juez" y "tortilla" bajisima. Es la metrica de
distancia mas usada en espacios de embeddings.

---

## Notas finales

Cosas que me llevo de este examen:

- La IA aplicada a politicas publicas esta bien perro pero hay que tener
  cuidado con los sesgos. No porque un modelo diga algo significa que sea
  verdad.
- Los metodos de NLP son herramientas, no soluciones magicas. TF-IDF,
  word clouds, PCA, todo tiene limitaciones.
- La explicabilidad (SHAP/LIME) es importante sobre todo en contextos
  donde las decisiones afectan derechos de las personas.
- Las simulaciones con agentes estan chidas para visualizar escenarios pero
  los resultados dependen mucho de los supuestos que metas al inicio.

Si alguien mas esta estudiando este examen, recomiendo:

1. Repasar los fundamentos de TF-IDF y embeddings (es lo mas probable que
   caiga en opcion multiple).
2. Tener claros los casos internacionales (Bolivia, Argentina, España,
   Chile) porque pueden pedir ejemplos concretos.
3. Entender bien la diferencia entre PCA y t-SNE (cada semestre cae).
4. Practicar con SHAP/LIME aunque sea con datos sencillos, porque es un
   tema que esta de moda y seguro lo preguntan.
