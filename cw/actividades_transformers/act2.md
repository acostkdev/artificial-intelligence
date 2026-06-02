Actividad manual Parte 2 — Entender Transformers
Objetivo
Que los alumnos profundicen en el mecanismo de atencion expandiendo la matriz completa, aplicando softmax numerico, mezclando vectores, y explorando mascaras, atencion cruzada, MLM, capas apiladas y escalamiento.

Nivel
Intermedio. Asume que se realizo la Parte I o que se conoce la idea de atencion como reparto de porcentajes.

Materiales
Hojas de esta actividad (una por alumno o por pareja).
Lapiz, colores opcionales.
Calculadora con exponencial (actividad 7) o tabla de apoyo del docente.
Regla o cuadricula para dibujar matrices.

Conceptos que se refuerzan
Actividad	Concepto Transformer
6	Matriz completa N x N: cada token tiene su propia fila de atencion
7	Softmax numerico exacto: de puntuaciones crudas a probabilidades
8	Mezcla ponderada de vectores: salida de atencion = suma alpha_i * V_i
9	Mascara de padding: ignorar relleno en lotes
10	Matriz rectangular: cross-attention (decoder hacia encoder)
11	Prediccion con hueco: Masked Language Modeling (BERT)
12	Dos rondas de atencion: capas apiladas refinan representacion
13	Conteo de enlaces: paralelismo vs camino secuencial (RNN)
14	Escalar por sqrt(d_k): evitar saturacion del softmax

---

### Solucion Actividad 6 — Matriz de atencion completa

**Frase de trabajo: LA NINA PEQUENA COME FRUTA** (5 palabras)

**Paso 1 — Puntuaciones 0-10 para cada fila**

Asignamos puntuaciones desde cada palabra hacia todas las demas, incluyendose a si misma:

| Desde \ Hacia | LA | NINA | PEQUENA | COME | FRUTA |
|---|---|---|---|---|---|
| LA | 6 | 9 | 7 | 2 | 1 |
| NINA | 8 | 7 | 9 | 6 | 3 |
| PEQUENA | 5 | 9 | 8 | 4 | 2 |
| COME | 2 | 8 | 4 | 5 | 9 |
| FRUTA | 1 | 4 | 2 | 8 | 7 |

**Justificacion de las puntuaciones:**

- **LA**: El articulo mira fuertemente a NINA (9) porque es el nucleo del sintagma nominal, y a PEQUENA (7) porque es el modificador directo. Baja puntuacion a COME y FRUTA porque un articulo tiene poca relacion directa con el verbo y el objeto.
- **NINA**: Como sujeto, mira a PEQUENA (9) porque la modifica, y tambien a LA (8) por ser su determinante. Mantiene atencion moderada a COME (6) porque es el verbo que ejecuta.
- **PEQUENA**: Similar a NINA: mira mucho a NINA (9) porque la califica, y a LA (5) por el articulo. Atiende moderadamente a COME (4).
- **COME**: Desde el verbo, mira fuerte a FRUTA (9) porque es el objeto directo (que se come), y a NINA (8) porque es el sujeto (quien come). Bajo interes en LA y PEQUENA.
- **FRUTA**: Desde el objeto, mira fuerte a COME (8) porque es el verbo que lo afecta. Moderada atencion a NINA (4) por ser la poseedora de la accion.

**Paso 2 — Normalizar cada fila (convertir a %).**
Sumas por fila: LA = 25, NINA = 33, PEQUENA = 28, COME = 28, FRUTA = 22.

| Desde \ Hacia | LA | NINA | PEQUENA | COME | FRUTA |
|---|---|---|---|---|---|
| LA | 6/25=24% | 9/25=36% | 7/25=28% | 2/25=8% | 1/25=4% |
| NINA | 8/33=24% | 7/33=21% | 9/33=27% | 6/33=18% | 3/33=9% |
| PEQUENA | 5/28=18% | 9/28=32% | 8/28=29% | 4/28=14% | 2/28=7% |
| COME | 2/28=7% | 8/28=29% | 4/28=14% | 5/28=18% | 9/28=32% |
| FRUTA | 1/22=5% | 4/22=18% | 2/22=9% | 8/22=36% | 7/22=32% |

Verificacion: cada fila suma 100% (con redondeo permitido ±1%).

**Paso 3 — Colorear la celda mas alta de cada fila (en rojo):**

| Desde \ Hacia | LA | NINA | PEQUENA | COME | FRUTA |
|---|---|---|---|---|---|
| LA | 24% | **36%** | 28% | 8% | 4% |
| NINA | 24% | 21% | **27%** | 18% | 9% |
| PEQUENA | 18% | **32%** | 29% | 14% | 7% |
| COME | 7% | 29% | 14% | 18% | **32%** |
| FRUTA | 5% | 18% | 9% | **36%** | 32% |

**Patron:** Cada palabra tiene su propio patron. Las palabras funcionales (LA) se enfocan en una palabra especifica, mientras que las palabras de contenido (NINA, COME) distribuyen la atencion en varias. El patron de NINA y PEQUENA se parece porque son adyacentes y se modifican entre si.

**Respuestas a las preguntas:**

1. **Fila de COME vs FRUTA:** Son distintas. COME mira principalmente a FRUTA (32%) y NINA (29%) porque necesita saber que se come y quien come. FRUTA mira principalmente a COME (36%) porque necesita saber que verbo la afecta. No son iguales porque cada palabra tiene un rol sintactico distinto.

2. **Fila con atencion pareja (~20% cada una):** Ninguna fila reparte exactamente parejo, pero NINA es la mas distribuida porque como sujeto necesita equilibrar atencion a su determinante (LA 24%), su modificador (PEQUENA 27%), y el verbo (COME 18%). En general, las palabras de contenido tienden a distribuir mas que las funcionales.

3. **Tabla de 100 palabras:** tendria 100 x 100 = 10,000 celdas. Textos largos cuestan mas memoria porque la matriz de atencion crece cuadraticamente O(n^2) con la longitud de la secuencia, no linealmente.

---

### Solucion Actividad 7 — Softmax a mano

**Situacion:** La palabra PEQUENA obtuvo estos puntajes brutos:

| Hacia | NINA | PEQUENA | COME | FRUTA |
|---|---|---|---|---|
| Puntaje s_i | 3.0 | 0.5 | 0.2 | 1.0 |

**Paso 1 — Exponencial:**

e^3.0 = 20.09, e^0.5 = 1.65, e^0.2 = 1.22, e^1.0 = 2.72.

Suma = 20.09 + 1.65 + 1.22 + 2.72 = **25.68**

**Paso 2 — Dividir entre la suma:**

| Palabra | e^{s_i} | / 25.68 | % |
|---|---|---|---|
| NINA | 20.09 | 20.09/25.68 | **78%** |
| PEQUENA | 1.65 | 1.65/25.68 | **6%** |
| COME | 1.22 | 1.22/25.68 | **5%** |
| FRUTA | 2.72 | 2.72/25.68 | **11%** |
| Suma | 25.68 | | **100%** |

**Paso 3 — Interpretacion:**

- Aunque COME tenia 0.2 y PEQUENA 0.5 (no tan diferentes), la palabra NINA domina con 78% porque su puntaje de 3.0, al elevarse a e^3.0, produce 20.09 que es desproporcionadamente mayor al resto. Esto es la propiedad clave de softmax: amplifica las diferencias.
- Si el puntaje de NINA fuera 10 en lugar de 3, la exponencial daria e^10 = 22026, haciendo que softmax diera practicamente 100% a NINA. Por eso en los Transformers se escala dividiendo entre sqrt(d_k): para que los puntajes no crezcan demasiado con dimensions grandes y el softmax no se sature.

**Respuesta a la pregunta:**

No basta con dividir entre la suma sin exponencial porque:
- Con puntajes negativos, la division simple podria dar porcentajes negativos, lo cual no tiene sentido como pesos de atencion.
- Softmax amplifica las diferencias: puntajes 3, 2, 1 con division simple dan 50%, 33%, 17%. Con softmax dan proporciones mas extremas porque la exponencial hace que los valores grandes dominen mucho mas.
- La exponencial garantiza que todos los pesos sean positivos, lo cual es necesario para interpretarlos como probabilidades.

---

### Solucion Actividad 8 — Mezcla de vectores (Values)

**Datos:** Vectores V = (x, y) de cada palabra:

| Palabra | V = (x, y) |
|---|---|
| LA | (1, 1) |
| NINA | (4, 5) |
| PEQUENA | (3, 4) |
| COME | (5, 1) |
| FRUTA | (6, 3) |

**Porcentajes desde COME** (tomados de la actividad 6):

| Hacia | % | Decimal |
|---|---|---|
| LA | 7% | 0.07 |
| NINA | 29% | 0.29 |
| PEQUENA | 14% | 0.14 |
| COME | 18% | 0.18 |
| FRUTA | 32% | 0.32 |

**Paso 1 — Convertir % a decimales (ya hecho arriba).**

**Paso 2 — Multiplicar cada vector por su peso y sumar:**

- 0.07 * (1, 1) = (0.07, 0.07)
- 0.29 * (4, 5) = (1.16, 1.45)
- 0.14 * (3, 4) = (0.42, 0.56)
- 0.18 * (5, 1) = (0.90, 0.18)
- 0.32 * (6, 3) = (1.92, 0.96)

**Suma total (salida de atencion para COME):**
x = 0.07 + 1.16 + 0.42 + 0.90 + 1.92 = **4.47**
y = 0.07 + 1.45 + 0.56 + 0.18 + 0.96 = **3.22**

**Resultado: salida = (4.47, 3.22)**

**Paso 3 — Interpretacion en el plano:**

- Los puntos originales: LA(1,1), NINA(4,5), PEQUENA(3,4), COME(5,1), FRUTA(6,3).
- La salida (4.47, 3.22) esta entre NINA (4,5) y FRUTA (6,3), mas cerca de NINA.
- Esto tiene sentido: despues de atender, COME "entiende" que NINA es quien come (sujeto) y FRUTA es lo que se come (objeto). El vector resultante es una mezcla contextualizada que incorpora informacion de las palabras mas relevantes.

**Conexion con el Transformer:** En un Transformer real los vectores V tienen cientos de dimensiones (ej. 768 en BERT-base) pero la operacion es la misma: promedio ponderado. La diferencia es que en vez de 2 coordenadas se manejan vectores de alta dimension, y los pesos de atencion se aprenden durante el entrenamiento.

---

### Solucion Actividad 9 — Mascara de padding

**Lote de dos frases (maximo 5 posiciones):**

Frase 1: EL   GATO   COME   —   —
Frase 2: LA   NINA   PEQUENA   COME   FRUTA
(— = PAD, relleno)

**Paso 1 — Matriz 5x5 solo para Frase 1:**

Marcamos con **P** las filas y columnas que corresponden a tokens PAD:

| Desde \ Hacia | EL | GATO | COME | PAD | PAD |
|---|---|---|---|---|---|
| **EL** | | | | P | P |
| **GATO** | | | | P | P |
| **COME** | | | | P | P |
| **PAD** | P | P | P | P | P |
| **PAD** | P | P | P | P | P |

**Paso 2 — Regla:** Las palabras reales no pueden prestar atencion a PAD. Se tachan esas celdas. Las filas de PAD no tienen sentido porque el token de relleno no deberia participar en el calculo (se enmascaran completamente, aunque en la practica tambien se ignoran).

Matriz real util (solo palabras reales atienden a palabras reales):

| Desde \ Hacia | EL | GATO | COME |
|---|---|---|---|
| **EL** | atencion normal | atencion normal | atencion normal |
| **GATO** | atencion normal | atencion normal | atencion normal |
| **COME** | atencion normal | atencion normal | atencion normal |

**Paso 3 — Preguntas:**

**Frase 2 no necesita tantas celdas tachadas** porque todas sus 5 posiciones son palabras reales, no tiene PAD. Solo si hubiera frases mas cortas en el lote necesitarian mascara.

**Si el modelo atendiera mucho a PAD:** aprenderia patrones falsos. El PAD es siempre el mismo token sin significado, y si el modelo le asigna atencion, estaria aprendiendo relaciones espurias (ej. que "GATO" se relaciona con un token vacio). Ademas, como las frases tienen longitudes distintas, la posicion del PAD varia, introduciendo ruido inconsistente en el entrenamiento.

---

### Solucion Actividad 10 — Atencion cruzada (decoder mirando encoder)

**Encoder (espanol, ya leido entero):** YO   QUIERO   CAFE

**Decoder (ingles, generando palabra a palabra):** I   WANT   ___  (aun no escribio COFFEE)

**Paso 1 — Matriz rectangular 3x3:**

Puntuamos la fila del hueco (Palabra 3, la proxima a generar en ingles):

| Desde (ingles) \ Espanol | YO | QUIERO | CAFE |
|---|---|---|---|
| Palabra 3 (por escribir) | 2 | 3 | 10 |

**Justificacion:** La palabra que falta es "COFFEE". En espanol, la palabra que significa "coffee" es CAFE. Por lo tanto, la atencion deberia ser casi total hacia CAFE. Le sigue QUIERO (3) porque da contexto de que se quiere algo, y YO (2) porque es el sujeto.

**Conversion a % (division simple para este ejercicio):** Suma = 15.

| Hacia | Puntaje | % |
|---|---|---|
| YO | 2 | 13% |
| QUIERO | 3 | 20% |
| CAFE | 10 | **67%** |

**Respuestas:**

1. **CAFE deberia ganar por mucho (67%).** Porque es la palabra espanola que corresponde directamente a "COFFEE", que es lo que el decoder necesita generar.

2. **La fila de I (primera palabra generada) probablemente miraria mucho a YO.** Tiene sentido porque "I" y "YO" son la misma palabra en distinto idioma (la primera persona del singular). En traduccion, los pronombres personales suelen alinearse fuertemente.

3. **Diferencia clave entre self-attention y cross-attention:** En self-attention, las columnas y las filas pertenecen al mismo idioma/misma secuencia. Aqui en cross-attention, las filas son del decoder (ingles, generandose) y las columnas del encoder (espanol, ya completa). Esto permite que cada palabra generada en ingles consulte la informacion relevante de toda la frase en espanol, sin estar limitada al orden de generacion.

---

### Solucion Actividad 11 — Adivinar la palabra tapada (BERT / MLM)

**Frase con hueco:** EL   GATO   ___   PESCADO

**Candidatos:** COME, DUERME, VERDE, RAPIDO

**Paso 2 — Puntuacion 0-10 segun compatibilidad con EL, GATO, PESCADO:**

| Candidato | Puntaje | Razon |
|---|---|---|
| COME | 10 | Tiene todo el sentido: el gato come pescado (sujeto-verbo-objeto). |
| DUERME | 6 | Podria tener sentido (el gato duerme... pero no con pescado). |
| VERDE | 2 | Un gato verde no es imposible pero es extrano, y menos con pescado. |
| RAPIDO | 3 | "El gato rapido pescado" no tiene coherencia gramatical directa. |

**Paso 3 — Conversion a % (softmax aproximado con e^x, usando e^10 ≈ 22026, e^6≈403, e^2≈7.4, e^3≈20.1):**

Suma exponencial = 22026 + 403 + 7.4 + 20.1 = **22456.5**

| Candidato | e^{s_i} | % |
|---|---|---|
| COME | 22026 | **98.1%** |
| DUERME | 403 | **1.8%** |
| VERDE | 7.4 | ~0.03% |
| RAPIDO | 20.1 | ~0.09% |

**Paso 4 — Reflexion:**

1. **Por que COME supera a VERDE:** COME completa la estructura sujeto-verbo-objeto "EL GATO COME PESCADO", que es una oracion gramatical y semanticamente coherente. VERDE es un adjetivo que modificaria a GATO, pero entonces "EL GATO VERDE PESCADO" no tiene verbo y "PESCADO" queda sin funcion gramatical clara.

2. **DUERME podria tener algo de sentido** si la frase fuera "EL GATO DUERME PESCADO"? No realmente, porque "dormir" es intransitivo y no toma objeto directo. Sin embargo, un hablante podria interpretarlo forzadamente como "el gato duerme junto al pescado". La atencion entre GATO y DUERME seria alta (sujeto-verbo), pero entre DUERME y PESCADO seria baja porque no hay relacion directa.

3. **BERT necesita ver PESCADO (a la derecha del hueco)** porque es un modelo bidireccional: utiliza contexto completo, no solo el izquierdo. Para predecir correctamente la palabra tapada, BERT considera tanto las palabras anteriores (EL, GATO) como las posteriores (PESCADO). En este caso, ver PESCADO es crucial porque solo con "EL GATO ___" podrian caber muchas opciones (COME, DUERME, CORRE, etc.), pero al ver PESCADO se reduce drasticamente a verbos compatibles con comer pescado.

---

### Solucion Actividad 12 — Dos capas de atencion

**Perfiles iniciales y tras capa 1:**

| Palabra | Perfil inicial | Tras capa 1 |
|---|---|---|
| LA | 1 | 2 |
| NINA | 4 | 6 |
| PEQUENA | 3 | 5 |
| COME | 5 | **7** |
| FRUTA | 6 | **8** |

**Paso 1 — Segunda ronda desde FRUTA:**

Usamos los perfiles tras capa 1 (no los iniciales). FRUTA (perfil 8) ahora debe repartir atencion hacia las demas palabras con sus nuevos perfiles.

Puntuaciones 0-10 desde FRUTA en capa 2:

| Desde FRUTA | LA | NINA | PEQUENA | COME | FRUTA |
|---|---|---|---|---|---|
| Puntaje 0-10 | 1 | 5 | 3 | 9 | 6 |

**Justificacion:** FRUTA ya "sabe" tras la capa 1 que COME tiene perfil 7 (mezclado con informacion del sujeto y objeto). Ahora en la capa 2, FRUTA puede mirar a COME con puntaje alto (9) porque la conexion verbo-objeto se reforzo. Tambien mira a NINA (5) porque es el sujeto间接o (quien come la fruta). LA sigue siendo poco relevante (1).

Conversion a % (suma = 24):

| Hacia | Puntaje | % |
|---|---|---|
| LA | 1 | 4% |
| NINA | 5 | 21% |
| PEQUENA | 3 | 13% |
| COME | 9 | **38%** |
| FRUTA | 6 | 25% |

**Paso 2 — Conclusion:**

"En la segunda capa, FRUTA ya 'sabe' que COME tiene perfil 7 porque la primera capa conecto verbo-objeto. Por eso en la segunda capa FRUTA puede atender mas fuertemente a COME (38%) que en la primera capa. La informacion se refina: cada capa construye sobre representaciones ya mezcladas, permitiendo captar relaciones mas indirectas o abstractas."

**Nota:** En un Transformer real no solo hay atencion apilada, sino tambien Feed-Forward Networks y LayerNorm entre capas. Aqui simplificamos a solo la atencion para entender la idea central: capas profundas ven relaciones que las superficiales no pueden ver porque trabajan sobre informacion ya contextualizada.

---

### Solucion Actividad 13 — RNN vs Transformer: contar conexiones

**Dibujo: A — B — C — D — E (5 nodos en linea)**

**Modo RNN (solo vecino anterior):**
Cada letra solo recibe mensaje directo de su anterior:
A -> B -> C -> D -> E
Enlaces para llegar de A a E: **4** (A→B, B→C, C→D, D→E).

La informacion viaja paso a paso, como un "telefono descompuesto": A influye en B, B influye en C, etc.

**Modo atencion (1 capa, todos miran todos):**
Cada letra puede mirar a las 5: 5 x 5 = **25** conexiones potenciales (incluye mirarse a si misma).

A puede mirar directamente a E en un solo paso.

**Respuestas:**

1. **Saltos de A a E:** En RNN se necesitan **4 saltos** (A→B→C→D→E). En una capa de atencion solo **1 salto** (A→E directamente porque en la matriz 5x5 la celda (A, E) existe y puede tener un peso alto).

2. **Crecimiento con 100 palabras:** RNN tendria 100 enlaces secuenciales (lineal, O(n)). Atencion tendria 100^2 = 10,000 celdas (cuadratico, O(n^2)). La atencion crece mucho mas rapido en numero de conexiones.

3. **Por que usamos Transformers y no solo RNN a pesar del costo cuadratico:**
   - **Paralelismo:** En atencion, todas las palabras pueden procesarse simultaneamente. En RNN, cada paso depende del anterior, forzando procesamiento secuencial. Esto hace que los Transformers sean mucho mas rapidos de entrenar en hardware moderno (GPUs).
   - **Calidad de dependencias:** El Transformer conecta palabras lejanas en un solo paso, sin tener que "recordar" a traves de multiples pasos como la RNN. Esto evita el problema del desvanecimiento del gradiente para relaciones de largo alcance.
   - **Trade-off:** El costo cuadratico O(n^2) en memoria es la desventaja principal, y por eso se investigan alternativas como Longformer, Linformer, etc. Pero para la mayoria de casos practicos, la ganancia en calidad y velocidad de entrenamiento compensa el costo.

---

### Solucion Actividad 14 — Escalar por sqrt(d_k)

**Ejercicio numerico:**

Dos palabras con vectores Q y K de dimension d_k = 4.
- Q·K de dos palabras principales = 8 (puntaje alto)
- Resto de palabras: puntaje 2 (tres palabras con puntaje 2)

**Sin escalar:** puntajes = [8, 2, 2, 2] (4 palabras)

Softmax aproximado con exponencial:
e^8 = 2981.0, e^2 = 7.4

Suma = 2981.0 + 7.4 + 7.4 + 7.4 = **3003.2**

| Palabra | e^{s_i} | % |
|---|---|---|
| Palabra A (puntaje 8) | 2981.0 | **99.3%** |
| Resto (puntaje 2 c/u) | 7.4 | ~0.25% c/u |

La palabra con puntaje 8 acapara practicamente toda la atencion (99.3%). Las demas son invisibles.

**Con escalar: dividir entre sqrt(d_k) = sqrt(4) = 2**

Puntajes escalados: [8/2, 2/2, 2/2, 2/2] = [4, 1, 1, 1]

Softmax aproximado:
e^4 = 54.6, e^1 = 2.72

Suma = 54.6 + 2.72 + 2.72 + 2.72 = **62.76**

| Palabra | e^{s_i} | % |
|---|---|---|
| Palabra A (puntaje 4) | 54.6 | **87.0%** |
| Resto (puntaje 1 c/u) | 2.72 | ~4.3% c/u |

**Comparacion:**

| | Sin escalar | Con escalar |
|---|---|---|
| Palabra ganadora | 99.3% | 87.0% |
| Resto (cada una) | ~0.25% | ~4.3% |

**Conclusion:** La palabra ganadora sigue ganando en ambos casos, pero con el escalamiento su porcentaje baja del 99.3% al 87.0%, permitiendo que las otras palabras tengan "voz" (4.3% cada una en vez de 0.25%). Esto evita la saturacion del softmax, donde una sola palabra se lleva casi toda la atencion y la informacion del resto se pierde.

**Frase para llevar:** "Dividir entre sqrt(d_k) es bajar el volumen antes de repartir atencion, para que varias palabras sigan teniendo voz."

Esto es especialmente importante cuando d_k es grande (ej. 64 o 128 en Transformers reales), porque los productos punto Q·K crecen proporcionalmente a sqrt(d_k), y sin escalar el softmax se saturaria.
