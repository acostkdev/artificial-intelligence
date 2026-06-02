Actividad manual — Entender Transformers sin computadora
Objetivo
Que los alumnos experimenten la idea central del Transformer — cada palabra decide a cuáles otras presta atención — usando solo papel, lápiz y una calculadora (o porcentajes a ojo).

Nivel
Principiantes. No se requiere programación ni álgebra lineal.

Materiales
Hojas de esta actividad (una por alumno o por pareja).
Lápiz, colores opcionales.
Calculadora (opcional; se puede redondear a enteros).
Conceptos que se refuerzan
Concepto	Actividad donde aparece
Atención como “reparto de importancia”	1 y 2
Contexto cambia el significado	2
Máscara causal (no ver el futuro)	3
Varias “cabezas” / varios criterios	4
Leer todo vs escribir paso a paso	5
Hoja para alumnos — Actividad 1: La matriz de atención
Enunciado
Frase corta (4 palabras):

EL   GATO   COME   PESCADO
Imagina que eres la palabra COME y quieres entender qué haces en la oración. Puntúa del 0 al 10 cuánto te “importa” cada palabra para entenderte (10 = muchísimo).

 	EL	GATO	COME	PESCADO
Desde COME →	 	 	?	 
Completa la fila de COME (solo esa fila en esta actividad).

Paso 2 — Convertir a porcentajes (mini-softmax)
Suma tus cuatro puntuaciones: __
Divide cada puntuación entre la suma y multiplica por 100 (o usa la tabla de apoyo del docente).
Palabra	Puntuación	÷ Suma	× 100 ≈ %
EL	 	 	 
GATO	 	 	 
COME	 	 	 
PESCADO	 	 	 
Total	 	 	100 %
Paso 3 — Interpretación
Responde en una frase: ¿A quién le diste más atención? ¿Tiene sentido para el verbo “come”?

Pregunta de cierre
Si fueras la palabra PESCADO, ¿crees que tu fila de porcentajes sería igual? ¿Por qué sí o por no?

---

### Solucion Actividad 1

**Paso 1 — Puntuaciones desde COME (0–10):**

| Palabra | Puntuacion | Justificacion |
|---------|------------|---------------|
| EL | 2 | El articulo tiene poca relevancia para entender que significa "come" |
| GATO | 9 | Muy importante: es quien realiza la accion de comer (el sujeto) |
| COME | 4 | La palabra misma, atencion moderada a si misma (autoatencion) |
| PESCADO | 10 | Muy importante: es lo que se come, el objeto directo |

**Paso 2 — Porcentajes (mini-softmax):**

Suma total = 2 + 9 + 4 + 10 = **25**

| Palabra | Puntuacion | / Suma | x 100 | % |
|---------|-----------|--------|-------|---|
| EL | 2 | 2/25 | 0.08 | 8 % |
| GATO | 9 | 9/25 | 0.36 | 36 % |
| COME | 4 | 4/25 | 0.16 | 16 % |
| PESCADO | 10 | 10/25 | 0.40 | 40 % |
| Total | 25 | | | 100 % |

**Paso 3 — Interpretacion:** Le dimos mas atencion a PESCADO (40 %) porque es el objeto directo del verbo "come" — saber que se come es esencial para entender la accion. Le sigue GATO (36 %) por ser el sujeto. Ambas son las mas relevantes, lo cual tiene sentido.

**Pregunta de cierre:** No, no seria igual. Si fueramos PESCADO, probablemente le dariamos mas atencion a COME (el verbo que explica que le pasa al pescado) y a EL (el articulo que lo introduce). Cada palabra tiene su propio contexto de atencion segun su rol en la oracion.

---

Hoja para alumnos — Actividad 2: La palabra ambigua (dos contextos)
Enunciado
La palabra BANCO aparece en dos frases. En parejas, completen solo la fila de BANCO (puntuación 0–10 y luego porcentajes) en cada caso.

Frase A
FUIMOS   AL   BANCO   DEL   RIO
Frase B
FUIMOS   AL   BANCO   A   SACAR   DINERO
Preguntas
¿En cuál frase BANCO le da más puntos a “RIO” / “DEL”?
¿En cuál le da más a “DINERO” / “SACAR”?
Esto imita lo que hace un Transformer: la misma palabra cambia de vecinos importantes según la oración. Escríbanlo con sus palabras (3 líneas máximo).

---

### Solucion Actividad 2

**Frase A — FUIMOS AL BANCO DEL RIO**

Puntuaciones desde BANCO (0–10):

| Palabra | Punt. | / Suma | x 100 | % |
|---------|-------|--------|-------|---|
| FUIMOS | 3 | 3/28 | 0.107 | 10.7 % |
| AL | 2 | 2/28 | 0.071 | 7.1 % |
| BANCO | 5 | 5/28 | 0.179 | 17.9 % |
| DEL | 8 | 8/28 | 0.286 | 28.6 % |
| RIO | 10 | 10/28 | 0.357 | 35.7 % |
| Total | **28** | | | **100 %** |

**Frase B — FUIMOS AL BANCO A SACAR DINERO**

Puntuaciones desde BANCO (0–10):

| Palabra | Punt. | / Suma | x 100 | % |
|---------|-------|--------|-------|---|
| FUIMOS | 3 | 3/29 | 0.103 | 10.3 % |
| AL | 2 | 2/29 | 0.069 | 6.9 % |
| BANCO | 4 | 4/29 | 0.138 | 13.8 % |
| A | 1 | 1/29 | 0.034 | 3.4 % |
| SACAR | 9 | 9/29 | 0.310 | 31.0 % |
| DINERO | 10 | 10/29 | 0.345 | 34.5 % |
| Total | **29** | | | **100 %** |

- En la **frase A**, BANCO le da mas puntos a RIO (35.7 %) y DEL (28.6 %) porque el contexto es geografico (banco del rio).
- En la **frase B**, BANCO le da mas a DINERO (34.5 %) y SACAR (31.0 %) porque el contexto es financiero (banco a sacar dinero).

**Explicacion:** La misma palabra (BANCO) cambia radicalmente a cuales otras les presta atencion segun la oracion completa. En un Transformer, esto se logra porque los pesos de atencion se calculan con base en todo el contexto, no por palabra aislada. Asi el modelo puede distinguir significados distintos del mismo token segun el contexto — algo fundamental para entender el lenguaje.

---

Hoja para alumnos — Actividad 3: Máscara causal (no hacer trampa)
Situación
Un modelo que escribe la frase palabra por palabra (como ChatGPT) no puede ver palabras del futuro al generar la actual.

Orden de generación:

1º EL  →  2º GATO  →  3º COME  →  4º PESCADO
Instrucción
Dibujen una cuadrícula 4×4 (filas = palabra que “pregunta”, columnas = palabra a la que mira).

Marquen con ✓ donde SÍ se permite mirar.
Dejen en blanco (o ✗) donde NO se permite (futuro).
Ejemplo de la fila cuando ya se escribió hasta “GATO” y ahora se genera la 3.ª palabra: solo puede mirar EL y GATO.

Preguntas
¿Cuántos ✓ hay en la fila de la última palabra (PESCADO)?
¿Cuántos ✓ hay en la fila de la primera palabra (EL)?
La forma de ✓ que queda (triángulo abajo) se llama máscara causal. ¿Por qué creen que es necesaria para escribir texto?

---

### Solucion Actividad 3

**Cuadricula 4x4 (mascara causal):**

| Fila: palabra que genera | EL | GATO | COME | PESCADO |
|--------------------------|----|------|------|---------|
| 1a: EL | ✓ | ✗ | ✗ | ✗ |
| 2a: GATO | ✓ | ✓ | ✗ | ✗ |
| 3a: COME | ✓ | ✓ | ✓ | ✗ |
| 4a: PESCADO | ✓ | ✓ | ✓ | ✓ |

**Respuestas:**
- Fila de la ultima palabra (PESCADO): **4 ✓** (puede mirar todas porque ya fueron generadas).
- Fila de la primera palabra (EL): **1 ✓** (solo puede mirarse a si misma, no hay palabras anteriores).

**La mascara causal forma un triangulo inferior de ✓.** Es necesaria porque al generar texto paso a paso (autoregresivamente), el modelo no debe tener acceso a palabras que aun no ha escrito. Si pudiera ver el futuro, "haria trampa": en lugar de aprender a predecir la siguiente palabra, simplemente copiaria la respuesta. Esto es esencial en modelos como ChatGPT que generan texto de izquierda a derecha.

---

Hoja para alumnos — Actividad 4: Varias cabezas (varios criterios)
En equipo
Usen la frase:

MARIA   NO   COMIO   PORQUE   ESTABA   ENFERMA
Cada persona del equipo es una cabeza de atención distinta y solo puntúa la fila de COMIO (0–10), con un criterio distinto:

Persona	Criterio (solo para COMIO)
A	¿Quién explica el porqué? (causa)
B	¿Quién es el sujeto de la acción?
C	¿Quién está junto al verbo (vecinos inmediatos)?
Después de puntuar
Cada uno convierte su fila a porcentajes (suma 100 %).
Comparen: ¿las tres filas son iguales?
En un Transformer real, esas “vistas” se juntan. ¿Qué ventaja tendría ver la frase desde tres criterios y no solo uno?

---

### Solucion Actividad 4

**Frase: MARIA NO COMIO PORQUE ESTABA ENFERMA**

**Persona A — Criterio: causa (quien explica el porque)**

| Palabra | Punt. | / Suma | % |
|---------|-------|--------|---|
| MARIA | 2 | 2/37 | 5.4 % |
| NO | 3 | 3/37 | 8.1 % |
| COMIO | 5 | 5/37 | 13.5 % |
| PORQUE | 10 | 10/37 | 27.0 % |
| ESTABA | 8 | 8/37 | 21.6 % |
| ENFERMA | 9 | 9/37 | 24.3 % |
| Total | **37** | | **100 %** |

**Persona B — Criterio: sujeto (quien ejecuta la accion)**

| Palabra | Punt. | / Suma | % |
|---------|-------|--------|---|
| MARIA | 10 | 10/27 | 37.0 % |
| NO | 6 | 6/27 | 22.2 % |
| COMIO | 5 | 5/27 | 18.5 % |
| PORQUE | 1 | 1/27 | 3.7 % |
| ESTABA | 2 | 2/27 | 7.4 % |
| ENFERMA | 3 | 3/27 | 11.1 % |
| Total | **27** | | **100 %** |

**Persona C — Criterio: vecinos inmediatos (palabras junto al verbo)**

| Palabra | Punt. | / Suma | % |
|---------|-------|--------|---|
| MARIA | 1 | 1/28 | 3.6 % |
| NO | 9 | 9/28 | 32.1 % |
| COMIO | 5 | 5/28 | 17.9 % |
| PORQUE | 8 | 8/28 | 28.6 % |
| ESTABA | 3 | 3/28 | 10.7 % |
| ENFERMA | 2 | 2/28 | 7.1 % |
| Total | **28** | | **100 %** |

**Comparacion:** Las tres filas son muy distintas entre si. La persona A concentra la atencion en PORQUE y ENFERMA (la causa). La persona B la concentra en MARIA (el sujeto). La persona C la distribuye entre NO y PORQUE (los vecinos inmediatos de COMIO).

**Ventaja de multiples cabezas:** Con un solo criterio solo capturariamos un tipo de relacion. Con varias cabezas, el modelo puede representar simultaneamente relaciones de causa, sujeto-verbo, proximidad sintactica, y muchas mas en paralelo. Esto es exactamente lo que hace el multi-head attention en los Transformers reales: cada cabeza aprende un "aspecto" distinto de la relacion entre palabras, y luego se concatenan para formar una representacion contextual rica.

---

Hoja para alumnos — Actividad 5: Encoder vs Decoder (role-play)
Sin papel — 10 minutos en plenario
Equipo “Encoder” (el que lee todo)
Reciben una frase secreta del docente (en sobre o pizarra oculta), por ejemplo: El examen fue el lunes.
Pueden leerla entera y anotar 3 “pistas” cortas (sustantivos, fecha, tema).
No dicen la frase literal; solo las pistas.
Equipo “Decoder” (el que escribe)
Deben reconstruir la frase palabra por palabra, preguntando de una en una.
Solo pueden usar lo que el Encoder les diga hasta ese momento (simulan no ver el futuro).
Después de cada palabra nueva, el Encoder puede dar una pista más (simula atención cruzada).
Debrief (preguntas del docente)
¿Qué fue más fácil: leer todo (encoder) o escribir de a poco (decoder)?
¿En qué momento el decoder “necesitó” mirar atrás?
¿Cómo se relaciona esto con traducir o con un chatbot?

---

### Solucion Actividad 5 (guia para el debrief)

**¿Que fue mas facil?** Leer todo (encoder) es mas facil porque tienes la informacion completa desde el inicio. Escribir de a poco (decoder) es mas dificil porque tienes que decidir la siguiente palabra sin saber lo que sigue, cargando con la incertidumbre de lo que falta.

**Cuando el decoder "necesita" mirar atras:** Tipicamente al empezar una frase nueva o al encontrar una palabra ambigua. Por ejemplo, si la frase es "El examen fue el lunes", al generar "el" el decoder podria necesitar confirmar si se refiere al articulo o a un dia de la semana. En ese momento las pistas del encoder (atencion cruzada) le ayudan a desambiguar.

**Relacion con traduccion y chatbots:** En traduccion, el encoder lee toda la frase origen (como el equipo Encoder) y el decoder genera palabra por palabra en el idioma destino consultando el contexto completo mediante atencion cruzada. En los chatbots como ChatGPT, solo hay decoder: genera texto paso a paso con mascara causal (no ve el futuro), pero tiene un contexto extenso de la conversacion previa.

---

Actividad extra, Tokenizar a mano
Enunciado
Los modelos no siempre cortan en palabras completas. Partan en “tokens” (pedazos útiles):

inteligencia
programación
des-apro-bado

---

### Solucion Actividad extra — Tokenizar a mano

Separacion en tokens (subpalabras) como lo haria un tokenizador BPE tipico:

| Palabra original | Tokens propuestos | Explicacion |
|------------------|-------------------|-------------|
| inteligencia | intelig-encia | "intelig" es la raiz, "encia" es el sufijo que forma sustantivos abstractos. Un tokenizador podria dividirlo asi porque "encia" aparece en muchas palabras (paciencia, conciencia). |
| programacion | progra-macion | "progra" abrevia programacion/programar, "macion" es un sufijo comun de accion y efecto. Tambien podria ser pro-grama-cion segun el vocabulario del modelo. |
| des-apro-bado | des-a-pro-ba-do | El enunciado ya lo sugiere como tres partes. "des-" es prefijo de negacion, "apro" podria ser raiz, "bado" es terminacion de participio. Un tokenizador real podria partirlo en aun mas piezas. |

**Nota:** En modelos como BERT o GPT, el tokenizador no siempre respeta las fronteras de las palabras. Por ejemplo, "inteligencia" podria ser un solo token si aparece frecuentemente en el corpus de entrenamiento, o partirse en varios si es menos comun. Esto lo decide el algoritmo BPE (Byte Pair Encoding) segun la frecuencia de los substrings en el corpus.
