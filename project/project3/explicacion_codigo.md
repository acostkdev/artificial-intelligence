# Explicacion del codigo — Actividad 3

Este documento explica linea por linea como funciona cada archivo del
proyecto. Lo escribimos en tono de bitacora para que quede claro que
hicimos y por que.

---

## 1. `dataset_c.txt` — El corpus de funciones C

Este archivo contiene 97 funciones escritas en C con un estilo
consistente. Decidimos ponerlo como archivo separado (no harcodeado en
el codigo) para poder modificarlo sin tener que tocar los scripts de
entrenamiento.

Cada funcion sigue el mismo formato:

```
// nombre: descripcion de lo que hace
tipo nombre(parametros) {
    ...
}
```

Escogimos snake_case y comentarios con `//` para que el modelo aprenda
un patron predecible. Incluimos funciones de distintas categorias:
aritmeticas, cadenas, arreglos, ordenamiento, matrices, estructuras,
memoria dinamica, listas enlazadas y arboles binarios. Esto le da
variedad al modelo para que no solo memorice un tipo de operacion.

El archivo se lee completo como una sola cadena de texto. No hay
separacion en lineas porque a nivel de caracteres eso no importa: el
modelo ve puras secuencias de caracteres, no "lineas".

---

## 2. `entrenar_rnn.py` — Entrenamiento de la RNN

### 2.1 Importaciones y semilla

```python
import numpy as np
import os
import pickle
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
```

Pusimos semilla 42 para que los resultados sean reproducibles. Sin
semilla, cada vez que entrenaras obtendrias pesos distintos y no
podrias comparar si una mejora funciona o fue casualidad.

### 2.2 Cargar el corpus

```python
ruta_dataset = os.path.join(os.path.dirname(__file__), "dataset_c.txt")
with open(ruta_dataset, "r", encoding="utf-8") as f:
    CORPUS = f.read()
```

Usamos `os.path.dirname(__file__)` para que funcione sin importar desde
donde se ejecute el script. Si pusieramos el path fijo ("dataset_c.txt")
y alguien corre el script desde otra carpeta, no lo encontraria.

### 2.3 Crear el vocabulario

```python
chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)
```

Esto es identico a como se hace en `apuntes_ia.md:6456-6458`.
`chars` es una lista con todos los caracteres distintos que aparecen
en el corpus (letras, numeros, espacios, signos). Son 75 caracteres
para nuestro dataset.

`stoi` mapea de caracter a indice (string to int). `itos` hace lo
contrario (int to string). Necesitamos ambos porque el modelo trabaja
con numeros, no con letras.

### 2.4 Guardar los mapeos

```python
with open(ruta_mappings, "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": VOCAB_SIZE}, f)
```

Estos mapeos los necesita el servidor (`servidor_api.py`) para poder
convertir el texto de entrada a indices y la salida del modelo de
vuelta a texto. Los guardamos con pickle para no tener que re-calcularlos.

### 2.5 Codificar el corpus completo

```python
SEQ = np.array(encode(CORPUS), dtype=np.int64)
```

Convertimos todo el corpus de texto a una secuencia de enteros de 64
bits. Usamos int64 porque TF a veces se queja con int32 en operaciones
de embedding.

### 2.6 Crear ventanas de entrenamiento

```python
block_size = 128
X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - block_size):
    X_rows.append(SEQ[i : i + block_size])
    Y_rows.append(SEQ[i + 1 : i + 1 + block_size])
```

Cada muestra de entrenamiento es un par (X, Y):
- X: ventana de 128 caracteres
- Y: la misma ventana pero desplazada 1 caracter a la derecha

O sea, para la posicion t en la ventana, la entrada es SEQ[t] y la
salida esperada es SEQ[t+1]. Esto es exactamente el enfoque de
"modelado de lenguaje causal" que se explica en `apuntes_ia.md:6646-6658`.

Escogimos 128 porque las funciones tienen nombres largos y queremos que
el modelo alcance a ver el nombre de la funcion cuando esta decidiendo
el cuerpo.

### 2.7 Construir el modelo

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(block_size,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, embed_dim),
    tf.keras.layers.SimpleRNN(hidden, activation="tanh", return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE)),
])
```

La arquitectura es la misma que en `apuntes_ia.md:6669-6676`:

1. **Input**: recibe vectores de 128 enteros (los indices de los
   caracteres). Usamos Input como capa separada para poder especificar
   el shape explicitamente.

2. **Embedding**: convierte cada indice entero en un vector denso de 64
   dimensiones (`apuntes_ia.md:9104-9150`). Sin embedding, cada caracter
   seria un one-hot de 75 posiciones (75=vocab_size), lo cual es muy
   disperso. El embedding aprende que caracteres son "similares" en
   contexto (ej: 'a' y 'b' suelen aparecer en contextos parecidos).

3. **SimpleRNN** con 128 unidades y tanh: es la RNN vanilla que procesa
   la secuencia de vectores de embedding. La funcion tanh comprime los
   valores entre -1 y 1 para evitar que crezcan sin control. `return_sequences=True`
   es necesario porque queremos la salida en CADA paso temporal, no solo
   en el ultimo (ver `apuntes_ia.md:6578-6586`).

4. **TimeDistributed(Dense(75))**: aplica la misma capa densa a cada
   uno de los 128 pasos temporales. Genera 75 logits (puntuaciones) por
   cada posicion, uno por cada caracter del vocabulario. El `TimeDistributed`
   asegura que los pesos de la densa se compartan entre todos los pasos
   (`apuntes_ia.md:6410-6416`).

En total son 39,179 parametros entrenables. Es un modelo pequeno
comparado con los millones de parametros de los Transformers, pero
suficiente para aprender los patrones basicos de las 97 funciones.

### 2.8 Compilar

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

- `Adam` con learning rate 0.001: el optimizador que usa momento
  adaptativo, combina ventajas de Momentum y RMSprop. Es el estandar
  para RNNs (`apuntes_ia.md:2216-2229`).
- `SparseCategoricalCrossentropy`: funcion de perdida para clasificacion
  con multiples clases. `from_logits=True` porque la salida del modelo
  son logits (sin softmax). Si pusieramos from_logits=False, TF esperaria
  probabilidades (con softmax) y el calculo seria numericamente menos
  estable.

### 2.9 Entrenar

```python
epochs = 150
batch_size = 32

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(...),
    tf.keras.callbacks.EarlyStopping(monitor="loss", patience=15, ...),
]

history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
```

- 150 epocas: cada epoca pasa todo el dataset una vez. Elegimos 150
  porque observamos que la perdida se estabiliza alrededor de la epoca
  120-130.
- batch_size=32: procesa 32 secuencias a la vez. Mas grande acelera el
  entrenamiento pero requiere mas memoria.
- EarlyStopping: si la perdida no mejora por 15 epocas seguidas, para
  el entrenamiento. Esto evita sobreentrenar si el modelo ya no aprende.
- ModelCheckpoint: guarda el modelo al final de cada epoca por si se
  interrumpe el entrenamiento (los borra al final).

### 2.10 Generacion (autocompletado)

```python
def muestrear(logits, temperatura=0.05, top_k=3):
    z = logits / max(temperatura, 1e-6)
    z = z - z.max()
    if top_k > 0:
        indices = np.argpartition(z, -top_k)[-top_k:]
        mask = np.full_like(z, -np.inf)
        mask[indices] = z[indices]
        z = mask
    e = np.exp(z)
    p = e / e.sum()
    return int(rng_completar.choice(len(p), p=p))
```

La funcion de muestreo es clave para generar texto de calidad:

1. **Temperatura**: divide los logits entre la temperatura. Temperatura
   baja (< 1) hace que las diferencias entre probabilidades se
   acentuen, dando mas peso a los caracteres mas probables. Temperatura
   alta (> 1) aplana la distribucion, dando mas variedad. Nosotros
   usamos 0.05, que es muy baja, para que el modelo sea bastante
   determinista y no invente caracteres raros.

2. **Top-k**: solo considera los k caracteres mas probables y pone
   -infinito en los demas. Esto evita que el modelo elija caracteres
   con probabilidad muy baja que romperian la sintaxis de C. Con k=3,
   el modelo solo puede elegir entre los 3 caracteres mas probables en
   cada paso.

3. **Softmax**: convierte los logits (despues de temperatura y top-k) a
   probabilidades que suman 1.

4. **Choice**: elige un indice segun esas probabilidades.

Esto es una mejora respecto a la generacion greedy (argmax) que
originalmente tenia el reporte. Con argmax, el modelo siempre elegia el
caracter mas probable y se atoraba en patrones repetitivos.

```python
def completar(prompt, max_nuevos=150):
    ids = encode(prompt)
    prompt_len = len(ids)
    for _ in range(max_nuevos):
        x = np.array(ids[-block_size:], dtype=np.int64)
        if x.shape[0] < block_size:
            pad = np.full(block_size - x.shape[0], ids[0], dtype=np.int64)
            x = np.concatenate([pad, x])
        x = x.reshape(1, block_size)
        logits = model(x, training=False).numpy()[0, -1, :]
        idx = muestrear(logits)
        ids.append(idx)
        if decode(ids[prompt_len:]).endswith("}\n\n"):
            break
    return decode(ids)
```

`completar` recibe un prompt (texto inicial) y genera caracteres uno
por uno:

1. Convierte el prompt a indices con `encode()`
2. Toma los ultimos `block_size` caracteres como contexto. Si el prompt
   es mas corto que block_size, rellena al inicio con el primer
   caracter (padding)
3. Pasa el contexto al modelo y obtiene logits del ultimo paso temporal
   (`[0, -1, :]` agarra la ultima posicion de la unica muestra del batch)
4. Aplica muestreo con temperatura+top-k
5. Agrega el caracter generado a la lista
6. Revisa si ya se completo la funcion (`}\n\n`)
7. Repite hasta completar o llegar al maximo

### 2.11 Guardar el modelo

```python
model.save(ruta_modelo)
```

Guarda el modelo completo (arquitectura + pesos + configuracion del
optimizador) en formato H5. Despues eliminamos los checkpoints
temporales para no llenar el disco.

---

## 3. `servidor_api.py` — API Flask

### 3.1 Cargar el modelo

```python
modelo_completo = tf.keras.models.load_model(ruta_modelo)
block_size = modelo_completo.input_shape[1]
```

Cargamos el modelo guardado por `entrenar_rnn.py`. Obtenemos el
`block_size` directamente del input del modelo para no tener que
hardcodearlo.

### 3.2 Optimizacion: modelo de inferencia separado

```python
embed_dim = modelo_completo.get_layer("embedding").output_dim
hidden = modelo_completo.get_layer("simple_rnn").units

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(block_size,), dtype=tf.int64),
    tf.keras.layers.Embedding(VOCAB_SIZE, embed_dim),
    tf.keras.layers.SimpleRNN(hidden, activation="tanh", return_sequences=False),
    tf.keras.layers.Dense(VOCAB_SIZE),
])
modelo.get_layer("embedding").set_weights(
    modelo_completo.get_layer("embedding").get_weights()
)
modelo.get_layer("simple_rnn").set_weights(
    modelo_completo.get_layer("simple_rnn").get_weights()
)
modelo.get_layer("dense").set_weights(
    modelo_completo.get_layer("time_distributed").layer.get_weights()
)
```

Aqui hicimos una optimizacion importante. El modelo original usa
`return_sequences=True` + `TimeDistributed(Dense)` porque durante el
entrenamiento necesitamos logits en CADA paso temporal. Pero para la
inferencia (generacion), solo nos interesa el ultimo paso, el que
predice el siguiente caracter.

Construimos un modelo de inferencia con `return_sequences=False` y
`Dense` directo (sin TimeDistributed). Esto evita que TensorFlow
calcule las capas densas para los 128 pasos temporales cuando solo
necesitamos el ultimo. Es un ahorro pequeno pero notable (~1.2 millones
de multiplicaciones de menos por inferencia).

Copiamos los pesos usando `get_layer()` por nombre en vez de por indice
numerico (`layers[2]`). Si la arquitectura cambia, es mas facil de
mantener porque los nombres son semanticos.

Nota: para el modelo de inferencia, la capa se llama `dense` directamente,
mientras que en el original esta envuelta en `TimeDistributed`, por eso
accedemos con `.layer` para sacar la capa interna.

### 3.3 Compilacion del forward pass

```python
@tf.function
def step(x):
    return modelo(x, training=False)
```

El decorador `@tf.function` compila la funcion en un grafo de
TensorFlow. Sin el, cada llamada a `modelo()` se ejecuta en modo eager
(Python puro), que es mas lento porque TF no puede optimizar el grafo
de computo. Con `@tf.function`, TF compila el grafo una vez y lo
reutiliza.

### 3.4 Funcion de generacion

```python
buf = np.empty(block_size, dtype=np.int64)

def completar(prompt, max_nuevos=350):
    ids = [stoi.get(c, 0) for c in prompt]
    prompt_len = len(ids)
    for i in range(max_nuevos):
        ctx = ids[-block_size:]
        start = block_size - len(ctx)
        buf[start:] = ctx
        if start > 0:
            buf[:start] = ids[0]
        logits = step(buf.reshape(1, block_size)).numpy()[0]
        idx = muestrear(logits)
        ids.append(idx)
        gen = "".join(itos[j] for j in ids[prompt_len:])
        if gen.endswith("}\n\n"):
            break
    return "".join(itos[i] for i in ids)
```

Es muy similar a la version en `entrenar_rnn.py` pero con una
optimizacion: usamos un buffer pre-asignado (`buf = np.empty(128)`) en
vez de crear un arreglo nuevo en cada iteracion. Esto evita allocaciones
de memoria innecesarias.

La generacion se detiene al encontrar `}\n\n` que es como terminan todas
las funciones del dataset. Si no encuentra ese patron en 350 caracteres,
se detiene forzosamente.

### 3.5 Endpoints HTTP

```python
@app.route("/autocompletar", methods=["POST"])
def autocompletar():
    datos = request.get_json(force=True)
    codigo = datos.get("codigo", "")
    max_nuevos = datos.get("max_tokens", 350)

    generado = completar(codigo, max_nuevos=max_nuevos)
    completado = generado[len(codigo):]

    return jsonify({
        "codigo_original": codigo,
        "completado": completado,
        "codigo_completo": generado,
    })
```

`force=True` permite recibir JSON aunque el Content-Type no sea
`application/json` (util para pruebas con curl).

El endpoint devuelve tres campos:
- `codigo_original`: el prompt que enviaste
- `completado`: solo la parte que genero el modelo (sin el prompt)
- `codigo_completo`: prompt + generado

Esto le da flexibilidad al cliente (la extension de VS Code) para
decidir como insertar el texto.

### 3.6 Health check

```python
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelo": "rnn_vanilla", "vocab_size": VOCAB_SIZE})
```

Endpoint simple para verificar que el servidor esta vivo y el modelo
cargo correctamente.

---

## 4. `extension_vscode/` — Extension de VS Code

### 4.1 `package.json`

```json
{
    "name": "autocompletado-rnn-c",
    "displayName": "Autocompletado RNN C",
    "engines": { "vscode": "^1.85.0" },
    "activationEvents": ["onLanguage:c"],
    "main": "./extension.js",
    "contributes": {
        "commands": [
            { "command": "autocompletado-rnn-c.autocompletar",
              "title": "Autocompletar con RNN" }
        ],
        "keybindings": [
            { "command": "...", "key": "ctrl+shift+l",
              "when": "editorLangId == c" }
        ]
    }
}
```

- `activationEvents: onLanguage:c`: la extension solo se activa cuando
  abres un archivo .c. Esto evita que consuma recursos en otros tipos
  de archivo.
- `keybindings: ctrl+shift+l`: el atajo de teclado, solo funciona en
  archivos C.

### 4.2 `extension.js`

```javascript
let comando = vscode.commands.registerCommand(
    "autocompletado-rnn-c.autocompletar",
    async function () {
        let editor = vscode.window.activeTextEditor;
        let codigo = documento.getText(rango);
        ...
        let respuesta = await fetch("http://127.0.0.1:5000/autocompletar", {
            method: "POST",
            body: JSON.stringify({ codigo: codigo, max_tokens: 200 }),
        });
        let datos = await respuesta.json();
        editor.edit(function (edicion) {
            edicion.insert(posicion, datos.completado);
        });
    }
);
```

La extension:
1. Toma el texto del archivo activo hasta la posicion del cursor
2. Lo envia al endpoint `/autocompletar` de nuestra API
3. Inserta el texto generado justo donde esta el cursor

Usa `fetch` nativo de Node.js 18+ (incluido en VS Code).

La extension es minima pero funcional. No maneja casos como
seleccion multiple porque el objetivo es mostrar el pipeline completo:
entrenamiento -> API -> editor.

---

## 5. `example.c` — Ejemplo de uso

Este archivo contiene 3 funciones de ejemplo (sumar, factorial, invertir
cadena). Sirve para probar el autocompletado: abres el archivo en VS
Code, pones el cursor al final de un comentario y presionas Ctrl+Shift+L,
o pruebas con curl pasando el comentario como prompt.

---

## Resumen del flujo completo

```
dataset_c.txt  -->  entrenar_rnn.py  -->  modelo_rnn.h5
                                              |
                                              v
                                     servidor_api.py
                                              |
                                    +---------+---------+
                                    |                   |
                               curl / HTTP        extension_vscode
                              (terminal)          (VS Code)
```

1. Escribimos 97 funciones C en `dataset_c.txt`
2. `entrenar_rnn.py` lee el corpus, construye el vocabulario, crea
   ventanas de 128 caracteres, entrena una RNN vanilla por 150 epocas,
   y guarda el modelo
3. `servidor_api.py` carga el modelo, construye una version optimizada
   para inferencia (sin TimeDistributed), la compila con @tf.function,
   y expone dos endpoints HTTP
4. La extension de VS Code o curl envian el codigo escrito hasta el
   momento, la RNN predice el siguiente caracter uno por uno, y lo
   devuelve para insertarlo en el editor
