# Reporte Actividad 3: Asistente de Codigo Personalizado con RNN

## Introduccion

En esta actividad diseñamos, entrenamos e integramos un modelo basico de
lenguaje utilizando Redes Neuronales Recurrentes (RNN Vanilla) en
TensorFlow/Keras para autocompletar codigo C. El modelo se desplego como una
API local con Flask y se conecto a una extension de VS Code para usarlo
directamente desde el editor.

---

## 1. Fundamento teorico

### Redes Neuronales Recurrentes (RNN)

Las RNN estan disenadas para trabajar con datos secuenciales. A diferencia de
una red densa tradicional (MLP) que ve cada entrada de forma independiente, una
RNN mantiene un estado oculto que funciona como "memoria" de los pasos
anteriores. La ecuacion fundamental de la RNN vanilla es:

    h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

Esto significa que la salida en el paso t depende tanto de la entrada actual
x_t como del estado oculto anterior h_{t-1} (`apuntes_ia.md:3595-3610`).

### Por que RNN y no MLP para autocompletar

Un MLP recibiria una ventana de caracteres fija y no tendria memoria de lo que
vino antes de esa ventana. La RNN, al mantener un estado oculto que se
actualiza con cada caracter, puede capturar dependencias de largo alcance en el
codigo (`apuntes_ia.md:3556-3570`).

### Arquitectura usada

- **Embedding** (`apuntes_ia.md:9104-9150`): convierte cada caracter (indice
  entero) en un vector denso de 64 dimensiones. Esto permite que la red aprenda
  relaciones semanticas entre caracteres (ej: letras que suelen aparecer juntas).
- **SimpleRNN** con tanh y 128 unidades ocultas: procesa la secuencia de
  vectores de embedding y mantiene el estado recurrente.
- **TimeDistributed(Dense)** (`apuntes_ia.md:6410-6416`): aplica la misma capa
  densa a cada paso temporal para producir logits (puntuaciones) para cada
  caracter del vocabulario.

La combinacion `return_sequences=True` + `TimeDistributed` es necesaria porque
entrenamos al modelo a predecir el siguiente caracter en cada posicion de la
ventana, no solo al final (`apuntes_ia.md:6578-6586`).

### Funcion de perdida y optimizador

Usamos `SparseCategoricalCrossentropy(from_logits=True)` porque tenemos
etiquetas como indices enteros (no one-hot) y los logits de salida no pasan por
softmax. El optimizador Adam (`apuntes_ia.md:2216-2229`) adapta la tasa de
aprendizaje por parametro, combinando ventajas de Momentum y RMSprop.

---

## 2. Estructura del proyecto

```
project/project3/
├── dataset_c.txt              # 97 funciones C (~30000 caracteres)
├── entrenar_rnn.py            # Entrenamiento de la RNN
├── modelo_rnn.h5              # Modelo entrenado (489 KB)
├── tokenizer_mappings.pkl     # Mapeos caracter→indice
├── example.c                  # Ejemplo de funciones C
├── playground.c               # Archivo de prueba rapida
├── servidor_api.py            # API Flask para servir el modelo
├── extension_vscode/          # Extension VS Code
│   ├── package.json
│   └── extension.js
├── reporte_rnn.md             # Reporte del proyecto
└── explicacion_codigo.md      # Explicacion detallada del codigo
```

---

## 3. Explicacion del codigo

### 3.1 dataset_c.txt

Contiene 97 funciones en C (originalmente 73, expandido a 97) con un estilo consistente:

- Nombres en español con snake_case: `sumar_dos_enteros`, `invertir_cadena`
- Comentario `//` antes de cada funcion explicando que hace
- Sangria de 4 espacios
- Llave de apertura en la misma linea

Tipos de funciones incluidas:

| Categoria | Ejemplos |
|-----------|----------|
| Aritmetica basica | sumar, restar, multiplicar, dividir, area, volumen |
| Matematicas | factorial, fibonacci, primo, MCD, MCM |
| Cadenas | longitud, copiar, invertir, buscar, mayusculas |
| Arreglos | maximo, minimo, suma, promedio, ocurrencias |
| Ordenamiento | burbuja, seleccion, insercion, quick sort, merge sort |
| Matrices | sumar, multiplicar, transponer |
| Estructuras | Punto, Rectangulo, Estudiante |
| Memoria dinamica | malloc, free, realloc, matriz 2D |
| Listas enlazadas | insertar, eliminar, buscar, imprimir |
| Arbol binario | insertar, buscar, inorden, altura |
| Miscelaneos | email, palindromo, binario, intercambiar |

### 3.2 entrenar_rnn.py

**Paso 1: Cargar el corpus**
Lee el archivo `dataset_c.txt` completo como una cadena de texto.

**Paso 2: Crear el vocabulario**
Extrae todos los caracteres unicos del corpus, crea dos diccionarios:
- `stoi`: de caracter a indice entero (ej: `'a' -> 5`)
- `itos`: de indice entero a caracter (ej: `5 -> 'a'`)

Esto es identico a como se hace en `apuntes_ia.md:6456-6458`.

**Paso 3: Crear ventanas de entrenamiento**
Cada muestra de entrenamiento es una ventana de 128 caracteres (X) y la misma
ventana desplazada un caracter a la derecha (Y). Esto ensena al modelo a
predecir el siguiente caracter dado el contexto (`apuntes_ia.md:6646-6658`).
El block_size se aumento de 64 a 128 para que el modelo alcance a ver el
nombre de la funcion cuando decide que operador usar en el cuerpo.

**Paso 4: Construir el modelo**
Arquitectura: Input(128) -> Embedding(75, 64) -> SimpleRNN(128, tanh) ->
TimeDistributed(Dense(75)). Total: 39,179 parametros.

**Paso 5: Entrenar**
150 epocas con lotes de 32. La perdida bajo de 1.24 a 0.116 (vs 0.172 del
modelo anterior). Incluye EarlyStopping (paciencia 15) por si el modelo
deja de mejorar.

**Paso 6: Guardar**
El modelo se guarda en `modelo_rnn.h5` y los mapeos en `tokenizer_mappings.pkl`.

### 3.3 servidor_api.py

Carga el modelo y los mapeos, y expone dos endpoints:

- **GET /health**: verifica que el servidor funciona
- **POST /autocompletar**: recibe JSON con:
   ```json
   {
     "codigo": "// sumar_dos_enteros",
      "max_tokens": 350
    }
    ```
    Devuelve:
    ```json
    {
      "codigo_original": "// sumar_dos_enteros",
      "completado": "(int a, int b) {\n    int resultado = a + b;\n...",
      "codigo_completo": "// sumar_dos_enteros(int a, int b) {\n..."
   }
   ```

La generacion usa muestreo con temperatura (`0.05`) y top-k (`3`) para
equilibrar determinismo y variedad: selecciona uno de los 3 caracteres mas
probables, escalando las probabilidades con temperatura baja para favorecer
opciones seguras. Se detiene automaticamente al completar la funcion (`}\n\n`).

Para hacer la inferencia mas rapida, el servidor no usa el modelo original
(`return_sequences=True` + `TimeDistributed`) directamente. En su lugar,
construye un modelo de inferencia separado con `return_sequences=False` y
`Dense` simple (sin TimeDistributed), copia los pesos por nombre de capa, y
compila el forward pass con `@tf.function` para que TensorFlow lo ejecute
como grafo compilado en vez de eager.

### 3.4 extension_vscode/

Extension minimima de VS Code que:

1. Lee el contenido del archivo .c activo
2. Lo envia al endpoint `/autocompletar` de la API local
3. Inserta el texto generado en la posicion del cursor

Atajo de teclado: `Ctrl+Shift+L` en archivos .c.

---

## 4. Como usar el proyecto

### 4.1 Ruta rapida (usar el modelo ya entrenado)

```bash
# 1. Iniciar el servidor API
cd project/project3
python3 servidor_api.py
# El servidor se queda escuchando en http://127.0.0.1:5000

# 2. Probar desde otra terminal con curl
curl -X POST http://127.0.0.1:5000/autocompletar \
  -H "Content-Type: application/json" \
  -d '{"codigo": "// sumar_dos_enteros", "max_tokens": 100}'

# 3. (Opcional) Instalar extension VS Code
cp -r extension_vscode ~/.vscode/extensions/autocompletado-rnn-c
# Recargar VS Code, abrir un .c y presionar Ctrl+Shift+L
```

### 4.2 Entrenar desde cero (si quieres modificar el dataset)

```bash
cd project/project3
python3 entrenar_rnn.py    # Toma ~50 min en CPU (150 epocas)
python3 servidor_api.py     # Iniciar API con el nuevo modelo
```

### 4.3 Probar el autocompletado

Una vez que el servidor corra, puedes hacer peticiones:

```bash
curl -X POST http://127.0.0.1:5000/autocompletar \
  -H "Content-Type: application/json" \
  -d '{"codigo": "// fibonacci", "max_tokens": 150}'
```

Para mejores resultados, da mas contexto en el prompt:

```bash
curl -X POST http://127.0.0.1:5000/autocompletar \
  -H "Content-Type: application/json" \
  -d '{"codigo": "// generar_fibonacci: llena un arreglo\nvoid generar_fibonacci(int n, int arreglo[])", "max_tokens": 200}'
```

---

## 5. Notas sobre el entrenamiento y optimizacion (v3)

- Se usaron 28,939 muestras de entrenamiento a partir de ~97 funciones C
- Vocabulario de 75 caracteres distintos (letras, numeros, signos, espacios)
- 39,179 parametros entrenables
- Entrenamiento: ~2.5 h en CPU (sin GPU), 150 epocas
- Perdida final: 0.1157 (vs 0.1717 de la v1, mejora del 33%)
- Block_size: 128 para mejor contexto del nombre de funcion
- Inferencia: temperatura 0.05 + top_k=3 en lugar de argmax greedy
- Optimizacion: modelo de inferencia separado con `return_sequences=False`
  (evita costo de TimeDistributed sobre los 128 pasos temporales) y forward
  compilado con `@tf.function`

---

## 6. Posibles mejoras

- ~~Aumentar block_size a 128 para capturar dependencias mas largas~~ (hecho en v2)
- ~~Inferencia con temperatura + top_k en vez de greedy~~ (hecho en v3)
- ~~Optimizar servidor con modelo de inferencia separado + @tf.function~~ (hecho en v3)
- Usar LSTM en lugar de SimpleRNN para mejor memoria a largo plazo
- Apilar varias capas RNN
- Entrenar con GPU para mas rapidez
- Agregar un endpoint `/sugerir` que devuelva varias opciones
- Mejorar la extension de VS Code con popup de seleccion multiple
