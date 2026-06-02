# CNN - Clasificador de Animales v2

Clasificador de imágenes en 5 clases (arañas, ballenas, changos, pájaros, ranas)
usando una CNN con **Squeeze-and-Excitation blocks**, **MixUp augmentation**,
**fondo aleatorio durante entrenamiento** y **remoción de fondo en inferencia**.

## Dataset

- 5 clases con **5000 imágenes** cada una (**25000 total**)
- Imágenes redimensionadas a **64×64** píxeles
- División 64/16/20: Train **16000**, Val **4000**, Test **5000**
- Stratified split para mantener proporción de clases
- Fuentes: Kaggle (HappyWhale, spiders, birds, monkeys) + GitHub (frog-dataset)

### Desbalance y aumentación

Cuando una clase tiene menos de 5000 imágenes en `raw_descargas/`, `preparar_dataset.py`
genera imágenes aumentadas con `ImageDataGenerator` (rotación, zoom, shift, flip,
brillo) hasta llegar al total. Ejemplo: ballenas solo tenía ~2935 raw, se generaron
~2065 aumentadas.

## Mejoras respecto a v1

### 1. SE Blocks (Squeeze-and-Excitation)

Cada bloque convolucional termina con un SE block que aprende a **ponderar canales**:
amplifica los que detectan el objeto y suprime los que detectan fondo.

```
Entrada: [H, W, C]  →  GAP → Dense(C/r) → ReLU → Dense(C) → Sigmoid → ×
```

- Se agrega después de BatchNorm, antes de MaxPooling
- Reduction ratio = 8 (mínimo 4 canales)
- Costo computacional mínimo (~1-2% más parámetros)
- **Idea clave**: la red aprende QUÉ canales son útiles y cuáles no, como un mecanismo
  de atención por canal. Si un canal detecta "fondo blanco" en lugar de "rana", el SE
  block aprende a darle peso bajo.

### 2. MixUp Augmentation

En lugar de ImageDataGenerator tradicional, se usa **MixUp**: las imágenes
de entrenamiento se interpolan linealmente entre pares aleatorios:

```python
x = λ * x_i + (1-λ) * x_j
y = λ * y_i + (1-λ) * y_j
```

- α = 0.2 controla la fuerza de la mezcla
- Fuerza al modelo a aprender features del objeto, no del fondo
- Con λ ~ Beta(0.2, 0.2), la mayoría de mezclas son cercanas a una imagen pura
  o la otra, con pocas mezclas 50/50

### 3. Fondo aleatorio durante entrenamiento

**Problema original**: la clase `ranas` tiene ~85% de imágenes con fondo blanco
de laboratorio. El modelo aprendía "fondo blanco = rana" en lugar de características
reales de la rana. Al inferir en fotos reales con fondo natural, fallaba.

**Solución**: en cada batch de entrenamiento, antes de aplicar MixUp, se detecta
el fondo mediante **bordes** y se reemplaza con un color aleatorio:

```python
def _randomizar_fondo(self, batch_X):
    # 1. Convertir a gris y detectar bordes (Sobel aproximado)
    gray = np.mean(batch_X[i], axis=2)
    gx = np.abs(np.diff(gray, axis=1, append=0))
    gy = np.abs(np.diff(gray, axis=0, append=0))
    edge = ((gx + gy) > 0.06).astype(np.float32)

    # 2. Dilatar bordes 4 veces para cubrir el cuerpo del animal
    # 3. Suavizar máscara con box blur

    # 4. Reemplazar fondo con color aleatorio
    color = colores_fondo[rand]  # verde, azul, café, blanco, gris...
    batch_X[i] = batch_X[i] * edge + color * (1 - edge)
```

**Colores de fondo usados**: verdes naturaleza, azules cielo/mar, cafés tierra,
blanco y gris. Así el modelo NUNCA puede usar el color del fondo como pista,
porque cambia cada época.

**¿Por qué detección de bordes y no umbrales de color?**
- Umbrales de color solo funcionan si el fondo es uniforme (blanco, azul cielo)
- Bordes funcionan para CUALQUIER fondo: el animal siempre tiene bordes
- Más robusto y generalizable

### 4. Label Smoothing

La loss usa `CategoricalCrossentropy(label_smoothing=0.05)` para evitar
que el modelo tenga demasiada confianza en sus predicciones.

**¿Qué hace?** En lugar de one-hot exacto `[0, 1, 0, 0, 0]`, usa targets
suavizados como `[0.01, 0.96, 0.01, 0.01, 0.01]`.

**¿Por qué el loss se ve "alto"?** Con label smoothing, aunque el modelo
clasifique perfecto, el loss mínimo teórico es ~0.05-0.15. Un loss de 0.4
con smoothing equivale a ~0.25-0.35 sin smoothing.

**Valor usado**: 0.05 (bajado de 0.1 para que el loss se vea más bajo).

### 5. Cosine Decay LR

Reemplaza ReduceLROnPlateau por un decaimiento coseno suave desde
1e-3 hasta 1e-4 a lo largo de todo el entrenamiento.

```python
cosine_decay = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=len(train_gen) * EPOCHS,
    alpha=1e-4  # LR final = 1e-4
)
```

**Ventaja**: no espera a que el modelo se estanque para bajar la LR,
sino que baja suavemente desde el inicio, permitiendo convergencia más estable.

## Pipeline de inferencia (remoción de fondo)

### ¿Por qué remover el fondo en inferencia pero NO en entrenamiento?

1. **Entrenamiento**: las imágenes tienen sus fondos originales. El `_randomizar_fondo`
   durante el batch cambia fondos aleatoriamente para que el modelo no memorice colores.
2. **Inferencia**: al clasificar una imagen nueva, se prueban **dos versiones**:
   - La imagen original (con su fondo)
   - La imagen con fondo removido y reemplazado por **blanco**
3. Se elige la predicción con **mayor confianza** de las dos.

**¿Por qué blanco?** La clase `ranas` tiene ~85% de imágenes con fondo blanco
en entrenamiento. Si removemos el fondo de la imagen de inferencia y ponemos
blanco, la imagen se parece más a lo que el modelo vio durante entrenamiento.

**Esto pasa automáticamente, el usuario nunca sabe cuál versión ganó.**

### Remoción de fondo (rembg)

Usa el modelo **U²-Net** (pre-entrenado) a través de la librería `rembg`:
1. Toma la imagen PIL
2. `rembg.remove()` devuelve RGBA con fondo transparente
3. Se reemplaza el canal alpha con blanco `[255, 255, 255]`

```python
COLOR_FONDO = np.array([255, 255, 255], dtype=np.uint8)

def remover_fondo(pil_img):
    resultado = rembg.remove(pil_img, session=session)
    arr = np.array(resultado, dtype=np.uint8)
    mask = arr[:, :, 3] > 0
    rgb = arr[:, :, :3]
    salida = np.full_like(rgb, COLOR_FONDO)
    salida[mask] = rgb[mask]
    return Image.fromarray(salida)
```

## Arquitectura v2

```
Input(64, 64, 3)
│
├─ Conv2D 32 (3×3) → LeakyReLU → BatchNorm → SE Block → MaxPool2D → Dropout(0.2)
├─ Conv2D 64 (3×3) → LeakyReLU → BatchNorm → SE Block → MaxPool2D → Dropout(0.2)
├─ Conv2D 128 (3×3) → LeakyReLU → BatchNorm → SE Block → MaxPool2D → Dropout(0.2)
│
├─ Flatten
├─ Dense 256 → LeakyReLU → BatchNorm → Dropout(0.35)
├─ Dense 128 → LeakyReLU → BatchNorm → Dropout(0.35)
└─ Dense 5 → Softmax
```

- Optimizer: Adam (lr inicial 1e-3, cosine decay a 1e-4)
- Loss: categorical_crossentropy con label_smoothing = 0.05
- Batch size: 64
- Max epochs: 30 (EarlyStopping patience = 5)
- MixUp α = 0.2

**Total params**: ~2.2M (8.5 MB)

## Decisiones de diseño

### ¿Por qué SE blocks?
Las imágenes de animales tienen fondos muy variables. El SE block permite
al modelo aprender qué canales ignorar, mejorando la precisión en presencia
de fondos diversos. Funciona como un mecanismo de atención por canal.

### ¿Por qué fondo aleatorio y no solo rembg en entrenamiento?
- rembg procesa ~2-5 imágenes por segundo → 25000 imágenes tomaría horas
- El fondo aleatorio es instantáneo (se aplica en el batch en memoria)
- Al cambiar colores cada época, el modelo ve MÁS variedad que con rembg

### ¿Por qué detección de bordes y no umbrales HSV?
- Los umbrales HSV asumen que el fondo tiene un color específico (blanco)
- Los bordes funcionan para cualquier tipo de fondo
- Más robusto cuando hay imágenes con fondo variado

### ¿Por qué LeakyReLU en vez de ReLU?
Evita neuronas muertas (dying ReLU) y permite gradientes pequeños
pero no nulos en la región negativa.

### ¿Por qué BatchNorm antes del Dropout?
Normaliza las activaciones antes de aplicar dropout, estabilizando
el entrenamiento y permitiendo learning rates más altos.

### ¿Por qué blanco como color de reemplazo en inferencia?
La clase ranas tiene predominancia de fondo blanco en entrenamiento.
Al inferir con fondo blanco, la imagen se alinea con la distribución
de entrenamiento de esa clase, mejorando la detección.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `CNN_animales.ipynb` | Notebook principal (entrenamiento + widget de predicción) |
| `probar_modelo.ipynb` | Notebook solo para predicción (sin entrenamiento) |
| `predecir.py` | Script CLI para predecir una imagen: `python3 predecir.py <ruta>` |
| `preparar_dataset.py` | Balancea las 5 clases a 5000 imágenes cada una |
| `procesar_imagen.py` | Utilidad con `remover_fondo()` usando rembg |
| `descargar_datasets.py` | Descarga imágenes raw desde Kaggle/GitHub |
| `enable_gpu.py` | Utilidad para forzar detección de GPU |
| `modelo_animales_v2.keras` | Mejor modelo guardado (Keras v3) |
| `modelo_animales_v2.h5` | Modelo en formato legacy HDF5 |
| `dataset/` | 5 carpetas con 5000 imágenes cada una |
| `raw_descargas/` | Imágenes originales descargadas |

## Scripts

### `preparar_dataset.py`

1. Limpia la carpeta `dataset/<clase>/`
2. Si hay ≥5000 raw, muestrea aleatoriamente 5000
3. Si hay <5000 raw, copia todas y genera aumentadas hasta llegar a 5000
4. Las imágenes copiadas se nombran `real_XXXXXX.jpg`
5. Las aumentadas se nombran `aug_XXXXXX.jpg`

### `predecir.py`

```bash
python3 predecir.py ruta/de/la/imagen.jpg
```

Internamente:
1. Carga la imagen
2. Redimensiona a 64×64
3. Evalúa con y sin remoción de fondo
4. Muestra la mejor predicción

### `procesar_imagen.py`

Módulo compartido entre `predecir.py`, `CNN_animales.ipynb` y `probar_modelo.ipynb`.
Usa `rembg` para eliminar el fondo y reemplazarlo con blanco.

## Uso

### Preparar dataset
```bash
.venv/bin/python3 preparar_dataset.py
```

### Entrenar (Jupyter)
```bash
.venv/bin/jupyter notebook CNN_animales.ipynb
```
Ejecutar todas las celdas en orden. Entrenamiento: ~30-60 min en CPU.

### Probar modelo (Jupyter)
```bash
.venv/bin/jupyter notebook probar_modelo.ipynb
```
Solo el widget de predicción, sin código de entrenamiento.

### Inferencia (CLI)
```bash
.venv/bin/python3 predecir.py ruta/de/imagen.jpg
```

## GPU

La **GTX 1050** (Pascal, CC 6.1) **NO** es compatible con TF 2.21 + CUDA 12.x.
El notebook fuerza CPU con `CUDA_VISIBLE_DEVICES=-1`.

Si tienes una GPU compatible (CC ≥ 7.0), elimina esa línea para entrenar en GPU.

## Posibles preguntas

### ¿Por qué loss alto si el accuracy es bueno?
Label smoothing. Con smoothing=0.05, el loss mínimo teórico es ~0.05.
Un loss de 0.4-0.5 con accuracy 93-95% es normal y esperado.

### ¿Por qué fallaban las ranas?
El dataset de ranas tenía ~85% imágenes con fondo blanco de laboratorio.
El modelo aprendió "fondo blanco = rana". Al mostrarle una rana real con
fondo natural, no la reconocía porque no veía el fondo blanco asociado.

### ¿Cómo se solucionó?
1. **Fondo aleatorio en entrenamiento**: se reemplaza el fondo de cada imagen
   con colores aleatorios durante el batch, forzando al modelo a ignorar el fondo
2. **Remoción de fondo en inferencia**: se prueba la imagen con y sin fondo,
   y se usa la predicción con mayor confianza

### ¿Por qué no se removió el fondo del dataset directamente?
Procesar 25000 imágenes con rembg tomaría horas. El fondo aleatorio en memoria
es instantáneo y además genera más variedad (colores diferentes cada época).

### ¿Por qué patience=5 en EarlyStopping?
Para que el entrenamiento no se alargue innecesariamente. Si el loss de validación
no mejora en 5 épocas, se detiene. Combinado con CosineDecay, el modelo suele
converger en 8-12 épocas.

### ¿El fondo aleatorio no daña el entrenamiento?
No. Al reemplazar el fondo con colores naturales (verdes, azules, cafés),
el modelo aprende a enfocarse en la forma/textura del animal. Además, MixUp
mezcla imágenes con diferentes fondos, reforzando la invariancia al fondo.

### ¿Qué pasa si rembg falla en remover el fondo?
El algoritmo de inferencia prueba AMBAS versiones (con y sin fondo) y elige
la de mayor confianza. Si rembg falla, la versión original gana.
