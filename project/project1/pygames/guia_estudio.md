# Guía de estudio — juego con MLP / Árbol de Decisión

## 1. Pipeline de Machine Learning

```
[ Juego manual ] → recolectar datos → entrenar modelo → [ Juego auto ]
                        ↑                              ↓
                   más datos ←── evaluar y repetir ←───┘
```

Cada paso del pipeline tiene problemas reales que hay que resolver. El juego es el pretexto; lo importante son las decisiones de diseño de ML.

---

## 2. El problema de clasificación

**Entrada (features):** 3 variables por frame

| Feature | Tipo | Rango típico | Descripción |
|---------|------|-------------|-------------|
| `velocidad_bala` | float | -18 a -4 | Rapidez de la bala (negativo = hacia la izquierda) |
| `distancia` | float | 0 a ~1000 | Pixeles entre el jugador y la bala |
| `altura_bala` | int | 0 o 1 | 0 = bala baja (a los pies), 1 = bala alta (a la cabeza) |

**Salida (clases):** 3 acciones discretas

| Código | Acción | Cuándo usarla |
|--------|--------|---------------|
| 0 | Quieto | Bala alta y lejos, o bala ya pasó |
| 1 | Salto | Bala baja y cerca (esquivar saltando) |
| 2 | Agachado | Bala alta y cerca (esquivar agachándose) |

Es un problema de **clasificación multiclase** (3 clases) con **features mixtas** (2 continuas, 1 categórica binaria).

---

## 3. Recolección de datos — calidad > cantidad

### 3.1 El problema del muestreo ingenuo

`registrar_decision_manual()` se llama **cada frame** (~45 veces/segundo).

En la primera versión, la acción se infería del **estado actual** (si `en_suelo == False → acción = 1`, si `agachado → acción = 2`, sino `acción = 0`).

**Problema:** un solo salto dura ~20-30 frames y genera 20-30 muestras con `acción = 1`, aunque el jugador haya presionado la tecla una sola vez. El dataset termina con:

```
quieto:  514  (89.4%)
salto:   23   (4.0%)
agachado: 38  (6.6%)
```

**Desbalance extremo (~22:1).** El árbol aprende "predice quieto siempre y aciertas 90%". No aprende a esquivar.

### 3.2 Solución — registro por evento + submuestreo

```
┌──────────────────────────────────────────────────────────┐
│  ¿Cómo se registra?           │ ¿Cuándo?                  │
├───────────────────────────────┼──────────────────────────┤
│  accion=1 (salto)             │ Solo al presionar UP      │
│  accion=2 (agachado)          │ Solo al presionar DOWN    │
│  accion=0 (quieto)            │ 1 de cada 30 frames (~0.67s) │
└───────────────────────────────┴──────────────────────────┘
```

- **Salto y agachado** se registran en el evento `KEYDOWN`, no en el loop. Cada acción = 1 muestra, no 20.
- **Quieto** se registra con submuestreo: contador `frames_subsampling` que solo guarda cada 30 frames. Esto reduce drásticamente la clase mayoritaria sin perder representación de distancias variadas.

Resultado con ~3 min de juego:

```
quieto:  172  (75.1%)
salto:   29   (12.7%)
agachado: 28  (12.2%)
```

Relación ~6:1. Mucho más saludable para cualquier clasificador.

### 3.3 Lección general

> En ML, la calidad del dataset importa más que la complejidad del modelo. Un árbol con buenos datos le gana a una red neuronal con datos sucios.

---

## 4. Modelos

### 4.1 MLP (Multi-Layer Perceptron) — tecla T

```python
MLPClassifier(
    hidden_layer_sizes=(8, 4),    # 2 capas ocultas: 8 → 4 neuronas
    activation="relu",            # no linealidad
    solver="adam",                # optimizador adaptativo
    max_iter=300000,              # suficientes épocas para converger
    random_state=42,
)
```

**Arquitectura:** `3 → 8 → 4 → 3`

```
Features    Capa oculta 1    Capa oculta 2    Salida (softmax)
┌─────┐     ┌────┐ ┌────┐    ┌────┐ ┌────┐    ┌─────┐
│vel  │────▶│ n1 │ │ n2 │...─▶│ n1 │ │ n2 │...─▶│quieto│
│dist │────▶│    │ │    │    │    │ │    │    │ salto │
│alt  │────▶│    │ │    │    │    │ │    │    │agachado│
└─────┘     └────┘ └────┘    └────┘ └────┘    └─────┘
```

**Preprocesamiento:** Requiere `StandardScaler` — normaliza cada feature a media 0, varianza 1. Sin esto, features con escalas grandes (distancia ~1000) dominan a las chicas (altura_bala 0/1).

**Forward pass:** combinación lineal → ReLU → combinación lineal → ReLU → combinación lineal → softmax → probabilidades.

**Backpropagation:** adam optimiza los pesos para minimizar cross-entropy loss.

### 4.2 Árbol de Decisión — tecla R

```python
DecisionTreeClassifier(
    max_depth=6,          # profundidad máxima (controla complejidad)
    min_samples_leaf=4,   # mín muestras por hoja (evita overfitting)
    random_state=42,
)
```

**Cómo funciona:** El árbol hace **preguntas binarias** en cada nodo:
- `¿distancia < 150?` → sí/no
- `¿altura_bala == 0?` → sí/no
- `¿velocidad_bala < -10?` → sí/no

Cada pregunta divide los datos en 2 subconjuntos más puros (menor entropía / Gini impurity).

**Visualización conceptual de un split:**

```
                      ┌──────────────────┐
                      │  distancia < 200? │
                      └────────┬─────────┘
                              / \
                             /   \
               sí (cerca)   /     \   no (lejos)
                          /         \
                 ┌─────────┐      ┌──────────────────┐
                 │altura==0?│     │     predecir      │
                 └────┬────┘      │    quieto (0)     │
                     / \          └──────────────────┘
                    /   \
          ┌─────────┐  ┌──────────┐
          │ salto(1) │  │agachado(2)│
          └──────────┘  └──────────┘
```

**Propiedades del árbol:**
- **No necesita normalización:** las escalas no importan, solo el orden de los valores.
- **Interpretable:** puedes visualizar las reglas de decisión.
- **Overfittea fácil:** si `max_depth` es muy grande, memoriza ruido en vez de generalizar.

### 4.3 Comparación MLP vs Árbol

| Aspecto | MLP | Árbol |
|---------|-----|-------|
| Interpretabilidad | Baja (caja negra) | Alta (reglas visibles) |
| Normalización | Necesaria | No necesaria |
| Datos necesarios | Más (≥100) | Menos (≥30 por clase) |
| Overfitting | Regularizable (dropout, capas chicas) | Fácil (controlar depth/leaf) |
| Frontera decisión | Suave, continua | Rígida, escalonada |
| Probabilidades | Suaves (softmax) | Discretas (proporción en hoja) |

**Para este juego:** el MLP da transiciones más suaves. El árbol es más brusco pero más fácil de diagnosticar y explicar.

---

## 5. De la predicción a la acción

### 5.1 `decision_auto()` — la inferencia

```python
def decision_auto(self) -> int:
    probas = self.modelo.predict_proba(Xs)[0]  # [p0, p1, p2]
    best_idx = int(probas.argmax())
    if probas[best_idx] < _UMBRAL:   # 0.35
        return -1                     # mantener estado actual
    return int(self.modelo.classes_[best_idx])
```

**Problema clásico de `argmax`:** `predict_proba` devuelve un array indexado de 0 a n_clases-1. `argmax` da el **índice**, no la etiqueta.

Si los datos solo tienen clases {0, 2}:
- Índice 0 → probabilidad para clase 0 (quieto)
- Índice 1 → probabilidad para clase 2 (agachado) ← ¡el índice 1 NO es clase 1!

Usar `modelo.classes_[best_idx]` traduce el índice a la clase real.

**Sin este fix:** si faltó clase 1 en entrenamiento, el modelo predecía agachado pero el código lo interpretaba como salto. Saltaba cuando debería agacharse.

### 5.2 Umbral de confianza

`_UMBRAL = 0.35`. Si ninguna clase alcanza 35% de probabilidad, devuelve `-1` (no hacer nada, mantener estado actual).

**Por qué:** con 3 clases, si todas están cerca de 33%, cualquier decisión es casi aleatoria. `-1` le dice "no cambies nada hasta que el modelo esté más seguro".

### 5.3 Suavizado temporal (buffer de decisiones)

```python
self.buffer_decision = deque(maxlen=3)  # últimas 3 predicciones
```

En modo auto, se acumulan las últimas 3 decisiones del modelo. Solo se ejecuta una acción si las 3 son idénticas:

```
Frame 1: modelo → salto     buffer = [salto]          → no se ejecuta
Frame 2: modelo → salto     buffer = [salto, salto]   → no se ejecuta
Frame 3: modelo → salto     buffer = [salto, salto, salto] → ¡SALTA!
Frame 4: modelo → quieto    buffer = [quieto]          → se reseteó, no hace nada
```

Esto filtra **predicciones espurias de un solo frame**. Si el modelo se equivoca un frame y acierta los siguientes, el error se ignora.

**Analogía:** es un filtro de medianía temporal sobre las decisiones.

### 5.4 Cooldown post-agachado

Cuando el timer de agachado expira, se activa un cooldown de 8 frames (~0.18s) que bloquea saltar O agacharse de nuevo:

```python
# En manejar_agachado():
if self.timer_agachado == 0:
    self.agachado = False
    self.cooldown_agachado = 8   # bloquea acciones por 8 frames

# En la decisión auto:
if decision == 1 and self.cooldown_agachado == 0:  # no salta en cooldown
if decision == 2 and self.cooldown_agachado == 0:  # no se agacha en cooldown
```

**Problema que resuelve:** cuando el timer de agachado expira, el modelo ve que la bala sigue siendo amenaza y predice agachado o salto inmediatamente de nuevo. Sin cooldown, el personaje rebota entre estados.

### 5.5 Orden de ejecución crítico

```python
if self.salto:
    self.manejar_salto()       # 1. aterrizar ANTES de decidir

if self.modo_auto:
    decision = self.decision_auto()  # 2. decidir con en_suelo actualizado
```

Si `manejar_salto()` aterriza al jugador en este frame, `en_suelo = True`. Luego el modelo puede decidir agacharse inmediatamente. Sin este orden, el modelo nunca puede agacharse en el mismo frame que aterriza.

---

## 6. Estrategias contra el "rebote" en Árbol de Decisión

El árbol produce fronteras de decisión **rígidas** (escalones). El MLP produce transiciones **suaves**. Esto causa problemas específicos con el árbol:

### Problema

1. Timer de agachado expira → `agachado = False`
2. Árbol ve: bala cerca + altura=1 → clasifica como "salto" (porque en los datos de entrenamiento, a esa distancia y altura, la acción era saltar)
3. Personaje: agachado → parado → salta inmediatamente

### Soluciones implementadas

| # | Solución | Qué hace | Cómo ayuda |
|---|----------|----------|------------|
| 1 | `_FRAMES_AGACHADO_AUTO = 45` | El auto se agacha 1s en vez de 0.4s | Cubre toda la aproximación de la bala, no se para antes de que pase |
| 2 | `not self.agachado` guard | Impide resetear el timer cada frame | Evita que el modelo extienda el agachado infinitamente |
| 3 | Umbral 0.35 | Solo actuar si el modelo está ≥35% seguro | Evita cambios con baja confianza |
| 4 | Buffer 3 frames | Requiere 3 predicciones iguales consecutivas | Filtra espurios de 1 frame |
| 5 | Cooldown post-agachado | ~0.18s sin acciones después de agacharse | Da tiempo a que la bala pase |
| 6 | Submuestreo de quieto 1:30 | Balancea el dataset | El árbol no biascea a predecir quieto siempre |

---

## 7. Flujo de juego recomendado

1. **M** en menú → modo manual (resetea datos)
2. Jugar **5-10 minutos** alternando quieto/salto/agachado según la bala. Intentar cubrir:
   - Balas rápidas y lentas
   - Balas altas y bajas
   - Distancias variadas (agacharse/saltar tanto temprano como tarde)
3. **R** → entrenar Árbol de Decisión
4. **A** → modo auto, ver cómo juega
5. Si hay problemas: **M** → más datos → **R** → **A**
6. Opcional: **C** exporta el CSV para analizarlo externamente

---

## 8. Posibles preguntas del profe (y respuestas)

### Sobre el pipeline

**P: ¿Por qué no usaste redes convolucionales (CNN) ya que es un juego?**

R: El estado del juego se puede representar con 3 features numéricas (velocidad, distancia, altura). No necesitamos extraer patrones espaciales de píxeles. Un clasificador tabular (MLP/Árbol) es más simple, más rápido de entrenar y más interpretable para este espacio de estados de baja dimensión.

**P: ¿Y si la bala tuviera trayectoria curva? ¿Seguirían alcanzando 3 features?**

R: No. Habría que agregar features como aceleración, ángulo, o incluso una ventana de posiciones anteriores (seq2seq con RNN). Pero para el caso lineal actual, 3 features bastan.

**P: ¿Por qué el MLP usa 2 capas ocultas (8 y 4)? ¿Cómo se eligieron?**

R: Fue empírico. 3 features de entrada → 8 → 4 → 3 clases. La capa de 8 captura combinaciones lineales de las features, la de 4 las refina. Muy pocas neuronas y no aprende; demasiadas y overfittea con el dataset chico. 8→4 dio el mejor balance con <500 muestras.

### Sobre el dataset

**P: ¿Por̃ qué 80 muestras como mínimo para entrenar?**

R: Con menos de 80, estratificar en train/test 80/20 dejaría ~16 de test, muy pocas para una evaluación confiable. También es probable que falten clases (ej. solo quieto y salto), lo que activa el "modelo trivial".

**P: ¿Qué pasa si solo registro datos de balas lentas y el auto recibe una rápida?**

R: El modelo no generaliza a velocidades no vistas. Las features están dentro del rango de entrenamiento. Si solo entrenó con velocidades -4 a -8 y recibe -18, la distancia a la que debe reaccionar es distinta. Solución: jugar variando velocidades (el juego genera velocidades aleatorias entre -18 y -4).

**P: Submuestrear quieto 1:30 — ¿no se pierde información valiosa?**

R: Se pierde información redundante, no valiosa. 30 frames seguidos de quieto con distancias muy parecidas no aportan diversidad. El submuestreo conserva la variabilidad (distintas distancias, alturas, velocidades) pero elimina la repetición.

### Sobre el Árbol de Decisión

**P: ¿Por qué el árbol no necesita normalización y el MLP sí?**

R: El árbol hace splits basados en umbrales (ej. `distancia < 150`), solo importa el orden relativo, no la magnitud. El MLP usa combinaciones lineales con pesos: si distancia está en escala 0-1000 y altura_bala en 0-1, la distancia domina el gradiente y la altura es ignorada. El scaler iguala las escalas.

**P: ¿Qué pasa si el profe pide visualizar el árbol?**

R: Podemos imprimir las reglas con `sklearn.tree.export_text(modelo)`, o exportar a PDF con `export_graphviz`. Se verían los splits: `distancia <= 150 → altura_bala <= 0.5 → clase=1 (salto)`.

**P: `max_depth=6` con solo 3 features — ¿no es mucha profundidad?**

R: Con 3 features binarias, un árbol de profundidad 6 tiene hasta 2^6 = 64 hojas. Con ~200 muestras, es razonable pero hay que controlar overfitting con `min_samples_leaf=4`. Si el accuracy de test >> accuracy de train, hay overfitting.

### Sobre el modo auto

**P: ¿Por qué hay veces que el personaje se queda quieto y la bala lo mata?**

R: Pueden ser 2 causas: (1) el umbral de confianza 0.35 — si ninguna clase llega a 35%, devuelve -1 y no hace nada, y (2) la clase quieto (0) sigue siendo mayoritaria en el dataset, el árbol se sesga a predecir 0.

**P: El buffer de 3 frames — ¿no introduce latencia?**

R: Sí, unos 67ms (3 frames a 45 FPS). Es imperceptible para el juego. A cambio, filtra predicciones ruidosas. Es un tradeoff clásico: estabilidad vs reactividad.

**P: ¿Por qué el auto usa 45 frames de agachado y el manual 18?**

R: En manual, el jugador decide cuándo agacharse y la bala suele estar cerca, 18 frames (~0.4s) alcanzan. En auto, el modelo puede predecir agachado cuando la bala todavía está lejos; si usara 18 frames, el timer expiraría antes de que la bala pase, el modelo predeciría agachado otra vez, y se crearía un "rebote". 45 frames (~1s) cubren toda la aproximación.

### Sobre bugs clásicos

**P: ¿Qué aprendiste del bug de `modelo.classes_`?**

R: Que `predict_proba` devuelve probabilidades indexadas por posición (0, 1, 2), NO por etiqueta de clase. Si faltan clases en entrenamiento, la posición 1 puede corresponder a la clase 2. Usar `modelo.classes_[best_idx]` es la forma correcta de traducir.

**P: ¿Cómo debugging las predicciones del modelo en tiempo real?**

R: El HUD amarillo arriba a la izquierda muestra las probabilidades en vivo: `quieto:0.45 | salto:0.30 | agachado:0.25`. Si una clase siempre da 0.00, es que no existe en los datos de entrenamiento.

---

## 9. Para exponer — puntos clave

1. **Pipeline completo de ML** — no solo el modelo, todo el ciclo datos→entrenamiento→inferencia
2. **Calidad del dataset** — el mayor impacto. Registro por evento + submuestreo
3. **Desbalance de clases** — problema ubicuo en ML, cómo mitigarlo sin tocar el modelo
4. **Suavizado temporal** — las predicciones frame a frame son ruidosas; el filtro de 3 frames lo estabiliza
5. **Árbol vs MLP** — tradeoff interpretabilidad vs suavidad
6. **`modelo.classes_`** — bug clásico de sklearn, mapear índices a etiquetas reales
7. **Los modelos no existen en el vacío** — el orden de ejecución, los timers, los cooldowns, todo afecta el resultado final
