# Reporte: Juego Bala y Salto — Arbol de Decision vs MLP

## Bitacora de implementacion

### 1. Planteamiento del problema

Decidimos adaptar el juego Phaser a Python con pygame para que un personaje esquive pelotas saltando. La mecanica central consiste en generar pelotas que cruzan la pantalla de derecha a izquierda con velocidad y altura variables; el jugador debe saltar en el momento preciso para evitar la colision.

El objetivo principal fue implementar dos clasificadores que aprendieran a jugar en modo automatico a partir de datos recolectados en modo manual:
- Arbol de decision con profundidad maxima 5
- Perceptron multicapa con una capa oculta de 12 neuronas

### 2. Diseno del juego

Creamos una ventana de 800x600 pixeles con un suelo en la coordenada y=500. El jugador se representa como un rectangulo azul con una cabeza circular, y la pelota como un circulo rojo. Las pelotas aparecen cada 1800 ms con velocidad aleatoria entre -8 y -4 pixeles/frame y altura aleatoria en un rango de 90 pixeles sobre el suelo.

Las tres caracteristicas (features) que usamos para entrenar los modelos fueron:
- `velocidad_bala`: velocidad horizontal de la pelota (negativa, hacia la izquierda)
- `distancia_jugador`: distancia en pixeles entre la pelota y el jugador
- `altura_pelota`: posicion vertical de la pelota

Decidimos incluir la altura de la pelota como tercera caracteristica porque observamos que la dificultad del salto depende de si la pelota viene a nivel del suelo o mas arriba. El juego registra una decision (saltar=1, no saltar=0) por cada frame en que la pelota esta activa y el jugador esta en el suelo.

### 3. Modo manual y recoleccion de datos

El modo manual se activa con la tecla M. En este modo:
- El jugador controla el salto con la barra espaciadora
- Cada vez que el jugador presiona espacio, se registra la decision como salto=1
- Cuando hay pelota activa y el jugador no salta, se registra salto=0
- Los datos se acumulan en una lista de objetos `Muestra`

Jugamos varias partidas para recolectar aproximadamente 150-200 muestras con variedad de situaciones. Intentamos mezclar momentos de salto temprano, salto tardio y ausencia de salto para que los modelos tuvieran ejemplos balanceados.

### 4. Entrenamiento del arbol de decision

Configuramos el `DecisionTreeClassifier` con `max_depth=5` para limitar la complejidad y evitar sobreajuste. Dividimos los datos en 80% entrenamiento y 20% prueba.

El accuracy obtenido fue de alrededor de 0.85-0.92, lo que indica que el arbol logro capturar las reglas de decision de manera aceptable. Al observar el comportamiento en modo auto, notamos que el arbol tomaba decisiones bastante coherentes, aunque en ocasiones saltaba demasiado tarde cuando la pelota venia a alta velocidad.

### 5. Entrenamiento del MLP

Configuramos el `MLPClassifier` con una sola capa oculta de 12 neuronas, activacion ReLU, optimizador Adam y maximo 2000 iteraciones. Usamos los mismos datos de entrenamiento que con el arbol.

El accuracy del MLP fue ligeramente inferior, alrededor de 0.78-0.85. Posiblemente esto se deba a que no normalizamos los datos (las caracteristicas tienen escalas muy distintas: la velocidad esta en [-8, -4], la distancia en cientos de pixeles y la altura en decenas). En implementaciones futuras convendria aplicar `StandardScaler` antes de entrenar.

### 6. Modo automatico

En modo auto, cada frame el modelo consulta las caracteristicas actuales de la pelota y decide si saltar. El modo auto con arbol se activa con la tecla A, y con MLP con la tecla N.

Observamos que:
- El arbol de decision tiende a ser mas reactivo y tomar decisiones mas "binarias" (salta o no salta con alta confianza)
- El MLP muestra probabilidades intermedias (visibles en el HUD como `proba_salto`) y a veces duda mas antes de saltar
- Ambos modelos logran esquivar la mayoria de las pelotas, pero fallan en casos extremos (pelota muy rapida o muy cerca)

### 7. Problemas encontrados

1. **Datos desbalanceados**: al jugar, naturalmente pasamos mas tiempo sin saltar que saltando, lo que genero mas muestras de clase 0 que de clase 1.
2. **Falta de normalizacion**: el MLP funciona mejor con datos normalizados; no hacerlo probablemente afecto su accuracy.
3. **Learning rate fijo**: el MLPClassifier usa el learning rate por defecto de Adam (0.001). No lo ajustamos, lo cual es un aspecto mejorable.
4. **Colisiones en modo auto**: ocasionalmente ambos modelos fallan y la pelota golpea al jugador, lo que revela que los datos de entrenamiento aun son insuficientes o sesgados.

### 8. Exportacion de datos

Implementamos la exportacion a CSV (tecla C) que genera `datos_juego.csv` con las columnas `velocidad_bala`, `distancia_jugador`, `altura_pelota` y `salto`. Esto permite analizar los datos externamente con pandas o herramientas de visualizacion.

### 9. Conclusiones

Ambos modelos lograron aprender a jugar de manera aceptable, aunque con limitaciones evidentes:

- El arbol de decision es mas simple, interpretable y rapido de entrenar, ademas de que no requiere normalizacion de datos. Sin embargo, su naturaleza binaria lo hace menos flexible ante situaciones marginales.
- El MLP puede capturar relaciones mas complejas pero requiere mayor cuidado en el preprocesamiento y ajuste de hiperparametros.

La principal limitacion del experimento es que los datos se recolectaron de un solo jugador en una sesion corta, por lo que los modelos heredaron sus sesgos y patrones especificos. Para obtener modelos mas robustos se necesitarian mas datos de multiples jugadores y sesiones de juego mas largas.
