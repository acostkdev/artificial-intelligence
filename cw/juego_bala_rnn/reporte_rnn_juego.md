# Reporte — Juego Bala y Salto con RNN

## Introduccion

Hemos adaptado el juego de bala y salto que originalmente usaba un MLP
(Multilayer Perceptron) para incorporar redes recurrentes (RNN). La idea
principal fue aprovechar la capacidad de las RNN para procesar secuencias
temporales, ya que el movimiento de la bala hacia el jugador es inherentemente
una secuencia: la distancia disminuye frame a frame y la velocidad puede variar.
Con el MLP clasico cada decision se toma viendo un solo instante, mientras que
con una RNN podemos considerar la historia completa de los ultimos N frames.

## Implementacion

Partimos del codigo base del juego MLP que ya teniamos en
`project/project1/pygames/juego_mpl.py` y lo extendimos en la clase
`JuegoRNN`. Los cambios principales fueron:

- Agregamos un buffer de secuencia (`deque` con maxlen configurable) que
  almacena los ultimos N frames. Por frame guardamos velocidad de la bala,
  distancia al jugador y si el jugador esta saltando o no.
- Cuando el buffer se llena por primera vez, guardamos la secuencia completa
  como una muestra de entrenamiento para la RNN. Como el buffer es una ventana
  deslizante, cada frame nuevo genera una secuencia nueva (con solapamiento).
- Implementamos `crear_modelo_rnn(tipo)` que construye un modelo Keras con
  una capa recurrente (SimpleRNN, GRU o LSTM segun el parametro), seguida de
  una capa densa de 16 neuronas con ReLU, dropout de 0.2 y una salida sigmoide
  para clasificacion binaria (saltar o no saltar).
- La funcion `entrenar_rnn()` prepara los datos llamando a
  `preparar_datos_rnn()`, que convierte las secuencias en tensores 3D de forma
  (num_secuencias, longitud_secuencia, 2) y normaliza con StandardScaler.
  Dividimos en 80% entrenamiento y 20% prueba, entrenamos por 60 epochs con
  batch size de 32, y evaluamos accuracy, precision, recall y F1.
- Para el modo automatico, implementamos `decision_auto_rnn()` que toma el
  buffer actual (debe tener longitud_secuencia frames), normaliza y predice
  con el modelo entrenado. Si la probabilidad de salto es mayor o igual a 0.5,
  el jugador salta.

Mantuvimos el MLP original con 3 clases (quieto, salto, agachado) usando
MLPClassifier de sklearn con capas ocultas (8, 4) y 300000 iteraciones maximas.
Esto nos permite comparar ambos enfoques desde el mismo juego.

## Actividad 1: Comparacion MLP vs RNN

Entrenamos ambos modelos con los mismos datos recolectados jugando
manualmente. Recolectamos aproximadamente 150 muestras para MLP (con
submuestreo de frames quietos cada 30 frames) y alrededor de 200 secuencias
para RNN (con longitud de secuencia de 10). Los resultados fueron los
siguientes:

| Modelo | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| MLP (3 clases) | 0.842 | 0.831 | 0.842 | 0.836 |
| RNN (GRU, seq=10) | 0.801 | 0.778 | 0.801 | 0.789 |

El MLP obtuvo mejores metricas numericas en este experimento. Sin embargo,
esto hay que tomarlo con reserva por varias razones. Primero, el MLP predice
3 clases mientras que la RNN solo predice 2 (saltar o no), por lo que las
metricas no son directamente comparables. Segundo, la RNN necesita mas datos
para entrenarse adecuadamente porque cada secuencia de 10 frames contiene
menos informacion independiente que 10 muestras individuales. Tercero, el
MLP tiene acceso a la altura de la bala como tercera caracteristica, mientras
que la RNN solo usa velocidad y distancia.

En modo automatico, observamos que el MLP tiende a reaccionar de forma mas
brusca: a veces salta demasiado tarde o demasiado temprano porque solo ve el
frame actual. La RNN, al tener contexto temporal, parece tomar decisiones mas
suaves y consistentes: si la distancia viene disminuyendo de manera constante,
la RNN anticipa el salto antes. Sin embargo, la RNN tambien tiene falsos
positivos: a veces "cree" que la bala se acerca cuando en realidad se esta
alejando (despues de un reset de bala, por ejemplo).

## Actividad 2: Experimentos con longitud de secuencia

Probamos tres longitudes de secuencia: 5, 10 y 20 frames, todas con GRU y
60 epochs.

| Longitud | Accuracy | Precision | Recall | F1 | Tiempo (s) |
|----------|----------|-----------|--------|-----|------------|
| 5 | 0.778 | 0.750 | 0.778 | 0.764 | 4.2 |
| 10 | 0.801 | 0.778 | 0.801 | 0.789 | 6.8 |
| 20 | 0.815 | 0.800 | 0.815 | 0.807 | 11.5 |

La longitud de 5 frames entrena mas rapido pero captura menos contexto.
Con 5 frames la RNN apenas alcanza a ver como la bala se acerca un poco,
lo que no es suficiente para tomar decisiones informadas. Con 10 frames
el rendimiento mejora notablemente. Con 20 frames obtenemos la mejor
precision, aunque el tiempo de entrenamiento casi se duplica respecto a 10.

En modo automatico, la diferencia mas notable fue con 5 frames: la RNN
tomaba decisiones erraticas porque no alcanzaba a ver el patron completo
de la bala acercandose. Con 20 frames el comportamiento era muy suave pero
a veces tardaba en reaccionar a cambios repentinos de velocidad (porque el
promedio de los 20 frames "suaviza" mucho la senal).

Consideramos que 10 frames es un buen balance: da suficiente contexto sin
ser demasiado lento.

## Actividad 3: Comparacion de tipos de RNN

Comparamos SimpleRNN, GRU y LSTM con longitud de secuencia fija de 10 y
60 epochs.

| Tipo | Accuracy | Precision | Recall | F1 | Tiempo (s) |
|------|----------|-----------|--------|-----|------------|
| SimpleRNN | 0.771 | 0.745 | 0.771 | 0.758 | 5.1 |
| GRU | 0.801 | 0.778 | 0.801 | 0.789 | 6.8 |
| LSTM | 0.795 | 0.770 | 0.795 | 0.782 | 9.2 |

SimpleRNN fue la mas rapida pero con menor rendimiento, lo cual era
esperado: la RNN vanilla sufre del problema del gradiente que desaparece,
y con secuencias de 10 frames ya empieza a mostrar sus limitaciones. GRU
dio el mejor balance entre velocidad y rendimiento, como mencionan los
apuntes. LSTM fue la mas lenta y sus metricas fueron ligeramente inferiores
a GRU en este experimento, probablemente porque con solo 10 frames no
necesitamos la memoria a largo plazo que ofrece LSTM.

En modo automatico, SimpleRNN se comportaba de forma mas nerviosa, con
saltos a destiempo. GRU y LSTM fueron muy similares en comportamiento,
aunque GRU respondia ligeramente mas rapido.

## Actividad 4: Analisis de secuencias

Exportamos las secuencias a CSV y las visualizamos con matplotlib. Al
graficar distancia vs tiempo para secuencias donde el jugador salta versus
donde no salta, observamos patrones claros:

- Las secuencias de salto tipicamente muestran una distancia que disminuye
  hasta aproximadamente 200-300 pixeles, momento en el que el jugador salta.
- Las secuencias de no salto generalmente corresponden a balas que pasan
  a mayor altura (altura_bala=1) o balas que van muy rapido y el jugador
  no alcanza a reaccionar.
- Tambien hay secuencias de no salto donde la bala va muy lenta y el
  jugador decide esperar.

Un patron interesante que identificamos es que cuando la velocidad es
alta (mayor a 15 en valor absoluto) y la distancia es menor a 400, la
probabilidad de salto aumenta drasticamente. Con la RNN, este patron se
captura naturalmente a traves de la secuencia, mientras que el MLP solo
ve el frame actual y no puede distinguir entre "una bala que viene rapido
desde lejos" y "una bala que viene lento desde cerca".

## Conclusiones

La RNN ofrece ventajas conceptuales importantes para este problema porque
el movimiento de la bala es temporal por naturaleza. Sin embargo, en la
practica con datos limitados, el MLP puede dar metricas comparables o
incluso superiores. La RNN brilla cuando hay suficientes datos de secuencia
y cuando el patron temporal es relevante. Para este juego en particular,
recomendariamos GRU con longitud de secuencia de 10 frames como el mejor
balance entre rendimiento y velocidad.

Quedan varias mejoras pendientes: aumentar el numero de epochs, probar
con mas neuronas en la capa recurrente, usar batch normalization, o
incluso probar RNN apiladas (stacked RNN). Tambien seria interesante
entrenar la RNN con 3 clases en vez de binaria para incluir la accion de
agacharse.
