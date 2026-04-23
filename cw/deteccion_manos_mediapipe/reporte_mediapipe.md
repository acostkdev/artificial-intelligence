# Deteccion de Manos y Gestos con MediaPipe — Reporte

## Que hicimos

Cuatro ejercicios con MediaPipe para deteccion de manos, gestos y
rostro en tiempo real.

## Ejercicios

### 1. Deteccion de manos (21 puntos clave)

Usamos `mp.solutions.hands` para detectar las manos en el feed de la
camara. MediaPipe regresa 21 landmarks por mano (muneca, punta de cada
dedo, falanges). Con `mp_drawing.draw_landmarks` dibujamos los puntos y
sus conexiones sobre el frame.

### 2. Reconocimiento de letras (A, B, C)

Partiendo de los 21 landmarks, calculamos la posicion de las puntas de
los dedos y las distancias entre ellas. Con reglas heuristicas
clasificamos la letra segun la configuracion de la mano:

- **A**: pulgar e indice cerca (< 30px), indice y medio separados (> 50px)
- **B**: todos los dedos estirados, punta del indice mas arriba que el
  medio y asi sucesivamente
- **C**: pulgar e indice separados (> 50px)

El reconocimiento no es perfecto porque las heuristicas son sencillas y
dependen de la distancia a la camara.

### 3. Face Mesh

Usamos `mp.solutions.face_mesh` para detectar 468 puntos faciales.
Probamos dos modos de dibujo: `FACEMESH_TESSELATION` (malla completa) y
`FACEMESH_CONTOURS` (solo contorno de ojos, cejas, labios). Con
`cv2.flip` ponemos la imagen en espejo para que los movimientos sean
naturales.

### 4. Mascara animada

Con `mp.solutions.face_detection` obtenemos la caja delimitadora de la
cara. Cargamos un PNG con transparencia (canal alpha) y lo redimensionamos
al tamano de la cara detectada. La funcion `overlay_mask` mezcla el PNG
con el frame usando el canal alpha como peso. El resultado es una
mascara que sigue la cara en tiempo real.

## Problemas

- El reconocimiento de letras es sensible a la distancia. A menos de
  50cm de la camara las distancias cambian mucho.
- Face Mesh pesa en CPU, se siente lag si no hay GPU.
- Para la mascara se necesita un PNG con transparencia. Sin el archivo
  el programa falla al hacer `cv2.imread`.
- En la mascara, si la cara esta cerca del borde la caja delimitadora
  puede salirse del frame y causar error de dimensiones.
