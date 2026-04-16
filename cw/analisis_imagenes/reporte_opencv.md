# Analisis de Imagenes con OpenCV — Reporte

## Que hicimos

Nueve ejercicios de vision por computadora usando OpenCV, desde carga
de imagenes hasta reconocimiento facial y template matching.

## Ejercicios

### 1. Carga de imagenes y propiedades

Cargamos una imagen con `imread`, vimos su shape (alto, ancho, canales)
y separamos los canales BGR con `split`. Mostramos cada canal por
separado y combinaciones como GRB (swap de canales).

### 2. Operadores puntuales

Recorrimos cada pixel de la imagen en gris con dos loops anidados. Si
el valor era mayor a 50 lo poniamos a 255 (blanco), si no a 0 (negro).
Resultado: una imagen binarizada con umbral fijo.

### 3. Modelos de color

Convertimos de BGR a RGB y separamos los canales. Creamos mascaras por
canal (R, G, B) combinando el canal correspondiente con una imagen de
ceros.

### 4. Segmentacion de color

Usamos `inRange()` en el espacio HSV para segmentar un rango de color.
Ajustando los umbrales bajo y alto en H (tono), S (saturacion) y V
(valor) podemos aislar objetos de cierto color. Usamos dos rangos
(rojo bajo y rojo alto) para cubrir el wrap-around del espacio HSV.

### 5. Haar Cascades

Usamos el clasificador pre-entrenado de OpenCV para detectar rostros.
`detectMultiScale` devuelve las cajas delimitadoras de las caras
detectadas. Dibujamos rectangulos alrededor de cada rostro en tiempo
real. El cascade file viene incluido en OpenCV.

### 6, 7, 8. Eigenfaces, Fisherfaces, LBPH

Los tres metodos de reconocimiento facial siguen el mismo patron:

1. **Entrenamiento**: Leer un dataset de carpetas (una por persona),
   cada imagen en gris, entrenar el reconocedor y guardar el modelo XML.
2. **Reconocimiento en vivo**: Cargar el modelo, usar Haar Cascade para
   detectar rostros, redimensionar cada rostro a 100x100, predecir con
   el reconocedor. Si la confianza supera un umbral, mostrar el nombre
   de la persona; si no, mostrar "Desconocido".

Diferencias:
- **Eigenfaces** usa PCA. Umbral de confianza: > 2800 (distancia mas
  alta = menos similar).
- **Fisherfaces** usa PCA + LDA. Umbral: < 500.
- **LBPH** usa patrones locales. Umbral: < 70.

### 9. Template Matching

Cargamos una imagen principal y una plantilla. Convertimos ambas a
gris, aplicamos `matchTemplate` con el metodo `TM_CCOEFF_NORMED`
(correlacion normalizada). `minMaxLoc` encuentra la mejor coincidencia
y dibujamos un rectangulo alrededor.

