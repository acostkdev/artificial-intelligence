# Clasificacion de Modelos de Autos con CNN

## Objetivo

Implementar una red neuronal convolucional para identificar cinco
modelos distintos de autos a partir de imagenes. El dataset se
construyo a partir del conjunto Stanford Cars, seleccionando 5 clases
con aproximadamente 40 imagenes cada una.

## Dataset

Los modelos elegidos fueron:

- Audi TT RS Coupe 2012
- BMW M3 Coupe 2012
- Ford F-150 Regular Cab 2012
- Lamborghini Aventador Coupe 2012
- Tesla Model S Sedan 2012

La actividad original pedia usar el archivo CNNriesgo de Dropbox del
profesor pero no se pudo localizar, asi que optamos por descargar
estos datos del dataset Stanford Cars desde Hugging Face usando la
libreria `datasets`. Las imagenes se organizaron en carpetas, una por
clase, y se cargaron con `image_dataset_from_directory` de Keras.

## Arquitectura de la CNN

Usamos una arquitectura secuencial con tres capas convolucionales:

1. Conv2D 32 filtros + MaxPooling
2. Conv2D 64 filtros + MaxPooling
3. Conv2D 128 filtros + MaxPooling
4. Capa densa de 128 neuronas con Dropout 0.5 para evitar sobreajuste
5. Capa de salida softmax con 5 neuronas

La funcion de activacion en las convolucionales es ReLU. Como
optimizador usamos Adam con entropia cruzada como funcion de perdida.

Las imagenes se redimensionaron a 128x128 y se normalizaron dividiendo
entre 255.

## Resultados

El modelo se entreno por 25 epocas con un split 80/20 para
entrenamiento y validacion. La grafica de precision muestra que
el modelo alcanzo una precision aceptable en validacion.

Ojo: con mas epocas probablemente mejore, pero tambien aumenta el
riesgo de sobreajuste porque el dataset es pequeno (solo ~200
imagenes en total). Una opcion seria aplicar aumento de datos (data
augmentation) para generar mas variedad.

## Conclusion

La CNN logra distinguir entre los 5 modelos aunque el dataset es
limitado. Para una version mas robusta convendria reunir mas imagenes
por clase o usar transfer learning con un modelo preentrenado como
MobileNet o ResNet.
