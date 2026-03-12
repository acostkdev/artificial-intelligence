# Clasificacion de Clientes con Red Multicapa

## Descripcion

Implementamos una red neuronal multicapa con Keras para clasificar
clientes en tres categorias de riesgo financiero: bajo, medio y alto.
Usamos tres caracteristicas de entrada: historial de pagos, ingresos
mensuales y relacion deuda-ingreso.

## Datos

Son 6 muestras con 3 caracteristicas cada una. La salida es one-hot:
[1,0,0] para riesgo bajo, [0,1,0] para medio y [0,0,1] para alto. Los
datos ya vienen normalizados entre 0 y 1.

## Arquitectura

La red tiene una capa oculta con 4 neuronas y activacion ReLU, y una
capa de salida con 3 neuronas y softmax para obtener probabilidades de
cada clase. Usamos Adam como optimizador y entropia cruzada como
funcion de perdida.

## Resultados

La red aprendio a clasificar correctamente las 6 muestras. A diferencia
del perceptron simple de la actividad anterior, aqui los datos no son
linealmente separables en 3 clases, por eso necesitamos la capa oculta
con activacion no lineal.

## Mejoras posibles

Con un dataset tan pequeno no podemos esperar buena generalizacion. Lo
ideal seria tener decenas o cientos de clientes etiquetados. Tambien se
podria ajustar el numero de neuronas en la capa oculta y agregar
regularizacion para evitar sobreajuste.
