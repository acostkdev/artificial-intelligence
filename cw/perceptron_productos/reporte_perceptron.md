# Clasificacion de Productos con Perceptron Simple

## Descripcion

Implementamos un perceptron simple desde cero con numpy para clasificar
productos como aceptados o rechazados segun su precio relativo y
calidad percibida. Usamos solo 6 muestras de entrenamiento con los
datos que vienen en los apuntes.

## Datos

Los criterios de aceptacion originales son: precio <= 0.6 y calidad >=
0.7. Con eso los datos quedaron etiquetados como se esperaba. Ya
venian normalizados entre 0 y 1 asi que no hicimos escalado extra.

## Implementacion

El perceptron usa una funcion de activacion escalon (step) que
devuelve 1 si z >= 0 y 0 en otro caso. Los pesos se actualizan con la
regla del perceptron: w += tasa * error * x. Entrenamos por hasta 100
epocas o hasta que no haya errores.

Los pesos encontrados fueron w1 y w2 para las caracteristicas y b para
el sesgo. La frontera de decision es una linea recta que separa las
dos clases.

## Resultados

El perceptron converge sin errores porque los datos son linealmente
separables - se puede trazar una linea recta que separe aceptados de
rechazados. La precision es del 100% en los datos de entrenamiento.

## Mejoras posibles

Con datos mas complejos o no lineales, este perceptron simple no
funcionaria. Se podria mejorar usando una red multicapa, funciones de
activacion como sigmoide o ReLU, y descenso de gradiente en lugar de
la regla clasica del perceptron.
