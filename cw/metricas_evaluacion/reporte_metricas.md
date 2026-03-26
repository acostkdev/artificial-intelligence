# Metricas de Evaluacion (Ejercicios)

## Descripcion

En esta actividad resolvimos ejercicios de matrices de confusion y
metricas de evaluacion para modelos de clasificacion. Implementamos
todo en Python con numpy y matplotlib, calculando cada metrica de
forma manual y generando graficas para visualizar el rendimiento.

## Ejercicio 1: Calculo de metricas desde matriz de confusion

Partimos de una matriz de confusion con VP=45, FP=10, FN=15, VN=30
para un problema de deteccion de fraude. Calculamos cada metrica
manualmente con las formulas de los apuntes.

El accuracy quedo en 75% - parece decente pero no cuenta toda la
historia. La precision de 81.82% nos dice que cuando el modelo
predice fraude, la mayoria de las veces acierta. El recall de 75%
indica que de todos los fraudes reales, el modelo encuentra 3 de
cada 4. La especificidad tambien dio 75%, asi que los negativos
tambien se clasifican similar.

El F1-Score de 78.26% balancea precision y recall. Decidimos
calcular todo manualmente sin usar sklearn para practicar las
formulas, aunque en la vida real usariamos las funciones de la
libreria.

## Ejercicio 2: Interpretacion de modelo

Nos dieron un modelo con Accuracy 85%, Precision 40% y Recall 90%.
A primera vista el accuracy parece bueno, pero la precision baja
(40%) revela que hay muchos falsos positivos. Esto es tipico en
problemas con clases desbalanceadas donde el modelo aprende a
predecir la mayoria bien pero falla en la minoria.

Para el juego de bala y salto de actividades anteriores, donde
saltar cuando no se necesita es molesto pero no fatal, quizas
podria funcionar con un ajuste de umbral. Pero si el juego
castiga mucho los saltos falsos, no es buen modelo. Decidimos que
depende del costo de los falsos positivos vs. falsos negativos.

## Ejercicio 3: Mejoras para Recall bajo

Un modelo con Recall de 0.5 significa que solo detecta la mitad de
los positivos reales. Propusimos varias mejoras:

- Bajar el umbral de decision para capturar mas positivos (aunque
  aumenten los falsos positivos)
- Balancear las clases con tecnicas como SMOTE si el dataset esta
  desbalanceado
- Recolectar mas datos de entrenamiento, especialmente de la clase
  positiva
- Revisar las caracteristicas: quizas faltan variables relevantes
- Probar con un modelo mas complejo o con regularizacion diferente

No hay solucion unica, depende del contexto del problema.

## Curva ROC y AUC

Generamos una curva ROC sintetica con datos aleatorios para
visualizar la relacion entre la tasa de verdaderos positivos y la
tasa de falsos positivos. El AUC nos dio un valor decente, lo que
indica que el modelo separa bien las clases en general.

La curva ROC es util para comparar modelos independientemente del
umbral de decision. Mientras mas cerca de la esquina superior
izquierda, mejor. La linea diagonal representa un clasificador
aleatorio.

## Curva Precision-Recall para clases desbalanceadas

Para simular un escenario realista con clases desbalanceadas,
creamos un dataset con solo 10% de ejemplos positivos (50 de 500).
En estos casos la curva Precision-Recall es mas informativa que la
ROC, porque la ROC puede verse optimista cuando hay muchos
negativos.

Graficamos la curva y vimos como la precision cae cuando el recall
aumenta, que es el comportamiento esperado. Para clases
desbalanceadas esta grafica es indispensable.

## Matthews Correlation Coefficient (MCC)

Calculamos el MCC para la matriz del ejercicio 1 y lo comparamos
con clasificadores perfecto, aleatorio e inverso. El MCC tiene la
ventaja de considerar los cuatro valores de la matriz de confusion
y funciona bien incluso con clases desbalanceadas.

El rango va de -1 (inverso) a +1 (perfecto), con 0 indicando
aleatorio. Nuestro modelo quedo en un valor intermedio, lejos del
perfecto pero mucho mejor que aleatorio.

## Conclusion

Las metricas de evaluacion no deben verse de forma aislada. Un
accuracy alto puede esconder un modelo inutil si las clases estan
desbalanceadas. Para el proyecto del juego, probablemente
usariamos Precision y Recall como metricas principales, con la
curva ROC para seleccionar el umbral optimo. El MCC es util como
medida unificada cuando no queremos depender de una sola metrica.
