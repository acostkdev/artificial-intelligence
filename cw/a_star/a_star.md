# A* - Busqueda de Caminos

## Que hicimos

Implementamos el algoritmo A* para encontrar la ruta mas corta entre dos puntos
dentro de una cuadricula de 10x10. Usamos Pygame para la parte visual, de esta
forma podemos ver paso a paso como el algoritmo va explorando los nodos hasta
encontrar el camino optimo.

## Fundamento

El algoritmo A* funciona con una funcion de evaluacion f(n) = g(n) + h(n). Aqui
g(n) es el costo que hemos acumulado desde el nodo inicial hasta el nodo actual,
y h(n) es una estimacion de lo que falta para llegar al destino. Usamos distancia
Manhattan como heuristica porque estamos en una cuadricula con movimiento en 4
direcciones solamente. La distancia Manhattan basicamente suma la diferencia
horizontal y vertical entre dos puntos, lo cual es admisible para este caso
porque jamas sobreestima el costo real del camino.

Hemos utilizado un heap (cola de prioridad) para ir extrayendo siempre el nodo
con el menor valor de f. Esto nos asegura que el algoritmo va expandiendo primero
las rutas mas prometedoras. Para la gestion de nodos ya procesados manejamos una
lista cerrada que se va llenando a medida que el algoritmo avanza. Al final se
imprime esta lista cerrada en la terminal para que podamos ver exactamente cuales
nodos fueron explorados.

### Detalle medio contraintuitivo

La heuristica tiene un efecto bien importante en el comportamiento del algoritmo.
Si h vale 0, A* se comporta practicamente como Dijkstra y explora todo sin
priorizar. Si h es muy grande, se vuelve greedy y puede encontrar un camino que
no sea el optimo. Para esta cuadricula la distancia Manhattan queda en un punto
balanceado, ni muy ni muy relajada. O sea, funciona.

## Como se uso

Para usar el programa se corre con `python cw/a_star/a_star.py`. Se abre una
ventana con una cuadricula de 100 nodos en blanco. Con click izquierdo colocamos
el nodo de inicio (naranja), luego el nodo final (morado), y despues podemos
poner paredes (negro) para obstaculizar el paso. Con click derecho sobre cualquier
nodo lo regresamos a su estado original. Cuando ya tenemos el escenario listo,
presionamos ESPACIO para que el algoritmo empiece a trabajar. Durante la ejecucion
los nodos en la lista abierta (por explorar) se pintan de verde, los que ya se
exploraron se vuelven rojos, y al final el camino encontrado se marca en azul.
Presionando R se reinicia toda la cuadricula.

### Codigo y estructura

La clase Nodo es la base de todo. Cada nodo guarda su posicion, color, tamaño y
una lista de vecinos. El metodo `actualizar_vecinos` se encarga de detectar los
4 nodos adyacentes (arriba, abajo, izquierda, derecha) recorriendo renglon y
columna con operaciones aritmeticas simples.

El algoritmo esta en la funcion `algoritmo_a_star`. Recibe una funcion callback
para dibujar, la grid completa, el nodo inicio y el nodo fin. Usamos un heap
para el open set y permitimos entradas repetidas. Esto es medio ineficiente
porque si encontramos un mejor camino a un nodo que ya estaba en la lista abierta,
lo volvemos a meter al heap sin eliminar la entrada anterior. Cuando sacamos un
nodo del heap, si ya tiene color rojo significa que ya lo procesamos y lo saltamos.
Es un truco comun para no tener que implementar decrease-key, y para una cuadricula
de 10x10 no hay problema de rendimiento.

Los scores g y f se almacenan en diccionarios donde todas las entradas empiezan
en infinito y se van actualizando conforme encontramos mejores caminos.

```python
# Fragmento del nucleo del algoritmo
if temp_g_score < g_score[vecino]:
    came_from[vecino] = actual
    g_score[vecino] = temp_g_score
    f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
    heapq.heappush(open_set, (f_score[vecino], contador, vecino))
```

La reconstruccion del camino se hace con la funcion `reconstruir_camino`, que
parte del nodo final y va hacia atras usando el diccionario came_from, pintando
cada nodo de azul.

## Posibles mejoras

- No tenemos control de velocidad, el algoritmo corre al ritmo del ciclo de
  Pygame. Se podria agregar un delay configurable.
- El movimiento es solo en 4 direcciones, no considera diagonales. Si agregaramos
  movimiento en 8 direcciones la heuristica tendria que ser euclidiana.
- El open_set con entradas duplicadas gasta memoria innecesariamente, pero para
  este caso no es problema.
- No hay indicacion visual de cuales nodos se intentaron cuando no se encuentra
  un camino, solo aparece el mensaje en terminal.

## Referencia

Nos basamos en el cascaron de Pygame que viene en los apuntes (Visualizacion de
Nodos, Cascaron A*). La implementacion del algoritmo sigue el pseudocodigo
estandar de A* con distancia Manhattan.
