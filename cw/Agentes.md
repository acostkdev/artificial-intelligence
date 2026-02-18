Sensores que le permiten actuar.
Termino **percepcion**, el agente puede recibir entradas para ejecutar una accion
**Secuencia de percepcion**, historial de percepciones. Todo lo que estamos captando

Un agente inteligente se basa en su secuencia de percepcion
Debemos de tener un fundamento matemático para que se describa la forma de actuar del agente, lo cual es alcanzable mediante el análisis de la secuencia de percepción

![[Pasted image 20260209103422.png]]
Ej secuencias de percepcion
## Planteamiento
1. Las reglas para poder ganar el 5 en línea (los algoritmos)
2. Tipos de movimientos en base al tiro del agente rival
3. Qué estrategias seguir dependiendo del tiro inicial. Si voy yo primero o el agente
Analizar todo lo que debemos de hacer para que un agente no nos gane en ajedrez o algún juego


![[Pasted image 20260209102816.png]]

## Analizando abstraccion de 3 fichas en tablero 5x5



## Programar un gato que nunca pueda perder

![[Pasted image 20260209104404.png]]
## Análisis
Primero que nada, tenemos que saber los fundamentos
- 3 en linea para poder ganar
- Dependiendo del tiro hay más o menos opciones de gane
### Primer turno mío
1. Gane. Al poner en una esquina y que el coloque ficha en un lugar que no sea el centro, buscamos un extremo que nos haga línea de 3 y ya ganamos
2. Gane. Al poner en el centro y que el coloque en cualquiera de los medios. Ponemos una esquina y ya ganamos. Si el pone esquina, vamos a la contraria y si el pone al lado de esa esquina, le tapas y ya ganas
Ahora si soy el turno 2 ahi está pero a la inversa pero para empatar.

Como segundo jugador la única manera de tener triunfo asegurado es que el primero se equivoque y aprovechar.