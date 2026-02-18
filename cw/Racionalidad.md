## Problemas
1. Canibales
2. Esposos celosos
3. Ranas

## Secciones de la racionalidad
1. Medida de rendimiento para exito. La meta para resolver mejor el problema
2. Conocimiento del medio acumulado. Cómo está distribuido, matriz, entorno de un río donde debemos de atravesar y hay dos posiciones una lancha y así
3. Acciones que puede realizar el agente. Los tipos de acciones a realizar
4. Todas las acciones posibles que se pueden hacer. Es como el análisis de los estados, pero cuidar las reglas, porque sino hay loops


### 1. Canibales
Tres misioneros y tres caníbales tienen que cruzar un río con una barca que solo puede llevar como máximo dos personas, con la limitación de si existen misioneros presente en tierra, no puede pueden estar con un mayor número de caníbales, porque los caníbales se comerían a los misioneros.
1. **Medida:** Cruzar a todos en el menor número de movimientos
2. **Conoc medio:** Tenemos 3 misioneros y 3 canibales. Tenemos un río por cruzar. C > M no es posible de cualuquier lado. Tenemos una balsa con capacidad para 2 personas
3. **Acciones posibles:** Cruzar un canibal, cruzar un monje, Cruzar 2 canibales, Cruzar 2 monjes, Cruzar 1 y 1
4. **Secuencia de percepcion:**  

| Secuencia de percepciones | Acción                   |
| ------------------------- | ------------------------ |
| (CCCMMM) - 0 - (000000)   | Mover C y C a la derecha |
| (CCCMMM) - 0 - (000000)   | Mover C y M a la derecha |
| (00CMMM) - 0 - (CC0000)   | Mover C a la izquierda   |
| (0CC0MM) - 0 - (C00M00)   | Mover M a la izquierda   |
| ...                       | ...                      |



### 2. Esposos
Hay tres matrimonios, con la limitación que ninguna mujer puede estar en la presencia de otro hombre a no ser que su marido también esté presente. Bajo esta limitación no puede haber más mujeres que hombres en un banco del Río, porque si es así, alguna iría sin su marido. Pueden cruzar como máximo 2 personas juntas
1. **Medida:** Cruzar a todos en el menor número de movimientos
2. **Conoc medio:** Tenemos 3 esposos y 3 esposas (MaridoA,MaridoB,MaridoC) y (EsposaA,EsposaB,EsposaC).Tenemos un río por cruzar. EsposaX no puede estar sin MaridoX. Marido X puede estar sin EsposaX.
3. **Acciones posibles:** Cruzar marido, cruzar esposa, Cruzar EsposaX+MaridoX, Cruzar EsposaX+EsposaY
4. **Secuencia de percepcion:**  

| Secuencia de percepciones     | Acción      |
| ----------------------------- | ----------- |
| (MaMbMcEaEbEc) - 0 - (000000) | Mover Ea+Eb |
| (MaMbMcEaEbEc) - 0 - (000000) | Mover Ma+Ea |
| (MaMbMcEaEbEc) - 0 - (000000) | Mover Mb+Eb |
| (MaMbMcEaEbEc) - 0 - (000000) | Mover Mc+Ec |
| ...                           | ...         |

### 3. Ranas
Tenemos 7 espacios, hay 3 ranas de cada extremo, dejando la casilla 4 desocupada. Debemos de cruzar todas las ranas de un lado al otro. Las ranas no pueden saltar hacia atrás y una rana puede saltar como maximo a una. 
1. **Medida:** Cruzar a todos en el menor número de movimientos
2. **Conoc medio:** Tenemos 3 ranas izquierdas (Ri) tenemos 3 ranas derechas (Rd). Tenemos un espacio libre en el hueco numero 4. Y tenemos un total de 7 huecos, de los cuales 6 están ocupados ya. Rana no puede saltar hacia atrás. Rana puede saltar solo 1 rana por encima.
3. **Acciones posibles:** Rx salta hacia delante al hueco que está enfrente de ella. Rx salta a (Ry o Rx). x y y pueden representar izquierda o derecha.
4. **Secuencia de percepcion:**  

El arreglo de ranas y el espacio lo llamaremos "Arr"

| Secuencia de percepciones | Acción                      |
| ------------------------- | --------------------------- |
| (Ri,Ri,Ri,0,Rd,Rd,Rd)     | Mover Ri de Arr[2] a Arr[3] |
| (Ri,Ri,Ri,0,Rd,Rd,Rd)     | Mover Ri de Arr[1] a Arr[3] |
| (Ri,Ri,Ri,0,Rd,Rd,Rd)     | Mover Rdde Arr[4] a Arr[3]  |
|                           | Mover Rd de Arr[5] a Arr[3] |
| ...                       | ...                         |
