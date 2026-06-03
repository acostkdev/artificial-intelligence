// Ejemplo de uso del autocompletado RNN
// Crea un archivo .c, escribe el comentario de una funcion
// y presiona Ctrl+Shift+L en VS Code (o llama a la API)

// sumar_dos_enteros: recibe dos valores enteros y entrega la suma
int sumar_dos_enteros(int a, int b) {
    int resultado = a + b;
    return resultado;
}

// calcular_factorial: calcula el factorial de n usando recursion
int calcular_factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    int resultado = n * calcular_factorial(n - 1);
    return resultado;
}

// invertir_cadena: voltea el orden de los caracteres de una cadena
void invertir_cadena(char cadena[]) {
    int largo = longitud_cadena(cadena);
    for (int i = 0; i < largo / 2; i++) {
        char temporal = cadena[i];
        cadena[i] = cadena[largo - 1 - i];
        cadena[largo - 1 - i] = temporal;
    }
}
