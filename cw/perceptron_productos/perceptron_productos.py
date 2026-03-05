import numpy as np
import matplotlib.pyplot as plt

# Datos: precio relativo, calidad percibida -> aceptado (1) o rechazado (0)
X = np.array([
    [0.5, 0.8],
    [0.6, 0.9],
    [0.7, 0.6],
    [0.4, 0.5],
    [0.3, 0.9],
    [0.8, 0.4]
])
y = np.array([1, 1, 0, 0, 1, 0])

# Los datos ya estan normalizados entre 0 y 1, no hace falta escalar

def funcion_activacion(z):
    return 1 if z >= 0 else 0

def entrenar_perceptron(X, y, tasa_aprendizaje=0.1, epocas=100):
    n_muestras, n_caracteristicas = X.shape
    w = np.random.randn(n_caracteristicas) * 0.1
    b = np.random.randn() * 0.1
    errores_por_epoca = []

    for epoca in range(epocas):
        errores = 0
        for i in range(n_muestras):
            z = np.dot(w, X[i]) + b
            prediccion = funcion_activacion(z)
            error = y[i] - prediccion
            if error != 0:
                w += tasa_aprendizaje * error * X[i]
                b += tasa_aprendizaje * error
                errores += 1
        errores_por_epoca.append(errores)
        if errores == 0:
            print(f"Convergio en epoca {epoca + 1}")
            break

    return w, b, errores_por_epoca

w, b, historial_errores = entrenar_perceptron(X, y, tasa_aprendizaje=0.1, epocas=100)
print(f"Pesos finales: w1 = {w[0]:.4f}, w2 = {w[1]:.4f}, b = {b:.4f}")

predicciones = np.array([funcion_activacion(np.dot(w, x) + b) for x in X])
print(f"Predicciones: {predicciones}")
print(f"Reales:       {y}")
print(f"Precision: {np.mean(predicciones == y) * 100:.1f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(historial_errores)
plt.xlabel('Epoca')
plt.ylabel('Errores')
plt.title('Errores por epoca')
plt.grid(True)

plt.subplot(1, 3, 2)
for i in range(len(X)):
    color = 'green' if y[i] == 1 else 'red'
    plt.scatter(X[i, 0], X[i, 1], c=color, s=100, edgecolors='black')
    etiqueta = 'Aceptado' if y[i] == 1 else 'Rechazado'
    plt.text(X[i, 0] + 0.02, X[i, 1] + 0.02, etiqueta, fontsize=9)

x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([funcion_activacion(np.dot(w, np.array([px, py])) + b) for px, py in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
if w[1] != 0:
    x_vals = np.array([0, 1])
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'b--', linewidth=2, label='Frontera de decision')
    plt.legend()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Precio relativo')
plt.ylabel('Calidad percibida')
plt.title('Frontera de decision')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.axis('off')
texto = "Partes del perceptron:\n\n"
texto += f"Entrada x1 = Precio relativo\n"
texto += f"Entrada x2 = Calidad percibida\n\n"
texto += f"Peso w1 = {w[0]:.4f}\n"
texto += f"Peso w2 = {w[1]:.4f}\n"
texto += f"Sesgo b = {b:.4f}\n\n"
texto += f"z = w1*x1 + w2*x2 + b\n"
texto += f"y = 1 si z >= 0, 0 si no\n\n"
texto += "Los datos son linealmente\nseparables? "
sep = all(funcion_activacion(np.dot(w, x) + b) == yi for x, yi in zip(X, y))
texto += f"{'Si' if sep else 'No'}"
plt.text(0.1, 0.9, texto, fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(__file__.replace('.py', '_resultados.png'))
plt.show()

print(f"\nLinealmente separables? {sep}")
print("\nPosibles mejoras:")
print("- Agregar mas datos de entrenamiento")
print("- Usar una funcion de activacion diferente (tanh, sigmoide)")
print("- Convertir a red multicapa para problemas no lineales")
print("- Probar con descenso de gradiente en lugar de regla del perceptron")
