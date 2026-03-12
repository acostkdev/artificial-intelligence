import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

X = np.array([
    [0.9, 0.8, 0.2],
    [0.7, 0.6, 0.5],
    [0.4, 0.4, 0.8],
    [0.8, 0.9, 0.3],
    [0.5, 0.7, 0.6],
    [0.3, 0.5, 0.9]
])

y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

etiquetas = ['Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto']

modelo = Sequential([
    Dense(4, input_dim=3, activation='relu'),
    Dense(3, activation='softmax')
])

modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

modelo.summary()

historial = modelo.fit(X, y, epochs=200, verbose=0)

perdida, precision = modelo.evaluate(X, y, verbose=0)
print(f"\nPrecision en entrenamiento: {precision:.4f}")

print("\nPredicciones:")
predicciones = modelo.predict(X)
for i in range(len(X)):
    pred = np.argmax(predicciones[i])
    real = np.argmax(y[i])
    print(f"  Cliente {i+1}: Real={etiquetas[real]}, Pred={etiquetas[pred]}, {'OK' if pred == real else 'X'}")

pesos_y_sesgos = modelo.get_weights()
print(f"\nPesos capa oculta: {pesos_y_sesgos[0].shape}")
print(f"Sesgos capa oculta: {pesos_y_sesgos[1].shape}")
print(f"Pesos capa salida: {pesos_y_sesgos[2].shape}")
print(f"Sesgos capa salida: {pesos_y_sesgos[3].shape}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(historial.history['loss'])
plt.title('Perdida durante entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(historial.history['accuracy'])
plt.title('Precision durante entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Precision')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.axis('off')
texto = "Arquitectura:\n\n"
texto += "Entrada: 3 neuronas\n"
texto += "  - Historial pagos\n"
texto += "  - Ingresos mensuales\n"
texto += "  - Relacion deuda/ingreso\n\n"
texto += "Capa oculta: 4 neuronas (ReLU)\n\n"
texto += "Salida: 3 neuronas (Softmax)\n"
texto += "  - [1,0,0] = Riesgo Bajo\n"
texto += "  - [0,1,0] = Riesgo Medio\n"
texto += "  - [0,0,1] = Riesgo Alto\n\n"
texto += "Optimizador: Adam\n"
texto += "Perdida: Cross-entropy"
plt.text(0.1, 0.95, texto, fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(__file__.replace('.py', '_resultados.png'))
plt.show()

print("\nSeparabilidad lineal? Los datos de 3 clases en 3D no son")
print("linealmente separables con una linea recta, por eso usamos")
print("una red con capa oculta y activacion no lineal (ReLU).")
print("\nPosibles mejoras:")
print("- Mas datos de clientes para generalizar mejor")
print("- Mas neuronas en la capa oculta")
print("- Validacion cruzada para evitar sobreajuste")
print("- Normalizar las caracteristicas si no lo estuvieran")
