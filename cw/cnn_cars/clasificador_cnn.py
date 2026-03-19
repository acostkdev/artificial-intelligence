import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

DIR_DATOS = os.path.join(os.path.dirname(__file__), "data")
ALTO = 128
ANCHO = 128
TAM_LOTE = 16
EPOCAS = 25

entrenamiento = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_DATOS,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(ALTO, ANCHO),
    batch_size=TAM_LOTE
)

prueba = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_DATOS,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(ALTO, ANCHO),
    batch_size=TAM_LOTE
)

nombres_clases = entrenamiento.class_names
print("Clases:", nombres_clases)

autotune = tf.data.AUTOTUNE
entrenamiento = entrenamiento.cache().shuffle(1000).prefetch(buffer_size=autotune)
prueba = prueba.cache().prefetch(buffer_size=autotune)

modelo = models.Sequential([
    layers.Rescaling(1./255, input_shape=(ALTO, ANCHO, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(nombres_clases), activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

historial = modelo.fit(
    entrenamiento,
    validation_data=prueba,
    epochs=EPOCAS
)

perdida, precision = modelo.evaluate(prueba)
print(f"\nPrecision en prueba: {precision:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Prueba')
plt.title('Precision por epoca')
plt.xlabel('Epoca')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Prueba')
plt.title('Perdida por epoca')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "graficas_entrenamiento.png"))
print("Grafica guardada como graficas_entrenamiento.png")

imagenes_prueba, etiquetas_prueba = next(iter(prueba))
predicciones = modelo.predict(imagenes_prueba)
predichas = np.argmax(predicciones, axis=1)

plt.figure(figsize=(12, 8))
for i in range(min(12, len(imagenes_prueba))):
    plt.subplot(3, 4, i+1)
    plt.imshow(imagenes_prueba[i].numpy().astype("uint8"))
    real = nombres_clases[etiquetas_prueba[i]]
    pred = nombres_clases[predichas[i]]
    color = "green" if real == pred else "red"
    plt.title(f"Real: {real}\nPred: {pred}", color=color, fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "predicciones_ejemplo.png"))
print("Predicciones guardadas como predicciones_ejemplo.png")
