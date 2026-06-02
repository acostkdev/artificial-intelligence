import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import enable_gpu
    enable_gpu.enable()
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, LeakyReLU, GlobalAveragePooling2D,
    Multiply, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

CARPETA_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TAMANO = 64
SEMILLA = 42
BATCH_SIZE = 64
EPOCHS = 30
MIXUP_ALPHA = 0.2
LABEL_SMOOTH = 0.05

np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

CLASES = sorted(os.listdir(CARPETA_DATASET))
print(f"Clases: {CLASES}")

X, y = [], []
for idx, clase in enumerate(CLASES):
    carpeta = os.path.join(CARPETA_DATASET, clase)
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  {clase}: {len(archivos)}")
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            img = load_img(ruta, target_size=(TAMANO, TAMANO))
            X.append(img_to_array(img))
            y.append(idx)
        except Exception as e:
            print(f"  [WARN] {archivo}: {e}")

X = np.array(X, dtype=np.float32) / 255.0
y_cat = to_categorical(y, num_classes=len(CLASES))
print(f"\nTotal: {len(X)}, forma: {X.shape}")

train_X, test_X, train_y, test_y = train_test_split(
    X, y_cat, test_size=0.2, random_state=SEMILLA, stratify=y
)
train_X, val_X, train_y, val_y = train_test_split(
    train_X, train_y, test_size=0.2, random_state=SEMILLA,
    stratify=np.argmax(train_y, axis=1)
)

print(f"Train: {train_X.shape}")
print(f"Val:   {val_X.shape}")
print(f"Test:  {test_X.shape}")

class MixUpSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, alpha=MIXUP_ALPHA, shuffle=True):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def _randomizar_fondo(self, batch_X):
        B, H, W, C = batch_X.shape
        colores_fondo = np.array([
            [0.3, 0.7, 0.3], [0.2, 0.5, 0.2], [0.4, 0.8, 0.4],
            [0.3, 0.6, 0.8], [0.2, 0.4, 0.6], [0.5, 0.7, 0.9],
            [0.6, 0.5, 0.3], [0.5, 0.4, 0.2], [0.7, 0.6, 0.4],
            [1.0, 1.0, 1.0], [0.9, 0.9, 0.9],
        ], dtype=np.float32)
        for i in range(B):
            gray = np.mean(batch_X[i], axis=2)
            gx = np.abs(np.diff(gray, axis=1, append=0))
            gy = np.abs(np.diff(gray, axis=0, append=0))
            edge = ((gx + gy) > 0.06).astype(np.float32)
            for _ in range(4):
                d = edge.copy()
                d[:-1] = np.maximum(d[:-1], edge[1:])
                d[1:] = np.maximum(d[1:], edge[:-1])
                d[:, :-1] = np.maximum(d[:, :-1], edge[:, 1:])
                d[:, 1:] = np.maximum(d[:, 1:], edge[:, :-1])
                edge = d
            for _ in range(2):
                edge = (edge[:-2] + edge[1:-1] + edge[2:]) / 3
                edge = (edge[:, :-2] + edge[:, 1:-1] + edge[:, 2:]) / 3
                if edge.shape[0] < H:
                    edge = np.pad(edge, ((0, H - edge.shape[0]), (0, W - edge.shape[1])), mode='edge')
            color = colores_fondo[np.random.randint(len(colores_fondo))].reshape(1, 1, 3)
            batch_X[i] = batch_X[i] * edge[:, :, np.newaxis] + color * (1 - edge[:, :, np.newaxis])
        return batch_X

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_idx].copy()
        batch_y = self.y[batch_idx].copy()

        batch_X = self._randomizar_fondo(batch_X)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(batch_X))
            batch_X = lam * batch_X + (1 - lam) * batch_X[perm]
            batch_y = lam * batch_y + (1 - lam) * batch_y[perm]

        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_gen = MixUpSequence(train_X, train_y, BATCH_SIZE, alpha=MIXUP_ALPHA)
print(f"MixUpSequence — steps_per_epoch: {len(train_gen)}")

def se_block(x, reduction=8):
    channels = int(x.shape[-1])
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(channels // reduction, 4), activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return Multiply()([x, se])

inp = Input((TAMANO, TAMANO, 3))

x = Conv2D(32, (3, 3), padding='same')(inp)
x = LeakyReLU(negative_slope=0.1)(x)
x = BatchNormalization()(x)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = BatchNormalization()(x)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(128, (3, 3), padding='same')(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = BatchNormalization()(x)
x = se_block(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(256)(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
x = Dense(128)(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
out = Dense(len(CLASES), activation='softmax')(x)

modelo = Model(inp, out)
modelo.summary()

modelo.compile(
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=len(train_gen) * EPOCHS,
    alpha=1e-4
)
modelo.optimizer.learning_rate = cosine_decay

callbacks = [
    ModelCheckpoint(
        'modelo_animales_v2.keras', monitor='val_loss',
        save_best_only=True, verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    )
]

print("\nIniciando entrenamiento...")
historial = modelo.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=EPOCHS,
    validation_data=(val_X, val_y),
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = modelo.evaluate(test_X, test_y, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

preds = modelo.predict(test_X)
preds_clase = np.argmax(preds, axis=1)
test_clase = np.argmax(test_y, axis=1)
print(classification_report(test_clase, preds_clase, target_names=CLASES))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Train')
plt.plot(historial.history['val_accuracy'], label='Val')
plt.title('Precision'); plt.xlabel('Epoca'); plt.ylabel('Precision')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Train')
plt.plot(historial.history['val_loss'], label='Val')
plt.title('Perdida'); plt.xlabel('Epoca'); plt.ylabel('Perdida')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('grafica_entrenamiento_v2.png', dpi=150)

modelo.save('modelo_animales_v2.keras')
modelo.save('modelo_animales_v2.h5')
print("\nModelos guardados: modelo_animales_v2.keras / .h5")
print("=== ENTRENAMIENTO COMPLETADO ===")
