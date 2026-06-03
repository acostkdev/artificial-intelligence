import numpy as np
import os
import pickle
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

ruta_dataset = os.path.join(os.path.dirname(__file__), "dataset_c.txt")
with open(ruta_dataset, "r", encoding="utf-8") as f:
    CORPUS = f.read()

chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

ruta_mappings = os.path.join(os.path.dirname(__file__), "tokenizer_mappings.pkl")
with open(ruta_mappings, "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": VOCAB_SIZE}, f)

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

SEQ = np.array(encode(CORPUS), dtype=np.int64)
print("VOCAB_SIZE:", VOCAB_SIZE, "| caracteres en corpus:", len(CORPUS))
print("Funciones aprox:", CORPUS.count("//"))


block_size = 128
X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - block_size):
    X_rows.append(SEQ[i : i + block_size])
    Y_rows.append(SEQ[i + 1 : i + 1 + block_size])

X = np.stack(X_rows)
Y = np.stack(Y_rows)
print("X:", X.shape, "Y:", Y.shape)

embed_dim = 64
hidden = 128

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(block_size,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, embed_dim),
        tf.keras.layers.SimpleRNN(hidden, activation="tanh", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE)),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.summary()

epochs = 150
batch_size = 32

ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_rnn.h5")
checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint_epoch_{epoch:03d}.h5")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=False, save_freq="epoch", verbose=0
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=15, min_delta=0.001, verbose=1
    ),
]

history = model.fit(
    X,
    Y,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=2,
)

print("epocas entrenadas:", len(history.history["loss"]))
print("perdida inicial:", round(float(history.history["loss"][0]), 4))
print("perdida final:", round(float(history.history["loss"][-1]), 4))

model.save(ruta_modelo)
print("Modelo guardado en:", ruta_modelo)

import glob
for f in glob.glob(os.path.join(os.path.dirname(__file__), "checkpoint_epoch_*.h5")):
    os.remove(f)
print("Checkpoints temporales eliminados")


rng_completar = np.random.default_rng(42)

def muestrear(logits, temperatura=0.05, top_k=3):
    z = logits / max(temperatura, 1e-6)
    z = z - z.max()
    if top_k > 0:
        indices = np.argpartition(z, -top_k)[-top_k:]
        mask = np.full_like(z, -np.inf)
        mask[indices] = z[indices]
        z = mask
    e = np.exp(z)
    p = e / e.sum()
    return int(rng_completar.choice(len(p), p=p))

def completar(prompt, max_nuevos=150):
    ids = encode(prompt)
    prompt_len = len(ids)
    for _ in range(max_nuevos):
        x = np.array(ids[-block_size:], dtype=np.int64)
        if x.shape[0] < block_size:
            pad = np.full(block_size - x.shape[0], ids[0], dtype=np.int64)
            x = np.concatenate([pad, x])
        x = x.reshape(1, block_size)
        logits = model(x, training=False).numpy()[0, -1, :]
        idx = muestrear(logits)
        ids.append(idx)
        if decode(ids[prompt_len:]).endswith("}\n\n"):
            break
    return decode(ids)


print("\n--- Prueba de generacion ---")
print(completar("// sumar", max_nuevos=120))

print("\n--- Prueba con inicio de funcion ---")
print(completar("// calcular", max_nuevos=150))
