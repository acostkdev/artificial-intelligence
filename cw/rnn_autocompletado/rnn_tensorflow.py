import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

CORPUS = r'''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distancia_origen(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

for i in range(10):
    print(i, fibonacci(i))
'''

chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

SEQ = np.array(encode(CORPUS), dtype=np.int64)
print("VOCAB_SIZE:", VOCAB_SIZE, "| caracteres en corpus:", len(CORPUS))

block_size = 32
X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - block_size):
    X_rows.append(SEQ[i : i + block_size])
    Y_rows.append(SEQ[i + 1 : i + 1 + block_size])

X = np.stack(X_rows)
Y = np.stack(Y_rows)
print("X:", X.shape, "Y:", Y.shape)

embed_dim = 48
hidden = 64

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

epochs = 120
batch_size = 16

history = model.fit(
    X,
    Y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=0,
)

print("epocas:", len(history.history["loss"]))
print("perdida inicial:", round(float(history.history["loss"][0]), 4))
print("perdida final:", round(float(history.history["loss"][-1]), 4))

def complete(prompt, max_new=120, temperature=0.75):
    ids = encode(prompt)
    rng = np.random.default_rng(42)
    for _ in range(max_new):
        x = np.array(ids[-block_size:], dtype=np.int64)
        if x.shape[0] < block_size:
            pad = np.full(block_size - x.shape[0], ids[0], dtype=np.int64)
            x = np.concatenate([pad, x])
        x = x.reshape(1, block_size)
        logits = model(x, training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))
    return decode(ids)

print("\n--- Generacion TF ---")
print(complete("def fac", max_new=90, temperature=0.75))

print("\n--- Generacion TF (class Punto) ---")
print(complete("class Pu", max_new=80, temperature=0.7))
