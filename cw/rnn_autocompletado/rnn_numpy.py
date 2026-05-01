import numpy as np

rng = np.random.default_rng(42)

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
block_size = 32

X_rows, Y_rows = [], []
for i in range(0, len(SEQ) - block_size):
    X_rows.append(SEQ[i : i + block_size])
    Y_rows.append(SEQ[i + 1 : i + 1 + block_size])

X_np = np.stack(X_rows)
Y_np = np.stack(Y_rows)

H, d_emb = 64, 48
scale_e = 0.02
scale_rnn = 1.0 / np.sqrt(H + d_emb)

def init_params(V, H, d, rng):
    E = rng.normal(0, scale_e, size=(V, d))
    Wxh = rng.normal(0, scale_rnn, size=(H, d))
    Whh = rng.normal(0, scale_rnn, size=(H, H))
    bh = np.zeros(H)
    Why = rng.normal(0, scale_rnn, size=(V, H))
    by = np.zeros(V)
    return {"E": E, "Wxh": Wxh, "Whh": Whh, "bh": bh, "Why": Why, "by": by}

params_np = init_params(VOCAB_SIZE, H, d_emb, rng)

def forward_sequence(x_indices, p, h0=None):
    T = len(x_indices)
    V = p["E"].shape[0]
    H = p["bh"].shape[0]
    if h0 is None:
        h0 = np.zeros(H)

    h_list = []
    pre_list = []
    emb_list = []
    logits_list = []
    h_prev = h0.copy()

    for t in range(T):
        idx = int(x_indices[t])
        emb = p["E"][idx].copy()
        pre = p["Wxh"] @ emb + p["Whh"] @ h_prev + p["bh"]
        h = np.tanh(pre)
        logits = p["Why"] @ h + p["by"]

        emb_list.append(emb)
        pre_list.append(pre)
        h_list.append(h)
        logits_list.append(logits)
        h_prev = h

    logits = np.stack(logits_list, axis=0)
    return logits, {"h": h_list, "pre": pre_list, "emb": emb_list, "x": x_indices}

def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_from_logits(logits, targets):
    probs = softmax_rows(logits)
    T = logits.shape[0]
    nll = 0.0
    for t in range(T):
        nll -= np.log(probs[t, int(targets[t])] + 1e-12)
    return nll / T

def backward_sequence(targets, p, cache):
    T = len(targets)
    V, H = p["Why"].shape
    d = p["E"].shape[1]

    probs = softmax_rows(np.stack([p["Why"] @ cache["h"][t] + p["by"] for t in range(T)]))
    d_logits = probs.copy()
    for t in range(T):
        d_logits[t, int(targets[t])] -= 1.0
    d_logits /= T

    g = {k: np.zeros_like(v) for k, v in p.items()}
    dh_next = np.zeros(H)

    for t in range(T - 1, -1, -1):
        h = cache["h"][t]
        pre = cache["pre"][t]
        emb = cache["emb"][t]
        xi = int(cache["x"][t])

        dh = p["Why"].T @ d_logits[t] + dh_next
        dpre = (1.0 - np.tanh(pre) ** 2) * dh

        g["Why"] += np.outer(d_logits[t], h)
        g["by"] += d_logits[t]
        g["Wxh"] += np.outer(dpre, emb)
        g["Whh"] += np.outer(dpre, cache["h"][t - 1] if t > 0 else np.zeros(H))
        g["bh"] += dpre

        row = np.zeros_like(p["E"][0])
        row += p["Wxh"].T @ dpre
        g["E"][xi] += row

        dh_next = p["Whh"].T @ dpre

    return g

def sgd_step(p, g, lr):
    for k in p:
        p[k] -= lr * g[k]

probe_x = X_np[0]
probe_y = Y_np[0]
logits0, cache0 = forward_sequence(probe_x, params_np)
loss0 = cross_entropy_from_logits(logits0, probe_y)
print("loss antes de entrenar (1 secuencia):", float(loss0))

N = X_np.shape[0]
batch_size = 8
epochs = 80
lr = 0.02
rng = np.random.default_rng(7)

for epoch in range(1, epochs + 1):
    perm = rng.permutation(N)
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, N, batch_size):
        batch_idx = perm[start : start + batch_size]
        g_acc = {k: np.zeros_like(v) for k, v in params_np.items()}
        batch_loss = 0.0
        count = 0

        for j in batch_idx:
            x_seq = X_np[j]
            y_seq = Y_np[j]
            logits, cache = forward_sequence(x_seq, params_np)
            batch_loss += cross_entropy_from_logits(logits, y_seq)
            g = backward_sequence(y_seq, params_np, cache)
            for k in g_acc:
                g_acc[k] += g[k]
            count += 1

        batch_loss /= max(count, 1)
        epoch_loss += batch_loss
        n_batches += 1

        for k in g_acc:
            g_acc[k] /= max(count, 1)
        sgd_step(params_np, g_acc, lr)

    if epoch == 1 or epoch % 20 == 0:
        print(f"epoch {epoch:3d} | perdida media ~ {epoch_loss / max(n_batches, 1):.4f}")

print("Entrenamiento NumPy terminado.")

def sample_next(logits, temperature=1.0, rng=None):
    rng = rng or np.random.default_rng()
    z = logits / max(temperature, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))

def complete_numpy(prompt, max_new=120, temperature=0.8, rng=None):
    rng = rng or np.random.default_rng(123)
    ids = list(encode(prompt))
    for _ in range(max_new):
        x_arr = np.array(ids[-block_size:], dtype=np.int64)
        if len(x_arr) < block_size:
            pad = np.full(block_size - len(x_arr), ids[0], dtype=np.int64)
            x_arr = np.concatenate([pad, x_arr])
        logits, _ = forward_sequence(x_arr, params_np)
        next_id = sample_next(logits[-1], temperature=temperature, rng=rng)
        ids.append(next_id)
    return decode(ids)

txt = complete_numpy("def fac", max_new=100, temperature=0.75)
print("\n--- Generacion NumPy ---")
print(txt[:400])
