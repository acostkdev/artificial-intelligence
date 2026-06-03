import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

rng = np.random.default_rng(42)

ruta_base = os.path.dirname(__file__)
ruta_modelo = os.path.join(ruta_base, "modelo_rnn.h5")
ruta_mappings = os.path.join(ruta_base, "tokenizer_mappings.pkl")

with open(ruta_mappings, "rb") as f:
    mappings = pickle.load(f)

stoi = mappings["stoi"]
itos = mappings["itos"]
VOCAB_SIZE = mappings["vocab_size"]

modelo_completo = tf.keras.models.load_model(ruta_modelo)
block_size = modelo_completo.input_shape[1]

embed_dim = modelo_completo.get_layer("embedding").output_dim
hidden = modelo_completo.get_layer("simple_rnn").units

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(block_size,), dtype=tf.int64),
    tf.keras.layers.Embedding(VOCAB_SIZE, embed_dim),
    tf.keras.layers.SimpleRNN(hidden, activation="tanh", return_sequences=False),
    tf.keras.layers.Dense(VOCAB_SIZE),
])
modelo.get_layer("embedding").set_weights(
    modelo_completo.get_layer("embedding").get_weights()
)
modelo.get_layer("simple_rnn").set_weights(
    modelo_completo.get_layer("simple_rnn").get_weights()
)
modelo.get_layer("dense").set_weights(
    modelo_completo.get_layer("time_distributed").layer.get_weights()
)
del modelo_completo

@tf.function
def step(x):
    return modelo(x, training=False)

TEMPERATURA = 0.05
TOP_K = 3
buf = np.empty(block_size, dtype=np.int64)

def muestrear(logits, temperatura=TEMPERATURA, top_k=TOP_K):
    z = logits / max(temperatura, 1e-6)
    z = z - z.max()
    if top_k > 0:
        indices = np.argpartition(z, -top_k)[-top_k:]
        mask = np.full_like(z, -np.inf)
        mask[indices] = z[indices]
        z = mask
    e = np.exp(z)
    p = e / e.sum()
    return int(rng.choice(len(p), p=p))

def completar(prompt, max_nuevos=350):
    ids = [stoi.get(c, 0) for c in prompt]
    prompt_len = len(ids)
    for i in range(max_nuevos):
        ctx = ids[-block_size:]
        start = block_size - len(ctx)
        buf[start:] = ctx
        if start > 0:
            buf[:start] = ids[0]
        logits = step(buf.reshape(1, block_size)).numpy()[0]
        idx = muestrear(logits)
        ids.append(idx)
        gen = "".join(itos[j] for j in ids[prompt_len:])
        if gen.endswith("}\n\n"):
            break
    return "".join(itos[i] for i in ids)

print(f"API lista. Vocab: {VOCAB_SIZE}, block_size: {block_size}")


@app.route("/autocompletar", methods=["POST"])
def autocompletar():
    datos = request.get_json(force=True)
    codigo = datos.get("codigo", "")
    max_nuevos = datos.get("max_tokens", 350)

    if not codigo:
        return jsonify({"error": "No se envio codigo"}), 400

    generado = completar(codigo, max_nuevos=max_nuevos)
    completado = generado[len(codigo):]

    return jsonify({
        "codigo_original": codigo,
        "completado": completado,
        "codigo_completo": generado,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelo": "rnn_vanilla", "vocab_size": VOCAB_SIZE})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
