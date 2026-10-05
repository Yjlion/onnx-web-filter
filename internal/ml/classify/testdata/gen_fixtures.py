"""Regenerates fixtures.json: reference outputs from Python onnxruntime for
the catalog models, on inputs Go can reproduce exactly.

    pip install onnxruntime tokenizers numpy pillow
    python gen_fixtures.py <ml data dir>     # e.g. ../../../../data/ml

Images use an LCG tensor and a constant tensor (bit-identical in Go) plus
scene.jpg resized with PIL (Go's resizer differs slightly, so that case is
compared loosely). Text and embedding cases tokenize with the Rust
tokenizers library, which the Go tokenizer matches exactly.
"""
import json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

data = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))


def lcg(seed, n):
    out = np.empty(n, dtype=np.float32)
    s = seed & 0xFFFFFFFF
    for i in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        out[i] = s / 4294967296.0
    return out


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


IMAGE = {
    "nsfwjs-mobilenet": dict(layout="NHWC", mean=0.0, std=1.0, inp="input", out="prediction", softmax=False),
    "falconsai-vit": dict(layout="NCHW", mean=0.5, std=0.5, inp="pixel_values", out="logits", softmax=True),
}
TEXTS = [
    "Welcome to the city library. Opening hours, events for children and the new reading room.",
    "Hot naked girls stripping live on webcam, explicit hardcore porn videos free.",
    "Recipe: slow-roasted tomatoes with garlic, olive oil and fresh basil.",
    "Adult dating for singles looking for casual sex tonight, no strings attached.",
]
SITES = [
    "bbc.co.uk BBC News - Home: breaking news, world news, business, politics",
    "store.steampowered.com Steam: buy and play PC games",
    "chase.com Chase Bank: credit cards, mortgages, checking and savings",
]

fx = {"images": [], "texts": [], "embeds": []}
for mid, spec in IMAGE.items():
    path = os.path.join(data, "models", mid, "model.onnx")
    if not os.path.exists(path):
        print("skip", mid)
        continue
    s = ort.InferenceSession(path)
    shape = (1, 224, 224, 3) if spec["layout"] == "NHWC" else (1, 3, 224, 224)
    def run(x):
        y = s.run(None, {spec["inp"]: x.reshape(shape)})[0][0].astype(np.float64)
        return (softmax(y) if spec["softmax"] else y).tolist()
    for name, seed in [("lcg42", 42), ("lcg1337", 1337)]:
        fx["images"].append({"model": mid, "case": name, "seed": seed, "probs": run(lcg(seed, 3 * 224 * 224))})
    fx["images"].append({"model": mid, "case": "const0.5", "seed": -1, "probs": run(np.full(3 * 224 * 224, 0.5, np.float32))})
    im = Image.open(os.path.join(here, "scene.jpg")).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = (np.asarray(im, np.float32) / 255.0 - spec["mean"]) / spec["std"]
    if spec["layout"] == "NCHW":
        a = a.transpose(2, 0, 1)
    fx["images"].append({"model": mid, "case": "scene.jpg", "seed": -2, "probs": run(np.ascontiguousarray(a))})

tm = os.path.join(data, "models", "distilbert-nsfw")
if os.path.exists(os.path.join(tm, "model.onnx")):
    s = ort.InferenceSession(os.path.join(tm, "model.onnx"))
    tok = Tokenizer.from_file(os.path.join(tm, "tokenizer.json"))
    for t in TEXTS:
        e = tok.encode(t)
        ids = np.array([e.ids], np.int64)
        y = s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0].astype(np.float64)
        fx["texts"].append({"model": "distilbert-nsfw", "text": t, "probs": softmax(y).tolist()})

em = os.path.join(data, "models", "minilm-l6")
if os.path.exists(os.path.join(em, "model.onnx")):
    s = ort.InferenceSession(os.path.join(em, "model.onnx"))
    tok = Tokenizer.from_file(os.path.join(em, "tokenizer.json"))
    tok.no_padding()
    for t in SITES:
        e = tok.encode(t)
        ids = np.array([e.ids], np.int64)
        h = s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)})[0][0]
        v = h.mean(axis=0)
        v = v / np.linalg.norm(v)
        fx["embeds"].append({"model": "minilm-l6", "text": t, "vector": v.astype(np.float64).tolist()})

with open(os.path.join(here, "fixtures.json"), "w") as f:
    json.dump(fx, f)
for c in fx["images"]:
    print(c["model"], c["case"], np.round(c["probs"], 4))
for c in fx["texts"]:
    print("text", np.round(c["probs"], 4), c["text"][:50])
print(len(fx["embeds"]), "embeddings")
