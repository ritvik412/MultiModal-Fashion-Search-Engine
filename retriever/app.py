# retriever/app.py
from flask import Flask, request, jsonify
import clip, torch, faiss, numpy as np, csv

app = Flask(__name__)
MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL, PREP = None, None
INDEX, META = None, None
TEMPLATES = ["A photo of a person wearing {}.", "A photo of {}.", "A photograph of a person in {}."]

def load_all(index_path, meta_csv):
    global MODEL, PREP, INDEX, META
    MODEL, PREP = clip.load(MODEL_NAME, device=DEVICE)
    INDEX = faiss.read_index(index_path)
    META = []
    with open(meta_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            META.append(r)

@app.route("/search", methods=["POST"])
def search_api():
    data = request.json
    query = data.get("query", "")
    topk = int(data.get("topk", 5))
    tokens = clip.tokenize([t.format(query) for t in TEMPLATES]).to(DEVICE)
    with torch.no_grad():
        feats = MODEL.encode_text(tokens).cpu().numpy()
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True)+1e-10)
    q = feats.mean(axis=0)
    q = q / (np.linalg.norm(q)+1e-10)
    D, I = INDEX.search(np.expand_dims(q.astype('float32'), axis=0), topk)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({"path": META[idx]["path"], "score": float(score)})
    return jsonify(results)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    load_all(args.index, args.meta)
    app.run(host=args.host, port=args.port)
