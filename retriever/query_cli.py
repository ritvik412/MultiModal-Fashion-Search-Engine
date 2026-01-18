# retriever/query_cli.py
import clip, torch, numpy as np, faiss, csv, json
from tqdm import tqdm

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_index(index_path, meta_csv):
    index = faiss.read_index(index_path)
    meta = []
    with open(meta_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta.append(r)
    return index, meta

def make_text_embedding(model, text, templates=None):
    if templates is None:
        templates = ["A photo of a person wearing {}.", "A photo of {}."]
    texts = [t.format(text) for t in templates]
    tokens = clip.tokenize(texts).to(DEVICE)  # [T, L]
    with torch.no_grad():
        txt_feats = model.encode_text(tokens)   # [T, D]
    txt_feats = txt_feats.cpu().numpy()
    # L2 normalize and average (ensembling across prompts)
    txt_feats = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True)+1e-10)
    avg = txt_feats.mean(axis=0)
    avg = avg / (np.linalg.norm(avg)+1e-10)
    return avg.astype('float32')

def search(index, q_emb, topk=10):
    # index expects (Nq, D)
    D, I = index.search(np.expand_dims(q_emb, axis=0), topk)
    return I[0], D[0]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    index, meta = load_index(args.index, args.meta)
    q_emb = make_text_embedding(model, args.query)
    ids, scores = search(index, q_emb, args.topk)
    for i, s in zip(ids, scores):
        print(f"{s:.4f}\t{meta[i]['path']}")
