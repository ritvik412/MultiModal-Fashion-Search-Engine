# indexer/extract_features.py
# Extract image embeddings using pretrained CLIP and save features + metadata.
import os
import csv
import torch
import clip   # openai CLIP repo (pip install git+https://github.com/openai/CLIP.git)
from PIL import Image
from tqdm import tqdm
import numpy as np

MODEL_NAME = "ViT-B/32"  # or "RN50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    return model, preprocess

def batch_iter(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]

def extract(image_dir, out_features="features.npy", out_meta="meta.csv", batch_size=64):
    model, preprocess = load_model()
    image_paths = [os.path.join(image_dir, p) for p in os.listdir(image_dir)
                   if p.lower().endswith((".jpg",".png",".jpeg"))]
    embeddings = []
    meta = []
    for batch in batch_iter(image_paths, batch_size):
        imgs = []
        for p in batch:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            img_features = model.encode_image(imgs)    # [B, D]
            img_features = img_features.cpu().numpy()
        embeddings.append(img_features)
        for p in batch:
            meta.append({"path": p})
    embeddings = np.vstack(embeddings)
    # L2 normalize (important for cosine via inner-product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    np.save(out_features, embeddings)
    # write CSV with metadata
    with open(out_meta, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path"])
        writer.writeheader()
        for row in meta:
            writer.writerow(row)
    print(f"Saved features: {out_features}, meta: {out_meta}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    print(f"{DEVICE} is selected.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_features", default="features.npy")
    p.add_argument("--out_meta", default="meta.csv")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    extract(args.image_dir, args.out_features, args.out_meta, args.batch_size)
