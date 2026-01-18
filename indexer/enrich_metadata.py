# indexer/enrich_metadata.py
import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Semantic vocabularies --------
COLORS = ["black", "white", "blue", "red", "green", "brown", "beige", "grey"]

CATEGORIES = [
    "jacket", "shirt", "dress", "jeans",
    "sweater", "hoodie", "coat", "top", "tie"
]

VIBES = ["formal", "casual", "streetwear", "sporty", "minimal", "bold"]

ENVIRONMENTS = ["office", "street", "park", "home", "indoor", "outdoor"]

# -------- Load CLIP --------
model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def encode_text(prompts):
    tokens = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# -------- Text embeddings (prompt-engineered) --------
color_emb = encode_text([f"a person wearing {c} clothing" for c in COLORS])

cat_emb = encode_text([
    f"a person wearing a {c}" for c in CATEGORIES
])

vibe_emb = encode_text([
    f"a {v} fashion outfit" for v in VIBES
])

env_emb = encode_text([
    "a person inside a modern office",
    "a person walking on an urban street",
    "a person sitting in a park",
    "a person inside a home",
    "indoor fashion scene",
    "outdoor fashion scene"
])

# -------- Load image embeddings + base metadata --------
features = np.load("data/fashionpedia_features.npy")
meta = pd.read_csv("data/fashionpedia_meta.csv")

def nearest_label(image_emb, label_embs, labels):
    sims = image_emb @ label_embs.T
    return labels[int(np.argmax(sims))]

# -------- Enrich metadata --------
colors, categories, vibes, environments = [], [], [], []

for img_emb in tqdm(features, desc="Enriching metadata"):
    colors.append(nearest_label(img_emb, color_emb, COLORS))
    categories.append(nearest_label(img_emb, cat_emb, CATEGORIES))
    vibes.append(nearest_label(img_emb, vibe_emb, VIBES))
    environments.append(nearest_label(img_emb, env_emb, ENVIRONMENTS))

meta["color"] = colors
meta["category"] = categories
meta["vibe"] = vibes
meta["environment"] = environments

# -------- Save --------
out_path = "data/fashionpedia_meta_enriched.csv"
meta.to_csv(out_path, index=False)

print(f"Enriched metadata saved to: {out_path}")
