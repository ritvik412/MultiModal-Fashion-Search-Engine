import sys
import os
import streamlit as st
import torch
import clip
from PIL import Image
from qdrant_client import QdrantClient
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from retriever.search import rerank_results
load_dotenv()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Fashion Semantic Search", layout="wide")
st.title("👗 Fashion Semantic Image Search")

# ---------------- LOAD CLIP ----------------
@st.cache_resource
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

model = load_clip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- QDRANT ----------------
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION = os.getenv("QDRANT_COLLECTION")

# ---------------- SEARCH INPUT ----------------
query = st.text_input("Search (e.g. 'streetwear bomber jacket')")

with st.expander("🔎 Filters"):
    category = st.selectbox(
        "Category",
        ["Any", "jacket", "shirt", "dress", "jeans", "sweater", "hoodie", "coat", "top"]
    )
    color = st.selectbox(
        "Color",
        ["Any", "black", "white", "blue", "red", "green", "brown", "beige", "grey"]
    )
    vibe = st.selectbox(
        "Vibe",
        ["Any", "formal", "casual", "streetwear", "sporty", "minimal", "bold"]
    )

topk = st.slider("Number of results", 6, 30, 12)

# ---------------- SEARCH LOGIC ----------------
if query:
    with torch.no_grad():
        prompts = [
            f"A photo of a person wearing {query}",
            f"A fashion outfit featuring {query}",
            f"A clothing item described as {query}"
        ]
        tokens = clip.tokenize(prompts).to(DEVICE)
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        query_vector = text_emb.mean(dim=0).cpu().numpy().tolist()

    qdrant_filter = {"must": []}

    if category != "Any":
        qdrant_filter["must"].append(
            {"key": "category", "match": {"value": category}}
        )
    if color != "Any":
        qdrant_filter["must"].append(
            {"key": "color", "match": {"value": color}}
        )
    if vibe != "Any":
        qdrant_filter["must"].append(
            {"key": "vibe", "match": {"value": vibe}}
        )

    if not qdrant_filter["must"]:
        qdrant_filter = None

    raw_results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=40,
        with_payload=True
    ).points

    candidates = []
    for p in raw_results:
        candidates.append({
            "score": p.score,
            "payload": p.payload
        })

    reranked = rerank_results(query, candidates)
    results = reranked[:topk]


    # ---------------- DISPLAY ----------------
    cols = st.columns(3)

    for i, p in enumerate(results):
        payload = p["payload"]
        img_path = payload.get("image_path", "")

        col = cols[i % 3]
        try:
            img = Image.open(img_path).convert("RGB")
            col.image(img, use_column_width=True)
        except:
            col.warning("Image not found")

        col.caption(
            f"Color: {payload.get('color', 'N/A')} | "
            f"Category: {payload.get('category', 'N/A')} | "
            f"Vibe: {payload.get('vibe', 'N/A')} | "
            f"Env: {payload.get('environment', 'N/A')}"
        )

