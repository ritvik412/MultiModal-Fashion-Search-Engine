# indexer/upload_to_qdrant_batched.py
import os, uuid, numpy as np, pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from dotenv import load_dotenv
load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COL = os.getenv("QDRANT_COLLECTION")
vectors = np.load("data/fashionpedia_features.npy")
meta = pd.read_csv("data/fashionpedia_meta_enriched.csv")
BATCH=256
points=[]
for i,row in meta.iterrows():
    payload = {
        "image_path": row["path"],
        "color": row["color"],
        "category": row["category"],
        "vibe": row["vibe"]
    }
    points.append(rest_models.PointStruct(id=str(uuid.uuid4()), vector=vectors[i].tolist(), payload=payload))
    if len(points) >= BATCH:
        client.upsert(collection_name=COL, points=points)
        points=[]
if points:
    client.upsert(collection_name=COL, points=points)
print("Uploaded all points.")
