# create the qdrant collection
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("QDRANT_COLLECTION")

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION}' created.")
else:
    print(f"Collection '{COLLECTION}' already exists.")
