# MultiModal-Fashion-Search-Engine
Fashion-based Search Engine with Qdrant Vector DB for semantic search

## 🛠️ Setup & Requirements

- Python 3.12
- Virtual environment recommended
.env file with:
```
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=
```
## ⚙️ How to Run

- Assumes a Python 3.12 virtual environment and .env configured with Qdrant credentials.

### Indexer 
```
# Extract CLIP features
python indexer/extract_features.py --image_dir data/fashionpedia --out_features data/fashionpedia_features.npy --out_meta data/fashionpedia_meta.csv --batch_size 32
```
```
# Enrich metadata (colors, category, vibe, environment)
python indexer/enrich_metadata.py
```
```
# Create collection on Qdrant (if not exists)
python indexer/create_collection.py
```
```
# Upload vectors + payloads to Qdrant
python indexer/upload_to_qdrant_batched.py
```

### Retriever
```# Run Streamlit demo (local)
streamlit run app/streamlit_app.py
```
```
# Or CLI (FAISS fallback, optional)
python retriever/query_cli.py --index index.faiss --meta meta.csv --query "a black jacket" --topk 5
```

## Code Structure and Modularity
- Root layout (relevant files)
```
indexer/ # offline pipelines (feature extraction, enrichment, upload)
retriever/ # online logic (parse, rerank, API)
app/ # Streamlit demo
.env # Qdrant credentials
requirements.txt
README.md
```

## 🧠 ML Highlights (Why this is better than vanilla CLIP)
- Attribute Enrichment
Zero-shot extraction of fashion attributes using prompt-engineered CLIP text embeddings.

- Query Decomposition
Explicit parsing of color, clothing type, vibe, and environment from user queries.

- Attribute-Aware Reranking
Final ranking combines: ```CLIP similarity + attribute consistency score```
