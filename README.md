# MultiModal-Fashion-Search-Engine
Fashion-based Search Engine with Qdrant Vector DB for semantic search

## How to run

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
