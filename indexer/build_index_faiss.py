# indexer/build_index_faiss.py
import numpy as np
import faiss
import argparse
import os

def build_index(features_path, index_path="index.faiss", use_ivf=False, nlist=100):
    x = np.load(features_path).astype('float32')  # shape (N, D)
    d = x.shape[1]
    # For cosine similarity use inner-product on normalized vectors
    if use_ivf:
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        index.add(x)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(x)
    faiss.write_index(index, index_path)
    print(f"Built index at {index_path} with {index.ntotal} vectors.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--index_path", default="index.faiss")
    p.add_argument("--use_ivf", action="store_true")
    p.add_argument("--nlist", type=int, default=100)
    args = p.parse_args()
    build_index(args.features, args.index_path, args.use_ivf, args.nlist)
