# retriever/search.py
import re

# Known vocabularies (must match indexer)
COLORS = ["black","white","blue","red","green","brown","beige","grey","yellow"]
CATEGORIES = ["jacket","shirt","dress","jeans","sweater","hoodie","coat","top","tie"]
VIBES = ["formal","casual","streetwear","sporty","minimal","bold"]
ENVIRONMENTS = ["office","street","park","home","indoor","outdoor"]

def parse_query(query: str):
    q = query.lower()
    return {
        "colors": [c for c in COLORS if c in q],
        "categories": [c for c in CATEGORIES if c in q],
        "vibe": next((v for v in VIBES if v in q), None),
        "environment": next((e for e in ENVIRONMENTS if e in q), None)
    }

def attribute_score(parsed, payload):
    score = 0.0

    # Color matching (multi-color aware)
    if parsed["colors"]:
        if payload.get("color") in parsed["colors"]:
            score += 1.0
        else:
            score -= 0.5

    # Category matching
    if parsed["categories"]:
        if payload.get("category") in parsed["categories"]:
            score += 1.0
        else:
            score -= 0.5

    # Vibe matching
    if parsed["vibe"]:
        if payload.get("vibe") == parsed["vibe"]:
            score += 0.8
        else:
            score -= 0.3

    # Environment matching
    if parsed["environment"]:
        if payload.get("environment") == parsed["environment"]:
            score += 0.8
        else:
            score -= 0.3

    return score

def rerank_results(query, candidates):
    """
    candidates: list of dicts with keys:
      - score (CLIP similarity)
      - payload (metadata)
    """
    parsed = parse_query(query)

    reranked = []
    for c in candidates:
        attr_s = attribute_score(parsed, c["payload"])
        final_score = (0.7 * c["score"]) + (0.3 * attr_s)
        c["final_score"] = final_score
        reranked.append(c)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked
