"""
jansahay/evaluation/metrics.py

Evaluation metric implementations for ranked retrieval.

Definitions:
  Hit@K   : 1 if at least one relevant scheme appears in the top-K results, else 0.
  Recall@K: |relevant ∩ retrieved@K| / |relevant|
  Prec@K  : |relevant ∩ retrieved@K| / K
  MRR     : Mean Reciprocal Rank — 1/rank of the first relevant hit
             (0 if no relevant item in results; averaged over all queries)

All metrics handle the case where a query has MULTIPLE relevant schemes.
"""

def hit_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """1.0 if any of the top-k retrieved IDs are in relevant_ids, else 0.0."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant schemes found in the top-k retrieved list."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k == 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """Reciprocal rank of the first relevant item in the retrieved list (0 if none)."""
    if not relevant_ids:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_metrics(records: list) -> dict:
    """
    Given a list of per-query metric dicts, compute the mean of each metric.
    """
    if not records:
        return {}
    keys = [k for k in records[0] if isinstance(records[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in records if r[k] is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["n_queries"] = len(records)
    return out
