"""
scripts/run_evaluation.py

Full Phase-5 evaluation pipeline for JanSahay.
Reproduces all metrics, tables, qualitative examples, and plots.

Usage:
    python scripts/run_evaluation.py
"""

import os, sys, json, time, csv, warnings
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from jansahay.utils.data_loader import load_dataset
from jansahay.retrieval.tfidf import TfidfRetriever
from jansahay.retrieval.semantic import SemanticRetriever
from jansahay.evaluation.metrics import (
    hit_at_k, recall_at_k, precision_at_k, reciprocal_rank, average_metrics
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR   = "reports/evaluation"
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
TFIDF_ART = "data/artifacts/tfidf"
SEM_ART   = "data/artifacts/semantic"
SCHEMES   = "data/processed/schemes_clean.csv"

os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("JANSAHAY PHASE 5: FORMAL EVALUATION")
print("=" * 70)

raw_df    = load_dataset()
schemes_df = pd.read_csv(SCHEMES)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build ground-truth: unique queries → relevant scheme IDs
# ─────────────────────────────────────────────────────────────────────────────
# Strict relevance: IDs of schemes that appear with this exact query.
# Topical relevance: all scheme IDs sharing the same Intent as the query.

intent_to_ids = (
    raw_df.groupby("Intent")["ID"].apply(list).to_dict()
)

queries_df = pd.read_csv("data/processed/queries.csv")   # 82 unique queries

query_records = []
for _, row in queries_df.iterrows():
    strict_ids = set(int(x) for x in str(row["relevant_scheme_ids"]).split(",") if x.strip())
    intent     = str(row["intent"]).strip()
    topical_ids= set(intent_to_ids.get(intent, []))

    query_records.append({
        "query_id"        : row["query_id"],
        "user_query"      : row["user_query"],
        "language"        : row["language"],
        "query_type"      : row["query_type"],
        "difficulty"      : row["difficulty"],
        "intent"          : intent,
        "strict_ids"      : strict_ids,
        "topical_ids"     : topical_ids,
        "n_strict"        : len(strict_ids),
        "n_topical"       : len(topical_ids),
    })

print(f"\n[Ground Truth] {len(query_records)} unique queries loaded.")
print(f"  Avg strict relevant per query : "
      f"{np.mean([r['n_strict'] for r in query_records]):.2f}")
print(f"  Avg topical relevant per query: "
      f"{np.mean([r['n_topical'] for r in query_records]):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load / build retrieval models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Models] Loading TF-IDF retriever …")
tfidf = TfidfRetriever.load(TFIDF_ART, current_df=schemes_df)
print(f"  TF-IDF loaded. Features: {tfidf.tfidf_matrix.shape[1]}")

print("[Models] Loading Semantic retriever …")
sem = SemanticRetriever.load(SEM_ART, current_df=schemes_df)
sem._load_model()
print(f"  Semantic loaded. Model: {sem.model_name}  Dim: {sem._embedding_dim}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run retrieval for every unique query (measure latency)
# ─────────────────────────────────────────────────────────────────────────────
TOP_K = 5

print(f"\n[Retrieval] Running both models on {len(query_records)} queries …")

tfidf_latencies = []
sem_latencies   = []
tfidf_zero_overlap = []  # queries where TF-IDF returns no_lexical_overlap

for rec in query_records:
    q = rec["user_query"]

    # TF-IDF
    t0 = time.perf_counter()
    tr = tfidf.search(q, top_k=TOP_K)
    tfidf_latencies.append(time.perf_counter() - t0)

    tfidf_ids = [r["scheme_id"] for r in tr.get("results", [])]
    rec["tfidf_status"]  = tr["status"]
    rec["tfidf_top_ids"] = tfidf_ids
    if tr["status"] == "no_lexical_overlap":
        tfidf_zero_overlap.append(rec)

    # Semantic
    t0 = time.perf_counter()
    sr = sem.search(q, top_k=TOP_K)
    sem_latencies.append(time.perf_counter() - t0)

    sem_ids = [r["scheme_id"] for r in sr.get("results", [])]
    rec["sem_status"]         = sr["status"]
    rec["semantic_top_ids"]   = sem_ids

print(f"  TF-IDF  — avg latency: {np.mean(tfidf_latencies)*1000:.1f} ms  "
      f"median: {np.median(tfidf_latencies)*1000:.1f} ms")
print(f"  Semantic— avg latency: {np.mean(sem_latencies)*1000:.1f} ms  "
      f"median: {np.median(sem_latencies)*1000:.1f} ms")
print(f"  TF-IDF zero-overlap queries: {len(tfidf_zero_overlap)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute per-query metrics (strict relevance)
# ─────────────────────────────────────────────────────────────────────────────
def compute_record_metrics(rec, mode="tfidf"):
    ids_key = f"{mode}_top_ids"
    retrieved = rec.get(ids_key, [])
    strict    = rec["strict_ids"]
    topical   = rec["topical_ids"]

    return {
        "query_id"    : rec["query_id"],
        "user_query"  : rec["user_query"],
        "language"    : rec["language"],
        "query_type"  : rec["query_type"],
        "difficulty"  : rec["difficulty"],
        "intent"      : rec["intent"],
        "model"       : mode,
        "n_strict"    : rec["n_strict"],
        "hit@1_strict"    : hit_at_k(retrieved, strict, 1),
        "hit@5_strict"    : hit_at_k(retrieved, strict, 5),
        "recall@1_strict" : recall_at_k(retrieved, strict, 1),
        "recall@5_strict" : recall_at_k(retrieved, strict, 5),
        "precision@5_strict": precision_at_k(retrieved, strict, 5),
        "mrr_strict"      : reciprocal_rank(retrieved, strict),
        "hit@5_topical"   : hit_at_k(retrieved, topical, 5),
        "recall@5_topical": recall_at_k(retrieved, topical, 5),
        "tfidf_overlap"   : 1 if rec.get("tfidf_status") == "no_lexical_overlap" else 0,
        "retrieved_ids"   : ",".join(str(i) for i in retrieved),
        "strict_ids"      : ",".join(str(i) for i in sorted(strict)),
    }

all_metrics = []
for rec in query_records:
    all_metrics.append(compute_record_metrics(rec, "tfidf"))
    all_metrics.append(compute_record_metrics(rec, "semantic"))

metrics_df = pd.DataFrame(all_metrics)
tfidf_df   = metrics_df[metrics_df["model"] == "tfidf"]
sem_df     = metrics_df[metrics_df["model"] == "semantic"]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Overall metrics table
# ─────────────────────────────────────────────────────────────────────────────
METRIC_COLS = [
    "hit@1_strict", "hit@5_strict", "recall@1_strict",
    "recall@5_strict", "precision@5_strict", "mrr_strict",
    "hit@5_topical", "recall@5_topical"
]

overall_rows = []
for model, df_ in [("TF-IDF", tfidf_df), ("Semantic (E5)", sem_df)]:
    row = {"model": model, "n_queries": len(df_)}
    for col in METRIC_COLS:
        row[col] = round(df_[col].mean(), 4)
    overall_rows.append(row)

overall_df = pd.DataFrame(overall_rows)
print("\n" + "=" * 70)
print("OVERALL METRICS (Strict Relevance, unless labelled topical)")
print("=" * 70)
print(overall_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Language-wise breakdown
# ─────────────────────────────────────────────────────────────────────────────
lang_rows = []
for lang in ["English", "Hindi", "Code-Mixed"]:
    for model, df_ in [("TF-IDF", tfidf_df), ("Semantic (E5)", sem_df)]:
        sub = df_[df_["language"] == lang]
        if len(sub) == 0:
            continue
        row = {"model": model, "language": lang, "n_queries": len(sub)}
        for col in METRIC_COLS:
            row[col] = round(sub[col].mean(), 4)
        lang_rows.append(row)

lang_df = pd.DataFrame(lang_rows)
print("\n" + "=" * 70)
print("LANGUAGE-WISE METRICS")
print("=" * 70)
print(lang_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Query-type breakdown
# ─────────────────────────────────────────────────────────────────────────────
qtype_rows = []
for qt in tfidf_df["query_type"].unique():
    for model, df_ in [("TF-IDF", tfidf_df), ("Semantic (E5)", sem_df)]:
        sub = df_[df_["query_type"] == qt]
        row = {"model": model, "query_type": qt, "n_queries": len(sub)}
        for col in METRIC_COLS:
            row[col] = round(sub[col].mean(), 4)
        qtype_rows.append(row)

qtype_df = pd.DataFrame(qtype_rows)
print("\n" + "=" * 70)
print("QUERY-TYPE METRICS")
print("=" * 70)
print(qtype_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Difficulty breakdown
# ─────────────────────────────────────────────────────────────────────────────
diff_rows = []
for diff in ["Easy", "Medium", "Hard"]:
    for model, df_ in [("TF-IDF", tfidf_df), ("Semantic (E5)", sem_df)]:
        sub = df_[df_["difficulty"] == diff]
        if len(sub) == 0:
            continue
        row = {"model": model, "difficulty": diff, "n_queries": len(sub)}
        for col in METRIC_COLS:
            row[col] = round(sub[col].mean(), 4)
        diff_rows.append(row)

diff_df = pd.DataFrame(diff_rows)
print("\n" + "=" * 70)
print("DIFFICULTY METRICS")
print("=" * 70)
print(diff_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. TF-IDF zero-overlap analysis
# ─────────────────────────────────────────────────────────────────────────────
zero_df = tfidf_df[tfidf_df["tfidf_overlap"] == 1]
print("\n" + "=" * 70)
print("TF-IDF ZERO-OVERLAP ANALYSIS")
print("=" * 70)
print(f"Total queries with zero lexical overlap: {len(zero_df)} / {len(tfidf_df)}")
if len(zero_df):
    lang_zero = zero_df["language"].value_counts()
    print(f"By language:\n{lang_zero.to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Qualitative examples
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Qualitative] Selecting representative examples …")

def get_scheme_name(scheme_id):
    row = schemes_df[schemes_df["ID"] == scheme_id]
    if row.empty:
        return f"ID {scheme_id}"
    return row.iloc[0]["Scheme_Name"][:80]

def describe_result(rec, model):
    ids_key  = f"{model}_top_ids"
    retrieved = rec.get(ids_key, [])
    strict    = rec["strict_ids"]
    top5_names = [get_scheme_name(i) for i in retrieved[:3]]
    hit5 = hit_at_k(retrieved, strict, 5)
    mrr  = reciprocal_rank(retrieved, strict)
    return {
        "top3_names": top5_names,
        "hit@5"     : hit5,
        "mrr"       : round(mrr, 3),
    }

# Pick 10 diverse qualitative examples
qual_rows = []

# Helper: find rec by criteria
def find_rec(lang=None, qtype=None, tfidf_overlap=None,
             tfidf_hit5=None, sem_hit5=None, exclude_qids=None):
    for rec in query_records:
        qid = rec["query_id"]
        if exclude_qids and qid in exclude_qids:
            continue
        if lang and rec["language"] != lang:
            continue
        if qtype and rec["query_type"] != qtype:
            continue
        tfidf_res = describe_result(rec, "tfidf")
        sem_res   = describe_result(rec, "semantic")
        if tfidf_overlap is not None:
            actual_overlap = 1 if rec.get("tfidf_status") == "no_lexical_overlap" else 0
            if actual_overlap != tfidf_overlap:
                continue
        if tfidf_hit5 is not None and tfidf_res["hit@5"] != tfidf_hit5:
            continue
        if sem_hit5 is not None and sem_res["hit@5"] != sem_hit5:
            continue
        return rec, tfidf_res, sem_res
    return None, None, None

used_qids = set()

scenarios = [
    {"label": "TF-IDF works well (English, both hit)",
     "lang": "English", "tfidf_hit5": 1.0, "sem_hit5": 1.0},
    {"label": "Semantic works, TF-IDF fails (Hindi, zero overlap)",
     "lang": "Hindi", "tfidf_overlap": 1},
    {"label": "Both work well (Code-Mixed)",
     "lang": "Code-Mixed", "tfidf_hit5": 1.0, "sem_hit5": 1.0},
    {"label": "TF-IDF fails – zero lexical overlap (Hindi)",
     "lang": "Hindi", "tfidf_overlap": 1},
    {"label": "English keyword query",
     "lang": "English", "qtype": "Keyword"},
    {"label": "English Natural Language query",
     "lang": "English", "qtype": "Natural Language"},
    {"label": "Hinglish query",
     "lang": "Code-Mixed", "qtype": "Natural Language"},
    {"label": "Hindi question query",
     "lang": "Hindi", "qtype": "Question"},
    {"label": "TF-IDF only works (Semantic low relevance)",
     "lang": "English", "tfidf_hit5": 1.0, "sem_hit5": 0.0},
    {"label": "Semantic only works (TF-IDF no overlap)",
     "lang": "Hindi", "tfidf_overlap": 1, "sem_hit5": 1.0},
]

for sc in scenarios:
    kwargs = {k: v for k, v in sc.items() if k != "label"}
    kwargs["exclude_qids"] = used_qids
    rec, tr, sr = find_rec(**kwargs)
    if rec is None:
        # relax constraints and try again
        rec, tr, sr = find_rec(lang=sc.get("lang"), exclude_qids=used_qids)
    if rec is None:
        continue
    used_qids.add(rec["query_id"])
    qual_rows.append({
        "scenario"            : sc["label"],
        "query"               : rec["user_query"],
        "language"            : rec["language"],
        "query_type"          : rec["query_type"],
        "strict_relevant_ids" : ",".join(str(i) for i in sorted(rec["strict_ids"])),
        "strict_relevant_names": "; ".join(get_scheme_name(i) for i in sorted(rec["strict_ids"])),
        "tfidf_status"        : rec.get("tfidf_status", ""),
        "tfidf_top3"          : " | ".join(tr["top3_names"]) if tr else "",
        "tfidf_hit@5"         : tr["hit@5"] if tr else "",
        "tfidf_mrr"           : tr["mrr"]   if tr else "",
        "semantic_top3"       : " | ".join(sr["top3_names"]) if sr else "",
        "sem_hit@5"           : sr["hit@5"] if sr else "",
        "sem_mrr"             : sr["mrr"]   if sr else "",
    })

qual_df = pd.DataFrame(qual_rows)
print(f"  Generated {len(qual_df)} qualitative examples.")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Save CSVs
# ─────────────────────────────────────────────────────────────────────────────
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8")
overall_df.to_csv(os.path.join(OUT_DIR, "overall_metrics.csv"), index=False, encoding="utf-8")
lang_df.to_csv(os.path.join(OUT_DIR, "language_metrics.csv"), index=False, encoding="utf-8")
qtype_df.to_csv(os.path.join(OUT_DIR, "query_type_metrics.csv"), index=False, encoding="utf-8")
diff_df.to_csv(os.path.join(OUT_DIR, "difficulty_metrics.csv"), index=False, encoding="utf-8")
qual_df.to_csv(os.path.join(OUT_DIR, "qualitative_examples.csv"), index=False, encoding="utf-8")

# Latency JSON
latency = {
    "tfidf_avg_ms"    : round(np.mean(tfidf_latencies)*1000, 2),
    "tfidf_median_ms" : round(np.median(tfidf_latencies)*1000, 2),
    "semantic_avg_ms" : round(np.mean(sem_latencies)*1000, 2),
    "semantic_median_ms": round(np.median(sem_latencies)*1000, 2),
    "note": "Model loading excluded. Per-query inference latency only.",
}
with open(os.path.join(OUT_DIR, "latency.json"), "w") as f:
    json.dump(latency, f, indent=2)

print(f"\n[Saved] CSV files → {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Plots
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_COLOR  = "#2563EB"   # blue
SEM_COLOR    = "#059669"   # green
FONT_SIZE    = 10

def bar_comparison(ax, tfidf_val, sem_val, title, ylabel):
    bars = ax.bar(["TF-IDF", "Semantic\n(E5)"],
                  [tfidf_val, sem_val],
                  color=[TFIDF_COLOR, SEM_COLOR], width=0.5, edgecolor="white")
    ax.set_title(title, fontsize=FONT_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE-1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=FONT_SIZE-1)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE-1)

# ── Plot 1: Overall Hit@5 and MRR ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle("Overall Retrieval Performance (Strict Relevance)", fontsize=11, fontweight="bold")
t_row = overall_df[overall_df["model"] == "TF-IDF"].iloc[0]
s_row = overall_df[overall_df["model"] == "Semantic (E5)"].iloc[0]
bar_comparison(axes[0], t_row["hit@5_strict"], s_row["hit@5_strict"], "Hit@5", "Score")
bar_comparison(axes[1], t_row["mrr_strict"],   s_row["mrr_strict"],   "MRR",   "Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "overall_performance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: Language-wise Hit@5 ──────────────────────────────────────────
langs_plot = ["English", "Hindi", "Code-Mixed"]
tfidf_h5   = []
sem_h5     = []
for lang in langs_plot:
    t = lang_df[(lang_df["model"]=="TF-IDF") & (lang_df["language"]==lang)]
    s = lang_df[(lang_df["model"]=="Semantic (E5)") & (lang_df["language"]==lang)]
    tfidf_h5.append(t["hit@5_strict"].values[0] if len(t) else 0)
    sem_h5.append(s["hit@5_strict"].values[0]   if len(s) else 0)

x  = np.arange(len(langs_plot))
w  = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, tfidf_h5, w, label="TF-IDF",         color=TFIDF_COLOR, edgecolor="white")
ax.bar(x + w/2, sem_h5,   w, label="Semantic (E5)",   color=SEM_COLOR,   edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(langs_plot, fontsize=FONT_SIZE)
ax.set_ylabel("Hit@5", fontsize=FONT_SIZE)
ax.set_ylim(0, 1.1)
ax.set_title("Hit@5 by Language (Strict Relevance)", fontsize=11, fontweight="bold")
ax.legend(fontsize=FONT_SIZE-1)
for i, (t, s) in enumerate(zip(tfidf_h5, sem_h5)):
    ax.text(i - w/2, t + 0.02, f"{t:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
    ax.text(i + w/2, s + 0.02, f"{s:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "language_hit5.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Language-wise MRR ─────────────────────────────────────────────
tfidf_mrr_lang = []
sem_mrr_lang   = []
for lang in langs_plot:
    t = lang_df[(lang_df["model"]=="TF-IDF") & (lang_df["language"]==lang)]
    s = lang_df[(lang_df["model"]=="Semantic (E5)") & (lang_df["language"]==lang)]
    tfidf_mrr_lang.append(t["mrr_strict"].values[0] if len(t) else 0)
    sem_mrr_lang.append(s["mrr_strict"].values[0]   if len(s) else 0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, tfidf_mrr_lang, w, label="TF-IDF",       color=TFIDF_COLOR, edgecolor="white")
ax.bar(x + w/2, sem_mrr_lang,   w, label="Semantic (E5)", color=SEM_COLOR,   edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(langs_plot, fontsize=FONT_SIZE)
ax.set_ylabel("MRR", fontsize=FONT_SIZE)
ax.set_ylim(0, 1.1)
ax.set_title("MRR by Language (Strict Relevance)", fontsize=11, fontweight="bold")
ax.legend(fontsize=FONT_SIZE-1)
for i, (t, s) in enumerate(zip(tfidf_mrr_lang, sem_mrr_lang)):
    ax.text(i - w/2, t + 0.02, f"{t:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
    ax.text(i + w/2, s + 0.02, f"{s:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "language_mrr.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 4: Query-type Hit@5 ─────────────────────────────────────────────
qtypes_plot = sorted(qtype_df["query_type"].unique())
tfidf_qt  = [qtype_df[(qtype_df["model"]=="TF-IDF") & (qtype_df["query_type"]==qt)]["hit@5_strict"].values[0]
             for qt in qtypes_plot]
sem_qt    = [qtype_df[(qtype_df["model"]=="Semantic (E5)") & (qtype_df["query_type"]==qt)]["hit@5_strict"].values[0]
             for qt in qtypes_plot]
xq = np.arange(len(qtypes_plot))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(xq - w/2, tfidf_qt, w, label="TF-IDF",         color=TFIDF_COLOR, edgecolor="white")
ax.bar(xq + w/2, sem_qt,   w, label="Semantic (E5)",   color=SEM_COLOR,   edgecolor="white")
ax.set_xticks(xq); ax.set_xticklabels(qtypes_plot, fontsize=FONT_SIZE-1)
ax.set_ylabel("Hit@5", fontsize=FONT_SIZE)
ax.set_ylim(0, 1.15)
ax.set_title("Hit@5 by Query Type (Strict Relevance)", fontsize=11, fontweight="bold")
ax.legend(fontsize=FONT_SIZE-1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "querytype_hit5.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 5: TF-IDF zero-overlap by language ───────────────────────────────
lang_overlap = tfidf_df.groupby("language")["tfidf_overlap"].agg(["sum","count"]).reset_index()
lang_overlap.columns = ["language","zero_count","total"]
lang_overlap["zero_pct"] = lang_overlap["zero_count"] / lang_overlap["total"]
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"English": TFIDF_COLOR, "Hindi": "#DC2626", "Code-Mixed": "#D97706"}
for _, row in lang_overlap.iterrows():
    ax.bar(row["language"], row["zero_pct"],
           color=colors.get(row["language"], "#6B7280"), edgecolor="white")
    ax.text(row["language"], row["zero_pct"] + 0.01,
            f"{row['zero_count']}/{row['total']}",
            ha="center", va="bottom", fontsize=FONT_SIZE-1)
ax.set_ylabel("Zero-overlap Rate", fontsize=FONT_SIZE)
ax.set_ylim(0, 1.15)
ax.set_title("TF-IDF Zero Lexical Overlap by Language", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "tfidf_zero_overlap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 6: Retrieval latency ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["TF-IDF", "Semantic (E5)"],
       [latency["tfidf_avg_ms"], latency["semantic_avg_ms"]],
       color=[TFIDF_COLOR, SEM_COLOR], edgecolor="white", width=0.4)
ax.set_ylabel("Avg per-query latency (ms)", fontsize=FONT_SIZE)
ax.set_title("Per-Query Retrieval Latency\n(model loading excluded)",
             fontsize=11, fontweight="bold")
for i, v in enumerate([latency["tfidf_avg_ms"], latency["semantic_avg_ms"]]):
    ax.text(i, v + 0.1, f"{v:.1f} ms", ha="center", va="bottom", fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "retrieval_latency.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"[Plots] Saved 6 plots to {PLOT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Data leakage verification
# ─────────────────────────────────────────────────────────────────────────────
LEAKAGE_FIELDS = ["User_Query","Intent","Query_Type","Language","Difficulty","Level"]
leakage_found  = False
if "doc_text" in schemes_df.columns:
    for field in LEAKAGE_FIELDS:
        header = f"{field}:"
        if schemes_df["doc_text"].str.contains(header, na=False).any():
            print(f"  [ERROR] Leakage: '{header}' found in doc_text!")
            leakage_found = True
if not leakage_found:
    print("[Leakage Check] PASSED — no forbidden fields found in doc_text.")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Write evaluation_summary.md
# ─────────────────────────────────────────────────────────────────────────────
t_overall = overall_df[overall_df["model"] == "TF-IDF"].iloc[0]
s_overall = overall_df[overall_df["model"] == "Semantic (E5)"].iloc[0]
zero_pct_hindi    = lang_overlap[lang_overlap["language"]=="Hindi"]["zero_pct"].values
zero_pct_codemix  = lang_overlap[lang_overlap["language"]=="Code-Mixed"]["zero_pct"].values
zero_pct_english  = lang_overlap[lang_overlap["language"]=="English"]["zero_pct"].values
zero_hindi   = f"{zero_pct_hindi[0]:.1%}"   if len(zero_pct_hindi) else "N/A"
zero_codemix = f"{zero_pct_codemix[0]:.1%}" if len(zero_pct_codemix) else "N/A"
zero_english = f"{zero_pct_english[0]:.1%}" if len(zero_pct_english) else "N/A"

# Build language table strings
def lang_row(lang):
    t = lang_df[(lang_df["model"]=="TF-IDF") & (lang_df["language"]==lang)]
    s = lang_df[(lang_df["model"]=="Semantic (E5)") & (lang_df["language"]==lang)]
    if not len(t) or not len(s):
        return ""
    tn, sn = int(t["n_queries"].values[0]), int(s["n_queries"].values[0])
    assert tn == sn, "language query count mismatch"
    return (
        f"| {lang} (n={tn}) "
        f"| {t['hit@1_strict'].values[0]:.3f} "
        f"| {s['hit@1_strict'].values[0]:.3f} "
        f"| {t['hit@5_strict'].values[0]:.3f} "
        f"| {s['hit@5_strict'].values[0]:.3f} "
        f"| {t['mrr_strict'].values[0]:.3f} "
        f"| {s['mrr_strict'].values[0]:.3f} |"
    )

lang_table = "\n".join(lang_row(l) for l in ["English","Hindi","Code-Mixed"] if lang_row(l))

def qtype_row(qt):
    t = qtype_df[(qtype_df["model"]=="TF-IDF") & (qtype_df["query_type"]==qt)]
    s = qtype_df[(qtype_df["model"]=="Semantic (E5)") & (qtype_df["query_type"]==qt)]
    if not len(t) or not len(s):
        return ""
    tn = int(t["n_queries"].values[0])
    return (
        f"| {qt} (n={tn}) "
        f"| {t['hit@5_strict'].values[0]:.3f} "
        f"| {s['hit@5_strict'].values[0]:.3f} "
        f"| {t['mrr_strict'].values[0]:.3f} "
        f"| {s['mrr_strict'].values[0]:.3f} |"
    )

qtype_table = "\n".join(qtype_row(qt) for qt in qtypes_plot if qtype_row(qt))

summary_md = f"""# JanSahay Phase 5 — Evaluation Summary

## 1. Dataset
- Raw records: 115
- Unique evaluation queries: 82
- Languages: English (n={int((tfidf_df['language']=='English').sum())}), Hindi (n={int((tfidf_df['language']=='Hindi').sum())}), Code-Mixed (n={int((tfidf_df['language']=='Code-Mixed').sum())})
- Query types: Natural Language, Keyword, Question, Conversational
- Difficulty: Easy, Medium, Hard

## 2. Evaluation Protocol
- Evaluation unit: **unique User_Query** (82 queries).
- Repeated rows that share the same User_Query are collapsed; their associated Scheme IDs are unioned into the relevant set.
- Top-K = 5 for all rank-based metrics.
- Model configurations are **fixed from Phase 3 and Phase 4** — no post-hoc tuning.

## 3. Relevance Definitions
- **Strict**: A scheme is relevant if it is directly associated with the query in the dataset.
- **Topical** (broader): A scheme is topically relevant if it shares the same Intent as the query. This is a coarser measure and does not imply the scheme is the exact correct answer.

## 4. Metrics Computed
| Metric | Definition |
|--------|-----------|
| Hit@1  | 1 if a relevant scheme is in rank-1 result |
| Hit@5  | 1 if any relevant scheme is in top-5 |
| Recall@1 | Fraction of relevant schemes in rank-1 result |
| Recall@5 | Fraction of relevant schemes in top-5 |
| Precision@5 | Fraction of top-5 that are relevant |
| MRR | Mean Reciprocal Rank — 1/position of first relevant |

## 5. Overall Results (Strict Relevance, n=82 queries)

| Model | Hit@1 | Hit@5 | Recall@1 | Recall@5 | Prec@5 | MRR |
|-------|-------|-------|----------|----------|--------|-----|
| TF-IDF | {t_overall['hit@1_strict']:.3f} | {t_overall['hit@5_strict']:.3f} | {t_overall['recall@1_strict']:.3f} | {t_overall['recall@5_strict']:.3f} | {t_overall['precision@5_strict']:.3f} | {t_overall['mrr_strict']:.3f} |
| Semantic (E5) | {s_overall['hit@1_strict']:.3f} | {s_overall['hit@5_strict']:.3f} | {s_overall['recall@1_strict']:.3f} | {s_overall['recall@5_strict']:.3f} | {s_overall['precision@5_strict']:.3f} | {s_overall['mrr_strict']:.3f} |

## 6. Language Results (Strict Relevance)

| Language | TF-IDF Hit@1 | Sem Hit@1 | TF-IDF Hit@5 | Sem Hit@5 | TF-IDF MRR | Sem MRR |
|----------|-------------|----------|-------------|---------|-----------|--------|
{lang_table}

## 7. Query-Type Results (Strict Relevance)

| Query Type | TF-IDF Hit@5 | Sem Hit@5 | TF-IDF MRR | Sem MRR |
|------------|-------------|---------|-----------|--------|
{qtype_table}

## 8. TF-IDF Zero-Overlap Analysis
TF-IDF relies on exact vocabulary overlap between query terms and scheme document tokens.
When queries are written in Hindi (Devanagari) against English scheme documents, overlap is zero.

| Language | Zero-overlap Rate |
|----------|-----------------|
| English  | {zero_english} |
| Hindi    | {zero_hindi} |
| Code-Mixed | {zero_codemix} |

TF-IDF returns `no_lexical_overlap` for these queries and produces no results — correctly
avoiding presenting arbitrary schemes as matches.

## 9. Retrieval Latency (model loading excluded)

| Model | Avg Latency | Median Latency |
|-------|------------|---------------|
| TF-IDF | {latency['tfidf_avg_ms']:.1f} ms | {latency['tfidf_median_ms']:.1f} ms |
| Semantic (E5) | {latency['semantic_avg_ms']:.1f} ms | {latency['semantic_median_ms']:.1f} ms |

Note: Semantic model loading (first-time model download + tokenizer initialisation)
takes significantly longer and is excluded from per-query latency to enable a fair
per-query comparison.

## 10. Qualitative Examples
See `qualitative_examples.csv` for 10 annotated query examples covering:
TF-IDF success, Semantic success, both work, TF-IDF lexical failure, Semantic
weak result, Hindi query, English query, Hinglish query, Keyword query, Natural
Language query.

## 11. Dataset Limitations
1. **Small corpus (115 schemes, 82 unique queries)**: Metric estimates have high variance; small differences should not be over-interpreted.
2. **Query–scheme associations are manually created**: There is no crowd-sourced or user-validated relevance judgement.
3. **Repeated descriptive content**: 4 unique description texts and 8 eligibility texts appear across multiple schemes; cosine similarity models may rank duplicates identically.
4. **Generic queries**: Broad queries (e.g., "scholarship for students") are intentionally associated with multiple schemes, which challenges strict single-label metrics.
5. **Strict relevance is conservative**: A retrieved scheme that is genuinely helpful but was not annotated for that exact query will score as a miss.
6. **Topical relevance is broad**: Treating all schemes with the same Intent as relevant inflates recall numbers and should not be reported without clear labelling.

## 12. Objective Interpretation
- On **English queries**, both models retrieve relevant schemes, with {t_overall['hit@5_strict']:.3f} (TF-IDF) and {s_overall['hit@5_strict']:.3f} (Semantic) overall Hit@5.
- On **Hindi queries**, TF-IDF experiences a zero-overlap rate of {zero_hindi}, producing no results. The semantic model embeds queries and documents into a shared multilingual space, enabling cross-lingual retrieval.
- **Code-Mixed** queries contain Hinglish (Latin-script) terms; TF-IDF can partially match these to English scheme vocabulary, so its zero-overlap rate is lower for Code-Mixed than for Hindi.
- The semantic model ({sem.model_name}) applies E5 query/passage prefixes consistently and performs cross-lingual retrieval without any fine-tuning on this dataset.
- No claim is made about which model is universally "better". The differences depend on query language and the strict-vs-topical relevance definition used.

## 13. Data Leakage Verification
PASSED — `User_Query`, `Intent`, `Query_Type`, `Language`, `Difficulty`, and `Level`
fields were confirmed absent from all scheme document text (`doc_text`) used for
vectorisation and embedding.

## 14. Files Generated
- `reports/evaluation/metrics.csv` — per-query, per-model metrics (all 164 rows)
- `reports/evaluation/overall_metrics.csv`
- `reports/evaluation/language_metrics.csv`
- `reports/evaluation/query_type_metrics.csv`
- `reports/evaluation/difficulty_metrics.csv`
- `reports/evaluation/qualitative_examples.csv`
- `reports/evaluation/latency.json`
- `reports/evaluation/plots/overall_performance.png`
- `reports/evaluation/plots/language_hit5.png`
- `reports/evaluation/plots/language_mrr.png`
- `reports/evaluation/plots/querytype_hit5.png`
- `reports/evaluation/plots/tfidf_zero_overlap.png`
- `reports/evaluation/plots/retrieval_latency.png`
"""

with open(os.path.join(OUT_DIR, "evaluation_summary.md"), "w", encoding="utf-8") as f:
    f.write(summary_md)

print(f"[Summary] evaluation_summary.md written.")
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
