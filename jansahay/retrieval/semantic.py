"""
SemanticRetriever — Multilingual Dense Embedding Retrieval for JanSahay.

Model: intfloat/multilingual-e5-base (primary)
       sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (fallback)

Why multilingual:
  User queries are written in Hindi, English, and Hinglish (code-mixed).
  Scheme documents are primarily in English (with some Hindi in Descriptions).
  A cross-lingual dense retrieval model embeds both queries and documents
  into a shared semantic space, enabling Hindi queries to match English
  scheme text — something impossible with lexical TF-IDF.

E5 Input Format:
  - Query  embeddings: prefix "query: <text>"
  - Document embeddings: prefix "passage: <text>"
  (Only for multilingual-e5-base; NOT applied to the MiniLM fallback.)

Data Leakage:
  Documents are built exclusively from Scheme_Name, Description, Benefits,
  Eligibility. User_Query, Intent, Query_Type, Language, Difficulty, and
  Level are NEVER embedded into document vectors.
"""

import os
import json
import hashlib
import numpy as np

from jansahay.preprocessing.text_cleaner import preprocess_for_semantic

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PRIMARY_MODEL   = "intfloat/multilingual-e5-base"
FALLBACK_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Heuristic low-relevance threshold (documented, not scientifically validated).
# Chosen conservatively so that clearly unrelated queries surface a warning
# without suppressing genuinely weak-but-valid matches.  Phase 5 will calibrate.
DEFAULT_LOW_SIM_THRESHOLD = 0.30


def _compute_doc_hash(texts: list) -> str:
    """SHA-256 of the concatenated preprocessed document corpus."""
    corpus = "||".join(texts)
    return hashlib.sha256(corpus.encode("utf-8")).hexdigest()


class SemanticRetriever:
    """
    Dense semantic retrieval using a pretrained multilingual sentence-embedding
    model.  Embeddings are built once and reused; the model is loaded once and
    kept in memory.
    """

    def __init__(self, model_name: str = PRIMARY_MODEL,
                 low_sim_threshold: float = DEFAULT_LOW_SIM_THRESHOLD):
        self.model_name        = model_name
        self.low_sim_threshold = low_sim_threshold
        self._model            = None          # loaded lazily / explicitly
        self._embeddings       = None          # (N, D) float32 numpy array
        self._scheme_ids       = None          # list[int]
        self._schemes_df       = None          # full metadata dataframe
        self._doc_hash         = None
        self._embedding_dim    = None
        self._use_e5_prefix    = (model_name == PRIMARY_MODEL)
        self._is_indexed       = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load the SentenceTransformer model (once)."""
        if self._model is not None:
            return  # already loaded — do NOT reload

        from sentence_transformers import SentenceTransformer
        print(f"[SemanticRetriever] Loading model: {self.model_name} …")
        try:
            self._model = SentenceTransformer(self.model_name)
            print(f"[SemanticRetriever] Model loaded successfully.")
        except Exception as exc:
            if self.model_name == PRIMARY_MODEL:
                print(f"[SemanticRetriever] Primary model failed ({exc}).")
                print(f"[SemanticRetriever] Falling back to {FALLBACK_MODEL} …")
                self.model_name     = FALLBACK_MODEL
                self._use_e5_prefix = False
                self._model         = SentenceTransformer(FALLBACK_MODEL)
                print(f"[SemanticRetriever] Fallback model loaded.")
            else:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name}': {exc}"
                ) from exc

    def _format_document(self, text: str) -> str:
        """Apply E5 'passage:' prefix only when using multilingual-e5-base."""
        if self._use_e5_prefix:
            return f"passage: {text}"
        return text

    def _format_query(self, text: str) -> str:
        """Apply E5 'query:' prefix only when using multilingual-e5-base."""
        if self._use_e5_prefix:
            return f"query: {text}"
        return text

    def _encode(self, texts: list, batch_size: int = 32,
                show_progress: bool = False) -> np.ndarray:
        """Encode texts and return L2-normalised float32 embeddings."""
        self._load_model()   # no-op if already loaded
        raw = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True   # L2-norm applied inside model
        )
        # Verify normalisation (unit vectors → dot product == cosine sim)
        norms = np.linalg.norm(raw, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            "Embeddings are NOT unit-normalised — check model encode settings."
        return raw.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_index(self, df, text_column: str = "doc_text_semantic"):
        """
        Generate and store document embeddings.
        Embeddings are created ONCE and reused for all subsequent searches.

        Parameters
        ----------
        df : pd.DataFrame  — must contain 'ID' and a document-text column.
        text_column        — defaults to 'doc_text_semantic' from Phase 2.
        """
        import pandas as pd

        self._load_model()

        if text_column not in df.columns:
            if "doc_text" in df.columns:
                text_column = "doc_text"
            else:
                raise ValueError(f"Column '{text_column}' not found.")

        self._schemes_df = df.reset_index(drop=True)
        self._scheme_ids = self._schemes_df["ID"].tolist()

        # Semantic preprocessing (light normalisation, preserves structure)
        raw_texts     = self._schemes_df[text_column].fillna("").tolist()
        clean_texts   = [preprocess_for_semantic(t) for t in raw_texts]
        doc_inputs    = [self._format_document(t) for t in clean_texts]

        print(f"[SemanticRetriever] Embedding {len(doc_inputs)} scheme documents …")
        self._embeddings    = self._encode(doc_inputs, show_progress=True)
        self._embedding_dim = self._embeddings.shape[1]
        self._doc_hash      = _compute_doc_hash(clean_texts)
        self._is_indexed    = True
        print(f"[SemanticRetriever] Index built. Shape: {self._embeddings.shape}")
        return self

    def search(self, query: str, top_k: int = 5,
               level_filter: str = None, intent_filter: str = None) -> dict:
        """
        Encode query → cosine similarity → ranked Top-K results.
        Does NOT rebuild document embeddings.
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index() or load().")

        # 1. Minimal semantic preprocessing + E5 prefix
        clean_q   = preprocess_for_semantic(query)
        q_input   = self._format_query(clean_q)
        q_vec     = self._encode([q_input])   # shape (1, D)

        # 2. Cosine similarity via dot product (both sides L2-normalised)
        scores = (self._embeddings @ q_vec.T).flatten()  # (N,)

        # 3. Filter candidates
        candidates = []
        for idx in range(len(self._schemes_df)):
            row = self._schemes_df.iloc[idx]
            if level_filter:
                if str(row.get("Level", "")).strip().lower() != level_filter.strip().lower():
                    continue
            if intent_filter:
                if str(row.get("Intent", "")).strip().lower() != intent_filter.strip().lower():
                    continue
            candidates.append(idx)

        # 4. Build result list
        results = []
        for idx in candidates:
            score = float(scores[idx])
            row   = self._schemes_df.iloc[idx]
            low   = score < self.low_sim_threshold
            results.append({
                "scheme_id"      : int(row.get("ID",          idx + 1)),
                "scheme_name"    : str(row.get("Scheme_Name", "")),
                "similarity_score": round(score, 4),
                "level"          : str(row.get("Level",       "")),
                "intent"         : str(row.get("Intent",      "")),
                "description"    : str(row.get("Description", "")),
                "benefits"       : str(row.get("Benefits",    "")),
                "eligibility"    : str(row.get("Eligibility", "")),
                "low_relevance"  : low,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top = results[:top_k]
        for rank, r in enumerate(top, 1):
            r["rank"] = rank

        all_low = all(r["low_relevance"] for r in top)
        return {
            "status"        : "low_relevance_warning" if all_low else "success",
            "query"         : query,
            "clean_query"   : clean_q,
            "model"         : self.model_name,
            "top_k"         : top_k,
            "total_candidates": len(candidates),
            "results"       : top,
        }

    def rank_all(self, query: str) -> list:
        """Return [(scheme_id, score)] for every document, sorted descending."""
        if not self._is_indexed:
            raise ValueError("Index not built.")
        clean_q = preprocess_for_semantic(query)
        q_vec   = self._encode([self._format_query(clean_q)])
        scores  = (self._embeddings @ q_vec.T).flatten()
        ranked  = sorted(
            [(int(self._scheme_ids[i]), float(scores[i]))
             for i in range(len(self._scheme_ids))],
            key=lambda x: x[1], reverse=True
        )
        return ranked

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, dir_path: str = "data/artifacts/semantic"):
        """
        Persist embeddings (NumPy) and config (JSON) to disk.
        No vector database used; 115 vectors fit trivially in RAM.
        """
        if not self._is_indexed:
            raise ValueError("Nothing to save — call build_index() first.")
        os.makedirs(dir_path, exist_ok=True)

        # Embeddings
        np.save(os.path.join(dir_path, "embeddings.npy"), self._embeddings)

        # Metadata CSV
        self._schemes_df.to_csv(
            os.path.join(dir_path, "schemes_metadata.csv"),
            index=False, encoding="utf-8"
        )

        # Config
        config = {
            "model_name"       : self.model_name,
            "use_e5_prefix"    : self._use_e5_prefix,
            "embedding_dim"    : int(self._embedding_dim),
            "num_schemes"      : len(self._scheme_ids),
            "normalization"    : "L2 (unit vectors)",
            "doc_hash"         : self._doc_hash,
            "low_sim_threshold": self.low_sim_threshold,
        }
        with open(os.path.join(dir_path, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[SemanticRetriever] Artifacts saved to '{dir_path}'.")

    @classmethod
    def load(cls, dir_path: str = "data/artifacts/semantic",
             current_df=None):
        """
        Load a previously saved index.
        Performs stale-model check if current_df is supplied.
        Does NOT reload the model — the model is loaded lazily on first search.
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                f"Artifact directory '{dir_path}' not found."
            )

        config_path = os.path.join(dir_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        instance                  = cls(
            model_name        = config["model_name"],
            low_sim_threshold = config.get("low_sim_threshold",
                                           DEFAULT_LOW_SIM_THRESHOLD)
        )
        instance._use_e5_prefix   = config.get("use_e5_prefix", True)
        instance._embedding_dim   = config["embedding_dim"]
        instance._doc_hash        = config["doc_hash"]

        import pandas as pd
        instance._embeddings  = np.load(
            os.path.join(dir_path, "embeddings.npy")
        ).astype(np.float32)
        instance._schemes_df  = pd.read_csv(
            os.path.join(dir_path, "schemes_metadata.csv")
        )
        instance._scheme_ids  = instance._schemes_df["ID"].tolist()
        instance._is_indexed  = True

        # Stale model protection
        if current_df is not None:
            col = ("doc_text_semantic" if "doc_text_semantic" in current_df.columns
                   else "doc_text")
            if col in current_df.columns:
                current_texts = [
                    preprocess_for_semantic(t)
                    for t in current_df[col].fillna("").tolist()
                ]
                current_hash = _compute_doc_hash(current_texts)
                if current_hash != instance._doc_hash:
                    raise ValueError(
                        "Stale semantic artifact detected! "
                        "The scheme documents have changed since the index was built. "
                        "Please rebuild with build_index()."
                    )

        return instance
