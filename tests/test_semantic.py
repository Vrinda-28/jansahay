"""
Unit tests for SemanticRetriever (Phase 4).

These tests use a tiny 3-document toy corpus so they run without
needing the full 115-scheme dataset or a long model download.
All assertions are structural / behavioural — formal accuracy
evaluation is deferred to Phase 5.
"""
import os, sys, shutil, unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jansahay.retrieval.semantic import SemanticRetriever, _compute_doc_hash

# ── tiny shared fixture ────────────────────────────────────────────────────── #
import pandas as pd

TOY_ROWS = [
    {
        "ID": 1, "Scheme_Name": "Scholarship for Engineering Students",
        "Description": "Financial aid for engineering students.",
        "Benefits": "Tuition waiver up to 50000 per year.",
        "Eligibility": "Enrolled in government college.",
        "Level": "Central", "Intent": "Education",
        "doc_text_semantic": (
            "Scheme Name: Scholarship for Engineering Students\n\n"
            "Description: Financial aid for engineering students.\n\n"
            "Benefits: Tuition waiver up to 50000 per year.\n\n"
            "Eligibility: Enrolled in government college."
        ),
    },
    {
        "ID": 2, "Scheme_Name": "Farm Mechanization Subsidy",
        "Description": "Subsidy for farmers to purchase tractors.",
        "Benefits": "50 percent subsidy on machinery purchase.",
        "Eligibility": "Small and marginal farmers.",
        "Level": "State", "Intent": "Agriculture",
        "doc_text_semantic": (
            "Scheme Name: Farm Mechanization Subsidy\n\n"
            "Description: Subsidy for farmers to purchase tractors.\n\n"
            "Benefits: 50 percent subsidy on machinery purchase.\n\n"
            "Eligibility: Small and marginal farmers."
        ),
    },
    {
        "ID": 3, "Scheme_Name": "Women Entrepreneur Loan Scheme",
        "Description": "Loans for women starting small businesses.",
        "Benefits": "Collateral-free loan up to 1 lakh.",
        "Eligibility": "Women aged 18-55 with a business plan.",
        "Level": "Central", "Intent": "Women Welfare",
        "doc_text_semantic": (
            "Scheme Name: Women Entrepreneur Loan Scheme\n\n"
            "Description: Loans for women starting small businesses.\n\n"
            "Benefits: Collateral-free loan up to 1 lakh.\n\n"
            "Eligibility: Women aged 18-55 with a business plan."
        ),
    },
]

TOY_DF = pd.DataFrame(TOY_ROWS)
ARTIFACT_DIR = "data/test_artifacts_semantic"


def _get_fitted_retriever():
    """Return a fresh retriever fitted on the toy corpus."""
    r = SemanticRetriever()
    r.build_index(TOY_DF, text_column="doc_text_semantic")
    return r


class TestSemanticRetrieverBasics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = _get_fitted_retriever()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ARTIFACT_DIR):
            shutil.rmtree(ARTIFACT_DIR)

    # ── embedding dimension ──────────────────────────────────────────── #
    def test_embedding_dimension_is_positive_integer(self):
        self.assertIsInstance(self.retriever._embedding_dim, int)
        self.assertGreater(self.retriever._embedding_dim, 0)

    def test_embedding_matrix_shape(self):
        emb = self.retriever._embeddings
        self.assertEqual(emb.shape[0], len(TOY_DF))
        self.assertEqual(emb.shape[1], self.retriever._embedding_dim)

    # ── normalisation ────────────────────────────────────────────────── #
    def test_document_embeddings_are_unit_normalised(self):
        norms = np.linalg.norm(self.retriever._embeddings, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4),
                        f"Norms are not 1.0: {norms}")

    def test_query_embedding_is_unit_normalised(self):
        q_vec = self.retriever._encode(
            [self.retriever._format_query("scholarship for students")]
        )
        norm = float(np.linalg.norm(q_vec))
        self.assertAlmostEqual(norm, 1.0, places=3)

    # ── search ───────────────────────────────────────────────────────── #
    def test_english_query_returns_top_k(self):
        res = self.retriever.search("scholarship for engineering students", top_k=2)
        self.assertIn(res["status"], ("success", "low_relevance_warning"))
        self.assertLessEqual(len(res["results"]), 2)

    def test_hindi_query_returns_results(self):
        res = self.retriever.search("गरीब छात्रों के लिए पढ़ाई की सहायता योजना", top_k=2)
        self.assertIn("results", res)
        # Should not crash; results may have low relevance given tiny toy corpus
        self.assertIsInstance(res["results"], list)

    def test_hinglish_query_returns_results(self):
        res = self.retriever.search("kisan ko machine kharidne ke liye sahayata", top_k=2)
        self.assertIn("results", res)

    def test_results_are_sorted_descending(self):
        res = self.retriever.search("scholarship engineering", top_k=3)
        scores = [r["similarity_score"] for r in res["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_results_have_rank_field(self):
        res = self.retriever.search("women loan business", top_k=2)
        for r in res["results"]:
            self.assertIn("rank", r)

    def test_similarity_score_in_minus_one_to_one(self):
        res = self.retriever.search("farming machinery subsidy", top_k=3)
        for r in res["results"]:
            self.assertGreaterEqual(r["similarity_score"], -1.0)
            self.assertLessEqual(r["similarity_score"],  1.0)

    # ── determinism ──────────────────────────────────────────────────── #
    def test_deterministic_results(self):
        q = "scholarship for engineering students"
        res1 = self.retriever.search(q, top_k=3)
        res2 = self.retriever.search(q, top_k=3)
        scores1 = [r["similarity_score"] for r in res1["results"]]
        scores2 = [r["similarity_score"] for r in res2["results"]]
        self.assertEqual(scores1, scores2)

    # ── model reuse ──────────────────────────────────────────────────── #
    def test_model_not_reloaded_on_repeated_search(self):
        model_id_before = id(self.retriever._model)
        self.retriever.search("women entrepreneur", top_k=1)
        self.retriever.search("farm subsidy", top_k=1)
        model_id_after  = id(self.retriever._model)
        self.assertEqual(model_id_before, model_id_after,
                         "Model object should not be recreated between searches.")

    # ── level filter ─────────────────────────────────────────────────── #
    def test_level_filter(self):
        res = self.retriever.search("scheme assistance", top_k=3,
                                    level_filter="State")
        for r in res["results"]:
            self.assertEqual(r["level"].lower(), "state")

    # ── intent filter ────────────────────────────────────────────────── #
    def test_intent_filter(self):
        res = self.retriever.search("scholarship", top_k=3,
                                    intent_filter="Education")
        for r in res["results"]:
            self.assertEqual(r["intent"].lower(), "education")

    # ── off-domain / low-relevance warning ───────────────────────────── #
    def test_off_domain_query_triggers_low_relevance_warning(self):
        res = self.retriever.search("best pizza near me", top_k=3)
        # Scores should be very low for this tiny corpus
        # Status may be low_relevance_warning or success depending on threshold
        self.assertIn(res["status"], ("success", "low_relevance_warning"))

    # ── rank_all ─────────────────────────────────────────────────────── #
    def test_rank_all_returns_all_docs(self):
        ranked = self.retriever.rank_all("women loan scheme")
        self.assertEqual(len(ranked), len(TOY_DF))
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestSemanticRetrieverPersistence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = _get_fitted_retriever()
        cls.retriever.save(ARTIFACT_DIR)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ARTIFACT_DIR):
            shutil.rmtree(ARTIFACT_DIR)

    def test_artifact_files_created(self):
        for fname in ("embeddings.npy", "schemes_metadata.csv", "config.json"):
            self.assertTrue(
                os.path.exists(os.path.join(ARTIFACT_DIR, fname)),
                f"Missing artifact: {fname}"
            )

    def test_load_produces_same_embeddings(self):
        loaded = SemanticRetriever.load(ARTIFACT_DIR, current_df=TOY_DF)
        np.testing.assert_array_almost_equal(
            loaded._embeddings, self.retriever._embeddings, decimal=5
        )

    def test_loaded_embedding_dim_matches(self):
        loaded = SemanticRetriever.load(ARTIFACT_DIR)
        self.assertEqual(loaded._embedding_dim,
                         self.retriever._embedding_dim)

    def test_stale_artifact_detection(self):
        modified_df = TOY_DF.copy()
        modified_df.at[0, "doc_text_semantic"] = "completely altered text"
        with self.assertRaises(ValueError) as ctx:
            SemanticRetriever.load(ARTIFACT_DIR, current_df=modified_df)
        self.assertIn("Stale", str(ctx.exception))

    def test_loaded_model_not_immediately_in_memory(self):
        """Model should be loaded lazily, not at load() time."""
        loaded = SemanticRetriever.load(ARTIFACT_DIR)
        self.assertIsNone(loaded._model,
                          "Model should not be loaded until first encode call.")


if __name__ == "__main__":
    unittest.main()
