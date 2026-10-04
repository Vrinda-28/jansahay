"""
Unit tests for jansahay/evaluation/metrics.py
Uses small manually-constructed examples to verify mathematical correctness.
"""
import sys, os, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jansahay.evaluation.metrics import (
    hit_at_k, recall_at_k, precision_at_k, reciprocal_rank, average_metrics
)

class TestHitAtK(unittest.TestCase):

    def test_hit1_when_first_result_is_relevant(self):
        self.assertEqual(hit_at_k([1, 2, 3], {1, 5}, 1), 1.0)

    def test_hit1_when_first_result_not_relevant(self):
        self.assertEqual(hit_at_k([2, 3, 4], {1, 5}, 1), 0.0)

    def test_hit5_relevant_at_rank5(self):
        self.assertEqual(hit_at_k([2, 3, 4, 6, 1], {1}, 5), 1.0)

    def test_hit5_relevant_beyond_k(self):
        self.assertEqual(hit_at_k([2, 3, 4, 6, 7, 1], {1}, 5), 0.0)

    def test_hit_empty_relevant(self):
        self.assertEqual(hit_at_k([1, 2, 3], set(), 5), 0.0)

    def test_hit_empty_retrieved(self):
        self.assertEqual(hit_at_k([], {1, 2}, 5), 0.0)

    def test_hit_multi_relevant(self):
        # query has two relevant; one appears at rank 3
        self.assertEqual(hit_at_k([5, 6, 10, 20, 30], {10, 25}, 5), 1.0)


class TestRecallAtK(unittest.TestCase):

    def test_recall_perfect(self):
        # All 2 relevant items in top 2
        self.assertAlmostEqual(recall_at_k([1, 2, 3, 4], {1, 2}, 2), 1.0)

    def test_recall_partial(self):
        # 1 of 2 relevant items in top 2
        self.assertAlmostEqual(recall_at_k([1, 3, 5, 2], {1, 2}, 2), 0.5)

    def test_recall_zero(self):
        self.assertAlmostEqual(recall_at_k([3, 4, 5], {1, 2}, 3), 0.0)

    def test_recall_empty_relevant(self):
        self.assertAlmostEqual(recall_at_k([1, 2, 3], set(), 3), 0.0)

    def test_recall_single_relevant_at_k5(self):
        self.assertAlmostEqual(recall_at_k([10, 2, 3, 4, 5], {10}, 5), 1.0)

    def test_recall_at1_single_relevant_not_at_1(self):
        self.assertAlmostEqual(recall_at_k([2, 10, 3], {10}, 1), 0.0)


class TestPrecisionAtK(unittest.TestCase):

    def test_precision_all_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3], {1, 2, 3}, 3), 1.0)

    def test_precision_half_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3, 4], {1, 2}, 4), 0.5)

    def test_precision_none_relevant(self):
        self.assertAlmostEqual(precision_at_k([3, 4, 5], {1, 2}, 3), 0.0)

    def test_precision_k_zero(self):
        self.assertAlmostEqual(precision_at_k([1, 2], {1}, 0), 0.0)

    def test_precision_at5_one_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 3, 4, 5, 6], {1}, 5), 0.2)

    def test_precision_empty_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3], set(), 3), 0.0)


class TestReciprocalRank(unittest.TestCase):

    def test_rr_relevant_at_rank1(self):
        self.assertAlmostEqual(reciprocal_rank([5, 2, 3], {5}), 1.0)

    def test_rr_relevant_at_rank2(self):
        self.assertAlmostEqual(reciprocal_rank([2, 5, 3], {5}), 0.5)

    def test_rr_relevant_at_rank5(self):
        self.assertAlmostEqual(reciprocal_rank([1, 2, 3, 4, 5], {5}), 0.2)

    def test_rr_not_found(self):
        self.assertAlmostEqual(reciprocal_rank([1, 2, 3], {9, 10}), 0.0)

    def test_rr_empty_relevant(self):
        self.assertAlmostEqual(reciprocal_rank([1, 2, 3], set()), 0.0)

    def test_rr_multi_relevant_first_at_rank2(self):
        # first relevant is rank 2 (id=10), second (id=5) is rank 3
        self.assertAlmostEqual(reciprocal_rank([1, 10, 5], {10, 5}), 0.5)

    def test_rr_empty_retrieved(self):
        self.assertAlmostEqual(reciprocal_rank([], {1, 2}), 0.0)


class TestAverageMetrics(unittest.TestCase):

    def test_average_basic(self):
        records = [
            {"hit@5": 1.0, "mrr": 0.5},
            {"hit@5": 0.0, "mrr": 0.0},
        ]
        result = average_metrics(records)
        self.assertAlmostEqual(result["hit@5"], 0.5)
        self.assertAlmostEqual(result["mrr"],   0.25)

    def test_average_empty(self):
        self.assertEqual(average_metrics([]), {})

    def test_n_queries(self):
        records = [{"mrr": 0.5}, {"mrr": 1.0}, {"mrr": 0.0}]
        self.assertEqual(average_metrics(records)["n_queries"], 3)


class TestMultiRelevantEdgeCases(unittest.TestCase):
    """
    Simulate dataset-realistic scenario:
    one query → two relevant schemes (IDs 12 and 27).
    """
    RELEVANT = {12, 27}

    def test_hit5_when_only_one_relevant_retrieved(self):
        retrieved = [5, 12, 8, 9, 10]
        self.assertEqual(hit_at_k(retrieved, self.RELEVANT, 5), 1.0)

    def test_recall5_partial(self):
        retrieved = [5, 12, 8, 9, 10]
        self.assertAlmostEqual(recall_at_k(retrieved, self.RELEVANT, 5), 0.5)

    def test_recall5_full(self):
        retrieved = [12, 27, 3, 4, 5]
        self.assertAlmostEqual(recall_at_k(retrieved, self.RELEVANT, 5), 1.0)

    def test_precision5_two_hits(self):
        retrieved = [12, 27, 3, 4, 5]
        self.assertAlmostEqual(precision_at_k(retrieved, self.RELEVANT, 5), 0.4)

    def test_mrr_first_relevant_at_rank2(self):
        retrieved = [5, 12, 27, 4, 6]
        self.assertAlmostEqual(reciprocal_rank(retrieved, self.RELEVANT), 0.5)

    def test_zero_relevant_schemes_all_metrics_zero(self):
        """Edge case: query has no annotated relevant schemes."""
        retrieved = [1, 2, 3, 4, 5]
        relevant  = set()
        self.assertEqual(hit_at_k(retrieved,       relevant, 5), 0.0)
        self.assertEqual(recall_at_k(retrieved,    relevant, 5), 0.0)
        self.assertEqual(precision_at_k(retrieved, relevant, 5), 0.0)
        self.assertEqual(reciprocal_rank(retrieved, relevant),    0.0)

    def test_duplicate_query_all_relevant_ids_unioned(self):
        """
        Verify that when a query appears for schemes 3 and 7,
        both are counted as relevant (simulates query reuse scenario).
        """
        relevant = {3, 7}
        retrieved = [7, 2, 3, 8, 9]
        self.assertEqual(recall_at_k(retrieved, relevant, 5), 1.0)


if __name__ == "__main__":
    unittest.main()
