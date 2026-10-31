"""Deterministic tests for the source-data-free Choro metric harness."""
from __future__ import annotations
import random, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from muse_choro_synthetic_harness import boundary_alignment_rate, conditional_transition_entropy, pearson_correlation, rotate_phrase, run_synthetic_suite, validate_phrases

class ChoroSyntheticHarnessTests(unittest.TestCase):
    def test_complete_suite_passes_without_empirical_claim(self):
        report = run_synthetic_suite()
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["schema"], "muse-choro-synthetic-harness-report-v1")
        self.assertFalse(report["source_data_used"])
        self.assertFalse(report["corpus_data_ingested"])
        self.assertIsNone(report["empirical_results"])
        self.assertFalse(report["runtime_capability_claimed"])

    def test_phrase_alignment_detects_authored_fixture_and_counterexample(self):
        aligned = [[4, 8, 12, 16] for _ in range(64)]
        counterexample = [[1, 2, 3, 4] for _ in range(64)]
        self.assertEqual(boundary_alignment_rate(aligned), 1.0)
        self.assertEqual(boundary_alignment_rate(counterexample), 0.25)
        rotations = [rotate_phrase(aligned[i], i % 16) for i in range(64)]
        self.assertEqual(boundary_alignment_rate(rotations), 0.25)
        self.assertEqual(sum(map(len, aligned)), sum(map(len, rotations)))

    def test_transition_entropy_uses_sequence_order_not_only_counts(self):
        ordered = ["I", "V", "vi", "IV"] * 16
        shuffled = list(ordered)
        random.Random(90210).shuffle(shuffled)
        self.assertEqual(sorted(ordered), sorted(shuffled))
        self.assertLess(conditional_transition_entropy(ordered), conditional_transition_entropy(shuffled))

    def test_trend_metric_detects_reversal(self):
        years = [1900 + 3*i for i in range(40)]
        values = [1 + i // 5 for i in range(40)]
        self.assertGreater(pearson_correlation(years, values), 0.95)
        self.assertLess(pearson_correlation(years, list(reversed(values))), -0.95)

    def test_malformed_phrase_inputs_fail_closed(self):
        for phrases in ([], [[]], [[0, 4]], [[1, 17]], [[1, 1]]):
            with self.subTest(phrases=phrases):
                with self.assertRaises(ValueError):
                    validate_phrases(phrases)

    def test_invalid_entropy_and_correlation_fail_closed(self):
        with self.assertRaises(ValueError):
            conditional_transition_entropy(["I"])
        with self.assertRaises(ValueError):
            conditional_transition_entropy(["I", ""])
        with self.assertRaises(ValueError):
            pearson_correlation([1, 1], [2, 3])
        with self.assertRaises(ValueError):
            pearson_correlation([1, 2], [4, 4])

if __name__ == "__main__":
    unittest.main()
