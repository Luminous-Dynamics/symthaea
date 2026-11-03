import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalogue_adequacy as a

class CatalogueAdequacyTests(unittest.TestCase):
    def test_same_data_scores_are_deterministic(self):
        params = a.base.generate_recipe(2)["_params"]
        prepared = a._prepare(params)
        data = a.sample_dataset(3, 4, 2, "candidate", "genuine_mechanism", prepared)
        self.assertEqual(a.best_catalogue_loglik(data, prepared), a.best_catalogue_loglik(data, prepared))
    def test_known_candidate_never_zero_likelihood_on_its_own_support(self):
        params = a.base.generate_recipe(4)["_params"]
        prepared = a._prepare(params)
        for model in a.base.MODELS:
            data = a.sample_dataset(4, 5, 3, "candidate", model, prepared)
            score, _ = a.best_catalogue_loglik(data, prepared)
            self.assertNotEqual(score, float("-inf"))
    def test_calibration_is_least_favourable(self):
        params = a.base.generate_recipe(5)["_params"]
        prepared = a._prepare(params)
        cal = a.calibrate_threshold(7, 2, 30, prepared, 0.05)
        self.assertEqual(cal["threshold"], min(cal["per_model_thresholds"].values()))
    def test_invalid_inputs_fail_closed(self):
        params = a.base.generate_recipe(5)["_params"]
        prepared = a._prepare(params)
        with self.assertRaises(ValueError): a.calibrate_threshold(0, 1, 10, prepared)
        with self.assertRaises(ValueError): a.calibrate_threshold(0, 1, 30, prepared, 0)
        with self.assertRaises(ValueError): a.sample_dataset(1, 0, 0, "ood", None, prepared)
        with self.assertRaises(ValueError): a.sample_dataset(1, 0, 1, "candidate", "bad", prepared)
    def test_small_evaluation_rate_sums(self):
        out = a.evaluate(seed_count=2, n_values=(1,), calibration_reps=20, evaluation_reps=10)
        row = out["aggregate"]["1"]
        self.assertAlmostEqual(row["ood_detection_rate"] + (1-row["ood_detection_rate"]),1.0)
        self.assertAlmostEqual(row["in_family_false_ood_rate_pooled"] + (1-row["in_family_false_ood_rate_pooled"]),1.0)
        self.assertEqual(set(row["in_family_false_ood_rate_by_model"]),set(a.base.MODELS))

if __name__ == "__main__": unittest.main(verbosity=2)
