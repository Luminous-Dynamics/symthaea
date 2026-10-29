import unittest
from open_world_reference_stress import fixtures, evaluate, action_components, run, self_test

class OpenWorldReferenceStressTests(unittest.TestCase):
    def setUp(self):
        self.models, self.narrow, self.broad, self.heldout, self.actions = fixtures()

    def test_reference_banks_exclude_exact_heldout_models(self):
        all_refs = self.narrow + self.broad
        for world in self.heldout.values():
            self.assertFalse(any((ref.parents, ref.truth_tables) == (world.parents, world.truth_tables) for ref in all_refs))

    def test_run_grid_complete_deterministic(self):
        first, second = run("test"), run("test")
        self.assertEqual(first, second)
        self.assertEqual(len(first["rows"]), 54)

    def test_auc_bounded_and_zero_support_explicit(self):
        data = run("test")
        self.assertTrue(all(0.0 <= row["ood_surprise_auc"] <= 1.0 for row in data["rows"]))
        bad = [row for row in data["rows"] if row["regime"] == "misspecified_partial_low_noise"]
        self.assertTrue(any(row["in_class_zero_support_mass"] > 0.0 for row in bad))

    def test_broad_reference_bank_is_used_in_divergence(self):
        differences = []
        for action in self.actions:
            _, narrow_jsd = action_components(self.models, (0.5, 0.5), self.narrow, action, 0.1, 1.0)
            _, broad_jsd = action_components(self.models, (0.5, 0.5), self.broad, action, 0.1, 1.0)
            differences.append(abs(narrow_jsd - broad_jsd))
        self.assertTrue(any(value > 1e-8 for value in differences))

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
