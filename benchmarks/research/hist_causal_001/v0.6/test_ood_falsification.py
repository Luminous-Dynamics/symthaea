import unittest
import sys
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ood_falsification as o

class OutOfFamilyTests(unittest.TestCase):
    def test_all_ood_probe_distributions_normalize(self):
        for seed in range(64):
            p=o.base.generate_recipe(seed)["_params"]
            for probe in o.base.PROBES:
                self.assertEqual(sum(o.ood_distribution(probe,p).values(),F(0)),F(1))
    def test_ood_signature_is_not_any_in_family_signature(self):
        for seed in range(64):
            result=o.validate_ood(seed)
            self.assertGreater(F(result["minimum_average_total_variation"]),0)
    def test_misspecification_is_an_allowed_agent_output(self):
        for seed in range(16):
            self.assertIn("model_misspecified",o.agent_task(seed)["allowed_outputs"])
    def test_target_components_and_mixture_weight_are_not_in_agent_view(self):
        view=o.agent_task(5)
        for forbidden in ("target_model","mixture_components","mixture_weight","recipe_parameters","seed"):
            self.assertNotIn(forbidden,view)
    def test_repeatable_task_and_evaluator_records(self):
        self.assertEqual(o.agent_task(42),o.agent_task(42))
        self.assertEqual(o.evaluator_key(42),o.evaluator_key(42))
    def test_ood_sampling_replays_and_is_supported(self):
        p=o.base.generate_recipe(4)["_params"]
        x=o.sample_ood_probe(4,"randomized_primary",3,p)
        y=o.sample_ood_probe(4,"randomized_primary",3,p)
        private=o.sample_ood_trial_evaluator(4,"randomized_primary",3,p)
        self.assertEqual(x,y)
        self.assertEqual(x,private[0])
        self.assertIn(x,set(o.ood_distribution("randomized_primary",p)))
    def test_weight_bounds_fail_closed(self):
        p=o.base.generate_recipe(3)["_params"]
        for w in (F(0),F(1),F(-1,2),F(3,2)):
            with self.assertRaises(ValueError): o.ood_distribution("randomized_primary",p,w)
    def test_development_matrix_reports_full_seed_count(self):
        result=o.run(100)
        self.assertEqual(result["development_seeds_validated"],100)
        self.assertTrue(result["all_ood_signatures_differ_from_all_six_candidates"])

if __name__=="__main__": unittest.main(verbosity=2)
