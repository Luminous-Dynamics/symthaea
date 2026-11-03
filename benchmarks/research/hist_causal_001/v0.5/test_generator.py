import unittest
from itertools import combinations
import generator as g

class SeededChallengeGeneratorTests(unittest.TestCase):
    def test_same_seed_replays_exact_recipe_and_target(self):
        self.assertEqual(g.generate_recipe(42),g.generate_recipe(42))
    def test_different_seeds_change_recipe_or_target(self):
        self.assertNotEqual(g.generate_recipe(1)["evaluator_key"],g.generate_recipe(2)["evaluator_key"])
    def test_target_and_parameters_not_leaked_to_agent_view(self):
        task=g.generate_recipe(9)
        for key in ("target_model","recipe_parameters","seed","latent_q","sensor_sensitivity"):
            self.assertNotIn(key,task["agent_view"])
    def test_all_distributions_normalize_for_seed_range(self):
        for seed in range(64):
            p=g.generate_recipe(seed)["_params"]
            for m in g.MODELS:
                for probe in g.PROBES:
                    self.assertEqual(sum(g._dist(m,probe,p).values()),1,(seed,m,probe))
    def test_all_pairs_identifiable_with_full_probe_suite(self):
        for seed in range(64):
            p=g.generate_recipe(seed)["_params"]
            for a,b in combinations(g.MODELS,2):
                self.assertFalse(g._equivalent(a,b,p),(seed,a,b))
    def test_observational_alias_is_preserved(self):
        for seed in range(64):
            p=g.generate_recipe(seed)["_params"]
            self.assertEqual(g._dist("genuine_mechanism","observational_primary",p),
                             g._dist("latent_common_cause","observational_primary",p))
    def test_execution_alias_is_preserved_until_readback(self):
        for seed in range(64):
            p=g.generate_recipe(seed)["_params"]
            self.assertEqual(g._dist("genuine_mechanism","randomized_primary",p),
                             g._dist("actuator_noncompliance","randomized_primary",p))
            self.assertNotEqual(g._dist("genuine_mechanism","actuator_readback",p),
                                g._dist("actuator_noncompliance","actuator_readback",p))
    def test_sampling_is_reproducible_and_in_support(self):
        for seed in range(32):
            task=g.generate_recipe(seed); target=task["evaluator_key"]["target_model"]; p=task["_params"]
            for probe in g.PROBES:
                x=g.sample_probe(seed,probe,2,target,p)
                self.assertEqual(x,g.sample_probe(seed,probe,2,target,p))
                self.assertIn(x,g._dist(target,probe,p))
    def test_seed_bounds_fail_closed(self):
        with self.assertRaises(ValueError): g.generate_recipe(-1)
        with self.assertRaises(ValueError): g.generate_recipe(1<<64)
    def test_generated_recipe_validator(self):
        for seed in range(20):
            v=g.validate_generated(g.generate_recipe(seed))
            self.assertTrue(v["valid_mass_and_parameters"])
            self.assertTrue(v["all_models_separated_by_full_suite"])
            self.assertFalse(v["target_in_agent_view"])

if __name__=="__main__": unittest.main(verbosity=2)
