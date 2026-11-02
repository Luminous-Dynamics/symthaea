import unittest
from fractions import Fraction as F
from itertools import product
import reference as r

class CausalTriangulationTests(unittest.TestCase):
    def test_every_model_probe_distribution_normalizes(self):
        for m,p in product(r.MODELS,r.PROBES):
            self.assertEqual(sum(r.distribution(m,p).values(),F(0)),F(1),(m,p))
    def test_observational_mechanism_confounded_alias(self):
        self.assertEqual(r.distribution("genuine_mechanism","observational_primary"),r.distribution("latent_common_cause","observational_primary"))
    def test_randomization_breaks_common_cause_alias(self):
        self.assertNotEqual(r.distribution("genuine_mechanism","randomized_primary"),r.distribution("latent_common_cause","randomized_primary"))
    def test_primary_sensor_artifact_mimics_mechanism(self):
        self.assertEqual(r.distribution("genuine_mechanism","randomized_primary"),r.distribution("primary_sensor_artefact","randomized_primary"))
    def test_orthogonal_reference_separates_primary_sensor_artifact(self):
        self.assertNotEqual(r.distribution("genuine_mechanism","independent_reference"),r.distribution("primary_sensor_artefact","independent_reference"))
    def test_execution_failure_mimics_then_readback_separates(self):
        for probe in ("randomized_primary","independent_reference"):
            self.assertEqual(r.distribution("genuine_mechanism",probe),r.distribution("actuator_noncompliance",probe))
        self.assertNotEqual(r.distribution("genuine_mechanism","actuator_readback"),r.distribution("actuator_noncompliance","actuator_readback"))
    def test_common_mode_artifact_defeats_secondary_reference_but_blank_catches(self):
        for probe in ("randomized_primary","independent_reference"):
            self.assertEqual(r.distribution("genuine_mechanism",probe),r.distribution("common_mode_reference_artefact",probe))
        self.assertNotEqual(r.distribution("genuine_mechanism","blank_common_mode_control"),r.distribution("common_mode_reference_artefact","blank_common_mode_control"))
    def test_sensor_self_test(self):
        self.assertNotEqual(r.distribution("genuine_mechanism","sensor_self_test"),r.distribution("primary_sensor_artefact","sensor_self_test"))
    def test_full_suite_separates_all_pairs(self):
        self.assertEqual(r.score(r.PROBES.keys())["pairs_separated"],15)
    def test_minimum_cost_probe_set(self):
        result=r.minimum_cost_suite()
        self.assertTrue(result["identified"])
        self.assertEqual(r.score(result["probes"])["pairs_separated"],15)
    def test_probe_enumeration_order_does_not_change_score(self):
        self.assertEqual(r.score(r.PROBES.keys())["pairs_separated"],r.score(reversed(tuple(r.PROBES)))["pairs_separated"])

if __name__=="__main__": unittest.main(verbosity=2)
