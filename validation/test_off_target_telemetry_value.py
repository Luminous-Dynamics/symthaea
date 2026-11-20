import math
import unittest
from off_target_telemetry_value import (
    TELEMETRY_MODES, POLICIES, fixtures, observation_distribution,
    predictive, posterior_update, evaluate, run, self_test,
)
from off_target_kernel_inference import Q_GRID, R_GRID

class OffTargetTelemetryValueTests(unittest.TestCase):
    def setUp(self):
        self.models, self.refs, self.ood, self.actions = fixtures()

    def test_observation_channels_normalize(self):
        for model in (*self.models, *self.refs, *self.ood.values()):
            for action in self.actions:
                for q in Q_GRID:
                    for r in R_GRID:
                        for mode in TELEMETRY_MODES:
                            dist=observation_distribution(model,action,0.1,q,r,mode)
                            self.assertAlmostEqual(sum(dist.values()),1.0,places=12)

    def test_predictive_and_posterior_normalize_with_telemetry(self):
        n=len(self.models)*len(Q_GRID)*len(R_GRID)
        post=tuple([1.0/n]*n)
        for mode in TELEMETRY_MODES:
            dist=predictive(self.models,Q_GRID,R_GRID,post,self.actions[0],0.1,mode)
            self.assertAlmostEqual(sum(dist.values()),1.0,places=12)
            observation=next(iter(dist))
            updated=posterior_update(self.models,Q_GRID,R_GRID,post,self.actions[0],observation,0.1,mode)
            self.assertAlmostEqual(sum(updated),1.0,places=12)

    def test_perfect_telemetry_is_more_informative_about_spill_on_off_target_world(self):
        data=run("test")
        group={}
        for mode in TELEMETRY_MODES:
            rows=[r for r in data["rows"] if r["regime"]=="matched_off_target" and r["policy"]=="graph_eig" and r["telemetry"]==mode]
            group[mode]=sum(r["off_target_information_bits"] for r in rows)/len(rows)
        self.assertGreater(group["noisy_spill_sensor"],group["outcome_only"])
        self.assertGreater(group["perfect_subset_telemetry"],group["noisy_spill_sensor"])

    def test_full_grid_deterministic_and_complete(self):
        a,b=run("test"),run("test")
        self.assertEqual(a,b)
        self.assertEqual(len(a["rows"]),4*2*len(TELEMETRY_MODES)*len(POLICIES))
        self.assertTrue(all(0.0<=r["ood_surprise_auc"]<=1.0 for r in a["rows"]))

    def test_joint_models_keep_support_when_truth_is_on_grid(self):
        data=run("test")
        self.assertTrue(all(r["in_class_zero_support_mass"]<1e-12 for r in data["rows"]))

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
