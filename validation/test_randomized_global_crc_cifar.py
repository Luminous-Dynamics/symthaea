import unittest
import numpy as np
from randomized_global_crc_cifar import (
    RandomizedCommonDepthCRC, finite_population_state_weights,
    residual_test_probabilities
)


class RandomizedGlobalCommonDepthTests(unittest.TestCase):
    def test_hypergeometric_state_weights_are_exactly_normalized(self):
        states,weights=finite_population_state_weights(5,(70,20,10))
        self.assertEqual(len(states),21)
        self.assertAlmostEqual(float(weights.sum()),1.0,places=12)
        self.assertTrue(np.all(weights>=0))

    def test_residual_test_distribution_depends_on_calibration_counts(self):
        q1=residual_test_probabilities((70,20,10),(5,0,0))
        q2=residual_test_probabilities((70,20,10),(0,5,0))
        self.assertFalse(np.allclose(q1,q2))
        self.assertAlmostEqual(float(q1.sum()),1.0,places=12)

    def test_randomized_policy_satisfies_every_augmented_constraint(self):
        engine=RandomizedCommonDepthCRC(5,.2,(70,20,10))
        result=engine.solve()
        self.assertTrue(result["verification"]["valid"])
        self.assertEqual(result["verification"]["augmented_states_checked"],len(engine.augmented))
        self.assertLessEqual(result["verification"]["max_budget_excess"],1e-9)

    def test_action_probabilities_are_normalized(self):
        engine=RandomizedCommonDepthCRC(4,.15,(70,20,10))
        result=engine.solve()
        P=np.asarray(result["probabilities"])
        self.assertEqual(P.shape,(len(engine.states),3))
        self.assertTrue(np.all(P>=-1e-9))
        self.assertTrue(np.allclose(P.sum(axis=1),1.0,atol=1e-9))

    def test_hierarchy_metrics_match_expected_nodes(self):
        engine=RandomizedCommonDepthCRC(4,.2,(70,20,10))
        result=engine.solve()
        metrics=result["expected_metrics"]
        self.assertGreaterEqual(metrics["correct_specificity"],0.)
        self.assertLessEqual(metrics["correct_specificity"],1.)
        self.assertGreaterEqual(metrics["mean_node_size"],1.)
        self.assertLessEqual(metrics["mean_node_size"],100.)

    def test_published_scale_has_1326_and_1378_states(self):
        engine=RandomizedCommonDepthCRC(50,.1,(7300,969,1731))
        self.assertEqual(len(engine.states),1326)
        self.assertEqual(len(engine.augmented),1378)

    def test_all_paper_alpha_levels_pass_exact_constraints(self):
        for alpha in (.02,.05,.10,.15):
            engine=RandomizedCommonDepthCRC(50,alpha,(7300,969,1731))
            result=engine.solve()
            self.assertTrue(result["verification"]["valid"])
            self.assertLessEqual(result["verification"]["max_budget_excess"],1e-8)

    def test_invalid_population_and_action_grid_fail_closed(self):
        with self.assertRaises(ValueError):
            RandomizedCommonDepthCRC(50,.1,(10,10,10))
        with self.assertRaises(ValueError):
            RandomizedCommonDepthCRC(5,.1,(70,20,10),action_grid=(0.,.5,2.))


if __name__=="__main__":
    unittest.main(verbosity=2)
