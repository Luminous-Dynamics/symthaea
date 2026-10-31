import unittest
import numpy as np
from global_crc_utility_lp import (
    GlobalLinearCRCOptimizer, multinomial_state_weights,
    equal_share_two_type_reference
)


class GlobalCRCUtilityLPTests(unittest.TestCase):
    def setUp(self):
        self.engine=GlobalLinearCRCOptimizer(2,[1.0,.5],.5)
        self.states=self.engine.states
        self.equal=equal_share_two_type_reference()
        self.equal_actions=np.asarray([self.equal[D] for D in self.states])

    def test_reference_weighted_lp_is_exactly_valid_for_every_augmented_state(self):
        _,weights=multinomial_state_weights(2,[.1,.9])
        result=self.engine.solve_reference_weighted(weights)
        self.assertTrue(result["verification"]["valid"])
        self.assertEqual(result["verification"]["augmented_states_checked"],4)

    def test_reference_weighted_objective_beats_equal_share_when_state_weights_skewed(self):
        _,weights=multinomial_state_weights(2,[.1,.9])
        result=self.engine.solve_reference_weighted(weights)
        equal_utility=float(np.dot(weights,1-self.equal_actions))
        self.assertAlmostEqual(equal_utility,.82625,places=10)
        self.assertAlmostEqual(result["expected_utility"],.905,places=10)
        self.assertGreater(result["expected_utility"],equal_utility+.07)
        by_state=dict(zip(self.states,result["actions"]))
        self.assertLess(by_state[(0,2)],self.equal[(0,2)])
        self.assertGreater(by_state[(1,1)],self.equal[(1,1)])
        self.assertAlmostEqual(by_state[(2,0)],self.equal[(2,0)],places=8)

    def test_reference_weight_sensitivity_and_tie_case(self):
        _,w50=multinomial_state_weights(2,[.5,.5])
        result=self.engine.solve_reference_weighted(w50)
        self.assertAlmostEqual(result["expected_utility"],.65625,places=9)
        self.assertTrue(np.allclose(result["actions"],self.equal_actions,atol=1e-9,rtol=0.0))
        _,w90=multinomial_state_weights(2,[.9,.1])
        result90=self.engine.solve_reference_weighted(w90)
        equal90=float(np.dot(w90,1-self.equal_actions))
        self.assertAlmostEqual(result90["expected_utility"],equal90,places=9)

    def test_maximin_is_distribution_free_over_calibration_state_weights(self):
        result=self.engine.solve_distribution_free_maximin()
        self.assertTrue(result["verification"]["valid"])
        self.assertAlmostEqual(result["worst_state_utility"],.5,places=9)

    def test_multi_risk_lp_honors_every_risk_simultaneously(self):
        engine=GlobalLinearCRCOptimizer(3,[[1.,.5,.2],[.2,.7,1.]], [.25,.3])
        _,weights=multinomial_state_weights(3,[.2,.3,.5])
        result=engine.solve_reference_weighted(weights)
        self.assertTrue(result["verification"]["valid"])
        self.assertEqual(len(result["verification"]["max_excess_by_risk"]),2)

    def test_randomized_weight_vectors_remain_exactly_valid(self):
        rng=np.random.default_rng(20261031)
        for _ in range(40):
            w=rng.dirichlet(np.ones(len(self.states)))
            result=self.engine.solve_reference_weighted(w)
            self.assertTrue(result["verification"]["valid"])
            self.assertLessEqual(max(result["verification"]["max_excess_by_risk"]),1e-8)

    def test_rejects_invalid_weight_and_risk_inputs(self):
        with self.assertRaises(ValueError):
            self.engine.solve_reference_weighted([.1,.2,.3])
        with self.assertRaises(ValueError):
            GlobalLinearCRCOptimizer(2,[1.,-.1],.5)
        with self.assertRaises(ValueError):
            GlobalLinearCRCOptimizer(2,[1.,.5],[.5,.4])

    def test_reference_weighted_solution_matches_direct_expected_utility(self):
        _,weights=multinomial_state_weights(2,[.2,.8])
        result=self.engine.solve_reference_weighted(weights)
        direct=float(sum(w*(1-a) for w,a in zip(weights,result["actions"])))
        self.assertAlmostEqual(result["expected_utility"],direct,places=12)


if __name__=="__main__":
    unittest.main(verbosity=2)
