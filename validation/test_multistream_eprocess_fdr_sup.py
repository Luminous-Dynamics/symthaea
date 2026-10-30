import math
import unittest

from multistream_eprocess_fdr_sup import (
    adjuster_a1, adjuster_a2, adjusted_running_max_ehb, async_order_witness,
    e_bonferroni_adjusted, ehb_rejections, log_adjuster_a1, log_adjuster_a2,
    simulate_paper_counterexample,
)


class MultiStreamEProcessTests(unittest.TestCase):
    def test_adjuster_boundary_values(self):
        self.assertAlmostEqual(adjuster_a1(1.0), 0.5, places=12)
        self.assertEqual(adjuster_a2(1.0), 0.0)
        self.assertAlmostEqual(adjuster_a1(2.0),
            (2.0-1.0-math.log(2.0))/math.log(2.0)**2, places=12)
        self.assertAlmostEqual(adjuster_a2(4.0), 1.0, places=12)

    def test_adjusters_are_monotone_on_domain(self):
        values = [1.0, 1.001, 1.1, 2.0, 10.0, 1000.0]
        a1 = [adjuster_a1(x) for x in values]
        a2 = [adjuster_a2(x) for x in values]
        self.assertEqual(a1, sorted(a1))
        self.assertEqual(a2, sorted(a2))

    def test_adjuster_log_forms_match_direct_values(self):
        for m in (1.0, 1.01, 2.0, 5.0, 100.0):
            self.assertAlmostEqual(math.exp(log_adjuster_a1(math.log(m))), adjuster_a1(m), places=10)
            if m > 1.0:
                self.assertAlmostEqual(math.exp(log_adjuster_a2(math.log(m))), adjuster_a2(m), places=10)

    def test_e_bh_rejects_top_rank_using_k_over_alpha_k_threshold(self):
        self.assertEqual(ehb_rejections([math.log(100.0), math.log(3.0)], 0.05), {0})
        self.assertEqual(ehb_rejections([math.log(30.0), math.log(25.0)], 0.05), {0, 1})
        self.assertEqual(ehb_rejections([math.log(10.0), math.log(5.0)], 0.05), set())

    def test_async_gathering_order_can_change_current_ebh_decision(self):
        witness = async_order_witness()
        self.assertEqual(witness["time_t"]["rejections"], [])
        self.assertEqual(witness["time_t_plus_1_both_streams_updated"]["rejections"], [])
        self.assertEqual(witness["time_t_plus_1_stream_1_frozen"]["rejections"], [0, 1])

    def test_adjusted_max_e_bh_and_bonferroni_validate_arguments(self):
        self.assertEqual(adjusted_running_max_ehb([0.0, 0.0], 0.05, "a2"), set())
        with self.assertRaises(ValueError):
            adjusted_running_max_ehb([0.0], 0.05, "bad")
        self.assertEqual(e_bonferroni_adjusted([math.log(1000.0), 0.0], 0.05, [0.5, 0.5]), set())
        with self.assertRaises(ValueError):
            e_bonferroni_adjusted([0.0, 0.0], 0.05, [0.2, 0.2])
        with self.assertRaises(ValueError):
            e_bonferroni_adjusted([0.0], 0.05, [1.0], "bad")

    def test_paper_counterexample_reproduces_running_maximum_fdr_inflation(self):
        result = simulate_paper_counterexample(repetitions=100_000, horizon=100, seed=20261030)
        methods = result["methods"]
        self.assertGreater(methods["naive_running_max_ebh"]["false_discovery_rate_all_null"], 0.05)
        self.assertLessEqual(methods["adjusted_maximum_a1_ebh"]["wilson95"][1], 0.05)
        self.assertLessEqual(methods["adjusted_maximum_a2_ebh"]["wilson95"][1], 0.05)
        self.assertLessEqual(methods["current_ebh"]["wilson95"][1], 0.05)

    def test_independent_per_stream_martingale_increments_have_unit_mean(self):
        # Each coordinate in {(1/2,1/2),(2,1/2),(1/2,2)} has mean one.
        self.assertAlmostEqual((0.5+2.0+0.5)/3.0, 1.0)
        self.assertAlmostEqual((0.5+0.5+2.0)/3.0, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
