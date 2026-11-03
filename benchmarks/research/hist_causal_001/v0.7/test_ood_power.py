import unittest
from fractions import Fraction as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ood_power as p

class OODPowerTests(unittest.TestCase):
    def test_js_is_symmetric_nonnegative_and_zero_on_equal(self):
        a = {(0,): F(1,3), (1,): F(2,3)}
        b = {(0,): F(1,3), (1,): F(2,3)}
        c = {(0,): F(1,2), (1,): F(1,2)}
        self.assertEqual(p.jensen_shannon(a,b), 0.0)
        self.assertAlmostEqual(p.jensen_shannon(a,c), p.jensen_shannon(c,a), places=12)
        self.assertGreaterEqual(p.jensen_shannon(a,c), 0.0)
    def test_nearest_candidate_is_in_catalogue_and_separated(self):
        for seed in range(32):
            model, js = p.nearest_candidate(seed)
            self.assertIn(model, p.base.MODELS)
            self.assertGreater(js, 0)
    def test_equal_prior_lr_threshold_corresponds_to_95_percent(self):
        self.assertAlmostEqual(19/(1+19), 0.95)
        self.assertEqual(p.classify_llr(0.0), "abstain")
        self.assertEqual(p.classify_llr(10.0), "ood")
        self.assertEqual(p.classify_llr(-10.0), "candidate")
    def test_log_bayes_factor_replays(self):
        candidate = p.nearest_candidate(4)[0]
        first = p.trial_log_bayes_factor(4,2,7,"ood",candidate)
        second = p.trial_log_bayes_factor(4,2,7,"ood",candidate)
        self.assertEqual(first, second)
    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError): p.trial_log_bayes_factor(2,0,0,"ood",p.base.MODELS[0])
        with self.assertRaises(ValueError): p.trial_log_bayes_factor(2,1,-1,"ood",p.base.MODELS[0])
        with self.assertRaises(ValueError): p.trial_log_bayes_factor(2,1,0,"other",p.base.MODELS[0])
        with self.assertRaises(ValueError): p.evaluate(seed_count=0)
        with self.assertRaises(ValueError): p.evaluate(replicates=0)
        with self.assertRaises(ValueError): p.evaluate(n_values=(1,0))
    def test_small_evaluation_normalizes_decision_rates(self):
        out = p.evaluate(seed_count=4, n_values=(1,2), replicates=10)
        for n in ("1","2"):
            for truth in ("ood","candidate"):
                row = out["outcomes"][n][truth]
                rates = row["rates"]
                self.assertAlmostEqual(sum(rates.values()),1.0,places=12)
                self.assertEqual(row["n_trials"],40)
                self.assertTrue(all(0<=v<=1 for v in rates.values()))
                self.assertEqual(set(row["marginal_95pct_wilson_intervals"]),set(rates))

if __name__ == "__main__":
    unittest.main(verbosity=2)
