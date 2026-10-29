import unittest
import numpy as np
from huber_catoni_reference import catoni_huber_cs

class CatoniReferenceTests(unittest.TestCase):
    def test_validation_rejects_invalid_epsilon(self):
        with self.assertRaises(ValueError): catoni_huber_cs([0,1],alpha=.05,eps=1.0,moment_bound=.25)

    def test_interval_is_ordered_and_in_support(self):
        xs=np.array([0,1,0,0,1,0,0,0,1,0]*30,dtype=float)
        for eps in (0.0,0.05,0.18):
            lo,hi=catoni_huber_cs(xs,alpha=.025,eps=eps,moment_bound=.25)
            self.assertGreaterEqual(lo,0.0); self.assertLessEqual(hi,1.0); self.assertLessEqual(lo,hi)

    def test_interval_contains_constant_mean_observations_at_small_n(self):
        xs=np.full(64,0.4)
        lo,hi=catoni_huber_cs(xs,alpha=.025,eps=.1,moment_bound=.25)
        self.assertLessEqual(lo,0.4); self.assertGreaterEqual(hi,0.4)

    def test_epsilon_increase_does_not_claim_narrower_interval(self):
        rng=np.random.default_rng(18); xs=rng.binomial(1,.32,size=1024).astype(float)
        intervals=[catoni_huber_cs(xs,alpha=.025,eps=e,moment_bound=.25) for e in (0.0,.05,.1,.18)]
        widths=[hi-lo for lo,hi in intervals]
        self.assertTrue(all(widths[i] <= widths[i+1]+1e-9 for i in range(len(widths)-1)))

if __name__=="__main__": unittest.main(verbosity=2)
