import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import composite_null_test as t

class CompositeNullTests(unittest.TestCase):
    def test_monte_carlo_pvalue_is_superuniform_for_exchangeable_ranks(self):
        b=19; values=np.arange(b+1,dtype=float); pvals=[]
        for held in range(b+1):
            pvals.append(t.lower_tail_mc_pvalues(np.delete(values,held),np.array([values[held]]))[0])
        self.assertEqual(sorted(pvals),[(k+1)/(b+1) for k in range(b+1)])
        self.assertEqual(sum(p<=0.05 for p in pvals),1)
    def test_ties_are_conservative(self):
        cal=np.array([1.0,1.0,2.0,3.0]); ev=np.array([1.0,1.5,4.0])
        self.assertTrue(np.allclose(t.lower_tail_mc_pvalues(cal,ev),[3/5,3/5,1.0]))
    def test_union_null_rejects_only_if_all_candidates_rejected(self):
        self.assertEqual(t.classify_from_model_pvalues(np.array([.01]*6),.05),"model_misspecified")
        for i in range(6):
            p=np.array([.01]*6); p[i]=.06
            self.assertEqual(t.classify_from_model_pvalues(p,.05),"catalogue_not_rejected")
    def test_full_likelihood_preparation_normalizes(self):
        p=t._prepare(t.base.generate_recipe(3)["_params"])
        for probe in t.base.PROBES:
            self.assertTrue(np.allclose(p["candidate_prob"][probe].sum(axis=1),1.0))
            self.assertAlmostEqual(p["ood_prob"][probe].sum(),1.0)
    def test_one_candidate_must_explain_all_probes(self):
        counts=[np.array([[1,0]]),np.array([[0,1]])]
        logs=[np.array([[0.,-10.],[-10.,0.]]),np.array([[0.,-10.],[-10.,0.]])]
        self.assertAlmostEqual(float(t._max_joint_loglik(counts,logs)[0]),-10.0)
    def test_simulation_shapes_and_determinism(self):
        p=t._prepare(t.base.generate_recipe(4)["_params"])
        a=t._simulate_scores(2,25,"ood",None,p,np.random.default_rng(101))
        b=t._simulate_scores(2,25,"ood",None,p,np.random.default_rng(101))
        self.assertEqual(a.shape,(25,)); self.assertTrue(np.array_equal(a,b))
    def test_each_candidate_has_independent_calibration_stream(self):
        seeds=[t._stable_seed("suite","cal",m) for m in t.base.MODELS]
        self.assertEqual(len(seeds),len(set(seeds)))
    def test_small_evaluation_has_per_model_false_alarm_metrics(self):
        r=t.evaluate(2,(1,),50,30)
        row=r["aggregate"]["1"]
        self.assertEqual(set(row["in_family_false_ood_by_model_mean_over_recipes"]),set(t.base.MODELS))
        self.assertTrue(0<=row["worst_candidate_false_ood_rate"]<=1)
        self.assertTrue(0<=row["ood_detection_rate_mean_over_recipes"]<=1)
    def test_fail_closed_inputs(self):
        with self.assertRaises(ValueError): t.lower_tail_mc_pvalues(np.array([]),np.array([1.]))
        with self.assertRaises(ValueError): t.classify_from_model_pvalues(np.array([.1]),.05)
        with self.assertRaises(ValueError): t.classify_from_model_pvalues(np.array([.1]*6),.5)
        with self.assertRaises(ValueError): t.evaluate(seed_count=0)

if __name__=="__main__": unittest.main(verbosity=2)
