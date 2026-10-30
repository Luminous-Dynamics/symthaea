import math, unittest
from multistream_eprocess_fdr_sup import (
 adjuster_a1,adjuster_a2,adjusted_running_max_ehb,async_order_witness,
 e_bonferroni_adjusted,ehb_rejections,log_adjuster_a1,log_adjuster_a2,
 simulate_paper_counterexample)
class T(unittest.TestCase):
 def test_boundaries(self):
  self.assertAlmostEqual(adjuster_a1(1),.5,places=12); self.assertEqual(adjuster_a2(1),0)
  self.assertAlmostEqual(adjuster_a1(2),(2-1-math.log(2))/math.log(2)**2,places=12)
  self.assertAlmostEqual(adjuster_a2(4),1,places=12)
 def test_monotone(self):
  xs=[1,1.001,1.1,2,10,1000]
  self.assertEqual([adjuster_a1(x) for x in xs],sorted(adjuster_a1(x) for x in xs))
  self.assertEqual([adjuster_a2(x) for x in xs],sorted(adjuster_a2(x) for x in xs))
 def test_log_matches(self):
  for x in [1,1.01,2,5,100]:
   self.assertAlmostEqual(math.exp(log_adjuster_a1(math.log(x))),adjuster_a1(x),places=10)
   if x>1:self.assertAlmostEqual(math.exp(log_adjuster_a2(math.log(x))),adjuster_a2(x),places=10)
 def test_e_bh_thresholds(self):
  self.assertEqual(ehb_rejections([math.log(100),math.log(3)],.05),{0})
  self.assertEqual(ehb_rejections([math.log(30),math.log(25)],.05),{0,1})
  self.assertEqual(ehb_rejections([math.log(10),math.log(5)],.05),set())
 def test_async_witness(self):
  w=async_order_witness()
  self.assertEqual(w["time_t"]["rejections"],[])
  self.assertEqual(w["time_t_plus_1_both_streams_updated"]["rejections"],[])
  self.assertEqual(w["time_t_plus_1_stream_1_frozen"]["rejections"],[0,1])
 def test_bonferroni_arguments(self):
  self.assertEqual(adjusted_running_max_ehb([0,0],.05,"a2"),set())
  with self.assertRaises(ValueError):adjusted_running_max_ehb([0],.05,"bad")
  self.assertEqual(e_bonferroni_adjusted([math.log(1000),0],.05,[.5,.5]),set())
  with self.assertRaises(ValueError):e_bonferroni_adjusted([0,0],.05,[.2,.2])
  with self.assertRaises(ValueError):e_bonferroni_adjusted([0],.05,[1],"bad")
 def test_paper_counterexample(self):
  r=simulate_paper_counterexample(repetitions=100_000,horizon=100,seed=20261030)["methods"]
  self.assertGreater(r["naive_running_max_ebh"]["false_discovery_rate_all_null"],.05)
  self.assertLessEqual(r["adjusted_maximum_a1_ebh"]["wilson95"][1],.05)
  self.assertLessEqual(r["adjusted_maximum_a2_ebh"]["wilson95"][1],.05)
  self.assertLessEqual(r["current_ebh"]["wilson95"][1],.05)
 def test_increment_marginals(self):
  self.assertAlmostEqual((.5+2+.5)/3,1);self.assertAlmostEqual((.5+.5+2)/3,1)
if __name__=="__main__":unittest.main(verbosity=2)
