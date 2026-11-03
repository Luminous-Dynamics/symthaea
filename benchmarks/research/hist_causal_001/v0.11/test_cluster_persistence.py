import unittest
import cluster_persistence as c

class ClusterPersistenceTests(unittest.TestCase):
    def test_reproducible_for_fixed_seed(self):
        cfg={"replicates":25,"permutations":99,"seed":7}
        self.assertEqual(c.run(cfg),c.run(cfg))

    def test_reports_all_three_analysis_units(self):
        out=c.run({"replicates":20,"permutations":49,"seed":9})
        self.assertEqual(set(out["methods"]),{"invalid_participant_level_welch","cluster_mean_welch","cluster_level_randomization_inference"})

    def test_cluster_ri_rate_is_bounded(self):
        out=c.run({"replicates":30,"permutations":99,"seed":11})
        for method in out["methods"].values():
            self.assertTrue(0 <= method["rejection_rate_at_0_05"] <= 1)

    def test_rejects_unbalanced_invalid_configuration(self):
        with self.assertRaises(ValueError):
            c.run({"clusters":60,"treated_clusters":20})

    def test_rejects_zero_permutations(self):
        with self.assertRaises(ValueError):
            c.run({"permutations":0})

if __name__=="__main__":
    unittest.main(verbosity=2)
