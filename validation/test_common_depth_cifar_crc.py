import unittest
from common_depth_cifar_crc import (
    CommonDepthCIFARCRC, largest_piecewise_linear_feasible, count_vectors, loss
)


class CommonDepthCRCTests(unittest.TestCase):
    def test_count_vector_table_sizes_match_paper(self):
        self.assertEqual(len(count_vectors(50,3)),1326)
        self.assertEqual(len(count_vectors(51,3)),1378)

    def test_piecewise_envelope_baseline_hand_cases(self):
        # For h1=0,h2=0 and budget=(n+1)*alpha=5.1, envelope can reach depth 2.
        self.assertEqual(largest_piecewise_linear_feasible(0,0,5.1,True),2.0)
        # With only the type-2 term and envelope coefficient, solve (h2+1)*a/2 <= B.
        self.assertAlmostEqual(largest_piecewise_linear_feasible(0,10,2.0,True),4/11,places=12)

    def test_loss_table_matches_appendix_b7(self):
        self.assertEqual(loss(0,1.7),0.)
        self.assertAlmostEqual(loss(1,.9),0.)
        self.assertAlmostEqual(loss(1,1.8),.4)
        self.assertAlmostEqual(loss(2,1.8),.9)

    def test_crc_baseline_valid_on_every_augmented_multiset(self):
        engine=CommonDepthCIFARCRC(50,.1)
        for kind in ("crc","envelope"):
            controller=engine.baseline(kind)
            report=engine.validity_report(controller)
            self.assertTrue(report["valid"],report)
            self.assertEqual(report["number_augmented_states"],1378)

    def test_equal_share_rounds_valid_and_never_reduce_depth(self):
        engine=CommonDepthCIFARCRC(50,.1)
        controller=engine.baseline("envelope")
        for _ in range(20):
            updated=engine.equal_share_round(controller)
            self.assertTrue(engine.validity_report(updated)["valid"])
            for h in engine.states:
                self.assertGreaterEqual(updated[h]+1e-12,controller[h])
            controller=updated

    def test_expected_prediction_metrics(self):
        engine=CommonDepthCIFARCRC(50,.1)
        root=engine.expected_metrics(0.,2)
        superclass=engine.expected_metrics(1.,1)
        leaf_correct=engine.expected_metrics(2.,0)
        self.assertEqual(root["expected_node_size"],100.)
        self.assertAlmostEqual(superclass["expected_node_size"],5.)
        self.assertEqual(leaf_correct["expected_node_size"],1.)
        self.assertEqual(root["risk"],0.)
        self.assertAlmostEqual(superclass["correct_specificity"],__import__("math").log(20)/__import__("math").log(100))
        self.assertEqual(leaf_correct["correct_specificity"],1.)


    def test_pool_sampler_uses_published_finite_population_counts(self):
        from common_depth_cifar_crc import _sample_finite_population_splits
        splits=_sample_finite_population_splits((7300,969,1731),50,16,300,20261031)
        self.assertEqual(len(splits),300)
        for calibration,test in splits:
            self.assertEqual(sum(calibration),50)
            self.assertEqual(sum(test),16)
            self.assertTrue(all(0<=x<=50 for x in calibration))
            self.assertTrue(all(0<=x<=16 for x in test))

    def test_common_depth_split_metric_aggregation(self):
        engine=CommonDepthCIFARCRC(50,.1)
        controller=engine.baseline("envelope")
        h=(48,1,1)
        test=(8,4,4)
        got=engine.aggregate_test_metrics(controller,h,test)
        depth=controller[h]
        manual={key:sum(test[z]*engine.expected_metrics(depth,z)[key] for z in range(3))/16
                for key in ("risk","expected_node_size","correct_specificity")}
        for key in manual:
            self.assertAlmostEqual(got[key],manual[key],places=12)

    def test_standard_and_envelope_baselines_are_valid_at_all_paper_levels(self):
        for alpha in (.02,.05,.10,.15):
            engine=CommonDepthCIFARCRC(50,alpha)
            self.assertTrue(engine.validity_report(engine.baseline("crc"))["valid"])
            self.assertTrue(engine.validity_report(engine.baseline("envelope"))["valid"])

    def test_invalid_alpha_and_n_rejected(self):
        with self.assertRaises(ValueError):CommonDepthCIFARCRC(0,.1)
        with self.assertRaises(ValueError):CommonDepthCIFARCRC(50,1.0)

    def test_crc_recertification_round_is_monotone(self):
        engine=CommonDepthCIFARCRC(3,.2)
        baseline=engine.baseline("envelope")
        once=engine.equal_share_round(baseline)
        twice=engine.equal_share_round(once)
        for h in engine.states:
            self.assertGreaterEqual(once[h]+1e-12,baseline[h])
            self.assertGreaterEqual(twice[h]+1e-12,once[h])
        self.assertTrue(engine.validity_report(twice)["valid"])


if __name__=="__main__":
    unittest.main(verbosity=2)
