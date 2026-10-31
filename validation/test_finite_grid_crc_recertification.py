import unittest
import numpy as np
from finite_grid_crc_recertification import FiniteGridCRC


class FiniteGridCRCGridTests(unittest.TestCase):
    def test_step_loss_table_no_convexity_assumption(self):
        # Nonincreasing but nonconvex/step-like losses over discrete actions.
        actions=[0.,1.,2.,3.]
        losses=np.array([[
            [1.,1.,.6,.2],
            [.8,.4,.4,0.]
        ]])
        engine=FiniteGridCRC(actions,losses,.5,n_calibration=2)
        baseline=engine.envelope_crc()
        self.assertTrue(engine.validity_report(baseline)["valid"])
        one=engine.recertify_equal_share(baseline)
        self.assertTrue(engine.validity_report(one)["valid"])
        sweep=engine.coordinate_sweep(baseline)
        self.assertTrue(engine.validity_report(sweep)["valid"])
        self.assertTrue(engine.is_coordinatewise_admissible(sweep))
        for D in engine.states:
            self.assertLessEqual(one[D],baseline[D])

    def test_multiple_risks_share_one_action_grid(self):
        actions=[0.,.5,1.]
        losses=np.array([
            [[1.,.5,0.],[.7,.4,0.]],
            [[.8,.6,0.],[1.,.4,0.]]
        ])
        engine=FiniteGridCRC(actions,losses,[.6,.5],n_calibration=2)
        baseline=engine.envelope_crc()
        updated=engine.recertify_equal_share(baseline)
        self.assertTrue(engine.validity_report(baseline)["valid"])
        self.assertTrue(engine.validity_report(updated)["valid"])


    def test_paper_nonconvex_grid_witness_equal_share_stalls_sweep_improves(self):
        # Bai & Ramdas Appendix B.5: grid Lambda={.1,.25,.4,.5,1}.
        # The envelope controller is a fixed point of equal-share updates,
        # while one coordinate sweep reaches a strictly less conservative
        # valid controller on the same finite grid.
        actions=[.1,.25,.4,.5,1.]
        losses=np.array([[
            [1.-a for a in actions],
            [.5*(1.-a) for a in actions]
        ]])
        engine=FiniteGridCRC(actions,losses,.5,n_calibration=2)
        baseline=engine.envelope_crc()
        def values(controller):
            return tuple(float(engine.actions[controller[D]])
                         for D in [(2,0),(1,1),(0,2)])
        self.assertEqual(values(baseline),(.5,.4,.25))
        share=engine.recertify_equal_share(baseline)
        self.assertEqual(values(share),(.5,.4,.25))
        sweep=engine.coordinate_sweep(baseline)
        self.assertEqual(values(sweep),(.5,.4,.1))
        self.assertTrue(engine.validity_report(sweep)["valid"])
        self.assertTrue(engine.is_coordinatewise_admissible(sweep))

    def test_rejects_nonmonotone_loss_table(self):
        actions=[0.,1.,2.]
        losses=np.array([[[1.,.5,.6]]])
        with self.assertRaises(ValueError):
            FiniteGridCRC(actions,losses,.5,2)

    def test_rejects_unsorted_action_grid(self):
        with self.assertRaises(ValueError):
            FiniteGridCRC([0.,2.,1.],np.ones((1,1,3)),.5,2)

    def test_every_coordinate_sweep_prefix_is_valid(self):
        actions=[0.,1.,2.,3.]
        losses=np.array([[[1.,.7,.4,0.],[.8,.6,.3,0.]]])
        engine=FiniteGridCRC(actions,losses,.5,3)
        start=engine.envelope_crc()
        result=engine.coordinate_sweep(start)
        self.assertTrue(engine.validity_report(result)["valid"])


if __name__=="__main__":
    unittest.main(verbosity=2)
