import unittest
import numpy as np

from confidence_adaptive_crc import (
    ConfidenceAdaptiveCRC, build_feature_adaptive
)


class ConfidenceAdaptiveCRCTests(unittest.TestCase):
    def test_state_count_matches_paper_formula(self):
        cal=[(.1,.3,1),(.2,.5,2),(.1,.3,1)]
        e=ConfidenceAdaptiveCRC(cal,(.2,.4),.2,cap_breakpoints=False,group_duplicates=False)
        r=e.num_active_points
        self.assertEqual(e.state_count,r*(2**(r-1)))

    def test_grouped_and_ungrouped_baseline_states_match_exhaustively(self):
        cal=[(.1,.4,1),(.1,.4,2),(.2,.5,1),(.1,.4,2)]
        query=(.1,.4)
        grouped=ConfidenceAdaptiveCRC(cal,query,.25,cap_breakpoints=False,group_duplicates=True)
        ungrouped=ConfidenceAdaptiveCRC(cal,query,.25,cap_breakpoints=False,group_duplicates=False)
        self.assertLess(grouped.state_count,ungrouped.state_count)
        grouped_table=dict(grouped.baseline_table)
        ungrouped_table=dict(ungrouped.baseline_table)
        for round_index in range(3):
            for j,counts in grouped.states:
                check=grouped.expand_group_state_against_ungrouped(
                    ungrouped,j,counts,grouped_table=grouped_table,
                    ungrouped_table=ungrouped_table)
                self.assertTrue(check["match"],check)
            if round_index<2:
                grouped_table=grouped.recertify_round(grouped_table)
                ungrouped_table=ungrouped.recertify_round(ungrouped_table)

    def test_breakpoint_capping_preserves_all_ungrouped_baseline_cells(self):
        cal=[
            (.01,.10,1),(.02,.20,2),(.03,.30,1),(.04,.40,2),
            (.05,.50,1),(.06,.60,2),(.07,.70,1)
        ]
        query=(.08,.80)
        uncapped=ConfidenceAdaptiveCRC(cal,query,.25,cap_breakpoints=False,group_duplicates=False)
        capped=ConfidenceAdaptiveCRC(cal,query,.25,cap_breakpoints=True,group_duplicates=False)
        self.assertIsNotNone(capped.cap_threshold)
        self.assertEqual(uncapped.states,capped.states)
        uncapped_table=dict(uncapped.baseline_table)
        capped_table=dict(capped.baseline_table)
        for round_index in range(4):
            for state in uncapped.states:
                self.assertAlmostEqual(uncapped_table[state],capped_table[state],places=12)
            if round_index<3:
                uncapped_table=uncapped.recertify_round(uncapped_table)
                capped_table=capped.recertify_round(capped_table)

    def test_every_round_keeps_all_hypothetical_type_constraints_valid(self):
        cal=[(.02,.3,1),(.08,.42,2),(.02,.3,1),(.11,.55,1),(.0,.0,0),(.08,.42,2)]
        engine=ConfidenceAdaptiveCRC(cal,(.02,.3),.25)
        current=dict(engine.baseline_table)
        for _ in range(8):
            updated=engine.recertify_round(current)
            self.assertTrue(engine.validity_report(updated)["valid"])
            for state in engine.states:
                self.assertGreaterEqual(updated[state]+1e-10,current[state])
            current=updated

    def test_query_depth_uses_actual_calibration_types(self):
        cal=[(.1,.4,1),(.2,.5,2),(.1,.4,0),(.3,.7,1)]
        a=ConfidenceAdaptiveCRC(cal,(.1,.4),.25,cap_breakpoints=False)
        depth=a.query_depth(a.baseline_table)
        self.assertIn(depth,(0.,1.,2.))
        self.assertEqual(a.group_sizes[ a.query_group ],2)

    def test_grouped_state_count_closed_form(self):
        cal=[(.1,.4,1),(.1,.4,2),(.2,.5,1),(.2,.5,2)]
        e=ConfidenceAdaptiveCRC(cal,(.1,.4),.2,group_duplicates=True)
        expected=sum(e.group_sizes[j]*np.prod([e.group_sizes[h]+1 for h in range(e.num_groups) if h!=j])
                     for j in range(e.num_groups))
        self.assertEqual(e.state_count,int(expected))


    def test_randomized_grouping_and_capping_parity(self):
        rng=np.random.default_rng(20261031)
        pairs=[(.02,.18),(.08,.40),(.14,.62),(.22,.85)]
        for case in range(12):
            n=int(rng.integers(4,8))
            calibration=[]
            for _ in range(n):
                u,v=pairs[int(rng.integers(0,len(pairs)))]
                error_type=int(rng.choice([1,2]))
                calibration.append((u,v,error_type))
            query=pairs[int(rng.integers(0,len(pairs)))]
            alpha=float(rng.choice([.2,.25,.3,.4]))
            grouped=ConfidenceAdaptiveCRC(calibration,query,alpha,
                cap_breakpoints=True,group_duplicates=True)
            ungrouped=ConfidenceAdaptiveCRC(calibration,query,alpha,
                cap_breakpoints=True,group_duplicates=False)
            grouped_table=dict(grouped.baseline_table)
            ungrouped_table=dict(ungrouped.baseline_table)
            for round_index in range(3):
                for j,counts in grouped.states:
                    check=grouped.expand_group_state_against_ungrouped(
                        ungrouped,j,counts,grouped_table=grouped_table,
                        ungrouped_table=ungrouped_table)
                    self.assertTrue(check["match"],(case,round_index,j,counts,check))
                self.assertTrue(grouped.validity_report(grouped_table)["valid"])
                self.assertTrue(ungrouped.validity_report(ungrouped_table)["valid"])
                if round_index<2:
                    grouped_table=grouped.recertify_round(grouped_table)
                    ungrouped_table=ungrouped.recertify_round(ungrouped_table)

            capped=ConfidenceAdaptiveCRC(calibration,query,alpha,
                cap_breakpoints=True,group_duplicates=False)
            uncapped=ConfidenceAdaptiveCRC(calibration,query,alpha,
                cap_breakpoints=False,group_duplicates=False)
            capped_table=dict(capped.baseline_table)
            uncapped_table=dict(uncapped.baseline_table)
            for round_index in range(3):
                for state in capped.states:
                    self.assertAlmostEqual(capped_table[state],uncapped_table[state],
                        places=10,msg=f"case={case}, round={round_index}, state={state}")
                if round_index<2:
                    capped_table=capped.recertify_round(capped_table)
                    uncapped_table=uncapped.recertify_round(uncapped_table)

    def test_invalid_breakpoint_order_and_type_rejected(self):
        with self.assertRaises(ValueError):
            ConfidenceAdaptiveCRC([(.5,.2,1)],(.1,.3),.2)
        with self.assertRaises(ValueError):
            ConfidenceAdaptiveCRC([(.1,.2,3)],(.1,.3),.2)

    def test_no_capping_when_too_few_active_points_or_negative_budget(self):
        a=ConfidenceAdaptiveCRC([(.1,.2,1)],(.1,.3),.05,cap_breakpoints=True)
        self.assertIsNone(a.cap_threshold)
        b=ConfidenceAdaptiveCRC([(.1,.2,1),(.2,.3,2)],(.1,.3),.001,cap_breakpoints=True)
        self.assertIsNone(b.cap_threshold)


if __name__=="__main__":
    unittest.main(verbosity=2)
