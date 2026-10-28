#!/usr/bin/env python3
import math, random, tempfile, unittest
from pathlib import Path
import benchmark as b

class RiskAwareTests(unittest.TestCase):
    def test_deterministic_generation(self):
        a=b.make_case(random.Random(11),17,"iid","test")
        c=b.make_case(random.Random(11),17,"iid","test")
        self.assertEqual(a,c)

    def test_policy_surface_has_only_features(self):
        c=b.make_case(random.Random(17),1,"iid","test")
        cfg={"similarity_threshold":.7,"evidence_threshold":.7,"risk_budget":.035,
             "direct_threshold":.6,"conflict_trigger":.7,"freshness_trigger":.3,"scope_trigger":.44,
             "calibrated_risk_thresholds_by_budget":{f"risk_control_{int(b*1000):02d}permille":{"answer":{"threshold":.02},"escalate":{"threshold":.02}} for b in b.RISK_BUDGETS}}
        models=b.train_models([b.make_case(random.Random(i),i,"iid","dev") for i in range(200)])
        for p in b.POLICIES:
            action=b.choose_policy_action(p,c["x"],cfg,models)
            self.assertIn(action,{"answer","escalate","abstain"})
        with self.assertRaises((TypeError, AttributeError, KeyError)):
            b.choose_policy_action("evidence_threshold",c,cfg)  # Case object is not a feature map.

    def test_policy_has_no_latent_key(self):
        c=b.make_case(random.Random(19),3,"scope_shift","test")
        x=dict(c["x"])
        self.assertNotIn("latent_class",x)
        self.assertNotIn("should_answer",x)
        self.assertEqual(b.evidence_score(x),b.evidence_score(c["x"]))

    def test_escalation_cost_charged_once(self):
        c=b.make_case(random.Random(2),5,"iid","test")
        action=c["deep_action"]
        out=c["outcomes"]["escalate"]
        gross=(b.ABSTAIN_GOOD if not c["should_answer"] else b.ABSTAIN_BAD) if action=="abstain" else (b.GOOD_UTILITY if c["should_answer"] and c["deep_correct"] else b.HARM_UTILITY)
        self.assertAlmostEqual(out["utility"],gross-b.TOKEN_COST["escalate"]*b.TOKEN_UTILITY_COST)

    def test_abstain_is_non_harmful(self):
        for i in range(80):
            c=b.make_case(random.Random(i),i,"iid","test")
            self.assertFalse(c["outcomes"]["abstain"]["harm"])

    def test_ridge_reproducible(self):
        dev=[b.make_case(random.Random(i),i,"iid","dev") for i in range(80)]
        m1=b.train_models(dev); m2=b.train_models(dev)
        self.assertEqual(m1,m2)

    def test_calibration_risk_upper_bound(self):
        rng=random.Random(712)
        calib=[b.make_case(rng,i,"iid","calibration") for i in range(1500)]
        train=[b.make_case(random.Random(9000+i),i,"iid","train") for i in range(800)]
        models=b.train_models(train)
        thresholds=b.calibrate_risk_thresholds(calib,models,risk_budget=.10,min_selected=50)
        for action in ("answer","escalate"):
            rec=thresholds[action]
            if rec["selected"]:
                self.assertLessEqual(rec["wilson_upper_95"],.10)
                self.assertLessEqual(rec["observed_rate"],.10)

    def test_group_bootstrap_paired_and_reproducible(self):
        a=[]; z=[]
        for g in range(5):
            for i in range(5):
                ident=f"{g}-{i}"
                a.append({"case_id":ident,"group_id":str(g),"net_utility":1.0})
                z.append({"case_id":ident,"group_id":str(g),"net_utility":0.5})
        self.assertEqual(b.grouped_bootstrap_delta(a,z,"net_utility",reps=40),b.grouped_bootstrap_delta(a,z,"net_utility",reps=40))
        self.assertAlmostEqual(b.grouped_bootstrap_delta(a,z,"net_utility",reps=40)[0],.5)

    def test_all_outcomes_bounded(self):
        for regime in b.REGIMES:
            for i in range(80):
                c=b.make_case(random.Random(i),i,regime,"test")
                self.assertTrue(set(c["outcomes"])=={"answer","escalate","abstain"})
                self.assertTrue(all(-3 <= o["utility"] <= 2 for o in c["outcomes"].values()))
                self.assertEqual(c["outcomes"]["escalate"]["harm"] in (True,False),True)

    def test_singular_pivot_fallback_is_finite(self):
        result=b.solve_linear([[0.0,0.0],[0.0,1.0]],[0.0,1.0])
        self.assertTrue(all(math.isfinite(v) for v in result))
        self.assertAlmostEqual(result[1],1.0,places=7)

if __name__=="__main__": unittest.main(verbosity=2)
