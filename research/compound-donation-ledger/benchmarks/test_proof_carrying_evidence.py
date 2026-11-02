import copy
import unittest
from fractions import Fraction as F

from proof_carrying_evidence import (
    EvidenceWriter,
    SCALE,
    adjusted_integer,
    build_spec,
    digest,
    generate_observations,
    independent_verify,
    negative_controls,
    validate_spec,
    writer_factor,
)


class ProofCarryingEvidenceTests(unittest.TestCase):
    def test_bernoulli_lr_factors_are_exact(self):
        spec = build_spec(n=3, rounds=3)
        c = spec.candidates[0]
        self.assertEqual(writer_factor(c, 1), F(7, 5))
        self.assertEqual(writer_factor(c, 0), F(3, 5))
        self.assertEqual(F(1, 2) * writer_factor(c, 1) +
                         F(1, 2) * writer_factor(c, 0), F(1))

    def test_adjuster_floor_is_exact_and_monotone(self):
        values = [F(1), F(9, 4), F(4), F(16), F(25)]
        adjusted = [adjusted_integer(x, SCALE) for x in values]
        self.assertEqual(adjusted[0], 0)
        self.assertEqual(adjusted, sorted(adjusted))
        # Verify floor_S(sqrt(x)-1) with exact squared inequalities.
        for x, value in zip(values, adjusted):
            lower = F(value, SCALE) + 1
            upper = F(value + 1, SCALE) + 1
            self.assertLessEqual(lower * lower, x)
            self.assertGreater(upper * upper, x)

    def test_complete_round_batches_replay(self):
        spec = build_spec(n=9, rounds=12)
        writer = EvidenceWriter(spec)
        receipts = []
        observations = generate_observations(spec, 12, "shared_uniform", 10146)
        for round_no, batch in enumerate(observations):
            receipts.append(writer.append_round(round_no, batch))
        verified = independent_verify(spec, receipts)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["rounds"], 12)
        self.assertEqual(verified["factor_updates"], sum(
            sum(c.arrival_round <= t for c in spec.candidates)
            for t in range(12)
        ))

    def test_incomplete_batch_and_round_gap_fail_closed(self):
        spec = build_spec(n=6, rounds=5)
        writer = EvidenceWriter(spec)
        active = {
            c.candidate_id: 0
            for c in spec.candidates if c.arrival_round == 0
        }
        with self.assertRaises(ValueError):
            writer.append_round(0, {next(iter(active)): 0})
        with self.assertRaises(ValueError):
            writer.append_round(1, active)

    def test_independent_verifier_rejects_all_tamper_controls(self):
        spec = build_spec(n=9, rounds=10)
        writer = EvidenceWriter(spec)
        receipts = []
        for t, batch in enumerate(generate_observations(spec, 10, "independent", 10147)):
            receipts.append(writer.append_round(t, batch))
        controls = negative_controls(spec, receipts)
        self.assertEqual(len(controls), 7)
        self.assertTrue(all(controls.values()))

    def test_process_precommitment_change_is_rejected(self):
        spec = build_spec(n=3, rounds=2)
        mutated = copy.deepcopy(spec)
        object.__setattr__(mutated.candidates[0], "p1_num", 3)
        with self.assertRaises(ValueError):
            validate_spec(mutated)

    def test_every_receipt_claim_is_recomputed_from_observations(self):
        spec = build_spec(n=6, rounds=8)
        writer = EvidenceWriter(spec)
        receipts = [
            writer.append_round(t, batch)
            for t, batch in enumerate(generate_observations(spec, 8, "balanced_negative", 10148))
        ]
        # The independent verifier must reject a false exact LR numerator.
        altered = copy.deepcopy(receipts)
        altered[2]["claims"][0]["factor_num"] += 1
        with self.assertRaises(ValueError):
            independent_verify(spec, altered)


if __name__ == "__main__":
    unittest.main()
