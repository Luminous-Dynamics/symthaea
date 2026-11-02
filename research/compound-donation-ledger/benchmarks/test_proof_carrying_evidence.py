import copy
import unittest
from fractions import Fraction as F

from proof_carrying_evidence import (
    EvidenceWriter,
    floor_a2,
    independent_verify,
    make_spec,
    negative_controls,
    sha256,
    generate_round,
    validate_spec,
)


class ProofCarryingEvidenceTests(unittest.TestCase):
    def test_registered_bernoulli_factors_and_mean_one(self):
        spec = make_spec(n=3)
        p0, p1 = F(*spec["candidates"][0]["p0"]), F(*spec["candidates"][0]["p1"])
        success = p1 / p0
        failure = (1-p1) / (1-p0)
        self.assertEqual(success, F(7, 5))
        self.assertEqual(failure, F(3, 5))
        self.assertEqual(p0*success + (1-p0)*failure, 1)

    def test_adjuster_floor_with_exact_square_bounds(self):
        scale = 1_000_000
        for maximum in (F(1), F(9, 4), F(4), F(16), F(25), F(7, 3)):
            adjusted = floor_a2(maximum, scale)
            lower = F(adjusted, scale) + 1
            upper = F(adjusted + 1, scale) + 1
            self.assertLessEqual(lower * lower, maximum)
            self.assertGreater(upper * upper, maximum)

    def test_writer_rounds_replay_independently_for_all_dependence_modes(self):
        for mode in ("independent", "shared_uniform", "balanced_negative"):
            with self.subTest(mode=mode):
                spec = make_spec(n=9)
                writer = EvidenceWriter(spec)
                receipts = []
                import random
                rng = random.Random(20261102)
                for t in range(20):
                    observations = generate_round(spec, t, mode, rng)
                    receipts.append(writer.append_round(t, observations))
                result = independent_verify(spec, receipts)
                self.assertTrue(result["verified"])
                expected = sum(
                    c["arrival_round"] <= t
                    for t in range(20)
                    for c in spec["candidates"]
                )
                self.assertEqual(result["evidence_updates"], expected)

    def test_writer_rejects_incomplete_and_out_of_order_batches(self):
        spec = make_spec(n=6)
        writer = EvidenceWriter(spec)
        active = [c["id"] for c in spec["candidates"] if c["arrival_round"] <= 0]
        with self.assertRaises(ValueError):
            writer.append_round(0, {i: 0 for i in active[:-1]})
        with self.assertRaises(ValueError):
            writer.append_round(1, {i: 0 for i in active})
        with self.assertRaises(ValueError):
            writer.append_round(0, {i: 0 for i in reversed(active)})

    def test_process_spec_tampering_is_rejected(self):
        spec = make_spec(n=4)
        changed = copy.deepcopy(spec)
        changed["candidates"][0]["p1"] = [3, 5]
        with self.assertRaises(ValueError):
            validate_spec(changed)

    def test_claim_tampering_is_rejected_by_independent_replay(self):
        spec = make_spec(n=6)
        writer = EvidenceWriter(spec)
        receipts = []
        import random
        rng = random.Random(4006)
        for t in range(10):
            receipts.append(writer.append_round(t, generate_round(spec, t, "independent", rng)))
        altered = copy.deepcopy(receipts)
        altered[0]["claims"][0]["factor"][0] += 1
        with self.assertRaises(ValueError):
            independent_verify(spec, altered)

    def test_seven_negative_controls_fail_closed(self):
        spec = make_spec(n=9)
        writer = EvidenceWriter(spec)
        receipts = []
        import random
        rng = random.Random(4007)
        for t in range(10):
            receipts.append(writer.append_round(t, generate_round(spec, t, "shared_uniform", rng)))
        controls = negative_controls(spec, receipts)
        self.assertEqual(len(controls), 7)
        self.assertTrue(all(controls.values()))

    def test_hash_chain_detects_receipt_mutation(self):
        spec = make_spec(n=6)
        writer = EvidenceWriter(spec)
        receipts = []
        import random
        rng = random.Random(4008)
        for t in range(5):
            receipts.append(writer.append_round(t, generate_round(spec, t, "balanced_negative", rng)))
        altered = copy.deepcopy(receipts)
        altered[2]["observations"][0]["value"] ^= 1
        with self.assertRaises(ValueError):
            independent_verify(spec, altered)


if __name__ == "__main__":
    unittest.main()
