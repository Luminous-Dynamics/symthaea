import random
import unittest
from fractions import Fraction

from donation_e_lond_reference import (
    DonationELONDIndex,
    donation_threshold_reference,
    donation_wealth_reference,
    run_donation_e_lond,
)


class DonationELONDReferenceTests(unittest.TestCase):
    def test_index_matches_scan_on_exact_tie_at_threshold(self):
        index = DonationELONDIndex(Fraction(1))
        first = index.threshold(Fraction(1, 2))
        self.assertEqual(first["e_value_threshold"], Fraction(2))
        index.add(Fraction(2), Fraction(1, 2), True)

        second = index.threshold(Fraction(1, 4))
        self.assertEqual(second["rejection_count"], 1)
        self.assertEqual(second["threshold_key"], Fraction(1, 2))
        self.assertEqual(second["prefix_count"], 1)  # equality is included
        self.assertEqual(second["prefix_gamma"], Fraction(1, 2))
        self.assertEqual(second["prefix_gamma_e"], Fraction(1))
        self.assertEqual(second["wealth"], Fraction(1, 2))
        self.assertEqual(second["denominator"], Fraction(0))
        self.assertEqual(second["e_value_threshold"], Fraction(0))
        self.assertTrue(index.verify())
        index.assert_valid_avl()

    def test_duplicate_keys_aggregate_in_one_avl_bucket(self):
        index = DonationELONDIndex(Fraction(1))
        # gamma=.25 and E=4 give identical x=gamma*(E-1)=.75.
        self.assertTrue(index.step(Fraction(4), Fraction(1, 4))["rejected"])
        self.assertTrue(index.step(Fraction(4), Fraction(1, 4))["rejected"])
        self.assertEqual(index.rejection_count, 2)
        self.assertEqual(index.root.bucket_count, 2)
        self.assertEqual(index.root.subtree_gamma, Fraction(1, 2))
        self.assertEqual(index.root.subtree_gamma_e, Fraction(2))
        self.assertTrue(index.verify())
        index.assert_valid_avl()

    def test_zero_weight_never_rejects(self):
        result = run_donation_e_lond(
            [Fraction(100), Fraction(0)],
            [Fraction(1), Fraction(0)],
            Fraction(1, 2),
        )
        self.assertEqual(result["donation_rejections"], (1,))
        self.assertEqual(result["donation"][1]["e_value_threshold"], None)

    def test_fixed_stream_matches_reference_threshold_each_round(self):
        delta = Fraction(1, 10)
        gamma = [Fraction(1, i * (i + 1)) for i in range(1, 101)]
        e_values = [
            Fraction((17 * i + 11) % 97, 13) for i in range(1, 101)
        ]
        index = DonationELONDIndex(delta)
        e_history = []
        gamma_history = []
        rejected = []
        for i, (e_value, weight) in enumerate(zip(e_values, gamma), start=1):
            reference = donation_threshold_reference(
                e_history, gamma_history, rejected, weight, delta
            )
            indexed = index.threshold(weight)
            for field in (
                "wealth",
                "rejection_count",
                "capped_wealth",
                "denominator",
                "test_level",
                "e_value_threshold",
                "baseline_e_lond_threshold",
            ):
                self.assertEqual(indexed[field], reference[field], (i, field))
            threshold = reference["e_value_threshold"]
            did_reject = threshold is not None and e_value >= threshold
            index.add(e_value, weight, did_reject)
            e_history.append(e_value)
            gamma_history.append(weight)
            if did_reject:
                rejected.append(i)
            self.assertTrue(index.verify(), i)
            index.assert_valid_avl()

    def test_randomized_exact_streams_match_scan(self):
        rng = random.Random(981301)
        for case in range(80):
            n = rng.randint(1, 80)
            delta = Fraction(rng.randint(1, 10), 10)
            # Telescoping gamma_i=1/(i(i+1)) has exactly summable mass one.
            gamma = [Fraction(1, i * (i + 1)) for i in range(1, n + 1)]
            e_values = [
                Fraction(rng.randint(0, 300), rng.randint(1, 30))
                for _ in range(n)
            ]
            result = run_donation_e_lond(e_values, gamma, delta)
            self.assertEqual(len(result["donation"]), n)
            self.assertEqual(len(result["e_lond"]), n)
            self.assertTrue(
                set(result["e_lond_rejections"]).issubset(
                    set(result["donation_rejections"])
                ),
                case,
            )

    def test_rejects_gamma_mass_above_one(self):
        index = DonationELONDIndex(Fraction(1, 2))
        index.add(Fraction(1), Fraction(3, 4), False)
        with self.assertRaisesRegex(ValueError, "sum to at most one"):
            index.add(Fraction(1), Fraction(1, 2), False)

    def test_scan_rejects_duplicate_or_out_of_range_rejection_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            donation_wealth_reference(
                [Fraction(2), Fraction(3)],
                [Fraction(1, 2), Fraction(1, 4)],
                [1, 1],
                Fraction(1, 2),
            )
        with self.assertRaisesRegex(ValueError, "1-based"):
            donation_wealth_reference(
                [Fraction(2)],
                [Fraction(1, 2)],
                [2],
                Fraction(1, 2),
            )

    def test_reference_rejects_invalid_delta_and_gamma_budget(self):
        with self.assertRaises(ValueError):
            donation_wealth_reference([], [], [], Fraction(0))
        with self.assertRaises(ValueError):
            donation_wealth_reference(
                [Fraction(2), Fraction(2)],
                [Fraction(3, 4), Fraction(3, 4)],
                [],
                Fraction(1, 2),
            )

    def test_rejects_zero_weight_rejection_and_non_boolean_flag(self):
        index = DonationELONDIndex(Fraction(1, 2))
        with self.assertRaisesRegex(ValueError, "zero-weight"):
            index.add(Fraction(10), Fraction(0), True)
        with self.assertRaises(TypeError):
            index.add(Fraction(1), Fraction(1, 2), 1)

    def test_rejects_invalid_evidence_or_gamma(self):
        with self.assertRaises(TypeError):
            run_donation_e_lond([0.1], [Fraction(1)], Fraction(1, 2))
        with self.assertRaises(ValueError):
            run_donation_e_lond([Fraction(-1)], [Fraction(1)], Fraction(1, 2))
        with self.assertRaises(ValueError):
            run_donation_e_lond(
                [Fraction(1), Fraction(1)],
                [Fraction(3, 4), Fraction(3, 4)],
                Fraction(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
