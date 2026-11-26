import unittest

from benchmark_countable_prior_penalties import (
    benchmark_rows,
    geometric_product_normalizer,
    polynomial_product_normalizer,
    prior_penalty_bits,
)


class CountablePriorPenaltyBenchmarkTests(unittest.TestCase):
    def test_product_prior_normalizers_are_finite_and_positive(self):
        self.assertGreater(polynomial_product_normalizer(), 0.0)
        self.assertGreater(geometric_product_normalizer(), 0.0)
        self.assertLess(geometric_product_normalizer(16), geometric_product_normalizer(1024))

    def test_gamma_gap_prior_rewards_high_rank_contiguous_clusters(self):
        contiguous = prior_penalty_bits((1000, 1001, 1002, 1003, 1004))
        sparse = prior_penalty_bits((1, 10, 100, 1000))
        self.assertLess(
            contiguous["code_bits"],
            contiguous["polynomial_product_bits"],
        )
        self.assertGreater(
            sparse["code_bits"],
            sparse["polynomial_product_bits"],
        )

    def test_benchmark_rows_are_deterministic_and_well_formed(self):
        first = benchmark_rows()
        second = benchmark_rows()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        for row in first:
            self.assertGreater(row["code_bits"], 0)
            self.assertGreater(row["polynomial_product_bits"], 0.0)
            self.assertGreater(row["geometric_product_bits"], 0.0)

    def test_bad_intersection_inputs_fail_closed(self):
        for bad in ((), (0,), (-1,), (True,), (1, 1), (1.0,)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    prior_penalty_bits(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
