import importlib.util
import unittest
from fractions import Fraction

from benchmark_affine_projective_closure_solvers import (
    _scipy_milp_inputs,
    exhaustive_optimum,
    make_instance,
    run_benchmark,
)
from affine_projective_closure_milp import build_affine_closure_milp


class AffineProjectiveClosureSolverBenchmarkTests(unittest.TestCase):
    def test_instance_generation_is_reproducible_and_has_known_closure_margin(self):
        first = make_instance(12, 20261113)
        second = make_instance(12, 20261113)
        self.assertEqual(first["instance_sha256"], second["instance_sha256"])
        self.assertEqual(first["utility"], second["utility"])
        self.assertEqual(first["costs"], second["costs"])
        for i in first["ids"]:
            weight = Fraction(1, 1 << i)
            self.assertEqual(
                weight * (first["evidence"][i] - 1),
                Fraction(10, 12),
            )

    def test_scipy_matrix_size_is_linear(self):
        self.skipTest("requires SciPy; solver-independent model-size test is below") if not importlib.util.find_spec("scipy") else None
        instance = make_instance(20, 20261121)
        model = build_affine_closure_milp(
            instance["ids"], instance["evidence"], instance["alpha"],
            scheme=instance["scheme"], values=instance["utility"],
            costs=instance["costs"], capacity=instance["capacity"],
        )
        inputs = _scipy_milp_inputs(model)
        self.assertEqual(inputs["matrix_shape"], (20 + 2 + 1, 41))
        self.assertLessEqual(inputs["nonzeros"], 5 * 20 + 5)
        self.assertEqual(inputs["integrality"].sum(), 20)

    def test_exact_exhaustive_action_oracle_finds_feasible_optimum(self):
        instance = make_instance(8, 20261109)
        model = build_affine_closure_milp(
            instance["ids"], instance["evidence"], instance["alpha"],
            scheme=instance["scheme"], values=instance["utility"],
            costs=instance["costs"], capacity=instance["capacity"],
        )
        optimum = exhaustive_optimum(model)
        self.assertTrue(optimum["feasible"])
        self.assertEqual(optimum["checked_actions"], 256)
        self.assertGreaterEqual(len(optimum["rejection_ids"]), 7)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is optional")
    def test_highs_incumbent_is_exactly_verified_and_matches_small_oracle(self):
        result = run_benchmark(
            (8,), seed=20261101, time_limit=10.0,
            relative_gap=0.0, include_exhaustive_up_to=8,
        )
        row = result["results"][0]
        self.assertTrue(row["accepted"], row)
        self.assertTrue(row["optimality_proven"], row)
        self.assertTrue(row["exact_verification"]["valid"], row)
        self.assertTrue(row["exhaustive_objective_match"], row)

    def test_benchmark_manifest_labels_itself_as_solver_only(self):
        if not importlib.util.find_spec("scipy"):
            self.skipTest("SciPy is optional")
        result = run_benchmark((5,), seed=42, time_limit=5.0)
        self.assertIn("not a statistical FDR validity or power benchmark", result["purpose"])
        self.assertEqual(result["configuration"]["base_seed"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
