#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from adapt_bakeoff_to_robustness import adapt_documents
from decision_robustness_eval import score_document, validate_document
from multifidelity_compositional_bakeoff import (
    BENCHMARK_ID,
    CompositionalScenarioWorld,
    _digest_candidate_pool,
)
from multifidelity_scenario_suite import normalize_fidelity_costs
from prepare_synthetic_robustness_manifests import (
    GENERATOR_BENCHMARK_ID,
    REFERENCE_POINT,
    SHIFT_FAMILIES,
    build_manifests,
)


def canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def build_sweep() -> tuple[dict, bytes]:
    selector = {
        "objective_family": "local_bias_island",
        "constraint_family": "constraint_boundary",
        "dimension": 2,
        "severity": 0.5,
        "seed": 17,
        "policy": "low_screen",
    }
    costs = normalize_fidelity_costs({"low": 1, "medium": 5, "high": 25})
    world = CompositionalScenarioWorld(
        selector["objective_family"], selector["constraint_family"],
        selector["dimension"], selector["severity"], selector["seed"],
        costs, 75,
    )
    # A tiny unaltered-format sweep containing one selected run. This is a
    # schema test, not a measured optimizer outcome.
    run = {
        **selector,
        "design_count": 75,
        "candidate_pool_sha256": _digest_candidate_pool(world),
        "fidelity_costs": costs,
        "cost_budget": 200,
        "cost_spent": 26,
        "evaluations": [
            {"candidate_id": 0, "fidelity": "low", "objectives": [0.2, 0.4], "constraint_margin": 0.1, "cost_units": 1},
            {"candidate_id": 0, "fidelity": "high", "objectives": [0.2, 0.4], "constraint_margin": 0.1, "cost_units": 25},
        ],
    }
    sweep = {
        "benchmark_id": BENCHMARK_ID,
        "synthetic_fixture_only": True,
        "objective_families": [selector["objective_family"]],
        "constraint_families": [selector["constraint_family"]],
        "dimensions": [2],
        "severities": [0.5],
        "seeds": [17],
        "policies": ["low_screen"],
        "design_count": 75,
        "cost_budget": 200,
        "fidelity_costs": costs,
        "source_sha256": {
            "tools/casting-benchmark/multifidelity_compositional_bakeoff.py": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "tools/casting-benchmark/multifidelity_compositional_bakeoff.py").read_bytes()
            ).hexdigest(),
            "tools/casting-benchmark/multifidelity_scenario_suite.py": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "tools/casting-benchmark/multifidelity_scenario_suite.py").read_bytes()
            ).hexdigest(),
        },
        "run_count": 1,
        "paired_runs": [run],
        "warning": "synthetic known-answer fixture only",
    }
    return sweep, canonical(sweep)


def make_artifacts():
    sweep, sweep_raw = build_sweep()
    source = Path(__file__).resolve().parent / "prepare_synthetic_robustness_manifests.py"
    selection, holdout = build_manifests(
        sweep,
        sweep_raw,
        {
            "objective_family": "local_bias_island",
            "constraint_family": "constraint_boundary",
            "dimension": 2,
            "severity": 0.5,
            "seed": 17,
            "policy": "low_screen",
        },
        source.read_bytes(),
    )
    selection_raw = (json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    holdout_raw = canonical(holdout)
    return sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw


class SyntheticShiftManifestTests(unittest.TestCase):
    def test_generates_complete_full_pool_selection_and_shift_manifests(self) -> None:
        _, _, selection, _, holdout, _ = make_artifacts()
        self.assertEqual(selection["selected_candidate_ids"], [0])
        self.assertEqual(len(selection["candidate_predictions"]), 75)
        self.assertEqual(len(holdout["shifts"]), len(SHIFT_FAMILIES))
        self.assertEqual([row["shift_id"] for row in holdout["shifts"]], list(SHIFT_FAMILIES))
        for shift in holdout["shifts"]:
            self.assertEqual(len(shift["evaluations"]), 75)
            self.assertEqual({row["candidate_id"] for row in shift["evaluations"]}, set(range(75)))

    def test_generator_labels_shifts_as_synthetic_not_independent_physical_evidence(self) -> None:
        _, _, _, _, holdout, _ = make_artifacts()
        self.assertEqual(holdout["holdout_role"], "synthetic_post_selection_diagnostic")
        self.assertEqual(holdout["evidence_class"], "synthetic")
        self.assertEqual(holdout["shift_suite_id"], GENERATOR_BENCHMARK_ID)

    def test_deterministic_bytes_for_same_inputs(self) -> None:
        first = make_artifacts()
        second = make_artifacts()
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[5], second[5])

    def test_manifest_generator_digest_is_bound(self) -> None:
        _, _, selection, _, holdout, _ = make_artifacts()
        expected = hashlib.sha256(
            (Path(__file__).resolve().parent / "prepare_synthetic_robustness_manifests.py").read_bytes()
        ).hexdigest()
        self.assertEqual(selection["manifest_generator_sha256"], expected)
        self.assertEqual(holdout["manifest_generator_sha256"], expected)

    def test_full_pipeline_adapts_and_scores_known_answer(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = make_artifacts()
        adapted = adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw)
        validate_document(adapted)
        receipt = score_document(adapted, canonical(adapted))
        self.assertEqual(receipt["source_run"]["evidence_class"], "synthetic")
        self.assertEqual(receipt["split_declarations"]["holdout_role"], "synthetic_post_selection_diagnostic")
        self.assertEqual(receipt["metrics"]["candidate_count"], 75)
        self.assertEqual(receipt["metrics"]["holdout_shift_count"], len(SHIFT_FAMILIES))

    def test_zero_shift_control_matches_normalized_high_truth_for_all_candidates(self) -> None:
        sweep, sweep_raw, selection, _, holdout, _ = make_artifacts()
        family = next(row for row in holdout["shifts"] if row["shift_id"] == "zero_shift_control")
        prediction = selection["candidate_predictions"][0]
        truth = family["evaluations"][0]
        self.assertTrue(all(0.0 <= value < REFERENCE_POINT[i] for i, value in enumerate(truth["objectives"])))
        self.assertNotEqual(prediction["objectives"], truth["objectives"])

    def test_shift_families_produce_distinct_outcomes(self) -> None:
        _, _, _, _, holdout, _ = make_artifacts()
        by_family = {row["shift_id"]: row for row in holdout["shifts"]}
        base = by_family["zero_shift_control"]["evaluations"]
        self.assertTrue(any(
            by_family["localized_defect"]["evaluations"][i]["objectives"] != base[i]["objectives"]
            for i in range(len(base))
        ))
        self.assertTrue(any(
            by_family["structural_discontinuity"]["evaluations"][i]["constraint_margin"] != base[i]["constraint_margin"]
            for i in range(len(base))
        ))

    def test_rejects_pool_digest_mismatch(self) -> None:
        sweep, sweep_raw = build_sweep()
        sweep["paired_runs"][0]["candidate_pool_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate pool digest does not match"):
            build_manifests(
                sweep, canonical(sweep),
                {
                    "objective_family": "local_bias_island",
                    "constraint_family": "constraint_boundary",
                    "dimension": 2,
                    "severity": 0.5,
                    "seed": 17,
                    "policy": "low_screen",
                },
                Path(__file__).resolve().parent.joinpath("prepare_synthetic_robustness_manifests.py").read_bytes(),
            )

    def test_rejects_run_without_high_fidelity_selection(self) -> None:
        sweep, _ = build_sweep()
        sweep["paired_runs"][0]["evaluations"] = [
            {"candidate_id": 0, "fidelity": "low", "objectives": [0.2, 0.4], "constraint_margin": 0.1, "cost_units": 1}
        ]
        sweep["paired_runs"][0]["cost_spent"] = 1
        raw = canonical(sweep)
        with self.assertRaisesRegex(ValueError, "no high-fidelity candidate decisions"):
            build_manifests(
                sweep, raw,
                {
                    "objective_family": "local_bias_island",
                    "constraint_family": "constraint_boundary",
                    "dimension": 2,
                    "severity": 0.5,
                    "seed": 17,
                    "policy": "low_screen",
                },
                Path(__file__).resolve().parent.joinpath("prepare_synthetic_robustness_manifests.py").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
