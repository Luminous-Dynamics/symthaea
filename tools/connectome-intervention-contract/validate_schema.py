#!/usr/bin/env python3
"""CI-only Draft 2020-12 schema conformance checks for connectome manifests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCHEMA = ROOT / "docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_MANIFEST.schema.json"
BENCHMARK_EXAMPLE = ROOT / "docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_EXAMPLE_V1.json"
CPG_SCHEMA = ROOT / "docs/neuroscience/CPG_NUMERICAL_RUN_RECEIPT_V1.schema.json"
CPG_EXAMPLE = ROOT / "docs/neuroscience/CPG_NUMERICAL_RUN_RECEIPT_EXAMPLE_V1.json"


def rejected(validator: Draft202012Validator, instance: dict, label: str) -> None:
    errors = list(validator.iter_errors(instance))
    if not errors:
        raise AssertionError(f"negative schema control unexpectedly passed: {label}")


def cpg_complete_synthetic_fixture() -> dict:
    h = lambda char: char * 64
    return {
        "schema_version": "CPG-NUMERICAL-RUN-RECEIPT-V1",
        "run_id": "synthetic-cpg-schema-complete-001",
        "run_state": "complete",
        "evidence_tier": "synthetic_known_answer",
        "claim_ceiling": "synthetic_only",
        "created_at_utc": "2026-11-02T00:00:00Z",
        "source": {
            "source_kind": "synthetic",
            "repository": None,
            "immutable_code_ref": None,
            "code_commit_sha": "a" * 40,
            "source_file_path": "synthetic://rate-equation",
            "source_file_sha256": h("b"),
            "data_revision": None,
            "data_manifest_sha256": None,
            "rights_state": "synthetic",
            "permission_receipt_sha256": None,
            "provenance_notes": "synthetic schema fixture",
        },
        "simulator": {
            "model_family": "rate_network",
            "equation_manifest_sha256": h("c"),
            "parameter_manifest_sha256": h("d"),
            "state_semantics": "Three-state synthetic bounded rate system.",
            "observation_model_id": "identity",
        },
        "solver": {
            "solver_id": "Dopri5",
            "library": "synthetic-reference-solver",
            "library_version": "1.0",
            "dependency_lock_sha256": h("e"),
            "backend": "cpu",
            "dtype": "float64",
            "rtol": 1e-8,
            "atol": 1e-10,
            "initial_step_seconds": 0.001,
            "max_steps": 10000,
            "throw_on_failure": True,
            "nonfinite_event_detection_enabled": True,
        },
        "simulation": {
            "start_time_seconds": 0.0,
            "end_time_seconds": 0.01,
            "nominal_save_interval_seconds": 0.01,
            "state_dimension": 3,
            "expected_saved_timepoints": 2,
            "actual_saved_timepoints": 2,
            "solver_status": "success",
            "attempted_steps": 1,
            "accepted_steps": 1,
            "rejected_steps": 0,
            "function_evaluations": 7,
            "random_seed": 7,
            "prng_id": "synthetic-counter-v1",
            "deterministic_mode": True,
        },
        "perturbation": {
            "protocol_sha256": h("f"),
            "waveform_sha256": h("1"),
            "stimulus_applied": True,
            "input_noise": {
                "requested_mean": 0.0,
                "requested_stddev": 0.0,
                "applied": False,
                "realized_mean": 0.0,
                "realized_stddev": 0.0,
                "sample_sha256": None,
            },
            "weight_noise": {
                "requested_mean": 0.0,
                "requested_stddev": 0.0,
                "applied": False,
                "realized_mean": 0.0,
                "realized_stddev": 0.0,
                "sample_sha256": None,
            },
        },
        "numerics": {
            "raw_state_sha256": h("2"),
            "raw_output_manifest_sha256": h("3"),
            "raw_state_finite_scan_performed": True,
            "non_finite_value_count": 0,
            "post_solver_sanitized_value_count": 0,
            "post_solver_clipped_value_count": 0,
            "post_solver_transformed_value_count": 0,
            "minimum_raw_value": 0.1,
            "maximum_raw_value": 0.6,
            "maximum_absolute_raw_value": 0.6,
            "raw_state_artifact": {
                "relative_path": "raw/state.f64le",
                "encoding": "f64le",
                "byte_length": 48,
                "sha256": h("2"),
                "capture_stage": "direct_solver_output_before_postprocessing",
                "layout": "state_major_time_minor",
            },
        },
        "resources": {
            "wall_time_seconds": 0.01,
            "peak_memory_bytes": 4096,
            "energy_joules": None,
        },
        "artifacts": {
            "resolved_config_sha256": h("4"),
            "environment_sha256": h("5"),
            "logs_sha256": h("6"),
        },
        "limitations": ["synthetic schema fixture only; no biological claim"],
    }


def main() -> None:
    benchmark_schema = json.loads(BENCHMARK_SCHEMA.read_text(encoding="utf-8"))
    benchmark_example = json.loads(BENCHMARK_EXAMPLE.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(benchmark_schema)
    benchmark_validator = Draft202012Validator(benchmark_schema)
    benchmark_validator.validate(benchmark_example)
    assert benchmark_example["run_state"] == "incomplete"
    assert benchmark_example["results"]["status"] == "not_run"

    wrong_claim = copy.deepcopy(benchmark_example)
    wrong_claim["claim_ceiling"] = "observational_prediction_only"
    rejected(benchmark_validator, wrong_claim, "benchmark evidence-tier / claim-ceiling mismatch")

    unknown_property = copy.deepcopy(benchmark_example)
    unknown_property["unreviewed_assertion"] = "pass"
    rejected(benchmark_validator, unknown_property, "benchmark unknown top-level property")

    forged_complete = copy.deepcopy(benchmark_example)
    forged_complete["run_state"] = "complete"
    forged_complete["results"]["status"] = "complete"
    rejected(benchmark_validator, forged_complete, "benchmark forged completion with null evidence")

    measured_without_permission = copy.deepcopy(benchmark_example)
    measured_without_permission["evidence_tier"] = "measured_intervention"
    measured_without_permission["claim_ceiling"] = "measured_intervention_prediction"
    measured_without_permission["source"]["source_kind"] = "measured_intervention"
    measured_without_permission["source"]["rights_state"] = "unknown"
    rejected(benchmark_validator, measured_without_permission, "measured intervention without permission")

    cpg_schema = json.loads(CPG_SCHEMA.read_text(encoding="utf-8"))
    cpg_example = json.loads(CPG_EXAMPLE.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(cpg_schema)
    cpg_validator = Draft202012Validator(cpg_schema)
    cpg_validator.validate(cpg_example)
    assert cpg_example["run_state"] == "incomplete"

    # Positive complete-shaped fixture exercises the schema's complete branch.
    cpg_complete = cpg_complete_synthetic_fixture()
    cpg_validator.validate(cpg_complete)

    cpg_forged = copy.deepcopy(cpg_complete)
    cpg_forged["numerics"]["non_finite_value_count"] = 1
    rejected(cpg_validator, cpg_forged, "CPG complete receipt with non-finite values")

    cpg_sanitized = copy.deepcopy(cpg_complete)
    cpg_sanitized["numerics"]["post_solver_sanitized_value_count"] = 1
    rejected(cpg_validator, cpg_sanitized, "CPG complete receipt with post-solver sanitization")

    cpg_clipped = copy.deepcopy(cpg_complete)
    cpg_clipped["numerics"]["post_solver_clipped_value_count"] = 1
    rejected(cpg_validator, cpg_clipped, "CPG complete receipt with post-solver clipping")

    cpg_failed = copy.deepcopy(cpg_complete)
    cpg_failed["simulation"]["solver_status"] = "max_steps_reached"
    rejected(cpg_validator, cpg_failed, "CPG complete receipt with failed solver")

    cpg_noise_noop = copy.deepcopy(cpg_example)
    cpg_noise_noop["perturbation"]["input_noise"]["requested_stddev"] = 0.1
    rejected(cpg_validator, cpg_noise_noop, "CPG requested non-zero noise marked unapplied")

    print(
        "OK: both Draft 2020-12 schemas are valid; both incomplete examples and a "
        "synthetic complete CPG fixture validate; nine adversarial mutations reject."
    )


if __name__ == "__main__":
    main()
