#!/usr/bin/env python3
"""Compare two Spore semantic-time evidence manifests without aesthetic scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

INPUT_SCHEMA = "spore-temporal-evidence-v1"
OUTPUT_SCHEMA = "spore-temporal-comparison-v1"
SCALAR_METRICS = (
    "mean_luma",
    "p95_luma",
    "bright_fraction",
    "very_bright_fraction",
    "non_near_black_fraction",
    "luminous_centroid_x",
    "luminous_centroid_y",
)


def load_report(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"{path}: expected schema {INPUT_SCHEMA}")
    return report, hashlib.sha256(raw).hexdigest()


def by_name(items: list[dict], key: str, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}: missing/non-string {key}")
        if value in result:
            raise ValueError(f"{label}: duplicate {key} {value!r}")
        result[value] = item
    return result


def require_same_set(control: set[str], treatment: set[str], label: str) -> None:
    if control != treatment:
        missing = sorted(control - treatment)
        added = sorted(treatment - control)
        raise ValueError(f"{label} mismatch: missing_in_treatment={missing}, added_in_treatment={added}")


def delta(control: object, treatment: object) -> float | int | None:
    if control is None or treatment is None:
        return None
    if not isinstance(control, (int, float)) or not isinstance(treatment, (int, float)):
        raise ValueError("metric values must be numeric or null")
    control_f = float(control)
    treatment_f = float(treatment)
    if not math.isfinite(control_f) or not math.isfinite(treatment_f):
        raise ValueError("metric values must be finite")
    value = treatment_f - control_f
    if isinstance(control, int) and isinstance(treatment, int):
        return int(value)
    return round(value, 8)


def compare_sample(control: dict, treatment: dict) -> dict:
    for key in ("sample_key", "role", "stage_index", "stage_kind"):
        if control.get(key) != treatment.get(key):
            raise ValueError(
                f"sample {control.get('sample_key')}: semantic identity differs for {key}: "
                f"{control.get(key)!r} != {treatment.get(key)!r}"
            )

    control_metrics = control.get("metrics")
    treatment_metrics = treatment.get("metrics")
    if not isinstance(control_metrics, dict) or not isinstance(treatment_metrics, dict):
        raise ValueError(f"sample {control['sample_key']}: missing metrics")
    if control_metrics.get("thresholds") != treatment_metrics.get("thresholds"):
        raise ValueError(f"sample {control['sample_key']}: metric thresholds differ")

    metric_deltas = {
        metric: delta(control_metrics.get(metric), treatment_metrics.get(metric))
        for metric in SCALAR_METRICS
    }

    control_hash = control.get("frame_sha256")
    treatment_hash = treatment.get("frame_sha256")
    if not isinstance(control_hash, str) or not isinstance(treatment_hash, str):
        raise ValueError(f"sample {control['sample_key']}: missing frame hash")

    return {
        "sample_key": control["sample_key"],
        "role": control["role"],
        "stage_index": control.get("stage_index"),
        "stage_kind": control.get("stage_kind"),
        "control": {
            "target_elapsed_ms": control.get("target_elapsed_ms"),
            "actual_elapsed_ms": control.get("actual_elapsed_ms"),
            "timing_error_ms": control.get("timing_error_ms"),
            "exact_semantic_time": control.get("exact_semantic_time"),
            "frame_sha256": control_hash,
            "metrics": control_metrics,
        },
        "treatment": {
            "target_elapsed_ms": treatment.get("target_elapsed_ms"),
            "actual_elapsed_ms": treatment.get("actual_elapsed_ms"),
            "timing_error_ms": treatment.get("timing_error_ms"),
            "exact_semantic_time": treatment.get("exact_semantic_time"),
            "frame_sha256": treatment_hash,
            "metrics": treatment_metrics,
        },
        "frame_identical": control_hash == treatment_hash,
        "metric_delta_treatment_minus_control": metric_deltas,
    }


def compare_reports(control: dict, treatment: dict, control_hash: str, treatment_hash: str) -> dict:
    for field in ("width", "height"):
        if control.get(field) != treatment.get(field):
            raise ValueError(f"{field} mismatch: {control.get(field)} != {treatment.get(field)}")

    control_policy = control.get("policy", {})
    treatment_policy = treatment.get("policy", {})
    for field in ("sample_rule", "selection"):
        if control_policy.get(field) != treatment_policy.get(field):
            raise ValueError(f"temporal policy mismatch for {field}")

    control_cases = by_name(control.get("cases", []), "name", "control cases")
    treatment_cases = by_name(treatment.get("cases", []), "name", "treatment cases")
    require_same_set(set(control_cases), set(treatment_cases), "lifecycle case set")

    cases = []
    changed_samples = 0
    total_samples = 0
    for case_name in sorted(control_cases):
        control_case = control_cases[case_name]
        treatment_case = treatment_cases[case_name]
        for field in ("family", "cue"):
            # Family is allowed to differ because the renderer treatment can
            # intentionally express a different visual grammar for the same
            # factual case. Cue must remain semantically identical.
            if field == "cue" and control_case.get(field) != treatment_case.get(field):
                raise ValueError(f"{case_name}: cue mismatch")

        control_samples = by_name(control_case.get("samples", []), "sample_key", case_name)
        treatment_samples = by_name(treatment_case.get("samples", []), "sample_key", case_name)
        require_same_set(set(control_samples), set(treatment_samples), f"{case_name} sample set")

        samples = []
        for sample_key in control_samples:
            compared = compare_sample(control_samples[sample_key], treatment_samples[sample_key])
            total_samples += 1
            if not compared["frame_identical"]:
                changed_samples += 1
            samples.append(compared)

        cases.append(
            {
                "name": case_name,
                "control_family": control_case.get("family"),
                "treatment_family": treatment_case.get("family"),
                "cue": control_case.get("cue"),
                "control_terminal_frame_exact": control_case.get("terminal_frame_exact"),
                "treatment_terminal_frame_exact": treatment_case.get("terminal_frame_exact"),
                "samples": samples,
            }
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "policy": {
            "purpose": "control-treatment-difference-report-not-aesthetic-judgement",
            "pairing": "lifecycle case + semantic sample key",
            "delta_direction": "treatment-minus-control",
            "metrics_are_scores": False,
            "no_winner_field": True,
        },
        "control_manifest_sha256": control_hash,
        "treatment_manifest_sha256": treatment_hash,
        "width": control.get("width"),
        "height": control.get("height"),
        "summary": {
            "case_count": len(cases),
            "sample_count": total_samples,
            "pixel_changed_sample_count": changed_samples,
            "pixel_identical_sample_count": total_samples - changed_samples,
        },
        "cases": cases,
    }


def compare_paths(control_path: Path, treatment_path: Path, output: Path) -> dict:
    control, control_hash = load_report(control_path)
    treatment, treatment_hash = load_report(treatment_path)
    report = compare_reports(control, treatment, control_hash, treatment_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return report


def synthetic_report(mean_luma: float, frame_hash: str) -> dict:
    metrics = {
        "mean_luma": mean_luma,
        "p95_luma": 100,
        "bright_fraction": 0.1,
        "very_bright_fraction": 0.01,
        "non_near_black_fraction": 0.5,
        "luminous_centroid_x": 0.5,
        "luminous_centroid_y": 0.5,
        "thresholds": {
            "near_black_luma": 18,
            "centroid_luma": 64,
            "bright_luma": 96,
            "very_bright_luma": 160,
        },
    }
    return {
        "schema": INPUT_SCHEMA,
        "width": 4,
        "height": 2,
        "policy": {
            "sample_rule": "sequence-start + every BootStage midpoint + sequence-final",
            "selection": "nearest existing exact renderer frame; ties choose earlier frame",
        },
        "cases": [
            {
                "name": "case-a",
                "family": "Synthetic",
                "cue": "Starting",
                "terminal_frame_exact": True,
                "samples": [
                    {
                        "sample_key": "00-sequence-start",
                        "role": "sequence-start",
                        "stage_index": None,
                        "stage_kind": None,
                        "target_elapsed_ms": 0,
                        "actual_elapsed_ms": 0,
                        "timing_error_ms": 0,
                        "exact_semantic_time": True,
                        "frame_sha256": frame_hash,
                        "metrics": metrics,
                    }
                ],
            }
        ],
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="spore-temporal-compare-") as directory:
        root = Path(directory)
        control_path = root / "control.json"
        treatment_path = root / "treatment.json"
        output = root / "comparison.json"
        control_path.write_text(json.dumps(synthetic_report(20.0, "a" * 64)))
        treatment_path.write_text(json.dumps(synthetic_report(27.5, "b" * 64)))

        report = compare_paths(control_path, treatment_path, output)
        assert report["schema"] == OUTPUT_SCHEMA
        assert report["summary"] == {
            "case_count": 1,
            "sample_count": 1,
            "pixel_changed_sample_count": 1,
            "pixel_identical_sample_count": 0,
        }
        sample = report["cases"][0]["samples"][0]
        assert sample["metric_delta_treatment_minus_control"]["mean_luma"] == 7.5
        assert report["policy"]["metrics_are_scores"] is False
        assert "winner" not in report

        mismatched = synthetic_report(27.5, "b" * 64)
        mismatched["cases"][0]["samples"][0]["stage_kind"] = "Repair"
        treatment_path.write_text(json.dumps(mismatched))
        try:
            compare_paths(control_path, treatment_path, output)
        except ValueError as error:
            assert "semantic identity differs" in str(error)
        else:
            raise AssertionError("semantic mismatch should fail closed")

    print("spore_temporal_compare self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", nargs="?", type=Path)
    parser.add_argument("treatment", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, default=Path("spore-temporal-comparison.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.control is None or args.treatment is None:
        parser.error("control and treatment manifests are required unless --self-test is used")
    compare_paths(args.control, args.treatment, args.out)


if __name__ == "__main__":
    main()
