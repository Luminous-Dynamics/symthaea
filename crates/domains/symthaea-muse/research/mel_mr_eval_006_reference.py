#!/usr/bin/env python3
"""Independent MEL-MR-EVAL-006 reference oracle.

Research/test tooling only. This file does not import or execute Rust and must
not be used as a production audio verifier. Its purpose is cross-language
numerical sanity checking of the documented synthetic fixtures and metrics.

Run:
  python3 crates/domains/symthaea-muse/research/mel_mr_eval_006_reference.py \
    --source-blob-sha <Rust-source-blob-sha> \
    --runner-blob-sha <Rust-runner-blob-sha> \
    --output /tmp/mel-mr-eval-006-reference.json

Requires Python 3 and NumPy. The emitted artifact records this script's SHA-256
and explicitly distinguishes reference execution from committed Rust execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE_HZ = 16_000
TOTAL_FRAMES = 8_192


def sine(frequency_hz: float = 440.0, frames: int = TOTAL_FRAMES) -> np.ndarray:
    indices = np.arange(frames, dtype=np.float32)
    return np.sin(2 * np.pi * frequency_hz * indices / SAMPLE_RATE_HZ).astype(np.float32)


def burst(start_frame: int, end_frame: int) -> np.ndarray:
    indices = np.arange(TOTAL_FRAMES, dtype=np.float32)
    samples = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    mask = (indices >= start_frame) & (indices < end_frame)
    samples[mask] = np.sin(2 * np.pi * 440.0 * indices[mask] / SAMPLE_RATE_HZ)
    return samples


def hann_window(size: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))).astype(np.float32)


def rms_db(samples: np.ndarray) -> float:
    mean_square = float(np.mean(samples.astype(np.float64) ** 2))
    return float(20 * np.log10(max(math.sqrt(mean_square), 1e-12)))


def merge_masks(
    intervals: list[tuple[int, int]], guard_frames: int, total_frames: int
) -> list[tuple[int, int]]:
    expanded = sorted(
        (max(0, start - guard_frames), min(total_frames, end + guard_frames))
        for start, end in intervals
    )
    merged: list[tuple[int, int]] = []
    for start, end in expanded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def audit_acoustic(
    reference: np.ndarray,
    candidate: np.ndarray,
    authorized_masks: list[tuple[int, int]],
) -> dict[str, Any]:
    frame_size = 256
    hop_size = 128
    rms_tolerance_db = 0.5
    spectral_tolerance = 0.08
    minimum_windows = 4
    masks = merge_masks(authorized_masks, 0, len(reference))
    window = hann_window(frame_size)
    measurements: list[tuple[float, float, int]] = []

    for start in range(0, len(reference) - frame_size + 1, hop_size):
        end = start + frame_size
        if any(start < mask_end and end > mask_start for mask_start, mask_end in masks):
            continue
        ref_window = reference[start:end]
        candidate_window = candidate[start:end]
        rms_delta = abs(rms_db(ref_window) - rms_db(candidate_window))
        ref_spectrum = np.abs(np.fft.rfft(ref_window * window))
        candidate_spectrum = np.abs(np.fft.rfft(candidate_window * window))
        denominator = float(np.sum(ref_spectrum + candidate_spectrum))
        spectral_distance = (
            0.0
            if denominator <= np.finfo(float).eps
            else float(np.clip(np.sum(np.abs(ref_spectrum - candidate_spectrum)) / denominator, 0, 1))
        )
        measurements.append((rms_delta, spectral_distance, start))

    violations = [
        item for item in measurements
        if item[0] > rms_tolerance_db or item[1] > spectral_tolerance
    ]
    verdict = (
        "fail" if violations
        else "pass" if len(measurements) >= minimum_windows
        else "abstain"
    )
    return {
        "actual_verdict": verdict,
        "protected_windows": len(measurements),
        "violating_windows": len(violations),
        "worst_rms_delta_db": max((item[0] for item in measurements), default=0.0),
        "worst_spectral_distance": max((item[1] for item in measurements), default=0.0),
    }


def detect_onsets(samples: np.ndarray) -> list[dict[str, float | int]]:
    frame_size = 128
    hop_size = 32
    novelty_threshold = 0.12
    minimum_separation_frames = max(1, int(8 * SAMPLE_RATE_HZ / 1000))
    window = hann_window(frame_size)
    novelty_points: list[tuple[int, float]] = []
    previous_magnitudes: np.ndarray | None = None

    for start in range(0, len(samples) - frame_size + 1, hop_size):
        magnitudes = np.abs(np.fft.rfft(samples[start : start + frame_size] * window))
        energy = float(np.sum(magnitudes))
        if previous_magnitudes is None:
            novelty = 0.0 if energy <= np.finfo(float).eps else 1.0
        else:
            novelty = (
                0.0
                if energy <= np.finfo(float).eps
                else float(
                    np.clip(
                        np.sum(np.maximum(magnitudes - previous_magnitudes, 0)) / energy,
                        0,
                        1,
                    )
                )
            )
        novelty_points.append((start + frame_size // 2, novelty))
        previous_magnitudes = magnitudes

    detected: list[dict[str, float | int]] = []
    for index, (frame, novelty) in enumerate(novelty_points):
        previous = novelty_points[index - 1][1] if index else 0.0
        following = novelty_points[index + 1][1] if index + 1 < len(novelty_points) else 0.0
        if novelty < novelty_threshold or novelty < previous or novelty <= following:
            continue
        if detected and frame - int(detected[-1]["frame"]) < minimum_separation_frames:
            continue
        detected.append({"frame": frame, "novelty": novelty})
    return detected


def audit_onsets(
    reference: np.ndarray,
    candidate: np.ndarray,
    authorized_masks: list[tuple[int, int]],
) -> dict[str, Any]:
    frame_size = 128
    hop_size = 32
    timing_tolerance_frames = int(4 * SAMPLE_RATE_HZ / 1000)
    minimum_protected_frames = math.ceil(100 * SAMPLE_RATE_HZ / 1000)
    effective_guard = frame_size // 2 + hop_size
    masks = merge_masks(authorized_masks, effective_guard, len(reference))
    protected_duration = len(reference) - sum(end - start for start, end in masks)

    ref_events = [
        event for event in detect_onsets(reference)
        if not any(start <= int(event["frame"]) < end for start, end in masks)
    ]
    candidate_events = [
        event for event in detect_onsets(candidate)
        if not any(start <= int(event["frame"]) < end for start, end in masks)
    ]

    matched_candidate_indices: set[int] = set()
    missing: list[dict[str, float | int]] = []
    matched = 0
    for ref_event in ref_events:
        options = [
            (abs(int(candidate_event["frame"]) - int(ref_event["frame"])), index, candidate_event)
            for index, candidate_event in enumerate(candidate_events)
            if index not in matched_candidate_indices
        ]
        if not options:
            missing.append(ref_event)
            continue
        displacement, candidate_index, _ = min(options, key=lambda item: item[0])
        if displacement <= timing_tolerance_frames:
            matched_candidate_indices.add(candidate_index)
            matched += 1
        else:
            missing.append(ref_event)
    extras = [
        event for index, event in enumerate(candidate_events)
        if index not in matched_candidate_indices
    ]

    if missing or extras:
        verdict = "fail"
    elif not ref_events or protected_duration < minimum_protected_frames:
        verdict = "abstain"
    else:
        verdict = "pass"

    return {
        "actual_verdict": verdict,
        "reference_onsets_frames": [int(event["frame"]) for event in ref_events],
        "candidate_onsets_frames": [int(event["frame"]) for event in candidate_events],
        "matched_events": matched,
        "missing_onsets": len(missing),
        "extra_onsets": len(extras),
        "effective_boundary_guard_frames": effective_guard,
    }


def wilson_95_interval(successes: int, trials: int) -> dict[str, float] | None:
    if trials == 0 or successes < 0 or successes > trials:
        return None
    n = float(trials)
    p = successes / n
    z = 1.959963984540054
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (p + z_squared / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z_squared / (4 * n * n)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return {"lower": lower, "upper": upper}


def build_case(
    case_id: str,
    family: str,
    critical: bool,
    expected: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "critical_mutation": critical,
        "expected_verdict": expected,
        **result,
        "expected_outcome_reproduced": result["actual_verdict"] == expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blob-sha", default="UNPINNED")
    parser.add_argument("--runner-blob-sha", default="UNPINNED")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_sine = sine()
    local_damage = reference_sine.copy()
    local_damage[1024:1280] = 0.0
    gain_drop = reference_sine.copy()
    gain_drop[1024:1280] *= 0.25
    frequency_swap = reference_sine.copy()
    indices = np.arange(1024, 1280, dtype=np.float32)
    frequency_swap[1024:1280] = np.sin(2 * np.pi * 880.0 * indices / SAMPLE_RATE_HZ)
    authorized_damage = reference_sine.copy()
    authorized_damage[4096:5120] = 0.0
    authorized_mask = [(4096, 5120)]

    reference_burst = burst(1024, 3072)
    delayed_burst = burst(1536, 3584)
    extra_transient = reference_burst.copy()
    extra_transient[5120:6144] = burst(5120, 6144)[5120:6144]
    silence = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    mask_hiding_onsets = [(512, 4096)]

    cases = [
        build_case("AP-IDENTITY-001", "identity_control", False, "pass",
                   audit_acoustic(reference_sine, reference_sine, authorized_mask)),
        build_case("AP-LOCAL-LEAK-001", "acoustic_protected_mutation", True, "fail",
                   audit_acoustic(reference_sine, local_damage, authorized_mask)),
        build_case("AP-GAIN-DROP-001", "acoustic_protected_mutation", True, "fail",
                   audit_acoustic(reference_sine, gain_drop, authorized_mask)),
        build_case("AP-FREQUENCY-SWAP-001", "acoustic_protected_mutation", True, "fail",
                   audit_acoustic(reference_sine, frequency_swap, authorized_mask)),
        build_case("AP-AUTHORIZED-ONLY-001", "authorization_scope_control", False, "pass",
                   audit_acoustic(reference_sine, authorized_damage, authorized_mask)),
        build_case("ONSET-IDENTITY-001", "identity_control", False, "pass",
                   audit_onsets(reference_burst, reference_burst, [])),
        build_case("ONSET-DELAY-001", "onset_event_mutation", True, "fail",
                   audit_onsets(reference_burst, delayed_burst, [])),
        build_case("ONSET-EXTRA-001", "onset_event_mutation", True, "fail",
                   audit_onsets(reference_burst, extra_transient, [])),
        build_case("ONSET-MASK-001", "evidence_coverage_control", False, "abstain",
                   audit_onsets(reference_burst, delayed_burst, mask_hiding_onsets)),
        build_case("ONSET-UNMEASURABLE-001", "evidence_coverage_control", False, "abstain",
                   audit_onsets(silence, silence, [])),
        build_case("ONSET-MISSING-001", "onset_event_mutation", True, "fail",
                   audit_onsets(reference_burst, silence, [])),
    ]

    passed = sum(case["expected_outcome_reproduced"] for case in cases)
    critical = [case for case in cases if case["critical_mutation"] and case["expected_verdict"] == "fail"]
    positive = [case for case in cases if case["expected_verdict"] == "pass"]
    false_accepts = sum(case["actual_verdict"] == "pass" for case in critical)
    false_rejections = sum(case["actual_verdict"] == "fail" for case in positive)

    family_rollup: dict[str, dict[str, int]] = {}
    for case in cases:
        stats = family_rollup.setdefault(
            case["family"],
            {"cases": 0, "reproduced": 0, "critical_mutations": 0, "critical_false_accepts": 0},
        )
        stats["cases"] += 1
        stats["reproduced"] += int(case["expected_outcome_reproduced"])
        if case["critical_mutation"] and case["expected_verdict"] == "fail":
            stats["critical_mutations"] += 1
            stats["critical_false_accepts"] += int(case["actual_verdict"] == "pass")
    for stats in family_rollup.values():
        stats["critical_false_accept_wilson_95_two_sided"] = wilson_95_interval(
            stats["critical_false_accepts"], stats["critical_mutations"]
        )

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact = {
        "schema_version": "1.0.0",
        "benchmark": "MEL-MR-EVAL-006",
        "artifact_kind": "independent_reference_sanity_check",
        "generated_date": "2026-10-29",
        "implementation_under_test": {
            "source_path": "crates/domains/symthaea-muse/src/audio_preservation.rs",
            "source_blob_sha": args.source_blob_sha,
            "runner_path": "crates/domains/symthaea-muse/examples/mel_mr_eval_006_conformance.rs",
            "runner_blob_sha": args.runner_blob_sha,
            "claimed_execution_of_rust": False,
        },
        "reference_oracle": {
            "script_path": "crates/domains/symthaea-muse/research/mel_mr_eval_006_reference.py",
            "script_sha256": script_sha,
            "language": "Python 3",
            "numpy_version": np.__version__,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frames_per_case": TOTAL_FRAMES,
            "channels": 1,
            "acoustic_thresholds": {
                "frame_size": 256, "hop_size": 128, "rms_delta_db_tolerance": 0.5,
                "spectral_distance_tolerance": 0.08, "minimum_protected_windows": 4,
            },
            "onset_thresholds": {
                "frame_size": 128, "hop_size": 32, "novelty_threshold": 0.12,
                "minimum_separation_ms": 8, "onset_time_tolerance_ms": 4,
                "automatic_analysis_guard_frames": 96, "minimum_protected_duration_ms": 100,
            },
            "limitations": [
                "Independent reference implementation; it does not import or execute Rust.",
                "Synthetic signals only; not a real music corpus, human listening study, or editor head-to-head benchmark.",
                "Thresholds are fixture settings, not production calibration.",
                "Results establish expected numerical behavior only on these fixed cases.",
            ],
        },
        "summary": {
            "cases_total": len(cases),
            "expected_verdicts_reproduced": passed,
            "mismatches": len(cases) - passed,
            "critical_mutations_total": len(critical),
            "critical_false_accepts": false_accepts,
            "critical_false_accept_rate": false_accepts / len(critical) if critical else None,
            "critical_false_accept_rate_wilson_95_two_sided": wilson_95_interval(false_accepts, len(critical)),
            "positive_controls_total": len(positive),
            "false_rejections": false_rejections,
            "false_rejection_rate": false_rejections / len(positive) if positive else None,
            "false_rejection_rate_wilson_95_two_sided": wilson_95_interval(false_rejections, len(positive)),
            "caveat": "The 0/6 false-accept panel has a two-sided 95% Wilson upper bound of about 39%; it is a smoke test, not a reliability claim.",
        },
        "per_family": family_rollup,
        "cases": cases,
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
