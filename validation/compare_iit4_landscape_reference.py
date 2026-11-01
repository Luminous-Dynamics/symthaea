#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed comparison of PyPhi and Rust IIT4 (2023) landscape artifacts.

Inputs are generated independently by the pinned PyPhi oracle and the Rust
per-cut diagnostic workflow. This tool compares common model identities,
selection states, all cut scores, MIP tie families, and final system values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise RuntimeError(f"{path}: missing CSV header")
        return list(reader)


def _float(value: Any, context: str) -> float:
    if value is None:
        raise RuntimeError(f"{context}: missing numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{context}: non-finite numeric value {value!r}")
    return result


def _close(left: Any, right: Any, tolerance: float, context: str) -> dict[str, Any] | None:
    a = _float(left, context + " left")
    b = _float(right, context + " right")
    difference = abs(a - b)
    allowed = tolerance * max(1.0, abs(a), abs(b))
    if difference > allowed:
        return {
            "field": context,
            "pyphi": a,
            "rust": b,
            "absolute_difference": difference,
            "allowed_difference": allowed,
        }
    return None


def _state(value: Any, context: str) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError(f"{context}: expected nonempty state vector")
    if any(bit not in (0, 1, False, True) for bit in value):
        raise RuntimeError(f"{context}: non-binary state value")
    return "".join("1" if bool(bit) else "0" for bit in value)


def compare(
    pyphi_path: Path,
    rust_landscape_path: Path,
    rust_partition_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    pyphi = json.loads(pyphi_path.read_text(encoding="utf-8"))
    if pyphi.get("status") != "PASS":
        raise RuntimeError("PyPhi reference artifact is not PASS")
    if pyphi.get("schema_version") != "pyphi-iit4-2023-landscape-reference-v0.1":
        raise RuntimeError("unexpected PyPhi landscape schema")
    if pyphi.get("provenance", {}).get("formalism") != "IIT_4_0_2023":
        raise RuntimeError("PyPhi landscape formalism is not IIT_4_0_2023")
    if pyphi.get("skipped_points") != []:
        raise RuntimeError("PyPhi landscape contains skipped points")

    rust_points = _read_csv(rust_landscape_path)
    rust_scores = _read_csv(rust_partition_path)
    if not rust_points or not rust_scores:
        raise RuntimeError("Rust landscape or per-partition artifact is empty")

    points_by_theta: dict[str, dict[str, str]] = {}
    for row in rust_points:
        theta = row.get("theta")
        if not theta or theta in points_by_theta:
            raise RuntimeError(f"Rust landscape has missing or duplicate theta {theta!r}")
        points_by_theta[theta] = row

    scores_by_theta: dict[str, list[dict[str, str]]] = {}
    for row in rust_scores:
        theta = row.get("theta")
        if not theta:
            raise RuntimeError("Rust per-cut record has no theta")
        scores_by_theta.setdefault(theta, []).append(row)

    mismatches: list[dict[str, Any]] = []
    compared_points = 0
    total_cut_rows = 0

    for point in pyphi["points"]:
        theta = float(point["theta"])
        theta_key = f"{theta:.8f}"
        rust_point = points_by_theta.get(theta_key)
        if rust_point is None:
            # Accept only an unambiguous numerically identical CSV theta.
            matches = [
                row for key, row in points_by_theta.items()
                if math.isclose(float(key), theta, rel_tol=0.0, abs_tol=1e-9)
            ]
            if len(matches) != 1:
                mismatches.append({"theta": theta, "field": "rust_point_presence"})
                continue
            rust_point = matches[0]

        point_mismatches: list[dict[str, Any]] = []
        identity_pairs = (
            ("model_sha256", "canonical_dense_model_sha256"),
            ("evaluation_sha256", "canonical_dense_evaluation_sha256"),
            ("analysis_request_sha256", "canonical_dense_analysis_request_sha256"),
        )
        for py_field, rust_field in identity_pairs:
            if point[py_field] != rust_point.get(rust_field):
                point_mismatches.append({
                    "field": py_field,
                    "pyphi": point[py_field],
                    "rust": rust_point.get(rust_field),
                })

        py_cause = _state(point["cause_state"], f"theta={theta} PyPhi cause")
        py_effect = _state(point["effect_state"], f"theta={theta} PyPhi effect")
        rust_cause = rust_point.get("selected_cause_state")
        rust_effect = rust_point.get("selected_effect_state")
        if py_cause != rust_cause:
            point_mismatches.append({"field": "cause_state", "pyphi": py_cause, "rust": rust_cause})
        if py_effect != rust_effect:
            point_mismatches.append({"field": "effect_state", "pyphi": py_effect, "rust": rust_effect})

        py_ties = sorted(point["mip_tie_cut_sha256s"])
        rust_tie_field = rust_point.get("tied_mip_cut_sha256s", "")
        rust_ties = sorted(x for x in rust_tie_field.split(";") if x and x != "NA")
        if py_ties != rust_ties:
            point_mismatches.append({"field": "mip_tie_cut_sha256s", "pyphi": py_ties, "rust": rust_ties})
        if rust_point.get("winning_mip_cut_sha256") not in py_ties:
            point_mismatches.append({
                "field": "winning_mip_representative_in_pyphi_tie_set",
                "pyphi": py_ties,
                "rust": rust_point.get("winning_mip_cut_sha256"),
            })
        try:
            rust_tie_count = int(rust_point.get("mip_tie_count", "0"))
        except ValueError as exc:
            raise RuntimeError(f"theta={theta}: invalid Rust MIP tie count") from exc
        if rust_tie_count != len(py_ties):
            point_mismatches.append({
                "field": "mip_tie_count",
                "pyphi": len(py_ties),
                "rust": rust_tie_count,
            })

        for py_field, rust_field in (
            ("phi", "phi_s"),
            ("signed_phi", "signed_phi"),
            ("normalized_phi", "normalized_phi"),
            ("signed_normalized_phi", "signed_normalized_phi"),
        ):
            mismatch = _close(
                point[py_field],
                rust_point.get(rust_field),
                tolerance,
                f"theta={theta} {py_field}",
            )
            if mismatch:
                point_mismatches.append(mismatch)

        py_regime = int(point["regime"])
        rust_regime = int(rust_point["selection_regime"])
        if py_regime != rust_regime:
            point_mismatches.append({"field": "selection_regime", "pyphi": py_regime, "rust": rust_regime})

        selected_scores = [
            row for row in scores_by_theta.get(theta_key, [])
            if row.get("cause_state") == py_cause and row.get("effect_state") == py_effect
        ]
        if not selected_scores:
            selected_scores = [
                row for key, rows in scores_by_theta.items()
                if math.isclose(float(key), theta, rel_tol=0.0, abs_tol=1e-9)
                for row in rows
                if row.get("cause_state") == py_cause and row.get("effect_state") == py_effect
            ]

        py_partition_scores = point["partition_scores"]
        py_by_cut = {row["cut_sha256"]: row for row in py_partition_scores}
        rust_by_cut: dict[str, dict[str, str]] = {}
        for row in selected_scores:
            cut = row["cut_sha256"]
            if cut in rust_by_cut:
                point_mismatches.append({"field": "duplicate_rust_cut_score", "cut_sha256": cut})
                continue
            rust_by_cut[cut] = row
        if set(py_by_cut) != set(rust_by_cut):
            point_mismatches.append({
                "field": "partition_cut_set",
                "pyphi_cut_count": len(py_by_cut),
                "rust_cut_count": len(rust_by_cut),
                "pyphi_missing_in_rust": sorted(set(py_by_cut) - set(rust_by_cut)),
                "rust_missing_in_pyphi": sorted(set(rust_by_cut) - set(py_by_cut)),
            })

        for cut in sorted(set(py_by_cut) & set(rust_by_cut)):
            py_score = py_by_cut[cut]
            rust_score = rust_by_cut[cut]
            for py_field, rust_field in (
                ("cause_signed_phi", "cause_signed_phi"),
                ("effect_signed_phi", "effect_signed_phi"),
                ("signed_phi", "signed_phi"),
                ("phi", "phi"),
                ("normalized_phi", "normalized_phi"),
                ("signed_normalized_phi", "signed_normalized_phi"),
            ):
                mismatch = _close(
                    py_score[py_field],
                    rust_score.get(rust_field),
                    tolerance,
                    f"theta={theta} cut={cut} {py_field}",
                )
                if mismatch:
                    point_mismatches.append({"cut_sha256": cut, **mismatch})
        total_cut_rows += len(set(py_by_cut) & set(rust_by_cut))
        compared_points += 1
        if point_mismatches:
            mismatches.append({
                "theta": theta,
                "mismatch_count": len(point_mismatches),
                "first_mismatches": point_mismatches[:12],
            })

    status = "PASS" if compared_points == 13 and not mismatches else "NOT_QUALIFIED"
    return {
        "schema_version": "iit4-2023-landscape-cross-engine-comparison-v0.1",
        "status": status,
        "qualification_scope": "same_input_pyphi_rust_landscape_and_per_cut_score_comparison",
        "consciousness_inference_permitted": False,
        "provenance": {
            "formalism": "IIT_4_0_2023",
            "pyphi_version": pyphi["provenance"]["version"],
            "pyphi_wheel_sha256": pyphi["provenance"]["wheel_sha256"],
            "tolerance_relative": tolerance,
            "pyphi_landscape_sha256": __import__("hashlib").sha256(
                pyphi_path.read_bytes()
            ).hexdigest(),
            "rust_landscape_sha256": __import__("hashlib").sha256(
                rust_landscape_path.read_bytes()
            ).hexdigest(),
            "rust_per_cut_sha256": __import__("hashlib").sha256(
                rust_partition_path.read_bytes()
            ).hexdigest(),
        },
        "summary": {
            "expected_point_count": 13,
            "compared_point_count": compared_points,
            "matched_cut_score_count": total_cut_rows,
            "mismatch_point_count": len(mismatches),
        },
        "mismatches": mismatches,
        "notes": [
            "Exact model/evaluation/analysis-request digests are compared before score values.",
            "Cause and effect contributions and combined signed/clamped/normalized scores are compared for every common cut.",
            "Complete MIP tie families are compared as sets of canonical directed cut-matrix digests.",
            "Selection regimes are sampled grid labels only; they do not certify continuous switch-free intervals.",
            "A failed, incomplete, missing, duplicate, or mismatched record cannot produce PASS.",
            "This comparison is scoped to the frozen 3-unit binary local-CPT IIT 4.0 (2023) landscape fixture only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyphi", type=Path, required=True)
    parser.add_argument("--rust-landscape", type=Path, required=True)
    parser.add_argument("--rust-partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if not (math.isfinite(args.tolerance) and 0.0 < args.tolerance <= 1e-3):
        raise SystemExit("tolerance must be finite and in (0, 1e-3]")
    try:
        result = compare(
            args.pyphi,
            args.rust_landscape,
            args.rust_partitions,
            args.tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    print(
        f"status={result['status']} "
        f"points={result['summary']['compared_point_count']}/13 "
        f"cut_scores={result['summary']['matched_cut_score_count']} "
        f"mismatches={result['summary']['mismatch_point_count']}"
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
