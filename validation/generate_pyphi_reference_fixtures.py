#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run a pinned upstream PyPhi literature-reference smoke test.

This test verifies the external oracle/API against published example outputs.
It does not measure Symthaea's HDC estimator and cannot qualify consciousness.
The result artifact binds the canonical TPM, declared connectivity, state,
formalism, package version, Python version, and all computed values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import pyphi


PINNED_PYPHI_VERSION = "2.0.0rc4"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pyphi-iit4-2023-fig1a-reference-v0.1.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _float_matrix(value: Any) -> list[list[float]]:
    return [[float(cell) for cell in row] for row in value]


def _model_digest(substrate: Any, state: tuple[int, ...], formalism: str) -> tuple[str, dict[str, Any]]:
    # PyPhi documents state-by-node TPMs and little-endian row ordering. Use
    # its canonical TPM export rather than serializing a display string.
    tpm_frame = substrate.tpm.to_pandas()
    tpm = _float_matrix(tpm_frame.values.tolist())
    row_states = [list(map(int, row if isinstance(row, tuple) else (row,))) for row in tpm_frame.index.tolist()]
    model = {
        "formalism": formalism,
        "node_labels": [str(label) for label in substrate.node_labels],
        "state": list(state),
        "state_order": row_states,
        "tpm_state_by_node": tpm,
        "connectivity_matrix": [[int(cell) for cell in row] for row in substrate.cm.tolist()],
        "time_grain": "one PyPhi discrete update",
    }
    return hashlib.sha256(_canonical_json(model)).hexdigest(), model


def _extract_measurements(substrate: Any, state: tuple[int, ...], formalism: str) -> dict[str, float]:
    analyses = {
        "system-phi-s-A": pyphi.analyze(substrate, state=state, subset=(0,), formalism=formalism),
        "system-phi-s-AB": pyphi.analyze(substrate, state=state, subset=(0, 1), formalism=formalism),
        "system-phi-s-ABC": pyphi.analyze(substrate, state=state, subset=(0, 1, 2), formalism=formalism),
    }
    ab = analyses["system-phi-s-AB"]
    values = {
        "system-phi-s-A": float(analyses["system-phi-s-A"].phi),
        "system-phi-s-AB": float(ab.phi),
        "AB-cause-phi": float(ab.sia.cause.phi),
        "AB-effect-phi": float(ab.sia.effect.phi),
        "AB-structure-Phi": float(ab.ces.big_phi),
        "system-phi-s-ABC": float(analyses["system-phi-s-ABC"].phi),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("oracle produced a non-finite quantity; refusing to write a passing artifact")
    return values


def run(output_path: Path) -> dict[str, Any]:
    try:
        installed_version = importlib.metadata.version("pyphi")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("PyPhi distribution metadata unavailable; install the pinned pyphi==2.0.0rc4 artifact") from exc

    if installed_version != PINNED_PYPHI_VERSION:
        raise RuntimeError(
            f"wrong PyPhi version: expected {PINNED_PYPHI_VERSION}, found {installed_version}; refusing an unpinned comparison"
        )

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    reference = fixture["reference"]
    if reference["exact_version"] != PINNED_PYPHI_VERSION:
        raise RuntimeError("fixture and runner version pins disagree")
    formalism = reference["formalism"]
    state = tuple(int(bit) for bit in reference["state"])

    pyphi.config.progress_bars = False
    substrate = pyphi.examples.iit4_2023_fig1a_substrate()
    model_sha256, canonical_model = _model_digest(substrate, state, formalism)
    measured = _extract_measurements(substrate, state, formalism)

    comparisons = []
    for expected in fixture["quantities"]:
        actual = measured[expected["id"]]
        error = abs(actual - float(expected["expected"]))
        tolerance = float(expected["absolute_tolerance"])
        comparisons.append({
            "id": expected["id"],
            "quantity": expected["kind"],
            "subset": expected["subset"],
            "expected_anchor": expected["expected"],
            "actual": actual,
            "absolute_error_to_documented_anchor": error,
            "absolute_tolerance": tolerance,
            "matches_anchor": error <= tolerance,
        })

    passed = all(item["matches_anchor"] for item in comparisons)
    artifact = {
        "schema_version": "pyphi-reference-run-v0.1",
        "status": "PASS" if passed else "MISMATCH",
        "qualification_scope": "upstream_reference_smoke_test_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed_version,
            "python_version": platform.python_version(),
            "formalism": formalism,
            "upstream_example": reference["api_example"],
            "state": list(state),
            "model_sha256": model_sha256,
            "model_digest_semantics": "SHA-256 of canonical JSON containing TPM, row order, node labels, declared connectivity, state, time grain, and formalism",
            "source_fixture": str(FIXTURE_PATH.name),
            "source_url": reference["source_url"],
        },
        "canonical_model": canonical_model,
        "comparisons": comparisons,
        "summary": {
            "expected_quantity_count": len(fixture["quantities"]),
            "computed_quantity_count": len(measured),
            "passed_anchor_count": sum(1 for item in comparisons if item["matches_anchor"]),
            "failed_anchor_count": sum(1 for item in comparisons if not item["matches_anchor"]),
        },
        "notes": [
            "This run checks only that pinned PyPhi reproduces its documented IIT 4.0 (2023) example anchors.",
            "It is not a Symthaea/HDC comparison, and does not validate consciousness.",
            "Any anchor mismatch remains visible and produces a nonzero exit code.",
            "The TPM digest is necessary provenance, not a proof of formal correctness.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("PYPHI_IIT4_2023_REFERENCE_RUN.json"))
    args = parser.parse_args()
    try:
        artifact = run(args.output)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2

    print(f"status={artifact['status']}")
    print(f"model_sha256={artifact['provenance']['model_sha256']}")
    print(f"anchors={artifact['summary']['passed_anchor_count']}/{artifact['summary']['expected_quantity_count']}")
    print(f"artifact={args.output}")
    return 0 if artifact["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
