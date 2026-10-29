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
import struct
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


MODEL_MAGIC = b"SYMTHAEA-IIT-BINARY-TPM\0v0.1\0"
EVALUATION_MAGIC = b"SYMTHAEA-IIT-EVALUATION\0v0.1\0"
ANALYSIS_REQUEST_MAGIC = b"SYMTHAEA-IIT-ANALYSIS-REQUEST\0v0.1\0"


def _canonical_model_bytes(tpm: list[list[float]], connectivity: list[list[int]]) -> bytes:
    """Match symthaea-iit's v0.1 little-endian binary TPM encoding."""
    n_units = len(tpm[0]) if tpm else 0
    if not 1 <= n_units <= 16 or len(tpm) != (1 << n_units):
        raise RuntimeError("TPM dimensions are outside the Rust v0.1 dense binary model contract")
    if any(len(row) != n_units for row in tpm):
        raise RuntimeError("TPM must be state-by-unit with exactly n_units columns")
    if len(connectivity) != n_units or any(len(row) != n_units for row in connectivity):
        raise RuntimeError("connectivity matrix shape does not match TPM unit count")

    encoded = bytearray(MODEL_MAGIC)
    encoded.extend(struct.pack("<I", n_units))
    for row in tpm:
        for cell in row:
            probability = float(cell)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError("TPM contains a non-finite or out-of-range probability")
            if probability == 0.0:
                probability = 0.0  # canonicalize negative zero like the Rust model
            encoded.extend(struct.pack("<d", probability))

    for row in connectivity:
        for edge in row:
            bit = int(edge)
            if bit not in (0, 1):
                raise RuntimeError("connectivity matrix must contain only binary entries")
            encoded.append(bit)
    return bytes(encoded)


def _evaluation_digest(model_bytes: bytes, state: tuple[int, ...]) -> str:
    if any(bit not in (0, 1) for bit in state):
        raise RuntimeError("evaluation state must contain only binary bits")
    return hashlib.sha256(EVALUATION_MAGIC + model_bytes + bytes(state)).hexdigest()


def _analysis_request_digest(
    model_bytes: bytes,
    state: tuple[int, ...],
    subset: list[int],
    formalism: str,
    quantity_id: str,
) -> str:
    """Match Rust AnalysisRequest::request_sha256 exactly."""
    encoded = bytearray(ANALYSIS_REQUEST_MAGIC)
    encoded.extend(model_bytes)
    encoded.extend(bytes(state))
    encoded.extend(struct.pack("<I", len(subset)))
    for unit in subset:
        encoded.extend(struct.pack("<I", int(unit)))
    encoded.extend(b"\0")
    encoded.extend(formalism.encode("utf-8"))
    encoded.extend(b"\0")
    encoded.extend(quantity_id.encode("utf-8"))
    return hashlib.sha256(encoded).hexdigest()


def _model_digest(
    substrate: Any, state: tuple[int, ...], formalism: str
) -> tuple[str, str, str, dict[str, Any], bytes]:
    # PyPhi documents state-by-node TPMs, little-endian rows, and source-to-target
    # connectivity. The binary digests now match the Rust core contract byte-for-byte.
    tpm_frame = substrate.tpm.to_pandas()
    tpm = _float_matrix(tpm_frame.values.tolist())
    connectivity = [[int(cell) for cell in row] for row in substrate.cm.tolist()]
    row_states = [
        list(map(int, row if isinstance(row, tuple) else (row,)))
        for row in tpm_frame.index.tolist()
    ]
    model = {
        "formalism": formalism,
        "node_labels": [str(label) for label in substrate.node_labels],
        "state": list(state),
        "state_order": row_states,
        "tpm_state_by_node": tpm,
        "connectivity_matrix": connectivity,
        "time_grain": "one PyPhi discrete update",
    }
    model_bytes = _canonical_model_bytes(tpm, connectivity)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    evaluation_sha256 = _evaluation_digest(model_bytes, state)
    reference_manifest_sha256 = hashlib.sha256(_canonical_json(model)).hexdigest()
    return model_sha256, evaluation_sha256, reference_manifest_sha256, model, model_bytes

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
    model_sha256, evaluation_sha256, reference_manifest_sha256, canonical_model, model_bytes = _model_digest(
        substrate, state, formalism
    )
    measured = _extract_measurements(substrate, state, formalism)

    comparisons = []
    for expected in fixture["quantities"]:
        actual = measured[expected["id"]]
        error = abs(actual - float(expected["expected"]))
        tolerance = float(expected["absolute_tolerance"])
        comparisons.append({
            "id": expected["id"],
            "quantity": expected["kind"],
            "quantity_id": expected["quantity_id"],
            "analysis_request_sha256": _analysis_request_digest(
                model_bytes, state, expected["subset"], formalism, expected["quantity_id"]
            ),
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
            "model_digest_semantics": "SHA-256 of SYMTHAEA-IIT-BINARY-TPM v0.1 bytes: n_units u32 LE, state-by-unit f64 LE TPM, then source-major connectivity bytes; negative zero normalized",
            "evaluation_sha256": evaluation_sha256,
            "reference_manifest_sha256": reference_manifest_sha256,
            "analysis_request_digest_semantics": "SHA-256 of SYMTHAEA-IIT-ANALYSIS-REQUEST v0.1 bytes: canonical model, state bits, candidate subset length/indices u32 LE, NUL, formalism id, NUL, quantity id",
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
