#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate pinned IIT 4.0 (2023) PyPhi primitive repertoires for Rust parity."""

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

import numpy as np

try:
    from .generate_pyphi_reference_fixtures import (
        EXPECTED_CONNECTIVITY,
        EXPECTED_EVALUATION_SHA256,
        EXPECTED_FORMALISM,
        EXPECTED_MODEL_SHA256,
        EXPECTED_PYPHI_WHEEL_SHA256,
        EXPECTED_STATE,
        EXPECTED_TPM_STATE_BY_NODE,
        PINNED_PYPHI_VERSION,
    )
    from .iit_digest_contract import (
        canonical_model_bytes,
        evaluation_digest,
        model_digest,
        primitive_request_digest,
    )
except ImportError:  # Direct execution: python validation/generate_pyphi_primitive_corpus.py
    from generate_pyphi_reference_fixtures import (
        EXPECTED_CONNECTIVITY,
        EXPECTED_EVALUATION_SHA256,
        EXPECTED_FORMALISM,
        EXPECTED_MODEL_SHA256,
        EXPECTED_PYPHI_WHEEL_SHA256,
        EXPECTED_STATE,
        EXPECTED_TPM_STATE_BY_NODE,
        PINNED_PYPHI_VERSION,
    )
    from iit_digest_contract import (
        canonical_model_bytes,
        evaluation_digest,
        model_digest,
        primitive_request_digest,
    )

CANDIDATE_CASES = (
    ("AB", (0, 1), ((), (0,), (1,), (0, 1))),
    ("ABC", (0, 1, 2), (
        (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
    )),
)
OPERATIONS = (
    ("cause_repertoire", "iit4_cause_repertoire", "cause_repertoire"),
    ("effect_repertoire", "iit4_effect_repertoire", "effect_repertoire"),
    (
        "forward_cause_likelihood",
        "iit4_forward_cause_likelihood",
        "forward_cause_repertoire",
    ),
)
OUTPUT_SCHEMA_VERSION = "pyphi-iit4-2023-primitive-corpus-v0.1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _flat_little_endian(array: Any) -> tuple[list[int], list[float]]:
    values = np.asarray(array, dtype=float)
    shape = [int(size) for size in values.shape]
    flat = values.ravel(order="F")
    result = [float(value) for value in flat.tolist()]
    if not result or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
        raise RuntimeError("PyPhi primitive returned an empty, non-finite, or invalid probability vector")
    return shape, result


def _vector_sha256(values: list[float]) -> str:
    import struct

    encoded = b"SYMTHAEA-IIT-PRIMITIVE-VECTOR\0v0.1\0"
    encoded += b"".join(struct.pack("<d", 0.0 if value == 0.0 else value) for value in values)
    return hashlib.sha256(encoded).hexdigest()


def generate(output_path: Path) -> dict[str, Any]:
    installed_version = importlib.metadata.version("pyphi")
    if installed_version != PINNED_PYPHI_VERSION:
        raise RuntimeError(
            f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed_version}"
        )

    import pyphi
    from pyphi.substrate import Substrate
    from pyphi.system import System

    pyphi.config.progress_bars = False
    state = tuple(EXPECTED_STATE)
    tpm = np.asarray(EXPECTED_TPM_STATE_BY_NODE, dtype=float)
    connectivity = np.asarray(EXPECTED_CONNECTIVITY, dtype=int)
    substrate = Substrate(tpm, cm=connectivity, node_labels=("A", "B", "C"))
    model_bytes = canonical_model_bytes(
        EXPECTED_TPM_STATE_BY_NODE, EXPECTED_CONNECTIVITY
    )
    model_sha256 = model_digest(model_bytes)
    evaluation_sha256 = evaluation_digest(model_bytes, state)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"model digest mismatch: {model_sha256}")
    if evaluation_sha256 != EXPECTED_EVALUATION_SHA256:
        raise RuntimeError(f"evaluation digest mismatch: {evaluation_sha256}")

    cases: list[dict[str, Any]] = []
    for case_name, candidate, mechanisms in CANDIDATE_CASES:
        system = System(
            substrate=substrate,
            state=state,
            subset=candidate,
            background_conditioning="CAUSAL_MARGINALIZATION",
        )
        for mechanism in mechanisms:
            primitive_results: dict[str, Any] = {}
            for case_operation, operation_id, method_name in OPERATIONS:
                method = getattr(system, method_name)
                result = method(mechanism, candidate)
                shape, values = _flat_little_endian(result)
                request_sha256 = primitive_request_digest(
                    model_bytes,
                    state,
                    candidate,
                    mechanism,
                    candidate,
                    EXPECTED_FORMALISM,
                    operation_id,
                )
                total = math.fsum(values)
                if case_operation in {"cause_repertoire", "effect_repertoire"}:
                    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError(
                            f"{case_name} mechanism={mechanism} {case_operation} "
                            f"is not normalized: sum={total:.17g}"
                        )
                primitive_results[case_operation] = {
                    "operation_id": operation_id,
                    "request_sha256": request_sha256,
                    "shape": shape,
                    "flatten_order": "Fortran/column-major; first listed purview unit is least-significant bit",
                    "values": values,
                    "sum": total,
                    "vector_sha256": _vector_sha256(values),
                }

            cases.append({
                "case_id": f"{case_name}-mechanism-" + (
                    "empty" if not mechanism else "-".join(map(str, mechanism))
                ),
                "candidate_label": case_name,
                "candidate_system": list(candidate),
                "mechanism": list(mechanism),
                "purview": list(candidate),
                "state": list(state),
                "background_conditioning": "CAUSAL_MARGINALIZATION",
                "primitives": primitive_results,
            })

    expected_case_count = sum(len(mechanisms) for _, _, mechanisms in CANDIDATE_CASES)
    expected_operation_count = expected_case_count * len(OPERATIONS)
    actual_operation_count = sum(len(case["primitives"]) for case in cases)
    if len(cases) != expected_case_count or actual_operation_count != expected_operation_count:
        raise RuntimeError(
            f"primitive corpus is incomplete: cases={len(cases)}/{expected_case_count}, "
            f"operations={actual_operation_count}/{expected_operation_count}"
        )

    artifact = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_primitive_reference_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed_version,
            "python_version": platform.python_version(),
            "formalism": EXPECTED_FORMALISM,
            "pyphi_wheel_sha256": EXPECTED_PYPHI_WHEEL_SHA256,
            "model_sha256": model_sha256,
            "evaluation_sha256": evaluation_sha256,
            "state": list(state),
            "model_input_semantics": "exact canonical binary TPM shared with the Rust Fig. 1A fixture",
            "system_background_semantics": "external units are causally marginalized conditional on the observed whole-substrate state (IIT 4.0 2023 Eq. 4)",
            "vector_semantics": "PyPhi repertoire axes are flattened in Fortran order; candidate unit order is canonical ascending; first listed purview unit is the least-significant output bit",
            "source_commit": "134a8f31b49db512322465ed3ba5da7194d362fd",
        },
        "cases": cases,
        "summary": {
            "expected_case_count": expected_case_count,
            "computed_case_count": len(cases),
            "operations_per_case": len(OPERATIONS),
            "expected_operation_count": expected_operation_count,
            "computed_operation_count": actual_operation_count,
        },
        "notes": [
            "This artifact contains PyPhi primitive repertoire and forward-cause likelihood outputs only.",
            "It does not establish Rust/PyPhi parity until the Rust engine independently reproduces every request and vector.",
            "The proper subsystem uses explicit causal marginalization for the external background.",
            "IIT 4.0 (2026) is a separate formalism and is not evaluated here.",
            "The forward-cause likelihood is unnormalized and is not interchangeable with a cause repertoire.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PYPHI_IIT4_2023_PRIMITIVES.json"),
    )
    args = parser.parse_args()
    try:
        artifact = generate(args.output)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    print(
        f"status={artifact['status']} "
        f"cases={artifact['summary']['computed_case_count']} "
        f"operations={artifact['summary']['computed_operation_count']} "
        f"model_sha256={artifact['provenance']['model_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
