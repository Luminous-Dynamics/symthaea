#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate a pinned PyPhi pointwise GID-operator reference corpus.

This is deliberately an operator-level fixture set. It does not construct a
causal partition or qualify a complete IIT system analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

PINNED_PYPHI_VERSION = "2.0.0rc4"
PINNED_PYPHI_WHEEL_SHA256 = (
    "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620"
)
PYPHI_SOURCE_COMMIT = "134a8f31b49db512322465ed3ba5da7194d362fd"
FORMALISM = "IIT_4_0_2023"
OPERATION_ID = "pyphi.measures.distribution.generalized_intrinsic_difference"
SCHEMA_VERSION = "pyphi-iit4-2023-gid-operator-v0.1"

# The arrays are normalized where required by the public Rust adapter:
# forward and partitioned are likelihood vectors; selectivity is a distribution.
CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "mixed-sign-ratios",
        "forward": [0.5, 0.25, 0.125, 0.125],
        "partitioned_forward": [0.25, 0.5, 0.25, 0.25],
        "selectivity": [0.4, 0.3, 0.2, 0.1],
        "expected_signs": ["positive", "negative", "negative", "negative"],
    },
    {
        "case_id": "one-sided-and-both-zero-support",
        "forward": [0.5, 0.0, 0.25, 0.0],
        "partitioned_forward": [0.25, 0.5, 0.0, 0.0],
        "selectivity": [0.4, 0.3, 0.2, 0.1],
        "expected_signs": ["positive", "zero", "zero", "zero"],
    },
    {
        "case_id": "equal-repertoires",
        "forward": [0.25, 0.5, 0.25],
        "partitioned_forward": [0.25, 0.5, 0.25],
        "selectivity": [0.2, 0.5, 0.3],
        "expected_signs": ["zero", "zero", "zero"],
    },
    {
        "case_id": "rare-but-representable",
        "forward": [1e-200, 2e-200, 4e-200],
        "partitioned_forward": [2e-200, 1e-200, 2e-200],
        "selectivity": [0.2, 0.3, 0.5],
        "expected_signs": ["negative", "positive", "positive"],
    },
    {
        "case_id": "nonuniform-weighting",
        "forward": [0.8, 0.1, 0.05, 0.05],
        "partitioned_forward": [0.4, 0.2, 0.1, 0.1],
        "selectivity": [0.1, 0.2, 0.3, 0.4],
        "expected_signs": ["positive", "negative", "negative", "negative"],
    },
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("pyphi")
    if installed != PINNED_PYPHI_VERSION:
        raise RuntimeError(
            f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed}"
        )

    import numpy as np
    from pyphi.measures.distribution import generalized_intrinsic_difference

    cases: list[dict[str, Any]] = []
    total_states = 0
    for definition in CASES:
        forward = np.asarray(definition["forward"], dtype=float)
        partitioned = np.asarray(definition["partitioned_forward"], dtype=float)
        selectivity = np.asarray(definition["selectivity"], dtype=float)
        if not (forward.ndim == partitioned.ndim == selectivity.ndim == 1):
            raise RuntimeError(f"{definition['case_id']}: inputs must be 1-D")
        if not (len(forward) == len(partitioned) == len(selectivity)):
            raise RuntimeError(f"{definition['case_id']}: input lengths differ")
        for name, vector in (
            ("forward", forward),
            ("partitioned_forward", partitioned),
            ("selectivity", selectivity),
        ):
            if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
                raise RuntimeError(f"{definition['case_id']}: invalid {name} values")
        if not np.isclose(float(np.sum(selectivity)), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{definition['case_id']}: selectivity is not normalized")

        result = np.asarray(
            generalized_intrinsic_difference(
                forward,
                partitioned,
                selectivity,
                state=None,
            ),
            dtype=float,
        ).reshape(-1)
        if len(result) != len(forward) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"{definition['case_id']}: invalid GID output")

        state_values = [
            float(
                generalized_intrinsic_difference(
                    forward,
                    partitioned,
                    selectivity,
                    state=index,
                )
            )
            for index in range(len(forward))
        ]
        if not np.allclose(result, np.asarray(state_values), rtol=0.0, atol=1e-14):
            raise RuntimeError(f"{definition['case_id']}: state-selected GID differs from vector")

        input_payload = {
            "forward": definition["forward"],
            "partitioned_forward": definition["partitioned_forward"],
            "selectivity": definition["selectivity"],
        }
        request_payload = {
            "formalism": FORMALISM,
            "operation_id": OPERATION_ID,
            "case_id": definition["case_id"],
            "inputs": input_payload,
        }
        cases.append({
            "case_id": definition["case_id"],
            "formalism": FORMALISM,
            "operation_id": OPERATION_ID,
            "input_sha256": sha256(canonical_json(input_payload)),
            "request_sha256": sha256(canonical_json(request_payload)),
            **input_payload,
            "gid_vector": result.tolist(),
            "state_values": state_values,
            "expected_signs": definition["expected_signs"],
        })
        total_states += len(result)

    if len(cases) != len(CASES):
        raise RuntimeError(f"incomplete GID corpus: {len(cases)}/{len(CASES)}")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_gid_operator_reference_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed,
            "python_version": platform.python_version(),
            "formalism": FORMALISM,
            "wheel_sha256": PINNED_PYPHI_WHEEL_SHA256,
            "source_commit": PYPHI_SOURCE_COMMIT,
            "operation_id": OPERATION_ID,
            "implementation_semantics": (
                "PyPhi pointwise GID = selectivity * nan_to_num(log2(forward / "
                "partitioned), nan=0, posinf=0, neginf=0)"
            ),
        },
        "cases": cases,
        "summary": {
            "expected_case_count": len(CASES),
            "computed_case_count": len(cases),
            "expected_state_count": total_states,
            "computed_state_count": total_states,
            "all_state_queries_match_vector": True,
        },
        "notes": [
            "Operator-level reference only: no causal partition or system search is performed.",
            "The corpus tests PyPhi's float64 GID operator on the supplied vectors.",
            "Rare-but-representable inputs test ratios whose input values are small but still nonzero in float64.",
            "Finite mathematical log likelihoods whose exponentiated probability views both underflow require a separate log-domain analytic oracle; they are not represented as PyPhi parity cases here.",
            "IIT 4.0 (2026) intrinsic-information semantics are not evaluated.",
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
        default=Path("PYPHI_IIT4_2023_GID_OPERATOR.json"),
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
        f"states={artifact['summary']['computed_state_count']} "
        f"pyphi={artifact['provenance']['version']} "
        f"wheel_sha256={artifact['provenance']['wheel_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
