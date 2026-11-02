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



PINNED_PYPHI_VERSION = "2.0.0rc4"
EXPECTED_EXAMPLE = "pyphi.examples.iit4_2023_fig1a_substrate()"
EXPECTED_FORMALISM = "IIT_4_0_2023"
EXPECTED_PYPHI_WHEEL_SHA256 = "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620"
EXPECTED_STATE = (0, 1, 1)
# Canonical binary TPM captured from the executed PyPhi 2.0.0rc4 Fig. 1A
# reference run. PyPhi's weight->TPM builder is independently checked against
# this matrix, but all cross-language computations use this exact matrix.
EXPECTED_TPM_STATE_BY_NODE = [
    [0.11920292202211757, 0.7685247834990178, 0.16798161486607552],
    [0.026596993576865863, 0.9988874639671398, 0.5],
    [0.973403006423134, 0.40131233988754816, 0.16798161486607552],
    [0.8807970779778823, 0.9945137011005495, 0.5],
    [0.11920292202211757, 0.0054862988994504045, 0.5],
    [0.026596993576865863, 0.5986876601124519, 0.8320183851339245],
    [0.973403006423134, 0.0011125360328603216, 0.5],
    [0.8807970779778823, 0.23147521650098227, 0.8320183851339245],
]
EXPECTED_CONNECTIVITY = [
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
]
EXAMPLE_TPM_ABSOLUTE_TOLERANCE = 1e-14
# Cross-language identity vectors captured from the successful pinned PyPhi
# run and asserted by the Rust Fig. 1A fixture. These make input parity strict:
# a one-ULP TPM change is a different model until both sides explicitly agree.
EXPECTED_MODEL_SHA256 = "6d67ba617b5d2ccbba711cb44f4753286a45ae6ded16fc1a544a77dc8e931ac1"
EXPECTED_EVALUATION_SHA256 = "1442d185bc5d2c23e3c6fc92e94f21d1c492cbe41a451cebd1d51fa36e862744"
EXPECTED_ANALYSIS_REQUEST_SHA256 = {
    "system-phi-s-A": "9b98c27017572e39052d64f69c82369a1d682efa0ad7823359c256faf2cc53eb",
    "system-phi-s-AB": "83e7343053f4747a1cea57749b9a94f46baa2617bf31add6f4a94b8578a2468f",
    "AB-cause-phi": "f4f89b92bda0672e4268a6bf55a2fccf1b6cf1dca634f4bead5c1c4b2a96b1dd",
    "AB-effect-phi": "e7d0791d0c96cdb3dc08c7ccdd98a49da3037cee7abc9e984bb20925acb2e390",
    "AB-structure-Phi": "3337af98aceae37804adb85dbc827d38f89f9b1c1943ffd22cb24ec753f706de",
    "system-phi-s-ABC": "ea88d7dc1612dc72bd51d999cf582199ad76e0fd8fb90dfcd65f4788aa16fbfa",
}
EXPECTED_QUANTITY_IDS = {
    "system-phi-s-A": "iit4_system_phi_s",
    "system-phi-s-AB": "iit4_system_phi_s",
    "AB-cause-phi": "iit4_cause_phi_c",
    "AB-effect-phi": "iit4_effect_phi_e",
    "AB-structure-Phi": "iit4_structure_big_phi",
    "system-phi-s-ABC": "iit4_system_phi_s",
}
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


try:
    from .iit_digest_contract import (
        analysis_request_digest,
        canonical_model_bytes,
        evaluation_digest,
        model_digest,
    )
except ImportError:  # Direct execution: python validation/generate_pyphi_reference_fixtures.py
    from iit_digest_contract import (
        analysis_request_digest,
        canonical_model_bytes,
        evaluation_digest,
        model_digest,
    )


def _model_digest(
    substrate: Any, state: tuple[int, ...], formalism: str
) -> tuple[str, str, str, dict[str, Any], bytes]:
    # PyPhi documents state-by-node TPMs, little-endian rows, and source-to-target
    # connectivity. Reuse the shared binary encoder covered by dependency-free tests.
    tpm_frame = substrate.tpm.to_pandas()
    tpm = _float_matrix(tpm_frame.values.tolist())
    connectivity = [[int(cell) for cell in row] for row in substrate.cm.tolist()]
    row_states = [
        list(map(int, row if isinstance(row, tuple) else (row,)))
        for row in tpm_frame.index.tolist()
    ]
    n_units = len(tpm[0]) if tpm else 0
    expected_row_states = [
        [(state_index >> unit) & 1 for unit in range(n_units)]
        for state_index in range(1 << n_units)
    ]
    if len(state) != n_units or any(bit not in (0, 1) for bit in state):
        raise RuntimeError("evaluation state does not match the binary TPM unit count")
    if row_states != expected_row_states:
        raise RuntimeError(
            "PyPhi TPM row order does not match symthaea-iit's declared little-endian state order"
        )
    model = {
        "formalism": formalism,
        "node_labels": [str(label) for label in substrate.node_labels],
        "state": list(state),
        "state_order": row_states,
        "tpm_state_by_node": tpm,
        "connectivity_matrix": connectivity,
        "time_grain": "one PyPhi discrete update",
    }
    model_bytes = canonical_model_bytes(tpm, connectivity)
    model_sha256 = model_digest(model_bytes)
    evaluation_sha256 = evaluation_digest(model_bytes, state)
    reference_manifest_sha256 = hashlib.sha256(_canonical_json(model)).hexdigest()
    return model_sha256, evaluation_sha256, reference_manifest_sha256, model, model_bytes

def _extract_measurements(
    pyphi_module: Any, substrate: Any, state: tuple[int, ...], formalism: str
) -> dict[str, float]:
    analyses = {
        "system-phi-s-A": pyphi_module.analyze(substrate, state=state, subset=(0,), formalism=formalism),
        "system-phi-s-AB": pyphi_module.analyze(substrate, state=state, subset=(0, 1), formalism=formalism),
        "system-phi-s-ABC": pyphi_module.analyze(substrate, state=state, subset=(0, 1, 2), formalism=formalism),
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
    reference = fixture.get("reference", {})
    if reference.get("exact_version") != PINNED_PYPHI_VERSION:
        raise RuntimeError("fixture and runner version pins disagree")
    if reference.get("api_example") != EXPECTED_EXAMPLE:
        raise RuntimeError("fixture points to an unrecognized PyPhi reference example")
    if reference.get("formalism") != EXPECTED_FORMALISM:
        raise RuntimeError("fixture formalism differs from the runner's reviewed reference profile")
    state = tuple(reference.get("state", ()))
    if state != EXPECTED_STATE:
        raise RuntimeError("fixture evaluation state differs from the reviewed Fig. 1A state")

    quantities = fixture.get("quantities")
    if not isinstance(quantities, list) or len(quantities) != len(EXPECTED_QUANTITY_IDS):
        raise RuntimeError("fixture must contain exactly the six reviewed reference quantities")
    ids = [item.get("id") for item in quantities if isinstance(item, dict)]
    if len(ids) != len(quantities) or len(set(ids)) != len(ids):
        raise RuntimeError("fixture quantity IDs are missing, malformed, or duplicated")
    if set(ids) != set(EXPECTED_QUANTITY_IDS):
        raise RuntimeError("fixture quantity set differs from the six reviewed reference quantities")
    for item in quantities:
        if item.get("quantity_id") != EXPECTED_QUANTITY_IDS[item["id"]]:
            raise RuntimeError(f"quantity id mismatch for {item['id']}")
        tolerance = item.get("absolute_tolerance")
        expected = item.get("expected")
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise RuntimeError(f"invalid absolute tolerance for {item['id']}")
        if not isinstance(expected, (int, float)) or not math.isfinite(expected):
            raise RuntimeError(f"invalid expected anchor for {item['id']}")

    try:
        import pyphi
    except ImportError as exc:
        raise RuntimeError("PyPhi is unavailable; install the pinned pyphi==2.0.0rc4 artifact") from exc
    formalism = EXPECTED_FORMALISM
    pyphi.config.progress_bars = False

    # Check the published upstream builder against a canonical, reviewed TPM.
    # The builder can differ by an ULP across floating-point execution paths;
    # that is recorded and bounded here, not allowed to alter the differential
    # input used by Rust and PyPhi.
    import numpy as np
    from pyphi.substrate import Substrate

    generated_substrate = pyphi.examples.iit4_2023_fig1a_substrate()
    generated_tpm = _float_matrix(
        generated_substrate.tpm.to_pandas().values.tolist()
    )
    generated_connectivity = [
        [int(cell) for cell in row]
        for row in generated_substrate.cm.tolist()
    ]
    if generated_connectivity != EXPECTED_CONNECTIVITY:
        raise RuntimeError(
            "upstream Fig. 1A builder connectivity differs from the reviewed canonical model"
        )
    if len(generated_tpm) != len(EXPECTED_TPM_STATE_BY_NODE) or any(
        len(actual_row) != len(expected_row)
        or any(
            abs(actual - expected) > EXAMPLE_TPM_ABSOLUTE_TOLERANCE
            for actual, expected in zip(actual_row, expected_row, strict=True)
        )
        for actual_row, expected_row in zip(
            generated_tpm, EXPECTED_TPM_STATE_BY_NODE, strict=True
        )
    ):
        raise RuntimeError(
            "upstream Fig. 1A builder TPM differs from the reviewed canonical model "
            f"by more than {EXAMPLE_TPM_ABSOLUTE_TOLERANCE}"
        )
    max_example_tpm_deviation = max(
        abs(actual - expected)
        for actual_row, expected_row in zip(
            generated_tpm, EXPECTED_TPM_STATE_BY_NODE, strict=True
        )
        for actual, expected in zip(actual_row, expected_row, strict=True)
    )

    # All reference calculations and shared model/request digests use this exact
    # TPM matrix, not a freshly recomputed floating-point approximation.
    substrate = Substrate(
        np.asarray(EXPECTED_TPM_STATE_BY_NODE, dtype=float),
        cm=np.asarray(EXPECTED_CONNECTIVITY, dtype=int),
        node_labels=generated_substrate.node_labels,
    )
    model_sha256, evaluation_sha256, reference_manifest_sha256, canonical_model, model_bytes = _model_digest(
        substrate, state, formalism
    )
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"PyPhi model digest differs from the shared Rust fixture: "
            f"expected {EXPECTED_MODEL_SHA256}, found {model_sha256}"
        )
    if evaluation_sha256 != EXPECTED_EVALUATION_SHA256:
        raise RuntimeError(
            f"PyPhi evaluation digest differs from the shared Rust fixture: "
            f"expected {EXPECTED_EVALUATION_SHA256}, found {evaluation_sha256}"
        )
    measured = _extract_measurements(pyphi, substrate, state, formalism)

    comparisons = []
    for expected in fixture["quantities"]:
        actual = measured[expected["id"]]
        error = abs(actual - float(expected["expected"]))
        tolerance = float(expected["absolute_tolerance"])
        comparisons.append({
            "id": expected["id"],
            "quantity": expected["kind"],
            "quantity_id": expected["quantity_id"],
            "analysis_request_sha256": analysis_request_digest(
                model_bytes, state, expected["subset"], formalism, expected["quantity_id"]
            ),
            "subset": expected["subset"],
            "expected_anchor": expected["expected"],
            "actual": actual,
            "absolute_error_to_documented_anchor": error,
            "absolute_tolerance": tolerance,
            "matches_anchor": error <= tolerance,
        })

    for comparison in comparisons:
        expected_request_sha256 = EXPECTED_ANALYSIS_REQUEST_SHA256[comparison["id"]]
        if comparison["analysis_request_sha256"] != expected_request_sha256:
            raise RuntimeError(
                f"analysis request digest differs for {comparison['id']}: "
                f"expected {expected_request_sha256}, "
                f"found {comparison['analysis_request_sha256']}"
            )

    passed = all(item["matches_anchor"] for item in comparisons)
    artifact = {
        "schema_version": "pyphi-reference-run-v0.1",
        "status": "PASS" if passed else "MISMATCH",
        "qualification_scope": "upstream_reference_smoke_test_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed_version,
            "pyphi_wheel_sha256": EXPECTED_PYPHI_WHEEL_SHA256,
            "python_version": platform.python_version(),
            "formalism": formalism,
            "upstream_example": reference["api_example"],
            "state": list(state),
            "upstream_builder_tpm_max_abs_deviation": max_example_tpm_deviation,
            "upstream_builder_tpm_tolerance": EXAMPLE_TPM_ABSOLUTE_TOLERANCE,
            "model_input_semantics": "exact canonical TPM shared with the Rust Fig. 1A fixture; the upstream builder is separately checked within the recorded absolute tolerance",
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
