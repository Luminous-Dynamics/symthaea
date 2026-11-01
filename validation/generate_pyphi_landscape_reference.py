#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the pinned PyPhi reference for the Rust sparse IIT4 landscape sweep.

The sweep uses the same canonical dense TPM wire contract, 3-unit model,
candidate, state, parameter values and IIT 4.0 (2023) profile as the Rust
example. Every point is one full PyPhi SIA. Regime boundaries are grid
brackets only; they are not certified continuous intervals.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import struct
import sys
from pathlib import Path
from typing import Any

from iit_digest_contract import (
    analysis_request_digest,
    canonical_model_bytes,
    evaluation_digest,
    model_digest,
)

PINNED_PYPHI_VERSION = "2.0.0rc4"
PINNED_PYPHI_WHEEL_SHA256 = (
    "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620"
)
PYPHI_SOURCE_COMMIT = "134a8f31b49db512322465ed3ba5da7194d362fd"
FORMALISM = "IIT_4_0_2023"
QUANTITY_ID = "iit4_system_phi_s"
SCHEMA_VERSION = "pyphi-iit4-2023-landscape-reference-v0.1"
STATE = (1, 0, 1)
CANDIDATE = (0, 1, 2)
GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

# Same parent scopes and local CPT row order as
# crates/core/symthaea-iit/examples/sparse_phi_landscape_sweep.rs.
PARENTS = ((0, 2), (0, 1), (1, 2))
BASE_TABLES = (
    (0.12, 0.38, 0.57, None),
    (0.18, 0.58, 0.31, 0.79),
    (0.21, 0.43, 0.62, 0.83),
)
PARAMETER_NAME = "target0_parent_state_3_probability"


def _model_inputs(theta: float) -> tuple[list[list[float]], list[list[int]]]:
    tables = [list(row) for row in BASE_TABLES]
    tables[0][3] = float(theta)
    tpm: list[list[float]] = []
    for state_index in range(1 << len(PARENTS)):
        state = [(state_index >> unit) & 1 for unit in range(len(PARENTS))]
        row: list[float] = []
        for target, parents in enumerate(PARENTS):
            local_index = sum(
                state[parent] << slot for slot, parent in enumerate(parents)
            )
            row.append(float(tables[target][local_index]))
        tpm.append(row)

    connectivity = [
        [int(source in PARENTS[target]) for target in range(len(PARENTS))]
        for source in range(len(PARENTS))
    ]
    return tpm, connectivity


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas scalars to strict JSON, mapping undefined to null."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported landscape output value: {type(value).__name__}")


def _mip_cut_identity(sia: Any) -> tuple[str | None, list[list[int]] | None]:
    partition = getattr(sia, "partition", None)
    if partition is None:
        return None, None
    removed = sorted(
        (int(source), int(target))
        for source, target in partition.removed_edges()
    )
    removed_set = set(removed)
    flat = [
        (source, target) in removed_set
        for source in CANDIDATE
        for target in CANDIDATE
    ]
    encoded = b"SYMTHAEA-IIT-LANDSCAPE-CUT\\0v0.1\\0"
    encoded += struct.pack("<Q", len(flat))
    encoded += bytes(int(value) for value in flat)
    import hashlib
    return hashlib.sha256(encoded).hexdigest(), [list(edge) for edge in removed]


def generate(output_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("pyphi")
    if installed != PINNED_PYPHI_VERSION:
        raise RuntimeError(
            f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed}"
        )

    import numpy as np
    import pyphi
    from pyphi.substrate import Substrate

    pyphi.config.progress_bars = False

    def builder(theta: float):
        tpm, connectivity = _model_inputs(float(theta))
        return Substrate(
            np.asarray(tpm, dtype=float),
            cm=np.asarray(connectivity, dtype=int),
            node_labels=("A", "B", "C"),
        )

    # Freeze and validate every exact model input independently of the
    # landscape analysis. The common digests allow byte-identity comparison
    # with the dense representation emitted by the Rust sweep.
    expected_inputs = []
    for point_index, theta in enumerate(GRID):
        tpm, connectivity = _model_inputs(theta)
        model_bytes = canonical_model_bytes(tpm, connectivity)
        expected_inputs.append({
            "point_index": point_index,
            "theta": theta,
            "tpm": tpm,
            "connectivity": connectivity,
            "model_sha256": model_digest(model_bytes),
            "evaluation_sha256": evaluation_digest(model_bytes, STATE),
            "analysis_request_sha256": analysis_request_digest(
                model_bytes, STATE, CANDIDATE, FORMALISM, QUANTITY_ID
            ),
        })

    section = pyphi.landscape_section(
        builder,
        state=STATE,
        grid=GRID,
        subset=CANDIDATE,
        formalism=FORMALISM,
        progress=False,
    )

    if len(section.skipped):
        raise RuntimeError(
            "one or more landscape grid points were dynamically unreachable: "
            + repr(section.skipped)
        )
    records = section.df.to_dict(orient="records")
    if len(records) != len(GRID):
        raise RuntimeError(f"landscape incomplete: {len(records)}/{len(GRID)}")

    points: list[dict[str, Any]] = []
    for expected, raw, sia in zip(expected_inputs, records, section.sias, strict=True):
        theta = float(raw["theta"])
        if not math.isclose(theta, expected["theta"], rel_tol=0.0, abs_tol=1e-14):
            raise RuntimeError(
                f"landscape output order/value mismatch: {theta} != {expected['theta']}"
            )
        mip_cut_sha256, mip_cut_removed_edges = _mip_cut_identity(sia)
        point = {
            **expected,
            "mip_cut_sha256": mip_cut_sha256,
            "mip_cut_removed_edges": mip_cut_removed_edges,
            "phi": _json_value(raw.get("phi")),
            "signed_phi": _json_value(raw.get("signed_phi")),
            "normalized_phi": _json_value(raw.get("normalized_phi")),
            "signed_normalized_phi": _json_value(raw.get("signed_normalized_phi")),
            "partition": _json_value(raw.get("partition")),
            "cause_state": _json_value(raw.get("cause_state")),
            "effect_state": _json_value(raw.get("effect_state")),
            "partition_margin": _json_value(raw.get("partition_margin")),
            "cause_state_margin": _json_value(raw.get("cause_state_margin")),
            "effect_state_margin": _json_value(raw.get("effect_state_margin")),
            "effectively_tied": _json_value(raw.get("effectively_tied")),
            "regime": int(raw["regime"]),
            "qualification": "EXACT_SAMPLE_ONLY",
        }
        points.append(point)

    boundaries = [
        {"left_theta": float(left), "right_theta": float(right),
         "qualification": "GRID_RESOLUTION_SWITCH_BRACKET_NOT_CERTIFIED"}
        for left, right in section.boundaries
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_iit4_2023_parameter_landscape_reference_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed,
            "python_version": platform.python_version(),
            "formalism": FORMALISM,
            "quantity_id": QUANTITY_ID,
            "wheel_sha256": PINNED_PYPHI_WHEEL_SHA256,
            "source_commit": PYPHI_SOURCE_COMMIT,
            "model_representation": "canonical dense binary TPM; state-by-unit; little-endian state index",
            "candidate_system": list(CANDIDATE),
            "state": list(STATE),
            "parameter_name": PARAMETER_NAME,
            "grid": list(GRID),
            "grid_semantics": "13 exact point evaluations; no interpolation or continuous interval certificate",
        },
        "points": points,
        "grid_resolution_boundaries": boundaries,
        "skipped_points": [],
        "summary": {
            "expected_point_count": len(GRID),
            "computed_point_count": len(points),
            "distinct_model_digest_count": len({p["model_sha256"] for p in points}),
            "distinct_request_digest_count": len({p["analysis_request_sha256"] for p in points}),
            "regime_count": len({p["regime"] for p in points}),
            "boundary_count": len(boundaries),
            "all_points_complete": True,
        },
        "notes": [
            "Every row is one exact PyPhi SIA sample at one parameter value; no interpolation is performed.",
            "Regime boundary entries are adjacent grid brackets where selection identity changes, not certified switch intervals.",
            "The oracle and Rust sweep bind identical canonical dense TPM, evaluation, candidate, state, formalism, quantity and parameter values.",
            "PyPhi and Rust may use distinct native sparse/dense representation identities; the canonical dense digests identify the shared mathematical input.",
            "This artifact does not claim cross-engine parity until a separate verifier compares every common field and selection.",
            "IIT 4.0 (2026) is not evaluated.",
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
        default=Path("PYPHI_IIT4_2023_LANDSCAPE.json"),
    )
    args = parser.parse_args()
    try:
        artifact = generate(args.output)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    print(
        f"status={artifact['status']} "
        f"points={artifact['summary']['computed_point_count']} "
        f"regimes={artifact['summary']['regime_count']} "
        f"boundaries={artifact['summary']['boundary_count']} "
        f"wheel_sha256={artifact['provenance']['wheel_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
