#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate an IIT 4.0 (2026) landscape paired with the frozen 2023 reference.

This measures the effect of the Mayner et al. (2026) intrinsic-information
requirement on exactly the 13 canonical models used in the IIT4-2023 sweep.
The two formalisms remain separately labeled; no 2023 result is relabeled.
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
FORMALISM_2023 = "IIT_4_0_2023"
FORMALISM_2026 = "IIT_4_0_2026"
QUANTITY_ID = "iit4_system_phi_s"
SCHEMA_VERSION = "pyphi-iit4-2026-landscape-reference-v0.1"
STATE = (1, 0, 1)
CANDIDATE = (0, 1, 2)
GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
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
            local_index = sum(state[parent] << slot for slot, parent in enumerate(parents))
            row.append(float(tables[target][local_index]))
        tpm.append(row)
    connectivity = [
        [int(source in PARENTS[target]) for target in range(len(PARENTS))]
        for source in range(len(PARENTS))
    ]
    return tpm, connectivity


def _json_value(value: Any) -> Any:
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
    if hasattr(value, "name"):
        return str(value.name)
    raise TypeError(f"unsupported IIT4-2026 output value: {type(value).__name__}")


def _direction_map(value: Any) -> dict[str, float | None]:
    if not value:
        return {}
    result = {}
    for direction, amount in value.items():
        name = getattr(direction, "name", str(direction))
        converted = _json_value(amount)
        result[str(name).lower()] = None if converted is None else float(converted)
    return result


def _cut_digest(sia: Any) -> tuple[str | None, list[list[int]] | None, str | None]:
    partition = getattr(sia, "partition", None)
    if partition is None:
        return None, None, None
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
    encoded = b"SYMTHAEA-IIT-LANDSCAPE-CUT\0v0.1\0"
    encoded += len(flat).to_bytes(8, "little")
    encoded += bytes(int(value) for value in flat)
    cut_sha = hashlib.sha256(encoded).hexdigest()
    lex_key_hex = partition.lex_key().hex()
    return cut_sha, [list(edge) for edge in removed], lex_key_hex


def generate(output_path: Path, reference_2023_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("pyphi")
    if installed != PINNED_PYPHI_VERSION:
        raise RuntimeError(f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed}")

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

    reference_2023 = json.loads(reference_2023_path.read_text(encoding="utf-8"))
    if reference_2023.get("schema_version") != "pyphi-iit4-2023-landscape-reference-v0.1":
        raise RuntimeError("paired 2023 landscape artifact has the wrong schema")
    if reference_2023.get("status") != "PASS":
        raise RuntimeError("paired 2023 landscape artifact is not PASS")
    if len(reference_2023.get("points", [])) != len(GRID):
        raise RuntimeError("paired 2023 landscape is incomplete")
    reference_by_theta = {float(point["theta"]): point for point in reference_2023["points"]}

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
                model_bytes, STATE, CANDIDATE, FORMALISM_2026, QUANTITY_ID
            ),
        })

    section = pyphi.landscape_section(
        builder,
        state=STATE,
        grid=GRID,
        subset=CANDIDATE,
        formalism=FORMALISM_2026,
        progress=False,
    )
    if len(section.skipped):
        raise RuntimeError(f"IIT4-2026 landscape skipped points: {section.skipped!r}")
    records = section.df.to_dict(orient="records")
    if len(records) != len(GRID) or len(section.sias) != len(GRID):
        raise RuntimeError(
            f"IIT4-2026 landscape incomplete: rows={len(records)} sias={len(section.sias)}"
        )

    points = []
    for expected, raw, sia in zip(expected_inputs, records, section.sias, strict=True):
        theta = float(raw["theta"])
        if not math.isclose(theta, expected["theta"], rel_tol=0.0, abs_tol=1e-14):
            raise RuntimeError("IIT4-2026 landscape grid order/value mismatch")
        reference = reference_by_theta.get(theta)
        if reference is None:
            raise RuntimeError(f"missing IIT4-2023 paired point at theta={theta}")
        for digest_field in ("model_sha256", "evaluation_sha256"):
            if expected[digest_field] != reference[digest_field]:
                raise RuntimeError(f"paired model identity mismatch at theta={theta}: {digest_field}")

        cause = getattr(sia, "cause", None)
        effect = getattr(sia, "effect", None)
        ii_system = getattr(sia, "intrinsic_information", None)
        if ii_system is None:
            raise RuntimeError(f"IIT4-2026 did not provide ii(s) at theta={theta}")
        phi_2026 = float(raw["phi"])
        phi_2023 = float(reference["phi"])
        if phi_2026 > phi_2023 + 1e-10:
            raise RuntimeError(
                f"IIT4-2026 monotonicity violation at theta={theta}: "
                f"phi2026={phi_2026} > phi2023={phi_2023}"
            )
        if phi_2026 > float(ii_system) + 1e-10:
            raise RuntimeError(
                f"IIT4-2026 intrinsic-information cap violated at theta={theta}"
            )

        cut_sha, removed_edges, lex_key_hex = _cut_digest(sia)
        points.append({
            **expected,
            "paired_2023_model_sha256": reference["model_sha256"],
            "paired_2023_evaluation_sha256": reference["evaluation_sha256"],
            "phi_2023": phi_2023,
            "signed_phi_2023": float(reference["signed_phi"]),
            "phi_2026": phi_2026,
            "signed_phi_2026": _json_value(raw.get("signed_phi")),
            "normalized_phi_2026": _json_value(raw.get("normalized_phi")),
            "signed_normalized_phi_2026": _json_value(raw.get("signed_normalized_phi")),
            "intrinsic_information": float(ii_system),
            "cause_phi": _json_value(getattr(cause, "phi", None)),
            "cause_signed_phi": _json_value(getattr(cause, "signed_phi", None)),
            "effect_phi": _json_value(getattr(effect, "phi", None)),
            "effect_signed_phi": _json_value(getattr(effect, "signed_phi", None)),
            "intrinsic_differentiation": _direction_map(
                getattr(sia, "intrinsic_differentiation", None)
            ),
            "cause_state": _json_value(raw.get("cause_state")),
            "effect_state": _json_value(raw.get("effect_state")),
            "partition": _json_value(raw.get("partition")),
            "mip_cut_sha256": cut_sha,
            "mip_cut_removed_edges": removed_edges,
            "mip_lex_key_hex": lex_key_hex,
            "partition_margin": _json_value(raw.get("partition_margin")),
            "cause_state_margin": _json_value(raw.get("cause_state_margin")),
            "effect_state_margin": _json_value(raw.get("effect_state_margin")),
            "effectively_tied": _json_value(raw.get("effectively_tied")),
            "regime": int(raw["regime"]),
            "delta_phi_2023_minus_2026": phi_2023 - phi_2026,
            "intrinsic_cap_binding": phi_2026 < min(
                float(getattr(cause, "phi", phi_2026)),
                float(getattr(effect, "phi", phi_2026)),
            ) - 1e-10,
            "selected_state_matches_2023": (
                _json_value(raw.get("cause_state")) == reference["cause_state"]
                and _json_value(raw.get("effect_state")) == reference["effect_state"]
            ),
            "qualification": "EXACT_SAMPLE_ONLY",
        })

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_iit4_2026_intrinsic_information_landscape_reference_only",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed,
            "python_version": platform.python_version(),
            "formalism": FORMALISM_2026,
            "paired_formalism": FORMALISM_2023,
            "quantity_id": QUANTITY_ID,
            "wheel_sha256": PINNED_PYPHI_WHEEL_SHA256,
            "source_commit": PYPHI_SOURCE_COMMIT,
            "candidate_system": list(CANDIDATE),
            "state": list(STATE),
            "parameter_name": PARAMETER_NAME,
            "grid": list(GRID),
            "system_phi_measure": "INTRINSIC_INFORMATION",
            "equation": "phi_s_2026 = min(phi_c, phi_e, ii(s))",
            "comparison_invariant": "phi_s_2026 <= phi_s_2023 at every identical model/state search",
            "grid_semantics": "13 exact point evaluations; no interpolation or continuous interval certificate",
        },
        "points": points,
        "skipped_points": [],
        "summary": {
            "expected_point_count": len(GRID),
            "computed_point_count": len(points),
            "distinct_model_digest_count": len({p["model_sha256"] for p in points}),
            "all_points_complete": True,
            "intrinsic_cap_binding_count": sum(bool(p["intrinsic_cap_binding"]) for p in points),
            "state_selection_change_count": sum(not p["selected_state_matches_2023"] for p in points),
            "mean_phi_reduction": sum(p["delta_phi_2023_minus_2026"] for p in points) / len(points),
            "max_phi_reduction": max(p["delta_phi_2023_minus_2026"] for p in points),
        },
        "notes": [
            "IIT4-2026 is evaluated as a distinct formalism; this artifact does not relabel IIT4-2023 results.",
            "Each 2026 point is paired to the same canonical TPM and current state as the 2023 reference.",
            "The intrinsic-information cap is applied at system level after partition selection, per the pinned PyPhi 2.0 formalism implementation.",
            "The monotonicity invariant phi_s_2026 <= phi_s_2023 is checked at every matched model point.",
            "The cause/effect state selection may differ between formalisms; it is recorded rather than assumed equal.",
            "All rows are exact point samples; no continuous interval certification is implied.",
            "This artifact does not imply that phi_s is a direct measurement of consciousness.",
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
    parser.add_argument("--reference-2023", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PYPHI_IIT4_2026_LANDSCAPE.json"),
    )
    args = parser.parse_args()
    try:
        artifact = generate(args.output, args.reference_2023)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    print(
        f"status={artifact['status']} points={artifact['summary']['computed_point_count']} "
        f"intrinsic_caps={artifact['summary']['intrinsic_cap_binding_count']} "
        f"state_changes={artifact['summary']['state_selection_change_count']} "
        f"mean_phi_reduction={artifact['summary']['mean_phi_reduction']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
