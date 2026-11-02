#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare IIT4-2023 and IIT4-2026 on PyPhi's published positive-phi Fig. 1A.

The key experiment fixes the exact cause/effect system state selected under
IIT4-2023, then recomputes the system MIP under both formalisms. This isolates
the 2026 intrinsic-information requirement from cross-formalism state-pair
selection. A second pair of full SIA searches records any state-selection
changes.
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
PINNED_WHEEL_SHA256 = (
    "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620"
)
PYPHI_SOURCE_COMMIT = "134a8f31b49db512322465ed3ba5da7194d362fd"
FORMALISM_2023 = "IIT_4_0_2023"
FORMALISM_2026 = "IIT_4_0_2026"
QUANTITY_ID = "iit4_system_phi_s"
SCHEMA_VERSION = "pyphi-iit4-dual-formalism-positive-anchor-v0.1"


def _finite(value: Any, context: str) -> float:
    if value is None:
        raise RuntimeError(f"{context}: missing numeric value")
    if hasattr(value, "item"):
        value = value.item()
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{context}: non-finite value {number}")
    return number


def _state_vector(value: Any) -> list[int]:
    if value is None:
        raise RuntimeError("selected system state is missing")
    if hasattr(value, "state"):
        value = value.state
    return [int(bit) for bit in value]


def _cut_identity(sia: Any, candidate: list[int]) -> tuple[str, list[list[int]], str]:
    partition = sia.partition
    removed = sorted(
        (int(source), int(target))
        for source, target in partition.removed_edges()
    )
    removed_set = set(removed)
    flat = [
        (source, target) in removed_set
        for source in candidate
        for target in candidate
    ]
    encoded = b"SYMTHAEA-IIT-LANDSCAPE-CUT\\0v0.1\\0"
    encoded += len(flat).to_bytes(8, "little")
    encoded += bytes(int(value) for value in flat)
    return (
        hashlib.sha256(encoded).hexdigest(),
        [list(edge) for edge in removed],
        partition.lex_key().hex(),
    )


def _direction_map(values: Any) -> dict[str, float]:
    if not values:
        return {}
    result = {}
    for direction, value in values.items():
        name = getattr(direction, "name", str(direction)).lower()
        result[name] = _finite(value, f"intrinsic_differentiation.{name}")
    return result


def _sia_summary(sia: Any, candidate: list[int]) -> dict[str, Any]:
    cut_sha, removed, lex_key = _cut_identity(sia, candidate)
    state = sia.system_state
    return {
        "phi_s": _finite(sia.phi, "phi_s"),
        "signed_phi_s": _finite(sia.signed_phi, "signed_phi_s"),
        "normalized_phi_s": _finite(sia.normalized_phi, "normalized_phi_s"),
        "signed_normalized_phi_s": _finite(sia.signed_normalized_phi, "signed_normalized_phi_s"),
        "cause_phi": _finite(sia.cause.phi, "cause_phi"),
        "cause_signed_phi": _finite(sia.cause.signed_phi, "cause_signed_phi"),
        "effect_phi": _finite(sia.effect.phi, "effect_phi"),
        "effect_signed_phi": _finite(sia.effect.signed_phi, "effect_signed_phi"),
        "intrinsic_information": (
            None if sia.intrinsic_information is None
            else _finite(sia.intrinsic_information, "intrinsic_information")
        ),
        "cause_intrinsic_specification": _finite(
            state.cause.intrinsic_specification,
            "cause_intrinsic_specification",
        ),
        "effect_intrinsic_specification": _finite(
            state.effect.intrinsic_specification,
            "effect_intrinsic_specification",
        ),
        "intrinsic_differentiation": _direction_map(sia.intrinsic_differentiation),
        "cause_state": _state_vector(state.cause.state),
        "effect_state": _state_vector(state.effect.state),
        "mip_cut_sha256": cut_sha,
        "mip_cut_removed_edges": removed,
        "mip_lex_key_hex": lex_key,
        "partition_margin": None if sia.partition_margin is None else _finite(sia.partition_margin, "partition_margin"),
        "cause_state_margin": None if state.cause.state_margin is None else _finite(state.cause.state_margin, "cause_state_margin"),
        "effect_state_margin": None if state.effect.state_margin is None else _finite(state.effect.state_margin, "effect_state_margin"),
        "effectively_tied": bool(sia.effectively_tied),
        "ties_count": len(sia.ties),
    }


def generate(anchor_path: Path, output_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("pyphi")
    if installed != PINNED_PYPHI_VERSION:
        raise RuntimeError(f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed}")
    import numpy as np
    import pyphi
    from pyphi.substrate import Substrate
    from pyphi.system import System

    pyphi.config.progress_bars = False
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if anchor.get("status") != "PASS":
        raise RuntimeError("published Fig. 1A anchor artifact is not PASS")
    canonical = anchor["canonical_model"]
    if canonical["formalism"] != FORMALISM_2023:
        raise RuntimeError("Fig. 1A source anchor must be IIT_4_0_2023")
    tpm = np.asarray(canonical["tpm_state_by_node"], dtype=float)
    cm = np.asarray(canonical["connectivity_matrix"], dtype=int)
    state = tuple(int(bit) for bit in canonical["state"])
    candidate = [0, 1, 2]
    if tpm.shape != (8, 3) or cm.shape != (3, 3):
        raise RuntimeError(f"unexpected Fig. 1A shape: TPM={tpm.shape}, CM={cm.shape}")

    def builder(_theta: float):
        return Substrate(
            tpm.copy(),
            cm=cm.copy(),
            node_labels=tuple(canonical["node_labels"]),
        )

    reference_bytes = canonical_model_bytes(
        canonical["tpm_state_by_node"],
        canonical["connectivity_matrix"],
    )
    canonical_model_sha256 = model_digest(reference_bytes)
    canonical_evaluation_sha256 = evaluation_digest(reference_bytes, state)
    request_2023 = analysis_request_digest(
        reference_bytes, state, candidate, FORMALISM_2023, QUANTITY_ID
    )
    request_2026 = analysis_request_digest(
        reference_bytes, state, candidate, FORMALISM_2026, QUANTITY_ID
    )

    landscape_2023 = pyphi.landscape_section(
        builder, state=state, grid=(0.0,), subset=candidate,
        formalism=FORMALISM_2023, progress=False,
    )
    landscape_2026 = pyphi.landscape_section(
        builder, state=state, grid=(0.0,), subset=candidate,
        formalism=FORMALISM_2026, progress=False,
    )
    if len(landscape_2023.sias) != 1 or len(landscape_2026.sias) != 1:
        raise RuntimeError("positive anchor full-system SIA did not return exactly one point")
    full_2023 = landscape_2023.sias[0]
    full_2026 = landscape_2026.sias[0]

    system = System(builder(0.0), state, subset=candidate)
    # Hold the 2023-selected cause/effect system state fixed. This isolates
    # the effect of Eq. 23 from the separate problem of selecting among
    # alternative cause/effect state pairs.
    fixed_state = full_2023.system_state
    fixed_2023 = system.sia(
        system_state=fixed_state,
        formalism=FORMALISM_2023,
        parallel=False,
        progress=False,
    )
    fixed_2026 = system.sia(
        system_state=fixed_state,
        formalism=FORMALISM_2026,
        parallel=False,
        progress=False,
    )

    full_summary_2023 = _sia_summary(full_2023, candidate)
    full_summary_2026 = _sia_summary(full_2026, candidate)
    fixed_summary_2023 = _sia_summary(fixed_2023, candidate)
    fixed_summary_2026 = _sia_summary(fixed_2026, candidate)

    if full_summary_2023["phi_s"] <= 0.0:
        raise RuntimeError("published Fig. 1A whole-system 2023 phi must be positive")
    if fixed_summary_2026["intrinsic_information"] is None:
        raise RuntimeError("IIT4-2026 did not return ii(s) for fixed selected state")
    tolerance = 1e-10
    if fixed_summary_2026["phi_s"] > fixed_summary_2023["phi_s"] + tolerance:
        raise RuntimeError("fixed-state IIT4-2026 phi exceeds IIT4-2023 phi")
    if fixed_summary_2026["phi_s"] > fixed_summary_2026["intrinsic_information"] + tolerance:
        raise RuntimeError("fixed-state IIT4-2026 phi exceeds ii(s)")
    if fixed_summary_2023["mip_cut_sha256"] != fixed_summary_2026["mip_cut_sha256"]:
        raise RuntimeError(
            "fixed-state MIP changed between IIT4-2023 and IIT4-2026; "
            "the 2026 requirement must be applied after MIP selection"
        )
    if full_summary_2026["phi_s"] > full_summary_2023["phi_s"] + tolerance:
        raise RuntimeError("full-search IIT4-2026 phi exceeds IIT4-2023 phi")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_positive_phi_fixed_state_dual_formalism_conformance",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed,
            "python_version": platform.python_version(),
            "source_commit": PYPHI_SOURCE_COMMIT,
            "wheel_sha256": PINNED_WHEEL_SHA256,
            "source_anchor": "PyPhi IIT 4.0 (2023) Fig. 1A",
            "source_anchor_schema": anchor["schema_version"],
            "canonical_model_sha256": canonical_model_sha256,
            "canonical_evaluation_sha256": canonical_evaluation_sha256,
            "analysis_request_sha256_iit4_2023": request_2023,
            "analysis_request_sha256_iit4_2026": request_2026,
            "current_state": list(state),
            "candidate_system": candidate,
            "quantity_id": QUANTITY_ID,
            "fixed_state_semantics": "cause/effect system state selected by the full IIT4-2023 SIA, then held unchanged under both formalisms",
        },
        "full_search": {
            "formalism_2023": FORMALISM_2023,
            "formalism_2026": FORMALISM_2026,
            "iit4_2023": full_summary_2023,
            "iit4_2026": full_summary_2026,
            "state_selection_changed": (
                full_summary_2023["cause_state"] != full_summary_2026["cause_state"]
                or full_summary_2023["effect_state"] != full_summary_2026["effect_state"]
            ),
            "phi_reduction_2023_minus_2026": (
                full_summary_2023["phi_s"] - full_summary_2026["phi_s"]
            ),
        },
        "fixed_state_comparison": {
            "same_specified_state": True,
            "formalism_2023": FORMALISM_2023,
            "formalism_2026": FORMALISM_2026,
            "iit4_2023": fixed_summary_2023,
            "iit4_2026": fixed_summary_2026,
            "mip_unchanged": (
                fixed_summary_2023["mip_cut_sha256"] == fixed_summary_2026["mip_cut_sha256"]
            ),
            "intrinsic_information_cap_binding": (
                fixed_summary_2026["intrinsic_information"]
                < min(fixed_summary_2026["cause_phi"], fixed_summary_2026["effect_phi"]) - tolerance
            ),
            "phi_reduction_2023_minus_2026": (
                fixed_summary_2023["phi_s"] - fixed_summary_2026["phi_s"]
            ),
        },
        "invariants": {
            "canonical_model_identical": True,
            "fixed_state_identical": True,
            "phi_2026_not_above_phi_2023": True,
            "phi_2026_not_above_intrinsic_information": True,
            "fixed_state_mip_unchanged": True,
            "positive_2023_anchor": True,
        },
        "notes": [
            "The fixed-state comparison isolates the 2026 intrinsic-information cap from cross-formalism cause/effect state selection.",
            "The same canonical TPM and current state are used for both formalisms.",
            "PyPhi 2.0.0rc4 applies the IIT4-2026 intrinsic-information requirement after fixed-state MIP selection.",
            "This is a formalism-conformance test, not evidence that phi_s measures phenomenal consciousness.",
            "The result is limited to one published positive-phi anchor; generality requires a broader model corpus.",
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
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PYPHI_IIT4_DUAL_FORMALISM_POSITIVE_ANCHOR.json"),
    )
    args = parser.parse_args()
    try:
        result = generate(args.anchor, args.output)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    fixed = result["fixed_state_comparison"]
    print(
        f"status={result['status']} "
        f"phi23={fixed['iit4_2023']['phi_s']:.12g} "
        f"phi26={fixed['iit4_2026']['phi_s']:.12g} "
        f"ii={fixed['iit4_2026']['intrinsic_information']:.12g} "
        f"cap_binds={fixed['intrinsic_information_cap_binding']} "
        f"mip_unchanged={fixed['mip_unchanged']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
