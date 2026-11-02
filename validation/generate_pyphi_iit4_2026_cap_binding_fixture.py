#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Search a deterministic small-system family for an IIT4-2026 cap-binding case.

The search is a reproducible fixture-discovery step, not an estimate of how
frequently the cap binds in nature. Candidates are generated with Python's
stable random.Random stream, evaluated under the pinned PyPhi wheel, and the
first candidate satisfying strict fixed-state conditions is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
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
SCHEMA_VERSION = "pyphi-iit4-2026-cap-binding-positive-control-v0.1"
STATE = (0, 1, 1)
CANDIDATE = (0, 1, 2)
MAX_SEEDS = 128
LOW = 0.025
HIGH = 0.975
TOLERANCE = 1e-10


def _cut_identity(sia: Any) -> tuple[str, list[list[int]], str]:
    partition = sia.partition
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
    return (
        hashlib.sha256(encoded).hexdigest(),
        [list(edge) for edge in removed],
        partition.lex_key().hex(),
    )


def _finite(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite: {number}")
    return number


def _state(value: Any) -> list[int]:
    if hasattr(value, "state"):
        value = value.state
    return [int(bit) for bit in value]


def _summary(sia: Any) -> dict[str, Any]:
    cut_sha, removed, lex_key = _cut_identity(sia)
    ii_value = getattr(sia, "intrinsic_information", None)
    return {
        "phi_s": _finite(sia.phi, "phi_s"),
        "signed_phi_s": _finite(sia.signed_phi, "signed_phi_s"),
        "normalized_phi_s": _finite(sia.normalized_phi, "normalized_phi_s"),
        "cause_phi": _finite(sia.cause.phi, "cause_phi"),
        "cause_signed_phi": _finite(sia.cause.signed_phi, "cause_signed_phi"),
        "effect_phi": _finite(sia.effect.phi, "effect_phi"),
        "effect_signed_phi": _finite(sia.effect.signed_phi, "effect_signed_phi"),
        "intrinsic_information": None if ii_value is None else _finite(ii_value, "ii(s)"),
        "cause_state": _state(sia.system_state.cause.state),
        "effect_state": _state(sia.system_state.effect.state),
        "mip_cut_sha256": cut_sha,
        "mip_cut_removed_edges": removed,
        "mip_lex_key_hex": lex_key,
        "partition_margin": None if sia.partition_margin is None else _finite(sia.partition_margin, "partition_margin"),
        "ties_count": len(sia.ties),
        "effectively_tied": bool(sia.effectively_tied),
    }


def _candidate_tpm(seed: int) -> list[list[float]]:
    stream = random.Random(seed)
    return [
        [stream.uniform(LOW, HIGH) for _ in range(3)]
        for _ in range(8)
    ]


def generate(output_path: Path, max_seeds: int) -> dict[str, Any]:
    installed = importlib.metadata.version("pyphi")
    if installed != PINNED_PYPHI_VERSION:
        raise RuntimeError(f"expected pyphi=={PINNED_PYPHI_VERSION}, found {installed}")

    import numpy as np
    import pyphi
    from pyphi.substrate import Substrate
    from pyphi.system import System

    pyphi.config.progress_bars = False
    connectivity = np.ones((3, 3), dtype=int)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for seed in range(max_seeds):
        tpm = _candidate_tpm(seed)
        substrate = Substrate(
            np.asarray(tpm, dtype=float),
            cm=connectivity.copy(),
            node_labels=("A", "B", "C"),
        )
        system = System(substrate, STATE, subset=CANDIDATE)
        try:
            full_2023 = system.sia(
                formalism=FORMALISM_2023,
                parallel=False,
                progress=False,
            )
            fixed_2023 = system.sia(
                system_state=full_2023.system_state,
                formalism=FORMALISM_2023,
                parallel=False,
                progress=False,
            )
            fixed_2026 = system.sia(
                system_state=full_2023.system_state,
                formalism=FORMALISM_2026,
                parallel=False,
                progress=False,
            )
        except Exception as exc:
            attempts.append({"seed": seed, "status": "ERROR", "error": repr(exc)})
            continue

        summary_2023 = _summary(fixed_2023)
        summary_2026 = _summary(fixed_2026)
        ii_value = summary_2026["intrinsic_information"]
        cause_effect_minimum = min(
            summary_2026["cause_phi"],
            summary_2026["effect_phi"],
        )
        cap_binds = (
            ii_value is not None
            and ii_value < cause_effect_minimum - TOLERANCE
            and summary_2026["phi_s"] < summary_2023["phi_s"] - TOLERANCE
            and abs(summary_2026["phi_s"] - ii_value) <= TOLERANCE
            and summary_2026["mip_cut_sha256"] == summary_2023["mip_cut_sha256"]
            and summary_2023["phi_s"] > TOLERANCE
        )
        attempt = {
            "seed": seed,
            "status": "CAP_BINDS" if cap_binds else "CAP_NOT_BINDING",
            "phi_2023": summary_2023["phi_s"],
            "phi_2026": summary_2026["phi_s"],
            "intrinsic_information": ii_value,
            "cause_effect_minimum_2026": cause_effect_minimum,
            "mip_unchanged": summary_2023["mip_cut_sha256"] == summary_2026["mip_cut_sha256"],
        }
        attempts.append(attempt)
        if cap_binds:
            model_bytes = canonical_model_bytes(tpm, connectivity.tolist())
            selected = {
                "seed": seed,
                "tpm_state_by_node": tpm,
                "connectivity_matrix": connectivity.tolist(),
                "state": list(STATE),
                "candidate_system": list(CANDIDATE),
                "model_sha256": model_digest(model_bytes),
                "evaluation_sha256": evaluation_digest(model_bytes, STATE),
                "analysis_request_sha256_iit4_2023": analysis_request_digest(
                    model_bytes, STATE, CANDIDATE, FORMALISM_2023, QUANTITY_ID
                ),
                "analysis_request_sha256_iit4_2026": analysis_request_digest(
                    model_bytes, STATE, CANDIDATE, FORMALISM_2026, QUANTITY_ID
                ),
                "iit4_2023": summary_2023,
                "iit4_2026": summary_2026,
                "cap_binding": True,
                "mip_unchanged": True,
                "phi_reduction_2023_minus_2026": (
                    summary_2023["phi_s"] - summary_2026["phi_s"]
                ),
                "cap_gap_to_cause_effect_minimum": (
                    cause_effect_minimum - ii_value
                ),
            }
            break

    status = "PASS" if selected is not None else "NOT_QUALIFIED"
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "qualification_scope": "pinned_pyphi_iit4_2026_cap_binding_positive_control_discovery",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": installed,
            "python_version": platform.python_version(),
            "source_commit": PYPHI_SOURCE_COMMIT,
            "wheel_sha256": PINNED_WHEEL_SHA256,
            "formalism_2023": FORMALISM_2023,
            "formalism_2026": FORMALISM_2026,
            "quantity_id": QUANTITY_ID,
            "generator": "python.random.Random(seed), uniform local TPM probability in [0.025,0.975]",
            "seed_order": "0 through max_seeds-1; first qualifying cap-binding seed is frozen",
            "max_seeds": max_seeds,
            "state": list(STATE),
            "candidate_system": list(CANDIDATE),
            "connectivity_matrix": connectivity.tolist(),
            "tolerance": TOLERANCE,
        },
        "selected_fixture": selected,
        "attempts": attempts,
        "summary": {
            "attempted_seed_count": len(attempts),
            "cap_binding_fixture_found": selected is not None,
            "selected_seed": None if selected is None else selected["seed"],
        },
        "notes": [
            "This deterministic search discovers one positive control; it does not estimate cap-binding prevalence.",
            "The same exact cause/effect system state is held fixed under IIT4-2023 and IIT4-2026.",
            "A valid fixture requires ii(s) to be strictly below both phi_c and phi_e, with IIT4-2026 phi_s equal to ii(s).",
            "The fixed-state MIP must remain unchanged; this tests the post-MIP intrinsic-information cap.",
            "The candidate TPM and model/evaluation/request identities are frozen in the artifact.",
            "No inference about phenomenal consciousness is permitted.",
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
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_seeds <= 2048:
        raise SystemExit("--max-seeds must be in [1, 2048]")
    try:
        artifact = generate(args.output, args.max_seeds)
    except Exception as exc:
        print(f"NOT QUALIFIED: {exc}", file=sys.stderr)
        return 2
    summary = artifact["summary"]
    selected = artifact["selected_fixture"]
    print(
        f"status={artifact['status']} attempts={summary['attempted_seed_count']} "
        f"selected_seed={summary['selected_seed']} "
        f"phi23={None if selected is None else selected['iit4_2023']['phi_s']} "
        f"phi26={None if selected is None else selected['iit4_2026']['phi_s']} "
        f"ii={None if selected is None else selected['iit4_2026']['intrinsic_information']}"
    )
    return 0 if artifact["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
