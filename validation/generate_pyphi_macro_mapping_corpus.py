#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate a wheel-pinned reference corpus for PyPhi macro mapping families."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import struct
import sys
from pathlib import Path
from typing import Any

PINNED_PYPHI_VERSION = "2.0.0rc4"
PINNED_PYPHI_WHEEL_SHA256 = (
    "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620"
)
PYPHI_SOURCE_COMMIT = "134a8f31b49db512322465ed3ba5da7194d362fd"
SCHEMA_VERSION = "pyphi-macro-mapping-corpus-v0.1"

CASES = (
    {
        "case_id": "coarse-2-on-count-2",
        "family": "coarse_grain",
        "constituents": [0, 1],
        "update_grain": 1,
        "on_counts": [2],
        "background_apportionment": [3],
    },
    {
        "case_id": "coarse-2-on-count-1",
        "family": "coarse_grain",
        "constituents": [0, 1],
        "update_grain": 1,
        "on_counts": [1],
    },
    {
        "case_id": "coarse-3-on-count-1-2",
        "family": "coarse_grain",
        "constituents": [0, 1, 2],
        "update_grain": 1,
        "on_counts": [1, 2],
    },
    {
        "case_id": "blackbox-2-grain-2-output-0",
        "family": "blackbox",
        "constituents": [0, 1],
        "update_grain": 2,
        "output_constituents": [0],
    },
    {
        "case_id": "blackbox-2-grain-2-outputs-0-1",
        "family": "blackbox",
        "constituents": [0, 1],
        "update_grain": 2,
        "output_constituents": [0, 1],
    },
    {
        "case_id": "blackbox-3-grain-2-outputs-1-2",
        "family": "blackbox",
        "constituents": [0, 1, 2],
        "update_grain": 2,
        "output_constituents": [1, 2],
    },
)


def _macro_mapping_digest(
    constituents: list[int], update_grain: int, mapping: list[int]
) -> str:
    encoded = bytearray(b"SYMTHAEA-IIT-MACRO-MAPPING\0v0.1\0")
    encoded.extend(struct.pack("<Q", len(constituents)))
    for unit in constituents:
        encoded.extend(struct.pack("<Q", unit))
    encoded.extend(struct.pack("<Q", update_grain))
    encoded.extend(struct.pack("<Q", len(mapping)))
    encoded.extend(bytes(mapping))
    return hashlib.sha256(encoded).hexdigest()


def _spec_digest(
    mapping_digest: str,
    complement_flipped: bool,
    background_apportionment: list[int],
) -> str:
    encoded = bytearray(b"SYMTHAEA-IIT-MACRO-UNIT-SPEC\0v0.2\0")
    encoded.extend(mapping_digest.encode("ascii"))
    encoded.extend(bytes([int(complement_flipped)]))
    encoded.extend(struct.pack("<Q", len(background_apportionment)))
    for unit in background_apportionment:
        encoded.extend(struct.pack("<Q", unit))
    return hashlib.sha256(encoded).hexdigest()


def generate(output: Path) -> dict[str, Any]:
    version = importlib.metadata.version("pyphi")
    if version != PINNED_PYPHI_VERSION:
        raise RuntimeError(f"expected pyphi=={PINNED_PYPHI_VERSION}, found {version}")

    from pyphi.macro.units import MacroUnit, blackbox, coarse_grain

    output_cases = []
    total_entries = 0
    for case in CASES:
        constituents = case["constituents"]
        family = case["family"]
        if family == "coarse_grain":
            mapping = list(coarse_grain(len(constituents), case["on_counts"]))
        elif family == "blackbox":
            mapping = list(
                blackbox(
                    len(constituents),
                    case["update_grain"],
                    case["output_constituents"],
                )
            )
        else:
            raise RuntimeError(f"unsupported fixture family: {family}")

        expected_entries = 1 << (len(constituents) * case["update_grain"])
        if len(mapping) != expected_entries:
            raise RuntimeError(
                f"{case['case_id']}: expected {expected_entries} mapping rows, "
                f"received {len(mapping)}"
            )
        if set(mapping) != {0, 1}:
            raise RuntimeError(f"{case['case_id']}: PyPhi produced a constant mapping")
        if any(value not in (0, 1) for value in mapping):
            raise RuntimeError(f"{case['case_id']}: PyPhi produced a non-binary mapping")

        background_apportionment = case.get("background_apportionment", [])
        pyphi_unit = MacroUnit(
            constituents=tuple(constituents),
            update_grain=case["update_grain"],
            mapping=tuple(mapping),
            background_apportionment=tuple(background_apportionment),
        )
        background_apportionment = list(pyphi_unit.background_apportionment)
        complement = [1 - value for value in mapping]
        complement_flipped = complement < mapping
        canonical_mapping = complement if complement_flipped else mapping
        mapping_sha256 = _macro_mapping_digest(
            constituents, case["update_grain"], canonical_mapping
        )
        spec_sha256 = _spec_digest(
            mapping_sha256, complement_flipped, background_apportionment
        )
        item = {
            **case,
            "mapping": mapping,
            "canonical_mapping": canonical_mapping,
            "background_apportionment": background_apportionment,
            "complement_flipped": complement_flipped,
            "mapping_sha256": mapping_sha256,
            "spec_sha256": spec_sha256,
            "mapping_entries": len(mapping),
            "sequence_state_bits": len(constituents) * case["update_grain"],
        }
        output_cases.append(item)
        total_entries += len(mapping)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "qualification_scope": "pinned_pyphi_macro_mapping_reference_only",
        "macro_framework": "Marshall_et_al_2026_intrinsic_units",
        "formalism_scope": "mapping_only_no_phi_or_macro_tpm_claim",
        "consciousness_inference_permitted": False,
        "provenance": {
            "distribution": "pyphi",
            "version": version,
            "python_version": platform.python_version(),
            "wheel_sha256": PINNED_PYPHI_WHEEL_SHA256,
            "source_commit": PYPHI_SOURCE_COMMIT,
            "mapping_order": (
                "mixed-radix little-endian; constituent order fixes digit order; "
                "temporal windows oldest-first"
            ),
        },
        "cases": output_cases,
        "summary": {
            "expected_case_count": len(CASES),
            "computed_case_count": len(output_cases),
            "total_mapping_entries": total_entries,
            "families": {
                "coarse_grain": sum(item["family"] == "coarse_grain" for item in output_cases),
                "blackbox": sum(item["family"] == "blackbox" for item in output_cases),
            },
        },
        "notes": [
            "This corpus pins PyPhi mapping-family truth tables only.",
            "It does not contain a macro TPM, a unithood verdict, a grain search, or phi_s.",
            "Rust parity must be established by a separate exact request and cellwise comparison.",
            "The macro mapping layer is independent of the IIT 4.0 (2023) versus IIT 4.0 (2026) system-phi distinction.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PYPHI_MACRO_MAPPING_REFERENCE.json"),
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
        f"mapping_entries={artifact['summary']['total_mapping_entries']} "
        f"pyphi={artifact['provenance']['version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
