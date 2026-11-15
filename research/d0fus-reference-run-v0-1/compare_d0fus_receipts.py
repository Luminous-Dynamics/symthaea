#!/usr/bin/env python3
"""Compare source-pinned D0FUS wheel/source ITER run receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    wheel = load(args.wheel_receipt)
    source = load(args.source_receipt)

    required_equal = (
        "solver_version",
        "source_git_commit",
        "deck_git_blob_sha",
        "deck_sha256",
        "global_config_dataclass_field_count",
        "run_output_tuple_length",
        "canonical_output_tuple_sha256",
        "core_observables",
        "power_balance",
    )
    mismatches = {
        field: {"wheel": wheel.get(field), "source": source.get(field)}
        for field in required_equal
        if wheel.get(field) != source.get(field)
    }
    if wheel.get("run_output_tuple_length", 0) <= 0:
        mismatches["wheel_run_output_tuple_length"] = wheel.get("run_output_tuple_length")
    if source.get("run_output_tuple_length", 0) <= 0:
        mismatches["source_run_output_tuple_length"] = source.get("run_output_tuple_length")

    module_inventory = wheel.get("wheel_source_module_parity", {})
    output_parity = "PASS" if not mismatches else "FAIL"
    receipt = {
        "receipt_version": "d0fus-wheel-source-parity/0.2",
        "solver": "D0FUS",
        "solver_version": wheel.get("solver_version"),
        "source_git_commit": wheel.get("source_git_commit"),
        "deck_git_blob_sha": wheel.get("deck_git_blob_sha"),
        "deck_sha256": wheel.get("deck_sha256"),
        "pypi_wheel_sha256": wheel.get("wheel_sha256"),
        "pypi_sdist_sha256": wheel.get("sdist_sha256"),
        "global_config_dataclass_field_count": {
            "wheel": wheel.get("global_config_dataclass_field_count"),
            "source": source.get("global_config_dataclass_field_count"),
        },
        "run_output_tuple_length": {
            "wheel": wheel.get("run_output_tuple_length"),
            "source": source.get("run_output_tuple_length"),
        },
        "canonical_output_tuple_sha256": {
            "wheel": wheel.get("canonical_output_tuple_sha256"),
            "source": source.get("canonical_output_tuple_sha256"),
        },
        "core_observables": {
            "wheel": wheel.get("core_observables"),
            "source": source.get("core_observables"),
        },
        "power_balance": {
            "wheel": wheel.get("power_balance"),
            "source": source.get("power_balance"),
        },
        "wheel_source_module_inventory": module_inventory,
        "runtime_versions": {
            "wheel": wheel.get("python_and_dependency_versions"),
            "source": source.get("python_and_dependency_versions"),
        },
        "wheel_output_report_sha256": wheel.get("output_report_sha256"),
        "source_output_report_sha256": source.get("output_report_sha256"),
        "output_mismatches": mismatches,
        "output_parity": output_parity,
        "claim_ceiling": (
            "pinned D0FUS ITER reference-deck execution and measured wheel/source "
            "output comparison; not independent physical validation"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("D0FUS_PARITY_RECEIPT=" + str(args.receipt))
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))

    if mismatches:
        raise SystemExit(
            "D0FUS PyPI-wheel/source output mismatch; diagnostic receipt preserved: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
