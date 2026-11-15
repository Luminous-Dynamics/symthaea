#!/usr/bin/env python3
"""Compare the two source-pinned D0FUS ITER run receipts."""

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
    )
    mismatches = {
        field: {
            "wheel": wheel.get(field),
            "source": source.get(field),
        }
        for field in required_equal
        if wheel.get(field) != source.get(field)
    }
    if mismatches:
        raise SystemExit(
            "D0FUS PyPI-wheel/source replay mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    if wheel.get("wheel_source_module_parity", {}).get("result") != "byte-identical":
        raise SystemExit("D0FUS wheel modules were not byte-identical to the pinned source tree")
    if wheel.get("run_output_tuple_length") != 165:
        raise SystemExit(
            "D0FUS v3.2.0 reference output tuple length is not the expected 165: "
            f"{wheel.get('run_output_tuple_length')}"
        )

    receipt = {
        "receipt_version": "d0fus-wheel-source-parity/0.1",
        "solver": "D0FUS",
        "solver_version": wheel["solver_version"],
        "source_git_commit": wheel["source_git_commit"],
        "deck_git_blob_sha": wheel["deck_git_blob_sha"],
        "deck_sha256": wheel["deck_sha256"],
        "pypi_wheel_sha256": wheel["wheel_sha256"],
        "pypi_sdist_sha256": wheel["sdist_sha256"],
        "global_config_dataclass_field_count": wheel["global_config_dataclass_field_count"],
        "run_output_tuple_length": wheel["run_output_tuple_length"],
        "canonical_output_tuple_sha256": wheel["canonical_output_tuple_sha256"],
        "core_observables": wheel["core_observables"],
        "wheel_source_module_parity": wheel["wheel_source_module_parity"],
        "runtime_versions": {
            "wheel": wheel["python_and_dependency_versions"],
            "source": source["python_and_dependency_versions"],
        },
        "wheel_output_report_sha256": wheel["output_report_sha256"],
        "source_output_report_sha256": source["output_report_sha256"],
        "parity": "PASS",
        "claim_ceiling": (
            "pinned D0FUS ITER reference-deck execution and wheel/source parity; "
            "not independent physical validation"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("D0FUS_PARITY_RECEIPT=" + str(args.receipt))
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
