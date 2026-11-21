"""Validate semantic evidence gates in CETB measurement qualification receipts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class QualificationError(ValueError):
    """Raised when a qualification receipt overstates or misstates its evidence."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def validate_semantics(receipt: dict[str, Any]) -> None:
    status = receipt["evidence_status"]
    data = receipt["data_receipt"]
    participant_count = data["participant_count"]

    if status in {"signal_analysis", "independent_replay"}:
        _need(
            data["access_state"] == "downloaded_verified",
            f"{status} requires downloaded_verified source data",
        )
        digest = data["file_manifest_sha256"]
        _need(
            isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            f"{status} requires a lowercase source manifest SHA-256",
        )
        _need(participant_count >= 2, f"{status} requires at least two participants")

    sufficiency = receipt["measurement_sufficiency"]
    if sufficiency["status"] == "adequate":
        _need(sufficiency["metric_value"] is not None, "adequate sufficiency requires metric_value")
        _need(sufficiency["threshold"] is not None, "adequate sufficiency requires a predeclared threshold")
        _need(
            math.isfinite(float(sufficiency["metric_value"]))
            and math.isfinite(float(sufficiency["threshold"])),
            "adequate sufficiency values must be finite",
        )

    repeatability = receipt["repeatability"]
    if repeatability["confidence_interval"] is not None:
        interval = repeatability["confidence_interval"]
        _need(interval["lower"] <= interval["upper"], "repeatability confidence interval is reversed")
    if repeatability["status"] == "adequate":
        _need(
            repeatability["within_state_estimate"] is not None,
            "adequate repeatability requires a within_state_estimate",
        )
        _need(
            repeatability["confidence_interval"] is not None,
            "adequate repeatability requires a confidence interval",
        )
        _need(
            repeatability["participant_count"] >= 2,
            "adequate repeatability requires at least two independent participants",
        )
        _need(
            repeatability["inferential_unit"] == "participant",
            "repeatability inference must use participants as the independent unit",
        )

    construct = receipt["construct_validity"]
    if construct["status"] == "adequate":
        _need(bool(construct["positive_controls"]), "adequate construct validity requires positive controls")
        _need(bool(construct["negative_controls"]), "adequate construct validity requires negative controls")
        _need(
            construct["reporting_scope"] != "phenomenal_experience_not_observed",
            "an unobserved phenomenal-experience endpoint cannot be marked construct-valid",
        )

    lanes = receipt["transportability"]
    participant_lane = lanes["participant_holdout"]
    if participant_lane["status"] == "adequate":
        _need(
            participant_lane["source_independence"] == "same_release_heldout_participants",
            "participant holdout must use participants absent from model/feature selection",
        )
        _need(participant_lane["sample_size"] >= 1, "participant holdout requires held-out participants")

    site_lane = lanes["site_holdout"]
    if site_lane["status"] == "adequate":
        _need(
            site_lane["source_independence"] in {
                "same_release_heldout_site", "independent_acquisition_same_cohort"
            },
            "site holdout cannot be claimed from a random participant split",
        )
        _need(site_lane["sample_size"] >= 1, "site holdout requires held-out site data")

    cohort_lane = lanes["independent_cohort"]
    if cohort_lane["status"] == "adequate":
        _need(
            cohort_lane["source_independence"] == "independent_cohort",
            "independent-cohort qualification requires genuinely independent participants",
        )
        _need(cohort_lane["sample_size"] >= 1, "independent-cohort qualification requires participants")


def validate_receipt(receipt: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise QualificationError("install research/rcaf/requirements-validation.txt") from exc
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).iter_errors(receipt),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise QualificationError(f"schema violation at {location}: {first.message}")
    validate_semantics(receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("schema", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        validate_receipt(receipt, schema)
    except (OSError, json.JSONDecodeError, QualificationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: CETB multidimensional measurement qualification receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
