"""Validate CETB defensible multiverse manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a defensible multiverse manifest fails validation."""


def validate_manifest(manifest: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise ManifestError("install research/rcaf/requirements-validation.txt") from exc

    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).iter_errors(manifest),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ManifestError(f"schema violation at {location}: {first.message}")

    status = manifest["status"]
    freeze = manifest["freeze"]
    if status == "preregistered_confirmatory" and freeze["confirmatory_outcomes_accessed"]:
        raise ManifestError("confirmatory manifest cannot be frozen after outcome access")
    if status == "synthetic_fixture" and freeze["confirmatory_outcomes_accessed"]:
        raise ManifestError("synthetic fixture cannot claim confirmatory outcome access")

    if manifest["generation"]["mode"] == "deterministic_sample":
        sampling_method = manifest["generation"].get("sampling_method", "")
        if not sampling_method.strip():
            raise ManifestError("deterministic sampling requires a declared sampling_method")

    node_ids: set[str] = set()
    option_ids: set[str] = set()
    primary_options = 0
    for node in manifest["decision_nodes"]:
        node_id = node["node_id"]
        if node_id in node_ids:
            raise ManifestError(f"duplicate decision node ID: {node_id}")
        node_ids.add(node_id)
        if node["applicability"] == "not_applicable_with_rationale":
            if not node.get("not_applicable_rationale", "").strip():
                raise ManifestError(f"not-applicable node {node_id!r} requires a rationale")
        for option in node["options"]:
            option_id = option["option_id"]
            if option_id in option_ids:
                raise ManifestError(f"duplicate option ID: {option_id}")
            option_ids.add(option_id)
            if not option["defensibility_rationale"].strip():
                raise ManifestError(f"option {option_id!r} requires a defensibility rationale")
            if option["lane"] == "primary":
                primary_options += 1

    if primary_options == 0:
        raise ManifestError("at least one independently justified primary-lane option is required")

    rule_ids: set[str] = set()
    for rule in manifest["compatibility_rules"]:
        if rule["rule_id"] in rule_ids:
            raise ManifestError(f"duplicate compatibility rule ID: {rule['rule_id']}")
        rule_ids.add(rule["rule_id"])
        unknown = set(rule["option_ids"]) - option_ids
        if unknown:
            raise ManifestError(
                f"compatibility rule {rule['rule_id']!r} references unknown options: {sorted(unknown)}"
            )
        if rule["effect"] == "separate_estimand" and len(set(rule["option_ids"])) < 2:
            raise ManifestError("separate_estimand rules must reference at least two distinct options")

    assessment_ids: set[str] = set()
    for assessment in manifest["equivalence_assessments"]:
        assessment_id = assessment["assessment_id"]
        if assessment_id in assessment_ids:
            raise ManifestError(f"duplicate equivalence assessment ID: {assessment_id}")
        assessment_ids.add(assessment_id)
        unknown = set(assessment["option_ids"]) - option_ids
        if unknown:
            raise ManifestError(
                f"equivalence assessment {assessment_id!r} references unknown options: {sorted(unknown)}"
            )
        if len(set(assessment["option_ids"])) < 2:
            raise ManifestError("equivalence assessments must compare at least two distinct options")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("schema", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        validate_manifest(manifest, schema)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: CETB defensible multiverse manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
