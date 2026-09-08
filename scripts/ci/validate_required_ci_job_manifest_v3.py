#!/usr/bin/env python3
"""Validate the base-owned required-job manifest against exact workflow bytes."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.evaluate_merge_admission_v3 import validate_policy  # noqa: E402
from scripts.ci.merge_admission_manifest_v2 import (  # noqa: E402
    extract_top_level_job_ids,
    git_blob_id,
    validate_manifest_against_workflow,
)


def validate(workflow_bytes: bytes, policy: dict, manifest: dict, *, allow_incomplete: bool = False) -> dict:
    validate_policy(policy)
    validate_manifest_against_workflow(manifest, workflow_bytes)

    integration = policy["full_integration"]
    if manifest["workflow_path"] != integration["workflow_path"]:
        raise ValueError("manifest workflow_path does not match policy")
    if set(manifest["profiles"]) != set(integration["accepted_events"]):
        raise ValueError("manifest event profiles do not equal policy accepted_events")

    governed = set(policy["control_plane"]["paths"])
    for path in (
        integration["workflow_path"],
        integration["required_job_manifest_path"],
        "scripts/ci/merge_admission_manifest_v2.py",
        "scripts/ci/validate_required_ci_job_manifest_v3.py",
    ):
        if path not in governed:
            raise ValueError(f"qualification-critical path is not governed: {path}")

    if manifest["complete"] is not True and not allow_incomplete:
        raise ValueError("manifest is intentionally incomplete; --allow-incomplete is staging-only")

    workflow_ids = extract_top_level_job_ids(workflow_bytes.decode("utf-8"))
    return {
        "schema": "symthaea.required-ci-job-manifest-validation.v3",
        "workflow_blob_sha": git_blob_id(workflow_bytes),
        "workflow_job_count": len(workflow_ids),
        "manifest_complete": manifest["complete"],
        "qualification_eligible": manifest["complete"] is True,
        "profiles": {
            event: {
                "top_level_job_count": len(profile["top_level_job_ids"]),
                "family_count": len(profile["families"]),
                "exact_workflow_job_census": set(profile["top_level_job_ids"]) == set(workflow_ids),
            }
            for event, profile in manifest["profiles"].items()
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument("--policy", default="scripts/ci/merge_admission_policy_v3.json")
    parser.add_argument("--manifest", default="scripts/ci/required_ci_job_manifest_v1.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        workflow = pathlib.Path(args.workflow).read_bytes()
        policy = json.loads(pathlib.Path(args.policy).read_text(encoding="utf-8"))
        manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(policy, dict) or not isinstance(manifest, dict):
            raise ValueError("policy and manifest must be JSON objects")
        result = validate(workflow, policy, manifest, allow_incomplete=args.allow_incomplete)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"required CI job manifest validation refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
