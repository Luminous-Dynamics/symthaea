#!/usr/bin/env python3
"""Validate the required CI job manifest against an exact workflow generation.

This validator is queue-neutral and does not call GitHub. It proves structural
consistency between the checked-in workflow, merge-admission policy, and
required-job manifest. By default an intentionally incomplete manifest is a
failure; --allow-incomplete permits staging/audit of a fail-closed scaffold.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from scripts.ci.evaluate_merge_admission_v2 import (
    git_blob_sha,
    validate_policy,
    validate_required_job_manifest,
)
from scripts.ci.validate_ci_lifecycle_registry import extract_job_blocks


def validate(
    workflow_bytes: bytes,
    policy: dict,
    manifest: dict,
    *,
    allow_incomplete: bool = False,
) -> dict:
    validate_policy(policy)
    validate_required_job_manifest(manifest)

    actual_workflow_blob = git_blob_sha(workflow_bytes)
    workflow_path = policy["full_integration"]["workflow_path"]
    manifest_path = policy["full_integration"]["required_job_manifest_path"]
    control_paths = set(policy["control_plane"]["paths"])

    if workflow_path not in control_paths:
        raise ValueError("workflow path is not governed by the merge-admission control plane")
    if manifest_path not in control_paths:
        raise ValueError("required-job manifest path is not governed by the merge-admission control plane")
    if manifest["workflow_path"] != workflow_path:
        raise ValueError("required-job manifest workflow_path does not match policy")
    if manifest["workflow_blob_sha"] != actual_workflow_blob:
        raise ValueError(
            "required-job manifest workflow blob drift: "
            f"expected {manifest['workflow_blob_sha']}, got {actual_workflow_blob}"
        )

    accepted_events = set(policy["full_integration"]["accepted_events"])
    profile_events = set(manifest["profiles"])
    if profile_events != accepted_events:
        missing = sorted(accepted_events - profile_events)
        extra = sorted(profile_events - accepted_events)
        details: list[str] = []
        if missing:
            details.append("missing profiles: " + ", ".join(missing))
        if extra:
            details.append("unexpected profiles: " + ", ".join(extra))
        raise ValueError(
            "required-job manifest event profiles do not match policy: "
            + "; ".join(details)
        )

    blocks = extract_job_blocks(workflow_bytes.decode("utf-8"))
    workflow_job_ids = set(blocks)
    profile_summaries: dict[str, dict] = {}

    for event, profile in manifest["profiles"].items():
        top_level = set(profile["top_level_job_ids"])
        family_ids = {family["job_id"] for family in profile["families"]}

        if manifest["complete"]:
            missing = sorted(workflow_job_ids - top_level)
            extra = sorted(top_level - workflow_job_ids)
            if missing or extra:
                details: list[str] = []
                if missing:
                    details.append("missing workflow jobs: " + ", ".join(missing))
                if extra:
                    details.append("unknown workflow jobs: " + ", ".join(extra))
                raise ValueError(
                    f"complete profile {event} does not census the exact workflow: "
                    + "; ".join(details)
                )
            if family_ids != workflow_job_ids:
                raise ValueError(
                    f"complete profile {event} family job IDs do not equal exact workflow job IDs"
                )

        profile_summaries[event] = {
            "top_level_job_count": len(top_level),
            "family_count": len(family_ids),
            "workflow_job_count": len(workflow_job_ids),
            "exact_workflow_job_census": top_level == workflow_job_ids,
        }

    if manifest["complete"] is not True and not allow_incomplete:
        raise ValueError(
            "required-job manifest is intentionally incomplete; "
            "use --allow-incomplete only for staging/audit, never qualification"
        )

    return {
        "schema": "symthaea.required-ci-job-manifest-validation.v1",
        "workflow_blob_sha": actual_workflow_blob,
        "workflow_job_count": len(workflow_job_ids),
        "manifest_complete": manifest["complete"],
        "qualification_eligible": manifest["complete"] is True,
        "profiles": profile_summaries,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument("--policy", default="scripts/ci/merge_admission_policy_v2.json")
    parser.add_argument("--manifest", default="scripts/ci/required_ci_job_manifest_v1.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        workflow_bytes = pathlib.Path(args.workflow).read_bytes()
        policy = json.loads(pathlib.Path(args.policy).read_text(encoding="utf-8"))
        manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(policy, dict) or not isinstance(manifest, dict):
            raise ValueError("policy and manifest must be JSON objects")
        result = validate(
            workflow_bytes,
            policy,
            manifest,
            allow_incomplete=args.allow_incomplete,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"required CI job manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
