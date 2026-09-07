#!/usr/bin/env python3
"""Pure, fail-closed merge-admission policy evaluator.

This module does not call GitHub, merge code, or emit a GitHub status. It accepts
an observation assembled by a trusted caller, evaluates it against an explicit
policy, and emits a content-addressed *unsigned* disposition receipt.

The v1 trust theorem is intentionally narrow:

    candidate-owned CI result != merge authority

For ordinary admission, the candidate must preserve the target-base CI control
plane byte-for-byte. A candidate that changes the authority/control plane is
classified BOOTSTRAP_REQUIRED and cannot self-authorize with the machinery it
is changing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

POLICY_SCHEMA = "symthaea.merge-admission-policy.v1"
OBSERVATION_SCHEMA = "symthaea.merge-admission-observation.v1"
RECEIPT_SCHEMA = "symthaea.merge-admission-receipt.v1"
HEX_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class Decision(str, Enum):
    ADMITTED = "admitted"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    BOOTSTRAP_REQUIRED = "bootstrap_required"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Evaluation:
    decision: Decision
    reasons: tuple[str, ...]
    receipt: dict[str, Any]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha(value: Any, name: str) -> str:
    value = _require_string(value, name)
    if not HEX_ID.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40- or 64-hex object id")
    return value


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("unexpected merge admission policy schema")
    _require_string(policy.get("repository"), "policy.repository")
    _require_string(policy.get("target_branch"), "policy.target_branch")
    if policy.get("decision_default") != "incomplete":
        raise ValueError("policy must default to incomplete")
    if policy.get("head_change_invalidates") is not True:
        raise ValueError("head changes must invalidate admission")
    if policy.get("base_change_invalidates") is not True:
        raise ValueError("base changes must invalidate admission")
    if policy.get("unknown_evidence_default") != "reject":
        raise ValueError("unknown evidence must fail closed")
    if policy.get("focused_evidence_can_substitute_for_full_integration") is not False:
        raise ValueError("focused evidence may not substitute for full integration")
    if policy.get("tier1_can_substitute_for_full_integration") is not False:
        raise ValueError("Tier-1 evidence may not substitute for full integration")

    control = _require_object(policy.get("control_plane"), "policy.control_plane")
    if control.get("mode") != "exact_base_equivalence":
        raise ValueError("v1 requires exact_base_equivalence control-plane mode")
    if control.get("candidate_changes_require_independent_bootstrap") is not True:
        raise ValueError("control-plane changes must require independent bootstrap")
    paths = control.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(v, str) and v for v in paths):
        raise ValueError("control_plane.paths must be a non-empty string list")
    if len(paths) != len(set(paths)):
        raise ValueError("control_plane.paths contains duplicates")

    integration = _require_object(policy.get("full_integration"), "policy.full_integration")
    workflow_path = _require_string(
        integration.get("workflow_path"), "policy.full_integration.workflow_path"
    )
    if workflow_path not in paths:
        raise ValueError("full integration workflow must be part of the control plane")
    events = integration.get("accepted_events")
    if not isinstance(events, list) or not events or not all(isinstance(v, str) for v in events):
        raise ValueError("full_integration.accepted_events must be a non-empty string list")
    if integration.get("required_status") != "completed":
        raise ValueError("full integration must require completed status")
    if integration.get("required_conclusion") != "success":
        raise ValueError("full integration must require success")
    for key in ("require_exact_head", "require_exact_base", "require_no_required_job_skips"):
        if integration.get(key) is not True:
            raise ValueError(f"full_integration.{key} must be true")


def validate_observation(observation: dict[str, Any]) -> None:
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("unexpected merge admission observation schema")
    _require_string(observation.get("repository"), "observation.repository")
    _require_string(observation.get("target_branch"), "observation.target_branch")
    _require_sha(observation.get("current_base_sha"), "observation.current_base_sha")
    _require_sha(observation.get("candidate_head_sha"), "observation.candidate_head_sha")
    _require_sha(observation.get("candidate_tree_sha"), "observation.candidate_tree_sha")
    control = observation.get("control_plane")
    if not isinstance(control, list):
        raise ValueError("observation.control_plane must be a list")
    for index, item in enumerate(control):
        item = _require_object(item, f"observation.control_plane[{index}]")
        _require_string(item.get("path"), f"observation.control_plane[{index}].path")
        for side in ("base_blob_sha", "candidate_blob_sha"):
            value = item.get(side)
            if value is not None:
                _require_sha(value, f"observation.control_plane[{index}].{side}")
    integration = observation.get("full_integration")
    if integration is not None and not isinstance(integration, dict):
        raise ValueError("observation.full_integration must be an object or null")


def _control_plane_disposition(
    policy: dict[str, Any], observation: dict[str, Any]
) -> tuple[Decision | None, list[str], dict[str, str | None]]:
    required_paths = list(policy["control_plane"]["paths"])
    rows = observation["control_plane"]
    by_path: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        path = row["path"]
        if path in by_path:
            duplicates.add(path)
        by_path[path] = row

    if duplicates:
        return Decision.REJECTED, ["duplicate control-plane observations: " + ", ".join(sorted(duplicates))], {}

    observed_paths = set(by_path)
    required_set = set(required_paths)
    missing = sorted(required_set - observed_paths)
    unexpected = sorted(observed_paths - required_set)
    if missing or unexpected:
        reasons: list[str] = []
        if missing:
            reasons.append("missing control-plane observations: " + ", ".join(missing))
        if unexpected:
            reasons.append("unexpected control-plane observations: " + ", ".join(unexpected))
        return Decision.REJECTED, reasons, {}

    base_blobs: dict[str, str | None] = {}
    changed: list[str] = []
    for path in required_paths:
        row = by_path[path]
        base_blob = row.get("base_blob_sha")
        candidate_blob = row.get("candidate_blob_sha")
        base_blobs[path] = base_blob
        if base_blob != candidate_blob:
            changed.append(path)

    if changed:
        return (
            Decision.BOOTSTRAP_REQUIRED,
            ["candidate changes merge-authority control plane: " + ", ".join(sorted(changed))],
            base_blobs,
        )
    return None, [], base_blobs


def _job_failure_disposition(jobs: Any) -> tuple[Decision | None, list[str]]:
    if not isinstance(jobs, list) or not jobs:
        return Decision.INCOMPLETE, ["required job observation set is absent or empty"]

    seen: set[str] = set()
    duplicate_names: set[str] = set()
    incomplete: list[str] = []
    failed: list[str] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            return Decision.REJECTED, [f"required_jobs[{index}] is not an object"]
        name = job.get("name")
        if not isinstance(name, str) or not name:
            return Decision.REJECTED, [f"required_jobs[{index}] has no valid name"]
        if name in seen:
            duplicate_names.add(name)
        seen.add(name)

        status = job.get("status")
        conclusion = job.get("conclusion")
        skipped = job.get("skipped")
        if skipped is True or conclusion == "skipped":
            incomplete.append(name + ":skipped")
        elif status != "completed":
            incomplete.append(name + ":" + str(status))
        elif conclusion != "success":
            failed.append(name + ":" + str(conclusion))

    if duplicate_names:
        return Decision.REJECTED, ["duplicate required job observations: " + ", ".join(sorted(duplicate_names))]
    if failed:
        return Decision.REJECTED, ["required jobs did not succeed: " + ", ".join(sorted(failed))]
    if incomplete:
        return Decision.INCOMPLETE, ["required jobs are not complete successes: " + ", ".join(sorted(incomplete))]
    return None, []


def evaluate(
    policy: dict[str, Any], observation: dict[str, Any], *, policy_bytes: bytes | None = None
) -> Evaluation:
    validate_policy(policy)
    validate_observation(observation)

    policy_digest = sha256_hex(policy_bytes if policy_bytes is not None else canonical_json(policy))
    reasons: list[str] = []

    if observation["repository"] != policy["repository"]:
        reasons.append("repository identity does not match policy")
    if observation["target_branch"] != policy["target_branch"]:
        reasons.append("target branch does not match policy")
    if reasons:
        decision = Decision.REJECTED
        return _finish(policy_digest, observation, decision, reasons)

    control_decision, control_reasons, base_blobs = _control_plane_disposition(policy, observation)
    if control_decision is not None:
        return _finish(policy_digest, observation, control_decision, control_reasons)

    integration = observation.get("full_integration")
    if integration is None:
        return _finish(
            policy_digest,
            observation,
            Decision.INCOMPLETE,
            ["full integration evidence is absent; Tier-1/focused evidence cannot substitute"],
        )

    required = policy["full_integration"]
    workflow_path = required["workflow_path"]
    workflow_base_blob = base_blobs.get(workflow_path)
    if workflow_base_blob is None:
        return _finish(
            policy_digest,
            observation,
            Decision.REJECTED,
            ["trusted full-integration workflow is absent from target-base control plane"],
        )

    identity_errors: list[str] = []
    if integration.get("workflow_path") != workflow_path:
        identity_errors.append("workflow path is not the trusted full-integration workflow")
    workflow_blob = integration.get("workflow_blob_sha")
    if workflow_blob != workflow_base_blob:
        identity_errors.append("workflow blob does not equal trusted target-base workflow blob")
    if integration.get("event") not in set(required["accepted_events"]):
        identity_errors.append("workflow event is not admitted by policy")
    run_id = integration.get("run_id")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        identity_errors.append("workflow run_id is not a positive integer")
    if identity_errors:
        return _finish(policy_digest, observation, Decision.REJECTED, identity_errors)

    stale: list[str] = []
    if integration.get("head_sha") != observation["candidate_head_sha"]:
        stale.append("full integration head does not equal current candidate head")
    if integration.get("base_sha") != observation["current_base_sha"]:
        stale.append("full integration base does not equal current target base")
    if stale:
        return _finish(policy_digest, observation, Decision.STALE, stale)

    status = integration.get("status")
    conclusion = integration.get("conclusion")
    if status != required["required_status"]:
        return _finish(
            policy_digest,
            observation,
            Decision.INCOMPLETE,
            [f"full integration status is {status!r}, not completed"],
        )
    if conclusion != required["required_conclusion"]:
        if conclusion in (None, "cancelled", "skipped", "neutral"):
            decision = Decision.INCOMPLETE
        else:
            decision = Decision.REJECTED
        return _finish(
            policy_digest,
            observation,
            decision,
            [f"full integration conclusion is {conclusion!r}, not success"],
        )

    if integration.get("job_set_complete") is not True:
        return _finish(
            policy_digest,
            observation,
            Decision.INCOMPLETE,
            ["trusted collector has not established the required job set is complete"],
        )

    job_decision, job_reasons = _job_failure_disposition(integration.get("required_jobs"))
    if job_decision is not None:
        return _finish(policy_digest, observation, job_decision, job_reasons)

    return _finish(policy_digest, observation, Decision.ADMITTED, ["all v1 admission predicates satisfied"])


def _finish(
    policy_digest: str,
    observation: dict[str, Any],
    decision: Decision,
    reasons: list[str],
) -> Evaluation:
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "policy_sha256": policy_digest,
        "repository": observation.get("repository"),
        "target_branch": observation.get("target_branch"),
        "current_base_sha": observation.get("current_base_sha"),
        "candidate_head_sha": observation.get("candidate_head_sha"),
        "candidate_tree_sha": observation.get("candidate_tree_sha"),
        "decision": decision.value,
        "reasons": list(reasons),
        "caveat": "unsigned policy-core disposition; enforceable authority requires a trusted external collector/check and repository merge rule",
    }
    receipt_digest = sha256_hex(canonical_json(body))
    receipt = {**body, "receipt_sha256": receipt_digest}
    return Evaluation(decision=decision, reasons=tuple(reasons), receipt=receipt)


def load_json_object(path: pathlib.Path, name: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed, raw


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="scripts/ci/merge_admission_policy_v1.json")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        policy, policy_bytes = load_json_object(pathlib.Path(args.policy), "policy")
        observation, _ = load_json_object(pathlib.Path(args.observation), "observation")
        result = evaluate(policy, observation, policy_bytes=policy_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"merge admission evaluation refused: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(result.receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result.decision is Decision.ADMITTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
