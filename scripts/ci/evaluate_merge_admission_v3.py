#!/usr/bin/env python3
"""Pure, fail-closed merge-admission evaluator v3.

V3 separates candidate failure, missing evidence, collector inconsistency, and
trusted-control-plane inconsistency. It never calls GitHub and never merges.
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

from scripts.ci.merge_admission_manifest_v2 import (
    ManifestDisposition,
    canonical_json,
    evaluate_job_census,
    git_blob_id,
    validate_manifest,
    validate_manifest_against_workflow,
)

POLICY_SCHEMA = "symthaea.merge-admission-policy.v3"
OBSERVATION_SCHEMA = "symthaea.merge-admission-observation.v3"
RECEIPT_SCHEMA = "symthaea.merge-admission-receipt.v3"
HEX_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
RUN_STATUSES = {"queued", "in_progress", "completed", "waiting", "requested", "pending"}
RUN_CONCLUSIONS = {None, "success", "failure", "neutral", "cancelled", "skipped", "timed_out", "action_required", "stale", "startup_failure"}
FAILURE_CONCLUSIONS = {"failure", "timed_out", "action_required", "startup_failure"}
INCOMPLETE_CONCLUSIONS = {None, "neutral", "cancelled", "skipped", "stale"}


class Decision(str, Enum):
    ADMITTED = "admitted"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    BOOTSTRAP_REQUIRED = "bootstrap_required"
    COLLECTOR_INVALID = "collector_invalid"
    CONTROL_PLANE_INVALID = "control_plane_invalid"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Evaluation:
    decision: Decision
    reasons: tuple[str, ...]
    receipt: dict[str, Any]


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
        raise ValueError(f"{name} must be lowercase 40- or 64-hex")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")


def validate_policy(policy: dict[str, Any]) -> None:
    allowed = {
        "schema", "repository", "target_branch", "policy_path", "enforcement_ready",
        "decision_default", "head_change_invalidates", "base_change_invalidates",
        "unknown_evidence_default", "control_plane", "full_integration",
        "focused_evidence_can_substitute_for_full_integration",
        "tier1_can_substitute_for_full_integration",
    }
    _reject_unknown(policy, allowed, "policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("unexpected merge admission policy schema")
    _require_string(policy.get("repository"), "policy.repository")
    _require_string(policy.get("target_branch"), "policy.target_branch")
    policy_path = _require_string(policy.get("policy_path"), "policy.policy_path")
    if policy.get("enforcement_ready") is not False:
        raise ValueError("v3 must remain enforcement_ready=false")
    if policy.get("decision_default") != "incomplete":
        raise ValueError("policy must default to incomplete")
    if policy.get("head_change_invalidates") is not True or policy.get("base_change_invalidates") is not True:
        raise ValueError("head and base changes must invalidate admission")
    if policy.get("unknown_evidence_default") != "reject":
        raise ValueError("unknown evidence must fail closed")
    if policy.get("focused_evidence_can_substitute_for_full_integration") is not False:
        raise ValueError("focused evidence may not substitute for full integration")
    if policy.get("tier1_can_substitute_for_full_integration") is not False:
        raise ValueError("Tier-1 evidence may not substitute for full integration")

    control = _require_object(policy.get("control_plane"), "policy.control_plane")
    _reject_unknown(control, {"mode", "paths", "candidate_changes_require_independent_bootstrap"}, "policy.control_plane")
    if control.get("mode") != "exact_base_equivalence":
        raise ValueError("v3 requires exact_base_equivalence control-plane mode")
    if control.get("candidate_changes_require_independent_bootstrap") is not True:
        raise ValueError("control-plane changes must require independent bootstrap")
    paths = control.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(v, str) and v for v in paths):
        raise ValueError("control_plane.paths must be a non-empty string list")
    if len(paths) != len(set(paths)):
        raise ValueError("control_plane.paths contains duplicates")
    if policy_path not in paths:
        raise ValueError("policy path must be governed")

    integration = _require_object(policy.get("full_integration"), "policy.full_integration")
    expected = {
        "workflow_path", "required_job_manifest_path", "accepted_events",
        "required_status", "required_conclusion", "require_exact_head",
        "require_exact_base", "require_complete_job_census",
    }
    _reject_unknown(integration, expected, "policy.full_integration")
    workflow_path = _require_string(integration.get("workflow_path"), "policy.full_integration.workflow_path")
    manifest_path = _require_string(integration.get("required_job_manifest_path"), "policy.full_integration.required_job_manifest_path")
    if workflow_path not in paths or manifest_path not in paths:
        raise ValueError("workflow and required-job manifest must be governed")
    events = integration.get("accepted_events")
    if not isinstance(events, list) or not events or not all(isinstance(v, str) and v for v in events):
        raise ValueError("accepted_events must be a non-empty string list")
    if len(events) != len(set(events)):
        raise ValueError("accepted_events contains duplicates")
    if integration.get("required_status") != "completed" or integration.get("required_conclusion") != "success":
        raise ValueError("full integration must require completed/success")
    for key in ("require_exact_head", "require_exact_base", "require_complete_job_census"):
        if integration.get(key) is not True:
            raise ValueError(f"full_integration.{key} must be true")


def validate_observation(observation: dict[str, Any]) -> None:
    _reject_unknown(
        observation,
        {"schema", "repository", "target_branch", "current_base_sha", "candidate_head_sha", "candidate_tree_sha", "control_plane", "full_integration"},
        "observation",
    )
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("unexpected merge admission observation schema")
    _require_string(observation.get("repository"), "observation.repository")
    _require_string(observation.get("target_branch"), "observation.target_branch")
    for field in ("current_base_sha", "candidate_head_sha", "candidate_tree_sha"):
        _require_sha(observation.get(field), f"observation.{field}")

    control = observation.get("control_plane")
    if not isinstance(control, list):
        raise ValueError("observation.control_plane must be a list")
    for index, row in enumerate(control):
        row = _require_object(row, f"observation.control_plane[{index}]")
        _reject_unknown(row, {"path", "base_blob_sha", "candidate_blob_sha"}, f"observation.control_plane[{index}]")
        _require_string(row.get("path"), f"observation.control_plane[{index}].path")
        for side in ("base_blob_sha", "candidate_blob_sha"):
            value = row.get(side)
            if value is not None:
                _require_sha(value, f"observation.control_plane[{index}].{side}")

    integration = observation.get("full_integration")
    if integration is None:
        return
    integration = _require_object(integration, "observation.full_integration")
    allowed = {
        "workflow_path", "workflow_blob_sha", "run_id", "run_attempt", "event", "status",
        "conclusion", "head_sha", "base_sha", "job_census_proof", "job_census",
    }
    _reject_unknown(integration, allowed, "observation.full_integration")
    _require_string(integration.get("workflow_path"), "observation.full_integration.workflow_path")
    _require_sha(integration.get("workflow_blob_sha"), "observation.full_integration.workflow_blob_sha")
    _require_positive_int(integration.get("run_id"), "observation.full_integration.run_id")
    _require_positive_int(integration.get("run_attempt"), "observation.full_integration.run_attempt")
    _require_string(integration.get("event"), "observation.full_integration.event")
    for field in ("head_sha", "base_sha"):
        _require_sha(integration.get(field), f"observation.full_integration.{field}")

    proof = _require_object(integration.get("job_census_proof"), "observation.full_integration.job_census_proof")
    _reject_unknown(proof, {"reported_total_count", "pages_fetched", "terminal_page_observed"}, "observation.full_integration.job_census_proof")
    total = proof.get("reported_total_count")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise ValueError("reported_total_count must be a non-negative integer")
    _require_positive_int(proof.get("pages_fetched"), "job_census_proof.pages_fetched")
    if not isinstance(proof.get("terminal_page_observed"), bool):
        raise ValueError("terminal_page_observed must be boolean")

    jobs = integration.get("job_census")
    if not isinstance(jobs, list):
        raise ValueError("observation.full_integration.job_census must be a list")
    for index, job in enumerate(jobs):
        job = _require_object(job, f"observation.full_integration.job_census[{index}]")
        _reject_unknown(job, {"job_id", "name", "status", "conclusion", "skipped"}, f"observation.full_integration.job_census[{index}]")
        _require_positive_int(job.get("job_id"), f"job_census[{index}].job_id")
        _require_string(job.get("name"), f"job_census[{index}].name")
        if job.get("status") not in RUN_STATUSES:
            raise ValueError(f"job_census[{index}].status is unknown")
        if job.get("conclusion") not in RUN_CONCLUSIONS:
            raise ValueError(f"job_census[{index}].conclusion is unknown")
        if not isinstance(job.get("skipped"), bool):
            raise ValueError(f"job_census[{index}].skipped must be boolean")


def _control_plane(policy: dict[str, Any], observation: dict[str, Any]) -> tuple[Decision | None, list[str], dict[str, str | None]]:
    required = list(policy["control_plane"]["paths"])
    rows = observation["control_plane"]
    by_path: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        path = row["path"]
        if path in by_path:
            duplicates.add(path)
        by_path[path] = row
    if duplicates:
        return Decision.COLLECTOR_INVALID, ["duplicate control-plane observations: " + ", ".join(sorted(duplicates))], {}
    missing = sorted(set(required) - set(by_path))
    extra = sorted(set(by_path) - set(required))
    if missing or extra:
        reasons: list[str] = []
        if missing:
            reasons.append("missing control-plane observations: " + ", ".join(missing))
        if extra:
            reasons.append("unexpected control-plane observations: " + ", ".join(extra))
        return Decision.COLLECTOR_INVALID, reasons, {}
    base_blobs: dict[str, str | None] = {}
    changed: list[str] = []
    for path in required:
        base_blob = by_path[path].get("base_blob_sha")
        candidate_blob = by_path[path].get("candidate_blob_sha")
        base_blobs[path] = base_blob
        if base_blob != candidate_blob:
            changed.append(path)
    if changed:
        return Decision.BOOTSTRAP_REQUIRED, ["candidate changes merge-authority control plane: " + ", ".join(sorted(changed))], base_blobs
    return None, [], base_blobs


def _normalized_jobs(observation: dict[str, Any]) -> list[dict[str, Any]]:
    integration = observation.get("full_integration")
    jobs = integration.get("job_census", []) if isinstance(integration, dict) else []
    return sorted(
        ({"job_id": j.get("job_id"), "name": j.get("name"), "status": j.get("status"), "conclusion": j.get("conclusion"), "skipped": j.get("skipped")} for j in jobs if isinstance(j, dict)),
        key=lambda j: (int(j.get("job_id") or 0), str(j.get("name"))),
    )


def _evidence_binding(observation: dict[str, Any], manifest_raw: bytes, manifest_blob: str, required_jobs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    control = sorted(
        ({"path": r.get("path"), "base_blob_sha": r.get("base_blob_sha"), "candidate_blob_sha": r.get("candidate_blob_sha")} for r in observation.get("control_plane", []) if isinstance(r, dict)),
        key=lambda r: str(r.get("path")),
    )
    integration = observation.get("full_integration")
    if not isinstance(integration, dict):
        ib = None
    else:
        census = _normalized_jobs(observation)
        required = sorted(required_jobs, key=lambda j: (int(j["job_id"]), str(j["name"])))
        ib = {
            "workflow_path": integration.get("workflow_path"),
            "workflow_blob_sha": integration.get("workflow_blob_sha"),
            "run_id": integration.get("run_id"),
            "run_attempt": integration.get("run_attempt"),
            "event": integration.get("event"),
            "status": integration.get("status"),
            "conclusion": integration.get("conclusion"),
            "head_sha": integration.get("head_sha"),
            "base_sha": integration.get("base_sha"),
            "job_census_proof": integration.get("job_census_proof"),
            "job_census_count": len(census),
            "job_census_sha256": sha256_hex(canonical_json(census)),
            "required_job_observation_count": len(required),
            "required_jobs_sha256": sha256_hex(canonical_json(required)),
            "required_job_manifest_sha256": sha256_hex(manifest_raw),
            "required_job_manifest_blob_sha": manifest_blob,
        }
    value = {"control_plane": control, "full_integration": ib}
    return {"sha256": sha256_hex(canonical_json(value)), "value": value}


def _finish(policy_raw: bytes, manifest_raw: bytes, manifest_blob: str, observation: dict[str, Any], decision: Decision, reasons: list[str], required_jobs: tuple[dict[str, Any], ...] = ()) -> Evaluation:
    evidence = _evidence_binding(observation, manifest_raw, manifest_blob, required_jobs)
    body = {
        "schema": RECEIPT_SCHEMA,
        "policy_sha256": sha256_hex(policy_raw),
        "policy_blob_sha": git_blob_id(policy_raw),
        "enforcement_ready": False,
        "repository": observation.get("repository"),
        "target_branch": observation.get("target_branch"),
        "current_base_sha": observation.get("current_base_sha"),
        "candidate_head_sha": observation.get("candidate_head_sha"),
        "candidate_tree_sha": observation.get("candidate_tree_sha"),
        "evidence_binding_sha256": evidence["sha256"],
        "evidence_binding": evidence["value"],
        "decision": decision.value,
        "reasons": list(reasons),
        "caveat": "unsigned policy-core disposition; enforcement requires a trusted collector/check and repository merge rule",
    }
    return Evaluation(decision, tuple(reasons), {**body, "receipt_sha256": sha256_hex(canonical_json(body))})


def evaluate(policy: dict[str, Any], observation: dict[str, Any], manifest: dict[str, Any], workflow_bytes: bytes, *, policy_bytes: bytes | None = None, manifest_bytes: bytes | None = None) -> Evaluation:
    validate_policy(policy)
    validate_observation(observation)
    validate_manifest(manifest)
    policy_raw = policy_bytes if policy_bytes is not None else canonical_json(policy)
    manifest_raw = manifest_bytes if manifest_bytes is not None else canonical_json(manifest)
    manifest_blob = git_blob_id(manifest_raw)

    if observation["repository"] != policy["repository"] or observation["target_branch"] != policy["target_branch"]:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.REJECTED, ["repository or target-branch identity does not match policy"])

    control_decision, control_reasons, base_blobs = _control_plane(policy, observation)
    if control_decision is not None:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, control_decision, control_reasons)

    policy_path = policy["policy_path"]
    required = policy["full_integration"]
    workflow_path = required["workflow_path"]
    manifest_path = required["required_job_manifest_path"]
    policy_base = base_blobs.get(policy_path)
    workflow_base = base_blobs.get(workflow_path)
    manifest_base = base_blobs.get(manifest_path)
    if policy_base is None or workflow_base is None or manifest_base is None:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["trusted control plane is missing policy, workflow, or required-job manifest"])
    if git_blob_id(policy_raw, len(policy_base)) != policy_base:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["loaded policy bytes do not equal target-base policy blob"])
    if git_blob_id(workflow_bytes, len(workflow_base)) != workflow_base:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["loaded workflow bytes do not equal target-base workflow blob"])
    if git_blob_id(manifest_raw, len(manifest_base)) != manifest_base:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["loaded manifest bytes do not equal target-base manifest blob"])
    try:
        validate_manifest_against_workflow(manifest, workflow_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, [f"required-job manifest/workflow inconsistency: {exc}"])
    if set(manifest["profiles"]) != set(required["accepted_events"]):
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["required-job manifest event profiles do not equal admitted workflow events"])
    if manifest.get("complete") is not True:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.CONTROL_PLANE_INVALID, ["trusted required-job manifest is not qualification-complete"])

    integration = observation.get("full_integration")
    if integration is None:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.INCOMPLETE, ["full integration evidence is absent"])
    identity_errors: list[str] = []
    if integration["workflow_path"] != workflow_path:
        identity_errors.append("workflow path is not the trusted full-integration workflow")
    if integration["workflow_blob_sha"] != workflow_base:
        identity_errors.append("workflow blob is not the trusted target-base workflow blob")
    if integration["event"] not in set(required["accepted_events"]):
        identity_errors.append("workflow event is not admitted by policy")
    if identity_errors:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.REJECTED, identity_errors)

    stale: list[str] = []
    if integration["head_sha"] != observation["candidate_head_sha"]:
        stale.append("full integration head does not equal current candidate head")
    if integration["base_sha"] != observation["current_base_sha"]:
        stale.append("full integration base does not equal current target base")
    if stale:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.STALE, stale)

    status = integration.get("status")
    conclusion = integration.get("conclusion")
    if status not in RUN_STATUSES or conclusion not in RUN_CONCLUSIONS:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, ["workflow run status/conclusion is outside the admitted GitHub vocabulary"])
    if status != required["required_status"]:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.INCOMPLETE, [f"full integration status is {status!r}, not completed"])
    if conclusion != required["required_conclusion"]:
        decision = Decision.REJECTED if conclusion in FAILURE_CONCLUSIONS else Decision.INCOMPLETE
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, decision, [f"full integration conclusion is {conclusion!r}, not success"])

    census = integration["job_census"]
    proof = integration["job_census_proof"]
    if proof["terminal_page_observed"] is not True:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.INCOMPLETE, ["collector has not observed the terminal jobs page"])
    if len(census) != proof["reported_total_count"]:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, ["job census length does not equal GitHub reported_total_count"])
    seen_ids: set[int] = set()
    for job in census:
        if job["job_id"] in seen_ids:
            return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, ["job census contains duplicate GitHub job IDs"])
        seen_ids.add(job["job_id"])
        skipped_by_conclusion = job["conclusion"] == "skipped"
        if job["skipped"] != skipped_by_conclusion:
            return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, [f"job {job['job_id']} skipped flag contradicts conclusion"])
        if job["status"] != "completed" and job["conclusion"] is not None:
            return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, [f"job {job['job_id']} has conclusion before completed status"])
        if job["status"] == "completed" and job["conclusion"] is None:
            return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.COLLECTOR_INVALID, [f"job {job['job_id']} completed without a conclusion"])

    manifest_result = evaluate_job_census(manifest, integration["event"], census)
    mapping = {
        ManifestDisposition.MANIFEST_INCOMPLETE: Decision.CONTROL_PLANE_INVALID,
        ManifestDisposition.MANIFEST_MISMATCH: Decision.CONTROL_PLANE_INVALID,
        ManifestDisposition.CENSUS_INCOMPLETE: Decision.INCOMPLETE,
        ManifestDisposition.JOB_FAILURE: Decision.REJECTED,
    }
    if manifest_result.disposition is not ManifestDisposition.SATISFIED:
        return _finish(policy_raw, manifest_raw, manifest_blob, observation, mapping[manifest_result.disposition], list(manifest_result.reasons), manifest_result.required_jobs)
    return _finish(policy_raw, manifest_raw, manifest_blob, observation, Decision.ADMITTED, ["all v3 admission predicates satisfied"], manifest_result.required_jobs)


def load_json_object(path: pathlib.Path, name: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed, raw


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="scripts/ci/merge_admission_policy_v3.json")
    parser.add_argument("--required-job-manifest", default="scripts/ci/required_ci_job_manifest_v1.json")
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        policy, policy_bytes = load_json_object(pathlib.Path(args.policy), "policy")
        manifest, manifest_bytes = load_json_object(pathlib.Path(args.required_job_manifest), "required-job manifest")
        observation, _ = load_json_object(pathlib.Path(args.observation), "observation")
        workflow_bytes = pathlib.Path(args.workflow).read_bytes()
        result = evaluate(policy, observation, manifest, workflow_bytes, policy_bytes=policy_bytes, manifest_bytes=manifest_bytes)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
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
