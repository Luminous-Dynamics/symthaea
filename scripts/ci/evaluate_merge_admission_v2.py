#!/usr/bin/env python3
"""Pure, fail-closed merge-admission policy evaluator.

This module does not call GitHub, merge code, or emit a GitHub status. A trusted
collector supplies exact repository/run metadata and a complete job census. The
policy core independently evaluates that census against a base-owned required-job
manifest and emits a content-addressed unsigned disposition receipt.

Trust theorem:

    candidate-owned CI result != merge authority

Ordinary admission additionally requires exact target-base equivalence for every
governed control-plane path. A candidate changing that plane is classified
BOOTSTRAP_REQUIRED and cannot self-authorize with the machinery it changes.
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

POLICY_SCHEMA = "symthaea.merge-admission-policy.v2"
OBSERVATION_SCHEMA = "symthaea.merge-admission-observation.v2"
RECEIPT_SCHEMA = "symthaea.merge-admission-receipt.v2"
MANIFEST_SCHEMA = "symthaea.required-ci-job-manifest.v1"
HEX_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

POLICY_KEYS = {
    "schema",
    "repository",
    "target_branch",
    "policy_path",
    "enforcement_ready",
    "decision_default",
    "head_change_invalidates",
    "base_change_invalidates",
    "unknown_evidence_default",
    "control_plane",
    "full_integration",
    "focused_evidence_can_substitute_for_full_integration",
    "tier1_can_substitute_for_full_integration",
}
CONTROL_POLICY_KEYS = {
    "mode",
    "paths",
    "candidate_changes_require_independent_bootstrap",
}
INTEGRATION_POLICY_KEYS = {
    "workflow_path",
    "required_job_manifest_path",
    "accepted_events",
    "required_status",
    "required_conclusion",
    "require_exact_head",
    "require_exact_base",
    "require_no_required_job_skips",
}
OBSERVATION_KEYS = {
    "schema",
    "repository",
    "target_branch",
    "current_base_sha",
    "candidate_head_sha",
    "candidate_tree_sha",
    "control_plane",
    "full_integration",
}
CONTROL_OBSERVATION_KEYS = {"path", "base_blob_sha", "candidate_blob_sha"}
INTEGRATION_OBSERVATION_KEYS = {
    "workflow_path",
    "workflow_blob_sha",
    "run_id",
    "run_attempt",
    "event",
    "status",
    "conclusion",
    "head_sha",
    "base_sha",
    "job_census_complete",
    "job_census",
}
JOB_OBSERVATION_KEYS = {"job_id", "name", "status", "conclusion", "skipped"}
MANIFEST_KEYS = {"schema", "workflow_path", "workflow_blob_sha", "complete", "profiles"}
PROFILE_KEYS = {"event", "top_level_job_ids", "families"}
FAMILY_KEYS = {
    "job_id",
    "api_name_regex",
    "min_instances",
    "max_instances",
    "required_disposition",
}


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


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


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


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")


def validate_policy(policy: dict[str, Any]) -> None:
    _reject_unknown_keys(policy, POLICY_KEYS, "policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("unexpected merge admission policy schema")
    _require_string(policy.get("repository"), "policy.repository")
    _require_string(policy.get("target_branch"), "policy.target_branch")
    policy_path = _require_string(policy.get("policy_path"), "policy.policy_path")
    if policy.get("enforcement_ready") is not False:
        raise ValueError("v2 is an executable policy core and must remain enforcement_ready=false")
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
    _reject_unknown_keys(control, CONTROL_POLICY_KEYS, "policy.control_plane")
    if control.get("mode") != "exact_base_equivalence":
        raise ValueError("v2 requires exact_base_equivalence control-plane mode")
    if control.get("candidate_changes_require_independent_bootstrap") is not True:
        raise ValueError("control-plane changes must require independent bootstrap")
    paths = control.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(v, str) and v for v in paths):
        raise ValueError("control_plane.paths must be a non-empty string list")
    if len(paths) != len(set(paths)):
        raise ValueError("control_plane.paths contains duplicates")
    if policy_path not in paths:
        raise ValueError("policy.policy_path must be part of the control plane")

    integration = _require_object(policy.get("full_integration"), "policy.full_integration")
    _reject_unknown_keys(integration, INTEGRATION_POLICY_KEYS, "policy.full_integration")
    workflow_path = _require_string(
        integration.get("workflow_path"), "policy.full_integration.workflow_path"
    )
    manifest_path = _require_string(
        integration.get("required_job_manifest_path"),
        "policy.full_integration.required_job_manifest_path",
    )
    for path, label in (
        (workflow_path, "full integration workflow"),
        (manifest_path, "required-job manifest"),
    ):
        if path not in paths:
            raise ValueError(f"{label} must be part of the control plane")

    events = integration.get("accepted_events")
    if not isinstance(events, list) or not events or not all(isinstance(v, str) and v for v in events):
        raise ValueError("full_integration.accepted_events must be a non-empty string list")
    if len(events) != len(set(events)):
        raise ValueError("full_integration.accepted_events contains duplicates")
    if integration.get("required_status") != "completed":
        raise ValueError("full integration must require completed status")
    if integration.get("required_conclusion") != "success":
        raise ValueError("full integration must require success")
    for key in ("require_exact_head", "require_exact_base", "require_no_required_job_skips"):
        if integration.get(key) is not True:
            raise ValueError(f"full_integration.{key} must be true")


def validate_observation(observation: dict[str, Any]) -> None:
    _reject_unknown_keys(observation, OBSERVATION_KEYS, "observation")
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
        _reject_unknown_keys(item, CONTROL_OBSERVATION_KEYS, f"observation.control_plane[{index}]")
        _require_string(item.get("path"), f"observation.control_plane[{index}].path")
        for side in ("base_blob_sha", "candidate_blob_sha"):
            value = item.get(side)
            if value is not None:
                _require_sha(value, f"observation.control_plane[{index}].{side}")

    integration = observation.get("full_integration")
    if integration is None:
        return
    integration = _require_object(integration, "observation.full_integration")
    _reject_unknown_keys(integration, INTEGRATION_OBSERVATION_KEYS, "observation.full_integration")
    _require_string(integration.get("workflow_path"), "observation.full_integration.workflow_path")
    _require_sha(integration.get("workflow_blob_sha"), "observation.full_integration.workflow_blob_sha")
    _require_positive_int(integration.get("run_id"), "observation.full_integration.run_id")
    _require_positive_int(integration.get("run_attempt"), "observation.full_integration.run_attempt")
    _require_string(integration.get("event"), "observation.full_integration.event")
    for field in ("head_sha", "base_sha"):
        _require_sha(integration.get(field), f"observation.full_integration.{field}")
    if not isinstance(integration.get("job_census_complete"), bool):
        raise ValueError("observation.full_integration.job_census_complete must be boolean")
    jobs = integration.get("job_census")
    if not isinstance(jobs, list):
        raise ValueError("observation.full_integration.job_census must be a list")
    for index, job in enumerate(jobs):
        job = _require_object(job, f"observation.full_integration.job_census[{index}]")
        _reject_unknown_keys(job, JOB_OBSERVATION_KEYS, f"observation.full_integration.job_census[{index}]")
        _require_positive_int(job.get("job_id"), f"observation.full_integration.job_census[{index}].job_id")
        _require_string(job.get("name"), f"observation.full_integration.job_census[{index}].name")
        if not isinstance(job.get("skipped"), bool):
            raise ValueError(f"observation.full_integration.job_census[{index}].skipped must be boolean")


def validate_required_job_manifest(manifest: dict[str, Any]) -> None:
    _reject_unknown_keys(manifest, MANIFEST_KEYS, "required_job_manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected required-job manifest schema")
    _require_string(manifest.get("workflow_path"), "required_job_manifest.workflow_path")
    _require_sha(manifest.get("workflow_blob_sha"), "required_job_manifest.workflow_blob_sha")
    if not isinstance(manifest.get("complete"), bool):
        raise ValueError("required_job_manifest.complete must be boolean")
    profiles = _require_object(manifest.get("profiles"), "required_job_manifest.profiles")
    if not profiles:
        raise ValueError("required_job_manifest.profiles must not be empty")

    for profile_name, profile_value in profiles.items():
        _require_string(profile_name, "required_job_manifest profile name")
        profile = _require_object(profile_value, f"required_job_manifest.profiles[{profile_name}]")
        _reject_unknown_keys(profile, PROFILE_KEYS, f"required_job_manifest.profiles[{profile_name}]")
        event = _require_string(profile.get("event"), f"required_job_manifest.profiles[{profile_name}].event")
        if event != profile_name:
            raise ValueError(f"required_job_manifest profile {profile_name} must use matching event")
        top_ids = profile.get("top_level_job_ids")
        if not isinstance(top_ids, list) or not all(isinstance(v, str) and v for v in top_ids):
            raise ValueError(f"required_job_manifest profile {profile_name} has invalid top_level_job_ids")
        if len(top_ids) != len(set(top_ids)):
            raise ValueError(f"required_job_manifest profile {profile_name} duplicates top_level_job_ids")

        families = profile.get("families")
        if not isinstance(families, list):
            raise ValueError(f"required_job_manifest profile {profile_name} families must be a list")
        family_ids: list[str] = []
        for index, family_value in enumerate(families):
            family = _require_object(family_value, f"required_job_manifest.profiles[{profile_name}].families[{index}]")
            _reject_unknown_keys(family, FAMILY_KEYS, f"required_job_manifest.profiles[{profile_name}].families[{index}]")
            job_id = _require_string(family.get("job_id"), f"required_job_manifest.profiles[{profile_name}].families[{index}].job_id")
            family_ids.append(job_id)
            pattern = _require_string(family.get("api_name_regex"), f"required_job_manifest.profiles[{profile_name}].families[{index}].api_name_regex")
            if not pattern.startswith("^") or not pattern.endswith("$"):
                raise ValueError(f"required-job regex for {job_id} must be explicitly anchored")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid required-job regex for {job_id}: {exc}") from exc
            minimum = family.get("min_instances")
            maximum = family.get("max_instances")
            if (
                not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or minimum < 0
                or not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < 0
                or minimum > maximum
            ):
                raise ValueError(f"invalid instance bounds for required-job family {job_id}")
            disposition = family.get("required_disposition")
            if disposition not in ("success", "allowed_skip"):
                raise ValueError(f"invalid required disposition for {job_id}")
            if disposition == "success" and minimum < 1:
                raise ValueError(f"success-required family {job_id} must require at least one instance")
        if len(family_ids) != len(set(family_ids)):
            raise ValueError(f"required_job_manifest profile {profile_name} duplicates family job_id")
        if set(family_ids) != set(top_ids):
            raise ValueError(f"required_job_manifest profile {profile_name} family IDs must exactly match top_level_job_ids")


def _control_plane_disposition(policy: dict[str, Any], observation: dict[str, Any]) -> tuple[Decision | None, list[str], dict[str, str | None]]:
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
        return Decision.BOOTSTRAP_REQUIRED, ["candidate changes merge-authority control plane: " + ", ".join(sorted(changed))], base_blobs
    return None, [], base_blobs


def _job_result_disposition(job: dict[str, Any], required_disposition: str) -> Decision | None:
    status = job.get("status")
    conclusion = job.get("conclusion")
    skipped = job.get("skipped") is True or conclusion == "skipped"
    if required_disposition == "allowed_skip" and skipped and status == "completed":
        return None
    if skipped or status != "completed":
        return Decision.INCOMPLETE
    if conclusion == "success":
        return None
    if conclusion in (None, "cancelled", "neutral", "stale"):
        return Decision.INCOMPLETE
    return Decision.REJECTED


def _manifest_job_disposition(manifest: dict[str, Any], event: str, jobs: list[dict[str, Any]]) -> tuple[Decision | None, list[str], list[dict[str, Any]]]:
    if manifest.get("complete") is not True:
        return Decision.INCOMPLETE, ["trusted required-job manifest is not marked qualification-complete"], []
    profiles = manifest["profiles"]
    profile = profiles.get(event)
    if not isinstance(profile, dict):
        return Decision.REJECTED, [f"trusted required-job manifest has no profile for event {event!r}"], []

    by_job_id: dict[int, dict[str, Any]] = {}
    duplicate_ids: set[int] = set()
    for job in jobs:
        job_id = job["job_id"]
        if job_id in by_job_id:
            duplicate_ids.add(job_id)
        by_job_id[job_id] = job
    if duplicate_ids:
        return Decision.REJECTED, ["duplicate GitHub job ids in census: " + ", ".join(str(v) for v in sorted(duplicate_ids))], []

    families = profile["families"]
    compiled = [(family, re.compile(family["api_name_regex"])) for family in families]
    matched_by_family: dict[str, list[dict[str, Any]]] = {family["job_id"]: [] for family in families}
    unknown_jobs: list[str] = []
    ambiguous_jobs: list[str] = []
    for job in jobs:
        matches = [family["job_id"] for family, pattern in compiled if pattern.fullmatch(job["name"]) is not None]
        if not matches:
            unknown_jobs.append(f"{job['job_id']}:{job['name']}")
        elif len(matches) > 1:
            ambiguous_jobs.append(f"{job['job_id']}:{job['name']}=>{','.join(sorted(matches))}")
        else:
            matched_by_family[matches[0]].append(job)
    if unknown_jobs or ambiguous_jobs:
        reasons: list[str] = []
        if unknown_jobs:
            reasons.append("job census contains jobs outside trusted manifest: " + "; ".join(sorted(unknown_jobs)))
        if ambiguous_jobs:
            reasons.append("job census contains ambiguously classified jobs: " + "; ".join(sorted(ambiguous_jobs)))
        return Decision.REJECTED, reasons, []

    required_jobs: list[dict[str, Any]] = []
    cardinality_errors: list[str] = []
    incomplete: list[str] = []
    failed: list[str] = []
    for family in families:
        family_id = family["job_id"]
        matches = matched_by_family[family_id]
        count = len(matches)
        minimum = family["min_instances"]
        maximum = family["max_instances"]
        if count < minimum or count > maximum:
            cardinality_errors.append(f"{family_id}: observed {count}, expected {minimum}..{maximum}")
            continue
        for job in matches:
            required_jobs.append(job)
            disposition = _job_result_disposition(job, family["required_disposition"])
            if disposition is Decision.INCOMPLETE:
                incomplete.append(f"{family_id}/{job['name']}:{job.get('status')}/{job.get('conclusion')}")
            elif disposition is Decision.REJECTED:
                failed.append(f"{family_id}/{job['name']}:{job.get('status')}/{job.get('conclusion')}")
    if cardinality_errors:
        return Decision.INCOMPLETE, ["required-job manifest cardinality not satisfied: " + "; ".join(sorted(cardinality_errors))], required_jobs
    if failed:
        return Decision.REJECTED, ["required jobs did not succeed: " + "; ".join(sorted(failed))], required_jobs
    if incomplete:
        return Decision.INCOMPLETE, ["required jobs are not complete successes: " + "; ".join(sorted(incomplete))], required_jobs
    return None, [], required_jobs


def _normalized_job_census(observation: dict[str, Any]) -> list[dict[str, Any]]:
    integration = observation.get("full_integration")
    if not isinstance(integration, dict):
        return []
    jobs = integration.get("job_census")
    if not isinstance(jobs, list):
        return []
    return sorted(
        ({"job_id": job.get("job_id"), "name": job.get("name"), "status": job.get("status"), "conclusion": job.get("conclusion"), "skipped": job.get("skipped")} for job in jobs if isinstance(job, dict)),
        key=lambda job: (int(job.get("job_id") or 0), str(job.get("name"))),
    )


def _evidence_binding(observation: dict[str, Any], manifest_sha256: str, manifest_blob_sha: str, required_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    control_rows = sorted(
        ({"path": row.get("path"), "base_blob_sha": row.get("base_blob_sha"), "candidate_blob_sha": row.get("candidate_blob_sha")} for row in observation.get("control_plane", []) if isinstance(row, dict)),
        key=lambda row: str(row.get("path")),
    )
    integration = observation.get("full_integration")
    if not isinstance(integration, dict):
        integration_binding = None
    else:
        census = _normalized_job_census(observation)
        required_normalized = sorted(required_jobs, key=lambda job: (int(job["job_id"]), str(job["name"])))
        integration_binding = {
            "workflow_path": integration.get("workflow_path"),
            "workflow_blob_sha": integration.get("workflow_blob_sha"),
            "run_id": integration.get("run_id"),
            "run_attempt": integration.get("run_attempt"),
            "event": integration.get("event"),
            "status": integration.get("status"),
            "conclusion": integration.get("conclusion"),
            "head_sha": integration.get("head_sha"),
            "base_sha": integration.get("base_sha"),
            "job_census_complete": integration.get("job_census_complete"),
            "job_census_count": len(census),
            "job_census_sha256": sha256_hex(canonical_json(census)),
            "required_job_observation_count": len(required_normalized),
            "required_jobs_sha256": sha256_hex(canonical_json(required_normalized)),
            "required_job_manifest_sha256": manifest_sha256,
            "required_job_manifest_blob_sha": manifest_blob_sha,
        }
    binding = {"control_plane": control_rows, "full_integration": integration_binding}
    return {"sha256": sha256_hex(canonical_json(binding)), "value": binding}


def evaluate(policy: dict[str, Any], observation: dict[str, Any], required_job_manifest: dict[str, Any], *, policy_bytes: bytes | None = None, manifest_bytes: bytes | None = None) -> Evaluation:
    validate_policy(policy)
    validate_observation(observation)
    validate_required_job_manifest(required_job_manifest)

    policy_raw = policy_bytes if policy_bytes is not None else canonical_json(policy)
    manifest_raw = manifest_bytes if manifest_bytes is not None else canonical_json(required_job_manifest)
    policy_digest = sha256_hex(policy_raw)
    policy_blob = git_blob_sha(policy_raw)
    manifest_digest = sha256_hex(manifest_raw)
    manifest_blob = git_blob_sha(manifest_raw)

    reasons: list[str] = []
    if observation["repository"] != policy["repository"]:
        reasons.append("repository identity does not match policy")
    if observation["target_branch"] != policy["target_branch"]:
        reasons.append("target branch does not match policy")
    if reasons:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, reasons, [])

    control_decision, control_reasons, base_blobs = _control_plane_disposition(policy, observation)
    if control_decision is not None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, control_decision, control_reasons, [])

    policy_path = policy["policy_path"]
    policy_base_blob = base_blobs.get(policy_path)
    if policy_base_blob is None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["trusted merge-admission policy is absent from target-base control plane"], [])
    if policy_blob != policy_base_blob:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["loaded merge-admission policy bytes do not equal target-base policy blob"], [])

    required = policy["full_integration"]
    workflow_path = required["workflow_path"]
    manifest_path = required["required_job_manifest_path"]
    workflow_base_blob = base_blobs.get(workflow_path)
    manifest_base_blob = base_blobs.get(manifest_path)
    if workflow_base_blob is None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["trusted full-integration workflow is absent from target-base control plane"], [])
    if manifest_base_blob is None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["trusted required-job manifest is absent from target-base control plane"], [])
    if manifest_blob != manifest_base_blob:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["loaded required-job manifest bytes do not equal target-base manifest blob"], [])
    if required_job_manifest["workflow_path"] != workflow_path:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["required-job manifest names the wrong workflow path"], [])
    if required_job_manifest["workflow_blob_sha"] != workflow_base_blob:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, ["required-job manifest is not bound to the trusted target-base workflow blob"], [])

    integration = observation.get("full_integration")
    if integration is None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.INCOMPLETE, ["full integration evidence is absent; Tier-1/focused evidence cannot substitute"], [])

    identity_errors: list[str] = []
    if integration.get("workflow_path") != workflow_path:
        identity_errors.append("workflow path is not the trusted full-integration workflow")
    if integration.get("workflow_blob_sha") != workflow_base_blob:
        identity_errors.append("workflow blob does not equal trusted target-base workflow blob")
    if integration.get("event") not in set(required["accepted_events"]):
        identity_errors.append("workflow event is not admitted by policy")
    if identity_errors:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.REJECTED, identity_errors, [])

    stale: list[str] = []
    if integration.get("head_sha") != observation["candidate_head_sha"]:
        stale.append("full integration head does not equal current candidate head")
    if integration.get("base_sha") != observation["current_base_sha"]:
        stale.append("full integration base does not equal current target base")
    if stale:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.STALE, stale, [])

    status = integration.get("status")
    conclusion = integration.get("conclusion")
    if status != required["required_status"]:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.INCOMPLETE, [f"full integration status is {status!r}, not completed"], [])
    if conclusion != required["required_conclusion"]:
        decision = Decision.INCOMPLETE if conclusion in (None, "cancelled", "skipped", "neutral", "stale") else Decision.REJECTED
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, decision, [f"full integration conclusion is {conclusion!r}, not success"], [])
    if integration.get("job_census_complete") is not True:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.INCOMPLETE, ["trusted collector has not established complete GitHub job pagination/census"], [])

    census = integration.get("job_census")
    assert isinstance(census, list)
    job_decision, job_reasons, required_jobs = _manifest_job_disposition(required_job_manifest, integration["event"], census)
    if job_decision is not None:
        return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, job_decision, job_reasons, required_jobs)
    return _finish(policy_digest, policy_blob, manifest_digest, manifest_blob, observation, Decision.ADMITTED, ["all v2 admission predicates satisfied"], required_jobs)


def _finish(policy_digest: str, policy_blob_sha: str, manifest_digest: str, manifest_blob_sha: str, observation: dict[str, Any], decision: Decision, reasons: list[str], required_jobs: list[dict[str, Any]]) -> Evaluation:
    evidence_binding = _evidence_binding(observation, manifest_digest, manifest_blob_sha, required_jobs)
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "policy_sha256": policy_digest,
        "policy_blob_sha": policy_blob_sha,
        "enforcement_ready": False,
        "repository": observation.get("repository"),
        "target_branch": observation.get("target_branch"),
        "current_base_sha": observation.get("current_base_sha"),
        "candidate_head_sha": observation.get("candidate_head_sha"),
        "candidate_tree_sha": observation.get("candidate_tree_sha"),
        "evidence_binding_sha256": evidence_binding["sha256"],
        "evidence_binding": evidence_binding["value"],
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
    parser.add_argument("--policy", default="scripts/ci/merge_admission_policy_v2.json")
    parser.add_argument("--required-job-manifest", default="scripts/ci/required_ci_job_manifest_v1.json")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        policy, policy_bytes = load_json_object(pathlib.Path(args.policy), "policy")
        manifest, manifest_bytes = load_json_object(pathlib.Path(args.required_job_manifest), "required-job manifest")
        observation, _ = load_json_object(pathlib.Path(args.observation), "observation")
        result = evaluate(policy, observation, manifest, policy_bytes=policy_bytes, manifest_bytes=manifest_bytes)
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
