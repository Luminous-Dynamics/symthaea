#!/usr/bin/env python3
"""Pure, fail-closed merge-admission policy evaluator.

This module does not call GitHub, merge code, or emit a GitHub status. It accepts
an observation assembled by a trusted caller, evaluates it against explicit
base-owned policy + required-job manifest inputs, and emits a content-addressed
*unsigned* disposition receipt.

The v1 trust theorem is intentionally narrow:

    candidate-owned CI result != merge authority

For ordinary admission, the candidate must preserve the target-base CI control
plane byte-for-byte. A candidate that changes the authority/control plane is
classified BOOTSTRAP_REQUIRED and cannot self-authorize with the machinery it is
changing.

The evaluator also distinguishes two different completeness claims:

    complete GitHub job census
        !=
    trusted required-job manifest satisfied

The collector supplies the exhaustive census. The pure evaluator derives
manifest satisfaction itself.
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
MANIFEST_SCHEMA = "symthaea.required-ci-job-manifest.v1"
HEX_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

POLICY_KEYS = {
    "schema",
    "policy_path",
    "repository",
    "target_branch",
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


@dataclass(frozen=True)
class ManifestSatisfaction:
    profile: str
    family_count: int
    required_job_count: int
    required_jobs_sha256: str
    family_summary_sha256: str
    value: dict[str, Any]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_oid(data: bytes, *, hex_length: int) -> str:
    payload = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    if hex_length == 40:
        return hashlib.sha1(payload).hexdigest()
    if hex_length == 64:
        return hashlib.sha256(payload).hexdigest()
    raise ValueError("unsupported git object-id width")


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


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")


def validate_policy(policy: dict[str, Any]) -> None:
    _reject_unknown_keys(policy, POLICY_KEYS, "policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("unexpected merge admission policy schema")
    policy_path = _require_string(policy.get("policy_path"), "policy.policy_path")
    _require_string(policy.get("repository"), "policy.repository")
    _require_string(policy.get("target_branch"), "policy.target_branch")
    if policy.get("enforcement_ready") is not False:
        raise ValueError("v1 is an executable policy core and must remain enforcement_ready=false")
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
        raise ValueError("v1 requires exact_base_equivalence control-plane mode")
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
    if workflow_path not in paths:
        raise ValueError("full integration workflow must be part of the control plane")
    if manifest_path not in paths:
        raise ValueError("required job manifest must be part of the control plane")
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


def validate_manifest(manifest: dict[str, Any]) -> None:
    _reject_unknown_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected required CI job manifest schema")
    _require_string(manifest.get("workflow_path"), "manifest.workflow_path")
    _require_sha(manifest.get("workflow_blob_sha"), "manifest.workflow_blob_sha")
    _require_bool(manifest.get("complete"), "manifest.complete")
    profiles = _require_object(manifest.get("profiles"), "manifest.profiles")
    if not profiles:
        raise ValueError("manifest.profiles must not be empty")

    for profile_name, profile_value in profiles.items():
        _require_string(profile_name, "manifest profile name")
        profile = _require_object(profile_value, f"manifest.profiles[{profile_name!r}]")
        _reject_unknown_keys(profile, PROFILE_KEYS, f"manifest.profiles[{profile_name!r}]")
        event = _require_string(
            profile.get("event"), f"manifest.profiles[{profile_name!r}].event"
        )
        if profile_name != event:
            raise ValueError("v1 manifest profile key must equal its event")
        job_ids = profile.get("top_level_job_ids")
        if not isinstance(job_ids, list) or not all(
            isinstance(job_id, str) and job_id for job_id in job_ids
        ):
            raise ValueError(f"manifest profile {profile_name!r} has invalid top_level_job_ids")
        if len(job_ids) != len(set(job_ids)):
            raise ValueError(f"manifest profile {profile_name!r} repeats a top-level job id")
        families = profile.get("families")
        if not isinstance(families, list):
            raise ValueError(f"manifest profile {profile_name!r} families must be a list")
        family_ids: list[str] = []
        for index, family_value in enumerate(families):
            family = _require_object(
                family_value, f"manifest.profiles[{profile_name!r}].families[{index}]"
            )
            _reject_unknown_keys(
                family,
                FAMILY_KEYS,
                f"manifest.profiles[{profile_name!r}].families[{index}]",
            )
            family_id = _require_string(
                family.get("job_id"),
                f"manifest.profiles[{profile_name!r}].families[{index}].job_id",
            )
            family_ids.append(family_id)
            pattern = _require_string(
                family.get("api_name_regex"),
                f"manifest.profiles[{profile_name!r}].families[{index}].api_name_regex",
            )
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"manifest profile {profile_name!r} has invalid API name regex: {exc}"
                ) from exc
            minimum = family.get("min_instances")
            maximum = family.get("max_instances")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
                raise ValueError("manifest family min_instances must be a non-negative integer")
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < minimum:
                raise ValueError("manifest family max_instances must be >= min_instances")
            if family.get("required_disposition") not in {"success", "allowed_skip"}:
                raise ValueError("manifest family has unknown required_disposition")
        if len(family_ids) != len(set(family_ids)):
            raise ValueError(f"manifest profile {profile_name!r} repeats a family job_id")
        if set(family_ids) - set(job_ids):
            raise ValueError("manifest family references a job_id absent from top_level_job_ids")
        if manifest["complete"] is True and set(family_ids) != set(job_ids):
            raise ValueError(
                "complete manifest profiles must account for every top-level job id exactly once"
            )


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
        _reject_unknown_keys(
            item,
            CONTROL_OBSERVATION_KEYS,
            f"observation.control_plane[{index}]",
        )
        _require_string(item.get("path"), f"observation.control_plane[{index}].path")
        for side in ("base_blob_sha", "candidate_blob_sha"):
            value = item.get(side)
            if value is not None:
                _require_sha(value, f"observation.control_plane[{index}].{side}")

    integration = observation.get("full_integration")
    if integration is None:
        return
    integration = _require_object(integration, "observation.full_integration")
    _reject_unknown_keys(
        integration,
        INTEGRATION_OBSERVATION_KEYS,
        "observation.full_integration",
    )
    _require_string(integration.get("workflow_path"), "observation.full_integration.workflow_path")
    workflow_blob = integration.get("workflow_blob_sha")
    if workflow_blob is not None:
        _require_sha(workflow_blob, "observation.full_integration.workflow_blob_sha")
    _require_positive_int(integration.get("run_id"), "observation.full_integration.run_id")
    _require_positive_int(
        integration.get("run_attempt"), "observation.full_integration.run_attempt"
    )
    _require_bool(
        integration.get("job_census_complete"),
        "observation.full_integration.job_census_complete",
    )
    for field in ("head_sha", "base_sha"):
        value = integration.get(field)
        if value is not None:
            _require_sha(value, f"observation.full_integration.{field}")
    jobs = integration.get("job_census")
    if not isinstance(jobs, list):
        raise ValueError("observation.full_integration.job_census must be a list")
    seen_job_ids: set[int] = set()
    for index, job_value in enumerate(jobs):
        job = _require_object(job_value, f"observation.full_integration.job_census[{index}]")
        _reject_unknown_keys(
            job,
            JOB_OBSERVATION_KEYS,
            f"observation.full_integration.job_census[{index}]",
        )
        job_id = _require_positive_int(
            job.get("job_id"), f"observation.full_integration.job_census[{index}].job_id"
        )
        if job_id in seen_job_ids:
            raise ValueError("observation.full_integration.job_census contains duplicate job_id")
        seen_job_ids.add(job_id)
        _require_string(
            job.get("name"), f"observation.full_integration.job_census[{index}].name"
        )
        _require_bool(
            job.get("skipped"), f"observation.full_integration.job_census[{index}].skipped"
        )


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
        return (
            Decision.REJECTED,
            ["duplicate control-plane observations: " + ", ".join(sorted(duplicates))],
            {},
        )

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


def _artifact_matches_base_blob(
    *, artifact_bytes: bytes, path: str, base_blobs: dict[str, str | None]
) -> bool:
    expected = base_blobs.get(path)
    if expected is None:
        return False
    return git_blob_oid(artifact_bytes, hex_length=len(expected)) == expected


def _job_disposition(job: dict[str, Any], required_disposition: str) -> Decision | None:
    status = job.get("status")
    conclusion = job.get("conclusion")
    skipped = job.get("skipped") is True or conclusion == "skipped"
    if required_disposition == "allowed_skip" and skipped:
        return None
    if skipped or status != "completed":
        return Decision.INCOMPLETE
    if conclusion != "success":
        return Decision.REJECTED
    return None


def _manifest_satisfaction(
    manifest: dict[str, Any],
    *,
    event: str,
    jobs: list[dict[str, Any]],
) -> tuple[Decision | None, list[str], ManifestSatisfaction | None]:
    if manifest.get("complete") is not True:
        return Decision.INCOMPLETE, ["trusted required-job manifest is not complete"], None

    profile_value = manifest["profiles"].get(event)
    if profile_value is None:
        return Decision.REJECTED, [f"trusted manifest has no profile for event {event!r}"], None
    profile = _require_object(profile_value, f"manifest.profiles[{event!r}]")
    families = profile["families"]

    match_by_job: dict[int, list[int]] = {job["job_id"]: [] for job in jobs}
    family_summaries: list[dict[str, Any]] = []
    required_jobs: list[dict[str, Any]] = []
    reasons_incomplete: list[str] = []
    reasons_rejected: list[str] = []

    for family_index, family in enumerate(families):
        pattern = re.compile(family["api_name_regex"])
        matched = [job for job in jobs if pattern.fullmatch(job["name"]) is not None]
        for job in matched:
            match_by_job[job["job_id"]].append(family_index)

        count = len(matched)
        minimum = family["min_instances"]
        maximum = family["max_instances"]
        if count < minimum:
            reasons_incomplete.append(
                f"required job family {family['job_id']!r} has {count} instances; "
                f"minimum is {minimum}"
            )
        if count > maximum:
            reasons_rejected.append(
                f"required job family {family['job_id']!r} has {count} instances; "
                f"maximum is {maximum}"
            )

        family_decisions = [
            _job_disposition(job, family["required_disposition"]) for job in matched
        ]
        if Decision.REJECTED in family_decisions:
            reasons_rejected.append(
                f"required job family {family['job_id']!r} contains a failed instance"
            )
        elif Decision.INCOMPLETE in family_decisions:
            reasons_incomplete.append(
                f"required job family {family['job_id']!r} contains an incomplete/skipped instance"
            )

        required_jobs.extend(matched)
        family_summaries.append(
            {
                "job_id": family["job_id"],
                "matched_job_ids": sorted(job["job_id"] for job in matched),
                "matched_count": count,
                "required_disposition": family["required_disposition"],
            }
        )

    ambiguous = sorted(job_id for job_id, matches in match_by_job.items() if len(matches) > 1)
    unmatched = sorted(job_id for job_id, matches in match_by_job.items() if not matches)
    if ambiguous:
        reasons_rejected.append(
            "job census contains jobs matching multiple manifest families: "
            + ", ".join(str(v) for v in ambiguous)
        )
    if unmatched:
        reasons_rejected.append(
            "job census contains jobs absent from the complete manifest profile: "
            + ", ".join(str(v) for v in unmatched)
        )

    if reasons_rejected:
        return Decision.REJECTED, reasons_rejected + reasons_incomplete, None
    if reasons_incomplete:
        return Decision.INCOMPLETE, reasons_incomplete, None

    required_jobs_sorted = sorted(required_jobs, key=lambda job: job["job_id"])
    family_summaries_sorted = sorted(family_summaries, key=lambda row: row["job_id"])
    value = {
        "profile": event,
        "family_count": len(family_summaries_sorted),
        "required_job_count": len(required_jobs_sorted),
        "required_jobs_sha256": sha256_hex(canonical_json(required_jobs_sorted)),
        "family_summary_sha256": sha256_hex(canonical_json(family_summaries_sorted)),
    }
    return (
        None,
        [],
        ManifestSatisfaction(
            profile=event,
            family_count=value["family_count"],
            required_job_count=value["required_job_count"],
            required_jobs_sha256=value["required_jobs_sha256"],
            family_summary_sha256=value["family_summary_sha256"],
            value=value,
        ),
    )


def _evidence_binding(
    observation: dict[str, Any],
    *,
    manifest_sha256: str,
    manifest_satisfaction: ManifestSatisfaction | None,
) -> dict[str, Any]:
    control_rows = sorted(
        (
            {
                "path": row.get("path"),
                "base_blob_sha": row.get("base_blob_sha"),
                "candidate_blob_sha": row.get("candidate_blob_sha"),
            }
            for row in observation.get("control_plane", [])
            if isinstance(row, dict)
        ),
        key=lambda row: str(row.get("path")),
    )
    integration = observation.get("full_integration")
    if not isinstance(integration, dict):
        integration_binding = None
    else:
        jobs = integration.get("job_census")
        jobs_for_digest = jobs if isinstance(jobs, list) else []
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
            "job_census_count": len(jobs_for_digest),
            "job_census_sha256": sha256_hex(canonical_json(jobs_for_digest)),
            "required_job_manifest_sha256": manifest_sha256,
            "manifest_satisfaction": (
                manifest_satisfaction.value if manifest_satisfaction is not None else None
            ),
        }
    binding = {
        "control_plane": control_rows,
        "full_integration": integration_binding,
    }
    return {
        "sha256": sha256_hex(canonical_json(binding)),
        "value": binding,
    }


def evaluate(
    policy: dict[str, Any],
    manifest: dict[str, Any],
    observation: dict[str, Any],
    *,
    policy_bytes: bytes | None = None,
    manifest_bytes: bytes | None = None,
) -> Evaluation:
    validate_policy(policy)
    validate_manifest(manifest)
    validate_observation(observation)

    exact_policy_bytes = policy_bytes if policy_bytes is not None else canonical_json(policy)
    exact_manifest_bytes = (
        manifest_bytes if manifest_bytes is not None else canonical_json(manifest)
    )
    policy_digest = sha256_hex(exact_policy_bytes)
    manifest_digest = sha256_hex(exact_manifest_bytes)
    reasons: list[str] = []

    if observation["repository"] != policy["repository"]:
        reasons.append("repository identity does not match policy")
    if observation["target_branch"] != policy["target_branch"]:
        reasons.append("target branch does not match policy")
    if reasons:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            reasons,
        )

    control_decision, control_reasons, base_blobs = _control_plane_disposition(
        policy, observation
    )
    if control_decision is not None:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            control_decision,
            control_reasons,
        )

    policy_path = policy["policy_path"]
    manifest_path = policy["full_integration"]["required_job_manifest_path"]
    if not _artifact_matches_base_blob(
        artifact_bytes=exact_policy_bytes, path=policy_path, base_blobs=base_blobs
    ):
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            ["policy bytes do not equal the trusted target-base policy blob"],
        )
    if not _artifact_matches_base_blob(
        artifact_bytes=exact_manifest_bytes, path=manifest_path, base_blobs=base_blobs
    ):
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            ["required-job manifest bytes do not equal the trusted target-base manifest blob"],
        )

    integration = observation.get("full_integration")
    if integration is None:
        return _finish(
            policy_digest,
            manifest_digest,
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
            manifest_digest,
            observation,
            Decision.REJECTED,
            ["trusted full-integration workflow is absent from target-base control plane"],
        )

    manifest_errors: list[str] = []
    if manifest["workflow_path"] != workflow_path:
        manifest_errors.append("required-job manifest names the wrong workflow path")
    if manifest["workflow_blob_sha"] != workflow_base_blob:
        manifest_errors.append("required-job manifest binds the wrong workflow blob")
    if manifest_errors:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            manifest_errors,
        )

    identity_errors: list[str] = []
    if integration.get("workflow_path") != workflow_path:
        identity_errors.append("workflow path is not the trusted full-integration workflow")
    if integration.get("workflow_blob_sha") != workflow_base_blob:
        identity_errors.append("workflow blob does not equal trusted target-base workflow blob")
    event = integration.get("event")
    if event not in set(required["accepted_events"]):
        identity_errors.append("workflow event is not admitted by policy")
    if identity_errors:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            identity_errors,
        )

    stale: list[str] = []
    if integration.get("head_sha") != observation["candidate_head_sha"]:
        stale.append("full integration head does not equal current candidate head")
    if integration.get("base_sha") != observation["current_base_sha"]:
        stale.append("full integration base does not equal current target base")
    if stale:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.STALE,
            stale,
        )

    status = integration.get("status")
    conclusion = integration.get("conclusion")
    if status != required["required_status"]:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.INCOMPLETE,
            [f"full integration status is {status!r}, not completed"],
        )
    if conclusion != required["required_conclusion"]:
        decision = (
            Decision.INCOMPLETE
            if conclusion in (None, "cancelled", "skipped", "neutral")
            else Decision.REJECTED
        )
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            decision,
            [f"full integration conclusion is {conclusion!r}, not success"],
        )

    if integration.get("job_census_complete") is not True:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.INCOMPLETE,
            ["trusted collector has not established a complete GitHub job census"],
        )

    jobs = integration.get("job_census")
    if not isinstance(jobs, list):
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            Decision.REJECTED,
            ["job census is malformed"],
        )
    manifest_decision, manifest_reasons, satisfaction = _manifest_satisfaction(
        manifest,
        event=str(event),
        jobs=jobs,
    )
    if manifest_decision is not None:
        return _finish(
            policy_digest,
            manifest_digest,
            observation,
            manifest_decision,
            manifest_reasons,
        )

    return _finish(
        policy_digest,
        manifest_digest,
        observation,
        Decision.ADMITTED,
        ["all v1 admission predicates satisfied"],
        manifest_satisfaction=satisfaction,
    )


def _finish(
    policy_digest: str,
    manifest_digest: str,
    observation: dict[str, Any],
    decision: Decision,
    reasons: list[str],
    *,
    manifest_satisfaction: ManifestSatisfaction | None = None,
) -> Evaluation:
    evidence_binding = _evidence_binding(
        observation,
        manifest_sha256=manifest_digest,
        manifest_satisfaction=manifest_satisfaction,
    )
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "policy_sha256": policy_digest,
        "required_job_manifest_sha256": manifest_digest,
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
        "caveat": (
            "unsigned policy-core disposition; enforceable authority requires a trusted "
            "external collector/check and repository merge rule"
        ),
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
    parser.add_argument("--manifest")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        policy_path = pathlib.Path(args.policy)
        policy, policy_bytes = load_json_object(policy_path, "policy")
        validate_policy(policy)
        if policy_path.as_posix() != policy["policy_path"]:
            raise ValueError("CLI policy path does not equal policy.policy_path")

        configured_manifest_path = policy["full_integration"]["required_job_manifest_path"]
        manifest_path = pathlib.Path(args.manifest or configured_manifest_path)
        if manifest_path.as_posix() != configured_manifest_path:
            raise ValueError(
                "CLI manifest path does not equal policy.full_integration.required_job_manifest_path"
            )
        manifest, manifest_bytes = load_json_object(manifest_path, "required job manifest")
        observation, _ = load_json_object(pathlib.Path(args.observation), "observation")
        result = evaluate(
            policy,
            manifest,
            observation,
            policy_bytes=policy_bytes,
            manifest_bytes=manifest_bytes,
        )
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
