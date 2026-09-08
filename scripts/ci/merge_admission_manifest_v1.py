#!/usr/bin/env python3
"""Closed-world required-job manifest utilities for merge admission.

The collector reports an API job census. This module, not the collector, decides
whether that census satisfies the trusted base-owned manifest.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

MANIFEST_SCHEMA = "symthaea.required-ci-job-manifest.v1"
JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


@dataclass(frozen=True)
class ManifestResult:
    complete: bool
    rejected: bool
    reasons: tuple[str, ...]
    required_jobs: tuple[dict[str, Any], ...]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git_blob_id(data: bytes, width: int = 40) -> str:
    payload = f"blob {len(data)}\0".encode("ascii") + data
    if width == 40:
        return hashlib.sha1(payload).hexdigest()
    if width == 64:
        return hashlib.sha256(payload).hexdigest()
    raise ValueError("unsupported Git object-id width")


def extract_top_level_job_ids(workflow_text: str) -> tuple[str, ...]:
    """Extract plain top-level job IDs from a GitHub Actions workflow.

    This intentionally understands only the conservative source shape currently
    used by ci.yml: `jobs:` at column zero and job IDs at exactly two spaces.
    A shape it cannot prove is refused rather than guessed.
    """
    in_jobs = False
    ids: list[str] = []
    for raw in workflow_text.splitlines():
        if not in_jobs:
            if raw == "jobs:":
                in_jobs = True
            continue
        if raw and not raw[0].isspace() and not raw.startswith("#"):
            break
        match = re.fullmatch(r"  ([A-Za-z0-9_-]+):\s*(?:#.*)?", raw)
        if match:
            ids.append(match.group(1))
    if not in_jobs:
        raise ValueError("workflow has no plain top-level jobs: section")
    if not ids:
        raise ValueError("workflow jobs section yielded no top-level job IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("workflow contains duplicate top-level job IDs")
    return tuple(ids)


def validate_manifest(manifest: dict[str, Any]) -> None:
    allowed = {"schema", "workflow_path", "workflow_blob_sha", "complete", "profiles"}
    unknown = sorted(set(manifest) - allowed)
    if unknown:
        raise ValueError("manifest contains unknown fields: " + ", ".join(unknown))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected required-job manifest schema")
    if not isinstance(manifest.get("workflow_path"), str) or not manifest["workflow_path"]:
        raise ValueError("manifest.workflow_path must be non-empty")
    blob = manifest.get("workflow_blob_sha")
    if not isinstance(blob, str) or not OBJECT_ID.fullmatch(blob):
        raise ValueError("manifest.workflow_blob_sha must be lowercase 40- or 64-hex")
    if not isinstance(manifest.get("complete"), bool):
        raise ValueError("manifest.complete must be boolean")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("manifest.profiles must be a non-empty object")
    for profile_name, profile in profiles.items():
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError("manifest profile names must be non-empty")
        if not isinstance(profile, dict):
            raise ValueError(f"profile {profile_name!r} must be an object")
        unknown_profile = sorted(set(profile) - {"event", "top_level_job_ids", "families"})
        if unknown_profile:
            raise ValueError(f"profile {profile_name!r} contains unknown fields: {', '.join(unknown_profile)}")
        if profile.get("event") != profile_name:
            raise ValueError(f"profile {profile_name!r} event must equal profile name")
        ids = profile.get("top_level_job_ids")
        families = profile.get("families")
        if not isinstance(ids, list) or not all(isinstance(v, str) and JOB_ID.fullmatch(v) for v in ids):
            raise ValueError(f"profile {profile_name!r} top_level_job_ids invalid")
        if len(ids) != len(set(ids)):
            raise ValueError(f"profile {profile_name!r} has duplicate top-level job IDs")
        if not isinstance(families, list):
            raise ValueError(f"profile {profile_name!r} families must be a list")
        seen_family_ids: set[str] = set()
        for index, family in enumerate(families):
            if not isinstance(family, dict):
                raise ValueError(f"profile {profile_name!r} family {index} must be an object")
            expected = {"job_id", "api_name_regex", "min_instances", "max_instances", "required_disposition"}
            if set(family) != expected:
                raise ValueError(f"profile {profile_name!r} family {index} has wrong fields")
            job_id = family["job_id"]
            if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
                raise ValueError(f"profile {profile_name!r} family {index} job_id invalid")
            if job_id in seen_family_ids:
                raise ValueError(f"profile {profile_name!r} repeats family job_id {job_id}")
            seen_family_ids.add(job_id)
            try:
                re.compile(family["api_name_regex"])
            except (TypeError, re.error) as exc:
                raise ValueError(f"profile {profile_name!r} family {job_id} regex invalid") from exc
            lo, hi = family["min_instances"], family["max_instances"]
            if not isinstance(lo, int) or isinstance(lo, bool) or lo < 0:
                raise ValueError(f"profile {profile_name!r} family {job_id} min_instances invalid")
            if not isinstance(hi, int) or isinstance(hi, bool) or hi < lo:
                raise ValueError(f"profile {profile_name!r} family {job_id} max_instances invalid")
            if family["required_disposition"] not in {"success", "allowed_skip"}:
                raise ValueError(f"profile {profile_name!r} family {job_id} disposition invalid")
        if manifest["complete"]:
            if not ids:
                raise ValueError(f"complete manifest profile {profile_name!r} has no top-level jobs")
            if set(ids) != seen_family_ids:
                raise ValueError(f"complete manifest profile {profile_name!r} must define exactly one family per top-level job")


def validate_manifest_against_workflow(manifest: dict[str, Any], workflow_bytes: bytes) -> None:
    validate_manifest(manifest)
    expected_blob = manifest["workflow_blob_sha"]
    actual_blob = git_blob_id(workflow_bytes, len(expected_blob))
    if actual_blob != expected_blob:
        raise ValueError("manifest workflow_blob_sha does not match workflow bytes")
    workflow_ids = set(extract_top_level_job_ids(workflow_bytes.decode("utf-8")))
    for profile_name, profile in manifest["profiles"].items():
        profile_ids = set(profile["top_level_job_ids"])
        if manifest["complete"] and profile_ids != workflow_ids:
            missing = sorted(workflow_ids - profile_ids)
            extra = sorted(profile_ids - workflow_ids)
            raise ValueError(
                f"complete profile {profile_name!r} does not equal workflow job IDs; missing={missing}, extra={extra}"
            )
        if not profile_ids.issubset(workflow_ids):
            raise ValueError(f"profile {profile_name!r} names top-level jobs absent from workflow")


def evaluate_job_census(manifest: dict[str, Any], event: str, jobs: list[dict[str, Any]]) -> ManifestResult:
    validate_manifest(manifest)
    if not manifest["complete"]:
        return ManifestResult(False, False, ("required-job manifest is not complete",), ())
    profile = manifest["profiles"].get(event)
    if profile is None:
        return ManifestResult(False, True, (f"no trusted required-job profile for event {event!r}",), ())
    if not isinstance(jobs, list) or not jobs:
        return ManifestResult(False, False, ("job census is absent or empty",), ())

    family_matches: dict[str, list[dict[str, Any]]] = {f["job_id"]: [] for f in profile["families"]}
    unmanifested: list[str] = []
    ambiguous: list[str] = []
    for job in jobs:
        name = job.get("name") if isinstance(job, dict) else None
        if not isinstance(name, str) or not name:
            return ManifestResult(False, True, ("job census contains an entry without a valid name",), ())
        matches = [f for f in profile["families"] if re.fullmatch(f["api_name_regex"], name)]
        if not matches:
            unmanifested.append(name)
        elif len(matches) > 1:
            ambiguous.append(name)
        else:
            family_matches[matches[0]["job_id"]].append(job)
    if unmanifested or ambiguous:
        reasons: list[str] = []
        if unmanifested:
            reasons.append("unmanifested jobs: " + ", ".join(sorted(unmanifested)))
        if ambiguous:
            reasons.append("jobs match multiple manifest families: " + ", ".join(sorted(ambiguous)))
        return ManifestResult(False, True, tuple(reasons), ())

    incomplete: list[str] = []
    failed: list[str] = []
    required_jobs: list[dict[str, Any]] = []
    for family in profile["families"]:
        matched = family_matches[family["job_id"]]
        count = len(matched)
        if count < family["min_instances"]:
            incomplete.append(f"{family['job_id']}: expected at least {family['min_instances']}, saw {count}")
            continue
        if count > family["max_instances"]:
            failed.append(f"{family['job_id']}: expected at most {family['max_instances']}, saw {count}")
            continue
        for job in matched:
            status = job.get("status")
            conclusion = job.get("conclusion")
            skipped = job.get("skipped") is True or conclusion == "skipped"
            disposition = family["required_disposition"]
            if status != "completed":
                incomplete.append(f"{job['name']}:{status}")
            elif conclusion == "failure" or conclusion in {"timed_out", "action_required", "startup_failure"}:
                failed.append(f"{job['name']}:{conclusion}")
            elif skipped:
                if disposition != "allowed_skip":
                    incomplete.append(f"{job['name']}:skipped")
            elif conclusion != "success":
                incomplete.append(f"{job['name']}:{conclusion}")
            required_jobs.append(job)
    if failed:
        return ManifestResult(False, True, ("required-job manifest violations: " + ", ".join(sorted(failed)),), tuple(required_jobs))
    if incomplete:
        return ManifestResult(False, False, ("required-job manifest incomplete: " + ", ".join(sorted(incomplete)),), tuple(required_jobs))
    return ManifestResult(True, False, ("job census satisfies trusted manifest",), tuple(required_jobs))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("manifest must be a JSON object")
    return parsed
