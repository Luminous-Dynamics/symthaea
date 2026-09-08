#!/usr/bin/env python3
"""Closed-world required-job manifest validation and census matching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

MANIFEST_SCHEMA = "symthaea.required-ci-job-manifest.v1"
JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
FAILURE_CONCLUSIONS = {"failure", "timed_out", "action_required", "startup_failure"}
INCOMPLETE_CONCLUSIONS = {None, "cancelled", "neutral", "skipped", "stale"}


class ManifestDisposition(str, Enum):
    SATISFIED = "satisfied"
    MANIFEST_INCOMPLETE = "manifest_incomplete"
    CENSUS_INCOMPLETE = "census_incomplete"
    MANIFEST_MISMATCH = "manifest_mismatch"
    JOB_FAILURE = "job_failure"


@dataclass(frozen=True)
class ManifestResult:
    disposition: ManifestDisposition
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
    if set(manifest) != {"schema", "workflow_path", "workflow_blob_sha", "complete", "profiles"}:
        raise ValueError("required-job manifest has wrong top-level fields")
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
        if not isinstance(profile_name, str) or not profile_name or not isinstance(profile, dict):
            raise ValueError("manifest profile identity invalid")
        if set(profile) != {"event", "top_level_job_ids", "families"}:
            raise ValueError(f"profile {profile_name!r} has wrong fields")
        if profile.get("event") != profile_name:
            raise ValueError(f"profile {profile_name!r} event must equal profile name")
        ids = profile.get("top_level_job_ids")
        families = profile.get("families")
        if not isinstance(ids, list) or not all(isinstance(v, str) and JOB_ID.fullmatch(v) for v in ids):
            raise ValueError(f"profile {profile_name!r} top_level_job_ids invalid")
        if len(ids) != len(set(ids)):
            raise ValueError(f"profile {profile_name!r} duplicates top-level job IDs")
        if not isinstance(families, list):
            raise ValueError(f"profile {profile_name!r} families must be a list")
        seen: set[str] = set()
        for index, family in enumerate(families):
            expected = {"job_id", "api_name_regex", "min_instances", "max_instances", "required_disposition"}
            if not isinstance(family, dict) or set(family) != expected:
                raise ValueError(f"profile {profile_name!r} family {index} has wrong fields")
            job_id = family["job_id"]
            if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id) or job_id in seen:
                raise ValueError(f"profile {profile_name!r} family {index} job_id invalid/duplicate")
            seen.add(job_id)
            pattern = family["api_name_regex"]
            if not isinstance(pattern, str) or not pattern.startswith("^") or not pattern.endswith("$"):
                raise ValueError(f"family {job_id} regex must be explicitly anchored")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"family {job_id} regex invalid") from exc
            lo, hi = family["min_instances"], family["max_instances"]
            if not isinstance(lo, int) or isinstance(lo, bool) or lo < 0:
                raise ValueError(f"family {job_id} min_instances invalid")
            if not isinstance(hi, int) or isinstance(hi, bool) or hi < lo:
                raise ValueError(f"family {job_id} max_instances invalid")
            disposition = family["required_disposition"]
            if disposition not in {"success", "allowed_skip"}:
                raise ValueError(f"family {job_id} disposition invalid")
            if disposition == "success" and lo < 1:
                raise ValueError(f"success-required family {job_id} must require at least one instance")
        if manifest["complete"]:
            if not ids:
                raise ValueError(f"complete profile {profile_name!r} has no top-level jobs")
            if set(ids) != seen:
                raise ValueError(f"complete profile {profile_name!r} must define exactly one family per top-level job")


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
            raise ValueError(f"complete profile {profile_name!r} does not equal workflow job IDs; missing={missing}, extra={extra}")
        if not profile_ids.issubset(workflow_ids):
            raise ValueError(f"profile {profile_name!r} names top-level jobs absent from workflow")


def evaluate_job_census(manifest: dict[str, Any], event: str, jobs: list[dict[str, Any]]) -> ManifestResult:
    validate_manifest(manifest)
    if not manifest["complete"]:
        return ManifestResult(ManifestDisposition.MANIFEST_INCOMPLETE, ("required-job manifest is not complete",), ())
    profile = manifest["profiles"].get(event)
    if profile is None:
        return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, (f"no trusted required-job profile for event {event!r}",), ())
    if not jobs:
        return ManifestResult(ManifestDisposition.CENSUS_INCOMPLETE, ("job census is empty",), ())

    by_id: dict[int, dict[str, Any]] = {}
    duplicate_ids: set[int] = set()
    for job in jobs:
        job_id = job.get("job_id") if isinstance(job, dict) else None
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, ("job census contains invalid job_id",), ())
        if job_id in by_id:
            duplicate_ids.add(job_id)
        by_id[job_id] = job
    if duplicate_ids:
        return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, ("duplicate GitHub job ids: " + ", ".join(str(v) for v in sorted(duplicate_ids)),), ())

    family_matches: dict[str, list[dict[str, Any]]] = {f["job_id"]: [] for f in profile["families"]}
    unmanifested: list[str] = []
    ambiguous: list[str] = []
    compiled = [(f, re.compile(f["api_name_regex"])) for f in profile["families"]]
    for job in jobs:
        name = job.get("name")
        if not isinstance(name, str) or not name:
            return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, ("job census contains invalid name",), ())
        matches = [f for f, pattern in compiled if pattern.fullmatch(name)]
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
        return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, tuple(reasons), ())

    incomplete: list[str] = []
    failures: list[str] = []
    required_jobs: list[dict[str, Any]] = []
    for family in profile["families"]:
        matched = family_matches[family["job_id"]]
        count = len(matched)
        if count < family["min_instances"]:
            incomplete.append(f"{family['job_id']}: expected at least {family['min_instances']}, saw {count}")
            continue
        if count > family["max_instances"]:
            return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, (f"{family['job_id']}: expected at most {family['max_instances']}, saw {count}",), tuple(required_jobs))
        for job in matched:
            required_jobs.append(job)
            status = job.get("status")
            conclusion = job.get("conclusion")
            skipped = job.get("skipped") is True or conclusion == "skipped"
            if status != "completed":
                incomplete.append(f"{job['name']}:{status}")
            elif conclusion in FAILURE_CONCLUSIONS:
                failures.append(f"{job['name']}:{conclusion}")
            elif skipped:
                if family["required_disposition"] != "allowed_skip":
                    incomplete.append(f"{job['name']}:skipped")
            elif conclusion == "success":
                pass
            elif conclusion in INCOMPLETE_CONCLUSIONS:
                incomplete.append(f"{job['name']}:{conclusion}")
            else:
                return ManifestResult(ManifestDisposition.MANIFEST_MISMATCH, (f"unknown job conclusion for {job['name']}: {conclusion!r}",), tuple(required_jobs))
    if failures:
        return ManifestResult(ManifestDisposition.JOB_FAILURE, ("required jobs failed: " + ", ".join(sorted(failures)),), tuple(required_jobs))
    if incomplete:
        return ManifestResult(ManifestDisposition.CENSUS_INCOMPLETE, ("required jobs incomplete: " + ", ".join(sorted(incomplete)),), tuple(required_jobs))
    return ManifestResult(ManifestDisposition.SATISFIED, ("job census satisfies trusted manifest",), tuple(required_jobs))
