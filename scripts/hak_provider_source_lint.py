#!/usr/bin/env python3
"""Audit-only HAK-012 provider source-observation and projection validator.

This module validates locally materialized provider observations and exact field
projection provenance. It does not authenticate GitHub cryptographically,
establish semantic claim truth, or grant runtime authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

OBS_SCHEMA = "hak.provider-source-observation.v1"
PROJECTION_SCHEMA = "hak.provider-projection-map.v1"
ASSURANCE_CLASSES = {"ProviderObserved", "ProviderBound", "CryptographicallyAttested"}
RESOURCE_KINDS = {
    "WorkflowRunObservation",
    "WorkflowJobsObservation",
    "WorkflowJobStepsObservation",
}
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REF = re.compile(r"^git:(?P<repo>[^@]+)@(?P<sha>[0-9a-f]{40}):(?P<path>.+)$")


class ProviderSourceLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderSourceLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")
    return value


def _digest(prefix: bytes, doc: Any) -> str:
    encoded = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(prefix + encoded).hexdigest()


def compute_payload_digest(payload: dict[str, Any]) -> str:
    return _digest(b"hak.provider-normalized-payload.v1\0", payload)


def compute_observation_digest(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "observation_digest"}
    return _digest(b"hak.provider-source-observation.v1\0", payload)


def compute_projection_digest(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "projection_digest"}
    return _digest(b"hak.provider-projection-map.v1\0", payload)


def compute_capsule_digest(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "capsule_digest"}
    return _digest(b"hak.real-provider-evidence-capsule.v1\0", payload)


def _expected_source_ref(doc: dict[str, Any]) -> str:
    repository = doc["repository"]
    identity = doc["resource_identity"]
    run_id = identity["run_id"]
    attempt = identity["run_attempt"]
    base = f"github-actions:{repository}:workflow-run/{run_id}:attempt/{attempt}"
    kind = doc["resource_kind"]
    if kind == "WorkflowRunObservation":
        return base
    if kind == "WorkflowJobsObservation":
        return base + ":jobs"
    return base + f":job/{identity.get('job_id')}:steps"


def validate_source_observation(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == OBS_SCHEMA, f"schema_version must be {OBS_SCHEMA}")
    _text(doc.get("observation_id"), "observation_id")
    _require(doc.get("provider") == "github-actions", "provider must be github-actions")
    assurance = doc.get("assurance_class")
    _require(assurance in ASSURANCE_CLASSES,
             f"assurance_class must be one of {sorted(ASSURANCE_CLASSES)}")
    _text(doc.get("repository"), "repository")
    kind = doc.get("resource_kind")
    _require(kind in RESOURCE_KINDS, f"resource_kind must be one of {sorted(RESOURCE_KINDS)}")

    identity = _obj(doc.get("resource_identity"), "resource_identity")
    run_id = _positive_int(identity.get("run_id"), "resource_identity.run_id")
    attempt = _positive_int(identity.get("run_attempt"), "resource_identity.run_attempt")
    job_id = identity.get("job_id")
    if kind == "WorkflowJobStepsObservation":
        _positive_int(job_id, "resource_identity.job_id")
    else:
        _require(job_id is None, "only WorkflowJobStepsObservation may carry resource_identity.job_id")

    _require(doc.get("source_ref") == _expected_source_ref(doc),
             "source_ref must exactly match provider, repository, resource kind, run, attempt, and job identity")

    window = _obj(doc.get("observation_window"), "observation_window")
    after = _text(window.get("after_or_at"), "observation_window.after_or_at")
    before = _text(window.get("before_or_at"), "observation_window.before_or_at")
    _require(window.get("precision") == "BoundedWindow",
             "observation_window.precision must be BoundedWindow")
    _text(window.get("basis"), "observation_window.basis")
    _require(after <= before, "observation window must satisfy after_or_at <= before_or_at")

    collector = _obj(doc.get("collector"), "collector")
    _text(collector.get("kind"), "collector.kind")
    _text(collector.get("identity"), "collector.identity")
    _text(collector.get("method"), "collector.method")

    _require(doc.get("payload_scope") == "NormalizedProjection", "payload_scope must be NormalizedProjection")
    normalization = _obj(doc.get("normalization"), "normalization")
    _require(normalization.get("method") == "FieldSelectionNoSemanticTransform",
             "normalization.method must be FieldSelectionNoSemanticTransform")
    replayability = normalization.get("replayability")
    _require(replayability in {"NotReplayableWithoutRawResponse", "ReplayableFromRetainedRawResponse"},
             "normalization.replayability is invalid")
    _text(normalization.get("profile_ref"), "normalization.profile_ref")

    payload = _obj(doc.get("normalized_payload"), "normalized_payload")
    payload_digest = doc.get("normalized_payload_digest")
    _require(isinstance(payload_digest, str) and DIGEST.fullmatch(payload_digest),
             "normalized_payload_digest must be sha256:<64 hex>")
    _require(payload_digest == compute_payload_digest(payload),
             "normalized_payload_digest does not match normalized_payload")

    retention = _obj(doc.get("retention"), "retention")
    _require(retention.get("normalized_payload_retained") is True,
             "normalized_payload_retained must be true")
    raw_retained = retention.get("raw_response_retained")
    _require(isinstance(raw_retained, bool), "raw_response_retained must be boolean")
    raw_digest = retention.get("raw_response_digest")
    if raw_retained:
        _require(isinstance(raw_digest, str) and DIGEST.fullmatch(raw_digest),
                 "retained raw response requires raw_response_digest")
        _require(replayability == "ReplayableFromRetainedRawResponse",
                 "raw-response retention requires replayable normalization disclosure")
    else:
        _require(raw_digest is None, "raw_response_digest must be null when raw response is not retained")
        _require(replayability == "NotReplayableWithoutRawResponse",
                 "normalization must not claim replayability when raw response is not retained")

    provider_binding_ref = doc.get("provider_binding_ref")
    attestation_ref = doc.get("attestation_ref")
    if assurance == "ProviderObserved":
        _require(provider_binding_ref is None and attestation_ref is None,
                 "ProviderObserved must not claim provider binding or cryptographic attestation")
    elif assurance == "ProviderBound":
        _text(provider_binding_ref, "provider_binding_ref")
        _require(attestation_ref is None, "ProviderBound must not claim cryptographic attestation")
    else:
        _text(provider_binding_ref, "provider_binding_ref")
        _text(attestation_ref, "attestation_ref")

    supersedes = doc.get("supersedes_observation_id")
    _require(supersedes is None or (isinstance(supersedes, str) and supersedes.strip()),
             "supersedes_observation_id must be null or non-empty")

    if kind == "WorkflowRunObservation":
        _require(payload.get("id") == run_id, "workflow-run payload id must match resource identity")
        _require(payload.get("run_attempt") == attempt,
                 "workflow-run payload run_attempt must match resource identity")
    elif kind == "WorkflowJobsObservation":
        jobs = _array(payload.get("jobs"), "normalized_payload.jobs")
        _require(payload.get("total_count") == len(jobs),
                 "workflow-jobs total_count must equal retained jobs length")
        for index, job in enumerate(jobs):
            item = _obj(job, f"normalized_payload.jobs[{index}]")
            _positive_int(item.get("id"), f"normalized_payload.jobs[{index}].id")
            _require(item.get("run_id") == run_id,
                     f"normalized_payload.jobs[{index}].run_id must match resource identity")
            _require(item.get("run_attempt") == attempt,
                     f"normalized_payload.jobs[{index}].run_attempt must match resource identity")
    else:
        _array(payload.get("steps"), "normalized_payload.steps")

    observation_digest = doc.get("observation_digest")
    _require(isinstance(observation_digest, str) and DIGEST.fullmatch(observation_digest),
             "observation_digest must be sha256:<64 hex>")
    _require(observation_digest == compute_observation_digest(doc),
             "observation_digest does not match canonical observation content")


def _resolve_path(doc: Any, path: str) -> Any:
    current = doc
    for segment in path.split("."):
        if isinstance(current, list):
            _require(segment.isdigit(), f"path segment {segment!r} must index an array")
            index = int(segment)
            _require(0 <= index < len(current), f"array index out of range in path {path!r}")
            current = current[index]
        elif isinstance(current, dict):
            _require(segment in current, f"missing source/target path segment {segment!r} in {path!r}")
            current = current[segment]
        else:
            raise ProviderSourceLintError(f"path {path!r} traverses a scalar at {segment!r}")
    return current


def _flatten_leaves(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        if not value:
            return {prefix}
        result: set[str] = set()
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_leaves(child, child_prefix))
        return result
    if isinstance(value, list):
        if not value:
            return {prefix}
        result: set[str] = set()
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten_leaves(child, child_prefix))
        return result
    return {prefix}


def _required_kind_for_target(target_path: str) -> str:
    if target_path == "job.steps" or target_path.startswith("job.steps."):
        return "WorkflowJobStepsObservation"
    if target_path.startswith("job."):
        return "WorkflowJobsObservation"
    return "WorkflowRunObservation"


def validate_provider_projection(
    capsule: dict[str, Any],
    observations: list[dict[str, Any]],
    projection: dict[str, Any],
    *,
    capsule_repo_path: str,
) -> None:
    _require(capsule.get("schema_version") == "hak.real-provider-evidence-capsule.v1",
             "target capsule must be hak.real-provider-evidence-capsule.v1")
    _require(capsule.get("capsule_digest") == compute_capsule_digest(capsule),
             "target capsule_digest must match canonical capsule content")

    for observation in observations:
        validate_source_observation(observation)
    obs_by_id: dict[str, dict[str, Any]] = {}
    for observation in observations:
        oid = observation["observation_id"]
        _require(oid not in obs_by_id, f"duplicate source observation id: {oid}")
        obs_by_id[oid] = observation

    _require(projection.get("schema_version") == PROJECTION_SCHEMA,
             f"schema_version must be {PROJECTION_SCHEMA}")
    _text(projection.get("projection_id"), "projection_id")
    declared_assurance = projection.get("declared_source_assurance")
    _require(declared_assurance in ASSURANCE_CLASSES,
             "declared_source_assurance must be a known assurance class")
    for observation in observations:
        _require(observation["assurance_class"] == declared_assurance,
                 "every source observation assurance_class must equal declared_source_assurance")

    target = _obj(projection.get("target"), "target")
    artifact_ref = _text(target.get("artifact_ref"), "target.artifact_ref")
    match = GIT_REF.fullmatch(artifact_ref)
    _require(match is not None, "target.artifact_ref must be git:<repo>@<40hex>:<path>")
    _require(match.group("path") == capsule_repo_path,
             "target.artifact_ref path must match loaded capsule repo path")
    _require(target.get("capsule_id") == capsule.get("capsule_id"),
             "target capsule_id must match loaded capsule")
    _require(target.get("capsule_digest") == capsule.get("capsule_digest"),
             "target capsule_digest must match loaded capsule")
    _require(target.get("target_root") == "provider_snapshot", "target_root must be provider_snapshot")

    bindings = _array(projection.get("source_observations"), "source_observations")
    bound: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(bindings):
        item = _obj(raw, f"source_observations[{index}]")
        oid = _text(item.get("observation_id"), f"source_observations[{index}].observation_id")
        _require(oid not in bound, f"duplicate source observation binding: {oid}")
        _require(oid in obs_by_id, f"projection references unknown source observation: {oid}")
        _require(item.get("observation_digest") == obs_by_id[oid]["observation_digest"],
                 f"source observation digest mismatch for {oid}")
        bound[oid] = item
    _require(set(bound) == set(obs_by_id),
             "projection source_observations must exactly match supplied observations")

    first = observations[0]
    common = (
        first["provider"], first["repository"], first["resource_identity"]["run_id"],
        first["resource_identity"]["run_attempt"],
    )
    for observation in observations[1:]:
        current = (
            observation["provider"], observation["repository"],
            observation["resource_identity"]["run_id"], observation["resource_identity"]["run_attempt"],
        )
        _require(current == common, "all source observations must belong to one provider/run lineage")
    _require(match.group("repo") == common[1],
             "target artifact repository must match source-observation repository")

    kinds = [observation["resource_kind"] for observation in observations]
    _require(kinds.count("WorkflowRunObservation") == 1,
             "projection requires exactly one WorkflowRunObservation")
    _require(kinds.count("WorkflowJobsObservation") == 1,
             "projection requires exactly one WorkflowJobsObservation")
    _require(kinds.count("WorkflowJobStepsObservation") == 1,
             "projection requires exactly one WorkflowJobStepsObservation")

    snapshot = _obj(capsule.get("provider_snapshot"), "provider_snapshot")
    step_obs = next(x for x in observations if x["resource_kind"] == "WorkflowJobStepsObservation")
    target_job = _obj(snapshot.get("job"), "provider_snapshot.job")
    _require(step_obs["resource_identity"]["job_id"] == target_job.get("job_id"),
             "WorkflowJobStepsObservation must bind the exact target job")

    mappings = _array(projection.get("mappings"), "mappings")
    by_target: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(mappings):
        item = _obj(raw, f"mappings[{index}]")
        target_path = _text(item.get("target_path"), f"mappings[{index}].target_path")
        _require(target_path not in by_target, f"duplicate target_path mapping: {target_path}")
        oid = _text(item.get("source_observation_id"), f"mappings[{index}].source_observation_id")
        _require(oid in obs_by_id, f"mapping references unknown source observation: {oid}")
        source_path = _text(item.get("source_path"), f"mappings[{index}].source_path")
        _require(item.get("transform") == "Identity",
                 "HAK-012 v1 supports only Identity projection transforms")
        required_kind = _required_kind_for_target(target_path)
        _require(obs_by_id[oid]["resource_kind"] == required_kind,
                 f"target {target_path} must source from {required_kind}")
        source_value = _resolve_path(obs_by_id[oid], source_path)
        target_value = _resolve_path(snapshot, target_path)
        _require(source_value == target_value,
                 f"projected value mismatch for target {target_path}")
        by_target[target_path] = item

    expected_leaves = _flatten_leaves(snapshot)
    _require(
        set(by_target) == expected_leaves,
        "projection mappings must cover every target leaf exactly once; "
        f"missing={sorted(expected_leaves - set(by_target))}, extra={sorted(set(by_target) - expected_leaves)}",
    )

    digest = projection.get("projection_digest")
    _require(isinstance(digest, str) and DIGEST.fullmatch(digest),
             "projection_digest must be sha256:<64 hex>")
    _require(digest == compute_projection_digest(projection),
             "projection_digest does not match canonical projection content")


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderSourceLintError(str(exc)) from exc
    _require(isinstance(doc, dict), f"{path} root must be an object")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate HAK provider-source observations and exact projection provenance."
    )
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--observation", action="append", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        capsule = _load(args.capsule)
        projection = _load(args.projection)
        observations = [_load(path) for path in args.observation]
        validate_provider_projection(
            capsule, observations, projection, capsule_repo_path=args.capsule.as_posix()
        )
    except ProviderSourceLintError as exc:
        print(f"FAIL {args.projection}: {exc}")
        return 1
    print(f"OK   {args.projection} (provider source observations + exact projection provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
