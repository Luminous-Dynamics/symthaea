#!/usr/bin/env python3
"""Validate qualification-attempt observations and coalesce them by admission subject."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as train

OBSERVATION_SCHEMA = "symthaea.qualification-attempt-observation.v1"
SET_SCHEMA = "symthaea.qualification-admission-subject-attempt-set.v1"
OBSERVATION_DOMAIN = b"symthaea.qualification-attempt-observation.v1\0"
SET_DOMAIN = b"symthaea.qualification-admission-subject-attempt-set.v1\0"
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

TERMINAL_DISPOSITIONS = {
    "Passed", "LockStale", "NamespaceInvalid", "FormattingFailed",
    "CompileFailed", "ClippyFailed", "TestsFailed", "DocTestsFailed",
    "InfrastructureUnavailable", "Cancelled", "OutcomeUnknown",
}
SOURCE_FAILURE_DISPOSITIONS = {
    "LockStale", "NamespaceInvalid", "FormattingFailed", "CompileFailed",
    "ClippyFailed", "TestsFailed", "DocTestsFailed",
}
INFRASTRUCTURE_DISPOSITIONS = {"InfrastructureUnavailable", "Cancelled", "OutcomeUnknown"}
THEOREM_DISPOSITIONS = {"Passed", "Failed", "NotExecuted"}


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise train.TrainManifestError(f"{where}: expected canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: text must use Unicode NFC")
    return value


def _id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected 40 lowercase hex Git SHA")
    return value


def _positive_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise train.TrainManifestError(f"{where}: expected positive integer")
    return value


def compute_observation_id(raw: Any) -> str:
    """Recompute the semantic ID from the complete observation payload.

    The observation ID intentionally commits fields that the compact attempt-set
    projection does not repeat (for example runner environment, timestamps,
    failure details, and explicit non-claims). Presentation whitespace is not
    semantic; object keys are canonicalized by JSON ordering.
    """
    if not isinstance(raw, dict):
        raise train.TrainManifestError("attempt observation: expected object")
    payload = {key: value for key, value in raw.items() if key != "observation_id"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(OBSERVATION_DOMAIN + encoded).hexdigest()


def _validate_progress(progress: list[Any], terminal: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    failed_indexes: list[int] = []

    for index, item in enumerate(progress):
        if not isinstance(item, dict):
            raise train.TrainManifestError(f"attempt observation.theorem_progress[{index}]: expected object")
        if set(item) != {"name", "disposition"}:
            raise train.TrainManifestError(
                f"attempt observation.theorem_progress[{index}]: expected exactly name/disposition"
            )
        name = _string(item.get("name"), f"attempt observation.theorem_progress[{index}].name")
        disposition = _string(item.get("disposition"), f"attempt observation.theorem_progress[{index}].disposition")
        if disposition not in THEOREM_DISPOSITIONS:
            raise train.TrainManifestError(
                f"attempt observation.theorem_progress[{index}].disposition: unsupported {disposition!r}"
            )
        if name in seen:
            raise train.TrainManifestError(f"attempt observation.theorem_progress: duplicate theorem {name!r}")
        seen.add(name)
        if disposition == "Failed":
            failed_indexes.append(index)
        normalized.append({"name": name, "disposition": disposition})

    if terminal == "Passed":
        if any(item["disposition"] != "Passed" for item in normalized):
            raise train.TrainManifestError(
                "attempt observation.theorem_progress: terminal Passed requires every theorem to be Passed"
            )
        return normalized

    if terminal in SOURCE_FAILURE_DISPOSITIONS:
        if len(failed_indexes) != 1:
            raise train.TrainManifestError(
                "attempt observation.theorem_progress: source failure requires exactly one Failed theorem"
            )
        failed_index = failed_indexes[0]
        if any(item["disposition"] != "NotExecuted" for item in normalized[failed_index + 1 :]):
            raise train.TrainManifestError(
                "attempt observation.theorem_progress: theorems after a source failure must be NotExecuted"
            )
        return normalized

    if terminal in INFRASTRUCTURE_DISPOSITIONS:
        if failed_indexes:
            raise train.TrainManifestError(
                "attempt observation.theorem_progress: infrastructure terminal state must not encode a source theorem failure"
            )
        passed_not_executed = [item["disposition"] for item in normalized]
        first_not_executed = next(
            (index for index, disposition in enumerate(passed_not_executed) if disposition == "NotExecuted"),
            None,
        )
        if first_not_executed is not None and any(
            disposition != "NotExecuted" for disposition in passed_not_executed[first_not_executed:]
        ):
            raise train.TrainManifestError(
                "attempt observation.theorem_progress: execution cannot resume after NotExecuted without a new attempt"
            )
        return normalized

    raise train.TrainManifestError(f"attempt observation.execution.terminal_disposition: unsupported {terminal!r}")


def normalize_observation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != OBSERVATION_SCHEMA:
        raise train.TrainManifestError(f"attempt observation: expected schema {OBSERVATION_SCHEMA!r}")

    declared_observation_id = _id(raw.get("observation_id"), "attempt observation.observation_id")
    computed_observation_id = compute_observation_id(raw)
    if declared_observation_id != computed_observation_id:
        raise train.TrainManifestError(
            "attempt observation.observation_id: does not match canonical observation contents"
        )

    workflow = raw.get("workflow")
    source = raw.get("source_subject")
    correlation = raw.get("admission_correlation")
    execution = raw.get("execution")
    progress = raw.get("theorem_progress")
    if not all(isinstance(value, dict) for value in (workflow, source, correlation, execution)):
        raise train.TrainManifestError("attempt observation: malformed workflow/source/correlation/execution")
    if not isinstance(progress, list) or not progress:
        raise train.TrainManifestError("attempt observation.theorem_progress: expected non-empty list")

    terminal = _string(execution.get("terminal_disposition"), "attempt observation.execution.terminal_disposition")
    if terminal not in TERMINAL_DISPOSITIONS:
        raise train.TrainManifestError(f"attempt observation.execution.terminal_disposition: unsupported {terminal!r}")

    normalized_progress = _validate_progress(progress, terminal)

    return {
        "observation_id": declared_observation_id,
        "admission_subject_id": _id(correlation.get("subject_id"), "attempt observation.admission_correlation.subject_id"),
        "provider": _string(raw.get("provider"), "attempt observation.provider"),
        "repository": _string(raw.get("repository"), "attempt observation.repository"),
        "workflow_run_id": _positive_int(workflow.get("run_id"), "attempt observation.workflow.run_id"),
        "workflow_run_number": _positive_int(workflow.get("run_number"), "attempt observation.workflow.run_number"),
        "job_id": _positive_int(workflow.get("job_id"), "attempt observation.workflow.job_id"),
        "requested_pr_head_sha": _sha(source.get("pr_head_sha"), "attempt observation.source_subject.pr_head_sha"),
        "requested_pr_base_sha": _sha(source.get("pr_base_sha"), "attempt observation.source_subject.pr_base_sha"),
        "checked_out_sha": _sha(source.get("checked_out_sha"), "attempt observation.source_subject.checked_out_sha"),
        "terminal_disposition": terminal,
        "theorem_progress": normalized_progress,
    }


def build_attempt_set(observations: list[Any]) -> dict[str, Any]:
    if not observations:
        raise train.TrainManifestError("attempt set: expected at least one observation")
    normalized = [normalize_observation(value) for value in observations]
    subjects = {value["admission_subject_id"] for value in normalized}
    if len(subjects) != 1:
        raise train.TrainManifestError("attempt set: observations target different admission subjects")

    ids = [value["observation_id"] for value in normalized]
    if len(ids) != len(set(ids)):
        raise train.TrainManifestError("attempt set: duplicate observation_id")

    provider_attempts = [
        (value["provider"], value["repository"], value["workflow_run_id"], value["job_id"])
        for value in normalized
    ]
    if len(provider_attempts) != len(set(provider_attempts)):
        raise train.TrainManifestError("attempt set: duplicate provider run/job identity")

    result = {
        "schema": SET_SCHEMA,
        "admission_subject_id": next(iter(subjects)),
        "attempt_observations": sorted(normalized, key=lambda value: value["observation_id"]),
        "non_claims": [
            "does not establish chronological ordering between attempts",
            "does not establish identical checked-out bytes across attempts",
            "does not include source-changing repair links",
            "does not establish hosted qualification unless an included attempt is Passed",
        ],
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    result["set_id"] = "sha256:" + hashlib.sha256(SET_DOMAIN + payload).hexdigest()
    return result


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_observation(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--print-normalized", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_attempt_set([load_observation(path) for path in args.observations])
    except train.TrainManifestError as error:
        print(f"qualification attempt set invalid: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.print_normalized else result["set_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
