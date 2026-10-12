#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed structural validator for the voice language capability registry.

This validates tag shape and evidence structure; it does not query GitHub, prove
that an Actions run passed, or measure pronunciation/voice quality. The registry
must not be treated as a live trust root merely because this script succeeds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(__file__).with_name("voice_language_registry.json")
CAPABILITIES = {
    "text_normalization", "asr", "translation", "g2p",
    "tts", "singing", "code_switching", "streaming",
}
STATUSES = {
    "unverified", "implementation_present", "automated_checks_passed",
    "speaker_reviewed", "production_qualified",
}
TAG_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
LANGUAGE_KEYS = {"tag", "display_name", "script", "capabilities"}
CAPABILITY_KEYS = {
    "status", "backend", "source_refs", "model", "dataset",
    "verification", "evaluation", "speaker_review",
}
MODEL_KEYS = {"name", "revision", "sha256", "license"}
DATASET_KEYS = {"name", "revision", "sha256", "license", "held_out_split"}
VERIFICATION_KEYS = {"commit_sha", "run_url", "checks"}
EVALUATION_KEYS = {
    "metric", "value", "threshold", "comparator", "sample_count",
    "held_out", "split_description", "outcome",
}
REVIEW_KEYS = {"reviewer_count", "method", "scope", "date"}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(value: Any, keys: set[str], where: str) -> None:
    _need(isinstance(value, dict), f"{where} must be an object")
    missing, extra = keys - set(value), set(value) - keys
    _need(not missing and not extra, f"{where} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def _read_json(path: Path) -> Any:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _validate_model(model: Any, where: str) -> None:
    if model is None:
        return
    _exact_keys(model, MODEL_KEYS, f"{where}.model")
    for key in ("name", "revision", "license"):
        _need(_nonempty(model[key]), f"{where}.model.{key} must be explicit and non-empty")
    _need(isinstance(model["sha256"], str) and SHA256_RE.fullmatch(model["sha256"]) is not None,
          f"{where}.model.sha256 must be a 64-hex SHA-256 digest")


def _validate_dataset(dataset: Any, where: str) -> None:
    if dataset is None:
        return
    _exact_keys(dataset, DATASET_KEYS, f"{where}.dataset")
    for key in ("name", "revision", "license", "held_out_split"):
        _need(_nonempty(dataset[key]), f"{where}.dataset.{key} must be explicit and non-empty")
    _need(isinstance(dataset["sha256"], str) and SHA256_RE.fullmatch(dataset["sha256"]) is not None,
          f"{where}.dataset.sha256 must be a 64-hex SHA-256 digest")


def _validate_verification(receipt: Any, where: str) -> None:
    if receipt is None:
        return
    _exact_keys(receipt, VERIFICATION_KEYS, f"{where}.verification")
    _need(isinstance(receipt["commit_sha"], str) and SHA1_RE.fullmatch(receipt["commit_sha"]) is not None,
          f"{where}.verification.commit_sha must be a 40-hex commit SHA")
    url = receipt["run_url"]
    _need(_nonempty(url) and url.startswith("https://github.com/") and "/actions/runs/" in url,
          f"{where}.verification.run_url must point to a GitHub Actions run")
    checks = receipt["checks"]
    _need(isinstance(checks, list) and checks and all(_nonempty(c) for c in checks),
          f"{where}.verification.checks must be a non-empty list of named checks")
    _need(len(checks) == len(set(checks)), f"{where}.verification.checks must be unique")


def _validate_evaluation(evaluation: Any, where: str) -> None:
    if evaluation is None:
        return
    _exact_keys(evaluation, EVALUATION_KEYS, f"{where}.evaluation")
    _need(_nonempty(evaluation["metric"]), f"{where}.evaluation.metric must be non-empty")
    for key in ("value", "threshold"):
        value = evaluation[key]
        _need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
              f"{where}.evaluation.{key} must be a finite number")
    comparator = evaluation["comparator"]
    _need(isinstance(comparator, str) and comparator in {"gte", "lte"}, f"{where}.evaluation.comparator must be gte or lte")
    count = evaluation["sample_count"]
    _need(isinstance(count, int) and not isinstance(count, bool) and count > 0,
          f"{where}.evaluation.sample_count must be a positive integer")
    _need(isinstance(evaluation["held_out"], bool), f"{where}.evaluation.held_out must be boolean")
    _need(_nonempty(evaluation["split_description"]), f"{where}.evaluation.split_description must explain the held-out split")
    expected = (evaluation["value"] >= evaluation["threshold"] if comparator == "gte"
                else evaluation["value"] <= evaluation["threshold"])
    _need(isinstance(evaluation["outcome"], str) and evaluation["outcome"] in {"pass", "fail"}, f"{where}.evaluation.outcome must be pass or fail")
    _need(evaluation["outcome"] == ("pass" if expected else "fail"),
          f"{where}.evaluation.outcome contradicts value/threshold/comparator")


def _validate_speaker_review(review: Any, where: str) -> None:
    if review is None:
        return
    _exact_keys(review, REVIEW_KEYS, f"{where}.speaker_review")
    count = review["reviewer_count"]
    _need(isinstance(count, int) and not isinstance(count, bool) and count >= 1,
          f"{where}.speaker_review.reviewer_count must be a positive integer")
    for key in ("method", "scope"):
        _need(_nonempty(review[key]), f"{where}.speaker_review.{key} must be non-empty")
    _need(isinstance(review["date"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", review["date"]) is not None,
          f"{where}.speaker_review.date must be ISO YYYY-MM-DD")
    try:
        date.fromisoformat(review["date"])
    except ValueError as exc:
        raise ValueError(f"{where}.speaker_review.date must be ISO YYYY-MM-DD") from exc


def validate_registry(registry: Any) -> dict[str, int]:
    _exact_keys(registry, {"schema_version", "capabilities", "languages"}, "registry")
    _need(isinstance(registry["schema_version"], int) and not isinstance(registry["schema_version"], bool) and registry["schema_version"] == 1, "registry.schema_version must equal integer 1")
    declared = registry["capabilities"]
    _need(isinstance(declared, list) and declared and all(_nonempty(x) for x in declared),
          "registry.capabilities must be a non-empty string array")
    _need(len(declared) == len(set(declared)), "registry.capabilities must be unique")
    _need(set(declared) == CAPABILITIES, f"registry.capabilities must match required set: {sorted(CAPABILITIES)}")
    languages = registry["languages"]
    _need(isinstance(languages, list) and languages, "registry.languages must be a non-empty array")
    seen_tags: set[str] = set()
    count_by_status = {status: 0 for status in STATUSES}
    for index, language in enumerate(languages):
        where = f"languages[{index}]"
        _exact_keys(language, LANGUAGE_KEYS, where)
        tag = language["tag"]
        _need(isinstance(tag, str) and TAG_RE.fullmatch(tag) is not None,
              f"{where}.tag must have a BCP 47-shaped language tag")
        _need(tag.casefold() not in seen_tags, f"{where}.tag duplicates another tag case-insensitively")
        seen_tags.add(tag.casefold())
        _need(_nonempty(language["display_name"]), f"{where}.display_name must be non-empty")
        script = language["script"]
        _need(script is None or (isinstance(script, str) and re.fullmatch(r"[A-Z][a-z]{3}", script) is not None),
              f"{where}.script must be a four-letter ISO 15924-style code or null")
        cap_map = language["capabilities"]
        _exact_keys(cap_map, CAPABILITIES, f"{where}.capabilities")
        for cap_name, record in cap_map.items():
            cw = f"{where}.capabilities.{cap_name}"
            _exact_keys(record, CAPABILITY_KEYS, cw)
            status = record["status"]
            _need(isinstance(status, str) and status in STATUSES, f"{cw}.status is not recognized")
            count_by_status[status] += 1
            backend = record["backend"]
            _need(backend is None or _nonempty(backend), f"{cw}.backend must be null or a non-empty string")
            refs = record["source_refs"]
            _need(isinstance(refs, list) and all(_nonempty(ref) for ref in refs), f"{cw}.source_refs must be a string array")
            for ref in refs:
                _need(not ref.startswith("/") and ".." not in Path(ref).parts and "\\" not in ref,
                      f"{cw}.source_refs must be safe repository-relative paths")
            _validate_model(record["model"], cw)
            _validate_dataset(record["dataset"], cw)
            _validate_verification(record["verification"], cw)
            _validate_evaluation(record["evaluation"], cw)
            _validate_speaker_review(record["speaker_review"], cw)
            if status == "unverified":
                _need(backend is None and not refs and record["model"] is None and record["dataset"] is None
                      and record["verification"] is None and record["evaluation"] is None
                      and record["speaker_review"] is None,
                      f"{cw}: unverified capability cannot include implementation or qualification claims")
            else:
                _need(_nonempty(backend) and refs, f"{cw}: status {status} requires backend and source_refs")
            if status in {"automated_checks_passed", "speaker_reviewed", "production_qualified"}:
                _need(record["verification"] is not None, f"{cw}: {status} requires a verification receipt")
            if status in {"speaker_reviewed", "production_qualified"}:
                _need(record["evaluation"] is not None and record["evaluation"]["held_out"]
                      and record["evaluation"]["outcome"] == "pass",
                      f"{cw}: {status} requires a passing held-out evaluation")
                _need(record["speaker_review"] is not None, f"{cw}: {status} requires speaker review evidence")
            if status == "production_qualified":
                _need(record["model"] is not None and record["dataset"] is not None,
                      f"{cw}: production_qualified requires pinned model and dataset metadata")
                _need(record["speaker_review"]["reviewer_count"] >= 2,
                      f"{cw}: production_qualified requires at least two speaker reviewers")
    return {
        "languages": len(languages),
        "capability_records": len(languages) * len(CAPABILITIES),
        **{f"status_{key}": value for key, value in sorted(count_by_status.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    try:
        summary = validate_registry(_read_json(args.registry))
    except ValueError as exc:
        parser.error(str(exc))
    print("voice language registry valid (schema only): " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
