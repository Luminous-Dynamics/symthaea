#!/usr/bin/env python3
"""Validate content-addressed generic qualification subjects.

V1 supports exact Git subjects only. Integration-train/catalog layers may resolve into one of
these subjects, but are not mandatory parts of the universal qualification ontology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as train

SCHEMA = "symthaea.qualification-subject.v1"
DOMAIN = b"symthaea.qualification-subject.v1\0"
_KIND = "GitCommit"
_OBJECT_FORMAT_LENGTH = {"sha1": 40, "sha256": 64}
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str], *, where: str) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise train.TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise train.TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise train.TrainManifestError(f"{where}: expected string")
    if value != value.strip() or not value:
        raise train.TrainManifestError(f"{where}: must be non-empty canonical text")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise train.TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise train.TrainManifestError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_object_format(value: Any, *, where: str) -> str:
    value = _require_string(value, where=where)
    if value not in _OBJECT_FORMAT_LENGTH:
        raise train.TrainManifestError(f"{where}: expected one of {sorted(_OBJECT_FORMAT_LENGTH)}")
    return value


def _require_object_id(value: Any, *, where: str, object_format: str | None = None) -> str:
    value = _require_string(value, where=where)
    if object_format is None:
        matches = [name for name, length in _OBJECT_FORMAT_LENGTH.items() if len(value) == length]
        if len(matches) != 1:
            raise train.TrainManifestError(f"{where}: object format is ambiguous or unsupported")
        object_format = matches[0]
    else:
        object_format = _require_object_format(object_format, where=f"{where}.object_format")
    expected = _OBJECT_FORMAT_LENGTH[object_format]
    if len(value) != expected or any(char not in "0123456789abcdef" for char in value):
        raise train.TrainManifestError(
            f"{where}: expected {expected}-character lowercase {object_format} Git object id"
        )
    return value


def _require_repository(value: Any, *, where: str) -> str:
    value = _require_string(value, where=where)
    if _REPOSITORY.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected canonical owner/repository")
    return value


def _canonical_payload(subject: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in subject.items() if key != "subject_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_subject_id(subject: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical_payload(subject)).hexdigest()


def normalize_subject(subject: Any, *, verify_declared_id: bool = True, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(subject, dict):
        raise train.TrainManifestError("qualification subject: expected object")
    _require_exact_keys(
        subject,
        {"schema", "kind", "repository", "object_format", "source_commit", "source_tree"},
        {"subject_id"},
        where="qualification subject",
    )
    if subject["schema"] != SCHEMA:
        raise train.TrainManifestError(f"qualification subject.schema: expected {SCHEMA!r}")
    if subject["kind"] != _KIND:
        raise train.TrainManifestError(f"qualification subject.kind: only {_KIND!r} is supported in V1")
    if require_id and "subject_id" not in subject:
        raise train.TrainManifestError("qualification subject.subject_id: required but absent")

    object_format = _require_object_format(subject["object_format"], where="qualification subject.object_format")
    normalized = {
        "schema": SCHEMA,
        "kind": _KIND,
        "repository": _require_repository(subject["repository"], where="qualification subject.repository"),
        "object_format": object_format,
        "source_commit": _require_object_id(subject["source_commit"], where="qualification subject.source_commit", object_format=object_format),
        "source_tree": _require_object_id(subject["source_tree"], where="qualification subject.source_tree", object_format=object_format),
    }
    subject_id = _compute_subject_id(normalized)
    normalized["subject_id"] = subject_id
    if verify_declared_id and "subject_id" in subject:
        declared = _require_string(subject["subject_id"], where="qualification subject.subject_id")
        if declared != subject_id:
            raise train.TrainManifestError(
                f"qualification subject.subject_id: expected {subject_id}, got {declared}"
            )
    return normalized


def compute_subject_id(subject: Any) -> str:
    return normalize_subject(subject, verify_declared_id=False)["subject_id"]


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise train.TrainManifestError(f"git {' '.join(args)}: {detail}")
    return result.stdout.strip()


def validate_git_binding(subject: Any, repo: Path) -> None:
    normalized = normalize_subject(subject, require_id=True)
    repo = repo.resolve()
    if _run_git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise train.TrainManifestError(f"repository is not a Git work tree: {repo}")
    actual_format = _run_git(repo, "rev-parse", "--show-object-format")
    if actual_format != normalized["object_format"]:
        raise train.TrainManifestError(
            f"qualification subject.object_format: repository uses {actual_format}, declared {normalized['object_format']}"
        )
    _run_git(repo, "cat-file", "-e", f"{normalized['source_commit']}^{{commit}}")
    actual_tree = _run_git(repo, "rev-parse", f"{normalized['source_commit']}^{{tree}}")
    if actual_tree != normalized["source_tree"]:
        raise train.TrainManifestError(
            f"qualification subject.source_tree: expected {actual_tree} for commit {normalized['source_commit']}, got {normalized['source_tree']}"
        )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_subject(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(f"{path}: subject exceeds {train.MAX_MANIFEST_BYTES} bytes")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except train.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_subject(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("--require-id", action="store_true")
    parser.add_argument("--verify-git", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--print-normalized", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_subject(args.subject, require_id=args.require_id)
        if args.verify_git:
            validate_git_binding(normalized, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification subject invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    else:
        print(normalized["subject_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
