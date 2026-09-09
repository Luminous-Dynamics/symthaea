#!/usr/bin/env python3
"""Validate immutable qualification-admission requests for integration trains."""

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

import integration_train_catalog as catalog
import integration_train_manifest as train

SCHEMA = "symthaea.qualification-admission-request.v1"
DOMAIN = b"symthaea.qualification-admission-request.v1\0"
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    if value != value.strip():
        raise train.TrainManifestError(f"{where}: leading/trailing whitespace is not canonical")
    if not value:
        raise train.TrainManifestError(f"{where}: must not be empty")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise train.TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise train.TrainManifestError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _canonical_payload_from_normalized(request: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in request.items() if key != "admission_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_admission_id_from_normalized(request: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical_payload_from_normalized(request)).hexdigest()


def normalize_request(request: Any, *, verify_declared_id: bool = True, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise train.TrainManifestError("admission: expected object")
    _require_exact_keys(
        request,
        {"schema", "program_id", "catalog_id", "catalog_branch", "catalog_path", "target_train", "qualification_profile", "reason", "evidence_refs", "non_claims"},
        {"admission_id"},
        where="admission",
    )
    if request["schema"] != SCHEMA:
        raise train.TrainManifestError(f"admission.schema: expected {SCHEMA!r}")
    if require_id and "admission_id" not in request:
        raise train.TrainManifestError("admission.admission_id: required but absent")

    raw_target = request["target_train"]
    if not isinstance(raw_target, dict):
        raise train.TrainManifestError("admission.target_train: expected object")
    _require_exact_keys(raw_target, {"name", "train_id", "subject_sha"}, set(), where="admission.target_train")

    normalized = {
        "schema": SCHEMA,
        "program_id": _require_string(request["program_id"], where="admission.program_id"),
        "catalog_id": _require_id(request["catalog_id"], where="admission.catalog_id"),
        "catalog_branch": catalog._require_branch(request["catalog_branch"], where="admission.catalog_branch"),
        "catalog_path": catalog._require_manifest_path(request["catalog_path"], where="admission.catalog_path"),
        "target_train": {
            "name": _require_string(raw_target["name"], where="admission.target_train.name"),
            "train_id": _require_id(raw_target["train_id"], where="admission.target_train.train_id"),
            "subject_sha": train._require_sha(raw_target["subject_sha"], where="admission.target_train.subject_sha"),
        },
        "qualification_profile": _require_string(request["qualification_profile"], where="admission.qualification_profile"),
        "reason": _require_string(request["reason"], where="admission.reason"),
        "evidence_refs": train._require_sorted_unique_strings(request["evidence_refs"], where="admission.evidence_refs"),
        "non_claims": train._require_sorted_unique_strings(request["non_claims"], where="admission.non_claims"),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError("admission.non_claims: must contain at least one explicit non-claim")

    admission_id = _compute_admission_id_from_normalized(normalized)
    normalized["admission_id"] = admission_id
    if verify_declared_id and "admission_id" in request:
        declared = _require_id(request["admission_id"], where="admission.admission_id")
        if declared != admission_id:
            raise train.TrainManifestError(f"admission.admission_id: expected {admission_id}, got {declared}")
    return normalized


def compute_admission_id(request: Any) -> str:
    normalized = normalize_request(request, verify_declared_id=False)
    return _compute_admission_id_from_normalized(normalized)


def validate_catalog_record(request: Any, catalog_value: Any) -> None:
    normalized = normalize_request(request)
    normalized_catalog = catalog.normalize_catalog(catalog_value, require_id=True)
    if normalized_catalog["program_id"] != normalized["program_id"]:
        raise train.TrainManifestError("admission.program_id: does not match referenced catalog")
    if normalized_catalog["catalog_id"] != normalized["catalog_id"]:
        raise train.TrainManifestError("admission.catalog_id: does not match referenced catalog")

    target = normalized["target_train"]
    matches = [entry for entry in normalized_catalog["trains"] if entry["name"] == target["name"]]
    if len(matches) != 1:
        raise train.TrainManifestError(f"admission.target_train.name: {target['name']!r} not uniquely present in catalog")
    entry = matches[0]
    if entry["train_id"] != target["train_id"]:
        raise train.TrainManifestError("admission.target_train.train_id: does not match referenced catalog")
    if entry["cumulative_tip_sha"] != target["subject_sha"]:
        raise train.TrainManifestError("admission.target_train.subject_sha: does not equal catalog train tip")


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise train.TrainManifestError(f"git {' '.join(args)}: {detail}")
    return result.stdout


def _require_git_repo(repo: Path) -> Path:
    repo = repo.resolve()
    result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise train.TrainManifestError(f"repository is not a Git work tree: {repo}")
    return repo


def validate_catalog_binding(request: Any, repo: Path) -> None:
    normalized = normalize_request(request)
    repo = _require_git_repo(repo)
    spec = f"{normalized['catalog_branch']}:{normalized['catalog_path']}"
    raw_text = _run_git(repo, "show", spec)
    try:
        raw_catalog = json.loads(raw_text, object_pairs_hook=train._object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise train.TrainManifestError(f"invalid catalog JSON at {spec}: {error}") from error
    validate_catalog_record(normalized, raw_catalog)
    catalog.validate_git_bindings(raw_catalog, repo)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_request(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(f"{path}: admission request exceeds {train.MAX_MANIFEST_BYTES} bytes")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except train.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_request(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="qualification admission JSON")
    parser.add_argument("--require-id", action="store_true", help="require the exact computed admission_id")
    parser.add_argument("--verify-catalog", action="store_true", help="resolve and recursively verify the referenced catalog")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository used by --verify-catalog")
    parser.add_argument("--print-normalized", action="store_true", help="print canonical normalized JSON including admission_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_request(args.request, require_id=args.require_id)
        if args.verify_catalog:
            validate_catalog_binding(normalized, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification admission request invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    else:
        print(normalized["admission_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
