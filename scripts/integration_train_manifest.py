#!/usr/bin/env python3
"""Validate deterministic integration-train manifests without mutating GitHub state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "symthaea.integration-train.v1"
DOMAIN = b"symthaea.integration-train.v1\0"
MAX_TRANCHES = 100
MAX_LIST_ITEMS = 512
MAX_TEXT_BYTES = 4096
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ORIGIN_PR = 2_147_483_647
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TrainManifestError(ValueError):
    """Raised when an integration-train manifest is malformed or inconsistent."""


def _require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    *,
    where: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")


def _require_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise TrainManifestError(f"{where}: expected lowercase 40-hex Git SHA")
    return value


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise TrainManifestError(f"{where}: expected string")
    if value != value.strip():
        raise TrainManifestError(f"{where}: leading/trailing whitespace is not canonical")
    if not value:
        raise TrainManifestError(f"{where}: must not be empty")
    if unicodedata.normalize("NFC", value) != value:
        raise TrainManifestError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TrainManifestError(f"{where}: exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_sorted_unique_strings(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list):
        raise TrainManifestError(f"{where}: expected list")
    if len(value) > MAX_LIST_ITEMS:
        raise TrainManifestError(f"{where}: exceeds {MAX_LIST_ITEMS} entries")
    result = [_require_string(item, where=f"{where}[]") for item in value]
    if result != sorted(result):
        raise TrainManifestError(f"{where}: must be lexicographically sorted")
    if len(result) != len(set(result)):
        raise TrainManifestError(f"{where}: duplicate entries are not allowed")
    return result


def _require_changed_files(value: Any, *, where: str) -> list[str]:
    result = _require_sorted_unique_strings(value, where=where)
    if not result:
        raise TrainManifestError(f"{where}: must not be empty")
    for item in result:
        path = PurePosixPath(item)
        if (
            path.is_absolute()
            or ".." in path.parts
            or item in {".", ".."}
            or path.as_posix() != item
        ):
            raise TrainManifestError(f"{where}: non-canonical repository path {item!r}")
        if "\\" in item:
            raise TrainManifestError(f"{where}: use '/' separators, got {item!r}")
    return result


def normalize_manifest(
    manifest: Any,
    *,
    verify_declared_id: bool = True,
    require_id: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise TrainManifestError("manifest: expected object")

    _require_exact_keys(
        manifest,
        {"schema", "base_subject", "ordered_tranches", "cumulative_tip_sha"},
        {"train_id"},
        where="manifest",
    )
    if manifest["schema"] != SCHEMA:
        raise TrainManifestError(f"manifest.schema: expected {SCHEMA!r}")
    if require_id and "train_id" not in manifest:
        raise TrainManifestError("manifest.train_id: required but absent")

    base_subject = _require_sha(manifest["base_subject"], where="manifest.base_subject")
    cumulative_tip_sha = _require_sha(
        manifest["cumulative_tip_sha"], where="manifest.cumulative_tip_sha"
    )

    tranches_raw = manifest["ordered_tranches"]
    if not isinstance(tranches_raw, list) or not tranches_raw:
        raise TrainManifestError("manifest.ordered_tranches: expected non-empty list")
    if len(tranches_raw) > MAX_TRANCHES:
        raise TrainManifestError(
            f"manifest.ordered_tranches: exceeds V1 bound of {MAX_TRANCHES}"
        )

    normalized_tranches: list[dict[str, Any]] = []
    seen_commits: set[str] = set()
    seen_theorems: set[str] = set()
    expected_predecessor = base_subject

    for index, raw in enumerate(tranches_raw):
        where = f"manifest.ordered_tranches[{index}]"
        if not isinstance(raw, dict):
            raise TrainManifestError(f"{where}: expected object")
        _require_exact_keys(
            raw,
            {
                "commit_sha",
                "predecessor_sha",
                "theorem_id",
                "claim",
                "changed_files",
                "evidence_refs",
                "non_claims",
            },
            {"origin_pr"},
            where=where,
        )

        commit_sha = _require_sha(raw["commit_sha"], where=f"{where}.commit_sha")
        predecessor_sha = _require_sha(
            raw["predecessor_sha"], where=f"{where}.predecessor_sha"
        )
        theorem_id = _require_string(raw["theorem_id"], where=f"{where}.theorem_id")
        claim = _require_string(raw["claim"], where=f"{where}.claim")
        changed_files = _require_changed_files(
            raw["changed_files"], where=f"{where}.changed_files"
        )
        evidence_refs = _require_sorted_unique_strings(
            raw["evidence_refs"], where=f"{where}.evidence_refs"
        )
        non_claims = _require_sorted_unique_strings(
            raw["non_claims"], where=f"{where}.non_claims"
        )

        if commit_sha in seen_commits:
            raise TrainManifestError(f"{where}.commit_sha: duplicate commit {commit_sha}")
        if theorem_id in seen_theorems:
            raise TrainManifestError(f"{where}.theorem_id: duplicate theorem {theorem_id!r}")
        if predecessor_sha != expected_predecessor:
            raise TrainManifestError(
                f"{where}.predecessor_sha: expected {expected_predecessor}, got {predecessor_sha}"
            )

        normalized: dict[str, Any] = {
            "commit_sha": commit_sha,
            "predecessor_sha": predecessor_sha,
            "theorem_id": theorem_id,
            "claim": claim,
            "changed_files": changed_files,
            "evidence_refs": evidence_refs,
            "non_claims": non_claims,
        }
        if "origin_pr" in raw:
            origin_pr = raw["origin_pr"]
            if (
                isinstance(origin_pr, bool)
                or not isinstance(origin_pr, int)
                or origin_pr <= 0
                or origin_pr > MAX_ORIGIN_PR
            ):
                raise TrainManifestError(
                    f"{where}.origin_pr: expected integer in 1..{MAX_ORIGIN_PR}"
                )
            normalized["origin_pr"] = origin_pr

        normalized_tranches.append(normalized)
        seen_commits.add(commit_sha)
        seen_theorems.add(theorem_id)
        expected_predecessor = commit_sha

    if cumulative_tip_sha != normalized_tranches[-1]["commit_sha"]:
        raise TrainManifestError(
            "manifest.cumulative_tip_sha: must equal the final tranche commit_sha"
        )

    normalized_manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "base_subject": base_subject,
        "ordered_tranches": normalized_tranches,
        "cumulative_tip_sha": cumulative_tip_sha,
    }
    train_id = _compute_train_id_from_normalized(normalized_manifest)
    normalized_manifest["train_id"] = train_id

    if verify_declared_id and "train_id" in manifest:
        declared = manifest["train_id"]
        if not isinstance(declared, str) or _ID_RE.fullmatch(declared) is None:
            raise TrainManifestError("manifest.train_id: expected sha256:<64 lowercase hex>")
        if declared != train_id:
            raise TrainManifestError(
                f"manifest.train_id: expected {train_id}, got {declared}"
            )

    return normalized_manifest


def _canonical_payload_from_normalized(manifest: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in manifest.items() if key != "train_id"}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _compute_train_id_from_normalized(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        DOMAIN + _canonical_payload_from_normalized(manifest)
    ).hexdigest()
    return f"sha256:{digest}"


def canonical_payload_bytes(manifest: Any) -> bytes:
    normalized = normalize_manifest(manifest, verify_declared_id=False)
    return _canonical_payload_from_normalized(normalized)


def compute_train_id(manifest: Any) -> str:
    normalized = normalize_manifest(manifest, verify_declared_id=False)
    return _compute_train_id_from_normalized(normalized)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise TrainManifestError(f"git {' '.join(args)}: {detail}")
    return result.stdout.strip()


def validate_git_chain(manifest: Any, repo: Path) -> None:
    normalized = normalize_manifest(manifest)
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise TrainManifestError(f"repository has no .git directory: {repo}")

    _run_git(repo, "cat-file", "-e", f"{normalized['base_subject']}^{{commit}}")

    for index, tranche in enumerate(normalized["ordered_tranches"]):
        commit_sha = tranche["commit_sha"]
        predecessor_sha = tranche["predecessor_sha"]
        parents = _run_git(repo, "show", "-s", "--format=%P", commit_sha).split()
        if parents != [predecessor_sha]:
            raise TrainManifestError(
                f"tranche {index} {commit_sha}: expected exactly one parent "
                f"{predecessor_sha}, got {parents}"
            )

        changed = sorted(
            line
            for line in _run_git(
                repo,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit_sha,
            ).splitlines()
            if line
        )
        if changed != tranche["changed_files"]:
            raise TrainManifestError(
                f"tranche {index} {commit_sha}: changed_files mismatch; "
                f"declared={tranche['changed_files']!r}, actual={changed!r}"
            )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_manifest(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise TrainManifestError(
                f"{path}: manifest exceeds {MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TrainManifestError(f"{path}: {error}") from error
    return normalize_manifest(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="integration-train JSON manifest")
    parser.add_argument(
        "--require-id",
        action="store_true",
        help="require the manifest to contain the exact computed train_id",
    )
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="verify direct-parent ancestry and changed_files against a local Git checkout",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Git repository used by --verify-git (default: current directory)",
    )
    parser.add_argument(
        "--print-normalized",
        action="store_true",
        help="print canonical normalized JSON including train_id",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_manifest(args.manifest, require_id=args.require_id)
        if args.verify_git:
            validate_git_chain(normalized, args.repo)
    except TrainManifestError as error:
        print(f"integration-train manifest invalid: {error}", file=sys.stderr)
        return 2

    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    else:
        print(normalized["train_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
