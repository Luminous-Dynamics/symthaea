#!/usr/bin/env python3
"""Derive finalized integration-train manifests from semantic author specs and Git facts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as manifest

AUTHOR_SCHEMA = "symthaea.integration-train-author.v1"


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
        raise manifest.TrainManifestError(
            f"{where}: missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise manifest.TrainManifestError(
            f"{where}: unknown fields: {', '.join(unknown)}"
        )


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise manifest.TrainManifestError(f"{where}: expected string")
    if value != value.strip():
        raise manifest.TrainManifestError(
            f"{where}: leading/trailing whitespace is not canonical"
        )
    if not value:
        raise manifest.TrainManifestError(f"{where}: must not be empty")
    if unicodedata.normalize("NFC", value) != value:
        raise manifest.TrainManifestError(
            f"{where}: text must use Unicode NFC normalization"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise manifest.TrainManifestError(
            f"{where}: control characters are not canonical"
        )
    if len(value.encode("utf-8")) > manifest.MAX_TEXT_BYTES:
        raise manifest.TrainManifestError(
            f"{where}: exceeds {manifest.MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return value


def _require_sha(value: Any, *, where: str) -> str:
    # Reuse finalized-manifest validation so SHA syntax has exactly one definition.
    return manifest._require_sha(value, where=where)


def _require_sorted_unique_strings(value: Any, *, where: str) -> list[str]:
    return manifest._require_sorted_unique_strings(value, where=where)


def _normalize_origin_pr(raw: dict[str, Any], *, where: str) -> int | None:
    if "origin_pr" not in raw:
        return None
    origin_pr = raw["origin_pr"]
    if (
        isinstance(origin_pr, bool)
        or not isinstance(origin_pr, int)
        or origin_pr <= 0
        or origin_pr > manifest.MAX_ORIGIN_PR
    ):
        raise manifest.TrainManifestError(
            f"{where}.origin_pr: expected integer in 1..{manifest.MAX_ORIGIN_PR}"
        )
    return origin_pr


def normalize_author_spec(spec: Any) -> dict[str, Any]:
    """Normalize semantic input; Git-derived predecessor/file fields are forbidden."""
    if not isinstance(spec, dict):
        raise manifest.TrainManifestError("author_spec: expected object")
    _require_exact_keys(
        spec,
        {"schema", "base_subject", "ordered_tranches"},
        set(),
        where="author_spec",
    )
    if spec["schema"] != AUTHOR_SCHEMA:
        raise manifest.TrainManifestError(
            f"author_spec.schema: expected {AUTHOR_SCHEMA!r}"
        )

    base_subject = _require_sha(
        spec["base_subject"], where="author_spec.base_subject"
    )
    raw_tranches = spec["ordered_tranches"]
    if not isinstance(raw_tranches, list) or not raw_tranches:
        raise manifest.TrainManifestError(
            "author_spec.ordered_tranches: expected non-empty list"
        )
    if len(raw_tranches) > manifest.MAX_TRANCHES:
        raise manifest.TrainManifestError(
            "author_spec.ordered_tranches: "
            f"exceeds V1 bound of {manifest.MAX_TRANCHES}"
        )

    normalized: list[dict[str, Any]] = []
    seen_commits: set[str] = set()
    seen_theorems: set[str] = set()

    for index, raw in enumerate(raw_tranches):
        where = f"author_spec.ordered_tranches[{index}]"
        if not isinstance(raw, dict):
            raise manifest.TrainManifestError(f"{where}: expected object")
        _require_exact_keys(
            raw,
            {"commit_sha", "theorem_id", "claim", "evidence_refs", "non_claims"},
            {"origin_pr"},
            where=where,
        )
        commit_sha = _require_sha(raw["commit_sha"], where=f"{where}.commit_sha")
        theorem_id = _require_string(raw["theorem_id"], where=f"{where}.theorem_id")
        claim = _require_string(raw["claim"], where=f"{where}.claim")
        evidence_refs = _require_sorted_unique_strings(
            raw["evidence_refs"], where=f"{where}.evidence_refs"
        )
        non_claims = _require_sorted_unique_strings(
            raw["non_claims"], where=f"{where}.non_claims"
        )
        if commit_sha in seen_commits:
            raise manifest.TrainManifestError(
                f"{where}.commit_sha: duplicate commit {commit_sha}"
            )
        if theorem_id in seen_theorems:
            raise manifest.TrainManifestError(
                f"{where}.theorem_id: duplicate theorem {theorem_id!r}"
            )

        tranche: dict[str, Any] = {
            "commit_sha": commit_sha,
            "theorem_id": theorem_id,
            "claim": claim,
            "evidence_refs": evidence_refs,
            "non_claims": non_claims,
        }
        origin_pr = _normalize_origin_pr(raw, where=where)
        if origin_pr is not None:
            tranche["origin_pr"] = origin_pr
        normalized.append(tranche)
        seen_commits.add(commit_sha)
        seen_theorems.add(theorem_id)

    return {
        "schema": AUTHOR_SCHEMA,
        "base_subject": base_subject,
        "ordered_tranches": normalized,
    }


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
        raise manifest.TrainManifestError(f"git {' '.join(args)}: {detail}")
    return result.stdout.strip()


def _require_git_repo(repo: Path) -> Path:
    repo = repo.resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise manifest.TrainManifestError(
            f"repository is not a Git work tree: {repo}"
        )
    return repo


def _commit_parent(repo: Path, commit_sha: str) -> str:
    parents = _run_git(repo, "show", "-s", "--format=%P", commit_sha).split()
    if len(parents) != 1:
        raise manifest.TrainManifestError(
            f"commit {commit_sha}: expected exactly one parent, got {parents}"
        )
    return parents[0]


def _changed_files(repo: Path, commit_sha: str) -> list[str]:
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
    return manifest._require_changed_files(
        changed, where=f"commit {commit_sha}.changed_files"
    )


def derive_manifest(spec: Any, repo: Path) -> dict[str, Any]:
    """Derive all Git-owned facts and return a canonical finalized manifest."""
    authored = normalize_author_spec(spec)
    repo = _require_git_repo(repo)
    base_subject = authored["base_subject"]
    _run_git(repo, "cat-file", "-e", f"{base_subject}^{{commit}}")

    expected_parent = base_subject
    derived: list[dict[str, Any]] = []
    for index, tranche in enumerate(authored["ordered_tranches"]):
        commit_sha = tranche["commit_sha"]
        _run_git(repo, "cat-file", "-e", f"{commit_sha}^{{commit}}")
        actual_parent = _commit_parent(repo, commit_sha)
        if actual_parent != expected_parent:
            raise manifest.TrainManifestError(
                f"author_spec.ordered_tranches[{index}].commit_sha: "
                f"expected parent {expected_parent}, got {actual_parent}"
            )
        item: dict[str, Any] = {
            "commit_sha": commit_sha,
            "predecessor_sha": actual_parent,
            "theorem_id": tranche["theorem_id"],
            "claim": tranche["claim"],
            "changed_files": _changed_files(repo, commit_sha),
            "evidence_refs": tranche["evidence_refs"],
            "non_claims": tranche["non_claims"],
        }
        if "origin_pr" in tranche:
            item["origin_pr"] = tranche["origin_pr"]
        derived.append(item)
        expected_parent = commit_sha

    finalized = {
        "schema": manifest.SCHEMA,
        "base_subject": base_subject,
        "ordered_tranches": derived,
        "cumulative_tip_sha": derived[-1]["commit_sha"],
    }
    return manifest.normalize_manifest(finalized, verify_declared_id=False)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise manifest.TrainManifestError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def load_author_spec(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > manifest.MAX_MANIFEST_BYTES:
            raise manifest.TrainManifestError(
                f"{path}: author spec exceeds {manifest.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except manifest.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise manifest.TrainManifestError(f"{path}: {error}") from error
    return normalize_author_spec(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("author_spec", type=Path, help="semantic train author JSON")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Git work tree used to derive ancestry and changed files",
    )
    parser.add_argument(
        "--print-normalized",
        action="store_true",
        help="print finalized canonical JSON including train_id",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        finalized = derive_manifest(load_author_spec(args.author_spec), args.repo)
    except manifest.TrainManifestError as error:
        print(f"integration-train author spec invalid: {error}", file=sys.stderr)
        return 2

    if args.print_normalized:
        print(json.dumps(finalized, indent=2, ensure_ascii=False))
    else:
        print(finalized["train_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
