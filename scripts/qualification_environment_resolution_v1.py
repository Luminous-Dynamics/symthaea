#!/usr/bin/env python3
"""Verify exact qualification-environment selection -> Nix derivation/output resolution.

This is the narrow bridge between a Git-bound `QualificationEnvironmentSelectionV1` and a later
realization witness. It executes Nix against an immutable local Git-flake reference to the exact
qualification subject commit, not against ambient working-tree bytes.

Positive theorem:

    EnvironmentSelectionGitBindingVerifiedOnly
    + exact local git+file flake reference at source_commit
    + successful Nix derivation/output resolution
        -> EnvironmentSelectionResolutionVerifiedOnly

This does not prove recursive store closure capture, tool bytes, qualification recipe execution,
provider authenticity, chronology, or scientific validity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import integration_train_manifest as train
import qualification_environment_selection_v1 as selection_mod
import qualification_environment_selection_verifier_v1 as selection_verifier
import qualification_framing_v1 as framing
import qualification_input_closure_v1 as closure_mod
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-environment-resolution.v1"
ID_DOMAIN = "symthaea.qualification-environment-resolution-id.v1"
POSITIVE_STATE = "EnvironmentSelectionResolutionVerifiedOnly"
_STORE_PATH = re.compile(r"^/nix/store/[0-9abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9+._?=-]+$")
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")

NON_CLAIMS = sorted([
    "does not prove recursive Nix store closure contents or references",
    "does not prove qualification tool executable bytes or version outputs",
    "does not prove the qualification recipe executed",
    "does not authenticate the capture/execution provider",
    "does not establish externally anchored chronology",
])


def _run(repo: Path, argv: list[str]) -> str:
    proc = subprocess.run(
        argv,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "NIX_CONFIG": os.environ.get("NIX_CONFIG", "")},
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise train.TrainManifestError(f"environment resolution command failed: {argv!r}: {detail}")
    return proc.stdout.strip()


def _require_store_path(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _STORE_PATH.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected canonical /nix/store path")
    return value


def _require_sha256_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def immutable_flake_ref(repo: Path, source_commit: str, flake_output: str) -> str:
    repo = repo.resolve()
    # URI path is transport only; semantic resolution identity commits the exact Git source and
    # returned derivation/output, not the host-local repository location.
    uri_path = quote(str(repo), safe="/")
    return f"git+file://{uri_path}?rev={source_commit}#{flake_output}"


def _fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("environment_selection_id", framing.encode_text(value["environment_selection_id"])),
        ("qualification_subject_id", framing.encode_text(value["qualification_subject_id"])),
        ("input_closure_id", framing.encode_text(value["input_closure_id"])),
        ("target_system", framing.encode_enum(value["target_system"])),
        ("flake_output", framing.encode_text(value["flake_output"])),
        ("derivation_path", framing.encode_text(value["derivation_path"])),
        (
            "output_paths",
            framing.encode_set(framing.encode_text(path) for path in value["output_paths"]),
        ),
        ("nix_version", framing.encode_text(value["nix_version"])),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(item) for item in value["non_claims"]),
        ),
    ]


def normalize_resolution(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification environment resolution: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "state",
            "environment_selection_id",
            "qualification_subject_id",
            "input_closure_id",
            "target_system",
            "flake_output",
            "derivation_path",
            "output_paths",
            "nix_version",
            "non_claims",
        },
        {"environment_resolution_id"},
        where="qualification environment resolution",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(
            f"qualification environment resolution.schema: expected {SCHEMA!r}"
        )
    if value["state"] != POSITIVE_STATE:
        raise train.TrainManifestError(
            "qualification environment resolution.state: positive V1 state required"
        )
    if require_id and "environment_resolution_id" not in value:
        raise train.TrainManifestError(
            "qualification environment resolution.environment_resolution_id: required"
        )

    output_paths_raw = value["output_paths"]
    if not isinstance(output_paths_raw, list) or not output_paths_raw:
        raise train.TrainManifestError(
            "qualification environment resolution.output_paths: expected non-empty array"
        )
    output_paths = [
        _require_store_path(path, where=f"qualification environment resolution.output_paths[{index}]")
        for index, path in enumerate(output_paths_raw)
    ]
    if output_paths != sorted(set(output_paths)):
        raise train.TrainManifestError(
            "qualification environment resolution.output_paths: must be sorted and unique"
        )

    non_claims = train._require_sorted_unique_strings(
        value["non_claims"], where="qualification environment resolution.non_claims"
    )
    if non_claims != NON_CLAIMS:
        raise train.TrainManifestError(
            "qualification environment resolution.non_claims: V1 theorem boundary must equal frozen set"
        )

    target_system = subject_mod._require_string(
        value["target_system"], where="qualification environment resolution.target_system"
    )
    if target_system not in selection_mod.SUPPORTED_SYSTEMS:
        raise train.TrainManifestError(
            "qualification environment resolution.target_system: unsupported system"
        )
    flake_output = subject_mod._require_string(
        value["flake_output"], where="qualification environment resolution.flake_output"
    )
    expected_prefix = f"devShells.{target_system}."
    if not flake_output.startswith(expected_prefix):
        raise train.TrainManifestError(
            "qualification environment resolution.flake_output: target system mismatch"
        )

    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "state": POSITIVE_STATE,
        "environment_selection_id": _require_sha256_id(
            value["environment_selection_id"],
            where="qualification environment resolution.environment_selection_id",
        ),
        "qualification_subject_id": _require_sha256_id(
            value["qualification_subject_id"],
            where="qualification environment resolution.qualification_subject_id",
        ),
        "input_closure_id": _require_sha256_id(
            value["input_closure_id"],
            where="qualification environment resolution.input_closure_id",
        ),
        "target_system": target_system,
        "flake_output": flake_output,
        "derivation_path": _require_store_path(
            value["derivation_path"],
            where="qualification environment resolution.derivation_path",
        ),
        "output_paths": output_paths,
        "nix_version": subject_mod._require_string(
            value["nix_version"], where="qualification environment resolution.nix_version"
        ),
        "non_claims": list(NON_CLAIMS),
    }
    expected_id = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    normalized["environment_resolution_id"] = expected_id
    if verify_declared_id and "environment_resolution_id" in value:
        declared = _require_sha256_id(
            value["environment_resolution_id"],
            where="qualification environment resolution.environment_resolution_id",
        )
        if declared != expected_id:
            raise train.TrainManifestError(
                "qualification environment resolution.environment_resolution_id: resolution semantics changed"
            )
    return normalized


def validate_against_context(
    resolution: Any, selection: Any, subject: Any, closure: Any
) -> dict[str, Any]:
    normalized = normalize_resolution(resolution, require_id=True)
    normalized_selection = selection_mod.normalize_selection(selection, require_id=True)
    normalized_subject = subject_mod.normalize_subject(subject, require_id=True)
    normalized_closure = closure_mod.validate_against_subject(closure, normalized_subject)

    if normalized["environment_selection_id"] != normalized_selection["environment_selection_id"]:
        raise train.TrainManifestError(
            "qualification environment resolution: selection identity mismatch"
        )
    if normalized["qualification_subject_id"] != normalized_subject["subject_id"]:
        raise train.TrainManifestError(
            "qualification environment resolution: subject identity mismatch"
        )
    if normalized["input_closure_id"] != normalized_closure["input_closure_id"]:
        raise train.TrainManifestError(
            "qualification environment resolution: input closure identity mismatch"
        )
    if normalized["target_system"] != normalized_selection["target_system"]:
        raise train.TrainManifestError(
            "qualification environment resolution: target system mismatch"
        )
    if normalized["flake_output"] != normalized_selection["flake_output"]:
        raise train.TrainManifestError(
            "qualification environment resolution: flake output mismatch"
        )
    return normalized


def resolve_environment_selection(
    selection: Any,
    subject: Any,
    closure: Any,
    repo: Path,
) -> dict[str, Any]:
    normalized_subject = subject_mod.normalize_subject(subject, require_id=True)
    normalized_closure = closure_mod.normalize_closure(closure, require_id=True)
    selection_witness = selection_verifier.verify_selection_git_binding(
        selection, normalized_subject, normalized_closure, repo
    )
    normalized_selection = selection_mod.normalize_selection(selection, require_id=True)

    flake_ref = immutable_flake_ref(
        repo,
        normalized_subject["source_commit"],
        normalized_selection["flake_output"],
    )

    # `.drvPath` evaluates the exact flake output without relying on caller-supplied drv identity.
    derivation_path = _require_store_path(
        _run(repo, ["nix", "eval", "--raw", f"{flake_ref}.drvPath"]),
        where="qualification environment resolution.derivation_path",
    )

    # Realize the exact selected output and capture every direct output path reported by Nix.
    raw_outputs = _run(repo, ["nix", "build", "--no-link", "--print-out-paths", flake_ref])
    output_paths = sorted(
        {
            _require_store_path(
                line, where="qualification environment resolution.output_path"
            )
            for line in raw_outputs.splitlines()
            if line
        }
    )
    if not output_paths:
        raise train.TrainManifestError(
            "qualification environment resolution: Nix returned no realized output path"
        )

    nix_version = _run(repo, ["nix", "--version"])
    if not nix_version:
        raise train.TrainManifestError("qualification environment resolution: empty nix version")

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "state": POSITIVE_STATE,
        "environment_selection_id": selection_witness["environment_selection_id"],
        "qualification_subject_id": normalized_subject["subject_id"],
        "input_closure_id": normalized_closure["input_closure_id"],
        "target_system": normalized_selection["target_system"],
        "flake_output": normalized_selection["flake_output"],
        "derivation_path": derivation_path,
        "output_paths": output_paths,
        "nix_version": nix_version,
        "non_claims": list(NON_CLAIMS),
    }
    return normalize_resolution(value, verify_declared_id=False)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = item
    return result


def load_resolution(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: environment resolution exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_resolution(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("selection", type=Path)
    resolve.add_argument("subject", type=Path)
    resolve.add_argument("closure", type=Path)
    resolve.add_argument("--repo", type=Path, default=Path.cwd())

    verify = subparsers.add_parser("verify")
    verify.add_argument("resolution", type=Path)
    verify.add_argument("selection", type=Path)
    verify.add_argument("subject", type=Path)
    verify.add_argument("closure", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = selection_mod.load_selection(args.selection, require_id=True)
        subject = subject_mod.load_subject(args.subject, require_id=True)
        closure = closure_mod.load_closure(args.closure, require_id=True)
        if args.command == "resolve":
            resolution = resolve_environment_selection(selection, subject, closure, args.repo)
        else:
            resolution = load_resolution(args.resolution, require_id=True)
            resolution = validate_against_context(resolution, selection, subject, closure)
    except train.TrainManifestError as error:
        print(f"qualification environment resolution invalid: {error}")
        return 2
    print(resolution["environment_resolution_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
