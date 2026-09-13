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
_STORE_PATH = re.compile(r"^/nix/store/[0-9a-z]{32}-[^\x00\n]+$")

NON_CLAIMS = [
    "does not prove recursive Nix store closure contents or references",
    "does not prove qualification tool executable bytes or version outputs",
    "does not prove the qualification recipe executed",
    "does not authenticate the capture/execution provider",
    "does not establish externally anchored chronology",
]


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


def _require_store_path(value: str, *, where: str) -> str:
    if not isinstance(value, str) or _STORE_PATH.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected canonical /nix/store path")
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
    output_paths = sorted({_require_store_path(line, where="qualification environment resolution.output_path")
                           for line in raw_outputs.splitlines() if line})
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
    value["environment_resolution_id"] = framing.semantic_sha256_id(ID_DOMAIN, _fields(value))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("closure", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = selection_mod.load_selection(args.selection, require_id=True)
        subject = subject_mod.load_subject(args.subject, require_id=True)
        closure = closure_mod.load_closure(args.closure, require_id=True)
        resolution = resolve_environment_selection(selection, subject, closure, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification environment resolution invalid: {error}")
        return 2
    print(resolution["environment_resolution_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
