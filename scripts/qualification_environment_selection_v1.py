#!/usr/bin/env python3
"""Exact Nix qualification-environment selection preimage V1.

This object proves only *what environment selection was requested* from exact source inputs:

    exact whole-tree InputClosureId
    + target system
    + exact flake output attribute
    + exact Git blobs for flake.nix / flake.lock / rust-toolchain.toml
        -> QualificationEnvironmentSelectionIdV1

It does NOT prove that Nix evaluated the output, realized its store closure, entered the shell,
or that observed compiler/tool binaries came from that realization. Those require a separate
execution-time realization witness before an Attempt V2 may claim an exact environment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_input_closure_v1 as closure_mod
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-environment-selection.v1"
ID_DOMAIN = "symthaea.qualification-environment-selection-id.v1"
KIND = "NixFlakeDevShell"
SELECTION_STRENGTH = "ImmutableSourceSelectionNotRealization"
SUPPORTED_SYSTEMS = {"x86_64-linux", "aarch64-linux"}
SELECTOR_PATHS = ("flake.lock", "flake.nix", "rust-toolchain.toml")
_FLAKE_OUTPUT = re.compile(r"^devShells\.[A-Za-z0-9_+.-]+\.[A-Za-z0-9_+.-]+$")

NON_CLAIMS = [
    "does not prove Nix evaluated the selected output",
    "does not prove the selected store closure was realized",
    "does not prove tool binaries observed at execution came from the selection",
    "does not prove the executor entered the selected environment",
]


def _require_selector_path(value: Any, *, where: str) -> str:
    value = subject_mod._require_string(value, where=where)
    if value not in SELECTOR_PATHS:
        raise train.TrainManifestError(f"{where}: unsupported V1 selector path {value!r}")
    return value


def _normalize_selector_blobs(
    value: Any, *, object_format: str
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise train.TrainManifestError(
            "qualification environment selection.selector_blobs: expected array"
        )
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        where = f"qualification environment selection.selector_blobs[{index}]"
        if not isinstance(item, dict):
            raise train.TrainManifestError(f"{where}: expected object")
        subject_mod._require_exact_keys(item, {"path", "git_blob"}, set(), where=where)
        path = _require_selector_path(item["path"], where=f"{where}.path")
        if path in seen:
            raise train.TrainManifestError(
                f"qualification environment selection.selector_blobs: duplicate path {path!r}"
            )
        seen.add(path)
        out.append(
            {
                "path": path,
                "git_blob": subject_mod._require_object_id(
                    item["git_blob"], where=f"{where}.git_blob", object_format=object_format
                ),
            }
        )
    out.sort(key=lambda item: item["path"])
    if tuple(item["path"] for item in out) != SELECTOR_PATHS:
        missing = sorted(set(SELECTOR_PATHS) - {item["path"] for item in out})
        extra = sorted({item["path"] for item in out} - set(SELECTOR_PATHS))
        raise train.TrainManifestError(
            f"qualification environment selection.selector_blobs: exact V1 selector set required: missing={missing}, extra={extra}"
        )
    return out


def _blob_frame(item: dict[str, str]) -> bytes:
    return framing.frame_record(
        "symthaea.qualification-environment-selector-blob.v1",
        [
            ("path", framing.encode_text(item["path"])),
            ("git_blob", framing.encode_text(item["git_blob"])),
        ],
    )


def _fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("kind", framing.encode_enum(value["kind"])),
        ("selection_strength", framing.encode_enum(value["selection_strength"])),
        ("input_closure_id", framing.encode_text(value["input_closure_id"])),
        ("object_format", framing.encode_enum(value["object_format"])),
        ("target_system", framing.encode_enum(value["target_system"])),
        ("flake_output", framing.encode_text(value["flake_output"])),
        (
            "selector_blobs",
            framing.encode_list([_blob_frame(item) for item in value["selector_blobs"]]),
        ),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(item) for item in value["non_claims"]),
        ),
    ]


def normalize_selection(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification environment selection: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "kind",
            "selection_strength",
            "input_closure_id",
            "object_format",
            "target_system",
            "flake_output",
            "selector_blobs",
            "non_claims",
        },
        {"environment_selection_id"},
        where="qualification environment selection",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(
            f"qualification environment selection.schema: expected {SCHEMA!r}"
        )
    if value["kind"] != KIND or value["selection_strength"] != SELECTION_STRENGTH:
        raise train.TrainManifestError(
            "qualification environment selection: unsupported V1 kind/selection strength"
        )
    if require_id and "environment_selection_id" not in value:
        raise train.TrainManifestError(
            "qualification environment selection.environment_selection_id: required"
        )

    input_closure_id = value["input_closure_id"]
    if (
        not isinstance(input_closure_id, str)
        or not input_closure_id.startswith("sha256:")
        or len(input_closure_id) != 71
        or any(char not in "0123456789abcdef" for char in input_closure_id[7:])
    ):
        raise train.TrainManifestError(
            "qualification environment selection.input_closure_id: expected sha256:<64 lowercase hex>"
        )
    object_format = subject_mod._require_object_format(
        value["object_format"], where="qualification environment selection.object_format"
    )
    target_system = subject_mod._require_string(
        value["target_system"], where="qualification environment selection.target_system"
    )
    if target_system not in SUPPORTED_SYSTEMS:
        raise train.TrainManifestError(
            f"qualification environment selection.target_system: unsupported V1 system {target_system!r}"
        )
    flake_output = subject_mod._require_string(
        value["flake_output"], where="qualification environment selection.flake_output"
    )
    if _FLAKE_OUTPUT.fullmatch(flake_output) is None:
        raise train.TrainManifestError(
            "qualification environment selection.flake_output: expected devShells.<system>.<name>"
        )
    expected_prefix = f"devShells.{target_system}."
    if not flake_output.startswith(expected_prefix):
        raise train.TrainManifestError(
            "qualification environment selection.flake_output: target system mismatch"
        )
    non_claims = train._require_sorted_unique_strings(
        value["non_claims"], where="qualification environment selection.non_claims"
    )
    if non_claims != NON_CLAIMS:
        raise train.TrainManifestError(
            "qualification environment selection.non_claims: V1 theorem boundary must equal the frozen set"
        )

    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "selection_strength": SELECTION_STRENGTH,
        "input_closure_id": input_closure_id,
        "object_format": object_format,
        "target_system": target_system,
        "flake_output": flake_output,
        "selector_blobs": _normalize_selector_blobs(
            value["selector_blobs"], object_format=object_format
        ),
        "non_claims": list(NON_CLAIMS),
    }
    expected_id = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    normalized["environment_selection_id"] = expected_id
    if verify_declared_id and "environment_selection_id" in value:
        declared = value["environment_selection_id"]
        if declared != expected_id:
            raise train.TrainManifestError(
                "qualification environment selection.environment_selection_id: selection semantics changed"
            )
    return normalized


def validate_against_closure(selection: Any, closure: Any) -> dict[str, Any]:
    normalized = normalize_selection(selection, require_id=True)
    normalized_closure = closure_mod.normalize_closure(closure, require_id=True)
    if normalized["input_closure_id"] != normalized_closure["input_closure_id"]:
        raise train.TrainManifestError(
            "qualification environment selection: input closure does not match selection"
        )
    if normalized["object_format"] != normalized_closure["object_format"]:
        raise train.TrainManifestError(
            "qualification environment selection: Git object format does not match closure"
        )
    return normalized


def build_from_git(
    *,
    subject: Any,
    closure: Any,
    repo: Path,
    target_system: str,
    flake_output: str,
) -> dict[str, Any]:
    """Resolve the three selector blobs from the exact subject commit in a local Git checkout."""

    normalized_subject = subject_mod.normalize_subject(subject, require_id=True)
    normalized_closure = closure_mod.validate_git_binding(closure, normalized_subject, repo)
    selector_blobs: list[dict[str, str]] = []
    for path in SELECTOR_PATHS:
        blob = subject_mod._run_git(
            repo, "rev-parse", f"{normalized_subject['source_commit']}:{path}"
        )
        subject_mod._run_git(repo, "cat-file", "-e", f"{blob}^{{blob}}")
        selector_blobs.append({"path": path, "git_blob": blob})

    raw = {
        "schema": SCHEMA,
        "kind": KIND,
        "selection_strength": SELECTION_STRENGTH,
        "input_closure_id": normalized_closure["input_closure_id"],
        "object_format": normalized_subject["object_format"],
        "target_system": target_system,
        "flake_output": flake_output,
        "selector_blobs": selector_blobs,
        "non_claims": list(NON_CLAIMS),
    }
    return normalize_selection(raw, verify_declared_id=False)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = item
    return result


def load_selection(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: environment selection exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_selection(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--require-id", action="store_true")
    parser.add_argument("--print-normalized", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_selection(args.selection, require_id=args.require_id)
    except train.TrainManifestError as error:
        print(f"qualification environment selection invalid: {error}")
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, sort_keys=True, indent=2))
    else:
        print(normalized["environment_selection_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
