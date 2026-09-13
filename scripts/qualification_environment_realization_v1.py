#!/usr/bin/env python3
"""Reconstruct one provider-neutral qualification environment from captured preimages.

This verifier consumes file-backed capture preimages rather than caller-provided content hashes.
It revalidates the exact Git subject/closure/selection/resolution chain, reconstructs a recursive
store-closure commitment and exact required-tool commitments, then derives one semantic
`QualificationEnvironmentId`.

Positive theorem:

    exact Git-bound environment selection
    + exact verified Nix selection resolution
    + complete captured closure/reference preimages
    + exact captured required-tool executable/version preimages
        -> QualificationEnvironmentRealizationCommitmentsVerifiedOnly

This remains weaker than recipe execution, PASS, provider authenticity, chronology, scientific
validity, merge readiness, or action authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_environment_resolution_v1 as resolution_mod
import qualification_environment_selection_verifier_v1 as selection_verifier
import qualification_framing_v1 as framing
import qualification_tool_requirements_v1 as requirements_mod
import qualification_subject as subject_mod

CAPTURE_SCHEMA = "symthaea.qualification-environment-realization-capture.v1"
WITNESS_SCHEMA = "symthaea.qualification-environment-realization.v1"
ENV_ID_DOMAIN = "symthaea.qualification-environment-id.v1"
POSITIVE_STATE = "QualificationEnvironmentRealizationCommitmentsVerifiedOnly"
MAX_CAPTURE_FILE_BYTES = 8 * 1024 * 1024 * 1024

NON_CLAIMS = sorted([
    "does not authenticate the capture or execution provider",
    "does not establish externally anchored chronology",
    "does not prove a PASS qualification result",
    "does not prove captured reference sets were derived from NAR bytes",
    "does not prove scientific validity",
    "does not prove the qualification recipe executed",
    "does not provide merge deployment or action authority",
])


def _sha256_file(path: Path) -> tuple[str, int]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise train.TrainManifestError(f"capture file unavailable: {path}: {error}") from error
    if size < 0 or size > MAX_CAPTURE_FILE_BYTES:
        raise train.TrainManifestError(
            f"capture file size outside V1 bound 0..{MAX_CAPTURE_FILE_BYTES}: {path}: {size}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise train.TrainManifestError(f"capture file unreadable: {path}: {error}") from error
    return "sha256:" + digest.hexdigest(), size


def _capture_path(root: Path, value: Any, *, where: str) -> Path:
    text = subject_mod._require_string(value, where=where)
    raw = Path(text)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise train.TrainManifestError(f"{where}: expected canonical relative capture path")
    root = root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise train.TrainManifestError(f"{where}: capture path escapes capture root") from error
    if not resolved.is_file():
        raise train.TrainManifestError(f"{where}: capture file does not exist: {resolved}")
    return resolved


def _normalize_store_capture(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError(f"{where}: expected object")
    subject_mod._require_exact_keys(
        value, {"store_path", "nar_file", "references"}, set(), where=where
    )
    store_path = resolution_mod._require_store_path(value["store_path"], where=f"{where}.store_path")
    nar_file = subject_mod._require_string(value["nar_file"], where=f"{where}.nar_file")
    references = value["references"]
    if not isinstance(references, list):
        raise train.TrainManifestError(f"{where}.references: expected array")
    references = [
        resolution_mod._require_store_path(ref, where=f"{where}.references[{index}]")
        for index, ref in enumerate(references)
    ]
    if references != sorted(set(references)):
        raise train.TrainManifestError(f"{where}.references: must be sorted and unique")
    return {"store_path": store_path, "nar_file": nar_file, "references": references}


def _normalize_tool_capture(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError(f"{where}: expected object")
    subject_mod._require_exact_keys(
        value,
        {"tool_name", "executable_path", "executable_file", "version_argv", "version_output_file"},
        set(),
        where=where,
    )
    version_argv = value["version_argv"]
    if not isinstance(version_argv, list) or not version_argv:
        raise train.TrainManifestError(f"{where}.version_argv: expected non-empty array")
    version_argv = [
        subject_mod._require_string(item, where=f"{where}.version_argv[{index}]")
        for index, item in enumerate(version_argv)
    ]
    return {
        "tool_name": subject_mod._require_string(value["tool_name"], where=f"{where}.tool_name"),
        "executable_path": subject_mod._require_string(
            value["executable_path"], where=f"{where}.executable_path"
        ),
        "executable_file": subject_mod._require_string(
            value["executable_file"], where=f"{where}.executable_file"
        ),
        "version_argv": version_argv,
        "version_output_file": subject_mod._require_string(
            value["version_output_file"], where=f"{where}.version_output_file"
        ),
    }


def normalize_capture(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("environment realization capture: expected object")
    subject_mod._require_exact_keys(
        value,
        {"schema", "environment_resolution_id", "store_objects", "tools", "evidence_refs"},
        set(),
        where="environment realization capture",
    )
    if value["schema"] != CAPTURE_SCHEMA:
        raise train.TrainManifestError(
            f"environment realization capture.schema: expected {CAPTURE_SCHEMA!r}"
        )
    environment_resolution_id = resolution_mod._require_sha256_id(
        value["environment_resolution_id"],
        where="environment realization capture.environment_resolution_id",
    )

    raw_objects = value["store_objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise train.TrainManifestError(
            "environment realization capture.store_objects: expected non-empty array"
        )
    store_objects = [
        _normalize_store_capture(item, where=f"environment realization capture.store_objects[{index}]")
        for index, item in enumerate(raw_objects)
    ]
    store_objects.sort(key=lambda item: item["store_path"])
    paths = [item["store_path"] for item in store_objects]
    if len(paths) != len(set(paths)):
        raise train.TrainManifestError(
            "environment realization capture.store_objects: duplicate store_path"
        )

    raw_tools = value["tools"]
    if not isinstance(raw_tools, list) or not raw_tools:
        raise train.TrainManifestError(
            "environment realization capture.tools: expected non-empty array"
        )
    tools = [
        _normalize_tool_capture(item, where=f"environment realization capture.tools[{index}]")
        for index, item in enumerate(raw_tools)
    ]
    tools.sort(key=lambda item: item["tool_name"])
    tool_names = [item["tool_name"] for item in tools]
    if len(tool_names) != len(set(tool_names)):
        raise train.TrainManifestError(
            "environment realization capture.tools: duplicate tool_name"
        )

    evidence_refs = train._require_sorted_unique_strings(
        value["evidence_refs"], where="environment realization capture.evidence_refs"
    )
    if not evidence_refs:
        raise train.TrainManifestError(
            "environment realization capture.evidence_refs: at least one occurrence/provenance reference required"
        )
    return {
        "schema": CAPTURE_SCHEMA,
        "environment_resolution_id": environment_resolution_id,
        "store_objects": store_objects,
        "tools": tools,
        "evidence_refs": evidence_refs,
    }


def _store_semantic_frame(item: dict[str, Any]) -> bytes:
    return framing.frame_record(
        "symthaea.qualification-realized-store-object.v1",
        [
            ("store_path", framing.encode_text(item["store_path"])),
            ("nar_sha256", framing.encode_text(item["nar_sha256"])),
            ("nar_size", framing.encode_u64(item["nar_size"])),
            (
                "references",
                framing.encode_set(framing.encode_text(ref) for ref in item["references"]),
            ),
        ],
    )


def _tool_semantic_frame(item: dict[str, Any]) -> bytes:
    return framing.frame_record(
        "symthaea.qualification-realized-tool.v1",
        [
            ("tool_name", framing.encode_text(item["tool_name"])),
            ("executable_path", framing.encode_text(item["executable_path"])),
            ("executable_sha256", framing.encode_text(item["executable_sha256"])),
            ("executable_size", framing.encode_u64(item["executable_size"])),
            (
                "version_argv",
                framing.encode_list([framing.encode_text(arg) for arg in item["version_argv"]]),
            ),
            ("version_output_sha256", framing.encode_text(item["version_output_sha256"])),
            ("version_output_size", framing.encode_u64(item["version_output_size"])),
        ],
    )


def _environment_fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("environment_resolution_id", framing.encode_text(value["environment_resolution_id"])),
        ("tool_requirements_id", framing.encode_text(value["tool_requirements_id"])),
        ("target_system", framing.encode_enum(value["target_system"])),
        (
            "output_paths",
            framing.encode_set(framing.encode_text(path) for path in value["output_paths"]),
        ),
        (
            "store_objects",
            framing.encode_set(_store_semantic_frame(item) for item in value["store_objects"]),
        ),
        ("tools", framing.encode_set(_tool_semantic_frame(item) for item in value["tools"])),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(item) for item in value["non_claims"]),
        ),
    ]


def _reachable_store_paths(
    roots: list[str], store_objects: list[dict[str, Any]]
) -> set[str]:
    refs_by_path = {item["store_path"]: item["references"] for item in store_objects}
    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        path = stack.pop()
        if path in reachable:
            continue
        if path not in refs_by_path:
            raise train.TrainManifestError(
                f"environment realization capture: reachable path absent from captured closure: {path}"
            )
        reachable.add(path)
        stack.extend(refs_by_path[path])
    return reachable


def verify_realization_capture(
    *,
    resolution: Any,
    selection: Any,
    subject: Any,
    closure: Any,
    repo: Path,
    tool_requirements: Any,
    profile: Any,
    recipes: list[Any],
    capture: Any,
    capture_root: Path,
) -> dict[str, Any]:
    selection_verifier.verify_selection_git_binding(selection, subject, closure, repo)
    normalized_resolution = resolution_mod.validate_against_context(
        resolution, selection, subject, closure
    )
    normalized_requirements = requirements_mod.normalize_requirements(
        tool_requirements,
        profile=profile,
        recipes=recipes,
        require_id=True,
    )
    normalized_capture = normalize_capture(capture)
    if normalized_capture["environment_resolution_id"] != normalized_resolution["environment_resolution_id"]:
        raise train.TrainManifestError(
            "environment realization capture: resolution identity mismatch"
        )

    captured_paths = {item["store_path"] for item in normalized_capture["store_objects"]}
    missing_roots = sorted(set(normalized_resolution["output_paths"]) - captured_paths)
    if missing_roots:
        raise train.TrainManifestError(
            f"environment realization capture: selected output roots absent from captured closure: {missing_roots}"
        )
    for item in normalized_capture["store_objects"]:
        missing_refs = sorted(set(item["references"]) - captured_paths)
        if missing_refs:
            raise train.TrainManifestError(
                f"environment realization capture: {item['store_path']} references objects absent from captured closure: {missing_refs}"
            )
    reachable = _reachable_store_paths(
        normalized_resolution["output_paths"], normalized_capture["store_objects"]
    )
    if reachable != captured_paths:
        extras = sorted(captured_paths - reachable)
        raise train.TrainManifestError(
            f"environment realization capture: captured closure contains unreachable extra store objects: {extras}"
        )

    semantic_store_objects: list[dict[str, Any]] = []
    for item in normalized_capture["store_objects"]:
        nar_path = _capture_path(
            capture_root,
            item["nar_file"],
            where=f"environment realization capture nar_file for {item['store_path']}",
        )
        nar_sha256, nar_size = _sha256_file(nar_path)
        if nar_size == 0:
            raise train.TrainManifestError(
                f"environment realization capture: NAR preimage is empty for {item['store_path']}"
            )
        semantic_store_objects.append(
            {
                "store_path": item["store_path"],
                "nar_sha256": nar_sha256,
                "nar_size": nar_size,
                "references": item["references"],
            }
        )
    semantic_store_objects.sort(key=lambda item: item["store_path"])

    requirements_by_name = {
        item["tool_name"]: item for item in normalized_requirements["tools"]
    }
    capture_by_name = {item["tool_name"]: item for item in normalized_capture["tools"]}
    if set(capture_by_name) != set(requirements_by_name):
        missing = sorted(set(requirements_by_name) - set(capture_by_name))
        extra = sorted(set(capture_by_name) - set(requirements_by_name))
        raise train.TrainManifestError(
            f"environment realization capture: exact required tool set mismatch: missing={missing}, extra={extra}"
        )

    semantic_tools: list[dict[str, Any]] = []
    for tool_name in sorted(requirements_by_name):
        requirement = requirements_by_name[tool_name]
        item = capture_by_name[tool_name]
        executable_path = item["executable_path"]
        if Path(executable_path).name != requirement["executable_basename"]:
            raise train.TrainManifestError(
                f"environment realization capture: tool {tool_name} executable basename mismatch"
            )
        expected_argv = [executable_path, *requirement["version_argv_tail"]]
        if item["version_argv"] != expected_argv:
            raise train.TrainManifestError(
                f"environment realization capture: tool {tool_name} version argv does not equal frozen requirement"
            )

        owners = [path for path in captured_paths if executable_path.startswith(path + "/")]
        if len(owners) != 1:
            raise train.TrainManifestError(
                f"environment realization capture: tool {tool_name} executable must belong to exactly one captured store object"
            )

        executable_file = _capture_path(
            capture_root,
            item["executable_file"],
            where=f"environment realization capture executable_file for {tool_name}",
        )
        version_output_file = _capture_path(
            capture_root,
            item["version_output_file"],
            where=f"environment realization capture version_output_file for {tool_name}",
        )
        executable_sha256, executable_size = _sha256_file(executable_file)
        version_output_sha256, version_output_size = _sha256_file(version_output_file)
        if executable_size == 0:
            raise train.TrainManifestError(
                f"environment realization capture: tool {tool_name} executable preimage is empty"
            )
        if version_output_size == 0:
            raise train.TrainManifestError(
                f"environment realization capture: tool {tool_name} version output is empty"
            )
        semantic_tools.append(
            {
                "tool_name": tool_name,
                "executable_path": executable_path,
                "executable_sha256": executable_sha256,
                "executable_size": executable_size,
                "version_argv": expected_argv,
                "version_output_sha256": version_output_sha256,
                "version_output_size": version_output_size,
            }
        )

    witness: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "state": POSITIVE_STATE,
        "environment_resolution_id": normalized_resolution["environment_resolution_id"],
        "tool_requirements_id": normalized_requirements["tool_requirements_id"],
        "target_system": normalized_resolution["target_system"],
        "output_paths": list(normalized_resolution["output_paths"]),
        "store_objects": semantic_store_objects,
        "tools": semantic_tools,
        "capture_evidence_refs": list(normalized_capture["evidence_refs"]),
        "non_claims": list(NON_CLAIMS),
    }
    witness["qualification_environment_id"] = framing.semantic_sha256_id(
        ENV_ID_DOMAIN, _environment_fields(witness)
    )
    return witness


def load_capture(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: environment realization capture exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_capture(raw)
