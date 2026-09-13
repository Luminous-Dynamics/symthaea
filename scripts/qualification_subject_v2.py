#!/usr/bin/env python3
"""Framed V2 generic qualification subject identity.

V1 subject records remain historical prototype evidence. V2 intentionally uses a distinct schema
and typed ID prefix so a V1 JSON-hashed identifier cannot be substituted for a V2 framed one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing as framing
import qualification_subject as v1

SCHEMA = "symthaea.qualification-subject.v2"
DOMAIN = "qualification-subject-v2"
ID_PREFIX = "qsubject-v2-sha256"
_ID_RE = re.compile(r"^qsubject-v2-sha256:[0-9a-f]{64}$")


def _preimage(normalized: dict[str, Any]) -> bytes:
    return framing.record(
        DOMAIN,
        [
            ("schema", framing.text(normalized["schema"])),
            ("kind", framing.text(normalized["kind"])),
            ("repository", framing.text(normalized["repository"])),
            ("object_format", framing.text(normalized["object_format"])),
            ("source_commit", framing.text(normalized["source_commit"])),
            ("source_tree", framing.text(normalized["source_tree"])),
        ],
    )


def normalize_subject(value: Any, *, verify_declared_id: bool = True, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification subject v2: expected object")
    v1._require_exact_keys(
        value,
        {"schema", "kind", "repository", "object_format", "source_commit", "source_tree"},
        {"subject_id"},
        where="qualification subject v2",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(f"qualification subject v2.schema: expected {SCHEMA!r}")
    if require_id and "subject_id" not in value:
        raise train.TrainManifestError("qualification subject v2.subject_id: required but absent")

    # Reuse V1's mature semantic/canonical validators without reusing its identity bytes.
    surrogate = dict(value)
    surrogate["schema"] = v1.SCHEMA
    surrogate.pop("subject_id", None)
    checked = v1.normalize_subject(surrogate, verify_declared_id=False)
    checked.pop("subject_id", None)
    checked["schema"] = SCHEMA

    subject_id = framing.sha256_hex_id(ID_PREFIX, _preimage(checked))
    checked["subject_id"] = subject_id
    if verify_declared_id and "subject_id" in value:
        declared = value["subject_id"]
        if not isinstance(declared, str) or _ID_RE.fullmatch(declared) is None:
            raise train.TrainManifestError(
                "qualification subject v2.subject_id: expected qsubject-v2-sha256:<64 lowercase hex>"
            )
        if declared != subject_id:
            raise train.TrainManifestError(
                f"qualification subject v2.subject_id: expected {subject_id}, got {declared}"
            )
    return checked


def compute_subject_id(value: Any) -> str:
    return normalize_subject(value, verify_declared_id=False)["subject_id"]


def framed_preimage(value: Any) -> bytes:
    normalized = normalize_subject(value, verify_declared_id=False)
    normalized.pop("subject_id", None)
    return _preimage(normalized)


def validate_git_binding(value: Any, repo: Path) -> None:
    normalized = normalize_subject(value, require_id=True)
    repo = repo.resolve()
    if v1._run_git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise train.TrainManifestError(f"repository is not a Git work tree: {repo}")
    actual_format = v1._run_git(repo, "rev-parse", "--show-object-format")
    if actual_format != normalized["object_format"]:
        raise train.TrainManifestError(
            f"qualification subject v2.object_format: repository uses {actual_format}, declared {normalized['object_format']}"
        )
    v1._run_git(repo, "cat-file", "-e", f"{normalized['source_commit']}^{{commit}}")
    actual_tree = v1._run_git(repo, "rev-parse", f"{normalized['source_commit']}^{{tree}}")
    if actual_tree != normalized["source_tree"]:
        raise train.TrainManifestError(
            f"qualification subject v2.source_tree: expected {actual_tree}, got {normalized['source_tree']}"
        )

    # Repository name is a semantic namespace label in V2; this local Git binding proves commit/tree
    # membership in the supplied object database, not authenticated remote ownership/provenance.
