#!/usr/bin/env python3
"""Canonical content-ID descriptors for qualification evidence.

These identifiers answer only "which exact content is being named?". This parser validates the
identifier format; it does NOT establish that referenced bytes exist, that a digest was recomputed
over those bytes, who produced them, whether the producer is trusted, whether the content is
correct, or whether the evidence is sufficient for a qualification theorem.
"""

from __future__ import annotations

import re
from typing import Any

import integration_train_manifest as train

_EVIDENCE_CONTENT_ID_RE = re.compile(
    r"^(?:sha256:[0-9a-f]{64}|git-blob-sha1:[0-9a-f]{40})$"
)


def require_evidence_content_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _EVIDENCE_CONTENT_ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(
            f"{where}: expected sha256:<64 lowercase hex> or "
            "git-blob-sha1:<40 lowercase hex>"
        )
    return value


def require_sorted_unique_evidence_content_ids(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list):
        raise train.TrainManifestError(f"{where}: expected array")
    normalized = [
        require_evidence_content_id(item, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    ]
    if normalized != sorted(normalized):
        raise train.TrainManifestError(f"{where}: content ids must be lexicographically sorted")
    if len(normalized) != len(set(normalized)):
        raise train.TrainManifestError(f"{where}: content ids must be unique")
    return normalized
