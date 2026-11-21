#!/usr/bin/env python3
"""Independent split-manifest validator for SEMI-BENCH-001.

Validates sample identity, split assignment, group-disjoint or forward-time
claims, and deterministic manifest identity. It does not download, inspect,
transform, train on, or redistribute source datasets.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

INPUT_SCHEMA = "luminous.semi-bench.split-manifest.v1"
OUTPUT_SCHEMA = "luminous.semi-bench.split-validation-report.v1"
SPLITS = ("train", "validation", "test")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SplitContractError(ValueError):
    """A split manifest violates the versioned admission contract."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitContractError(f"{field}: expected non-empty string")
    return value


def _parse_timestamp(value: Any, field: str, time_basis: str) -> datetime:
    raw = _text(value, field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SplitContractError(f"{field}: expected ISO-8601 timestamp") from exc
    has_offset = parsed.tzinfo is not None and parsed.utcoffset() is not None
    if time_basis == "offset_aware" and not has_offset:
        raise SplitContractError(f"{field}: timezone offset is required for offset_aware")
    if time_basis == "source_local_naive" and has_offset:
        raise SplitContractError(f"{field}: must be timezone-free for source_local_naive")
    return parsed


def validate_split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SplitContractError("manifest: expected object")
    if manifest.get("schema") != INPUT_SCHEMA:
        raise SplitContractError("schema: unsupported or missing split-manifest schema")

    dataset_id = _text(manifest.get("dataset_id"), "dataset_id")
    snapshot_digest = _text(manifest.get("dataset_snapshot_sha256"), "dataset_snapshot_sha256")
    if not SHA256_RE.fullmatch(snapshot_digest):
        raise SplitContractError("dataset_snapshot_sha256: expected lowercase SHA-256 hex")

    scheme = manifest.get("split_scheme")
    if scheme not in ("group_disjoint", "forward_time", "stratified_sample"):
        raise SplitContractError("split_scheme: unsupported scheme")

    time_basis = None
    clock_domain_id = None
    if scheme == "forward_time":
        time_basis = manifest.get("time_basis")
        if time_basis not in ("offset_aware", "source_local_naive"):
            raise SplitContractError("time_basis: expected offset_aware or source_local_naive")
        if time_basis == "source_local_naive":
            clock_domain_id = _text(manifest.get("clock_domain_id"), "clock_domain_id")

    rows = manifest.get("assignments")
    if not isinstance(rows, list) or not rows:
        raise SplitContractError("assignments: expected non-empty array")

    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    split_timestamps: dict[str, list[datetime]] = {split: [] for split in SPLITS}
    counts: dict[str, dict[str, int]] = {
        split: {"samples": 0, "positive": 0, "negative": 0} for split in SPLITS
    }
    canonical_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SplitContractError(f"assignments[{index}]: expected object")
        sample_id = _text(row.get("sample_id"), f"assignments[{index}].sample_id")
        if sample_id in seen_ids:
            raise SplitContractError(f"assignments[{index}].sample_id: duplicate {sample_id!r}")
        seen_ids.add(sample_id)

        split = row.get("split")
        if split not in SPLITS:
            raise SplitContractError(f"assignments[{index}].split: expected train, validation or test")
        positive = row.get("positive")
        if type(positive) is not bool:
            raise SplitContractError(f"assignments[{index}].positive: expected boolean")

        canonical = {"sample_id": sample_id, "split": split, "positive": positive}
        counts[split]["samples"] += 1
        counts[split]["positive" if positive else "negative"] += 1

        if scheme == "group_disjoint":
            group_id = _text(row.get("group_id"), f"assignments[{index}].group_id")
            canonical["group_id"] = group_id
            group_splits.setdefault(group_id, set()).add(split)
        elif scheme == "forward_time":
            timestamp = _parse_timestamp(
                row.get("observed_at"),
                f"assignments[{index}].observed_at",
                time_basis,
            )
            canonical["observed_at"] = timestamp.isoformat()
            split_timestamps[split].append(timestamp)
        canonical_rows.append(canonical)

    empty = [split for split in SPLITS if counts[split]["samples"] == 0]
    if empty:
        raise SplitContractError(f"assignments: empty required split(s): {', '.join(empty)}")

    group_leakage = sorted(group for group, assigned in group_splits.items() if len(assigned) > 1)
    if scheme == "group_disjoint" and group_leakage:
        raise SplitContractError(f"group leakage across splits: {group_leakage[0]!r}")

    temporal_bounds: dict[str, dict[str, str] | None] = {}
    if scheme == "forward_time":
        for split in SPLITS:
            if not split_timestamps[split]:
                raise SplitContractError(f"forward_time: missing timestamps in {split}")
            ordered = sorted(split_timestamps[split])
            temporal_bounds[split] = {
                "minimum": ordered[0].isoformat(),
                "maximum": ordered[-1].isoformat(),
            }
        if not (
            max(split_timestamps["train"]) < min(split_timestamps["validation"])
            and max(split_timestamps["validation"]) < min(split_timestamps["test"])
        ):
            raise SplitContractError("forward_time: train, validation and test windows must be strictly chronological and non-overlapping")

    canonical_rows.sort(key=lambda row: row["sample_id"])
    canonical_payload = {
        "schema": INPUT_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_snapshot_sha256": snapshot_digest,
        "split_scheme": scheme,
        "assignments": canonical_rows,
    }
    # Preserve the v1 digest/report for group-disjoint and stratified manifests.
    # Time-basis metadata is added only when it is relevant to forward-time
    # validation, so older known-answer fixtures remain stable.
    if scheme == "forward_time":
        canonical_payload["time_basis"] = time_basis
        if time_basis == "source_local_naive":
            canonical_payload["clock_domain_id"] = clock_domain_id

    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    split_digest = hashlib.sha256(serialized).hexdigest()

    report = {
        "schema": OUTPUT_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_snapshot_sha256": snapshot_digest,
        "split_scheme": scheme,
        "split_manifest_sha256": split_digest,
        "split_counts": counts,
        "guarantees": {
            "unique_sample_identity": True,
            "all_splits_non_empty": True,
            "group_disjoint": scheme == "group_disjoint",
            "strict_forward_time": scheme == "forward_time",
        },
        "temporal_bounds": temporal_bounds if scheme == "forward_time" else None,
        "claim_limits": {
            "group_independence": "established-for-this-manifest" if scheme == "group_disjoint" else "not-established",
            "temporal_independence": (
                "established-for-this-manifest"
                if scheme == "forward_time" and time_basis == "offset_aware"
                else "established-within-declared-source-clock-domain-only"
                if scheme == "forward_time" and time_basis == "source_local_naive"
                else "not-established"
            ),
            "stratified_sample_split": scheme == "stratified_sample",
        },
    }
    if scheme == "forward_time":
        report["time_basis"] = time_basis
        if time_basis == "source_local_naive":
            report["clock_domain_id"] = clock_domain_id
    return report
