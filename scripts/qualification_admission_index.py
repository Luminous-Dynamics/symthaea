#!/usr/bin/env python3
"""Coalesce immutable qualification-admission requests by exact work subject."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_admission as admission

SCHEMA = "symthaea.qualification-admission-index.v1"
DOMAIN = b"symthaea.qualification-admission-index.v1\0"
SUBJECT_DOMAIN = b"symthaea.qualification-admission-subject.v1\0"

NON_CLAIMS = [
    "does not establish hosted qualification",
    "does not merge or erase request provenance",
    "does not schedule qualification",
]


def _subject_payload(normalized_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "program_id": normalized_request["program_id"],
        "catalog_id": normalized_request["catalog_id"],
        "target_train": normalized_request["target_train"],
        "qualification_profile": normalized_request["qualification_profile"],
    }


def compute_admission_subject_id(request: Any) -> str:
    normalized = admission.normalize_request(request)
    payload = json.dumps(
        _subject_payload(normalized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(SUBJECT_DOMAIN + payload).hexdigest()


def _canonical_payload(index: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in index.items() if key != "index_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_index_id(index: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical_payload(index)).hexdigest()


def build_index(requests: list[Any]) -> dict[str, Any]:
    if not requests:
        raise train.TrainManifestError("admission index: expected at least one request")

    groups: dict[str, dict[str, Any]] = {}
    seen_request_ids: set[str] = set()
    for position, raw in enumerate(requests):
        normalized = admission.normalize_request(raw, require_id=True)
        request_id = normalized["admission_id"]
        if request_id in seen_request_ids:
            raise train.TrainManifestError(
                f"admission index request[{position}]: duplicate admission_id {request_id}"
            )
        seen_request_ids.add(request_id)

        subject_id = compute_admission_subject_id(normalized)
        payload = _subject_payload(normalized)
        group = groups.get(subject_id)
        if group is None:
            group = {"subject_id": subject_id, **payload, "request_ids": []}
            groups[subject_id] = group
        else:
            expected = {
                "program_id": group["program_id"],
                "catalog_id": group["catalog_id"],
                "target_train": group["target_train"],
                "qualification_profile": group["qualification_profile"],
            }
            if payload != expected:
                raise train.TrainManifestError(
                    f"admission index: subject-id collision for {subject_id}"
                )
        group["request_ids"].append(request_id)

    normalized_groups = []
    for subject_id in sorted(groups):
        group = groups[subject_id]
        group["request_ids"] = sorted(group["request_ids"])
        normalized_groups.append(group)

    index: dict[str, Any] = {
        "schema": SCHEMA,
        "subjects": normalized_groups,
        "non_claims": NON_CLAIMS,
    }
    index["index_id"] = _compute_index_id(index)
    return index


def load_requests(paths: list[Path]) -> list[dict[str, Any]]:
    return [admission.load_request(path, require_id=True) for path in paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="+", type=Path, help="immutable qualification-admission request JSON files")
    parser.add_argument("--print-normalized", action="store_true", help="print the canonical coalesced index")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        index = build_index(load_requests(args.requests))
    except train.TrainManifestError as error:
        print(f"qualification admission index invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(index, indent=2, ensure_ascii=False))
    else:
        print(index["index_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
