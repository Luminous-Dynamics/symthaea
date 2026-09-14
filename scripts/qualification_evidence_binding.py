#!/usr/bin/env python3
"""Verify exact evidence bytes against a qualification evidence-content descriptor.

This is a pre-receipt byte-binding theorem. It establishes only that the supplied bytes recompute
to the supplied content identifier under the named algorithm. It does not establish trustworthy
acquisition, provider/run provenance, producer authenticity, evidence correctness/sufficiency,
chronology, or qualification authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_evidence_id as evidence_id_mod

SCHEMA = "symthaea.qualification-evidence-byte-binding.v1"
DOMAIN = b"symthaea.qualification-evidence-byte-binding.v1\0"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _binding_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "binding_id"}
    return "sha256:" + hashlib.sha256(DOMAIN + _canon(payload)).hexdigest()


def _recompute(content_id: str, data: bytes) -> tuple[str, str]:
    descriptor = evidence_id_mod.require_evidence_content_id(
        content_id, where="evidence byte binding.content_id"
    )
    algorithm, expected_hex = descriptor.split(":", 1)
    if algorithm == "sha256":
        actual_hex = hashlib.sha256(data).hexdigest()
        method = "RawSha256V1"
    elif algorithm == "git-blob-sha1":
        framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        # SHA-1 is used here only to reproduce Git's existing blob object identity,
        # not as a new security-strength claim.
        actual_hex = hashlib.sha1(framed, usedforsecurity=False).hexdigest()
        method = "GitBlobSha1V1"
    else:  # guarded by descriptor validator
        raise train.TrainManifestError("evidence byte binding: unsupported content-id algorithm")
    if actual_hex != expected_hex:
        raise train.TrainManifestError(
            f"evidence byte binding: digest mismatch for {algorithm}: expected {expected_hex}, got {actual_hex}"
        )
    return descriptor, method


def verify_bytes(content_id: str, data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise train.TrainManifestError("evidence byte binding.data: expected bytes")
    if len(data) > MAX_EVIDENCE_BYTES:
        raise train.TrainManifestError(
            f"evidence byte binding.data: exceeds {MAX_EVIDENCE_BYTES} bytes"
        )
    descriptor, method = _recompute(content_id, data)
    normalized = {
        "schema": SCHEMA,
        "content_id": descriptor,
        "byte_length": len(data),
        "verification_method": method,
        "non_claims": [
            "byte equality under the declared digest does not establish evidence correctness or sufficiency",
            "does not establish chronology, qualification PASS, current admission, or merge authority",
            "does not establish provider/run provenance or producer authenticity",
            "does not establish trustworthy acquisition of the verified bytes",
            "git-blob-sha1 reproduces Git object identity and is not a new cryptographic-strength claim",
        ],
    }
    normalized["binding_id"] = _binding_id(normalized)
    return normalized


def verify_file(content_id: str, path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    if size > MAX_EVIDENCE_BYTES:
        raise train.TrainManifestError(
            f"{path}: evidence object exceeds {MAX_EVIDENCE_BYTES} bytes"
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    if len(data) != size:
        raise train.TrainManifestError(
            f"{path}: size changed while reading; refuse non-atomic evidence binding"
        )
    return verify_bytes(content_id, data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_id", help="sha256:... or git-blob-sha1:... content descriptor")
    parser.add_argument("file", type=Path, help="exact local evidence bytes to verify")
    parser.add_argument("--json", action="store_true", help="print the complete binding object")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        binding = verify_file(args.content_id, args.file)
    except train.TrainManifestError as error:
        print(f"qualification evidence byte binding failed: {error}")
        return 2
    if args.json:
        print(json.dumps(binding, indent=2, ensure_ascii=False))
    else:
        print(binding["binding_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
