#!/usr/bin/env python3
"""Trust-anchored Lineage-B v2 candidate run capture.

This is the qualified operator-facing wrapper. The reusable capture library validates
closed-world structure; this wrapper additionally fixes the exact hosted-qualified
#976 profile and retained verification roots before publication.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import capture_hcpmmp1_neuromaps_lineage_b_verified_run as capture

QUALIFIED_CAPTURE_PROFILE = "symthaea-hcpmmp1-lineage-b-qualified-run-capture-v2"
QUALIFIED_PROFILE_FILE_SHA256 = "sha256:975c7727ecb9faf7e3d46a039aa4e6601b7e2498b36443bf9012c51b4f2e9499"
QUALIFIED_VERIFICATION_FILE_SHA256 = "sha256:bd79a0280f0a9e5ad2308d6824ccda2f28d0b5cffa3bb4f89b3afa7089afe4d9"
QUALIFIED_WORKBENCH_HEAD = "5da8729fbe2d6017739c859c486a58f4d457c852"
QUALIFIED_CLOSURE_DIGEST = "sha256:4b9820c088e3ab1481833c1659c449b0acc4b168f172d8f43abf383fae6b8a6a"

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data/neuroscience/hcpmmp1_lineage_b_verified_run_capture_profile_v2.json"
VERIFICATION_PATH = ROOT / "data/neuroscience/evidence/workbench_isolated_version_verification_5da8729f.json"


def _require_trust_anchors(metadata: dict[str, Any]) -> None:
    expected = {
        "verified_run_profile_sha256": QUALIFIED_PROFILE_FILE_SHA256,
        "workbench_verification_file_sha256": QUALIFIED_VERIFICATION_FILE_SHA256,
        "qualified_workbench_head": QUALIFIED_WORKBENCH_HEAD,
        "closure_digest": QUALIFIED_CLOSURE_DIGEST,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise capture.CaptureError(f"qualified run capture: trust-anchor mismatch: {key}")

    authority = metadata.get("authority")
    if authority != capture.CAPTURE_AUTHORITY:
        raise capture.CaptureError("qualified run capture: capture authority state mismatch")


def capture_qualified_manifest(
    method_manifest: Path,
    input_items: Iterable[str],
    execution_id: str,
    authorization_reference: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    doc, metadata = capture.capture_manifest(
        method_manifest,
        PROFILE_PATH,
        VERIFICATION_PATH,
        input_items,
        execution_id,
        authorization_reference,
    )
    _require_trust_anchors(metadata)
    return doc, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a Lineage-B candidate run using the exact retained #976 "
            "Workbench verification roots; never execute Workbench."
        )
    )
    parser.add_argument("--method-manifest", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--input", action="append", required=True, metavar="ROLE=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        doc, metadata = capture_qualified_manifest(
            args.method_manifest,
            args.input,
            args.execution_id,
            args.authorization_reference,
        )
        target = capture.write_new(args.output, doc)
        receipt = {
            "profile": QUALIFIED_CAPTURE_PROFILE,
            "run_manifest_file_sha256": capture.digest_file(target),
            **metadata,
        }
    except (capture.CaptureError, capture.ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
