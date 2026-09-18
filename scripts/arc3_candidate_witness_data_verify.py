#!/usr/bin/env python3
"""Compose ARC3 candidate run metadata, artifact, and receipt verification.

This is the last data-verification layer before external recipe admission. It
performs no network access, executes no candidate code, and cannot create a
trusted-recipe decision. Inputs are saved GitHub REST JSON plus the downloaded
artifact ZIP and independently supplied expected identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import arc3_candidate_artifact_verify as artifact_verify
import arc3_candidate_run_metadata_verify as metadata_verify


class WitnessDataError(ValueError):
    pass


def fail(message: str) -> None:
    raise WitnessDataError(message)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        run_json=args.run_json,
        jobs_json=args.jobs_json,
        artifacts_json=args.artifacts_json,
        expected_repository=args.expected_repository,
        expected_run_id=args.expected_run_id,
        expected_run_attempt=args.expected_run_attempt,
        expected_workflow_id=args.expected_workflow_id,
        expected_event=args.expected_event,
        expected_head_branch=args.expected_head_branch,
        expected_helper_sha=args.expected_helper_sha,
        expected_pr_number=args.expected_pr_number,
        expected_subject_sha=args.expected_subject_sha,
        expected_runner_label=args.expected_runner_class,
    )


def build_artifact_args(
    args: argparse.Namespace, artifact_digest: str
) -> argparse.Namespace:
    return argparse.Namespace(
        artifact=args.artifact,
        expected_artifact_sha256=artifact_digest,
        expected_repository=args.expected_repository,
        expected_run_id=args.expected_run_id,
        expected_run_attempt=args.expected_run_attempt,
        expected_helper_sha=args.expected_helper_sha,
        expected_helper_tree=args.expected_helper_tree,
        expected_workflow_blob=args.expected_workflow_blob,
        expected_subject_sha=args.expected_subject_sha,
        expected_subject_tree=args.expected_subject_tree,
        expected_subject_binding_sha256=args.expected_subject_binding_sha256,
        expected_cargo_lock_sha256=args.expected_cargo_lock_sha256,
        expected_oracle_sha256=args.expected_oracle_sha256,
        expected_fixture_sha256=args.expected_fixture_sha256,
        expected_vector_sha256=args.expected_vector_sha256,
        expected_runner_class=args.expected_runner_class,
    )


def verify(args: argparse.Namespace) -> dict[str, object]:
    job_id, artifact_id, artifact_digest, artifact_size = metadata_verify.verify(
        build_metadata_args(args)
    )

    observed_artifact_sha256 = file_sha256(args.artifact)
    metadata_artifact_sha256 = artifact_verify.normalize_sha256(
        artifact_digest, "metadata artifact digest"
    )
    if observed_artifact_sha256 != metadata_artifact_sha256:
        fail(
            "downloaded artifact does not match GitHub metadata digest: "
            f"metadata={metadata_artifact_sha256} downloaded={observed_artifact_sha256}"
        )

    artifact_sha256, receipt_sha256, receipt_bytes = artifact_verify.verify(
        build_artifact_args(args, artifact_digest)
    )

    result = {
        "schema": "symthaea.arc3.protocol.bootstrap-witness-data.v1",
        "disposition": "WITNESS_DATA_PASS",
        "repository": args.expected_repository,
        "run_id": int(args.expected_run_id),
        "run_attempt": int(args.expected_run_attempt),
        "workflow_id": int(args.expected_workflow_id),
        "pr_number": int(args.expected_pr_number),
        "job_id": job_id,
        "helper_sha": args.expected_helper_sha,
        "helper_tree": args.expected_helper_tree,
        "workflow_blob": args.expected_workflow_blob,
        "subject_sha": args.expected_subject_sha,
        "subject_tree": args.expected_subject_tree,
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha256,
        "artifact_size": artifact_size,
        "receipt_sha256": receipt_sha256,
        "receipt_bytes": receipt_bytes,
    }
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-json", required=True)
    p.add_argument("--jobs-json", required=True)
    p.add_argument("--artifacts-json", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--expected-repository", required=True)
    p.add_argument("--expected-run-id", required=True)
    p.add_argument("--expected-run-attempt", required=True)
    p.add_argument("--expected-workflow-id", required=True)
    p.add_argument("--expected-event", default="pull_request")
    p.add_argument("--expected-head-branch", required=True)
    p.add_argument("--expected-helper-sha", required=True)
    p.add_argument("--expected-helper-tree", required=True)
    p.add_argument("--expected-workflow-blob", required=True)
    p.add_argument("--expected-pr-number", required=True)
    p.add_argument("--expected-subject-sha", required=True)
    p.add_argument("--expected-subject-tree", required=True)
    p.add_argument("--expected-subject-binding-sha256", required=True)
    p.add_argument("--expected-cargo-lock-sha256", required=True)
    p.add_argument("--expected-oracle-sha256", required=True)
    p.add_argument("--expected-fixture-sha256", required=True)
    p.add_argument("--expected-vector-sha256", required=True)
    p.add_argument("--expected-runner-class", default="ubuntu-slim")
    return p


def main() -> int:
    try:
        result = verify(parser().parse_args())
    except (
        OSError,
        ValueError,
        metadata_verify.MetadataError,
        artifact_verify.ArtifactError,
        artifact_verify.receipt_verify.ReceiptError,
    ) as exc:
        print(f"ARC3 candidate witness-data verification FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
