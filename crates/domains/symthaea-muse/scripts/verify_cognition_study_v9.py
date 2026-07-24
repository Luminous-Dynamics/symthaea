#!/usr/bin/env python3
"""Independent V9 artifact/session commitment verifier.

Uses only the Python standard library. It intentionally does not know policy
arms and cannot reveal the private codebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def safe_join(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    return root / path


def artifact_commitment(bundle: dict[str, Any]) -> str:
    committed = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    return digest(committed)


def verify_artifacts(root: Path, bundle_path: Path) -> list[str]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if artifact_commitment(bundle) != bundle.get("bundle_sha256"):
        issues.append("bundle_sha256 mismatch")
    seen: set[str] = set()
    for record in bundle.get("records", []):
        presentation_id = record.get("presentation_id", "")
        if presentation_id in seen:
            issues.append(f"duplicate presentation: {presentation_id}")
        seen.add(presentation_id)
        for field in ("audio", "recipe", "score", "validation_report", "renderer_log"):
            evidence = record.get(field)
            if not isinstance(evidence, dict):
                issues.append(f"{presentation_id}: missing {field}")
                continue
            try:
                path = safe_join(root, evidence["relative_path"])
                size, sha = file_digest(path)
            except (OSError, KeyError, ValueError) as error:
                issues.append(f"{presentation_id}: {field}: {error}")
                continue
            if size != evidence.get("byte_count"):
                issues.append(f"{presentation_id}: {field}: byte_count mismatch")
            if sha != evidence.get("sha256"):
                issues.append(f"{presentation_id}: {field}: sha256 mismatch")
        midi = record.get("midi")
        if midi is not None:
            try:
                path = safe_join(root, midi["relative_path"])
                size, sha = file_digest(path)
            except (OSError, KeyError, ValueError) as error:
                issues.append(f"{presentation_id}: midi: {error}")
            else:
                if size != midi.get("byte_count") or sha != midi.get("sha256"):
                    issues.append(f"{presentation_id}: midi evidence mismatch")
    return issues


def package_commitment(package: dict[str, Any]) -> str:
    committed = {key: value for key, value in package.items() if key != "package_sha256"}
    return digest(committed)


def event_commitment(package_sha256: str, envelope: dict[str, Any]) -> str:
    return digest(
        {
            "package_sha256": package_sha256,
            "sequence": envelope["sequence"],
            "previous_event_sha256": envelope["previous_event_sha256"],
            "server_received_unix_ms": envelope["server_received_unix_ms"],
            "client_elapsed_ms": envelope["client_elapsed_ms"],
            "event": envelope["event"],
        }
    )


def log_commitment(log: dict[str, Any]) -> str:
    committed = {key: value for key, value in log.items() if key != "log_sha256"}
    return digest(committed)


def verify_session(package_path: Path, log_path: Path) -> list[str]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    log = json.loads(log_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if package_commitment(package) != package.get("package_sha256"):
        issues.append("package_sha256 mismatch")
    if log.get("package_sha256") != package.get("package_sha256"):
        issues.append("log/package mismatch")
    previous = GENESIS
    server_time = 0
    client_time = 0
    for index, envelope in enumerate(log.get("events", [])):
        if envelope.get("sequence") != index:
            issues.append(f"event {index}: sequence mismatch")
        if envelope.get("previous_event_sha256") != previous:
            issues.append(f"event {index}: previous digest mismatch")
        expected = event_commitment(package["package_sha256"], envelope)
        if envelope.get("event_sha256") != expected:
            issues.append(f"event {index}: event digest mismatch")
        if index and envelope.get("server_received_unix_ms", 0) < server_time:
            issues.append(f"event {index}: server time regressed")
        if index and envelope.get("client_elapsed_ms", 0) < client_time:
            issues.append(f"event {index}: client time regressed")
        previous = envelope.get("event_sha256", "")
        server_time = envelope.get("server_received_unix_ms", 0)
        client_time = envelope.get("client_elapsed_ms", 0)
    if log.get("events") and log_commitment(log) != log.get("log_sha256"):
        issues.append("log_sha256 mismatch")
    return issues



def release_commitment(bundle: dict[str, Any]) -> str:
    committed = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    return digest(committed)


def verify_release(root: Path, plan_path: Path, bundle_path: Path) -> list[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if digest(plan) != bundle.get("release_plan_sha256"):
        issues.append("release_plan_sha256 mismatch")
    if release_commitment(bundle) != bundle.get("bundle_sha256"):
        issues.append("release bundle_sha256 mismatch")
    planned = {
        (entry.get("role"), entry.get("relative_path")): entry
        for entry in plan.get("entries", [])
    }
    observed: set[tuple[str | None, str | None]] = set()
    for evidence in bundle.get("files", []):
        identity = (evidence.get("role"), evidence.get("relative_path"))
        observed.add(identity)
        if identity not in planned:
            issues.append(f"unexpected release file: {identity}")
            continue
        try:
            path = safe_join(root, evidence["relative_path"])
            size, sha = file_digest(path)
        except (OSError, KeyError, ValueError) as error:
            issues.append(f"release file {identity}: {error}")
            continue
        if size != evidence.get("byte_count"):
            issues.append(f"release file {identity}: byte_count mismatch")
        if sha != evidence.get("sha256"):
            issues.append(f"release file {identity}: sha256 mismatch")
        if evidence.get("visibility") != planned[identity].get("visibility"):
            issues.append(f"release file {identity}: visibility mismatch")
    for identity in sorted(set(planned) - observed):
        issues.append(f"missing release file: {identity}")
    return issues

def self_test() -> list[str]:
    issues: list[str] = []
    known = digest({"b": 2, "a": 1})
    expected = "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    if known != expected:
        issues.append(f"canonical digest mismatch: {known}")
    package = {
        "package_version": "test",
        "participant_schedule_sha256": "a" * 64,
        "artifact_bundle_sha256": "b" * 64,
        "block_id": "block",
        "participant_token": "participant",
        "key": {"fixture_id": "fixture", "seed": 1},
        "protocol": {},
        "presentations": [],
        "package_sha256": "",
    }
    package["package_sha256"] = package_commitment(package)
    if package_commitment(package) != package["package_sha256"]:
        issues.append("package self-test failed")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    artifacts = sub.add_parser("artifacts")
    artifacts.add_argument("root", type=Path)
    artifacts.add_argument("bundle", type=Path)
    session = sub.add_parser("session")
    session.add_argument("package", type=Path)
    session.add_argument("log", type=Path)
    release = sub.add_parser("release")
    release.add_argument("root", type=Path)
    release.add_argument("plan", type=Path)
    release.add_argument("bundle", type=Path)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "artifacts":
        issues = verify_artifacts(args.root, args.bundle)
    elif args.command == "session":
        issues = verify_session(args.package, args.log)
    elif args.command == "release":
        issues = verify_release(args.root, args.plan, args.bundle)
    else:
        issues = self_test()
    json.dump(issues, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
