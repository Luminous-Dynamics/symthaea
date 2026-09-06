#!/usr/bin/env python3
"""Validate exact source ownership for Spore extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "docs/architecture/spore-product-source-lineage-v1.json"
SOURCE_SCHEMA_PATH = ROOT / "docs/architecture/spore-product-source-lineage-v1.schema.json"
MIGRATION_PATH = ROOT / "docs/architecture/spore-migration-manifest-v1.json"
PARITY_PATH = ROOT / "docs/architecture/spore-golden-parity-v1.json"

HOST_COMMIT = "5d80360768ee329c50756e71fbce4692ac3a8e45"
HOST_TREE = "51c04910b3a97586ecf88a46699b7de22e3e1b0b"
SOURCE_COMMIT = "4fe8b1e2ca5fb60463de16c0b9ec649e1fc059a2"
SOURCE_TREE = "d53df873dd204738fe669066ca9aa22db88b22c1"

EXPECTED_ARTIFACTS = {
    "spore-boot-tools-package": (
        "nix/packages/spore-boot-tools.nix",
        "ab753efcfb377ee49d652ef9ec419a99389760e5",
        "mixed-recovery-presentation-package",
        "split-required",
        "split",
    ),
    "boot-state-cargo": (
        "crates/core/symthaea-boot-state/Cargo.toml",
        "b8c07286e8ca6618e4b71e15578270cdf9a58832",
        "recovery-package-metadata",
        "spore",
        "transform",
    ),
    "boot-state-lib": (
        "crates/core/symthaea-boot-state/src/lib.rs",
        "a4d8f43b00c506dc723aaf146fe6295b3bcaef2a",
        "recovery-state-with-presentation-coupling",
        "spore",
        "transform",
    ),
    "linux-recovery": (
        "crates/core/symthaea-boot-state/src/linux_recovery.rs",
        "693c5c0bb704119bdcd134600be83bda5ed20ffa",
        "linux-recovery-adapter",
        "spore",
        "move-or-transform",
    ),
    "recovery-planner": (
        "crates/core/symthaea-boot-state/src/recovery.rs",
        "62f9540f9ade5a42442f795bf20df868b17e9a73",
        "pure-recovery-planner",
        "spore",
        "move-or-transform",
    ),
    "recovery-executor": (
        "crates/core/symthaea-boot-state/src/recovery_executor.rs",
        "b465ea10c2066097e433302191366b8835a41663",
        "fault-injectable-recovery-executor",
        "spore",
        "move-or-transform",
    ),
    "recovery-cli": (
        "crates/core/symthaea-boot-state/src/bin/spore_recovery_linux.rs",
        "36ad29330608ddf691b6cc45f2b1ea76ffaf411e",
        "linux-recovery-cli",
        "spore",
        "transform",
    ),
    "boot-ecology-cargo": (
        "crates/core/symthaea-boot-ecology/Cargo.toml",
        "39328407fcfb8e0a10862fa3c816e156815cbe38",
        "mixed-protocol-presentation-metadata",
        "split-required",
        "split",
    ),
    "boot-ecology-lib": (
        "crates/core/symthaea-boot-ecology/src/lib.rs",
        "88b8c8ea9715d13e39fb02d23356fe94e4ebb3ff",
        "mixed-factual-protocol-and-visual-composer",
        "split-required",
        "split",
    ),
}
EXPECTED_FINDINGS = {
    "SRC-001": "architectural",
    "SRC-002": "architectural",
    "SRC-003": "authority",
    "SRC-004": "authority",
}
EXPECTED_REPAIRS = {f"REPAIR-{n:03d}" for n in range(1, 5)}


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be object")
    return data


def validate(source: dict[str, Any], migration: dict[str, Any], parity: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    constants = {
        "schema": "spore-product-source-lineage-v1",
        "destination_repository": "Luminous-Dynamics/spore",
        "destination_repository_status": "not-created",
        "qualification_transfer_policy": "never-inherit",
    }
    for key, expected in constants.items():
        if source.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")

    if migration.get("destination_repository") != source.get("destination_repository"):
        errors.append("source audit destination differs from migration contract")
    if migration.get("destination_repository_status") != "not-created":
        errors.append("parent migration destination status is no longer not-created")
    if migration.get("qualification_transfer_policy") != "never-inherit":
        errors.append("parent migration contract no longer forbids qualification inheritance")
    if parity.get("destination_repository") != source.get("destination_repository"):
        errors.append("source audit destination differs from parity contract")
    if parity.get("qualification_transfer_policy") != "never-inherit":
        errors.append("parity contract no longer forbids qualification inheritance")

    host = source.get("host_pin")
    if not isinstance(host, dict):
        errors.append("host_pin must be object")
        host = {}
    expected_host = {
        "repository": "Tristan-Stoltz-ERC/nixos-config",
        "branch": "spore/runtime-expendability-v1.3.2-proof",
        "commit": HOST_COMMIT,
        "tree": HOST_TREE,
        "flake_lock_path": "flake.lock",
        "flake_lock_blob_sha1": "2ff122e7f9cc7fde5ad6cb8fcc18453513915b67",
        "input_name": "symthaea",
        "input_flake": False,
        "input_repository": "Luminous-Dynamics/luminous-dynamics",
        "input_revision": SOURCE_COMMIT,
        "input_nar_hash": "sha256-G/SffFTFiEyLmQ4Y9O6vEhrEM0kLLi/7lwQ6b22/1JA=",
    }
    if host != expected_host:
        errors.append("host pin differs from exact reviewed flake.lock identity")

    recovery = source.get("recovery_source")
    if not isinstance(recovery, dict):
        errors.append("recovery_source must be object")
        recovery = {}
    expected_recovery = {
        "repository": "Luminous-Dynamics/luminous-dynamics",
        "branch": "spore/recovery-v0.3.4f-qualified",
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "historical_qualification": "source-lineage-only",
        "destination_qualification": "NOT_ESTABLISHED",
    }
    if recovery != expected_recovery:
        errors.append("recovery source differs from exact host-consumed qualified lineage")

    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be array")
        artifacts = []
    by_id = {
        item.get("id"): item for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(by_id) != len(artifacts):
        errors.append("artifact ids must be unique")
    if set(by_id) != set(EXPECTED_ARTIFACTS):
        errors.append(
            f"artifact set drifted: missing={sorted(set(EXPECTED_ARTIFACTS)-set(by_id))} "
            f"extra={sorted(set(by_id)-set(EXPECTED_ARTIFACTS))}"
        )
    for artifact_id, expected in EXPECTED_ARTIFACTS.items():
        item = by_id.get(artifact_id)
        if item is None:
            continue
        path, blob, role, owner, migration_kind = expected
        if item.get("path") != path:
            errors.append(f"{artifact_id}: path drifted")
        if item.get("blob_sha1") != blob:
            errors.append(f"{artifact_id}: source blob drifted")
        if item.get("role") != role:
            errors.append(f"{artifact_id}: role drifted")
        if item.get("target_owner") != owner:
            errors.append(f"{artifact_id}: target owner drifted")
        if item.get("migration_kind") != migration_kind:
            errors.append(f"{artifact_id}: migration kind drifted")
        if item.get("destination_path") is not None:
            errors.append(f"{artifact_id}: destination_path impossible before destination repo exists")

    findings = source.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be array")
        findings = []
    finding_by_id = {
        item.get("id"): item for item in findings
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(finding_by_id) != set(EXPECTED_FINDINGS):
        errors.append("finding set drifted")
    for finding_id, severity in EXPECTED_FINDINGS.items():
        item = finding_by_id.get(finding_id)
        if item is None:
            continue
        if item.get("severity") != severity:
            errors.append(f"{finding_id}: severity drifted")
        if item.get("status") != "OPEN":
            errors.append(f"{finding_id}: cannot close finding without a new source-audit version")
        if not isinstance(item.get("statement"), str) or len(item["statement"].strip()) < 20:
            errors.append(f"{finding_id}: statement is missing")
    src003 = finding_by_id.get("SRC-003", {})
    if src003.get("tracking_issue") != "Luminous-Dynamics/luminous-dynamics#51":
        errors.append("SRC-003 must remain bound to source authority issue #51")
    src004 = finding_by_id.get("SRC-004", {})
    if src004.get("tracking_issue") != "Luminous-Dynamics/luminous-dynamics#56":
        errors.append("SRC-004 must remain bound to presentation-veto authority issue #56")
    src004_statement = src004.get("statement", "")
    if not isinstance(src004_statement, str) or not all(
        phrase in src004_statement
        for phrase in (
            "AlreadyKnownGood",
            "exact current boot",
            "last_boot_blessed",
        )
    ):
        errors.append("SRC-004 must retain the healthy AlreadyKnownGood per-boot blessing defect")

    repairs = source.get("required_pre_extraction_repairs")
    if not isinstance(repairs, list):
        errors.append("required_pre_extraction_repairs must be array")
        repairs = []
    repair_by_id = {
        item.get("id"): item for item in repairs
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(repair_by_id) != EXPECTED_REPAIRS:
        errors.append("required repair set drifted")
    for repair_id in EXPECTED_REPAIRS:
        item = repair_by_id.get(repair_id)
        if item is None:
            continue
        if item.get("status") != "REQUIRED":
            errors.append(f"{repair_id}: cannot advance before a versioned repaired-source lineage exists")
        if not isinstance(item.get("statement"), str) or len(item["statement"].strip()) < 20:
            errors.append(f"{repair_id}: statement is missing")

    repair002 = repair_by_id.get("REPAIR-002", {})
    repair002_statement = repair002.get("statement", "")
    if not isinstance(repair002_statement, str) or not all(
        phrase in repair002_statement
        for phrase in (
            "factual recovery preparation, qualification and LKG-commit authority",
            "recovery-native prepared-boot",
            "morphology history",
            "every healthy prepared boot",
            "AlreadyKnownGood",
        )
    ):
        errors.append(
            "REPAIR-002 must remove presentation authority and preserve exact per-boot qualification truth"
        )

    # Explicitly prohibit the easiest ownership and authority mistakes.
    if host.get("input_repository") == "Luminous-Dynamics/symthaea":
        errors.append("host recovery pin must not be rewritten to the current Symthaea repository")
    if by_id.get("spore-boot-tools-package", {}).get("target_owner") == "spore":
        errors.append("mixed quicken/recovery package cannot be assigned wholesale to Spore")
    if by_id.get("boot-ecology-lib", {}).get("target_owner") == "spore":
        errors.append("mixed Boot Ecology crate cannot be assigned wholesale to Spore")

    return errors


def validate_files() -> list[str]:
    load(SOURCE_SCHEMA_PATH)  # parse/ship strict structural schema
    return validate(load(SOURCE_PATH), load(MIGRATION_PATH), load(PARITY_PATH))


def main() -> int:
    try:
        errors = validate_files()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: Spore product source lineage and ownership boundary are exact")
    print(f"host_commit={HOST_COMMIT}")
    print(f"recovery_source_commit={SOURCE_COMMIT}")
    print(f"recovery_source_tree={SOURCE_TREE}")
    print("pre_extraction_repairs=4")
    print("destination_qualification=NOT_ESTABLISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
