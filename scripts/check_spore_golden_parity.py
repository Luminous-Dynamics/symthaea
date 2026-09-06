#!/usr/bin/env python3
"""Validate the Spore golden parity corpus against its parent migration manifest."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARITY_PATH = ROOT / "docs/architecture/spore-golden-parity-v1.json"
PARITY_SCHEMA_PATH = ROOT / "docs/architecture/spore-golden-parity-v1.schema.json"
RECEIPT_SCHEMA_PATH = ROOT / "docs/architecture/spore-parity-receipt-v1.schema.json"
MIGRATION_PATH = ROOT / "docs/architecture/spore-migration-manifest-v1.json"
MARKDOWN_PATH = ROOT / "docs/architecture/SPORE_GOLDEN_PARITY_V1.md"

OBLIGATION_RE = re.compile(r"^### (GP-[A-Z]+-[0-9]{3})\b", re.MULTILINE)
ID_RE = re.compile(r"^GP-[A-Z]+-[0-9]{3}$")

EXPECTED_TOP_KEYS = {
    "$schema", "schema", "source_manifest_schema", "source_lineage_id",
    "source_commit", "source_tree", "qualification_transfer_policy",
    "destination_repository", "destination_parity_policy", "obligations",
}
EXPECTED_OBLIGATION_KEYS = {
    "id", "domain", "source_artifact_id", "statement", "execution_class",
    "status", "source_expectation_only", "destination_receipt", "status_justification",
}
ALLOWED_DOMAINS = {"boot", "identity", "helper", "authority", "firmware"}
ALLOWED_EXECUTION = {"vm", "unit", "firmware-vm"}
ALLOWED_STATUS = {
    "DEFINED", "MIGRATED_UNEXECUTED", "EXECUTED_FAILED", "EXECUTED_PASS",
    "SUPERSEDED_WITH_JUSTIFICATION",
}
SOURCE_LINEAGE_ID = "nixos-config-runtime-expendability-v1.3.2"
SOURCE_REPOSITORY = "Tristan-Stoltz-ERC/nixos-config"
SOURCE_BRANCH = "spore/runtime-expendability-v1.3.2-proof"
SOURCE_COMMIT = "5d80360768ee329c50756e71fbce4692ac3a8e45"
SOURCE_TREE = "51c04910b3a97586ecf88a46699b7de22e3e1b0b"

# artifact id -> (allowed domains, execution class, exact path, exact Git blob)
PERMITTED_SOURCE = {
    "fail-open-vm": (
        {"boot", "identity"}, "vm", "tests/spore-boot-fail-open.nix",
        "4fcc1618b2e993ca38f0c7a988a7afae65dd9ede",
    ),
    "helper-expendability-vm": (
        {"helper"}, "vm", "tests/spore-boot-helper-expendability.nix",
        "0fc089e9677d0398d79028baf65ddfe0682a4493",
    ),
    "systemd-authority-tests": (
        {"authority"}, "unit", "tests/test_spore_systemd_authority.py",
        "4db5d9af7f942a978c2c786ffcbb8f6b1ad35e26",
    ),
    "ovmf-recovery-vm": (
        {"firmware"}, "firmware-vm", "tests/spore-boot-ovmf-recovery.nix",
        "b2f645694c53c579d43b2b8f8781b0e6b8a4fbd8",
    ),
}


class ContractError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ContractError(f"{path}: top level must be an object")
    return data


def markdown_obligation_ids(text: str) -> list[str]:
    return OBLIGATION_RE.findall(text)


def validate_receipt_schema(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("additionalProperties") is not False:
        errors.append("parity receipt schema must reject additional properties")
    props = schema.get("properties")
    if not isinstance(props, dict):
        return errors + ["parity receipt schema properties must be an object"]
    expected_consts = {
        "schema": "spore-parity-receipt-v1",
        "evidence_tier": "parity-only",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "destination_repository": "Luminous-Dynamics/spore",
        "destination_qualification": "NOT_ESTABLISHED",
    }
    for key, expected in expected_consts.items():
        prop = props.get(key)
        if not isinstance(prop, dict) or prop.get("const") != expected:
            errors.append(f"parity receipt {key} must be const {expected!r}")
    result = props.get("result")
    if not isinstance(result, dict) or set(result.get("enum", [])) != {"PASS", "FAIL"}:
        errors.append("parity receipt result must be exactly PASS or FAIL")
    required = set(schema.get("required", []))
    for key in {
        "obligation_id", "source_artifact_id", "source_blob_sha1",
        "destination_commit", "destination_tree", "destination_artifact_sha256",
        "runner_identity", "toolchain_identity", "execution_class", "result",
        "destination_qualification",
    }:
        if key not in required:
            errors.append(f"parity receipt schema must require {key}")
    return errors


def validate(parity: dict[str, Any], migration: dict[str, Any], markdown: str) -> list[str]:
    errors: list[str] = []

    if set(parity) != EXPECTED_TOP_KEYS:
        errors.append(
            "parity top-level keys differ from frozen v1 contract: "
            f"expected={sorted(EXPECTED_TOP_KEYS)} actual={sorted(parity)}"
        )

    constants = {
        "$schema": "./spore-golden-parity-v1.schema.json",
        "schema": "spore-golden-parity-v1",
        "source_manifest_schema": "spore-migration-manifest-v1",
        "source_lineage_id": SOURCE_LINEAGE_ID,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "qualification_transfer_policy": "never-inherit",
        "destination_repository": "Luminous-Dynamics/spore",
        "destination_parity_policy": "fresh-execution-required",
    }
    for key, expected in constants.items():
        if parity.get(key) != expected:
            errors.append(f"{key} must be {expected!r}, got {parity.get(key)!r}")

    if migration.get("schema") != parity.get("source_manifest_schema"):
        errors.append("parent migration schema does not match parity source_manifest_schema")
    if migration.get("qualification_transfer_policy") != "never-inherit":
        errors.append("parent migration manifest no longer forbids qualification inheritance")
    if migration.get("destination_repository") != parity.get("destination_repository"):
        errors.append("destination repository differs between migration and parity contracts")

    lineages = migration.get("lineages")
    lineage_by_id = {
        item.get("id"): item for item in lineages
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(lineages, list) else {}
    lineage = lineage_by_id.get(parity.get("source_lineage_id"))
    if lineage is None:
        errors.append("source_lineage_id is absent from parent migration manifest")
    else:
        if lineage.get("source_repository") != SOURCE_REPOSITORY:
            errors.append("source lineage repository drifted")
        if lineage.get("source_branch") != SOURCE_BRANCH:
            errors.append("source lineage branch drifted")
        if lineage.get("source_commit") != parity.get("source_commit"):
            errors.append("parity source_commit does not match parent lineage")
        if lineage.get("source_tree") != parity.get("source_tree"):
            errors.append("parity source_tree does not match parent lineage")
        if lineage.get("qualification_boundary") != "transformed-candidate":
            errors.append("source lineage must remain classified transformed-candidate")
        if lineage.get("source_mutation_in_qualification") is not True:
            errors.append("source lineage must retain source_mutation_in_qualification=true")

    artifacts = migration.get("artifacts")
    artifact_by_id = {
        item.get("id"): item for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(artifacts, list) else {}

    obligations = parity.get("obligations")
    if not isinstance(obligations, list):
        errors.append("obligations must be an array")
        obligations = []
    if len(obligations) != 25:
        errors.append(f"v1 corpus must contain exactly 25 obligations, got {len(obligations)}")

    seen: set[str] = set()
    used_sources: set[str] = set()

    for index, obligation in enumerate(obligations):
        prefix = f"obligations[{index}]"
        if not isinstance(obligation, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(obligation) != EXPECTED_OBLIGATION_KEYS:
            errors.append(
                f"{prefix} keys differ from frozen v1 obligation contract: "
                f"expected={sorted(EXPECTED_OBLIGATION_KEYS)} actual={sorted(obligation)}"
            )

        oid = obligation.get("id")
        if not isinstance(oid, str) or not ID_RE.fullmatch(oid):
            errors.append(f"{prefix}.id is invalid: {oid!r}")
        elif oid in seen:
            errors.append(f"duplicate obligation id: {oid}")
        else:
            seen.add(oid)

        domain = obligation.get("domain")
        if domain not in ALLOWED_DOMAINS:
            errors.append(f"{oid or prefix}: invalid domain {domain!r}")
        execution_class = obligation.get("execution_class")
        if execution_class not in ALLOWED_EXECUTION:
            errors.append(f"{oid or prefix}: invalid execution_class {execution_class!r}")
        statement = obligation.get("statement")
        if not isinstance(statement, str) or len(statement.strip()) < 10:
            errors.append(f"{oid or prefix}: statement is too short")
        if obligation.get("source_expectation_only") is not True:
            errors.append(f"{oid or prefix}: source_expectation_only must remain true")

        status = obligation.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{oid or prefix}: invalid status {status!r}")
        receipt = obligation.get("destination_receipt")
        justification = obligation.get("status_justification")
        if receipt is not None and (not isinstance(receipt, str) or not receipt.strip()):
            errors.append(f"{oid or prefix}: destination_receipt must be null or non-empty string")
        if justification is not None and (
            not isinstance(justification, str) or not justification.strip()
        ):
            errors.append(f"{oid or prefix}: status_justification must be null or non-empty string")
        if status in {"EXECUTED_PASS", "EXECUTED_FAILED"} and not receipt:
            errors.append(f"{oid or prefix}: executed status requires destination_receipt")
        if status not in {"EXECUTED_PASS", "EXECUTED_FAILED"} and receipt is not None:
            errors.append(f"{oid or prefix}: non-executed status cannot carry destination_receipt")
        if status == "SUPERSEDED_WITH_JUSTIFICATION" and not justification:
            errors.append(f"{oid or prefix}: superseded status requires status_justification")
        if status != "SUPERSEDED_WITH_JUSTIFICATION" and justification is not None:
            errors.append(f"{oid or prefix}: status_justification is only valid for superseded status")

        source_id = obligation.get("source_artifact_id")
        source_rule = PERMITTED_SOURCE.get(source_id)
        if source_rule is None:
            errors.append(f"{oid or prefix}: source_artifact_id {source_id!r} is not parity-approved")
        else:
            used_sources.add(source_id)
            allowed_domains, expected_execution, expected_path, expected_blob = source_rule
            if domain not in allowed_domains:
                errors.append(f"{oid or prefix}: domain {domain!r} incompatible with source {source_id!r}")
            if execution_class != expected_execution:
                errors.append(
                    f"{oid or prefix}: execution_class {execution_class!r} must be "
                    f"{expected_execution!r} for source {source_id!r}"
                )

        artifact = artifact_by_id.get(source_id)
        if artifact is None:
            errors.append(f"{oid or prefix}: source artifact {source_id!r} absent from migration manifest")
        else:
            if artifact.get("lineage_id") != SOURCE_LINEAGE_ID:
                errors.append(f"{oid or prefix}: source artifact moved to an unexpected lineage")
            if artifact.get("source_role") != "test-fixture":
                errors.append(f"{oid or prefix}: parity source must remain a test-fixture")
            if artifact.get("authority_class") != "test-evidence":
                errors.append(f"{oid or prefix}: parity source must remain test-evidence")
            if artifact.get("target_owner") != "spore":
                errors.append(f"{oid or prefix}: parity source target owner must remain spore")
            if artifact.get("migration_kind") != "move":
                errors.append(f"{oid or prefix}: parity fixture migration_kind must remain move")
            if artifact.get("destination_qualification") != "required":
                errors.append(f"{oid or prefix}: destination qualification must remain required")
            if source_rule is not None:
                _, _, expected_path, expected_blob = source_rule
                if artifact.get("source_path") != expected_path:
                    errors.append(f"{oid or prefix}: source artifact path drifted")
                if artifact.get("source_blob_sha1") != expected_blob:
                    errors.append(f"{oid or prefix}: source artifact blob drifted")

    if used_sources != set(PERMITTED_SOURCE):
        errors.append(
            "v1 corpus must cover every approved source fixture as a family: "
            f"used={sorted(used_sources)} expected={sorted(PERMITTED_SOURCE)}"
        )

    md_ids = markdown_obligation_ids(markdown)
    if len(md_ids) != len(set(md_ids)):
        errors.append("markdown contains duplicate GP obligation headings")
    if set(md_ids) != seen:
        errors.append(
            "markdown/manifest obligation sets differ: "
            f"markdown_only={sorted(set(md_ids) - seen)} manifest_only={sorted(seen - set(md_ids))}"
        )

    # Until the destination repository exists, parity is definition-only.
    if migration.get("destination_repository_status") == "not-created":
        advanced = [
            item.get("id") for item in obligations
            if isinstance(item, dict) and item.get("status") != "DEFINED"
        ]
        if advanced:
            errors.append(
                "destination repository is not created; no obligation may advance beyond DEFINED: "
                + ", ".join(str(x) for x in advanced)
            )
        receipts = [
            item.get("id") for item in obligations
            if isinstance(item, dict) and item.get("destination_receipt") is not None
        ]
        if receipts:
            errors.append(
                "destination repository is not created; destination receipts are impossible: "
                + ", ".join(str(x) for x in receipts)
            )

    return errors


def validate_files() -> list[str]:
    load_json(PARITY_SCHEMA_PATH)
    parity = load_json(PARITY_PATH)
    migration = load_json(MIGRATION_PATH)
    markdown = MARKDOWN_PATH.read_text()
    receipt_schema = load_json(RECEIPT_SCHEMA_PATH)
    return validate(parity, migration, markdown) + validate_receipt_schema(receipt_schema)


def main() -> int:
    try:
        errors = validate_files()
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: Spore golden parity corpus is relationally consistent")
    print("obligations=25")
    print(f"source_commit={SOURCE_COMMIT}")
    print(f"source_tree={SOURCE_TREE}")
    print("evidence_tier=parity-only")
    print("destination_qualification=NOT_ESTABLISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
