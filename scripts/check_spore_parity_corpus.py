#!/usr/bin/env python3
"""Validate the Spore extraction behavioral parity corpus.

This checker validates catalog structure and provenance linkage only. It does not
claim that historical tests passed in the destination repository. Destination
qualification must execute migrated tests against exact committed destination
bytes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "docs" / "architecture" / "spore-parity-corpus-v1.json"
MANIFEST_PATH = ROOT / "docs" / "architecture" / "spore-migration-manifest-v1.json"

SCHEMA = "spore-parity-corpus-v1"
TRANSFER_POLICY = "never-inherit"
BEHAVIOR_ID = re.compile(r"^SPORE-PARITY-\d{3}$")
INVARIANT_ID = re.compile(r"^SPORE-\d{3}$")

REQUIRED_DOMAINS = {
    "availability",
    "boot-identity",
    "qualification",
    "lkg",
    "helper-expendability",
    "lifecycle",
    "firmware-recovery",
    "systemd-authority",
}
REQUIRED_ARTIFACTS = {
    "fail-open-vm",
    "helper-expendability-vm",
    "ovmf-recovery-vm",
    "systemd-authority-tests",
}
ALLOWED_INVARIANTS = {f"SPORE-{i:03d}" for i in range(1, 13)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(corpus: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if corpus.get("schema") != SCHEMA:
        errors.append(f"corpus schema must be {SCHEMA!r}")
    if corpus.get("qualification_transfer_policy") != TRANSFER_POLICY:
        errors.append("corpus qualification_transfer_policy must be never-inherit")
    if manifest.get("qualification_transfer_policy") != TRANSFER_POLICY:
        errors.append("migration manifest qualification_transfer_policy must be never-inherit")
    if corpus.get("source_manifest") != "docs/architecture/spore-migration-manifest-v1.json":
        errors.append("corpus must bind the canonical migration manifest path")

    required_domains = corpus.get("required_domains")
    if not isinstance(required_domains, list) or set(required_domains) != REQUIRED_DOMAINS:
        errors.append("required_domains must exactly match the frozen parity domain set")

    required_artifacts = corpus.get("required_artifacts")
    if not isinstance(required_artifacts, list) or set(required_artifacts) != REQUIRED_ARTIFACTS:
        errors.append("required_artifacts must exactly match the frozen historical fixture set")

    artifacts_raw = manifest.get("artifacts")
    if not isinstance(artifacts_raw, list):
        errors.append("migration manifest artifacts must be a list")
        artifacts_raw = []

    artifacts: dict[str, dict[str, Any]] = {}
    for artifact in artifacts_raw:
        if not isinstance(artifact, dict):
            errors.append("migration manifest contains a non-object artifact")
            continue
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append("migration manifest artifact has an invalid id")
            continue
        if artifact_id in artifacts:
            errors.append(f"duplicate migration manifest artifact id: {artifact_id}")
        artifacts[artifact_id] = artifact

    behaviors = corpus.get("behaviors")
    if not isinstance(behaviors, list) or not behaviors:
        errors.append("behaviors must be a non-empty list")
        behaviors = []

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    covered_domains: set[str] = set()
    referenced_artifacts: set[str] = set()

    for index, behavior in enumerate(behaviors):
        where = f"behaviors[{index}]"
        if not isinstance(behavior, dict):
            errors.append(f"{where} must be an object")
            continue

        behavior_id = behavior.get("id")
        if not isinstance(behavior_id, str) or not BEHAVIOR_ID.fullmatch(behavior_id):
            errors.append(f"{where}.id must match {BEHAVIOR_ID.pattern}")
        elif behavior_id in seen_ids:
            errors.append(f"duplicate behavior id: {behavior_id}")
        else:
            seen_ids.add(behavior_id)

        name = behavior.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{where}.name must be a non-empty string")
        elif name in seen_names:
            errors.append(f"duplicate behavior name: {name}")
        else:
            seen_names.add(name)

        selector = behavior.get("selector")
        if not isinstance(selector, str) or not selector.strip():
            errors.append(f"{where}.selector must be a non-empty source-test selector")

        artifact_id = behavior.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append(f"{where}.artifact_id must be a non-empty string")
        else:
            referenced_artifacts.add(artifact_id)
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                errors.append(f"{where} references unknown manifest artifact {artifact_id!r}")
            else:
                if artifact.get("target_owner") != "spore":
                    errors.append(f"{where} source artifact {artifact_id!r} is not owned by Spore")
                if artifact.get("destination_qualification") != "required":
                    errors.append(
                        f"{where} source artifact {artifact_id!r} must require destination qualification"
                    )
                if artifact.get("source_role") != "test-fixture":
                    errors.append(
                        f"{where} source artifact {artifact_id!r} must be a test-fixture"
                    )
                if not artifact.get("source_blob_sha1"):
                    errors.append(
                        f"{where} source artifact {artifact_id!r} lacks an exact source blob"
                    )

        domains = behavior.get("domains")
        if not isinstance(domains, list) or not domains:
            errors.append(f"{where}.domains must be a non-empty list")
        else:
            domain_set = set(domains)
            unknown = domain_set - REQUIRED_DOMAINS
            if unknown:
                errors.append(f"{where}.domains contains unknown values: {sorted(unknown)}")
            if len(domain_set) != len(domains):
                errors.append(f"{where}.domains contains duplicates")
            covered_domains.update(domain_set)

        invariants = behavior.get("invariants")
        if not isinstance(invariants, list) or not invariants:
            errors.append(f"{where}.invariants must be a non-empty list")
        else:
            for invariant in invariants:
                if (
                    not isinstance(invariant, str)
                    or not INVARIANT_ID.fullmatch(invariant)
                    or invariant not in ALLOWED_INVARIANTS
                ):
                    errors.append(f"{where} references unknown constitutional invariant {invariant!r}")

        must_hold = behavior.get("must_hold")
        if not isinstance(must_hold, list) or not must_hold:
            errors.append(f"{where}.must_hold must be a non-empty list")
        elif any(not isinstance(item, str) or not item.strip() for item in must_hold):
            errors.append(f"{where}.must_hold entries must be non-empty strings")

    if covered_domains != REQUIRED_DOMAINS:
        missing = REQUIRED_DOMAINS - covered_domains
        extra = covered_domains - REQUIRED_DOMAINS
        if missing:
            errors.append(f"parity corpus does not cover required domains: {sorted(missing)}")
        if extra:
            errors.append(f"parity corpus covers undeclared domains: {sorted(extra)}")

    if not REQUIRED_ARTIFACTS.issubset(referenced_artifacts):
        errors.append(
            "parity corpus does not reference all required historical fixtures: "
            f"{sorted(REQUIRED_ARTIFACTS - referenced_artifacts)}"
        )

    return errors


def main() -> int:
    corpus = load_json(CORPUS_PATH)
    manifest = load_json(MANIFEST_PATH)
    if not isinstance(corpus, dict) or not isinstance(manifest, dict):
        print("FAIL: corpus and manifest roots must both be JSON objects", file=sys.stderr)
        return 1

    errors = validate(corpus, manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "PASS: Spore parity corpus is structurally valid, provenance-linked, "
        "and still requires fresh destination qualification"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
