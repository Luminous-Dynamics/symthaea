#!/usr/bin/env python3
"""Fail-closed inventory for legacy epistemic authority labels.

EPI-SEM-001A is intentionally audit-only.  This script does not reinterpret,
repair, or authorize any epistemic claim.  It freezes the currently known Rust
surfaces where legacy E3/E4 labels or scalar E/N/M quality collapse exist so
new uses cannot spread without an explicit review of the inventory.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCHEMA = "epi-sem-001a-audit-v1"
AUTHORITY_SCOPE = "legacy-epistemic-semantic-inventory-only"

LEGACY_AUTHORITY_LABELS = (
    "E3CryptographicallyProven",
    "E4PubliclyReproducible",
)

EXPECTED_LABEL_FILES = {
    "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs",
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs",
    "src/cognitive_loop/cycle_consciousness.rs",
    "src/consciousness/causal_explanation.rs",
    "src/consciousness/epistemic_tiers.rs",
}

# These are semantic-debt witnesses, not positive authority assertions.
# A later repair is expected to make one or more of them disappear; that repair
# must update this audit and explain the changed authority boundary explicitly.
EXPECTED_WITNESSES = {
    "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs": (
        "EmpiricalTier::from_z_score(z_score)",
        "EmpiricalTier::E3CryptographicallyProven",
        "EmpiricalTier::E4PubliclyReproducible",
    ),
    "src/consciousness/causal_explanation.rs": (
        "self.evidence.len() >= 20 && self.confidence > 0.9",
        "EmpiricalTier::E3CryptographicallyProven",
    ),
    "src/cognitive_loop/cycle_consciousness.rs": (
        "self.stats.total_cycles > 1000",
        "EmpiricalTier::E3CryptographicallyProven",
    ),
    "src/consciousness/epistemic_tiers.rs": (
        "pub fn quality_score(&self) -> f64",
        "pub fn contextual_quality_score(&self, context: EpistemicContext) -> f64",
        "EmpiricalTier::E3CryptographicallyProven",
        "EmpiricalTier::E4PubliclyReproducible",
    ),
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs": (
        "E3CryptographicallyProven = 3",
        "E4PubliclyReproducible = 4",
    ),
}

SCAN_ROOTS = ("src", "crates", "apps", "tests", "benches", "examples")
SKIP_PARTS = {"target", ".git", ".direnv", "node_modules"}


def rust_files(repo_root: Path):
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.rs"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            yield path


def rel(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    observed_label_files: set[str] = set()
    for path in rust_files(repo_root):
        text = path.read_text(encoding="utf-8")
        if any(label in text for label in LEGACY_AUTHORITY_LABELS):
            observed_label_files.add(rel(repo_root, path))

    missing_files = sorted(EXPECTED_LABEL_FILES - observed_label_files)
    unexpected_files = sorted(observed_label_files - EXPECTED_LABEL_FILES)

    missing_witnesses: dict[str, list[str]] = {}
    for file_name, witnesses in EXPECTED_WITNESSES.items():
        path = repo_root / file_name
        if not path.is_file():
            missing_witnesses[file_name] = ["<file missing>"]
            continue
        text = path.read_text(encoding="utf-8")
        missing = [witness for witness in witnesses if witness not in text]
        if missing:
            missing_witnesses[file_name] = missing

    ok = not missing_files and not unexpected_files and not missing_witnesses

    report = {
        "schema": SCHEMA,
        "authority_scope": AUTHORITY_SCOPE,
        "legacy_authority_labels": list(LEGACY_AUTHORITY_LABELS),
        "observed_label_files": sorted(observed_label_files),
        "expected_label_files": sorted(EXPECTED_LABEL_FILES),
        "missing_expected_files": missing_files,
        "unexpected_new_files": unexpected_files,
        "missing_semantic_debt_witnesses": missing_witnesses,
        "result": "PASS_INVENTORY" if ok else "REVIEW_REQUIRED",
        "nonclaims": [
            "inventory pass does not validate legacy epistemic semantics",
            "inventory pass does not establish cryptographic proof",
            "inventory pass does not establish reproducibility",
            "inventory pass does not establish statistical or causal validity",
            "inventory pass does not grant MEL-EPI claim authority",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not ok:
        print(
            "EPI-SEM-001A inventory changed: review the semantic boundary and "
            "update the audit deliberately; do not silently extend the allowlist.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
