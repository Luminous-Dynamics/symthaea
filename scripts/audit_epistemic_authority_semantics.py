#!/usr/bin/env python3
"""Fail-closed inventory for legacy epistemic authority semantics.

EPI-SEM-001A-R2 is deliberately audit-only. It inventories authority-bearing
E/N/M vocabulary, proxy-to-authority producers, cross-system propagation seams,
and one already-hardened positive control. It does not validate or repair the
legacy semantics and cannot mint epistemic authority.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCHEMA = "epi-sem-001a-r2-audit-v1"
AUTHORITY_SCOPE = "legacy-epistemic-semantic-inventory-only"

# Exact spelling differs across three coexisting E/N/M implementations. Scan
# all known aliases so a new authority-bearing use cannot hide behind spelling.
AUTHORITY_MARKERS = (
    "E3CryptographicallyProven",
    "E4PubliclyReproducible",
    "CryptographicallyVerifiable",
    "PubliclyReproducible",
    "E3Cryptographic",
    "E4PublicRepro",
)

# A vocabulary occurrence is not automatically a defect. This set includes
# compatibility definitions, tests, known unsafe producers, and the hardened
# physics bridge used as a positive control. New files using these authority
# terms require deliberate review rather than silently expanding the surface.
ALLOWED_AUTHORITY_VOCABULARY_FILES = {
    "crates/bridges/symthaea-physics-bridge/src/discovery.rs",
    "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs",
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs",
    "crates/domains/symthaea-physics-catalog/src/discovery.rs",
    "src/cognitive_loop/cycle_consciousness.rs",
    "src/consciousness/causal_explanation.rs",
    "src/consciousness/epistemic_tiers.rs",
    "src/mycelix/mapper.rs",
    "src/mycelix/types.rs",
    "stubs/mycelix-sdk/src/epistemic/mod.rs",
    "tests/mycelix_integration.rs",
}

# These witnesses freeze known semantic debt. Their presence is NOT approval;
# it makes the migration boundary explicit. A later repair should intentionally
# remove a witness and update this audit in a fresh subject.
UNSAFE_PRODUCER_WITNESSES = {
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
    "src/mycelix/types.rs": (
        "pub fn from_phi(phi: f32, has_reproducibility: bool) -> Self",
        "p if p >= 0.4 && has_reproducibility => Self::PubliclyReproducible",
        "p if p >= 0.3 => Self::CryptographicallyVerifiable",
        "impl From<EmpiricalLevel> for mycelix_sdk::epistemic::EmpiricalLevel",
        "EmpiricalLevel::CryptographicallyVerifiable => E::E3Cryptographic",
        "EmpiricalLevel::PubliclyReproducible => E::E4PublicRepro",
    ),
    "src/mycelix/mapper.rs": (
        "pub fn phi_to_empirical(&self, phi: f32, is_reproducible: bool) -> EmpiricalLevel",
        "phi >= self.config.e4_threshold && can_reach_e4",
        "phi >= self.config.e3_threshold",
        "EmpiricalLevel::CryptographicallyVerifiable",
        "e.evidence_type == EvidenceType::CryptographicProof",
    ),
    "crates/domains/symthaea-physics-catalog/src/discovery.rs": (
        "if is_open_source && simulation_confidence > 0.8",
        "if similarity > 0.9",
        "assert_eq!(d.lem.empirical, 4)",
        "assert_eq!(d.lem.normative, 3)",
    ),
}

# Propagation/gating surfaces are distinct from producers. They can preserve or
# amplify a misclassification, so the audit freezes them separately.
PROPAGATION_WITNESSES = {
    "src/consciousness/epistemic_tiers.rs": (
        "pub fn quality_score(&self) -> f64",
        "pub fn contextual_quality_score(&self, context: EpistemicContext) -> f64",
        "EmpiricalTier::E3CryptographicallyProven =>",
        "EmpiricalTier::E4PubliclyReproducible =>",
    ),
    "src/mycelix/types.rs": (
        "pub fn from_code(code: &str) -> Option<Self>",
        "3 => EmpiricalLevel::CryptographicallyVerifiable",
        "4 => EmpiricalLevel::PubliclyReproducible",
    ),
    "stubs/mycelix-sdk/src/epistemic/mod.rs": (
        "self.empirical >= min_empirical && self.normative >= min_normative",
    ),
    "tests/mycelix_integration.rs": (
        "mapper.phi_to_empirical(0.35, false)",
        "EmpiricalLevel::CryptographicallyVerifiable",
        "mapper.phi_to_empirical(0.5, true)",
        "EmpiricalLevel::PubliclyReproducible",
    ),
}

# There is also an axis-definition collision, especially on N: one shared type
# says N means who agrees a claim is valid, while the Mycelix bridge says N is
# who should have access/distribution scope. This must be resolved separately
# from E3/E4 repair rather than hidden by a numeric N0..N3 coincidence.
AXIS_COLLISION_WITNESSES = {
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs": (
        "/// N-Axis: WHO agrees this knowledge claim is valid?",
        "/// M-Axis: How PERMANENT is this knowledge?",
    ),
    "src/mycelix/types.rs": (
        "/// Normative Level - Who should have access to this?",
        "/// Materiality Level - How long should this persist?",
        "pub fn from_scope(scope: WorkspaceScope) -> Self",
        "pub fn from_importance(importance: f32) -> Self",
    ),
}

# The bridge implementation already demonstrates the desired bounded style:
# local confidence/open-source availability do not self-promote to E3/E4.
POSITIVE_CONTROL_WITNESSES = {
    "crates/bridges/symthaea-physics-bridge/src/discovery.rs": (
        "A local simulation result supports at most E1",
        "open_source_and_confidence_do_not_imply_reproduction",
        "assert_eq!(d.lem.empirical, 1)",
    ),
}

SCAN_ROOTS = ("src", "crates", "apps", "tests", "benches", "examples", "stubs")
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


def missing_witnesses(repo_root: Path, contract: dict[str, tuple[str, ...]]):
    missing: dict[str, list[str]] = {}
    for file_name, witnesses in contract.items():
        path = repo_root / file_name
        if not path.is_file():
            missing[file_name] = ["<file missing>"]
            continue
        text = path.read_text(encoding="utf-8")
        absent = [witness for witness in witnesses if witness not in text]
        if absent:
            missing[file_name] = absent
    return missing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    observed_vocabulary_files: set[str] = set()
    for path in rust_files(repo_root):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in AUTHORITY_MARKERS):
            observed_vocabulary_files.add(rel(repo_root, path))

    unexpected_vocabulary_files = sorted(
        observed_vocabulary_files - ALLOWED_AUTHORITY_VOCABULARY_FILES
    )

    missing_unsafe = missing_witnesses(repo_root, UNSAFE_PRODUCER_WITNESSES)
    missing_propagation = missing_witnesses(repo_root, PROPAGATION_WITNESSES)
    missing_axis = missing_witnesses(repo_root, AXIS_COLLISION_WITNESSES)
    missing_control = missing_witnesses(repo_root, POSITIVE_CONTROL_WITNESSES)

    ok = not any(
        (
            unexpected_vocabulary_files,
            missing_unsafe,
            missing_propagation,
            missing_axis,
            missing_control,
        )
    )

    report = {
        "schema": SCHEMA,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_markers": list(AUTHORITY_MARKERS),
        "observed_authority_vocabulary_files": sorted(observed_vocabulary_files),
        "allowed_authority_vocabulary_files": sorted(ALLOWED_AUTHORITY_VOCABULARY_FILES),
        "unexpected_authority_vocabulary_files": unexpected_vocabulary_files,
        "missing_unsafe_producer_witnesses": missing_unsafe,
        "missing_propagation_witnesses": missing_propagation,
        "missing_axis_collision_witnesses": missing_axis,
        "missing_positive_control_witnesses": missing_control,
        "result": "PASS_INVENTORY" if ok else "REVIEW_REQUIRED",
        "nonclaims": [
            "inventory pass does not validate legacy E/N/M semantics",
            "inventory pass does not establish cryptographic proof or authentication",
            "inventory pass does not establish reproducibility or replication",
            "inventory pass does not establish statistical or causal validity",
            "inventory pass does not grant MEL-EPI or action authority",
            "an allowed vocabulary occurrence is not approval of its semantics",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not ok:
        print(
            "EPI-SEM-001A-R2 inventory changed: review the semantic boundary; "
            "do not silently extend aliases, producers, conversions, or gates.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
