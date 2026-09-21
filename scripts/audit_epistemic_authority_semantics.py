#!/usr/bin/env python3
"""Fail-closed inventory for legacy epistemic authority semantics.

EPI-SEM-001A-R4 is audit-only. It inventories three independent discovery
surfaces plus the known producer/propagation contracts:

1. empirical proof/reproduction authority aliases;
2. normative/material high-authority aliases (axiomatic/foundational);
3. numeric/local E/N/M shapes that can hide authority behind integers.

It does not validate or repair legacy semantics and cannot mint authority.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCHEMA = "epi-sem-001a-r4-audit-v1"
AUTHORITY_SCOPE = "legacy-epistemic-semantic-inventory-only"

EMPIRICAL_AUTHORITY_MARKERS = (
    "E3CryptographicallyProven",
    "E4PubliclyReproducible",
    "CryptographicallyVerifiable",
    "PubliclyReproducible",
    "E3Cryptographic",
    "E4PublicRepro",
)

HIGH_AXIS_AUTHORITY_MARKERS = (
    "N3Axiomatic",
    "M3Foundational",
    "NormativeLevel::Foundational",
    "MaterialityLevel::Permanent",
)

NUMERIC_SHAPE_MARKERS = (
    "pub empirical: u8",
    "pub e_tier: u8",
    "struct LocalEpistemicClassification",
    "enum LocalEmpiricalLevel",
)

ALLOWED_EMPIRICAL_AUTHORITY_FILES = {
    "crates/bridges/symthaea-physics-bridge/src/discovery.rs",
    "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs",
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs",
    "crates/domains/symthaea-physics-catalog/src/discovery.rs",
    "src/cognitive_loop/cycle_consciousness.rs",
    "src/consciousness/causal_explanation.rs",
    "src/consciousness/epistemic_tiers.rs",
    "src/consciousness/mycelix_bridge.rs",
    "src/mycelix/mapper.rs",
    "src/mycelix/types.rs",
    "stubs/mycelix-sdk/src/epistemic/mod.rs",
    "tests/mycelix_integration.rs",
}

ALLOWED_HIGH_AXIS_AUTHORITY_FILES = {
    "crates/bridges/symthaea-mycelix-bridge/src/lib.rs",
    "crates/domains/symthaea-epistemic-types/src/global_ledger.rs",
    "src/consciousness/epistemic_tiers.rs",
    "src/consciousness/mycelix_bridge.rs",
    "src/mycelix/mapper.rs",
    "src/mycelix/types.rs",
    "stubs/mycelix-sdk/src/epistemic/mod.rs",
    "tests/mycelix_integration.rs",
}

ALLOWED_NUMERIC_SHAPE_FILES = {
    "crates/bridges/symthaea-mycelix-bridge/src/lib.rs",
    "crates/bridges/symthaea-physics-bridge/src/discovery.rs",
    "crates/domains/symthaea-physics-catalog/src/discovery.rs",
    "src/consciousness/mycelix_bridge.rs",
}

# Presence freezes known semantic debt; it is not approval.
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
        "EmpiricalLevel::CryptographicallyVerifiable => E::E3Cryptographic",
        "EmpiricalLevel::PubliclyReproducible => E::E4PublicRepro",
    ),
    "src/mycelix/mapper.rs": (
        "pub fn phi_to_empirical(&self, phi: f32, is_reproducible: bool) -> EmpiricalLevel",
        "!self.config.require_reproducibility_for_e4 || is_reproducible",
        "phi >= self.config.e4_threshold",
        "phi >= self.config.e3_threshold",
        "e.evidence_type == EvidenceType::CryptographicProof",
    ),
    "src/consciousness/mycelix_bridge.rs": (
        "pub fn from_factcheck(",
        "e if e >= 0.8 => 4, // reproducible",
        "e if e >= 0.6 => 3, // proven",
        "\"True\" => (0.9, 0.7, 0.6)",
        "pub fn create_epistemic_claim(",
        "eval.authenticity > 0.8",
        "EmpiricalLevel::E3Cryptographic",
    ),
    "crates/domains/symthaea-physics-catalog/src/discovery.rs": (
        "if is_open_source && simulation_confidence > 0.8",
        "if similarity > 0.9",
        "assert_eq!(d.lem.empirical, 4)",
        "assert_eq!(d.lem.normative, 3)",
    ),
}

HIGH_AXIS_PRODUCER_WITNESSES = {
    "src/consciousness/mycelix_bridge.rs": (
        "n if n >= 0.75 => 3, // axiomatic",
        "m if m >= 0.75 => 3, // foundational",
        "ActionType::Constitutional => NormativeLevel::N3Axiomatic",
    ),
    "src/mycelix/mapper.rs": (
        "output.importance > 0.8",
        "materiality: MaterialityLevel::Permanent",
        "e.evidence_type == EvidenceType::Consensus",
        "empirical: EmpiricalLevel::PrivatelyVerifiable",
    ),
    "crates/bridges/symthaea-mycelix-bridge/src/lib.rs": (
        "WorkspaceScope::Universal => Self::Foundational",
        "if importance >= 0.75",
        "LocalNormativeLevel::Foundational =>",
        "mycelix_sdk::epistemic::NormativeLevel::N3Axiomatic",
        "LocalMaterialityLevel::Permanent =>",
        "mycelix_sdk::epistemic::MaterialityLevel::M3Foundational",
    ),
}

SCALAR_COLLAPSE_WITNESSES = {
    "src/consciousness/epistemic_tiers.rs": (
        "pub fn quality_score(&self) -> f64",
        "pub fn contextual_quality_score(&self, context: EpistemicContext) -> f64",
    ),
    "src/consciousness/mycelix_bridge.rs": (
        "let quality = (e_tier as f32 / 4.0) * 0.40",
        "+ (n_tier as f32 / 3.0) * 0.35",
        "+ (m_tier as f32 / 3.0) * 0.25",
    ),
}

PROPAGATION_WITNESSES = {
    "src/consciousness/epistemic_tiers.rs": (
        "EmpiricalTier::E3CryptographicallyProven =>",
        "EmpiricalTier::E4PubliclyReproducible =>",
        "NormativeTier::N3Axiomatic => NormativeLevel::N3Axiomatic",
        "MaterialityTier::M3Foundational => MaterialityLevel::M3Foundational",
    ),
    "src/mycelix/types.rs": (
        "pub fn from_code(code: &str) -> Option<Self>",
        "3 => EmpiricalLevel::CryptographicallyVerifiable",
        "4 => EmpiricalLevel::PubliclyReproducible",
        "NormativeLevel::Foundational => N::N3Axiomatic",
        "MaterialityLevel::Permanent => M::M3Foundational",
    ),
    "stubs/mycelix-sdk/src/epistemic/mod.rs": (
        "self.empirical >= min_empirical && self.normative >= min_normative",
    ),
    "tests/mycelix_integration.rs": (
        "mapper.phi_to_empirical(0.35, false)",
        "EmpiricalLevel::CryptographicallyVerifiable",
        "NormativeLevel::Foundational",
        "MaterialityLevel::Permanent",
    ),
    "crates/domains/symthaea-physics-catalog/src/wasm_api.rs": (
        "pub fn classify_physics_claim(description: &str, confidence: f64, is_open_source: bool) -> String",
        "crate::discovery::classify_claim(",
        "crate::catalog::Domain::ModifiedGravity",
    ),
}

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
    "src/consciousness/mycelix_bridge.rs": (
        "Maps the 3D EpistemicPosition (empirical, normative, mythic)",
        "`normative` (0.0-1.0) → N-tier (0-3): source consensus",
        "`mythic` (0.0-1.0) → M-tier (0-3): persistence/foundationality",
    ),
}

POSITIVE_CONTROL_WITNESSES = {
    "crates/bridges/symthaea-physics-bridge/src/discovery.rs": (
        "A local simulation result supports at most E1",
        "open_source_and_confidence_do_not_imply_reproduction",
        "assert_eq!(d.lem.empirical, 1)",
    ),
    "crates/bridges/symthaea-mycelix-bridge/src/lib.rs": (
        "supports an E1 testimonial claim only",
        "stronger levels require replay artifacts, cryptographic verification, or a",
        "LocalEmpiricalLevel::Testimonial",
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


def observed_files(repo_root: Path, markers: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for path in rust_files(repo_root):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in markers):
            found.add(rel(repo_root, path))
    return found


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

    observed_empirical = observed_files(repo_root, EMPIRICAL_AUTHORITY_MARKERS)
    observed_high_axis = observed_files(repo_root, HIGH_AXIS_AUTHORITY_MARKERS)
    observed_numeric = observed_files(repo_root, NUMERIC_SHAPE_MARKERS)

    unexpected_empirical = sorted(observed_empirical - ALLOWED_EMPIRICAL_AUTHORITY_FILES)
    unexpected_high_axis = sorted(observed_high_axis - ALLOWED_HIGH_AXIS_AUTHORITY_FILES)
    unexpected_numeric = sorted(observed_numeric - ALLOWED_NUMERIC_SHAPE_FILES)

    missing_unsafe = missing_witnesses(repo_root, UNSAFE_PRODUCER_WITNESSES)
    missing_high_axis = missing_witnesses(repo_root, HIGH_AXIS_PRODUCER_WITNESSES)
    missing_scalar = missing_witnesses(repo_root, SCALAR_COLLAPSE_WITNESSES)
    missing_propagation = missing_witnesses(repo_root, PROPAGATION_WITNESSES)
    missing_axis = missing_witnesses(repo_root, AXIS_COLLISION_WITNESSES)
    missing_control = missing_witnesses(repo_root, POSITIVE_CONTROL_WITNESSES)

    failures = (
        unexpected_empirical,
        unexpected_high_axis,
        unexpected_numeric,
        missing_unsafe,
        missing_high_axis,
        missing_scalar,
        missing_propagation,
        missing_axis,
        missing_control,
    )
    ok = not any(failures)

    report = {
        "schema": SCHEMA,
        "authority_scope": AUTHORITY_SCOPE,
        "empirical_authority_markers": list(EMPIRICAL_AUTHORITY_MARKERS),
        "high_axis_authority_markers": list(HIGH_AXIS_AUTHORITY_MARKERS),
        "numeric_shape_markers": list(NUMERIC_SHAPE_MARKERS),
        "observed_empirical_authority_files": sorted(observed_empirical),
        "allowed_empirical_authority_files": sorted(ALLOWED_EMPIRICAL_AUTHORITY_FILES),
        "unexpected_empirical_authority_files": unexpected_empirical,
        "observed_high_axis_authority_files": sorted(observed_high_axis),
        "allowed_high_axis_authority_files": sorted(ALLOWED_HIGH_AXIS_AUTHORITY_FILES),
        "unexpected_high_axis_authority_files": unexpected_high_axis,
        "observed_numeric_shape_files": sorted(observed_numeric),
        "allowed_numeric_shape_files": sorted(ALLOWED_NUMERIC_SHAPE_FILES),
        "unexpected_numeric_shape_files": unexpected_numeric,
        "missing_unsafe_producer_witnesses": missing_unsafe,
        "missing_high_axis_producer_witnesses": missing_high_axis,
        "missing_scalar_collapse_witnesses": missing_scalar,
        "missing_propagation_witnesses": missing_propagation,
        "missing_axis_collision_witnesses": missing_axis,
        "missing_positive_control_witnesses": missing_control,
        "result": "PASS_INVENTORY" if ok else "REVIEW_REQUIRED",
        "nonclaims": [
            "inventory pass does not validate legacy E/N/M semantics",
            "inventory pass does not establish cryptographic proof or authentication",
            "inventory pass does not establish reproducibility or replication",
            "inventory pass does not establish normative consensus or axiomatic truth",
            "inventory pass does not establish foundational materiality or durability",
            "inventory pass does not establish statistical or causal validity",
            "inventory pass does not grant MEL-EPI, governance, or action authority",
            "an allowed occurrence is not approval of its semantics",
            "marker coverage is a regression ratchet, not proof of complete semantic discovery",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not ok:
        print(
            "EPI-SEM-001A-R4 inventory changed: review empirical aliases, high-axis aliases, "
            "numeric shapes, producers, scalar collapses, exports, axis definitions, or controls; "
            "do not silently expand the surface.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
