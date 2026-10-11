// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formal safety obligations for evidence-first domain awareness.

use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};

/// Construct the baseline safety case for a domain-awareness deployment.
///
/// These obligations intentionally separate observation, classification, risk,
/// and physical authority. They are suitable for simulation, runtime monitors,
/// formal methods, telemetry evidence, and regulatory/engineering review.
pub fn domain_awareness_safety_case(subject: impl Into<String>) -> SafetyCase {
    let mut safety_case = SafetyCase::new(subject);

    for (claim, evidence_kind) in [
        (
            "observation presence cannot by itself establish identity or intent",
            EvidenceKind::FormalProof,
        ),
        (
            "cooperative identity assertions remain evidence and cannot directly grant physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "unknown, conflicting, and out-of-distribution states remain representable without forced classification",
            EvidenceKind::Test,
        ),
        (
            "sensor degradation, missing quorum, stale evidence, or model divergence cannot increase confidence or authority",
            EvidenceKind::FormalProof,
        ),
        (
            "correlated observations derived from one physical source are not counted as independent witnesses",
            EvidenceKind::Test,
        ),
        (
            "observation timestamps, clock uncertainty, freshness bounds, and evidence lineage are preserved across fusion",
            EvidenceKind::Telemetry,
        ),
        (
            "risk assessment is consequence-oriented and does not itself authorize hazardous physical action",
            EvidenceKind::FormalProof,
        ),
        (
            "physical actions remain behind an independent fail-closed safety and authorization boundary",
            EvidenceKind::Test,
        ),
        (
            "loss of communication or external consensus cannot increase local physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "incident evidence preserves contradictory observations and provenance for later audit",
            EvidenceKind::Telemetry,
        ),
        (
            "operational design limits and degraded modes are explicitly documented and reviewed",
            EvidenceKind::Standard,
        ),
        (
            "recovery from a fail-closed state follows an explicit reviewed procedure",
            EvidenceKind::Test,
        ),
    ] {
        safety_case.add_obligation(ProofObligation::new(claim, evidence_kind));
    }

    safety_case
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_case_is_nonempty_and_fail_closed_until_evidence_exists() {
        let case = domain_awareness_safety_case("harbor-awareness-node");
        assert!(case.obligations.len() >= 10);
        assert!(!case.is_discharged());
    }

    #[test]
    fn authority_separation_is_an_explicit_obligation() {
        let case = domain_awareness_safety_case("airspace-awareness-node");
        assert!(case.obligations.iter().any(|obligation| {
            obligation.claim.contains("independent fail-closed safety and authorization boundary")
        }));
    }
}
