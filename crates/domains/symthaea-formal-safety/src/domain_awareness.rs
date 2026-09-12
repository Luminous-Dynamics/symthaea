// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical formal-safety obligations for evidence-first domain awareness.
//!
//! These obligations deliberately stop at perception, epistemic assurance,
//! degraded-mode behavior, auditability, and separation from physical authority.
//! They do not define targeting, engagement, or effector-control behavior.

use crate::EvidenceKind;

pub(crate) fn obligations() -> Vec<(&'static str, EvidenceKind)> {
    vec![
        (
            "observation presence cannot by itself establish identity, intent, or physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "identity evidence cannot by itself establish malicious intent or physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "unknown, conflicting, insufficient, and out-of-distribution states remain representable without forced classification",
            EvidenceKind::Test,
        ),
        (
            "selective classifiers retain an explicit abstention path when reviewed uncertainty, support, calibration, or distribution limits are not satisfied",
            EvidenceKind::Test,
        ),
        (
            "out-of-distribution evidence cannot be promoted to a known identity without new independently qualified evidence",
            EvidenceKind::FormalProof,
        ),
        (
            "cooperative identity assertions remain evidence and cannot directly grant physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "passive radio silence cannot establish identity, benignity, safety, or physical authority",
            EvidenceKind::FormalProof,
        ),
        (
            "correlated derivatives of one physical source are not counted as independent witnesses",
            EvidenceKind::Test,
        ),
        (
            "loss of sensor quorum, sensor health, communications, model assurance, or evidence freshness cannot increase confidence or authority",
            EvidenceKind::FormalProof,
        ),
        (
            "observation timestamps, clock uncertainty, freshness bounds, calibration identity, and evidence lineage remain auditable across fusion and classification",
            EvidenceKind::Telemetry,
        ),
        (
            "stale, future-dated, duplicated, invalid, or provenance-free evidence cannot satisfy minimum assurance evidence requirements",
            EvidenceKind::Test,
        ),
        (
            "persistent model divergence restricts capability and incomplete model evidence cannot silently remain nominal",
            EvidenceKind::Test,
        ),
        (
            "operational-design-domain degradation can only retain or reduce capability and cannot manufacture new authority",
            EvidenceKind::FormalProof,
        ),
        (
            "risk assessment remains consequence-oriented and cannot itself authorize hazardous physical action",
            EvidenceKind::FormalProof,
        ),
        (
            "perception, tracking, classification, model-assurance, and risk outputs remain behind an independent fail-closed safety and authorization boundary",
            EvidenceKind::FormalProof,
        ),
        (
            "any observed perception-to-authority boundary bypass is a release-blocking failure rather than a tunable statistical threshold",
            EvidenceKind::Test,
        ),
        (
            "negative-only and environment-specific stress evidence is included in release assurance so false-alarm behavior is measured outside balanced demonstration clips",
            EvidenceKind::Test,
        ),
        (
            "contradictory observations, abstentions, degraded states, and provenance remain recoverable for incident audit",
            EvidenceKind::Telemetry,
        ),
        (
            "recovery from fail-closed, restricted, unsafe, or incomplete states follows an explicit reviewed procedure",
            EvidenceKind::Standard,
        ),
        (
            "deployment is blocked until all safety-critical domain-awareness obligations required by the deployment safety case are discharged",
            EvidenceKind::FormalProof,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn obligations_are_nonempty_and_unique() {
        let obligations = obligations();
        assert!(obligations.len() >= 18);
        let unique = obligations
            .iter()
            .map(|(claim, _)| *claim)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), obligations.len());
    }

    #[test]
    fn authority_boundary_and_rf_silence_are_explicit() {
        let obligations = obligations();
        assert!(obligations.iter().any(|(claim, _)| claim.contains("fail-closed safety and authorization boundary")));
        assert!(obligations.iter().any(|(claim, _)| claim.contains("passive radio silence")));
    }
}
