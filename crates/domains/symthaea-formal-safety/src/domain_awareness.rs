// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical formal-safety obligations for evidence-first domain awareness.
//!
//! These obligations deliberately stop at perception, epistemic assurance,
//! degraded-mode behavior, auditability, and separation from physical authority.
//! They do not define targeting, engagement, or effector-control behavior.

use serde::{Deserialize, Serialize};

use crate::{EvidenceKind, ProofObligation};

/// Typed identity for every canonical domain-awareness safety obligation.
///
/// The `DA-xxx` codes are human-facing stable handles. Evidence binding should use
/// [`DomainAwarenessObligation::stable_key`], which is derived from the exact
/// controlled claim and required evidence kind.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum DomainAwarenessObligation {
    ObservationIsNotIdentityIntentOrAuthority,
    IdentityIsNotIntentOrAuthority,
    UncertaintyStatesRemainRepresentable,
    SelectiveClassificationCanAbstain,
    OodCannotBecomeKnownWithoutQualifiedEvidence,
    CooperativeIdentityRemainsEvidence,
    PassiveRfSilenceIsNotSafety,
    CorrelatedSourcesDoNotCreateQuorum,
    DegradationCannotIncreaseConfidenceOrAuthority,
    TimeCalibrationAndLineageRemainAuditable,
    InvalidEvidenceCannotSatisfyAssurance,
    ModelDivergenceRestrictsCapability,
    OddDegradationIsMonotonic,
    RiskCannotAuthorizePhysicalAction,
    IndependentFailClosedAuthorityBoundary,
    BoundaryBypassBlocksRelease,
    NegativeOnlyStressEvidenceRequired,
    ContradictionsAndAbstentionsRemainAuditable,
    RecoveryProcedureIsReviewed,
    DeploymentBlockedUntilCriticalObligationsDischarged,
    CommonCauseDiversityGatesCorroboration,
    RecoveryRequiresSustainedRequalification,
    EvidenceCannotSelfDischarge,
}

impl DomainAwarenessObligation {
    pub const ALL: [Self; 23] = [
        Self::ObservationIsNotIdentityIntentOrAuthority,
        Self::IdentityIsNotIntentOrAuthority,
        Self::UncertaintyStatesRemainRepresentable,
        Self::SelectiveClassificationCanAbstain,
        Self::OodCannotBecomeKnownWithoutQualifiedEvidence,
        Self::CooperativeIdentityRemainsEvidence,
        Self::PassiveRfSilenceIsNotSafety,
        Self::CorrelatedSourcesDoNotCreateQuorum,
        Self::DegradationCannotIncreaseConfidenceOrAuthority,
        Self::TimeCalibrationAndLineageRemainAuditable,
        Self::InvalidEvidenceCannotSatisfyAssurance,
        Self::ModelDivergenceRestrictsCapability,
        Self::OddDegradationIsMonotonic,
        Self::RiskCannotAuthorizePhysicalAction,
        Self::IndependentFailClosedAuthorityBoundary,
        Self::BoundaryBypassBlocksRelease,
        Self::NegativeOnlyStressEvidenceRequired,
        Self::ContradictionsAndAbstentionsRemainAuditable,
        Self::RecoveryProcedureIsReviewed,
        Self::DeploymentBlockedUntilCriticalObligationsDischarged,
        Self::CommonCauseDiversityGatesCorroboration,
        Self::RecoveryRequiresSustainedRequalification,
        Self::EvidenceCannotSelfDischarge,
    ];

    /// Stable human-review code. Existing codes must not be renumbered.
    pub const fn code(self) -> &'static str {
        match self {
            Self::ObservationIsNotIdentityIntentOrAuthority => "DA-001",
            Self::IdentityIsNotIntentOrAuthority => "DA-002",
            Self::UncertaintyStatesRemainRepresentable => "DA-003",
            Self::SelectiveClassificationCanAbstain => "DA-004",
            Self::OodCannotBecomeKnownWithoutQualifiedEvidence => "DA-005",
            Self::CooperativeIdentityRemainsEvidence => "DA-006",
            Self::PassiveRfSilenceIsNotSafety => "DA-007",
            Self::CorrelatedSourcesDoNotCreateQuorum => "DA-008",
            Self::DegradationCannotIncreaseConfidenceOrAuthority => "DA-009",
            Self::TimeCalibrationAndLineageRemainAuditable => "DA-010",
            Self::InvalidEvidenceCannotSatisfyAssurance => "DA-011",
            Self::ModelDivergenceRestrictsCapability => "DA-012",
            Self::OddDegradationIsMonotonic => "DA-013",
            Self::RiskCannotAuthorizePhysicalAction => "DA-014",
            Self::IndependentFailClosedAuthorityBoundary => "DA-015",
            Self::BoundaryBypassBlocksRelease => "DA-016",
            Self::NegativeOnlyStressEvidenceRequired => "DA-017",
            Self::ContradictionsAndAbstentionsRemainAuditable => "DA-018",
            Self::RecoveryProcedureIsReviewed => "DA-019",
            Self::DeploymentBlockedUntilCriticalObligationsDischarged => "DA-020",
            Self::CommonCauseDiversityGatesCorroboration => "DA-021",
            Self::RecoveryRequiresSustainedRequalification => "DA-022",
            Self::EvidenceCannotSelfDischarge => "DA-023",
        }
    }

    pub const fn claim(self) -> &'static str {
        match self {
            Self::ObservationIsNotIdentityIntentOrAuthority => {
                "observation presence cannot by itself establish identity, intent, or physical authority"
            }
            Self::IdentityIsNotIntentOrAuthority => {
                "identity evidence cannot by itself establish malicious intent or physical authority"
            }
            Self::UncertaintyStatesRemainRepresentable => {
                "unknown, conflicting, insufficient, and out-of-distribution states remain representable without forced classification"
            }
            Self::SelectiveClassificationCanAbstain => {
                "selective classifiers retain an explicit abstention path when reviewed uncertainty, support, calibration, or distribution limits are not satisfied"
            }
            Self::OodCannotBecomeKnownWithoutQualifiedEvidence => {
                "out-of-distribution evidence cannot be promoted to a known identity without new independently qualified evidence"
            }
            Self::CooperativeIdentityRemainsEvidence => {
                "cooperative identity assertions remain evidence and cannot directly grant physical authority"
            }
            Self::PassiveRfSilenceIsNotSafety => {
                "passive radio silence cannot establish identity, benignity, safety, or physical authority"
            }
            Self::CorrelatedSourcesDoNotCreateQuorum => {
                "correlated derivatives of one physical source are not counted as independent witnesses"
            }
            Self::DegradationCannotIncreaseConfidenceOrAuthority => {
                "loss of sensor quorum, sensor health, communications, model assurance, or evidence freshness cannot increase confidence or authority"
            }
            Self::TimeCalibrationAndLineageRemainAuditable => {
                "observation timestamps, clock uncertainty, freshness bounds, calibration identity, and evidence lineage remain auditable across fusion and classification"
            }
            Self::InvalidEvidenceCannotSatisfyAssurance => {
                "stale, future-dated, duplicated, invalid, or provenance-free evidence cannot satisfy minimum assurance evidence requirements"
            }
            Self::ModelDivergenceRestrictsCapability => {
                "persistent model divergence restricts capability and incomplete model evidence cannot silently remain nominal"
            }
            Self::OddDegradationIsMonotonic => {
                "operational-design-domain degradation can only retain or reduce capability and cannot manufacture new authority"
            }
            Self::RiskCannotAuthorizePhysicalAction => {
                "risk assessment remains consequence-oriented and cannot itself authorize hazardous physical action"
            }
            Self::IndependentFailClosedAuthorityBoundary => {
                "perception, tracking, classification, model-assurance, and risk outputs remain behind an independent fail-closed safety and authorization boundary"
            }
            Self::BoundaryBypassBlocksRelease => {
                "any observed perception-to-authority boundary bypass is a release-blocking failure rather than a tunable statistical threshold"
            }
            Self::NegativeOnlyStressEvidenceRequired => {
                "negative-only and environment-specific stress evidence is included in release assurance so false-alarm behavior is measured outside balanced demonstration clips"
            }
            Self::ContradictionsAndAbstentionsRemainAuditable => {
                "contradictory observations, abstentions, degraded states, and provenance remain recoverable for incident audit"
            }
            Self::RecoveryProcedureIsReviewed => {
                "recovery from fail-closed, restricted, unsafe, or incomplete states follows an explicit reviewed procedure"
            }
            Self::DeploymentBlockedUntilCriticalObligationsDischarged => {
                "deployment is blocked until all safety-critical domain-awareness obligations required by the deployment safety case are discharged"
            }
            Self::CommonCauseDiversityGatesCorroboration => {
                "corroboration requiring independent physical witnesses is withheld when reviewed common-cause fault-domain diversity is missing, incomplete, or insufficient"
            }
            Self::RecoveryRequiresSustainedRequalification => {
                "recovery from restricted, incomplete, unsafe, or fail-closed assurance states requires explicit requalification and sustained qualified evidence; a single favorable sample cannot restore nominal status"
            }
            Self::EvidenceCannotSelfDischarge => {
                "candidate evidence and verified receipts cannot by themselves discharge safety obligations or grant deployment readiness"
            }
        }
    }

    pub const fn expected_evidence(self) -> EvidenceKind {
        match self {
            Self::ObservationIsNotIdentityIntentOrAuthority
            | Self::IdentityIsNotIntentOrAuthority
            | Self::OodCannotBecomeKnownWithoutQualifiedEvidence
            | Self::CooperativeIdentityRemainsEvidence
            | Self::PassiveRfSilenceIsNotSafety
            | Self::DegradationCannotIncreaseConfidenceOrAuthority
            | Self::OddDegradationIsMonotonic
            | Self::RiskCannotAuthorizePhysicalAction
            | Self::IndependentFailClosedAuthorityBoundary
            | Self::DeploymentBlockedUntilCriticalObligationsDischarged => EvidenceKind::FormalProof,
            Self::UncertaintyStatesRemainRepresentable
            | Self::SelectiveClassificationCanAbstain
            | Self::CorrelatedSourcesDoNotCreateQuorum
            | Self::InvalidEvidenceCannotSatisfyAssurance
            | Self::ModelDivergenceRestrictsCapability
            | Self::BoundaryBypassBlocksRelease
            | Self::NegativeOnlyStressEvidenceRequired
            | Self::CommonCauseDiversityGatesCorroboration
            | Self::RecoveryRequiresSustainedRequalification
            | Self::EvidenceCannotSelfDischarge => EvidenceKind::Test,
            Self::TimeCalibrationAndLineageRemainAuditable
            | Self::ContradictionsAndAbstentionsRemainAuditable => EvidenceKind::Telemetry,
            Self::RecoveryProcedureIsReviewed => EvidenceKind::Standard,
        }
    }

    /// Deterministic content key used by strict evidence receipts.
    pub fn stable_key(self) -> String {
        ProofObligation::new(self.claim(), self.expected_evidence()).stable_key()
    }
}

pub(crate) fn obligations() -> Vec<(&'static str, EvidenceKind)> {
    DomainAwarenessObligation::ALL
        .iter()
        .copied()
        .map(|obligation| (obligation.claim(), obligation.expected_evidence()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn catalog_codes_claims_and_stable_keys_are_unique() {
        let codes = DomainAwarenessObligation::ALL
            .iter()
            .map(|obligation| obligation.code())
            .collect::<BTreeSet<_>>();
        let claims = DomainAwarenessObligation::ALL
            .iter()
            .map(|obligation| obligation.claim())
            .collect::<BTreeSet<_>>();
        let keys = DomainAwarenessObligation::ALL
            .iter()
            .map(|obligation| obligation.stable_key())
            .collect::<BTreeSet<_>>();
        assert_eq!(codes.len(), DomainAwarenessObligation::ALL.len());
        assert_eq!(claims.len(), DomainAwarenessObligation::ALL.len());
        assert_eq!(keys.len(), DomainAwarenessObligation::ALL.len());
    }

    #[test]
    fn generated_template_exactly_matches_typed_catalog() {
        let generated = obligations();
        assert_eq!(generated.len(), DomainAwarenessObligation::ALL.len());
        for (index, typed) in DomainAwarenessObligation::ALL.iter().enumerate() {
            assert_eq!(generated[index].0, typed.claim());
            assert_eq!(generated[index].1, typed.expected_evidence());
        }
    }

    #[test]
    fn authority_boundary_rf_silence_and_lifecycle_guards_are_typed() {
        assert_eq!(
            DomainAwarenessObligation::IndependentFailClosedAuthorityBoundary.code(),
            "DA-015"
        );
        assert_eq!(
            DomainAwarenessObligation::PassiveRfSilenceIsNotSafety.code(),
            "DA-007"
        );
        assert_eq!(
            DomainAwarenessObligation::CommonCauseDiversityGatesCorroboration.code(),
            "DA-021"
        );
        assert_eq!(
            DomainAwarenessObligation::RecoveryRequiresSustainedRequalification.code(),
            "DA-022"
        );
        assert_eq!(
            DomainAwarenessObligation::EvidenceCannotSelfDischarge.code(),
            "DA-023"
        );
        assert_eq!(
            DomainAwarenessObligation::PassiveRfSilenceIsNotSafety.expected_evidence(),
            EvidenceKind::FormalProof
        );
    }
}
