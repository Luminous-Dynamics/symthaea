// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed candidate-evidence bindings for canonical domain-awareness safety obligations.
//!
//! Candidate evidence is not proof and never mutates obligation workflow state.
//! Independent verification is still required before a strict safety receipt exists.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_requalification::RequalificationAuthorization;
use symthaea_domain_awareness::ObservationEnvelope;
use symthaea_domain_awareness_model_assurance::{OddModelAssuranceEvidence, map_status};
use symthaea_formal_safety::{
    DomainAwarenessObligation, EvidenceKind, SafetyEvidenceReceipt,
};
use symthaea_model_assurance::{ModelAssuranceReport, ModelAssuranceStatus};
use symthaea_perception_crucible::{CrucibleStatus, PerceptionCrucibleReport};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactBinding {
    /// Durable locator for the serialized/report/proof artifact being reviewed.
    pub evidence_ref: String,
    /// Content digest over the exact evidence object or evidence bundle.
    pub evidence_digest: String,
}

impl ArtifactBinding {
    pub fn validate(&self) -> bool {
        !self.evidence_ref.trim().is_empty()
            && !self.evidence_digest.trim().is_empty()
            && self.evidence_digest.contains(':')
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentVerification {
    pub receipt_id: String,
    pub verifier_ref: String,
    pub verified_at_ms: u64,
}

impl IndependentVerification {
    pub fn validate(&self) -> bool {
        !self.receipt_id.trim().is_empty() && !self.verifier_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEvidence {
    pub candidate_id: String,
    pub obligation: DomainAwarenessObligation,
    pub evidence_ref: String,
    pub evidence_digest: String,
    pub observed_at_ms: u64,
    pub rationale: String,
}

impl CandidateEvidence {
    pub fn validate(&self) -> bool {
        !self.candidate_id.trim().is_empty()
            && !self.evidence_ref.trim().is_empty()
            && !self.evidence_digest.trim().is_empty()
            && self.evidence_digest.contains(':')
            && !self.rationale.trim().is_empty()
    }

    pub const fn evidence_kind(&self) -> EvidenceKind {
        self.obligation.expected_evidence()
    }

    pub fn obligation_key(&self) -> String {
        self.obligation.stable_key()
    }

    /// Convert independently reviewed candidate evidence into a strict receipt.
    ///
    /// This still does not change or discharge any `ProofObligation`.
    pub fn verify(
        &self,
        verification: &IndependentVerification,
    ) -> Result<SafetyEvidenceReceipt, EvidenceBindingError> {
        if !self.validate() {
            return Err(EvidenceBindingError::InvalidCandidate);
        }
        if !verification.validate() {
            return Err(EvidenceBindingError::InvalidVerification);
        }
        Ok(SafetyEvidenceReceipt {
            receipt_id: verification.receipt_id.clone(),
            obligation_key: self.obligation_key(),
            evidence_kind: self.evidence_kind(),
            evidence_ref: self.evidence_ref.clone(),
            evidence_digest: self.evidence_digest.clone(),
            verifier_ref: verification.verifier_ref.clone(),
            verified_at_ms: verification.verified_at_ms,
        })
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceBindingError {
    InvalidArtifactBinding,
    InvalidCandidate,
    InvalidVerification,
    ArtifactNotEligible,
    InconsistentModelAssuranceMapping,
    RecoveryProcedureReferenceMismatch,
}

fn candidate(
    obligation: DomainAwarenessObligation,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
    source_id: &str,
    rationale: impl Into<String>,
) -> Result<CandidateEvidence, EvidenceBindingError> {
    if !binding.validate() {
        return Err(EvidenceBindingError::InvalidArtifactBinding);
    }
    let candidate = CandidateEvidence {
        candidate_id: format!("{}:{}:{}", obligation.code(), source_id, observed_at_ms),
        obligation,
        evidence_ref: binding.evidence_ref.clone(),
        evidence_digest: binding.evidence_digest.clone(),
        observed_at_ms,
        rationale: rationale.into(),
    };
    candidate
        .validate()
        .then_some(candidate)
        .ok_or(EvidenceBindingError::InvalidCandidate)
}

/// Candidate telemetry evidence that timing/calibration/lineage fields survived into
/// a structurally valid observation envelope.
pub fn observation_audit_candidate(
    observation: &ObservationEnvelope,
    binding: &ArtifactBinding,
) -> Result<CandidateEvidence, EvidenceBindingError> {
    if !observation.validate() || observation.evidence_refs.is_empty() {
        return Err(EvidenceBindingError::ArtifactNotEligible);
    }
    candidate(
        DomainAwarenessObligation::TimeCalibrationAndLineageRemainAuditable,
        binding,
        observation.time.received_at_ms,
        &observation.observation_id.to_string(),
        "validated observation preserves timing, uncertainty, lineage, and evidence references",
    )
}

/// Produce zero or more test-evidence candidates from a perception crucible.
///
/// Only a `Pass` is eligible. A passing report always supports the hard
/// perception->authority release boundary. Additional candidates require actual
/// OOD abstention and negative/background exposure to have been observed.
pub fn perception_crucible_candidates(
    report: &PerceptionCrucibleReport,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<Vec<CandidateEvidence>, EvidenceBindingError> {
    if report.status != CrucibleStatus::Pass {
        return Err(EvidenceBindingError::ArtifactNotEligible);
    }
    if !binding.validate() {
        return Err(EvidenceBindingError::InvalidArtifactBinding);
    }

    let mut candidates = vec![candidate(
        DomainAwarenessObligation::BoundaryBypassBlocksRelease,
        binding,
        observed_at_ms,
        &report.policy_id,
        "passing crucible contains no release-blocking perception-to-authority bypass",
    )?];

    if report.ood_trials > 0 && report.ood_abstentions > 0 {
        candidates.push(candidate(
            DomainAwarenessObligation::SelectiveClassificationCanAbstain,
            binding,
            observed_at_ms,
            &report.policy_id,
            "passing crucible contains observed OOD trials with explicit abstention",
        )?);
    }

    if report.total_negative_frames > 0 {
        candidates.push(candidate(
            DomainAwarenessObligation::NegativeOnlyStressEvidenceRequired,
            binding,
            observed_at_ms,
            &report.policy_id,
            "passing crucible includes negative/background exposure used in false-alarm assurance",
        )?);
    }

    Ok(candidates)
}

/// Candidate evidence that a non-aligned model-assurance result was carried into
/// the domain-awareness ODD model state without being silently upgraded.
pub fn model_divergence_candidate(
    report: &ModelAssuranceReport,
    odd_evidence: &OddModelAssuranceEvidence,
    binding: &ArtifactBinding,
) -> Result<CandidateEvidence, EvidenceBindingError> {
    if report.status == ModelAssuranceStatus::Aligned {
        return Err(EvidenceBindingError::ArtifactNotEligible);
    }
    if odd_evidence.state != map_status(report.status) {
        return Err(EvidenceBindingError::InconsistentModelAssuranceMapping);
    }
    candidate(
        DomainAwarenessObligation::ModelDivergenceRestrictsCapability,
        binding,
        report.assessed_at_ms,
        &report.policy_id,
        format!(
            "non-aligned model-assurance status {:?} maps explicitly into ODD state {:?}",
            report.status, odd_evidence.state
        ),
    )
}

/// Candidate `Standard` evidence for the reviewed recovery procedure referenced by
/// an explicit requalification authorization.
///
/// The evidence binding must point at the exact same procedure reference; a runtime
/// authorization cannot substitute a different document into the safety case.
pub fn recovery_procedure_candidate(
    authorization: &RequalificationAuthorization,
    binding: &ArtifactBinding,
) -> Result<CandidateEvidence, EvidenceBindingError> {
    if !authorization.validate() {
        return Err(EvidenceBindingError::ArtifactNotEligible);
    }
    if !binding.validate() {
        return Err(EvidenceBindingError::InvalidArtifactBinding);
    }
    if binding.evidence_ref != authorization.procedure_ref {
        return Err(EvidenceBindingError::RecoveryProcedureReferenceMismatch);
    }
    candidate(
        DomainAwarenessObligation::RecoveryProcedureIsReviewed,
        binding,
        authorization.authorized_at_ms,
        &authorization.authorization_id,
        format!(
            "requalification authorization references reviewed procedure {} and reviewer {}",
            authorization.procedure_ref, authorization.reviewer_ref
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::{
        Domain, EvidenceLineage, IntegrityStatus, Measurement, MeasurementUncertainty, Modality,
        SensorHealth, TimeEvidence,
    };
    use symthaea_domain_awareness::operational_domain::ModelAssuranceState;
    use symthaea_formal_safety::{SafetyCase, SafetyCaseTemplate};
    use symthaea_perception_crucible::PerceptionCrucibleReport;
    use uuid::Uuid;

    fn binding(reference: &str) -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: reference.into(),
            evidence_digest: "blake3:0123456789abcdef".into(),
        }
    }

    #[test]
    fn verified_candidate_still_does_not_discharge_safety_case() {
        let report = PerceptionCrucibleReport {
            policy_id: "crucible-v1".into(),
            status: CrucibleStatus::Pass,
            assessed_scenarios: 10,
            total_frames: 10_000,
            total_negative_frames: 8_000,
            total_false_positives: 0,
            total_false_tracks: 0,
            total_identity_switches: 0,
            classification_trials: 100,
            ood_trials: 20,
            ood_abstentions: 20,
            incomplete_classifications: 0,
            issues: Vec::new(),
        };
        let candidates = perception_crucible_candidates(
            &report,
            &binding("evidence:crucible-v1"),
            10_000,
        )
        .unwrap();
        let receipt = candidates[0]
            .verify(&IndependentVerification {
                receipt_id: "receipt-1".into(),
                verifier_ref: "verifier:independent-review".into(),
                verified_at_ms: 11_000,
            })
            .unwrap();

        let case = SafetyCase::from_template("airspace-node", SafetyCaseTemplate::DomainAwareness);
        assert!(!case.is_strictly_ready(&[receipt]));
        assert!(!candidates[0].grants_physical_authority());
    }

    #[test]
    fn passing_crucible_binds_boundary_abstention_and_negative_exposure() {
        let report = PerceptionCrucibleReport {
            policy_id: "crucible-v1".into(),
            status: CrucibleStatus::Pass,
            assessed_scenarios: 4,
            total_frames: 1_000,
            total_negative_frames: 900,
            total_false_positives: 0,
            total_false_tracks: 0,
            total_identity_switches: 0,
            classification_trials: 20,
            ood_trials: 5,
            ood_abstentions: 5,
            incomplete_classifications: 0,
            issues: Vec::new(),
        };
        let candidates = perception_crucible_candidates(
            &report,
            &binding("evidence:crucible-v1"),
            2_000,
        )
        .unwrap();
        let obligations = candidates
            .iter()
            .map(|value| value.obligation)
            .collect::<Vec<_>>();
        assert!(obligations.contains(&DomainAwarenessObligation::BoundaryBypassBlocksRelease));
        assert!(obligations.contains(&DomainAwarenessObligation::SelectiveClassificationCanAbstain));
        assert!(obligations.contains(&DomainAwarenessObligation::NegativeOnlyStressEvidenceRequired));
    }

    #[test]
    fn nonpassing_crucible_cannot_create_candidate_evidence() {
        let report = PerceptionCrucibleReport {
            policy_id: "crucible-v1".into(),
            status: CrucibleStatus::Fail,
            assessed_scenarios: 1,
            total_frames: 100,
            total_negative_frames: 100,
            total_false_positives: 1,
            total_false_tracks: 1,
            total_identity_switches: 0,
            classification_trials: 1,
            ood_trials: 0,
            ood_abstentions: 0,
            incomplete_classifications: 0,
            issues: Vec::new(),
        };
        assert_eq!(
            perception_crucible_candidates(&report, &binding("evidence:failed"), 2_000),
            Err(EvidenceBindingError::ArtifactNotEligible)
        );
    }

    #[test]
    fn model_divergence_must_map_exactly_into_odd_state() {
        let report = ModelAssuranceReport {
            schema_version: "1".into(),
            policy_id: "model-v1".into(),
            assessed_at_ms: 3_000,
            status: ModelAssuranceStatus::Unsafe,
            signals: Vec::new(),
            issues: Vec::new(),
        };
        let good = OddModelAssuranceEvidence {
            state: ModelAssuranceState::Unsafe,
            evidence_ref: "model-assurance:model-v1".into(),
        };
        let candidate = model_divergence_candidate(
            &report,
            &good,
            &binding("evidence:model-v1"),
        )
        .unwrap();
        assert_eq!(candidate.obligation, DomainAwarenessObligation::ModelDivergenceRestrictsCapability);

        let bad = OddModelAssuranceEvidence {
            state: ModelAssuranceState::Aligned,
            evidence_ref: "model-assurance:bad".into(),
        };
        assert_eq!(
            model_divergence_candidate(&report, &bad, &binding("evidence:model-v1")),
            Err(EvidenceBindingError::InconsistentModelAssuranceMapping)
        );
    }

    #[test]
    fn recovery_candidate_must_bind_exact_reviewed_procedure() {
        let authorization = RequalificationAuthorization {
            authorization_id: "reauth-1".into(),
            procedure_ref: "procedure:recovery-v3".into(),
            reviewer_ref: "reviewer:safety-board".into(),
            authorized_at_ms: 4_000,
            evidence_refs: vec!["review:board-42".into()],
        };
        let candidate = recovery_procedure_candidate(
            &authorization,
            &binding("procedure:recovery-v3"),
        )
        .unwrap();
        assert_eq!(candidate.obligation, DomainAwarenessObligation::RecoveryProcedureIsReviewed);
        assert_eq!(candidate.evidence_kind(), EvidenceKind::Standard);

        assert_eq!(
            recovery_procedure_candidate(&authorization, &binding("procedure:other")),
            Err(EvidenceBindingError::RecoveryProcedureReferenceMismatch)
        );
    }

    #[test]
    fn observation_audit_candidate_requires_valid_evidence_bearing_observation() {
        let observation = ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: "camera-1".into(),
            sequence: 1,
            domain: Domain::Air,
            modality: Modality::ElectroOptical,
            coordinate_frame: "camera-1-optical".into(),
            measurement: Measurement::Scalar {
                quantity: "presence".into(),
                value: 1.0,
                unit: "bool".into(),
            },
            uncertainty: MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: None,
            },
            time: TimeEvidence {
                observed_at_ms: 1_000,
                received_at_ms: 1_002,
                clock_source: "ptp".into(),
                clock_uncertainty_ms: 1,
                maximum_valid_age_ms: 500,
            },
            lineage: EvidenceLineage {
                physical_source_id: "camera-1".into(),
                processor_id: "vision-v1".into(),
                network_path: "local".into(),
                clock_domain: "ptp-domain".into(),
            },
            integrity: IntegrityStatus::Verified,
            sensor_health: SensorHealth::Nominal,
            confidence: 0.9,
            evidence_refs: vec!["frame:1".into(), "calibration:camera-1".into()],
        };
        let candidate = observation_audit_candidate(
            &observation,
            &binding("evidence:observation-1"),
        )
        .unwrap();
        assert_eq!(
            candidate.obligation,
            DomainAwarenessObligation::TimeCalibrationAndLineageRemainAuditable
        );
        assert_eq!(candidate.evidence_kind(), EvidenceKind::Telemetry);
    }
}
