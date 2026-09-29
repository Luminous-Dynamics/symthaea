// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic structural audit of the Millennium evidence chain.
//! This verifies provenance and ancestry only; it never evaluates scientific truth.
//!
//! The audit is deliberately a projection over immutable records. It is suitable
//! for indexing in a distributed knowledge graph (DKG), while the underlying
//! record/event layer remains the source of truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::criterion_evidence::CriterionEvidenceEligibility;
use crate::external_observation::ExternalExperimentalObservation;
use crate::independent_assessment::IndependentAssessment;
use crate::replication::ReplicationRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceChainStatus {
    Complete,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceChainDiagnosticCode {
    InvalidObservation,
    InvalidAssessment,
    InvalidReplication,
    InvalidCriterionEvidence,
    AssessmentObservationMismatch,
    AssessmentCommitmentMismatch,
    ReplicationObservationMismatch,
    ReplicationAssessmentMismatch,
    ReplicationCommitmentMismatch,
    CriterionLineageMismatch,
    EvidenceObservationMismatch,
    EvidenceAssessmentMismatch,
    EvidenceReplicationMismatch,
    EvidenceCommitmentMismatch,
}

impl EvidenceChainDiagnosticCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidObservation => "invalid_observation",
            Self::InvalidAssessment => "invalid_assessment",
            Self::InvalidReplication => "invalid_replication",
            Self::InvalidCriterionEvidence => "invalid_criterion_evidence",
            Self::AssessmentObservationMismatch => "assessment_observation_mismatch",
            Self::AssessmentCommitmentMismatch => "assessment_commitment_mismatch",
            Self::ReplicationObservationMismatch => "replication_observation_mismatch",
            Self::ReplicationAssessmentMismatch => "replication_assessment_mismatch",
            Self::ReplicationCommitmentMismatch => "replication_commitment_mismatch",
            Self::CriterionLineageMismatch => "criterion_lineage_mismatch",
            Self::EvidenceObservationMismatch => "evidence_observation_mismatch",
            Self::EvidenceAssessmentMismatch => "evidence_assessment_mismatch",
            Self::EvidenceReplicationMismatch => "evidence_replication_mismatch",
            Self::EvidenceCommitmentMismatch => "evidence_commitment_mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceChainDiagnostic {
    pub code: EvidenceChainDiagnosticCode,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceChainAudit {
    pub status: EvidenceChainStatus,
    pub challenge_id: String,
    pub criterion_id: String,
    pub criterion_generation: String,
    pub observation_id: String,
    pub assessment_id: String,
    pub replication_id: String,
    pub evidence_id: String,
    pub diagnostics: Vec<EvidenceChainDiagnostic>,
}

impl EvidenceChainAudit {
    /// Audit one complete chain. Complete means only that all structural
    /// envelopes and cross-record links are internally consistent.
    pub fn audit(
        observation: &ExternalExperimentalObservation,
        assessment: &IndependentAssessment,
        replication: &ReplicationRecord,
        evidence: &CriterionEvidenceEligibility,
    ) -> Self {
        let mut diagnostics = Vec::new();

        if !observation.verify_integrity() {
            diagnostics.push(d(EvidenceChainDiagnosticCode::InvalidObservation, "observation envelope integrity failed"));
        }
        if !assessment.verify_integrity() {
            diagnostics.push(d(EvidenceChainDiagnosticCode::InvalidAssessment, "assessment envelope integrity failed"));
        }
        if !replication.verify_integrity() {
            diagnostics.push(d(EvidenceChainDiagnosticCode::InvalidReplication, "replication envelope integrity failed"));
        }
        if !evidence.verify_integrity() {
            diagnostics.push(d(EvidenceChainDiagnosticCode::InvalidCriterionEvidence, "criterion-evidence envelope integrity failed"));
        }

        if assessment.observation_id != observation.observation_id
            || assessment.observation_record_digest != observation.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::AssessmentObservationMismatch, "assessment does not point to the exact observation record"));
        }
        if assessment.commitment_event_id != observation.commitment_event_id
            || assessment.candidate_id != observation.candidate_id
            || assessment.binding_digest != observation.binding_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::AssessmentCommitmentMismatch, "assessment commitment/candidate/binding ancestry differs from observation"));
        }
        if replication.observation_id != observation.observation_id
            || replication.observation_record_digest != observation.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::ReplicationObservationMismatch, "replication does not point to the exact observation record"));
        }
        if replication.assessment_id != assessment.assessment_id
            || replication.assessment_record_digest != assessment.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::ReplicationAssessmentMismatch, "replication does not point to the exact assessment record"));
        }
        if replication.commitment_event_id != observation.commitment_event_id
            || replication.candidate_id != observation.candidate_id
            || replication.binding_digest != observation.binding_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::ReplicationCommitmentMismatch, "replication commitment/candidate/binding ancestry differs from observation"));
        }
        if evidence.challenge_id != observation.challenge_id
            || evidence.criterion_id != observation.criterion_id
            || evidence.criterion_generation != observation.criterion_generation
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::CriterionLineageMismatch, "criterion-evidence metadata differs from immutable observation lineage"));
        }
        if evidence.observation_id != observation.observation_id
            || evidence.observation_record_digest != observation.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::EvidenceObservationMismatch, "criterion evidence does not point to the exact observation record"));
        }
        if evidence.assessment_id != assessment.assessment_id
            || evidence.assessment_record_digest != assessment.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::EvidenceAssessmentMismatch, "criterion evidence does not point to the exact assessment record"));
        }
        if evidence.replication_id != replication.replication_id
            || evidence.replication_record_digest != replication.record_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::EvidenceReplicationMismatch, "criterion evidence does not point to the exact replication record"));
        }
        if evidence.commitment_event_id != observation.commitment_event_id
            || evidence.candidate_id != observation.candidate_id
            || evidence.binding_digest != observation.binding_digest
        {
            diagnostics.push(d(EvidenceChainDiagnosticCode::EvidenceCommitmentMismatch, "criterion evidence commitment/candidate/binding ancestry differs from observation"));
        }

        let status = if diagnostics.is_empty() {
            EvidenceChainStatus::Complete
        } else {
            EvidenceChainStatus::Invalid
        };

        Self {
            status,
            challenge_id: observation.challenge_id.clone(),
            criterion_id: observation.criterion_id.clone(),
            criterion_generation: observation.criterion_generation.clone(),
            observation_id: observation.observation_id.clone(),
            assessment_id: assessment.assessment_id.clone(),
            replication_id: replication.replication_id.clone(),
            evidence_id: evidence.evidence_id.clone(),
            diagnostics,
        }
    }

    pub fn is_structurally_complete(&self) -> bool {
        self.status == EvidenceChainStatus::Complete
    }

    /// Stable content identity for this audit projection.
    ///
    /// This is not a signature and does not authenticate any actor. It commits
    /// to the normalized audit fields and diagnostic set so independent
    /// indexers can derive the same projection identity.
    pub fn audit_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:evidence-chain-audit:v1 ");
        for s in [
            status_code(self.status),
            self.challenge_id.as_str(),
            self.criterion_id.as_str(),
            self.criterion_generation.as_str(),
            self.observation_id.as_str(),
            self.assessment_id.as_str(),
            self.replication_id.as_str(),
            self.evidence_id.as_str(),
        ] {
            put_string(&mut h, s);
        }
        h.update((self.diagnostics.len() as u64).to_be_bytes());
        for diagnostic in &self.diagnostics {
            put_string(&mut h, diagnostic.code.as_str());
            put_string(&mut h, &diagnostic.detail);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn d(code: EvidenceChainDiagnosticCode, detail: &str) -> EvidenceChainDiagnostic {
    EvidenceChainDiagnostic {
        code,
        detail: detail.into(),
    }
}

fn status_code(status: EvidenceChainStatus) -> &'static str {
    match status {
        EvidenceChainStatus::Complete => "complete",
        EvidenceChainStatus::Invalid => "invalid",
    }
}

fn put_string(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(status: EvidenceChainStatus, diagnostics: Vec<EvidenceChainDiagnostic>) -> EvidenceChainAudit {
        EvidenceChainAudit {
            status,
            challenge_id: "challenge:1".into(),
            criterion_id: "criterion:1".into(),
            criterion_generation: "generation:1".into(),
            observation_id: "observation:1".into(),
            assessment_id: "assessment:1".into(),
            replication_id: "replication:1".into(),
            evidence_id: "evidence:1".into(),
            diagnostics,
        }
    }

    #[test]
    fn diagnostic_codes_are_stable_and_typed() {
        assert_eq!(EvidenceChainDiagnosticCode::CriterionLineageMismatch.as_str(), "criterion_lineage_mismatch");
        let diagnostic = EvidenceChainDiagnostic {
            code: EvidenceChainDiagnosticCode::CriterionLineageMismatch,
            detail: "lineage differs".into(),
        };
        let json = serde_json::to_string(&diagnostic).unwrap();
        assert!(json.contains("criterion_lineage_mismatch"));
        let decoded: EvidenceChainDiagnostic = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, diagnostic);
    }

    #[test]
    fn audit_digest_is_deterministic() {
        let a = sample(EvidenceChainStatus::Complete, Vec::new());
        let b = a.clone();
        assert_eq!(a.audit_digest(), b.audit_digest());
    }

    #[test]
    fn audit_digest_changes_when_diagnostic_changes() {
        let a = sample(EvidenceChainStatus::Invalid, vec![EvidenceChainDiagnostic {
            code: EvidenceChainDiagnosticCode::InvalidObservation,
            detail: "observation envelope integrity failed".into(),
        }]);
        let mut b = a.clone();
        b.diagnostics[0].detail = "changed".into();
        assert_ne!(a.audit_digest(), b.audit_digest());
    }

    #[test]
    fn audit_digest_changes_when_lineage_changes() {
        let a = sample(EvidenceChainStatus::Complete, Vec::new());
        let mut b = a.clone();
        b.criterion_generation = "generation:2".into();
        assert_ne!(a.audit_digest(), b.audit_digest());
    }

    #[test]
    fn audit_round_trips_without_losing_typed_diagnostics() {
        let a = sample(EvidenceChainStatus::Invalid, vec![
            EvidenceChainDiagnostic {
                code: EvidenceChainDiagnosticCode::EvidenceCommitmentMismatch,
                detail: "ancestry differs".into(),
            },
        ]);
        let json = serde_json::to_string(&a).unwrap();
        let b: EvidenceChainAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.audit_digest(), b.audit_digest());
    }
}
