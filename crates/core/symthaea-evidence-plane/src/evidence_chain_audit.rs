// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic structural audit of the Millennium evidence chain.
//! This verifies provenance and ancestry only; it never evaluates scientific truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::external_observation::ExternalExperimentalObservation;
use crate::independent_assessment::IndependentAssessment;
use crate::replication::ReplicationRecord;
use crate::criterion_evidence::CriterionEvidenceEligibility;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum EvidenceChainStatus { Complete, Invalid }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceChainDiagnostic { pub code: String, pub detail: String }

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
    /// Content-addressing digest for this exact audit payload.
    pub audit_digest: String,
}

impl EvidenceChainAudit {
    /// Audit one complete chain. Complete means only that all structural
    /// envelopes and cross-record links are internally consistent.
    pub fn audit(observation: &ExternalExperimentalObservation, assessment: &IndependentAssessment, replication: &ReplicationRecord, evidence: &CriterionEvidenceEligibility) -> Self {
        let mut diagnostics = Vec::new();
        if !observation.verify_integrity() { diagnostics.push(d("invalid_observation", "observation envelope integrity failed")); }
        if !assessment.verify_integrity() { diagnostics.push(d("invalid_assessment", "assessment envelope integrity failed")); }
        if !replication.verify_integrity() { diagnostics.push(d("invalid_replication", "replication envelope integrity failed")); }
        if !evidence.verify_integrity() { diagnostics.push(d("invalid_criterion_evidence", "criterion-evidence envelope integrity failed")); }
        if assessment.observation_id != observation.observation_id || assessment.observation_record_digest != observation.record_digest { diagnostics.push(d("assessment_observation_mismatch", "assessment does not point to the exact observation record")); }
        if assessment.commitment_event_id != observation.commitment_event_id || assessment.candidate_id != observation.candidate_id || assessment.binding_digest != observation.binding_digest { diagnostics.push(d("assessment_commitment_mismatch", "assessment commitment/candidate/binding ancestry differs from observation")); }
        if replication.observation_id != observation.observation_id || replication.observation_record_digest != observation.record_digest { diagnostics.push(d("replication_observation_mismatch", "replication does not point to the exact observation record")); }
        if replication.assessment_id != assessment.assessment_id || replication.assessment_record_digest != assessment.record_digest { diagnostics.push(d("replication_assessment_mismatch", "replication does not point to the exact assessment record")); }
        if replication.commitment_event_id != observation.commitment_event_id || replication.candidate_id != observation.candidate_id || replication.binding_digest != observation.binding_digest { diagnostics.push(d("replication_commitment_mismatch", "replication commitment/candidate/binding ancestry differs from observation")); }
        if evidence.challenge_id != observation.challenge_id || evidence.criterion_id != observation.criterion_id || evidence.criterion_generation != observation.criterion_generation { diagnostics.push(d("criterion_lineage_mismatch", "criterion-evidence metadata differs from immutable observation lineage")); }
        if evidence.observation_id != observation.observation_id || evidence.observation_record_digest != observation.record_digest { diagnostics.push(d("evidence_observation_mismatch", "criterion evidence does not point to the exact observation record")); }
        if evidence.assessment_id != assessment.assessment_id || evidence.assessment_record_digest != assessment.record_digest { diagnostics.push(d("evidence_assessment_mismatch", "criterion evidence does not point to the exact assessment record")); }
        if evidence.replication_id != replication.replication_id || evidence.replication_record_digest != replication.record_digest { diagnostics.push(d("evidence_replication_mismatch", "criterion evidence does not point to the exact replication record")); }
        if evidence.commitment_event_id != observation.commitment_event_id || evidence.candidate_id != observation.candidate_id || evidence.binding_digest != observation.binding_digest { diagnostics.push(d("evidence_commitment_mismatch", "criterion evidence commitment/candidate/binding ancestry differs from observation")); }
        let status = if diagnostics.is_empty() { EvidenceChainStatus::Complete } else { EvidenceChainStatus::Invalid };
        let mut audit = Self {
            status,
            challenge_id: observation.challenge_id.clone(),
            criterion_id: observation.criterion_id.clone(),
            criterion_generation: observation.criterion_generation.clone(),
            observation_id: observation.observation_id.clone(),
            assessment_id: assessment.assessment_id.clone(),
            replication_id: replication.replication_id.clone(),
            evidence_id: evidence.evidence_id.clone(),
            diagnostics,
            audit_digest: String::new(),
        };
        audit.audit_digest = audit.compute_digest();
        audit
    }

    /// Return the canonical content digest for this audit record.
    pub fn audit_digest(&self) -> String { self.compute_digest() }

    /// Verify the audit envelope and its content-addressing digest.
    pub fn verify_integrity(&self) -> bool {
        !self.challenge_id.trim().is_empty()
            && !self.criterion_id.trim().is_empty()
            && !self.criterion_generation.trim().is_empty()
            && !self.observation_id.trim().is_empty()
            && !self.assessment_id.trim().is_empty()
            && !self.replication_id.trim().is_empty()
            && !self.evidence_id.trim().is_empty()
            && !self.audit_digest.trim().is_empty()
            && self.audit_digest == self.compute_digest()
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:evidence-chain-audit:v1\0");
        put(&mut h, match self.status {
            EvidenceChainStatus::Complete => "complete",
            EvidenceChainStatus::Invalid => "invalid",
        });
        put(&mut h, &self.challenge_id);
        put(&mut h, &self.criterion_id);
        put(&mut h, &self.criterion_generation);
        put(&mut h, &self.observation_id);
        put(&mut h, &self.assessment_id);
        put(&mut h, &self.replication_id);
        put(&mut h, &self.evidence_id);
        h.update((self.diagnostics.len() as u64).to_be_bytes());
        for diagnostic in &self.diagnostics {
            put(&mut h, &diagnostic.code);
            put(&mut h, &diagnostic.detail);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn put(h: &mut Sha256, s: &str) {
    h.update((s.len() as u64).to_be_bytes());
    h.update(s.as_bytes());
}

fn d(code:&str, detail:&str)->EvidenceChainDiagnostic { EvidenceChainDiagnostic { code: code.into(), detail: detail.into() } }
