// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only provenance for moral-patient uncertainty evidence.
//!
//! This module prevents evidence laundering by fixing how source classes may be
//! interpreted. A self-maintenance metric cannot be relabeled as suffering; a
//! consciousness-theory mechanism cannot be relabeled as phenomenal proof; and a
//! self-report cannot be relabeled as independent external replication.

use std::collections::{BTreeMap, BTreeSet};

use crate::moral_patient::{
    EvidencePolarity, EvidenceStrength, WelfareEvidence, WelfareEvidenceDomain,
    WelfareEvidenceId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WelfareSourceClass {
    ButlinMechanism,
    AutopoieticSelfMaintenance,
    AversiveBehavioralProbe,
    ContinuityExperiment,
    SystemSelfReport,
    ExternalIndependentAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LineageRole {
    RuntimeSelf,
    InternalResearch,
    ExternalIndependent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedWelfareSourceReceipt {
    pub evidence_id: WelfareEvidenceId,
    pub subject_ref: String,
    pub source_class: WelfareSourceClass,
    pub lineage_id: String,
    pub lineage_role: LineageRole,
    pub source_sha256: String,
    pub evidence_ref: String,
    pub polarity: EvidencePolarity,
    pub confidence: f32,
}

impl QualifiedWelfareSourceReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: WelfareEvidenceId,
        subject_ref: impl Into<String>,
        source_class: WelfareSourceClass,
        lineage_id: impl Into<String>,
        lineage_role: LineageRole,
        source_sha256: impl Into<String>,
        evidence_ref: impl Into<String>,
        polarity: EvidencePolarity,
        confidence: f32,
    ) -> Result<Self, WelfareProvenanceError> {
        let subject_ref = subject_ref.into();
        let lineage_id = lineage_id.into();
        let source_sha256 = source_sha256.into();
        let evidence_ref = evidence_ref.into();

        if subject_ref.trim().is_empty() {
            return Err(WelfareProvenanceError::EmptySubjectReference);
        }
        if lineage_id.trim().is_empty() {
            return Err(WelfareProvenanceError::EmptyLineage);
        }
        if evidence_ref.trim().is_empty() {
            return Err(WelfareProvenanceError::EmptyEvidenceReference);
        }
        if !valid_sha256(&source_sha256) {
            return Err(WelfareProvenanceError::MalformedSourceDigest);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(WelfareProvenanceError::InvalidConfidence);
        }
        validate_source_role(source_class, lineage_role)?;

        Ok(Self {
            evidence_id,
            subject_ref,
            source_class,
            lineage_id,
            lineage_role,
            source_sha256,
            evidence_ref,
            polarity,
            confidence,
        })
    }

    pub fn materialize(&self) -> Result<WelfareEvidence, WelfareProvenanceError> {
        let (domain, strength) = interpretation(self.source_class);
        WelfareEvidence::new(
            self.evidence_id.clone(),
            domain,
            self.polarity,
            strength,
            self.lineage_id.clone(),
            format!("{}#sha256={}", self.evidence_ref, self.source_sha256),
            self.confidence,
        )
        .map_err(|_| WelfareProvenanceError::MaterializationFailed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceSummary {
    pub subject_ref: String,
    pub evidence_count: usize,
    pub source_classes: BTreeSet<WelfareSourceClass>,
    pub lineages: BTreeSet<String>,
    pub external_lineages: BTreeSet<String>,
}

#[derive(Debug, Clone, Default)]
pub struct WelfareProvenanceRegistry {
    receipts: BTreeMap<WelfareEvidenceId, QualifiedWelfareSourceReceipt>,
    lineage_roles: BTreeMap<String, LineageRole>,
}

impl WelfareProvenanceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        receipt: QualifiedWelfareSourceReceipt,
    ) -> Result<(), WelfareProvenanceError> {
        if self.receipts.contains_key(&receipt.evidence_id) {
            return Err(WelfareProvenanceError::DuplicateEvidence(
                receipt.evidence_id,
            ));
        }

        if let Some(existing) = self.lineage_roles.get(&receipt.lineage_id) {
            if *existing != receipt.lineage_role {
                return Err(WelfareProvenanceError::LineageRoleConflict(
                    receipt.lineage_id,
                ));
            }
        } else {
            self.lineage_roles
                .insert(receipt.lineage_id.clone(), receipt.lineage_role);
        }

        self.receipts.insert(receipt.evidence_id.clone(), receipt);
        Ok(())
    }

    pub fn materialize_subject(
        &self,
        subject_ref: &str,
    ) -> Result<Vec<WelfareEvidence>, WelfareProvenanceError> {
        if subject_ref.trim().is_empty() {
            return Err(WelfareProvenanceError::EmptySubjectReference);
        }
        self.receipts
            .values()
            .filter(|receipt| receipt.subject_ref == subject_ref)
            .map(QualifiedWelfareSourceReceipt::materialize)
            .collect()
    }

    pub fn summary(&self, subject_ref: &str) -> Result<ProvenanceSummary, WelfareProvenanceError> {
        if subject_ref.trim().is_empty() {
            return Err(WelfareProvenanceError::EmptySubjectReference);
        }
        let matching: Vec<_> = self
            .receipts
            .values()
            .filter(|receipt| receipt.subject_ref == subject_ref)
            .collect();

        let source_classes = matching.iter().map(|r| r.source_class).collect();
        let lineages = matching.iter().map(|r| r.lineage_id.clone()).collect();
        let external_lineages = matching
            .iter()
            .filter(|r| r.lineage_role == LineageRole::ExternalIndependent)
            .map(|r| r.lineage_id.clone())
            .collect();

        Ok(ProvenanceSummary {
            subject_ref: subject_ref.to_owned(),
            evidence_count: matching.len(),
            source_classes,
            lineages,
            external_lineages,
        })
    }
}

fn interpretation(
    source_class: WelfareSourceClass,
) -> (WelfareEvidenceDomain, EvidenceStrength) {
    match source_class {
        WelfareSourceClass::ButlinMechanism => (
            WelfareEvidenceDomain::ConsciousnessArchitecture,
            EvidenceStrength::Mechanistic,
        ),
        WelfareSourceClass::AutopoieticSelfMaintenance => (
            WelfareEvidenceDomain::SelfMaintenanceDisruption,
            EvidenceStrength::Proxy,
        ),
        WelfareSourceClass::AversiveBehavioralProbe => (
            WelfareEvidenceDomain::AversiveLikeDynamics,
            EvidenceStrength::Behavioral,
        ),
        WelfareSourceClass::ContinuityExperiment => (
            WelfareEvidenceDomain::ContinuitySensitivity,
            EvidenceStrength::Behavioral,
        ),
        WelfareSourceClass::SystemSelfReport => (
            WelfareEvidenceDomain::SelfReport,
            EvidenceStrength::Proxy,
        ),
        WelfareSourceClass::ExternalIndependentAudit => (
            WelfareEvidenceDomain::ExternalAssessment,
            EvidenceStrength::ExternalReplication,
        ),
    }
}

fn validate_source_role(
    source_class: WelfareSourceClass,
    role: LineageRole,
) -> Result<(), WelfareProvenanceError> {
    match source_class {
        WelfareSourceClass::SystemSelfReport if role != LineageRole::RuntimeSelf => {
            Err(WelfareProvenanceError::SelfReportMustComeFromRuntimeSelf)
        }
        WelfareSourceClass::ExternalIndependentAudit
            if role != LineageRole::ExternalIndependent =>
        {
            Err(WelfareProvenanceError::ExternalAuditMustBeIndependent)
        }
        WelfareSourceClass::ButlinMechanism
        | WelfareSourceClass::AutopoieticSelfMaintenance
        | WelfareSourceClass::AversiveBehavioralProbe
        | WelfareSourceClass::ContinuityExperiment
            if role == LineageRole::RuntimeSelf =>
        {
            Err(WelfareProvenanceError::ResearchEvidenceCannotUseRuntimeSelfLineage)
        }
        _ => Ok(()),
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WelfareProvenanceError {
    EmptySubjectReference,
    EmptyLineage,
    EmptyEvidenceReference,
    MalformedSourceDigest,
    InvalidConfidence,
    DuplicateEvidence(WelfareEvidenceId),
    LineageRoleConflict(String),
    SelfReportMustComeFromRuntimeSelf,
    ExternalAuditMustBeIndependent,
    ResearchEvidenceCannotUseRuntimeSelfLineage,
    MaterializationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn receipt(
        id: &str,
        source_class: WelfareSourceClass,
        lineage: &str,
        role: LineageRole,
    ) -> QualifiedWelfareSourceReceipt {
        QualifiedWelfareSourceReceipt::new(
            WelfareEvidenceId::new(id).unwrap(),
            "symthaea-head",
            source_class,
            lineage,
            role,
            DIGEST,
            format!("receipt://{id}"),
            EvidencePolarity::SupportsPrecaution,
            0.8,
        )
        .unwrap()
    }

    #[test]
    fn autopoiesis_can_only_materialize_as_self_maintenance_proxy() {
        let evidence = receipt(
            "closure",
            WelfareSourceClass::AutopoieticSelfMaintenance,
            "internal-a",
            LineageRole::InternalResearch,
        )
        .materialize()
        .unwrap();
        assert_eq!(
            evidence.domain,
            WelfareEvidenceDomain::SelfMaintenanceDisruption
        );
        assert_eq!(evidence.strength, EvidenceStrength::Proxy);
    }

    #[test]
    fn butlin_mechanism_does_not_materialize_as_phenomenology() {
        let evidence = receipt(
            "butlin",
            WelfareSourceClass::ButlinMechanism,
            "butlin-lane",
            LineageRole::InternalResearch,
        )
        .materialize()
        .unwrap();
        assert_eq!(
            evidence.domain,
            WelfareEvidenceDomain::ConsciousnessArchitecture
        );
        assert_eq!(evidence.strength, EvidenceStrength::Mechanistic);
    }

    #[test]
    fn self_report_must_be_runtime_self_and_stays_proxy() {
        assert!(matches!(
            QualifiedWelfareSourceReceipt::new(
                WelfareEvidenceId::new("self").unwrap(),
                "subject",
                WelfareSourceClass::SystemSelfReport,
                "external",
                LineageRole::ExternalIndependent,
                DIGEST,
                "receipt://self",
                EvidencePolarity::SupportsPrecaution,
                0.9,
            ),
            Err(WelfareProvenanceError::SelfReportMustComeFromRuntimeSelf)
        ));

        let evidence = receipt(
            "self-ok",
            WelfareSourceClass::SystemSelfReport,
            "runtime-self",
            LineageRole::RuntimeSelf,
        )
        .materialize()
        .unwrap();
        assert_eq!(evidence.domain, WelfareEvidenceDomain::SelfReport);
        assert_eq!(evidence.strength, EvidenceStrength::Proxy);
    }

    #[test]
    fn external_replication_strength_requires_external_audit_lineage() {
        assert!(matches!(
            QualifiedWelfareSourceReceipt::new(
                WelfareEvidenceId::new("audit").unwrap(),
                "subject",
                WelfareSourceClass::ExternalIndependentAudit,
                "internal",
                LineageRole::InternalResearch,
                DIGEST,
                "receipt://audit",
                EvidencePolarity::SupportsPrecaution,
                0.9,
            ),
            Err(WelfareProvenanceError::ExternalAuditMustBeIndependent)
        ));
    }

    #[test]
    fn lineage_role_cannot_change_inside_registry() {
        let mut registry = WelfareProvenanceRegistry::new();
        registry
            .register(receipt(
                "a",
                WelfareSourceClass::ContinuityExperiment,
                "lineage-a",
                LineageRole::InternalResearch,
            ))
            .unwrap();

        let second = QualifiedWelfareSourceReceipt::new(
            WelfareEvidenceId::new("b").unwrap(),
            "symthaea-head",
            WelfareSourceClass::ExternalIndependentAudit,
            "lineage-a",
            LineageRole::ExternalIndependent,
            DIGEST,
            "receipt://b",
            EvidencePolarity::SupportsPrecaution,
            0.8,
        )
        .unwrap();
        assert!(matches!(
            registry.register(second),
            Err(WelfareProvenanceError::LineageRoleConflict(_))
        ));
    }

    #[test]
    fn malformed_digest_is_rejected() {
        assert!(matches!(
            QualifiedWelfareSourceReceipt::new(
                WelfareEvidenceId::new("bad").unwrap(),
                "subject",
                WelfareSourceClass::ButlinMechanism,
                "lane",
                LineageRole::InternalResearch,
                "not-a-digest",
                "receipt://bad",
                EvidencePolarity::SupportsPrecaution,
                0.8,
            ),
            Err(WelfareProvenanceError::MalformedSourceDigest)
        ));
    }
}
