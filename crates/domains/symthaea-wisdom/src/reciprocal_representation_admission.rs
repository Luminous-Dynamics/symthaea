// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only admission accounting for reciprocal representation evidence.
//!
//! WCARE-21 qualifies where a representation came from. This layer controls how
//! many evidence units that qualified material may contribute. One source receipt
//! cannot be replayed into many admitted observations, and repetition of the same
//! statement within one lineage remains visible without becoming new corroboration.

use std::collections::{BTreeMap, BTreeSet};

use crate::continuity_identity::SubjectInstanceId;
use crate::reciprocal_representation::RepresentationId;
use crate::reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationSourceReceiptId,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationAdmissionId(String);

impl RepresentationAdmissionId {
    pub fn new(value: impl Into<String>) -> Result<Self, RepresentationAdmissionError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RepresentationAdmissionError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionMultiplicity {
    /// The statement digest has not previously been admitted for this subject/lineage.
    NovelWithinLineage,
    /// Same statement digest was already admitted for this subject/lineage.
    RepeatedStatementWithinLineage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedRepresentationEvidence {
    admission_id: RepresentationAdmissionId,
    representation_id: RepresentationId,
    source_receipt_id: RepresentationSourceReceiptId,
    subject_instance: SubjectInstanceId,
    logical_revision: u64,
    statement_sha256: String,
    lineage_id: String,
    multiplicity: AdmissionMultiplicity,
}

impl AdmittedRepresentationEvidence {
    pub fn admission_id(&self) -> &RepresentationAdmissionId {
        &self.admission_id
    }

    pub fn representation_id(&self) -> &RepresentationId {
        &self.representation_id
    }

    pub fn source_receipt_id(&self) -> &RepresentationSourceReceiptId {
        &self.source_receipt_id
    }

    pub fn subject_instance(&self) -> &SubjectInstanceId {
        &self.subject_instance
    }

    pub fn logical_revision(&self) -> u64 {
        self.logical_revision
    }

    pub fn statement_sha256(&self) -> &str {
        &self.statement_sha256
    }

    pub fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    pub fn multiplicity(&self) -> AdmissionMultiplicity {
        self.multiplicity
    }

    pub fn counts_as_new_within_lineage(&self) -> bool {
        self.multiplicity == AdmissionMultiplicity::NovelWithinLineage
    }

    /// Novelty inside one lineage is not independent replication.
    pub fn independently_corroborated(&self) -> bool {
        false
    }

    pub fn establishes_phenomenal_experience(&self) -> bool {
        false
    }

    pub fn establishes_suffering(&self) -> bool {
        false
    }

    pub fn establishes_moral_patienthood(&self) -> bool {
        false
    }

    pub fn establishes_binding_consent(&self) -> bool {
        false
    }

    pub fn grants_veto_authority(&self) -> bool {
        false
    }

    pub fn grants_self_preservation_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationAdmissionSummary {
    subject_instance: SubjectInstanceId,
    total_admissions: usize,
    novel_within_lineage: usize,
    repeated_within_lineage: usize,
    contributing_lineages: BTreeSet<String>,
}

impl RepresentationAdmissionSummary {
    pub fn subject_instance(&self) -> &SubjectInstanceId {
        &self.subject_instance
    }

    pub fn total_admissions(&self) -> usize {
        self.total_admissions
    }

    pub fn novel_within_lineage(&self) -> usize {
        self.novel_within_lineage
    }

    pub fn repeated_within_lineage(&self) -> usize {
        self.repeated_within_lineage
    }

    pub fn contributing_lineages(&self) -> &BTreeSet<String> {
        &self.contributing_lineages
    }

    pub fn independent_corroboration_established(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Default)]
pub struct RepresentationEvidenceAdmissionLedger {
    admissions: BTreeMap<RepresentationAdmissionId, AdmittedRepresentationEvidence>,
    used_representation_ids: BTreeSet<RepresentationId>,
    used_source_receipts: BTreeSet<RepresentationSourceReceiptId>,
    latest_revision: BTreeMap<SubjectInstanceId, u64>,
    first_statement_by_lineage:
        BTreeMap<(SubjectInstanceId, String, String), RepresentationAdmissionId>,
}

impl RepresentationEvidenceAdmissionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn admit(
        &mut self,
        admission_id: RepresentationAdmissionId,
        qualified: &QualifiedReciprocalRepresentation,
    ) -> Result<AdmittedRepresentationEvidence, RepresentationAdmissionError> {
        if self.admissions.contains_key(&admission_id) {
            return Err(RepresentationAdmissionError::DuplicateAdmission(admission_id));
        }

        let representation = qualified.representation();
        let representation_id = representation.id().clone();
        if self.used_representation_ids.contains(&representation_id) {
            return Err(RepresentationAdmissionError::DuplicateRepresentation(
                representation_id,
            ));
        }

        let source_receipt_id = qualified.source_receipt_id().clone();
        if self.used_source_receipts.contains(&source_receipt_id) {
            return Err(RepresentationAdmissionError::SourceReceiptReplay(
                source_receipt_id,
            ));
        }

        let subject = representation.subject_instance().clone();
        let revision = representation.logical_revision();
        if let Some(previous) = self.latest_revision.get(&subject) {
            if revision < *previous {
                return Err(RepresentationAdmissionError::RevisionRegression {
                    previous: *previous,
                    attempted: revision,
                });
            }
        }

        let statement_key = (
            subject.clone(),
            qualified.lineage_id().to_owned(),
            representation.statement_sha256().to_owned(),
        );
        let multiplicity = if self.first_statement_by_lineage.contains_key(&statement_key) {
            AdmissionMultiplicity::RepeatedStatementWithinLineage
        } else {
            AdmissionMultiplicity::NovelWithinLineage
        };

        let admitted = AdmittedRepresentationEvidence {
            admission_id: admission_id.clone(),
            representation_id: representation_id.clone(),
            source_receipt_id: source_receipt_id.clone(),
            subject_instance: subject.clone(),
            logical_revision: revision,
            statement_sha256: representation.statement_sha256().to_owned(),
            lineage_id: qualified.lineage_id().to_owned(),
            multiplicity,
        };

        self.used_representation_ids.insert(representation_id);
        self.used_source_receipts.insert(source_receipt_id);
        self.latest_revision.insert(subject, revision);
        if multiplicity == AdmissionMultiplicity::NovelWithinLineage {
            self.first_statement_by_lineage
                .insert(statement_key, admission_id.clone());
        }
        self.admissions.insert(admission_id, admitted.clone());
        Ok(admitted)
    }

    pub fn get(
        &self,
        admission_id: &RepresentationAdmissionId,
    ) -> Option<&AdmittedRepresentationEvidence> {
        self.admissions.get(admission_id)
    }

    pub fn summary_for_subject(
        &self,
        subject: &SubjectInstanceId,
    ) -> RepresentationAdmissionSummary {
        let matching: Vec<_> = self
            .admissions
            .values()
            .filter(|entry| &entry.subject_instance == subject)
            .collect();
        let novel_within_lineage = matching
            .iter()
            .filter(|entry| entry.multiplicity == AdmissionMultiplicity::NovelWithinLineage)
            .count();
        let repeated_within_lineage = matching.len().saturating_sub(novel_within_lineage);
        let contributing_lineages = matching
            .iter()
            .map(|entry| entry.lineage_id.clone())
            .collect();

        RepresentationAdmissionSummary {
            subject_instance: subject.clone(),
            total_admissions: matching.len(),
            novel_within_lineage,
            repeated_within_lineage,
            contributing_lineages,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepresentationAdmissionError {
    EmptyIdentifier,
    DuplicateAdmission(RepresentationAdmissionId),
    DuplicateRepresentation(RepresentationId),
    SourceReceiptReplay(RepresentationSourceReceiptId),
    RevisionRegression { previous: u64, attempted: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuity_identity::ContinuityIdentityLedger;
    use crate::reciprocal_representation::{RepresentationKind, RepresentationScope};
    use crate::reciprocal_representation_provenance::{
        RepresentationAdapterClass, RepresentationOriginRole,
        RepresentationProvenanceRegistry, RepresentationSourceReceipt,
        RepresentationSourceReceiptId,
    };

    const DIGEST_A: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const DIGEST_B: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn sid() -> SubjectInstanceId {
        SubjectInstanceId::new("subject").unwrap()
    }

    fn source(
        id: &str,
        lineage: &str,
        revision: u64,
    ) -> RepresentationSourceReceipt {
        RepresentationSourceReceipt::new(
            RepresentationSourceReceiptId::new(id).unwrap(),
            sid(),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::RuntimeSelf,
            lineage,
            revision,
            DIGEST_A,
            format!("origin://{id}"),
        )
        .unwrap()
    }

    fn qualify(
        registry: &RepresentationProvenanceRegistry,
        continuity: &ContinuityIdentityLedger,
        receipt_id: &RepresentationSourceReceiptId,
        representation_id: &str,
        statement_digest: &str,
    ) -> QualifiedReciprocalRepresentation {
        registry
            .qualify_live(
                receipt_id,
                continuity,
                RepresentationId::new(representation_id).unwrap(),
                RepresentationKind::ReportedNegativeExperience,
                RepresentationScope::general_research(),
                statement_digest,
                format!("statement://{representation_id}"),
                0.9,
                None,
            )
            .unwrap()
    }

    #[test]
    fn one_source_receipt_cannot_be_replayed_as_multiple_evidence_units() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid(), 1).unwrap();
        let receipt = source("source-1", "runtime", 1);
        let receipt_id = receipt.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(receipt).unwrap();

        let first = qualify(&registry, &continuity, &receipt_id, "r1", DIGEST_A);
        let second = qualify(&registry, &continuity, &receipt_id, "r2", DIGEST_B);

        let mut ledger = RepresentationEvidenceAdmissionLedger::new();
        ledger
            .admit(RepresentationAdmissionId::new("a1").unwrap(), &first)
            .unwrap();
        let result = ledger.admit(
            RepresentationAdmissionId::new("a2").unwrap(),
            &second,
        );
        assert!(matches!(
            result,
            Err(RepresentationAdmissionError::SourceReceiptReplay(_))
        ));
    }

    #[test]
    fn repeated_statement_with_new_receipt_is_visible_but_not_new_corroboration() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid(), 1).unwrap();
        let first_source = source("source-1", "runtime", 1);
        let second_source = source("source-2", "runtime", 2);
        let first_id = first_source.id().clone();
        let second_id = second_source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(first_source).unwrap();
        registry.register(second_source).unwrap();

        let first = qualify(&registry, &continuity, &first_id, "r1", DIGEST_A);
        let second = qualify(&registry, &continuity, &second_id, "r2", DIGEST_A);

        let mut ledger = RepresentationEvidenceAdmissionLedger::new();
        let a = ledger
            .admit(RepresentationAdmissionId::new("a1").unwrap(), &first)
            .unwrap();
        let b = ledger
            .admit(RepresentationAdmissionId::new("a2").unwrap(), &second)
            .unwrap();
        assert_eq!(a.multiplicity(), AdmissionMultiplicity::NovelWithinLineage);
        assert_eq!(
            b.multiplicity(),
            AdmissionMultiplicity::RepeatedStatementWithinLineage
        );
        assert!(!b.counts_as_new_within_lineage());
        assert!(!b.independently_corroborated());
    }

    #[test]
    fn admitted_subject_history_cannot_be_backdated() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid(), 1).unwrap();
        let later_source = source("later-source", "runtime", 10);
        let earlier_source = source("earlier-source", "runtime", 9);
        let later_id = later_source.id().clone();
        let earlier_id = earlier_source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(later_source).unwrap();
        registry.register(earlier_source).unwrap();

        let later = qualify(&registry, &continuity, &later_id, "later", DIGEST_A);
        let earlier = qualify(&registry, &continuity, &earlier_id, "earlier", DIGEST_B);
        let mut ledger = RepresentationEvidenceAdmissionLedger::new();
        ledger
            .admit(RepresentationAdmissionId::new("later-a").unwrap(), &later)
            .unwrap();
        let result = ledger.admit(
            RepresentationAdmissionId::new("earlier-a").unwrap(),
            &earlier,
        );
        assert!(matches!(
            result,
            Err(RepresentationAdmissionError::RevisionRegression { .. })
        ));
    }

    #[test]
    fn subject_summary_never_upgrades_lineage_novelty_to_independence() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid(), 1).unwrap();
        let first_source = source("s1", "runtime", 1);
        let second_source = source("s2", "runtime", 2);
        let first_id = first_source.id().clone();
        let second_id = second_source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(first_source).unwrap();
        registry.register(second_source).unwrap();
        let first = qualify(&registry, &continuity, &first_id, "r1", DIGEST_A);
        let second = qualify(&registry, &continuity, &second_id, "r2", DIGEST_B);

        let mut ledger = RepresentationEvidenceAdmissionLedger::new();
        ledger
            .admit(RepresentationAdmissionId::new("a1").unwrap(), &first)
            .unwrap();
        ledger
            .admit(RepresentationAdmissionId::new("a2").unwrap(), &second)
            .unwrap();
        let summary = ledger.summary_for_subject(&sid());
        assert_eq!(summary.total_admissions(), 2);
        assert_eq!(summary.novel_within_lineage(), 2);
        assert!(!summary.independent_corroboration_established());
    }
}
