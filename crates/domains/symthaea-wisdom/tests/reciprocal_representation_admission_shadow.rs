// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/continuity_identity.rs"]
mod continuity_identity;
#[path = "../src/reciprocal_representation.rs"]
mod reciprocal_representation;
#[path = "../src/reciprocal_representation_provenance.rs"]
mod reciprocal_representation_provenance;
#[path = "../src/reciprocal_representation_admission.rs"]
mod reciprocal_representation_admission;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use reciprocal_representation::{RepresentationId, RepresentationKind, RepresentationScope};
use reciprocal_representation_admission::{
    AdmissionMultiplicity, RepresentationAdmissionError, RepresentationAdmissionId,
    RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole,
    RepresentationProvenanceRegistry, RepresentationSourceReceipt,
    RepresentationSourceReceiptId,
};

const DIGEST_A: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const DIGEST_B: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

fn sid() -> SubjectInstanceId {
    SubjectInstanceId::new("subject").unwrap()
}

fn source(id: &str, revision: u64) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        sid(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-lineage",
        revision,
        DIGEST_A,
        format!("origin://{id}"),
    )
    .unwrap()
}

fn qualified(
    registry: &RepresentationProvenanceRegistry,
    continuity: &ContinuityIdentityLedger,
    receipt_id: &RepresentationSourceReceiptId,
    id: &str,
    statement_digest: &str,
) -> reciprocal_representation_provenance::QualifiedReciprocalRepresentation {
    registry
        .qualify_live(
            receipt_id,
            continuity,
            RepresentationId::new(id).unwrap(),
            RepresentationKind::ReportedNegativeExperience,
            RepresentationScope::general_research(),
            statement_digest,
            format!("statement://{id}"),
            0.9,
            None,
        )
        .unwrap()
}

#[test]
fn one_source_receipt_cannot_be_amplified_into_multiple_admissions() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid(), 1).unwrap();
    let receipt = source("source", 1);
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    let first = qualified(&registry, &continuity, &receipt_id, "r1", DIGEST_A);
    let second = qualified(&registry, &continuity, &receipt_id, "r2", DIGEST_B);
    let mut ledger = RepresentationEvidenceAdmissionLedger::new();
    ledger
        .admit(RepresentationAdmissionId::new("a1").unwrap(), &first)
        .unwrap();
    let result = ledger.admit(RepresentationAdmissionId::new("a2").unwrap(), &second);
    assert!(matches!(
        result,
        Err(RepresentationAdmissionError::SourceReceiptReplay(_))
    ));
}

#[test]
fn repeated_same_statement_is_persistence_not_new_corroboration() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid(), 1).unwrap();
    let first_source = source("s1", 1);
    let second_source = source("s2", 2);
    let first_id = first_source.id().clone();
    let second_id = second_source.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(first_source).unwrap();
    registry.register(second_source).unwrap();

    let first = qualified(&registry, &continuity, &first_id, "r1", DIGEST_A);
    let second = qualified(&registry, &continuity, &second_id, "r2", DIGEST_A);
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
    assert!(!b.independently_corroborated());
}

#[test]
fn admission_summary_never_promotes_lineage_novelty_to_independence() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid(), 1).unwrap();
    let first_source = source("s1", 1);
    let second_source = source("s2", 2);
    let first_id = first_source.id().clone();
    let second_id = second_source.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(first_source).unwrap();
    registry.register(second_source).unwrap();

    let first = qualified(&registry, &continuity, &first_id, "r1", DIGEST_A);
    let second = qualified(&registry, &continuity, &second_id, "r2", DIGEST_B);
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
