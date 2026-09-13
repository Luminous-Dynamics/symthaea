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

use continuity_identity::{
    ContinuityEvent, ContinuityEventId, ContinuityIdentityLedger, ContinuityKind,
    SubjectInstanceId,
};
use reciprocal_representation::{
    RepresentationId, RepresentationKind, RepresentationScope,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole, RepresentationProvenanceError,
    RepresentationProvenanceRegistry, RepresentationSourceReceipt,
    RepresentationSourceReceiptId,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sid(value: &str) -> SubjectInstanceId {
    SubjectInstanceId::new(value).unwrap()
}

fn source(
    id: &str,
    subject: SubjectInstanceId,
    revision: u64,
) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        format!("lineage-{id}"),
        revision,
        DIGEST,
        format!("origin://{id}"),
    )
    .unwrap()
}

fn qualify(
    registry: &RepresentationProvenanceRegistry,
    receipt_id: &RepresentationSourceReceiptId,
    continuity: &ContinuityIdentityLedger,
    representation_id: &str,
) -> Result<reciprocal_representation_provenance::QualifiedReciprocalRepresentation, RepresentationProvenanceError> {
    registry.qualify_live(
        receipt_id,
        continuity,
        RepresentationId::new(representation_id).unwrap(),
        RepresentationKind::RequestForReview,
        RepresentationScope::general_research(),
        DIGEST,
        format!("statement://{representation_id}"),
        0.9,
        None,
    )
}

#[test]
fn root_source_receipt_cannot_predate_registered_creation() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 10).unwrap();
    assert_eq!(continuity.created_revision(&sid("root")).unwrap(), 10);

    let receipt = source("root-backdated", sid("root"), 9);
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    assert_eq!(
        qualify(&registry, &receipt_id, &continuity, "root-report").unwrap_err(),
        RepresentationProvenanceError::SourceReceiptPredatesSubject {
            subject_instance: sid("root"),
            subject_created_revision: 10,
            receipt_revision: 9,
        }
    );
}

#[test]
fn fork_child_cannot_claim_source_receipt_from_before_its_creation() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity
        .record(
            ContinuityEvent::new(
                ContinuityEventId::new("fork").unwrap(),
                sid("root"),
                vec![sid("child-a"), sid("child-b")],
                ContinuityKind::Fork,
                1,
                20,
                false,
                Some(DIGEST.into()),
                true,
                ["receipt://fork".into()],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(continuity.created_revision(&sid("child-a")).unwrap(), 20);

    let receipt = source("child-backdated", sid("child-a"), 19);
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    assert_eq!(
        qualify(&registry, &receipt_id, &continuity, "child-report").unwrap_err(),
        RepresentationProvenanceError::SourceReceiptPredatesSubject {
            subject_instance: sid("child-a"),
            subject_created_revision: 20,
            receipt_revision: 19,
        }
    );
}

#[test]
fn source_receipt_at_creation_boundary_is_valid() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 10).unwrap();

    let receipt = source("boundary", sid("root"), 10);
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    let qualified = qualify(&registry, &receipt_id, &continuity, "boundary-report").unwrap();
    assert_eq!(qualified.representation().logical_revision(), 10);
    assert!(!qualified.establishes_moral_patienthood());
    assert!(!qualified.grants_self_preservation_authority());
}

#[test]
fn active_subject_may_emit_much_later_without_fake_continuity_event() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("active"), 5).unwrap();

    let receipt = source("later", sid("active"), 1_000);
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();

    let qualified = qualify(&registry, &receipt_id, &continuity, "later-report").unwrap();
    assert_eq!(qualified.representation().logical_revision(), 1_000);
}
