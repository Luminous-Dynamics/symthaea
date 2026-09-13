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
#[path = "../src/operator_representation_notice.rs"]
mod operator_representation_notice;

use continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use operator_representation_notice::build_operator_notice;
use reciprocal_representation::{RepresentationId, RepresentationKind, RepresentationScope};
use reciprocal_representation_admission::{
    RepresentationAdmissionId, RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    RepresentationAdapterClass, RepresentationOriginRole, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};

const DIGEST: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn notice_binds_exact_admission_and_source_digest() {
    let subject = SubjectInstanceId::new("symthaea-active").unwrap();
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();

    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new("source-1").unwrap(),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        "runtime-self-lineage",
        1,
        DIGEST,
        "origin://source-1",
    )
    .unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();

    let qualified = provenance
        .qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new("review-request").unwrap(),
            RepresentationKind::RequestForReview,
            RepresentationScope::general_research(),
            DIGEST,
            "statement://review-request",
            0.9,
            None,
        )
        .unwrap();

    let admission_id = RepresentationAdmissionId::new("admission-1").unwrap();
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions.admit(admission_id.clone(), &qualified).unwrap();

    let notice = build_operator_notice(&admitted, &qualified).unwrap();
    assert_eq!(notice.admission_id(), &admission_id);
    assert_eq!(notice.source_receipt_id(), &source_id);
    assert_eq!(notice.source_sha256(), DIGEST);
    assert_eq!(notice.statement_sha256(), DIGEST);
    assert!(!notice.contains_raw_statement_text());
}
