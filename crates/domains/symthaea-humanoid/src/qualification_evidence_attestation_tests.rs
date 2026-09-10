// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::evidence_digest::HumanoidEvidenceDigest;
use crate::morphology::HumanoidMorphology;
use crate::qualification::HumanoidQualificationSubject;
use crate::qualification_evidence_attestation::{
    HumanoidQualificationEvidenceAuthentication, HumanoidQualificationEvidenceKind,
    HumanoidUnsignedQualificationEvidenceClaim,
};
use crate::types::{ActuationMode, HumanoidTask};

fn subject(backend: &str) -> HumanoidQualificationSubject {
    HumanoidQualificationSubject::new(
        HumanoidMorphology::Dexterous53,
        HumanoidTask::Grasp,
        ActuationMode::NormalizedTorque,
        backend,
    )
}

fn unsigned(scheme: &str) -> HumanoidUnsignedQualificationEvidenceClaim {
    HumanoidUnsignedQualificationEvidenceClaim::new(
        subject("grasp-qualification-attestation-test"),
        HumanoidQualificationEvidenceKind::GraspMeasuredControllerTrial,
        HumanoidEvidenceDigest::from_bytes([1; 32]),
        HumanoidEvidenceDigest::from_bytes([2; 32]),
        "reference-recorder-v1",
        HumanoidEvidenceDigest::from_bytes([3; 32]),
        "recorder-key-1",
        7,
        1_000,
        2_000,
        scheme,
    )
    .unwrap()
}

#[test]
fn attached_signature_preserves_prepared_statement_digest() {
    let unsigned = unsigned("ml-dsa-87");
    let prepared = unsigned.statement_digest();
    let claim = unsigned.attach_signature(vec![0xA5; 96]).unwrap();

    assert_eq!(claim.statement_digest(), prepared);
    assert_eq!(claim.authentication().scheme_id(), "ml-dsa-87");
    assert_eq!(claim.authentication().signature(), &[0xA5; 96]);
}

#[test]
fn authentication_scheme_is_part_of_the_signed_statement() {
    let ml_dsa = unsigned("ml-dsa-87");
    let test_scheme = unsigned("test-ed25519-v1");

    assert_ne!(ml_dsa.statement_digest(), test_scheme.statement_digest());
}

#[test]
fn claim_is_bound_to_exact_qualification_subject() {
    let claim = unsigned("ml-dsa-87")
        .attach_signature(vec![0x5A; 64])
        .unwrap();

    assert!(claim.validate_for(&subject("grasp-qualification-attestation-test")));
    assert!(!claim.validate_for(&subject("different-backend")));
}

#[test]
fn empty_authentication_is_rejected() {
    assert!(HumanoidQualificationEvidenceAuthentication::new("ml-dsa-87", Vec::new()).is_none());
}
