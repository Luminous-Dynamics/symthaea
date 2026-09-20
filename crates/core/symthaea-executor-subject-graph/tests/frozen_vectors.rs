use symthaea_executor_identity::{
    ExecutorEvidenceSubjectId, ExecutorIdentityDimension, ExecutorRuntimeIncarnationId,
    ExecutorVerifierProfileId,
};
use symthaea_executor_subject_graph::{
    SubjectGraphRelationPolicyV1, SubjectNodeCandidateV1, SubjectRelationAssurance,
    SubjectRelationCandidateV1, SubjectRelationClass,
};
use symthaea_interaction_core::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

#[test]
fn public_canonical_vectors_match_independent_oracle() {
    let challenge = digest(0x22);
    let runtime = ExecutorRuntimeIncarnationId::new(digest(0x33)).unwrap();

    let workload = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Workload,
        ExecutorEvidenceSubjectId::new(digest(0x11)).unwrap(),
        challenge,
        runtime,
        ExecutorVerifierProfileId::new(digest(0x44)).unwrap(),
        digest(0x55),
    )
    .unwrap();
    assert_eq!(
        workload.id().digest().to_hex(),
        "04d28d04dcc1d9b9b7f5a59572fe17dc7f648f592c2eb18720af5f5f282f0455"
    );

    let software = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Software,
        ExecutorEvidenceSubjectId::new(digest(0x66)).unwrap(),
        challenge,
        runtime,
        ExecutorVerifierProfileId::new(digest(0x77)).unwrap(),
        digest(0x88),
    )
    .unwrap();
    assert_eq!(
        software.id().digest().to_hex(),
        "5629ba7839e08f4e9d1cfb03f4c3c007749698a8b2dc7f45afd7a0f83986bfa3"
    );

    let relation = SubjectRelationCandidateV1::new(
        SubjectRelationClass::WorkloadSoftware,
        SubjectRelationAssurance::MeasuredProcess,
        &workload,
        &software,
        ExecutorVerifierProfileId::new(digest(0x99)).unwrap(),
        digest(0xAA),
    )
    .unwrap();
    assert_eq!(
        relation.id().digest().to_hex(),
        "fd1176cf74330b36a23da8eb43c259631e9689f29bc194e8ed0b0ee486a7a554"
    );

    let policy = SubjectGraphRelationPolicyV1::new(
        SubjectRelationAssurance::AuthenticatedProcess,
        SubjectRelationAssurance::MeasuredProcess,
        SubjectRelationAssurance::MeasuredProcess,
        SubjectRelationAssurance::HardwareAnchoredProcess,
    );
    assert_eq!(
        policy.digest().to_hex(),
        "5eeb222be4f932efb6071e7b44ebe8c3226797ac1232a43690c0fca712fbe1d0"
    );
}

#[test]
fn evidence_mutation_changes_node_and_relation_identity() {
    let challenge = digest(0x22);
    let runtime = ExecutorRuntimeIncarnationId::new(digest(0x33)).unwrap();
    let workload = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Workload,
        ExecutorEvidenceSubjectId::new(digest(0x11)).unwrap(),
        challenge,
        runtime,
        ExecutorVerifierProfileId::new(digest(0x44)).unwrap(),
        digest(0x55),
    )
    .unwrap();
    let workload_changed = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Workload,
        ExecutorEvidenceSubjectId::new(digest(0x11)).unwrap(),
        challenge,
        runtime,
        ExecutorVerifierProfileId::new(digest(0x44)).unwrap(),
        digest(0x56),
    )
    .unwrap();
    assert_ne!(workload.id(), workload_changed.id());

    let software = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Software,
        ExecutorEvidenceSubjectId::new(digest(0x66)).unwrap(),
        challenge,
        runtime,
        ExecutorVerifierProfileId::new(digest(0x77)).unwrap(),
        digest(0x88),
    )
    .unwrap();
    let relation = SubjectRelationCandidateV1::new(
        SubjectRelationClass::WorkloadSoftware,
        SubjectRelationAssurance::MeasuredProcess,
        &workload,
        &software,
        ExecutorVerifierProfileId::new(digest(0x99)).unwrap(),
        digest(0xAA),
    )
    .unwrap();
    let relation_changed = SubjectRelationCandidateV1::new(
        SubjectRelationClass::WorkloadSoftware,
        SubjectRelationAssurance::MeasuredProcess,
        &workload,
        &software,
        ExecutorVerifierProfileId::new(digest(0x99)).unwrap(),
        digest(0xAB),
    )
    .unwrap();
    assert_ne!(relation.id(), relation_changed.id());
}
