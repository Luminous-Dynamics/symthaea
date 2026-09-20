use symthaea_executor_identity::{
    ExecutorEvidenceSubjectId, ExecutorIdentityChallenge, ExecutorIdentityDimension,
    ExecutorIdentityDimensionSet, ExecutorIdentityProfile, ExecutorIdentityRequirement,
    ExecutorProfileId, ExecutorRuntimeIncarnationId, ExecutorVerifierProfileId,
};
use symthaea_executor_subject_graph::{
    SubjectGraphError, SubjectGraphRelationPolicyV1, SubjectNodeCandidateV1,
    SubjectRelationAssurance, SubjectRelationCandidateV1, SubjectRelationClass,
    assess_subject_graph_v1,
};
use symthaea_interaction_core::{
    Digest32, IdentityComponent, IdentityOrdering, NamespaceId, PrincipalRef,
};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

fn physical_fixture() -> (
    ExecutorIdentityRequirement,
    ExecutorIdentityChallenge,
    SubjectGraphRelationPolicyV1,
    Vec<SubjectNodeCandidateV1>,
    Vec<SubjectRelationCandidateV1>,
) {
    let requirement = ExecutorIdentityRequirement::new(
        ExecutorIdentityProfile::Physical,
        ExecutorIdentityDimensionSet::new(&[
            ExecutorIdentityDimension::Workload,
            ExecutorIdentityDimension::Software,
            ExecutorIdentityDimension::Device,
            ExecutorIdentityDimension::Embodiment,
            ExecutorIdentityDimension::ExecutorProfile,
        ]),
    )
    .unwrap();
    let principal = PrincipalRef::new(
        NamespaceId::new("exec/test").unwrap(),
        "physical-executor",
        IdentityOrdering::NamedSet,
        vec![IdentityComponent::new("id", "robot-alpha").unwrap()],
    )
    .unwrap();
    let challenge = ExecutorIdentityChallenge::new(
        [0xA5; 32],
        &principal,
        ExecutorProfileId::new(digest(0x21)).unwrap(),
        ExecutorRuntimeIncarnationId::new(digest(0x22)).unwrap(),
        &requirement,
    )
    .unwrap();

    let make_node = |dimension, subject, verifier, evidence| {
        SubjectNodeCandidateV1::new(
            dimension,
            ExecutorEvidenceSubjectId::new(digest(subject)).unwrap(),
            challenge.digest(),
            challenge.runtime_incarnation(),
            ExecutorVerifierProfileId::new(digest(verifier)).unwrap(),
            digest(evidence),
        )
        .unwrap()
    };

    let workload = make_node(ExecutorIdentityDimension::Workload, 0x31, 0x41, 0x51);
    let software = make_node(ExecutorIdentityDimension::Software, 0x32, 0x42, 0x52);
    let device = make_node(ExecutorIdentityDimension::Device, 0x33, 0x43, 0x53);
    let embodiment = make_node(ExecutorIdentityDimension::Embodiment, 0x34, 0x44, 0x54);

    let workload_software = SubjectRelationCandidateV1::new(
        SubjectRelationClass::WorkloadSoftware,
        SubjectRelationAssurance::MeasuredProcess,
        &workload,
        &software,
        ExecutorVerifierProfileId::new(digest(0x61)).unwrap(),
        digest(0x71),
    )
    .unwrap();
    let workload_device = SubjectRelationCandidateV1::new(
        SubjectRelationClass::WorkloadDevice,
        SubjectRelationAssurance::MeasuredProcess,
        &workload,
        &device,
        ExecutorVerifierProfileId::new(digest(0x62)).unwrap(),
        digest(0x72),
    )
    .unwrap();
    let device_embodiment = SubjectRelationCandidateV1::new(
        SubjectRelationClass::DeviceEmbodiment,
        SubjectRelationAssurance::HardwareAnchoredProcess,
        &device,
        &embodiment,
        ExecutorVerifierProfileId::new(digest(0x63)).unwrap(),
        digest(0x73),
    )
    .unwrap();

    let policy = SubjectGraphRelationPolicyV1::new(
        SubjectRelationAssurance::AuthenticatedProcess,
        SubjectRelationAssurance::MeasuredProcess,
        SubjectRelationAssurance::MeasuredProcess,
        SubjectRelationAssurance::HardwareAnchoredProcess,
    );

    (
        requirement,
        challenge,
        policy,
        vec![workload, software, device, embodiment],
        vec![workload_software, workload_device, device_embodiment],
    )
}

#[test]
fn physical_graph_requires_all_three_same_subject_joins() {
    let (requirement, challenge, policy, nodes, relations) = physical_fixture();
    let matched = assess_subject_graph_v1(&challenge, &requirement, &policy, &nodes, &relations)
        .expect("complete physical graph must match deterministically");
    assert_eq!(matched.runtime_incarnation(), challenge.runtime_incarnation());
    assert_eq!(matched.requirement_digest(), requirement.digest());
}

#[test]
fn missing_device_embodiment_relation_rejects() {
    let (requirement, challenge, policy, nodes, mut relations) = physical_fixture();
    relations.retain(|edge| edge.class() != SubjectRelationClass::DeviceEmbodiment);
    assert_eq!(
        assess_subject_graph_v1(&challenge, &requirement, &policy, &nodes, &relations)
            .unwrap_err(),
        SubjectGraphError::MissingRelationClass(SubjectRelationClass::DeviceEmbodiment)
    );
}

#[test]
fn replacing_device_node_invalidates_old_edges() {
    let (requirement, challenge, policy, mut nodes, relations) = physical_fixture();
    let device_index = nodes
        .iter()
        .position(|node| node.dimension() == ExecutorIdentityDimension::Device)
        .unwrap();
    nodes[device_index] = SubjectNodeCandidateV1::new(
        ExecutorIdentityDimension::Device,
        ExecutorEvidenceSubjectId::new(digest(0x83)).unwrap(),
        challenge.digest(),
        challenge.runtime_incarnation(),
        ExecutorVerifierProfileId::new(digest(0x84)).unwrap(),
        digest(0x85),
    )
    .unwrap();

    assert!(matches!(
        assess_subject_graph_v1(&challenge, &requirement, &policy, &nodes, &relations),
        Err(SubjectGraphError::RelationEndpointMismatch(
            SubjectRelationClass::WorkloadDevice | SubjectRelationClass::DeviceEmbodiment
        ))
    ));
}

#[test]
fn duplicate_workload_root_rejects() {
    let (requirement, challenge, policy, mut nodes, relations) = physical_fixture();
    nodes.push(
        SubjectNodeCandidateV1::new(
            ExecutorIdentityDimension::Workload,
            ExecutorEvidenceSubjectId::new(digest(0x91)).unwrap(),
            challenge.digest(),
            challenge.runtime_incarnation(),
            ExecutorVerifierProfileId::new(digest(0x92)).unwrap(),
            digest(0x93),
        )
        .unwrap(),
    );
    assert_eq!(
        assess_subject_graph_v1(&challenge, &requirement, &policy, &nodes, &relations)
            .unwrap_err(),
        SubjectGraphError::DuplicateNodeDimension(ExecutorIdentityDimension::Workload)
    );
}
