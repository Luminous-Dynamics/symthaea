// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic same-subject graph matching for executor identity.
//!
//! This crate is deliberately **not** an identity verifier. Provider evidence
//! and relation candidates are ordinary inputs. A successful graph match says
//! only that those candidates satisfy a deterministic topology/assurance
//! contract for one exact EXEC-ID challenge.
//!
//! ```text
//! SubjectNodeCandidateV1
//! + SubjectRelationCandidateV1
//! + deterministic graph policy
//!     -> SubjectGraphMatchCandidateV1
//!
//! SubjectGraphMatchCandidateV1
//!     != verified provider evidence
//!     != verifier-owned live relation
//!     != VerifiedExecutorBinding
//!     != execution authority
//! ```

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_executor_identity::{
    ExecutorEvidenceSubjectId, ExecutorIdentityChallenge, ExecutorIdentityDimension,
    ExecutorIdentityRequirement, ExecutorRuntimeIncarnationId, ExecutorVerifierProfileId,
};
use symthaea_interaction_core::Digest32;
use thiserror::Error;

pub const SUBJECT_GRAPH_SCHEMA_VERSION: u16 = 1;

const NODE_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.node.v1\0";
const RELATION_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.relation.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.policy.v1\0";
const NODE_SET_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.node-set.v1\0";
const RELATION_SET_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.relation-set.v1\0";
const MATCH_DOMAIN: &[u8] = b"symthaea.executor.subject-graph.match-candidate.v1\0";

const GRAPH_DIMENSIONS: [ExecutorIdentityDimension; 5] = [
    ExecutorIdentityDimension::SessionPeer,
    ExecutorIdentityDimension::Workload,
    ExecutorIdentityDimension::Software,
    ExecutorIdentityDimension::Device,
    ExecutorIdentityDimension::Embodiment,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubjectRelationClass {
    EndpointWorkload,
    WorkloadSoftware,
    WorkloadDevice,
    DeviceEmbodiment,
}

impl SubjectRelationClass {
    const ALL: [Self; 4] = [
        Self::EndpointWorkload,
        Self::WorkloadSoftware,
        Self::WorkloadDevice,
        Self::DeviceEmbodiment,
    ];

    const fn code(self) -> u16 {
        match self {
            Self::EndpointWorkload => 0,
            Self::WorkloadSoftware => 1,
            Self::WorkloadDevice => 2,
            Self::DeviceEmbodiment => 3,
        }
    }

    const fn endpoint_dimensions(
        self,
    ) -> (ExecutorIdentityDimension, ExecutorIdentityDimension) {
        match self {
            Self::EndpointWorkload => (
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
            ),
            Self::WorkloadSoftware => (
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
            ),
            Self::WorkloadDevice => (
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Device,
            ),
            Self::DeviceEmbodiment => (
                ExecutorIdentityDimension::Device,
                ExecutorIdentityDimension::Embodiment,
            ),
        }
    }
}

/// Evidence strength for one same-subject relation candidate. This is not an
/// authority or consequence lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubjectRelationAssurance {
    DevelopmentObserved,
    AuthenticatedProcess,
    MeasuredProcess,
    HardwareAnchoredProcess,
}

impl SubjectRelationAssurance {
    const fn code(self) -> u16 {
        match self {
            Self::DevelopmentObserved => 0,
            Self::AuthenticatedProcess => 1,
            Self::MeasuredProcess => 2,
            Self::HardwareAnchoredProcess => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubjectNodeIdV1(Digest32);

impl SubjectNodeIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubjectRelationIdV1(Digest32);

impl SubjectRelationIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Ordinary provider-node candidate. Construction computes semantic identity;
/// it does not verify the provider evidence named by the commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectNodeCandidateV1 {
    dimension: ExecutorIdentityDimension,
    subject: ExecutorEvidenceSubjectId,
    challenge: Digest32,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    verifier_profile: ExecutorVerifierProfileId,
    evidence_commitment: Digest32,
    id: SubjectNodeIdV1,
}

impl SubjectNodeCandidateV1 {
    pub fn new(
        dimension: ExecutorIdentityDimension,
        subject: ExecutorEvidenceSubjectId,
        challenge: Digest32,
        runtime_incarnation: ExecutorRuntimeIncarnationId,
        verifier_profile: ExecutorVerifierProfileId,
        evidence_commitment: Digest32,
    ) -> Result<Self, SubjectGraphError> {
        if !is_graph_dimension(dimension) {
            return Err(SubjectGraphError::UnsupportedGraphDimension(dimension));
        }
        reject_zero("node challenge", challenge)?;
        reject_zero("node evidence commitment", evidence_commitment)?;

        let mut transcript = Transcript::new(NODE_DOMAIN);
        transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
        transcript.u16(dimension_code(dimension));
        transcript.digest(subject.digest());
        transcript.digest(challenge);
        transcript.digest(runtime_incarnation.digest());
        transcript.digest(verifier_profile.digest());
        transcript.digest(evidence_commitment);
        let id = SubjectNodeIdV1(transcript.finish());

        Ok(Self {
            dimension,
            subject,
            challenge,
            runtime_incarnation,
            verifier_profile,
            evidence_commitment,
            id,
        })
    }

    pub const fn dimension(&self) -> ExecutorIdentityDimension {
        self.dimension
    }

    pub const fn subject(&self) -> ExecutorEvidenceSubjectId {
        self.subject
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge
    }

    pub const fn runtime_incarnation(&self) -> ExecutorRuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn verifier_profile(&self) -> ExecutorVerifierProfileId {
        self.verifier_profile
    }

    pub const fn evidence_commitment(&self) -> Digest32 {
        self.evidence_commitment
    }

    pub const fn id(&self) -> SubjectNodeIdV1 {
        self.id
    }
}

/// Ordinary relation candidate. It commits the exact endpoint-node identities;
/// equality of subject IDs, hostnames, paths, PIDs, or other caller data never
/// creates a relation implicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectRelationCandidateV1 {
    class: SubjectRelationClass,
    assurance: SubjectRelationAssurance,
    left: SubjectNodeIdV1,
    right: SubjectNodeIdV1,
    challenge: Digest32,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    verifier_profile: ExecutorVerifierProfileId,
    evidence_commitment: Digest32,
    id: SubjectRelationIdV1,
}

impl SubjectRelationCandidateV1 {
    pub fn new(
        class: SubjectRelationClass,
        assurance: SubjectRelationAssurance,
        left: &SubjectNodeCandidateV1,
        right: &SubjectNodeCandidateV1,
        verifier_profile: ExecutorVerifierProfileId,
        evidence_commitment: Digest32,
    ) -> Result<Self, SubjectGraphError> {
        let (expected_left, expected_right) = class.endpoint_dimensions();
        if left.dimension != expected_left || right.dimension != expected_right {
            return Err(SubjectGraphError::InvalidRelationEndpointDimensions { class });
        }
        if left.challenge != right.challenge {
            return Err(SubjectGraphError::RelationEndpointChallengeMismatch { class });
        }
        if left.runtime_incarnation != right.runtime_incarnation {
            return Err(SubjectGraphError::RelationEndpointRuntimeMismatch { class });
        }
        reject_zero("relation evidence commitment", evidence_commitment)?;

        let challenge = left.challenge;
        let runtime_incarnation = left.runtime_incarnation;
        let mut transcript = Transcript::new(RELATION_DOMAIN);
        transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
        transcript.u16(class.code());
        transcript.u16(assurance.code());
        transcript.digest(left.id.digest());
        transcript.digest(right.id.digest());
        transcript.digest(challenge);
        transcript.digest(runtime_incarnation.digest());
        transcript.digest(verifier_profile.digest());
        transcript.digest(evidence_commitment);
        let id = SubjectRelationIdV1(transcript.finish());

        Ok(Self {
            class,
            assurance,
            left: left.id,
            right: right.id,
            challenge,
            runtime_incarnation,
            verifier_profile,
            evidence_commitment,
            id,
        })
    }

    pub const fn class(&self) -> SubjectRelationClass {
        self.class
    }

    pub const fn assurance(&self) -> SubjectRelationAssurance {
        self.assurance
    }

    pub const fn left(&self) -> SubjectNodeIdV1 {
        self.left
    }

    pub const fn right(&self) -> SubjectNodeIdV1 {
        self.right
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge
    }

    pub const fn runtime_incarnation(&self) -> ExecutorRuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn verifier_profile(&self) -> ExecutorVerifierProfileId {
        self.verifier_profile
    }

    pub const fn evidence_commitment(&self) -> Digest32 {
        self.evidence_commitment
    }

    pub const fn id(&self) -> SubjectRelationIdV1 {
        self.id
    }
}

/// Deterministic minimum-assurance policy for the four closed relation classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectGraphRelationPolicyV1 {
    endpoint_workload: SubjectRelationAssurance,
    workload_software: SubjectRelationAssurance,
    workload_device: SubjectRelationAssurance,
    device_embodiment: SubjectRelationAssurance,
}

impl SubjectGraphRelationPolicyV1 {
    pub const fn new(
        endpoint_workload: SubjectRelationAssurance,
        workload_software: SubjectRelationAssurance,
        workload_device: SubjectRelationAssurance,
        device_embodiment: SubjectRelationAssurance,
    ) -> Self {
        Self {
            endpoint_workload,
            workload_software,
            workload_device,
            device_embodiment,
        }
    }

    pub const fn minimum(&self, class: SubjectRelationClass) -> SubjectRelationAssurance {
        match class {
            SubjectRelationClass::EndpointWorkload => self.endpoint_workload,
            SubjectRelationClass::WorkloadSoftware => self.workload_software,
            SubjectRelationClass::WorkloadDevice => self.workload_device,
            SubjectRelationClass::DeviceEmbodiment => self.device_embodiment,
        }
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
        for class in SubjectRelationClass::ALL {
            transcript.u16(class.code());
            transcript.u16(self.minimum(class).code());
        }
        transcript.finish()
    }
}

/// Positive deterministic topology result. This is intentionally an ordinary
/// candidate and has no conversion path to `VerifiedExecutorBinding`.
#[derive(Debug, PartialEq, Eq)]
pub struct SubjectGraphMatchCandidateV1 {
    challenge: Digest32,
    requirement: Digest32,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    relation_policy: Digest32,
    workload_root_subject: ExecutorEvidenceSubjectId,
    workload_root_node: SubjectNodeIdV1,
    node_set: Digest32,
    relation_set: Digest32,
    digest: Digest32,
}

impl SubjectGraphMatchCandidateV1 {
    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge
    }

    pub const fn requirement_digest(&self) -> Digest32 {
        self.requirement
    }

    pub const fn runtime_incarnation(&self) -> ExecutorRuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn relation_policy_digest(&self) -> Digest32 {
        self.relation_policy
    }

    pub const fn workload_root_subject(&self) -> ExecutorEvidenceSubjectId {
        self.workload_root_subject
    }

    pub const fn workload_root_node(&self) -> SubjectNodeIdV1 {
        self.workload_root_node
    }

    pub const fn node_set_digest(&self) -> Digest32 {
        self.node_set
    }

    pub const fn relation_set_digest(&self) -> Digest32 {
        self.relation_set
    }

    pub const fn digest(&self) -> Digest32 {
        self.digest
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SubjectGraphError {
    #[error("{0} must not use an all-zero digest")]
    ZeroDigest(&'static str),
    #[error("dimension {0:?} is not part of the same-live-subject graph")]
    UnsupportedGraphDimension(ExecutorIdentityDimension),
    #[error("relation {class:?} has invalid endpoint dimensions")]
    InvalidRelationEndpointDimensions { class: SubjectRelationClass },
    #[error("relation {class:?} endpoints bind different challenges")]
    RelationEndpointChallengeMismatch { class: SubjectRelationClass },
    #[error("relation {class:?} endpoints bind different runtime incarnations")]
    RelationEndpointRuntimeMismatch { class: SubjectRelationClass },
    #[error("the challenge was created for a different executor identity requirement")]
    ChallengeRequirementMismatch,
    #[error("the selected requirement does not require a same-subject graph")]
    GraphNotRequired,
    #[error("node dimension {0:?} is not required by this graph")]
    UnexpectedNodeDimension(ExecutorIdentityDimension),
    #[error("more than one node was supplied for dimension {0:?}")]
    DuplicateNodeDimension(ExecutorIdentityDimension),
    #[error("required graph node dimension {0:?} is missing")]
    MissingNodeDimension(ExecutorIdentityDimension),
    #[error("node {0:?} belongs to another EXEC-ID challenge")]
    NodeChallengeMismatch(ExecutorIdentityDimension),
    #[error("node {0:?} belongs to another runtime incarnation")]
    NodeRuntimeMismatch(ExecutorIdentityDimension),
    #[error("relation class {0:?} is not required by this graph")]
    UnexpectedRelationClass(SubjectRelationClass),
    #[error("more than one relation was supplied for class {0:?}")]
    DuplicateRelationClass(SubjectRelationClass),
    #[error("required relation class {0:?} is missing")]
    MissingRelationClass(SubjectRelationClass),
    #[error("relation {0:?} belongs to another EXEC-ID challenge")]
    RelationChallengeMismatch(SubjectRelationClass),
    #[error("relation {0:?} belongs to another runtime incarnation")]
    RelationRuntimeMismatch(SubjectRelationClass),
    #[error("relation {0:?} does not connect the exact selected endpoint nodes")]
    RelationEndpointMismatch(SubjectRelationClass),
    #[error("relation {class:?} assurance {actual:?} is below required {required:?}")]
    InsufficientRelationAssurance {
        class: SubjectRelationClass,
        required: SubjectRelationAssurance,
        actual: SubjectRelationAssurance,
    },
    #[error("the required graph is not connected to the unique Workload root")]
    DisconnectedGraph,
}

/// Deterministically assess ordinary same-subject graph candidates.
///
/// This function does not verify provider evidence or promote a match into a
/// live executor binding.
pub fn assess_subject_graph_v1(
    challenge: &ExecutorIdentityChallenge,
    requirement: &ExecutorIdentityRequirement,
    policy: &SubjectGraphRelationPolicyV1,
    nodes: &[SubjectNodeCandidateV1],
    relations: &[SubjectRelationCandidateV1],
) -> Result<SubjectGraphMatchCandidateV1, SubjectGraphError> {
    if challenge.requirement_digest() != requirement.digest() {
        return Err(SubjectGraphError::ChallengeRequirementMismatch);
    }

    let support = support_dimensions(requirement);
    if support.is_empty() {
        return Err(SubjectGraphError::GraphNotRequired);
    }

    let challenge_digest = challenge.digest();
    let runtime_incarnation = challenge.runtime_incarnation();
    let mut selected = BTreeMap::new();

    for node in nodes {
        if !support.contains(&node.dimension) {
            return Err(SubjectGraphError::UnexpectedNodeDimension(node.dimension));
        }
        if node.challenge != challenge_digest {
            return Err(SubjectGraphError::NodeChallengeMismatch(node.dimension));
        }
        if node.runtime_incarnation != runtime_incarnation {
            return Err(SubjectGraphError::NodeRuntimeMismatch(node.dimension));
        }
        if selected.insert(node.dimension, node).is_some() {
            return Err(SubjectGraphError::DuplicateNodeDimension(node.dimension));
        }
    }

    for dimension in &support {
        if !selected.contains_key(dimension) {
            return Err(SubjectGraphError::MissingNodeDimension(*dimension));
        }
    }

    let expected_relations = expected_relation_classes(&support);
    let mut selected_relations = BTreeMap::new();
    for relation in relations {
        if !expected_relations.contains(&relation.class) {
            return Err(SubjectGraphError::UnexpectedRelationClass(relation.class));
        }
        if relation.challenge != challenge_digest {
            return Err(SubjectGraphError::RelationChallengeMismatch(relation.class));
        }
        if relation.runtime_incarnation != runtime_incarnation {
            return Err(SubjectGraphError::RelationRuntimeMismatch(relation.class));
        }
        if selected_relations.insert(relation.class, relation).is_some() {
            return Err(SubjectGraphError::DuplicateRelationClass(relation.class));
        }
    }

    for class in &expected_relations {
        let relation = selected_relations
            .get(class)
            .copied()
            .ok_or(SubjectGraphError::MissingRelationClass(*class))?;
        let (left_dimension, right_dimension) = class.endpoint_dimensions();
        let left = selected
            .get(&left_dimension)
            .copied()
            .ok_or(SubjectGraphError::MissingNodeDimension(left_dimension))?;
        let right = selected
            .get(&right_dimension)
            .copied()
            .ok_or(SubjectGraphError::MissingNodeDimension(right_dimension))?;
        if relation.left != left.id || relation.right != right.id {
            return Err(SubjectGraphError::RelationEndpointMismatch(*class));
        }
        let minimum = policy.minimum(*class);
        if relation.assurance < minimum {
            return Err(SubjectGraphError::InsufficientRelationAssurance {
                class: *class,
                required: minimum,
                actual: relation.assurance,
            });
        }
    }

    let workload = selected
        .get(&ExecutorIdentityDimension::Workload)
        .copied()
        .ok_or(SubjectGraphError::MissingNodeDimension(
            ExecutorIdentityDimension::Workload,
        ))?;

    if !connected_to_workload(&selected, &selected_relations, workload.id) {
        return Err(SubjectGraphError::DisconnectedGraph);
    }

    let node_set = node_set_digest(selected.values().copied());
    let relation_set = relation_set_digest(selected_relations.values().copied());
    let relation_policy = policy.digest();
    let requirement_digest = requirement.digest();

    let mut transcript = Transcript::new(MATCH_DOMAIN);
    transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
    transcript.digest(challenge_digest);
    transcript.digest(requirement_digest);
    transcript.digest(runtime_incarnation.digest());
    transcript.digest(relation_policy);
    transcript.digest(workload.subject.digest());
    transcript.digest(workload.id.digest());
    transcript.digest(node_set);
    transcript.digest(relation_set);
    let digest = transcript.finish();

    Ok(SubjectGraphMatchCandidateV1 {
        challenge: challenge_digest,
        requirement: requirement_digest,
        runtime_incarnation,
        relation_policy,
        workload_root_subject: workload.subject,
        workload_root_node: workload.id,
        node_set,
        relation_set,
        digest,
    })
}

fn support_dimensions(requirement: &ExecutorIdentityRequirement) -> BTreeSet<ExecutorIdentityDimension> {
    let required = requirement.required_dimensions();
    let mut support = BTreeSet::new();
    for dimension in GRAPH_DIMENSIONS {
        if required.contains(dimension) {
            support.insert(dimension);
        }
    }

    // Necessary join support is structural only; it does not add verified
    // dimensions to the executor identity requirement.
    if support.contains(&ExecutorIdentityDimension::SessionPeer)
        || support.contains(&ExecutorIdentityDimension::Software)
        || support.contains(&ExecutorIdentityDimension::Device)
        || support.contains(&ExecutorIdentityDimension::Embodiment)
    {
        support.insert(ExecutorIdentityDimension::Workload);
    }
    if support.contains(&ExecutorIdentityDimension::Embodiment) {
        support.insert(ExecutorIdentityDimension::Device);
    }
    support
}

fn expected_relation_classes(
    support: &BTreeSet<ExecutorIdentityDimension>,
) -> BTreeSet<SubjectRelationClass> {
    let mut expected = BTreeSet::new();
    if support.contains(&ExecutorIdentityDimension::SessionPeer) {
        expected.insert(SubjectRelationClass::EndpointWorkload);
    }
    if support.contains(&ExecutorIdentityDimension::Software) {
        expected.insert(SubjectRelationClass::WorkloadSoftware);
    }
    if support.contains(&ExecutorIdentityDimension::Device) {
        expected.insert(SubjectRelationClass::WorkloadDevice);
    }
    if support.contains(&ExecutorIdentityDimension::Embodiment) {
        expected.insert(SubjectRelationClass::DeviceEmbodiment);
    }
    expected
}

fn connected_to_workload(
    nodes: &BTreeMap<ExecutorIdentityDimension, &SubjectNodeCandidateV1>,
    relations: &BTreeMap<SubjectRelationClass, &SubjectRelationCandidateV1>,
    root: SubjectNodeIdV1,
) -> bool {
    let mut visited = BTreeSet::from([root]);
    loop {
        let before = visited.len();
        for relation in relations.values() {
            if visited.contains(&relation.left) {
                visited.insert(relation.right);
            }
            if visited.contains(&relation.right) {
                visited.insert(relation.left);
            }
        }
        if visited.len() == before {
            break;
        }
    }
    nodes.values().all(|node| visited.contains(&node.id))
}

fn node_set_digest<'a>(nodes: impl Iterator<Item = &'a SubjectNodeCandidateV1>) -> Digest32 {
    let mut ids: Vec<_> = nodes.map(|node| node.id).collect();
    ids.sort();
    let mut transcript = Transcript::new(NODE_SET_DOMAIN);
    transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
    transcript.u32(ids.len() as u32);
    for id in ids {
        transcript.digest(id.digest());
    }
    transcript.finish()
}

fn relation_set_digest<'a>(
    relations: impl Iterator<Item = &'a SubjectRelationCandidateV1>,
) -> Digest32 {
    let mut ids: Vec<_> = relations.map(|relation| relation.id).collect();
    ids.sort();
    let mut transcript = Transcript::new(RELATION_SET_DOMAIN);
    transcript.u16(SUBJECT_GRAPH_SCHEMA_VERSION);
    transcript.u32(ids.len() as u32);
    for id in ids {
        transcript.digest(id.digest());
    }
    transcript.finish()
}

const fn is_graph_dimension(dimension: ExecutorIdentityDimension) -> bool {
    matches!(
        dimension,
        ExecutorIdentityDimension::SessionPeer
            | ExecutorIdentityDimension::Workload
            | ExecutorIdentityDimension::Software
            | ExecutorIdentityDimension::Device
            | ExecutorIdentityDimension::Embodiment
    )
}

const fn dimension_code(dimension: ExecutorIdentityDimension) -> u16 {
    match dimension {
        ExecutorIdentityDimension::SessionPeer => 0,
        ExecutorIdentityDimension::Operator => 1,
        ExecutorIdentityDimension::Workload => 2,
        ExecutorIdentityDimension::Software => 3,
        ExecutorIdentityDimension::Device => 4,
        ExecutorIdentityDimension::Embodiment => 5,
        ExecutorIdentityDimension::ExecutorProfile => 6,
    }
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), SubjectGraphError> {
    if value.as_bytes() == &[0; 32] {
        Err(SubjectGraphError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn finish(self) -> Digest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Digest32::new(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_executor_identity::{
        ExecutorIdentityDimensionSet, ExecutorIdentityProfile, ExecutorProfileId,
    };
    use symthaea_interaction_core::{
        IdentityComponent, IdentityOrdering, NamespaceId, PrincipalRef,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn principal() -> PrincipalRef {
        PrincipalRef::new(
            NamespaceId::new("exec/test").unwrap(),
            "executor",
            IdentityOrdering::NamedSet,
            vec![IdentityComponent::new("id", "alpha").unwrap()],
        )
        .unwrap()
    }

    fn requirement(extra: &[ExecutorIdentityDimension]) -> ExecutorIdentityRequirement {
        let mut dimensions = vec![
            ExecutorIdentityDimension::Workload,
            ExecutorIdentityDimension::Software,
            ExecutorIdentityDimension::ExecutorProfile,
        ];
        dimensions.extend_from_slice(extra);
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&dimensions),
        )
        .unwrap()
    }

    fn challenge(
        requirement: &ExecutorIdentityRequirement,
        runtime_byte: u8,
    ) -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xA5; 32],
            &principal(),
            ExecutorProfileId::new(digest(0x22)).unwrap(),
            ExecutorRuntimeIncarnationId::new(digest(runtime_byte)).unwrap(),
            requirement,
        )
        .unwrap()
    }

    fn node(
        challenge: &ExecutorIdentityChallenge,
        dimension: ExecutorIdentityDimension,
        byte: u8,
    ) -> SubjectNodeCandidateV1 {
        SubjectNodeCandidateV1::new(
            dimension,
            ExecutorEvidenceSubjectId::new(digest(byte)).unwrap(),
            challenge.digest(),
            challenge.runtime_incarnation(),
            ExecutorVerifierProfileId::new(digest(byte.wrapping_add(40))).unwrap(),
            digest(byte.wrapping_add(80)),
        )
        .unwrap()
    }

    fn relation(
        class: SubjectRelationClass,
        assurance: SubjectRelationAssurance,
        left: &SubjectNodeCandidateV1,
        right: &SubjectNodeCandidateV1,
        byte: u8,
    ) -> SubjectRelationCandidateV1 {
        SubjectRelationCandidateV1::new(
            class,
            assurance,
            left,
            right,
            ExecutorVerifierProfileId::new(digest(byte.wrapping_add(100))).unwrap(),
            digest(byte.wrapping_add(120)),
        )
        .unwrap()
    }

    fn policy() -> SubjectGraphRelationPolicyV1 {
        SubjectGraphRelationPolicyV1::new(
            SubjectRelationAssurance::AuthenticatedProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::HardwareAnchoredProcess,
        )
    }

    #[test]
    fn local_workload_software_graph_matches() {
        let requirement = requirement(&[]);
        let challenge = challenge(&requirement, 0x33);
        let workload = node(&challenge, ExecutorIdentityDimension::Workload, 1);
        let software = node(&challenge, ExecutorIdentityDimension::Software, 2);
        let edge = relation(
            SubjectRelationClass::WorkloadSoftware,
            SubjectRelationAssurance::MeasuredProcess,
            &workload,
            &software,
            3,
        );
        let matched = assess_subject_graph_v1(
            &challenge,
            &requirement,
            &policy(),
            &[workload.clone(), software.clone()],
            &[edge],
        )
        .unwrap();
        assert_eq!(matched.workload_root_node(), workload.id());
        assert_eq!(matched.workload_root_subject(), workload.subject());
    }

    #[test]
    fn matching_subject_ids_do_not_create_an_edge() {
        let requirement = requirement(&[]);
        let challenge = challenge(&requirement, 0x33);
        let same_subject = ExecutorEvidenceSubjectId::new(digest(7)).unwrap();
        let workload = SubjectNodeCandidateV1::new(
            ExecutorIdentityDimension::Workload,
            same_subject,
            challenge.digest(),
            challenge.runtime_incarnation(),
            ExecutorVerifierProfileId::new(digest(8)).unwrap(),
            digest(9),
        )
        .unwrap();
        let software = SubjectNodeCandidateV1::new(
            ExecutorIdentityDimension::Software,
            same_subject,
            challenge.digest(),
            challenge.runtime_incarnation(),
            ExecutorVerifierProfileId::new(digest(10)).unwrap(),
            digest(11),
        )
        .unwrap();
        assert_eq!(
            assess_subject_graph_v1(
                &challenge,
                &requirement,
                &policy(),
                &[workload, software],
                &[],
            )
            .unwrap_err(),
            SubjectGraphError::MissingRelationClass(SubjectRelationClass::WorkloadSoftware)
        );
    }

    #[test]
    fn weak_relation_cannot_satisfy_stronger_policy() {
        let requirement = requirement(&[]);
        let challenge = challenge(&requirement, 0x33);
        let workload = node(&challenge, ExecutorIdentityDimension::Workload, 1);
        let software = node(&challenge, ExecutorIdentityDimension::Software, 2);
        let edge = relation(
            SubjectRelationClass::WorkloadSoftware,
            SubjectRelationAssurance::AuthenticatedProcess,
            &workload,
            &software,
            3,
        );
        assert!(matches!(
            assess_subject_graph_v1(
                &challenge,
                &requirement,
                &policy(),
                &[workload, software],
                &[edge],
            ),
            Err(SubjectGraphError::InsufficientRelationAssurance { .. })
        ));
    }

    #[test]
    fn runtime_incarnation_mismatch_fails_closed() {
        let requirement = requirement(&[]);
        let challenge_a = challenge(&requirement, 0x33);
        let challenge_b = challenge(&requirement, 0x34);
        let workload = node(&challenge_a, ExecutorIdentityDimension::Workload, 1);
        let software = node(&challenge_b, ExecutorIdentityDimension::Software, 2);
        assert!(matches!(
            assess_subject_graph_v1(
                &challenge_a,
                &requirement,
                &policy(),
                &[workload, software],
                &[],
            ),
            Err(SubjectGraphError::NodeChallengeMismatch(
                ExecutorIdentityDimension::Software
            ))
        ));
    }

    #[test]
    fn endpoint_requires_explicit_endpoint_workload_relation() {
        let requirement = requirement(&[ExecutorIdentityDimension::SessionPeer]);
        let challenge = challenge(&requirement, 0x33);
        let endpoint = node(&challenge, ExecutorIdentityDimension::SessionPeer, 1);
        let workload = node(&challenge, ExecutorIdentityDimension::Workload, 2);
        let software = node(&challenge, ExecutorIdentityDimension::Software, 3);
        let software_edge = relation(
            SubjectRelationClass::WorkloadSoftware,
            SubjectRelationAssurance::MeasuredProcess,
            &workload,
            &software,
            4,
        );
        assert_eq!(
            assess_subject_graph_v1(
                &challenge,
                &requirement,
                &policy(),
                &[endpoint, workload, software],
                &[software_edge],
            )
            .unwrap_err(),
            SubjectGraphError::MissingRelationClass(SubjectRelationClass::EndpointWorkload)
        );
    }

    #[test]
    fn insertion_order_is_non_semantic() {
        let requirement = requirement(&[]);
        let challenge = challenge(&requirement, 0x33);
        let workload = node(&challenge, ExecutorIdentityDimension::Workload, 1);
        let software = node(&challenge, ExecutorIdentityDimension::Software, 2);
        let edge = relation(
            SubjectRelationClass::WorkloadSoftware,
            SubjectRelationAssurance::MeasuredProcess,
            &workload,
            &software,
            3,
        );
        let a = assess_subject_graph_v1(
            &challenge,
            &requirement,
            &policy(),
            &[workload.clone(), software.clone()],
            std::slice::from_ref(&edge),
        )
        .unwrap();
        let b = assess_subject_graph_v1(
            &challenge,
            &requirement,
            &policy(),
            &[software, workload],
            &[edge],
        )
        .unwrap();
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn unrelated_extra_node_is_rejected_not_ignored() {
        let requirement = requirement(&[]);
        let challenge = challenge(&requirement, 0x33);
        let workload = node(&challenge, ExecutorIdentityDimension::Workload, 1);
        let software = node(&challenge, ExecutorIdentityDimension::Software, 2);
        let device = node(&challenge, ExecutorIdentityDimension::Device, 3);
        assert_eq!(
            assess_subject_graph_v1(
                &challenge,
                &requirement,
                &policy(),
                &[workload, software, device],
                &[],
            )
            .unwrap_err(),
            SubjectGraphError::UnexpectedNodeDimension(ExecutorIdentityDimension::Device)
        );
    }

    #[test]
    fn development_profile_has_no_same_subject_graph_requirement() {
        let requirement = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::DevelopmentSimulation,
            ExecutorIdentityDimensionSet::new(&[ExecutorIdentityDimension::ExecutorProfile]),
        )
        .unwrap();
        let challenge = challenge(&requirement, 0x33);
        assert_eq!(
            assess_subject_graph_v1(&challenge, &requirement, &policy(), &[], &[]).unwrap_err(),
            SubjectGraphError::GraphNotRequired
        );
    }
}
