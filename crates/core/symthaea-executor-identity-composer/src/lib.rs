// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trusted ownership kernel for live executor identity.
//!
//! This crate puts the opaque live binding in the same Rust crate as the
//! future concrete composition logic that is allowed to construct it. Rust has
//! no friend-crate privacy, so the live type deliberately does not live in the
//! neutral `symthaea-executor-identity` ABI.
//!
//! D3B stops before provider-positive production verification:
//!
//! ```text
//! ExecutorIdentityChallenge
//! + ExecutorCompositionPolicyV1
//! + future concrete provider verification
//! + future internal deterministic graph assessment
//!     -> VerifiedExecutorBinding
//!
//! D3B ownership kernel alone
//!     != provider verification
//!     != live composition
//!     != authority
//!     != DispatchPermit
//! ```
//!
//! `VerifiedExecutorBinding` has private fields, no public constructor, no
//! Clone/Copy, and no serialization path. Future concrete verifier/composer
//! modules must remain inside this crate's trusted implementation boundary if
//! they need to construct live state.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use symthaea_executor_identity::{
    ExecutorEvidenceSubjectId, ExecutorIdentityChallenge, ExecutorIdentityDimension,
    ExecutorIdentityDimensionSet, ExecutorIdentityProfile, ExecutorIdentityRequirement,
    ExecutorProfileId, ExecutorRuntimeIncarnationId,
};
use symthaea_executor_subject_graph::SubjectGraphRelationPolicyV1;
use symthaea_interaction_core::Digest32;
use thiserror::Error;

pub const EXECUTOR_COMPOSER_SCHEMA_VERSION: u16 = 1;

const COMPOSITION_POLICY_DOMAIN: &[u8] = b"symthaea.executor.identity.composer.policy.v1\0";
const BINDING_DOMAIN: &[u8] = b"symthaea.executor.identity.composer.binding.v1\0";

const ALL_DIMENSIONS: [ExecutorIdentityDimension; 7] = [
    ExecutorIdentityDimension::SessionPeer,
    ExecutorIdentityDimension::Operator,
    ExecutorIdentityDimension::Workload,
    ExecutorIdentityDimension::Software,
    ExecutorIdentityDimension::Device,
    ExecutorIdentityDimension::Embodiment,
    ExecutorIdentityDimension::ExecutorProfile,
];

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExecutorComposerError {
    #[error("{0} must not use an all-zero digest")]
    ZeroDigest(&'static str),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExecutorBindingMismatch {
    #[error("live executor binding belongs to a different composition challenge")]
    ChallengeMismatch,
    #[error("the supplied challenge was created for a different identity requirement")]
    RequirementMismatch,
    #[error("composition policy was created for a different identity requirement")]
    CompositionPolicyRequirementMismatch,
    #[error("live executor binding was produced under a different composition policy")]
    CompositionPolicyMismatch,
    #[error("live executor binding names a different semantic principal")]
    PrincipalMismatch,
    #[error("live executor binding names a different executor-profile identity")]
    ExecutorProfileMismatch,
    #[error("live executor binding names a different runtime incarnation")]
    RuntimeIncarnationMismatch,
    #[error("identity profile mismatch: expected {expected:?}, got {actual:?}")]
    IdentityProfileMismatch {
        expected: ExecutorIdentityProfile,
        actual: ExecutorIdentityProfile,
    },
    #[error("live executor binding lacks required dimension {0:?}")]
    MissingRequiredDimension(ExecutorIdentityDimension),
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), ExecutorComposerError> {
    if value.as_bytes() == &[0; 32] {
        Err(ExecutorComposerError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

/// Semantic identity of one exact executor-composition policy.
///
/// This identifies configured policy only; it is not a verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutorCompositionPolicyId(Digest32);

impl ExecutorCompositionPolicyId {
    const fn from_digest(value: Digest32) -> Self {
        Self(value)
    }

    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Semantic identity of verifier-owned appraisal/currentness context.
///
/// Constructing this wrapper identifies context bytes only. It does not make
/// those bytes trusted or current. Only the private live-binding construction
/// boundary may give the identity verified meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutorVerificationContextId(Digest32);

impl ExecutorVerificationContextId {
    pub fn new(value: Digest32) -> Result<Self, ExecutorComposerError> {
        reject_zero("executor verification context identity", value)?;
        Ok(Self(value))
    }

    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Ordinary configured composition policy.
///
/// This value binds the exact neutral identity requirement, exact D3A relation
/// policy, and one configuration-owned provider-policy commitment. It is policy
/// identity only and cannot create live verified state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorCompositionPolicyV1 {
    requirement: Digest32,
    relation_policy: Digest32,
    provider_policy_commitment: Digest32,
    id: ExecutorCompositionPolicyId,
}

impl ExecutorCompositionPolicyV1 {
    pub fn new(
        requirement: &ExecutorIdentityRequirement,
        relation_policy: &SubjectGraphRelationPolicyV1,
        provider_policy_commitment: Digest32,
    ) -> Result<Self, ExecutorComposerError> {
        reject_zero("provider policy commitment", provider_policy_commitment)?;

        let requirement_digest = requirement.digest();
        let relation_policy_digest = relation_policy.digest();
        let mut transcript = Transcript::new(COMPOSITION_POLICY_DOMAIN);
        transcript.u16(EXECUTOR_COMPOSER_SCHEMA_VERSION);
        transcript.digest(requirement_digest);
        transcript.digest(relation_policy_digest);
        transcript.digest(provider_policy_commitment);
        let id = ExecutorCompositionPolicyId::from_digest(transcript.finish());

        Ok(Self {
            requirement: requirement_digest,
            relation_policy: relation_policy_digest,
            provider_policy_commitment,
            id,
        })
    }

    pub const fn requirement_digest(&self) -> Digest32 {
        self.requirement
    }

    pub const fn relation_policy_digest(&self) -> Digest32 {
        self.relation_policy
    }

    pub const fn provider_policy_commitment(&self) -> Digest32 {
        self.provider_policy_commitment
    }

    pub const fn id(&self) -> ExecutorCompositionPolicyId {
        self.id
    }
}

/// Composer-owned opaque live executor identity.
///
/// There is intentionally no public constructor and no public `from_*`
/// function. Future concrete provider-verification/composition modules inside
/// this crate may construct the private fields only after their own trust,
/// appraisal, currentness, and same-live-subject theorems are satisfied.
///
/// This type is intentionally move-only and non-serializable.
#[derive(Debug, PartialEq, Eq)]
pub struct VerifiedExecutorBinding {
    challenge: Digest32,
    principal: Digest32,
    profile: ExecutorIdentityProfile,
    executor_profile: ExecutorProfileId,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    root_subject: ExecutorEvidenceSubjectId,
    verified_dimensions: ExecutorIdentityDimensionSet,
    graph_match: Digest32,
    composition_policy: ExecutorCompositionPolicyId,
    evidence_commitment: Digest32,
    verification_context: ExecutorVerificationContextId,
}

impl VerifiedExecutorBinding {
    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge
    }

    pub const fn principal_digest(&self) -> Digest32 {
        self.principal
    }

    pub const fn profile(&self) -> ExecutorIdentityProfile {
        self.profile
    }

    pub const fn executor_profile(&self) -> ExecutorProfileId {
        self.executor_profile
    }

    pub const fn runtime_incarnation(&self) -> ExecutorRuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn root_subject(&self) -> ExecutorEvidenceSubjectId {
        self.root_subject
    }

    pub const fn verified_dimensions(&self) -> ExecutorIdentityDimensionSet {
        self.verified_dimensions
    }

    /// Deterministic D3A graph-match commitment used by the trusted composer.
    /// The digest is audit identity only; callers cannot use it to mint this
    /// live type.
    pub const fn graph_match_digest(&self) -> Digest32 {
        self.graph_match
    }

    pub const fn composition_policy(&self) -> ExecutorCompositionPolicyId {
        self.composition_policy
    }

    pub const fn evidence_commitment(&self) -> Digest32 {
        self.evidence_commitment
    }

    pub const fn verification_context(&self) -> ExecutorVerificationContextId {
        self.verification_context
    }

    /// Structural comparison only.
    ///
    /// Success means this already-live composer-owned binding names the exact
    /// challenge, requirement, principal, profile, runtime, composition policy,
    /// and every required identity dimension. It does not independently prove
    /// provider trust/currentness or create execution authority.
    pub fn structural_match(
        &self,
        challenge: &ExecutorIdentityChallenge,
        requirement: &ExecutorIdentityRequirement,
        policy: &ExecutorCompositionPolicyV1,
    ) -> Result<(), ExecutorBindingMismatch> {
        if self.challenge != challenge.digest() {
            return Err(ExecutorBindingMismatch::ChallengeMismatch);
        }
        if challenge.requirement_digest() != requirement.digest() {
            return Err(ExecutorBindingMismatch::RequirementMismatch);
        }
        if policy.requirement_digest() != requirement.digest() {
            return Err(ExecutorBindingMismatch::CompositionPolicyRequirementMismatch);
        }
        if self.composition_policy != policy.id() {
            return Err(ExecutorBindingMismatch::CompositionPolicyMismatch);
        }
        if self.principal != challenge.principal_digest() {
            return Err(ExecutorBindingMismatch::PrincipalMismatch);
        }
        if self.executor_profile != challenge.executor_profile() {
            return Err(ExecutorBindingMismatch::ExecutorProfileMismatch);
        }
        if self.runtime_incarnation != challenge.runtime_incarnation() {
            return Err(ExecutorBindingMismatch::RuntimeIncarnationMismatch);
        }
        if self.profile != requirement.profile() {
            return Err(ExecutorBindingMismatch::IdentityProfileMismatch {
                expected: requirement.profile(),
                actual: self.profile,
            });
        }
        for dimension in ALL_DIMENSIONS {
            if requirement.required_dimensions().contains(dimension)
                && !self.verified_dimensions.contains(dimension)
            {
                return Err(ExecutorBindingMismatch::MissingRequiredDimension(dimension));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(BINDING_DOMAIN);
        transcript.u16(EXECUTOR_COMPOSER_SCHEMA_VERSION);
        transcript.digest(self.challenge);
        transcript.digest(self.principal);
        transcript.u16(profile_code(self.profile));
        transcript.digest(self.executor_profile.digest());
        transcript.digest(self.runtime_incarnation.digest());
        transcript.digest(self.root_subject.digest());
        transcript.dimension_set(self.verified_dimensions);
        transcript.digest(self.graph_match);
        transcript.digest(self.composition_policy.digest());
        transcript.digest(self.evidence_commitment);
        transcript.digest(self.verification_context.digest());
        transcript.finish()
    }
}

const fn profile_code(profile: ExecutorIdentityProfile) -> u16 {
    match profile {
        ExecutorIdentityProfile::DevelopmentSimulation => 0,
        ExecutorIdentityProfile::LocalDigital => 1,
        ExecutorIdentityProfile::ConsequentialDigital => 2,
        ExecutorIdentityProfile::Physical => 3,
    }
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

    fn dimension_set(&mut self, dimensions: ExecutorIdentityDimensionSet) {
        let count = ALL_DIMENSIONS
            .iter()
            .filter(|dimension| dimensions.contains(**dimension))
            .count();
        self.u32(count as u32);
        for dimension in ALL_DIMENSIONS {
            if dimensions.contains(dimension) {
                self.u16(dimension_code(dimension));
            }
        }
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
    use symthaea_executor_subject_graph::SubjectRelationAssurance;
    use symthaea_interaction_core::{
        IdentityComponent, IdentityOrdering, NamespaceId, PrincipalRef,
    };

    const EXPECTED_POLICY: &str =
        "1787c698b21e79a45dffbe09cd8ffcb6ea409924a668022acefc442ace60db09";
    const EXPECTED_BINDING: &str =
        "19112f42350e223d34e7d0d4bf46be23783e929877850ceee2873b9578ab8223";

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn principal_named(name: &str) -> PrincipalRef {
        PrincipalRef::new(
            NamespaceId::new("intx/workload").expect("namespace"),
            "policy-pdp",
            IdentityOrdering::NamedSet,
            vec![IdentityComponent::new("name", name).expect("component")],
        )
        .expect("principal")
    }

    fn principal() -> PrincipalRef {
        principal_named("pdp-1")
    }

    fn requirement() -> ExecutorIdentityRequirement {
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect("requirement")
    }

    fn local_requirement() -> ExecutorIdentityRequirement {
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::LocalDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect("local requirement")
    }

    fn challenge_for(
        principal: &PrincipalRef,
        requirement: &ExecutorIdentityRequirement,
        profile_byte: u8,
        runtime_byte: u8,
    ) -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xA3; 32],
            principal,
            ExecutorProfileId::new(digest(profile_byte)).expect("profile"),
            ExecutorRuntimeIncarnationId::new(digest(runtime_byte)).expect("runtime"),
            requirement,
        )
        .expect("challenge")
    }

    fn challenge(
        requirement: &ExecutorIdentityRequirement,
        runtime_byte: u8,
    ) -> ExecutorIdentityChallenge {
        challenge_for(&principal(), requirement, 0xA1, runtime_byte)
    }

    fn relation_policy() -> SubjectGraphRelationPolicyV1 {
        SubjectGraphRelationPolicyV1::new(
            SubjectRelationAssurance::AuthenticatedProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::MeasuredProcess,
            SubjectRelationAssurance::HardwareAnchoredProcess,
        )
    }

    fn policy(
        requirement: &ExecutorIdentityRequirement,
        provider_policy_byte: u8,
    ) -> ExecutorCompositionPolicyV1 {
        ExecutorCompositionPolicyV1::new(
            requirement,
            &relation_policy(),
            digest(provider_policy_byte),
        )
        .expect("policy")
    }

    fn binding_fixture(
        challenge: &ExecutorIdentityChallenge,
        requirement: &ExecutorIdentityRequirement,
        policy: &ExecutorCompositionPolicyV1,
        dimensions: ExecutorIdentityDimensionSet,
        graph_match_byte: u8,
        evidence_byte: u8,
        verification_context_byte: u8,
    ) -> VerifiedExecutorBinding {
        VerifiedExecutorBinding {
            challenge: challenge.digest(),
            principal: challenge.principal_digest(),
            profile: requirement.profile(),
            executor_profile: challenge.executor_profile(),
            runtime_incarnation: challenge.runtime_incarnation(),
            root_subject: ExecutorEvidenceSubjectId::new(digest(0xA4)).expect("subject"),
            verified_dimensions: dimensions,
            graph_match: digest(graph_match_byte),
            composition_policy: policy.id(),
            evidence_commitment: digest(evidence_byte),
            verification_context: ExecutorVerificationContextId::new(digest(
                verification_context_byte,
            ))
            .expect("verification context"),
        }
    }

    fn exact_binding() -> (
        ExecutorIdentityRequirement,
        ExecutorIdentityChallenge,
        ExecutorCompositionPolicyV1,
        VerifiedExecutorBinding,
    ) {
        let requirement = requirement();
        let challenge = challenge(&requirement, 0xA2);
        let policy = policy(&requirement, 0xB1);
        let binding = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            requirement.required_dimensions(),
            0xA8,
            0xA5,
            0xA6,
        );
        (requirement, challenge, policy, binding)
    }

    #[test]
    fn frozen_policy_and_binding_vectors() {
        let (_, _, policy, binding) = exact_binding();
        assert_eq!(policy.id().digest().to_hex(), EXPECTED_POLICY);
        assert_eq!(binding.digest().to_hex(), EXPECTED_BINDING);
    }

    #[test]
    fn exact_structure_matches() {
        let (requirement, challenge, policy, binding) = exact_binding();
        binding
            .structural_match(&challenge, &requirement, &policy)
            .expect("exact live binding must structurally match");
    }

    #[test]
    fn challenge_substitution_rejects() {
        let (requirement, _, policy, binding) = exact_binding();
        let changed = challenge_for(&principal(), &requirement, 0xA1, 0xA7);
        assert_eq!(
            binding.structural_match(&changed, &requirement, &policy),
            Err(ExecutorBindingMismatch::ChallengeMismatch)
        );
    }

    #[test]
    fn supplied_requirement_must_match_challenge_requirement() {
        let (requirement, challenge, policy, binding) = exact_binding();
        let other_requirement = local_requirement();
        assert_ne!(requirement.digest(), other_requirement.digest());
        assert_eq!(
            binding.structural_match(&challenge, &other_requirement, &policy),
            Err(ExecutorBindingMismatch::RequirementMismatch)
        );
    }

    #[test]
    fn composition_policy_requirement_substitution_rejects() {
        let (requirement, challenge, _, binding) = exact_binding();
        let other_policy = policy(&local_requirement(), 0xB1);
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &other_policy),
            Err(ExecutorBindingMismatch::CompositionPolicyRequirementMismatch)
        );
    }

    #[test]
    fn composition_policy_identity_substitution_rejects() {
        let (requirement, challenge, _, binding) = exact_binding();
        let changed_policy = policy(&requirement, 0xB2);
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &changed_policy),
            Err(ExecutorBindingMismatch::CompositionPolicyMismatch)
        );
    }

    #[test]
    fn principal_substitution_rejects_even_under_matching_challenge_digest_field() {
        let (requirement, challenge, policy, mut binding) = exact_binding();
        binding.principal = principal_named("pdp-2").digest();
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &policy),
            Err(ExecutorBindingMismatch::PrincipalMismatch)
        );
    }

    #[test]
    fn executor_profile_substitution_rejects() {
        let (requirement, challenge, policy, mut binding) = exact_binding();
        binding.executor_profile = ExecutorProfileId::new(digest(0xAF)).expect("profile");
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &policy),
            Err(ExecutorBindingMismatch::ExecutorProfileMismatch)
        );
    }

    #[test]
    fn runtime_incarnation_field_substitution_rejects() {
        let (requirement, challenge, policy, mut binding) = exact_binding();
        binding.runtime_incarnation =
            ExecutorRuntimeIncarnationId::new(digest(0xA7)).expect("runtime");
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &policy),
            Err(ExecutorBindingMismatch::RuntimeIncarnationMismatch)
        );
    }

    #[test]
    fn identity_profile_substitution_rejects() {
        let (requirement, challenge, policy, mut binding) = exact_binding();
        binding.profile = ExecutorIdentityProfile::LocalDigital;
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &policy),
            Err(ExecutorBindingMismatch::IdentityProfileMismatch {
                expected: ExecutorIdentityProfile::ConsequentialDigital,
                actual: ExecutorIdentityProfile::LocalDigital,
            })
        );
    }

    #[test]
    fn missing_required_dimension_rejects() {
        let requirement = requirement();
        let challenge = challenge(&requirement, 0xA2);
        let policy = policy(&requirement, 0xB1);
        let binding = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
            0xA8,
            0xA5,
            0xA6,
        );
        assert_eq!(
            binding.structural_match(&challenge, &requirement, &policy),
            Err(ExecutorBindingMismatch::MissingRequiredDimension(
                ExecutorIdentityDimension::Software,
            ))
        );
    }

    #[test]
    fn graph_policy_evidence_and_currentness_are_binding_semantics() {
        let requirement = requirement();
        let challenge = challenge(&requirement, 0xA2);
        let policy = policy(&requirement, 0xB1);
        let dimensions = requirement.required_dimensions();
        let baseline = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            dimensions,
            0xA8,
            0xA5,
            0xA6,
        );
        let graph_changed = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            dimensions,
            0xA9,
            0xA5,
            0xA6,
        );
        let evidence_changed = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            dimensions,
            0xA8,
            0xA7,
            0xA6,
        );
        let context_changed = binding_fixture(
            &challenge,
            &requirement,
            &policy,
            dimensions,
            0xA8,
            0xA5,
            0xA7,
        );
        assert_ne!(baseline.digest(), graph_changed.digest());
        assert_ne!(baseline.digest(), evidence_changed.digest());
        assert_ne!(baseline.digest(), context_changed.digest());
    }

    #[test]
    fn provider_policy_commitment_is_policy_identity() {
        let requirement = requirement();
        assert_ne!(
            policy(&requirement, 0xB1).id(),
            policy(&requirement, 0xB2).id()
        );
    }

    #[test]
    fn zero_context_and_provider_policy_commitments_reject() {
        let requirement = requirement();
        assert_eq!(
            ExecutorVerificationContextId::new(Digest32::new([0; 32])),
            Err(ExecutorComposerError::ZeroDigest(
                "executor verification context identity"
            ))
        );
        assert_eq!(
            ExecutorCompositionPolicyV1::new(
                &requirement,
                &relation_policy(),
                Digest32::new([0; 32]),
            ),
            Err(ExecutorComposerError::ZeroDigest(
                "provider policy commitment"
            ))
        );
    }
}
