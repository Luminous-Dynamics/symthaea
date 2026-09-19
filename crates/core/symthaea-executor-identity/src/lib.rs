// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provider-neutral executor identity contracts.
//!
//! This crate freezes the semantic requirement/challenge/output ABI for
//! consequential executor identity without implementing a provider verifier or
//! public positive composer.
//!
//! ```text
//! PrincipalRef
//!     != VerifiedExecutorBinding
//!
//! evidence reference / digest
//!     != verified identity dimension
//!
//! VerifiedExecutorBinding
//!     != credential
//!     != CapabilityGrant
//!     != current authority
//!     != DispatchPermit
//! ```
//!
//! The live verified type has private fields, no public constructor, no Clone,
//! and no serialization path. A future trusted composition service must own the
//! transition from provider evidence into this type.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{Digest32, PrincipalRef};

pub const EXECUTOR_IDENTITY_SCHEMA_VERSION: u16 = 1;

const REQUIREMENT_DOMAIN: &[u8] = b"symthaea.executor.identity.requirement.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.executor.identity.challenge.v1\0";
const BINDING_DOMAIN: &[u8] = b"symthaea.executor.identity.binding.v1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorIdentityError {
    ZeroDigest(&'static str),
    ZeroChallengeNonce,
    MissingMandatoryDimension {
        profile: ExecutorIdentityProfile,
        dimension: ExecutorIdentityDimension,
    },
}

impl fmt::Display for ExecutorIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDigest(field) => write!(formatter, "{field} must not use an all-zero digest"),
            Self::ZeroChallengeNonce => write!(formatter, "executor identity challenge nonce must be non-zero"),
            Self::MissingMandatoryDimension { profile, dimension } => write!(
                formatter,
                "executor identity profile {profile:?} requires dimension {dimension:?}"
            ),
        }
    }
}

impl Error for ExecutorIdentityError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorBindingMismatch {
    ChallengeMismatch,
    RequirementMismatch,
    PrincipalMismatch,
    ExecutorProfileMismatch,
    RuntimeIncarnationMismatch,
    IdentityProfileMismatch {
        expected: ExecutorIdentityProfile,
        actual: ExecutorIdentityProfile,
    },
    MissingRequiredDimension(ExecutorIdentityDimension),
}

impl fmt::Display for ExecutorBindingMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeMismatch => write!(formatter, "executor identity challenge does not match"),
            Self::RequirementMismatch => write!(formatter, "executor identity requirement does not match challenge"),
            Self::PrincipalMismatch => write!(formatter, "executor principal does not match challenge"),
            Self::ExecutorProfileMismatch => write!(formatter, "executor profile identity does not match challenge"),
            Self::RuntimeIncarnationMismatch => write!(formatter, "executor runtime incarnation does not match challenge"),
            Self::IdentityProfileMismatch { expected, actual } => write!(
                formatter,
                "executor identity profile mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::MissingRequiredDimension(dimension) => {
                write!(formatter, "verified executor binding lacks required dimension {dimension:?}")
            }
        }
    }
}

impl Error for ExecutorBindingMismatch {}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), ExecutorIdentityError> {
    if value.as_bytes() == &[0; 32] {
        Err(ExecutorIdentityError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

/// Closed provider-neutral identity dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutorIdentityDimension {
    SessionPeer,
    Operator,
    Workload,
    Software,
    Device,
    Embodiment,
    ExecutorProfile,
}

impl ExecutorIdentityDimension {
    const ALL: [Self; 7] = [
        Self::SessionPeer,
        Self::Operator,
        Self::Workload,
        Self::Software,
        Self::Device,
        Self::Embodiment,
        Self::ExecutorProfile,
    ];

    const fn code(self) -> u16 {
        match self {
            Self::SessionPeer => 0,
            Self::Operator => 1,
            Self::Workload => 2,
            Self::Software => 3,
            Self::Device => 4,
            Self::Embodiment => 5,
            Self::ExecutorProfile => 6,
        }
    }

    const fn bit(self) -> u16 {
        1_u16 << self.code()
    }
}

/// Closed set of executor identity dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutorIdentityDimensionSet {
    bits: u16,
}

impl ExecutorIdentityDimensionSet {
    pub fn new(dimensions: &[ExecutorIdentityDimension]) -> Self {
        let mut bits = 0_u16;
        for dimension in dimensions {
            bits |= dimension.bit();
        }
        Self { bits }
    }

    pub const fn contains(self, dimension: ExecutorIdentityDimension) -> bool {
        self.bits & dimension.bit() != 0
    }

    pub const fn is_superset_of(self, required: Self) -> bool {
        self.bits & required.bits == required.bits
    }

    fn count(self) -> u32 {
        self.bits.count_ones()
    }
}

/// Consequence-sensitive identity profile. This is identity policy only, not
/// execution authority or a consequence-class grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorIdentityProfile {
    DevelopmentSimulation,
    LocalDigital,
    ConsequentialDigital,
    Physical,
}

impl ExecutorIdentityProfile {
    const fn code(self) -> u16 {
        match self {
            Self::DevelopmentSimulation => 0,
            Self::LocalDigital => 1,
            Self::ConsequentialDigital => 2,
            Self::Physical => 3,
        }
    }

    fn mandatory_dimensions(self) -> ExecutorIdentityDimensionSet {
        use ExecutorIdentityDimension::{
            Device, Embodiment, ExecutorProfile, Software, Workload,
        };
        match self {
            Self::DevelopmentSimulation => ExecutorIdentityDimensionSet::new(&[ExecutorProfile]),
            Self::LocalDigital | Self::ConsequentialDigital => {
                ExecutorIdentityDimensionSet::new(&[Workload, Software, ExecutorProfile])
            }
            Self::Physical => ExecutorIdentityDimensionSet::new(&[
                Workload,
                Software,
                Device,
                Embodiment,
                ExecutorProfile,
            ]),
        }
    }
}

/// Exact executor identity requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorIdentityRequirement {
    profile: ExecutorIdentityProfile,
    required_dimensions: ExecutorIdentityDimensionSet,
}

impl ExecutorIdentityRequirement {
    pub fn new(
        profile: ExecutorIdentityProfile,
        required_dimensions: ExecutorIdentityDimensionSet,
    ) -> Result<Self, ExecutorIdentityError> {
        let mandatory = profile.mandatory_dimensions();
        for dimension in ExecutorIdentityDimension::ALL {
            if mandatory.contains(dimension) && !required_dimensions.contains(dimension) {
                return Err(ExecutorIdentityError::MissingMandatoryDimension {
                    profile,
                    dimension,
                });
            }
        }
        Ok(Self {
            profile,
            required_dimensions,
        })
    }

    pub const fn profile(&self) -> ExecutorIdentityProfile {
        self.profile
    }

    pub const fn required_dimensions(&self) -> ExecutorIdentityDimensionSet {
        self.required_dimensions
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(REQUIREMENT_DOMAIN);
        transcript.u16(EXECUTOR_IDENTITY_SCHEMA_VERSION);
        transcript.u16(self.profile.code());
        transcript.dimension_set(self.required_dimensions);
        transcript.finish()
    }
}

macro_rules! digest_identity {
    ($name:ident, $field:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Digest32);

        impl $name {
            pub fn new(value: Digest32) -> Result<Self, ExecutorIdentityError> {
                reject_zero($field, value)?;
                Ok(Self(value))
            }

            pub const fn digest(self) -> Digest32 {
                self.0
            }
        }
    };
}

digest_identity!(ExecutorProfileId, "executor profile identity");
digest_identity!(ExecutorRuntimeIncarnationId, "executor runtime incarnation identity");
digest_identity!(ExecutorEvidenceSubjectId, "executor evidence subject identity");
digest_identity!(ExecutorVerifierProfileId, "executor verifier profile identity");

/// Ephemeral composition context. The nonce is a freshness input only; merely
/// constructing this value does not prove nonce unpredictability or trusted
/// freshness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorIdentityChallenge {
    nonce: [u8; 32],
    principal: Digest32,
    executor_profile: ExecutorProfileId,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    requirement: Digest32,
}

impl ExecutorIdentityChallenge {
    pub fn new(
        nonce: [u8; 32],
        principal: &PrincipalRef,
        executor_profile: ExecutorProfileId,
        runtime_incarnation: ExecutorRuntimeIncarnationId,
        requirement: &ExecutorIdentityRequirement,
    ) -> Result<Self, ExecutorIdentityError> {
        if nonce == [0; 32] {
            return Err(ExecutorIdentityError::ZeroChallengeNonce);
        }
        Ok(Self {
            nonce,
            principal: principal.digest(),
            executor_profile,
            runtime_incarnation,
            requirement: requirement.digest(),
        })
    }

    pub const fn nonce(&self) -> &[u8; 32] {
        &self.nonce
    }

    pub const fn principal_digest(&self) -> Digest32 {
        self.principal
    }

    pub const fn executor_profile(&self) -> ExecutorProfileId {
        self.executor_profile
    }

    pub const fn runtime_incarnation(&self) -> ExecutorRuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn requirement_digest(&self) -> Digest32 {
        self.requirement
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN);
        transcript.u16(EXECUTOR_IDENTITY_SCHEMA_VERSION);
        transcript.bytes32(self.nonce);
        transcript.digest(self.principal);
        transcript.digest(self.executor_profile.digest());
        transcript.digest(self.runtime_incarnation.digest());
        transcript.digest(self.requirement);
        transcript.finish()
    }
}

/// Opaque verifier-owned same-live-subject identity result.
///
/// This type deliberately has no public constructor, no Clone, and no
/// serialization implementation. A future trusted composition service inside
/// the executor-identity verification boundary must own production creation.
#[derive(Debug, PartialEq, Eq)]
pub struct VerifiedExecutorBinding {
    challenge: Digest32,
    principal: Digest32,
    profile: ExecutorIdentityProfile,
    executor_profile: ExecutorProfileId,
    runtime_incarnation: ExecutorRuntimeIncarnationId,
    subject: ExecutorEvidenceSubjectId,
    verified_dimensions: ExecutorIdentityDimensionSet,
    verifier_profile: ExecutorVerifierProfileId,
    evidence_commitment: Digest32,
    verification_context: Digest32,
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

    pub const fn subject(&self) -> ExecutorEvidenceSubjectId {
        self.subject
    }

    pub const fn verified_dimensions(&self) -> ExecutorIdentityDimensionSet {
        self.verified_dimensions
    }

    pub const fn verifier_profile(&self) -> ExecutorVerifierProfileId {
        self.verifier_profile
    }

    pub const fn evidence_commitment(&self) -> Digest32 {
        self.evidence_commitment
    }

    pub const fn verification_context(&self) -> Digest32 {
        self.verification_context
    }

    /// Structural comparison only. Success means this already verifier-owned
    /// binding names the exact challenge/profile/incarnation and contains every
    /// required dimension. It does not independently prove currentness.
    pub fn structural_match(
        &self,
        challenge: &ExecutorIdentityChallenge,
        requirement: &ExecutorIdentityRequirement,
    ) -> Result<(), ExecutorBindingMismatch> {
        if self.challenge != challenge.digest() {
            return Err(ExecutorBindingMismatch::ChallengeMismatch);
        }
        if challenge.requirement != requirement.digest() {
            return Err(ExecutorBindingMismatch::RequirementMismatch);
        }
        if self.principal != challenge.principal {
            return Err(ExecutorBindingMismatch::PrincipalMismatch);
        }
        if self.executor_profile != challenge.executor_profile {
            return Err(ExecutorBindingMismatch::ExecutorProfileMismatch);
        }
        if self.runtime_incarnation != challenge.runtime_incarnation {
            return Err(ExecutorBindingMismatch::RuntimeIncarnationMismatch);
        }
        if self.profile != requirement.profile {
            return Err(ExecutorBindingMismatch::IdentityProfileMismatch {
                expected: requirement.profile,
                actual: self.profile,
            });
        }
        for dimension in ExecutorIdentityDimension::ALL {
            if requirement.required_dimensions.contains(dimension)
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
        transcript.u16(EXECUTOR_IDENTITY_SCHEMA_VERSION);
        transcript.digest(self.challenge);
        transcript.digest(self.principal);
        transcript.u16(self.profile.code());
        transcript.digest(self.executor_profile.digest());
        transcript.digest(self.runtime_incarnation.digest());
        transcript.digest(self.subject.digest());
        transcript.dimension_set(self.verified_dimensions);
        transcript.digest(self.verifier_profile.digest());
        transcript.digest(self.evidence_commitment);
        transcript.digest(self.verification_context);
        transcript.finish()
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

    fn bytes32(&mut self, value: [u8; 32]) {
        self.hasher.update(value);
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn dimension_set(&mut self, dimensions: ExecutorIdentityDimensionSet) {
        self.u32(dimensions.count());
        for dimension in ExecutorIdentityDimension::ALL {
            if dimensions.contains(dimension) {
                self.u16(dimension.code());
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
    use symthaea_interaction_core::{IdentityComponent, IdentityOrdering, NamespaceId};

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn principal() -> PrincipalRef {
        PrincipalRef::new(
            NamespaceId::new("intx/workload").expect("namespace"),
            "policy-pdp",
            IdentityOrdering::NamedSet,
            vec![IdentityComponent::new("name", "pdp-1").expect("component")],
        )
        .expect("principal")
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

    fn challenge() -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xa3; 32],
            &principal(),
            ExecutorProfileId::new(digest(0xa1)).expect("profile"),
            ExecutorRuntimeIncarnationId::new(digest(0xa2)).expect("runtime"),
            &requirement(),
        )
        .expect("challenge")
    }

    fn verified_binding(dimensions: ExecutorIdentityDimensionSet) -> VerifiedExecutorBinding {
        VerifiedExecutorBinding {
            challenge: challenge().digest(),
            principal: principal().digest(),
            profile: ExecutorIdentityProfile::ConsequentialDigital,
            executor_profile: ExecutorProfileId::new(digest(0xa1)).unwrap(),
            runtime_incarnation: ExecutorRuntimeIncarnationId::new(digest(0xa2)).unwrap(),
            subject: ExecutorEvidenceSubjectId::new(digest(0xa4)).unwrap(),
            verified_dimensions: dimensions,
            verifier_profile: ExecutorVerifierProfileId::new(digest(0xa7)).unwrap(),
            evidence_commitment: digest(0xa5),
            verification_context: digest(0xa6),
        }
    }

    fn full_dimensions() -> ExecutorIdentityDimensionSet {
        requirement().required_dimensions()
    }

    #[test]
    fn canonical_vectors_are_stable() {
        assert_eq!(
            principal().digest().to_hex(),
            "6247c633b5234adba7b0403112772997a3bc433de490f147efb10668c0df074a"
        );
        assert_eq!(
            requirement().digest().to_hex(),
            "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411"
        );
        assert_eq!(
            challenge().digest().to_hex(),
            "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41"
        );
        assert_eq!(
            verified_binding(full_dimensions()).digest().to_hex(),
            "5715b3a04b88c47f91278ee43bb782ca2754a076609d6518eda5c7d898b03259"
        );
    }

    #[test]
    fn profile_mandatory_dimensions_fail_closed() {
        let error = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::Physical,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::Device,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect_err("physical requirement without embodiment must fail");
        assert_eq!(
            error,
            ExecutorIdentityError::MissingMandatoryDimension {
                profile: ExecutorIdentityProfile::Physical,
                dimension: ExecutorIdentityDimension::Embodiment,
            }
        );
    }

    #[test]
    fn exact_challenge_and_requirement_structurally_match() {
        verified_binding(full_dimensions())
            .structural_match(&challenge(), &requirement())
            .expect("structural match");
    }

    #[test]
    fn missing_dimension_rejects() {
        let incomplete = ExecutorIdentityDimensionSet::new(&[
            ExecutorIdentityDimension::Workload,
            ExecutorIdentityDimension::Software,
            ExecutorIdentityDimension::ExecutorProfile,
        ]);
        let error = verified_binding(incomplete)
            .structural_match(&challenge(), &requirement())
            .expect_err("missing session peer must reject");
        assert_eq!(
            error,
            ExecutorBindingMismatch::MissingRequiredDimension(
                ExecutorIdentityDimension::SessionPeer
            )
        );
    }

    #[test]
    fn runtime_incarnation_changes_challenge_identity() {
        let changed = ExecutorIdentityChallenge::new(
            [0xa3; 32],
            &principal(),
            ExecutorProfileId::new(digest(0xa1)).unwrap(),
            ExecutorRuntimeIncarnationId::new(digest(0xb2)).unwrap(),
            &requirement(),
        )
        .unwrap();
        assert_ne!(challenge().digest(), changed.digest());
        assert_eq!(
            verified_binding(full_dimensions()).structural_match(&changed, &requirement()),
            Err(ExecutorBindingMismatch::ChallengeMismatch)
        );
    }

    #[test]
    fn profile_is_not_implicitly_widened() {
        let physical = ExecutorIdentityRequirement::new(
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
        let error = verified_binding(full_dimensions())
            .structural_match(&challenge(), &physical)
            .expect_err("profile mismatch must reject");
        assert_eq!(error, ExecutorBindingMismatch::RequirementMismatch);
    }

    #[test]
    fn verifier_or_evidence_context_changes_binding_identity() {
        let baseline = verified_binding(full_dimensions());
        let changed_verifier = VerifiedExecutorBinding {
            verifier_profile: ExecutorVerifierProfileId::new(digest(0xb7)).unwrap(),
            ..verified_binding(full_dimensions())
        };
        let changed_evidence = VerifiedExecutorBinding {
            evidence_commitment: digest(0xb5),
            ..verified_binding(full_dimensions())
        };
        let changed_context = VerifiedExecutorBinding {
            verification_context: digest(0xb6),
            ..verified_binding(full_dimensions())
        };
        assert_ne!(baseline.digest(), changed_verifier.digest());
        assert_ne!(baseline.digest(), changed_evidence.digest());
        assert_ne!(baseline.digest(), changed_context.digest());
    }

    #[test]
    fn zero_identities_and_nonce_fail_closed() {
        assert_eq!(
            ExecutorProfileId::new(Digest32::new([0; 32])).expect_err("zero profile"),
            ExecutorIdentityError::ZeroDigest("executor profile identity")
        );
        assert_eq!(
            ExecutorIdentityChallenge::new(
                [0; 32],
                &principal(),
                ExecutorProfileId::new(digest(0xa1)).unwrap(),
                ExecutorRuntimeIncarnationId::new(digest(0xa2)).unwrap(),
                &requirement(),
            )
            .expect_err("zero nonce"),
            ExecutorIdentityError::ZeroChallengeNonce
        );
    }
}
