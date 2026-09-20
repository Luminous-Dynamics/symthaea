// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provider-neutral executor identity requirement and challenge contracts.
//!
//! This crate deliberately stops **before** live verified executor state.
//! It freezes the common vocabulary a trusted composition service and concrete
//! provider verifiers may share without making this neutral crate a verifier,
//! relying party, provider registry, or execution-authority source.
//!
//! ```text
//! PrincipalRef
//!     != verified executor identity
//!
//! ExecutorIdentityChallenge
//!     != freshness proof
//!     != provider verification
//!     != execution authority
//!
//! evidence reference / digest
//!     != verified identity dimension
//! ```
//!
//! The eventual live `VerifiedExecutorBinding` type belongs in the concrete
//! trusted composer crate, not here. That ownership is security-significant:
//! Rust has no friend-crate mechanism, so keeping a private constructor in this
//! neutral crate would make a separate trusted composer unable to construct the
//! type, while exposing a public constructor would weaken the trust boundary.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{Digest32, PrincipalRef};

pub const EXECUTOR_IDENTITY_SCHEMA_VERSION: u16 = 1;

const REQUIREMENT_DOMAIN: &[u8] = b"symthaea.executor.identity.requirement.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.executor.identity.challenge.v1\0";

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
            Self::ZeroChallengeNonce => {
                write!(formatter, "executor identity challenge nonce must be non-zero")
            }
            Self::MissingMandatoryDimension { profile, dimension } => write!(
                formatter,
                "executor identity profile {profile:?} requires dimension {dimension:?}"
            ),
        }
    }
}

impl Error for ExecutorIdentityError {}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), ExecutorIdentityError> {
    if value.as_bytes() == &[0; 32] {
        Err(ExecutorIdentityError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

/// Closed provider-neutral identity dimensions.
///
/// No dimension is inferred from another. In particular:
///
/// ```text
/// SessionPeer != Workload
/// Workload    != Software
/// Device      != Embodiment
/// ```
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

/// Closed deterministic set of executor identity dimensions.
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
/// an authority/consequence grant.
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
///
/// Construction ensures the selected profile's conservative mandatory
/// dimensions cannot be omitted. Additional dimensions remain explicit policy.
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
digest_identity!(
    ExecutorRuntimeIncarnationId,
    "executor runtime incarnation identity"
);
digest_identity!(
    ExecutorEvidenceSubjectId,
    "executor evidence subject identity"
);
digest_identity!(
    ExecutorVerifierProfileId,
    "executor verifier profile identity"
);

/// Ephemeral composition context shared by provider verifiers and the future
/// trusted composer.
///
/// The nonce is a freshness input only. Constructing this object does not prove
/// the nonce was generated unpredictably, recently, or by a trusted source.
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

    const EXPECTED_PRINCIPAL: &str =
        "6247c633b5234adba7b0403112772997a3bc433de490f147efb10668c0df074a";
    const EXPECTED_REQUIREMENT: &str =
        "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411";
    const EXPECTED_CHALLENGE: &str =
        "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41";

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

    fn consequential_requirement() -> ExecutorIdentityRequirement {
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

    fn challenge(
        requirement: &ExecutorIdentityRequirement,
        runtime: u8,
    ) -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xA3; 32],
            &principal(),
            ExecutorProfileId::new(digest(0xA1)).expect("profile"),
            ExecutorRuntimeIncarnationId::new(digest(runtime)).expect("runtime"),
            requirement,
        )
        .expect("challenge")
    }

    #[test]
    fn frozen_requirement_and_challenge_vectors() {
        let principal = principal();
        let requirement = consequential_requirement();
        let challenge = challenge(&requirement, 0xA2);
        assert_eq!(principal.digest().to_hex(), EXPECTED_PRINCIPAL);
        assert_eq!(requirement.digest().to_hex(), EXPECTED_REQUIREMENT);
        assert_eq!(challenge.digest().to_hex(), EXPECTED_CHALLENGE);
    }

    #[test]
    fn profile_mandatory_dimensions_fail_closed() {
        let missing_embodiment = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::Physical,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::Device,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        );
        assert_eq!(
            missing_embodiment.unwrap_err(),
            ExecutorIdentityError::MissingMandatoryDimension {
                profile: ExecutorIdentityProfile::Physical,
                dimension: ExecutorIdentityDimension::Embodiment,
            }
        );
    }

    #[test]
    fn dimension_order_and_duplicates_are_non_semantic() {
        let left = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .unwrap();
        let right = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::ExecutorProfile,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Software,
            ]),
        )
        .unwrap();
        assert_eq!(left.digest(), right.digest());
    }

    #[test]
    fn runtime_incarnation_changes_challenge_identity() {
        let requirement = consequential_requirement();
        let first = challenge(&requirement, 0xA2);
        let second = challenge(&requirement, 0xA4);
        assert_ne!(first.digest(), second.digest());
        assert_ne!(first.runtime_incarnation(), second.runtime_incarnation());
    }

    #[test]
    fn requirement_change_changes_challenge_identity() {
        let first_requirement = consequential_requirement();
        let second_requirement = ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .unwrap();
        assert_ne!(first_requirement.digest(), second_requirement.digest());
        assert_ne!(
            challenge(&first_requirement, 0xA2).digest(),
            challenge(&second_requirement, 0xA2).digest()
        );
    }

    #[test]
    fn zero_semantic_identities_fail_closed() {
        let zero = Digest32::new([0; 32]);
        assert!(ExecutorProfileId::new(zero).is_err());
        assert!(ExecutorRuntimeIncarnationId::new(zero).is_err());
        assert!(ExecutorEvidenceSubjectId::new(zero).is_err());
        assert!(ExecutorVerifierProfileId::new(zero).is_err());
    }

    #[test]
    fn zero_challenge_nonce_fails_closed() {
        let requirement = consequential_requirement();
        let result = ExecutorIdentityChallenge::new(
            [0; 32],
            &principal(),
            ExecutorProfileId::new(digest(0xA1)).unwrap(),
            ExecutorRuntimeIncarnationId::new(digest(0xA2)).unwrap(),
            &requirement,
        );
        assert_eq!(result.unwrap_err(), ExecutorIdentityError::ZeroChallengeNonce);
    }

    #[test]
    fn challenge_binds_exact_principal_and_executor_profile() {
        let requirement = consequential_requirement();
        let challenge = challenge(&requirement, 0xA2);
        assert_eq!(challenge.principal_digest(), principal().digest());
        assert_eq!(challenge.executor_profile().digest(), digest(0xA1));
        assert_eq!(challenge.requirement_digest(), requirement.digest());
    }
}
