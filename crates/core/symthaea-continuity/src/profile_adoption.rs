// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lineage-bearing authority subjects for verifier-profile adoption.
//!
//! A [`VerifierProfileV1`] is configuration. It describes an exact verifier root,
//! root epoch, profile name, and maximum evidence class, but construction does not
//! grant that verifier semantic authority. This module defines the exact untrusted
//! subject and predecessor lineage that a separate authority verifier must later
//! authenticate before a production path may derive an authorized verifier profile.
//!
//! Core theorem:
//!
//! `VerifierProfileV1 != AdoptionTransition != AuthorizedVerifierProfile`.
//!
//! No type in this module grants verification, witness, migration, or execution
//! authority by itself.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{ContinuityContractId, ContinuityRequirementId};
use crate::verifier::{VerifierProfileId, VerifierProfileV1};
use crate::witness::EvidenceClass;

/// Stable schema for one verifier-profile adoption claim.
pub const VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-subject-v1";
/// Stable schema for the lineage-bearing transition that must be authenticated.
pub const VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-transition-v1";

const SUBJECT_DOMAIN: &[u8] = b"symthaea.continuity.verifier-profile-adoption.subject.v1\0";
const TRANSITION_DOMAIN: &[u8] = b"symthaea.continuity.verifier-profile-adoption.transition.v1\0";

/// Content identity of one canonical verifier-profile adoption subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionSubjectId([u8; 32]);

impl VerifierProfileAdoptionSubjectId {
    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Content identity of one exact lineage-bearing adoption transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionTransitionDigest([u8; 32]);

impl VerifierProfileAdoptionTransitionDigest {
    /// Raw BLAKE3-256 transition bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Scope within which one adopted verifier may issue continuity evidence.
///
/// Scope is separate from evidence strength: a verifier may be authorized for a
/// narrow requirement set even when its profile is capable of stronger evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VerifierAdoptionScopeV1 {
    /// All continuity-verification requirements under the authority subject.
    AllContinuityVerification,
    /// Every requirement in one exact validated continuity contract.
    Contract {
        /// Exact contract identity.
        contract_id: ContinuityContractId,
    },
    /// A canonical non-empty subset of requirements in one exact contract.
    Requirements {
        /// Exact contract identity containing the requirements.
        contract_id: ContinuityContractId,
        /// Sorted unique requirement identities.
        requirement_ids: Vec<ContinuityRequirementId>,
    },
}

impl VerifierAdoptionScopeV1 {
    /// Construct a canonical exact-requirement scope.
    pub fn requirements(
        contract_id: ContinuityContractId,
        mut requirement_ids: Vec<ContinuityRequirementId>,
    ) -> Result<Self, VerifierProfileAdoptionError> {
        requirement_ids.sort();
        requirement_ids.dedup();
        if requirement_ids.is_empty() {
            return Err(VerifierProfileAdoptionError::EmptyRequirementScope);
        }
        Ok(Self::Requirements {
            contract_id,
            requirement_ids,
        })
    }

    fn validate(&self) -> Result<(), VerifierProfileAdoptionError> {
        if let Self::Requirements {
            requirement_ids, ..
        } = self
        {
            if requirement_ids.is_empty() {
                return Err(VerifierProfileAdoptionError::EmptyRequirementScope);
            }
            if requirement_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(VerifierProfileAdoptionError::NonCanonicalRequirementScope);
            }
        }
        Ok(())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::AllContinuityVerification => out.push(0),
            Self::Contract { contract_id } => {
                out.push(1);
                out.extend_from_slice(contract_id.as_bytes());
            }
            Self::Requirements {
                contract_id,
                requirement_ids,
            } => {
                out.push(2);
                out.extend_from_slice(contract_id.as_bytes());
                put_len(out, requirement_ids.len());
                for requirement_id in requirement_ids {
                    out.extend_from_slice(requirement_id.as_bytes());
                }
            }
        }
    }
}

/// Untrusted claim that an external adoption authority intends to grant one exact
/// verifier profile a bounded evidence role.
///
/// This value is serializable transport/configuration, not authorized state. A
/// future admission layer must compare the lineage-bearing transition against an
/// externally trusted adoption-authority root/head and cryptographic proof before
/// deriving any opaque authorized-profile type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionSubjectV1 {
    schema_version: String,
    adoption_id: String,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    verifier_role_id: String,
    verifier_profile_id: VerifierProfileId,
    generation: u64,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    evidence_class_ceiling: EvidenceClass,
    scope: VerifierAdoptionScopeV1,
    subject_id: VerifierProfileAdoptionSubjectId,
}

impl VerifierProfileAdoptionSubjectV1 {
    /// Construct one adoption subject against an exact verifier-profile snapshot.
    ///
    /// The grant may restrict the profile's evidence class but cannot exceed it.
    /// The adoption-authority root must also differ from the verifier's own root,
    /// preventing the direct self-adoption shape from being created normally.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adoption_id: impl Into<String>,
        authority_subject: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: [u8; 32],
        profile: &VerifierProfileV1,
        generation: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
        evidence_class_ceiling: EvidenceClass,
        scope: VerifierAdoptionScopeV1,
    ) -> Result<Self, VerifierProfileAdoptionError> {
        profile.validate()?;
        let adoption_id = checked_text("adoption_id", adoption_id.into())?;
        let authority_subject = checked_text("authority_subject", authority_subject.into())?;
        let authority_root_id = checked_text("authority_root_id", authority_root_id.into())?;
        let verifier_role_id = checked_text("verifier_role_id", profile.profile_name().to_owned())?;
        if authority_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionError::ZeroAuthorityRootDigest);
        }
        if authority_root_digest == profile.root_digest() {
            return Err(VerifierProfileAdoptionError::VerifierCannotSelfAdopt);
        }
        if generation == 0 {
            return Err(VerifierProfileAdoptionError::ZeroGeneration);
        }
        if valid_from_unix_ms == 0 || valid_until_unix_ms == 0 {
            return Err(VerifierProfileAdoptionError::ZeroValidityBound);
        }
        if valid_from_unix_ms >= valid_until_unix_ms {
            return Err(VerifierProfileAdoptionError::InvalidValidityInterval);
        }
        if evidence_class_ceiling > profile.evidence_class() {
            return Err(VerifierProfileAdoptionError::EvidenceClassExceedsProfile {
                profile: profile.evidence_class(),
                requested: evidence_class_ceiling,
            });
        }
        scope.validate()?;

        let verifier_profile_id = profile.id();
        let subject_id = VerifierProfileAdoptionSubjectId(hash_subject_fields(
            &adoption_id,
            &authority_subject,
            &authority_root_id,
            authority_root_digest,
            &verifier_role_id,
            verifier_profile_id,
            generation,
            valid_from_unix_ms,
            valid_until_unix_ms,
            evidence_class_ceiling,
            &scope,
        ));

        Ok(Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1.to_owned(),
            adoption_id,
            authority_subject,
            authority_root_id,
            authority_root_digest,
            verifier_role_id,
            verifier_profile_id,
            generation,
            valid_from_unix_ms,
            valid_until_unix_ms,
            evidence_class_ceiling,
            scope,
            subject_id,
        })
    }

    /// Validate intrinsic structure and stored content identity.
    ///
    /// This does not establish that `verifier_profile_id` corresponds to any
    /// currently provisioned profile. Use [`Self::validate_against_profile`] when
    /// that exact profile snapshot is available.
    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionError::UnsupportedSubjectSchema(
                self.schema_version.clone(),
            ));
        }
        checked_text("adoption_id", self.adoption_id.clone())?;
        checked_text("authority_subject", self.authority_subject.clone())?;
        checked_text("authority_root_id", self.authority_root_id.clone())?;
        checked_text("verifier_role_id", self.verifier_role_id.clone())?;
        if self.authority_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionError::ZeroAuthorityRootDigest);
        }
        if self.generation == 0 {
            return Err(VerifierProfileAdoptionError::ZeroGeneration);
        }
        if self.valid_from_unix_ms == 0 || self.valid_until_unix_ms == 0 {
            return Err(VerifierProfileAdoptionError::ZeroValidityBound);
        }
        if self.valid_from_unix_ms >= self.valid_until_unix_ms {
            return Err(VerifierProfileAdoptionError::InvalidValidityInterval);
        }
        self.scope.validate()?;
        let expected = VerifierProfileAdoptionSubjectId(hash_subject_fields(
            &self.adoption_id,
            &self.authority_subject,
            &self.authority_root_id,
            self.authority_root_digest,
            &self.verifier_role_id,
            self.verifier_profile_id,
            self.generation,
            self.valid_from_unix_ms,
            self.valid_until_unix_ms,
            self.evidence_class_ceiling,
            &self.scope,
        ));
        if expected != self.subject_id {
            return Err(VerifierProfileAdoptionError::SubjectIdentityMismatch);
        }
        Ok(())
    }

    /// Re-bind the transport subject to the exact local verifier profile it claims.
    pub fn validate_against_profile(
        &self,
        profile: &VerifierProfileV1,
    ) -> Result<(), VerifierProfileAdoptionError> {
        self.validate()?;
        profile.validate()?;
        if self.verifier_profile_id != profile.id() {
            return Err(VerifierProfileAdoptionError::VerifierProfileMismatch);
        }
        if self.verifier_role_id != profile.profile_name() {
            return Err(VerifierProfileAdoptionError::VerifierRoleMismatch);
        }
        if self.authority_root_digest == profile.root_digest() {
            return Err(VerifierProfileAdoptionError::VerifierCannotSelfAdopt);
        }
        if self.evidence_class_ceiling > profile.evidence_class() {
            return Err(VerifierProfileAdoptionError::EvidenceClassExceedsProfile {
                profile: profile.evidence_class(),
                requested: self.evidence_class_ceiling,
            });
        }
        Ok(())
    }

    /// Canonical application bytes later authenticated by a detached proof.
    pub fn canonical_signing_bytes(&self) -> Result<Vec<u8>, VerifierProfileAdoptionError> {
        self.validate()?;
        let mut out = Vec::with_capacity(384);
        out.extend_from_slice(SUBJECT_DOMAIN);
        put_str(&mut out, VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1);
        put_str(&mut out, &self.adoption_id);
        put_str(&mut out, &self.authority_subject);
        put_str(&mut out, &self.authority_root_id);
        out.extend_from_slice(&self.authority_root_digest);
        put_str(&mut out, &self.verifier_role_id);
        out.extend_from_slice(self.verifier_profile_id.as_bytes());
        out.extend_from_slice(&self.generation.to_le_bytes());
        out.extend_from_slice(&self.valid_from_unix_ms.to_le_bytes());
        out.extend_from_slice(&self.valid_until_unix_ms.to_le_bytes());
        out.push(evidence_class_tag(self.evidence_class_ceiling));
        self.scope.encode(&mut out);
        out.extend_from_slice(self.subject_id.as_bytes());
        Ok(out)
    }

    pub fn id(&self) -> VerifierProfileAdoptionSubjectId {
        self.subject_id
    }
    pub fn adoption_id(&self) -> &str {
        &self.adoption_id
    }
    pub fn authority_subject(&self) -> &str {
        &self.authority_subject
    }
    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }
    pub fn authority_root_digest(&self) -> [u8; 32] {
        self.authority_root_digest
    }
    pub fn verifier_role_id(&self) -> &str {
        &self.verifier_role_id
    }
    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn valid_from_unix_ms(&self) -> u64 {
        self.valid_from_unix_ms
    }
    pub fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
    pub fn evidence_class_ceiling(&self) -> EvidenceClass {
        self.evidence_class_ceiling
    }
    pub fn scope(&self) -> &VerifierAdoptionScopeV1 {
        &self.scope
    }
}

/// Exact predecessor state for one logical verifier-role adoption lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerifierProfileAdoptionPredecessorV1 {
    /// Generation 1 only. The adoption-authority root must already be trusted by a
    /// separate bootstrap/provisioning policy.
    Bootstrap,
    /// Exact digest of the immediately preceding transition.
    Previous(VerifierProfileAdoptionTransitionDigest),
}

/// Lineage-bearing adoption transition that a separate authority verifier must
/// authenticate before local admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionTransitionV1 {
    schema_version: String,
    predecessor: VerifierProfileAdoptionPredecessorV1,
    subject: VerifierProfileAdoptionSubjectV1,
}

impl VerifierProfileAdoptionTransitionV1 {
    /// Build generation 1 under an already externally trusted adoption authority.
    pub fn bootstrap(
        subject: VerifierProfileAdoptionSubjectV1,
    ) -> Result<Self, VerifierProfileAdoptionError> {
        subject.validate()?;
        if subject.generation() != 1 {
            return Err(VerifierProfileAdoptionError::BootstrapGenerationMustBeOne {
                observed: subject.generation(),
            });
        }
        Ok(Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1.to_owned(),
            predecessor: VerifierProfileAdoptionPredecessorV1::Bootstrap,
            subject,
        })
    }

    /// Build an exact successor in the same organizational verifier-role lineage.
    ///
    /// The adoption authority and logical verifier role cannot change inside an
    /// ordinary successor. The exact `VerifierProfileId` may change: that is the
    /// mechanism by which the external adoption authority can rotate/restrict the
    /// verifier root without allowing the verifier to self-authorize the change.
    pub fn successor(
        subject: VerifierProfileAdoptionSubjectV1,
        predecessor: &VerifierProfileAdoptionTransitionV1,
    ) -> Result<Self, VerifierProfileAdoptionError> {
        predecessor.validate()?;
        subject.validate()?;
        let expected_generation = predecessor.generation().checked_add(1).ok_or(
            VerifierProfileAdoptionError::GenerationExhausted {
                current: predecessor.generation(),
            },
        )?;
        if subject.generation() != expected_generation {
            return Err(VerifierProfileAdoptionError::GenerationNotSuccessor {
                current: predecessor.generation(),
                expected: expected_generation,
                observed: subject.generation(),
            });
        }
        if subject.authority_subject() != predecessor.subject.authority_subject() {
            return Err(VerifierProfileAdoptionError::AuthoritySubjectChanged);
        }
        if subject.authority_root_id() != predecessor.subject.authority_root_id() {
            return Err(VerifierProfileAdoptionError::AuthorityRootIdChanged);
        }
        if subject.authority_root_digest() != predecessor.subject.authority_root_digest() {
            return Err(VerifierProfileAdoptionError::AuthorityRootChanged);
        }
        if subject.verifier_role_id() != predecessor.subject.verifier_role_id() {
            return Err(VerifierProfileAdoptionError::VerifierRoleChanged);
        }
        Ok(Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1.to_owned(),
            predecessor: VerifierProfileAdoptionPredecessorV1::Previous(
                predecessor.transition_digest()?,
            ),
            subject,
        })
    }

    /// Validate intrinsic transition structure. This is not signature verification
    /// and does not compare against any persisted/current adoption head.
    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionError::UnsupportedTransitionSchema(
                self.schema_version.clone(),
            ));
        }
        self.subject.validate()?;
        match (self.subject.generation(), self.predecessor) {
            (1, VerifierProfileAdoptionPredecessorV1::Bootstrap) => Ok(()),
            (1, VerifierProfileAdoptionPredecessorV1::Previous(_)) => {
                Err(VerifierProfileAdoptionError::GenerationOneHasPredecessor)
            }
            (_, VerifierProfileAdoptionPredecessorV1::Bootstrap) => Err(
                VerifierProfileAdoptionError::NonInitialGenerationUsesBootstrap {
                    generation: self.subject.generation(),
                },
            ),
            (_, VerifierProfileAdoptionPredecessorV1::Previous(_)) => Ok(()),
        }
    }

    /// Canonical exact bytes a future authority verifier must authenticate.
    pub fn canonical_signing_bytes(&self) -> Result<Vec<u8>, VerifierProfileAdoptionError> {
        self.validate()?;
        let subject_bytes = self.subject.canonical_signing_bytes()?;
        let mut out = Vec::with_capacity(subject_bytes.len() + 128);
        out.extend_from_slice(TRANSITION_DOMAIN);
        put_str(&mut out, VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1);
        match self.predecessor {
            VerifierProfileAdoptionPredecessorV1::Bootstrap => out.push(0),
            VerifierProfileAdoptionPredecessorV1::Previous(digest) => {
                out.push(1);
                out.extend_from_slice(digest.as_bytes());
            }
        }
        put_len(&mut out, subject_bytes.len());
        out.extend_from_slice(&subject_bytes);
        Ok(out)
    }

    /// BLAKE3-256 identity of the exact canonical transition bytes.
    pub fn transition_digest(
        &self,
    ) -> Result<VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionError> {
        Ok(VerifierProfileAdoptionTransitionDigest(
            *blake3::hash(&self.canonical_signing_bytes()?).as_bytes(),
        ))
    }

    pub fn predecessor(&self) -> VerifierProfileAdoptionPredecessorV1 {
        self.predecessor
    }
    pub fn subject(&self) -> &VerifierProfileAdoptionSubjectV1 {
        &self.subject
    }
    pub fn generation(&self) -> u64 {
        self.subject.generation()
    }
}

/// Fail-closed construction/lineage errors for verifier adoption claims.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionError {
    #[error(transparent)]
    VerifierProfile(#[from] crate::verifier::VerificationAdmissionError),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("adoption-authority root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("a verifier root cannot directly adopt itself")]
    VerifierCannotSelfAdopt,
    #[error("adoption generation must be non-zero")]
    ZeroGeneration,
    #[error("adoption validity bounds must be non-zero")]
    ZeroValidityBound,
    #[error("adoption validity interval must satisfy valid_from < valid_until")]
    InvalidValidityInterval,
    #[error("requested evidence class {requested:?} exceeds verifier profile maximum {profile:?}")]
    EvidenceClassExceedsProfile {
        profile: EvidenceClass,
        requested: EvidenceClass,
    },
    #[error("exact requirement adoption scope must not be empty")]
    EmptyRequirementScope,
    #[error("exact requirement adoption scope must be sorted and unique")]
    NonCanonicalRequirementScope,
    #[error("unsupported verifier-profile adoption subject schema {0}")]
    UnsupportedSubjectSchema(String),
    #[error("stored verifier-profile adoption subject identity is not canonical")]
    SubjectIdentityMismatch,
    #[error("adoption subject does not bind the supplied exact verifier profile")]
    VerifierProfileMismatch,
    #[error("adoption subject verifier role does not match the supplied profile name")]
    VerifierRoleMismatch,
    #[error("unsupported verifier-profile adoption transition schema {0}")]
    UnsupportedTransitionSchema(String),
    #[error("bootstrap verifier-profile adoption generation must be 1, observed {observed}")]
    BootstrapGenerationMustBeOne { observed: u64 },
    #[error("generation 1 verifier-profile adoption must not carry a predecessor")]
    GenerationOneHasPredecessor,
    #[error("verifier-profile adoption generation {generation} cannot use bootstrap predecessor")]
    NonInitialGenerationUsesBootstrap { generation: u64 },
    #[error("verifier-profile adoption generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error(
        "verifier-profile adoption generation must be exact successor of {current}: expected {expected}, observed {observed}"
    )]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
    },
    #[error("adoption authority subject changed across ordinary successor")]
    AuthoritySubjectChanged,
    #[error("adoption authority root id changed across ordinary successor")]
    AuthorityRootIdChanged,
    #[error("adoption authority root changed across ordinary successor")]
    AuthorityRootChanged,
    #[error("logical verifier role changed across ordinary successor")]
    VerifierRoleChanged,
}

#[allow(clippy::too_many_arguments)]
fn hash_subject_fields(
    adoption_id: &str,
    authority_subject: &str,
    authority_root_id: &str,
    authority_root_digest: [u8; 32],
    verifier_role_id: &str,
    verifier_profile_id: VerifierProfileId,
    generation: u64,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    evidence_class_ceiling: EvidenceClass,
    scope: &VerifierAdoptionScopeV1,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, adoption_id);
    put_str(&mut bytes, authority_subject);
    put_str(&mut bytes, authority_root_id);
    bytes.extend_from_slice(&authority_root_digest);
    put_str(&mut bytes, verifier_role_id);
    bytes.extend_from_slice(verifier_profile_id.as_bytes());
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(&valid_from_unix_ms.to_le_bytes());
    bytes.extend_from_slice(&valid_until_unix_ms.to_le_bytes());
    bytes.push(evidence_class_tag(evidence_class_ceiling));
    scope.encode(&mut bytes);
    let mut hasher = blake3::Hasher::new();
    hasher.update(SUBJECT_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn evidence_class_tag(class: EvidenceClass) -> u8 {
    match class {
        EvidenceClass::Declared => 1,
        EvidenceClass::Observed => 2,
        EvidenceClass::StaticAnalysis => 3,
        EvidenceClass::Simulated => 4,
        EvidenceClass::DifferentiallyVerified => 5,
        EvidenceClass::HardwareVerified => 6,
        EvidenceClass::IndependentlyReplicated => 7,
    }
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };

    fn profile(root: u8, epoch: u64, class: EvidenceClass) -> VerifierProfileV1 {
        VerifierProfileV1::new("hardware-verifier-v1", [root; 32], epoch, class).unwrap()
    }

    fn subject(
        profile: &VerifierProfileV1,
        adoption_id: &str,
        generation: u64,
        authority_root: u8,
        ceiling: EvidenceClass,
    ) -> VerifierProfileAdoptionSubjectV1 {
        VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            "organization:test",
            "adoption-root-1",
            [authority_root; 32],
            profile,
            generation,
            1_000,
            2_000,
            ceiling,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap()
    }

    fn contract_with_two_requirements() -> crate::ValidatedContinuityContractV1 {
        let mut requirements = Vec::new();
        for seed in [1u8, 2u8] {
            let observation = ObservationEnvelopeV1::new(
                format!("machine-{seed}"),
                "workflow.dependency",
                "fixture",
                "1",
                1_700_000_000_000 + seed as u64,
                ObservationCoverage::Complete,
                EvidenceBasis::Tested,
                [seed; 32],
                vec![],
            )
            .unwrap();
            let dependency = DependencyClaimV1::new(
                "role:research",
                "requires",
                format!("capability-{seed}"),
                DependencyBasis::Observed,
                vec![observation.id()],
                vec![],
            )
            .unwrap();
            requirements.push(
                ContinuityRequirementV1::new(
                    dependency.id(),
                    format!("capability-{seed}"),
                    RequirementCriticality::Must,
                    EquivalencePredicate::BehavioralScenario {
                        scenario_id: format!("scenario-{seed}"),
                    },
                    ApprovalBasis::ExplicitPolicy,
                    [seed.wrapping_add(20); 32],
                )
                .unwrap(),
            );
        }
        ContinuityContractV1::new("research-fleet", [44; 32], requirements)
            .unwrap()
            .validate()
            .unwrap()
    }

    #[test]
    fn verifier_cannot_directly_self_adopt() {
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        assert_eq!(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-1",
                "organization:test",
                "adoption-root-1",
                profile.root_digest(),
                &profile,
                1,
                1_000,
                2_000,
                EvidenceClass::HardwareVerified,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            )
            .unwrap_err(),
            VerifierProfileAdoptionError::VerifierCannotSelfAdopt
        );
    }

    #[test]
    fn adoption_cannot_grant_more_strength_than_profile_declares() {
        let profile = profile(9, 7, EvidenceClass::Observed);
        assert!(matches!(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-1",
                "organization:test",
                "adoption-root-1",
                [4; 32],
                &profile,
                1,
                1_000,
                2_000,
                EvidenceClass::HardwareVerified,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            ),
            Err(VerifierProfileAdoptionError::EvidenceClassExceedsProfile { .. })
        ));
    }

    #[test]
    fn bootstrap_is_generation_one_only() {
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let first = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &profile,
            "adopt-1",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        assert_eq!(first.generation(), 1);
        assert_eq!(
            first.predecessor(),
            VerifierProfileAdoptionPredecessorV1::Bootstrap
        );
        assert!(matches!(
            VerifierProfileAdoptionTransitionV1::bootstrap(subject(
                &profile,
                "adopt-2",
                2,
                4,
                EvidenceClass::HardwareVerified
            )),
            Err(VerifierProfileAdoptionError::BootstrapGenerationMustBeOne { observed: 2 })
        ));
    }

    #[test]
    fn successor_binds_exact_predecessor() {
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let first = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &profile,
            "adopt-a",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        let second = VerifierProfileAdoptionTransitionV1::successor(
            subject(&profile, "adopt-b", 2, 4, EvidenceClass::HardwareVerified),
            &first,
        )
        .unwrap();
        assert_eq!(
            second.predecessor(),
            VerifierProfileAdoptionPredecessorV1::Previous(first.transition_digest().unwrap())
        );
    }

    #[test]
    fn adoption_authority_can_rotate_exact_verifier_profile_under_same_role() {
        let old_profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let new_profile = profile(10, 8, EvidenceClass::HardwareVerified);
        let first = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &old_profile,
            "adopt-a",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        let second = VerifierProfileAdoptionTransitionV1::successor(
            subject(
                &new_profile,
                "adopt-b",
                2,
                4,
                EvidenceClass::DifferentiallyVerified,
            ),
            &first,
        )
        .unwrap();
        assert_ne!(
            first.subject().verifier_profile_id(),
            second.subject().verifier_profile_id()
        );
        assert_eq!(
            first.subject().verifier_role_id(),
            second.subject().verifier_role_id()
        );
        assert_eq!(
            second.subject().evidence_class_ceiling(),
            EvidenceClass::DifferentiallyVerified
        );
    }

    #[test]
    fn adoption_authority_cannot_change_inside_ordinary_lineage() {
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let first = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &profile,
            "adopt-a",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        let candidate = subject(&profile, "adopt-b", 2, 5, EvidenceClass::HardwareVerified);
        assert_eq!(
            VerifierProfileAdoptionTransitionV1::successor(candidate, &first).unwrap_err(),
            VerifierProfileAdoptionError::AuthorityRootChanged
        );
    }

    #[test]
    fn conflicting_generation_one_heads_create_distinct_successor_lineages() {
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let first_a = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &profile,
            "adopt-a",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        let first_b = VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            &profile,
            "adopt-b",
            1,
            4,
            EvidenceClass::HardwareVerified,
        ))
        .unwrap();
        assert_ne!(
            first_a.transition_digest().unwrap(),
            first_b.transition_digest().unwrap()
        );
        let next = subject(&profile, "adopt-c", 2, 4, EvidenceClass::HardwareVerified);
        let second_a =
            VerifierProfileAdoptionTransitionV1::successor(next.clone(), &first_a).unwrap();
        let second_b = VerifierProfileAdoptionTransitionV1::successor(next, &first_b).unwrap();
        assert_ne!(second_a.predecessor(), second_b.predecessor());
        assert_ne!(
            second_a.canonical_signing_bytes().unwrap(),
            second_b.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn exact_requirement_scope_canonicalizes_real_requirement_ids() {
        let contract = contract_with_two_requirements();
        let first = contract.requirements()[0].id();
        let second = contract.requirements()[1].id();
        let scope =
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![second, first, second])
                .unwrap();
        match scope {
            VerifierAdoptionScopeV1::Requirements {
                contract_id,
                requirement_ids,
            } => {
                assert_eq!(contract_id, contract.id());
                assert_eq!(requirement_ids, vec![first, second]);
            }
            _ => panic!("expected exact requirement scope"),
        }
        assert_eq!(
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![]).unwrap_err(),
            VerifierProfileAdoptionError::EmptyRequirementScope
        );
    }

    #[test]
    fn noncanonical_transport_scope_is_rejected_by_subject_constructor() {
        let contract = contract_with_two_requirements();
        let first = contract.requirements()[0].id();
        let second = contract.requirements()[1].id();
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let noncanonical = VerifierAdoptionScopeV1::Requirements {
            contract_id: contract.id(),
            requirement_ids: vec![second, first],
        };
        assert_eq!(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-1",
                "organization:test",
                "adoption-root-1",
                [4; 32],
                &profile,
                1,
                1_000,
                2_000,
                EvidenceClass::HardwareVerified,
                noncanonical,
            )
            .unwrap_err(),
            VerifierProfileAdoptionError::NonCanonicalRequirementScope
        );
    }
}
