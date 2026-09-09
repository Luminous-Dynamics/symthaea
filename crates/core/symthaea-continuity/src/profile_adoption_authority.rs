// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded local capability for verifier-profile adoption authorities.
//!
//! A trusted adoption signer answers who may authenticate an adoption transition.
//! It does not answer how much verifier authority that signer may grant. This module
//! keeps those powers separate with a non-Serde local capability envelope.
//!
//! Core theorem:
//!
//! `trusted signer != bounded grant capability != current verifier authority`.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::VerifierAdoptionScopeV1;
use crate::profile_adoption_admission::PolicyCheckedVerifierProfileAdoptionV1;
use crate::witness::EvidenceClass;

const GRANT_DOMAIN: &[u8] = b"symthaea.continuity.verifier-adoption-authority-grant.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierAdoptionAuthorityGrantIdV1([u8; 32]);

impl VerifierAdoptionAuthorityGrantIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact local capability envelope for one verifier-adoption signer/role slot.
///
/// This type is deliberately non-Serde. `grant_epoch` is a local monotone policy
/// generation and is independent of both adoption generation and root provisioning
/// epoch. Replacing/narrowing this capability can therefore invalidate future use
/// without rotating the signer key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierAdoptionAuthorityGrantV1 {
    grant_slot_id: String,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    verifier_role_id: String,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: VerifierAdoptionScopeV1,
    id: VerifierAdoptionAuthorityGrantIdV1,
}

impl VerifierAdoptionAuthorityGrantV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grant_slot_id: impl Into<String>,
        authority_subject: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: [u8; 32],
        verifier_role_id: impl Into<String>,
        grant_epoch: u64,
        maximum_evidence_class: EvidenceClass,
        allowed_scope: VerifierAdoptionScopeV1,
    ) -> Result<Self, VerifierAdoptionAuthorityGrantError> {
        let grant_slot_id = checked_text("grant_slot_id", grant_slot_id.into())?;
        let authority_subject = checked_text("authority_subject", authority_subject.into())?;
        let authority_root_id = checked_text("authority_root_id", authority_root_id.into())?;
        let verifier_role_id = checked_text("verifier_role_id", verifier_role_id.into())?;
        if authority_root_digest == [0; 32] {
            return Err(VerifierAdoptionAuthorityGrantError::ZeroAuthorityRootDigest);
        }
        if grant_epoch == 0 {
            return Err(VerifierAdoptionAuthorityGrantError::ZeroGrantEpoch);
        }
        validate_scope_shape(&allowed_scope)?;
        let id = VerifierAdoptionAuthorityGrantIdV1(hash_grant(
            &grant_slot_id,
            &authority_subject,
            &authority_root_id,
            authority_root_digest,
            &verifier_role_id,
            grant_epoch,
            maximum_evidence_class,
            &allowed_scope,
        ));
        Ok(Self {
            grant_slot_id,
            authority_subject,
            authority_root_id,
            authority_root_digest,
            verifier_role_id,
            grant_epoch,
            maximum_evidence_class,
            allowed_scope,
            id,
        })
    }

    pub fn validate(&self) -> Result<(), VerifierAdoptionAuthorityGrantError> {
        checked_text("grant_slot_id", self.grant_slot_id.clone())?;
        checked_text("authority_subject", self.authority_subject.clone())?;
        checked_text("authority_root_id", self.authority_root_id.clone())?;
        checked_text("verifier_role_id", self.verifier_role_id.clone())?;
        if self.authority_root_digest == [0; 32] {
            return Err(VerifierAdoptionAuthorityGrantError::ZeroAuthorityRootDigest);
        }
        if self.grant_epoch == 0 {
            return Err(VerifierAdoptionAuthorityGrantError::ZeroGrantEpoch);
        }
        validate_scope_shape(&self.allowed_scope)?;
        let expected = VerifierAdoptionAuthorityGrantIdV1(hash_grant(
            &self.grant_slot_id,
            &self.authority_subject,
            &self.authority_root_id,
            self.authority_root_digest,
            &self.verifier_role_id,
            self.grant_epoch,
            self.maximum_evidence_class,
            &self.allowed_scope,
        ));
        if expected != self.id {
            return Err(VerifierAdoptionAuthorityGrantError::GrantIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> VerifierAdoptionAuthorityGrantIdV1 {
        self.id
    }
    pub fn grant_slot_id(&self) -> &str {
        &self.grant_slot_id
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
    pub fn grant_epoch(&self) -> u64 {
        self.grant_epoch
    }
    pub fn maximum_evidence_class(&self) -> EvidenceClass {
        self.maximum_evidence_class
    }
    pub fn allowed_scope(&self) -> &VerifierAdoptionScopeV1 {
        &self.allowed_scope
    }
}

/// Admission result additionally bounded by the local signer's grant capability.
///
/// Still non-cryptographic and non-persistent. A future registry write must combine
/// this value with the root/head/time preconditions and exact Xenia proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityGrantedVerifierProfileAdoptionV1 {
    checked: PolicyCheckedVerifierProfileAdoptionV1,
    authority_grant: VerifierAdoptionAuthorityGrantV1,
}

impl AuthorityGrantedVerifierProfileAdoptionV1 {
    pub fn checked(&self) -> &PolicyCheckedVerifierProfileAdoptionV1 {
        &self.checked
    }
    pub fn authority_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 {
        &self.authority_grant
    }
    pub fn canonical_transition_bytes(&self) -> &[u8] {
        self.checked.canonical_transition_bytes()
    }
}

pub fn bind_policy_checked_adoption_to_authority_grant(
    checked: PolicyCheckedVerifierProfileAdoptionV1,
    authority_grant: &VerifierAdoptionAuthorityGrantV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<AuthorityGrantedVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantError> {
    authority_grant.validate()?;
    let subject = checked.transition().subject();
    require_same_authority(
        subject.authority_subject(),
        subject.authority_root_id(),
        subject.authority_root_digest(),
        subject.verifier_role_id(),
        authority_grant,
    )?;
    if subject.evidence_class_ceiling() > authority_grant.maximum_evidence_class() {
        return Err(VerifierAdoptionAuthorityGrantError::EvidenceClassExceedsAuthorityGrant {
            maximum: authority_grant.maximum_evidence_class(),
            requested: subject.evidence_class_ceiling(),
        });
    }
    ground_scope(subject.scope(), scope_contract, ScopeOwner::Candidate)?;
    ground_scope(authority_grant.allowed_scope(), scope_contract, ScopeOwner::Grant)?;
    if !scope_is_subset(subject.scope(), authority_grant.allowed_scope()) {
        return Err(VerifierAdoptionAuthorityGrantError::ScopeExceedsAuthorityGrant);
    }
    Ok(AuthorityGrantedVerifierProfileAdoptionV1 {
        checked,
        authority_grant: authority_grant.clone(),
    })
}

pub(crate) fn require_same_authority(
    authority_subject: &str,
    authority_root_id: &str,
    authority_root_digest: [u8; 32],
    verifier_role_id: &str,
    grant: &VerifierAdoptionAuthorityGrantV1,
) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    if authority_subject != grant.authority_subject() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthoritySubjectMismatch);
    }
    if authority_root_id != grant.authority_root_id() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthorityRootIdMismatch);
    }
    if authority_root_digest != grant.authority_root_digest() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthorityRootDigestMismatch);
    }
    if verifier_role_id != grant.verifier_role_id() {
        return Err(VerifierAdoptionAuthorityGrantError::VerifierRoleMismatch);
    }
    Ok(())
}

pub(crate) fn intersect_scopes(
    left: &VerifierAdoptionScopeV1,
    right: &VerifierAdoptionScopeV1,
) -> Option<VerifierAdoptionScopeV1> {
    match (left, right) {
        (VerifierAdoptionScopeV1::AllContinuityVerification, other)
        | (other, VerifierAdoptionScopeV1::AllContinuityVerification) => Some(other.clone()),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Contract { contract_id: b },
        ) if a == b => Some(VerifierAdoptionScopeV1::Contract { contract_id: *a }),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids },
        )
        | (
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids },
            VerifierAdoptionScopeV1::Contract { contract_id: a },
        ) if a == b => VerifierAdoptionScopeV1::requirements(*a, requirement_ids.clone()).ok(),
        (
            VerifierAdoptionScopeV1::Requirements { contract_id: a, requirement_ids: left },
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids: right },
        ) if a == b => {
            let overlap: Vec<_> = left
                .iter()
                .copied()
                .filter(|id| right.binary_search(id).is_ok())
                .collect();
            if overlap.is_empty() {
                None
            } else {
                VerifierAdoptionScopeV1::requirements(*a, overlap).ok()
            }
        }
        _ => None,
    }
}

pub(crate) fn ground_scope(
    scope: &VerifierAdoptionScopeV1,
    contract: Option<&ValidatedContinuityContractV1>,
    owner: ScopeOwner,
) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            let contract = contract.ok_or(VerifierAdoptionAuthorityGrantError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierAdoptionAuthorityGrantError::ScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids } => {
            let contract = contract.ok_or(VerifierAdoptionAuthorityGrantError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierAdoptionAuthorityGrantError::ScopeContractMismatch);
            }
            for requirement in requirement_ids {
                if !contract.requirements().iter().any(|item| item.id() == *requirement) {
                    return Err(match owner {
                        ScopeOwner::Grant => VerifierAdoptionAuthorityGrantError::UnknownGrantRequirement { requirement: *requirement },
                        ScopeOwner::Candidate => VerifierAdoptionAuthorityGrantError::UnknownCandidateRequirement { requirement: *requirement },
                    });
                }
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ScopeOwner {
    Grant,
    Candidate,
}

fn scope_is_subset(candidate: &VerifierAdoptionScopeV1, allowed: &VerifierAdoptionScopeV1) -> bool {
    match allowed {
        VerifierAdoptionScopeV1::AllContinuityVerification => true,
        VerifierAdoptionScopeV1::Contract { contract_id: allowed_contract } => match candidate {
            VerifierAdoptionScopeV1::AllContinuityVerification => false,
            VerifierAdoptionScopeV1::Contract { contract_id }
            | VerifierAdoptionScopeV1::Requirements { contract_id, .. } => contract_id == allowed_contract,
        },
        VerifierAdoptionScopeV1::Requirements { contract_id: allowed_contract, requirement_ids: allowed } => match candidate {
            VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids }
                if contract_id == allowed_contract => requirement_ids.iter().all(|id| allowed.binary_search(id).is_ok()),
            _ => false,
        },
    }
}

fn validate_scope_shape(scope: &VerifierAdoptionScopeV1) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    if let VerifierAdoptionScopeV1::Requirements { requirement_ids, .. } = scope {
        if requirement_ids.is_empty() {
            return Err(VerifierAdoptionAuthorityGrantError::EmptyRequirementScope);
        }
        if requirement_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(VerifierAdoptionAuthorityGrantError::NonCanonicalRequirementScope);
        }
    }
    Ok(())
}

fn hash_grant(
    grant_slot_id: &str,
    authority_subject: &str,
    authority_root_id: &str,
    authority_root_digest: [u8; 32],
    verifier_role_id: &str,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: &VerifierAdoptionScopeV1,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(GRANT_DOMAIN);
    hash_text(&mut hasher, grant_slot_id);
    hash_text(&mut hasher, authority_subject);
    hash_text(&mut hasher, authority_root_id);
    hasher.update(&authority_root_digest);
    hash_text(&mut hasher, verifier_role_id);
    hasher.update(&grant_epoch.to_le_bytes());
    hasher.update(&[evidence_class_tag(maximum_evidence_class)]);
    hash_scope(&mut hasher, allowed_scope);
    *hasher.finalize().as_bytes()
}

fn hash_scope(hasher: &mut blake3::Hasher, scope: &VerifierAdoptionScopeV1) {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => {
            hasher.update(&[0]);
        }
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            hasher.update(&[1]);
            hasher.update(contract_id.as_bytes());
        }
        VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids } => {
            hasher.update(&[2]);
            hasher.update(contract_id.as_bytes());
            hasher.update(&(requirement_ids.len() as u64).to_le_bytes());
            for requirement in requirement_ids {
                hasher.update(requirement.as_bytes());
            }
        }
    }
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

fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn checked_text(field: &'static str, value: String) -> Result<String, VerifierAdoptionAuthorityGrantError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierAdoptionAuthorityGrantError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierAdoptionAuthorityGrantError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierAdoptionAuthorityGrantError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierAdoptionAuthorityGrantError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("authority-grant root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("authority-grant epoch must be greater than zero")]
    ZeroGrantEpoch,
    #[error("stored verifier-adoption authority grant identity is not canonical")]
    GrantIdentityMismatch,
    #[error("authority grant belongs to a different subject")]
    AuthoritySubjectMismatch,
    #[error("authority grant belongs to a different root id")]
    AuthorityRootIdMismatch,
    #[error("authority grant belongs to different root key material")]
    AuthorityRootDigestMismatch,
    #[error("authority grant belongs to a different verifier role")]
    VerifierRoleMismatch,
    #[error("requested evidence class {requested:?} exceeds authority-grant maximum {maximum:?}")]
    EvidenceClassExceedsAuthorityGrant { maximum: EvidenceClass, requested: EvidenceClass },
    #[error("requested verifier scope exceeds the local authority grant")]
    ScopeExceedsAuthorityGrant,
    #[error("contract/requirement scope requires the exact validated contract")]
    MissingScopeContract,
    #[error("scope references a different continuity contract")]
    ScopeContractMismatch,
    #[error("authority grant references requirement outside the exact contract: {requirement:?}")]
    UnknownGrantRequirement { requirement: ContinuityRequirementId },
    #[error("candidate adoption references requirement outside the exact contract: {requirement:?}")]
    UnknownCandidateRequirement { requirement: ContinuityRequirementId },
    #[error("exact requirement scope must not be empty")]
    EmptyRequirementScope,
    #[error("exact requirement scope must be sorted and unique")]
    NonCanonicalRequirementScope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_adoption::{VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1};
    use crate::profile_adoption_admission::{VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionHeadV1};
    use crate::verifier::VerifierProfileV1;

    #[test]
    fn trusted_signer_cannot_choose_strength_above_local_grant() {
        let profile = VerifierProfileV1::new("hardware-verifier-v1", [9; 32], 7, EvidenceClass::HardwareVerified).unwrap();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1", "organization:test", "adoption-root-1", [0x55; 32], &profile,
            1, 1_000, 2_000, EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        ).unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap().check(1_500, &transition, &profile, None).unwrap();
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1", "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", 1, EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        ).unwrap();
        assert!(matches!(
            bind_policy_checked_adoption_to_authority_grant(checked, &grant, None),
            Err(VerifierAdoptionAuthorityGrantError::EvidenceClassExceedsAuthorityGrant { .. })
        ));
    }
}
