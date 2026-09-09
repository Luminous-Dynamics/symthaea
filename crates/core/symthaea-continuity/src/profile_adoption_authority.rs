// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded local authority grants for verifier-profile adoption.
//!
//! A trusted adoption-authority key is an identity/authentication fact, not an
//! unlimited capability. This module closes that distinction by requiring an
//! independent local grant envelope that caps both evidence strength and scope.
//!
//! Core theorem:
//!
//! `trusted signer != bounded grant capability != adopted verifier authority`.
//!
//! The grant is deliberately non-Serde. It represents trusted local policy, not a
//! wire claim the signer can manufacture for itself.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::VerifierAdoptionScopeV1;
use crate::profile_adoption_admission::PolicyCheckedVerifierProfileAdoptionV1;
use crate::witness::EvidenceClass;

const GRANT_DOMAIN: &[u8] = b"symthaea.continuity.verifier-adoption-authority-grant.v1\0";

/// Content identity of one exact local verifier-adoption authority grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierAdoptionAuthorityGrantIdV1([u8; 32]);

impl VerifierAdoptionAuthorityGrantIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Local capability envelope limiting what one trusted adoption authority may grant.
///
/// This type is intentionally not serializable. `grant_epoch` is a monotone local
/// policy generation, separate from both root-provisioning epoch and adoption
/// transition generation. Changing this envelope must invalidate future use of an
/// old grant even when the authority key itself remains unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierAdoptionAuthorityGrantV1 {
    grant_id_text: String,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    verifier_role_id: String,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: VerifierAdoptionScopeV1,
    grant_id: VerifierAdoptionAuthorityGrantIdV1,
}

impl VerifierAdoptionAuthorityGrantV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grant_id_text: impl Into<String>,
        authority_subject: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: [u8; 32],
        verifier_role_id: impl Into<String>,
        grant_epoch: u64,
        maximum_evidence_class: EvidenceClass,
        allowed_scope: VerifierAdoptionScopeV1,
    ) -> Result<Self, VerifierAdoptionAuthorityGrantError> {
        let grant_id_text = checked_text("grant_id", grant_id_text.into())?;
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
        let grant_id = VerifierAdoptionAuthorityGrantIdV1(hash_grant(
            &grant_id_text,
            &authority_subject,
            &authority_root_id,
            authority_root_digest,
            &verifier_role_id,
            grant_epoch,
            maximum_evidence_class,
            &allowed_scope,
        ));
        Ok(Self {
            grant_id_text,
            authority_subject,
            authority_root_id,
            authority_root_digest,
            verifier_role_id,
            grant_epoch,
            maximum_evidence_class,
            allowed_scope,
            grant_id,
        })
    }

    pub fn id(&self) -> VerifierAdoptionAuthorityGrantIdV1 {
        self.grant_id
    }

    pub fn grant_id_text(&self) -> &str {
        &self.grant_id_text
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

/// Opaque result that one policy-checked adoption is also inside an independently
/// provisioned authority capability envelope.
///
/// This is the minimum adoption-policy type future cryptographic composition should
/// accept. A bare [`PolicyCheckedVerifierProfileAdoptionV1`] proves local lineage
/// admission but does not prove that the signer was delegated enough grant power.
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

/// Bind a locally admitted adoption to an independent capability envelope.
///
/// `scope_contract` is required whenever either the candidate or authority grant is
/// contract/requirement scoped. It grounds both scope expressions in the exact same
/// validated contract before subset comparison succeeds.
pub fn bind_policy_checked_adoption_to_authority_grant(
    checked: PolicyCheckedVerifierProfileAdoptionV1,
    authority_grant: &VerifierAdoptionAuthorityGrantV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<AuthorityGrantedVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantError> {
    let subject = checked.transition().subject();

    if subject.authority_subject() != authority_grant.authority_subject() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthoritySubjectMismatch);
    }
    if subject.authority_root_id() != authority_grant.authority_root_id() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthorityRootIdMismatch);
    }
    if subject.authority_root_digest() != authority_grant.authority_root_digest() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthorityRootDigestMismatch);
    }
    if subject.verifier_role_id() != authority_grant.verifier_role_id() {
        return Err(VerifierAdoptionAuthorityGrantError::VerifierRoleMismatch);
    }
    if subject.evidence_class_ceiling() > authority_grant.maximum_evidence_class() {
        return Err(
            VerifierAdoptionAuthorityGrantError::EvidenceClassExceedsAuthorityGrant {
                maximum: authority_grant.maximum_evidence_class(),
                requested: subject.evidence_class_ceiling(),
            },
        );
    }

    validate_scope_shape(subject.scope())?;
    validate_scope_shape(authority_grant.allowed_scope())?;
    if !scope_is_subset(subject.scope(), authority_grant.allowed_scope()) {
        return Err(VerifierAdoptionAuthorityGrantError::ScopeExceedsAuthorityGrant);
    }
    ground_scopes_in_contract(
        subject.scope(),
        authority_grant.allowed_scope(),
        scope_contract,
    )?;

    Ok(AuthorityGrantedVerifierProfileAdoptionV1 {
        checked,
        authority_grant: authority_grant.clone(),
    })
}

fn scope_is_subset(candidate: &VerifierAdoptionScopeV1, allowed: &VerifierAdoptionScopeV1) -> bool {
    match allowed {
        VerifierAdoptionScopeV1::AllContinuityVerification => true,
        VerifierAdoptionScopeV1::Contract {
            contract_id: allowed_contract,
        } => match candidate {
            VerifierAdoptionScopeV1::AllContinuityVerification => false,
            VerifierAdoptionScopeV1::Contract { contract_id }
            | VerifierAdoptionScopeV1::Requirements { contract_id, .. } => {
                contract_id == allowed_contract
            }
        },
        VerifierAdoptionScopeV1::Requirements {
            contract_id: allowed_contract,
            requirement_ids: allowed_requirements,
        } => match candidate {
            VerifierAdoptionScopeV1::Requirements {
                contract_id,
                requirement_ids,
            } if contract_id == allowed_contract => requirement_ids
                .iter()
                .all(|candidate_id| allowed_requirements.binary_search(candidate_id).is_ok()),
            _ => false,
        },
    }
}

fn ground_scopes_in_contract(
    candidate: &VerifierAdoptionScopeV1,
    allowed: &VerifierAdoptionScopeV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    if matches!(candidate, VerifierAdoptionScopeV1::AllContinuityVerification)
        && matches!(allowed, VerifierAdoptionScopeV1::AllContinuityVerification)
    {
        return Ok(());
    }

    let contract = scope_contract.ok_or(VerifierAdoptionAuthorityGrantError::MissingScopeContract)?;
    ground_one_scope(candidate, contract, false)?;
    ground_one_scope(allowed, contract, true)?;
    Ok(())
}

fn ground_one_scope(
    scope: &VerifierAdoptionScopeV1,
    contract: &ValidatedContinuityContractV1,
    is_authority_grant: bool,
) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            if contract.id() != *contract_id {
                return Err(VerifierAdoptionAuthorityGrantError::ScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            if contract.id() != *contract_id {
                return Err(VerifierAdoptionAuthorityGrantError::ScopeContractMismatch);
            }
            for requirement_id in requirement_ids {
                if !contract
                    .requirements()
                    .iter()
                    .any(|requirement| requirement.id() == *requirement_id)
                {
                    return Err(if is_authority_grant {
                        VerifierAdoptionAuthorityGrantError::UnknownAuthorityGrantRequirement {
                            requirement: *requirement_id,
                        }
                    } else {
                        VerifierAdoptionAuthorityGrantError::UnknownCandidateRequirement {
                            requirement: *requirement_id,
                        }
                    });
                }
            }
            Ok(())
        }
    }
}

fn validate_scope_shape(
    scope: &VerifierAdoptionScopeV1,
) -> Result<(), VerifierAdoptionAuthorityGrantError> {
    if let VerifierAdoptionScopeV1::Requirements {
        requirement_ids, ..
    } = scope
    {
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
    grant_id_text: &str,
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
    hash_text(&mut hasher, grant_id_text);
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
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            hasher.update(&[2]);
            hasher.update(contract_id.as_bytes());
            hasher.update(&(requirement_ids.len() as u64).to_le_bytes());
            for requirement_id in requirement_ids {
                hasher.update(requirement_id.as_bytes());
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

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierAdoptionAuthorityGrantError> {
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
    #[error("authority-grant exact requirement scope must not be empty")]
    EmptyRequirementScope,
    #[error("authority-grant exact requirement scope must be sorted and unique")]
    NonCanonicalRequirementScope,
    #[error("policy-checked adoption authority subject is outside this authority grant")]
    AuthoritySubjectMismatch,
    #[error("policy-checked adoption authority root id is outside this authority grant")]
    AuthorityRootIdMismatch,
    #[error("policy-checked adoption authority root digest is outside this authority grant")]
    AuthorityRootDigestMismatch,
    #[error("policy-checked adoption verifier role is outside this authority grant")]
    VerifierRoleMismatch,
    #[error("requested evidence class {requested:?} exceeds authority-grant maximum {maximum:?}")]
    EvidenceClassExceedsAuthorityGrant {
        maximum: EvidenceClass,
        requested: EvidenceClass,
    },
    #[error("requested verifier-adoption scope exceeds the local authority grant")]
    ScopeExceedsAuthorityGrant,
    #[error("contract/requirement authority grant requires the exact validated scope contract")]
    MissingScopeContract,
    #[error("authority-grant/candidate scope does not match the supplied exact contract")]
    ScopeContractMismatch,
    #[error("candidate scope names a requirement absent from the exact contract")]
    UnknownCandidateRequirement { requirement: ContinuityRequirementId },
    #[error("local authority grant names a requirement absent from the exact contract")]
    UnknownAuthorityGrantRequirement { requirement: ContinuityRequirementId },
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
    use crate::profile_adoption::{
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_admission::{
        VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionHeadV1,
    };
    use crate::VerifierProfileV1;

    fn profile(class: EvidenceClass) -> VerifierProfileV1 {
        VerifierProfileV1::new("hardware-verifier-v1", [9; 32], 7, class).unwrap()
    }

    fn contract(seed_base: u8) -> crate::ValidatedContinuityContractV1 {
        let mut requirements = Vec::new();
        for offset in [1u8, 2u8] {
            let seed = seed_base.wrapping_add(offset);
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
        ContinuityContractV1::new(
            format!("research-fleet-{seed_base}"),
            [seed_base.wrapping_add(40); 32],
            requirements,
        )
        .unwrap()
        .validate()
        .unwrap()
    }

    fn checked(
        profile: &VerifierProfileV1,
        ceiling: EvidenceClass,
        scope: VerifierAdoptionScopeV1,
    ) -> PolicyCheckedVerifierProfileAdoptionV1 {
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            profile,
            1,
            1_000,
            2_000,
            ceiling,
            scope,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        policy.check(1_500, &transition, profile, None).unwrap_or_else(|error| {
            panic!("test fixture local admission failed unexpectedly: {error}")
        })
    }

    fn checked_scoped(
        profile: &VerifierProfileV1,
        ceiling: EvidenceClass,
        scope: VerifierAdoptionScopeV1,
        contract: &ValidatedContinuityContractV1,
    ) -> PolicyCheckedVerifierProfileAdoptionV1 {
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-scoped",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            profile,
            1,
            1_000,
            2_000,
            ceiling,
            scope,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        policy
            .check(1_500, &transition, profile, Some(contract))
            .unwrap()
    }

    fn grant(
        maximum: EvidenceClass,
        allowed_scope: VerifierAdoptionScopeV1,
    ) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            1,
            maximum,
            allowed_scope,
        )
        .unwrap()
    }

    #[test]
    fn trusted_signer_cannot_choose_its_own_evidence_strength() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let checked = checked(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        let bounded = grant(
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );

        assert_eq!(
            bind_policy_checked_adoption_to_authority_grant(checked, &bounded, None),
            Err(
                VerifierAdoptionAuthorityGrantError::EvidenceClassExceedsAuthorityGrant {
                    maximum: EvidenceClass::DifferentiallyVerified,
                    requested: EvidenceClass::HardwareVerified,
                }
            )
        );
    }

    #[test]
    fn all_scope_grant_can_be_narrowed_by_candidate() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let contract = contract(10);
        let checked = checked_scoped(
            &profile,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::Contract {
                contract_id: contract.id(),
            },
            &contract,
        );
        let broad = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );

        bind_policy_checked_adoption_to_authority_grant(checked, &broad, Some(&contract))
            .unwrap();
    }

    #[test]
    fn contract_grant_rejects_global_candidate_but_allows_requirement_subset() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let contract = contract(10);
        let contract_grant = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract {
                contract_id: contract.id(),
            },
        );

        let global = checked(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        assert_eq!(
            bind_policy_checked_adoption_to_authority_grant(
                global,
                &contract_grant,
                Some(&contract),
            ),
            Err(VerifierAdoptionAuthorityGrantError::ScopeExceedsAuthorityGrant)
        );

        let requirement = contract.requirements()[0].id();
        let narrow = checked_scoped(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![requirement]).unwrap(),
            &contract,
        );
        bind_policy_checked_adoption_to_authority_grant(
            narrow,
            &contract_grant,
            Some(&contract),
        )
        .unwrap();
    }

    #[test]
    fn requirement_grant_requires_candidate_subset() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let contract = contract(10);
        let r1 = contract.requirements()[0].id();
        let r2 = contract.requirements()[1].id();
        let one_requirement_grant = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![r1]).unwrap(),
        );

        let exact = checked_scoped(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![r1]).unwrap(),
            &contract,
        );
        bind_policy_checked_adoption_to_authority_grant(
            exact,
            &one_requirement_grant,
            Some(&contract),
        )
        .unwrap();

        let wider = checked_scoped(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![r1, r2]).unwrap(),
            &contract,
        );
        assert_eq!(
            bind_policy_checked_adoption_to_authority_grant(
                wider,
                &one_requirement_grant,
                Some(&contract),
            ),
            Err(VerifierAdoptionAuthorityGrantError::ScopeExceedsAuthorityGrant)
        );
    }

    #[test]
    fn both_candidate_and_local_grant_require_real_contract_membership() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let contract_a = contract(10);
        let contract_b = contract(20);
        let foreign = contract_b.requirements()[0].id();

        // #1104 correctly refuses a foreign candidate requirement before this layer.
        let candidate_subject = VerifierProfileAdoptionSubjectV1::new(
            "candidate-foreign",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract_a.id(), vec![foreign]).unwrap(),
        )
        .unwrap();
        let candidate_transition =
            VerifierProfileAdoptionTransitionV1::bootstrap(candidate_subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        assert!(policy
            .check(1_500, &candidate_transition, &profile, Some(&contract_a))
            .is_err());

        // A malformed local capability envelope is independently rejected here.
        let local_foreign = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(contract_a.id(), vec![foreign]).unwrap(),
        );
        let valid_candidate = checked_scoped(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::requirements(
                contract_a.id(),
                vec![contract_a.requirements()[0].id()],
            )
            .unwrap(),
            &contract_a,
        );
        assert_eq!(
            bind_policy_checked_adoption_to_authority_grant(
                valid_candidate,
                &local_foreign,
                Some(&contract_a),
            ),
            Err(
                VerifierAdoptionAuthorityGrantError::UnknownAuthorityGrantRequirement {
                    requirement: foreign,
                }
            )
        );
    }

    #[test]
    fn grant_is_bound_to_exact_authority_and_role() {
        let profile = profile(EvidenceClass::HardwareVerified);
        let checked = checked(
            &profile,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        let wrong_root = VerifierAdoptionAuthorityGrantV1::new(
            "grant-wrong-root",
            "organization:test",
            "other-root",
            [0x55; 32],
            "hardware-verifier-v1",
            1,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        assert_eq!(
            bind_policy_checked_adoption_to_authority_grant(checked, &wrong_root, None),
            Err(VerifierAdoptionAuthorityGrantError::AuthorityRootIdMismatch)
        );
    }

    #[test]
    fn grant_identity_changes_with_epoch_strength_or_scope() {
        let contract = contract(10);
        let base = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        let epoch_two = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            2,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let weaker = grant(
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        let narrower = grant(
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract {
                contract_id: contract.id(),
            },
        );

        assert_ne!(base.id(), epoch_two.id());
        assert_ne!(base.id(), weaker.id());
        assert_ne!(base.id(), narrower.id());
    }

    #[test]
    fn direct_noncanonical_local_requirement_scope_is_rejected() {
        let contract = contract(10);
        let mut ids = vec![
            contract.requirements()[0].id(),
            contract.requirements()[1].id(),
        ];
        ids.sort();
        ids.reverse();
        assert!(matches!(
            VerifierAdoptionAuthorityGrantV1::new(
                "grant-bad-scope",
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                "hardware-verifier-v1",
                1,
                EvidenceClass::HardwareVerified,
                VerifierAdoptionScopeV1::Requirements {
                    contract_id: contract.id(),
                    requirement_ids: ids,
                },
            ),
            Err(VerifierAdoptionAuthorityGrantError::NonCanonicalRequirementScope)
        ));
    }
}
