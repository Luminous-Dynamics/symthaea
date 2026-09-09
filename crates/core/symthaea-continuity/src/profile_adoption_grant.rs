// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent local capability grants for verifier-profile adoption.
//!
//! A provisioned adoption-authority root identifies which authority may sign an
//! adoption transition. It does not, by itself, define how much evidence strength
//! or scope that authority is locally delegated to grant.
//!
//! Core theorem:
//!
//! `trusted adoption root != bounded grant capability != adopted verifier authority`.
//!
//! Grants are intentionally non-Serde local policy objects. The wire adoption
//! authority cannot manufacture or widen this envelope for itself.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::VerifierAdoptionScopeV1;
use crate::profile_adoption_root::{
    RootBoundPolicyCheckedVerifierProfileAdoptionV1, VerifierProfileAdoptionAuthorityRootSnapshotV1,
};
use crate::witness::EvidenceClass;

const GRANT_DOMAIN: &[u8] = b"symthaea.continuity.verifier-adoption-authority-grant.canonical.v1\0";

/// Content identity of one exact locally provisioned grant capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierAdoptionAuthorityGrantIdV1([u8; 32]);

impl VerifierAdoptionAuthorityGrantIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Local capability envelope limiting what one exact adoption-authority root
/// provisioning snapshot may grant to one logical verifier role.
///
/// `grant_epoch` is a local monotone policy generation distinct from both the
/// authority-root provisioning epoch and verifier-adoption transition generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierAdoptionAuthorityGrantV1 {
    grant_slot_id: String,
    authority_root_snapshot: VerifierProfileAdoptionAuthorityRootSnapshotV1,
    verifier_role_id: String,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: VerifierAdoptionScopeV1,
    grant_id: VerifierAdoptionAuthorityGrantIdV1,
}

impl VerifierAdoptionAuthorityGrantV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grant_slot_id: impl Into<String>,
        authority_root_snapshot: VerifierProfileAdoptionAuthorityRootSnapshotV1,
        verifier_role_id: impl Into<String>,
        grant_epoch: u64,
        maximum_evidence_class: EvidenceClass,
        allowed_scope: VerifierAdoptionScopeV1,
    ) -> Result<Self, VerifierAdoptionAuthorityGrantError> {
        let grant_slot_id = checked_text("grant_slot_id", grant_slot_id.into())?;
        let verifier_role_id = checked_text("verifier_role_id", verifier_role_id.into())?;
        if grant_epoch == 0 {
            return Err(VerifierAdoptionAuthorityGrantError::ZeroGrantEpoch);
        }
        validate_scope_shape(&allowed_scope)?;

        let grant_id = VerifierAdoptionAuthorityGrantIdV1(hash_grant(
            &grant_slot_id,
            &authority_root_snapshot,
            &verifier_role_id,
            grant_epoch,
            maximum_evidence_class,
            &allowed_scope,
        ));

        Ok(Self {
            grant_slot_id,
            authority_root_snapshot,
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

    pub fn grant_slot_id(&self) -> &str {
        &self.grant_slot_id
    }

    pub fn authority_root_snapshot(&self) -> &VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        &self.authority_root_snapshot
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

/// Opaque result proving that a root-bound, policy-checked adoption also fits
/// inside one independently provisioned local grant capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantBoundRootBoundVerifierProfileAdoptionV1 {
    root_bound: RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    authority_grant: VerifierAdoptionAuthorityGrantV1,
}

impl GrantBoundRootBoundVerifierProfileAdoptionV1 {
    pub fn root_bound(&self) -> &RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
        &self.root_bound
    }

    pub fn authority_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 {
        &self.authority_grant
    }

    pub fn effective_evidence_class(&self) -> EvidenceClass {
        // The adoption subject was already validated against the verifier profile,
        // and binding below proves the subject ceiling is <= local grant ceiling.
        self.root_bound
            .transition()
            .subject()
            .evidence_class_ceiling()
    }

    pub fn effective_scope(&self) -> &VerifierAdoptionScopeV1 {
        // Binding proves the adopted scope is a subset of the local grant scope.
        self.root_bound.transition().subject().scope()
    }
}

/// Bind one root-bound adoption to an independent local grant capability.
///
/// `scope_contract` is required whenever either the adopted or locally granted
/// scope is contract/requirement-specific. Both trusted policy and candidate scope
/// are grounded in the exact validated contract before subset comparison.
pub fn bind_root_bound_adoption_to_authority_grant(
    root_bound: RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    authority_grant: &VerifierAdoptionAuthorityGrantV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<GrantBoundRootBoundVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantError> {
    let subject = root_bound.transition().subject();

    if root_bound.authority_root_snapshot() != authority_grant.authority_root_snapshot() {
        return Err(VerifierAdoptionAuthorityGrantError::AuthorityRootSnapshotMismatch);
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
    ground_scopes_in_contract(
        subject.scope(),
        authority_grant.allowed_scope(),
        scope_contract,
    )?;
    if !scope_is_subset(subject.scope(), authority_grant.allowed_scope()) {
        return Err(VerifierAdoptionAuthorityGrantError::ScopeExceedsAuthorityGrant);
    }

    Ok(GrantBoundRootBoundVerifierProfileAdoptionV1 {
        root_bound,
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
    if matches!(
        candidate,
        VerifierAdoptionScopeV1::AllContinuityVerification
    ) && matches!(allowed, VerifierAdoptionScopeV1::AllContinuityVerification)
    {
        return Ok(());
    }

    let contract =
        scope_contract.ok_or(VerifierAdoptionAuthorityGrantError::MissingScopeContract)?;
    ground_one_scope(allowed, contract, true)?;
    ground_one_scope(candidate, contract, false)?;
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
    grant_slot_id: &str,
    authority_root_snapshot: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
    verifier_role_id: &str,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: &VerifierAdoptionScopeV1,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(GRANT_DOMAIN);
    hash_text(&mut hasher, grant_slot_id);
    hasher.update(authority_root_snapshot.id().as_bytes());
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

fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierAdoptionAuthorityGrantError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("verifier-adoption authority grant epoch must be greater than zero")]
    ZeroGrantEpoch,
    #[error("authority grant belongs to a different root provisioning snapshot")]
    AuthorityRootSnapshotMismatch,
    #[error("authority grant belongs to a different logical verifier role")]
    VerifierRoleMismatch,
    #[error(
        "requested evidence class {requested:?} exceeds local authority-grant maximum {maximum:?}"
    )]
    EvidenceClassExceedsAuthorityGrant {
        maximum: EvidenceClass,
        requested: EvidenceClass,
    },
    #[error("exact contract context is required for scoped verifier authority")]
    MissingScopeContract,
    #[error("verifier authority scope references a different continuity contract")]
    ScopeContractMismatch,
    #[error("authority grant references requirement outside the exact contract: {requirement:?}")]
    UnknownAuthorityGrantRequirement {
        requirement: ContinuityRequirementId,
    },
    #[error(
        "adoption candidate references requirement outside the exact contract: {requirement:?}"
    )]
    UnknownCandidateRequirement {
        requirement: ContinuityRequirementId,
    },
    #[error("requirement-scoped authority grant must not be empty")]
    EmptyRequirementScope,
    #[error("requirement-scoped authority grant must be strictly sorted and unique")]
    NonCanonicalRequirementScope,
    #[error("adoption candidate scope exceeds the independent local authority grant")]
    ScopeExceedsAuthorityGrant,
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
    use crate::verifier::VerifierProfileV1;

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            epoch,
        )
        .unwrap()
    }

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [1; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            "role:research",
            "requires",
            "capability:cuda",
            DependencyBasis::Observed,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            "cuda-workflow",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "cuda-fixture-v1".into(),
            },
            ApprovalBasis::ExplicitPolicy,
            [2; 32],
        )
        .unwrap();
        ContinuityContractV1::new("research-fleet", [3; 32], vec![requirement])
            .unwrap()
            .validate()
            .unwrap()
    }

    fn root_bound(
        root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
        scope: VerifierAdoptionScopeV1,
        ceiling: EvidenceClass,
    ) -> RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
        let profile = profile();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            &profile,
            1,
            1_000,
            2_000,
            ceiling,
            scope.clone(),
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile.profile_name(),
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        let scope_contract = match scope {
            VerifierAdoptionScopeV1::AllContinuityVerification => None,
            _ => panic!("scoped fixture must be built by dedicated test"),
        };
        let checked = policy
            .check(1_500, &transition, &profile, scope_contract)
            .unwrap();
        RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap()
    }

    #[test]
    fn exact_local_grant_binds_root_bound_adoption() {
        let root = root(4);
        let adoption = root_bound(
            &root,
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::DifferentiallyVerified,
        );
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root,
            "hardware-verifier-v1",
            1,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();

        let bound = bind_root_bound_adoption_to_authority_grant(adoption, &grant, None).unwrap();
        assert_eq!(
            bound.effective_evidence_class(),
            EvidenceClass::DifferentiallyVerified
        );
    }

    #[test]
    fn trusted_root_does_not_imply_unlimited_grant_strength() {
        let root = root(4);
        let adoption = root_bound(
            &root,
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::HardwareVerified,
        );
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root,
            "hardware-verifier-v1",
            1,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();

        assert!(matches!(
            bind_root_bound_adoption_to_authority_grant(adoption, &grant, None),
            Err(VerifierAdoptionAuthorityGrantError::EvidenceClassExceedsAuthorityGrant { .. })
        ));
    }

    #[test]
    fn grant_is_bound_to_exact_root_provisioning_epoch() {
        let root_a = root(4);
        let root_b = root(5);
        let adoption = root_bound(
            &root_a,
            VerifierAdoptionScopeV1::AllContinuityVerification,
            EvidenceClass::Observed,
        );
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root_b,
            "hardware-verifier-v1",
            1,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();

        assert_eq!(
            bind_root_bound_adoption_to_authority_grant(adoption, &grant, None).unwrap_err(),
            VerifierAdoptionAuthorityGrantError::AuthorityRootSnapshotMismatch
        );
    }

    #[test]
    fn provisioning_epoch_changes_grant_identity_even_for_same_key() {
        let a = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root(4),
            "hardware-verifier-v1",
            1,
            EvidenceClass::Observed,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let b = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root(5),
            "hardware-verifier-v1",
            1,
            EvidenceClass::Observed,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn requirement_scope_is_grounded_in_exact_contract() {
        let contract = contract();
        let requirement = contract.requirements()[0].id();
        let root = root(4);
        let profile = profile();
        let candidate_scope =
            VerifierAdoptionScopeV1::requirements(contract.id(), vec![requirement]).unwrap();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::Observed,
            candidate_scope.clone(),
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile.profile_name(),
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        let checked = policy
            .check(1_500, &transition, &profile, Some(&contract))
            .unwrap();
        let root_bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap();
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root,
            profile.profile_name(),
            1,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract {
                contract_id: contract.id(),
            },
        )
        .unwrap();

        let bound =
            bind_root_bound_adoption_to_authority_grant(root_bound, &grant, Some(&contract))
                .unwrap();
        assert_eq!(bound.effective_scope(), &candidate_scope);
    }
}
