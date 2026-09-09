// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for lineage-bearing verifier-profile adoption.
//!
//! Cryptographic authentication and adoption policy are deliberately separate.
//! A signature can prove which key authenticated an adoption transition; this
//! module proves whether that exact transition is admissible for the local
//! authority subject, provisioned adoption root, logical verifier role, exact
//! verifier profile, persisted lineage head, current time, and declared scope.
//!
//! Success still does not create an authorized verifier profile. The returned
//! [`PolicyCheckedVerifierProfileAdoptionV1`] is non-serializable and must later be
//! paired with verifier-owned cryptographic proof and commit/currentness checks.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::{
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::verifier::{VerifierProfileId, VerifierProfileV1};

/// Exact identity of one locally observed verifier-adoption head.
///
/// Fields are private and production construction derives all of them from one
/// validated transition. Runtime code therefore cannot synthesize a generation
/// from one lineage with a digest, authority root, role, or profile from another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionHeadIdentityV1 {
    generation: u64,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    verifier_role_id: String,
    verifier_profile_id: VerifierProfileId,
}

impl VerifierProfileAdoptionHeadIdentityV1 {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.transition_digest
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
}

/// Persisted verifier-adoption lineage state observed before candidate admission.
///
/// A generation counter alone is not sufficient because conflicting transitions
/// can share a generation. `Current` therefore retains the exact transition
/// digest plus the authority/role/profile identity derived from that transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierProfileAdoptionHeadV1 {
    /// The adoption authority is externally provisioned, but no transition has
    /// been admitted yet. Only generation-1 Bootstrap may enter.
    Uninitialized,
    /// Exact current adoption head.
    Current(VerifierProfileAdoptionHeadIdentityV1),
}

impl VerifierProfileAdoptionHeadV1 {
    /// Derive one internally consistent head from an exact validated transition.
    pub fn from_transition(
        transition: &VerifierProfileAdoptionTransitionV1,
    ) -> Result<Self, VerifierProfileAdoptionError> {
        transition.validate()?;
        let subject = transition.subject();
        Ok(Self::Current(VerifierProfileAdoptionHeadIdentityV1 {
            generation: transition.generation(),
            transition_digest: transition.transition_digest()?,
            authority_subject: subject.authority_subject().to_owned(),
            authority_root_id: subject.authority_root_id().to_owned(),
            authority_root_digest: subject.authority_root_digest(),
            verifier_role_id: subject.verifier_role_id().to_owned(),
            verifier_profile_id: subject.verifier_profile_id(),
        }))
    }

    pub fn identity(&self) -> Option<&VerifierProfileAdoptionHeadIdentityV1> {
        match self {
            Self::Uninitialized => None,
            Self::Current(identity) => Some(identity),
        }
    }
}

/// Trusted local policy inputs for verifier-profile adoption admission.
///
/// Root epoch/currentness is intentionally not modeled here. A later commit/CAS
/// tranche should capture a root provisioning epoch and recheck it immediately
/// before persistence, mirroring the safety-profile authority stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionAdmissionPolicyV1 {
    expected_authority_subject: String,
    trusted_authority_root_id: String,
    trusted_authority_root_digest: [u8; 32],
    expected_verifier_role_id: String,
    current_head: VerifierProfileAdoptionHeadV1,
}

impl VerifierProfileAdoptionAdmissionPolicyV1 {
    pub fn new(
        expected_authority_subject: impl Into<String>,
        trusted_authority_root_id: impl Into<String>,
        trusted_authority_root_digest: [u8; 32],
        expected_verifier_role_id: impl Into<String>,
        current_head: VerifierProfileAdoptionHeadV1,
    ) -> Result<Self, VerifierProfileAdoptionAdmissionError> {
        let expected_authority_subject = checked_policy_text(
            "expected_authority_subject",
            expected_authority_subject.into(),
        )?;
        let trusted_authority_root_id = checked_policy_text(
            "trusted_authority_root_id",
            trusted_authority_root_id.into(),
        )?;
        let expected_verifier_role_id = checked_policy_text(
            "expected_verifier_role_id",
            expected_verifier_role_id.into(),
        )?;
        if trusted_authority_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionAdmissionError::ZeroTrustedAuthorityRootDigest);
        }

        if let Some(identity) = current_head.identity() {
            if identity.authority_subject() != expected_authority_subject {
                return Err(
                    VerifierProfileAdoptionAdmissionError::PersistedHeadAuthoritySubjectMismatch {
                        expected: expected_authority_subject,
                        observed: identity.authority_subject().to_owned(),
                    },
                );
            }
            if identity.authority_root_id() != trusted_authority_root_id {
                return Err(
                    VerifierProfileAdoptionAdmissionError::PersistedHeadAuthorityRootIdMismatch {
                        expected: trusted_authority_root_id,
                        observed: identity.authority_root_id().to_owned(),
                    },
                );
            }
            if identity.authority_root_digest() != trusted_authority_root_digest {
                return Err(
                    VerifierProfileAdoptionAdmissionError::PersistedHeadAuthorityRootMismatch,
                );
            }
            if identity.verifier_role_id() != expected_verifier_role_id {
                return Err(
                    VerifierProfileAdoptionAdmissionError::PersistedHeadVerifierRoleMismatch {
                        expected: expected_verifier_role_id,
                        observed: identity.verifier_role_id().to_owned(),
                    },
                );
            }
        }

        Ok(Self {
            expected_authority_subject,
            trusted_authority_root_id,
            trusted_authority_root_digest,
            expected_verifier_role_id,
            current_head,
        })
    }

    pub fn expected_authority_subject(&self) -> &str {
        &self.expected_authority_subject
    }

    pub fn trusted_authority_root_id(&self) -> &str {
        &self.trusted_authority_root_id
    }

    pub fn trusted_authority_root_digest(&self) -> [u8; 32] {
        self.trusted_authority_root_digest
    }

    pub fn expected_verifier_role_id(&self) -> &str {
        &self.expected_verifier_role_id
    }

    pub fn current_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.current_head
    }

    /// Check one adoption transition against exact local policy/state.
    ///
    /// Validity is half-open: `valid_from <= now < valid_until`.
    ///
    /// `scope_contract` is required for contract- or requirement-scoped grants so
    /// admission can prove the exact contract identity and, for requirement scopes,
    /// membership of every granted requirement. It is unnecessary for
    /// `AllContinuityVerification`.
    pub fn check(
        &self,
        now_unix_ms: u64,
        transition: &VerifierProfileAdoptionTransitionV1,
        profile: &VerifierProfileV1,
        scope_contract: Option<&ValidatedContinuityContractV1>,
    ) -> Result<PolicyCheckedVerifierProfileAdoptionV1, VerifierProfileAdoptionAdmissionError> {
        transition.validate()?;
        let subject = transition.subject();
        subject.validate_against_profile(profile)?;

        if subject.authority_subject() != self.expected_authority_subject {
            return Err(
                VerifierProfileAdoptionAdmissionError::AuthoritySubjectMismatch {
                    expected: self.expected_authority_subject.clone(),
                    observed: subject.authority_subject().to_owned(),
                },
            );
        }
        if subject.authority_root_id() != self.trusted_authority_root_id {
            return Err(
                VerifierProfileAdoptionAdmissionError::TrustedAuthorityRootIdMismatch {
                    expected: self.trusted_authority_root_id.clone(),
                    observed: subject.authority_root_id().to_owned(),
                },
            );
        }
        if subject.authority_root_digest() != self.trusted_authority_root_digest {
            return Err(VerifierProfileAdoptionAdmissionError::TrustedAuthorityRootMismatch);
        }
        if subject.verifier_role_id() != self.expected_verifier_role_id {
            return Err(
                VerifierProfileAdoptionAdmissionError::VerifierRoleMismatch {
                    expected: self.expected_verifier_role_id.clone(),
                    observed: subject.verifier_role_id().to_owned(),
                },
            );
        }
        if profile.profile_name() != self.expected_verifier_role_id {
            return Err(
                VerifierProfileAdoptionAdmissionError::LocalProfileRoleMismatch {
                    expected: self.expected_verifier_role_id.clone(),
                    observed: profile.profile_name().to_owned(),
                },
            );
        }

        if now_unix_ms < subject.valid_from_unix_ms() {
            return Err(VerifierProfileAdoptionAdmissionError::NotYetValid {
                now_unix_ms,
                valid_from_unix_ms: subject.valid_from_unix_ms(),
            });
        }
        if now_unix_ms >= subject.valid_until_unix_ms() {
            return Err(VerifierProfileAdoptionAdmissionError::Expired {
                now_unix_ms,
                valid_until_unix_ms: subject.valid_until_unix_ms(),
            });
        }

        match &self.current_head {
            VerifierProfileAdoptionHeadV1::Uninitialized => {
                if transition.generation() != 1
                    || transition.predecessor() != VerifierProfileAdoptionPredecessorV1::Bootstrap
                {
                    return Err(VerifierProfileAdoptionAdmissionError::ExpectedBootstrap {
                        observed_generation: transition.generation(),
                        observed_predecessor: transition.predecessor(),
                    });
                }
            }
            VerifierProfileAdoptionHeadV1::Current(identity) => {
                let expected_generation = identity.generation().checked_add(1).ok_or(
                    VerifierProfileAdoptionAdmissionError::GenerationExhausted {
                        current: identity.generation(),
                    },
                )?;
                if transition.generation() != expected_generation {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::GenerationNotSuccessor {
                            current: identity.generation(),
                            expected: expected_generation,
                            observed: transition.generation(),
                        },
                    );
                }

                let expected_predecessor =
                    VerifierProfileAdoptionPredecessorV1::Previous(identity.transition_digest());
                if transition.predecessor() != expected_predecessor {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::PredecessorHeadMismatch {
                            expected: expected_predecessor,
                            observed: transition.predecessor(),
                        },
                    );
                }

                if subject.authority_subject() != identity.authority_subject() {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::PersistedLineageAuthoritySubjectMismatch,
                    );
                }
                if subject.authority_root_id() != identity.authority_root_id() {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::PersistedLineageAuthorityRootIdMismatch,
                    );
                }
                if subject.authority_root_digest() != identity.authority_root_digest() {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::PersistedLineageAuthorityRootMismatch,
                    );
                }
                if subject.verifier_role_id() != identity.verifier_role_id() {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::PersistedLineageVerifierRoleMismatch,
                    );
                }
            }
        }

        check_scope(subject.scope(), scope_contract)?;

        let canonical_transition_bytes = transition.canonical_signing_bytes()?;
        let candidate_head = VerifierProfileAdoptionHeadV1::from_transition(transition)?;

        Ok(PolicyCheckedVerifierProfileAdoptionV1 {
            transition: transition.clone(),
            profile: profile.clone(),
            canonical_transition_bytes,
            expected_predecessor_head: self.current_head.clone(),
            candidate_head,
        })
    }
}

/// Non-serializable result of exact local verifier-adoption admission.
///
/// This type has no public constructor and proves no signature. A future Xenia
/// adapter must authenticate [`Self::canonical_transition_bytes`] under the exact
/// externally trusted adoption-authority root before any registry commit can derive
/// an authorized verifier profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCheckedVerifierProfileAdoptionV1 {
    transition: VerifierProfileAdoptionTransitionV1,
    profile: VerifierProfileV1,
    canonical_transition_bytes: Vec<u8>,
    expected_predecessor_head: VerifierProfileAdoptionHeadV1,
    candidate_head: VerifierProfileAdoptionHeadV1,
}

impl PolicyCheckedVerifierProfileAdoptionV1 {
    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 {
        &self.transition
    }

    pub fn profile(&self) -> &VerifierProfileV1 {
        &self.profile
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        &self.canonical_transition_bytes
    }

    /// Exact adoption head that must still be current at commit time.
    pub fn expected_predecessor_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.expected_predecessor_head
    }

    /// Candidate head derived from the exact checked transition.
    pub fn candidate_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.candidate_head
    }

    pub fn candidate_generation(&self) -> u64 {
        self.transition.generation()
    }
}

fn check_scope(
    scope: &VerifierAdoptionScopeV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<(), VerifierProfileAdoptionAdmissionError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            let contract = scope_contract
                .ok_or(VerifierProfileAdoptionAdmissionError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileAdoptionAdmissionError::ScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            let contract = scope_contract
                .ok_or(VerifierProfileAdoptionAdmissionError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileAdoptionAdmissionError::ScopeContractMismatch);
            }
            for requirement_id in requirement_ids {
                if !contract
                    .requirements()
                    .iter()
                    .any(|requirement| requirement.id() == *requirement_id)
                {
                    return Err(
                        VerifierProfileAdoptionAdmissionError::UnknownScopedRequirement {
                            requirement: *requirement_id,
                        },
                    );
                }
            }
            Ok(())
        }
    }
}

fn checked_policy_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionAdmissionError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionAdmissionError::BlankPolicyText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionAdmissionError::PolicyTextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionAdmissionError::PolicyControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

/// Fail-closed local admission failures for verifier-profile adoption.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionAdmissionError {
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error("{field} policy text must not be blank")]
    BlankPolicyText { field: &'static str },
    #[error("{field} policy text exceeds 1024 bytes")]
    PolicyTextTooLong { field: &'static str },
    #[error("{field} policy text contains control characters")]
    PolicyControlCharacters { field: &'static str },
    #[error("trusted verifier-adoption authority root digest must be non-zero")]
    ZeroTrustedAuthorityRootDigest,
    #[error(
        "persisted verifier-adoption head authority subject mismatch: expected {expected}, observed {observed}"
    )]
    PersistedHeadAuthoritySubjectMismatch { expected: String, observed: String },
    #[error(
        "persisted verifier-adoption head authority root id mismatch: expected {expected}, observed {observed}"
    )]
    PersistedHeadAuthorityRootIdMismatch { expected: String, observed: String },
    #[error("persisted verifier-adoption head does not use the externally trusted authority root")]
    PersistedHeadAuthorityRootMismatch,
    #[error(
        "persisted verifier-adoption head role mismatch: expected {expected}, observed {observed}"
    )]
    PersistedHeadVerifierRoleMismatch { expected: String, observed: String },
    #[error(
        "verifier-adoption authority subject mismatch: expected {expected}, observed {observed}"
    )]
    AuthoritySubjectMismatch { expected: String, observed: String },
    #[error(
        "verifier-adoption authority root id mismatch: expected {expected}, observed {observed}"
    )]
    TrustedAuthorityRootIdMismatch { expected: String, observed: String },
    #[error("verifier-adoption authority root digest does not match externally trusted root")]
    TrustedAuthorityRootMismatch,
    #[error("verifier-adoption role mismatch: expected {expected}, observed {observed}")]
    VerifierRoleMismatch { expected: String, observed: String },
    #[error("local verifier profile role mismatch: expected {expected}, observed {observed}")]
    LocalProfileRoleMismatch { expected: String, observed: String },
    #[error(
        "verifier-profile adoption is not yet valid: now {now_unix_ms}, valid from {valid_from_unix_ms}"
    )]
    NotYetValid {
        now_unix_ms: u64,
        valid_from_unix_ms: u64,
    },
    #[error(
        "verifier-profile adoption is expired: now {now_unix_ms}, valid until {valid_until_unix_ms}"
    )]
    Expired {
        now_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("uninitialized verifier-adoption head requires generation-1 Bootstrap transition")]
    ExpectedBootstrap {
        observed_generation: u64,
        observed_predecessor: VerifierProfileAdoptionPredecessorV1,
    },
    #[error("verifier-adoption generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error(
        "verifier-adoption generation must be exact successor of {current}: expected {expected}, observed {observed}"
    )]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
    },
    #[error("verifier-adoption predecessor does not match the exact persisted head")]
    PredecessorHeadMismatch {
        expected: VerifierProfileAdoptionPredecessorV1,
        observed: VerifierProfileAdoptionPredecessorV1,
    },
    #[error("candidate authority subject does not match persisted verifier-adoption lineage")]
    PersistedLineageAuthoritySubjectMismatch,
    #[error("candidate authority root id does not match persisted verifier-adoption lineage")]
    PersistedLineageAuthorityRootIdMismatch,
    #[error("candidate authority root does not match persisted verifier-adoption lineage")]
    PersistedLineageAuthorityRootMismatch,
    #[error("candidate verifier role does not match persisted verifier-adoption lineage")]
    PersistedLineageVerifierRoleMismatch,
    #[error(
        "contract- or requirement-scoped verifier adoption requires the exact validated contract"
    )]
    MissingScopeContract,
    #[error("verifier-adoption scope contract does not match the supplied exact contract")]
    ScopeContractMismatch,
    #[error("verifier-adoption scope names a requirement absent from the exact contract")]
    UnknownScopedRequirement {
        requirement: ContinuityRequirementId,
    },
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
    use crate::witness::EvidenceClass;

    fn profile_named(name: &str, root: u8, epoch: u64) -> VerifierProfileV1 {
        VerifierProfileV1::new(name, [root; 32], epoch, EvidenceClass::HardwareVerified).unwrap()
    }

    fn profile(root: u8, epoch: u64) -> VerifierProfileV1 {
        profile_named("hardware-verifier-v1", root, epoch)
    }

    #[allow(clippy::too_many_arguments)]
    fn subject(
        profile: &VerifierProfileV1,
        adoption_id: &str,
        generation: u64,
        authority_subject: &str,
        authority_root_id: &str,
        authority_root: u8,
        scope: VerifierAdoptionScopeV1,
    ) -> crate::VerifierProfileAdoptionSubjectV1 {
        crate::VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            authority_subject,
            authority_root_id,
            [authority_root; 32],
            profile,
            generation,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            scope,
        )
        .unwrap()
    }

    fn bootstrap_for(
        profile: &VerifierProfileV1,
        adoption_id: &str,
        authority_subject: &str,
        authority_root_id: &str,
        authority_root: u8,
        scope: VerifierAdoptionScopeV1,
    ) -> VerifierProfileAdoptionTransitionV1 {
        VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            profile,
            adoption_id,
            1,
            authority_subject,
            authority_root_id,
            authority_root,
            scope,
        ))
        .unwrap()
    }

    fn bootstrap(
        profile: &VerifierProfileV1,
        adoption_id: &str,
    ) -> VerifierProfileAdoptionTransitionV1 {
        bootstrap_for(
            profile,
            adoption_id,
            "organization:test",
            "adoption-root-1",
            0x55,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
    }

    fn head(transition: &VerifierProfileAdoptionTransitionV1) -> VerifierProfileAdoptionHeadV1 {
        VerifierProfileAdoptionHeadV1::from_transition(transition).unwrap()
    }

    fn policy(
        current_head: VerifierProfileAdoptionHeadV1,
    ) -> VerifierProfileAdoptionAdmissionPolicyV1 {
        VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            current_head,
        )
        .unwrap()
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

    #[test]
    fn current_head_identity_is_derived_from_one_exact_transition() {
        let profile = profile(9, 7);
        let transition = bootstrap(&profile, "adopt-1");
        let current = head(&transition);
        let identity = current.identity().unwrap();

        assert_eq!(identity.generation(), 1);
        assert_eq!(
            identity.transition_digest(),
            transition.transition_digest().unwrap()
        );
        assert_eq!(identity.authority_subject(), "organization:test");
        assert_eq!(identity.authority_root_id(), "adoption-root-1");
        assert_eq!(identity.authority_root_digest(), [0x55; 32]);
        assert_eq!(identity.verifier_role_id(), "hardware-verifier-v1");
        assert_eq!(identity.verifier_profile_id(), profile.id());
    }

    #[test]
    fn exact_bootstrap_is_policy_checked_but_not_authorized() {
        let profile = profile(9, 7);
        let transition = bootstrap(&profile, "adopt-1");
        let checked = policy(VerifierProfileAdoptionHeadV1::Uninitialized)
            .check(1_000, &transition, &profile, None)
            .unwrap();

        assert_eq!(
            checked.expected_predecessor_head(),
            &VerifierProfileAdoptionHeadV1::Uninitialized
        );
        assert_eq!(checked.candidate_generation(), 1);
        assert_eq!(checked.candidate_head(), &head(&transition));
        assert_eq!(checked.profile().id(), profile.id());
        assert_eq!(
            checked.canonical_transition_bytes(),
            transition.canonical_signing_bytes().unwrap()
        );
    }

    #[test]
    fn exact_persisted_head_digest_is_required_for_successor() {
        let profile = profile(9, 7);
        let head_a = bootstrap(&profile, "adopt-a");
        let head_b = bootstrap(&profile, "adopt-b");
        assert_ne!(
            head_a.transition_digest().unwrap(),
            head_b.transition_digest().unwrap()
        );

        let candidate = VerifierProfileAdoptionTransitionV1::successor(
            subject(
                &profile,
                "adopt-2",
                2,
                "organization:test",
                "adoption-root-1",
                0x55,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            ),
            &head_a,
        )
        .unwrap();

        policy(head(&head_a))
            .check(1_500, &candidate, &profile, None)
            .unwrap();

        assert!(matches!(
            policy(head(&head_b)).check(1_500, &candidate, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::PredecessorHeadMismatch { .. })
        ));
    }

    #[test]
    fn authority_mediated_verifier_profile_rotation_is_admissible() {
        let profile_a = profile(9, 7);
        let profile_b = profile(10, 8);
        let first = bootstrap(&profile_a, "adopt-1");
        let second = VerifierProfileAdoptionTransitionV1::successor(
            subject(
                &profile_b,
                "adopt-2",
                2,
                "organization:test",
                "adoption-root-1",
                0x55,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            ),
            &first,
        )
        .unwrap();

        let checked = policy(head(&first))
            .check(1_500, &second, &profile_b, None)
            .unwrap();
        assert_eq!(checked.profile().id(), profile_b.id());
        assert_eq!(
            checked
                .candidate_head()
                .identity()
                .unwrap()
                .verifier_profile_id(),
            profile_b.id()
        );
        assert_ne!(
            checked
                .expected_predecessor_head()
                .identity()
                .unwrap()
                .verifier_profile_id(),
            profile_b.id()
        );
    }

    #[test]
    fn exact_local_profile_rebinding_is_required() {
        let profile_a = profile(9, 7);
        let profile_b = profile(10, 8);
        let transition = bootstrap(&profile_a, "adopt-1");

        assert!(matches!(
            policy(VerifierProfileAdoptionHeadV1::Uninitialized).check(
                1_500,
                &transition,
                &profile_b,
                None
            ),
            Err(VerifierProfileAdoptionAdmissionError::Adoption(
                VerifierProfileAdoptionError::VerifierProfileMismatch
            ))
        ));
    }

    #[test]
    fn policy_constructor_rejects_inconsistent_persisted_authority_or_role() {
        let profile = profile(9, 7);
        let wrong_subject = bootstrap_for(
            &profile,
            "subject",
            "organization:other",
            "adoption-root-1",
            0x55,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        assert!(matches!(
            VerifierProfileAdoptionAdmissionPolicyV1::new(
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                "hardware-verifier-v1",
                head(&wrong_subject),
            ),
            Err(
                VerifierProfileAdoptionAdmissionError::PersistedHeadAuthoritySubjectMismatch { .. }
            )
        ));

        let wrong_root = bootstrap_for(
            &profile,
            "root",
            "organization:test",
            "adoption-root-other",
            0x66,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        assert!(matches!(
            VerifierProfileAdoptionAdmissionPolicyV1::new(
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                "hardware-verifier-v1",
                head(&wrong_root),
            ),
            Err(VerifierProfileAdoptionAdmissionError::PersistedHeadAuthorityRootIdMismatch { .. })
                | Err(VerifierProfileAdoptionAdmissionError::PersistedHeadAuthorityRootMismatch)
        ));

        let other_role_profile = profile_named("other-role", 9, 7);
        let wrong_role = bootstrap_for(
            &other_role_profile,
            "role",
            "organization:test",
            "adoption-root-1",
            0x55,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        );
        assert!(matches!(
            VerifierProfileAdoptionAdmissionPolicyV1::new(
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                "hardware-verifier-v1",
                head(&wrong_role),
            ),
            Err(VerifierProfileAdoptionAdmissionError::PersistedHeadVerifierRoleMismatch { .. })
        ));
    }

    #[test]
    fn authority_root_subject_and_time_all_fail_closed() {
        let profile = profile(9, 7);
        let transition = bootstrap(&profile, "adopt-1");

        let wrong_subject = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:other",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        assert!(matches!(
            wrong_subject.check(1_500, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::AuthoritySubjectMismatch { .. })
        ));

        let wrong_root_id = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-other",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        assert!(matches!(
            wrong_root_id.check(1_500, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::TrustedAuthorityRootIdMismatch { .. })
        ));

        let wrong_root = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x66; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        assert_eq!(
            wrong_root.check(1_500, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::TrustedAuthorityRootMismatch)
        );

        let exact = policy(VerifierProfileAdoptionHeadV1::Uninitialized);
        assert!(matches!(
            exact.check(999, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::NotYetValid { .. })
        ));
        assert!(matches!(
            exact.check(2_000, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::Expired { .. })
        ));
    }

    #[test]
    fn contract_scope_requires_the_exact_validated_contract() {
        let profile = profile(9, 7);
        let contract_a = contract(10);
        let contract_b = contract(20);
        let transition = bootstrap_for(
            &profile,
            "contract-scope",
            "organization:test",
            "adoption-root-1",
            0x55,
            VerifierAdoptionScopeV1::Contract {
                contract_id: contract_a.id(),
            },
        );
        let exact = policy(VerifierProfileAdoptionHeadV1::Uninitialized);

        exact
            .check(1_500, &transition, &profile, Some(&contract_a))
            .unwrap();
        assert_eq!(
            exact.check(1_500, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::MissingScopeContract)
        );
        assert_eq!(
            exact.check(1_500, &transition, &profile, Some(&contract_b)),
            Err(VerifierProfileAdoptionAdmissionError::ScopeContractMismatch)
        );
    }

    #[test]
    fn requirement_scope_requires_real_membership_in_exact_contract() {
        let profile = profile(9, 7);
        let contract_a = contract(10);
        let contract_b = contract(20);
        let required = contract_a.requirements()[0].id();
        let exact_scope =
            VerifierAdoptionScopeV1::requirements(contract_a.id(), vec![required]).unwrap();
        let transition = bootstrap_for(
            &profile,
            "requirement-scope",
            "organization:test",
            "adoption-root-1",
            0x55,
            exact_scope,
        );
        policy(VerifierProfileAdoptionHeadV1::Uninitialized)
            .check(1_500, &transition, &profile, Some(&contract_a))
            .unwrap();

        let foreign_requirement = contract_b.requirements()[0].id();
        let forged_scope =
            VerifierAdoptionScopeV1::requirements(contract_a.id(), vec![foreign_requirement])
                .unwrap();
        let forged = bootstrap_for(
            &profile,
            "foreign-requirement",
            "organization:test",
            "adoption-root-1",
            0x55,
            forged_scope,
        );
        assert_eq!(
            policy(VerifierProfileAdoptionHeadV1::Uninitialized).check(
                1_500,
                &forged,
                &profile,
                Some(&contract_a)
            ),
            Err(
                VerifierProfileAdoptionAdmissionError::UnknownScopedRequirement {
                    requirement: foreign_requirement,
                }
            )
        );
    }

    #[test]
    fn generation_overflow_fails_closed() {
        let profile = profile(9, 7);
        let transition = bootstrap(&profile, "adopt-1");
        let forged_boundary_head =
            VerifierProfileAdoptionHeadV1::Current(VerifierProfileAdoptionHeadIdentityV1 {
                generation: u64::MAX,
                transition_digest: transition.transition_digest().unwrap(),
                authority_subject: "organization:test".to_owned(),
                authority_root_id: "adoption-root-1".to_owned(),
                authority_root_digest: [0x55; 32],
                verifier_role_id: "hardware-verifier-v1".to_owned(),
                verifier_profile_id: profile.id(),
            });

        assert_eq!(
            policy(forged_boundary_head).check(1_500, &transition, &profile, None),
            Err(VerifierProfileAdoptionAdmissionError::GenerationExhausted { current: u64::MAX })
        );
    }
}
