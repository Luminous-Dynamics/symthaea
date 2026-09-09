// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact verifier-profile binding for continuity verification policy.
//!
//! A human-readable verifier name is diagnostic metadata, not a trust root. The
//! public policy surface in this module binds the witness-layer policy to the exact
//! `VerifierProfileId`, which commits the verifier root digest, root epoch, evidence
//! class, and profile name.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::verifier::{
    ProfilePolicyCheckedVerificationEvidenceV1, VerificationAdmissionError,
    VerificationEvidenceClaimV1, VerifierProfileId, VerifierProfileV1,
    policy_check_verification_evidence,
};
use crate::witness::{VerificationPolicyEntryV1, VerificationPolicyV1, WitnessError};
use crate::{TargetRealizationId, ValidatedContinuityContractV1};

const EXACT_POLICY_DOMAIN: &[u8] = b"symthaea.continuity.exact-verification-policy.v1\0";

/// Content identity of a verification policy bound to one exact verifier profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExactVerificationPolicyId([u8; 32]);

impl ExactVerificationPolicyId {
    /// Return the raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Constructor-only policy bound to one exact verifier-profile identity.
///
/// This type intentionally implements neither `Serialize` nor `Deserialize`.
/// Transport/config bytes therefore cannot manufacture the trust-bearing wrapper;
/// callers reconstruct it from a validated `VerifierProfileV1` plus explicit
/// requirement evidence floors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactVerificationPolicyV1 {
    verifier_profile_id: VerifierProfileId,
    verifier_profile_name: String,
    inner: VerificationPolicyV1,
    exact_policy_id: ExactVerificationPolicyId,
}

impl ExactVerificationPolicyV1 {
    /// Construct a policy for one exact validated verifier profile and requirement set.
    pub fn new(
        profile: &VerifierProfileV1,
        entries: Vec<VerificationPolicyEntryV1>,
    ) -> Result<Self, ExactVerificationPolicyError> {
        profile.validate()?;
        let inner = VerificationPolicyV1::new(profile.profile_name(), entries)?;
        let verifier_profile_id = profile.id();
        let verifier_profile_name = profile.profile_name().to_string();
        let exact_policy_id = ExactVerificationPolicyId(hash_exact_policy(
            inner.id().as_bytes(),
            verifier_profile_id.as_bytes(),
        ));
        Ok(Self {
            verifier_profile_id,
            verifier_profile_name,
            inner,
            exact_policy_id,
        })
    }

    /// Stable exact policy identity.
    pub fn id(&self) -> ExactVerificationPolicyId {
        self.exact_policy_id
    }

    /// Exact verifier profile snapshot required by this policy.
    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    /// Human-readable verifier profile name retained for diagnostics only.
    pub fn verifier_profile_name(&self) -> &str {
        &self.verifier_profile_name
    }

    /// Requirement-specific evidence floors.
    pub fn entries(&self) -> &[VerificationPolicyEntryV1] {
        self.inner.entries()
    }

    /// Require the currently provisioned profile to be the exact profile pinned here.
    pub fn validate_against_profile(
        &self,
        profile: &VerifierProfileV1,
    ) -> Result<(), ExactVerificationPolicyError> {
        profile.validate()?;
        if self.verifier_profile_id != profile.id() {
            return Err(ExactVerificationPolicyError::VerifierProfileIdentityMismatch);
        }
        if self.verifier_profile_name != profile.profile_name() {
            return Err(ExactVerificationPolicyError::VerifierProfileNameMismatch);
        }
        if self.inner.verifier_profile_id() != self.verifier_profile_name {
            return Err(ExactVerificationPolicyError::InnerProfileNameMismatch);
        }
        let expected = ExactVerificationPolicyId(hash_exact_policy(
            self.inner.id().as_bytes(),
            self.verifier_profile_id.as_bytes(),
        ));
        if expected != self.exact_policy_id {
            return Err(ExactVerificationPolicyError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub(crate) fn inner(&self) -> &VerificationPolicyV1 {
        &self.inner
    }
}

/// Exact profile-policy admission used by future authority/authentication adapters.
///
/// A same-name profile with a different root, epoch, or evidence class is rejected
/// before the lower-level name-based witness policy is consulted. Success is still
/// only `ProfilePolicyCheckedVerificationEvidenceV1`; it is deliberately not
/// organizational authority and cannot be authenticated directly.
pub(crate) fn policy_check_exact_verification_evidence(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &ExactVerificationPolicyV1,
    profile: &VerifierProfileV1,
    expected_challenge: [u8; 32],
    claim: VerificationEvidenceClaimV1,
) -> Result<ProfilePolicyCheckedVerificationEvidenceV1, ExactVerificationPolicyError> {
    policy.validate_against_profile(profile)?;
    Ok(policy_check_verification_evidence(
        contract,
        target,
        policy.inner(),
        profile,
        expected_challenge,
        claim,
    )?)
}

/// Exact verifier-policy failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExactVerificationPolicyError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    Admission(#[from] VerificationAdmissionError),
    #[error("verification policy pins a different exact verifier profile identity")]
    VerifierProfileIdentityMismatch,
    #[error("verification policy verifier profile name mismatch")]
    VerifierProfileNameMismatch,
    #[error("internal verification policy profile name mismatch")]
    InnerProfileNameMismatch,
    #[error("stored exact verification policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
}

fn hash_exact_policy(inner_policy_id: &[u8; 32], verifier_profile_id: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EXACT_POLICY_DOMAIN);
    hasher.update(inner_policy_id);
    hasher.update(verifier_profile_id);
    *hasher.finalize().as_bytes()
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
    use crate::verifier::{VerificationOutcomeV1, VerifierProfileV1};
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1};

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

    fn profile(root: u8, epoch: u64, class: EvidenceClass) -> VerifierProfileV1 {
        VerifierProfileV1::new("hardware-verifier-v1", [root; 32], epoch, class).unwrap()
    }

    fn entries(contract: &ValidatedContinuityContractV1) -> Vec<VerificationPolicyEntryV1> {
        vec![VerificationPolicyEntryV1::new(
            contract.requirements()[0].id(),
            EvidenceClass::Simulated,
        )]
    }

    #[test]
    fn same_exact_profile_is_deterministic() {
        let contract = contract();
        let profile = profile(9, 7, EvidenceClass::HardwareVerified);
        let a = ExactVerificationPolicyV1::new(&profile, entries(&contract)).unwrap();
        let b = ExactVerificationPolicyV1::new(&profile, entries(&contract)).unwrap();
        assert_eq!(a.id(), b.id());
        assert_eq!(a.verifier_profile_id(), profile.id());
    }

    #[test]
    fn same_name_different_root_epoch_or_class_changes_policy_identity() {
        let contract = contract();
        let base = profile(9, 7, EvidenceClass::HardwareVerified);
        let changed = [
            profile(10, 7, EvidenceClass::HardwareVerified),
            profile(9, 8, EvidenceClass::HardwareVerified),
            profile(9, 7, EvidenceClass::Observed),
        ];
        let base_policy = ExactVerificationPolicyV1::new(&base, entries(&contract)).unwrap();
        for candidate in changed {
            let candidate_policy =
                ExactVerificationPolicyV1::new(&candidate, entries(&contract)).unwrap();
            assert_ne!(base_policy.id(), candidate_policy.id());
            assert_eq!(
                base_policy
                    .validate_against_profile(&candidate)
                    .unwrap_err(),
                ExactVerificationPolicyError::VerifierProfileIdentityMismatch
            );
        }
    }

    #[test]
    fn same_name_root_substitution_fails_before_claim_admission() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let trusted = profile(9, 7, EvidenceClass::HardwareVerified);
        let substituted = profile(10, 7, EvidenceClass::HardwareVerified);
        let policy = ExactVerificationPolicyV1::new(&trusted, entries(&contract)).unwrap();
        let challenge = [5; 32];
        let claim = VerificationEvidenceClaimV1::new(
            contract.id(),
            target,
            contract.requirements()[0].id(),
            substituted.id(),
            challenge,
            1_700_000_000_111,
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        )
        .unwrap();
        assert_eq!(
            policy_check_exact_verification_evidence(
                &contract,
                target,
                &policy,
                &substituted,
                challenge,
                claim,
            )
            .unwrap_err(),
            ExactVerificationPolicyError::VerifierProfileIdentityMismatch
        );
    }
}
