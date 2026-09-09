// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact verifier and freshness policy for distributed transition currentness.
//!
//! Authentication alone is not sufficient for a safety decision: evidence can be
//! stale, come from the wrong verifier profile snapshot, or drift too far apart in
//! time to describe one coherent distributed state. This module defines the exact
//! policy for those admission bounds without itself evaluating evidence.
//!
//! Core theorem:
//!
//! `AuthenticatedEvidence != CurrentEvidence != QualifiedDistributedTransition`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, ValidatedDistributedChangeBudgetV1};
use crate::failure_domain::FailureDomainPolicyId;
use crate::verifier::VerifierProfileId;

pub const DISTRIBUTED_CURRENTNESS_POLICY_SCHEMA_V1: &str =
    "symthaea-continuity-distributed-currentness-policy-v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.distributed-currentness-policy.v1\0";
const MAX_PROFILE_IDS: usize = 4096;
const MAX_FAILURE_DOMAIN_POLICIES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DistributedCurrentnessPolicyId([u8; 32]);

impl DistributedCurrentnessPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable exact policy for which verifier snapshots may establish each
/// currentness role and how fresh/coherent their evidence must be.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedCurrentnessPolicyV1 {
    schema_version: String,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    policy_generation: u64,
    participant_verifier_profile_ids: Vec<VerifierProfileId>,
    failure_domain_verifier_profile_ids: Vec<VerifierProfileId>,
    recovery_verifier_profile_ids: Vec<VerifierProfileId>,
    required_failure_domain_policy_ids: Vec<FailureDomainPolicyId>,
    max_participant_state_age_ms: u64,
    max_failure_domain_age_ms: u64,
    max_recovery_path_age_ms: u64,
    max_future_skew_ms: u64,
    max_cross_evidence_skew_ms: u64,
    policy_id: DistributedCurrentnessPolicyId,
}

impl DistributedCurrentnessPolicyV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: &ValidatedDistributedChangeBudgetV1,
        policy_generation: u64,
        mut participant_verifier_profile_ids: Vec<VerifierProfileId>,
        mut failure_domain_verifier_profile_ids: Vec<VerifierProfileId>,
        mut recovery_verifier_profile_ids: Vec<VerifierProfileId>,
        mut required_failure_domain_policy_ids: Vec<FailureDomainPolicyId>,
        max_participant_state_age_ms: u64,
        max_failure_domain_age_ms: u64,
        max_recovery_path_age_ms: u64,
        max_future_skew_ms: u64,
        max_cross_evidence_skew_ms: u64,
    ) -> Result<Self, DistributedCurrentnessPolicyError> {
        if policy_generation == 0 {
            return Err(DistributedCurrentnessPolicyError::ZeroPolicyGeneration);
        }
        canonicalize_profiles(
            "participant",
            &mut participant_verifier_profile_ids,
            true,
        )?;
        canonicalize_profiles(
            "failure-domain",
            &mut failure_domain_verifier_profile_ids,
            false,
        )?;
        canonicalize_profiles("recovery", &mut recovery_verifier_profile_ids, true)?;
        canonicalize_failure_policies(&mut required_failure_domain_policy_ids)?;
        validate_failure_policy_profile_relation(
            &required_failure_domain_policy_ids,
            &failure_domain_verifier_profile_ids,
        )?;
        validate_time_bounds(
            max_participant_state_age_ms,
            max_failure_domain_age_ms,
            max_recovery_path_age_ms,
            max_cross_evidence_skew_ms,
        )?;

        let budget_id = budget.id();
        let budget_generation = budget.generation();
        let policy_id = DistributedCurrentnessPolicyId(hash_policy(
            budget_id,
            budget_generation,
            policy_generation,
            &participant_verifier_profile_ids,
            &failure_domain_verifier_profile_ids,
            &recovery_verifier_profile_ids,
            &required_failure_domain_policy_ids,
            max_participant_state_age_ms,
            max_failure_domain_age_ms,
            max_recovery_path_age_ms,
            max_future_skew_ms,
            max_cross_evidence_skew_ms,
        ));

        Ok(Self {
            schema_version: DISTRIBUTED_CURRENTNESS_POLICY_SCHEMA_V1.to_owned(),
            budget_id,
            budget_generation,
            policy_generation,
            participant_verifier_profile_ids,
            failure_domain_verifier_profile_ids,
            recovery_verifier_profile_ids,
            required_failure_domain_policy_ids,
            max_participant_state_age_ms,
            max_failure_domain_age_ms,
            max_recovery_path_age_ms,
            max_future_skew_ms,
            max_cross_evidence_skew_ms,
            policy_id,
        })
    }

    pub fn validate(&self) -> Result<(), DistributedCurrentnessPolicyError> {
        if self.schema_version != DISTRIBUTED_CURRENTNESS_POLICY_SCHEMA_V1 {
            return Err(DistributedCurrentnessPolicyError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.budget_generation == 0 {
            return Err(DistributedCurrentnessPolicyError::ZeroBudgetGeneration);
        }
        if self.policy_generation == 0 {
            return Err(DistributedCurrentnessPolicyError::ZeroPolicyGeneration);
        }
        validate_canonical_profiles(
            "participant",
            &self.participant_verifier_profile_ids,
            true,
        )?;
        validate_canonical_profiles(
            "failure-domain",
            &self.failure_domain_verifier_profile_ids,
            false,
        )?;
        validate_canonical_profiles("recovery", &self.recovery_verifier_profile_ids, true)?;
        validate_canonical_failure_policies(&self.required_failure_domain_policy_ids)?;
        validate_failure_policy_profile_relation(
            &self.required_failure_domain_policy_ids,
            &self.failure_domain_verifier_profile_ids,
        )?;
        validate_time_bounds(
            self.max_participant_state_age_ms,
            self.max_failure_domain_age_ms,
            self.max_recovery_path_age_ms,
            self.max_cross_evidence_skew_ms,
        )?;

        let expected = DistributedCurrentnessPolicyId(hash_policy(
            self.budget_id,
            self.budget_generation,
            self.policy_generation,
            &self.participant_verifier_profile_ids,
            &self.failure_domain_verifier_profile_ids,
            &self.recovery_verifier_profile_ids,
            &self.required_failure_domain_policy_ids,
            self.max_participant_state_age_ms,
            self.max_failure_domain_age_ms,
            self.max_recovery_path_age_ms,
            self.max_future_skew_ms,
            self.max_cross_evidence_skew_ms,
        ));
        if expected != self.policy_id {
            return Err(DistributedCurrentnessPolicyError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn validate_against_budget(
        &self,
        budget: &ValidatedDistributedChangeBudgetV1,
    ) -> Result<ValidatedDistributedCurrentnessPolicyV1, DistributedCurrentnessPolicyError> {
        self.validate()?;
        if self.budget_id != budget.id() {
            return Err(DistributedCurrentnessPolicyError::BudgetIdentityMismatch);
        }
        if self.budget_generation != budget.generation() {
            return Err(DistributedCurrentnessPolicyError::BudgetGenerationMismatch);
        }
        Ok(ValidatedDistributedCurrentnessPolicyV1 {
            budget: budget.clone(),
            inner: self.clone(),
        })
    }

    pub fn id(&self) -> DistributedCurrentnessPolicyId {
        self.policy_id
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.budget_id
    }

    pub fn budget_generation(&self) -> u64 {
        self.budget_generation
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn participant_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        &self.participant_verifier_profile_ids
    }

    pub fn failure_domain_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        &self.failure_domain_verifier_profile_ids
    }

    pub fn recovery_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        &self.recovery_verifier_profile_ids
    }

    pub fn required_failure_domain_policy_ids(&self) -> &[FailureDomainPolicyId] {
        &self.required_failure_domain_policy_ids
    }

    pub fn max_participant_state_age_ms(&self) -> u64 {
        self.max_participant_state_age_ms
    }

    pub fn max_failure_domain_age_ms(&self) -> u64 {
        self.max_failure_domain_age_ms
    }

    pub fn max_recovery_path_age_ms(&self) -> u64 {
        self.max_recovery_path_age_ms
    }

    pub fn max_future_skew_ms(&self) -> u64 {
        self.max_future_skew_ms
    }

    pub fn max_cross_evidence_skew_ms(&self) -> u64 {
        self.max_cross_evidence_skew_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedDistributedCurrentnessPolicyV1 {
    budget: ValidatedDistributedChangeBudgetV1,
    inner: DistributedCurrentnessPolicyV1,
}

impl ValidatedDistributedCurrentnessPolicyV1 {
    pub fn id(&self) -> DistributedCurrentnessPolicyId {
        self.inner.id()
    }

    pub fn budget(&self) -> &ValidatedDistributedChangeBudgetV1 {
        &self.budget
    }

    pub fn budget_id(&self) -> DistributedChangeBudgetId {
        self.inner.budget_id()
    }

    pub fn budget_generation(&self) -> u64 {
        self.inner.budget_generation()
    }

    pub fn policy_generation(&self) -> u64 {
        self.inner.policy_generation()
    }

    pub fn participant_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        self.inner.participant_verifier_profile_ids()
    }

    pub fn failure_domain_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        self.inner.failure_domain_verifier_profile_ids()
    }

    pub fn recovery_verifier_profile_ids(&self) -> &[VerifierProfileId] {
        self.inner.recovery_verifier_profile_ids()
    }

    pub fn required_failure_domain_policy_ids(&self) -> &[FailureDomainPolicyId] {
        self.inner.required_failure_domain_policy_ids()
    }

    pub fn max_participant_state_age_ms(&self) -> u64 {
        self.inner.max_participant_state_age_ms()
    }

    pub fn max_failure_domain_age_ms(&self) -> u64 {
        self.inner.max_failure_domain_age_ms()
    }

    pub fn max_recovery_path_age_ms(&self) -> u64 {
        self.inner.max_recovery_path_age_ms()
    }

    pub fn max_future_skew_ms(&self) -> u64 {
        self.inner.max_future_skew_ms()
    }

    pub fn max_cross_evidence_skew_ms(&self) -> u64 {
        self.inner.max_cross_evidence_skew_ms()
    }

    pub fn as_raw(&self) -> &DistributedCurrentnessPolicyV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DistributedCurrentnessPolicyError {
    #[error("unsupported distributed currentness policy schema: {0}")]
    UnsupportedSchema(String),
    #[error("distributed currentness policy budget generation must be non-zero")]
    ZeroBudgetGeneration,
    #[error("distributed currentness policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("{role} verifier profile set must not be empty")]
    EmptyVerifierProfiles { role: &'static str },
    #[error("{role} verifier profile set exceeds the profile bound")]
    TooManyVerifierProfiles { role: &'static str },
    #[error("{role} verifier profiles must be in canonical sorted unique order")]
    NonCanonicalVerifierProfiles { role: &'static str },
    #[error("required failure-domain policy set exceeds the policy bound")]
    TooManyFailureDomainPolicies,
    #[error("required failure-domain policy ids must be in canonical sorted unique order")]
    NonCanonicalFailureDomainPolicies,
    #[error("required failure-domain policies need at least one approved failure-domain verifier profile")]
    MissingFailureDomainVerifierProfiles,
    #[error("failure-domain verifier profiles are configured but no failure-domain policy is required")]
    UnusedFailureDomainVerifierProfiles,
    #[error("participant-state maximum age must be non-zero")]
    ZeroParticipantAge,
    #[error("failure-domain maximum age must be non-zero")]
    ZeroFailureDomainAge,
    #[error("recovery-path maximum age must be non-zero")]
    ZeroRecoveryAge,
    #[error("cross-evidence skew bound must be non-zero")]
    ZeroCrossEvidenceSkew,
    #[error("distributed currentness policy references a different distributed budget")]
    BudgetIdentityMismatch,
    #[error("distributed currentness policy references a different budget generation")]
    BudgetGenerationMismatch,
    #[error("stored distributed currentness policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
}

fn canonicalize_profiles(
    role: &'static str,
    values: &mut Vec<VerifierProfileId>,
    required: bool,
) -> Result<(), DistributedCurrentnessPolicyError> {
    if values.len() > MAX_PROFILE_IDS {
        return Err(DistributedCurrentnessPolicyError::TooManyVerifierProfiles { role });
    }
    values.sort();
    values.dedup();
    if required && values.is_empty() {
        return Err(DistributedCurrentnessPolicyError::EmptyVerifierProfiles { role });
    }
    Ok(())
}

fn validate_canonical_profiles(
    role: &'static str,
    values: &[VerifierProfileId],
    required: bool,
) -> Result<(), DistributedCurrentnessPolicyError> {
    if values.len() > MAX_PROFILE_IDS {
        return Err(DistributedCurrentnessPolicyError::TooManyVerifierProfiles { role });
    }
    if required && values.is_empty() {
        return Err(DistributedCurrentnessPolicyError::EmptyVerifierProfiles { role });
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DistributedCurrentnessPolicyError::NonCanonicalVerifierProfiles { role });
    }
    Ok(())
}

fn canonicalize_failure_policies(
    values: &mut Vec<FailureDomainPolicyId>,
) -> Result<(), DistributedCurrentnessPolicyError> {
    if values.len() > MAX_FAILURE_DOMAIN_POLICIES {
        return Err(DistributedCurrentnessPolicyError::TooManyFailureDomainPolicies);
    }
    values.sort();
    values.dedup();
    Ok(())
}

fn validate_canonical_failure_policies(
    values: &[FailureDomainPolicyId],
) -> Result<(), DistributedCurrentnessPolicyError> {
    if values.len() > MAX_FAILURE_DOMAIN_POLICIES {
        return Err(DistributedCurrentnessPolicyError::TooManyFailureDomainPolicies);
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DistributedCurrentnessPolicyError::NonCanonicalFailureDomainPolicies);
    }
    Ok(())
}

fn validate_failure_policy_profile_relation(
    policies: &[FailureDomainPolicyId],
    profiles: &[VerifierProfileId],
) -> Result<(), DistributedCurrentnessPolicyError> {
    match (policies.is_empty(), profiles.is_empty()) {
        (false, true) => Err(DistributedCurrentnessPolicyError::MissingFailureDomainVerifierProfiles),
        (true, false) => Err(DistributedCurrentnessPolicyError::UnusedFailureDomainVerifierProfiles),
        _ => Ok(()),
    }
}

fn validate_time_bounds(
    max_participant_state_age_ms: u64,
    max_failure_domain_age_ms: u64,
    max_recovery_path_age_ms: u64,
    max_cross_evidence_skew_ms: u64,
) -> Result<(), DistributedCurrentnessPolicyError> {
    if max_participant_state_age_ms == 0 {
        return Err(DistributedCurrentnessPolicyError::ZeroParticipantAge);
    }
    if max_failure_domain_age_ms == 0 {
        return Err(DistributedCurrentnessPolicyError::ZeroFailureDomainAge);
    }
    if max_recovery_path_age_ms == 0 {
        return Err(DistributedCurrentnessPolicyError::ZeroRecoveryAge);
    }
    if max_cross_evidence_skew_ms == 0 {
        return Err(DistributedCurrentnessPolicyError::ZeroCrossEvidenceSkew);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_policy(
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    policy_generation: u64,
    participant_verifier_profile_ids: &[VerifierProfileId],
    failure_domain_verifier_profile_ids: &[VerifierProfileId],
    recovery_verifier_profile_ids: &[VerifierProfileId],
    required_failure_domain_policy_ids: &[FailureDomainPolicyId],
    max_participant_state_age_ms: u64,
    max_failure_domain_age_ms: u64,
    max_recovery_path_age_ms: u64,
    max_future_skew_ms: u64,
    max_cross_evidence_skew_ms: u64,
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(512);
    bytes.extend_from_slice(budget_id.as_bytes());
    bytes.extend_from_slice(&budget_generation.to_le_bytes());
    bytes.extend_from_slice(&policy_generation.to_le_bytes());
    put_profile_ids(&mut bytes, participant_verifier_profile_ids);
    put_profile_ids(&mut bytes, failure_domain_verifier_profile_ids);
    put_profile_ids(&mut bytes, recovery_verifier_profile_ids);
    put_len(&mut bytes, required_failure_domain_policy_ids.len());
    for id in required_failure_domain_policy_ids {
        bytes.extend_from_slice(id.as_bytes());
    }
    bytes.extend_from_slice(&max_participant_state_age_ms.to_le_bytes());
    bytes.extend_from_slice(&max_failure_domain_age_ms.to_le_bytes());
    bytes.extend_from_slice(&max_recovery_path_age_ms.to_le_bytes());
    bytes.extend_from_slice(&max_future_skew_ms.to_le_bytes());
    bytes.extend_from_slice(&max_cross_evidence_skew_ms.to_le_bytes());
    let mut hasher = blake3::Hasher::new();
    hasher.update(POLICY_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_profile_ids(out: &mut Vec<u8>, values: &[VerifierProfileId]) {
    put_len(out, values.len());
    for value in values {
        out.extend_from_slice(value.as_bytes());
    }
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedChangeBudgetV1, RecoveryPathClassV1};
    use crate::failure_domain::FailureDomainPolicyId;
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};
    use crate::verifier::VerifierProfileV1;
    use crate::witness::EvidenceClass;

    fn budget() -> ValidatedDistributedChangeBudgetV1 {
        let aggregate = ContinuitySubjectV1::new(
            "org.example",
            "cluster-a",
            ContinuityScopeV1::Cluster,
            None,
        )
        .unwrap();
        let a = ContinuitySubjectV1::new(
            "org.example",
            "node-a",
            ContinuityScopeV1::Machine,
            None,
        )
        .unwrap();
        let b = ContinuitySubjectV1::new(
            "org.example",
            "node-b",
            ContinuityScopeV1::Machine,
            None,
        )
        .unwrap();
        let raw = DistributedChangeBudgetV1::new(
            &aggregate,
            3,
            vec![a.id(), b.id()],
            1,
            1,
            vec![],
            vec![RecoveryPathClassV1::OutOfBandManagement],
        )
        .unwrap();
        raw.validate_against_subject(&aggregate).unwrap()
    }

    fn profile(seed: u8) -> VerifierProfileId {
        VerifierProfileV1::new(
            format!("profile-{seed}"),
            [seed; 32],
            1,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
        .id()
    }

    fn fake_failure_id(seed: u8) -> FailureDomainPolicyId {
        // IDs are opaque outside the defining module, so obtain one from JSON only
        // in this local structural test through serde round-trip of the byte tuple.
        serde_json::from_value(serde_json::json!([seed; 32])).unwrap()
    }

    #[test]
    fn constructor_canonicalizes_verifier_profile_sets() {
        let budget = budget();
        let p1 = profile(1);
        let p2 = profile(2);
        let policy = DistributedCurrentnessPolicyV1::new(
            &budget,
            1,
            vec![p2, p1, p1],
            vec![],
            vec![p2, p1],
            vec![],
            5_000,
            5_000,
            10_000,
            250,
            2_000,
        )
        .unwrap();
        assert_eq!(policy.participant_verifier_profile_ids(), &[p1, p2]);
        policy.validate_against_budget(&budget).unwrap();
    }

    #[test]
    fn required_failure_policy_requires_failure_verifier_profile() {
        let budget = budget();
        assert_eq!(
            DistributedCurrentnessPolicyV1::new(
                &budget,
                1,
                vec![profile(1)],
                vec![],
                vec![profile(2)],
                vec![fake_failure_id(9)],
                5_000,
                5_000,
                10_000,
                250,
                2_000,
            )
            .unwrap_err(),
            DistributedCurrentnessPolicyError::MissingFailureDomainVerifierProfiles
        );
    }

    #[test]
    fn unused_failure_verifier_profile_is_rejected() {
        let budget = budget();
        assert_eq!(
            DistributedCurrentnessPolicyV1::new(
                &budget,
                1,
                vec![profile(1)],
                vec![profile(3)],
                vec![profile(2)],
                vec![],
                5_000,
                5_000,
                10_000,
                250,
                2_000,
            )
            .unwrap_err(),
            DistributedCurrentnessPolicyError::UnusedFailureDomainVerifierProfiles
        );
    }

    #[test]
    fn freshness_bounds_are_identity_material() {
        let budget = budget();
        let a = DistributedCurrentnessPolicyV1::new(
            &budget,
            1,
            vec![profile(1)],
            vec![],
            vec![profile(2)],
            vec![],
            5_000,
            5_000,
            10_000,
            250,
            2_000,
        )
        .unwrap();
        let b = DistributedCurrentnessPolicyV1::new(
            &budget,
            1,
            vec![profile(1)],
            vec![],
            vec![profile(2)],
            vec![],
            6_000,
            5_000,
            10_000,
            250,
            2_000,
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
    }
}
