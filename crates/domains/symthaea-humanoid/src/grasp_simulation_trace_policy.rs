// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precommitted trust policy for authenticated Grasp simulation traces.
//!
//! A valid signature is not by itself sufficient qualification provenance. The
//! campaign must also precommit which producer software, key, authentication
//! scheme, revocation floor, claim lifetime and verifier trust-root policy are
//! acceptable. This module binds those facts without granting motor authority.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::grasp_simulation_controller::causal_trace::HumanoidGraspSimulationCausalTrace;
use crate::grasp_simulation_trace_attestation::{
    HumanoidGraspSimulationTraceClaim, HumanoidGraspSimulationTraceVerificationFailure,
    HumanoidGraspSimulationTraceVerifier, HumanoidVerifiedGraspSimulationTrace,
    verify_humanoid_grasp_simulation_trace_producer,
};

pub const HUMANOID_GRASP_SIMULATION_TRACE_TRUST_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_POLICY_VERIFIED_GRASP_SIMULATION_TRACE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidGraspSimulationTraceTrustPolicy {
    schema_version: u32,
    policy_id: String,
    producer_id: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    key_id: String,
    scheme_id: String,
    minimum_revocation_epoch: u64,
    maximum_claim_lifetime_millis: u64,
    verifier_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspSimulationTraceTrustPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        policy_id: impl Into<String>,
        producer_id: impl Into<String>,
        producer_artifact_digest: HumanoidEvidenceDigest,
        key_id: impl Into<String>,
        scheme_id: impl Into<String>,
        minimum_revocation_epoch: u64,
        maximum_claim_lifetime_millis: u64,
        verifier_digest: HumanoidEvidenceDigest,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_GRASP_SIMULATION_TRACE_TRUST_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            producer_id: producer_id.into(),
            producer_artifact_digest,
            key_id: key_id.into(),
            scheme_id: scheme_id.into(),
            minimum_revocation_epoch,
            maximum_claim_lifetime_millis,
            verifier_digest,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.base_valid() {
            return None;
        }
        value.policy_digest = digest_policy(&value);
        value.validate().then_some(value)
    }

    fn base_valid(&self) -> bool {
        self.schema_version == HUMANOID_GRASP_SIMULATION_TRACE_TRUST_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && valid_id(&self.producer_id)
            && !self.producer_artifact_digest.is_zero()
            && valid_id(&self.key_id)
            && valid_id(&self.scheme_id)
            && self.maximum_claim_lifetime_millis > 0
            && !self.verifier_digest.is_zero()
    }

    pub fn validate(&self) -> bool {
        self.base_valid()
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_policy(self)
    }

    pub fn admits_claim(&self, claim: &HumanoidGraspSimulationTraceClaim) -> bool {
        let Some(lifetime_millis) = claim
            .valid_until_unix_millis()
            .checked_sub(claim.attested_at_unix_millis())
        else {
            return false;
        };

        self.validate()
            && claim.validate()
            && claim.producer_id() == self.producer_id
            && claim.producer_artifact_digest() == self.producer_artifact_digest
            && claim.key_id() == self.key_id
            && claim.authentication().scheme_id() == self.scheme_id
            && claim.revocation_epoch() >= self.minimum_revocation_epoch
            && lifetime_millis <= self.maximum_claim_lifetime_millis
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.verifier_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspSimulationTracePolicyVerificationFailure {
    InvalidPolicy,
    ClaimRejectedByPolicy,
    TraceVerification(HumanoidGraspSimulationTraceVerificationFailure),
    VerifierPolicyMismatch,
    InvalidPolicyBoundEvidence,
}

/// Opaque trace provenance that passed both cryptographic verification and the
/// exact precommitted campaign trust policy.
///
/// The exact signed claim is retained so future import/serialization validation
/// can re-run producer/key/scheme/revocation/lifetime admission rather than merely
/// trusting that this wrapper must once have been constructed through the binder.
pub struct HumanoidPolicyVerifiedGraspSimulationTrace {
    schema_version: u32,
    evidence_policy_digest: HumanoidEvidenceDigest,
    claim: HumanoidGraspSimulationTraceClaim,
    inner: HumanoidVerifiedGraspSimulationTrace,
    binding_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidPolicyVerifiedGraspSimulationTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidPolicyVerifiedGraspSimulationTrace")
            .field("evidence_policy_digest", &self.evidence_policy_digest)
            .field("trace_digest", &self.inner.trace_digest())
            .field("statement_digest", &self.claim.statement_digest())
            .field("producer_id", &self.claim.producer_id())
            .field("verifier_digest", &self.inner.verifier_digest())
            .field("verification_digest", &self.inner.verification_digest())
            .field("binding_digest", &self.binding_digest)
            .finish()
    }
}

impl HumanoidPolicyVerifiedGraspSimulationTrace {
    pub const fn evidence_policy_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_policy_digest
    }

    pub const fn trace_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.trace_digest()
    }

    pub const fn verifier_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.verifier_digest()
    }

    pub const fn verification_digest(&self) -> HumanoidEvidenceDigest {
        self.inner.verification_digest()
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    pub const fn valid_until_unix_millis(&self) -> u64 {
        self.inner.valid_until_unix_millis()
    }

    pub fn claim(&self) -> &HumanoidGraspSimulationTraceClaim {
        &self.claim
    }

    pub fn validate_for(
        &self,
        trace: &HumanoidGraspSimulationCausalTrace,
        policy: &HumanoidGraspSimulationTraceTrustPolicy,
        now_unix_millis: u64,
    ) -> bool {
        self.schema_version == HUMANOID_POLICY_VERIFIED_GRASP_SIMULATION_TRACE_SCHEMA_VERSION
            && policy.validate()
            && self.evidence_policy_digest == policy.policy_digest()
            && self.claim.validate()
            && policy.admits_claim(&self.claim)
            && now_unix_millis > 0
            && now_unix_millis >= self.claim.attested_at_unix_millis()
            && now_unix_millis <= self.claim.valid_until_unix_millis()
            && self.claim.trace_digest() == trace.trace_digest()
            && self.claim.trace_digest() == self.inner.trace_digest()
            && self.claim.producer_id() == self.inner.producer_id()
            && self.claim.producer_artifact_digest() == self.inner.producer_artifact_digest()
            && self.inner.verifier_digest() == policy.verifier_digest()
            && self.inner.validate_for(trace, now_unix_millis)
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_policy_bound_verification(self)
    }
}

pub fn verify_policy_bound_humanoid_grasp_simulation_trace(
    trace: &HumanoidGraspSimulationCausalTrace,
    claim: &HumanoidGraspSimulationTraceClaim,
    verifier: &dyn HumanoidGraspSimulationTraceVerifier,
    policy: &HumanoidGraspSimulationTraceTrustPolicy,
    now_unix_millis: u64,
) -> Result<HumanoidPolicyVerifiedGraspSimulationTrace, HumanoidGraspSimulationTracePolicyVerificationFailure> {
    if !policy.validate() {
        return Err(HumanoidGraspSimulationTracePolicyVerificationFailure::InvalidPolicy);
    }
    if !policy.admits_claim(claim) {
        return Err(HumanoidGraspSimulationTracePolicyVerificationFailure::ClaimRejectedByPolicy);
    }

    let inner = verify_humanoid_grasp_simulation_trace_producer(
        trace,
        claim,
        verifier,
        now_unix_millis,
    )
    .map_err(HumanoidGraspSimulationTracePolicyVerificationFailure::TraceVerification)?;

    // Compare against the verifier identity captured by the accepted decision,
    // not a second verifier.verifier_digest() call.
    if inner.verifier_digest() != policy.verifier_digest() {
        return Err(HumanoidGraspSimulationTracePolicyVerificationFailure::VerifierPolicyMismatch);
    }

    let mut value = HumanoidPolicyVerifiedGraspSimulationTrace {
        schema_version: HUMANOID_POLICY_VERIFIED_GRASP_SIMULATION_TRACE_SCHEMA_VERSION,
        evidence_policy_digest: policy.policy_digest(),
        claim: claim.clone(),
        inner,
        binding_digest: HumanoidEvidenceDigest::ZERO,
    };
    value.binding_digest = digest_policy_bound_verification(&value);
    value
        .validate_for(trace, policy, now_unix_millis)
        .then_some(value)
        .ok_or(HumanoidGraspSimulationTracePolicyVerificationFailure::InvalidPolicyBoundEvidence)
}

fn digest_policy(value: &HumanoidGraspSimulationTraceTrustPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-trace-trust-policy.v1");
    h.u32(value.schema_version)
        .string(&value.policy_id)
        .string(&value.producer_id)
        .digest(value.producer_artifact_digest)
        .string(&value.key_id)
        .string(&value.scheme_id)
        .u64(value.minimum_revocation_epoch)
        .u64(value.maximum_claim_lifetime_millis)
        .digest(value.verifier_digest);
    h.finish()
}

fn digest_policy_bound_verification(
    value: &HumanoidPolicyVerifiedGraspSimulationTrace,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-simulation-trace-policy-binding.v1");
    h.u32(value.schema_version)
        .digest(value.evidence_policy_digest)
        .digest(value.claim.statement_digest())
        .digest(value.inner.verification_digest());
    h.finish()
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grasp_simulation_trace_attestation::{
        HumanoidGraspSimulationTraceAuthentication, HumanoidGraspSimulationTraceClaim,
    };

    fn digest(byte: u8) -> HumanoidEvidenceDigest {
        HumanoidEvidenceDigest::from_bytes([byte; 32])
    }

    fn claim(
        producer: &str,
        producer_artifact: HumanoidEvidenceDigest,
        key: &str,
        scheme: &str,
        revocation_epoch: u64,
        attested_at: u64,
        valid_until: u64,
    ) -> HumanoidGraspSimulationTraceClaim {
        HumanoidGraspSimulationTraceClaim::new(
            digest(1),
            producer,
            producer_artifact,
            key,
            revocation_epoch,
            attested_at,
            valid_until,
            HumanoidGraspSimulationTraceAuthentication::new(scheme, vec![0xAA; 64]).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn policy_identity_changes_with_trust_root_or_producer() {
        let a = HumanoidGraspSimulationTraceTrustPolicy::new(
            "trace-policy",
            "sim-executor-a",
            digest(2),
            "key-a",
            "ml-dsa-87",
            7,
            60_000,
            digest(3),
        )
        .unwrap();
        let different_producer = HumanoidGraspSimulationTraceTrustPolicy::new(
            "trace-policy",
            "sim-executor-b",
            digest(2),
            "key-a",
            "ml-dsa-87",
            7,
            60_000,
            digest(3),
        )
        .unwrap();
        let different_verifier = HumanoidGraspSimulationTraceTrustPolicy::new(
            "trace-policy",
            "sim-executor-a",
            digest(2),
            "key-a",
            "ml-dsa-87",
            7,
            60_000,
            digest(4),
        )
        .unwrap();
        assert_ne!(a.policy_digest(), different_producer.policy_digest());
        assert_ne!(a.policy_digest(), different_verifier.policy_digest());
    }

    #[test]
    fn policy_rejects_wrong_producer_key_scheme_and_revocation() {
        let policy = HumanoidGraspSimulationTraceTrustPolicy::new(
            "trace-policy",
            "sim-executor-a",
            digest(2),
            "key-a",
            "ml-dsa-87",
            7,
            60_000,
            digest(3),
        )
        .unwrap();

        assert!(policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-a", "ml-dsa-87", 7, 1_000, 2_000,
        )));
        assert!(!policy.admits_claim(&claim(
            "sim-executor-b", digest(2), "key-a", "ml-dsa-87", 7, 1_000, 2_000,
        )));
        assert!(!policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-b", "ml-dsa-87", 7, 1_000, 2_000,
        )));
        assert!(!policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-a", "ed25519", 7, 1_000, 2_000,
        )));
        assert!(!policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-a", "ml-dsa-87", 6, 1_000, 2_000,
        )));
    }

    #[test]
    fn policy_bounds_signed_claim_lifetime() {
        let policy = HumanoidGraspSimulationTraceTrustPolicy::new(
            "trace-policy",
            "sim-executor-a",
            digest(2),
            "key-a",
            "ml-dsa-87",
            0,
            1_000,
            digest(3),
        )
        .unwrap();
        assert!(policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-a", "ml-dsa-87", 0, 5_000, 6_000,
        )));
        assert!(!policy.admits_claim(&claim(
            "sim-executor-a", digest(2), "key-a", "ml-dsa-87", 0, 5_000, 6_001,
        )));
    }
}
