// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-bound clock policies and whole-envelope evaluation permits.
//!
//! V4 binds both candidate-window quorum semantics and successor-continuity
//! semantics into the externally authenticated evaluation policy. A permit
//! retains both validated policy records so later consumers do not need either
//! policy supplied by the caller.

use crate::clock::{ClockQuorumPolicy, MAX_CLOCK_OBSERVATIONS};
use crate::clock_bootstrap_authority::{
    ClockBootstrapAuthorityError, ClockBootstrapClaimV2, VerifiedClockBootstrapAuthorityV2,
};
use crate::continuity::ClockContinuityPolicy;
use crate::digest::{domain_hash, Sha256Digest};
use crate::signature::SignatureAlgorithm;
use crate::trust::{
    digest_trust_snapshot, KeyLifecycleStatus, KeyUsage, TrustSnapshot, TrustSnapshotError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CLOCK_QUORUM_POLICY_SCHEMA: &str = "symthaea.trust.clock-quorum-policy.v1";
pub const CLOCK_CONTINUITY_POLICY_SCHEMA: &str = "symthaea.trust.clock-continuity-policy.v1";
pub const CLOCK_EVALUATION_POLICY_SCHEMA: &str = "symthaea.trust.clock-evaluation-policy.v4";
pub const CLOCK_BOOTSTRAP_ANCHOR_SCHEMA: &str = "symthaea.trust.clock-bootstrap-anchor.v2";
pub const CLOCK_EVALUATION_PERMIT_SCHEMA: &str = "symthaea.trust.clock-evaluation-permit.v4";

const CLOCK_QUORUM_POLICY_DOMAIN: &[u8] = b"symthaea.trust.clock-quorum-policy.v1\0";
const CLOCK_CONTINUITY_POLICY_DOMAIN: &[u8] = b"symthaea.trust.clock-continuity-policy.v1\0";
const CLOCK_EVALUATION_POLICY_DOMAIN: &[u8] = b"symthaea.trust.clock-evaluation-policy.v4\0";
const CLOCK_BOOTSTRAP_ANCHOR_DOMAIN: &[u8] = b"symthaea.trust.clock-bootstrap-anchor.v2\0";
const CLOCK_EVALUATION_PERMIT_DOMAIN: &[u8] = b"symthaea.trust.clock-evaluation-permit.v4\0";
const BOOTSTRAP_ANCHOR_KIND: &str = "ExternalBootstrap";
const PERMIT_BASIS_KIND: &str = "BootstrapAnchor";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockQuorumPolicyRevisionV1 {
    schema: String,
    minimum_distinct_sources: usize,
    maximum_observations: usize,
    maximum_uncertainty_ms: u64,
    maximum_consensus_width_ms: u64,
    require_algorithm_diversity: bool,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockContinuityPolicyRevisionV1 {
    schema: String,
    maximum_epoch_step: u64,
    maximum_forward_gap_ms: u64,
    maximum_consensus_jump_ms: u64,
    minimum_shared_sources: usize,
    require_shared_algorithm: bool,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockEvaluationPolicyV4 {
    schema: String,
    policy_record_digest: Sha256Digest,
    clock_quorum_policy_id: Sha256Digest,
    clock_continuity_policy_id: Sha256Digest,
    max_transition_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockQuorumPolicyRevisionIdV1(Sha256Digest);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockContinuityPolicyRevisionIdV1(Sha256Digest);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockEvaluationPolicyIdV4(Sha256Digest);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockBootstrapAnchorIdV2(Sha256Digest);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockEvaluationPermitIdV4(Sha256Digest);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockAuthorityKeyV1 {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockEvaluationPermitV4 {
    id: ClockEvaluationPermitIdV4,
    basis_id: ClockBootstrapAnchorIdV2,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: Vec<ClockAuthorityKeyV1>,
    clock_quorum_policy: ClockQuorumPolicyRevisionV1,
    clock_continuity_policy: ClockContinuityPolicyRevisionV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockEvaluationPermitError {
    InvalidQuorumPolicy,
    QuorumPolicyIdentityMismatch,
    InvalidContinuityPolicy,
    ContinuityPolicyIdentityMismatch,
    InvalidEvaluationPolicy,
    EvaluationPolicyIdentityMismatch,
    BootstrapClaimInvalid(ClockBootstrapAuthorityError),
    BootstrapAuthorityClaimMismatch,
    BootstrapAuthorityPolicyMismatch,
    BootstrapPolicyMismatch,
    QuorumPolicyMismatch,
    ContinuityPolicyMismatch,
    BootstrapSnapshotMismatch,
    TrustSnapshotInvalid(TrustSnapshotError),
    TransitionEnvelopeOverflow,
    TimeScaleOverflow,
    SnapshotNotValidForEnvelope,
    InsufficientClockAuthorityKeys { actual: usize, required: usize },
    EligibleAlgorithmDiversityMissing,
    Encoding(String),
}

impl ClockQuorumPolicyRevisionV1 {
    pub fn new(
        minimum_distinct_sources: usize,
        maximum_observations: usize,
        maximum_uncertainty_ms: u64,
        maximum_consensus_width_ms: u64,
        require_algorithm_diversity: bool,
    ) -> Result<Self, ClockEvaluationPermitError> {
        if minimum_distinct_sources == 0
            || maximum_observations < minimum_distinct_sources
            || maximum_observations > MAX_CLOCK_OBSERVATIONS
            || maximum_consensus_width_ms == 0
        {
            return Err(ClockEvaluationPermitError::InvalidQuorumPolicy);
        }
        let mut value = Self {
            schema: CLOCK_QUORUM_POLICY_SCHEMA.to_string(),
            minimum_distinct_sources,
            maximum_observations,
            maximum_uncertainty_ms,
            maximum_consensus_width_ms,
            require_algorithm_diversity,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_quorum_policy_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockEvaluationPermitError> {
        if self.schema != CLOCK_QUORUM_POLICY_SCHEMA
            || self.minimum_distinct_sources == 0
            || self.maximum_observations < self.minimum_distinct_sources
            || self.maximum_observations > MAX_CLOCK_OBSERVATIONS
            || self.maximum_consensus_width_ms == 0
        {
            return Err(ClockEvaluationPermitError::InvalidQuorumPolicy);
        }
        if compute_quorum_policy_id(self)? != self.id {
            return Err(ClockEvaluationPermitError::QuorumPolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ClockQuorumPolicyRevisionIdV1 { ClockQuorumPolicyRevisionIdV1(self.id) }
    pub fn minimum_distinct_sources(&self) -> usize { self.minimum_distinct_sources }
    pub fn maximum_observations(&self) -> usize { self.maximum_observations }
    pub fn maximum_uncertainty_ms(&self) -> u64 { self.maximum_uncertainty_ms }
    pub fn maximum_consensus_width_ms(&self) -> u64 { self.maximum_consensus_width_ms }
    pub fn require_algorithm_diversity(&self) -> bool { self.require_algorithm_diversity }
    pub fn to_runtime_policy(&self) -> Result<ClockQuorumPolicy, ClockEvaluationPermitError> {
        self.validate()?;
        Ok(ClockQuorumPolicy {
            minimum_distinct_sources: self.minimum_distinct_sources,
            maximum_observations: self.maximum_observations,
            maximum_uncertainty_ms: self.maximum_uncertainty_ms,
            maximum_consensus_width_ms: self.maximum_consensus_width_ms,
            require_algorithm_diversity: self.require_algorithm_diversity,
        })
    }
}

impl ClockContinuityPolicyRevisionV1 {
    pub fn new(
        maximum_epoch_step: u64,
        maximum_forward_gap_ms: u64,
        maximum_consensus_jump_ms: u64,
        minimum_shared_sources: usize,
        require_shared_algorithm: bool,
    ) -> Result<Self, ClockEvaluationPermitError> {
        if maximum_epoch_step == 0 || maximum_consensus_jump_ms == 0 || minimum_shared_sources == 0 {
            return Err(ClockEvaluationPermitError::InvalidContinuityPolicy);
        }
        let mut value = Self {
            schema: CLOCK_CONTINUITY_POLICY_SCHEMA.to_string(),
            maximum_epoch_step,
            maximum_forward_gap_ms,
            maximum_consensus_jump_ms,
            minimum_shared_sources,
            require_shared_algorithm,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_continuity_policy_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockEvaluationPermitError> {
        if self.schema != CLOCK_CONTINUITY_POLICY_SCHEMA
            || self.maximum_epoch_step == 0
            || self.maximum_consensus_jump_ms == 0
            || self.minimum_shared_sources == 0
        {
            return Err(ClockEvaluationPermitError::InvalidContinuityPolicy);
        }
        if compute_continuity_policy_id(self)? != self.id {
            return Err(ClockEvaluationPermitError::ContinuityPolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ClockContinuityPolicyRevisionIdV1 { ClockContinuityPolicyRevisionIdV1(self.id) }
    pub fn maximum_epoch_step(&self) -> u64 { self.maximum_epoch_step }
    pub fn maximum_forward_gap_ms(&self) -> u64 { self.maximum_forward_gap_ms }
    pub fn maximum_consensus_jump_ms(&self) -> u64 { self.maximum_consensus_jump_ms }
    pub fn minimum_shared_sources(&self) -> usize { self.minimum_shared_sources }
    pub fn require_shared_algorithm(&self) -> bool { self.require_shared_algorithm }
    pub fn to_runtime_policy(&self) -> Result<ClockContinuityPolicy, ClockEvaluationPermitError> {
        self.validate()?;
        Ok(ClockContinuityPolicy {
            maximum_epoch_step: self.maximum_epoch_step,
            maximum_forward_gap_ms: self.maximum_forward_gap_ms,
            maximum_consensus_jump_ms: self.maximum_consensus_jump_ms,
            minimum_shared_sources: self.minimum_shared_sources,
            require_shared_algorithm: self.require_shared_algorithm,
        })
    }
}

impl ClockEvaluationPolicyV4 {
    pub fn new(
        policy_record_digest: Sha256Digest,
        clock_quorum_policy: &ClockQuorumPolicyRevisionV1,
        clock_continuity_policy: &ClockContinuityPolicyRevisionV1,
        max_transition_ms: u64,
        minimum_eligible_clock_authority_keys: usize,
        require_eligible_algorithm_diversity: bool,
    ) -> Result<Self, ClockEvaluationPermitError> {
        clock_quorum_policy.validate()?;
        clock_continuity_policy.validate()?;
        if policy_record_digest.0 == [0; 32] || max_transition_ms == 0 || minimum_eligible_clock_authority_keys < 2 {
            return Err(ClockEvaluationPermitError::InvalidEvaluationPolicy);
        }
        let mut value = Self {
            schema: CLOCK_EVALUATION_POLICY_SCHEMA.to_string(),
            policy_record_digest,
            clock_quorum_policy_id: clock_quorum_policy.id().as_digest(),
            clock_continuity_policy_id: clock_continuity_policy.id().as_digest(),
            max_transition_ms,
            minimum_eligible_clock_authority_keys,
            require_eligible_algorithm_diversity,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_evaluation_policy_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockEvaluationPermitError> {
        if self.schema != CLOCK_EVALUATION_POLICY_SCHEMA
            || self.policy_record_digest.0 == [0; 32]
            || self.max_transition_ms == 0
            || self.minimum_eligible_clock_authority_keys < 2
        {
            return Err(ClockEvaluationPermitError::InvalidEvaluationPolicy);
        }
        if compute_evaluation_policy_id(self)? != self.id {
            return Err(ClockEvaluationPermitError::EvaluationPolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ClockEvaluationPolicyIdV4 { ClockEvaluationPolicyIdV4(self.id) }
    pub fn policy_record_digest(&self) -> Sha256Digest { self.policy_record_digest }
    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 { ClockQuorumPolicyRevisionIdV1(self.clock_quorum_policy_id) }
    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 { ClockContinuityPolicyRevisionIdV1(self.clock_continuity_policy_id) }
    pub fn max_transition_ms(&self) -> u64 { self.max_transition_ms }
    pub fn minimum_eligible_clock_authority_keys(&self) -> usize { self.minimum_eligible_clock_authority_keys }
    pub fn require_eligible_algorithm_diversity(&self) -> bool { self.require_eligible_algorithm_diversity }
}

macro_rules! digest_id_impl {
    ($type:ty) => {
        impl $type {
            pub fn as_digest(self) -> Sha256Digest { self.0 }
            pub fn to_hex(self) -> String { self.0.to_hex() }
        }
    };
}

digest_id_impl!(ClockQuorumPolicyRevisionIdV1);
digest_id_impl!(ClockContinuityPolicyRevisionIdV1);
digest_id_impl!(ClockEvaluationPolicyIdV4);
digest_id_impl!(ClockBootstrapAnchorIdV2);
digest_id_impl!(ClockEvaluationPermitIdV4);

impl ClockAuthorityKeyV1 {
    pub fn algorithm(&self) -> &SignatureAlgorithm { &self.algorithm }
    pub fn key_id(&self) -> &str { &self.key_id }
}

impl ClockEvaluationPermitV4 {
    pub fn id(&self) -> ClockEvaluationPermitIdV4 { self.id }
    pub fn basis_id(&self) -> ClockBootstrapAnchorIdV2 { self.basis_id }
    pub fn evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 { self.evaluation_policy_id }
    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 { self.clock_quorum_policy_id }
    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 { self.clock_continuity_policy_id }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn evaluation_lower_unix_ms(&self) -> u64 { self.evaluation_lower_unix_ms }
    pub fn evaluation_upper_unix_ms(&self) -> u64 { self.evaluation_upper_unix_ms }
    pub fn minimum_eligible_clock_authority_keys(&self) -> usize { self.minimum_eligible_clock_authority_keys }
    pub fn require_eligible_algorithm_diversity(&self) -> bool { self.require_eligible_algorithm_diversity }
    pub fn eligible_clock_keys(&self) -> &[ClockAuthorityKeyV1] { &self.eligible_clock_keys }
    pub fn clock_quorum_policy(&self) -> &ClockQuorumPolicyRevisionV1 { &self.clock_quorum_policy }
    pub fn clock_continuity_policy(&self) -> &ClockContinuityPolicyRevisionV1 { &self.clock_continuity_policy }
    pub fn runtime_quorum_policy(&self) -> Result<ClockQuorumPolicy, ClockEvaluationPermitError> { self.clock_quorum_policy.to_runtime_policy() }
    pub fn runtime_continuity_policy(&self) -> Result<ClockContinuityPolicy, ClockEvaluationPermitError> { self.clock_continuity_policy.to_runtime_policy() }
}

pub fn derive_bootstrap_clock_evaluation_permit_v4(
    authority: &VerifiedClockBootstrapAuthorityV2,
    claim: &ClockBootstrapClaimV2,
    evaluation_policy: &ClockEvaluationPolicyV4,
    quorum_policy: &ClockQuorumPolicyRevisionV1,
    continuity_policy: &ClockContinuityPolicyRevisionV1,
    trust_snapshot: &TrustSnapshot,
) -> Result<ClockEvaluationPermitV4, ClockEvaluationPermitError> {
    claim.validate().map_err(ClockEvaluationPermitError::BootstrapClaimInvalid)?;
    evaluation_policy.validate()?;
    quorum_policy.validate()?;
    continuity_policy.validate()?;

    if authority.claim_id() != claim.id() { return Err(ClockEvaluationPermitError::BootstrapAuthorityClaimMismatch); }
    if authority.clock_evaluation_policy_id() != claim.clock_evaluation_policy_id() { return Err(ClockEvaluationPermitError::BootstrapAuthorityPolicyMismatch); }
    if claim.clock_evaluation_policy_id() != evaluation_policy.id().as_digest() { return Err(ClockEvaluationPermitError::BootstrapPolicyMismatch); }
    if evaluation_policy.clock_quorum_policy_id() != quorum_policy.id() { return Err(ClockEvaluationPermitError::QuorumPolicyMismatch); }
    if evaluation_policy.clock_continuity_policy_id() != continuity_policy.id() { return Err(ClockEvaluationPermitError::ContinuityPolicyMismatch); }

    let snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(ClockEvaluationPermitError::TrustSnapshotInvalid)?;
    if claim.trust_snapshot_digest() != snapshot_digest { return Err(ClockEvaluationPermitError::BootstrapSnapshotMismatch); }

    let evaluation_lower_unix_ms = claim.trusted_lower_unix_ms();
    let evaluation_upper_unix_ms = claim.trusted_upper_unix_ms().checked_add(evaluation_policy.max_transition_ms()).ok_or(ClockEvaluationPermitError::TransitionEnvelopeOverflow)?;
    let eligible_clock_keys = eligible_clock_authority_keys_for_envelope(trust_snapshot, evaluation_lower_unix_ms, evaluation_upper_unix_ms, evaluation_policy)?;

    let basis_id = ClockBootstrapAnchorIdV2(compute_bootstrap_anchor_id(authority.id().as_digest(), snapshot_digest, claim.trusted_lower_unix_ms(), claim.trusted_upper_unix_ms())?);
    let permit_id = ClockEvaluationPermitIdV4(compute_permit_id(
        basis_id,
        evaluation_policy.id(),
        quorum_policy.id(),
        continuity_policy.id(),
        snapshot_digest,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        evaluation_policy.minimum_eligible_clock_authority_keys(),
        evaluation_policy.require_eligible_algorithm_diversity(),
        &eligible_clock_keys,
    )?);

    Ok(ClockEvaluationPermitV4 {
        id: permit_id,
        basis_id,
        evaluation_policy_id: evaluation_policy.id(),
        clock_quorum_policy_id: quorum_policy.id(),
        clock_continuity_policy_id: continuity_policy.id(),
        trust_snapshot_digest: snapshot_digest,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        minimum_eligible_clock_authority_keys: evaluation_policy.minimum_eligible_clock_authority_keys(),
        require_eligible_algorithm_diversity: evaluation_policy.require_eligible_algorithm_diversity(),
        eligible_clock_keys,
        clock_quorum_policy: quorum_policy.clone(),
        clock_continuity_policy: continuity_policy.clone(),
    })
}

fn eligible_clock_authority_keys_for_envelope(
    trust_snapshot: &TrustSnapshot,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    policy: &ClockEvaluationPolicyV4,
) -> Result<Vec<ClockAuthorityKeyV1>, ClockEvaluationPermitError> {
    let snapshot_lower_ms = seconds_to_millis(trust_snapshot.issued_at_unix_s)?;
    let snapshot_upper_ms = seconds_to_millis(trust_snapshot.expires_at_unix_s)?;
    if lower_unix_ms > upper_unix_ms || lower_unix_ms < snapshot_lower_ms || upper_unix_ms >= snapshot_upper_ms {
        return Err(ClockEvaluationPermitError::SnapshotNotValidForEnvelope);
    }
    let mut eligible = Vec::new();
    for key in &trust_snapshot.keys {
        if key.status != KeyLifecycleStatus::Active || !key.usages.contains(&KeyUsage::ClockAuthority) { continue; }
        let key_lower_ms = seconds_to_millis(key.not_before_unix_s)?;
        if lower_unix_ms < key_lower_ms { continue; }
        if let Some(not_after_unix_s) = key.not_after_unix_s {
            let key_upper_ms = seconds_to_millis(not_after_unix_s)?;
            if upper_unix_ms >= key_upper_ms { continue; }
        }
        eligible.push(ClockAuthorityKeyV1 { algorithm: key.algorithm.clone(), key_id: key.key_id.clone() });
    }
    eligible.sort_by(|left, right| (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str())));
    if eligible.len() < policy.minimum_eligible_clock_authority_keys() {
        return Err(ClockEvaluationPermitError::InsufficientClockAuthorityKeys { actual: eligible.len(), required: policy.minimum_eligible_clock_authority_keys() });
    }
    if policy.require_eligible_algorithm_diversity() {
        let algorithms = eligible.iter().map(|key| key.algorithm.clone()).collect::<BTreeSet<_>>();
        if algorithms.len() < 2 { return Err(ClockEvaluationPermitError::EligibleAlgorithmDiversityMissing); }
    }
    Ok(eligible)
}

fn compute_quorum_policy_id(policy: &ClockQuorumPolicyRevisionV1) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let bytes = canonical_json_bytes([
        ("maximum_consensus_width_ms", Value::from(policy.maximum_consensus_width_ms)),
        ("maximum_observations", Value::from(policy.maximum_observations as u64)),
        ("maximum_uncertainty_ms", Value::from(policy.maximum_uncertainty_ms)),
        ("minimum_distinct_sources", Value::from(policy.minimum_distinct_sources as u64)),
        ("require_algorithm_diversity", Value::Bool(policy.require_algorithm_diversity)),
        ("schema", Value::String(policy.schema.clone())),
    ])?;
    Ok(domain_hash(CLOCK_QUORUM_POLICY_DOMAIN, &bytes))
}

fn compute_continuity_policy_id(policy: &ClockContinuityPolicyRevisionV1) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let bytes = canonical_json_bytes([
        ("maximum_consensus_jump_ms", Value::from(policy.maximum_consensus_jump_ms)),
        ("maximum_epoch_step", Value::from(policy.maximum_epoch_step)),
        ("maximum_forward_gap_ms", Value::from(policy.maximum_forward_gap_ms)),
        ("minimum_shared_sources", Value::from(policy.minimum_shared_sources as u64)),
        ("require_shared_algorithm", Value::Bool(policy.require_shared_algorithm)),
        ("schema", Value::String(policy.schema.clone())),
    ])?;
    Ok(domain_hash(CLOCK_CONTINUITY_POLICY_DOMAIN, &bytes))
}

fn compute_evaluation_policy_id(policy: &ClockEvaluationPolicyV4) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let bytes = canonical_json_bytes([
        ("clock_continuity_policy_id", Value::String(policy.clock_continuity_policy_id.to_hex())),
        ("clock_quorum_policy_id", Value::String(policy.clock_quorum_policy_id.to_hex())),
        ("max_transition_ms", Value::from(policy.max_transition_ms)),
        ("minimum_eligible_clock_authority_keys", Value::from(policy.minimum_eligible_clock_authority_keys as u64)),
        ("policy_record_digest", Value::String(policy.policy_record_digest.to_hex())),
        ("require_eligible_algorithm_diversity", Value::Bool(policy.require_eligible_algorithm_diversity)),
        ("schema", Value::String(policy.schema.clone())),
    ])?;
    Ok(domain_hash(CLOCK_EVALUATION_POLICY_DOMAIN, &bytes))
}

fn compute_bootstrap_anchor_id(
    authority_record_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    trusted_lower_unix_ms: u64,
    trusted_upper_unix_ms: u64,
) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let bytes = canonical_json_bytes([
        ("authority_record_digest", Value::String(authority_record_digest.to_hex())),
        ("kind", Value::String(BOOTSTRAP_ANCHOR_KIND.to_string())),
        ("schema", Value::String(CLOCK_BOOTSTRAP_ANCHOR_SCHEMA.to_string())),
        ("trust_snapshot_digest", Value::String(trust_snapshot_digest.to_hex())),
        ("trusted_lower_unix_ms", Value::from(trusted_lower_unix_ms)),
        ("trusted_upper_unix_ms", Value::from(trusted_upper_unix_ms)),
    ])?;
    Ok(domain_hash(CLOCK_BOOTSTRAP_ANCHOR_DOMAIN, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn compute_permit_id(
    basis_id: ClockBootstrapAnchorIdV2,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: &[ClockAuthorityKeyV1],
) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let eligible_keys_json = eligible_clock_keys.iter().map(|key| Value::Array(vec![Value::String(algorithm_identity(&key.algorithm)), Value::String(key.key_id.clone())])).collect::<Vec<_>>();
    let bytes = canonical_json_bytes([
        ("basis_id", Value::String(basis_id.to_hex())),
        ("basis_kind", Value::String(PERMIT_BASIS_KIND.to_string())),
        ("clock_continuity_policy_id", Value::String(continuity_policy_id.to_hex())),
        ("clock_quorum_policy_id", Value::String(quorum_policy_id.to_hex())),
        ("eligible_clock_keys", Value::Array(eligible_keys_json)),
        ("evaluation_lower_unix_ms", Value::from(evaluation_lower_unix_ms)),
        ("evaluation_policy_id", Value::String(evaluation_policy_id.to_hex())),
        ("evaluation_upper_unix_ms", Value::from(evaluation_upper_unix_ms)),
        ("minimum_eligible_clock_authority_keys", Value::from(minimum_eligible_clock_authority_keys as u64)),
        ("require_eligible_algorithm_diversity", Value::Bool(require_eligible_algorithm_diversity)),
        ("schema", Value::String(CLOCK_EVALUATION_PERMIT_SCHEMA.to_string())),
        ("trust_snapshot_digest", Value::String(trust_snapshot_digest.to_hex())),
    ])?;
    Ok(domain_hash(CLOCK_EVALUATION_PERMIT_DOMAIN, &bytes))
}

fn algorithm_identity(algorithm: &SignatureAlgorithm) -> String {
    match algorithm {
        SignatureAlgorithm::Ed25519 => "Ed25519".to_string(),
        SignatureAlgorithm::MlDsa65 => "MlDsa65".to_string(),
        SignatureAlgorithm::MlDsa87 => "MlDsa87".to_string(),
        SignatureAlgorithm::Other(name) => format!("Other:{name}"),
    }
}

fn seconds_to_millis(value: u64) -> Result<u64, ClockEvaluationPermitError> {
    value.checked_mul(1_000).ok_or(ClockEvaluationPermitError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(entries: [(&str, Value); N]) -> Result<Vec<u8>, ClockEvaluationPermitError> {
    let map = entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ClockEvaluationPermitError::Encoding(error.to_string()))
}
