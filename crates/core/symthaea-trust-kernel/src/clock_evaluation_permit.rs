// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-bound, whole-envelope clock evaluation permits.
//!
//! This module deliberately stops before clock-observation admission. A permit
//! proves only which trust snapshot, policy, envelope, and clock-authority keys
//! are eligible to participate in a later evaluation step.
//!
//! ```text
//! verified bootstrap authority
//! + exact bootstrap claim
//! + exact evaluation policy
//! + exact trust snapshot
//! != accepted clock evidence
//! ```

use crate::clock_bootstrap_authority::{
    ClockBootstrapAuthorityError, ClockBootstrapClaimV2, VerifiedClockBootstrapAuthorityV2,
};
use crate::digest::{Sha256Digest, domain_hash};
use crate::signature::SignatureAlgorithm;
use crate::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, TrustSnapshotError, digest_trust_snapshot,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CLOCK_EVALUATION_POLICY_SCHEMA: &str = "symthaea.trust.clock-evaluation-policy.v2";
pub const CLOCK_BOOTSTRAP_ANCHOR_SCHEMA: &str = "symthaea.trust.clock-bootstrap-anchor.v2";
pub const CLOCK_EVALUATION_PERMIT_SCHEMA: &str = "symthaea.trust.clock-evaluation-permit.v2";

const CLOCK_EVALUATION_POLICY_DOMAIN: &[u8] = b"symthaea.trust.clock-evaluation-policy.v2\0";
const CLOCK_BOOTSTRAP_ANCHOR_DOMAIN: &[u8] = b"symthaea.trust.clock-bootstrap-anchor.v2\0";
const CLOCK_EVALUATION_PERMIT_DOMAIN: &[u8] = b"symthaea.trust.clock-evaluation-permit.v2\0";
const BOOTSTRAP_ANCHOR_KIND: &str = "ExternalBootstrap";
const PERMIT_BASIS_KIND: &str = "BootstrapAnchor";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockEvaluationPolicyV2 {
    schema: String,
    policy_record_digest: Sha256Digest,
    max_transition_ms: u64,
    minimum_clock_authority_keys: usize,
    require_algorithm_diversity: bool,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockEvaluationPolicyIdV2(Sha256Digest);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockBootstrapAnchorIdV2(Sha256Digest);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockEvaluationPermitIdV2(Sha256Digest);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockAuthorityKeyV2 {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockEvaluationPermitV2 {
    id: ClockEvaluationPermitIdV2,
    basis_id: ClockBootstrapAnchorIdV2,
    policy_id: ClockEvaluationPolicyIdV2,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_clock_authority_keys: usize,
    require_algorithm_diversity: bool,
    eligible_clock_keys: Vec<ClockAuthorityKeyV2>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockEvaluationPermitError {
    InvalidPolicy,
    PolicyIdentityMismatch,
    BootstrapClaimInvalid(ClockBootstrapAuthorityError),
    BootstrapAuthorityClaimMismatch,
    BootstrapAuthorityPolicyMismatch,
    BootstrapPolicyMismatch,
    BootstrapSnapshotMismatch,
    TrustSnapshotInvalid(TrustSnapshotError),
    TransitionEnvelopeOverflow,
    TimeScaleOverflow,
    SnapshotNotValidForEnvelope,
    InsufficientClockAuthorityKeys { actual: usize, required: usize },
    EligibleAlgorithmDiversityMissing,
    Encoding(String),
}

impl ClockEvaluationPolicyV2 {
    pub fn new(
        policy_record_digest: Sha256Digest,
        max_transition_ms: u64,
        minimum_clock_authority_keys: usize,
        require_algorithm_diversity: bool,
    ) -> Result<Self, ClockEvaluationPermitError> {
        if max_transition_ms == 0 || minimum_clock_authority_keys < 2 {
            return Err(ClockEvaluationPermitError::InvalidPolicy);
        }
        let mut value = Self {
            schema: CLOCK_EVALUATION_POLICY_SCHEMA.to_string(),
            policy_record_digest,
            max_transition_ms,
            minimum_clock_authority_keys,
            require_algorithm_diversity,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_policy_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockEvaluationPermitError> {
        if self.schema != CLOCK_EVALUATION_POLICY_SCHEMA
            || self.max_transition_ms == 0
            || self.minimum_clock_authority_keys < 2
        {
            return Err(ClockEvaluationPermitError::InvalidPolicy);
        }
        if compute_policy_id(self)? != self.id {
            return Err(ClockEvaluationPermitError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ClockEvaluationPolicyIdV2 {
        ClockEvaluationPolicyIdV2(self.id)
    }

    pub fn policy_record_digest(&self) -> Sha256Digest {
        self.policy_record_digest
    }

    pub fn max_transition_ms(&self) -> u64 {
        self.max_transition_ms
    }

    pub fn minimum_clock_authority_keys(&self) -> usize {
        self.minimum_clock_authority_keys
    }

    pub fn require_algorithm_diversity(&self) -> bool {
        self.require_algorithm_diversity
    }
}

macro_rules! digest_id_impl {
    ($type:ty) => {
        impl $type {
            pub fn as_digest(self) -> Sha256Digest {
                self.0
            }

            pub fn to_hex(self) -> String {
                self.0.to_hex()
            }
        }
    };
}

digest_id_impl!(ClockEvaluationPolicyIdV2);
digest_id_impl!(ClockBootstrapAnchorIdV2);
digest_id_impl!(ClockEvaluationPermitIdV2);

impl ClockAuthorityKeyV2 {
    pub fn algorithm(&self) -> &SignatureAlgorithm {
        &self.algorithm
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

impl ClockEvaluationPermitV2 {
    pub fn id(&self) -> ClockEvaluationPermitIdV2 {
        self.id
    }

    pub fn basis_id(&self) -> ClockBootstrapAnchorIdV2 {
        self.basis_id
    }

    pub fn policy_id(&self) -> ClockEvaluationPolicyIdV2 {
        self.policy_id
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn evaluation_lower_unix_ms(&self) -> u64 {
        self.evaluation_lower_unix_ms
    }

    pub fn evaluation_upper_unix_ms(&self) -> u64 {
        self.evaluation_upper_unix_ms
    }

    pub fn minimum_clock_authority_keys(&self) -> usize {
        self.minimum_clock_authority_keys
    }

    pub fn require_algorithm_diversity(&self) -> bool {
        self.require_algorithm_diversity
    }

    pub fn eligible_clock_keys(&self) -> &[ClockAuthorityKeyV2] {
        &self.eligible_clock_keys
    }
}

pub fn derive_bootstrap_clock_evaluation_permit_v2(
    authority: &VerifiedClockBootstrapAuthorityV2,
    claim: &ClockBootstrapClaimV2,
    policy: &ClockEvaluationPolicyV2,
    trust_snapshot: &TrustSnapshot,
) -> Result<ClockEvaluationPermitV2, ClockEvaluationPermitError> {
    claim
        .validate()
        .map_err(ClockEvaluationPermitError::BootstrapClaimInvalid)?;
    policy.validate()?;

    if authority.claim_id() != claim.id() {
        return Err(ClockEvaluationPermitError::BootstrapAuthorityClaimMismatch);
    }
    if authority.clock_evaluation_policy_id() != claim.clock_evaluation_policy_id() {
        return Err(ClockEvaluationPermitError::BootstrapAuthorityPolicyMismatch);
    }
    if claim.clock_evaluation_policy_id() != policy.id().as_digest() {
        return Err(ClockEvaluationPermitError::BootstrapPolicyMismatch);
    }

    let snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(ClockEvaluationPermitError::TrustSnapshotInvalid)?;
    if claim.trust_snapshot_digest() != snapshot_digest {
        return Err(ClockEvaluationPermitError::BootstrapSnapshotMismatch);
    }

    let evaluation_lower_unix_ms = claim.trusted_lower_unix_ms();
    let evaluation_upper_unix_ms = claim
        .trusted_upper_unix_ms()
        .checked_add(policy.max_transition_ms())
        .ok_or(ClockEvaluationPermitError::TransitionEnvelopeOverflow)?;

    let eligible_clock_keys = eligible_clock_authority_keys_for_envelope(
        trust_snapshot,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        policy,
    )?;

    let basis_id = ClockBootstrapAnchorIdV2(compute_bootstrap_anchor_id(
        authority.id().as_digest(),
        snapshot_digest,
        claim.trusted_lower_unix_ms(),
        claim.trusted_upper_unix_ms(),
    )?);

    let permit_id = ClockEvaluationPermitIdV2(compute_permit_id(
        basis_id,
        policy.id(),
        snapshot_digest,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        policy.minimum_clock_authority_keys(),
        policy.require_algorithm_diversity(),
        &eligible_clock_keys,
    )?);

    Ok(ClockEvaluationPermitV2 {
        id: permit_id,
        basis_id,
        policy_id: policy.id(),
        trust_snapshot_digest: snapshot_digest,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        minimum_clock_authority_keys: policy.minimum_clock_authority_keys(),
        require_algorithm_diversity: policy.require_algorithm_diversity(),
        eligible_clock_keys,
    })
}

fn eligible_clock_authority_keys_for_envelope(
    trust_snapshot: &TrustSnapshot,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    policy: &ClockEvaluationPolicyV2,
) -> Result<Vec<ClockAuthorityKeyV2>, ClockEvaluationPermitError> {
    let snapshot_lower_ms = seconds_to_millis(trust_snapshot.issued_at_unix_s)?;
    let snapshot_upper_ms = seconds_to_millis(trust_snapshot.expires_at_unix_s)?;
    if lower_unix_ms > upper_unix_ms
        || lower_unix_ms < snapshot_lower_ms
        || upper_unix_ms >= snapshot_upper_ms
    {
        return Err(ClockEvaluationPermitError::SnapshotNotValidForEnvelope);
    }

    let mut eligible = Vec::new();
    for key in &trust_snapshot.keys {
        if key.status != KeyLifecycleStatus::Active || !key.usages.contains(&KeyUsage::ClockAuthority)
        {
            continue;
        }

        let key_lower_ms = seconds_to_millis(key.not_before_unix_s)?;
        if lower_unix_ms < key_lower_ms {
            continue;
        }
        if let Some(not_after_unix_s) = key.not_after_unix_s {
            let key_upper_ms = seconds_to_millis(not_after_unix_s)?;
            if upper_unix_ms >= key_upper_ms {
                continue;
            }
        }

        eligible.push(ClockAuthorityKeyV2 {
            algorithm: key.algorithm.clone(),
            key_id: key.key_id.clone(),
        });
    }

    eligible.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });

    if eligible.len() < policy.minimum_clock_authority_keys() {
        return Err(ClockEvaluationPermitError::InsufficientClockAuthorityKeys {
            actual: eligible.len(),
            required: policy.minimum_clock_authority_keys(),
        });
    }

    if policy.require_algorithm_diversity() {
        let algorithms = eligible
            .iter()
            .map(|key| key.algorithm.clone())
            .collect::<BTreeSet<_>>();
        if algorithms.len() < 2 {
            return Err(ClockEvaluationPermitError::EligibleAlgorithmDiversityMissing);
        }
    }

    Ok(eligible)
}

fn compute_policy_id(
    policy: &ClockEvaluationPolicyV2,
) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let bytes = canonical_json_bytes([
        (
            "max_transition_ms",
            Value::from(policy.max_transition_ms),
        ),
        (
            "minimum_clock_authority_keys",
            Value::from(policy.minimum_clock_authority_keys as u64),
        ),
        (
            "policy_record_digest",
            Value::String(policy.policy_record_digest.to_hex()),
        ),
        (
            "require_algorithm_diversity",
            Value::Bool(policy.require_algorithm_diversity),
        ),
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
        (
            "authority_record_digest",
            Value::String(authority_record_digest.to_hex()),
        ),
        ("kind", Value::String(BOOTSTRAP_ANCHOR_KIND.to_string())),
        ("schema", Value::String(CLOCK_BOOTSTRAP_ANCHOR_SCHEMA.to_string())),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
        ("trusted_lower_unix_ms", Value::from(trusted_lower_unix_ms)),
        ("trusted_upper_unix_ms", Value::from(trusted_upper_unix_ms)),
    ])?;
    Ok(domain_hash(CLOCK_BOOTSTRAP_ANCHOR_DOMAIN, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn compute_permit_id(
    basis_id: ClockBootstrapAnchorIdV2,
    policy_id: ClockEvaluationPolicyIdV2,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_clock_authority_keys: usize,
    require_algorithm_diversity: bool,
    eligible_clock_keys: &[ClockAuthorityKeyV2],
) -> Result<Sha256Digest, ClockEvaluationPermitError> {
    let eligible_keys_json = eligible_clock_keys
        .iter()
        .map(|key| {
            Value::Array(vec![
                Value::String(algorithm_identity(&key.algorithm)),
                Value::String(key.key_id.clone()),
            ])
        })
        .collect::<Vec<_>>();

    let bytes = canonical_json_bytes([
        ("basis_id", Value::String(basis_id.to_hex())),
        ("basis_kind", Value::String(PERMIT_BASIS_KIND.to_string())),
        ("eligible_clock_keys", Value::Array(eligible_keys_json)),
        (
            "evaluation_lower_unix_ms",
            Value::from(evaluation_lower_unix_ms),
        ),
        (
            "evaluation_upper_unix_ms",
            Value::from(evaluation_upper_unix_ms),
        ),
        (
            "minimum_clock_authority_keys",
            Value::from(minimum_clock_authority_keys as u64),
        ),
        ("policy_id", Value::String(policy_id.to_hex())),
        (
            "require_algorithm_diversity",
            Value::Bool(require_algorithm_diversity),
        ),
        ("schema", Value::String(CLOCK_EVALUATION_PERMIT_SCHEMA.to_string())),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
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

fn seconds_to_millis(seconds: u64) -> Result<u64, ClockEvaluationPermitError> {
    seconds
        .checked_mul(1_000)
        .ok_or(ClockEvaluationPermitError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(
    fields: [(&str, Value); N],
) -> Result<Vec<u8>, ClockEvaluationPermitError> {
    let map = fields
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ClockEvaluationPermitError::Encoding(error.to_string()))
}
