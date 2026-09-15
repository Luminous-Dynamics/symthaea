// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold authorization for one exact lineage-bound upgrade finalization context.
//!
//! This crate authorizes intent only. It freezes the exact lineage-bound context plus the exact
//! threshold policy/floor into a prepared payload and accepts only an interval-safe threshold
//! ceremony over that payload. A later fresh execution theorem remains mandatory.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::KeyUsage;
use symthaea_fabrication_lineage_bound_finalization_context::LineageBoundUpgradeFinalizationContextV1;
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};

pub const LINEAGE_BOUND_UPGRADE_FINALIZATION_PURPOSE: &str =
    "lineage-bound-upgrade-finalization-v1";
pub const PREPARED_LINEAGE_BOUND_FINALIZATION_AUTHORIZATION_SCHEMA: &str =
    "symthaea.fabrication.prepared-lineage-bound-finalization-authorization.v1";
pub const AUTHORIZED_LINEAGE_BOUND_FINALIZATION_SCHEMA: &str =
    "symthaea.fabrication.authorized-lineage-bound-finalization.v1";

const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const THRESHOLD_FLOOR_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-threshold-floor.v1\0";
const PREPARED_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-lineage-bound-finalization-authorization.v1\0";
const AUTHORIZED_DOMAIN: &[u8] =
    b"symthaea.fabrication.authorized-lineage-bound-finalization.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LineageBoundFinalizationThresholdFloorV1 {
    pub minimum_distinct_signers: usize,
    pub require_algorithm_diversity: bool,
}

impl Default for LineageBoundFinalizationThresholdFloorV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_signers: 2,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedLineageBoundFinalizationAuthorizationIdV1(Sha256Digest);

impl PreparedLineageBoundFinalizationAuthorizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorizedLineageBoundFinalizationIdV1(Sha256Digest);

impl AuthorizedLineageBoundFinalizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedLineageBoundFinalizationAuthorizationV1 {
    id: PreparedLineageBoundFinalizationAuthorizationIdV1,
    context: LineageBoundUpgradeFinalizationContextV1,
    threshold_policy_digest: Sha256Digest,
    threshold_floor_digest: Sha256Digest,
    minimum_distinct_signers: usize,
    require_algorithm_diversity: bool,
}

impl PreparedLineageBoundFinalizationAuthorizationV1 {
    pub fn id(&self) -> PreparedLineageBoundFinalizationAuthorizationIdV1 { self.id }
    pub fn signing_payload_digest(&self) -> Sha256Digest { self.id.as_digest() }
    pub fn context(&self) -> &LineageBoundUpgradeFinalizationContextV1 { &self.context }
    pub fn threshold_policy_digest(&self) -> Sha256Digest { self.threshold_policy_digest }
    pub fn threshold_floor_digest(&self) -> Sha256Digest { self.threshold_floor_digest }
    pub fn minimum_distinct_signers(&self) -> usize { self.minimum_distinct_signers }
    pub fn require_algorithm_diversity(&self) -> bool { self.require_algorithm_diversity }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct AuthorizedLineageBoundFinalizationV1 {
    id: AuthorizedLineageBoundFinalizationIdV1,
    prepared_id: PreparedLineageBoundFinalizationAuthorizationIdV1,
    context: LineageBoundUpgradeFinalizationContextV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    threshold_floor_digest: Sha256Digest,
    signer_count: usize,
    algorithm_count: usize,
}

impl AuthorizedLineageBoundFinalizationV1 {
    pub fn id(&self) -> AuthorizedLineageBoundFinalizationIdV1 { self.id }
    pub fn prepared_id(&self) -> PreparedLineageBoundFinalizationAuthorizationIdV1 { self.prepared_id }
    pub fn context(&self) -> &LineageBoundUpgradeFinalizationContextV1 { &self.context }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 { self.threshold_ceremony_id }
    pub fn threshold_ceremony_digest(&self) -> Sha256Digest { self.threshold_ceremony_digest }
    pub fn threshold_policy_digest(&self) -> Sha256Digest { self.threshold_policy_digest }
    pub fn threshold_floor_digest(&self) -> Sha256Digest { self.threshold_floor_digest }
    pub fn signer_count(&self) -> usize { self.signer_count }
    pub fn algorithm_count(&self) -> usize { self.algorithm_count }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizationAuthorizationError {
    InvalidThresholdFloor,
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    ThresholdPolicyBelowFloor,
    Encoding(String),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    InsufficientCeremonySigners { actual: usize, required: usize },
    MissingCeremonyAlgorithmDiversity,
}

#[derive(Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}

#[derive(Debug, Clone, Serialize)]
struct ThresholdFloorCommitment {
    minimum_distinct_signers: usize,
    require_algorithm_diversity: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PreparedAuthorizationCommitment {
    schema: &'static str,
    context_id: String,
    context_policy_digest: String,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    operational_state_digest: String,
    operational_lineage_digest: String,
    hardware_authority_set_digest: String,
    current_transparency_log_digest: String,
    current_checkpoint_digest: String,
    threshold_policy_digest: String,
    threshold_floor_digest: String,
    trust_snapshot_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct AuthorizedFinalizationCommitment {
    schema: &'static str,
    prepared_id: String,
    context_id: String,
    predecessor_root_digest: String,
    upgrade_cycle_sequence: u64,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
    threshold_policy_digest: String,
    threshold_floor_digest: String,
    signer_count: usize,
    algorithm_count: usize,
}

pub fn prepare_lineage_bound_finalization_authorization_v1(
    context: LineageBoundUpgradeFinalizationContextV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    threshold_floor: &LineageBoundFinalizationThresholdFloorV1,
) -> Result<PreparedLineageBoundFinalizationAuthorizationV1, LineageBoundFinalizationAuthorizationError> {
    if threshold_floor.minimum_distinct_signers == 0
        || threshold_floor.minimum_distinct_signers > MAX_THRESHOLD_APPROVALS
    {
        return Err(LineageBoundFinalizationAuthorizationError::InvalidThresholdFloor);
    }
    if !valid_threshold_policy(threshold_policy) {
        return Err(LineageBoundFinalizationAuthorizationError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        return Err(LineageBoundFinalizationAuthorizationError::ThresholdPolicyUsageMismatch);
    }
    if threshold_policy.minimum_distinct_signers < threshold_floor.minimum_distinct_signers
        || (threshold_floor.require_algorithm_diversity && !threshold_policy.require_algorithm_diversity)
    {
        return Err(LineageBoundFinalizationAuthorizationError::ThresholdPolicyBelowFloor);
    }

    let threshold_policy_digest = digest_threshold_policy(threshold_policy)?;
    let threshold_floor_digest = hash_serializable(
        THRESHOLD_FLOOR_DOMAIN,
        &ThresholdFloorCommitment {
            minimum_distinct_signers: threshold_floor.minimum_distinct_signers,
            require_algorithm_diversity: threshold_floor.require_algorithm_diversity,
        },
    )?;
    let commitment = PreparedAuthorizationCommitment {
        schema: PREPARED_LINEAGE_BOUND_FINALIZATION_AUTHORIZATION_SCHEMA,
        context_id: context.id().to_hex(),
        context_policy_digest: context.context_policy_digest().to_hex(),
        predecessor_root_digest: context.predecessor_root_digest().to_hex(),
        predecessor_current_head_digest: context.predecessor_current_head_digest().to_hex(),
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        upgrade_cycle_sequence: context.upgrade_cycle_sequence(),
        operational_state_digest: context.operational_state_digest().to_hex(),
        operational_lineage_digest: context.operational_lineage_digest().to_hex(),
        hardware_authority_set_digest: context.hardware_authority_set_digest().to_hex(),
        current_transparency_log_digest: context.current_transparency_log_digest().to_hex(),
        current_checkpoint_digest: context.current_checkpoint_digest().to_hex(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        threshold_floor_digest: threshold_floor_digest.to_hex(),
        trust_snapshot_digest: context.current_trust_snapshot_digest().to_hex(),
        compromise_tracker_digest: context.current_compromise_tracker_digest().to_hex(),
        clock_envelope_id: context.current_clock_envelope_id().to_hex(),
    };
    let id = PreparedLineageBoundFinalizationAuthorizationIdV1(
        hash_serializable(PREPARED_DOMAIN, &commitment)?,
    );

    Ok(PreparedLineageBoundFinalizationAuthorizationV1 {
        id,
        context,
        threshold_policy_digest,
        threshold_floor_digest,
        minimum_distinct_signers: threshold_floor.minimum_distinct_signers,
        require_algorithm_diversity: threshold_floor.require_algorithm_diversity,
    })
}

pub fn authorize_lineage_bound_finalization_v1(
    prepared: PreparedLineageBoundFinalizationAuthorizationV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<AuthorizedLineageBoundFinalizationV1, LineageBoundFinalizationAuthorizationError> {
    if ceremony.purpose() != LINEAGE_BOUND_UPGRADE_FINALIZATION_PURPOSE {
        return Err(LineageBoundFinalizationAuthorizationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(LineageBoundFinalizationAuthorizationError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(LineageBoundFinalizationAuthorizationError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.context.current_trust_snapshot_digest() {
        return Err(LineageBoundFinalizationAuthorizationError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.context.current_compromise_tracker_digest() {
        return Err(LineageBoundFinalizationAuthorizationError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.context.current_clock_envelope_id() {
        return Err(LineageBoundFinalizationAuthorizationError::ClockEnvelopeMismatch);
    }
    if ceremony.signers().len() < prepared.minimum_distinct_signers {
        return Err(LineageBoundFinalizationAuthorizationError::InsufficientCeremonySigners {
            actual: ceremony.signers().len(), required: prepared.minimum_distinct_signers,
        });
    }
    let algorithms = ceremony.signers().iter()
        .map(|(algorithm, _)| algorithm.clone())
        .collect::<BTreeSet<_>>();
    if prepared.require_algorithm_diversity && algorithms.len() < 2 {
        return Err(LineageBoundFinalizationAuthorizationError::MissingCeremonyAlgorithmDiversity);
    }

    let commitment = AuthorizedFinalizationCommitment {
        schema: AUTHORIZED_LINEAGE_BOUND_FINALIZATION_SCHEMA,
        prepared_id: prepared.id().to_hex(),
        context_id: prepared.context.id().to_hex(),
        predecessor_root_digest: prepared.context.predecessor_root_digest().to_hex(),
        upgrade_cycle_sequence: prepared.context.upgrade_cycle_sequence(),
        threshold_ceremony_id: ceremony.id().to_hex(),
        threshold_ceremony_digest: ceremony.ceremony_digest().to_hex(),
        threshold_policy_digest: prepared.threshold_policy_digest.to_hex(),
        threshold_floor_digest: prepared.threshold_floor_digest.to_hex(),
        signer_count: ceremony.signers().len(),
        algorithm_count: algorithms.len(),
    };
    let id = AuthorizedLineageBoundFinalizationIdV1(
        hash_serializable(AUTHORIZED_DOMAIN, &commitment)?,
    );

    Ok(AuthorizedLineageBoundFinalizationV1 {
        id,
        prepared_id: prepared.id,
        context: prepared.context,
        threshold_ceremony_id: ceremony.id(),
        threshold_ceremony_digest: ceremony.ceremony_digest(),
        threshold_policy_digest: prepared.threshold_policy_digest,
        threshold_floor_digest: prepared.threshold_floor_digest,
        signer_count: ceremony.signers().len(),
        algorithm_count: algorithms.len(),
    })
}

fn valid_threshold_policy(policy: &ThresholdCeremonyPolicy) -> bool {
    policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy.required_algorithms.iter().all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !id.trim().is_empty()
                    && id == id.trim()
                    && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
                    && !id.chars().any(char::is_control)
            })
        })
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, LineageBoundFinalizationAuthorizationError> {
    hash_serializable(
        THRESHOLD_POLICY_DOMAIN,
        &ThresholdPolicyCommitment {
            minimum_distinct_signers: policy.minimum_distinct_signers,
            maximum_approvals: policy.maximum_approvals,
            require_algorithm_diversity: policy.require_algorithm_diversity,
            required_algorithms: &policy.required_algorithms,
            allowed_key_ids: &policy.allowed_key_ids,
            key_usage: policy.key_usage,
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizationAuthorizationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizationAuthorizationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
