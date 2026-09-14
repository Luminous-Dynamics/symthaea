// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold authorization for one exact hardened upgrade finalization context.
//!
//! This crate still does not execute irreversible finalization. It freezes the exact finalization
//! context plus exact threshold policy into a prepared payload, then accepts only an interval-safe
//! `ClockGovernedThresholdCeremonyV1` over that payload under the exact current trust, containment
//! and clock identities committed by the context. A separate fresh execution theorem remains
//! required before predecessor authority can actually be retired.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::KeyUsage;
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_upgrade_finalization_context::{
    QualifiedUpgradeFinalizationContextIdV1, QualifiedUpgradeFinalizationContextV1,
};

pub const CLOCK_GOVERNED_UPGRADE_FINALIZATION_PURPOSE: &str =
    "clock-governed-upgrade-finalization-v1";
pub const PREPARED_UPGRADE_FINALIZATION_AUTHORIZATION_SCHEMA: &str =
    "symthaea.fabrication.prepared-upgrade-finalization-authorization.v1";
pub const AUTHORIZED_CLOCK_GOVERNED_UPGRADE_FINALIZATION_SCHEMA: &str =
    "symthaea.fabrication.authorized-clock-governed-upgrade-finalization.v1";

const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const THRESHOLD_FLOOR_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-threshold-floor.v1\0";
const PREPARED_AUTHORIZATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-upgrade-finalization-authorization.v1\0";
const AUTHORIZED_FINALIZATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.authorized-clock-governed-upgrade-finalization.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradeFinalizationThresholdFloorV1 {
    pub minimum_distinct_signers: usize,
    pub require_algorithm_diversity: bool,
}

impl Default for UpgradeFinalizationThresholdFloorV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_signers: 2,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedUpgradeFinalizationAuthorizationIdV1(Sha256Digest);

impl PreparedUpgradeFinalizationAuthorizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorizedClockGovernedUpgradeFinalizationIdV1(Sha256Digest);

impl AuthorizedClockGovernedUpgradeFinalizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Exact finalization context plus exact threshold policy, ready for signatures.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedUpgradeFinalizationAuthorizationV1 {
    id: PreparedUpgradeFinalizationAuthorizationIdV1,
    context: QualifiedUpgradeFinalizationContextV1,
    threshold_policy_digest: Sha256Digest,
    threshold_floor_digest: Sha256Digest,
    minimum_distinct_signers: usize,
    require_algorithm_diversity: bool,
}

impl PreparedUpgradeFinalizationAuthorizationV1 {
    pub fn id(&self) -> PreparedUpgradeFinalizationAuthorizationIdV1 {
        self.id
    }

    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }

    pub fn context(&self) -> &QualifiedUpgradeFinalizationContextV1 {
        &self.context
    }

    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }

    pub fn threshold_floor_digest(&self) -> Sha256Digest {
        self.threshold_floor_digest
    }
}

/// Threshold-authorized finalization intent. This is deliberately not an execution permit.
#[derive(Debug, Clone)]
#[must_use]
pub struct AuthorizedClockGovernedUpgradeFinalizationV1 {
    id: AuthorizedClockGovernedUpgradeFinalizationIdV1,
    prepared_id: PreparedUpgradeFinalizationAuthorizationIdV1,
    context: QualifiedUpgradeFinalizationContextV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    threshold_floor_digest: Sha256Digest,
    signer_count: usize,
    algorithm_count: usize,
}

impl AuthorizedClockGovernedUpgradeFinalizationV1 {
    pub fn id(&self) -> AuthorizedClockGovernedUpgradeFinalizationIdV1 {
        self.id
    }

    pub fn prepared_id(&self) -> PreparedUpgradeFinalizationAuthorizationIdV1 {
        self.prepared_id
    }

    pub fn context(&self) -> &QualifiedUpgradeFinalizationContextV1 {
        &self.context
    }

    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.threshold_ceremony_id
    }

    pub fn threshold_ceremony_digest(&self) -> Sha256Digest {
        self.threshold_ceremony_digest
    }

    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }

    pub fn threshold_floor_digest(&self) -> Sha256Digest {
        self.threshold_floor_digest
    }

    pub fn signer_count(&self) -> usize {
        self.signer_count
    }

    pub fn algorithm_count(&self) -> usize {
        self.algorithm_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeFinalizationAuthorizationError {
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
    governance_checkpoint_digest: String,
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
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
    threshold_policy_digest: String,
    threshold_floor_digest: String,
    signer_count: usize,
    algorithm_count: usize,
}

pub fn prepare_upgrade_finalization_authorization_v1(
    context: QualifiedUpgradeFinalizationContextV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    threshold_floor: &UpgradeFinalizationThresholdFloorV1,
) -> Result<PreparedUpgradeFinalizationAuthorizationV1, UpgradeFinalizationAuthorizationError> {
    if threshold_floor.minimum_distinct_signers == 0
        || threshold_floor.minimum_distinct_signers > MAX_THRESHOLD_APPROVALS
    {
        return Err(UpgradeFinalizationAuthorizationError::InvalidThresholdFloor);
    }
    if !valid_threshold_policy(threshold_policy) {
        return Err(UpgradeFinalizationAuthorizationError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        return Err(UpgradeFinalizationAuthorizationError::ThresholdPolicyUsageMismatch);
    }
    if threshold_policy.minimum_distinct_signers < threshold_floor.minimum_distinct_signers
        || (threshold_floor.require_algorithm_diversity
            && !threshold_policy.require_algorithm_diversity)
    {
        return Err(UpgradeFinalizationAuthorizationError::ThresholdPolicyBelowFloor);
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
        schema: PREPARED_UPGRADE_FINALIZATION_AUTHORIZATION_SCHEMA,
        context_id: context.id().to_hex(),
        context_policy_digest: context.context_policy_digest().to_hex(),
        governance_checkpoint_digest: context.governance_checkpoint_digest().to_hex(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        threshold_floor_digest: threshold_floor_digest.to_hex(),
        trust_snapshot_digest: context.current_trust_snapshot_digest().to_hex(),
        compromise_tracker_digest: context.current_compromise_tracker_digest().to_hex(),
        clock_envelope_id: context.current_clock_envelope_id().to_hex(),
    };
    let id = PreparedUpgradeFinalizationAuthorizationIdV1(hash_serializable(
        PREPARED_AUTHORIZATION_DOMAIN,
        &commitment,
    )?);

    Ok(PreparedUpgradeFinalizationAuthorizationV1 {
        id,
        context,
        threshold_policy_digest,
        threshold_floor_digest,
        minimum_distinct_signers: threshold_floor.minimum_distinct_signers,
        require_algorithm_diversity: threshold_floor.require_algorithm_diversity,
    })
}

pub fn authorize_clock_governed_upgrade_finalization_v1(
    prepared: PreparedUpgradeFinalizationAuthorizationV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<AuthorizedClockGovernedUpgradeFinalizationV1, UpgradeFinalizationAuthorizationError> {
    if ceremony.purpose() != CLOCK_GOVERNED_UPGRADE_FINALIZATION_PURPOSE {
        return Err(UpgradeFinalizationAuthorizationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.id().as_digest() {
        return Err(UpgradeFinalizationAuthorizationError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(UpgradeFinalizationAuthorizationError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.context.current_trust_snapshot_digest() {
        return Err(UpgradeFinalizationAuthorizationError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.context.current_compromise_tracker_digest() {
        return Err(UpgradeFinalizationAuthorizationError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.context.current_clock_envelope_id() {
        return Err(UpgradeFinalizationAuthorizationError::ClockEnvelopeMismatch);
    }
    if ceremony.signers().len() < prepared.minimum_distinct_signers {
        return Err(UpgradeFinalizationAuthorizationError::InsufficientCeremonySigners {
            actual: ceremony.signers().len(),
            required: prepared.minimum_distinct_signers,
        });
    }
    let algorithms = ceremony
        .signers()
        .iter()
        .map(|(algorithm, _)| algorithm.clone())
        .collect::<BTreeSet<_>>();
    if prepared.require_algorithm_diversity && algorithms.len() < 2 {
        return Err(UpgradeFinalizationAuthorizationError::MissingCeremonyAlgorithmDiversity);
    }

    let commitment = AuthorizedFinalizationCommitment {
        schema: AUTHORIZED_CLOCK_GOVERNED_UPGRADE_FINALIZATION_SCHEMA,
        prepared_id: prepared.id().to_hex(),
        context_id: prepared.context.id().to_hex(),
        threshold_ceremony_id: ceremony.id().to_hex(),
        threshold_ceremony_digest: ceremony.ceremony_digest().to_hex(),
        threshold_policy_digest: prepared.threshold_policy_digest.to_hex(),
        threshold_floor_digest: prepared.threshold_floor_digest.to_hex(),
        signer_count: ceremony.signers().len(),
        algorithm_count: algorithms.len(),
    };
    let id = AuthorizedClockGovernedUpgradeFinalizationIdV1(hash_serializable(
        AUTHORIZED_FINALIZATION_DOMAIN,
        &commitment,
    )?);

    Ok(AuthorizedClockGovernedUpgradeFinalizationV1 {
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
        && policy
            .required_algorithms
            .iter()
            .all(SignatureAlgorithm::is_canonical)
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
) -> Result<Sha256Digest, UpgradeFinalizationAuthorizationError> {
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
) -> Result<Sha256Digest, UpgradeFinalizationAuthorizationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeFinalizationAuthorizationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
