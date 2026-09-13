// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe live governance authority for fabrication.
//!
//! Portable evidence objects remain in `symthaea-fabrication-kernel`. This crate
//! mints separate opaque, non-deserializable live-authority capabilities after
//! composing exact fabrication semantics with trust-kernel clock authority.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::policy_migration::{
    MAX_MIGRATION_RATIONALE_BYTES, POLICY_MIGRATION_SCHEMA, PolicyInvariantDisposition,
    PolicyMigrationError, PolicyMigrationPlan, PolicyMigrationPolicy,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{KeyUsage, TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError,
};

pub const CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE: &str =
    "clock-governed-policy-migration-v1";
pub const PREPARED_CLOCK_GOVERNED_POLICY_MIGRATION_SCHEMA: &str =
    "symthaea.fabrication.prepared-clock-governed-policy-migration.v1";
pub const CLOCK_GOVERNED_POLICY_MIGRATION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-policy-migration.v1";

const LEGACY_POLICY_MIGRATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-migration-digest.v1\0";
const MIGRATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-migration-policy.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_MIGRATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-clock-governed-policy-migration.v1\0";
const AUTHORIZED_MIGRATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-migration.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedClockGovernedPolicyMigrationIdV1(Sha256Digest);

impl PreparedClockGovernedPolicyMigrationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedPolicyMigrationIdV1(Sha256Digest);

impl ClockGovernedPolicyMigrationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque pre-authorization proposal.
///
/// The proposal proves that the raw migration semantics are valid for every
/// possible true time in the trusted clock envelope and commits the complete
/// authority context that the threshold quorum must sign. It is not itself
/// authorization.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedClockGovernedPolicyMigrationV1 {
    id: PreparedClockGovernedPolicyMigrationIdV1,
    plan: PolicyMigrationPlan,
    plan_digest: Sha256Digest,
    migration_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
}

impl PreparedClockGovernedPolicyMigrationV1 {
    pub fn id(&self) -> PreparedClockGovernedPolicyMigrationIdV1 {
        self.id
    }

    /// Exact payload the threshold approvals must sign.
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }

    pub fn plan(&self) -> &PolicyMigrationPlan {
        &self.plan
    }

    pub fn plan_digest(&self) -> Sha256Digest {
        self.plan_digest
    }

    pub fn migration_policy_digest(&self) -> Sha256Digest {
        self.migration_policy_digest
    }

    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }

    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }

    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }

    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
}

/// Opaque live authority for one exact policy migration under one exact trusted
/// clock interval and one exact governance context.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedPolicyMigrationV1 {
    id: ClockGovernedPolicyMigrationIdV1,
    prepared_id: PreparedClockGovernedPolicyMigrationIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
    plan: PolicyMigrationPlan,
    plan_digest: Sha256Digest,
    migration_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
}

impl ClockGovernedPolicyMigrationV1 {
    pub fn id(&self) -> ClockGovernedPolicyMigrationIdV1 {
        self.id
    }

    pub fn prepared_id(&self) -> PreparedClockGovernedPolicyMigrationIdV1 {
        self.prepared_id
    }

    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.threshold_ceremony_id
    }

    pub fn threshold_ceremony_digest(&self) -> Sha256Digest {
        self.threshold_ceremony_digest
    }

    pub fn plan(&self) -> &PolicyMigrationPlan {
        &self.plan
    }

    pub fn plan_digest(&self) -> Sha256Digest {
        self.plan_digest
    }

    pub fn migration_policy_digest(&self) -> Sha256Digest {
        self.migration_policy_digest
    }

    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }

    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }

    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }

    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedPolicyMigrationError {
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    Migration(PolicyMigrationError),
    Clock(ClockGovernanceTimeError),
    TrustSnapshotInvalid(String),
    ContainmentStateInvalid(String),
    CompromiseTrackerInvalid(String),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    Encoding(String),
}

/// Validate and freeze the full context that a migration quorum must approve.
///
/// No caller-selected current time participates. Activation safety is proven
/// against the complete trusted clock interval. The returned ID, rather than
/// the legacy bare plan digest, is the threshold-signing payload.
pub fn prepare_clock_governed_policy_migration_v1(
    plan: PolicyMigrationPlan,
    migration_policy: &PolicyMigrationPolicy,
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
) -> Result<PreparedClockGovernedPolicyMigrationV1, ClockGovernedPolicyMigrationError> {
    validate_threshold_policy(threshold_policy)?;
    if threshold_policy.key_usage != KeyUsage::PolicyMigration {
        return Err(ClockGovernedPolicyMigrationError::ThresholdPolicyUsageMismatch);
    }

    validate_migration_plan_across_clock(&plan, migration_policy, clock)?;
    let plan_digest = digest_validated_migration_plan(&plan)?;
    let migration_policy_digest = digest_migration_policy(migration_policy)?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy)?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        ClockGovernedPolicyMigrationError::TrustSnapshotInvalid(format!("{error:?}"))
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        ClockGovernedPolicyMigrationError::ContainmentStateInvalid(format!("{error:?}"))
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| ClockGovernedPolicyMigrationError::CompromiseTrackerInvalid(format!("{error:?}")),
        )?;

    let id = PreparedClockGovernedPolicyMigrationIdV1(digest_prepared_migration(
        plan_digest,
        migration_policy_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_state.generation,
        clock.id(),
    )?);

    Ok(PreparedClockGovernedPolicyMigrationV1 {
        id,
        plan,
        plan_digest,
        migration_policy_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_generation: containment_state.generation,
        clock_envelope_id: clock.id(),
    })
}

/// Convert a fully interval-qualified threshold ceremony into live migration
/// authority for the exact prepared proposal it approved.
///
/// Every authority-context commitment is cross-checked. A threshold proof made
/// under another clock envelope, trust snapshot, threshold policy, or compromise
/// tracker cannot be relabeled as authority for this proposal.
pub fn authorize_clock_governed_policy_migration_v1(
    prepared: PreparedClockGovernedPolicyMigrationV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedPolicyMigrationV1, ClockGovernedPolicyMigrationError> {
    if ceremony.purpose() != CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE {
        return Err(ClockGovernedPolicyMigrationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(ClockGovernedPolicyMigrationError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(ClockGovernedPolicyMigrationError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(ClockGovernedPolicyMigrationError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(ClockGovernedPolicyMigrationError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(ClockGovernedPolicyMigrationError::ClockEnvelopeMismatch);
    }

    let id = ClockGovernedPolicyMigrationIdV1(digest_authorized_migration(
        prepared.id,
        ceremony.id(),
        ceremony.ceremony_digest(),
    )?);

    Ok(ClockGovernedPolicyMigrationV1 {
        id,
        prepared_id: prepared.id,
        threshold_ceremony_id: ceremony.id(),
        threshold_ceremony_digest: ceremony.ceremony_digest(),
        plan: prepared.plan,
        plan_digest: prepared.plan_digest,
        migration_policy_digest: prepared.migration_policy_digest,
        threshold_policy_digest: prepared.threshold_policy_digest,
        trust_snapshot_digest: prepared.trust_snapshot_digest,
        containment_state_digest: prepared.containment_state_digest,
        compromise_tracker_digest: prepared.compromise_tracker_digest,
        containment_generation: prepared.containment_generation,
        clock_envelope_id: prepared.clock_envelope_id,
    })
}

fn validate_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<(), ClockGovernedPolicyMigrationError> {
    let valid = policy.minimum_distinct_signers > 0
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
        });
    if valid {
        Ok(())
    } else {
        Err(ClockGovernedPolicyMigrationError::InvalidThresholdPolicy)
    }
}

fn validate_migration_plan_across_clock(
    plan: &PolicyMigrationPlan,
    policy: &PolicyMigrationPolicy,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
) -> Result<(), ClockGovernedPolicyMigrationError> {
    if plan.schema_version != POLICY_MIGRATION_SCHEMA {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::UnsupportedSchema,
        ));
    }
    plan.predecessor
        .validate()
        .map_err(ClockGovernedPolicyMigrationError::Migration)?;
    plan.successor
        .validate()
        .map_err(ClockGovernedPolicyMigrationError::Migration)?;
    if plan.predecessor.domain != plan.successor.domain {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::InvalidDomain,
        ));
    }
    if plan.predecessor.policy_digest == plan.successor.policy_digest
        || plan.predecessor.policy_version == plan.successor.policy_version
    {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::SamePolicy,
        ));
    }
    if plan.rollback_deadline_unix_s <= plan.activates_at_unix_s {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::InvalidActivationWindow,
        ));
    }
    clock
        .require_activation_within_delay_seconds(
            plan.activates_at_unix_s,
            policy.maximum_activation_delay_s,
        )
        .map_err(ClockGovernedPolicyMigrationError::Clock)?;
    if plan.rollback_deadline_unix_s - plan.activates_at_unix_s
        > policy.maximum_rollback_window_s
    {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::RollbackWindowTooLong,
        ));
    }
    if plan.rationale.trim().is_empty()
        || plan.rationale != plan.rationale.trim()
        || plan.rationale.len() > MAX_MIGRATION_RATIONALE_BYTES
        || plan.rationale.chars().any(char::is_control)
    {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::InvalidRationale,
        ));
    }
    if plan.migrations.len() != plan.predecessor.invariants.len() {
        return Err(ClockGovernedPolicyMigrationError::Migration(
            PolicyMigrationError::MigrationCountMismatch,
        ));
    }

    let predecessor = plan.predecessor.invariant_map();
    let successor = plan.successor.invariant_map();
    let mut migrated = BTreeSet::new();
    for migration in &plan.migrations {
        let Some(expected_predecessor) = predecessor.get(migration.name.as_str()) else {
            return Err(ClockGovernedPolicyMigrationError::Migration(
                PolicyMigrationError::UnknownPredecessorInvariant(migration.name.clone()),
            ));
        };
        if !migrated.insert(migration.name.clone()) {
            return Err(ClockGovernedPolicyMigrationError::Migration(
                PolicyMigrationError::DuplicateMigration(migration.name.clone()),
            ));
        }
        if migration.predecessor_digest != *expected_predecessor {
            return Err(ClockGovernedPolicyMigrationError::Migration(
                PolicyMigrationError::PredecessorDigestMismatch(migration.name.clone()),
            ));
        }
        match &migration.disposition {
            PolicyInvariantDisposition::Retained => {
                let Some(successor_digest) = successor.get(migration.name.as_str()) else {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::SuccessorInvariantMissing(migration.name.clone()),
                    ));
                };
                if migration.successor_digest != Some(*successor_digest) {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::SuccessorDigestMismatch(migration.name.clone()),
                    ));
                }
                if *successor_digest != migration.predecessor_digest {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::RetainedInvariantChanged(migration.name.clone()),
                    ));
                }
            }
            PolicyInvariantDisposition::Strengthened => {
                let Some(successor_digest) = successor.get(migration.name.as_str()) else {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::SuccessorInvariantMissing(migration.name.clone()),
                    ));
                };
                if migration.successor_digest != Some(*successor_digest) {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::SuccessorDigestMismatch(migration.name.clone()),
                    ));
                }
                if *successor_digest == migration.predecessor_digest {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::StrengthenedInvariantUnchanged(migration.name.clone()),
                    ));
                }
            }
            PolicyInvariantDisposition::Waived {
                incident_digest,
                expires_at_unix_s,
            } => {
                if !policy.allow_waivers {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::WaiverForbidden(migration.name.clone()),
                    ));
                }
                if migration.successor_digest.is_some()
                    || successor.contains_key(migration.name.as_str())
                {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::WaiverHasSuccessorDigest(migration.name.clone()),
                    ));
                }
                if incident_digest.0 == [0; 32] || *expires_at_unix_s <= plan.activates_at_unix_s {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::WaiverExpired(migration.name.clone()),
                    ));
                }
                if *expires_at_unix_s - plan.activates_at_unix_s
                    > policy.maximum_waiver_lifetime_s
                {
                    return Err(ClockGovernedPolicyMigrationError::Migration(
                        PolicyMigrationError::WaiverTooLong(migration.name.clone()),
                    ));
                }
            }
        }
    }
    for invariant in predecessor.keys() {
        if !migrated.contains(*invariant) {
            return Err(ClockGovernedPolicyMigrationError::Migration(
                PolicyMigrationError::MissingPredecessorInvariant((*invariant).into()),
            ));
        }
    }
    Ok(())
}

fn digest_validated_migration_plan(
    plan: &PolicyMigrationPlan,
) -> Result<Sha256Digest, ClockGovernedPolicyMigrationError> {
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| ClockGovernedPolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(LEGACY_POLICY_MIGRATION_DIGEST_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct MigrationPolicyCommitment {
    maximum_activation_delay_s: u64,
    maximum_rollback_window_s: u64,
    maximum_waiver_lifetime_s: u64,
    allow_waivers: bool,
}

fn digest_migration_policy(
    policy: &PolicyMigrationPolicy,
) -> Result<Sha256Digest, ClockGovernedPolicyMigrationError> {
    let bytes = serde_json::to_vec(&MigrationPolicyCommitment {
        maximum_activation_delay_s: policy.maximum_activation_delay_s,
        maximum_rollback_window_s: policy.maximum_rollback_window_s,
        maximum_waiver_lifetime_s: policy.maximum_waiver_lifetime_s,
        allow_waivers: policy.allow_waivers,
    })
    .map_err(|error| ClockGovernedPolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(MIGRATION_POLICY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
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

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, ClockGovernedPolicyMigrationError> {
    let bytes = serde_json::to_vec(&ThresholdPolicyCommitment {
        minimum_distinct_signers: policy.minimum_distinct_signers,
        maximum_approvals: policy.maximum_approvals,
        require_algorithm_diversity: policy.require_algorithm_diversity,
        required_algorithms: &policy.required_algorithms,
        allowed_key_ids: &policy.allowed_key_ids,
        key_usage: policy.key_usage,
    })
    .map_err(|error| ClockGovernedPolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(THRESHOLD_POLICY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct PreparedMigrationCommitment {
    schema: &'static str,
    purpose: &'static str,
    plan_digest: String,
    migration_policy_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    clock_envelope_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_migration(
    plan_digest: Sha256Digest,
    migration_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
) -> Result<Sha256Digest, ClockGovernedPolicyMigrationError> {
    let commitment = PreparedMigrationCommitment {
        schema: PREPARED_CLOCK_GOVERNED_POLICY_MIGRATION_SCHEMA,
        purpose: CLOCK_GOVERNED_POLICY_MIGRATION_PURPOSE,
        plan_digest: plan_digest.to_hex(),
        migration_policy_digest: migration_policy_digest.to_hex(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        containment_generation,
        clock_envelope_id: clock_envelope_id.to_hex(),
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|error| ClockGovernedPolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(PREPARED_MIGRATION_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct AuthorizedMigrationCommitment {
    schema: &'static str,
    prepared_id: String,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
}

fn digest_authorized_migration(
    prepared_id: PreparedClockGovernedPolicyMigrationIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
) -> Result<Sha256Digest, ClockGovernedPolicyMigrationError> {
    let bytes = serde_json::to_vec(&AuthorizedMigrationCommitment {
        schema: CLOCK_GOVERNED_POLICY_MIGRATION_SCHEMA,
        prepared_id: prepared_id.to_hex(),
        threshold_ceremony_id: threshold_ceremony_id.to_hex(),
        threshold_ceremony_digest: threshold_ceremony_digest.to_hex(),
    })
    .map_err(|error| ClockGovernedPolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(AUTHORIZED_MIGRATION_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
