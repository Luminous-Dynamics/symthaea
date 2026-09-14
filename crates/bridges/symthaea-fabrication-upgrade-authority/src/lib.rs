// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clock-governed upgrade handoff authorization bound to hardened policy currentness.
//!
//! This crate deliberately separates upgrade *authorization* from upgrade *activation*.
//! Authorization may approve a future executable handoff whose required policy migrations are
//! authorized but not yet active. A later runtime layer must prove those migrations actually
//! became the current policy lineages before executable authority transfers.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_governance_bridge::ClockGovernedPolicyMigrationV1;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::policy_migration::digest_policy_binding;
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{KeyUsage, TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_kernel::upgrade_handoff::{
    MAX_UPGRADE_MIGRATIONS, MAX_UPGRADE_REASON_BYTES, UpgradeEndpoint, UpgradeHandoffPolicy,
};
use symthaea_fabrication_policy_exact_evidence::ExactEvidenceBoundPolicyHeadV1;
use symthaea_fabrication_policy_head_observation::QuorumObservedPolicyHeadV1;
use symthaea_fabrication_policy_lineage::ClockGovernedPolicyLineageV1;
use symthaea_fabrication_policy_temporal_validity::ClockGovernedPolicyTemporalValidityPermitV1;
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_witness_authority::RegistryBoundPolicyHeadV1;
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_UPGRADE_HANDOFF_PURPOSE: &str =
    "clock-governed-upgrade-handoff-v1";
pub const CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-handoff-plan.v1";
pub const PREPARED_CLOCK_GOVERNED_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.prepared-clock-governed-upgrade-handoff.v1";
pub const CLOCK_GOVERNED_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-handoff.v1";

const HANDOFF_PLAN_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-handoff-plan.v1\0";
const HANDOFF_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-handoff-policy.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const POLICY_AUTHORITY_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-policy-authority-set.v1\0";
const PREPARED_HANDOFF_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-clock-governed-upgrade-handoff.v1\0";
const AUTHORIZED_HANDOFF_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-handoff.v1\0";

/// Portable statement of one policy migration that the future upgrade activation must observe as
/// current before executable authority transfers. This is evidence data, not live authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradePolicyRequirementV1 {
    pub policy_domain: String,
    pub migration_authority_id: String,
    pub migration_plan_digest: Sha256Digest,
    pub current_lineage_id: String,
    pub current_lineage_sequence: u64,
    pub current_policy_binding_digest: Sha256Digest,
    pub successor_policy_binding_digest: Sha256Digest,
    pub temporal_validity_permit_id: String,
    pub observed_head_id: String,
    pub registry_bound_head_id: String,
    pub exact_evidence_head_id: String,
    pub migration_activates_at_unix_s: u64,
    pub migration_rollback_deadline_unix_s: u64,
}

/// New portable upgrade plan. Unlike the legacy plan, it carries no scalar clock-evidence digest
/// and no `prepared_at` timestamp. The authoritative preparation interval is committed only by the
/// opaque prepared capability below.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockGovernedUpgradeHandoffPlanV1 {
    pub schema_version: String,
    pub predecessor: UpgradeEndpoint,
    pub successor: UpgradeEndpoint,
    pub activates_at_unix_ms: u64,
    pub finalization_deadline_unix_ms: u64,
    pub rollback_target_digest: Sha256Digest,
    pub policy_requirements: Vec<UpgradePolicyRequirementV1>,
    pub evidence_checkpoint_digest: Sha256Digest,
    pub recovery_key_set_digest: Sha256Digest,
    pub reason: String,
}

/// Borrowed live-authority bundle used only while preparing a handoff. It is deliberately not
/// serializable and is not itself authority.
pub struct UpgradePolicyAuthorityInputV1<'a> {
    pub migration: &'a ClockGovernedPolicyMigrationV1,
    pub lineage: &'a ClockGovernedPolicyLineageV1,
    pub temporal: &'a ClockGovernedPolicyTemporalValidityPermitV1,
    pub observed: &'a QuorumObservedPolicyHeadV1,
    pub registry_bound: &'a RegistryBoundPolicyHeadV1,
    pub exact_evidence: &'a ExactEvidenceBoundPolicyHeadV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedClockGovernedUpgradeHandoffIdV1(Sha256Digest);

impl PreparedClockGovernedUpgradeHandoffIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedUpgradeHandoffIdV1(Sha256Digest);

impl ClockGovernedUpgradeHandoffIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque full-context proposal whose ID is the payload the upgrade quorum must approve.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedClockGovernedUpgradeHandoffV1 {
    id: PreparedClockGovernedUpgradeHandoffIdV1,
    plan: ClockGovernedUpgradeHandoffPlanV1,
    plan_digest: Sha256Digest,
    handoff_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    policy_authority_set_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl PreparedClockGovernedUpgradeHandoffV1 {
    pub fn id(&self) -> PreparedClockGovernedUpgradeHandoffIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn plan(&self) -> &ClockGovernedUpgradeHandoffPlanV1 {
        &self.plan
    }
    pub fn plan_digest(&self) -> Sha256Digest {
        self.plan_digest
    }
    pub fn handoff_policy_digest(&self) -> Sha256Digest {
        self.handoff_policy_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn policy_authority_set_digest(&self) -> Sha256Digest {
        self.policy_authority_set_digest
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
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

/// Opaque authorization for one exact future upgrade handoff. This does not prove the handoff may
/// execute now; runtime activation must separately prove trusted time has reached activation and
/// every required policy migration became the current hardened policy lineage.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedUpgradeHandoffV1 {
    id: ClockGovernedUpgradeHandoffIdV1,
    prepared_id: PreparedClockGovernedUpgradeHandoffIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
    plan: ClockGovernedUpgradeHandoffPlanV1,
    plan_digest: Sha256Digest,
    handoff_policy_digest: Sha256Digest,
    policy_authority_set_digest: Sha256Digest,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
}

impl ClockGovernedUpgradeHandoffV1 {
    pub fn id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.id
    }
    pub fn prepared_id(&self) -> PreparedClockGovernedUpgradeHandoffIdV1 {
        self.prepared_id
    }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.threshold_ceremony_id
    }
    pub fn threshold_ceremony_digest(&self) -> Sha256Digest {
        self.threshold_ceremony_digest
    }
    pub fn plan(&self) -> &ClockGovernedUpgradeHandoffPlanV1 {
        &self.plan
    }
    pub fn plan_digest(&self) -> Sha256Digest {
        self.plan_digest
    }
    pub fn handoff_policy_digest(&self) -> Sha256Digest {
        self.handoff_policy_digest
    }
    pub fn policy_authority_set_digest(&self) -> Sha256Digest {
        self.policy_authority_set_digest
    }
    pub fn authorization_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.authorization_clock_envelope_id
    }
    pub fn authorization_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.authorization_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedUpgradeHandoffError {
    UnsupportedSchema,
    EndpointInvalid(String),
    SameVersion,
    SameSourceTree,
    SameExecutable,
    SameDurableState,
    AuthorityEpochInvalid,
    InvalidWindow,
    ActivationBeforeTrustedInterval,
    ActivationTooLate,
    FinalizationWindowTooLong,
    ZeroDigest(&'static str),
    MissingPolicyMigration,
    TooManyPolicyMigrations { actual: usize, maximum: usize },
    DuplicatePolicyDomain(String),
    DuplicateMigrationAuthority(String),
    PolicyRequirementsMismatch,
    PolicyAuthorityChainMismatch(String),
    PolicyPredecessorMismatch(String),
    PolicyMigrationTargetsWrongWindow(String),
    InvalidReason,
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    Clock(ClockGovernanceTimeError),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    Encoding(String),
}

/// Build a canonical portable handoff plan directly from the live policy-authority bundles that
/// the handoff will rely upon. This prevents a caller from hand-writing a friendlier migration list.
#[allow(clippy::too_many_arguments)]
pub fn build_clock_governed_upgrade_handoff_plan_v1(
    predecessor: UpgradeEndpoint,
    successor: UpgradeEndpoint,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
    rollback_target_digest: Sha256Digest,
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
    evidence_checkpoint_digest: Sha256Digest,
    recovery_key_set_digest: Sha256Digest,
    reason: impl Into<String>,
) -> Result<ClockGovernedUpgradeHandoffPlanV1, ClockGovernedUpgradeHandoffError> {
    let mut policy_requirements = derive_policy_requirements(policy_authorities)?;
    policy_requirements.sort_by(|left, right| left.policy_domain.cmp(&right.policy_domain));
    Ok(ClockGovernedUpgradeHandoffPlanV1 {
        schema_version: CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA.into(),
        predecessor,
        successor,
        activates_at_unix_ms,
        finalization_deadline_unix_ms,
        rollback_target_digest,
        policy_requirements,
        evidence_checkpoint_digest,
        recovery_key_set_digest,
        reason: reason.into(),
    })
}

/// Validate the complete handoff context under a trusted operational-clock interval and freeze the
/// exact payload an `UpgradeHandoff` threshold quorum must sign.
#[allow(clippy::too_many_arguments)]
pub fn prepare_clock_governed_upgrade_handoff_v1(
    plan: ClockGovernedUpgradeHandoffPlanV1,
    handoff_policy: &UpgradeHandoffPolicy,
    threshold_policy: &ThresholdCeremonyPolicy,
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedClockGovernedUpgradeHandoffV1, Vec<ClockGovernedUpgradeHandoffError>> {
    let mut violations = Vec::new();

    if !valid_threshold_policy(threshold_policy) {
        violations.push(ClockGovernedUpgradeHandoffError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::UpgradeHandoff {
        violations.push(ClockGovernedUpgradeHandoffError::ThresholdPolicyUsageMismatch);
    }

    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedUpgradeHandoffError::Clock(error));
            return Err(violations);
        }
    };

    if let Err(error) = validate_plan_across_clock(
        &plan,
        handoff_policy,
        policy_authorities,
        clock.lower_unix_ms(),
        clock.upper_unix_ms(),
    ) {
        violations.push(error);
    }

    if let Err(error) = trust_snapshot.validate() {
        violations.push(ClockGovernedUpgradeHandoffError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            ClockGovernedUpgradeHandoffError::TrustSnapshotNotValidAcrossEnvelope(reason),
        );
    }
    if let Err(error) = containment_state.validate() {
        violations.push(ClockGovernedUpgradeHandoffError::ContainmentStateInvalid(format!(
            "{error:?}"
        )));
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let plan_digest = digest_upgrade_handoff_plan_v1(&plan).map_err(|error| vec![error])?;
    let handoff_policy_digest =
        digest_handoff_policy(handoff_policy).map_err(|error| vec![error])?;
    let threshold_policy_digest =
        digest_threshold_policy(threshold_policy).map_err(|error| vec![error])?;
    let policy_authority_set_digest =
        digest_policy_authority_set(&plan.policy_requirements).map_err(|error| vec![error])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![ClockGovernedUpgradeHandoffError::TrustSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ClockGovernedUpgradeHandoffError::ContainmentStateInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ClockGovernedUpgradeHandoffError::ContainmentStateInvalid(format!(
                    "{error:?}"
                ))]
            },
        )?;

    let id = PreparedClockGovernedUpgradeHandoffIdV1(
        digest_prepared_handoff(
            plan_digest,
            handoff_policy_digest,
            threshold_policy_digest,
            policy_authority_set_digest,
            trust_snapshot_digest,
            containment_state_digest,
            compromise_tracker_digest,
            containment_state.generation,
            clock.id(),
            operational_basis.id(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(PreparedClockGovernedUpgradeHandoffV1 {
        id,
        plan,
        plan_digest,
        handoff_policy_digest,
        threshold_policy_digest,
        policy_authority_set_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_generation: containment_state.generation,
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
    })
}

/// Convert the exact interval-qualified `UpgradeHandoff` threshold quorum into opaque upgrade
/// authorization. This still is not runtime activation permission.
pub fn authorize_clock_governed_upgrade_handoff_v1(
    prepared: PreparedClockGovernedUpgradeHandoffV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedUpgradeHandoffV1, ClockGovernedUpgradeHandoffError> {
    if ceremony.purpose() != CLOCK_GOVERNED_UPGRADE_HANDOFF_PURPOSE {
        return Err(ClockGovernedUpgradeHandoffError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(ClockGovernedUpgradeHandoffError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(ClockGovernedUpgradeHandoffError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(ClockGovernedUpgradeHandoffError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(ClockGovernedUpgradeHandoffError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(ClockGovernedUpgradeHandoffError::ClockEnvelopeMismatch);
    }

    let id = ClockGovernedUpgradeHandoffIdV1(digest_authorized_handoff(
        prepared.id,
        ceremony.id(),
        ceremony.ceremony_digest(),
    )?);

    Ok(ClockGovernedUpgradeHandoffV1 {
        id,
        prepared_id: prepared.id,
        threshold_ceremony_id: ceremony.id(),
        threshold_ceremony_digest: ceremony.ceremony_digest(),
        plan: prepared.plan,
        plan_digest: prepared.plan_digest,
        handoff_policy_digest: prepared.handoff_policy_digest,
        policy_authority_set_digest: prepared.policy_authority_set_digest,
        authorization_clock_envelope_id: prepared.clock_envelope_id,
        authorization_operational_basis_id: prepared.operational_basis_id,
    })
}

pub fn digest_upgrade_handoff_plan_v1(
    plan: &ClockGovernedUpgradeHandoffPlanV1,
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    if plan.schema_version != CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA {
        return Err(ClockGovernedUpgradeHandoffError::UnsupportedSchema);
    }
    hash_serializable(HANDOFF_PLAN_DOMAIN, plan)
}

fn validate_plan_across_clock(
    plan: &ClockGovernedUpgradeHandoffPlanV1,
    policy: &UpgradeHandoffPolicy,
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
    clock_lower_unix_ms: u64,
    clock_upper_unix_ms: u64,
) -> Result<(), ClockGovernedUpgradeHandoffError> {
    if plan.schema_version != CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA {
        return Err(ClockGovernedUpgradeHandoffError::UnsupportedSchema);
    }
    plan.predecessor
        .validate()
        .map_err(|error| ClockGovernedUpgradeHandoffError::EndpointInvalid(format!("{error:?}")))?;
    plan.successor
        .validate()
        .map_err(|error| ClockGovernedUpgradeHandoffError::EndpointInvalid(format!("{error:?}")))?;
    if plan.predecessor.software_version == plan.successor.software_version {
        return Err(ClockGovernedUpgradeHandoffError::SameVersion);
    }
    if plan.predecessor.source_tree_digest == plan.successor.source_tree_digest {
        return Err(ClockGovernedUpgradeHandoffError::SameSourceTree);
    }
    if plan.predecessor.executable_digest == plan.successor.executable_digest {
        return Err(ClockGovernedUpgradeHandoffError::SameExecutable);
    }
    if plan.predecessor.durable_state_digest == plan.successor.durable_state_digest {
        return Err(ClockGovernedUpgradeHandoffError::SameDurableState);
    }
    let advanced = plan
        .successor
        .authority_epoch
        .dominates(&plan.predecessor.authority_epoch)
        .map_err(|_| ClockGovernedUpgradeHandoffError::AuthorityEpochInvalid)?;
    if !advanced {
        return Err(ClockGovernedUpgradeHandoffError::AuthorityEpochInvalid);
    }
    if plan.activates_at_unix_ms >= plan.finalization_deadline_unix_ms {
        return Err(ClockGovernedUpgradeHandoffError::InvalidWindow);
    }
    if plan.activates_at_unix_ms < clock_upper_unix_ms {
        return Err(ClockGovernedUpgradeHandoffError::ActivationBeforeTrustedInterval);
    }
    let latest_activation = clock_lower_unix_ms
        .checked_add(policy.maximum_activation_delay_ms)
        .ok_or(ClockGovernedUpgradeHandoffError::ActivationTooLate)?;
    if plan.activates_at_unix_ms > latest_activation {
        return Err(ClockGovernedUpgradeHandoffError::ActivationTooLate);
    }
    if plan.finalization_deadline_unix_ms - plan.activates_at_unix_ms
        > policy.maximum_finalization_window_ms
    {
        return Err(ClockGovernedUpgradeHandoffError::FinalizationWindowTooLong);
    }
    for (name, digest) in [
        ("rollback_target_digest", plan.rollback_target_digest),
        ("evidence_checkpoint_digest", plan.evidence_checkpoint_digest),
        ("recovery_key_set_digest", plan.recovery_key_set_digest),
    ] {
        if digest == Sha256Digest([0; 32]) {
            return Err(ClockGovernedUpgradeHandoffError::ZeroDigest(name));
        }
    }
    if policy.require_policy_migration && plan.policy_requirements.is_empty() {
        return Err(ClockGovernedUpgradeHandoffError::MissingPolicyMigration);
    }
    if plan.policy_requirements.len() > MAX_UPGRADE_MIGRATIONS {
        return Err(ClockGovernedUpgradeHandoffError::TooManyPolicyMigrations {
            actual: plan.policy_requirements.len(),
            maximum: MAX_UPGRADE_MIGRATIONS,
        });
    }
    if policy_authorities.len() > MAX_UPGRADE_MIGRATIONS {
        return Err(ClockGovernedUpgradeHandoffError::TooManyPolicyMigrations {
            actual: policy_authorities.len(),
            maximum: MAX_UPGRADE_MIGRATIONS,
        });
    }

    let mut expected = derive_policy_requirements(policy_authorities)?;
    expected.sort_by(|left, right| left.policy_domain.cmp(&right.policy_domain));
    if plan.policy_requirements != expected {
        return Err(ClockGovernedUpgradeHandoffError::PolicyRequirementsMismatch);
    }

    let mut domains = BTreeSet::new();
    let mut migrations = BTreeSet::new();
    for requirement in &plan.policy_requirements {
        if !domains.insert(requirement.policy_domain.clone()) {
            return Err(ClockGovernedUpgradeHandoffError::DuplicatePolicyDomain(
                requirement.policy_domain.clone(),
            ));
        }
        if !migrations.insert(requirement.migration_authority_id.clone()) {
            return Err(
                ClockGovernedUpgradeHandoffError::DuplicateMigrationAuthority(
                    requirement.migration_authority_id.clone(),
                ),
            );
        }
        let migration_activation_ms = requirement
            .migration_activates_at_unix_s
            .checked_mul(1_000)
            .ok_or_else(|| {
                ClockGovernedUpgradeHandoffError::PolicyMigrationTargetsWrongWindow(
                    requirement.policy_domain.clone(),
                )
            })?;
        let migration_rollback_ms = requirement
            .migration_rollback_deadline_unix_s
            .checked_mul(1_000)
            .ok_or_else(|| {
                ClockGovernedUpgradeHandoffError::PolicyMigrationTargetsWrongWindow(
                    requirement.policy_domain.clone(),
                )
            })?;
        if migration_activation_ms > plan.activates_at_unix_ms
            || migration_rollback_ms < plan.finalization_deadline_unix_ms
        {
            return Err(
                ClockGovernedUpgradeHandoffError::PolicyMigrationTargetsWrongWindow(
                    requirement.policy_domain.clone(),
                ),
            );
        }
    }

    if plan.reason.trim().is_empty()
        || plan.reason != plan.reason.trim()
        || plan.reason.len() > MAX_UPGRADE_REASON_BYTES
        || plan.reason.chars().any(char::is_control)
    {
        return Err(ClockGovernedUpgradeHandoffError::InvalidReason);
    }
    Ok(())
}

fn derive_policy_requirements(
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
) -> Result<Vec<UpgradePolicyRequirementV1>, ClockGovernedUpgradeHandoffError> {
    let mut requirements = Vec::with_capacity(policy_authorities.len());
    let mut domains = BTreeSet::new();
    let mut migration_ids = BTreeSet::new();

    for authority in policy_authorities {
        let domain = authority.lineage.domain().to_string();
        if !domains.insert(domain.clone()) {
            return Err(ClockGovernedUpgradeHandoffError::DuplicatePolicyDomain(domain));
        }
        if !migration_ids.insert(authority.migration.id().to_hex()) {
            return Err(
                ClockGovernedUpgradeHandoffError::DuplicateMigrationAuthority(
                    authority.migration.id().to_hex(),
                ),
            );
        }

        if authority.temporal.lineage_id() != authority.lineage.id()
            || authority.temporal.lineage_sequence() != authority.lineage.sequence()
            || authority.temporal.current_policy_binding_digest()
                != authority.lineage.current_policy_binding_digest()
            || authority.observed.lineage_id() != authority.lineage.id()
            || authority.observed.lineage_sequence() != authority.lineage.sequence()
            || authority.observed.temporal_validity_permit_id() != authority.temporal.id()
            || authority.observed.clock_envelope_id()
                != authority.temporal.current_clock_envelope_id()
            || authority.registry_bound.observed_head_id() != authority.observed.id()
            || authority.exact_evidence.observed_head_id() != authority.observed.id()
            || authority.exact_evidence.registry_bound_head_id() != authority.registry_bound.id()
        {
            return Err(ClockGovernedUpgradeHandoffError::PolicyAuthorityChainMismatch(
                domain,
            ));
        }

        let migration_plan = authority.migration.plan();
        if migration_plan.predecessor != *authority.lineage.current_policy()
            || migration_plan.predecessor.domain != authority.lineage.domain()
            || migration_plan.successor.domain != authority.lineage.domain()
        {
            return Err(ClockGovernedUpgradeHandoffError::PolicyPredecessorMismatch(
                authority.lineage.domain().to_string(),
            ));
        }

        let successor_policy_binding_digest = digest_policy_binding(&migration_plan.successor)
            .map_err(|error| {
                ClockGovernedUpgradeHandoffError::PolicyAuthorityChainMismatch(format!(
                    "{}: {error:?}",
                    authority.lineage.domain()
                ))
            })?;

        requirements.push(UpgradePolicyRequirementV1 {
            policy_domain: authority.lineage.domain().to_string(),
            migration_authority_id: authority.migration.id().to_hex(),
            migration_plan_digest: authority.migration.plan_digest(),
            current_lineage_id: authority.lineage.id().to_hex(),
            current_lineage_sequence: authority.lineage.sequence(),
            current_policy_binding_digest: authority.lineage.current_policy_binding_digest(),
            successor_policy_binding_digest,
            temporal_validity_permit_id: authority.temporal.id().to_hex(),
            observed_head_id: authority.observed.id().to_hex(),
            registry_bound_head_id: authority.registry_bound.id().to_hex(),
            exact_evidence_head_id: authority.exact_evidence.id().to_hex(),
            migration_activates_at_unix_s: migration_plan.activates_at_unix_s,
            migration_rollback_deadline_unix_s: migration_plan.rollback_deadline_unix_s,
        });
    }
    Ok(requirements)
}

#[derive(Serialize)]
struct HandoffPolicyCommitment {
    maximum_activation_delay_ms: u64,
    maximum_finalization_window_ms: u64,
    require_policy_migration: bool,
}

fn digest_handoff_policy(
    policy: &UpgradeHandoffPolicy,
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    hash_serializable(
        HANDOFF_POLICY_DOMAIN,
        &HandoffPolicyCommitment {
            maximum_activation_delay_ms: policy.maximum_activation_delay_ms,
            maximum_finalization_window_ms: policy.maximum_finalization_window_ms,
            require_policy_migration: policy.require_policy_migration,
        },
    )
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
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    if !valid_threshold_policy(policy) {
        return Err(ClockGovernedUpgradeHandoffError::InvalidThresholdPolicy);
    }
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

fn digest_policy_authority_set(
    requirements: &[UpgradePolicyRequirementV1],
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    let mut canonical = requirements.to_vec();
    canonical.sort_by(|left, right| left.policy_domain.cmp(&right.policy_domain));
    hash_serializable(POLICY_AUTHORITY_SET_DOMAIN, &canonical)
}

#[derive(Serialize)]
struct PreparedHandoffCommitment {
    schema: &'static str,
    plan_digest: String,
    handoff_policy_digest: String,
    threshold_policy_digest: String,
    policy_authority_set_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    clock_envelope_id: String,
    operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_handoff(
    plan_digest: Sha256Digest,
    handoff_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    policy_authority_set_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    hash_serializable(
        PREPARED_HANDOFF_DOMAIN,
        &PreparedHandoffCommitment {
            schema: PREPARED_CLOCK_GOVERNED_UPGRADE_HANDOFF_SCHEMA,
            plan_digest: plan_digest.to_hex(),
            handoff_policy_digest: handoff_policy_digest.to_hex(),
            threshold_policy_digest: threshold_policy_digest.to_hex(),
            policy_authority_set_digest: policy_authority_set_digest.to_hex(),
            trust_snapshot_digest: trust_snapshot_digest.to_hex(),
            containment_state_digest: containment_state_digest.to_hex(),
            compromise_tracker_digest: compromise_tracker_digest.to_hex(),
            containment_generation,
            clock_envelope_id: clock_envelope_id.to_hex(),
            operational_basis_id: operational_basis_id.to_hex(),
        },
    )
}

#[derive(Serialize)]
struct AuthorizedHandoffCommitment {
    schema: &'static str,
    prepared_id: String,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
}

fn digest_authorized_handoff(
    prepared_id: PreparedClockGovernedUpgradeHandoffIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    hash_serializable(
        AUTHORIZED_HANDOFF_DOMAIN,
        &AuthorizedHandoffCommitment {
            schema: CLOCK_GOVERNED_UPGRADE_HANDOFF_SCHEMA,
            prepared_id: prepared_id.to_hex(),
            threshold_ceremony_id: threshold_ceremony_id.to_hex(),
            threshold_ceremony_digest: threshold_ceremony_digest.to_hex(),
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedUpgradeHandoffError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedUpgradeHandoffError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::authority_epoch::AuthorityEpochVector;
    use symthaea_fabrication_kernel::crypto_digest::sha256;

    fn endpoint(version: &str, gateway_generation: u64) -> UpgradeEndpoint {
        UpgradeEndpoint {
            schema_version: "symthaea.fabrication.upgrade-endpoint.v1".into(),
            software_version: version.into(),
            source_tree_digest: sha256(format!("source-{version}").as_bytes()),
            executable_digest: sha256(format!("exe-{version}").as_bytes()),
            durable_state_digest: sha256(format!("state-{version}").as_bytes()),
            replay_contract_digest: sha256(format!("replay-{version}").as_bytes()),
            authority_epoch: AuthorityEpochVector::new(
                2,
                2,
                gateway_generation,
                1,
                1,
                5,
                3,
                2,
            )
            .unwrap(),
        }
    }

    #[test]
    fn portable_plan_digest_binds_policy_requirement_identity() {
        let requirement = UpgradePolicyRequirementV1 {
            policy_domain: "fabrication.release".into(),
            migration_authority_id: "11".repeat(32),
            migration_plan_digest: sha256(b"migration"),
            current_lineage_id: "22".repeat(32),
            current_lineage_sequence: 4,
            current_policy_binding_digest: sha256(b"current"),
            successor_policy_binding_digest: sha256(b"successor"),
            temporal_validity_permit_id: "33".repeat(32),
            observed_head_id: "44".repeat(32),
            registry_bound_head_id: "55".repeat(32),
            exact_evidence_head_id: "66".repeat(32),
            migration_activates_at_unix_s: 2_000,
            migration_rollback_deadline_unix_s: 3_000,
        };
        let mut first = ClockGovernedUpgradeHandoffPlanV1 {
            schema_version: CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA.into(),
            predecessor: endpoint("0.17.0", 4),
            successor: endpoint("0.18.0", 5),
            activates_at_unix_ms: 2_000_000,
            finalization_deadline_unix_ms: 2_500_000,
            rollback_target_digest: sha256(b"rollback"),
            policy_requirements: vec![requirement.clone()],
            evidence_checkpoint_digest: sha256(b"checkpoint"),
            recovery_key_set_digest: sha256(b"recovery"),
            reason: "upgrade".into(),
        };
        let first_digest = digest_upgrade_handoff_plan_v1(&first).unwrap();
        first.policy_requirements[0].exact_evidence_head_id = "77".repeat(32);
        let second_digest = digest_upgrade_handoff_plan_v1(&first).unwrap();
        assert_ne!(first_digest, second_digest);
    }

    #[test]
    fn handoff_policy_digest_binds_activation_and_finalization_limits() {
        let first = UpgradeHandoffPolicy {
            maximum_activation_delay_ms: 10_000,
            maximum_finalization_window_ms: 20_000,
            require_policy_migration: true,
        };
        let second = UpgradeHandoffPolicy {
            maximum_activation_delay_ms: 10_001,
            ..first.clone()
        };
        assert_ne!(
            digest_handoff_policy(&first).unwrap(),
            digest_handoff_policy(&second).unwrap()
        );
    }
}
