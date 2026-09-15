// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed, interval-safe probation authority that preserves global predecessor lineage.
//!
//! This independently re-runs the probation theorem from the lineage-bound handoff and activation
//! capabilities. Portable signed observation/profile formats are reused, but ordinary live handoff,
//! activation and clearance capabilities are never accepted or emitted.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_kernel::upgrade_probation::{
    MAX_PROBATION_OBSERVATIONS, UpgradeProbationPolicy, digest_upgrade_probation_observation,
};
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundClockGovernedUpgradeHandoffIdV1, LineageBoundClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_lineage_bound_upgrade_runtime::{
    LineageBoundUpgradeActivationPermitIdV1, LineageBoundUpgradeActivationPermitV1,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_upgrade_probation_authority::{
    ProbationObservationVerificationPolicyV1, SignedUpgradeProbationObservationV1,
    UpgradeProbationObservationVerifierV1, UpgradeProbationObserverProfileV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_PURPOSE: &str =
    "lineage-bound-upgrade-probation-clearance-v1";
pub const PREPARED_LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.prepared-lineage-bound-upgrade-probation-clearance.v1";
pub const LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-upgrade-probation-clearance.v1";
pub const MAX_LINEAGE_BOUND_PROBATION_CLOCK_HOPS: usize = 4096;
pub const MAX_LINEAGE_BOUND_PROBATION_VERIFIERS: usize = 8;

const OBSERVATION_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-observation-signature.v1\0";
const SIGNED_OBSERVATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-evidence.v1\0";
const SIGNED_OBSERVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-observation-set.v1\0";
const OBSERVER_PROFILE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-observer-profile-set.v1\0";
const VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-verifier-set.v1\0";
const PROBATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-policy.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-clock-lineage.v1\0";
const PREPARED_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-lineage-bound-upgrade-probation-clearance.v1\0";
const AUTHORIZED_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-probation-clearance.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedLineageBoundUpgradeProbationClearanceIdV1(Sha256Digest);

impl PreparedLineageBoundUpgradeProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundUpgradeProbationClearanceIdV1(Sha256Digest);

impl LineageBoundUpgradeProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedLineageBoundUpgradeProbationClearanceV1 {
    id: PreparedLineageBoundUpgradeProbationClearanceIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: LineageBoundUpgradeActivationPermitIdV1,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    governance_view_digest: Sha256Digest,
    registry_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    observation_set_digest: Sha256Digest,
    observer_profile_set_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    probation_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
    observation_count: usize,
    machine_count: usize,
    failure_domain_count: usize,
    observation_started_at_unix_ms: u64,
    observation_ended_at_unix_ms: u64,
    clearance_expires_at_unix_ms: u64,
}

impl PreparedLineageBoundUpgradeProbationClearanceV1 {
    pub fn id(&self) -> PreparedLineageBoundUpgradeProbationClearanceIdV1 { self.id }
    pub fn signing_payload_digest(&self) -> Sha256Digest { self.id.as_digest() }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
    pub fn activation_permit_id(&self) -> LineageBoundUpgradeActivationPermitIdV1 { self.activation_permit_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_digest }
    pub fn governance_view_digest(&self) -> Sha256Digest { self.governance_view_digest }
    pub fn registry_head_digest(&self) -> Sha256Digest { self.registry_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn observation_set_digest(&self) -> Sha256Digest { self.observation_set_digest }
    pub fn observer_profile_set_digest(&self) -> Sha256Digest { self.observer_profile_set_digest }
    pub fn verifier_set_digest(&self) -> Sha256Digest { self.verifier_set_digest }
    pub fn probation_policy_digest(&self) -> Sha256Digest { self.probation_policy_digest }
    pub fn threshold_policy_digest(&self) -> Sha256Digest { self.threshold_policy_digest }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn containment_state_digest(&self) -> Sha256Digest { self.containment_state_digest }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest { self.compromise_tracker_digest }
    pub fn activation_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.activation_operational_basis_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn clock_lineage_digest(&self) -> Sha256Digest { self.clock_lineage_digest }
    pub fn clock_hop_count(&self) -> usize { self.clock_hop_count }
    pub fn observation_count(&self) -> usize { self.observation_count }
    pub fn machine_count(&self) -> usize { self.machine_count }
    pub fn failure_domain_count(&self) -> usize { self.failure_domain_count }
    pub fn observation_started_at_unix_ms(&self) -> u64 { self.observation_started_at_unix_ms }
    pub fn observation_ended_at_unix_ms(&self) -> u64 { self.observation_ended_at_unix_ms }
    pub fn clearance_expires_at_unix_ms(&self) -> u64 { self.clearance_expires_at_unix_ms }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundUpgradeProbationClearanceV1 {
    id: LineageBoundUpgradeProbationClearanceIdV1,
    prepared_id: PreparedLineageBoundUpgradeProbationClearanceIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: LineageBoundUpgradeActivationPermitIdV1,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    governance_view_digest: Sha256Digest,
    registry_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    observation_set_digest: Sha256Digest,
    observer_profile_set_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    probation_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    operational_basis_id: OperationalClockBasisIdV1,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_count: usize,
    machine_count: usize,
    failure_domain_count: usize,
    observation_started_at_unix_ms: u64,
    observation_ended_at_unix_ms: u64,
    clearance_expires_at_unix_ms: u64,
}

impl LineageBoundUpgradeProbationClearanceV1 {
    pub fn id(&self) -> LineageBoundUpgradeProbationClearanceIdV1 { self.id }
    pub fn prepared_id(&self) -> PreparedLineageBoundUpgradeProbationClearanceIdV1 { self.prepared_id }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
    pub fn activation_permit_id(&self) -> LineageBoundUpgradeActivationPermitIdV1 { self.activation_permit_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_digest }
    pub fn governance_view_digest(&self) -> Sha256Digest { self.governance_view_digest }
    pub fn registry_head_digest(&self) -> Sha256Digest { self.registry_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 { self.threshold_ceremony_id }
    pub fn observation_set_digest(&self) -> Sha256Digest { self.observation_set_digest }
    pub fn observer_profile_set_digest(&self) -> Sha256Digest { self.observer_profile_set_digest }
    pub fn verifier_set_digest(&self) -> Sha256Digest { self.verifier_set_digest }
    pub fn probation_policy_digest(&self) -> Sha256Digest { self.probation_policy_digest }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn containment_state_digest(&self) -> Sha256Digest { self.containment_state_digest }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest { self.compromise_tracker_digest }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn observation_count(&self) -> usize { self.observation_count }
    pub fn machine_count(&self) -> usize { self.machine_count }
    pub fn failure_domain_count(&self) -> usize { self.failure_domain_count }
    pub fn observation_started_at_unix_ms(&self) -> u64 { self.observation_started_at_unix_ms }
    pub fn observation_ended_at_unix_ms(&self) -> u64 { self.observation_ended_at_unix_ms }
    pub fn clearance_expires_at_unix_ms(&self) -> u64 { self.clearance_expires_at_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundProbationError {
    ActivationLineageMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage { hop: usize, expected_predecessor: String, actual_predecessor: Option<String> },
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    InvalidProbationPolicy,
    InvalidVerificationPolicy,
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    InsufficientVerificationProviders { actual: usize, required: usize },
    TooManyVerificationProviders { actual: usize, maximum: usize },
    InvalidProvider(String),
    DuplicateProvider(String),
    EmptyObservations,
    TooManyObservations { actual: usize, maximum: usize },
    InvalidSignedObservation(String),
    ObservationDigestMismatch(String),
    ObservationHandoffMismatch(String),
    ObservationSuccessorMismatch(String),
    ObservationBeforeActivation(String),
    ObservationMayStillBeRunning(String),
    DuplicateObservation,
    MissingObserverProfile(String),
    DuplicateObserverProfile(String),
    UnexpectedObserverProfile(String),
    ObserverMachineMismatch(String),
    ObserverFailureDomainMismatch(String),
    ObserverUnknown(String),
    ObserverNotActive(String),
    ObserverUsageNotAllowed(String),
    ObserverKeyInvalidForObservation(String),
    ObserverNotValidAcrossEnvelope { key_id: String, reason: ClockGovernanceTimeError },
    ObserverCompromisedAcrossEnvelope(String),
    TrustSnapshotPostdatesObservation(String),
    ObservationSignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    ObservationTooShort,
    InsufficientMachines { actual: usize, required: usize },
    InsufficientFailureDomains { actual: usize, required: usize },
    InsufficientSuccessfulJobs,
    FailureBudgetExceeded,
    UncertainBudgetExceeded,
    EmergencyStopBudgetExceeded,
    ContainmentBudgetExceeded,
    InvalidJobAccounting,
    TrustSnapshotInvalid(String),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    ClearanceWindowUnavailable,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct VerifierCommitment { provider_id: String, verification_policy_digest: String }
#[derive(Debug, Clone, Serialize)]
struct ProbationAggregateCommitment {
    observation_set_digest: String,
    observer_profile_set_digest: String,
    verifier_set_digest: String,
    observation_count: usize,
    machine_count: usize,
    failure_domain_count: usize,
    observation_started_at_unix_ms: u64,
    observation_ended_at_unix_ms: u64,
    attempted_jobs: u64,
    successful_jobs: u64,
    failed_jobs: u64,
    uncertain_jobs: u64,
    emergency_stops: u64,
    containment_actions: u64,
}
#[derive(Debug, Clone, Serialize)]
struct ProbationPolicyCommitment {
    minimum_observation_duration_ms: u64,
    minimum_distinct_machines: usize,
    minimum_distinct_regions: usize,
    minimum_successful_jobs: u64,
    maximum_failure_basis_points: u32,
    maximum_uncertain_basis_points: u32,
    maximum_emergency_stops: u64,
    maximum_containment_actions: u64,
    maximum_observations: usize,
    maximum_clearance_duration_ms: u64,
}
#[derive(Debug, Clone, Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}
#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment { activation_basis_id: String, bridge_basis_ids: Vec<String>, current_basis_id: String }
#[derive(Debug, Clone, Serialize)]
struct PreparedCommitment {
    schema: &'static str,
    lineage_handoff_id: String,
    activation_permit_id: String,
    predecessor_root_digest: String,
    current_head_digest: String,
    governance_view_digest: String,
    registry_head_digest: String,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: String,
    aggregate: ProbationAggregateCommitment,
    probation_policy_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    activation_operational_basis_id: String,
    operational_basis_id: String,
    clock_envelope_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
    clearance_expires_at_unix_ms: u64,
}
#[derive(Debug, Clone, Serialize)]
struct AuthorizedCommitment {
    schema: &'static str,
    prepared_id: String,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_lineage_bound_upgrade_probation_clearance_v1(
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
    activation: &LineageBoundUpgradeActivationPermitV1,
    activation_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
    signed_observations: &[SignedUpgradeProbationObservationV1],
    observer_profiles: &[UpgradeProbationObserverProfileV1],
    probation_policy: &UpgradeProbationPolicy,
    verification_policy: &ProbationObservationVerificationPolicyV1,
    verification_providers: &[&dyn UpgradeProbationObservationVerifierV1],
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
) -> Result<PreparedLineageBoundUpgradeProbationClearanceV1, Vec<LineageBoundProbationError>> {
    let mut violations = Vec::new();

    if activation.lineage_handoff_id() != handoff.id()
        || activation.inner_handoff_id() != handoff.inner_handoff_id()
        || activation.handoff_plan_digest() != handoff.plan_digest()
        || activation.predecessor_root_digest() != handoff.predecessor_root_id().as_digest()
        || activation.current_head_digest() != handoff.current_head_id().as_digest()
        || activation.governance_view_digest() != handoff.governance_view_id().as_digest()
        || activation.registry_head_digest() != handoff.registry_head_id().as_digest()
        || activation.predecessor_endpoint_digest() != handoff.predecessor_endpoint_digest()
        || activation.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
        || activation.predecessor_checkpoint_digest() != handoff.predecessor_checkpoint_digest()
        || activation.predecessor_transparency_log_digest() != handoff.predecessor_transparency_log_digest()
    {
        violations.push(LineageBoundProbationError::ActivationLineageMismatch);
    }
    if activation_basis.id() != activation.current_operational_basis_id() {
        violations.push(LineageBoundProbationError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundProbationError::Clock(error)); return Err(violations); }
    };
    if activation_clock.id() != activation.current_clock_envelope_id() {
        violations.push(LineageBoundProbationError::ActivationEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_LINEAGE_BOUND_PROBATION_CLOCK_HOPS {
        violations.push(LineageBoundProbationError::TooManyClockHops { actual: clock_bridge.len(), maximum: MAX_LINEAGE_BOUND_PROBATION_CLOCK_HOPS });
    } else if let Err(error) = verify_clock_lineage(activation_basis.id(), clock_bridge, current_basis) {
        violations.push(error);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundProbationError::Clock(error)); return Err(violations); }
    };
    if current_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms {
        violations.push(LineageBoundProbationError::FinalizationMayBeClosed);
    }

    if !valid_probation_policy(probation_policy) { violations.push(LineageBoundProbationError::InvalidProbationPolicy); }
    if !valid_verification_policy(verification_policy) { violations.push(LineageBoundProbationError::InvalidVerificationPolicy); }
    if !valid_threshold_policy(threshold_policy) { violations.push(LineageBoundProbationError::InvalidThresholdPolicy); }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony { violations.push(LineageBoundProbationError::ThresholdPolicyUsageMismatch); }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(LineageBoundProbationError::InsufficientVerificationProviders { actual: verification_providers.len(), required: verification_policy.minimum_distinct_providers });
    }
    if verification_providers.len() > verification_policy.maximum_providers {
        violations.push(LineageBoundProbationError::TooManyVerificationProviders { actual: verification_providers.len(), maximum: verification_policy.maximum_providers });
    }
    if signed_observations.is_empty() { violations.push(LineageBoundProbationError::EmptyObservations); }
    if signed_observations.len() > probation_policy.maximum_observations || signed_observations.len() > MAX_PROBATION_OBSERVATIONS {
        violations.push(LineageBoundProbationError::TooManyObservations { actual: signed_observations.len(), maximum: probation_policy.maximum_observations.min(MAX_PROBATION_OBSERVATIONS) });
    }

    if let Err(error) = trust_snapshot.validate() { violations.push(LineageBoundProbationError::TrustSnapshotInvalid(format!("{error:?}"))); }
    if let Err(reason) = current_clock.require_valid_across_seconds_window(trust_snapshot.issued_at_unix_s, trust_snapshot.expires_at_unix_s) {
        violations.push(LineageBoundProbationError::TrustSnapshotNotValidAcrossEnvelope(reason));
    }
    if let Err(error) = containment_state.validate() { violations.push(LineageBoundProbationError::ContainmentStateInvalid(format!("{error:?}"))); }

    let profile_map = match canonical_profile_map(observer_profiles) { Ok(value) => value, Err(mut errors) => { violations.append(&mut errors); BTreeMap::new() } };
    let verifier_commitments = match validate_verification_providers(verification_providers) { Ok(value) => value, Err(mut errors) => { violations.append(&mut errors); Vec::new() } };

    let snapshot_issued_ms = trust_snapshot.issued_at_unix_s.checked_mul(1_000);
    if snapshot_issued_ms.is_none() { violations.push(LineageBoundProbationError::InvalidJobAccounting); }
    let mut used_profiles = BTreeSet::new();
    let mut evidence_digests = Vec::new();
    let mut observation_digests = BTreeSet::new();
    let mut machine_ids = BTreeSet::new();
    let mut failure_domains = BTreeSet::new();
    let mut started_at = u64::MAX;
    let mut ended_at = 0u64;
    let mut attempted = 0u64;
    let mut successful = 0u64;
    let mut failed = 0u64;
    let mut uncertain = 0u64;
    let mut emergency = 0u64;
    let mut containment = 0u64;

    for signed in signed_observations.iter().take(probation_policy.maximum_observations) {
        let key_id = signed.key_id.clone();
        if !signed.algorithm.is_canonical() || invalid_identifier(&signed.key_id) || signed.signature.is_empty() || signed.signature.len() > 64 * 1024 {
            violations.push(LineageBoundProbationError::InvalidSignedObservation(key_id)); continue;
        }
        if let Err(error) = signed.observation.validate() { violations.push(LineageBoundProbationError::InvalidSignedObservation(format!("{}: {error:?}", signed.key_id))); continue; }
        let expected_digest = match digest_upgrade_probation_observation(&signed.observation) { Ok(value) => value, Err(error) => { violations.push(LineageBoundProbationError::InvalidSignedObservation(format!("{}: {error:?}", signed.key_id))); continue; } };
        if expected_digest != signed.observation_digest { violations.push(LineageBoundProbationError::ObservationDigestMismatch(key_id)); continue; }
        if !observation_digests.insert(signed.observation_digest) { violations.push(LineageBoundProbationError::DuplicateObservation); continue; }
        if signed.observation.handoff_digest != handoff.plan_digest() { violations.push(LineageBoundProbationError::ObservationHandoffMismatch(signed.key_id.clone())); }
        if signed.observation.successor_state_digest != handoff.plan().successor.durable_state_digest { violations.push(LineageBoundProbationError::ObservationSuccessorMismatch(signed.key_id.clone())); }
        if signed.observation.started_at_unix_ms < handoff.plan().activates_at_unix_ms { violations.push(LineageBoundProbationError::ObservationBeforeActivation(signed.key_id.clone())); }
        if signed.observation.ended_at_unix_ms > current_clock.lower_unix_ms() { violations.push(LineageBoundProbationError::ObservationMayStillBeRunning(signed.key_id.clone())); }
        if snapshot_issued_ms.is_none_or(|issued| issued > signed.observation.started_at_unix_ms) { violations.push(LineageBoundProbationError::TrustSnapshotPostdatesObservation(signed.key_id.clone())); }

        let identity = (signed.algorithm.clone(), signed.key_id.clone());
        let Some(profile) = profile_map.get(&identity) else { violations.push(LineageBoundProbationError::MissingObserverProfile(signed.key_id.clone())); continue; };
        used_profiles.insert(identity);
        if profile.machine_id != signed.observation.machine_id { violations.push(LineageBoundProbationError::ObserverMachineMismatch(signed.key_id.clone())); }
        if profile.failure_domain != signed.observation.region_id { violations.push(LineageBoundProbationError::ObserverFailureDomainMismatch(signed.key_id.clone())); }
        requalify_observer_key(signed, trust_snapshot, containment_state, &current_clock, &mut violations);

        let message = observation_signature_message(signed.observation_digest);
        for provider in verification_providers {
            match provider.verify_observation_signature(&signed.algorithm, &signed.key_id, &message, &signed.signature) {
                Ok(true) => {}
                Ok(false) => violations.push(LineageBoundProbationError::ObservationSignatureRejected { provider: provider.provider_id().to_string(), key_id: signed.key_id.clone() }),
                Err(reason) => violations.push(LineageBoundProbationError::VerificationProviderError { provider: provider.provider_id().to_string(), reason }),
            }
        }

        machine_ids.insert(profile.machine_id.clone()); failure_domains.insert(profile.failure_domain.clone());
        started_at = started_at.min(signed.observation.started_at_unix_ms); ended_at = ended_at.max(signed.observation.ended_at_unix_ms);
        attempted = checked_add(attempted, signed.observation.attempted_jobs, &mut violations);
        successful = checked_add(successful, signed.observation.successful_jobs, &mut violations);
        failed = checked_add(failed, signed.observation.failed_jobs, &mut violations);
        uncertain = checked_add(uncertain, signed.observation.uncertain_jobs, &mut violations);
        emergency = checked_add(emergency, signed.observation.emergency_stops, &mut violations);
        containment = checked_add(containment, signed.observation.containment_actions, &mut violations);
        match hash_serializable(SIGNED_OBSERVATION_EVIDENCE_DOMAIN, signed) { Ok(value) => evidence_digests.push(value), Err(error) => violations.push(error) }
    }

    for identity in profile_map.keys() {
        if !used_profiles.contains(identity) { violations.push(LineageBoundProbationError::UnexpectedObserverProfile(identity.1.clone())); }
    }
    if !signed_observations.is_empty() && ended_at.checked_sub(started_at).is_none_or(|duration| duration < probation_policy.minimum_observation_duration_ms) { violations.push(LineageBoundProbationError::ObservationTooShort); }
    if machine_ids.len() < probation_policy.minimum_distinct_machines { violations.push(LineageBoundProbationError::InsufficientMachines { actual: machine_ids.len(), required: probation_policy.minimum_distinct_machines }); }
    if failure_domains.len() < probation_policy.minimum_distinct_regions { violations.push(LineageBoundProbationError::InsufficientFailureDomains { actual: failure_domains.len(), required: probation_policy.minimum_distinct_regions }); }
    if successful < probation_policy.minimum_successful_jobs { violations.push(LineageBoundProbationError::InsufficientSuccessfulJobs); }
    let accounted = successful.checked_add(failed).and_then(|v| v.checked_add(uncertain));
    if accounted != Some(attempted) { violations.push(LineageBoundProbationError::InvalidJobAccounting); }
    match checked_basis_points(failed, attempted) { Some(bp) if bp > u64::from(probation_policy.maximum_failure_basis_points) => violations.push(LineageBoundProbationError::FailureBudgetExceeded), None => violations.push(LineageBoundProbationError::InvalidJobAccounting), _ => {} }
    match checked_basis_points(uncertain, attempted) { Some(bp) if bp > u64::from(probation_policy.maximum_uncertain_basis_points) => violations.push(LineageBoundProbationError::UncertainBudgetExceeded), None => violations.push(LineageBoundProbationError::InvalidJobAccounting), _ => {} }
    if emergency > probation_policy.maximum_emergency_stops { violations.push(LineageBoundProbationError::EmergencyStopBudgetExceeded); }
    if containment > probation_policy.maximum_containment_actions { violations.push(LineageBoundProbationError::ContainmentBudgetExceeded); }

    let clearance_expires_at_unix_ms = match current_clock.lower_unix_ms().checked_add(probation_policy.maximum_clearance_duration_ms) {
        Some(value) => value.min(handoff.plan().finalization_deadline_unix_ms), None => { violations.push(LineageBoundProbationError::ClearanceWindowUnavailable); 0 }
    };
    if clearance_expires_at_unix_ms <= current_clock.upper_unix_ms() { violations.push(LineageBoundProbationError::ClearanceWindowUnavailable); }
    if !violations.is_empty() { return Err(violations); }

    evidence_digests.sort();
    let observation_set_digest = hash_serializable(SIGNED_OBSERVATION_SET_DOMAIN, &evidence_digests).map_err(|e| vec![e])?;
    let mut canonical_profiles = observer_profiles.to_vec();
    canonical_profiles.sort_by(|left, right| (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str())));
    let observer_profile_set_digest = hash_serializable(OBSERVER_PROFILE_SET_DOMAIN, &canonical_profiles).map_err(|e| vec![e])?;
    let verifier_set_digest = hash_serializable(VERIFIER_SET_DOMAIN, &verifier_commitments).map_err(|e| vec![e])?;
    let probation_policy_digest = digest_probation_policy(probation_policy).map_err(|e| vec![e])?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy).map_err(|e| vec![e])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| vec![LineageBoundProbationError::TrustSnapshotInvalid(format!("{error:?}"))])?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| vec![LineageBoundProbationError::ContainmentStateInvalid(format!("{error:?}"))])?;
    let compromise_tracker_digest = digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(|error| vec![LineageBoundProbationError::ContainmentStateInvalid(format!("{error:?}"))])?;
    let aggregate = ProbationAggregateCommitment { observation_set_digest: observation_set_digest.to_hex(), observer_profile_set_digest: observer_profile_set_digest.to_hex(), verifier_set_digest: verifier_set_digest.to_hex(), observation_count: signed_observations.len(), machine_count: machine_ids.len(), failure_domain_count: failure_domains.len(), observation_started_at_unix_ms: started_at, observation_ended_at_unix_ms: ended_at, attempted_jobs: attempted, successful_jobs: successful, failed_jobs: failed, uncertain_jobs: uncertain, emergency_stops: emergency, containment_actions: containment };
    let lineage = ClockLineageCommitment { activation_basis_id: activation_basis.id().to_hex(), bridge_basis_ids: clock_bridge.iter().map(|basis| basis.id().to_hex()).collect(), current_basis_id: current_basis.id().to_hex() };
    let clock_lineage_digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &lineage).map_err(|e| vec![e])?;
    let commitment = PreparedCommitment { schema: PREPARED_LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA, lineage_handoff_id: handoff.id().to_hex(), activation_permit_id: activation.id().to_hex(), predecessor_root_digest: activation.predecessor_root_digest().to_hex(), current_head_digest: activation.current_head_digest().to_hex(), governance_view_digest: activation.governance_view_digest().to_hex(), registry_head_digest: activation.registry_head_digest().to_hex(), predecessor_finalization_sequence: activation.predecessor_finalization_sequence(), handoff_plan_digest: handoff.plan_digest().to_hex(), aggregate, probation_policy_digest: probation_policy_digest.to_hex(), threshold_policy_digest: threshold_policy_digest.to_hex(), trust_snapshot_digest: trust_snapshot_digest.to_hex(), containment_state_digest: containment_state_digest.to_hex(), compromise_tracker_digest: compromise_tracker_digest.to_hex(), activation_operational_basis_id: activation_basis.id().to_hex(), operational_basis_id: current_basis.id().to_hex(), clock_envelope_id: current_clock.id().to_hex(), clock_lineage_digest: clock_lineage_digest.to_hex(), clock_hop_count: clock_bridge.len(), clearance_expires_at_unix_ms };
    let id = PreparedLineageBoundUpgradeProbationClearanceIdV1(hash_serializable(PREPARED_DOMAIN, &commitment).map_err(|e| vec![e])?);

    Ok(PreparedLineageBoundUpgradeProbationClearanceV1 { id, lineage_handoff_id: handoff.id(), activation_permit_id: activation.id(), predecessor_root_digest: activation.predecessor_root_digest(), current_head_digest: activation.current_head_digest(), governance_view_digest: activation.governance_view_digest(), registry_head_digest: activation.registry_head_digest(), predecessor_finalization_sequence: activation.predecessor_finalization_sequence(), handoff_plan_digest: handoff.plan_digest(), observation_set_digest, observer_profile_set_digest, verifier_set_digest, probation_policy_digest, threshold_policy_digest, trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, activation_operational_basis_id: activation_basis.id(), operational_basis_id: current_basis.id(), clock_envelope_id: current_clock.id(), clock_lineage_digest, clock_hop_count: clock_bridge.len(), observation_count: signed_observations.len(), machine_count: machine_ids.len(), failure_domain_count: failure_domains.len(), observation_started_at_unix_ms: started_at, observation_ended_at_unix_ms: ended_at, clearance_expires_at_unix_ms })
}

pub fn authorize_lineage_bound_upgrade_probation_clearance_v1(
    prepared: PreparedLineageBoundUpgradeProbationClearanceV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<LineageBoundUpgradeProbationClearanceV1, LineageBoundProbationError> {
    if ceremony.purpose() != LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_PURPOSE { return Err(LineageBoundProbationError::CeremonyPurposeMismatch); }
    if ceremony.payload_digest() != prepared.signing_payload_digest() { return Err(LineageBoundProbationError::CeremonyPayloadMismatch); }
    if ceremony.policy_digest() != prepared.threshold_policy_digest { return Err(LineageBoundProbationError::ThresholdPolicyDigestMismatch); }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest { return Err(LineageBoundProbationError::TrustSnapshotDigestMismatch); }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest { return Err(LineageBoundProbationError::CompromiseTrackerDigestMismatch); }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id { return Err(LineageBoundProbationError::ClockEnvelopeMismatch); }
    let commitment = AuthorizedCommitment { schema: LINEAGE_BOUND_UPGRADE_PROBATION_CLEARANCE_SCHEMA, prepared_id: prepared.id.to_hex(), threshold_ceremony_id: ceremony.id().to_hex(), threshold_ceremony_digest: ceremony.ceremony_digest().to_hex() };
    let id = LineageBoundUpgradeProbationClearanceIdV1(hash_serializable(AUTHORIZED_DOMAIN, &commitment)?);
    Ok(LineageBoundUpgradeProbationClearanceV1 { id, prepared_id: prepared.id, lineage_handoff_id: prepared.lineage_handoff_id, activation_permit_id: prepared.activation_permit_id, predecessor_root_digest: prepared.predecessor_root_digest, current_head_digest: prepared.current_head_digest, governance_view_digest: prepared.governance_view_digest, registry_head_digest: prepared.registry_head_digest, predecessor_finalization_sequence: prepared.predecessor_finalization_sequence, handoff_plan_digest: prepared.handoff_plan_digest, threshold_ceremony_id: ceremony.id(), observation_set_digest: prepared.observation_set_digest, observer_profile_set_digest: prepared.observer_profile_set_digest, verifier_set_digest: prepared.verifier_set_digest, probation_policy_digest: prepared.probation_policy_digest, trust_snapshot_digest: prepared.trust_snapshot_digest, containment_state_digest: prepared.containment_state_digest, compromise_tracker_digest: prepared.compromise_tracker_digest, operational_basis_id: prepared.operational_basis_id, clock_envelope_id: prepared.clock_envelope_id, observation_count: prepared.observation_count, machine_count: prepared.machine_count, failure_domain_count: prepared.failure_domain_count, observation_started_at_unix_ms: prepared.observation_started_at_unix_ms, observation_ended_at_unix_ms: prepared.observation_ended_at_unix_ms, clearance_expires_at_unix_ms: prepared.clearance_expires_at_unix_ms })
}

fn verify_clock_lineage(prior: OperationalClockBasisIdV1, bridge: &[OperationalClockBasisV1], current: &OperationalClockBasisV1) -> Result<(), LineageBoundProbationError> {
    if current.id() == prior {
        if bridge.is_empty() { return Ok(()); }
        return Err(LineageBoundProbationError::BrokenClockLineage { hop: 1, expected_predecessor: prior.to_hex(), actual_predecessor: bridge[0].predecessor_operational_basis_id().map(|v| v.to_hex()) });
    }
    let mut expected = prior;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) { return Err(LineageBoundProbationError::BrokenClockLineage { hop: index + 1, expected_predecessor: expected.to_hex(), actual_predecessor: actual.map(|v| v.to_hex()) }); }
        expected = basis.id();
    }
    let actual = current.predecessor_operational_basis_id();
    if actual != Some(expected) { return Err(LineageBoundProbationError::BrokenClockLineage { hop: bridge.len() + 1, expected_predecessor: expected.to_hex(), actual_predecessor: actual.map(|v| v.to_hex()) }); }
    Ok(())
}

fn canonical_profile_map(profiles: &[UpgradeProbationObserverProfileV1]) -> Result<BTreeMap<(SignatureAlgorithm, String), UpgradeProbationObserverProfileV1>, Vec<LineageBoundProbationError>> {
    let mut errors = Vec::new(); let mut map = BTreeMap::new();
    for profile in profiles {
        if !profile.algorithm.is_canonical() || invalid_identifier(&profile.key_id) || invalid_identifier(&profile.machine_id) || invalid_identifier(&profile.failure_domain) { errors.push(LineageBoundProbationError::MissingObserverProfile(profile.key_id.clone())); continue; }
        let key = (profile.algorithm.clone(), profile.key_id.clone());
        if map.insert(key, profile.clone()).is_some() { errors.push(LineageBoundProbationError::DuplicateObserverProfile(profile.key_id.clone())); }
    }
    if errors.is_empty() { Ok(map) } else { Err(errors) }
}

fn validate_verification_providers(providers: &[&dyn UpgradeProbationObservationVerifierV1]) -> Result<Vec<VerifierCommitment>, Vec<LineageBoundProbationError>> {
    let mut errors = Vec::new(); let mut seen = BTreeSet::new(); let mut commitments = Vec::new();
    for provider in providers {
        let id = provider.provider_id().to_string(); let policy = provider.verification_policy_digest();
        if invalid_identifier(&id) || policy == Sha256Digest([0; 32]) { errors.push(LineageBoundProbationError::InvalidProvider(id)); continue; }
        if !seen.insert(id.clone()) { errors.push(LineageBoundProbationError::DuplicateProvider(id)); continue; }
        commitments.push(VerifierCommitment { provider_id: id, verification_policy_digest: policy.to_hex() });
    }
    commitments.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
    if errors.is_empty() { Ok(commitments) } else { Err(errors) }
}

fn requalify_observer_key(signed: &SignedUpgradeProbationObservationV1, trust: &TrustSnapshot, containment: &FabricationContainmentState, clock: &ClockGovernanceEvaluationEnvelopeV1, errors: &mut Vec<LineageBoundProbationError>) {
    let Some(record) = trust.keys.iter().find(|record| record.algorithm == signed.algorithm && record.key_id == signed.key_id) else { errors.push(LineageBoundProbationError::ObserverUnknown(signed.key_id.clone())); return; };
    if record.status != KeyLifecycleStatus::Active { errors.push(LineageBoundProbationError::ObserverNotActive(signed.key_id.clone())); }
    if !record.usages.contains(&KeyUsage::UpgradeProbation) { errors.push(LineageBoundProbationError::ObserverUsageNotAllowed(signed.key_id.clone())); }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(record.not_before_unix_s, record.not_after_unix_s) { errors.push(LineageBoundProbationError::ObserverNotValidAcrossEnvelope { key_id: signed.key_id.clone(), reason }); }
    let not_before_ms = record.not_before_unix_s.checked_mul(1_000);
    let not_after_ms = record.not_after_unix_s.and_then(|value| value.checked_mul(1_000));
    if not_before_ms.is_none_or(|value| signed.observation.started_at_unix_ms < value) || (record.not_after_unix_s.is_some() && not_after_ms.is_none_or(|value| signed.observation.ended_at_unix_ms >= value)) { errors.push(LineageBoundProbationError::ObserverKeyInvalidForObservation(signed.key_id.clone())); }
    for compromise in containment.signer_compromise_tracker.records().iter().filter(|c| c.signer.algorithm == signed.algorithm && c.signer.key_id == signed.key_id && c.affected_usages.contains(&KeyUsage::UpgradeProbation)) {
        if clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s).is_err() { errors.push(LineageBoundProbationError::ObserverCompromisedAcrossEnvelope(signed.key_id.clone())); }
    }
}

fn observation_signature_message(digest: Sha256Digest) -> Vec<u8> { let mut message = OBSERVATION_SIGNATURE_DOMAIN.to_vec(); message.extend_from_slice(&digest.0); message }
fn checked_add(current: u64, value: u64, errors: &mut Vec<LineageBoundProbationError>) -> u64 { match current.checked_add(value) { Some(v) => v, None => { errors.push(LineageBoundProbationError::InvalidJobAccounting); current } } }
fn checked_basis_points(part: u64, whole: u64) -> Option<u64> { if whole == 0 { None } else { part.checked_mul(10_000).map(|v| v / whole) } }
fn invalid_identifier(value: &str) -> bool { value.trim().is_empty() || value != value.trim() || value.len() > MAX_THRESHOLD_KEY_ID_BYTES || value.chars().any(char::is_control) }
fn valid_probation_policy(policy: &UpgradeProbationPolicy) -> bool { policy.minimum_observation_duration_ms > 0 && policy.minimum_distinct_machines > 0 && policy.minimum_distinct_regions > 0 && policy.minimum_successful_jobs > 0 && policy.maximum_failure_basis_points <= 10_000 && policy.maximum_uncertain_basis_points <= 10_000 && policy.maximum_observations > 0 && policy.maximum_observations <= MAX_PROBATION_OBSERVATIONS && policy.maximum_clearance_duration_ms > 0 }
fn valid_verification_policy(policy: &ProbationObservationVerificationPolicyV1) -> bool { policy.minimum_distinct_providers > 0 && policy.maximum_providers > 0 && policy.minimum_distinct_providers <= policy.maximum_providers && policy.maximum_providers <= MAX_LINEAGE_BOUND_PROBATION_VERIFIERS }
fn valid_threshold_policy(policy: &ThresholdCeremonyPolicy) -> bool { policy.minimum_distinct_signers > 0 && policy.maximum_approvals > 0 && policy.minimum_distinct_signers <= policy.maximum_approvals && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS && policy.required_algorithms.iter().all(SignatureAlgorithm::is_canonical) && policy.allowed_key_ids.as_ref().is_none_or(|ids| ids.iter().all(|id| !invalid_identifier(id))) }

fn digest_probation_policy(policy: &UpgradeProbationPolicy) -> Result<Sha256Digest, LineageBoundProbationError> { hash_serializable(PROBATION_POLICY_DOMAIN, &ProbationPolicyCommitment { minimum_observation_duration_ms: policy.minimum_observation_duration_ms, minimum_distinct_machines: policy.minimum_distinct_machines, minimum_distinct_regions: policy.minimum_distinct_regions, minimum_successful_jobs: policy.minimum_successful_jobs, maximum_failure_basis_points: policy.maximum_failure_basis_points, maximum_uncertain_basis_points: policy.maximum_uncertain_basis_points, maximum_emergency_stops: policy.maximum_emergency_stops, maximum_containment_actions: policy.maximum_containment_actions, maximum_observations: policy.maximum_observations, maximum_clearance_duration_ms: policy.maximum_clearance_duration_ms }) }
fn digest_threshold_policy(policy: &ThresholdCeremonyPolicy) -> Result<Sha256Digest, LineageBoundProbationError> { hash_serializable(THRESHOLD_POLICY_DOMAIN, &ThresholdPolicyCommitment { minimum_distinct_signers: policy.minimum_distinct_signers, maximum_approvals: policy.maximum_approvals, require_algorithm_diversity: policy.require_algorithm_diversity, required_algorithms: &policy.required_algorithms, allowed_key_ids: &policy.allowed_key_ids, key_usage: policy.key_usage }) }
fn hash_serializable<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<Sha256Digest, LineageBoundProbationError> { let bytes = serde_json::to_vec(value).map_err(|error| LineageBoundProbationError::Encoding(error.to_string()))?; let mut hasher = Sha256::new(); hasher.update(domain); hasher.update(&bytes); Ok(hasher.finalize()) }
