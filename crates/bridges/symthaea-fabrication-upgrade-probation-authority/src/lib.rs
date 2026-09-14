// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed, interval-safe operational probation authority for clock-governed upgrades.
//!
//! The legacy probation DTO remains portable evidence. This bridge adds the missing live-authority
//! theorem: each operational summary is signed by an explicitly `UpgradeProbation`-authorized key,
//! bound to a governance-approved machine/failure-domain profile, independently verified by exact
//! signature bytes, and aggregated only after all observations are definitely complete under a
//! fresh descendant operational clock.
//!
//! `telemetry_evidence_digest` remains a signed, non-zero reference in this version. This crate does
//! not claim that the referenced raw telemetry set itself has been rebound; that is a separate
//! evidence theorem rather than something hidden behind the clearance type.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
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
    MAX_PROBATION_OBSERVATIONS, UpgradeProbationObservation, UpgradeProbationPolicy,
    digest_upgrade_probation_observation,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_runtime::{
    ClockGovernedUpgradeActivationPermitIdV1, ClockGovernedUpgradeActivationPermitV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA: &str =
    "symthaea.fabrication.signed-upgrade-probation-observation.v1";
pub const CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_PURPOSE: &str =
    "clock-governed-upgrade-probation-clearance-v1";
pub const PREPARED_UPGRADE_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.prepared-upgrade-probation-clearance.v1";
pub const CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-probation-clearance.v1";
pub const MAX_PROBATION_OBSERVATION_VERIFIERS: usize = 8;
pub const MAX_PROBATION_CLOCK_HOPS: usize = 4096;

const OBSERVATION_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-observation-signature.v1\0";
const SIGNED_OBSERVATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-evidence.v1\0";
const SIGNED_OBSERVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-upgrade-probation-observation-set.v1\0";
const OBSERVER_PROFILE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-observer-profile-set.v1\0";
const VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-verifier-set.v1\0";
const PROBATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-probation-policy.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_CLEARANCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-upgrade-probation-clearance.v1\0";
const AUTHORIZED_CLEARANCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-probation-clearance.v1\0";

/// Portable exact signature wrapper for one existing probation observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedUpgradeProbationObservationV1 {
    pub schema_version: String,
    pub observation: UpgradeProbationObservation,
    pub observation_digest: Sha256Digest,
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

/// Portable signer identity metadata. It gains authority only because the full profile set is
/// committed into the governance threshold payload for one exact clearance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeProbationObserverProfileV1 {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub machine_id: String,
    pub failure_domain: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbationObservationVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ProbationObservationVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: MAX_PROBATION_OBSERVATION_VERIFIERS,
        }
    }
}

/// Runtime verifier for exact observation signature bytes.
pub trait UpgradeProbationObservationVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;
    fn verify_observation_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedUpgradeProbationClearanceIdV1(Sha256Digest);

impl PreparedUpgradeProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedUpgradeProbationClearanceIdV1(Sha256Digest);

impl ClockGovernedUpgradeProbationClearanceIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proposal for clearance. The threshold payload commits the exact signed observations,
/// governed observer mapping, verifier set, policy, trust/containment context, and fresh clock.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedUpgradeProbationClearanceV1 {
    id: PreparedUpgradeProbationClearanceIdV1,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: ClockGovernedUpgradeActivationPermitIdV1,
    observation_set_digest: Sha256Digest,
    observer_profile_set_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    probation_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    observation_count: usize,
    machine_count: usize,
    failure_domain_count: usize,
    observation_started_at_unix_ms: u64,
    observation_ended_at_unix_ms: u64,
    clearance_expires_at_unix_ms: u64,
}

impl PreparedUpgradeProbationClearanceV1 {
    pub fn id(&self) -> PreparedUpgradeProbationClearanceIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }
    pub fn activation_permit_id(&self) -> ClockGovernedUpgradeActivationPermitIdV1 {
        self.activation_permit_id
    }
    pub fn observation_set_digest(&self) -> Sha256Digest {
        self.observation_set_digest
    }
    pub fn observer_profile_set_digest(&self) -> Sha256Digest {
        self.observer_profile_set_digest
    }
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn probation_policy_digest(&self) -> Sha256Digest {
        self.probation_policy_digest
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
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
    pub fn clearance_expires_at_unix_ms(&self) -> u64 {
        self.clearance_expires_at_unix_ms
    }
}

/// Opaque short-lived probation clearance for one exact activated handoff.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedUpgradeProbationClearanceV1 {
    id: ClockGovernedUpgradeProbationClearanceIdV1,
    prepared_id: PreparedUpgradeProbationClearanceIdV1,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: ClockGovernedUpgradeActivationPermitIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    observation_set_digest: Sha256Digest,
    observer_profile_set_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    probation_policy_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    observation_count: usize,
    machine_count: usize,
    failure_domain_count: usize,
    observation_started_at_unix_ms: u64,
    observation_ended_at_unix_ms: u64,
    clearance_expires_at_unix_ms: u64,
}

impl ClockGovernedUpgradeProbationClearanceV1 {
    pub fn id(&self) -> ClockGovernedUpgradeProbationClearanceIdV1 {
        self.id
    }
    pub fn prepared_id(&self) -> PreparedUpgradeProbationClearanceIdV1 {
        self.prepared_id
    }
    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }
    pub fn activation_permit_id(&self) -> ClockGovernedUpgradeActivationPermitIdV1 {
        self.activation_permit_id
    }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.threshold_ceremony_id
    }
    pub fn observation_set_digest(&self) -> Sha256Digest {
        self.observation_set_digest
    }
    pub fn observer_profile_set_digest(&self) -> Sha256Digest {
        self.observer_profile_set_digest
    }
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn probation_policy_digest(&self) -> Sha256Digest {
        self.probation_policy_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }
    pub fn machine_count(&self) -> usize {
        self.machine_count
    }
    pub fn failure_domain_count(&self) -> usize {
        self.failure_domain_count
    }
    pub fn observation_started_at_unix_ms(&self) -> u64 {
        self.observation_started_at_unix_ms
    }
    pub fn observation_ended_at_unix_ms(&self) -> u64 {
        self.observation_ended_at_unix_ms
    }
    pub fn clearance_expires_at_unix_ms(&self) -> u64 {
        self.clearance_expires_at_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeProbationAuthorityError {
    ActivationHandoffMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    InvalidProbationPolicy,
    InvalidVerificationPolicy,
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
    ObserverMachineMismatch(String),
    ObserverFailureDomainMismatch(String),
    ObserverUnknown(String),
    ObserverNotActive(String),
    ObserverUsageNotAllowed(String),
    ObserverKeyInvalidForObservation(String),
    ObserverNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    ObserverCompromisedAcrossEnvelope(String),
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
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
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
struct VerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

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

/// Prepare short-lived probation clearance authority under fresh descendant trusted time.
#[allow(clippy::too_many_arguments)]
pub fn prepare_clock_governed_upgrade_probation_clearance_v1(
    handoff: &ClockGovernedUpgradeHandoffV1,
    activation: &ClockGovernedUpgradeActivationPermitV1,
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
) -> Result<PreparedUpgradeProbationClearanceV1, Vec<UpgradeProbationAuthorityError>> {
    let mut violations = Vec::new();

    if activation.handoff_id() != handoff.id() || activation.handoff_plan_digest() != handoff.plan_digest() {
        violations.push(UpgradeProbationAuthorityError::ActivationHandoffMismatch);
    }
    if activation_basis.id() != activation.current_operational_basis_id() {
        violations.push(UpgradeProbationAuthorityError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeProbationAuthorityError::Clock(error));
            return Err(violations);
        }
    };
    if activation_clock.id() != activation.current_clock_envelope_id() {
        violations.push(UpgradeProbationAuthorityError::ActivationEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_PROBATION_CLOCK_HOPS {
        violations.push(UpgradeProbationAuthorityError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_PROBATION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(activation_basis.id(), clock_bridge, current_basis) {
        violations.push(error);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeProbationAuthorityError::Clock(error));
            return Err(violations);
        }
    };
    if current_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms {
        violations.push(UpgradeProbationAuthorityError::FinalizationMayBeClosed);
    }

    if !valid_probation_policy(probation_policy) {
        violations.push(UpgradeProbationAuthorityError::InvalidProbationPolicy);
    }
    if !valid_verification_policy(verification_policy) {
        violations.push(UpgradeProbationAuthorityError::InvalidVerificationPolicy);
    }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(UpgradeProbationAuthorityError::InsufficientVerificationProviders {
            actual: verification_providers.len(),
            required: verification_policy.minimum_distinct_providers,
        });
    }
    if verification_providers.len() > verification_policy.maximum_providers {
        violations.push(UpgradeProbationAuthorityError::TooManyVerificationProviders {
            actual: verification_providers.len(),
            maximum: verification_policy.maximum_providers,
        });
    }
    if !valid_threshold_policy(threshold_policy) {
        violations.push(UpgradeProbationAuthorityError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        violations.push(UpgradeProbationAuthorityError::ThresholdPolicyUsageMismatch);
    }

    if signed_observations.is_empty() {
        violations.push(UpgradeProbationAuthorityError::EmptyObservations);
    }
    if signed_observations.len() > probation_policy.maximum_observations
        || signed_observations.len() > MAX_PROBATION_OBSERVATIONS
    {
        violations.push(UpgradeProbationAuthorityError::TooManyObservations {
            actual: signed_observations.len(),
            maximum: probation_policy.maximum_observations.min(MAX_PROBATION_OBSERVATIONS),
        });
    }

    if let Err(error) = trust_snapshot.validate() {
        violations.push(UpgradeProbationAuthorityError::TrustSnapshotInvalid(format!("{error:?}")));
    }
    if let Err(reason) = current_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(UpgradeProbationAuthorityError::TrustSnapshotNotValidAcrossEnvelope(reason));
    }
    if let Err(error) = containment_state.validate() {
        violations.push(UpgradeProbationAuthorityError::ContainmentStateInvalid(format!("{error:?}")));
    }

    let profile_map = match canonical_profile_map(observer_profiles) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            BTreeMap::new()
        }
    };
    let verifier_commitments = match validate_verification_providers(verification_providers) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            Vec::new()
        }
    };

    let mut observation_evidence_digests = Vec::new();
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
        if signed.schema_version != SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA
            || !signed.algorithm.is_canonical()
            || invalid_identifier(&signed.key_id)
            || signed.signature.is_empty()
            || signed.signature.len() > 64 * 1024
        {
            violations.push(UpgradeProbationAuthorityError::InvalidSignedObservation(key_id));
            continue;
        }
        if let Err(error) = signed.observation.validate() {
            violations.push(UpgradeProbationAuthorityError::InvalidSignedObservation(format!(
                "{}: {error:?}", signed.key_id
            )));
            continue;
        }
        let expected_digest = match digest_upgrade_probation_observation(&signed.observation) {
            Ok(value) => value,
            Err(error) => {
                violations.push(UpgradeProbationAuthorityError::InvalidSignedObservation(format!(
                    "{}: {error:?}", signed.key_id
                )));
                continue;
            }
        };
        if expected_digest != signed.observation_digest {
            violations.push(UpgradeProbationAuthorityError::ObservationDigestMismatch(key_id));
            continue;
        }
        if !observation_digests.insert(signed.observation_digest) {
            violations.push(UpgradeProbationAuthorityError::DuplicateObservation);
            continue;
        }
        if signed.observation.handoff_digest != handoff.plan_digest() {
            violations.push(UpgradeProbationAuthorityError::ObservationHandoffMismatch(
                signed.key_id.clone(),
            ));
        }
        if signed.observation.successor_state_digest != handoff.plan().successor.durable_state_digest {
            violations.push(UpgradeProbationAuthorityError::ObservationSuccessorMismatch(
                signed.key_id.clone(),
            ));
        }
        if signed.observation.started_at_unix_ms < handoff.plan().activates_at_unix_ms {
            violations.push(UpgradeProbationAuthorityError::ObservationBeforeActivation(
                signed.key_id.clone(),
            ));
        }
        if signed.observation.ended_at_unix_ms > current_clock.lower_unix_ms() {
            violations.push(UpgradeProbationAuthorityError::ObservationMayStillBeRunning(
                signed.key_id.clone(),
            ));
        }

        let identity = (signed.algorithm.clone(), signed.key_id.clone());
        let Some(profile) = profile_map.get(&identity) else {
            violations.push(UpgradeProbationAuthorityError::MissingObserverProfile(
                signed.key_id.clone(),
            ));
            continue;
        };
        if profile.machine_id != signed.observation.machine_id {
            violations.push(UpgradeProbationAuthorityError::ObserverMachineMismatch(
                signed.key_id.clone(),
            ));
        }
        if profile.failure_domain != signed.observation.region_id {
            violations.push(UpgradeProbationAuthorityError::ObserverFailureDomainMismatch(
                signed.key_id.clone(),
            ));
        }
        requalify_observer_key(
            signed,
            trust_snapshot,
            containment_state,
            &current_clock,
            &mut violations,
        );

        let message = observation_signature_message(signed.observation_digest);
        for provider in verification_providers {
            match provider.verify_observation_signature(
                &signed.algorithm,
                &signed.key_id,
                &message,
                &signed.signature,
            ) {
                Ok(true) => {}
                Ok(false) => violations.push(
                    UpgradeProbationAuthorityError::ObservationSignatureRejected {
                        provider: provider.provider_id().to_string(),
                        key_id: signed.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    UpgradeProbationAuthorityError::VerificationProviderError {
                        provider: provider.provider_id().to_string(),
                        reason,
                    },
                ),
            }
        }

        machine_ids.insert(profile.machine_id.clone());
        failure_domains.insert(profile.failure_domain.clone());
        started_at = started_at.min(signed.observation.started_at_unix_ms);
        ended_at = ended_at.max(signed.observation.ended_at_unix_ms);
        attempted = match attempted.checked_add(signed.observation.attempted_jobs) {
            Some(value) => value,
            None => {
                violations.push(UpgradeProbationAuthorityError::InvalidJobAccounting);
                attempted
            }
        };
        successful = checked_add_or_record(successful, signed.observation.successful_jobs, &mut violations);
        failed = checked_add_or_record(failed, signed.observation.failed_jobs, &mut violations);
        uncertain = checked_add_or_record(uncertain, signed.observation.uncertain_jobs, &mut violations);
        emergency = checked_add_or_record(emergency, signed.observation.emergency_stops, &mut violations);
        containment = checked_add_or_record(containment, signed.observation.containment_actions, &mut violations);
        if let Ok(digest) = hash_serializable(SIGNED_OBSERVATION_EVIDENCE_DOMAIN, signed) {
            observation_evidence_digests.push(digest);
        } else {
            violations.push(UpgradeProbationAuthorityError::Encoding(
                "signed observation evidence".into(),
            ));
        }
    }

    if !signed_observations.is_empty() {
        let duration = ended_at.checked_sub(started_at).unwrap_or_default();
        if duration < probation_policy.minimum_observation_duration_ms {
            violations.push(UpgradeProbationAuthorityError::ObservationTooShort);
        }
    }
    if machine_ids.len() < probation_policy.minimum_distinct_machines {
        violations.push(UpgradeProbationAuthorityError::InsufficientMachines {
            actual: machine_ids.len(),
            required: probation_policy.minimum_distinct_machines,
        });
    }
    if failure_domains.len() < probation_policy.minimum_distinct_regions {
        violations.push(UpgradeProbationAuthorityError::InsufficientFailureDomains {
            actual: failure_domains.len(),
            required: probation_policy.minimum_distinct_regions,
        });
    }
    if successful < probation_policy.minimum_successful_jobs {
        violations.push(UpgradeProbationAuthorityError::InsufficientSuccessfulJobs);
    }
    if basis_points(failed, attempted) > u64::from(probation_policy.maximum_failure_basis_points) {
        violations.push(UpgradeProbationAuthorityError::FailureBudgetExceeded);
    }
    if basis_points(uncertain, attempted) > u64::from(probation_policy.maximum_uncertain_basis_points) {
        violations.push(UpgradeProbationAuthorityError::UncertainBudgetExceeded);
    }
    if emergency > probation_policy.maximum_emergency_stops {
        violations.push(UpgradeProbationAuthorityError::EmergencyStopBudgetExceeded);
    }
    if containment > probation_policy.maximum_containment_actions {
        violations.push(UpgradeProbationAuthorityError::ContainmentBudgetExceeded);
    }

    let clearance_horizon = match current_clock
        .lower_unix_ms()
        .checked_add(probation_policy.maximum_clearance_duration_ms)
    {
        Some(value) => value,
        None => {
            violations.push(UpgradeProbationAuthorityError::ClearanceWindowUnavailable);
            0
        }
    };
    let clearance_expires_at_unix_ms =
        clearance_horizon.min(handoff.plan().finalization_deadline_unix_ms);
    if clearance_expires_at_unix_ms <= current_clock.upper_unix_ms() {
        violations.push(UpgradeProbationAuthorityError::ClearanceWindowUnavailable);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    observation_evidence_digests.sort();
    let observation_set_digest =
        hash_serializable(SIGNED_OBSERVATION_SET_DOMAIN, &observation_evidence_digests)
            .map_err(|error| vec![error])?;
    let mut canonical_profiles = observer_profiles.to_vec();
    canonical_profiles.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    let observer_profile_set_digest =
        hash_serializable(OBSERVER_PROFILE_SET_DOMAIN, &canonical_profiles)
            .map_err(|error| vec![error])?;
    let verifier_set_digest = hash_serializable(VERIFIER_SET_DOMAIN, &verifier_commitments)
        .map_err(|error| vec![error])?;
    let probation_policy_digest = digest_probation_policy(probation_policy).map_err(|error| vec![error])?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy).map_err(|error| vec![error])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![UpgradeProbationAuthorityError::TrustSnapshotInvalid(format!("{error:?}"))]
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![UpgradeProbationAuthorityError::ContainmentStateInvalid(format!("{error:?}"))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| vec![UpgradeProbationAuthorityError::ContainmentStateInvalid(format!("{error:?}"))],
        )?;

    let aggregate = ProbationAggregateCommitment {
        observation_set_digest: observation_set_digest.to_hex(),
        observer_profile_set_digest: observer_profile_set_digest.to_hex(),
        verifier_set_digest: verifier_set_digest.to_hex(),
        observation_count: signed_observations.len(),
        machine_count: machine_ids.len(),
        failure_domain_count: failure_domains.len(),
        observation_started_at_unix_ms: started_at,
        observation_ended_at_unix_ms: ended_at,
        attempted_jobs: attempted,
        successful_jobs: successful,
        failed_jobs: failed,
        uncertain_jobs: uncertain,
        emergency_stops: emergency,
        containment_actions: containment,
    };

    let id = PreparedUpgradeProbationClearanceIdV1(
        digest_prepared_clearance(
            handoff.id(),
            activation.id(),
            &aggregate,
            probation_policy_digest,
            threshold_policy_digest,
            trust_snapshot_digest,
            containment_state_digest,
            compromise_tracker_digest,
            current_clock.id(),
            current_basis.id(),
            clearance_expires_at_unix_ms,
        )
        .map_err(|error| vec![error])?,
    );

    Ok(PreparedUpgradeProbationClearanceV1 {
        id,
        handoff_id: handoff.id(),
        activation_permit_id: activation.id(),
        observation_set_digest,
        observer_profile_set_digest,
        verifier_set_digest,
        probation_policy_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        clock_envelope_id: current_clock.id(),
        operational_basis_id: current_basis.id(),
        observation_count: signed_observations.len(),
        machine_count: machine_ids.len(),
        failure_domain_count: failure_domains.len(),
        observation_started_at_unix_ms: started_at,
        observation_ended_at_unix_ms: ended_at,
        clearance_expires_at_unix_ms,
    })
}

pub fn authorize_clock_governed_upgrade_probation_clearance_v1(
    prepared: PreparedUpgradeProbationClearanceV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedUpgradeProbationClearanceV1, UpgradeProbationAuthorityError> {
    if ceremony.purpose() != CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_PURPOSE {
        return Err(UpgradeProbationAuthorityError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(UpgradeProbationAuthorityError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(UpgradeProbationAuthorityError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(UpgradeProbationAuthorityError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(UpgradeProbationAuthorityError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(UpgradeProbationAuthorityError::ClockEnvelopeMismatch);
    }

    let id = ClockGovernedUpgradeProbationClearanceIdV1(
        digest_authorized_clearance(prepared.id, ceremony.id(), ceremony.ceremony_digest())?,
    );
    Ok(ClockGovernedUpgradeProbationClearanceV1 {
        id,
        prepared_id: prepared.id,
        handoff_id: prepared.handoff_id,
        activation_permit_id: prepared.activation_permit_id,
        threshold_ceremony_id: ceremony.id(),
        observation_set_digest: prepared.observation_set_digest,
        observer_profile_set_digest: prepared.observer_profile_set_digest,
        verifier_set_digest: prepared.verifier_set_digest,
        probation_policy_digest: prepared.probation_policy_digest,
        clock_envelope_id: prepared.clock_envelope_id,
        operational_basis_id: prepared.operational_basis_id,
        observation_count: prepared.observation_count,
        machine_count: prepared.machine_count,
        failure_domain_count: prepared.failure_domain_count,
        observation_started_at_unix_ms: prepared.observation_started_at_unix_ms,
        observation_ended_at_unix_ms: prepared.observation_ended_at_unix_ms,
        clearance_expires_at_unix_ms: prepared.clearance_expires_at_unix_ms,
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), UpgradeProbationAuthorityError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(UpgradeProbationAuthorityError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: prior_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }
    let mut expected = prior_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(UpgradeProbationAuthorityError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(UpgradeProbationAuthorityError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn canonical_profile_map(
    profiles: &[UpgradeProbationObserverProfileV1],
) -> Result<
    BTreeMap<(SignatureAlgorithm, String), UpgradeProbationObserverProfileV1>,
    Vec<UpgradeProbationAuthorityError>,
> {
    let mut errors = Vec::new();
    let mut map = BTreeMap::new();
    for profile in profiles {
        if !profile.algorithm.is_canonical()
            || invalid_identifier(&profile.key_id)
            || invalid_identifier(&profile.machine_id)
            || invalid_identifier(&profile.failure_domain)
        {
            errors.push(UpgradeProbationAuthorityError::MissingObserverProfile(
                profile.key_id.clone(),
            ));
            continue;
        }
        let identity = (profile.algorithm.clone(), profile.key_id.clone());
        if map.insert(identity, profile.clone()).is_some() {
            errors.push(UpgradeProbationAuthorityError::DuplicateObserverProfile(
                profile.key_id.clone(),
            ));
        }
    }
    if errors.is_empty() { Ok(map) } else { Err(errors) }
}

fn validate_verification_providers(
    providers: &[&dyn UpgradeProbationObservationVerifierV1],
) -> Result<Vec<VerifierCommitment>, Vec<UpgradeProbationAuthorityError>> {
    let mut errors = Vec::new();
    let mut seen = BTreeSet::new();
    let mut commitments = Vec::new();
    for provider in providers {
        let id = provider.provider_id().to_string();
        let policy_digest = provider.verification_policy_digest();
        if invalid_identifier(&id) || policy_digest == Sha256Digest([0; 32]) {
            errors.push(UpgradeProbationAuthorityError::InvalidProvider(id));
            continue;
        }
        if !seen.insert(id.clone()) {
            errors.push(UpgradeProbationAuthorityError::DuplicateProvider(id));
            continue;
        }
        commitments.push(VerifierCommitment {
            provider_id: id,
            verification_policy_digest: policy_digest.to_hex(),
        });
    }
    commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    if errors.is_empty() { Ok(commitments) } else { Err(errors) }
}

fn requalify_observer_key(
    signed: &SignedUpgradeProbationObservationV1,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<UpgradeProbationAuthorityError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| record.algorithm == signed.algorithm && record.key_id == signed.key_id)
    else {
        violations.push(UpgradeProbationAuthorityError::ObserverUnknown(signed.key_id.clone()));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(UpgradeProbationAuthorityError::ObserverNotActive(signed.key_id.clone()));
    }
    if !record.usages.contains(&KeyUsage::UpgradeProbation) {
        violations.push(UpgradeProbationAuthorityError::ObserverUsageNotAllowed(
            signed.key_id.clone(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(UpgradeProbationAuthorityError::ObserverNotValidAcrossEnvelope {
            key_id: signed.key_id.clone(),
            reason,
        });
    }

    let not_before_ms = record.not_before_unix_s.checked_mul(1_000);
    let not_after_ms = record.not_after_unix_s.and_then(|value| value.checked_mul(1_000));
    if not_before_ms.is_none_or(|value| signed.observation.started_at_unix_ms < value)
        || record.not_after_unix_s.is_some()
            && not_after_ms.is_none_or(|value| signed.observation.ended_at_unix_ms >= value)
    {
        violations.push(UpgradeProbationAuthorityError::ObserverKeyInvalidForObservation(
            signed.key_id.clone(),
        ));
    }

    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            compromise.signer.algorithm == signed.algorithm
                && compromise.signer.key_id == signed.key_id
                && compromise.affected_usages.contains(&KeyUsage::UpgradeProbation)
        })
    {
        if clock
            .require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s)
            .is_err()
        {
            violations.push(UpgradeProbationAuthorityError::ObserverCompromisedAcrossEnvelope(
                signed.key_id.clone(),
            ));
        }
    }
}

fn observation_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = OBSERVATION_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn checked_add_or_record(
    current: u64,
    value: u64,
    violations: &mut Vec<UpgradeProbationAuthorityError>,
) -> u64 {
    match current.checked_add(value) {
        Some(total) => total,
        None => {
            violations.push(UpgradeProbationAuthorityError::InvalidJobAccounting);
            current
        }
    }
}

fn basis_points(part: u64, whole: u64) -> u64 {
    if whole == 0 {
        return 10_000;
    }
    part.saturating_mul(10_000) / whole
}

fn valid_probation_policy(policy: &UpgradeProbationPolicy) -> bool {
    policy.minimum_observation_duration_ms > 0
        && policy.minimum_distinct_machines > 0
        && policy.minimum_distinct_regions > 0
        && policy.minimum_successful_jobs > 0
        && policy.maximum_failure_basis_points <= 10_000
        && policy.maximum_uncertain_basis_points <= 10_000
        && policy.maximum_observations > 0
        && policy.maximum_observations <= MAX_PROBATION_OBSERVATIONS
        && policy.maximum_clearance_duration_ms > 0
}

fn valid_verification_policy(policy: &ProbationObservationVerificationPolicyV1) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_PROBATION_OBSERVATION_VERIFIERS
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

#[derive(Serialize)]
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

fn digest_probation_policy(
    policy: &UpgradeProbationPolicy,
) -> Result<Sha256Digest, UpgradeProbationAuthorityError> {
    if !valid_probation_policy(policy) {
        return Err(UpgradeProbationAuthorityError::InvalidProbationPolicy);
    }
    hash_serializable(
        PROBATION_POLICY_DOMAIN,
        &ProbationPolicyCommitment {
            minimum_observation_duration_ms: policy.minimum_observation_duration_ms,
            minimum_distinct_machines: policy.minimum_distinct_machines,
            minimum_distinct_regions: policy.minimum_distinct_regions,
            minimum_successful_jobs: policy.minimum_successful_jobs,
            maximum_failure_basis_points: policy.maximum_failure_basis_points,
            maximum_uncertain_basis_points: policy.maximum_uncertain_basis_points,
            maximum_emergency_stops: policy.maximum_emergency_stops,
            maximum_containment_actions: policy.maximum_containment_actions,
            maximum_observations: policy.maximum_observations,
            maximum_clearance_duration_ms: policy.maximum_clearance_duration_ms,
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

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, UpgradeProbationAuthorityError> {
    if !valid_threshold_policy(policy) {
        return Err(UpgradeProbationAuthorityError::InvalidThresholdPolicy);
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

#[derive(Serialize)]
struct PreparedClearanceCommitment<'a> {
    schema: &'static str,
    handoff_id: String,
    activation_permit_id: String,
    aggregate: &'a ProbationAggregateCommitment,
    probation_policy_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
    clearance_expires_at_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_clearance(
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: ClockGovernedUpgradeActivationPermitIdV1,
    aggregate: &ProbationAggregateCommitment,
    probation_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    clearance_expires_at_unix_ms: u64,
) -> Result<Sha256Digest, UpgradeProbationAuthorityError> {
    hash_serializable(
        PREPARED_CLEARANCE_DOMAIN,
        &PreparedClearanceCommitment {
            schema: PREPARED_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
            handoff_id: handoff_id.to_hex(),
            activation_permit_id: activation_permit_id.to_hex(),
            aggregate,
            probation_policy_digest: probation_policy_digest.to_hex(),
            threshold_policy_digest: threshold_policy_digest.to_hex(),
            trust_snapshot_digest: trust_snapshot_digest.to_hex(),
            containment_state_digest: containment_state_digest.to_hex(),
            compromise_tracker_digest: compromise_tracker_digest.to_hex(),
            clock_envelope_id: clock_envelope_id.to_hex(),
            operational_basis_id: operational_basis_id.to_hex(),
            clearance_expires_at_unix_ms,
        },
    )
}

#[derive(Serialize)]
struct AuthorizedClearanceCommitment {
    schema: &'static str,
    prepared_id: String,
    threshold_ceremony_id: String,
    threshold_ceremony_digest: String,
}

fn digest_authorized_clearance(
    prepared_id: PreparedUpgradeProbationClearanceIdV1,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    threshold_ceremony_digest: Sha256Digest,
) -> Result<Sha256Digest, UpgradeProbationAuthorityError> {
    hash_serializable(
        AUTHORIZED_CLEARANCE_DOMAIN,
        &AuthorizedClearanceCommitment {
            schema: CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
            prepared_id: prepared_id.to_hex(),
            threshold_ceremony_id: threshold_ceremony_id.to_hex(),
            threshold_ceremony_digest: threshold_ceremony_digest.to_hex(),
        },
    )
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, UpgradeProbationAuthorityError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeProbationAuthorityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
