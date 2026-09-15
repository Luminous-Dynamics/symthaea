// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe hardware reauthorization that preserves global upgrade-predecessor provenance.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{FabricationContainmentState, digest_containment_state};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::hardware_reauthorization::{
    HardwareReauthorizationPolicy, HardwareReauthorizationStatement, MAX_HARDWARE_ID_BYTES,
    MAX_HARDWARE_SIGNATURE_BYTES, SignedHardwareReauthorization,
    digest_hardware_reauthorization_statement,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_lineage_bound_probation_telemetry::{
    LineageBoundTelemetryProbationClearanceIdV1, LineageBoundTelemetryProbationClearanceV1,
};
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundClockGovernedUpgradeHandoffIdV1, LineageBoundClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_lineage_bound_upgrade_probation::{
    LineageBoundUpgradeProbationClearanceIdV1, LineageBoundUpgradeProbationClearanceV1,
};
use symthaea_fabrication_upgrade_hardware_authority::{
    ExactHardwareReauthorizationVerificationPolicyV1, ExactHardwareReauthorizationVerifierV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const LINEAGE_BOUND_HARDWARE_REAUTHORIZATION_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-hardware-reauthorization.v1";
pub const MAX_LINEAGE_BOUND_HARDWARE_CLOCK_HOPS: usize = 4096;
pub const MAX_LINEAGE_BOUND_HARDWARE_VERIFIERS: usize = 8;

const SIGNED_HARDWARE_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-hardware-reauthorization-evidence.v1\0";
const HARDWARE_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-hardware-reauthorization-policy.v1\0";
const VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.hardware-reauthorization-verifier-set.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-hardware-clock-lineage.v1\0";
const CAPABILITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-hardware-reauthorization.v1\0";
const HARDWARE_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.hardware-reauthorization-signature.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundHardwareReauthorizationIdV1(Sha256Digest);
impl LineageBoundHardwareReauthorizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundHardwareReauthorizationV1 {
    id: LineageBoundHardwareReauthorizationIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    probation_clearance_id: LineageBoundUpgradeProbationClearanceIdV1,
    telemetry_clearance_id: LineageBoundTelemetryProbationClearanceIdV1,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    governance_view_digest: Sha256Digest,
    registry_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    statement: HardwareReauthorizationStatement,
    statement_digest: Sha256Digest,
    signed_evidence_digest: Sha256Digest,
    hardware_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    clearance_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
}

impl LineageBoundHardwareReauthorizationV1 {
    pub fn id(&self) -> LineageBoundHardwareReauthorizationIdV1 { self.id }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
    pub fn probation_clearance_id(&self) -> LineageBoundUpgradeProbationClearanceIdV1 { self.probation_clearance_id }
    pub fn telemetry_clearance_id(&self) -> LineageBoundTelemetryProbationClearanceIdV1 { self.telemetry_clearance_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_digest }
    pub fn governance_view_digest(&self) -> Sha256Digest { self.governance_view_digest }
    pub fn registry_head_digest(&self) -> Sha256Digest { self.registry_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn statement(&self) -> &HardwareReauthorizationStatement { &self.statement }
    pub fn statement_digest(&self) -> Sha256Digest { self.statement_digest }
    pub fn signed_evidence_digest(&self) -> Sha256Digest { self.signed_evidence_digest }
    pub fn hardware_policy_digest(&self) -> Sha256Digest { self.hardware_policy_digest }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn containment_state_digest(&self) -> Sha256Digest { self.containment_state_digest }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest { self.compromise_tracker_digest }
    pub fn verifier_set_digest(&self) -> Sha256Digest { self.verifier_set_digest }
    pub fn clearance_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.clearance_operational_basis_id }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.current_operational_basis_id }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.current_clock_envelope_id }
    pub fn clock_lineage_digest(&self) -> Sha256Digest { self.clock_lineage_digest }
    pub fn clock_hop_count(&self) -> usize { self.clock_hop_count }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundHardwareError {
    ProbationLineageMismatch,
    TelemetryLineageMismatch,
    ClearanceBasisMismatch,
    ClearanceEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage { hop: usize, expected_predecessor: String, actual_predecessor: Option<String> },
    Clock(ClockGovernanceTimeError),
    ClearanceExpired,
    InvalidPolicy,
    InvalidVerificationPolicy,
    InsufficientVerificationProviders { actual: usize, required: usize },
    TooManyVerificationProviders { actual: usize, maximum: usize },
    InvalidProvider(String),
    DuplicateProvider(String),
    InvalidStatement(String),
    HandoffDigestMismatch,
    SuccessorMismatch,
    StatementBeforeProbation,
    StatementMayBeFuture,
    StatementExpired,
    StatementTooOld,
    StatementWindowTooLong,
    StatementOutlivesHandoff,
    StatementDigestMismatch,
    InvalidSigner,
    TrustSnapshotInvalid(String),
    TrustSnapshotMismatch,
    TrustSnapshotPostdatesStatement,
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    ContainmentStateMismatch,
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerInvalidForStatementWindow(String),
    SignerCompromisedAcrossEnvelope(String),
    SignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct VerifierCommitment { provider_id: String, verification_policy_digest: String }
#[derive(Debug, Clone, Serialize)]
struct HardwarePolicyCommitment { maximum_authorization_duration_s: u64, maximum_statement_age_s: u64 }
#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment { clearance_basis_id: String, bridge_basis_ids: Vec<String>, current_basis_id: String }
#[derive(Debug, Clone, Serialize)]
struct CapabilityCommitment {
    schema: &'static str,
    lineage_handoff_id: String,
    probation_clearance_id: String,
    telemetry_clearance_id: String,
    predecessor_root_digest: String,
    current_head_digest: String,
    governance_view_digest: String,
    registry_head_digest: String,
    predecessor_finalization_sequence: u64,
    handoff_plan_digest: String,
    statement_digest: String,
    signed_evidence_digest: String,
    hardware_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    verifier_set_digest: String,
    clearance_operational_basis_id: String,
    current_operational_basis_id: String,
    current_clock_envelope_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_lineage_bound_hardware_reauthorization_v1(
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
    probation: &LineageBoundUpgradeProbationClearanceV1,
    telemetry: &LineageBoundTelemetryProbationClearanceV1,
    signed: &SignedHardwareReauthorization,
    hardware_policy: &HardwareReauthorizationPolicy,
    clearance_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    verification_policy: &ExactHardwareReauthorizationVerificationPolicyV1,
    verification_providers: &[&dyn ExactHardwareReauthorizationVerifierV1],
) -> Result<LineageBoundHardwareReauthorizationV1, Vec<LineageBoundHardwareError>> {
    let mut violations = Vec::new();

    if probation.lineage_handoff_id() != handoff.id()
        || probation.handoff_plan_digest() != handoff.plan_digest()
        || probation.predecessor_root_digest() != handoff.predecessor_root_id().as_digest()
        || probation.current_head_digest() != handoff.current_head_id().as_digest()
        || probation.governance_view_digest() != handoff.governance_view_id().as_digest()
        || probation.registry_head_digest() != handoff.registry_head_id().as_digest()
        || probation.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
    {
        violations.push(LineageBoundHardwareError::ProbationLineageMismatch);
    }
    if telemetry.clearance_id() != probation.id()
        || telemetry.lineage_handoff_digest() != handoff.id().as_digest()
        || telemetry.predecessor_root_digest() != probation.predecessor_root_digest()
        || telemetry.current_head_digest() != probation.current_head_digest()
        || telemetry.governance_view_digest() != probation.governance_view_digest()
        || telemetry.registry_head_digest() != probation.registry_head_digest()
        || telemetry.predecessor_finalization_sequence() != probation.predecessor_finalization_sequence()
        || telemetry.handoff_plan_digest() != handoff.plan_digest()
    {
        violations.push(LineageBoundHardwareError::TelemetryLineageMismatch);
    }

    if clearance_basis.id() != probation.operational_basis_id() {
        violations.push(LineageBoundHardwareError::ClearanceBasisMismatch);
    }
    let clearance_clock = match derive_clock_governance_evaluation_envelope_v1(clearance_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundHardwareError::Clock(error)); return Err(violations); }
    };
    if clearance_clock.id() != probation.clock_envelope_id() {
        violations.push(LineageBoundHardwareError::ClearanceEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_LINEAGE_BOUND_HARDWARE_CLOCK_HOPS {
        violations.push(LineageBoundHardwareError::TooManyClockHops { actual: clock_bridge.len(), maximum: MAX_LINEAGE_BOUND_HARDWARE_CLOCK_HOPS });
    } else if let Err(error) = verify_clock_lineage(clearance_basis.id(), clock_bridge, current_basis) {
        violations.push(error);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundHardwareError::Clock(error)); return Err(violations); }
    };
    if current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms() {
        violations.push(LineageBoundHardwareError::ClearanceExpired);
    }

    if hardware_policy.maximum_authorization_duration_s == 0 || hardware_policy.maximum_statement_age_s == 0 {
        violations.push(LineageBoundHardwareError::InvalidPolicy);
    }
    if !valid_verification_policy(verification_policy) { violations.push(LineageBoundHardwareError::InvalidVerificationPolicy); }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(LineageBoundHardwareError::InsufficientVerificationProviders { actual: verification_providers.len(), required: verification_policy.minimum_distinct_providers });
    }
    if verification_providers.len() > verification_policy.maximum_providers {
        violations.push(LineageBoundHardwareError::TooManyVerificationProviders { actual: verification_providers.len(), maximum: verification_policy.maximum_providers });
    }

    if let Err(error) = signed.statement.validate() { violations.push(LineageBoundHardwareError::InvalidStatement(format!("{error:?}"))); }
    if signed.statement.handoff_digest != handoff.plan_digest() { violations.push(LineageBoundHardwareError::HandoffDigestMismatch); }
    if signed.statement.successor_source_tree_digest != handoff.plan().successor.source_tree_digest
        || signed.statement.successor_executable_digest != handoff.plan().successor.executable_digest
    { violations.push(LineageBoundHardwareError::SuccessorMismatch); }

    let issued_ms = match signed.statement.issued_at_unix_s.checked_mul(1_000) { Some(v) => v, None => { violations.push(LineageBoundHardwareError::TimeScaleOverflow); 0 } };
    let expires_ms = match signed.statement.expires_at_unix_s.checked_mul(1_000) { Some(v) => v, None => { violations.push(LineageBoundHardwareError::TimeScaleOverflow); 0 } };
    if issued_ms < clearance_clock.upper_unix_ms() { violations.push(LineageBoundHardwareError::StatementBeforeProbation); }
    if issued_ms > current_clock.lower_unix_ms() { violations.push(LineageBoundHardwareError::StatementMayBeFuture); }
    if expires_ms <= current_clock.upper_unix_ms() { violations.push(LineageBoundHardwareError::StatementExpired); }
    if signed.statement.expires_at_unix_s.checked_sub(signed.statement.issued_at_unix_s).is_none_or(|duration| duration > hardware_policy.maximum_authorization_duration_s) {
        violations.push(LineageBoundHardwareError::StatementWindowTooLong);
    }
    let maximum_age_ms = match hardware_policy.maximum_statement_age_s.checked_mul(1_000) { Some(v) => v, None => { violations.push(LineageBoundHardwareError::TimeScaleOverflow); 0 } };
    if current_clock.upper_unix_ms().checked_sub(issued_ms).is_none_or(|age| age > maximum_age_ms) { violations.push(LineageBoundHardwareError::StatementTooOld); }
    if expires_ms > handoff.plan().finalization_deadline_unix_ms { violations.push(LineageBoundHardwareError::StatementOutlivesHandoff); }

    let statement_digest = match digest_hardware_reauthorization_statement(&signed.statement) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundHardwareError::InvalidStatement(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if statement_digest != signed.statement_digest { violations.push(LineageBoundHardwareError::StatementDigestMismatch); }
    if !signed.signature.algorithm.is_canonical() || invalid_identifier(&signed.signature.key_id) || signed.signature.signature.is_empty() || signed.signature.signature.len() > MAX_HARDWARE_SIGNATURE_BYTES {
        violations.push(LineageBoundHardwareError::InvalidSigner);
    }

    if let Err(error) = trust_snapshot.validate() { violations.push(LineageBoundHardwareError::TrustSnapshotInvalid(format!("{error:?}"))); }
    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) { Ok(value) => value, Err(error) => { violations.push(LineageBoundHardwareError::TrustSnapshotInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    if trust_snapshot_digest != probation.trust_snapshot_digest() { violations.push(LineageBoundHardwareError::TrustSnapshotMismatch); }
    if trust_snapshot.issued_at_unix_s > signed.statement.issued_at_unix_s { violations.push(LineageBoundHardwareError::TrustSnapshotPostdatesStatement); }
    if let Err(reason) = current_clock.require_valid_across_seconds_window(trust_snapshot.issued_at_unix_s, trust_snapshot.expires_at_unix_s) { violations.push(LineageBoundHardwareError::TrustSnapshotNotValidAcrossEnvelope(reason)); }

    if let Err(error) = containment_state.validate() { violations.push(LineageBoundHardwareError::ContainmentStateInvalid(format!("{error:?}"))); }
    let containment_state_digest = match digest_containment_state(containment_state) { Ok(value) => value, Err(error) => { violations.push(LineageBoundHardwareError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    let compromise_tracker_digest = match digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker) { Ok(value) => value, Err(error) => { violations.push(LineageBoundHardwareError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    if containment_state_digest != probation.containment_state_digest()
        || compromise_tracker_digest != probation.compromise_tracker_digest()
        || containment_state_digest != telemetry.containment_state_digest()
        || compromise_tracker_digest != telemetry.compromise_tracker_digest()
    { violations.push(LineageBoundHardwareError::ContainmentStateMismatch); }

    match trust_snapshot.keys.iter().find(|record| record.algorithm == signed.signature.algorithm && record.key_id == signed.signature.key_id) {
        None => violations.push(LineageBoundHardwareError::SignerUnknown(signed.signature.key_id.clone())),
        Some(record) => {
            if record.status != KeyLifecycleStatus::Active { violations.push(LineageBoundHardwareError::SignerNotActive(signed.signature.key_id.clone())); }
            if !record.usages.contains(&KeyUsage::HardwareReauthorization) { violations.push(LineageBoundHardwareError::SignerUsageNotAllowed(signed.signature.key_id.clone())); }
            if record.not_before_unix_s > signed.statement.issued_at_unix_s || record.not_after_unix_s.is_some_and(|end| end < signed.statement.expires_at_unix_s) {
                violations.push(LineageBoundHardwareError::SignerInvalidForStatementWindow(signed.signature.key_id.clone()));
            }
        }
    }
    for compromise in containment_state.signer_compromise_tracker.records().iter().filter(|c| c.signer.algorithm == signed.signature.algorithm && c.signer.key_id == signed.signature.key_id && c.affected_usages.contains(&KeyUsage::HardwareReauthorization)) {
        if current_clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s).is_err() {
            violations.push(LineageBoundHardwareError::SignerCompromisedAcrossEnvelope(signed.signature.key_id.clone()));
        }
    }

    let verifier_commitments = match validate_verification_providers(verification_providers) { Ok(value) => value, Err(mut errors) => { violations.append(&mut errors); Vec::new() } };
    let message = hardware_signature_message(statement_digest);
    for provider in verification_providers {
        match provider.verify_hardware_reauthorization_signature(&signed.signature.algorithm, &signed.signature.key_id, &message, &signed.signature.signature) {
            Ok(true) => {}
            Ok(false) => violations.push(LineageBoundHardwareError::SignatureRejected { provider: provider.provider_id().to_string(), key_id: signed.signature.key_id.clone() }),
            Err(reason) => violations.push(LineageBoundHardwareError::VerificationProviderError { provider: provider.provider_id().to_string(), reason }),
        }
    }
    if !violations.is_empty() { return Err(violations); }

    let signed_evidence_digest = hash_serializable(SIGNED_HARDWARE_EVIDENCE_DOMAIN, signed).map_err(|e| vec![e])?;
    let hardware_policy_digest = hash_serializable(HARDWARE_POLICY_DOMAIN, &HardwarePolicyCommitment { maximum_authorization_duration_s: hardware_policy.maximum_authorization_duration_s, maximum_statement_age_s: hardware_policy.maximum_statement_age_s }).map_err(|e| vec![e])?;
    let verifier_set_digest = hash_serializable(VERIFIER_SET_DOMAIN, &verifier_commitments).map_err(|e| vec![e])?;
    let lineage = ClockLineageCommitment { clearance_basis_id: clearance_basis.id().to_hex(), bridge_basis_ids: clock_bridge.iter().map(|b| b.id().to_hex()).collect(), current_basis_id: current_basis.id().to_hex() };
    let clock_lineage_digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &lineage).map_err(|e| vec![e])?;
    let clock_hop_count = if current_basis.id() == clearance_basis.id() { 0 } else { clock_bridge.len() + 1 };
    let commitment = CapabilityCommitment {
        schema: LINEAGE_BOUND_HARDWARE_REAUTHORIZATION_SCHEMA,
        lineage_handoff_id: handoff.id().to_hex(), probation_clearance_id: probation.id().to_hex(), telemetry_clearance_id: telemetry.id().to_hex(),
        predecessor_root_digest: probation.predecessor_root_digest().to_hex(), current_head_digest: probation.current_head_digest().to_hex(), governance_view_digest: probation.governance_view_digest().to_hex(), registry_head_digest: probation.registry_head_digest().to_hex(), predecessor_finalization_sequence: probation.predecessor_finalization_sequence(), handoff_plan_digest: handoff.plan_digest().to_hex(),
        statement_digest: statement_digest.to_hex(), signed_evidence_digest: signed_evidence_digest.to_hex(), hardware_policy_digest: hardware_policy_digest.to_hex(), trust_snapshot_digest: trust_snapshot_digest.to_hex(), containment_state_digest: containment_state_digest.to_hex(), compromise_tracker_digest: compromise_tracker_digest.to_hex(), verifier_set_digest: verifier_set_digest.to_hex(), clearance_operational_basis_id: clearance_basis.id().to_hex(), current_operational_basis_id: current_basis.id().to_hex(), current_clock_envelope_id: current_clock.id().to_hex(), clock_lineage_digest: clock_lineage_digest.to_hex(), clock_hop_count,
    };
    let id = LineageBoundHardwareReauthorizationIdV1(hash_serializable(CAPABILITY_DOMAIN, &commitment).map_err(|e| vec![e])?);
    Ok(LineageBoundHardwareReauthorizationV1 { id, lineage_handoff_id: handoff.id(), probation_clearance_id: probation.id(), telemetry_clearance_id: telemetry.id(), predecessor_root_digest: probation.predecessor_root_digest(), current_head_digest: probation.current_head_digest(), governance_view_digest: probation.governance_view_digest(), registry_head_digest: probation.registry_head_digest(), predecessor_finalization_sequence: probation.predecessor_finalization_sequence(), handoff_plan_digest: handoff.plan_digest(), statement: signed.statement.clone(), statement_digest, signed_evidence_digest, hardware_policy_digest, trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, verifier_set_digest, clearance_operational_basis_id: clearance_basis.id(), current_operational_basis_id: current_basis.id(), current_clock_envelope_id: current_clock.id(), clock_lineage_digest, clock_hop_count })
}

fn verify_clock_lineage(prior: OperationalClockBasisIdV1, bridge: &[OperationalClockBasisV1], current: &OperationalClockBasisV1) -> Result<(), LineageBoundHardwareError> {
    if current.id() == prior { if bridge.is_empty() { return Ok(()); } return Err(LineageBoundHardwareError::BrokenClockLineage { hop: 1, expected_predecessor: prior.to_hex(), actual_predecessor: bridge[0].predecessor_operational_basis_id().map(|v| v.to_hex()) }); }
    let mut expected = prior;
    for (index, basis) in bridge.iter().enumerate() { let actual = basis.predecessor_operational_basis_id(); if actual != Some(expected) { return Err(LineageBoundHardwareError::BrokenClockLineage { hop: index + 1, expected_predecessor: expected.to_hex(), actual_predecessor: actual.map(|v| v.to_hex()) }); } expected = basis.id(); }
    let actual = current.predecessor_operational_basis_id(); if actual != Some(expected) { return Err(LineageBoundHardwareError::BrokenClockLineage { hop: bridge.len() + 1, expected_predecessor: expected.to_hex(), actual_predecessor: actual.map(|v| v.to_hex()) }); } Ok(())
}
fn valid_verification_policy(policy: &ExactHardwareReauthorizationVerificationPolicyV1) -> bool { policy.minimum_distinct_providers > 0 && policy.maximum_providers > 0 && policy.minimum_distinct_providers <= policy.maximum_providers && policy.maximum_providers <= MAX_LINEAGE_BOUND_HARDWARE_VERIFIERS }
fn validate_verification_providers(providers: &[&dyn ExactHardwareReauthorizationVerifierV1]) -> Result<Vec<VerifierCommitment>, Vec<LineageBoundHardwareError>> { let mut errors = Vec::new(); let mut seen = BTreeSet::new(); let mut commitments = Vec::new(); for provider in providers { let id = provider.provider_id().to_string(); let digest = provider.verification_policy_digest(); if invalid_identifier(&id) || digest == Sha256Digest([0; 32]) { errors.push(LineageBoundHardwareError::InvalidProvider(id)); continue; } if !seen.insert(id.clone()) { errors.push(LineageBoundHardwareError::DuplicateProvider(id)); continue; } commitments.push(VerifierCommitment { provider_id: id, verification_policy_digest: digest.to_hex() }); } commitments.sort_by(|a,b| a.provider_id.cmp(&b.provider_id)); if errors.is_empty() { Ok(commitments) } else { Err(errors) } }
fn hardware_signature_message(digest: Sha256Digest) -> Vec<u8> { let mut message = HARDWARE_SIGNATURE_DOMAIN.to_vec(); message.extend_from_slice(&digest.0); message }
fn invalid_identifier(value: &str) -> bool { value.trim().is_empty() || value != value.trim() || value.len() > MAX_HARDWARE_ID_BYTES || value.chars().any(char::is_control) }
fn hash_serializable<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<Sha256Digest, LineageBoundHardwareError> { let bytes = serde_json::to_vec(value).map_err(|e| LineageBoundHardwareError::Encoding(e.to_string()))?; let mut hasher = Sha256::new(); hasher.update(domain); hasher.update(&bytes); Ok(hasher.finalize()) }
