// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe machine hardware reauthorization for hardened fabrication upgrades.
//!
//! The portable `SignedHardwareReauthorization` format remains unchanged. This bridge replaces its
//! scalar live-authority verifier with a theorem over the clock-governed handoff, authenticated and
//! telemetry-bound probation clearance, recursive operational clock lineage, real trust records,
//! persistent compromise containment, and exact raw signature bytes.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::hardware_reauthorization::{
    HardwareReauthorizationPolicy, HardwareReauthorizationStatement,
    MAX_HARDWARE_ID_BYTES, MAX_HARDWARE_SIGNATURE_BYTES, SignedHardwareReauthorization,
    digest_hardware_reauthorization_statement,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_probation_authority::{
    ClockGovernedUpgradeProbationClearanceIdV1, ClockGovernedUpgradeProbationClearanceV1,
};
use symthaea_fabrication_upgrade_probation_telemetry::{
    TelemetryBoundUpgradeProbationClearanceIdV1, TelemetryBoundUpgradeProbationClearanceV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_HARDWARE_REAUTHORIZATION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-hardware-reauthorization.v1";
pub const MAX_HARDWARE_REAUTHORIZATION_CLOCK_HOPS: usize = 4096;
pub const MAX_HARDWARE_REAUTHORIZATION_VERIFIERS: usize = 8;

const SIGNED_HARDWARE_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.signed-hardware-reauthorization-evidence.v1\0";
const HARDWARE_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-hardware-reauthorization-policy.v1\0";
const HARDWARE_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.hardware-reauthorization-verifier-set.v1\0";
const HARDWARE_AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-hardware-reauthorization.v1\0";
const HARDWARE_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.hardware-reauthorization-signature.v1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactHardwareReauthorizationVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactHardwareReauthorizationVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: MAX_HARDWARE_REAUTHORIZATION_VERIFIERS,
        }
    }
}

pub trait ExactHardwareReauthorizationVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;
    fn verify_hardware_reauthorization_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedHardwareReauthorizationIdV1(Sha256Digest);

impl ClockGovernedHardwareReauthorizationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedHardwareReauthorizationV1 {
    id: ClockGovernedHardwareReauthorizationIdV1,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    probation_clearance_id: ClockGovernedUpgradeProbationClearanceIdV1,
    telemetry_bound_clearance_id: TelemetryBoundUpgradeProbationClearanceIdV1,
    statement: HardwareReauthorizationStatement,
    statement_digest: Sha256Digest,
    signed_evidence_digest: Sha256Digest,
    hardware_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    clock_hop_count: usize,
}

impl ClockGovernedHardwareReauthorizationV1 {
    pub fn id(&self) -> ClockGovernedHardwareReauthorizationIdV1 {
        self.id
    }
    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }
    pub fn probation_clearance_id(&self) -> ClockGovernedUpgradeProbationClearanceIdV1 {
        self.probation_clearance_id
    }
    pub fn telemetry_bound_clearance_id(&self) -> TelemetryBoundUpgradeProbationClearanceIdV1 {
        self.telemetry_bound_clearance_id
    }
    pub fn statement(&self) -> &HardwareReauthorizationStatement {
        &self.statement
    }
    pub fn statement_digest(&self) -> Sha256Digest {
        self.statement_digest
    }
    pub fn signed_evidence_digest(&self) -> Sha256Digest {
        self.signed_evidence_digest
    }
    pub fn hardware_policy_digest(&self) -> Sha256Digest {
        self.hardware_policy_digest
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
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.current_clock_envelope_id
    }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.current_operational_basis_id
    }
    pub fn clock_hop_count(&self) -> usize {
        self.clock_hop_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedHardwareReauthorizationError {
    HandoffMismatch,
    ProbationTelemetryMismatch,
    ClearanceBasisMismatch,
    ClearanceEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
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
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
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
struct VerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct HardwarePolicyCommitment {
    maximum_authorization_duration_s: u64,
    maximum_statement_age_s: u64,
}

#[derive(Debug, Clone, Serialize)]
struct HardwareAuthorityCommitment {
    schema: &'static str,
    handoff_id: String,
    probation_clearance_id: String,
    telemetry_bound_clearance_id: String,
    statement_digest: String,
    signed_evidence_digest: String,
    hardware_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    verifier_set_digest: String,
    current_clock_envelope_id: String,
    current_operational_basis_id: String,
    clock_hop_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_clock_governed_hardware_reauthorization_v1(
    handoff: &ClockGovernedUpgradeHandoffV1,
    probation: &ClockGovernedUpgradeProbationClearanceV1,
    telemetry_bound_probation: &TelemetryBoundUpgradeProbationClearanceV1,
    signed: &SignedHardwareReauthorization,
    hardware_policy: &HardwareReauthorizationPolicy,
    clearance_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    verification_policy: &ExactHardwareReauthorizationVerificationPolicyV1,
    verification_providers: &[&dyn ExactHardwareReauthorizationVerifierV1],
) -> Result<ClockGovernedHardwareReauthorizationV1, Vec<ClockGovernedHardwareReauthorizationError>> {
    let mut violations = Vec::new();

    if probation.handoff_id() != handoff.id() {
        violations.push(ClockGovernedHardwareReauthorizationError::HandoffMismatch);
    }
    if telemetry_bound_probation.clearance_id() != probation.id() {
        violations.push(ClockGovernedHardwareReauthorizationError::ProbationTelemetryMismatch);
    }
    if clearance_basis.id() != probation.operational_basis_id() {
        violations.push(ClockGovernedHardwareReauthorizationError::ClearanceBasisMismatch);
    }
    let clearance_clock = match derive_clock_governance_evaluation_envelope_v1(clearance_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedHardwareReauthorizationError::Clock(error));
            return Err(violations);
        }
    };
    if clearance_clock.id() != probation.clock_envelope_id() {
        violations.push(ClockGovernedHardwareReauthorizationError::ClearanceEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_HARDWARE_REAUTHORIZATION_CLOCK_HOPS {
        violations.push(ClockGovernedHardwareReauthorizationError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_HARDWARE_REAUTHORIZATION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(clearance_basis.id(), clock_bridge, current_basis) {
        violations.push(error);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedHardwareReauthorizationError::Clock(error));
            return Err(violations);
        }
    };
    if current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms() {
        violations.push(ClockGovernedHardwareReauthorizationError::ClearanceExpired);
    }

    if hardware_policy.maximum_authorization_duration_s == 0
        || hardware_policy.maximum_statement_age_s == 0
    {
        violations.push(ClockGovernedHardwareReauthorizationError::InvalidPolicy);
    }
    if !valid_verification_policy(verification_policy) {
        violations.push(ClockGovernedHardwareReauthorizationError::InvalidVerificationPolicy);
    }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(
            ClockGovernedHardwareReauthorizationError::InsufficientVerificationProviders {
                actual: verification_providers.len(),
                required: verification_policy.minimum_distinct_providers,
            },
        );
    }
    if verification_providers.len() > verification_policy.maximum_providers {
        violations.push(ClockGovernedHardwareReauthorizationError::TooManyVerificationProviders {
            actual: verification_providers.len(),
            maximum: verification_policy.maximum_providers,
        });
    }

    if let Err(error) = signed.statement.validate() {
        violations.push(ClockGovernedHardwareReauthorizationError::InvalidStatement(format!(
            "{error:?}"
        )));
    }
    if signed.statement.handoff_digest != handoff.plan_digest() {
        violations.push(ClockGovernedHardwareReauthorizationError::HandoffDigestMismatch);
    }
    if signed.statement.successor_source_tree_digest != handoff.plan().successor.source_tree_digest
        || signed.statement.successor_executable_digest
            != handoff.plan().successor.executable_digest
    {
        violations.push(ClockGovernedHardwareReauthorizationError::SuccessorMismatch);
    }

    let issued_ms = match signed.statement.issued_at_unix_s.checked_mul(1_000) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedHardwareReauthorizationError::TimeScaleOverflow);
            0
        }
    };
    let expires_ms = match signed.statement.expires_at_unix_s.checked_mul(1_000) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedHardwareReauthorizationError::TimeScaleOverflow);
            0
        }
    };
    if issued_ms < clearance_clock.upper_unix_ms() {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementBeforeProbation);
    }
    if issued_ms > current_clock.lower_unix_ms() {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementMayBeFuture);
    }
    if expires_ms <= current_clock.upper_unix_ms() {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementExpired);
    }
    if signed
        .statement
        .expires_at_unix_s
        .checked_sub(signed.statement.issued_at_unix_s)
        .is_none_or(|duration| duration > hardware_policy.maximum_authorization_duration_s)
    {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementWindowTooLong);
    }
    let maximum_age_ms = match hardware_policy.maximum_statement_age_s.checked_mul(1_000) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedHardwareReauthorizationError::TimeScaleOverflow);
            0
        }
    };
    if current_clock
        .upper_unix_ms()
        .checked_sub(issued_ms)
        .is_none_or(|age| age > maximum_age_ms)
    {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementTooOld);
    }
    if expires_ms > handoff.plan().finalization_deadline_unix_ms {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementOutlivesHandoff);
    }

    let statement_digest = match digest_hardware_reauthorization_statement(&signed.statement) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedHardwareReauthorizationError::InvalidStatement(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if statement_digest != signed.statement_digest {
        violations.push(ClockGovernedHardwareReauthorizationError::StatementDigestMismatch);
    }
    if !signed.signature.algorithm.is_canonical()
        || invalid_identifier(&signed.signature.key_id)
        || signed.signature.signature.is_empty()
        || signed.signature.signature.len() > MAX_HARDWARE_SIGNATURE_BYTES
    {
        violations.push(ClockGovernedHardwareReauthorizationError::InvalidSigner);
    }

    if let Err(error) = trust_snapshot.validate() {
        violations.push(ClockGovernedHardwareReauthorizationError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(reason) = current_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            ClockGovernedHardwareReauthorizationError::TrustSnapshotNotValidAcrossEnvelope(reason),
        );
    }
    let signer_record = trust_snapshot.keys.iter().find(|record| {
        record.algorithm == signed.signature.algorithm && record.key_id == signed.signature.key_id
    });
    match signer_record {
        None => violations.push(ClockGovernedHardwareReauthorizationError::SignerUnknown(
            signed.signature.key_id.clone(),
        )),
        Some(record) => {
            if record.status != KeyLifecycleStatus::Active {
                violations.push(ClockGovernedHardwareReauthorizationError::SignerNotActive(
                    signed.signature.key_id.clone(),
                ));
            }
            if !record.usages.contains(&KeyUsage::HardwareReauthorization) {
                violations.push(ClockGovernedHardwareReauthorizationError::SignerUsageNotAllowed(
                    signed.signature.key_id.clone(),
                ));
            }
            if record.not_before_unix_s > signed.statement.issued_at_unix_s
                || record
                    .not_after_unix_s
                    .is_some_and(|not_after| not_after < signed.statement.expires_at_unix_s)
            {
                violations.push(
                    ClockGovernedHardwareReauthorizationError::SignerInvalidForStatementWindow(
                        signed.signature.key_id.clone(),
                    ),
                );
            }
        }
    }

    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            compromise.signer.algorithm == signed.signature.algorithm
                && compromise.signer.key_id == signed.signature.key_id
                && compromise
                    .affected_usages
                    .contains(&KeyUsage::HardwareReauthorization)
        })
    {
        if current_clock
            .require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s)
            .is_err()
        {
            violations.push(
                ClockGovernedHardwareReauthorizationError::SignerCompromisedAcrossEnvelope(
                    signed.signature.key_id.clone(),
                ),
            );
        }
    }

    let verifier_commitments = match validate_verification_providers(verification_providers) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            Vec::new()
        }
    };
    let message = hardware_signature_message(statement_digest);
    for provider in verification_providers {
        match provider.verify_hardware_reauthorization_signature(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            &message,
            &signed.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => violations.push(ClockGovernedHardwareReauthorizationError::SignatureRejected {
                provider: provider.provider_id().to_string(),
                key_id: signed.signature.key_id.clone(),
            }),
            Err(reason) => violations.push(
                ClockGovernedHardwareReauthorizationError::VerificationProviderError {
                    provider: provider.provider_id().to_string(),
                    reason,
                },
            ),
        }
    }

    if let Err(error) = containment_state.validate() {
        violations.push(ClockGovernedHardwareReauthorizationError::InvalidStatement(format!(
            "containment state: {error:?}"
        )));
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let signed_evidence_digest = hash_serializable(SIGNED_HARDWARE_EVIDENCE_DOMAIN, signed)
        .map_err(|error| vec![error])?;
    let hardware_policy_digest = digest_hardware_policy(hardware_policy).map_err(|error| vec![error])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![ClockGovernedHardwareReauthorizationError::TrustSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ClockGovernedHardwareReauthorizationError::InvalidStatement(format!(
            "containment digest: {error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ClockGovernedHardwareReauthorizationError::InvalidStatement(format!(
                    "compromise digest: {error:?}"
                ))]
            },
        )?;
    let verifier_set_digest =
        hash_serializable(HARDWARE_VERIFIER_SET_DOMAIN, &verifier_commitments)
            .map_err(|error| vec![error])?;

    let commitment = HardwareAuthorityCommitment {
        schema: CLOCK_GOVERNED_HARDWARE_REAUTHORIZATION_SCHEMA,
        handoff_id: handoff.id().to_hex(),
        probation_clearance_id: probation.id().to_hex(),
        telemetry_bound_clearance_id: telemetry_bound_probation.id().to_hex(),
        statement_digest: statement_digest.to_hex(),
        signed_evidence_digest: signed_evidence_digest.to_hex(),
        hardware_policy_digest: hardware_policy_digest.to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        verifier_set_digest: verifier_set_digest.to_hex(),
        current_clock_envelope_id: current_clock.id().to_hex(),
        current_operational_basis_id: current_basis.id().to_hex(),
        clock_hop_count: clock_bridge.len() + usize::from(current_basis.id() != clearance_basis.id()),
    };
    let id = ClockGovernedHardwareReauthorizationIdV1(
        hash_serializable(HARDWARE_AUTHORITY_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedHardwareReauthorizationV1 {
        id,
        handoff_id: handoff.id(),
        probation_clearance_id: probation.id(),
        telemetry_bound_clearance_id: telemetry_bound_probation.id(),
        statement: signed.statement.clone(),
        statement_digest,
        signed_evidence_digest,
        hardware_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        verifier_set_digest,
        current_clock_envelope_id: current_clock.id(),
        current_operational_basis_id: current_basis.id(),
        clock_hop_count: commitment.clock_hop_count,
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), ClockGovernedHardwareReauthorizationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(ClockGovernedHardwareReauthorizationError::BrokenClockLineage {
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
            return Err(ClockGovernedHardwareReauthorizationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(ClockGovernedHardwareReauthorizationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn valid_verification_policy(policy: &ExactHardwareReauthorizationVerificationPolicyV1) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_HARDWARE_REAUTHORIZATION_VERIFIERS
}

fn validate_verification_providers(
    providers: &[&dyn ExactHardwareReauthorizationVerifierV1],
) -> Result<Vec<VerifierCommitment>, Vec<ClockGovernedHardwareReauthorizationError>> {
    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    let mut commitments = Vec::with_capacity(providers.len());
    for provider in providers {
        let id = provider.provider_id().to_string();
        let policy_digest = provider.verification_policy_digest();
        if invalid_identifier(&id) || policy_digest == Sha256Digest([0; 32]) {
            violations.push(ClockGovernedHardwareReauthorizationError::InvalidProvider(id));
            continue;
        }
        if !seen.insert(id.clone()) {
            violations.push(ClockGovernedHardwareReauthorizationError::DuplicateProvider(id));
            continue;
        }
        commitments.push(VerifierCommitment {
            provider_id: id,
            verification_policy_digest: policy_digest.to_hex(),
        });
    }
    commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    if violations.is_empty() {
        Ok(commitments)
    } else {
        Err(violations)
    }
}

fn digest_hardware_policy(
    policy: &HardwareReauthorizationPolicy,
) -> Result<Sha256Digest, ClockGovernedHardwareReauthorizationError> {
    if policy.maximum_authorization_duration_s == 0 || policy.maximum_statement_age_s == 0 {
        return Err(ClockGovernedHardwareReauthorizationError::InvalidPolicy);
    }
    hash_serializable(
        HARDWARE_POLICY_DOMAIN,
        &HardwarePolicyCommitment {
            maximum_authorization_duration_s: policy.maximum_authorization_duration_s,
            maximum_statement_age_s: policy.maximum_statement_age_s,
        },
    )
}

fn hardware_signature_message(statement_digest: Sha256Digest) -> Vec<u8> {
    let mut message = HARDWARE_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&statement_digest.0);
    message
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_HARDWARE_ID_BYTES
        || value.chars().any(char::is_control)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedHardwareReauthorizationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedHardwareReauthorizationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_verifier_policy_defaults_to_two_providers() {
        let policy = ExactHardwareReauthorizationVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_verification_policy(&policy));
    }

    #[test]
    fn policy_digest_changes_with_age_limit() {
        let first = HardwareReauthorizationPolicy {
            maximum_authorization_duration_s: 100,
            maximum_statement_age_s: 10,
        };
        let second = HardwareReauthorizationPolicy {
            maximum_statement_age_s: 11,
            ..first.clone()
        };
        assert_ne!(
            digest_hardware_policy(&first).unwrap(),
            digest_hardware_policy(&second).unwrap()
        );
    }
}
