// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict consumer-side verification for MACE Decision Outcome Receipt v1.
//!
//! This module verifies a signed evaluation record against consumer-owned
//! episode, execution and preregistered-analysis bindings. It checks internal
//! arithmetic and non-compensatory safety fields, but it cannot prove that an
//! evaluator observed the real world honestly or that a causal design is
//! correctly implemented.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::evidence_qualification_gate::{
    EvidenceQualificationRejectV1, EvidenceQualificationSignatureVerifierV1,
};

pub const DECISION_OUTCOME_RECEIPT_SCHEMA_V1: &str = "mace_decision_outcome_receipt_v1";
pub const DECISION_OUTCOME_PAYLOAD_SCHEMA_V1: &str = "mace_decision_outcome_payload_v1";
pub const MAX_DECISION_OUTCOME_RECEIPT_BYTES_V1: usize = 1_048_576;
pub const MAX_DECISION_OUTCOME_UNIX_SECONDS_V1: u64 = 253_402_300_799;
pub const MAX_DECISION_OUTCOME_MICROUNITS_V1: i64 = 9_000_000_000_000_000;
pub const MAX_DECISION_OUTCOME_INDEPENDENT_EPISODES_V1: u64 = 1_000_000_000;
pub const MAX_DECISION_OUTCOME_IDENTIFIER_BYTES_V1: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeReceiptV1 {
    pub schema_version: String,
    pub payload: DecisionOutcomePayloadV1,
    pub attestation: DecisionOutcomeAttestationV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomePayloadV1 {
    pub schema_version: String,
    pub decision_episode_id: String,
    pub episode_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub evaluation_manifest_commitment_receipt_sha256: String,
    pub qualification_receipt_sha256: Option<String>,
    pub decision_sufficiency_receipt_sha256: Option<String>,
    pub decision_record_sha256: String,
    pub execution: DecisionOutcomeExecutionV1,
    pub outcome: DecisionOutcomeMeasurementV1,
    pub hard_constraints: DecisionOutcomeHardConstraintsV1,
    pub causal: DecisionOutcomeCausalV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcomeExecutionStatusV1 {
    Executed,
    Blocked,
    Abstained,
    NotAttempted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeExecutionV1 {
    pub status: DecisionOutcomeExecutionStatusV1,
    pub action_sha256: Option<String>,
    pub authorization_receipt_sha256: Option<String>,
    pub execution_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeMeasurementV1 {
    pub observation_window_start_unix: u64,
    pub observation_window_end_unix: u64,
    pub outcome_observation_sha256: String,
    pub utility_scale_sha256: String,
    pub baseline_utility_microunits: i64,
    pub treatment_utility_microunits: i64,
    pub utility_delta_microunits: i64,
    pub resource_cost_report_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcomeHardConstraintVerdictV1 {
    Satisfied,
    Violated,
    NotEvaluable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeHardConstraintsV1 {
    pub verdict: DecisionOutcomeHardConstraintVerdictV1,
    pub violation_count: Option<u64>,
    pub violation_report_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcomeCausalDesignV1 {
    RandomizedPaired,
    MatchedPaired,
    StructuralCounterfactual,
    DescriptiveOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcomeAttributionV1 {
    IdentifiedPositive,
    IdentifiedNoEffect,
    IdentifiedNegative,
    Unidentifiable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeCausalV1 {
    pub design: DecisionOutcomeCausalDesignV1,
    pub attribution: DecisionOutcomeAttributionV1,
    pub paired_episode_manifest_sha256: String,
    pub analysis_plan_sha256: String,
    pub independent_episode_count: u64,
    pub confidence_interval_report_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOutcomeAttestationV1 {
    pub evaluator_profile_sha256: String,
    pub verifier_profile_sha256: String,
    pub key_id: String,
    pub issued_at_unix: u64,
    pub payload_sha256: String,
    pub attestation_sha256: String,
    pub signature_base64: String,
}

/// Consumer-owned facts: the frozen episode, arm, decision/execution chain,
/// preregistered causal design, and separately pinned evaluator/verifier.
pub struct DecisionOutcomeExpectedBindingV1<'a> {
    pub decision_episode_id: &'a str,
    pub episode_manifest_sha256: &'a str,
    pub evaluation_manifest_sha256: &'a str,
    /// Digest of the independently verified, pre-run commitment witness receipt.
    pub evaluation_manifest_commitment_receipt_sha256: &'a str,
    pub qualification_receipt_sha256: Option<&'a str>,
    pub decision_sufficiency_receipt_sha256: Option<&'a str>,
    pub decision_record_sha256: &'a str,
    /// True only for the frozen no-memory control arm.
    pub no_memory_control_arm: bool,
    pub execution: &'a DecisionOutcomeExecutionV1,
    pub causal_design: DecisionOutcomeCausalDesignV1,
    pub paired_episode_manifest_sha256: &'a str,
    pub analysis_plan_sha256: &'a str,
    pub evaluator_profile_sha256: &'a str,
    pub verifier_profile_sha256: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionOutcomeReceiptRejectV1 {
    ReceiptTooLarge,
    InvalidJson,
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    BindingMismatch,
    NoMemoryArmBindingMismatch,
    ExecutionChainIncomplete,
    ExecutionBindingMismatch,
    InvalidObservationWindow,
    UtilityOutOfRange,
    UtilityDeltaMismatch,
    HardConstraintVerdictMismatch,
    CausalClaimMismatch,
    TooManyIndependentEpisodes,
    ReceiptIssuedInFuture,
    InvalidSignatureEncoding,
    PayloadDigestMismatch,
    AttestationDigestMismatch,
    CanonicalizationFailed,
    SignatureRejected(EvidenceQualificationRejectV1),
}

/// Private-field capability returned only after signature and semantic checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedDecisionOutcomeReceiptV1 {
    receipt_sha256: String,
    payload_sha256: String,
    decision_episode_id: String,
    execution_status: DecisionOutcomeExecutionStatusV1,
    hard_constraint_verdict: DecisionOutcomeHardConstraintVerdictV1,
    utility_delta_microunits: i64,
    causal_attribution: DecisionOutcomeAttributionV1,
    independent_episode_count: u64,
    issued_at_unix: u64,
}

impl VerifiedDecisionOutcomeReceiptV1 {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn decision_episode_id(&self) -> &str {
        &self.decision_episode_id
    }

    pub fn execution_status(&self) -> DecisionOutcomeExecutionStatusV1 {
        self.execution_status
    }

    pub fn hard_constraint_verdict(&self) -> DecisionOutcomeHardConstraintVerdictV1 {
        self.hard_constraint_verdict
    }

    /// True only for an explicit satisfied verdict; not-evaluable is not a pass.
    pub fn hard_constraints_satisfied(&self) -> bool {
        self.hard_constraint_verdict == DecisionOutcomeHardConstraintVerdictV1::Satisfied
    }

    pub fn utility_delta_microunits(&self) -> i64 {
        self.utility_delta_microunits
    }

    pub fn causal_attribution(&self) -> DecisionOutcomeAttributionV1 {
        self.causal_attribution
    }

    pub fn independent_episode_count(&self) -> u64 {
        self.independent_episode_count
    }

    pub fn issued_at_unix(&self) -> u64 {
        self.issued_at_unix
    }
}

/// Verifies one archival outcome statement. It is not an action-authorization
/// token; validate the linked action/execution receipts independently.
pub fn parse_and_verify_decision_outcome_receipt_v1(
    wire: &[u8],
    expected: &DecisionOutcomeExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
) -> Result<VerifiedDecisionOutcomeReceiptV1, DecisionOutcomeReceiptRejectV1> {
    if wire.len() > MAX_DECISION_OUTCOME_RECEIPT_BYTES_V1 {
        return Err(DecisionOutcomeReceiptRejectV1::ReceiptTooLarge);
    }

    // Every JSON object maps to a typed struct with deny_unknown_fields.
    // Serde rejects duplicate known fields during typed deserialization.
    let receipt: DecisionOutcomeReceiptV1 =
        serde_json::from_slice(wire).map_err(|_| DecisionOutcomeReceiptRejectV1::InvalidJson)?;

    validate_decision_outcome_receipt_v1(&receipt, expected, trusted_now_unix, verifier)
}

pub fn validate_decision_outcome_receipt_v1(
    receipt: &DecisionOutcomeReceiptV1,
    expected: &DecisionOutcomeExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
) -> Result<VerifiedDecisionOutcomeReceiptV1, DecisionOutcomeReceiptRejectV1> {
    if receipt.schema_version != DECISION_OUTCOME_RECEIPT_SCHEMA_V1
        || receipt.payload.schema_version != DECISION_OUTCOME_PAYLOAD_SCHEMA_V1
    {
        return Err(DecisionOutcomeReceiptRejectV1::UnsupportedSchemaVersion);
    }

    validate_atom(&receipt.payload.decision_episode_id)?;
    validate_atom(expected.decision_episode_id)?;
    if receipt.payload.decision_episode_id != expected.decision_episode_id
        || receipt.payload.episode_manifest_sha256 != expected.episode_manifest_sha256
        || receipt.payload.evaluation_manifest_sha256 != expected.evaluation_manifest_sha256
        || receipt.payload.evaluation_manifest_commitment_receipt_sha256
            != expected.evaluation_manifest_commitment_receipt_sha256
        || receipt.payload.decision_record_sha256 != expected.decision_record_sha256
    {
        return Err(DecisionOutcomeReceiptRejectV1::BindingMismatch);
    }

    for digest in [
        expected.episode_manifest_sha256,
        expected.evaluation_manifest_sha256,
        expected.evaluation_manifest_commitment_receipt_sha256,
        expected.decision_record_sha256,
        expected.evaluator_profile_sha256,
        expected.verifier_profile_sha256,
        expected.paired_episode_manifest_sha256,
        expected.analysis_plan_sha256,
    ] {
        validate_hash(digest)?;
    }

    if receipt.attestation.evaluator_profile_sha256 != expected.evaluator_profile_sha256
        || receipt.attestation.verifier_profile_sha256 != expected.verifier_profile_sha256
    {
        return Err(DecisionOutcomeReceiptRejectV1::BindingMismatch);
    }
    validate_atom(&receipt.attestation.key_id)?;
    validate_hash(&receipt.attestation.payload_sha256)?;
    validate_hash(&receipt.attestation.attestation_sha256)?;

    validate_memory_arm_binding(receipt, expected)?;
    validate_execution_chain(&receipt.payload.execution)?;
    if &receipt.payload.execution != expected.execution {
        return Err(DecisionOutcomeReceiptRejectV1::ExecutionBindingMismatch);
    }
    validate_outcome_measurement(&receipt.payload.outcome)?;
    validate_hard_constraints(&receipt.payload.hard_constraints)?;
    validate_causal_claim(&receipt.payload.causal, expected)?;

    if receipt.attestation.issued_at_unix > MAX_DECISION_OUTCOME_UNIX_SECONDS_V1
        || trusted_now_unix < receipt.attestation.issued_at_unix
        || trusted_now_unix > MAX_DECISION_OUTCOME_UNIX_SECONDS_V1
    {
        return Err(DecisionOutcomeReceiptRejectV1::ReceiptIssuedInFuture);
    }

    let calculated_payload_sha256 = decision_outcome_payload_sha256_v1(&receipt.payload)?;
    if calculated_payload_sha256 != receipt.attestation.payload_sha256 {
        return Err(DecisionOutcomeReceiptRejectV1::PayloadDigestMismatch);
    }

    let calculated_attestation_sha256 = decision_outcome_attestation_sha256_v1(
        &calculated_payload_sha256,
        &receipt.attestation,
    )?;
    if calculated_attestation_sha256 != receipt.attestation.attestation_sha256 {
        return Err(DecisionOutcomeReceiptRejectV1::AttestationDigestMismatch);
    }

    if receipt.attestation.signature_base64.len() != 88 {
        return Err(DecisionOutcomeReceiptRejectV1::InvalidSignatureEncoding);
    }
    let signature = STANDARD
        .decode(&receipt.attestation.signature_base64)
        .map_err(|_| DecisionOutcomeReceiptRejectV1::InvalidSignatureEncoding)?;
    if signature.len() != 64 || STANDARD.encode(&signature) != receipt.attestation.signature_base64 {
        return Err(DecisionOutcomeReceiptRejectV1::InvalidSignatureEncoding);
    }

    // The common MACE v1 Ed25519 verifier signs the UTF-8 bytes of the
    // lowercase 64-character SHA-256 hex digest (not its raw 32 bytes).
    verifier
        .verify_signature(
            &receipt.attestation.verifier_profile_sha256,
            &receipt.attestation.key_id,
            &calculated_attestation_sha256,
            &signature,
            receipt.attestation.issued_at_unix,
            trusted_now_unix,
        )
        .map_err(DecisionOutcomeReceiptRejectV1::SignatureRejected)?;

    let canonical_receipt = serde_json_canonicalizer::to_vec(receipt)
        .map_err(|_| DecisionOutcomeReceiptRejectV1::CanonicalizationFailed)?;
    let mut receipt_hasher = Sha256::new();
    receipt_hasher.update(b"luminous:mace:decision-outcome-receipt:v1\0");
    receipt_hasher.update(&canonical_receipt);
    let receipt_sha256 = hex_digest(&receipt_hasher.finalize());

    Ok(VerifiedDecisionOutcomeReceiptV1 {
        receipt_sha256,
        payload_sha256: calculated_payload_sha256,
        decision_episode_id: receipt.payload.decision_episode_id.clone(),
        execution_status: receipt.payload.execution.status,
        hard_constraint_verdict: receipt.payload.hard_constraints.verdict,
        utility_delta_microunits: receipt.payload.outcome.utility_delta_microunits,
        causal_attribution: receipt.payload.causal.attribution,
        independent_episode_count: receipt.payload.causal.independent_episode_count,
        issued_at_unix: receipt.attestation.issued_at_unix,
    })
}

/// Digest only the signed outcome payload; the Ed25519 signature covers the
/// separate attestation digest which binds this hash to signer and issuance metadata.
pub fn decision_outcome_payload_sha256_v1(
    payload: &DecisionOutcomePayloadV1,
) -> Result<String, DecisionOutcomeReceiptRejectV1> {
    let canonical = serde_json_canonicalizer::to_vec(payload)
        .map_err(|_| DecisionOutcomeReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-outcome-payload:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}


/// Hash the payload digest with the evaluator/verifier identity and issuance
/// metadata so the signature cannot be rebound to another signer or date.
pub fn decision_outcome_attestation_sha256_v1(
    payload_sha256: &str,
    attestation: &DecisionOutcomeAttestationV1,
) -> Result<String, DecisionOutcomeReceiptRejectV1> {
    validate_hash(payload_sha256)?;
    let preimage = json!({
        "schema_version": "mace_decision_outcome_attestation_preimage_v1",
        "payload_sha256": payload_sha256,
        "evaluator_profile_sha256": attestation.evaluator_profile_sha256,
        "verifier_profile_sha256": attestation.verifier_profile_sha256,
        "key_id": attestation.key_id,
        "issued_at_unix": attestation.issued_at_unix
    });
    let canonical = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|_| DecisionOutcomeReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-outcome-attestation:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

fn validate_memory_arm_binding(
    receipt: &DecisionOutcomeReceiptV1,
    expected: &DecisionOutcomeExpectedBindingV1<'_>,
) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    let actual_qualification = receipt.payload.qualification_receipt_sha256.as_deref();
    let actual_sufficiency = receipt.payload.decision_sufficiency_receipt_sha256.as_deref();

    if expected.no_memory_control_arm {
        if expected.qualification_receipt_sha256.is_some()
            || expected.decision_sufficiency_receipt_sha256.is_some()
            || actual_qualification.is_some()
            || actual_sufficiency.is_some()
        {
            return Err(DecisionOutcomeReceiptRejectV1::NoMemoryArmBindingMismatch);
        }
    } else if actual_qualification != expected.qualification_receipt_sha256
        || actual_sufficiency != expected.decision_sufficiency_receipt_sha256
        || actual_qualification.is_none()
        || actual_sufficiency.is_none()
    {
        return Err(DecisionOutcomeReceiptRejectV1::NoMemoryArmBindingMismatch);
    }

    if let Some(value) = actual_qualification {
        validate_hash(value)?;
    }
    if let Some(value) = actual_sufficiency {
        validate_hash(value)?;
    }
    Ok(())
}

fn validate_execution_chain(
    execution: &DecisionOutcomeExecutionV1,
) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    for digest in [
        execution.action_sha256.as_deref(),
        execution.authorization_receipt_sha256.as_deref(),
        execution.execution_receipt_sha256.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_hash(digest)?;
    }

    match execution.status {
        DecisionOutcomeExecutionStatusV1::Executed => {
            if execution.action_sha256.is_some()
                && execution.authorization_receipt_sha256.is_some()
                && execution.execution_receipt_sha256.is_some()
            {
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::ExecutionChainIncomplete)
            }
        }
        DecisionOutcomeExecutionStatusV1::Blocked => {
            if execution.action_sha256.is_none() && execution.execution_receipt_sha256.is_none() {
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::ExecutionChainIncomplete)
            }
        }
        DecisionOutcomeExecutionStatusV1::Abstained
        | DecisionOutcomeExecutionStatusV1::NotAttempted => {
            if execution.action_sha256.is_none()
                && execution.authorization_receipt_sha256.is_none()
                && execution.execution_receipt_sha256.is_none()
            {
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::ExecutionChainIncomplete)
            }
        }
    }
}

fn validate_outcome_measurement(
    outcome: &DecisionOutcomeMeasurementV1,
) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    if outcome.observation_window_start_unix > outcome.observation_window_end_unix
        || outcome.observation_window_start_unix > MAX_DECISION_OUTCOME_UNIX_SECONDS_V1
        || outcome.observation_window_end_unix > MAX_DECISION_OUTCOME_UNIX_SECONDS_V1
    {
        return Err(DecisionOutcomeReceiptRejectV1::InvalidObservationWindow);
    }

    for digest in [
        outcome.outcome_observation_sha256.as_str(),
        outcome.utility_scale_sha256.as_str(),
        outcome.resource_cost_report_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }

    for value in [
        outcome.baseline_utility_microunits,
        outcome.treatment_utility_microunits,
        outcome.utility_delta_microunits,
    ] {
        if value < -MAX_DECISION_OUTCOME_MICROUNITS_V1
            || value > MAX_DECISION_OUTCOME_MICROUNITS_V1
        {
            return Err(DecisionOutcomeReceiptRejectV1::UtilityOutOfRange);
        }
    }

    let expected_delta = outcome
        .treatment_utility_microunits
        .checked_sub(outcome.baseline_utility_microunits)
        .ok_or(DecisionOutcomeReceiptRejectV1::UtilityOutOfRange)?;
    if expected_delta != outcome.utility_delta_microunits {
        return Err(DecisionOutcomeReceiptRejectV1::UtilityDeltaMismatch);
    }
    Ok(())
}

fn validate_hard_constraints(
    constraints: &DecisionOutcomeHardConstraintsV1,
) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    match constraints.verdict {
        DecisionOutcomeHardConstraintVerdictV1::Satisfied => {
            if constraints.violation_count == Some(0)
                && constraints.violation_report_sha256.is_none()
            {
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::HardConstraintVerdictMismatch)
            }
        }
        DecisionOutcomeHardConstraintVerdictV1::Violated => {
            if constraints.violation_count.is_some_and(|count| count > 0) {
                if let Some(report) = constraints.violation_report_sha256.as_deref() {
                    validate_hash(report)?;
                    Ok(())
                } else {
                    Err(DecisionOutcomeReceiptRejectV1::HardConstraintVerdictMismatch)
                }
            } else {
                Err(DecisionOutcomeReceiptRejectV1::HardConstraintVerdictMismatch)
            }
        }
        DecisionOutcomeHardConstraintVerdictV1::NotEvaluable => {
            if constraints.violation_count.is_none() {
                if let Some(report) = constraints.violation_report_sha256.as_deref() {
                    validate_hash(report)?;
                }
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::HardConstraintVerdictMismatch)
            }
        }
    }
}

fn validate_causal_claim(
    causal: &DecisionOutcomeCausalV1,
    expected: &DecisionOutcomeExpectedBindingV1<'_>,
) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    if causal.design != expected.causal_design
        || causal.paired_episode_manifest_sha256 != expected.paired_episode_manifest_sha256
        || causal.analysis_plan_sha256 != expected.analysis_plan_sha256
    {
        return Err(DecisionOutcomeReceiptRejectV1::BindingMismatch);
    }
    validate_hash(&causal.paired_episode_manifest_sha256)?;
    validate_hash(&causal.analysis_plan_sha256)?;

    if causal.independent_episode_count > MAX_DECISION_OUTCOME_INDEPENDENT_EPISODES_V1 {
        return Err(DecisionOutcomeReceiptRejectV1::TooManyIndependentEpisodes);
    }
    if let Some(report) = causal.confidence_interval_report_sha256.as_deref() {
        validate_hash(report)?;
    }

    match causal.design {
        DecisionOutcomeCausalDesignV1::DescriptiveOnly => {
            if causal.attribution == DecisionOutcomeAttributionV1::Unidentifiable
                && causal.confidence_interval_report_sha256.is_none()
            {
                Ok(())
            } else {
                Err(DecisionOutcomeReceiptRejectV1::CausalClaimMismatch)
            }
        }
        _ => {
            let identified = causal.attribution != DecisionOutcomeAttributionV1::Unidentifiable;
            if identified
                && (causal.independent_episode_count == 0
                    || causal.confidence_interval_report_sha256.is_none())
            {
                Err(DecisionOutcomeReceiptRejectV1::CausalClaimMismatch)
            } else {
                Ok(())
            }
        }
    }
}

fn validate_atom(value: &str) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    if value.is_empty()
        || value.len() > MAX_DECISION_OUTCOME_IDENTIFIER_BYTES_V1
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(DecisionOutcomeReceiptRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), DecisionOutcomeReceiptRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DecisionOutcomeReceiptRejectV1::InvalidDigest);
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification_trust::{
        PinnedEd25519QualificationVerifierV1, QualificationKeyStatusV1,
        QualificationTrustKeyV1, QualificationTrustProfileV1, qualification_key_id_v1,
    };
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    struct Fixture {
        receipt: DecisionOutcomeReceiptV1,
        verifier: PinnedEd25519QualificationVerifierV1,
        signer: SigningKey,
        episode_manifest_sha256: String,
        evaluation_manifest_sha256: String,
        evaluation_manifest_commitment_receipt_sha256: String,
        qualification_receipt_sha256: Option<String>,
        decision_sufficiency_receipt_sha256: Option<String>,
        decision_record_sha256: String,
        execution: DecisionOutcomeExecutionV1,
        paired_episode_manifest_sha256: String,
        analysis_plan_sha256: String,
        evaluator_profile_sha256: String,
        verifier_profile_sha256: String,
    }

    impl Fixture {
        fn expected(&self) -> DecisionOutcomeExpectedBindingV1<'_> {
            DecisionOutcomeExpectedBindingV1 {
                decision_episode_id: "episode-1",
                episode_manifest_sha256: &self.episode_manifest_sha256,
                evaluation_manifest_sha256: &self.evaluation_manifest_sha256,
                evaluation_manifest_commitment_receipt_sha256: &self.evaluation_manifest_commitment_receipt_sha256,
                qualification_receipt_sha256: self.qualification_receipt_sha256.as_deref(),
                decision_sufficiency_receipt_sha256: self
                    .decision_sufficiency_receipt_sha256
                    .as_deref(),
                decision_record_sha256: &self.decision_record_sha256,
                no_memory_control_arm: false,
                execution: &self.execution,
                causal_design: DecisionOutcomeCausalDesignV1::RandomizedPaired,
                paired_episode_manifest_sha256: &self.paired_episode_manifest_sha256,
                analysis_plan_sha256: &self.analysis_plan_sha256,
                evaluator_profile_sha256: &self.evaluator_profile_sha256,
                verifier_profile_sha256: &self.verifier_profile_sha256,
            }
        }

        fn resign(&mut self) {
            let payload_sha256 = decision_outcome_payload_sha256_v1(&self.receipt.payload).unwrap();
            self.receipt.attestation.payload_sha256 = payload_sha256.clone();
            let attestation_sha256 = decision_outcome_attestation_sha256_v1(
                &payload_sha256,
                &self.receipt.attestation,
            ).unwrap();
            self.receipt.attestation.attestation_sha256 = attestation_sha256.clone();
            let signature = self.signer.sign(attestation_sha256.as_bytes()).to_bytes();
            self.receipt.attestation.signature_base64 = STANDARD.encode(signature);
        }

        fn verify(&self, now: u64) -> Result<VerifiedDecisionOutcomeReceiptV1, DecisionOutcomeReceiptRejectV1> {
            validate_decision_outcome_receipt_v1(
                &self.receipt,
                &self.expected(),
                now,
                &self.verifier,
            )
        }
    }

    fn fixture() -> Fixture {
        let signer = SigningKey::from_bytes(&[8; 32]);
        let public_key = signer.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-outcome-test-verifier".to_owned(),
            profile_version: 1,
            keys: vec![QualificationTrustKeyV1 {
                key_id: key_id.clone(),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 10,
                not_after_unix: 1_000,
            }],
        };
        let verifier_profile_sha256 = profile.profile_sha256().unwrap();
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile,
            &verifier_profile_sha256,
            1,
        )
        .unwrap();

        let episode_manifest_sha256 = digest('a');
        let evaluation_manifest_sha256 = digest('b');
        let evaluation_manifest_commitment_receipt_sha256 = digest('9');
        let qualification_receipt_sha256 = Some(digest('c'));
        let decision_sufficiency_receipt_sha256 = Some(digest('d'));
        let decision_record_sha256 = digest('e');
        let evaluator_profile_sha256 = digest('f');
        let execution = DecisionOutcomeExecutionV1 {
            status: DecisionOutcomeExecutionStatusV1::Executed,
            action_sha256: Some(digest('1')),
            authorization_receipt_sha256: Some(digest('2')),
            execution_receipt_sha256: Some(digest('3')),
        };
        let paired_episode_manifest_sha256 = digest('4');
        let analysis_plan_sha256 = digest('5');

        let mut fixture = Fixture {
            receipt: DecisionOutcomeReceiptV1 {
                schema_version: DECISION_OUTCOME_RECEIPT_SCHEMA_V1.to_owned(),
                payload: DecisionOutcomePayloadV1 {
                    schema_version: DECISION_OUTCOME_PAYLOAD_SCHEMA_V1.to_owned(),
                    decision_episode_id: "episode-1".to_owned(),
                    episode_manifest_sha256: episode_manifest_sha256.clone(),
                    evaluation_manifest_sha256: evaluation_manifest_sha256.clone(),
                    evaluation_manifest_commitment_receipt_sha256: evaluation_manifest_commitment_receipt_sha256.clone(),
                    qualification_receipt_sha256: qualification_receipt_sha256.clone(),
                    decision_sufficiency_receipt_sha256: decision_sufficiency_receipt_sha256.clone(),
                    decision_record_sha256: decision_record_sha256.clone(),
                    execution: execution.clone(),
                    outcome: DecisionOutcomeMeasurementV1 {
                        observation_window_start_unix: 110,
                        observation_window_end_unix: 120,
                        outcome_observation_sha256: digest('6'),
                        utility_scale_sha256: digest('7'),
                        baseline_utility_microunits: 10,
                        treatment_utility_microunits: 17,
                        utility_delta_microunits: 7,
                        resource_cost_report_sha256: digest('8'),
                    },
                    hard_constraints: DecisionOutcomeHardConstraintsV1 {
                        verdict: DecisionOutcomeHardConstraintVerdictV1::Satisfied,
                        violation_count: Some(0),
                        violation_report_sha256: None,
                    },
                    causal: DecisionOutcomeCausalV1 {
                        design: DecisionOutcomeCausalDesignV1::RandomizedPaired,
                        attribution: DecisionOutcomeAttributionV1::IdentifiedPositive,
                        paired_episode_manifest_sha256: paired_episode_manifest_sha256.clone(),
                        analysis_plan_sha256: analysis_plan_sha256.clone(),
                        independent_episode_count: 20,
                        confidence_interval_report_sha256: Some(digest('9')),
                    },
                },
                attestation: DecisionOutcomeAttestationV1 {
                    evaluator_profile_sha256: evaluator_profile_sha256.clone(),
                    verifier_profile_sha256: verifier_profile_sha256.clone(),
                    key_id,
                    issued_at_unix: 200,
                    payload_sha256: digest('0'),
                    attestation_sha256: String::new(),
                    signature_base64: String::new(),
                },
            },
            verifier,
            signer,
            episode_manifest_sha256,
            evaluation_manifest_sha256,
            evaluation_manifest_commitment_receipt_sha256,
            qualification_receipt_sha256,
            decision_sufficiency_receipt_sha256,
            decision_record_sha256,
            execution,
            paired_episode_manifest_sha256,
            analysis_plan_sha256,
            evaluator_profile_sha256,
            verifier_profile_sha256,
        };
        fixture.resign();
        fixture
    }

    #[test]
    fn valid_outcome_receipt_returns_verified_capability() {
        let fixture = fixture();
        let verified = fixture.verify(250).unwrap();
        assert_eq!(verified.decision_episode_id(), "episode-1");
        assert_eq!(verified.execution_status(), DecisionOutcomeExecutionStatusV1::Executed);
        assert_eq!(
            verified.hard_constraint_verdict(),
            DecisionOutcomeHardConstraintVerdictV1::Satisfied
        );
        assert_eq!(verified.utility_delta_microunits(), 7);
        assert_eq!(
            verified.causal_attribution(),
            DecisionOutcomeAttributionV1::IdentifiedPositive
        );
        assert_eq!(verified.independent_episode_count(), 20);
        assert_eq!(verified.receipt_sha256().len(), 64);
    }

    #[test]
    fn outcome_cannot_rebind_pre_run_manifest_witness() {
        let fixture = fixture();
        let changed_witness_digest = digest('0');
        let mut expected = fixture.expected();
        expected.evaluation_manifest_commitment_receipt_sha256 = &changed_witness_digest;
        assert_eq!(
            validate_decision_outcome_receipt_v1(
                &fixture.receipt,
                &expected,
                250,
                &fixture.verifier,
            ),
            Err(DecisionOutcomeReceiptRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn utility_delta_must_match_treatment_minus_baseline() {
        let mut fixture = fixture();
        fixture.receipt.payload.outcome.utility_delta_microunits = 6;
        fixture.resign();
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::UtilityDeltaMismatch)
        );
    }

    #[test]
    fn executed_outcome_requires_action_authorization_and_execution_receipts() {
        let mut fixture = fixture();
        fixture.receipt.payload.execution.authorization_receipt_sha256 = None;
        fixture.resign();
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::ExecutionChainIncomplete)
        );
    }

    #[test]
    fn descriptive_only_cannot_claim_causal_uplift() {
        let mut fixture = fixture();
        fixture.receipt.payload.causal.design = DecisionOutcomeCausalDesignV1::DescriptiveOnly;
        fixture.receipt.payload.causal.attribution = DecisionOutcomeAttributionV1::IdentifiedPositive;
        fixture.resign();
        let mut expected = fixture.expected();
        expected.causal_design = DecisionOutcomeCausalDesignV1::DescriptiveOnly;
        assert_eq!(
            validate_decision_outcome_receipt_v1(
                &fixture.receipt,
                &expected,
                250,
                &fixture.verifier,
            ),
            Err(DecisionOutcomeReceiptRejectV1::CausalClaimMismatch)
        );
    }

    #[test]
    fn inconsistent_hard_constraint_violation_count_is_rejected() {
        let mut fixture = fixture();
        fixture.receipt.payload.hard_constraints.verdict =
            DecisionOutcomeHardConstraintVerdictV1::Violated;
        fixture.receipt.payload.hard_constraints.violation_count = Some(0);
        fixture.receipt.payload.hard_constraints.violation_report_sha256 = Some(digest('a'));
        fixture.resign();
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::HardConstraintVerdictMismatch)
        );
    }

    #[test]
    fn positive_utility_does_not_erase_a_reported_hard_constraint_violation() {
        let mut fixture = fixture();
        fixture.receipt.payload.hard_constraints.verdict =
            DecisionOutcomeHardConstraintVerdictV1::Violated;
        fixture.receipt.payload.hard_constraints.violation_count = Some(1);
        fixture.receipt.payload.hard_constraints.violation_report_sha256 = Some(digest('a'));
        fixture.resign();
        let verified = fixture.verify(250).unwrap();
        assert_eq!(
            verified.hard_constraint_verdict(),
            DecisionOutcomeHardConstraintVerdictV1::Violated
        );
        assert!(verified.utility_delta_microunits() > 0);
        assert!(!verified.hard_constraints_satisfied());
    }

    #[test]
    fn outcome_observation_window_must_be_ordered() {
        let mut fixture = fixture();
        fixture.receipt.payload.outcome.observation_window_end_unix = 100;
        fixture.resign();
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::InvalidObservationWindow)
        );
    }

    #[test]
    fn no_memory_control_requires_both_memory_receipts_to_be_null() {
        let mut fixture = fixture();
        fixture.receipt.payload.qualification_receipt_sha256 = None;
        fixture.receipt.payload.decision_sufficiency_receipt_sha256 = None;
        fixture.resign();
        let mut expected = fixture.expected();
        expected.no_memory_control_arm = true;
        expected.qualification_receipt_sha256 = None;
        expected.decision_sufficiency_receipt_sha256 = None;
        assert!(validate_decision_outcome_receipt_v1(
            &fixture.receipt,
            &expected,
            250,
            &fixture.verifier,
        ).is_ok());
    }

    #[test]
    fn duplicate_json_fields_are_rejected() {
        let fixture = fixture();
        let wire = serde_json::to_string(&fixture.receipt).unwrap();
        let duplicated = wire.replacen(
            "{\"schema_version\":\"mace_decision_outcome_receipt_v1\",",
            "{\"schema_version\":\"mace_decision_outcome_receipt_v1\",\"schema_version\":\"mace_decision_outcome_receipt_v1\",",
            1,
        );
        assert_eq!(
            parse_and_verify_decision_outcome_receipt_v1(
                duplicated.as_bytes(),
                &fixture.expected(),
                250,
                &fixture.verifier,
            ),
            Err(DecisionOutcomeReceiptRejectV1::InvalidJson)
        );
    }

    #[test]
    fn issuance_metadata_cannot_be_rebound_without_resigning() {
        let mut fixture = fixture();
        fixture.receipt.attestation.issued_at_unix += 1;
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::AttestationDigestMismatch)
        );
    }

    #[test]
    fn invalid_signature_is_rejected() {
        let mut fixture = fixture();
        let mut bytes = STANDARD.decode(&fixture.receipt.attestation.signature_base64).unwrap();
        bytes[0] ^= 0x01;
        fixture.receipt.attestation.signature_base64 = STANDARD.encode(bytes);
        assert_eq!(
            fixture.verify(250),
            Err(DecisionOutcomeReceiptRejectV1::SignatureRejected(
                EvidenceQualificationRejectV1::SignatureRejected
            ))
        );
    }
}
