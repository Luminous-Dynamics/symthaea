// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only witness-policy epochs and dual-quorum policy rotation.
//!
//! A checkpoint witness policy cannot remain immutable forever: keys expire,
//! organizations leave, and compromised witnesses must be removed.  This
//! module preserves continuity by requiring each non-genesis policy epoch to
//! be authorized by both the outgoing and incoming witness quorums.  Signature
//! algorithms, key custody, and witness trust remain external to this crate.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::evidence_calibration::{
    CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION,
    CALIBRATION_PUBLICATION_CHECKPOINT_WITNESS_POLICY_VERSION,
    CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationCheckpointWitnessPolicy,
    CalibrationSignerIdentity,
    calibration_publication_catalog_checkpoint_sha256,
    calibration_publication_checkpoint_witness_policy_sha256,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

pub const CALIBRATION_PUBLICATION_WITNESS_POLICY_EPOCH_VERSION: &str =
    "score-evidence-calibration-publication-witness-policy-epoch-v1";
pub const CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_PAYLOAD_VERSION: &str =
    "score-evidence-calibration-publication-witness-policy-rotation-payload-v1";
pub const CALIBRATION_SIGNED_PUBLICATION_WITNESS_POLICY_ROTATION_VERSION: &str =
    "score-evidence-calibration-signed-publication-witness-policy-rotation-v1";
pub const CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_SET_VERSION: &str =
    "score-evidence-calibration-publication-witness-policy-rotation-set-v1";
pub const CALIBRATION_PUBLICATION_WITNESS_POLICY_LEDGER_VERSION: &str =
    "score-evidence-calibration-publication-witness-policy-ledger-v1";
pub const CALIBRATION_PUBLICATION_WITNESS_POLICY_AUDIT_VERSION: &str =
    "score-evidence-calibration-publication-witness-policy-audit-v1";

const EPOCH_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-witness-policy-epoch.v1\0";
const ROTATION_PAYLOAD_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-witness-policy-rotation-payload.v1\0";
const ROTATION_ENVELOPE_DOMAIN: &[u8] =
    b"symthaea.score-evidence.signed-publication-witness-policy-rotation.v1\0";
const ROTATION_SET_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-witness-policy-rotation-set.v1\0";
const LEDGER_DOMAIN: &[u8] =
    b"symthaea.score-evidence.publication-witness-policy-ledger.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyEpoch {
    pub epoch_version: String,
    pub ordinal: u64,
    pub policy: CalibrationPublicationCheckpointWitnessPolicy,
    pub activation_checkpoint: CalibrationPublicationCatalogCheckpoint,
    pub previous_epoch_sha256: Option<String>,
    pub issued_epoch: u64,
    pub epoch_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyRotationPayload {
    pub payload_version: String,
    pub catalog_id: String,
    pub authority_id: String,
    pub rotation_ordinal: u64,
    pub from_epoch_sha256: String,
    pub to_epoch_sha256: String,
    pub activation_checkpoint_sha256: String,
    pub issued_epoch: u64,
    pub payload_sha256: String,
}

impl CalibrationPublicationWitnessPolicyRotationPayload {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(ROTATION_PAYLOAD_DOMAIN);
        push_field(&mut bytes, &self.payload_version);
        push_field(&mut bytes, &self.catalog_id);
        push_field(&mut bytes, &self.authority_id);
        bytes.extend_from_slice(&self.rotation_ordinal.to_le_bytes());
        push_field(&mut bytes, &self.from_epoch_sha256);
        push_field(&mut bytes, &self.to_epoch_sha256);
        push_field(&mut bytes, &self.activation_checkpoint_sha256);
        bytes.extend_from_slice(&self.issued_epoch.to_le_bytes());
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSignedPublicationWitnessPolicyRotation {
    pub envelope_version: String,
    pub payload: CalibrationPublicationWitnessPolicyRotationPayload,
    pub signer: CalibrationSignerIdentity,
    pub signature_hex: String,
    pub envelope_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyRotationSet {
    pub set_version: String,
    pub payload_sha256: String,
    pub outgoing_policy_sha256: String,
    pub incoming_policy_sha256: String,
    pub outgoing_statements: Vec<CalibrationSignedPublicationWitnessPolicyRotation>,
    pub incoming_statements: Vec<CalibrationSignedPublicationWitnessPolicyRotation>,
    pub set_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyLedger {
    pub ledger_version: String,
    pub catalog_id: String,
    pub authority_id: String,
    pub epochs: Vec<CalibrationPublicationWitnessPolicyEpoch>,
    pub rotations: Vec<CalibrationPublicationWitnessPolicyRotationSet>,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPublicationWitnessPolicyIssueCode {
    LedgerVersionMismatch,
    EmptyIdentity,
    MissingGenesis,
    EpochRotationCountMismatch,
    EpochVersionMismatch,
    EpochOrdinalMismatch,
    PreviousEpochMismatch,
    PolicyVersionMismatch,
    ZeroWitnessThreshold,
    EmptyAcceptedWitness,
    NonCanonicalAcceptedWitnessOrder,
    InsufficientAcceptedWitnesses,
    PolicySha256Mismatch,
    CheckpointVersionMismatch,
    CheckpointSha256Mismatch,
    CheckpointIdentityMismatch,
    ActivationCountRegression,
    EpochBeforeCheckpoint,
    EpochSha256Mismatch,
    RotationSetVersionMismatch,
    RotationPayloadVersionMismatch,
    RotationOrdinalMismatch,
    RotationIdentityMismatch,
    RotationEpochMismatch,
    RotationCheckpointMismatch,
    RotationPayloadSha256Mismatch,
    RotationPolicyMismatch,
    PolicyUnchanged,
    EmptySignerIdentity,
    InvalidSignatureHex,
    RotationEnvelopeVersionMismatch,
    RotationEnvelopeSha256Mismatch,
    DuplicateRotationSigner,
    UnacceptedOutgoingSigner,
    UnacceptedIncomingSigner,
    OutgoingSignatureRejected,
    IncomingSignatureRejected,
    OutgoingThresholdNotMet,
    IncomingThresholdNotMet,
    RotationSetSha256Mismatch,
    LedgerSha256Mismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyIssue {
    pub code: CalibrationPublicationWitnessPolicyIssueCode,
    pub ordinal: Option<u64>,
    pub signer_key_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPublicationWitnessPolicyAuditReport {
    pub audit_version: String,
    pub structurally_valid: bool,
    pub authenticated_rotations: u64,
    pub total_rotations: u64,
    pub rotations_authenticated: bool,
    pub issues: Vec<CalibrationPublicationWitnessPolicyIssue>,
}

impl CalibrationPublicationWitnessPolicyAuditReport {
    pub fn valid(&self) -> bool {
        self.structurally_valid && self.issues.is_empty()
    }

    pub fn accepted(&self) -> bool {
        self.valid() && self.rotations_authenticated
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "details", rename_all = "snake_case")]
pub enum CalibrationPublicationWitnessPolicyError {
    InvalidLedger { issues: usize },
    InvalidCheckpoint,
    CatalogIdentityMismatch,
    ActivationCountRegression,
    PolicyUnchanged,
    InvalidRotation { issues: usize },
}

impl std::fmt::Display for CalibrationPublicationWitnessPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLedger { issues } => {
                write!(formatter, "witness-policy ledger audit failed with {issues} issues")
            }
            Self::InvalidCheckpoint => write!(formatter, "activation checkpoint is invalid"),
            Self::CatalogIdentityMismatch => {
                write!(formatter, "witness-policy ledger and checkpoint identities differ")
            }
            Self::ActivationCountRegression => {
                write!(formatter, "witness-policy activation checkpoint must advance the catalog")
            }
            Self::PolicyUnchanged => write!(formatter, "incoming witness policy is unchanged"),
            Self::InvalidRotation { issues } => {
                write!(formatter, "witness-policy rotation audit failed with {issues} issues")
            }
        }
    }
}

impl std::error::Error for CalibrationPublicationWitnessPolicyError {}

pub trait CalibrationPublicationWitnessPolicyRotationVerifier {
    type Error: std::fmt::Display;

    fn verify(
        &self,
        payload: &[u8],
        signer: &CalibrationSignerIdentity,
        signature: &[u8],
    ) -> Result<(), Self::Error>;
}

pub fn build_calibration_publication_witness_policy_genesis(
    checkpoint: CalibrationPublicationCatalogCheckpoint,
    policy: CalibrationPublicationCheckpointWitnessPolicy,
    issued_epoch: u64,
) -> Result<CalibrationPublicationWitnessPolicyLedger, CalibrationPublicationWitnessPolicyError> {
    if !checkpoint_structurally_valid(&checkpoint) {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidCheckpoint);
    }
    let mut epoch = CalibrationPublicationWitnessPolicyEpoch {
        epoch_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_EPOCH_VERSION.into(),
        ordinal: 0,
        policy,
        activation_checkpoint: checkpoint.clone(),
        previous_epoch_sha256: None,
        issued_epoch,
        epoch_sha256: String::new(),
    };
    epoch.epoch_sha256 = calibration_publication_witness_policy_epoch_sha256(&epoch);
    let mut ledger = CalibrationPublicationWitnessPolicyLedger {
        ledger_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_LEDGER_VERSION.into(),
        catalog_id: checkpoint.catalog_id,
        authority_id: checkpoint.authority_id,
        epochs: vec![epoch],
        rotations: Vec::new(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = calibration_publication_witness_policy_ledger_sha256(&ledger);
    let audit = audit_calibration_publication_witness_policy_ledger(&ledger);
    if !audit.valid() {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidLedger {
            issues: audit.issues.len(),
        });
    }
    Ok(ledger)
}

pub fn plan_calibration_publication_witness_policy_rotation(
    ledger: &CalibrationPublicationWitnessPolicyLedger,
    activation_checkpoint: CalibrationPublicationCatalogCheckpoint,
    incoming_policy: CalibrationPublicationCheckpointWitnessPolicy,
    issued_epoch: u64,
) -> Result<(
    CalibrationPublicationWitnessPolicyEpoch,
    CalibrationPublicationWitnessPolicyRotationPayload,
), CalibrationPublicationWitnessPolicyError> {
    let audit = audit_calibration_publication_witness_policy_ledger(ledger);
    if !audit.valid() {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidLedger {
            issues: audit.issues.len(),
        });
    }
    if !checkpoint_structurally_valid(&activation_checkpoint) {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidCheckpoint);
    }
    if activation_checkpoint.catalog_id != ledger.catalog_id
        || activation_checkpoint.authority_id != ledger.authority_id
    {
        return Err(CalibrationPublicationWitnessPolicyError::CatalogIdentityMismatch);
    }
    let previous = ledger.epochs.last().ok_or(
        CalibrationPublicationWitnessPolicyError::InvalidLedger { issues: 1 },
    )?;
    if activation_checkpoint.event_count <= previous.activation_checkpoint.event_count {
        return Err(CalibrationPublicationWitnessPolicyError::ActivationCountRegression);
    }
    if incoming_policy.policy_sha256 == previous.policy.policy_sha256 {
        return Err(CalibrationPublicationWitnessPolicyError::PolicyUnchanged);
    }
    let mut epoch = CalibrationPublicationWitnessPolicyEpoch {
        epoch_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_EPOCH_VERSION.into(),
        ordinal: previous.ordinal + 1,
        policy: incoming_policy,
        activation_checkpoint: activation_checkpoint.clone(),
        previous_epoch_sha256: Some(previous.epoch_sha256.clone()),
        issued_epoch,
        epoch_sha256: String::new(),
    };
    epoch.epoch_sha256 = calibration_publication_witness_policy_epoch_sha256(&epoch);
    let mut payload = CalibrationPublicationWitnessPolicyRotationPayload {
        payload_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_PAYLOAD_VERSION.into(),
        catalog_id: ledger.catalog_id.clone(),
        authority_id: ledger.authority_id.clone(),
        rotation_ordinal: epoch.ordinal,
        from_epoch_sha256: previous.epoch_sha256.clone(),
        to_epoch_sha256: epoch.epoch_sha256.clone(),
        activation_checkpoint_sha256: activation_checkpoint.checkpoint_sha256,
        issued_epoch,
        payload_sha256: String::new(),
    };
    payload.payload_sha256 = calibration_publication_witness_policy_rotation_payload_sha256(&payload);
    Ok((epoch, payload))
}

pub fn build_calibration_signed_publication_witness_policy_rotation(
    payload: CalibrationPublicationWitnessPolicyRotationPayload,
    signer: CalibrationSignerIdentity,
    signature: &[u8],
) -> CalibrationSignedPublicationWitnessPolicyRotation {
    let mut envelope = CalibrationSignedPublicationWitnessPolicyRotation {
        envelope_version: CALIBRATION_SIGNED_PUBLICATION_WITNESS_POLICY_ROTATION_VERSION.into(),
        payload,
        signer,
        signature_hex: encode_hex(signature),
        envelope_sha256: String::new(),
    };
    envelope.envelope_sha256 = calibration_signed_publication_witness_policy_rotation_sha256(&envelope);
    envelope
}

pub fn build_calibration_publication_witness_policy_rotation_set(
    payload: &CalibrationPublicationWitnessPolicyRotationPayload,
    outgoing_policy: &CalibrationPublicationCheckpointWitnessPolicy,
    incoming_policy: &CalibrationPublicationCheckpointWitnessPolicy,
    outgoing_statements: Vec<CalibrationSignedPublicationWitnessPolicyRotation>,
    incoming_statements: Vec<CalibrationSignedPublicationWitnessPolicyRotation>,
) -> CalibrationPublicationWitnessPolicyRotationSet {
    let mut set = CalibrationPublicationWitnessPolicyRotationSet {
        set_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_SET_VERSION.into(),
        payload_sha256: payload.payload_sha256.clone(),
        outgoing_policy_sha256: outgoing_policy.policy_sha256.clone(),
        incoming_policy_sha256: incoming_policy.policy_sha256.clone(),
        outgoing_statements,
        incoming_statements,
        set_sha256: String::new(),
    };
    set.set_sha256 = calibration_publication_witness_policy_rotation_set_sha256(&set);
    set
}

pub fn append_calibration_publication_witness_policy_rotation<
    V: CalibrationPublicationWitnessPolicyRotationVerifier,
>(
    ledger: &mut CalibrationPublicationWitnessPolicyLedger,
    epoch: CalibrationPublicationWitnessPolicyEpoch,
    rotation: CalibrationPublicationWitnessPolicyRotationSet,
    verifier: &V,
) -> Result<(), CalibrationPublicationWitnessPolicyError> {
    let existing = audit_calibration_publication_witness_policy_ledger(ledger);
    if !existing.valid() {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidLedger {
            issues: existing.issues.len(),
        });
    }
    let mut candidate = ledger.clone();
    candidate.epochs.push(epoch);
    candidate.rotations.push(rotation);
    candidate.ledger_sha256 = calibration_publication_witness_policy_ledger_sha256(&candidate);
    let audit = verify_calibration_publication_witness_policy_ledger(&candidate, verifier);
    if !audit.accepted() {
        return Err(CalibrationPublicationWitnessPolicyError::InvalidRotation {
            issues: audit.issues.len(),
        });
    }
    *ledger = candidate;
    Ok(())
}

pub fn active_calibration_publication_witness_policy_epoch<'a>(
    ledger: &'a CalibrationPublicationWitnessPolicyLedger,
    checkpoint: &CalibrationPublicationCatalogCheckpoint,
) -> Option<&'a CalibrationPublicationWitnessPolicyEpoch> {
    if ledger.catalog_id != checkpoint.catalog_id || ledger.authority_id != checkpoint.authority_id {
        return None;
    }
    ledger
        .epochs
        .iter()
        .rev()
        .find(|epoch| epoch.activation_checkpoint.event_count <= checkpoint.event_count)
}

pub fn audit_calibration_publication_witness_policy_ledger(
    ledger: &CalibrationPublicationWitnessPolicyLedger,
) -> CalibrationPublicationWitnessPolicyAuditReport {
    audit_ledger_inner(ledger, None::<&NeverVerifier>)
}

pub fn verify_calibration_publication_witness_policy_ledger<
    V: CalibrationPublicationWitnessPolicyRotationVerifier,
>(
    ledger: &CalibrationPublicationWitnessPolicyLedger,
    verifier: &V,
) -> CalibrationPublicationWitnessPolicyAuditReport {
    audit_ledger_inner(ledger, Some(verifier))
}

pub fn calibration_publication_witness_policy_epoch_sha256(
    epoch: &CalibrationPublicationWitnessPolicyEpoch,
) -> String {
    let mut hash = Sha256::new();
    hash.update(EPOCH_DOMAIN);
    hash_field(&mut hash, &epoch.epoch_version);
    hash.update(&epoch.ordinal.to_le_bytes());
    hash_field(&mut hash, &epoch.policy.policy_sha256);
    hash_field(&mut hash, &epoch.activation_checkpoint.checkpoint_sha256);
    hash_optional_field(&mut hash, epoch.previous_epoch_sha256.as_deref());
    hash.update(&epoch.issued_epoch.to_le_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_witness_policy_rotation_payload_sha256(
    payload: &CalibrationPublicationWitnessPolicyRotationPayload,
) -> String {
    let mut hash = Sha256::new();
    hash.update(&payload.canonical_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_signed_publication_witness_policy_rotation_sha256(
    envelope: &CalibrationSignedPublicationWitnessPolicyRotation,
) -> String {
    let mut hash = Sha256::new();
    hash.update(ROTATION_ENVELOPE_DOMAIN);
    hash_field(&mut hash, &envelope.envelope_version);
    hash_field(&mut hash, &envelope.payload.payload_sha256);
    hash_field(&mut hash, &envelope.signer.key_id);
    hash_field(&mut hash, &envelope.signer.algorithm);
    hash_optional_field(&mut hash, envelope.signer.issuer.as_deref());
    hash_field(&mut hash, &envelope.signature_hex);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_witness_policy_rotation_set_sha256(
    set: &CalibrationPublicationWitnessPolicyRotationSet,
) -> String {
    let mut hash = Sha256::new();
    hash.update(ROTATION_SET_DOMAIN);
    hash_field(&mut hash, &set.set_version);
    hash_field(&mut hash, &set.payload_sha256);
    hash_field(&mut hash, &set.outgoing_policy_sha256);
    hash_field(&mut hash, &set.incoming_policy_sha256);
    hash_envelopes(&mut hash, &set.outgoing_statements);
    hash_envelopes(&mut hash, &set.incoming_statements);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_witness_policy_ledger_sha256(
    ledger: &CalibrationPublicationWitnessPolicyLedger,
) -> String {
    let mut hash = Sha256::new();
    hash.update(LEDGER_DOMAIN);
    hash_field(&mut hash, &ledger.ledger_version);
    hash_field(&mut hash, &ledger.catalog_id);
    hash_field(&mut hash, &ledger.authority_id);
    hash.update(&(ledger.epochs.len() as u64).to_le_bytes());
    for epoch in &ledger.epochs {
        hash_field(&mut hash, &epoch.epoch_sha256);
    }
    hash.update(&(ledger.rotations.len() as u64).to_le_bytes());
    for rotation in &ledger.rotations {
        hash_field(&mut hash, &rotation.set_sha256);
    }
    sha256_hex(&hash.finalize())
}

fn audit_ledger_inner<V: CalibrationPublicationWitnessPolicyRotationVerifier>(
    ledger: &CalibrationPublicationWitnessPolicyLedger,
    verifier: Option<&V>,
) -> CalibrationPublicationWitnessPolicyAuditReport {
    let mut report = CalibrationPublicationWitnessPolicyAuditReport {
        audit_version: CALIBRATION_PUBLICATION_WITNESS_POLICY_AUDIT_VERSION.into(),
        structurally_valid: true,
        authenticated_rotations: 0,
        total_rotations: ledger.rotations.len() as u64,
        rotations_authenticated: ledger.rotations.is_empty(),
        issues: Vec::new(),
    };
    if ledger.ledger_version != CALIBRATION_PUBLICATION_WITNESS_POLICY_LEDGER_VERSION {
        issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::LedgerVersionMismatch, None, None, "witness-policy ledger version mismatch");
    }
    if ledger.catalog_id.trim().is_empty() || ledger.authority_id.trim().is_empty() {
        issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EmptyIdentity, None, None, "catalog and authority identities must not be empty");
    }
    if ledger.epochs.is_empty() {
        issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::MissingGenesis, None, None, "witness-policy ledger requires a genesis epoch");
    }
    if ledger.epochs.len().saturating_sub(1) != ledger.rotations.len() {
        issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EpochRotationCountMismatch, None, None, "every non-genesis policy epoch requires one rotation set");
    }

    for (index, epoch) in ledger.epochs.iter().enumerate() {
        let ordinal = index as u64;
        if epoch.epoch_version != CALIBRATION_PUBLICATION_WITNESS_POLICY_EPOCH_VERSION {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EpochVersionMismatch, Some(ordinal), None, "policy-epoch version mismatch");
        }
        if epoch.ordinal != ordinal {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EpochOrdinalMismatch, Some(ordinal), None, "policy-epoch ordinal is not contiguous");
        }
        let expected_previous = if index == 0 {
            None
        } else {
            Some(ledger.epochs[index - 1].epoch_sha256.as_str())
        };
        if epoch.previous_epoch_sha256.as_deref() != expected_previous {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::PreviousEpochMismatch, Some(ordinal), None, "policy-epoch predecessor hash mismatch");
        }
        audit_policy(&epoch.policy, ordinal, &mut report);
        if epoch.activation_checkpoint.checkpoint_version != CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::CheckpointVersionMismatch, Some(ordinal), None, "activation checkpoint version mismatch");
        }
        if epoch.activation_checkpoint.checkpoint_sha256
            != calibration_publication_catalog_checkpoint_sha256(&epoch.activation_checkpoint)
        {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::CheckpointSha256Mismatch, Some(ordinal), None, "activation checkpoint SHA-256 mismatch");
        }
        if epoch.activation_checkpoint.catalog_id != ledger.catalog_id
            || epoch.activation_checkpoint.authority_id != ledger.authority_id
        {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::CheckpointIdentityMismatch, Some(ordinal), None, "activation checkpoint identity differs from the ledger");
        }
        if index > 0
            && epoch.activation_checkpoint.event_count
                <= ledger.epochs[index - 1].activation_checkpoint.event_count
        {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::ActivationCountRegression, Some(ordinal), None, "policy activation must advance the catalog event count");
        }
        if epoch.issued_epoch < epoch.activation_checkpoint.issued_epoch {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EpochBeforeCheckpoint, Some(ordinal), None, "policy epoch was issued before its activation checkpoint");
        }
        if epoch.epoch_sha256 != calibration_publication_witness_policy_epoch_sha256(epoch) {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::EpochSha256Mismatch, Some(ordinal), None, "policy-epoch SHA-256 mismatch");
        }
    }

    for index in 0..ledger.rotations.len() {
        if index + 1 >= ledger.epochs.len() {
            break;
        }
        let outgoing = &ledger.epochs[index];
        let incoming = &ledger.epochs[index + 1];
        if outgoing.policy.policy_sha256 == incoming.policy.policy_sha256 {
            issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::PolicyUnchanged, Some(incoming.ordinal), None, "adjacent witness-policy epochs are identical");
        }
        let rotation = &ledger.rotations[index];
        let authenticated = audit_rotation(
            outgoing,
            incoming,
            rotation,
            verifier,
            &mut report,
        );
        if authenticated {
            report.authenticated_rotations += 1;
        }
    }
    report.rotations_authenticated = verifier.is_some()
        && report.authenticated_rotations == report.total_rotations;
    if report.total_rotations == 0 {
        report.rotations_authenticated = true;
    }
    if ledger.ledger_sha256 != calibration_publication_witness_policy_ledger_sha256(ledger) {
        issue(&mut report, CalibrationPublicationWitnessPolicyIssueCode::LedgerSha256Mismatch, None, None, "witness-policy ledger SHA-256 mismatch");
    }
    report.structurally_valid = report.issues.iter().all(|item| {
        !matches!(
            item.code,
            CalibrationPublicationWitnessPolicyIssueCode::OutgoingSignatureRejected
                | CalibrationPublicationWitnessPolicyIssueCode::IncomingSignatureRejected
                | CalibrationPublicationWitnessPolicyIssueCode::OutgoingThresholdNotMet
                | CalibrationPublicationWitnessPolicyIssueCode::IncomingThresholdNotMet
        )
    });
    report
}

fn audit_policy(
    policy: &CalibrationPublicationCheckpointWitnessPolicy,
    ordinal: u64,
    report: &mut CalibrationPublicationWitnessPolicyAuditReport,
) {
    if policy.policy_version != CALIBRATION_PUBLICATION_CHECKPOINT_WITNESS_POLICY_VERSION {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::PolicyVersionMismatch, Some(ordinal), None, "witness policy version mismatch");
    }
    if policy.minimum_distinct_witnesses == 0 {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::ZeroWitnessThreshold, Some(ordinal), None, "witness threshold must be positive");
    }
    if policy.accepted_key_ids.iter().any(|value| value.trim().is_empty()) {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::EmptyAcceptedWitness, Some(ordinal), None, "accepted witness key IDs must not be empty");
    }
    if policy.accepted_key_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::NonCanonicalAcceptedWitnessOrder, Some(ordinal), None, "accepted witness key IDs must be strictly increasing");
    }
    if policy.accepted_key_ids.len() < policy.minimum_distinct_witnesses as usize {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::InsufficientAcceptedWitnesses, Some(ordinal), None, "accepted witness list is smaller than its threshold");
    }
    if policy.policy_sha256 != calibration_publication_checkpoint_witness_policy_sha256(policy) {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::PolicySha256Mismatch, Some(ordinal), None, "witness-policy SHA-256 mismatch");
    }
}

fn audit_rotation<V: CalibrationPublicationWitnessPolicyRotationVerifier>(
    outgoing: &CalibrationPublicationWitnessPolicyEpoch,
    incoming: &CalibrationPublicationWitnessPolicyEpoch,
    rotation: &CalibrationPublicationWitnessPolicyRotationSet,
    verifier: Option<&V>,
    report: &mut CalibrationPublicationWitnessPolicyAuditReport,
) -> bool {
    let ordinal = incoming.ordinal;
    if rotation.set_version != CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_SET_VERSION {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationSetVersionMismatch, Some(ordinal), None, "rotation-set version mismatch");
    }
    if rotation.outgoing_policy_sha256 != outgoing.policy.policy_sha256
        || rotation.incoming_policy_sha256 != incoming.policy.policy_sha256
    {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationPolicyMismatch, Some(ordinal), None, "rotation set does not bind the outgoing and incoming policies");
    }
    let payload = rotation
        .outgoing_statements
        .first()
        .map(|statement| &statement.payload)
        .or_else(|| rotation.incoming_statements.first().map(|statement| &statement.payload));
    let Some(payload) = payload else {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::OutgoingThresholdNotMet, Some(ordinal), None, "rotation set contains no outgoing statements");
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::IncomingThresholdNotMet, Some(ordinal), None, "rotation set contains no incoming statements");
        if rotation.set_sha256 != calibration_publication_witness_policy_rotation_set_sha256(rotation) {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationSetSha256Mismatch, Some(ordinal), None, "rotation-set SHA-256 mismatch");
        }
        return false;
    };
    if payload.payload_version != CALIBRATION_PUBLICATION_WITNESS_POLICY_ROTATION_PAYLOAD_VERSION {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationPayloadVersionMismatch, Some(ordinal), None, "rotation payload version mismatch");
    }
    if payload.catalog_id != incoming.activation_checkpoint.catalog_id
        || payload.authority_id != incoming.activation_checkpoint.authority_id
    {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationIdentityMismatch, Some(ordinal), None, "rotation payload identity mismatch");
    }
    if payload.rotation_ordinal != ordinal {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationOrdinalMismatch, Some(ordinal), None, "rotation payload ordinal mismatch");
    }
    if payload.from_epoch_sha256 != outgoing.epoch_sha256
        || payload.to_epoch_sha256 != incoming.epoch_sha256
    {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationEpochMismatch, Some(ordinal), None, "rotation payload does not bind adjacent policy epochs");
    }
    if payload.activation_checkpoint_sha256 != incoming.activation_checkpoint.checkpoint_sha256 {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationCheckpointMismatch, Some(ordinal), None, "rotation payload activation checkpoint mismatch");
    }
    if payload.issued_epoch != incoming.issued_epoch {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationEpochMismatch, Some(ordinal), None, "rotation payload issuance epoch mismatch");
    }
    if payload.payload_sha256 != calibration_publication_witness_policy_rotation_payload_sha256(payload)
        || rotation.payload_sha256 != payload.payload_sha256
    {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationPayloadSha256Mismatch, Some(ordinal), None, "rotation payload SHA-256 mismatch");
    }
    let outgoing_authenticated = audit_rotation_side(
        payload,
        &outgoing.policy,
        &rotation.outgoing_statements,
        true,
        verifier,
        ordinal,
        report,
    );
    let incoming_authenticated = audit_rotation_side(
        payload,
        &incoming.policy,
        &rotation.incoming_statements,
        false,
        verifier,
        ordinal,
        report,
    );
    if rotation.set_sha256 != calibration_publication_witness_policy_rotation_set_sha256(rotation) {
        issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationSetSha256Mismatch, Some(ordinal), None, "rotation-set SHA-256 mismatch");
    }
    outgoing_authenticated && incoming_authenticated
}

#[allow(clippy::too_many_arguments)]
fn audit_rotation_side<V: CalibrationPublicationWitnessPolicyRotationVerifier>(
    payload: &CalibrationPublicationWitnessPolicyRotationPayload,
    policy: &CalibrationPublicationCheckpointWitnessPolicy,
    statements: &[CalibrationSignedPublicationWitnessPolicyRotation],
    outgoing: bool,
    verifier: Option<&V>,
    ordinal: u64,
    report: &mut CalibrationPublicationWitnessPolicyAuditReport,
) -> bool {
    let accepted = policy.accepted_key_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut authenticated = BTreeSet::new();
    for statement in statements {
        let key_id = statement.signer.key_id.clone();
        if statement.envelope_version != CALIBRATION_SIGNED_PUBLICATION_WITNESS_POLICY_ROTATION_VERSION {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationEnvelopeVersionMismatch, Some(ordinal), Some(key_id.clone()), "signed rotation-envelope version mismatch");
        }
        if statement.payload != *payload {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationPayloadSha256Mismatch, Some(ordinal), Some(key_id.clone()), "rotation statement carries a different payload");
        }
        if key_id.trim().is_empty() || statement.signer.algorithm.trim().is_empty() {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::EmptySignerIdentity, Some(ordinal), Some(key_id.clone()), "rotation signer identity and algorithm must not be empty");
        }
        let signature = match decode_hex(&statement.signature_hex) {
            Some(value) if !value.is_empty() => value,
            _ => {
                issue(report, CalibrationPublicationWitnessPolicyIssueCode::InvalidSignatureHex, Some(ordinal), Some(key_id.clone()), "rotation signature is not non-empty hexadecimal");
                Vec::new()
            }
        };
        if statement.envelope_sha256
            != calibration_signed_publication_witness_policy_rotation_sha256(statement)
        {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::RotationEnvelopeSha256Mismatch, Some(ordinal), Some(key_id.clone()), "signed rotation-envelope SHA-256 mismatch");
        }
        if !seen.insert(key_id.clone()) {
            issue(report, CalibrationPublicationWitnessPolicyIssueCode::DuplicateRotationSigner, Some(ordinal), Some(key_id.clone()), "rotation side contains duplicate signer identity");
        }
        if !accepted.contains(&key_id) {
            issue(
                report,
                if outgoing {
                    CalibrationPublicationWitnessPolicyIssueCode::UnacceptedOutgoingSigner
                } else {
                    CalibrationPublicationWitnessPolicyIssueCode::UnacceptedIncomingSigner
                },
                Some(ordinal),
                Some(key_id.clone()),
                "rotation signer is not accepted by this policy",
            );
        }
        if let Some(verifier) = verifier {
            if accepted.contains(&key_id)
                && !signature.is_empty()
                && verifier
                    .verify(&payload.canonical_bytes(), &statement.signer, &signature)
                    .is_ok()
            {
                authenticated.insert(key_id);
            } else if accepted.contains(&key_id) {
                issue(
                    report,
                    if outgoing {
                        CalibrationPublicationWitnessPolicyIssueCode::OutgoingSignatureRejected
                    } else {
                        CalibrationPublicationWitnessPolicyIssueCode::IncomingSignatureRejected
                    },
                    Some(ordinal),
                    Some(key_id),
                    "external rotation signature verifier rejected the statement",
                );
            }
        }
    }
    let threshold_met = authenticated.len() >= policy.minimum_distinct_witnesses as usize;
    if verifier.is_some() && !threshold_met {
        issue(
            report,
            if outgoing {
                CalibrationPublicationWitnessPolicyIssueCode::OutgoingThresholdNotMet
            } else {
                CalibrationPublicationWitnessPolicyIssueCode::IncomingThresholdNotMet
            },
            Some(ordinal),
            None,
            "authenticated rotation quorum did not meet the policy threshold",
        );
    }
    verifier.is_some() && threshold_met
}

fn checkpoint_structurally_valid(checkpoint: &CalibrationPublicationCatalogCheckpoint) -> bool {
    checkpoint.checkpoint_version == CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION
        && checkpoint.checkpoint_sha256
            == calibration_publication_catalog_checkpoint_sha256(checkpoint)
        && !checkpoint.catalog_id.trim().is_empty()
        && !checkpoint.authority_id.trim().is_empty()
}

fn hash_envelopes(
    hash: &mut Sha256,
    statements: &[CalibrationSignedPublicationWitnessPolicyRotation],
) {
    let mut statements = statements.to_vec();
    statements.sort_by(|left, right| {
        left.signer
            .key_id
            .cmp(&right.signer.key_id)
            .then_with(|| left.envelope_sha256.cmp(&right.envelope_sha256))
    });
    hash.update(&(statements.len() as u64).to_le_bytes());
    for statement in statements {
        hash_field(hash, &statement.envelope_sha256);
    }
}

fn issue(
    report: &mut CalibrationPublicationWitnessPolicyAuditReport,
    code: CalibrationPublicationWitnessPolicyIssueCode,
    ordinal: Option<u64>,
    signer_key_id: Option<String>,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationWitnessPolicyIssue {
        code,
        ordinal,
        signer_key_id,
        detail: detail.into(),
    });
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

fn hash_optional_field(hash: &mut Sha256, value: Option<&str>) {
    match value {
        None => hash.update(&[0]),
        Some(value) => {
            hash.update(&[1]);
            hash_field(hash, value);
        }
    }
}

fn push_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 {
        return None;
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let high = hex_nibble(bytes[index])?;
        let low = hex_nibble(bytes[index + 1])?;
        output.push((high << 4) | low);
        index += 2;
    }
    Some(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

struct NeverVerifier;

impl CalibrationPublicationWitnessPolicyRotationVerifier for NeverVerifier {
    type Error = &'static str;

    fn verify(
        &self,
        _payload: &[u8],
        _signer: &CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Err("verification not attempted")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptVerifier;

    impl CalibrationPublicationWitnessPolicyRotationVerifier for AcceptVerifier {
        type Error = &'static str;

        fn verify(
            &self,
            _payload: &[u8],
            _signer: &CalibrationSignerIdentity,
            _signature: &[u8],
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    fn checkpoint(event_count: u64, issued_epoch: u64) -> CalibrationPublicationCatalogCheckpoint {
        let mut checkpoint = CalibrationPublicationCatalogCheckpoint {
            checkpoint_version: CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION.into(),
            catalog_id: "catalog".into(),
            authority_id: "authority".into(),
            catalog_version: "catalog-v1".into(),
            catalog_sha256: format!("{:064x}", event_count + 1),
            record_count: event_count,
            event_count,
            head_event_sha256: Some(format!("{:064x}", event_count + 2)),
            previous_checkpoint_sha256: None,
            issued_epoch,
            checkpoint_sha256: String::new(),
        };
        checkpoint.checkpoint_sha256 = calibration_publication_catalog_checkpoint_sha256(&checkpoint);
        checkpoint
    }

    fn policy(threshold: u64, keys: &[&str]) -> CalibrationPublicationCheckpointWitnessPolicy {
        let mut accepted_key_ids = keys.iter().map(|value| (*value).to_string()).collect::<Vec<_>>();
        accepted_key_ids.sort();
        let mut policy = CalibrationPublicationCheckpointWitnessPolicy {
            policy_version: CALIBRATION_PUBLICATION_CHECKPOINT_WITNESS_POLICY_VERSION.into(),
            minimum_distinct_witnesses: threshold,
            accepted_key_ids,
            policy_sha256: String::new(),
        };
        policy.policy_sha256 = calibration_publication_checkpoint_witness_policy_sha256(&policy);
        policy
    }

    fn signer(key_id: &str) -> CalibrationSignerIdentity {
        CalibrationSignerIdentity {
            key_id: key_id.into(),
            algorithm: "test".into(),
            issuer: None,
        }
    }

    #[test]
    fn genesis_selects_active_policy() {
        let ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(1, 1),
            policy(1, &["a"]),
            1,
        )
        .expect("genesis");
        let active = active_calibration_publication_witness_policy_epoch(&ledger, &checkpoint(4, 4))
            .expect("active policy");
        assert_eq!(active.ordinal, 0);
        assert!(audit_calibration_publication_witness_policy_ledger(&ledger).valid());
    }

    #[test]
    fn dual_quorum_rotation_is_accepted() {
        let mut ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(1, 1),
            policy(1, &["a"]),
            1,
        )
        .expect("genesis");
        let (epoch, payload) = plan_calibration_publication_witness_policy_rotation(
            &ledger,
            checkpoint(3, 3),
            policy(1, &["b"]),
            3,
        )
        .expect("plan");
        let outgoing = build_calibration_signed_publication_witness_policy_rotation(
            payload.clone(),
            signer("a"),
            &[1],
        );
        let incoming = build_calibration_signed_publication_witness_policy_rotation(
            payload.clone(),
            signer("b"),
            &[2],
        );
        let set = build_calibration_publication_witness_policy_rotation_set(
            &payload,
            &ledger.epochs[0].policy,
            &epoch.policy,
            vec![outgoing],
            vec![incoming],
        );
        append_calibration_publication_witness_policy_rotation(
            &mut ledger,
            epoch,
            set,
            &AcceptVerifier,
        )
        .expect("append");
        let report = verify_calibration_publication_witness_policy_ledger(&ledger, &AcceptVerifier);
        assert!(report.accepted(), "{:?}", report.issues);
        assert_eq!(
            active_calibration_publication_witness_policy_epoch(&ledger, &checkpoint(5, 5))
                .expect("active")
                .ordinal,
            1
        );
    }

    #[test]
    fn incoming_quorum_is_required() {
        let mut ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(1, 1),
            policy(1, &["a"]),
            1,
        )
        .expect("genesis");
        let (epoch, payload) = plan_calibration_publication_witness_policy_rotation(
            &ledger,
            checkpoint(2, 2),
            policy(1, &["b"]),
            2,
        )
        .expect("plan");
        let outgoing = build_calibration_signed_publication_witness_policy_rotation(
            payload.clone(),
            signer("a"),
            &[1],
        );
        let set = build_calibration_publication_witness_policy_rotation_set(
            &payload,
            &ledger.epochs[0].policy,
            &epoch.policy,
            vec![outgoing],
            Vec::new(),
        );
        assert!(append_calibration_publication_witness_policy_rotation(
            &mut ledger,
            epoch,
            set,
            &AcceptVerifier,
        )
        .is_err());
        assert_eq!(ledger.epochs.len(), 1);
    }

    #[test]
    fn unchanged_policy_rotation_is_rejected() {
        let unchanged = policy(1, &["a"]);
        let ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(1, 1),
            unchanged.clone(),
            1,
        )
        .expect("genesis");
        let result = plan_calibration_publication_witness_policy_rotation(
            &ledger,
            checkpoint(2, 2),
            unchanged,
            2,
        );
        assert!(matches!(
            result,
            Err(CalibrationPublicationWitnessPolicyError::PolicyUnchanged)
        ));
    }

    #[test]
    fn activation_must_advance_event_count() {
        let ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(2, 2),
            policy(1, &["a"]),
            2,
        )
        .expect("genesis");
        let result = plan_calibration_publication_witness_policy_rotation(
            &ledger,
            checkpoint(2, 3),
            policy(1, &["b"]),
            3,
        );
        assert!(matches!(
            result,
            Err(CalibrationPublicationWitnessPolicyError::ActivationCountRegression)
        ));
    }

    #[test]
    fn ledger_hash_tampering_is_detected() {
        let mut ledger = build_calibration_publication_witness_policy_genesis(
            checkpoint(1, 1),
            policy(1, &["a"]),
            1,
        )
        .expect("genesis");
        ledger.ledger_sha256 = "00".repeat(32);
        let report = audit_calibration_publication_witness_policy_ledger(&ledger);
        assert!(!report.valid());
        assert!(report.issues.iter().any(|issue| {
            issue.code == CalibrationPublicationWitnessPolicyIssueCode::LedgerSha256Mismatch
        }));
    }
}
