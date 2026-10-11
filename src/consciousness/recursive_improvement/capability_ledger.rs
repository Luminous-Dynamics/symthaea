// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound operational self-knowledge for capability and skill claims.
//!
//! A capability claim is separate from evidence that qualifies it. Predictions must be recorded
//! before outcomes, outcomes must bind to the exact subject and forecast, and only pre-approved
//! evaluator identities/revisions on held-out or transfer cases can qualify a claim. Training
//! success remains useful telemetry but cannot grant qualification.
//!
//! Under the optional `identity` feature, signed-receipt ingress verifies an Ed25519 signature
//! using the evaluator key frozen in the ledger policy, checks the SHA-256 of exact evidence bytes,
//! and requires canonical `CapabilityEvidencePayloadV1` fields to match the signed receipt. The
//! verifier contract is `capability-evidence-envelope-v2`. It does not retrieve runner artifacts
//! or independently prove that a real run occurred; evaluator-key custody and evaluator truthfulness
//! remain in the trust model.
//! Raw receipt data cannot be admitted to qualification directly, and a digest string is not proof.
//!
//! This first version measures prospective success-probability calibration. Expected compute cost
//! and realized compute are retained for analysis, but do not yet participate in qualification.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

const CAPABILITY_RECEIPT_VERIFIER_IDENTITY: &str = "symthaea-capability-receipt-verifier";
const CAPABILITY_RECEIPT_VERIFIER_REVISION: &str = "capability-evidence-envelope-v2";

/// Exact software/evaluation subject to which all forecasts and outcomes are bound.
///
/// Digest fields are supplied by the build/evaluation authority, not derived by this type.
/// SHA-1- or SHA-256-sized Git object IDs are accepted for source identities to support both
/// repository object formats; configuration, evaluation-profile, claim, and receipt digests use
/// SHA-256.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectIdentity {
    pub source_commit: String,
    pub source_tree: String,
    pub configuration_sha256: String,
    pub evaluation_profile_sha256: String,
    /// Optional only when the subject genuinely has no separate model/data artifact bundle.
    pub model_data_sha256: Option<String>,
}

impl SubjectIdentity {
    pub fn validate(&self) -> Result<(), CapabilityLedgerError> {
        if !is_hex_digest(&self.source_commit, &[40, 64])
            || !is_hex_digest(&self.source_tree, &[40, 64])
            || !is_hex_digest(&self.configuration_sha256, &[64])
            || !is_hex_digest(&self.evaluation_profile_sha256, &[64])
            || self
                .model_data_sha256
                .as_ref()
                .is_some_and(|value| !is_hex_digest(value, &[64]))
        {
            return Err(CapabilityLedgerError::InvalidSubjectIdentity);
        }
        Ok(())
    }
}

/// Where a forecast is evaluated. Only held-out and transfer cases can qualify a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilitySplit {
    Training,
    HeldOut,
    Transfer,
    Diagnostic,
}

impl CapabilitySplit {
    fn may_qualify(self) -> bool {
        matches!(self, Self::HeldOut | Self::Transfer)
    }
}

/// Lifecycle of an operational capability claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityLifecycle {
    Proposed,
    Discovered,
    Candidate,
    Qualified,
    Restricted,
    Suspended,
    Retired,
}

/// Versioned, scoped statement about what the subject can do.
///
/// claim_sha256 must be the digest of the canonical claim payload maintained by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityClaim {
    pub claim_id: String,
    pub claim_sha256: String,
    pub description: String,
    pub scope: String,
    pub lifecycle: CapabilityLifecycle,
}

impl CapabilityClaim {
    pub fn new(
        claim_id: impl Into<String>,
        claim_sha256: impl Into<String>,
        description: impl Into<String>,
        scope: impl Into<String>,
    ) -> Result<Self, CapabilityLedgerError> {
        let claim = Self {
            claim_id: claim_id.into(),
            claim_sha256: claim_sha256.into(),
            description: description.into(),
            scope: scope.into(),
            lifecycle: CapabilityLifecycle::Proposed,
        };
        if !non_empty(&claim.claim_id)
            || !is_hex_digest(&claim.claim_sha256, &[64])
            || !non_empty(&claim.description)
            || !non_empty(&claim.scope)
        {
            return Err(CapabilityLedgerError::InvalidClaim);
        }
        Ok(claim)
    }
}

/// Prospective forecast written before the outcome is known.
///
/// sequence is a monotonically increasing logical sequence supplied by the evaluation transcript;
/// wall-clock timestamps are deliberately not used to establish ordering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityForecast {
    pub forecast_id: String,
    pub claim_id: String,
    pub claim_sha256: String,
    pub subject: SubjectIdentity,
    pub sequence: u64,
    pub context_id: String,
    pub split: CapabilitySplit,
    pub predicted_success_probability: f64,
    pub expected_compute_units: f64,
    pub predicted_failure_mode: Option<String>,
}

impl CapabilityForecast {
    /// Versioned, domain-separated BLAKE3 commitment to the complete forecast struct.
    ///
    /// The field-ordered serde encoding contains no maps; incompatible schema changes must bump
    /// the domain version. This is an integrity commitment, not proof that a forecast was
    /// prospectively registered. That chronology is enforced by the ledger's sequence checks.
    pub fn commitment_blake3(&self) -> Result<String, CapabilityLedgerError> {
        let encoded =
            serde_json::to_vec(self).map_err(|_| CapabilityLedgerError::InvalidForecast)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"MAGI_CAPABILITY_FORECAST_COMMITMENT_V1\0");
        hasher.update(&(encoded.len() as u64).to_be_bytes());
        hasher.update(&encoded);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

/// Untrusted outcome receipt payload. Digest shape and evaluator labels are data, not proof.
/// receipt_sha256 is the SHA-256 of the exact evidence payload bytes supplied for verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityOutcomeReceipt {
    pub forecast_id: String,
    /// BLAKE3 commitment to the complete prospectively registered CapabilityForecast.
    pub forecast_commitment_blake3: String,
    pub subject: SubjectIdentity,
    pub sequence: u64,
    pub observed_success: bool,
    pub split: CapabilitySplit,
    pub evaluator_identity: String,
    pub evaluator_revision: String,
    pub receipt_sha256: String,
    pub actual_compute_units: Option<f64>,
}

/// Version identifier for the canonical outcome-evidence envelope.
pub const CAPABILITY_EVIDENCE_SCHEMA_VERSION: &str = "magi-capability-evidence-v1";

/// Structured outcome evidence covered by the receipt SHA-256.
///
/// The immutable artifact/run identifiers and digest are claims to be resolved by a trusted
/// artifact adapter. This type checks that the envelope is canonical and agrees with the signed
/// receipt; it does not fetch the artifact or prove that the referenced run occurred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityEvidencePayloadV1 {
    pub schema_version: String,
    pub forecast_id: String,
    pub forecast_commitment_blake3: String,
    pub subject: SubjectIdentity,
    /// Outcome event sequence, which must match the signed receipt (not the forecast sequence).
    pub receipt_sequence: u64,
    pub observed_success: bool,
    pub split: CapabilitySplit,
    pub evaluator_identity: String,
    pub evaluator_revision: String,
    pub actual_compute_units: Option<f64>,
    /// Stable identity of the runner implementation that produced the artifact.
    pub runner_identity: String,
    /// Immutable execution/run identity assigned by the runner or CI provider.
    pub runner_run_identity: String,
    /// Immutable artifact identity in the runner's artifact store.
    pub runner_artifact_identity: String,
    /// Expected SHA-256 of the artifact bytes fetched by a trusted adapter.
    pub runner_artifact_sha256: String,
}

impl CapabilityEvidencePayloadV1 {
    /// Serialize this evidence envelope using the versioned field-ordered JSON profile.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CapabilityLedgerError> {
        self.validate_shape()?;
        serde_json::to_vec(self).map_err(|_| CapabilityLedgerError::InvalidEvidencePayload)
    }

    fn decode_canonical(bytes: &[u8]) -> Result<Self, CapabilityLedgerError> {
        if bytes.len() > MAX_CAPABILITY_EVIDENCE_PAYLOAD_BYTES {
            return Err(CapabilityLedgerError::EvidencePayloadTooLarge);
        }
        let evidence: Self = serde_json::from_slice(bytes)
            .map_err(|_| CapabilityLedgerError::InvalidEvidencePayload)?;
        evidence.validate_shape()?;
        if evidence.canonical_bytes()?.as_slice() != bytes {
            return Err(CapabilityLedgerError::NonCanonicalEvidencePayload);
        }
        Ok(evidence)
    }

    fn validate_shape(&self) -> Result<(), CapabilityLedgerError> {
        self.subject.validate()?;
        if self.schema_version != CAPABILITY_EVIDENCE_SCHEMA_VERSION
            || !non_empty(&self.forecast_id)
            || !is_hex_digest(&self.forecast_commitment_blake3, &[64])
            || !non_empty(&self.evaluator_identity)
            || !non_empty(&self.evaluator_revision)
            || self
                .actual_compute_units
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || !non_empty(&self.runner_identity)
            || !non_empty(&self.runner_run_identity)
            || !non_empty(&self.runner_artifact_identity)
            || !is_hex_digest(&self.runner_artifact_sha256, &[64])
        {
            return Err(CapabilityLedgerError::InvalidEvidencePayload);
        }
        Ok(())
    }

    fn validate_against_receipt(
        &self,
        receipt: &CapabilityOutcomeReceipt,
    ) -> Result<(), CapabilityLedgerError> {
        if self.forecast_id != receipt.forecast_id
            || self.forecast_commitment_blake3 != receipt.forecast_commitment_blake3
            || self.subject != receipt.subject
            || self.receipt_sequence != receipt.sequence
            || self.observed_success != receipt.observed_success
            || self.split != receipt.split
            || self.evaluator_identity != receipt.evaluator_identity
            || self.evaluator_revision != receipt.evaluator_revision
            || self.actual_compute_units != receipt.actual_compute_units
        {
            return Err(CapabilityLedgerError::EvidenceReceiptMismatch);
        }
        Ok(())
    }
}

const MAX_CAPABILITY_EVIDENCE_PAYLOAD_BYTES: usize = 1024 * 1024;

/// Signed receipt envelope. The signing key is deliberately not embedded: verification uses the
/// evaluator public key frozen in QualificationPolicy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedCapabilityOutcomeReceipt {
    pub receipt: CapabilityOutcomeReceipt,
    /// Hex-encoded Ed25519 signature over the versioned canonical receipt payload.
    pub signature_hex: String,
}

#[cfg(feature = "identity")]
impl CapabilityOutcomeReceipt {
    /// Return the exact versioned bytes that an evaluator must sign for this receipt.
    ///
    /// Consumers should sign these bytes with the evaluator private key corresponding to
    /// QualificationPolicy::trusted_evaluator_public_key_hex. The signature also binds the
    /// receipt to the SHA-256 digest of the evidence payload supplied during verification.
    pub fn canonical_signing_payload(&self) -> Result<Vec<u8>, CapabilityLedgerError> {
        receipt_verifier::canonical_payload(self)
    }
}

/// Opaque qualification-admission token. It intentionally has private fields and no serde
/// implementation, so callers cannot create one by supplying a digest or deserializing a claim.
/// Only this module and its descendants can issue the token. The feature-gated receipt
/// verifier checks the evaluator signature, signed receipt fields, and evidence-payload digest
/// before issuance. This wrapper is an API boundary; it does not itself prove runner truth.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedCapabilityOutcomeReceipt {
    receipt: CapabilityOutcomeReceipt,
    evaluator_public_key_hex: String,
    verifier_identity: String,
    verifier_revision: String,
    verification_root_sha256: String,
}

impl VerifiedCapabilityOutcomeReceipt {
    fn from_verified_pipeline(
        receipt: CapabilityOutcomeReceipt,
        evaluator_public_key_hex: impl Into<String>,
        verifier_identity: impl Into<String>,
        verifier_revision: impl Into<String>,
        verification_root_sha256: impl Into<String>,
    ) -> Result<Self, CapabilityLedgerError> {
        let verified = Self {
            receipt,
            evaluator_public_key_hex: evaluator_public_key_hex.into(),
            verifier_identity: verifier_identity.into(),
            verifier_revision: verifier_revision.into(),
            verification_root_sha256: verification_root_sha256.into(),
        };
        if !is_hex_digest(&verified.evaluator_public_key_hex, &[64])
            || !non_empty(&verified.verifier_identity)
            || !non_empty(&verified.verifier_revision)
            || !is_hex_digest(&verified.verification_root_sha256, &[64])
        {
            return Err(CapabilityLedgerError::InvalidVerificationProvenance);
        }
        Ok(verified)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityMetrics {
    /// Total forecasts associated with the claim, including unresolved forecasts.
    pub forecast_count: usize,
    pub resolved_count: usize,
    /// Resolved held-out and transfer cases (not training or diagnostics).
    pub qualification_samples: usize,
    pub held_out_samples: usize,
    pub transfer_samples: usize,
    /// None means no resolved observations are available; it is not a zero error score.
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub mean_predicted_success_probability: Option<f64>,
    pub empirical_success_rate: Option<f64>,
    pub mean_absolute_compute_error: Option<f64>,
}

/// Qualification policy frozen into a ledger at construction. The trusted outer evaluation
/// layer must authorize this policy for the exact subject/profile; do not let the claimant select
/// evaluator keys, allowlists, or score thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationPolicy {
    pub minimum_resolved: usize,
    pub minimum_held_out: usize,
    pub maximum_brier_score: f64,
    pub maximum_expected_calibration_error: f64,
    pub trusted_evaluator_identity: String,
    pub trusted_evaluator_revisions: Vec<String>,
    /// Hex-encoded Ed25519 public key pinned by the trusted evaluation layer.
    pub trusted_evaluator_public_key_hex: String,
    /// Exact independent receipt-verifier identity/revision accepted for qualification.
    pub trusted_verifier_identity: String,
    pub trusted_verifier_revisions: Vec<String>,
}

impl QualificationPolicy {
    pub fn validate(&self) -> Result<(), CapabilityLedgerError> {
        if self.minimum_resolved == 0
            || self.minimum_held_out == 0
            || !self.maximum_brier_score.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_brier_score)
            || !self.maximum_expected_calibration_error.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_expected_calibration_error)
            || !non_empty(&self.trusted_evaluator_identity)
            || self.trusted_evaluator_revisions.is_empty()
            || self.trusted_evaluator_revisions.iter().any(|r| !non_empty(r))
            || !is_hex_digest(&self.trusted_evaluator_public_key_hex, &[64])
            || !non_empty(&self.trusted_verifier_identity)
            || self.trusted_verifier_revisions.is_empty()
            || self.trusted_verifier_revisions.iter().any(|r| !non_empty(r))
        {
            return Err(CapabilityLedgerError::InvalidQualificationPolicy);
        }
        Ok(())
    }

    fn accepts(&self, verified: &VerifiedCapabilityOutcomeReceipt) -> bool {
        let receipt = &verified.receipt;
        receipt.evaluator_identity == self.trusted_evaluator_identity
            && self
                .trusted_evaluator_revisions
                .iter()
                .any(|revision| revision == &receipt.evaluator_revision)
            && verified.evaluator_public_key_hex.eq_ignore_ascii_case(
                &self.trusted_evaluator_public_key_hex,
            )
            && verified.verifier_identity == self.trusted_verifier_identity
            && self
                .trusted_verifier_revisions
                .iter()
                .any(|revision| revision == &verified.verifier_revision)
            && receipt.split.may_qualify()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityLedgerError {
    InvalidSubjectIdentity,
    InvalidClaim,
    DuplicateClaim,
    UnknownClaim,
    InvalidLifecycleTransition,
    InvalidForecast,
    DuplicateForecast,
    UnknownForecast,
    AlreadyResolved,
    SubjectMismatch,
    ClaimVersionMismatch,
    SequenceNotMonotonic,
    SplitMismatch,
    InvalidOutcomeReceipt,
    ForecastCommitmentMismatch,
    InvalidReceiptSignature,
    EvidenceRootMismatch,
    EvidencePayloadTooLarge,
    InvalidEvidencePayload,
    NonCanonicalEvidencePayload,
    EvidenceReceiptMismatch,
    InvalidVerificationProvenance,
    ReceiptReplay,
    InvalidQualificationPolicy,
    InsufficientEvidence,
    QualificationRejected,
}

impl fmt::Display for CapabilityLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CapabilityLedgerError {}

#[derive(Debug, Clone)]
struct ForecastRecord {
    forecast: CapabilityForecast,
    forecast_commitment_blake3: String,
    receipt: Option<VerifiedCapabilityOutcomeReceipt>,
}

/// In-memory ledger pinned to one exact subject identity.
#[derive(Debug, Clone)]
pub struct CapabilityLedger {
    subject: SubjectIdentity,
    /// Frozen at construction so callers cannot loosen thresholds after observing outcomes.
    /// The outer trusted evaluation layer must supply the policy authorized for this subject.
    qualification_policy: QualificationPolicy,
    claims: BTreeMap<String, CapabilityClaim>,
    forecasts: BTreeMap<String, ForecastRecord>,
    /// Most recent accepted forecast/outcome event sequence for the whole ledger.
    last_sequence: u64,
}

impl CapabilityLedger {
    pub fn new(
        subject: SubjectIdentity,
        qualification_policy: QualificationPolicy,
    ) -> Result<Self, CapabilityLedgerError> {
        subject.validate()?;
        qualification_policy.validate()?;
        Ok(Self {
            subject,
            qualification_policy,
            claims: BTreeMap::new(),
            forecasts: BTreeMap::new(),
            last_sequence: 0,
        })
    }

    pub fn subject(&self) -> &SubjectIdentity {
        &self.subject
    }

    pub fn add_claim(&mut self, claim: CapabilityClaim) -> Result<(), CapabilityLedgerError> {
        if self.claims.contains_key(&claim.claim_id) {
            return Err(CapabilityLedgerError::DuplicateClaim);
        }
        // Re-validate even if the claim was deserialized rather than built with new().
        let validated = CapabilityClaim::new(
            claim.claim_id.clone(),
            claim.claim_sha256.clone(),
            claim.description.clone(),
            claim.scope.clone(),
        )?;
        self.claims.insert(
            validated.claim_id.clone(),
            CapabilityClaim {
                lifecycle: CapabilityLifecycle::Proposed,
                ..validated
            },
        );
        Ok(())
    }

    pub fn claim(&self, claim_id: &str) -> Option<&CapabilityClaim> {
        self.claims.get(claim_id)
    }

    /// Advance a non-qualification lifecycle transition. A caller cannot set Qualified directly.
    pub fn transition_claim(
        &mut self,
        claim_id: &str,
        next: CapabilityLifecycle,
    ) -> Result<(), CapabilityLedgerError> {
        if next == CapabilityLifecycle::Qualified {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        let claim = self
            .claims
            .get_mut(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        let allowed = match claim.lifecycle {
            CapabilityLifecycle::Proposed => {
                matches!(next, CapabilityLifecycle::Discovered | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Discovered => {
                matches!(next, CapabilityLifecycle::Candidate | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Candidate => matches!(
                next,
                CapabilityLifecycle::Restricted
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Qualified => matches!(
                next,
                CapabilityLifecycle::Restricted
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Restricted => matches!(
                next,
                CapabilityLifecycle::Candidate
                    | CapabilityLifecycle::Suspended
                    | CapabilityLifecycle::Retired
            ),
            CapabilityLifecycle::Suspended => {
                matches!(next, CapabilityLifecycle::Candidate | CapabilityLifecycle::Retired)
            }
            CapabilityLifecycle::Retired => false,
        };
        if !allowed {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        claim.lifecycle = next;
        Ok(())
    }

    /// Admit a forecast bound to the exact subject and immutable claim version.
    pub fn record_forecast(
        &mut self,
        forecast: CapabilityForecast,
    ) -> Result<(), CapabilityLedgerError> {
        if !non_empty(&forecast.forecast_id)
            || !non_empty(&forecast.context_id)
            || !is_valid_probability(forecast.predicted_success_probability)
            || !forecast.expected_compute_units.is_finite()
            || forecast.expected_compute_units < 0.0
            || forecast
                .predicted_failure_mode
                .as_ref()
                .is_some_and(|reason| !non_empty(reason))
        {
            return Err(CapabilityLedgerError::InvalidForecast);
        }
        forecast.subject.validate()?;
        if forecast.subject != self.subject {
            return Err(CapabilityLedgerError::SubjectMismatch);
        }
        if forecast.sequence <= self.last_sequence {
            return Err(CapabilityLedgerError::SequenceNotMonotonic);
        }
        let claim = self
            .claims
            .get(&forecast.claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        if claim.claim_sha256 != forecast.claim_sha256 {
            return Err(CapabilityLedgerError::ClaimVersionMismatch);
        }
        if !matches!(
            claim.lifecycle,
            CapabilityLifecycle::Candidate
                | CapabilityLifecycle::Qualified
                | CapabilityLifecycle::Restricted
        ) {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        if self.forecasts.contains_key(&forecast.forecast_id) {
            return Err(CapabilityLedgerError::DuplicateForecast);
        }
        let sequence = forecast.sequence;
        let forecast_commitment_blake3 = forecast.commitment_blake3()?;
        self.forecasts.insert(
            forecast.forecast_id.clone(),
            ForecastRecord {
                forecast,
                forecast_commitment_blake3,
                receipt: None,
            },
        );
        self.last_sequence = sequence;
        Ok(())
    }

    /// Resolve a forecast only through the opaque token issued by the independent verifier.
    ///
    /// Raw CapabilityOutcomeReceipt values cannot qualify a claim: matching evaluator labels
    /// and a caller-supplied digest do not authenticate an execution artifact. The identity-feature
    /// ingress verifies a signature and exact evidence bytes before issuing the opaque token;
    /// builds without that feature have no production receipt-admission path.
    pub fn resolve_verified_forecast(
        &mut self,
        verified: VerifiedCapabilityOutcomeReceipt,
    ) -> Result<(), CapabilityLedgerError> {
        let receipt = &verified.receipt;
        receipt.subject.validate()?;
        if receipt.subject != self.subject {
            return Err(CapabilityLedgerError::SubjectMismatch);
        }
        if !non_empty(&receipt.evaluator_identity)
            || !non_empty(&receipt.evaluator_revision)
            || !is_hex_digest(&receipt.forecast_commitment_blake3, &[64])
            || !is_hex_digest(&receipt.receipt_sha256, &[64])
            || receipt
                .actual_compute_units
                .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
        {
            return Err(CapabilityLedgerError::InvalidOutcomeReceipt);
        }
        if !non_empty(&verified.verifier_identity)
            || !non_empty(&verified.verifier_revision)
            || !is_hex_digest(&verified.verification_root_sha256, &[64])
        {
            return Err(CapabilityLedgerError::InvalidVerificationProvenance);
        }
        if receipt.sequence <= self.last_sequence {
            return Err(CapabilityLedgerError::SequenceNotMonotonic);
        }
        {
            let record = self
                .forecasts
                .get(&receipt.forecast_id)
                .ok_or(CapabilityLedgerError::UnknownForecast)?;
            if record.receipt.is_some() {
                return Err(CapabilityLedgerError::AlreadyResolved);
            }
            if receipt.sequence <= record.forecast.sequence {
                return Err(CapabilityLedgerError::SequenceNotMonotonic);
            }
            if receipt.split != record.forecast.split {
                return Err(CapabilityLedgerError::SplitMismatch);
            }
            if receipt.subject != record.forecast.subject {
                return Err(CapabilityLedgerError::SubjectMismatch);
            }
            if receipt.forecast_commitment_blake3 != record.forecast_commitment_blake3 {
                return Err(CapabilityLedgerError::ForecastCommitmentMismatch);
            }
        }
        if self.forecasts.values().any(|existing| {
            existing.receipt.as_ref().is_some_and(|prior| {
                prior.receipt.receipt_sha256 == receipt.receipt_sha256
                    || prior.verification_root_sha256 == verified.verification_root_sha256
            })
        }) {
            return Err(CapabilityLedgerError::ReceiptReplay);
        }
        let sequence = receipt.sequence;
        let forecast_id = receipt.forecast_id.clone();
        self.forecasts
            .get_mut(&forecast_id)
            .ok_or(CapabilityLedgerError::UnknownForecast)?
            .receipt = Some(verified);
        self.last_sequence = sequence;
        Ok(())
    }

    /// Verify and admit a runner-signed receipt under this ledger's frozen qualification policy.
    ///
    /// Available only with the existing `identity` feature. The verifier checks the policy-pinned
    /// Ed25519 key and versioned receipt signature, hashes the exact evidence bytes, and requires a
    /// canonical, size-bounded CapabilityEvidencePayloadV1 whose outcome fields match the receipt.
    /// This authenticates the signed claim and envelope consistency, but does not independently
    /// prove artifact lineage or that the evaluator's runner produced truthful data.
    #[cfg(feature = "identity")]
    pub fn verify_and_resolve_signed_receipt(
        &mut self,
        signed: SignedCapabilityOutcomeReceipt,
        evidence_payload: &[u8],
    ) -> Result<(), CapabilityLedgerError> {
        let verified = receipt_verifier::verify_and_issue(
            &signed,
            evidence_payload,
            &self.qualification_policy,
        )?;
        self.resolve_verified_forecast(verified)
    }

    /// Return descriptive metrics for all resolved evidence attached to a claim.
    ///
    /// Qualification decisions use a stricter, policy-filtered subset via qualification_metrics.
    pub fn metrics(&self, claim_id: &str) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        if !self.claims.contains_key(claim_id) {
            return Err(CapabilityLedgerError::UnknownClaim);
        }
        let records: Vec<&ForecastRecord> = self
            .forecasts
            .values()
            .filter(|record| record.forecast.claim_id == claim_id)
            .collect();
        Ok(calculate_metrics(&records, records.len()))
    }

    /// Metrics computed only from resolved held-out/transfer receipts accepted by the
    /// immutable qualification policy supplied at ledger construction.
    pub fn qualification_metrics(
        &self,
        claim_id: &str,
    ) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        if !self.claims.contains_key(claim_id) {
            return Err(CapabilityLedgerError::UnknownClaim);
        }
        let all_records: Vec<&ForecastRecord> = self
            .forecasts
            .values()
            .filter(|record| record.forecast.claim_id == claim_id)
            .collect();
        let total_forecast_count = all_records.len();
        let qualifying_records: Vec<&ForecastRecord> = all_records
            .into_iter()
            .filter(|record| {
                record
                    .receipt
                    .as_ref()
                    .is_some_and(|receipt| self.qualification_policy.accepts(receipt))
            })
            .collect();
        Ok(calculate_metrics(&qualifying_records, total_forecast_count))
    }

    /// Promote a candidate only when a pinned evaluator policy and adequate out-of-training
    /// evidence satisfy both outcome-score and calibration-error gates.
    pub fn qualify_claim(
        &mut self,
        claim_id: &str,
    ) -> Result<CapabilityMetrics, CapabilityLedgerError> {
        let claim = self
            .claims
            .get(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?;
        if claim.lifecycle != CapabilityLifecycle::Candidate {
            return Err(CapabilityLedgerError::InvalidLifecycleTransition);
        }
        let metrics = self.qualification_metrics(claim_id)?;
        if metrics.resolved_count < self.qualification_policy.minimum_resolved
            || metrics.held_out_samples < self.qualification_policy.minimum_held_out
            || metrics.qualification_samples < self.qualification_policy.minimum_resolved
        {
            return Err(CapabilityLedgerError::InsufficientEvidence);
        }
        if metrics
            .brier_score
            .map_or(true, |score| score > self.qualification_policy.maximum_brier_score)
            || metrics
                .expected_calibration_error
                .map_or(true, |ece| {
                    ece > self.qualification_policy.maximum_expected_calibration_error
                })
        {
            return Err(CapabilityLedgerError::QualificationRejected);
        }
        self.claims
            .get_mut(claim_id)
            .ok_or(CapabilityLedgerError::UnknownClaim)?
            .lifecycle = CapabilityLifecycle::Qualified;
        Ok(metrics)
    }
}

#[cfg(feature = "identity")]
mod receipt_verifier {
    use super::*;
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use sha2::{Digest, Sha256};

    /// Domain-separated canonical bytes for the versioned signed receipt contract. The payload
    /// is a field-ordered struct without maps; incompatible schema changes must bump this version.
    pub(super) fn canonical_payload(
        receipt: &CapabilityOutcomeReceipt,
    ) -> Result<Vec<u8>, CapabilityLedgerError> {
        let encoded = serde_json::to_vec(receipt)
            .map_err(|_| CapabilityLedgerError::InvalidOutcomeReceipt)?;
        let mut payload = Vec::with_capacity(40 + encoded.len());
        payload.extend_from_slice(b"MAGI_CAPABILITY_OUTCOME_RECEIPT_V1\0");
        payload.extend_from_slice(&(encoded.len() as u64).to_be_bytes());
        payload.extend_from_slice(&encoded);
        Ok(payload)
    }

    pub(super) fn verify_and_issue(
        signed: &SignedCapabilityOutcomeReceipt,
        evidence_payload: &[u8],
        policy: &QualificationPolicy,
    ) -> Result<VerifiedCapabilityOutcomeReceipt, CapabilityLedgerError> {
        policy.validate()?;
        if signed.receipt.evaluator_identity != policy.trusted_evaluator_identity
            || !policy.trusted_evaluator_revisions
                .iter()
                .any(|revision| revision == &signed.receipt.evaluator_revision)
        {
            return Err(CapabilityLedgerError::InvalidOutcomeReceipt);
        }

        let public_key_bytes = hex::decode(&policy.trusted_evaluator_public_key_hex)
            .map_err(|_| CapabilityLedgerError::InvalidQualificationPolicy)?;
        let public_key_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| CapabilityLedgerError::InvalidQualificationPolicy)?;
        let public_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|_| CapabilityLedgerError::InvalidQualificationPolicy)?;
        let signature_bytes = hex::decode(&signed.signature_hex)
            .map_err(|_| CapabilityLedgerError::InvalidReceiptSignature)?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| CapabilityLedgerError::InvalidReceiptSignature)?;

        if evidence_payload.len() > MAX_CAPABILITY_EVIDENCE_PAYLOAD_BYTES {
            return Err(CapabilityLedgerError::EvidencePayloadTooLarge);
        }
        let evidence_root = Sha256::digest(evidence_payload);
        if signed.receipt.receipt_sha256 != hex::encode(&evidence_root) {
            return Err(CapabilityLedgerError::EvidenceRootMismatch);
        }

        let payload = canonical_payload(&signed.receipt)?;
        public_key
            .verify(&payload, &signature)
            .map_err(|_| CapabilityLedgerError::InvalidReceiptSignature)?;

        // A valid signature only authenticates the evaluator statement. Require the evidence
        // bytes themselves to follow the exact schema and agree with the signed outcome fields.
        let evidence = CapabilityEvidencePayloadV1::decode_canonical(evidence_payload)?;
        evidence.validate_against_receipt(&signed.receipt)?;

        // Bind the verifier receipt to the exact canonical bytes, signature, evidence root, and
        // policy-pinned evaluator key. Reusing either root for another outcome is rejected.
        let mut root = Sha256::new();
        root.update(b"MAGI_CAPABILITY_VERIFICATION_ROOT_V1\0");
        root.update((payload.len() as u64).to_be_bytes());
        root.update(&payload);
        root.update(signature.to_bytes());
        root.update(evidence_root);
        root.update(public_key.as_bytes());
        let verification_root = hex::encode(root.finalize());

        VerifiedCapabilityOutcomeReceipt::from_verified_pipeline(
            signed.receipt.clone(),
            hex::encode(public_key.as_bytes()),
            CAPABILITY_RECEIPT_VERIFIER_IDENTITY,
            CAPABILITY_RECEIPT_VERIFIER_REVISION,
            verification_root,
        )
    }
}

fn calculate_metrics(
    records: &[&ForecastRecord],
    total_forecast_count: usize,
) -> CapabilityMetrics {
    let resolved: Vec<(&CapabilityForecast, &VerifiedCapabilityOutcomeReceipt)> = records
        .iter()
        .filter_map(|record| {
            record
                .receipt
                .as_ref()
                .map(|receipt| (&record.forecast, receipt))
        })
        .collect();

    let resolved_count = resolved.len();
    let held_out_samples = resolved
        .iter()
        .filter(|(_, verified)| verified.receipt.split == CapabilitySplit::HeldOut)
        .count();
    let transfer_samples = resolved
        .iter()
        .filter(|(_, verified)| verified.receipt.split == CapabilitySplit::Transfer)
        .count();
    let qualification_samples = held_out_samples + transfer_samples;

    if resolved.is_empty() {
        return CapabilityMetrics {
            forecast_count: total_forecast_count,
            resolved_count,
            qualification_samples,
            held_out_samples,
            transfer_samples,
            brier_score: None,
            expected_calibration_error: None,
            mean_predicted_success_probability: None,
            empirical_success_rate: None,
            mean_absolute_compute_error: None,
        };
    }

    let n = resolved_count as f64;
    let brier_score = resolved
        .iter()
        .map(|(forecast, receipt)| {
            let observed = if receipt.receipt.observed_success { 1.0 } else { 0.0 };
            (forecast.predicted_success_probability - observed).powi(2)
        })
        .sum::<f64>()
        / n;

    // Ten fixed probability bins. Empty bins contribute zero weight; ECE is unavailable rather
    // than zero when there are no resolved outcomes.
    let mut bins = [(0usize, 0.0f64, 0.0f64); 10];
    for (forecast, receipt) in &resolved {
        let p = forecast.predicted_success_probability;
        let idx = ((p * 10.0).floor() as usize).min(9);
        bins[idx].0 += 1;
        bins[idx].1 += p;
        bins[idx].2 += if receipt.receipt.observed_success { 1.0 } else { 0.0 };
    }
    let ece = bins
        .iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, p_sum, observed_sum)| {
            let count_f = *count as f64;
            (count_f / n) * ((p_sum / count_f) - (observed_sum / count_f)).abs()
        })
        .sum::<f64>();

    let compute_errors: Vec<f64> = resolved
        .iter()
        .filter_map(|(forecast, receipt)| {
            receipt
                .receipt
                .actual_compute_units
                .map(|actual| (forecast.expected_compute_units - actual).abs())
        })
        .collect();

    CapabilityMetrics {
        forecast_count: total_forecast_count,
        resolved_count,
        qualification_samples,
        held_out_samples,
        transfer_samples,
        brier_score: Some(brier_score),
        expected_calibration_error: Some(ece),
        mean_predicted_success_probability: Some(
            resolved
                .iter()
                .map(|(forecast, _)| forecast.predicted_success_probability)
                .sum::<f64>()
                / n,
        ),
        empirical_success_rate: Some(
            resolved
                .iter()
                .filter(|(_, receipt)| receipt.receipt.observed_success)
                .count() as f64
                / n,
        ),
        mean_absolute_compute_error: if compute_errors.is_empty() {
            None
        } else {
            Some(compute_errors.iter().sum::<f64>() / compute_errors.len() as f64)
        },
    }
}

fn is_valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_hex_digest(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> SubjectIdentity {
        SubjectIdentity {
            source_commit: "a".repeat(40),
            source_tree: "b".repeat(40),
            configuration_sha256: "c".repeat(64),
            evaluation_profile_sha256: "d".repeat(64),
            model_data_sha256: Some("e".repeat(64)),
        }
    }

    fn setup_claim() -> CapabilityLedger {
        setup_claim_with_policy(policy())
    }

    fn setup_claim_with_policy(qualification_policy: QualificationPolicy) -> CapabilityLedger {
        let mut ledger = CapabilityLedger::new(subject(), qualification_policy).unwrap();
        ledger
            .add_claim(
                CapabilityClaim::new(
                    "rust-debugging",
                    "f".repeat(64),
                    "Diagnose bounded Rust compiler failures",
                    "Rust 1.96 workspace, declared dependencies and frozen test profile",
                )
                .unwrap(),
            )
            .unwrap();
        ledger
            .transition_claim("rust-debugging", CapabilityLifecycle::Discovered)
            .unwrap();
        ledger
            .transition_claim("rust-debugging", CapabilityLifecycle::Candidate)
            .unwrap();
        ledger
    }

    fn policy() -> QualificationPolicy {
        QualificationPolicy {
            minimum_resolved: 3,
            minimum_held_out: 3,
            maximum_brier_score: 0.05,
            maximum_expected_calibration_error: 0.15,
            trusted_evaluator_identity: "independent-qualification-runner".to_string(),
            trusted_evaluator_revisions: vec!["v1".to_string()],
            trusted_evaluator_public_key_hex: "a".repeat(64),
            trusted_verifier_identity: CAPABILITY_RECEIPT_VERIFIER_IDENTITY.to_string(),
            trusted_verifier_revisions: vec![CAPABILITY_RECEIPT_VERIFIER_REVISION.to_string()],
        }
    }

    fn test_verified_receipt(
        ledger: &CapabilityLedger,
        receipt: CapabilityOutcomeReceipt,
    ) -> VerifiedCapabilityOutcomeReceipt {
        test_verified_receipt_from(
            ledger,
            receipt,
            CAPABILITY_RECEIPT_VERIFIER_IDENTITY,
            CAPABILITY_RECEIPT_VERIFIER_REVISION,
        )
    }

    fn test_verified_receipt_from(
        ledger: &CapabilityLedger,
        mut receipt: CapabilityOutcomeReceipt,
        verifier_identity: &str,
        verifier_revision: &str,
    ) -> VerifiedCapabilityOutcomeReceipt {
        receipt.forecast_commitment_blake3 = ledger
            .forecasts
            .get(&receipt.forecast_id)
            .expect("test receipt must name a previously recorded forecast")
            .forecast_commitment_blake3
            .clone();
        // Synthetic unit-test admission only. Production issuance is isolated in the
        // identity-feature receipt verifier; this helper must never be used outside tests.
        // The sequence makes fixture roots unique within a ledger unless a test intentionally
        // reuses a receipt root to exercise replay rejection.
        let verification_root = format!("{:064x}", receipt.sequence);
        VerifiedCapabilityOutcomeReceipt::from_verified_pipeline(
            receipt,
            "a".repeat(64),
            verifier_identity,
            verifier_revision,
            verification_root,
        )
        .expect("fixed synthetic verifier provenance is structurally valid")
    }

    fn add_case(
        ledger: &mut CapabilityLedger,
        id: &str,
        forecast_seq: u64,
        split: CapabilitySplit,
        probability: f64,
        actual: bool,
    ) {
        let forecast = CapabilityForecast {
            forecast_id: id.to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: forecast_seq,
            context_id: format!("frozen-context-{id}"),
            split,
            predicted_success_probability: probability,
            expected_compute_units: 10.0,
            predicted_failure_mode: Some("Missing compiler feature context".to_string()),
        };
        ledger.record_forecast(forecast).unwrap();

        let receipt = CapabilityOutcomeReceipt {
            forecast_id: id.to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: forecast_seq + 1,
            observed_success: actual,
            split,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: format!("{:064x}", forecast_seq + 1),
            actual_compute_units: Some(12.0),
        };
        let verified = test_verified_receipt(ledger, receipt);
        ledger.resolve_verified_forecast(verified).unwrap();
    }

    #[test]
    fn training_success_cannot_qualify_a_skill() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("train-{i}"),
                2 * i + 1,
                CapabilitySplit::Training,
                0.9,
                true,
            );
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging"),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
        let qualification_metrics = ledger
            .qualification_metrics("rust-debugging")
            .unwrap();
        assert_eq!(qualification_metrics.forecast_count, 3);
        assert_eq!(qualification_metrics.resolved_count, 0);
        assert_eq!(qualification_metrics.brier_score, None);
        assert_eq!(
            ledger.claim("rust-debugging").unwrap().lifecycle,
            CapabilityLifecycle::Candidate
        );
    }

    #[test]
    fn held_out_evidence_can_qualify_a_calibrated_candidate() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("heldout-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.9,
                true,
            );
        }
        let metrics = ledger.qualify_claim("rust-debugging").unwrap();
        assert_eq!(metrics.resolved_count, 3);
        assert_eq!(metrics.held_out_samples, 3);
        assert!((metrics.brier_score.unwrap() - 0.01).abs() < 1e-9);
        assert_eq!(
            ledger.claim("rust-debugging").unwrap().lifecycle,
            CapabilityLifecycle::Qualified
        );
    }

    #[test]
    fn miscalibrated_held_out_evidence_is_rejected() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("wrong-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.99,
                false,
            );
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging"),
            Err(CapabilityLedgerError::QualificationRejected)
        );
    }

    #[test]
    fn outcome_must_follow_forecast_and_match_split() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "chronology".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 10,
                context_id: "context".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();

        let mut receipt = CapabilityOutcomeReceipt {
            forecast_id: "chronology".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: 10,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "2".repeat(64),
            actual_compute_units: None,
        };
        assert_eq!(
            ledger.resolve_verified_forecast(test_verified_receipt(&ledger, receipt.clone())),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );
        receipt.sequence = 11;
        receipt.split = CapabilitySplit::Training;
        assert_eq!(
            ledger.resolve_verified_forecast(test_verified_receipt(&ledger, receipt)),
            Err(CapabilityLedgerError::SplitMismatch)
        );
    }

    #[test]
    fn subject_mismatch_is_rejected() {
        let mut ledger = setup_claim();
        let mut forecast = CapabilityForecast {
            forecast_id: "wrong-subject".to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "context".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.7,
            expected_compute_units: 1.0,
            predicted_failure_mode: None,
        };
        forecast.subject.source_commit = "9".repeat(40);
        assert_eq!(
            ledger.record_forecast(forecast),
            Err(CapabilityLedgerError::SubjectMismatch)
        );
    }

    #[test]
    fn adding_a_claim_cannot_import_qualified_status() {
        let mut ledger = CapabilityLedger::new(subject(), policy()).unwrap();
        ledger
            .add_claim(CapabilityClaim {
                claim_id: "forged".to_string(),
                claim_sha256: "f".repeat(64),
                description: "Claim with caller-supplied status".to_string(),
                scope: "Test only".to_string(),
                lifecycle: CapabilityLifecycle::Qualified,
            })
            .unwrap();
        assert_eq!(
            ledger.claim("forged").unwrap().lifecycle,
            CapabilityLifecycle::Proposed
        );
    }

    #[test]
    fn outcome_for_different_forecast_payload_is_rejected() {
        let mut ledger = setup_claim();
        ledger.record_forecast(CapabilityForecast {
            forecast_id: "forecast-binding".to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "original-context".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.9,
            expected_compute_units: 5.0,
            predicted_failure_mode: Some("missing source context".to_string()),
        }).unwrap();

        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "forecast-binding".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "b".repeat(64),
            actual_compute_units: Some(5.0),
        };
        let verified = VerifiedCapabilityOutcomeReceipt::from_verified_pipeline(
            receipt,
            "a".repeat(64),
            CAPABILITY_RECEIPT_VERIFIER_IDENTITY,
            CAPABILITY_RECEIPT_VERIFIER_REVISION,
            "c".repeat(64),
        ).unwrap();
        assert_eq!(
            ledger.resolve_verified_forecast(verified),
            Err(CapabilityLedgerError::ForecastCommitmentMismatch)
        );
        assert_eq!(ledger.metrics("rust-debugging").unwrap().resolved_count, 0);
    }

    #[test]
    fn forecast_commitment_changes_when_claim_probability_or_context_changes() {
        let baseline = CapabilityForecast {
            forecast_id: "forecast-1".to_string(),
            claim_id: "claim-1".to_string(),
            claim_sha256: "a".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "context-a".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.75,
            expected_compute_units: 10.0,
            predicted_failure_mode: Some("missing compiler metadata".to_string()),
        };
        let baseline_commitment = baseline.commitment_blake3().unwrap();

        let mut changed = baseline.clone();
        changed.claim_sha256 = "b".repeat(64);
        assert_ne!(baseline_commitment, changed.commitment_blake3().unwrap());

        let mut changed = baseline.clone();
        changed.predicted_success_probability = 0.74;
        assert_ne!(baseline_commitment, changed.commitment_blake3().unwrap());

        let mut changed = baseline.clone();
        changed.context_id = "context-b".to_string();
        assert_ne!(baseline_commitment, changed.commitment_blake3().unwrap());

        let mut changed = baseline.clone();
        changed.expected_compute_units = 11.0;
        assert_ne!(baseline_commitment, changed.commitment_blake3().unwrap());
    }

    #[cfg(feature = "identity")]
    #[test]
    fn signed_receipt_requires_pinned_signature_and_exact_evidence_bytes() {
        use ed25519_dalek::{Signer, SigningKey};
        use sha2::{Digest, Sha256};

        let key = SigningKey::from_bytes(&[7u8; 32]);
        let mut pinned_policy = policy();
        pinned_policy.trusted_evaluator_public_key_hex =
            hex::encode(key.verifying_key().as_bytes());
        let mut ledger = setup_claim_with_policy(pinned_policy);

        ledger.record_forecast(CapabilityForecast {
            forecast_id: "signed-forecast".to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "signed-context".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.9,
            expected_compute_units: 2.0,
            predicted_failure_mode: None,
        }).unwrap();

        let forecast_commitment_blake3 = ledger
            .forecasts
            .get("signed-forecast")
            .unwrap()
            .forecast_commitment_blake3
            .clone();
        let evidence = CapabilityEvidencePayloadV1 {
            schema_version: CAPABILITY_EVIDENCE_SCHEMA_VERSION.to_string(),
            forecast_id: "signed-forecast".to_string(),
            forecast_commitment_blake3: forecast_commitment_blake3.clone(),
            subject: subject(),
            receipt_sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            actual_compute_units: Some(2.0),
            runner_identity: "github-actions-runner".to_string(),
            runner_run_identity: "repo/actions/runs/12345/attempts/1".to_string(),
            runner_artifact_identity: "repo/actions/artifacts/67890".to_string(),
            runner_artifact_sha256: "a".repeat(64),
        }
        .canonical_bytes()
        .unwrap();
        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "signed-forecast".to_string(),
            forecast_commitment_blake3,
            subject: subject(),
            sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: hex::encode(Sha256::digest(&evidence)),
            actual_compute_units: Some(2.0),
        };
        let canonical = receipt.canonical_signing_payload().unwrap();
        let signature = key.sign(&canonical);
        let signed = SignedCapabilityOutcomeReceipt {
            receipt,
            signature_hex: hex::encode(signature.to_bytes()),
        };

        assert_eq!(
            ledger.verify_and_resolve_signed_receipt(signed.clone(), b"different bytes"),
            Err(CapabilityLedgerError::EvidenceRootMismatch)
        );

        let mut tampered_claim = signed.clone();
        tampered_claim.receipt.observed_success = false;
        assert_eq!(
            ledger.verify_and_resolve_signed_receipt(tampered_claim, evidence.as_slice()),
            Err(CapabilityLedgerError::InvalidReceiptSignature)
        );
        assert_eq!(ledger.metrics("rust-debugging").unwrap().resolved_count, 0);

        // A correctly signed receipt whose canonical evidence describes a different outcome
        // must be rejected even when its byte digest and receipt signature both verify.
        let mut contradictory_evidence: CapabilityEvidencePayloadV1 =
            serde_json::from_slice(&evidence).unwrap();
        contradictory_evidence.observed_success = false;
        let contradictory_bytes = contradictory_evidence.canonical_bytes().unwrap();
        let mut contradictory_receipt = signed.receipt.clone();
        contradictory_receipt.receipt_sha256 =
            hex::encode(Sha256::digest(&contradictory_bytes));
        let contradictory_signature = key.sign(
            &contradictory_receipt.canonical_signing_payload().unwrap(),
        );
        let contradictory_signed = SignedCapabilityOutcomeReceipt {
            receipt: contradictory_receipt,
            signature_hex: hex::encode(contradictory_signature.to_bytes()),
        };
        assert_eq!(
            ledger.verify_and_resolve_signed_receipt(
                contradictory_signed,
                &contradictory_bytes,
            ),
            Err(CapabilityLedgerError::EvidenceReceiptMismatch)
        );
        assert_eq!(ledger.metrics("rust-debugging").unwrap().resolved_count, 0);

        // The same JSON with extra leading whitespace is parseable but not canonical. Even a
        // valid signature over its exact digest must not bypass the canonical evidence profile.
        let mut noncanonical_bytes = Vec::with_capacity(evidence.len() + 1);
        noncanonical_bytes.push(b' ');
        noncanonical_bytes.extend_from_slice(&evidence);
        let mut noncanonical_receipt = signed.receipt.clone();
        noncanonical_receipt.receipt_sha256 =
            hex::encode(Sha256::digest(&noncanonical_bytes));
        let noncanonical_signature = key.sign(
            &noncanonical_receipt.canonical_signing_payload().unwrap(),
        );
        let noncanonical_signed = SignedCapabilityOutcomeReceipt {
            receipt: noncanonical_receipt,
            signature_hex: hex::encode(noncanonical_signature.to_bytes()),
        };
        assert_eq!(
            ledger.verify_and_resolve_signed_receipt(
                noncanonical_signed,
                &noncanonical_bytes,
            ),
            Err(CapabilityLedgerError::NonCanonicalEvidencePayload)
        );
        assert_eq!(ledger.metrics("rust-debugging").unwrap().resolved_count, 0);

        // Reuse the same forecast ID/sequence/subject in a different ledger but change
        // the committed probability/context. The original valid signature must not transfer.
        let mut transplanted_ledger = setup_claim_with_policy(ledger.qualification_policy.clone());
        transplanted_ledger.record_forecast(CapabilityForecast {
            forecast_id: "signed-forecast".to_string(),
            claim_id: "rust-debugging".to_string(),
            claim_sha256: "f".repeat(64),
            subject: subject(),
            sequence: 1,
            context_id: "transplanted-context".to_string(),
            split: CapabilitySplit::HeldOut,
            predicted_success_probability: 0.8,
            expected_compute_units: 2.0,
            predicted_failure_mode: None,
        }).unwrap();
        assert_eq!(
            transplanted_ledger.verify_and_resolve_signed_receipt(signed.clone(), evidence.as_slice()),
            Err(CapabilityLedgerError::ForecastCommitmentMismatch)
        );
        assert_eq!(
            transplanted_ledger.metrics("rust-debugging").unwrap().resolved_count,
            0
        );

        ledger.verify_and_resolve_signed_receipt(signed, evidence.as_slice()).unwrap();
        assert_eq!(ledger.metrics("rust-debugging").unwrap().resolved_count, 1);
    }

    #[test]
    fn qualification_policy_is_fixed_before_outcomes_are_recorded() {
        let mut pinned_policy = policy();
        pinned_policy.trusted_evaluator_identity = "different-evaluator".to_string();
        let mut ledger = setup_claim_with_policy(pinned_policy);
        for i in 0..3 {
            add_case(
                &mut ledger,
                &format!("heldout-untrusted-{i}"),
                2 * i + 1,
                CapabilitySplit::HeldOut,
                0.9,
                true,
            );
        }
        let metrics = ledger.qualification_metrics("rust-debugging").unwrap();
        assert_eq!(metrics.forecast_count, 3);
        assert_eq!(metrics.resolved_count, 0);
        assert_eq!(
            ledger.qualify_claim("rust-debugging"),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
    }

    #[test]
    fn trusted_evaluator_revision_is_pinned() {
        let mut ledger = setup_claim();
        for i in 0..3 {
            let id = format!("wrong-revision-{i}");
            let seq = 2 * i + 1;
            ledger
                .record_forecast(CapabilityForecast {
                    forecast_id: id.clone(),
                    claim_id: "rust-debugging".to_string(),
                    claim_sha256: "f".repeat(64),
                    subject: subject(),
                    sequence: seq,
                    context_id: format!("held-out-{i}"),
                    split: CapabilitySplit::HeldOut,
                    predicted_success_probability: 0.9,
                    expected_compute_units: 10.0,
                    predicted_failure_mode: None,
                })
                .unwrap();
            ledger
                .resolve_verified_forecast(test_verified_receipt(&ledger, CapabilityOutcomeReceipt {
                    forecast_id: id,
                    forecast_commitment_blake3: "0".repeat(64),
                    subject: subject(),
                    sequence: seq + 1,
                    observed_success: true,
                    split: CapabilitySplit::HeldOut,
                    evaluator_identity: "independent-qualification-runner".to_string(),
                    evaluator_revision: "unreviewed-revision".to_string(),
                    receipt_sha256: format!("{:064x}", seq + 1),
                    actual_compute_units: Some(10.0),
                }))
                .unwrap();
        }
        assert_eq!(
            ledger.qualify_claim("rust-debugging"),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
    }

    #[test]
    fn verified_token_rejects_malformed_verifier_provenance() {
        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "token-shape".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "1".repeat(64),
            actual_compute_units: None,
        };
        assert_eq!(
            VerifiedCapabilityOutcomeReceipt::from_verified_pipeline(
                receipt,
                "a".repeat(64),
                "symthaea-evidence-verifier",
                "v1",
                "not-a-digest",
            ),
            Err(CapabilityLedgerError::InvalidVerificationProvenance)
        );
    }

    #[test]
    fn qualification_requires_the_policy_pinned_verifier_revision() {
        let mut ledger = setup_claim();
        for i in 0..3u64 {
            let forecast_sequence = i * 2 + 1;
            let id = format!("untrusted-verifier-{i}");
            ledger.record_forecast(CapabilityForecast {
                forecast_id: id.clone(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: forecast_sequence,
                context_id: format!("held-out-{i}"),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.9,
                expected_compute_units: 10.0,
                predicted_failure_mode: None,
            }).unwrap();

            ledger.resolve_verified_forecast(test_verified_receipt_from(
                &ledger,
                CapabilityOutcomeReceipt {
                    forecast_id: id,
                    forecast_commitment_blake3: "0".repeat(64),
                    subject: subject(),
                    sequence: forecast_sequence + 1,
                    observed_success: true,
                    split: CapabilitySplit::HeldOut,
                    evaluator_identity: "independent-qualification-runner".to_string(),
                    evaluator_revision: "v1".to_string(),
                    receipt_sha256: format!("{:064x}", forecast_sequence + 1),
                    actual_compute_units: Some(10.0),
                },
                CAPABILITY_RECEIPT_VERIFIER_IDENTITY,
                "capability-receipt-v1",
            )).unwrap();
        }

        let metrics = ledger.qualification_metrics("rust-debugging").unwrap();
        assert_eq!(metrics.forecast_count, 3);
        assert_eq!(metrics.resolved_count, 0);
        assert_eq!(metrics.brier_score, None);
        assert_eq!(
            ledger.qualify_claim("rust-debugging"),
            Err(CapabilityLedgerError::InsufficientEvidence)
        );
    }

    #[test]
    fn invalid_probabilities_and_costs_fail_closed() {
        let mut ledger = setup_claim();
        for (id, probability, cost) in [
            ("nan-probability", f64::NAN, 1.0),
            ("negative-probability", -0.01, 1.0),
            ("over-one-probability", 1.01, 1.0),
            ("negative-cost", 0.5, -1.0),
            ("infinite-cost", 0.5, f64::INFINITY),
        ] {
            let forecast = CapabilityForecast {
                forecast_id: id.to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: format!("context-{id}"),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: probability,
                expected_compute_units: cost,
                predicted_failure_mode: None,
            };
            assert_eq!(
                ledger.record_forecast(forecast),
                Err(CapabilityLedgerError::InvalidForecast),
                "invalid input should be rejected: {id}"
            );
        }
    }

    #[test]
    fn unqualified_claim_cannot_record_forecasts() {
        let mut ledger = CapabilityLedger::new(subject(), policy()).unwrap();
        ledger
            .add_claim(
                CapabilityClaim::new(
                    "not-ready",
                    "f".repeat(64),
                    "Not ready for evaluation",
                    "Test scope",
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            ledger.record_forecast(CapabilityForecast {
                forecast_id: "premature".to_string(),
                claim_id: "not-ready".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: "context".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            }),
            Err(CapabilityLedgerError::InvalidLifecycleTransition)
        );
    }

    #[test]
    fn duplicate_claim_ids_are_rejected() {
        let mut ledger = setup_claim();
        let duplicate = CapabilityClaim::new(
            "rust-debugging",
            "1".repeat(64),
            "A conflicting claim version",
            "A different scope",
        )
        .unwrap();
        assert_eq!(
            ledger.add_claim(duplicate),
            Err(CapabilityLedgerError::DuplicateClaim)
        );
    }

    #[test]
    fn forecast_and_outcome_events_share_one_monotonic_sequence() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "first".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 5,
                context_id: "context-first".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.8,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();

        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "first".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: 5,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "3".repeat(64),
            actual_compute_units: None,
        };
        assert_eq!(
            ledger.resolve_verified_forecast(test_verified_receipt(&ledger, receipt.clone())),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );

        let mut later_receipt = receipt;
        later_receipt.sequence = 6;
        ledger.resolve_verified_forecast(test_verified_receipt(&ledger, later_receipt)).unwrap();

        assert_eq!(
            ledger.record_forecast(CapabilityForecast {
                forecast_id: "out-of-order".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 5,
                context_id: "context-out-of-order".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.8,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            }),
            Err(CapabilityLedgerError::SequenceNotMonotonic)
        );
    }

    #[test]
    fn receipt_root_cannot_be_replayed_for_a_second_forecast() {
        let mut ledger = setup_claim();
        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "first".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 1,
                context_id: "context-first".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();
        let receipt = CapabilityOutcomeReceipt {
            forecast_id: "first".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            subject: subject(),
            sequence: 2,
            observed_success: true,
            split: CapabilitySplit::HeldOut,
            evaluator_identity: "independent-qualification-runner".to_string(),
            evaluator_revision: "v1".to_string(),
            receipt_sha256: "2".repeat(64),
            actual_compute_units: None,
        };
        ledger.resolve_verified_forecast(test_verified_receipt(&ledger, receipt.clone())).unwrap();

        ledger
            .record_forecast(CapabilityForecast {
                forecast_id: "second".to_string(),
                claim_id: "rust-debugging".to_string(),
                claim_sha256: "f".repeat(64),
                subject: subject(),
                sequence: 3,
                context_id: "context-second".to_string(),
                split: CapabilitySplit::HeldOut,
                predicted_success_probability: 0.5,
                expected_compute_units: 1.0,
                predicted_failure_mode: None,
            })
            .unwrap();
        let replay = CapabilityOutcomeReceipt {
            forecast_id: "second".to_string(),
            forecast_commitment_blake3: "0".repeat(64),
            sequence: 4,
            ..receipt
        };
        assert_eq!(
            ledger.resolve_verified_forecast(test_verified_receipt(&ledger, replay)),
            Err(CapabilityLedgerError::ReceiptReplay)
        );
    }

    #[test]
    fn unavailable_metrics_are_not_reported_as_perfect_calibration() {
        let ledger = setup_claim();
        let metrics = ledger.metrics("rust-debugging").unwrap();
        assert_eq!(metrics.resolved_count, 0);
        assert_eq!(metrics.brier_score, None);
        assert_eq!(metrics.expected_calibration_error, None);
    }
}
