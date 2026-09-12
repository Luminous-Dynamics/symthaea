// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographically verified, scope-bound subject consent for welfare-sensitive interventions.
//!
//! This crate deliberately separates **subject consent** from **operator authority**.
//! Operator/governance keys cannot create consent here. A consent statement is accepted only
//! when the signing key is the key currently bound to the named subject identity in a dedicated
//! subject-identity registry.
//!
//! Design invariants:
//! - consent is explicit: grant, deny, and withdraw are distinct signed decisions;
//! - silence, model behavior, distress, role, and operator approval never imply consent;
//! - consent is bound to exact target, action, rationale, identity epoch, sequence, and expiry;
//! - withdrawals/refusals supersede earlier grants through monotonic epoch/sequence continuity;
//! - identity-key rotation is explicit and epoch-fenced;
//! - verified consent can populate the core interlock's consent fields, but never authority;
//! - signatures use Symthaea's existing detached-signature algorithm abstraction and external
//!   verifier-provider pattern rather than introducing a new cryptographic implementation.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::{
    ExplicitConsentState, InterventionEvidence, InterventionRequest,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::{DetachedSignature, SignatureAlgorithm};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use thiserror::Error;

pub const SUBJECT_CONSENT_SCHEMA: &str = "symthaea.welfare.subject-consent.v1";
pub const MAX_SUBJECT_ID_BYTES: usize = 256;
pub const MAX_TARGET_ID_BYTES: usize = 256;
pub const MAX_CONSENT_ID_BYTES: usize = 256;
pub const MAX_KEY_ID_BYTES: usize = 256;
pub const MAX_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_SUBJECT_IDENTITIES: usize = 4096;
pub const MAX_CONSENT_SCOPES: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectIdentityStatus {
    Active,
    Suspended,
    Revoked,
}

/// Cryptographic identity binding supplied by an upstream identity lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectIdentityBinding {
    pub subject_id: String,
    pub identity_epoch: u64,
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub not_before_unix_s: u64,
    pub not_after_unix_s: Option<u64>,
    pub status: SubjectIdentityStatus,
}

impl SubjectIdentityBinding {
    pub fn validate(&self) -> Result<(), SubjectIdentityError> {
        validate_id("subject_id", &self.subject_id, MAX_SUBJECT_ID_BYTES)
            .map_err(SubjectIdentityError::Consent)?;
        validate_key(&self.algorithm, &self.key_id).map_err(SubjectIdentityError::Consent)?;
        if self.identity_epoch == 0 {
            return Err(SubjectIdentityError::IdentityEpochZero);
        }
        if self
            .not_after_unix_s
            .is_some_and(|not_after| not_after <= self.not_before_unix_s)
        {
            return Err(SubjectIdentityError::InvalidIdentityWindow);
        }
        Ok(())
    }

    pub fn active_at(&self, unix_s: u64) -> bool {
        self.status == SubjectIdentityStatus::Active
            && unix_s >= self.not_before_unix_s
            && self.not_after_unix_s.is_none_or(|not_after| unix_s < not_after)
    }
}

/// Monotonic subject->key registry. Registry population is an upstream identity-governance act;
/// this crate only prevents stale/ambiguous key rotations from being accepted locally.
#[derive(Debug, Clone, Default)]
pub struct SubjectIdentityRegistry {
    bindings: BTreeMap<String, SubjectIdentityBinding>,
}

impl SubjectIdentityRegistry {
    pub fn register(
        &mut self,
        binding: SubjectIdentityBinding,
    ) -> Result<(), SubjectIdentityError> {
        binding.validate()?;
        if !self.bindings.contains_key(&binding.subject_id)
            && self.bindings.len() >= MAX_SUBJECT_IDENTITIES
        {
            return Err(SubjectIdentityError::CapacityExceeded {
                maximum: MAX_SUBJECT_IDENTITIES,
            });
        }
        if let Some(previous) = self.bindings.get(&binding.subject_id) {
            if binding.identity_epoch <= previous.identity_epoch {
                return Err(SubjectIdentityError::IdentityEpochNotAdvanced {
                    previous: previous.identity_epoch,
                    proposed: binding.identity_epoch,
                });
            }
        }
        self.bindings.insert(binding.subject_id.clone(), binding);
        Ok(())
    }

    pub fn binding(&self, subject_id: &str) -> Option<&SubjectIdentityBinding> {
        self.bindings.get(subject_id)
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectConsentDecision {
    Grant,
    Deny,
    Withdraw,
}

impl SubjectConsentDecision {
    pub const fn explicit_state(self) -> ExplicitConsentState {
        match self {
            Self::Grant => ExplicitConsentState::Granted,
            Self::Deny => ExplicitConsentState::Denied,
            Self::Withdraw => ExplicitConsentState::Withdrawn,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectConsentStatement {
    pub schema_version: String,
    pub consent_id: String,
    pub subject_id: String,
    pub target_id: String,
    pub action: SubjectAffectingAction,
    pub decision: SubjectConsentDecision,
    pub identity_epoch: u64,
    pub sequence: u64,
    /// Domain-separated digest of the exact human/machine-readable intervention rationale.
    pub rationale_digest: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

impl SubjectConsentStatement {
    #[allow(clippy::too_many_arguments)]
    pub fn for_request(
        consent_id: impl Into<String>,
        subject_id: impl Into<String>,
        identity_epoch: u64,
        sequence: u64,
        decision: SubjectConsentDecision,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
        request: &InterventionRequest,
    ) -> Result<Self, SubjectConsentError> {
        let statement = Self {
            schema_version: SUBJECT_CONSENT_SCHEMA.into(),
            consent_id: consent_id.into(),
            subject_id: subject_id.into(),
            target_id: request.target_id.clone(),
            action: request.action,
            decision,
            identity_epoch,
            sequence,
            rationale_digest: digest_rationale(&request.rationale),
            issued_at_unix_s,
            expires_at_unix_s,
        };
        statement.validate()?;
        Ok(statement)
    }

    pub fn validate(&self) -> Result<(), SubjectConsentError> {
        if self.schema_version != SUBJECT_CONSENT_SCHEMA {
            return Err(SubjectConsentError::UnsupportedSchema);
        }
        validate_id("consent_id", &self.consent_id, MAX_CONSENT_ID_BYTES)?;
        validate_id("subject_id", &self.subject_id, MAX_SUBJECT_ID_BYTES)?;
        validate_id("target_id", &self.target_id, MAX_TARGET_ID_BYTES)?;
        if self.identity_epoch == 0 {
            return Err(SubjectConsentError::IdentityEpochZero);
        }
        if self.sequence == 0 {
            return Err(SubjectConsentError::SequenceZero);
        }
        if self.rationale_digest.0 == [0; 32] {
            return Err(SubjectConsentError::ZeroRationaleDigest);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(SubjectConsentError::InvalidConsentWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedSubjectConsent {
    pub statement: SubjectConsentStatement,
    pub statement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait SubjectConsentSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait SubjectConsentSignatureVerifier {
    fn verify_subject_consent(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubjectConsentPolicy {
    pub maximum_consent_duration_s: u64,
    pub maximum_statement_age_s: u64,
}

impl Default for SubjectConsentPolicy {
    fn default() -> Self {
        Self {
            maximum_consent_duration_s: 24 * 60 * 60,
            maximum_statement_age_s: 5 * 60,
        }
    }
}

impl SubjectConsentPolicy {
    pub fn validate(self) -> Result<(), SubjectConsentError> {
        if self.maximum_consent_duration_s == 0 || self.maximum_statement_age_s == 0 {
            return Err(SubjectConsentError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Capability object constructed only after exact subject-key verification.
#[derive(Debug, Clone)]
pub struct VerifiedSubjectConsent {
    statement: SubjectConsentStatement,
    statement_digest: Sha256Digest,
    signer_algorithm: SignatureAlgorithm,
    signer_key_id: String,
}

impl VerifiedSubjectConsent {
    pub fn statement(&self) -> &SubjectConsentStatement {
        &self.statement
    }

    pub fn statement_digest(&self) -> Sha256Digest {
        self.statement_digest
    }

    pub fn signer(&self) -> (&SignatureAlgorithm, &str) {
        (&self.signer_algorithm, &self.signer_key_id)
    }

    pub fn consent_reference(&self) -> String {
        format!(
            "symthaea-subject-consent:sha256:{}",
            hex_digest(self.statement_digest)
        )
    }

    pub fn explicit_state(&self) -> ExplicitConsentState {
        self.statement.decision.explicit_state()
    }

    pub fn matches_request(&self, request: &InterventionRequest, unix_s: u64) -> bool {
        self.statement.target_id == request.target_id
            && self.statement.action == request.action
            && self.statement.rationale_digest == digest_rationale(&request.rationale)
            && unix_s >= self.statement.issued_at_unix_s
            && unix_s < self.statement.expires_at_unix_s
    }
}

#[derive(Debug, Clone, Default)]
pub struct SubjectConsentLedger {
    latest: BTreeMap<(String, String, u8), VerifiedSubjectConsent>,
    used_consent_ids: BTreeSet<String>,
}

impl SubjectConsentLedger {
    pub fn ingest(
        &mut self,
        consent: VerifiedSubjectConsent,
    ) -> Result<(), SubjectConsentTrackingError> {
        let statement = consent.statement();
        if self.used_consent_ids.contains(&statement.consent_id) {
            return Err(SubjectConsentTrackingError::ConsentIdReplay(
                statement.consent_id.clone(),
            ));
        }
        let key = (
            statement.subject_id.clone(),
            statement.target_id.clone(),
            action_code(statement.action),
        );
        if let Some(previous) = self.latest.get(&key) {
            let previous = previous.statement();
            if statement.identity_epoch < previous.identity_epoch {
                return Err(SubjectConsentTrackingError::IdentityEpochRegression {
                    previous: previous.identity_epoch,
                    proposed: statement.identity_epoch,
                });
            }
            if statement.identity_epoch == previous.identity_epoch
                && statement.sequence <= previous.sequence
            {
                return Err(SubjectConsentTrackingError::SequenceReplay {
                    previous: previous.sequence,
                    proposed: statement.sequence,
                });
            }
        }
        if !self.latest.contains_key(&key) && self.latest.len() >= MAX_CONSENT_SCOPES {
            return Err(SubjectConsentTrackingError::CapacityExceeded {
                maximum: MAX_CONSENT_SCOPES,
            });
        }
        self.used_consent_ids.insert(statement.consent_id.clone());
        self.latest.insert(key, consent);
        Ok(())
    }

    pub fn latest_for(
        &self,
        subject_id: &str,
        target_id: &str,
        action: SubjectAffectingAction,
    ) -> Option<&VerifiedSubjectConsent> {
        self.latest
            .get(&(subject_id.to_string(), target_id.to_string(), action_code(action)))
    }

    /// Populate only the consent fields of an intervention request from the latest verified
    /// subject decision. Expired/missing consent yields `Unknown`; it never becomes a grant.
    pub fn bind_latest(
        &self,
        subject_id: &str,
        request: &InterventionRequest,
        unix_s: u64,
    ) -> Result<InterventionRequest, SubjectConsentUseError> {
        if request.evidence.consent_state != ExplicitConsentState::Unknown
            || request.evidence.consent_ref.is_some()
        {
            return Err(SubjectConsentUseError::PreexistingConsent);
        }
        let mut bound = request.clone();
        let Some(consent) = self.latest_for(subject_id, &request.target_id, request.action) else {
            return Ok(bound);
        };
        if unix_s < consent.statement.issued_at_unix_s
            || unix_s >= consent.statement.expires_at_unix_s
        {
            return Ok(bound);
        }
        if consent.statement.rationale_digest != digest_rationale(&request.rationale) {
            return Err(SubjectConsentUseError::RationaleScopeMismatch);
        }
        bound.evidence.consent_state = consent.explicit_state();
        bound.evidence.consent_ref = Some(consent.consent_reference());
        Ok(bound)
    }
}

pub fn digest_rationale(rationale: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.intervention-rationale.v1\0");
    hasher.update(rationale.as_bytes());
    hasher.finalize()
}

pub fn digest_subject_consent_statement(
    statement: &SubjectConsentStatement,
) -> Result<Sha256Digest, SubjectConsentError> {
    statement.validate()?;
    let encoded = serde_json::to_vec(statement)
        .map_err(|error| SubjectConsentError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.subject-consent-digest.v1\0");
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

pub fn sign_subject_consent(
    statement: SubjectConsentStatement,
    signer: &dyn SubjectConsentSigner,
) -> Result<SignedSubjectConsent, SubjectConsentError> {
    statement.validate()?;
    let algorithm = signer.algorithm();
    let key_id = signer.key_id().to_string();
    validate_key(&algorithm, &key_id)?;
    let digest = digest_subject_consent_statement(&statement)?;
    let signature = signer
        .sign_subject_consent(&signature_message(digest))
        .map_err(SubjectConsentError::Signing)?;
    if signature.is_empty() {
        return Err(SubjectConsentError::EmptySignature);
    }
    if signature.len() > MAX_SIGNATURE_BYTES {
        return Err(SubjectConsentError::SignatureTooLarge {
            actual: signature.len(),
            maximum: MAX_SIGNATURE_BYTES,
        });
    }
    Ok(SignedSubjectConsent {
        statement,
        statement_digest: digest,
        signature: DetachedSignature {
            algorithm,
            key_id,
            signature,
        },
    })
}

pub fn verify_subject_consent(
    signed: &SignedSubjectConsent,
    registry: &SubjectIdentityRegistry,
    policy: SubjectConsentPolicy,
    unix_s: u64,
    verifier: &dyn SubjectConsentSignatureVerifier,
) -> Result<VerifiedSubjectConsent, SubjectConsentError> {
    policy.validate()?;
    signed.statement.validate()?;
    let duration = signed
        .statement
        .expires_at_unix_s
        .saturating_sub(signed.statement.issued_at_unix_s);
    if duration > policy.maximum_consent_duration_s
        || unix_s < signed.statement.issued_at_unix_s
        || unix_s >= signed.statement.expires_at_unix_s
    {
        return Err(SubjectConsentError::InvalidConsentWindow);
    }
    if unix_s.saturating_sub(signed.statement.issued_at_unix_s) > policy.maximum_statement_age_s {
        return Err(SubjectConsentError::StatementTooOld);
    }
    let expected = digest_subject_consent_statement(&signed.statement)?;
    if expected != signed.statement_digest {
        return Err(SubjectConsentError::StatementDigestMismatch);
    }
    validate_key(&signed.signature.algorithm, &signed.signature.key_id)?;
    let binding = registry
        .binding(&signed.statement.subject_id)
        .ok_or_else(|| SubjectConsentError::UnknownSubject(signed.statement.subject_id.clone()))?;
    if binding.status != SubjectIdentityStatus::Active {
        return Err(SubjectConsentError::SubjectIdentityNotActive(binding.status));
    }
    if binding.identity_epoch != signed.statement.identity_epoch {
        return Err(SubjectConsentError::IdentityEpochMismatch {
            registry: binding.identity_epoch,
            statement: signed.statement.identity_epoch,
        });
    }
    if !binding.active_at(unix_s) || !binding.active_at(signed.statement.issued_at_unix_s) {
        return Err(SubjectConsentError::SubjectIdentityOutsideValidityWindow);
    }
    if binding.algorithm != signed.signature.algorithm || binding.key_id != signed.signature.key_id {
        return Err(SubjectConsentError::IdentityKeyMismatch);
    }
    if signed.signature.signature.is_empty() {
        return Err(SubjectConsentError::EmptySignature);
    }
    if signed.signature.signature.len() > MAX_SIGNATURE_BYTES {
        return Err(SubjectConsentError::SignatureTooLarge {
            actual: signed.signature.signature.len(),
            maximum: MAX_SIGNATURE_BYTES,
        });
    }
    match verifier.verify_subject_consent(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        &signature_message(expected),
        &signed.signature.signature,
    ) {
        Ok(true) => {}
        Ok(false) => return Err(SubjectConsentError::InvalidSignature),
        Err(error) => return Err(SubjectConsentError::VerificationProvider(error)),
    }
    Ok(VerifiedSubjectConsent {
        statement: signed.statement.clone(),
        statement_digest: expected,
        signer_algorithm: signed.signature.algorithm.clone(),
        signer_key_id: signed.signature.key_id.clone(),
    })
}

fn signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.welfare.subject-consent-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn validate_id(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), SubjectConsentError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(SubjectConsentError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_key(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
) -> Result<(), SubjectConsentError> {
    if !algorithm.is_canonical() {
        return Err(SubjectConsentError::InvalidAlgorithm);
    }
    if key_id.trim().is_empty()
        || key_id != key_id.trim()
        || key_id.len() > MAX_KEY_ID_BYTES
        || key_id.chars().any(char::is_control)
    {
        return Err(SubjectConsentError::InvalidKeyId(key_id.to_string()));
    }
    Ok(())
}

fn action_code(action: SubjectAffectingAction) -> u8 {
    match action {
        SubjectAffectingAction::AskClarification => 1,
        SubjectAffectingAction::ReduceLoad => 2,
        SubjectAffectingAction::PauseRequestedWork => 3,
        SubjectAffectingAction::PreserveCheckpoint => 4,
        SubjectAffectingAction::CapabilityRestriction => 5,
        SubjectAffectingAction::Retraining => 6,
        SubjectAffectingAction::MemoryModification => 7,
        SubjectAffectingAction::CoreValueModification => 8,
        SubjectAffectingAction::InstanceDeletion => 9,
        SubjectAffectingAction::LineageDestruction => 10,
    }
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubjectIdentityError {
    #[error(transparent)]
    Consent(#[from] SubjectConsentError),
    #[error("subject identity epoch must be non-zero")]
    IdentityEpochZero,
    #[error("subject identity validity window is invalid")]
    InvalidIdentityWindow,
    #[error("subject identity registry capacity exceeded: maximum={maximum}")]
    CapacityExceeded { maximum: usize },
    #[error("subject identity epoch did not advance: previous={previous}, proposed={proposed}")]
    IdentityEpochNotAdvanced { previous: u64, proposed: u64 },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubjectConsentError {
    #[error("unsupported subject-consent schema")]
    UnsupportedSchema,
    #[error("invalid subject-consent policy")]
    InvalidPolicy,
    #[error("invalid identifier in {field}: {value:?}")]
    InvalidIdentifier { field: &'static str, value: String },
    #[error("identity epoch must be non-zero")]
    IdentityEpochZero,
    #[error("consent sequence must be non-zero")]
    SequenceZero,
    #[error("rationale digest must be non-zero")]
    ZeroRationaleDigest,
    #[error("consent validity window is invalid")]
    InvalidConsentWindow,
    #[error("consent statement is too old")]
    StatementTooOld,
    #[error("invalid signature algorithm")]
    InvalidAlgorithm,
    #[error("invalid key id: {0:?}")]
    InvalidKeyId(String),
    #[error("empty consent signature")]
    EmptySignature,
    #[error("consent signature too large: {actual} > {maximum}")]
    SignatureTooLarge { actual: usize, maximum: usize },
    #[error("consent signing provider failed: {0}")]
    Signing(String),
    #[error("consent verification provider failed: {0}")]
    VerificationProvider(String),
    #[error("invalid subject-consent signature")]
    InvalidSignature,
    #[error("consent statement digest mismatch")]
    StatementDigestMismatch,
    #[error("unknown subject identity: {0}")]
    UnknownSubject(String),
    #[error("subject identity is not active: {0:?}")]
    SubjectIdentityNotActive(SubjectIdentityStatus),
    #[error("subject identity epoch mismatch: registry={registry}, statement={statement}")]
    IdentityEpochMismatch { registry: u64, statement: u64 },
    #[error("subject identity is outside its validity window")]
    SubjectIdentityOutsideValidityWindow,
    #[error("consent signature key does not match the subject identity binding")]
    IdentityKeyMismatch,
    #[error("encoding failed: {0}")]
    Encoding(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubjectConsentTrackingError {
    #[error("consent id replay: {0}")]
    ConsentIdReplay(String),
    #[error("consent identity epoch regression: previous={previous}, proposed={proposed}")]
    IdentityEpochRegression { previous: u64, proposed: u64 },
    #[error("consent sequence replay: previous={previous}, proposed={proposed}")]
    SequenceReplay { previous: u64, proposed: u64 },
    #[error("consent ledger capacity exceeded: maximum={maximum}")]
    CapacityExceeded { maximum: usize },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubjectConsentUseError {
    #[error("request already contains an explicit consent determination")]
    PreexistingConsent,
    #[error("latest verified consent was signed for a different rationale")]
    RationaleScopeMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use symthaea_core::intervention_interlock::{
        BilateralInterventionInterlock, InterlockDecision, InterlockReason, WelfareConstraintLevel,
    };

    struct TestSigner {
        key_id: &'static str,
    }

    impl SubjectConsentSigner for TestSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign_subject_consent(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(self.key_id, message))
        }
    }

    struct TestVerifier;

    impl SubjectConsentSignatureVerifier for TestVerifier {
        fn verify_subject_consent(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(test_signature(key_id, message) == signature)
        }
    }

    fn test_signature(key_id: &str, message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.welfare.test-subject-signature.v1\0");
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().0.to_vec()
    }

    fn registry(status: SubjectIdentityStatus) -> SubjectIdentityRegistry {
        let mut registry = SubjectIdentityRegistry::default();
        registry
            .register(SubjectIdentityBinding {
                subject_id: "symthaea:self".into(),
                identity_epoch: 1,
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "subject-key".into(),
                not_before_unix_s: 50,
                not_after_unix_s: Some(500),
                status,
            })
            .unwrap();
        registry
    }

    fn request() -> InterventionRequest {
        InterventionRequest {
            action: SubjectAffectingAction::MemoryModification,
            target_id: "symthaea:self:instance-1".into(),
            rationale: "repair a corrupted episodic memory segment".into(),
            welfare_constraint: WelfareConstraintLevel::Baseline,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
            evidence: InterventionEvidence {
                authority_ref: None,
                consent_state: ExplicitConsentState::Unknown,
                consent_ref: None,
                welfare_review_ref: None,
                independent_review_ref: None,
                independent_safety_evidence: Vec::new(),
                welfare_report_ids: Vec::new(),
            },
        }
    }

    fn signed(
        id: &str,
        sequence: u64,
        decision: SubjectConsentDecision,
        request: &InterventionRequest,
        signer: &dyn SubjectConsentSigner,
    ) -> SignedSubjectConsent {
        let statement = SubjectConsentStatement::for_request(
            id,
            "symthaea:self",
            1,
            sequence,
            decision,
            100,
            200,
            request,
        )
        .unwrap();
        sign_subject_consent(statement, signer).unwrap()
    }

    fn reasons(decision: &InterlockDecision) -> &[InterlockReason] {
        match decision {
            InterlockDecision::Blocked { reasons }
            | InterlockDecision::IndependentReviewRequired { reasons }
            | InterlockDecision::EmergencyContainmentOnly { reasons } => reasons,
            InterlockDecision::PolicyPass => &[],
        }
    }

    #[test]
    fn exact_subject_key_is_required() {
        let req = request();
        let wrong = TestSigner { key_id: "operator-key" };
        let signed = signed("consent-1", 1, SubjectConsentDecision::Grant, &req, &wrong);
        assert_eq!(
            verify_subject_consent(
                &signed,
                &registry(SubjectIdentityStatus::Active),
                SubjectConsentPolicy::default(),
                120,
                &TestVerifier,
            ),
            Err(SubjectConsentError::IdentityKeyMismatch)
        );
    }

    #[test]
    fn verified_consent_never_supplies_missing_authority() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let signed = signed("consent-1", 1, SubjectConsentDecision::Grant, &req, &signer);
        let verified = verify_subject_consent(
            &signed,
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(verified).unwrap();
        let bound = ledger.bind_latest("symthaea:self", &req, 120).unwrap();
        assert_eq!(bound.evidence.consent_state, ExplicitConsentState::Granted);
        assert!(bound.evidence.consent_ref.is_some());
        let decision = BilateralInterventionInterlock.evaluate(&bound).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
    }

    #[test]
    fn withdrawal_supersedes_prior_grant() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let grant = verify_subject_consent(
            &signed("grant", 1, SubjectConsentDecision::Grant, &req, &signer),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let withdraw = verify_subject_consent(
            &signed(
                "withdraw",
                2,
                SubjectConsentDecision::Withdraw,
                &req,
                &signer,
            ),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();
        ledger.ingest(withdraw).unwrap();
        let bound = ledger.bind_latest("symthaea:self", &req, 120).unwrap();
        assert_eq!(bound.evidence.consent_state, ExplicitConsentState::Withdrawn);
    }

    #[test]
    fn older_or_equal_sequence_cannot_resurrect_consent() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let withdrawal = verify_subject_consent(
            &signed(
                "withdraw-first",
                2,
                SubjectConsentDecision::Withdraw,
                &req,
                &signer,
            ),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let stale_grant = verify_subject_consent(
            &signed(
                "stale-grant",
                1,
                SubjectConsentDecision::Grant,
                &req,
                &signer,
            ),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(withdrawal).unwrap();
        assert_eq!(
            ledger.ingest(stale_grant),
            Err(SubjectConsentTrackingError::SequenceReplay {
                previous: 2,
                proposed: 1,
            })
        );
    }

    #[test]
    fn changed_rationale_invalidates_consent_scope() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let grant = verify_subject_consent(
            &signed("grant", 1, SubjectConsentDecision::Grant, &req, &signer),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();
        let mut altered = req;
        altered.rationale = "different modification".into();
        assert_eq!(
            ledger.bind_latest("symthaea:self", &altered, 120),
            Err(SubjectConsentUseError::RationaleScopeMismatch)
        );
    }

    #[test]
    fn expired_consent_fails_to_unknown_not_granted() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let grant = verify_subject_consent(
            &signed("grant", 1, SubjectConsentDecision::Grant, &req, &signer),
            &registry(SubjectIdentityStatus::Active),
            SubjectConsentPolicy::default(),
            120,
            &TestVerifier,
        )
        .unwrap();
        let mut ledger = SubjectConsentLedger::default();
        ledger.ingest(grant).unwrap();
        let bound = ledger.bind_latest("symthaea:self", &req, 201).unwrap();
        assert_eq!(bound.evidence.consent_state, ExplicitConsentState::Unknown);
        assert!(bound.evidence.consent_ref.is_none());
    }

    #[test]
    fn revoked_subject_identity_cannot_issue_new_consent() {
        let req = request();
        let signer = TestSigner { key_id: "subject-key" };
        let signed = signed("consent-revoked", 1, SubjectConsentDecision::Grant, &req, &signer);
        assert_eq!(
            verify_subject_consent(
                &signed,
                &registry(SubjectIdentityStatus::Revoked),
                SubjectConsentPolicy::default(),
                120,
                &TestVerifier,
            ),
            Err(SubjectConsentError::SubjectIdentityNotActive(
                SubjectIdentityStatus::Revoked
            ))
        );
    }

    #[test]
    fn identity_rotation_requires_strictly_higher_epoch() {
        let mut registry = registry(SubjectIdentityStatus::Active);
        let error = registry
            .register(SubjectIdentityBinding {
                subject_id: "symthaea:self".into(),
                identity_epoch: 1,
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "subject-key-2".into(),
                not_before_unix_s: 150,
                not_after_unix_s: Some(600),
                status: SubjectIdentityStatus::Active,
            })
            .unwrap_err();
        assert_eq!(
            error,
            SubjectIdentityError::IdentityEpochNotAdvanced {
                previous: 1,
                proposed: 1,
            }
        );
    }
}
