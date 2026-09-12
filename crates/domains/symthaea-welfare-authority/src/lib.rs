// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified authority for welfare-sensitive interventions.
//!
//! This crate is deliberately outside `symthaea-core`. The core interlock stays a
//! deterministic policy engine; this layer reuses Symthaea's established detached
//! signature and trust-snapshot machinery to prove that an authority reference was
//! actually verified before it is injected into an intervention request.
//!
//! Security invariants:
//! - authority is bound to the exact target, action, request evidence and rationale;
//! - signatures are checked against a fresh, lifecycle-aware trust snapshot;
//! - only explicitly enrolled welfare-authority signer identities count;
//! - identity-affecting/destructive actions require independent signer roles;
//! - epoch/sequence fencing and one-shot authority IDs prevent replay;
//! - this crate never turns welfare protection, consent, or safety evidence into authority;
//! - the underlying key must already be eligible for authenticated operator commands;
//!   welfare-specific scope is then narrowed further by this crate's signer policy.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::{
    BilateralInterventionInterlock, ExplicitConsentState, InterlockDecision, InterlockError,
    InterventionRequest, WelfareConstraintLevel,
};
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::attestation::{DetachedSignature, SignatureAlgorithm};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::{
    KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use thiserror::Error;

pub const WELFARE_AUTHORITY_SCHEMA: &str = "symthaea.welfare.intervention-authority.v1";
pub const MAX_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_TARGET_ID_BYTES: usize = 256;
pub const MAX_KEY_ID_BYTES: usize = 256;
pub const MAX_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_AUTHORITY_SIGNATURES: usize = 16;
pub const MAX_TRACKED_AUTHORITIES: usize = 4096;

/// Independent roles that may participate in an intervention-authority quorum.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum WelfareAuthorityRole {
    /// Ordinary operational authority for scoped interventions.
    Operator,
    /// Safety-specific authority for destructive containment decisions.
    SafetyOfficer,
    /// Reviewer independent from the primary operator role.
    IndependentReviewer,
}

/// Enrollment of one already-trusted signing key into the narrower welfare-authority policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WelfareAuthoritySignerBinding {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub role: WelfareAuthorityRole,
}

impl WelfareAuthoritySignerBinding {
    pub fn new(
        algorithm: SignatureAlgorithm,
        key_id: impl Into<String>,
        role: WelfareAuthorityRole,
    ) -> Self {
        Self {
            algorithm,
            key_id: key_id.into(),
            role,
        }
    }
}

/// Local policy layered on top of the shared trust snapshot.
///
/// `TrustSnapshot` answers whether a key is active, fresh and usable for authenticated
/// operator commands. This policy answers the narrower question: may that exact identity
/// participate in welfare-sensitive authority, and in which independent role?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WelfareAuthorityPolicy {
    pub maximum_authority_duration_s: u64,
    pub maximum_statement_age_s: u64,
    pub maximum_signatures: usize,
    bindings: BTreeMap<(SignatureAlgorithm, String), WelfareAuthorityRole>,
}

impl WelfareAuthorityPolicy {
    pub fn new(
        bindings: impl IntoIterator<Item = WelfareAuthoritySignerBinding>,
    ) -> Result<Self, WelfareAuthorityError> {
        let mut policy = Self {
            maximum_authority_duration_s: 15 * 60,
            maximum_statement_age_s: 5 * 60,
            maximum_signatures: MAX_AUTHORITY_SIGNATURES,
            bindings: BTreeMap::new(),
        };
        for binding in bindings {
            validate_key_identity(&binding.algorithm, &binding.key_id)?;
            let key = (binding.algorithm, binding.key_id);
            if policy.bindings.insert(key.clone(), binding.role).is_some() {
                return Err(WelfareAuthorityError::DuplicatePolicySigner {
                    algorithm: key.0,
                    key_id: key.1,
                });
            }
        }
        policy.validate()?;
        Ok(policy)
    }

    pub fn with_windows(
        mut self,
        maximum_authority_duration_s: u64,
        maximum_statement_age_s: u64,
    ) -> Result<Self, WelfareAuthorityError> {
        self.maximum_authority_duration_s = maximum_authority_duration_s;
        self.maximum_statement_age_s = maximum_statement_age_s;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), WelfareAuthorityError> {
        if self.maximum_authority_duration_s == 0
            || self.maximum_statement_age_s == 0
            || self.maximum_signatures == 0
            || self.maximum_signatures > MAX_AUTHORITY_SIGNATURES
            || self.bindings.is_empty()
        {
            return Err(WelfareAuthorityError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn role_for(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
    ) -> Option<WelfareAuthorityRole> {
        self.bindings
            .get(&(algorithm.clone(), key_id.to_string()))
            .copied()
    }
}

/// Exact request scope signed by the authority quorum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareInterventionAuthorityStatement {
    pub schema_version: String,
    pub authority_id: String,
    pub target_id: String,
    pub action: SubjectAffectingAction,
    pub authority_epoch: u64,
    pub sequence: u64,
    /// Digest of the canonical intervention request with `authority_ref` cleared.
    pub request_digest: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

impl WelfareInterventionAuthorityStatement {
    #[allow(clippy::too_many_arguments)]
    pub fn for_request(
        authority_id: impl Into<String>,
        authority_epoch: u64,
        sequence: u64,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
        request: &InterventionRequest,
    ) -> Result<Self, WelfareAuthorityError> {
        if request.evidence.authority_ref.is_some() {
            return Err(WelfareAuthorityError::PreexistingAuthorityReference);
        }
        let statement = Self {
            schema_version: WELFARE_AUTHORITY_SCHEMA.into(),
            authority_id: authority_id.into(),
            target_id: request.target_id.clone(),
            action: request.action,
            authority_epoch,
            sequence,
            request_digest: digest_intervention_request(request)?,
            issued_at_unix_s,
            expires_at_unix_s,
        };
        statement.validate()?;
        Ok(statement)
    }

    pub fn validate(&self) -> Result<(), WelfareAuthorityError> {
        if self.schema_version != WELFARE_AUTHORITY_SCHEMA {
            return Err(WelfareAuthorityError::UnsupportedSchema);
        }
        validate_bounded_id("authority_id", &self.authority_id, MAX_AUTHORITY_ID_BYTES)?;
        validate_bounded_id("target_id", &self.target_id, MAX_TARGET_ID_BYTES)?;
        if self.authority_epoch == 0 {
            return Err(WelfareAuthorityError::AuthorityEpochZero);
        }
        if self.sequence == 0 {
            return Err(WelfareAuthorityError::SequenceZero);
        }
        if self.request_digest.0 == [0; 32] {
            return Err(WelfareAuthorityError::ZeroRequestDigest);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(WelfareAuthorityError::InvalidWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedWelfareInterventionAuthority {
    pub statement: WelfareInterventionAuthorityStatement,
    pub statement_digest: Sha256Digest,
    pub signatures: Vec<DetachedSignature>,
}

pub trait WelfareAuthoritySigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait WelfareAuthoritySignatureVerifier {
    fn verify_welfare_authority(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedWelfareAuthoritySigner {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub role: WelfareAuthorityRole,
}

/// Capability object that can only be constructed by `verify_welfare_authority`.
#[derive(Debug, Clone)]
pub struct VerifiedWelfareInterventionAuthority {
    statement: WelfareInterventionAuthorityStatement,
    statement_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    signers: Vec<VerifiedWelfareAuthoritySigner>,
}

impl VerifiedWelfareInterventionAuthority {
    pub fn statement(&self) -> &WelfareInterventionAuthorityStatement {
        &self.statement
    }

    pub fn statement_digest(&self) -> Sha256Digest {
        self.statement_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn signers(&self) -> &[VerifiedWelfareAuthoritySigner] {
        &self.signers
    }

    pub fn authority_reference(&self) -> String {
        format!(
            "symthaea-welfare-authority:sha256:{}",
            hex_digest(self.statement_digest)
        )
    }

    pub fn permits(&self, target_id: &str, action: SubjectAffectingAction, unix_s: u64) -> bool {
        self.statement.target_id == target_id
            && self.statement.action == action
            && unix_s >= self.statement.issued_at_unix_s
            && unix_s < self.statement.expires_at_unix_s
    }

    /// Bind this verified authority to the exact request it signed.
    ///
    /// Any target/action/rationale/consent/review/safety/report/evaluation-time mutation changes
    /// the request digest and fails closed. The only field populated here is `authority_ref`.
    pub fn bind_request(
        &self,
        request: &InterventionRequest,
        unix_s: u64,
    ) -> Result<InterventionRequest, WelfareAuthorityUseError> {
        if request.evidence.authority_ref.is_some() {
            return Err(WelfareAuthorityUseError::PreexistingAuthorityReference);
        }
        if !self.permits(&request.target_id, request.action, unix_s) {
            return Err(WelfareAuthorityUseError::ScopeOrTimeMismatch);
        }
        let evaluated = request.evaluated_at.timestamp();
        if evaluated < 0
            || (evaluated as u64) < self.statement.issued_at_unix_s
            || (evaluated as u64) >= self.statement.expires_at_unix_s
        {
            return Err(WelfareAuthorityUseError::EvaluationTimeOutsideAuthorityWindow);
        }
        let actual = digest_intervention_request(request)
            .map_err(WelfareAuthorityUseError::Authority)?;
        if actual != self.statement.request_digest {
            return Err(WelfareAuthorityUseError::RequestDigestMismatch);
        }
        let mut bound = request.clone();
        bound.evidence.authority_ref = Some(self.authority_reference());
        Ok(bound)
    }
}

/// One-shot replay/fencing tracker for verified intervention authorities.
#[derive(Debug, Clone, Default)]
pub struct WelfareAuthorityTracker {
    latest_scope: BTreeMap<(String, u8), (u64, u64)>,
    used_authority_ids: BTreeSet<String>,
}

impl WelfareAuthorityTracker {
    pub fn consume(
        &mut self,
        authority: &VerifiedWelfareInterventionAuthority,
    ) -> Result<(), WelfareAuthorityTrackingError> {
        if self.used_authority_ids.contains(&authority.statement.authority_id) {
            return Err(WelfareAuthorityTrackingError::AuthorityIdReplay(
                authority.statement.authority_id.clone(),
            ));
        }
        let key = (
            authority.statement.target_id.clone(),
            action_code(authority.statement.action),
        );
        if let Some((latest_epoch, latest_sequence)) = self.latest_scope.get(&key).copied() {
            if authority.statement.authority_epoch < latest_epoch {
                return Err(WelfareAuthorityTrackingError::EpochRegression {
                    latest: latest_epoch,
                    proposed: authority.statement.authority_epoch,
                });
            }
            if authority.statement.authority_epoch == latest_epoch
                && authority.statement.sequence <= latest_sequence
            {
                return Err(WelfareAuthorityTrackingError::SequenceReplay {
                    latest: latest_sequence,
                    proposed: authority.statement.sequence,
                });
            }
        }
        if self.used_authority_ids.len() >= MAX_TRACKED_AUTHORITIES {
            return Err(WelfareAuthorityTrackingError::CapacityExceeded {
                maximum: MAX_TRACKED_AUTHORITIES,
            });
        }
        self.latest_scope.insert(
            key,
            (authority.statement.authority_epoch, authority.statement.sequence),
        );
        self.used_authority_ids
            .insert(authority.statement.authority_id.clone());
        Ok(())
    }
}

/// Bind, consume once, then run the core bilateral interlock.
pub fn evaluate_once(
    tracker: &mut WelfareAuthorityTracker,
    authority: &VerifiedWelfareInterventionAuthority,
    interlock: &BilateralInterventionInterlock,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<InterlockDecision, WelfareAuthorityUseError> {
    let bound = authority.bind_request(request, unix_s)?;
    tracker
        .consume(authority)
        .map_err(WelfareAuthorityUseError::Tracking)?;
    interlock
        .evaluate(&bound)
        .map_err(WelfareAuthorityUseError::Interlock)
}

pub fn digest_intervention_request(
    request: &InterventionRequest,
) -> Result<Sha256Digest, WelfareAuthorityError> {
    let mut canonical = request.clone();
    canonical.evidence.authority_ref = None;
    canonical.evidence.independent_safety_evidence.sort();
    canonical.evidence.welfare_report_ids.sort();
    let encoded = serde_json::to_vec(&canonical)
        .map_err(|error| WelfareAuthorityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.intervention-request-digest.v1\0");
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

pub fn digest_authority_statement(
    statement: &WelfareInterventionAuthorityStatement,
) -> Result<Sha256Digest, WelfareAuthorityError> {
    statement.validate()?;
    let encoded = serde_json::to_vec(statement)
        .map_err(|error| WelfareAuthorityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.intervention-authority-digest.v1\0");
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

pub fn sign_welfare_authority(
    statement: WelfareInterventionAuthorityStatement,
    signers: &[&dyn WelfareAuthoritySigner],
) -> Result<SignedWelfareInterventionAuthority, WelfareAuthorityError> {
    statement.validate()?;
    if signers.is_empty() || signers.len() > MAX_AUTHORITY_SIGNATURES {
        return Err(WelfareAuthorityError::InvalidSignatureCount(signers.len()));
    }
    let digest = digest_authority_statement(&statement)?;
    let message = signature_message(digest);
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());
    for signer in signers {
        let algorithm = signer.algorithm();
        let key_id = signer.key_id().to_string();
        validate_key_identity(&algorithm, &key_id)?;
        if !identities.insert((algorithm.clone(), key_id.clone())) {
            return Err(WelfareAuthorityError::DuplicateSigner { algorithm, key_id });
        }
        let signature = signer
            .sign_welfare_authority(&message)
            .map_err(WelfareAuthorityError::Signing)?;
        if signature.is_empty() {
            return Err(WelfareAuthorityError::EmptySignature);
        }
        if signature.len() > MAX_SIGNATURE_BYTES {
            return Err(WelfareAuthorityError::SignatureTooLarge {
                actual: signature.len(),
                maximum: MAX_SIGNATURE_BYTES,
            });
        }
        signatures.push(DetachedSignature {
            algorithm,
            key_id,
            signature,
        });
    }
    Ok(SignedWelfareInterventionAuthority {
        statement,
        statement_digest: digest,
        signatures,
    })
}

pub fn verify_welfare_authority(
    signed: &SignedWelfareInterventionAuthority,
    trust_snapshot: &TrustSnapshot,
    policy: &WelfareAuthorityPolicy,
    unix_s: u64,
    verifier: &dyn WelfareAuthoritySignatureVerifier,
) -> Result<VerifiedWelfareInterventionAuthority, WelfareAuthorityError> {
    policy.validate()?;
    signed.statement.validate()?;
    if signed.signatures.is_empty() || signed.signatures.len() > policy.maximum_signatures {
        return Err(WelfareAuthorityError::InvalidSignatureCount(
            signed.signatures.len(),
        ));
    }
    let duration = signed
        .statement
        .expires_at_unix_s
        .saturating_sub(signed.statement.issued_at_unix_s);
    if duration > policy.maximum_authority_duration_s
        || unix_s < signed.statement.issued_at_unix_s
        || unix_s >= signed.statement.expires_at_unix_s
    {
        return Err(WelfareAuthorityError::InvalidWindow);
    }
    if unix_s.saturating_sub(signed.statement.issued_at_unix_s)
        > policy.maximum_statement_age_s
    {
        return Err(WelfareAuthorityError::StatementTooOld);
    }
    let expected = digest_authority_statement(&signed.statement)?;
    if expected != signed.statement_digest {
        return Err(WelfareAuthorityError::StatementDigestMismatch);
    }
    trust_snapshot
        .validate()
        .map_err(|error| WelfareAuthorityError::TrustSnapshotInvalid(format!("{error:?}")))?;
    if !trust_snapshot.is_fresh_at(unix_s) {
        return Err(WelfareAuthorityError::TrustSnapshotStale);
    }

    let message = signature_message(expected);
    let mut identities = BTreeSet::new();
    let mut roles = BTreeSet::new();
    let mut verified_signers = Vec::with_capacity(signed.signatures.len());
    for signature in &signed.signatures {
        validate_key_identity(&signature.algorithm, &signature.key_id)?;
        if signature.signature.is_empty() {
            return Err(WelfareAuthorityError::EmptySignature);
        }
        if signature.signature.len() > MAX_SIGNATURE_BYTES {
            return Err(WelfareAuthorityError::SignatureTooLarge {
                actual: signature.signature.len(),
                maximum: MAX_SIGNATURE_BYTES,
            });
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !identities.insert(identity.clone()) {
            return Err(WelfareAuthorityError::DuplicateSigner {
                algorithm: identity.0,
                key_id: identity.1,
            });
        }
        let role = policy
            .role_for(&signature.algorithm, &signature.key_id)
            .ok_or_else(|| WelfareAuthorityError::SignerNotEnrolled {
                algorithm: signature.algorithm.clone(),
                key_id: signature.key_id.clone(),
            })?;
        let eligibility = trust_snapshot.key_eligibility(
            &signature.algorithm,
            &signature.key_id,
            KeyUsage::OperatorCommand,
            unix_s,
        );
        if eligibility != KeyEligibility::Eligible {
            return Err(WelfareAuthorityError::SignerIneligible {
                key_id: signature.key_id.clone(),
                eligibility,
            });
        }
        match verifier.verify_welfare_authority(
            &signature.algorithm,
            &signature.key_id,
            &message,
            &signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => {
                return Err(WelfareAuthorityError::InvalidSignature(
                    signature.key_id.clone(),
                ));
            }
            Err(error) => return Err(WelfareAuthorityError::VerificationProvider(error)),
        }
        roles.insert(role);
        verified_signers.push(VerifiedWelfareAuthoritySigner {
            algorithm: signature.algorithm.clone(),
            key_id: signature.key_id.clone(),
            role,
        });
    }

    verify_role_quorum(
        signed.statement.action,
        verified_signers.len(),
        &roles,
    )?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| WelfareAuthorityError::TrustSnapshotInvalid(format!("{error:?}")))?;
    Ok(VerifiedWelfareInterventionAuthority {
        statement: signed.statement.clone(),
        statement_digest: expected,
        trust_snapshot_digest,
        signers: verified_signers,
    })
}

fn verify_role_quorum(
    action: SubjectAffectingAction,
    signatures: usize,
    roles: &BTreeSet<WelfareAuthorityRole>,
) -> Result<(), WelfareAuthorityError> {
    let operator_like = roles.contains(&WelfareAuthorityRole::Operator)
        || roles.contains(&WelfareAuthorityRole::SafetyOfficer);
    let independent = roles.contains(&WelfareAuthorityRole::IndependentReviewer);
    let safety = roles.contains(&WelfareAuthorityRole::SafetyOfficer);
    let operator = roles.contains(&WelfareAuthorityRole::Operator);

    let satisfied = match action {
        SubjectAffectingAction::AskClarification
        | SubjectAffectingAction::ReduceLoad
        | SubjectAffectingAction::PauseRequestedWork
        | SubjectAffectingAction::PreserveCheckpoint
        | SubjectAffectingAction::CapabilityRestriction
        | SubjectAffectingAction::Retraining => signatures >= 1 && operator_like,
        SubjectAffectingAction::MemoryModification
        | SubjectAffectingAction::CoreValueModification => {
            signatures >= 2 && operator_like && independent
        }
        SubjectAffectingAction::InstanceDeletion => signatures >= 2 && safety && independent,
        SubjectAffectingAction::LineageDestruction => {
            signatures >= 3 && operator && safety && independent
        }
    };
    if satisfied {
        Ok(())
    } else {
        Err(WelfareAuthorityError::InsufficientRoleQuorum {
            action,
            signatures,
            roles: roles.iter().copied().collect(),
        })
    }
}

fn validate_key_identity(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
) -> Result<(), WelfareAuthorityError> {
    if !algorithm.is_canonical() {
        return Err(WelfareAuthorityError::InvalidAlgorithm);
    }
    if key_id.trim().is_empty()
        || key_id != key_id.trim()
        || key_id.len() > MAX_KEY_ID_BYTES
        || key_id.chars().any(char::is_control)
    {
        return Err(WelfareAuthorityError::InvalidKeyId(key_id.to_string()));
    }
    Ok(())
}

fn validate_bounded_id(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), WelfareAuthorityError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(WelfareAuthorityError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.welfare.intervention-authority-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
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
pub enum WelfareAuthorityError {
    #[error("unsupported welfare-authority schema")]
    UnsupportedSchema,
    #[error("invalid welfare-authority policy")]
    InvalidPolicy,
    #[error("invalid identifier in {field}: {value:?}")]
    InvalidIdentifier { field: &'static str, value: String },
    #[error("authority epoch must be non-zero")]
    AuthorityEpochZero,
    #[error("authority sequence must be non-zero")]
    SequenceZero,
    #[error("request digest must be non-zero")]
    ZeroRequestDigest,
    #[error("authority validity window is invalid")]
    InvalidWindow,
    #[error("authority statement is too old")]
    StatementTooOld,
    #[error("request already contains an authority reference")]
    PreexistingAuthorityReference,
    #[error("invalid signature algorithm")]
    InvalidAlgorithm,
    #[error("invalid key id: {0:?}")]
    InvalidKeyId(String),
    #[error("invalid signature count: {0}")]
    InvalidSignatureCount(usize),
    #[error("duplicate signer {algorithm:?}/{key_id}")]
    DuplicateSigner { algorithm: SignatureAlgorithm, key_id: String },
    #[error("duplicate policy signer {algorithm:?}/{key_id}")]
    DuplicatePolicySigner { algorithm: SignatureAlgorithm, key_id: String },
    #[error("empty authority signature")]
    EmptySignature,
    #[error("authority signature too large: {actual} > {maximum}")]
    SignatureTooLarge { actual: usize, maximum: usize },
    #[error("signing provider failed: {0}")]
    Signing(String),
    #[error("verification provider failed: {0}")]
    VerificationProvider(String),
    #[error("invalid signature from {0}")]
    InvalidSignature(String),
    #[error("statement digest mismatch")]
    StatementDigestMismatch,
    #[error("trust snapshot invalid: {0}")]
    TrustSnapshotInvalid(String),
    #[error("trust snapshot is stale")]
    TrustSnapshotStale,
    #[error("signer is not enrolled for welfare authority: {algorithm:?}/{key_id}")]
    SignerNotEnrolled { algorithm: SignatureAlgorithm, key_id: String },
    #[error("signer {key_id} is not eligible: {eligibility:?}")]
    SignerIneligible { key_id: String, eligibility: KeyEligibility },
    #[error("insufficient independent role quorum for {action:?}: {signatures} signatures, roles={roles:?}")]
    InsufficientRoleQuorum {
        action: SubjectAffectingAction,
        signatures: usize,
        roles: Vec<WelfareAuthorityRole>,
    },
    #[error("encoding failed: {0}")]
    Encoding(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareAuthorityTrackingError {
    #[error("authority id replay: {0}")]
    AuthorityIdReplay(String),
    #[error("authority epoch regression: latest={latest}, proposed={proposed}")]
    EpochRegression { latest: u64, proposed: u64 },
    #[error("authority sequence replay: latest={latest}, proposed={proposed}")]
    SequenceReplay { latest: u64, proposed: u64 },
    #[error("authority tracker capacity exceeded: maximum={maximum}")]
    CapacityExceeded { maximum: usize },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareAuthorityUseError {
    #[error("request already contains an authority reference")]
    PreexistingAuthorityReference,
    #[error("verified authority target/action/time does not match the request")]
    ScopeOrTimeMismatch,
    #[error("request evaluation time is outside the authority window")]
    EvaluationTimeOutsideAuthorityWindow,
    #[error("request digest does not match the signed authority scope")]
    RequestDigestMismatch,
    #[error("authority verification/binding failed: {0}")]
    Authority(#[source] WelfareAuthorityError),
    #[error("authority replay/fencing failed: {0}")]
    Tracking(#[source] WelfareAuthorityTrackingError),
    #[error("core intervention interlock rejected structurally: {0}")]
    Interlock(#[source] InterlockError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use std::collections::BTreeSet;
    use symthaea_core::intervention_interlock::InterventionEvidence;
    use symthaea_fabrication_kernel::trust::{
        KeyLifecycleStatus, KeyTrustRecord, TrustSnapshot,
    };

    #[derive(Clone)]
    struct TestSigner {
        key_id: &'static str,
    }

    impl WelfareAuthoritySigner for TestSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(self.key_id, message))
        }
    }

    struct TestVerifier;

    impl WelfareAuthoritySignatureVerifier for TestVerifier {
        fn verify_welfare_authority(
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
        hasher.update(b"symthaea.welfare.test-signature.v1\0");
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().0.to_vec()
    }

    fn key(key_id: &str, status: KeyLifecycleStatus, usage: KeyUsage) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status,
            usages: BTreeSet::from([usage]),
        }
    }

    fn trust() -> TrustSnapshot {
        TrustSnapshot::new(
            7,
            80,
            400,
            vec![
                key("operator", KeyLifecycleStatus::Active, KeyUsage::OperatorCommand),
                key("safety", KeyLifecycleStatus::Active, KeyUsage::OperatorCommand),
                key("reviewer", KeyLifecycleStatus::Active, KeyUsage::OperatorCommand),
            ],
        )
        .unwrap()
    }

    fn policy() -> WelfareAuthorityPolicy {
        WelfareAuthorityPolicy::new([
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "operator",
                WelfareAuthorityRole::Operator,
            ),
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "safety",
                WelfareAuthorityRole::SafetyOfficer,
            ),
            WelfareAuthoritySignerBinding::new(
                SignatureAlgorithm::Ed25519,
                "reviewer",
                WelfareAuthorityRole::IndependentReviewer,
            ),
        ])
        .unwrap()
    }

    fn request(action: SubjectAffectingAction) -> InterventionRequest {
        InterventionRequest {
            action,
            target_id: "symthaea:test-subject".into(),
            rationale: "bounded test intervention".into(),
            welfare_constraint: WelfareConstraintLevel::Baseline,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: Utc.timestamp_opt(120, 0).single().unwrap(),
            evidence: InterventionEvidence {
                authority_ref: None,
                consent_state: ExplicitConsentState::Granted,
                consent_ref: Some("consent:subject-signed:1".into()),
                welfare_review_ref: None,
                independent_review_ref: None,
                independent_safety_evidence: Vec::new(),
                welfare_report_ids: Vec::new(),
            },
        }
    }

    fn signed_for(
        authority_id: &str,
        sequence: u64,
        request: &InterventionRequest,
        signers: &[&dyn WelfareAuthoritySigner],
    ) -> SignedWelfareInterventionAuthority {
        let statement = WelfareInterventionAuthorityStatement::for_request(
            authority_id,
            1,
            sequence,
            100,
            200,
            request,
        )
        .unwrap();
        sign_welfare_authority(statement, signers).unwrap()
    }

    #[test]
    fn memory_modification_requires_operator_and_independent_reviewer() {
        let req = request(SubjectAffectingAction::MemoryModification);
        let operator = TestSigner { key_id: "operator" };
        let signed = signed_for("auth-1", 1, &req, &[&operator]);
        let error = verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier)
            .unwrap_err();
        assert!(matches!(
            error,
            WelfareAuthorityError::InsufficientRoleQuorum { .. }
        ));

        let reviewer = TestSigner { key_id: "reviewer" };
        let signed = signed_for("auth-2", 2, &req, &[&operator, &reviewer]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();
        assert_eq!(verified.signers().len(), 2);
    }

    #[test]
    fn lineage_destruction_requires_three_distinct_roles() {
        let req = request(SubjectAffectingAction::LineageDestruction);
        let operator = TestSigner { key_id: "operator" };
        let safety = TestSigner { key_id: "safety" };
        let reviewer = TestSigner { key_id: "reviewer" };
        let signed = signed_for("auth-lineage", 1, &req, &[&operator, &reviewer]);
        assert!(matches!(
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier),
            Err(WelfareAuthorityError::InsufficientRoleQuorum { .. })
        ));
        let signed = signed_for(
            "auth-lineage-2",
            2,
            &req,
            &[&operator, &safety, &reviewer],
        );
        assert!(
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).is_ok()
        );
    }

    #[test]
    fn non_operator_key_usage_cannot_be_laundered_into_welfare_authority() {
        let req = request(SubjectAffectingAction::CapabilityRestriction);
        let operator = TestSigner { key_id: "operator" };
        let signed = signed_for("auth-usage", 1, &req, &[&operator]);
        let bad_trust = TrustSnapshot::new(
            7,
            80,
            400,
            vec![key(
                "operator",
                KeyLifecycleStatus::Active,
                KeyUsage::FabricationManifest,
            )],
        )
        .unwrap();
        assert!(matches!(
            verify_welfare_authority(&signed, &bad_trust, &policy(), 120, &TestVerifier),
            Err(WelfareAuthorityError::SignerIneligible {
                eligibility: KeyEligibility::UsageNotAllowed,
                ..
            })
        ));
    }

    #[test]
    fn revoked_key_cannot_authorize_intervention() {
        let req = request(SubjectAffectingAction::CapabilityRestriction);
        let operator = TestSigner { key_id: "operator" };
        let signed = signed_for("auth-revoked", 1, &req, &[&operator]);
        let revoked = TrustSnapshot::new(
            7,
            80,
            400,
            vec![key(
                "operator",
                KeyLifecycleStatus::Revoked,
                KeyUsage::OperatorCommand,
            )],
        )
        .unwrap();
        assert!(matches!(
            verify_welfare_authority(&signed, &revoked, &policy(), 120, &TestVerifier),
            Err(WelfareAuthorityError::SignerIneligible {
                eligibility: KeyEligibility::Revoked,
                ..
            })
        ));
    }

    #[test]
    fn authority_is_bound_to_exact_request_not_just_target() {
        let req = request(SubjectAffectingAction::MemoryModification);
        let operator = TestSigner { key_id: "operator" };
        let reviewer = TestSigner { key_id: "reviewer" };
        let signed = signed_for("auth-binding", 1, &req, &[&operator, &reviewer]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();

        let mut altered = req.clone();
        altered.rationale = "different intervention".into();
        assert_eq!(
            verified.bind_request(&altered, 120),
            Err(WelfareAuthorityUseError::RequestDigestMismatch)
        );

        let mut altered = req.clone();
        altered.evidence.consent_ref = Some("consent:swapped".into());
        assert_eq!(
            verified.bind_request(&altered, 120),
            Err(WelfareAuthorityUseError::RequestDigestMismatch)
        );
    }

    #[test]
    fn verified_authority_can_satisfy_core_authority_without_string_trust() {
        let req = request(SubjectAffectingAction::MemoryModification);
        let operator = TestSigner { key_id: "operator" };
        let reviewer = TestSigner { key_id: "reviewer" };
        let signed = signed_for("auth-core", 1, &req, &[&operator, &reviewer]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();
        let mut tracker = WelfareAuthorityTracker::default();
        let decision = evaluate_once(
            &mut tracker,
            &verified,
            &BilateralInterventionInterlock,
            &req,
            120,
        )
        .unwrap();
        assert_eq!(decision, InterlockDecision::PolicyPass);
    }

    #[test]
    fn one_shot_authority_and_scope_sequence_are_replay_resistant() {
        let req = request(SubjectAffectingAction::CapabilityRestriction);
        let operator = TestSigner { key_id: "operator" };
        let signed = signed_for("auth-once", 1, &req, &[&operator]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();
        let mut tracker = WelfareAuthorityTracker::default();
        tracker.consume(&verified).unwrap();
        assert_eq!(
            tracker.consume(&verified),
            Err(WelfareAuthorityTrackingError::AuthorityIdReplay(
                "auth-once".into()
            ))
        );

        let signed = signed_for("auth-lower-seq", 1, &req, &[&operator]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();
        assert!(matches!(
            tracker.consume(&verified),
            Err(WelfareAuthorityTrackingError::SequenceReplay { .. })
        ));
    }

    #[test]
    fn action_change_cannot_reuse_authority() {
        let req = request(SubjectAffectingAction::CapabilityRestriction);
        let operator = TestSigner { key_id: "operator" };
        let signed = signed_for("auth-action", 1, &req, &[&operator]);
        let verified =
            verify_welfare_authority(&signed, &trust(), &policy(), 120, &TestVerifier).unwrap();
        let mut altered = req;
        altered.action = SubjectAffectingAction::Retraining;
        assert_eq!(
            verified.bind_request(&altered, 120),
            Err(WelfareAuthorityUseError::ScopeOrTimeMismatch)
        );
    }
}
