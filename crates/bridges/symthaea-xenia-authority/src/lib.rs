// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent Xenia authority verifier for the Symthaea Agency Kernel.
//!
//! Xenia signs an opaque commitment to a Symthaea [`CapabilityGrant`]. This
//! crate independently reconstructs Xenia V1 canonical bytes, verifies the
//! ledger authority signature, binds it to exact executor workload identity,
//! requires the exact current Symthaea anti-rollback checkpoint, and requires a
//! fresh signed Xenia ledger checkpoint at the same frontier.
//!
//! Signature validity is necessary but not sufficient for live authority. A
//! stale signed authorization is rejected when its ledger frontier no longer
//! matches the fresh Xenia checkpoint supplied for admission. Wall-clock time is
//! not accepted from the caller: time-sensitive checks consume a short-lived,
//! challenge-bound [`VerifiedAuthorityTime`] fact. Current authority epoch and
//! negative-authority facts likewise arrive as one indivisible, fresh
//! [`VerifiedAuthorityState`] snapshot rather than caller-selected inputs.

#![deny(unsafe_code)]

mod protocol;
mod workload;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{CapabilityGrant, Digest32, PrincipalId};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityState};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub use protocol::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, AGENT_CAPABILITY_AUTHORIZATION_DOMAIN,
    AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION, ED25519_SIGNATURE_ALGORITHM,
    ProtocolError, TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA,
    XeniaAgentAuthorizationV1, XeniaAgentCapabilityAttestationV1, XeniaCheckpointAnchorV1,
    XeniaLedgerCheckpointV1, XeniaSignatureEnvelopeV1,
};
pub use workload::{ExecutorWorkloadV1, WorkloadIdentityError};

/// Session provenance that the Xenia authority statement must bind exactly.
///
/// This type does not itself verify a Xenia handshake transcript signature.
/// It is the expected session identity supplied by the authenticated Xenia
/// integration boundary and is cryptographically covered by the agent
/// capability attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaSessionExpectationV1 {
    pub session_id: [u8; 16],
    pub transcript_hash: [u8; 32],
    pub transcript_signature_suite: TranscriptSignatureSuiteV1,
}

/// Freshness policy for the independently signed Xenia ledger checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaFreshnessPolicyV1 {
    /// Maximum checkpoint age accepted at effect admission.
    pub max_checkpoint_age_s: u64,
    /// Maximum tolerated checkpoint timestamp ahead of verified authority time.
    pub max_future_skew_s: u64,
}

impl XeniaFreshnessPolicyV1 {
    pub fn strict(max_checkpoint_age_s: u64, max_future_skew_s: u64) -> Self {
        Self {
            max_checkpoint_age_s,
            max_future_skew_s,
        }
    }
}

/// Affine proof that one exact Xenia attestation passed all V1 admission checks.
///
/// This type intentionally does not implement `Clone`. It is still not a
/// durable one-use reservation by itself; the Action Runtime must consume it in
/// the same transaction that reserves the underlying `CapabilityGrant`.
#[derive(Debug)]
pub struct VerifiedXeniaCapability {
    authorization_id: [u8; 16],
    session_id: [u8; 16],
    grant_digest: Digest32,
    workload_digest: Digest32,
    xenia_ledger_entry_count: u64,
    xenia_ledger_head_hash: [u8; 32],
    prior_checkpoint: CheckpointHead,
    authority_state_digest: Digest32,
    authority_state_sequence: u64,
    expires_at_unix_s: u64,
}

impl VerifiedXeniaCapability {
    pub fn authorization_id(&self) -> [u8; 16] {
        self.authorization_id
    }

    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn workload_digest(&self) -> Digest32 {
        self.workload_digest
    }

    pub fn xenia_frontier(&self) -> (u64, [u8; 32]) {
        (self.xenia_ledger_entry_count, self.xenia_ledger_head_hash)
    }

    pub fn prior_checkpoint(&self) -> CheckpointHead {
        self.prior_checkpoint
    }

    pub fn authority_state_digest(&self) -> Digest32 {
        self.authority_state_digest
    }

    pub fn authority_state_sequence(&self) -> u64 {
        self.authority_state_sequence
    }

    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }
}

/// Verify a Xenia-bound Symthaea capability before the Action Runtime may
/// reserve it for consequential execution.
///
/// `authority_time` must be challenge-bound to the exact capability digest.
/// Time is an interval, not a scalar: earliest-plausible time is used to prove
/// that an authorization/checkpoint is not from the future, while the
/// latest-plausible time is used to prove it has not expired or gone stale.
///
/// `authority_state` must be a fresh threshold-authenticated snapshot for the
/// same grant. Its source frontier must be exactly the fresh Xenia ledger
/// frontier used in this verification. This prevents mixing a fresh Xenia proof
/// with a stale epoch or stale revocation set.
#[allow(clippy::too_many_arguments)]
pub fn verify_xenia_capability_v1(
    attestation: &XeniaAgentCapabilityAttestationV1,
    fresh_xenia_checkpoint: &XeniaLedgerCheckpointV1,
    trusted_xenia_ledger_public_key: [u8; 32],
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    expected_session: XeniaSessionExpectationV1,
    current_agent_checkpoint: CheckpointHead,
    authority_time: &VerifiedAuthorityTime,
    authority_state: &VerifiedAuthorityState,
    freshness: XeniaFreshnessPolicyV1,
) -> Result<VerifiedXeniaCapability, XeniaAuthorityError> {
    let authorization = &attestation.authorization;
    authorization.validate()?;

    if attestation.schema != AGENT_CAPABILITY_ATTESTATION_SCHEMA {
        return Err(XeniaAuthorityError::UnsupportedAttestationSchema);
    }
    if attestation.signature.algorithm != ED25519_SIGNATURE_ALGORITHM {
        return Err(XeniaAuthorityError::UnsupportedAttestationSignatureSuite);
    }

    let grant_digest = grant.digest();
    authority_time.require_subject(grant_digest.0)?;
    authority_state.ensure_fresh(grant, authority_time)?;
    if authority_state.authority_epoch() != grant.authority_epoch {
        return Err(XeniaAuthorityError::GrantEpochStaleAgainstCurrentState);
    }
    let (state_frontier_sequence, state_frontier_digest) = authority_state.source_frontier();
    if state_frontier_sequence != fresh_xenia_checkpoint.entry_count
        || state_frontier_digest.0 != fresh_xenia_checkpoint.head_hash
    {
        return Err(XeniaAuthorityError::AuthorityStateFrontierMismatch);
    }

    // The verification-time lower bound remains a valid conservative lower
    // bound during this fact's very short lifetime because real time can only
    // advance. The moving upper bound accounts for elapsed Linux boot time.
    let (not_before_unix_s, _) = authority_time.interval_at_verification();
    let not_after_unix_s = authority_time.conservative_now_unix_s()?;

    verify_fresh_xenia_checkpoint(
        fresh_xenia_checkpoint,
        trusted_xenia_ledger_public_key,
        not_before_unix_s,
        not_after_unix_s,
        freshness,
    )?;

    if fresh_xenia_checkpoint.entry_count != authorization.ledger_entry_count
        || fresh_xenia_checkpoint.head_hash != authorization.ledger_head_hash
    {
        return Err(XeniaAuthorityError::AuthorizationFrontierStale);
    }
    if fresh_xenia_checkpoint.timestamp_unix_secs < authorization.issued_at_unix_s {
        return Err(XeniaAuthorityError::FreshnessCheckpointPredatesAuthorization);
    }

    let expected_fingerprint = *blake3::hash(&trusted_xenia_ledger_public_key).as_bytes();
    if attestation.ledger_public_key_fingerprint != expected_fingerprint {
        return Err(XeniaAuthorityError::SignerFingerprintMismatch);
    }
    verify_ed25519(
        trusted_xenia_ledger_public_key,
        &authorization.canonical_message()?,
        &attestation.signature.signature,
    )?;

    // To prove "already valid", even the earliest plausible current time must
    // be at/after issuance. To prove "not expired", even the latest plausible
    // current time must be at/before expiry.
    if not_before_unix_s < authorization.issued_at_unix_s {
        return Err(XeniaAuthorityError::AuthorizationNotYetValid);
    }
    if not_after_unix_s > authorization.expires_at_unix_s {
        return Err(XeniaAuthorityError::AuthorizationExpired);
    }

    if authorization.capability_digest != grant_digest.0 {
        return Err(XeniaAuthorityError::CapabilityDigestMismatch);
    }
    if authorization.authority_epoch != grant.authority_epoch.0 {
        return Err(XeniaAuthorityError::AuthorityEpochMismatch);
    }
    if grant
        .expires_at_unix_s
        .is_some_and(|grant_expiry| authorization.expires_at_unix_s > grant_expiry)
    {
        return Err(XeniaAuthorityError::AuthorizationOutlivesGrant);
    }

    workload.validate()?;
    let expected_executor = grant
        .audience
        .as_ref()
        .ok_or(XeniaAuthorityError::GrantMissingExecutorAudience)?;
    if &workload.executor != expected_executor {
        return Err(XeniaAuthorityError::WorkloadExecutorMismatch);
    }
    let workload_digest = workload.digest()?;
    if authorization.executor_workload_digest != workload_digest.0 {
        return Err(XeniaAuthorityError::WorkloadDigestMismatch);
    }

    if authorization.session_id != expected_session.session_id
        || authorization.session_transcript_hash != expected_session.transcript_hash
        || authorization.session_signature_suite != expected_session.transcript_signature_suite
    {
        return Err(XeniaAuthorityError::SessionBindingMismatch);
    }

    let expected_checkpoint = XeniaCheckpointAnchorV1 {
        sequence: current_agent_checkpoint.sequence,
        digest: current_agent_checkpoint.digest.0,
    };
    if authorization.prior_checkpoint != Some(expected_checkpoint) {
        return Err(XeniaAuthorityError::AgentCheckpointMismatch);
    }

    Ok(VerifiedXeniaCapability {
        authorization_id: authorization.authorization_id,
        session_id: authorization.session_id,
        grant_digest,
        workload_digest,
        xenia_ledger_entry_count: authorization.ledger_entry_count,
        xenia_ledger_head_hash: authorization.ledger_head_hash,
        prior_checkpoint: current_agent_checkpoint,
        authority_state_digest: authority_state.snapshot_digest(),
        authority_state_sequence: authority_state.state_sequence(),
        expires_at_unix_s: authorization.expires_at_unix_s,
    })
}

fn verify_fresh_xenia_checkpoint(
    checkpoint: &XeniaLedgerCheckpointV1,
    trusted_public_key: [u8; 32],
    not_before_unix_s: u64,
    not_after_unix_s: u64,
    freshness: XeniaFreshnessPolicyV1,
) -> Result<(), XeniaAuthorityError> {
    if checkpoint.ledger_public_key != trusted_public_key {
        return Err(XeniaAuthorityError::LedgerCheckpointKeyMismatch);
    }
    if checkpoint.entry_count == 0 || checkpoint.head_hash == [0; 32] {
        return Err(XeniaAuthorityError::PreGenesisFreshnessCheckpoint);
    }
    // A checkpoint is acceptably non-future only if it is no later than the
    // earliest plausible current time plus explicitly tolerated skew.
    if checkpoint.timestamp_unix_secs
        > not_before_unix_s.saturating_add(freshness.max_future_skew_s)
    {
        return Err(XeniaAuthorityError::LedgerCheckpointFromFuture);
    }
    // A checkpoint is acceptably fresh only if it remains within age even at
    // the latest plausible current time.
    if not_after_unix_s.saturating_sub(checkpoint.timestamp_unix_secs)
        > freshness.max_checkpoint_age_s
    {
        return Err(XeniaAuthorityError::LedgerCheckpointStale);
    }
    verify_ed25519(
        trusted_public_key,
        &checkpoint.signature_message()?,
        &checkpoint.signature,
    )
}

fn verify_ed25519(
    public_key: [u8; 32],
    message: &[u8],
    signature: &[u8],
) -> Result<(), XeniaAuthorityError> {
    let verifying_key = VerifyingKey::from_bytes(&public_key)
        .map_err(|_| XeniaAuthorityError::BadEd25519PublicKey)?;
    let signature: [u8; 64] = signature
        .try_into()
        .map_err(|_| XeniaAuthorityError::BadEd25519SignatureLength)?;
    verifying_key
        .verify(message, &Signature::from_bytes(&signature))
        .map_err(|_| XeniaAuthorityError::BadEd25519Signature)
}

#[derive(Debug, Error)]
pub enum XeniaAuthorityError {
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("verified authority state failed: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("invalid Xenia protocol object: {0}")]
    Protocol(#[from] ProtocolError),
    #[error("invalid executor workload: {0}")]
    Workload(#[from] WorkloadIdentityError),
    #[error("unsupported Xenia capability attestation schema")]
    UnsupportedAttestationSchema,
    #[error("V1 accepts only Ed25519 Xenia capability attestations")]
    UnsupportedAttestationSignatureSuite,
    #[error("trusted Xenia ledger public key does not match freshness checkpoint")]
    LedgerCheckpointKeyMismatch,
    #[error("freshness checkpoint is pre-genesis")]
    PreGenesisFreshnessCheckpoint,
    #[error("freshness checkpoint timestamp is implausibly in the future")]
    LedgerCheckpointFromFuture,
    #[error("freshness checkpoint exceeded the configured maximum age")]
    LedgerCheckpointStale,
    #[error("fresh authority-state source frontier differs from the signed Xenia checkpoint")]
    AuthorityStateFrontierMismatch,
    #[error("capability grant epoch is stale relative to fresh authority-state evidence")]
    GrantEpochStaleAgainstCurrentState,
    #[error("fresh Xenia frontier differs from signed authorization frontier")]
    AuthorizationFrontierStale,
    #[error("freshness checkpoint predates the signed authorization")]
    FreshnessCheckpointPredatesAuthorization,
    #[error("attestation signer fingerprint does not match trusted Xenia key")]
    SignerFingerprintMismatch,
    #[error("malformed Ed25519 public key")]
    BadEd25519PublicKey,
    #[error("Ed25519 signature must be exactly 64 bytes")]
    BadEd25519SignatureLength,
    #[error("Ed25519 signature verification failed")]
    BadEd25519Signature,
    #[error("Xenia authorization is not yet valid")]
    AuthorizationNotYetValid,
    #[error("Xenia authorization has expired")]
    AuthorizationExpired,
    #[error("Xenia authorization does not bind this exact CapabilityGrant")]
    CapabilityDigestMismatch,
    #[error("Xenia authorization authority epoch does not match the grant")]
    AuthorityEpochMismatch,
    #[error("Xenia authorization outlives the capability grant")]
    AuthorizationOutlivesGrant,
    #[error("capability grant must have an exact executor audience")]
    GrantMissingExecutorAudience,
    #[error("measured workload executor does not match capability audience")]
    WorkloadExecutorMismatch,
    #[error("Xenia authorization does not bind this exact measured workload")]
    WorkloadDigestMismatch,
    #[error("Xenia authorization does not bind the expected session provenance")]
    SessionBindingMismatch,
    #[error("Xenia authorization does not bind the exact current Agency Kernel checkpoint")]
    AgentCheckpointMismatch,
}

/// Convenience helper for callers constructing workload identities.
pub fn principal(value: impl Into<String>) -> PrincipalId {
    PrincipalId(value.into())
}
