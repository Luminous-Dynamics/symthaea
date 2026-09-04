// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent Xenia authority verifier for the Symthaea Agency Kernel.
//!
//! Xenia signs opaque commitments that Symthaea verifies independently. Live
//! capability authority and qualification-witness chronology remain separate
//! verification domains: both may use the trusted Xenia ledger key, but witness
//! evidence never creates or restores execution authority.
//!
//! Signature validity is necessary but not sufficient for live authority or
//! current chronology. Trusted time and source state remain separate
//! environmental facts.

#![deny(unsafe_code)]

mod protocol;
mod witness_frontier;
mod workload;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{CapabilityGrant, Digest32, PrincipalId};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityState};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub use protocol::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, AGENT_CAPABILITY_AUTHORIZATION_DOMAIN,
    AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION, ED25519_SIGNATURE_ALGORITHM, ProtocolError,
    TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA, XeniaAgentAuthorizationV1,
    XeniaAgentCapabilityAttestationV1, XeniaCheckpointAnchorV1, XeniaLedgerCheckpointV1,
    XeniaSignatureEnvelopeV1,
};
pub use witness_frontier::{
    SYMTHAEA_WITNESS_ANCHOR_OPERATION_DOMAIN, SYMTHAEA_WITNESS_FRONTIER_STATEMENT_DOMAIN,
    SYMTHAEA_WITNESS_FRONTIER_STATEMENT_SCHEMA_VERSION, VerifiedXeniaWitnessFrontierV1,
    XENIA_WITNESS_FRONTIER_ANCHOR_DOMAIN, XENIA_WITNESS_FRONTIER_ANCHOR_FINGERPRINT_DOMAIN,
    XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION, XENIA_WITNESS_FRONTIER_OBSERVATION_DOMAIN,
    XENIA_WITNESS_FRONTIER_OBSERVATION_FINGERPRINT_DOMAIN, XENIA_WITNESS_FRONTIER_SOURCE_DOMAIN,
    XeniaSignedWitnessFrontierAnchorV1, XeniaSignedWitnessFrontierObservationV1,
    XeniaWitnessFrontierAnchorSummaryV1, XeniaWitnessFrontierAnchorTargetV1,
    XeniaWitnessFrontierError, XeniaWitnessFrontierExpectationV1,
    derive_xenia_witness_frontier_source_id, witness_frontier_statement_digest,
};
pub use workload::{ExecutorWorkloadV1, VerifiedExecutorWorkload, WorkloadIdentityError};

/// Authority-time subject domain for one exact Xenia witness-currentness check.
pub const XENIA_WITNESS_FRONTIER_TIME_SUBJECT_DOMAIN: &[u8] =
    b"symthaea.xenia-witness-frontier.time-subject.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaSessionExpectationV1 {
    pub session_id: [u8; 16],
    pub transcript_hash: [u8; 32],
    pub transcript_signature_suite: TranscriptSignatureSuiteV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaFreshnessPolicyV1 {
    pub max_checkpoint_age_s: u64,
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

/// Reviewed freshness limits for challenge-bound Xenia witness observations.
///
/// The values must be configured together with the trusted Xenia
/// `anchor_policy_digest`; this type does not reinterpret that opaque source
/// policy commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XeniaWitnessFrontierFreshnessPolicyV1 {
    pub max_observation_age_s: u64,
    pub max_future_skew_s: u64,
}

impl XeniaWitnessFrontierFreshnessPolicyV1 {
    pub fn strict(max_observation_age_s: u64, max_future_skew_s: u64) -> Self {
        Self {
            max_observation_age_s,
            max_future_skew_s,
        }
    }

    fn validate(self) -> Result<(), XeniaWitnessFrontierVerificationError> {
        if self.max_observation_age_s == 0 {
            return Err(XeniaWitnessFrontierVerificationError::InvalidFreshnessPolicy);
        }
        Ok(())
    }
}

/// Build the exact subject that a [`VerifiedAuthorityTime`] fact must bind before
/// it may establish freshness for one Xenia witness-frontier observation.
///
/// The subject commits the trusted Xenia key, derived source namespace, source
/// epoch, reviewed anchor policy, witness, verifier challenge, and exact signed
/// durable-anchor fingerprint. The anchor signature is verified before the
/// subject is returned.
pub fn xenia_witness_frontier_time_subject_digest_v1(
    anchor: &XeniaSignedWitnessFrontierAnchorV1,
    expected: XeniaWitnessFrontierExpectationV1,
) -> Result<[u8; 32], XeniaWitnessFrontierVerificationError> {
    if expected.trusted_ledger_public_key == [0; 32]
        || expected.source_epoch == 0
        || expected.anchor_policy_digest == [0; 32]
        || expected.witness_id == [0; 16]
        || expected.challenge == [0; 32]
    {
        return Err(XeniaWitnessFrontierVerificationError::InvalidExpectation);
    }

    anchor
        .verify_with_trusted_key(expected.trusted_ledger_public_key)
        .map_err(XeniaWitnessFrontierVerificationError::Evidence)?;
    let source_id = derive_xenia_witness_frontier_source_id(
        expected.trusted_ledger_public_key,
        expected.anchor_policy_digest,
    )
    .map_err(XeniaWitnessFrontierVerificationError::Evidence)?;
    if anchor.target.source_id != source_id
        || anchor.target.source_epoch != expected.source_epoch
        || anchor.target.anchor_policy_digest != expected.anchor_policy_digest
        || anchor.target.witness_id != expected.witness_id
    {
        return Err(XeniaWitnessFrontierVerificationError::InvalidExpectation);
    }
    let anchor_fingerprint = anchor
        .fingerprint()
        .map_err(XeniaWitnessFrontierVerificationError::Evidence)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(XENIA_WITNESS_FRONTIER_TIME_SUBJECT_DOMAIN);
    hasher.update(&XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION.to_be_bytes());
    hasher.update(&expected.trusted_ledger_public_key);
    hasher.update(&source_id);
    hasher.update(&expected.source_epoch.to_be_bytes());
    hasher.update(&expected.anchor_policy_digest);
    hasher.update(&expected.witness_id);
    hasher.update(&expected.challenge);
    hasher.update(&anchor_fingerprint);
    Ok(*hasher.finalize().as_bytes())
}

/// Public Xenia witness-frontier verification boundary.
///
/// Unlike the protocol-only checker inside the private module, this API does not
/// accept a caller-selected wall-clock interval. It requires a short-lived
/// [`VerifiedAuthorityTime`] fact bound to this exact Xenia key/source/witness,
/// verifier challenge, and signed durable anchor.
pub fn verify_xenia_witness_frontier_v1(
    anchor: &XeniaSignedWitnessFrontierAnchorV1,
    observation: &XeniaSignedWitnessFrontierObservationV1,
    expected: XeniaWitnessFrontierExpectationV1,
    authority_time: &VerifiedAuthorityTime,
    freshness: XeniaWitnessFrontierFreshnessPolicyV1,
) -> Result<VerifiedXeniaWitnessFrontierV1, XeniaWitnessFrontierVerificationError> {
    freshness.validate()?;
    let time_subject = xenia_witness_frontier_time_subject_digest_v1(anchor, expected)?;
    authority_time
        .require_subject(time_subject)
        .map_err(|_| XeniaWitnessFrontierVerificationError::AuthorityTimeRejected)?;

    let (earliest_now_unix_s, _) = authority_time.interval_at_verification();
    let latest_now_unix_s = authority_time
        .conservative_now_unix_s()
        .map_err(|_| XeniaWitnessFrontierVerificationError::AuthorityTimeRejected)?;

    witness_frontier::verify_xenia_witness_frontier_v1(
        anchor,
        observation,
        expected,
        witness_frontier::XeniaWitnessObservationFreshnessV1 {
            earliest_now_unix_s,
            latest_now_unix_s,
            max_age_s: freshness.max_observation_age_s,
            max_future_skew_s: freshness.max_future_skew_s,
        },
    )
    .map_err(XeniaWitnessFrontierVerificationError::Evidence)
}

#[derive(Debug, Error)]
pub enum XeniaWitnessFrontierVerificationError {
    #[error("Xenia witness-frontier expectation is invalid")]
    InvalidExpectation,
    #[error("Xenia witness-frontier freshness policy is invalid")]
    InvalidFreshnessPolicy,
    #[error("verified authority-time fact rejected the witness-currentness check")]
    AuthorityTimeRejected,
    #[error("Xenia witness-frontier evidence rejected: {0}")]
    Evidence(#[from] XeniaWitnessFrontierError),
}

/// Affine proof that one exact Xenia attestation passed all V1 admission checks.
///
/// The proof owns both environmental objects whose substitution would otherwise
/// be dangerous: current authority state and the exact witnessed executor
/// workload. It remains distinct from durable one-use accounting, which is
/// provided by the Action Runtime/CAS frontier.
#[derive(Debug)]
pub struct VerifiedXeniaCapability {
    authorization_id: [u8; 16],
    session_id: [u8; 16],
    grant_digest: Digest32,
    workload_digest: Digest32,
    executor_workload: VerifiedExecutorWorkload,
    xenia_ledger_entry_count: u64,
    xenia_ledger_head_hash: [u8; 32],
    prior_checkpoint: CheckpointHead,
    authority_state: VerifiedAuthorityState,
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
    pub fn executor_workload(&self) -> &VerifiedExecutorWorkload {
        &self.executor_workload
    }
    pub fn xenia_frontier(&self) -> (u64, [u8; 32]) {
        (self.xenia_ledger_entry_count, self.xenia_ledger_head_hash)
    }
    pub fn prior_checkpoint(&self) -> CheckpointHead {
        self.prior_checkpoint
    }
    pub fn authority_state(&self) -> &VerifiedAuthorityState {
        &self.authority_state
    }
    pub fn authority_state_digest(&self) -> Digest32 {
        self.authority_state.snapshot_digest()
    }
    pub fn authority_state_sequence(&self) -> u64 {
        self.authority_state.state_sequence()
    }
    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }
}

/// Verify a Xenia-bound Symthaea capability before the Action Runtime may
/// reserve it for consequential execution.
///
/// Both `executor_workload` and `authority_state` are consumed by value. On
/// success the returned proof owns the exact workload/state objects that effect
/// admission must re-check. Xenia verifies cryptographic/provenance consistency;
/// final negative-authority evaluation and durable use accounting remain in the
/// Agency Kernel broker.
#[allow(clippy::too_many_arguments)]
pub fn verify_xenia_capability_v1(
    attestation: &XeniaAgentCapabilityAttestationV1,
    fresh_xenia_checkpoint: &XeniaLedgerCheckpointV1,
    trusted_xenia_ledger_public_key: [u8; 32],
    grant: &CapabilityGrant,
    executor_workload: VerifiedExecutorWorkload,
    expected_session: XeniaSessionExpectationV1,
    current_agent_checkpoint: CheckpointHead,
    authority_time: &VerifiedAuthorityTime,
    authority_state: VerifiedAuthorityState,
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

    executor_workload.ensure_fresh(grant, authority_time)?;
    let expected_executor = grant
        .audience
        .as_ref()
        .ok_or(XeniaAuthorityError::GrantMissingExecutorAudience)?;
    if &executor_workload.workload().executor != expected_executor {
        return Err(XeniaAuthorityError::WorkloadExecutorMismatch);
    }
    let workload_digest = executor_workload.workload_digest()?;

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
        executor_workload,
        xenia_ledger_entry_count: authorization.ledger_entry_count,
        xenia_ledger_head_hash: authorization.ledger_head_hash,
        prior_checkpoint: current_agent_checkpoint,
        authority_state,
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
    if checkpoint.timestamp_unix_secs
        > not_before_unix_s.saturating_add(freshness.max_future_skew_s)
    {
        return Err(XeniaAuthorityError::LedgerCheckpointFromFuture);
    }
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
    #[error("verified executor workload failed: {0}")]
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
    #[error("verified executor does not match capability audience")]
    WorkloadExecutorMismatch,
    #[error("Xenia authorization does not bind this exact verified workload")]
    WorkloadDigestMismatch,
    #[error("Xenia authorization does not bind the expected session provenance")]
    SessionBindingMismatch,
    #[error("Xenia authorization does not bind the exact current Agency Kernel checkpoint")]
    AgentCheckpointMismatch,
}

pub fn principal(value: impl Into<String>) -> PrincipalId {
    PrincipalId(value.into())
}
