// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed service boundary for trusted co-creation operations.
//!
//! This adapter deliberately wraps the existing creative_ecology session
//! ledger rather than introducing a second contribution or consent registry.
//! It adds the service checks missing from that ledger: authenticated actor,
//! trusted operation-scoped consent receipt, and exact-byte artifact reads.
//!
//! No identity provider, signing key, or consent authority is invented here.
//! Production services must inject trusted implementations. The provided
//! unavailable implementations reject all operations.
//!
//! The returned verified operation is an in-process evidence object, not a
//! durable signed journal. Studio must commit it atomically with the updated
//! session snapshot in its append-only/checkpointed transaction store.

use crate::creative_ecology::{
    CoCreativeActionV1, CoCreationEventV1, CoCreationSessionV1, ConsentPurposeV1,
    CreativeEcologyError,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

/// Opaque result of a configured actor-authentication provider.
///
/// A conforming implementation must prove that the credential resolves to
/// the exact expected participant identifier. Returning success for an
/// unbound credential violates this provider contract.
pub trait ActorAuthenticatorV1: Send + Sync {
    /// Stable identifier for the configured identity provider/verifier.
    fn verifier_id(&self) -> &str;

    /// Authenticate the credential as the expected registered participant.
    fn authenticate_actor(
        &self,
        credential_bytes: &[u8],
        expected_participant_id: &str,
    ) -> Result<(), TrustedBoundaryFailureV1>;
}

/// Verifies a signed or otherwise trusted consent receipt against the exact
/// proposed operation. The implementation must validate issuer/key trust,
/// signature, subject, grant, session, purpose, operation digest, artifact
/// scope, validity window, and current revocation state.
pub trait ConsentReceiptVerifierV1: Send + Sync {
    /// Stable identifier for the configured consent authority/verifier.
    fn verifier_id(&self) -> &str;

    /// Reject a receipt unless every expected claim is authorized.
    fn verify_receipt(
        &self,
        receipt_bytes: &[u8],
        expected: &ConsentReceiptExpectationV1,
    ) -> Result<(), TrustedBoundaryFailureV1>;
}

/// Reads immutable artifact bytes using their claimed SHA-256 content address.
/// A content-addressed implementation must not silently substitute a
/// different object for the requested digest.
pub trait ArtifactReaderV1: Send + Sync {
    /// Return the exact stored bytes, or None when the digest is absent.
    fn read_exact_by_sha256(
        &self,
        sha256_hex: &str,
    ) -> Result<Option<Vec<u8>>, TrustedBoundaryFailureV1>;
}

/// Exact scope that the trusted consent verifier must match to the signed
/// receipt. Artifact digests are sorted and unique.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentReceiptExpectationV1 {
    /// Contract version.
    pub schema_version: u16,
    /// Exact co-creation session.
    pub session_id: String,
    /// Authenticated participant/subject.
    pub participant_id: String,
    /// Existing session grant identifier.
    pub grant_id: String,
    /// Action-derived consent purpose.
    pub purpose: ConsentPurposeV1,
    /// Sequence the ledger will assign if the append succeeds.
    pub sequence: u64,
    /// Domain-separated digest of the exact operation envelope.
    pub operation_sha256: String,
    /// Exact source/result artifact digests consumed or produced.
    pub artifact_sha256s: Vec<String>,
    /// Purposes in the active ledger grant at verification time.
    pub granted_purposes: Vec<ConsentPurposeV1>,
    /// Grant sequence expiry, if one was declared.
    pub valid_through_sequence: Option<u64>,
}

/// SHA-256 evidence for one exact artifact read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactBytesV1 {
    /// Content address requested from the artifact store.
    pub sha256_hex: String,
    /// Number of exact bytes independently hashed.
    pub byte_len: u64,
}

/// In-process verified operation returned only after actor, artifact and
/// consent checks all succeed and the session ledger append succeeds.
///
/// Fields are private and there is no Deserialize implementation. A caller
/// cannot construct a verified result from untrusted serialized data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCoCreationOperationV1 {
    event: CoCreationEventV1,
    actor_verifier_id: String,
    consent_verifier_id: String,
    consent_receipt_sha256: String,
    consent_receipt_byte_len: u64,
    operation_sha256: String,
    artifact_evidence: Vec<VerifiedArtifactBytesV1>,
}

impl VerifiedCoCreationOperationV1 {
    /// The exact event committed to the in-memory co-creation session.
    pub fn event(&self) -> &CoCreationEventV1 {
        &self.event
    }

    /// Identity provider that authenticated the actor.
    pub fn actor_verifier_id(&self) -> &str {
        &self.actor_verifier_id
    }

    /// Trusted consent-receipt verifier that approved the operation.
    pub fn consent_verifier_id(&self) -> &str {
        &self.consent_verifier_id
    }

    /// Digest of the exact consent-receipt bytes.
    pub fn consent_receipt_sha256(&self) -> &str {
        &self.consent_receipt_sha256
    }

    /// Length of the exact consent-receipt bytes.
    pub fn consent_receipt_byte_len(&self) -> u64 {
        self.consent_receipt_byte_len
    }

    /// Domain-separated digest of the operation and its artifact scope.
    pub fn operation_sha256(&self) -> &str {
        &self.operation_sha256
    }

    /// Exact-byte evidence for every source/result artifact verified.
    pub fn artifact_evidence(&self) -> &[VerifiedArtifactBytesV1] {
        &self.artifact_evidence
    }
}

/// Error returned by a configured trust provider or artifact source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustedBoundaryFailureV1 {
    /// A configured provider is absent, disabled or not provisioned.
    VerifierUnavailable,
    /// Actor credential did not authenticate the requested participant.
    ActorRejected,
    /// Consent receipt signature, scope, issuer, expiry or revocation failed.
    ConsentRejected,
    /// Content-addressed artifact was not found.
    ArtifactNotFound,
    /// Bytes returned by the artifact store do not match the requested digest.
    ArtifactDigestMismatch,
    /// Receipt bytes were empty or malformed before provider verification.
    InvalidReceipt,
    /// Provider identity is missing or invalid.
    InvalidVerifierIdentity,
    /// An artifact store returned an empty artifact payload.
    EmptyArtifact,
}

impl fmt::Display for TrustedBoundaryFailureV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TrustedBoundaryFailureV1 {}

/// Adapter-level errors. Ledger validation errors remain distinguishable from
/// authentication, consent, and artifact-verification failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedCoCreationErrorV1 {
    /// Existing session/grant/lineage invariant failed.
    Ledger(CreativeEcologyError),
    /// A trusted provider or exact artifact read failed.
    Boundary(TrustedBoundaryFailureV1),
    /// Operation could not be serialized under this versioned contract.
    OperationSerialization,
    /// Sequence arithmetic cannot be represented.
    SequenceExhausted,
}

impl fmt::Display for TrustedCoCreationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ledger(error) => write!(f, "co-creation ledger rejected operation: {error}"),
            Self::Boundary(error) => {
                write!(f, "trusted co-creation boundary rejected operation: {error}")
            }
            Self::OperationSerialization => {
                write!(f, "operation envelope could not be serialized")
            }
            Self::SequenceExhausted => write!(f, "co-creation event sequence exhausted"),
        }
    }
}

impl std::error::Error for TrustedCoCreationErrorV1 {}

impl From<CreativeEcologyError> for TrustedCoCreationErrorV1 {
    fn from(value: CreativeEcologyError) -> Self {
        Self::Ledger(value)
    }
}

/// Fail-closed actor provider. Replace with the service's authenticated
/// identity-provider adapter; this type never accepts credentials.
pub struct UnavailableActorAuthenticatorV1;

impl ActorAuthenticatorV1 for UnavailableActorAuthenticatorV1 {
    fn verifier_id(&self) -> &str {
        "unavailable"
    }

    fn authenticate_actor(
        &self,
        _credential_bytes: &[u8],
        _expected_participant_id: &str,
    ) -> Result<(), TrustedBoundaryFailureV1> {
        Err(TrustedBoundaryFailureV1::VerifierUnavailable)
    }
}

/// Fail-closed consent provider. Replace with a trusted signature/key and
/// consent-state verifier; this type never accepts receipts.
pub struct UnavailableConsentReceiptVerifierV1;

impl ConsentReceiptVerifierV1 for UnavailableConsentReceiptVerifierV1 {
    fn verifier_id(&self) -> &str {
        "unavailable"
    }

    fn verify_receipt(
        &self,
        _receipt_bytes: &[u8],
        _expected: &ConsentReceiptExpectationV1,
    ) -> Result<(), TrustedBoundaryFailureV1> {
        Err(TrustedBoundaryFailureV1::VerifierUnavailable)
    }
}

/// Fail-closed artifact reader. Replace with the Studio-owned immutable
/// content-addressed store adapter.
pub struct UnavailableArtifactReaderV1;

impl ArtifactReaderV1 for UnavailableArtifactReaderV1 {
    fn read_exact_by_sha256(
        &self,
        _sha256_hex: &str,
    ) -> Result<Option<Vec<u8>>, TrustedBoundaryFailureV1> {
        Err(TrustedBoundaryFailureV1::VerifierUnavailable)
    }
}

/// Authenticated service adapter over the existing co-creation ledger.
///
/// All external checks happen before CoCreationSessionV1::append_action.
/// The ledger remains responsible for sequence, grant-owner, purpose and
/// event-lineage invariants. The host Studio transaction must atomically
/// persist the verified operation and updated session snapshot; this
/// in-memory adapter is not itself a durable journal.
pub struct TrustedCoCreationAdapterV1<'a> {
    session: &'a mut CoCreationSessionV1,
    actor_authenticator: &'a dyn ActorAuthenticatorV1,
    consent_verifier: &'a dyn ConsentReceiptVerifierV1,
    artifact_reader: &'a dyn ArtifactReaderV1,
}

impl<'a> TrustedCoCreationAdapterV1<'a> {
    /// Construct an adapter with explicit trusted providers.
    pub fn new(
        session: &'a mut CoCreationSessionV1,
        actor_authenticator: &'a dyn ActorAuthenticatorV1,
        consent_verifier: &'a dyn ConsentReceiptVerifierV1,
        artifact_reader: &'a dyn ArtifactReaderV1,
    ) -> Self {
        Self {
            session,
            actor_authenticator,
            consent_verifier,
            artifact_reader,
        }
    }

    /// Authenticate the actor, re-read and hash exact referenced artifact
    /// bytes, verify a purpose- and operation-scoped receipt, and only then
    /// append the action to the existing ledger.
    pub fn append_action(
        &mut self,
        credential_bytes: &[u8],
        participant_id: &str,
        grant_id: &str,
        consent_receipt_bytes: &[u8],
        action: CoCreativeActionV1,
    ) -> Result<VerifiedCoCreationOperationV1, TrustedCoCreationErrorV1> {
        self.session.validate_snapshot()?;
        validate_verifier_id(self.actor_authenticator.verifier_id())?;
        validate_verifier_id(self.consent_verifier.verifier_id())?;
        if consent_receipt_bytes.is_empty() {
            return Err(TrustedBoundaryFailureV1::InvalidReceipt.into());
        }

        let sequence = u64::try_from(self.session.events().len())
            .ok()
            .and_then(|length| length.checked_add(1))
            .ok_or(TrustedCoCreationErrorV1::SequenceExhausted)?;
        let purpose = action_purpose(&action);

        let grant = self
            .session
            .grants()
            .iter()
            .find(|grant| grant.grant_id == grant_id)
            .ok_or(CreativeEcologyError::UnknownConsentGrant)?;
        if grant.participant_id != participant_id {
            return Err(CreativeEcologyError::ConsentOwnerMismatch.into());
        }
        if !grant.purposes.contains(&purpose) {
            return Err(CreativeEcologyError::ConsentPurposeNotGranted(purpose).into());
        }
        if sequence < grant.granted_at_sequence
            || grant
                .valid_through_sequence
                .is_some_and(|end| sequence > end)
            || self.session.revocations().iter().any(|revocation| {
                revocation.grant_id == grant_id && sequence >= revocation.effective_sequence
            })
        {
            return Err(CreativeEcologyError::ConsentNotActive.into());
        }

        // Authenticate before artifact reads; an unauthenticated caller must
        // not use this adapter as an artifact-existence oracle.
        self.actor_authenticator
            .authenticate_actor(credential_bytes, participant_id)
            .map_err(TrustedCoCreationErrorV1::Boundary)?;

        let artifact_sha256s = referenced_artifact_digests(self.session, &action)?;
        let mut artifact_evidence = Vec::with_capacity(artifact_sha256s.len());
        for digest in &artifact_sha256s {
            let bytes = self
                .artifact_reader
                .read_exact_by_sha256(digest)
                .map_err(TrustedCoCreationErrorV1::Boundary)?
                .ok_or(TrustedBoundaryFailureV1::ArtifactNotFound)?;
            if bytes.is_empty() {
                return Err(TrustedBoundaryFailureV1::EmptyArtifact.into());
            }
            let actual = sha256_hex(&bytes);
            if &actual != digest {
                return Err(TrustedBoundaryFailureV1::ArtifactDigestMismatch.into());
            }
            artifact_evidence.push(VerifiedArtifactBytesV1 {
                sha256_hex: actual,
                byte_len: bytes.len() as u64,
            });
        }

        let operation_sha256 = operation_digest(
            &self.session.session_id,
            sequence,
            participant_id,
            grant_id,
            purpose,
            &action,
            &artifact_sha256s,
        )?;
        let expectation = ConsentReceiptExpectationV1 {
            schema_version: 1,
            session_id: self.session.session_id.clone(),
            participant_id: participant_id.to_owned(),
            grant_id: grant_id.to_owned(),
            purpose,
            sequence,
            operation_sha256: operation_sha256.clone(),
            artifact_sha256s,
            granted_purposes: grant.purposes.clone(),
            valid_through_sequence: grant.valid_through_sequence,
        };
        self.consent_verifier
            .verify_receipt(consent_receipt_bytes, &expectation)
            .map_err(TrustedCoCreationErrorV1::Boundary)?;

        // This is the first mutation. append_action repeats the ledger's
        // sequence, grant, purpose and lineage checks before committing.
        let event = self
            .session
            .append_action(participant_id, grant_id, action)?;
        Ok(VerifiedCoCreationOperationV1 {
            event,
            actor_verifier_id: self.actor_authenticator.verifier_id().to_owned(),
            consent_verifier_id: self.consent_verifier.verifier_id().to_owned(),
            consent_receipt_sha256: sha256_hex(consent_receipt_bytes),
            consent_receipt_byte_len: consent_receipt_bytes.len() as u64,
            operation_sha256,
            artifact_evidence,
        })
    }
}

fn validate_verifier_id(id: &str) -> Result<(), TrustedCoCreationErrorV1> {
    if id.trim().is_empty() || id == "unavailable" {
        return Err(TrustedBoundaryFailureV1::InvalidVerifierIdentity.into());
    }
    Ok(())
}

fn action_purpose(action: &CoCreativeActionV1) -> ConsentPurposeV1 {
    match action {
        CoCreativeActionV1::Propose { .. } => ConsentPurposeV1::Contribute,
        CoCreativeActionV1::Accept { .. }
        | CoCreativeActionV1::Reject { .. }
        | CoCreativeActionV1::Abstain { .. } => ConsentPurposeV1::Review,
        CoCreativeActionV1::Adapt { .. } | CoCreativeActionV1::Branch { .. } => {
            ConsentPurposeV1::Transform
        }
    }
}

fn add_digest(
    digests: &mut BTreeSet<String>,
    digest: &str,
) -> Result<(), TrustedCoCreationErrorV1> {
    if !is_sha256_hex(digest) {
        return Err(CreativeEcologyError::InvalidDigest("artifact_sha256").into());
    }
    digests.insert(digest.to_ascii_lowercase());
    Ok(())
}

fn add_event_artifact(
    session: &CoCreationSessionV1,
    digests: &mut BTreeSet<String>,
    event_id: &str,
) -> Result<(), TrustedCoCreationErrorV1> {
    let event = session
        .events()
        .iter()
        .find(|event| event.event_id == event_id)
        .ok_or_else(|| CreativeEcologyError::UnknownLineageEvent(event_id.to_owned()))?;
    let digest = match &event.action {
        CoCreativeActionV1::Propose {
            result_artifact_sha256,
            ..
        }
        | CoCreativeActionV1::Adapt {
            result_artifact_sha256,
            ..
        }
        | CoCreativeActionV1::Branch {
            result_artifact_sha256,
            ..
        } => result_artifact_sha256,
        CoCreativeActionV1::Accept { .. }
        | CoCreativeActionV1::Reject { .. }
        | CoCreativeActionV1::Abstain { .. } => {
            return Err(CreativeEcologyError::TargetHasNoArtifact.into());
        }
    };
    add_digest(digests, digest)
}

fn referenced_artifact_digests(
    session: &CoCreationSessionV1,
    action: &CoCreativeActionV1,
) -> Result<Vec<String>, TrustedCoCreationErrorV1> {
    let mut digests = BTreeSet::new();
    match action {
        CoCreativeActionV1::Propose {
            parent_artifact_sha256,
            result_artifact_sha256,
            influence_event_ids,
        } => {
            if let Some(parent) = parent_artifact_sha256 {
                add_digest(&mut digests, parent)?;
            }
            add_digest(&mut digests, result_artifact_sha256)?;
            for event_id in influence_event_ids {
                add_event_artifact(session, &mut digests, event_id)?;
            }
        }
        CoCreativeActionV1::Adapt {
            proposal_event_id,
            result_artifact_sha256,
            influence_event_ids,
        }
        | CoCreativeActionV1::Branch {
            proposal_event_id,
            result_artifact_sha256,
            influence_event_ids,
        } => {
            add_event_artifact(session, &mut digests, proposal_event_id)?;
            add_digest(&mut digests, result_artifact_sha256)?;
            for event_id in influence_event_ids {
                add_event_artifact(session, &mut digests, event_id)?;
            }
        }
        CoCreativeActionV1::Accept { proposal_event_id }
        | CoCreativeActionV1::Reject {
            proposal_event_id, ..
        }
        | CoCreativeActionV1::Abstain {
            proposal_event_id, ..
        } => add_event_artifact(session, &mut digests, proposal_event_id)?,
    }
    Ok(digests.into_iter().collect())
}

#[derive(Serialize)]
struct OperationEnvelopeV1<'a> {
    schema_version: u16,
    session_id: &'a str,
    sequence: u64,
    participant_id: &'a str,
    grant_id: &'a str,
    purpose: ConsentPurposeV1,
    action: &'a CoCreativeActionV1,
    artifact_sha256s: &'a [String],
}

fn operation_digest(
    session_id: &str,
    sequence: u64,
    participant_id: &str,
    grant_id: &str,
    purpose: ConsentPurposeV1,
    action: &CoCreativeActionV1,
    artifact_sha256s: &[String],
) -> Result<String, TrustedCoCreationErrorV1> {
    let envelope = OperationEnvelopeV1 {
        schema_version: 1,
        session_id,
        sequence,
        participant_id,
        grant_id,
        purpose,
        action,
        artifact_sha256s,
    };
    let serialized = serde_json::to_vec(&envelope)
        .map_err(|_| TrustedCoCreationErrorV1::OperationSerialization)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-muse/trusted-co-creation-operation/v1\0");
    hasher.update(serialized);
    Ok(encode_hex(&hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Shared test support for host-provided implementations.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::creative_ecology::{CoCreationParticipantV1, ContributorKindV1};
    use std::collections::BTreeMap;

    struct AllowActor;

    impl ActorAuthenticatorV1 for AllowActor {
        fn verifier_id(&self) -> &str {
            "test-identity-provider"
        }

        fn authenticate_actor(
            &self,
            credential_bytes: &[u8],
            expected_participant_id: &str,
        ) -> Result<(), TrustedBoundaryFailureV1> {
            if credential_bytes == b"valid-credential" && expected_participant_id == "artist" {
                Ok(())
            } else {
                Err(TrustedBoundaryFailureV1::ActorRejected)
            }
        }
    }

    struct AllowConsent;

    impl ConsentReceiptVerifierV1 for AllowConsent {
        fn verifier_id(&self) -> &str {
            "test-consent-verifier"
        }

        fn verify_receipt(
            &self,
            receipt_bytes: &[u8],
            expected: &ConsentReceiptExpectationV1,
        ) -> Result<(), TrustedBoundaryFailureV1> {
            if receipt_bytes == b"trusted-receipt"
                && expected.session_id == "studio-session"
                && expected.participant_id == "artist"
                && expected.purpose == ConsentPurposeV1::Contribute
                && expected.sequence == 1
                && expected.artifact_sha256s.len() == 1
            {
                Ok(())
            } else {
                Err(TrustedBoundaryFailureV1::ConsentRejected)
            }
        }
    }

    struct MemoryArtifacts {
        entries: BTreeMap<String, Vec<u8>>,
    }

    impl MemoryArtifacts {
        fn with_bytes(bytes: &[u8]) -> (Self, String) {
            let digest = sha256_hex(bytes);
            let mut entries = BTreeMap::new();
            entries.insert(digest.clone(), bytes.to_vec());
            (Self { entries }, digest)
        }
    }

    impl ArtifactReaderV1 for MemoryArtifacts {
        fn read_exact_by_sha256(
            &self,
            sha256_hex: &str,
        ) -> Result<Option<Vec<u8>>, TrustedBoundaryFailureV1> {
            Ok(self.entries.get(sha256_hex).cloned())
        }
    }

    struct CorruptArtifacts;

    impl ArtifactReaderV1 for CorruptArtifacts {
        fn read_exact_by_sha256(
            &self,
            _sha256_hex: &str,
        ) -> Result<Option<Vec<u8>>, TrustedBoundaryFailureV1> {
            Ok(Some(b"not-the-requested-bytes".to_vec()))
        }
    }

    struct RejectConsent;

    impl ConsentReceiptVerifierV1 for RejectConsent {
        fn verifier_id(&self) -> &str {
            "reject-consent-verifier"
        }

        fn verify_receipt(
            &self,
            _receipt_bytes: &[u8],
            _expected: &ConsentReceiptExpectationV1,
        ) -> Result<(), TrustedBoundaryFailureV1> {
            Err(TrustedBoundaryFailureV1::ConsentRejected)
        }
    }

    fn session_with_grant(purposes: Vec<ConsentPurposeV1>) -> CoCreationSessionV1 {
        let mut session = CoCreationSessionV1::new("studio-session").unwrap();
        session
            .register_participant(CoCreationParticipantV1 {
                participant_id: "artist".into(),
                kind: ContributorKindV1::Human,
            })
            .unwrap();
        session
            .register_participant(CoCreationParticipantV1 {
                participant_id: "melothaea".into(),
                kind: ContributorKindV1::Melothaea,
            })
            .unwrap();
        session
            .grant_consent("artist-grant", "artist", purposes, None)
            .unwrap();
        session
    }

    fn adapter<'a>(
        session: &'a mut CoCreationSessionV1,
        actor: &'a dyn ActorAuthenticatorV1,
        consent: &'a dyn ConsentReceiptVerifierV1,
        artifacts: &'a dyn ArtifactReaderV1,
    ) -> TrustedCoCreationAdapterV1<'a> {
        TrustedCoCreationAdapterV1::new(session, actor, consent, artifacts)
    }

    #[test]
    fn authenticated_consent_and_exact_bytes_are_required_before_append() {
        let bytes = br#"{"score":"A-minor"}"#;
        let (artifacts, digest) = MemoryArtifacts::with_bytes(bytes);
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = AllowActor;
        let consent = AllowConsent;
        let operation = adapter(&mut session, &actor, &consent, &artifacts)
            .append_action(
                b"valid-credential",
                "artist",
                "artist-grant",
                b"trusted-receipt",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: digest.clone(),
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        assert_eq!(operation.event().event_id, "studio-session:1");
        assert_eq!(operation.event().purpose, ConsentPurposeV1::Contribute);
        assert_eq!(operation.artifact_evidence()[0].sha256_hex, digest);
        assert_eq!(operation.artifact_evidence()[0].byte_len, bytes.len() as u64);
        assert_eq!(session.events().len(), 1);
    }

    #[test]
    fn unavailable_actor_provider_fails_closed_before_append() {
        let (artifacts, digest) = MemoryArtifacts::with_bytes(b"score");
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = UnavailableActorAuthenticatorV1;
        let consent = AllowConsent;
        let result = adapter(&mut session, &actor, &consent, &artifacts).append_action(
            b"valid-credential",
            "artist",
            "artist-grant",
            b"trusted-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: digest,
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Boundary(
                TrustedBoundaryFailureV1::InvalidVerifierIdentity
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn unavailable_consent_provider_fails_closed_before_append() {
        let (artifacts, digest) = MemoryArtifacts::with_bytes(b"score");
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = AllowActor;
        let consent = UnavailableConsentReceiptVerifierV1;
        let result = adapter(&mut session, &actor, &consent, &artifacts).append_action(
            b"valid-credential",
            "artist",
            "artist-grant",
            b"trusted-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: digest,
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Boundary(
                TrustedBoundaryFailureV1::InvalidVerifierIdentity
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn actor_authentication_failure_does_not_read_or_append_artifact() {
        let artifacts = CorruptArtifacts;
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = AllowActor;
        let consent = AllowConsent;
        let result = adapter(&mut session, &actor, &consent, &artifacts).append_action(
            b"wrong-credential",
            "artist",
            "artist-grant",
            b"trusted-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: "a".repeat(64),
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Boundary(
                TrustedBoundaryFailureV1::ActorRejected
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn artifact_store_bytes_are_rehashed_before_consent_verification() {
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = AllowActor;
        let consent = AllowConsent;
        let result = adapter(&mut session, &actor, &consent, &CorruptArtifacts).append_action(
            b"valid-credential",
            "artist",
            "artist-grant",
            b"trusted-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: "a".repeat(64),
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Boundary(
                TrustedBoundaryFailureV1::ArtifactDigestMismatch
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn grant_without_action_purpose_is_rejected_before_verifier_calls() {
        let mut session = session_with_grant(vec![ConsentPurposeV1::Review]);
        let actor = AllowActor;
        let consent = AllowConsent;
        let artifacts = CorruptArtifacts;
        let result = adapter(&mut session, &actor, &consent, &artifacts).append_action(
            b"valid-credential",
            "artist",
            "artist-grant",
            b"trusted-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: "a".repeat(64),
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Ledger(
                CreativeEcologyError::ConsentPurposeNotGranted(ConsentPurposeV1::Contribute)
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn rejected_receipt_does_not_append_event() {
        let (artifacts, digest) = MemoryArtifacts::with_bytes(b"score");
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let result = adapter(&mut session, &AllowActor, &RejectConsent, &artifacts).append_action(
            b"valid-credential",
            "artist",
            "artist-grant",
            b"forged-or-wrong-scope-receipt",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: digest,
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(TrustedCoCreationErrorV1::Boundary(
                TrustedBoundaryFailureV1::ConsentRejected
            ))
        );
        assert!(session.events().is_empty());
    }

    #[test]
    fn consent_verifier_receives_exact_operation_and_artifact_scope() {
        struct RecordingConsent {
            expected_sha: String,
        }
        impl ConsentReceiptVerifierV1 for RecordingConsent {
            fn verifier_id(&self) -> &str {
                "recording-consent-verifier"
            }
            fn verify_receipt(
                &self,
                receipt_bytes: &[u8],
                expected: &ConsentReceiptExpectationV1,
            ) -> Result<(), TrustedBoundaryFailureV1> {
                if receipt_bytes == b"signed-receipt"
                    && expected.artifact_sha256s == vec![self.expected_sha.clone()]
                    && expected.operation_sha256.len() == 64
                    && expected.granted_purposes == vec![ConsentPurposeV1::Contribute]
                {
                    Ok(())
                } else {
                    Err(TrustedBoundaryFailureV1::ConsentRejected)
                }
            }
        }

        let (artifacts, digest) = MemoryArtifacts::with_bytes(b"score-v2");
        let mut session = session_with_grant(vec![ConsentPurposeV1::Contribute]);
        let actor = AllowActor;
        let consent = RecordingConsent {
            expected_sha: digest.clone(),
        };
        let operation = adapter(&mut session, &actor, &consent, &artifacts)
            .append_action(
                b"valid-credential",
                "artist",
                "artist-grant",
                b"signed-receipt",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: digest,
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        assert_eq!(operation.consent_verifier_id(), "recording-consent-verifier");
        assert_eq!(operation.consent_receipt_sha256().len(), 64);
        assert_eq!(operation.operation_sha256().len(), 64);
    }
}
