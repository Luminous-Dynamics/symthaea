// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! RMI trial artifact verification at the Studio trust boundary.
//!
//! This module rehashes exact artifact bytes against event-bound SHA-256
//! descriptors and requires an injected receipt verifier before issuing a
//! fully verified Studio evidence result. The default verifier is unavailable.
//!
//! A successful result means that supplied bytes match the committed
//! manifest and that the configured verifier accepted the receipt, actor,
//! consent scope, and signature. It does not prove authorship, music quality,
//! human preference, or physical playback.

use crate::evidence_digest::{canonical_json_sha256, decode_hex_32, sha256_hex};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_muse_protocol::PieceProvenanceBundle;

/// Version of the RMI event-to-artifact evidence manifest.
pub const RMI_ARTIFACT_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// Direction of an event relative to one RMI trial.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiArtifactSideV1 {
    /// Partner observation consumed by the agent.
    Input,
    /// Agent response or other committed output.
    Output,
}

/// Descriptive artifact class. Content digests, not these labels, define identity.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiArtifactKindV1 {
    /// Serialized composition recipe or intent.
    Recipe,
    /// Exact symbolic score content.
    SymbolicScore,
    /// Performed MIDI bytes.
    PerformedMidi,
    /// Rendered audio bytes.
    RenderedAudio,
    /// Composition/listen bundle bytes.
    CompositionBundle,
    /// Performance bundle bytes.
    PerformanceBundle,
    /// Another explicitly named artifact type.
    Other,
}

/// One content-addressed artifact attached to one exact RMI event.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiArtifactBindingV1 {
    /// Exact event identifier from the RMI event ledger.
    pub event_id: String,
    /// Whether the event is a trial input or output.
    pub side: RmiArtifactSideV1,
    /// Descriptive content class.
    pub kind: RmiArtifactKindV1,
    /// Stable slot within one event and artifact kind (for stems/voices).
    pub slot: String,
    /// Content identity in the form sha256:<lowercase-hex>.
    pub artifact_id: String,
    /// MIME type of the exact bytes.
    pub media_type: String,
    /// Exact byte count.
    pub byte_len: u64,
    /// Lowercase SHA-256 of the exact bytes.
    pub sha256: String,
}

/// Replayable RMI manifest binding event IDs to exact artifacts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiTrialArtifactManifestV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Session identity (pseudonymous in published research artifacts).
    pub session_id: String,
    /// Trial/turn identity.
    pub turn_id: String,
    /// Task identity.
    pub task_id: String,
    /// Candidate policy identity.
    pub policy_id: String,
    /// Candidate policy/configuration version.
    pub policy_version: String,
    /// Exact 40-character lowercase Git revision.
    pub code_revision: String,
    /// Input event identities in ledger order.
    pub input_event_ids: Vec<String>,
    /// Output event identities in ledger order.
    pub output_event_ids: Vec<String>,
    /// Event-to-artifact bindings. Every event must have at least one.
    pub artifacts: Vec<RmiArtifactBindingV1>,
}

impl RmiTrialArtifactManifestV1 {
    /// Validate event coverage, content-address identity, and schema invariants.
    pub fn validate(&self) -> Result<(), RmiArtifactEvidenceError> {
        let valid_revision = self.code_revision.len() == 40
            && self
                .code_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if self.schema_version != RMI_ARTIFACT_EVIDENCE_SCHEMA_VERSION
            || self.session_id.trim().is_empty()
            || self.turn_id.trim().is_empty()
            || self.task_id.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.policy_version.trim().is_empty()
            || !valid_revision
            || self.input_event_ids.is_empty()
            || self.output_event_ids.is_empty()
            || self.artifacts.is_empty()
        {
            return Err(RmiArtifactEvidenceError::InvalidManifest);
        }

        let input_ids = unique_nonempty_ids(&self.input_event_ids)?;
        let output_ids = unique_nonempty_ids(&self.output_event_ids)?;
        if input_ids.iter().any(|event_id| output_ids.contains(event_id)) {
            return Err(RmiArtifactEvidenceError::DuplicateEventId);
        }

        let mut covered_inputs = BTreeSet::new();
        let mut covered_outputs = BTreeSet::new();
        let mut slots = BTreeSet::new();

        for artifact in &self.artifacts {
            if artifact.event_id.trim().is_empty()
                || artifact.slot.trim().is_empty()
                || artifact.media_type.trim().is_empty()
                || artifact.byte_len == 0
            {
                return Err(RmiArtifactEvidenceError::InvalidArtifactBinding);
            }
            validate_sha256(&artifact.sha256)?;
            if artifact.artifact_id != format!("sha256:{}", artifact.sha256) {
                return Err(RmiArtifactEvidenceError::ArtifactIdentityMismatch);
            }

            let is_known_event = match artifact.side {
                RmiArtifactSideV1::Input => {
                    covered_inputs.insert(artifact.event_id.as_str());
                    input_ids.contains(artifact.event_id.as_str())
                }
                RmiArtifactSideV1::Output => {
                    covered_outputs.insert(artifact.event_id.as_str());
                    output_ids.contains(artifact.event_id.as_str())
                }
            };
            if !is_known_event {
                return Err(RmiArtifactEvidenceError::UnexpectedEventBinding);
            }

            if !slots.insert((
                artifact.event_id.as_str(),
                artifact.side,
                artifact.kind,
                artifact.slot.as_str(),
            )) {
                return Err(RmiArtifactEvidenceError::DuplicateArtifactSlot);
            }
        }

        if input_ids
            .iter()
            .any(|event_id| !covered_inputs.contains(event_id))
            || output_ids
                .iter()
                .any(|event_id| !covered_outputs.contains(event_id))
        {
            return Err(RmiArtifactEvidenceError::MissingEventBinding);
        }

        Ok(())
    }

    /// Hash the canonical JSON representation of this manifest.
    pub fn sha256(&self) -> Result<String, RmiArtifactEvidenceError> {
        self.validate()?;
        canonical_json_sha256(self).map_err(|_| RmiArtifactEvidenceError::CanonicalEncoding)
    }

    /// Return sorted, unique artifact IDs committed by this manifest.
    pub fn artifact_ids(&self) -> Vec<String> {
        self.artifacts
            .iter()
            .map(|artifact| artifact.artifact_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Exact bytes resolved from Studio-owned storage for one content identity.
///
/// This verifier does not open arbitrary paths or treat a storage locator as
/// proof. The integration adapter must resolve the artifact from the
/// authoritative store and pass those exact bytes here.
#[derive(Clone, Copy, Debug)]
pub struct RmiArtifactPayloadV1<'a> {
    /// Content identity used by the manifest.
    pub artifact_id: &'a str,
    /// Exact bytes read from the authoritative artifact store.
    pub bytes: &'a [u8],
}

/// Successful byte-level verification result. It does not certify a signed
/// transaction or a consent decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRmiArtifactBytesV1 {
    /// Canonical digest of the validated manifest.
    pub manifest_sha256: String,
    /// Number of unique content identities rehashed.
    pub unique_artifact_count: usize,
    /// Number of events with verified bindings.
    pub event_count: usize,
    /// Total bytes hashed (unique artifacts only).
    pub total_verified_bytes: u64,
}

/// Recompute every artifact digest from supplied bytes and enforce exact
/// manifest/payload set equality. Missing, substituted, duplicate and
/// unreferenced payloads all fail closed.
pub fn verify_rmi_artifact_bytes<'a>(
    manifest: &RmiTrialArtifactManifestV1,
    payloads: &[RmiArtifactPayloadV1<'a>],
) -> Result<VerifiedRmiArtifactBytesV1, RmiArtifactEvidenceError> {
    manifest.validate()?;
    let expected_ids: BTreeSet<&str> = manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.artifact_id.as_str())
        .collect();

    let mut supplied = BTreeMap::new();
    for payload in payloads {
        if supplied.insert(payload.artifact_id, payload.bytes).is_some() {
            return Err(RmiArtifactEvidenceError::DuplicatePayload);
        }
    }

    if supplied.keys().any(|artifact_id| !expected_ids.contains(*artifact_id)) {
        return Err(RmiArtifactEvidenceError::UnexpectedPayload);
    }
    for artifact_id in &expected_ids {
        if !supplied.contains_key(artifact_id) {
            return Err(RmiArtifactEvidenceError::MissingPayload);
        }
    }

    for artifact in &manifest.artifacts {
        let bytes = supplied
            .get(artifact.artifact_id.as_str())
            .ok_or(RmiArtifactEvidenceError::MissingPayload)?;
        let actual_len = u64::try_from(bytes.len())
            .map_err(|_| RmiArtifactEvidenceError::ByteLengthMismatch)?;
        if actual_len != artifact.byte_len {
            return Err(RmiArtifactEvidenceError::ByteLengthMismatch);
        }
        if sha256_hex(bytes) != artifact.sha256 {
            return Err(RmiArtifactEvidenceError::DigestMismatch);
        }
    }

    let mut total_verified_bytes = 0u64;
    for artifact_id in &expected_ids {
        let bytes = supplied
            .get(artifact_id)
            .ok_or(RmiArtifactEvidenceError::MissingPayload)?;
        let len = u64::try_from(bytes.len())
            .map_err(|_| RmiArtifactEvidenceError::ByteLengthMismatch)?;
        total_verified_bytes = total_verified_bytes
            .checked_add(len)
            .ok_or(RmiArtifactEvidenceError::ByteLengthMismatch)?;
    }

    let event_count = manifest.input_event_ids.len() + manifest.output_event_ids.len();
    Ok(VerifiedRmiArtifactBytesV1 {
        manifest_sha256: manifest.sha256()?,
        unique_artifact_count: expected_ids.len(),
        event_count,
        total_verified_bytes,
    })
}

/// Exact byte set corresponding to the three mandatory fields in
/// PieceProvenanceBundle. Recipe and score bytes must use the same serialized
/// forms used when Studio produced its provenance hashes.
#[derive(Clone, Copy, Debug)]
pub struct PieceProvenanceBytesV1<'a> {
    /// Serialized PieceRecipe bytes.
    pub recipe: &'a [u8],
    /// Serialized symbolic Score bytes.
    pub score: &'a [u8],
    /// Exact rendered WAV bytes.
    pub audio: &'a [u8],
    /// Exact performed MIDI bytes, when the source exposes them.
    pub midi: Option<&'a [u8]>,
}

/// Independently rehashed PieceProvenanceBundle artifacts. MIDI remains absent
/// where the source provenance does not commit MIDI bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedPieceProvenanceBytesV1 {
    /// Recipe digest, recomputed from bytes.
    pub recipe_sha256: String,
    /// Symbolic-score digest, recomputed from bytes.
    pub score_sha256: String,
    /// Audio digest, recomputed from bytes.
    pub audio_sha256: String,
    /// MIDI digest when both bytes and a provenance digest exist.
    pub midi_sha256: Option<String>,
    /// Exact byte lengths for verified mandatory artifacts.
    pub recipe_byte_len: u64,
    /// Exact score byte length.
    pub score_byte_len: u64,
    /// Exact audio byte length.
    pub audio_byte_len: u64,
}

/// Recompute mandatory Studio provenance hashes and any supplied MIDI hash.
/// A missing MIDI digest is not inferred from a MIDI URI or boolean.
pub fn verify_piece_provenance_bytes(
    bundle: &PieceProvenanceBundle,
    bytes: PieceProvenanceBytesV1<'_>,
) -> Result<VerifiedPieceProvenanceBytesV1, RmiArtifactEvidenceError> {
    validate_sha256(&bundle.recipe_sha256)?;
    validate_sha256(&bundle.score_sha256)?;
    validate_sha256(&bundle.audio_sha256)?;

    let recipe_sha256 = sha256_hex(bytes.recipe);
    let score_sha256 = sha256_hex(bytes.score);
    let audio_sha256 = sha256_hex(bytes.audio);
    if recipe_sha256 != bundle.recipe_sha256
        || score_sha256 != bundle.score_sha256
        || audio_sha256 != bundle.audio_sha256
    {
        return Err(RmiArtifactEvidenceError::DigestMismatch);
    }

    let midi_sha256 = match bytes.midi {
        Some(midi_bytes) => {
            let expected = bundle
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == "performed-midi")
                .and_then(|artifact| artifact.sha256.as_deref())
                .ok_or(RmiArtifactEvidenceError::ExpectedDigestUnavailable)?;
            validate_sha256(expected)?;
            let actual = sha256_hex(midi_bytes);
            if actual != expected {
                return Err(RmiArtifactEvidenceError::DigestMismatch);
            }
            Some(actual)
        }
        None => None,
    };

    Ok(VerifiedPieceProvenanceBytesV1 {
        recipe_sha256,
        score_sha256,
        audio_sha256,
        midi_sha256,
        recipe_byte_len: u64::try_from(bytes.recipe.len())
            .map_err(|_| RmiArtifactEvidenceError::ByteLengthMismatch)?,
        score_byte_len: u64::try_from(bytes.score.len())
            .map_err(|_| RmiArtifactEvidenceError::ByteLengthMismatch)?,
        audio_byte_len: u64::try_from(bytes.audio.len())
            .map_err(|_| RmiArtifactEvidenceError::ByteLengthMismatch)?,
    })
}

/// Purposes are explicit and non-interchangeable. An evaluation receipt does
/// not authorize memory, training, publication or federation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiArtifactPurposeV1 {
    /// Evaluate the exact session/trial and committed artifacts.
    EvaluateSession,
    /// Store a contributor-specific artifact in personal memory.
    PersonalMemory,
    /// Use artifacts to train or update a model.
    ModelTraining,
    /// Publish artifacts or derived research data.
    Publication,
    /// Federate artifacts or learning state to another network.
    Federation,
}

/// Signed receipt body. This structure alone is only a claim; signature and
/// consent verification are delegated to a configured trust provider.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiStudioReceiptBodyV1 {
    /// Unique Studio transaction identifier.
    pub transaction_id: String,
    /// Authenticated actor identity or approved pseudonym.
    pub actor_id: String,
    /// Session identity.
    pub session_id: String,
    /// Trial/turn identity.
    pub turn_id: String,
    /// Task identity.
    pub task_id: String,
    /// Candidate policy identity.
    pub policy_id: String,
    /// Exact purpose authorized by the receipt.
    pub purpose: RmiArtifactPurposeV1,
    /// Canonical RMI artifact manifest digest.
    pub manifest_sha256: String,
    /// Sorted unique content identities covered by the receipt.
    pub artifact_ids: Vec<String>,
    /// Purpose-specific consent receipt references.
    pub consent_receipt_ids: Vec<String>,
    /// Durable commit sequence or event identity.
    pub commit_sequence: u64,
    /// Idempotency key for crash-safe retry.
    pub idempotency_key: String,
    /// Studio commit time in Unix milliseconds.
    pub committed_at_unix_ms: u64,
}

/// Envelope for a signed Studio transaction claim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiStudioReceiptEnvelopeV1 {
    /// Receipt claims.
    pub body: RmiStudioReceiptBodyV1,
    /// Canonical SHA-256 of body.
    pub body_sha256: String,
    /// Identifier for the trusted signing key.
    pub signer_key_id: String,
    /// Opaque signature bytes; encoding/algorithm is verifier-specific.
    pub signature: Vec<u8>,
}

/// Trust-provider boundary for Studio actor, signature and consent checks.
///
/// An implementation must authenticate the actor, validate the signature
/// against a trusted key, verify each consent receipt for the exact session,
/// action, artifact set and purpose, and check expiry/revocation at the
/// supplied verification time. Merely checking that receipt IDs are present
/// is not sufficient.
pub trait RmiStudioReceiptVerifier: Send + Sync {
    /// Stable verifier implementation identifier.
    fn verifier_id(&self) -> &str;

    /// Verify cryptographic authenticity and current purpose-specific
    /// authorization. Implementations must fail closed on missing trust data.
    fn verify(
        &self,
        envelope: &RmiStudioReceiptEnvelopeV1,
        manifest: &RmiTrialArtifactManifestV1,
        verification_time_unix_ms: u64,
    ) -> Result<(), RmiArtifactEvidenceError>;
}

/// Explicit unavailable verifier. Use this until a real identity/signature
/// and consent authority is connected; it never accepts a claim.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnavailableRmiStudioReceiptVerifier;

impl RmiStudioReceiptVerifier for UnavailableRmiStudioReceiptVerifier {
    fn verifier_id(&self) -> &str {
        "unavailable"
    }

    fn verify(
        &self,
        _envelope: &RmiStudioReceiptEnvelopeV1,
        _manifest: &RmiTrialArtifactManifestV1,
        _verification_time_unix_ms: u64,
    ) -> Result<(), RmiArtifactEvidenceError> {
        Err(RmiArtifactEvidenceError::ReceiptVerifierUnavailable)
    }
}

/// Successful Studio evidence result. Returned only after exact artifact-byte
/// verification and acceptance by the configured receipt verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRmiStudioEvidenceV1 {
    /// Canonical RMI manifest digest.
    pub manifest_sha256: String,
    /// Canonical signed receipt-body digest.
    pub receipt_body_sha256: String,
    /// Stable Studio transaction identifier.
    pub transaction_id: String,
    /// Receipt verifier that accepted signature and consent scope.
    pub verifier_id: String,
    /// Number of unique artifact identities byte-verified.
    pub artifact_count: usize,
    /// Total unique bytes independently hashed.
    pub total_verified_bytes: u64,
}

/// Verify exact artifact bytes, receipt/manifest equality, and the delegated
/// signature + consent decision. No receipt verifier means no verified result.
pub fn verify_rmi_studio_evidence(
    manifest: &RmiTrialArtifactManifestV1,
    payloads: &[RmiArtifactPayloadV1<'_>],
    envelope: &RmiStudioReceiptEnvelopeV1,
    verifier: &dyn RmiStudioReceiptVerifier,
    verification_time_unix_ms: u64,
) -> Result<VerifiedRmiStudioEvidenceV1, RmiArtifactEvidenceError> {
    let byte_evidence = verify_rmi_artifact_bytes(manifest, payloads)?;
    let body = &envelope.body;
    let expected_artifact_ids = manifest.artifact_ids();

    if body.transaction_id.trim().is_empty()
        || body.actor_id.trim().is_empty()
        || body.session_id != manifest.session_id
        || body.turn_id != manifest.turn_id
        || body.task_id != manifest.task_id
        || body.policy_id != manifest.policy_id
        || body.purpose != RmiArtifactPurposeV1::EvaluateSession
        || body.manifest_sha256 != byte_evidence.manifest_sha256
        || body.artifact_ids != expected_artifact_ids
        || body.consent_receipt_ids.is_empty()
        || unique_nonempty_ids(&body.consent_receipt_ids)?.len() != body.consent_receipt_ids.len()
        || body.commit_sequence == 0
        || body.idempotency_key.trim().is_empty()
        || body.committed_at_unix_ms == 0
        || verification_time_unix_ms < body.committed_at_unix_ms
        || envelope.signer_key_id.trim().is_empty()
        || envelope.signature.is_empty()
    {
        return Err(RmiArtifactEvidenceError::ReceiptContextMismatch);
    }

    validate_sha256(&envelope.body_sha256)?;
    let actual_body_sha256 =
        canonical_json_sha256(body).map_err(|_| RmiArtifactEvidenceError::CanonicalEncoding)?;
    if actual_body_sha256 != envelope.body_sha256 {
        return Err(RmiArtifactEvidenceError::ReceiptDigestMismatch);
    }

    let verifier_id = verifier.verifier_id();
    if verifier_id.trim().is_empty() || verifier_id == "unavailable" {
        return Err(RmiArtifactEvidenceError::ReceiptVerifierUnavailable);
    }
    verifier.verify(envelope, manifest, verification_time_unix_ms)?;

    Ok(VerifiedRmiStudioEvidenceV1 {
        manifest_sha256: byte_evidence.manifest_sha256,
        receipt_body_sha256: actual_body_sha256,
        transaction_id: body.transaction_id.clone(),
        verifier_id: verifier_id.to_owned(),
        artifact_count: byte_evidence.unique_artifact_count,
        total_verified_bytes: byte_evidence.total_verified_bytes,
    })
}

fn unique_nonempty_ids(ids: &[String]) -> Result<BTreeSet<&str>, RmiArtifactEvidenceError> {
    let values: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
    if values.len() != ids.len() || ids.iter().any(|value| value.trim().is_empty()) {
        return Err(RmiArtifactEvidenceError::DuplicateEventId);
    }
    Ok(values)
}

fn validate_sha256(value: &str) -> Result<(), RmiArtifactEvidenceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || decode_hex_32(value).is_none()
    {
        return Err(RmiArtifactEvidenceError::InvalidDigest);
    }
    Ok(())
}

/// Fail-closed errors for artifact, manifest and receipt verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmiArtifactEvidenceError {
    /// Unsupported schema or missing required manifest fields.
    InvalidManifest,
    /// Duplicate or empty event identity.
    DuplicateEventId,
    /// Artifact metadata is empty or has zero byte length.
    InvalidArtifactBinding,
    /// Artifact ID is not derived from its declared SHA-256.
    ArtifactIdentityMismatch,
    /// Artifact binding names an event not present on the declared side.
    UnexpectedEventBinding,
    /// One or more manifest events have no artifact binding.
    MissingEventBinding,
    /// Duplicate slot within an event and artifact kind.
    DuplicateArtifactSlot,
    /// Digest is not exactly 64 lowercase hexadecimal characters.
    InvalidDigest,
    /// Payload byte count does not match the manifest.
    ByteLengthMismatch,
    /// Content digest does not match the descriptor.
    DigestMismatch,
    /// The same payload identity was supplied more than once.
    DuplicatePayload,
    /// Payload has no corresponding manifest descriptor.
    UnexpectedPayload,
    /// A manifest artifact has no supplied payload.
    MissingPayload,
    /// Provenance does not commit an optional artifact digest.
    ExpectedDigestUnavailable,
    /// Canonical JSON serialization failed.
    CanonicalEncoding,
    /// Receipt context differs from manifest or lacks required claims.
    ReceiptContextMismatch,
    /// Receipt body digest does not match its canonical body.
    ReceiptDigestMismatch,
    /// No trusted receipt verifier is configured.
    ReceiptVerifierUnavailable,
}

impl fmt::Display for RmiArtifactEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RmiArtifactEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::{ProvenanceArtifact, ReproducibilityClaim};

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn binding(
        event_id: &str,
        side: RmiArtifactSideV1,
        kind: RmiArtifactKindV1,
        slot: &str,
        bytes: &[u8],
    ) -> RmiArtifactBindingV1 {
        let digest = sha256_hex(bytes);
        RmiArtifactBindingV1 {
            event_id: event_id.to_owned(),
            side,
            kind,
            slot: slot.to_owned(),
            artifact_id: format!("sha256:{digest}"),
            media_type: "application/octet-stream".to_owned(),
            byte_len: bytes.len() as u64,
            sha256: digest,
        }
    }

    fn manifest() -> RmiTrialArtifactManifestV1 {
        RmiTrialArtifactManifestV1 {
            schema_version: RMI_ARTIFACT_EVIDENCE_SCHEMA_VERSION,
            session_id: "session-pseudonym-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            task_id: "role-exchange".to_owned(),
            policy_id: "melothaea-candidate".to_owned(),
            policy_version: "candidate-v1".to_owned(),
            code_revision: REVISION.to_owned(),
            input_event_ids: vec!["partner-event-1".to_owned()],
            output_event_ids: vec!["agent-event-1".to_owned()],
            artifacts: vec![
                binding(
                    "partner-event-1",
                    RmiArtifactSideV1::Input,
                    RmiArtifactKindV1::PerformedMidi,
                    "main",
                    b"partner-midi",
                ),
                binding(
                    "agent-event-1",
                    RmiArtifactSideV1::Output,
                    RmiArtifactKindV1::PerformedMidi,
                    "main",
                    b"agent-midi",
                ),
            ],
        }
    }

    fn payloads() -> Vec<RmiArtifactPayloadV1<'static>> {
        vec![
            RmiArtifactPayloadV1 {
                artifact_id: Box::leak(
                    format!("sha256:{}", sha256_hex(b"partner-midi")).into_boxed_str(),
                ),
                bytes: b"partner-midi",
            },
            RmiArtifactPayloadV1 {
                artifact_id: Box::leak(
                    format!("sha256:{}", sha256_hex(b"agent-midi")).into_boxed_str(),
                ),
                bytes: b"agent-midi",
            },
        ]
    }

    struct AcceptTestReceipt;

    impl RmiStudioReceiptVerifier for AcceptTestReceipt {
        fn verifier_id(&self) -> &str {
            "test-only-verifier"
        }

        fn verify(
            &self,
            _envelope: &RmiStudioReceiptEnvelopeV1,
            _manifest: &RmiTrialArtifactManifestV1,
            _verification_time_unix_ms: u64,
        ) -> Result<(), RmiArtifactEvidenceError> {
            Ok(())
        }
    }

    fn receipt(manifest: &RmiTrialArtifactManifestV1) -> RmiStudioReceiptEnvelopeV1 {
        let body = RmiStudioReceiptBodyV1 {
            transaction_id: "studio-tx-1".to_owned(),
            actor_id: "performer-pseudonym-1".to_owned(),
            session_id: manifest.session_id.clone(),
            turn_id: manifest.turn_id.clone(),
            task_id: manifest.task_id.clone(),
            policy_id: manifest.policy_id.clone(),
            purpose: RmiArtifactPurposeV1::EvaluateSession,
            manifest_sha256: manifest.sha256().unwrap(),
            artifact_ids: manifest.artifact_ids(),
            consent_receipt_ids: vec!["consent-evaluation-1".to_owned()],
            commit_sequence: 12,
            idempotency_key: "rmi-trial-1".to_owned(),
            committed_at_unix_ms: 100,
        };
        let body_sha256 = canonical_json_sha256(&body).unwrap();
        RmiStudioReceiptEnvelopeV1 {
            body,
            body_sha256,
            signer_key_id: "test-key".to_owned(),
            signature: vec![0xA5],
        }
    }

    fn provenance_bundle(recipe: &[u8], score: &[u8], audio: &[u8]) -> PieceProvenanceBundle {
        PieceProvenanceBundle {
            recipe_schema_version: 1,
            recipe_sha256: sha256_hex(recipe),
            score_sha256: sha256_hex(score),
            audio_sha256: sha256_hex(audio),
            seed: 1,
            style_name: "fixture".to_owned(),
            renderer_name: "fixture-renderer".to_owned(),
            renderer_version: Some("1".to_owned()),
            muse_engine_version: "test".to_owned(),
            theory_engine_version: "test".to_owned(),
            muse_source_revision: Some(REVISION.to_owned()),
            theory_source_revision: Some(REVISION.to_owned()),
            soundfont_sha256: None,
            renderer_binary_sha256: None,
            performance_model_sha256: None,
            render_environment_sha256: None,
            diversity_plan: None,
            reproduction: ReproducibilityClaim {
                symbolic_score_exact: true,
                midi_exact: false,
                rendered_audio_exact: true,
                limitations: Vec::new(),
            },
            artifacts: vec![ProvenanceArtifact {
                kind: "rendered-audio".to_owned(),
                media_type: "audio/wav".to_owned(),
                uri: "/api/audio/1".to_owned(),
                sha256: Some(sha256_hex(audio)),
            }],
        }
    }

    #[test]
    fn artifact_manifest_rehashes_exact_input_and_output_bytes() {
        let verified = verify_rmi_artifact_bytes(&manifest(), &payloads()).unwrap();
        assert_eq!(verified.unique_artifact_count, 2);
        assert_eq!(verified.event_count, 2);
        assert_eq!(verified.total_verified_bytes, 22);
    }

    #[test]
    fn altered_input_bytes_are_rejected() {
        let manifest = manifest();
        let mut inputs = payloads();
        inputs[0] = RmiArtifactPayloadV1 {
            artifact_id: inputs[0].artifact_id,
            bytes: b"changed-midi",
        };
        assert_eq!(
            verify_rmi_artifact_bytes(&manifest, &inputs),
            Err(RmiArtifactEvidenceError::ByteLengthMismatch)
        );
    }

    #[test]
    fn equal_length_substitution_is_rejected_by_digest() {
        let manifest = manifest();
        let mut inputs = payloads();
        inputs[0] = RmiArtifactPayloadV1 {
            artifact_id: inputs[0].artifact_id,
            bytes: b"partner-midX",
        };
        assert_eq!(
            verify_rmi_artifact_bytes(&manifest, &inputs),
            Err(RmiArtifactEvidenceError::DigestMismatch)
        );
    }

    #[test]
    fn missing_payload_is_rejected() {
        assert_eq!(
            verify_rmi_artifact_bytes(&manifest(), &payloads()[..1]),
            Err(RmiArtifactEvidenceError::MissingPayload)
        );
    }

    #[test]
    fn unreferenced_payload_is_rejected() {
        let mut inputs = payloads();
        inputs.push(RmiArtifactPayloadV1 {
            artifact_id: "sha256:extra",
            bytes: b"extra",
        });
        assert_eq!(
            verify_rmi_artifact_bytes(&manifest(), &inputs),
            Err(RmiArtifactEvidenceError::UnexpectedPayload)
        );
    }

    #[test]
    fn duplicate_payload_identity_is_rejected() {
        let mut inputs = payloads();
        inputs.push(inputs[0]);
        assert_eq!(
            verify_rmi_artifact_bytes(&manifest(), &inputs),
            Err(RmiArtifactEvidenceError::DuplicatePayload)
        );
    }

    #[test]
    fn manifest_requires_exact_event_coverage_on_both_sides() {
        let mut invalid = manifest();
        invalid.output_event_ids.push("missing-output".to_owned());
        assert_eq!(
            invalid.validate(),
            Err(RmiArtifactEvidenceError::MissingEventBinding)
        );
    }

    #[test]
    fn manifest_rejects_binding_on_the_wrong_side() {
        let mut invalid = manifest();
        invalid.artifacts[0].side = RmiArtifactSideV1::Output;
        assert_eq!(
            invalid.validate(),
            Err(RmiArtifactEvidenceError::UnexpectedEventBinding)
        );
    }

    #[test]
    fn manifest_rejects_duplicate_event_identity_across_sides() {
        let mut invalid = manifest();
        invalid.output_event_ids[0] = invalid.input_event_ids[0].clone();
        assert_eq!(
            invalid.validate(),
            Err(RmiArtifactEvidenceError::DuplicateEventId)
        );
    }

    #[test]
    fn manifest_rejects_artifact_id_not_derived_from_digest() {
        let mut invalid = manifest();
        invalid.artifacts[0].artifact_id = "studio://keeper/123".to_owned();
        assert_eq!(
            invalid.validate(),
            Err(RmiArtifactEvidenceError::ArtifactIdentityMismatch)
        );
    }

    #[test]
    fn piece_provenance_rehashes_recipe_score_and_audio_bytes() {
        let recipe = b"serialized-recipe";
        let score = b"serialized-score";
        let audio = b"RIFF-wav-fixture";
        let bundle = provenance_bundle(recipe, score, audio);
        let verified = verify_piece_provenance_bytes(
            &bundle,
            PieceProvenanceBytesV1 {
                recipe,
                score,
                audio,
                midi: None,
            },
        )
        .unwrap();
        assert_eq!(verified.recipe_sha256, bundle.recipe_sha256);
        assert_eq!(verified.score_sha256, bundle.score_sha256);
        assert_eq!(verified.audio_sha256, bundle.audio_sha256);
        assert_eq!(verified.midi_sha256, None);
    }

    #[test]
    fn piece_provenance_rejects_changed_score_bytes() {
        let bundle = provenance_bundle(b"recipe", b"score", b"audio");
        assert_eq!(
            verify_piece_provenance_bytes(
                &bundle,
                PieceProvenanceBytesV1 {
                    recipe: b"recipe",
                    score: b"score-substituted",
                    audio: b"audio",
                    midi: None,
                },
            ),
            Err(RmiArtifactEvidenceError::DigestMismatch)
        );
    }

    #[test]
    fn midi_bytes_without_a_committed_midi_digest_are_not_claimed_as_verified() {
        let bundle = provenance_bundle(b"recipe", b"score", b"audio");
        assert_eq!(
            verify_piece_provenance_bytes(
                &bundle,
                PieceProvenanceBytesV1 {
                    recipe: b"recipe",
                    score: b"score",
                    audio: b"audio",
                    midi: Some(b"midi"),
                },
            ),
            Err(RmiArtifactEvidenceError::ExpectedDigestUnavailable)
        );
    }

    #[test]
    fn signed_receipt_and_consent_verifier_are_required() {
        let manifest = manifest();
        let envelope = receipt(&manifest);
        let verified = verify_rmi_studio_evidence(
            &manifest,
            &payloads(),
            &envelope,
            &AcceptTestReceipt,
            200,
        )
        .unwrap();
        assert_eq!(verified.transaction_id, "studio-tx-1");
        assert_eq!(verified.verifier_id, "test-only-verifier");

        assert_eq!(
            verify_rmi_studio_evidence(
                &manifest,
                &payloads(),
                &envelope,
                &UnavailableRmiStudioReceiptVerifier,
                200,
            ),
            Err(RmiArtifactEvidenceError::ReceiptVerifierUnavailable)
        );
    }

    #[test]
    fn receipt_cannot_authorize_a_different_artifact_set() {
        let manifest = manifest();
        let mut envelope = receipt(&manifest);
        envelope.body.artifact_ids.pop();
        envelope.body_sha256 = canonical_json_sha256(&envelope.body).unwrap();
        assert_eq!(
            verify_rmi_studio_evidence(
                &manifest,
                &payloads(),
                &envelope,
                &AcceptTestReceipt,
                200,
            ),
            Err(RmiArtifactEvidenceError::ReceiptContextMismatch)
        );
    }

    #[test]
    fn receipt_cannot_substitute_another_purpose() {
        let manifest = manifest();
        let mut envelope = receipt(&manifest);
        envelope.body.purpose = RmiArtifactPurposeV1::ModelTraining;
        envelope.body_sha256 = canonical_json_sha256(&envelope.body).unwrap();
        assert_eq!(
            verify_rmi_studio_evidence(
                &manifest,
                &payloads(),
                &envelope,
                &AcceptTestReceipt,
                200,
            ),
            Err(RmiArtifactEvidenceError::ReceiptContextMismatch)
        );
    }

    #[test]
    fn receipt_body_tampering_is_detected_before_verifier_call() {
        let manifest = manifest();
        let mut envelope = receipt(&manifest);
        envelope.body.transaction_id = "other-transaction".to_owned();
        assert_eq!(
            verify_rmi_studio_evidence(
                &manifest,
                &payloads(),
                &envelope,
                &AcceptTestReceipt,
                200,
            ),
            Err(RmiArtifactEvidenceError::ReceiptDigestMismatch)
        );
    }

    #[test]
    fn receipt_verification_time_cannot_precede_commit() {
        let manifest = manifest();
        let envelope = receipt(&manifest);
        assert_eq!(
            verify_rmi_studio_evidence(
                &manifest,
                &payloads(),
                &envelope,
                &AcceptTestReceipt,
                99,
            ),
            Err(RmiArtifactEvidenceError::ReceiptContextMismatch)
        );
    }

    #[test]
    fn unsupported_midi_is_not_inferred_from_a_midi_uri_or_boolean() {
        let recipe = b"recipe";
        let score = b"score";
        let audio = b"audio";
        let mut bundle = provenance_bundle(recipe, score, audio);
        bundle.reproduction.midi_exact = true;
        let verified = verify_piece_provenance_bytes(
            &bundle,
            PieceProvenanceBytesV1 {
                recipe,
                score,
                audio,
                midi: None,
            },
        )
        .unwrap();
        assert_eq!(verified.midi_sha256, None);
    }
}
