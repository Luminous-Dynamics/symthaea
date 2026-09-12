// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified Xenia state-anchor adapter for exact Symthaea episodic continuity.
//!
//! The adapter intentionally does not depend on the `xenia-peer` repository. A deployment-specific
//! transport verifies Xenia signatures/key continuity/freshness and returns a compact verified
//! artifact. Symthaea then independently validates its own serialized continuity snapshot and
//! requires its exact commitment to equal the opaque state commitment authenticated by Xenia.
//!
//! Two predecessor chains remain distinct:
//! - Symthaea revision N binds the previous `ContinuityAnchorSnapshot::commitment()`.
//! - Xenia revision N binds the fingerprint of the exact previous signed Xenia artifact.
//!
//! This is purpose separation, not duplicated cryptography.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_episodic_continuity_anchor::{
    CONTINUITY_ANCHOR_SCHEMA, ContinuityAnchorSnapshot, ContinuityHeadAnchor,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use thiserror::Error;

/// Purpose-separated Xenia namespace reserved for Symthaea episodic continuity.
pub const XENIA_SYMTHAEA_CONTINUITY_NAMESPACE: &str =
    "symthaea.episodic-continuity.xenia-anchor.v1";
const POLICY_CONTEXT_DOMAIN: &[u8] =
    b"symthaea.episodic-continuity.xenia-policy-context.v1\0";
const MAX_NAMESPACE_BYTES: usize = 256;
const MAX_OBJECT_ID_BYTES: usize = 512;
const MAX_RETAINED_REF_BYTES: usize = 2048;
const MAX_SNAPSHOT_SIDECAR_BYTES: usize = 64 * 1024;

/// Stable commitment to the exact Symthaea/Xenia interoperability policy context.
///
/// Xenia signs this value as its `policy_commitment`; a valid state commitment therefore cannot be
/// silently replayed under an unrelated Xenia anchor profile or relying-system namespace.
pub fn symthaea_xenia_policy_commitment() -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(POLICY_CONTEXT_DOMAIN);
    hasher.update(&(CONTINUITY_ANCHOR_SCHEMA.len() as u64).to_le_bytes());
    hasher.update(CONTINUITY_ANCHOR_SCHEMA.as_bytes());
    hasher.update(&(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.len() as u64).to_le_bytes());
    hasher.update(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.as_bytes());
    hasher.finalize()
}

/// A Xenia state-anchor artifact already verified cryptographically and against deployment policy.
///
/// The first seven fields map directly to Xenia `StateAnchorRecord` semantics. The signed artifact
/// fingerprint identifies the exact signature-bearing object. `snapshot_sidecar` is untrusted
/// application data until its Symthaea commitment is checked against `state_commitment`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedXeniaStateAnchor {
    pub namespace: String,
    pub object_id: String,
    pub revision: u64,
    pub previous_artifact_fingerprint: Option<[u8; 32]>,
    pub state_commitment: [u8; 32],
    pub policy_commitment: Option<[u8; 32]>,
    pub timestamp_unix_secs: u64,
    pub artifact_fingerprint: [u8; 32],
    pub snapshot_sidecar: Vec<u8>,
    pub retained_ref: String,
}

/// Exact semantic proposal a real transport maps to Xenia `StateAnchorRecord` before signing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaStateAnchorProposal {
    pub namespace: String,
    pub object_id: String,
    pub revision: u64,
    pub previous_artifact_fingerprint: Option<[u8; 32]>,
    pub state_commitment: [u8; 32],
    pub policy_commitment: Option<[u8; 32]>,
    pub timestamp_unix_secs: u64,
    pub snapshot_sidecar: Vec<u8>,
}

/// Boundary to a real Xenia state-anchor implementation.
///
/// Implementations MUST verify signatures, trusted signer/key-transition policy, exact target,
/// revision continuity, and freshness before returning `VerifiedXeniaStateAnchor`. CAS must
/// atomically reject when the retained artifact fingerprint differs from
/// `expected_current_artifact`.
pub trait VerifiedXeniaStateAnchorTransport {
    type Error: StdError + Send + Sync + 'static;

    fn load_verified(
        &self,
        namespace: &str,
        object_id: &str,
    ) -> Result<Option<VerifiedXeniaStateAnchor>, Self::Error>;

    fn compare_and_swap_verified(
        &mut self,
        namespace: &str,
        object_id: &str,
        expected_current_artifact: Option<[u8; 32]>,
        proposal: &XeniaStateAnchorProposal,
    ) -> Result<VerifiedXeniaStateAnchor, Self::Error>;
}

/// Implements Symthaea's `ContinuityHeadAnchor` over a verified Xenia transport.
pub struct XeniaContinuityHeadAnchor<T> {
    transport: T,
}

impl<T> XeniaContinuityHeadAnchor<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    pub fn into_inner(self) -> T {
        self.transport
    }
}

impl<T> XeniaContinuityHeadAnchor<T>
where
    T: VerifiedXeniaStateAnchorTransport,
{
    fn decode_verified_artifact(
        &self,
        expected_object_id: &str,
        artifact: &VerifiedXeniaStateAnchor,
    ) -> Result<ContinuityAnchorSnapshot, XeniaAnchorAdapterError<T::Error>> {
        validate_artifact_shape(expected_object_id, artifact)?;
        let snapshot: ContinuityAnchorSnapshot = bincode::deserialize(&artifact.snapshot_sidecar)
            .map_err(|error| XeniaAnchorAdapterError::SnapshotDecode(error.to_string()))?;
        snapshot
            .validate()
            .map_err(|error| XeniaAnchorAdapterError::Snapshot(error.to_string()))?;
        if snapshot.store_target_id != expected_object_id {
            return Err(XeniaAnchorAdapterError::SnapshotTargetMismatch);
        }
        if snapshot.revision != artifact.revision {
            return Err(XeniaAnchorAdapterError::RevisionMismatch {
                snapshot: snapshot.revision,
                xenia: artifact.revision,
            });
        }
        if snapshot.committed_at_unix_s != artifact.timestamp_unix_secs {
            return Err(XeniaAnchorAdapterError::TimestampMismatch {
                snapshot: snapshot.committed_at_unix_s,
                xenia: artifact.timestamp_unix_secs,
            });
        }
        let commitment = snapshot
            .commitment()
            .map_err(|error| XeniaAnchorAdapterError::Snapshot(error.to_string()))?;
        if commitment.0 != artifact.state_commitment {
            return Err(XeniaAnchorAdapterError::StateCommitmentMismatch);
        }
        Ok(snapshot)
    }
}

impl<T> ContinuityHeadAnchor for XeniaContinuityHeadAnchor<T>
where
    T: VerifiedXeniaStateAnchorTransport,
{
    type Error = XeniaAnchorAdapterError<T::Error>;

    fn load(
        &self,
        store_target_id: &str,
    ) -> Result<Option<ContinuityAnchorSnapshot>, Self::Error> {
        validate_text("store_target_id", store_target_id, MAX_OBJECT_ID_BYTES)?;
        self.transport
            .load_verified(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE, store_target_id)
            .map_err(XeniaAnchorAdapterError::Transport)?
            .as_ref()
            .map(|artifact| self.decode_verified_artifact(store_target_id, artifact))
            .transpose()
    }

    fn compare_and_swap(
        &mut self,
        store_target_id: &str,
        expected_current: Option<Sha256Digest>,
        next: &ContinuityAnchorSnapshot,
    ) -> Result<String, Self::Error> {
        validate_text("store_target_id", store_target_id, MAX_OBJECT_ID_BYTES)?;
        next.validate()
            .map_err(|error| XeniaAnchorAdapterError::Snapshot(error.to_string()))?;
        if next.store_target_id != store_target_id {
            return Err(XeniaAnchorAdapterError::SnapshotTargetMismatch);
        }

        let current_artifact = self
            .transport
            .load_verified(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE, store_target_id)
            .map_err(XeniaAnchorAdapterError::Transport)?;

        let expected_current_artifact = match (expected_current, current_artifact.as_ref()) {
            (None, None) => {
                if next.revision != 1 || next.previous_anchor_commitment.is_some() {
                    return Err(XeniaAnchorAdapterError::InvalidBootstrapRevision);
                }
                None
            }
            (None, Some(_)) => return Err(XeniaAnchorAdapterError::ExpectedEmptyAnchor),
            (Some(_), None) => return Err(XeniaAnchorAdapterError::MissingCurrentAnchor),
            (Some(expected), Some(current_artifact)) => {
                let current_snapshot =
                    self.decode_verified_artifact(store_target_id, current_artifact)?;
                let actual = current_snapshot
                    .commitment()
                    .map_err(|error| XeniaAnchorAdapterError::Snapshot(error.to_string()))?;
                if actual != expected {
                    return Err(XeniaAnchorAdapterError::StaleSymthaeaCommitment {
                        expected,
                        actual,
                    });
                }
                let expected_revision = current_snapshot
                    .revision
                    .checked_add(1)
                    .ok_or(XeniaAnchorAdapterError::RevisionOverflow)?;
                if next.revision != expected_revision {
                    return Err(XeniaAnchorAdapterError::NonSequentialRevision {
                        expected: expected_revision,
                        actual: next.revision,
                    });
                }
                if next.previous_anchor_commitment != Some(expected) {
                    return Err(XeniaAnchorAdapterError::PreviousSymthaeaCommitmentMismatch);
                }
                Some(current_artifact.artifact_fingerprint)
            }
        };

        let next_commitment = next
            .commitment()
            .map_err(|error| XeniaAnchorAdapterError::Snapshot(error.to_string()))?;
        let snapshot_sidecar = bincode::serialize(next)
            .map_err(|error| XeniaAnchorAdapterError::SnapshotEncode(error.to_string()))?;
        if snapshot_sidecar.len() > MAX_SNAPSHOT_SIDECAR_BYTES {
            return Err(XeniaAnchorAdapterError::SnapshotTooLarge {
                actual: snapshot_sidecar.len(),
                max: MAX_SNAPSHOT_SIDECAR_BYTES,
            });
        }

        let proposal = XeniaStateAnchorProposal {
            namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
            object_id: store_target_id.into(),
            revision: next.revision,
            previous_artifact_fingerprint: expected_current_artifact,
            state_commitment: next_commitment.0,
            policy_commitment: Some(symthaea_xenia_policy_commitment().0),
            timestamp_unix_secs: next.committed_at_unix_s,
            snapshot_sidecar,
        };
        let retained = self
            .transport
            .compare_and_swap_verified(
                XENIA_SYMTHAEA_CONTINUITY_NAMESPACE,
                store_target_id,
                expected_current_artifact,
                &proposal,
            )
            .map_err(XeniaAnchorAdapterError::Transport)?;
        let returned_snapshot = self.decode_verified_artifact(store_target_id, &retained)?;
        if returned_snapshot != *next {
            return Err(XeniaAnchorAdapterError::ReturnedSnapshotMismatch);
        }
        if retained.previous_artifact_fingerprint != expected_current_artifact {
            return Err(XeniaAnchorAdapterError::ReturnedPredecessorMismatch);
        }
        validate_text(
            "retained_ref",
            &retained.retained_ref,
            MAX_RETAINED_REF_BYTES,
        )?;
        Ok(retained.retained_ref)
    }
}

fn validate_artifact_shape<E>(
    expected_object_id: &str,
    artifact: &VerifiedXeniaStateAnchor,
) -> Result<(), XeniaAnchorAdapterError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    validate_text("namespace", &artifact.namespace, MAX_NAMESPACE_BYTES)?;
    validate_text("object_id", &artifact.object_id, MAX_OBJECT_ID_BYTES)?;
    validate_text(
        "retained_ref",
        &artifact.retained_ref,
        MAX_RETAINED_REF_BYTES,
    )?;
    if artifact.namespace != XENIA_SYMTHAEA_CONTINUITY_NAMESPACE {
        return Err(XeniaAnchorAdapterError::NamespaceMismatch);
    }
    if artifact.object_id != expected_object_id {
        return Err(XeniaAnchorAdapterError::ObjectMismatch);
    }
    if artifact.revision == 0 {
        return Err(XeniaAnchorAdapterError::ZeroRevision);
    }
    match (artifact.revision, artifact.previous_artifact_fingerprint) {
        (1, None) => {}
        (1, Some(_)) => return Err(XeniaAnchorAdapterError::UnexpectedRemotePredecessor),
        (_, None) => return Err(XeniaAnchorAdapterError::MissingRemotePredecessor),
        (_, Some([0; 32])) => return Err(XeniaAnchorAdapterError::ZeroRemotePredecessor),
        (_, Some(_)) => {}
    }
    if artifact.state_commitment == [0; 32] {
        return Err(XeniaAnchorAdapterError::ZeroStateCommitment);
    }
    if artifact.policy_commitment != Some(symthaea_xenia_policy_commitment().0) {
        return Err(XeniaAnchorAdapterError::PolicyCommitmentMismatch);
    }
    if artifact.artifact_fingerprint == [0; 32] {
        return Err(XeniaAnchorAdapterError::ZeroArtifactFingerprint);
    }
    if artifact.snapshot_sidecar.len() > MAX_SNAPSHOT_SIDECAR_BYTES {
        return Err(XeniaAnchorAdapterError::SnapshotTooLarge {
            actual: artifact.snapshot_sidecar.len(),
            max: MAX_SNAPSHOT_SIDECAR_BYTES,
        });
    }
    Ok(())
}

fn validate_text<E>(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), XeniaAnchorAdapterError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(XeniaAnchorAdapterError::InvalidText { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum XeniaAnchorAdapterError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("verified Xenia state-anchor transport failed: {0}")]
    Transport(#[source] E),
    #[error("invalid Xenia adapter text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("Xenia state-anchor namespace mismatch")]
    NamespaceMismatch,
    #[error("Xenia state-anchor object mismatch")]
    ObjectMismatch,
    #[error("Xenia state-anchor revision must be nonzero")]
    ZeroRevision,
    #[error("revision 1 Xenia anchor unexpectedly has a predecessor fingerprint")]
    UnexpectedRemotePredecessor,
    #[error("non-genesis Xenia anchor is missing its predecessor fingerprint")]
    MissingRemotePredecessor,
    #[error("Xenia predecessor fingerprint must not be zero")]
    ZeroRemotePredecessor,
    #[error("Xenia state commitment must not be zero")]
    ZeroStateCommitment,
    #[error("Xenia policy commitment does not match the Symthaea continuity profile")]
    PolicyCommitmentMismatch,
    #[error("Xenia artifact fingerprint must not be zero")]
    ZeroArtifactFingerprint,
    #[error("Symthaea snapshot sidecar is too large: actual={actual}, max={max}")]
    SnapshotTooLarge { actual: usize, max: usize },
    #[error("could not decode Symthaea continuity snapshot sidecar: {0}")]
    SnapshotDecode(String),
    #[error("could not encode Symthaea continuity snapshot sidecar: {0}")]
    SnapshotEncode(String),
    #[error("Symthaea continuity snapshot failed validation: {0}")]
    Snapshot(String),
    #[error("Symthaea snapshot target does not match the Xenia object")]
    SnapshotTargetMismatch,
    #[error("Symthaea/Xenia revision mismatch: snapshot={snapshot}, xenia={xenia}")]
    RevisionMismatch { snapshot: u64, xenia: u64 },
    #[error("Symthaea/Xenia timestamp mismatch: snapshot={snapshot}, xenia={xenia}")]
    TimestampMismatch { snapshot: u64, xenia: u64 },
    #[error("Xenia-authenticated state commitment does not match the Symthaea snapshot commitment")]
    StateCommitmentMismatch,
    #[error("bootstrap requires an absent remote Xenia anchor")]
    ExpectedEmptyAnchor,
    #[error("expected a current remote Xenia anchor but none exists")]
    MissingCurrentAnchor,
    #[error("bootstrap must write Symthaea revision 1 with no previous commitment")]
    InvalidBootstrapRevision,
    #[error("stale Symthaea continuity commitment: expected={expected:?}, actual={actual:?}")]
    StaleSymthaeaCommitment {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("Symthaea continuity revision overflow")]
    RevisionOverflow,
    #[error("non-sequential Symthaea continuity revision: expected={expected}, actual={actual}")]
    NonSequentialRevision { expected: u64, actual: u64 },
    #[error("next Symthaea snapshot does not bind the expected previous Symthaea commitment")]
    PreviousSymthaeaCommitmentMismatch,
    #[error("verified Xenia CAS returned a different Symthaea snapshot than proposed")]
    ReturnedSnapshotMismatch,
    #[error("verified Xenia CAS returned the wrong previous artifact fingerprint")]
    ReturnedPredecessorMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockTransportError(&'static str);

    impl fmt::Display for MockTransportError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl StdError for MockTransportError {}

    #[derive(Default)]
    struct MockVerifiedTransport {
        current: Option<VerifiedXeniaStateAnchor>,
        cas_calls: usize,
        corrupt_return_state_commitment: bool,
    }

    impl VerifiedXeniaStateAnchorTransport for MockVerifiedTransport {
        type Error = MockTransportError;

        fn load_verified(
            &self,
            namespace: &str,
            object_id: &str,
        ) -> Result<Option<VerifiedXeniaStateAnchor>, Self::Error> {
            if namespace != XENIA_SYMTHAEA_CONTINUITY_NAMESPACE {
                return Err(MockTransportError("wrong namespace"));
            }
            if let Some(current) = &self.current {
                if current.object_id != object_id {
                    return Ok(None);
                }
            }
            Ok(self.current.clone())
        }

        fn compare_and_swap_verified(
            &mut self,
            namespace: &str,
            object_id: &str,
            expected_current_artifact: Option<[u8; 32]>,
            proposal: &XeniaStateAnchorProposal,
        ) -> Result<VerifiedXeniaStateAnchor, Self::Error> {
            self.cas_calls += 1;
            if namespace != proposal.namespace || object_id != proposal.object_id {
                return Err(MockTransportError("target mismatch"));
            }
            let actual = self.current.as_ref().map(|entry| entry.artifact_fingerprint);
            if actual != expected_current_artifact
                || proposal.previous_artifact_fingerprint != expected_current_artifact
            {
                return Err(MockTransportError("stale remote CAS"));
            }
            let mut state_commitment = proposal.state_commitment;
            if self.corrupt_return_state_commitment {
                state_commitment[0] ^= 0x55;
            }
            let retained = VerifiedXeniaStateAnchor {
                namespace: proposal.namespace.clone(),
                object_id: proposal.object_id.clone(),
                revision: proposal.revision,
                previous_artifact_fingerprint: proposal.previous_artifact_fingerprint,
                state_commitment,
                policy_commitment: proposal.policy_commitment,
                timestamp_unix_secs: proposal.timestamp_unix_secs,
                artifact_fingerprint: [proposal.revision as u8; 32],
                snapshot_sidecar: proposal.snapshot_sidecar.clone(),
                retained_ref: format!("mock-xenia-anchor:{}", proposal.revision),
            };
            self.current = Some(retained.clone());
            Ok(retained)
        }
    }

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn snapshot(
        revision: u64,
        previous_anchor_commitment: Option<Sha256Digest>,
        seed: u8,
    ) -> ContinuityAnchorSnapshot {
        ContinuityAnchorSnapshot {
            schema_version: CONTINUITY_ANCHOR_SCHEMA.into(),
            store_target_id: "symthaea:self:episodic-memory".into(),
            revision,
            intent_generation: revision - 1,
            intent_head: digest(seed),
            quarantine_generation: revision - 1,
            quarantine_head: digest(seed.wrapping_add(1)),
            continuity_manifest_digest: digest(seed.wrapping_add(2)),
            committed_at_unix_s: 1_800_000_000 + revision,
            previous_anchor_commitment,
        }
    }

    #[test]
    fn bootstrap_and_load_round_trip_exact_snapshot() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport::default());
        let first = snapshot(1, None, 10);
        let retained = adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        assert_eq!(retained, "mock-xenia-anchor:1");
        let remote = adapter.transport().current.as_ref().unwrap();
        assert_eq!(
            remote.policy_commitment,
            Some(symthaea_xenia_policy_commitment().0)
        );
        assert_eq!(remote.timestamp_unix_secs, first.committed_at_unix_s);
        assert_eq!(adapter.load(&first.store_target_id).unwrap(), Some(first));
    }

    #[test]
    fn stale_symthaea_commitment_is_rejected_before_remote_cas() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport::default());
        let first = snapshot(1, None, 20);
        adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        let first_commitment = first.commitment().unwrap();
        let second = snapshot(2, Some(first_commitment), 21);
        let before = adapter.transport().cas_calls;
        let error = adapter
            .compare_and_swap(&second.store_target_id, Some(digest(99)), &second)
            .unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::StaleSymthaeaCommitment { .. }
        ));
        assert_eq!(adapter.transport().cas_calls, before);
    }

    #[test]
    fn policy_context_substitution_is_rejected() {
        let first = snapshot(1, None, 30);
        let artifact = VerifiedXeniaStateAnchor {
            namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
            object_id: first.store_target_id.clone(),
            revision: 1,
            previous_artifact_fingerprint: None,
            state_commitment: first.commitment().unwrap().0,
            policy_commitment: Some(digest(99).0),
            timestamp_unix_secs: first.committed_at_unix_s,
            artifact_fingerprint: [7; 32],
            snapshot_sidecar: bincode::serialize(&first).unwrap(),
            retained_ref: "retained:1".into(),
        };
        let adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport {
            current: Some(artifact),
            ..Default::default()
        });
        let error = adapter.load(&first.store_target_id).unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::PolicyCommitmentMismatch
        ));
    }

    #[test]
    fn signed_timestamp_must_match_snapshot_timestamp() {
        let first = snapshot(1, None, 35);
        let artifact = VerifiedXeniaStateAnchor {
            namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
            object_id: first.store_target_id.clone(),
            revision: 1,
            previous_artifact_fingerprint: None,
            state_commitment: first.commitment().unwrap().0,
            policy_commitment: Some(symthaea_xenia_policy_commitment().0),
            timestamp_unix_secs: first.committed_at_unix_s + 1,
            artifact_fingerprint: [8; 32],
            snapshot_sidecar: bincode::serialize(&first).unwrap(),
            retained_ref: "retained:1".into(),
        };
        let adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport {
            current: Some(artifact),
            ..Default::default()
        });
        let error = adapter.load(&first.store_target_id).unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::TimestampMismatch { .. }
        ));
    }

    #[test]
    fn xenia_state_commitment_substitution_is_rejected_on_load() {
        let first = snapshot(1, None, 40);
        let artifact = VerifiedXeniaStateAnchor {
            namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
            object_id: first.store_target_id.clone(),
            revision: 1,
            previous_artifact_fingerprint: None,
            state_commitment: digest(88).0,
            policy_commitment: Some(symthaea_xenia_policy_commitment().0),
            timestamp_unix_secs: first.committed_at_unix_s,
            artifact_fingerprint: [9; 32],
            snapshot_sidecar: bincode::serialize(&first).unwrap(),
            retained_ref: "retained:1".into(),
        };
        let adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport {
            current: Some(artifact),
            ..Default::default()
        });
        let error = adapter.load(&first.store_target_id).unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::StateCommitmentMismatch
        ));
    }

    #[test]
    fn xenia_and_symthaea_predecessor_chains_advance_together() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport::default());
        let first = snapshot(1, None, 50);
        adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        let first_commitment = first.commitment().unwrap();
        let first_xenia_fingerprint = adapter
            .transport()
            .current
            .as_ref()
            .unwrap()
            .artifact_fingerprint;
        let second = snapshot(2, Some(first_commitment), 51);
        adapter
            .compare_and_swap(
                &second.store_target_id,
                Some(first_commitment),
                &second,
            )
            .unwrap();
        let current = adapter.transport().current.as_ref().unwrap();
        assert_eq!(
            current.previous_artifact_fingerprint,
            Some(first_xenia_fingerprint)
        );
        assert_eq!(adapter.load(&second.store_target_id).unwrap(), Some(second));
    }

    #[test]
    fn malformed_return_from_transport_is_not_accepted() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport {
            corrupt_return_state_commitment: true,
            ..Default::default()
        });
        let first = snapshot(1, None, 60);
        let error = adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::StateCommitmentMismatch
        ));
    }

    #[test]
    fn skipped_symthaea_revision_is_rejected_before_remote_write() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport::default());
        let first = snapshot(1, None, 70);
        adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        let first_commitment = first.commitment().unwrap();
        let third = snapshot(3, Some(first_commitment), 71);
        let before = adapter.transport().cas_calls;
        let error = adapter
            .compare_and_swap(&third.store_target_id, Some(first_commitment), &third)
            .unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::NonSequentialRevision {
                expected: 2,
                actual: 3
            }
        ));
        assert_eq!(adapter.transport().cas_calls, before);
    }
}
