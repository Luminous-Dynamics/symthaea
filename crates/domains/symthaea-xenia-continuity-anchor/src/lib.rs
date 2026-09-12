// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified Xenia state-anchor adapter for exact Symthaea episodic continuity.
//!
//! This crate deliberately does not depend on the `xenia-peer` repository. Instead it defines a
//! narrow transport boundary whose implementations must return **already cryptographically
//! verified** Xenia state-anchor artifacts. Symthaea then verifies its own semantics independently:
//! the sidecar `ContinuityAnchorSnapshot` must validate and its exact commitment must equal the
//! opaque state commitment authenticated by Xenia.
//!
//! This preserves purpose separation in both directions:
//! - Xenia authenticates signer identity, target, revision/fork continuity, and the opaque state
//!   commitment, without learning Symthaea's memory/governance schema.
//! - Symthaea authenticates its own continuity snapshot and never needs to parse Xenia signature
//!   envelopes or key-transition artifacts here.
//!
//! A transport implementation is security-sensitive. It must not label an unverified network or
//! disk object as `VerifiedXeniaStateAnchor`. Real implementations should verify against the
//! deployment's retained Xenia key/key-transition and freshness policy before returning.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_episodic_continuity_anchor::{
    ContinuityAnchorSnapshot, ContinuityHeadAnchor,
};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use thiserror::Error;

/// Purpose-separated Xenia namespace reserved for Symthaea episodic continuity.
pub const XENIA_SYMTHAEA_CONTINUITY_NAMESPACE: &str =
    "symthaea.episodic-continuity.xenia-anchor.v1";
const MAX_NAMESPACE_BYTES: usize = 256;
const MAX_OBJECT_ID_BYTES: usize = 512;
const MAX_RETAINED_REF_BYTES: usize = 2048;
const MAX_SNAPSHOT_SIDECAR_BYTES: usize = 64 * 1024;

/// A Xenia artifact that the transport has already verified cryptographically and against its
/// deployment trust policy.
///
/// `snapshot_sidecar` is intentionally not trusted by Xenia. It is untrusted application data
/// whose exact Symthaea commitment is checked against `state_commitment` before it is returned to
/// the continuity protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedXeniaStateAnchor {
    /// Purpose-separated namespace authenticated by Xenia.
    pub namespace: String,
    /// Exact relying-system object/target authenticated by Xenia.
    pub object_id: String,
    /// Strict monotonic Xenia state-anchor revision.
    pub revision: u64,
    /// Fingerprint of the previous signed Xenia artifact, or `None` at revision 1.
    pub previous_artifact_fingerprint: Option<[u8; 32]>,
    /// Opaque state commitment authenticated by the Xenia signature.
    pub state_commitment: [u8; 32],
    /// Fingerprint of this exact signed Xenia artifact.
    pub artifact_fingerprint: [u8; 32],
    /// Serialized Symthaea snapshot sidecar. Its authenticity comes only from commitment equality.
    pub snapshot_sidecar: Vec<u8>,
    /// Durable/retained evidence reference returned by the Xenia backend.
    pub retained_ref: String,
}

/// Proposed Xenia state-anchor revision produced by the Symthaea adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaStateAnchorProposal {
    /// Purpose-separated namespace to sign.
    pub namespace: String,
    /// Exact target/object identifier to sign.
    pub object_id: String,
    /// Strict next revision.
    pub revision: u64,
    /// Fingerprint of the exact previous verified Xenia artifact.
    pub previous_artifact_fingerprint: Option<[u8; 32]>,
    /// Exact Symthaea continuity snapshot commitment to authenticate.
    pub state_commitment: [u8; 32],
    /// Untrusted sidecar retained with the signed artifact for Symthaea reconstruction.
    pub snapshot_sidecar: Vec<u8>,
}

/// Transport boundary to a Xenia state-anchor implementation.
///
/// Implementations MUST cryptographically verify returned artifacts under the deployment's trusted
/// Xenia identity/key-transition and freshness policy. CAS must atomically reject when the retained
/// artifact fingerprint differs from `expected_current_artifact`.
pub trait VerifiedXeniaStateAnchorTransport {
    type Error: StdError + Send + Sync + 'static;

    /// Load the latest verified artifact for one purpose-separated object.
    fn load_verified(
        &self,
        namespace: &str,
        object_id: &str,
    ) -> Result<Option<VerifiedXeniaStateAnchor>, Self::Error>;

    /// Atomically sign/retain `proposal` only if the backend's latest artifact fingerprint matches
    /// `expected_current_artifact` (`None` means the object must not yet exist).
    fn compare_and_swap_verified(
        &mut self,
        namespace: &str,
        object_id: &str,
        expected_current_artifact: Option<[u8; 32]>,
        proposal: &XeniaStateAnchorProposal,
    ) -> Result<VerifiedXeniaStateAnchor, Self::Error>;
}

/// Adapter implementing Symthaea's external continuity anchor contract over verified Xenia state
/// anchors.
pub struct XeniaContinuityHeadAnchor<T> {
    transport: T,
}

impl<T> XeniaContinuityHeadAnchor<T> {
    /// Wrap one verified Xenia state-anchor transport.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Borrow the underlying transport for diagnostics.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Mutably borrow the underlying transport for deployment-specific administration.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Consume the adapter and return its transport.
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
        let artifact = self
            .transport
            .load_verified(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE, store_target_id)
            .map_err(XeniaAnchorAdapterError::Transport)?;
        artifact
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
    if artifact.revision == 1 && artifact.previous_artifact_fingerprint.is_some() {
        return Err(XeniaAnchorAdapterError::UnexpectedRemotePredecessor);
    }
    if artifact.revision > 1 && artifact.previous_artifact_fingerprint.is_none() {
        return Err(XeniaAnchorAdapterError::MissingRemotePredecessor);
    }
    if artifact.state_commitment == [0; 32] {
        return Err(XeniaAnchorAdapterError::ZeroStateCommitment);
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

/// Semantic/transport failures surfaced by the verified Xenia adapter.
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
    #[error("Xenia state commitment must not be zero")]
    ZeroStateCommitment,
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
            if namespace != XENIA_SYMTHAEA_CONTINUITY_NAMESPACE
                || proposal.namespace != namespace
                || proposal.object_id != object_id
            {
                return Err(MockTransportError("target mismatch"));
            }
            let actual = self.current.as_ref().map(|entry| entry.artifact_fingerprint);
            if actual != expected_current_artifact {
                return Err(MockTransportError("stale remote CAS"));
            }
            if proposal.previous_artifact_fingerprint != expected_current_artifact {
                return Err(MockTransportError("proposal predecessor mismatch"));
            }
            let fingerprint = [proposal.revision as u8; 32];
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
                artifact_fingerprint: fingerprint,
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
            schema_version:
                symthaea_episodic_continuity_anchor::CONTINUITY_ANCHOR_SCHEMA.into(),
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
        let transport = MockVerifiedTransport::default();
        let mut adapter = XeniaContinuityHeadAnchor::new(transport);
        let first = snapshot(1, None, 10);
        let retained = adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        assert_eq!(retained, "mock-xenia-anchor:1");
        assert_eq!(
            adapter.load(&first.store_target_id).unwrap(),
            Some(first)
        );
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
    fn xenia_state_commitment_substitution_is_rejected_on_load() {
        let first = snapshot(1, None, 30);
        let sidecar = bincode::serialize(&first).unwrap();
        let transport = MockVerifiedTransport {
            current: Some(VerifiedXeniaStateAnchor {
                namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
                object_id: first.store_target_id.clone(),
                revision: 1,
                previous_artifact_fingerprint: None,
                state_commitment: digest(88).0,
                artifact_fingerprint: [7; 32],
                snapshot_sidecar: sidecar,
                retained_ref: "retained:1".into(),
            }),
            ..Default::default()
        };
        let adapter = XeniaContinuityHeadAnchor::new(transport);
        let error = adapter.load(&first.store_target_id).unwrap_err();
        assert!(matches!(
            error,
            XeniaAnchorAdapterError::StateCommitmentMismatch
        ));
    }

    #[test]
    fn xenia_and_symthaea_predecessor_chains_advance_together() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport::default());
        let first = snapshot(1, None, 40);
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
        let second = snapshot(2, Some(first_commitment), 41);
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
    fn malformed_return_from_transport_is_not_accepted_as_anchor_evidence() {
        let mut adapter = XeniaContinuityHeadAnchor::new(MockVerifiedTransport {
            corrupt_return_state_commitment: true,
            ..Default::default()
        });
        let first = snapshot(1, None, 50);
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
        let first = snapshot(1, None, 60);
        adapter
            .compare_and_swap(&first.store_target_id, None, &first)
            .unwrap();
        let first_commitment = first.commitment().unwrap();
        let third = snapshot(3, Some(first_commitment), 61);
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
