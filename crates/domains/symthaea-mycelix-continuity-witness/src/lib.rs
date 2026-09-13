// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-system verification of independently retained Xenia continuity checkpoints.
//!
//! This crate composes two deliberately different evidence systems without making either one
//! impersonate the other:
//!
//! - **Xenia** authenticates the checkpoint artifact, signer/key policy, target and freshness.
//! - **Mycelix** supplies independent append-only retention/observation evidence for the exact
//!   checkpoint bytes.
//!
//! A checkpoint is accepted only when exactly one candidate exists for the requested object/revision,
//! the exact Mycelix byte digest recomputes, the caller's distinct-witness policy is satisfied, and
//! Xenia verification of those same bytes agrees on target fingerprint, revision and anchor
//! fingerprint. Witness count never resolves a fork.
//!
//! `IndependentlyWitnessedCheckpoint` is evidence, **not authority** and not an execution permit.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_xenia_continuity_anchor::XENIA_SYMTHAEA_CONTINUITY_NAMESPACE;
use thiserror::Error;

/// Mycelix entry schema for retained external checkpoints.
pub const MYCELIX_CONTINUITY_CHECKPOINT_SCHEMA: &str =
    "mycelix.continuity-witness.checkpoint.v1";
/// Xenia checkpoint protocol retained by the Mycelix v1 witness profile.
pub const XENIA_STATE_ANCHOR_CHECKPOINT_SCHEMA: &str = "xenia-state-anchor-checkpoint-v1";
/// Digest domain used by Mycelix `continuity-witness` for opaque checkpoint bytes.
pub const MYCELIX_CHECKPOINT_DIGEST_DOMAIN: &[u8] =
    b"mycelix.continuity-witness.checkpoint-digest.v1\0";

const MAX_TARGET_ID_BYTES: usize = 512;
const MAX_CHECKPOINT_ACTION_REF_BYTES: usize = 2048;
const MAX_WITNESS_ID_BYTES: usize = 1024;
const MAX_CHECKPOINT_BYTES: usize = 64 * 1024;
const MAX_WITNESSES: usize = 4096;

/// Explicit relying-system policy for independent Mycelix observation.
///
/// Distinct agent identifiers are evidence of distinct Holochain keys only. They are not, by
/// themselves, proof of distinct human operators or administrative domains. Higher deployments
/// should layer DID/council/reputation/operator-diversity policy above this primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MycelixWitnessPolicy {
    pub minimum_distinct_witnesses: u32,
}

impl MycelixWitnessPolicy {
    pub fn validate(self) -> Result<(), WitnessPolicyError> {
        if self.minimum_distinct_witnesses == 0 {
            return Err(WitnessPolicyError::ZeroWitnessThreshold);
        }
        if self.minimum_distinct_witnesses as usize > MAX_WITNESSES {
            return Err(WitnessPolicyError::WitnessThresholdTooLarge {
                actual: self.minimum_distinct_witnesses,
                max: MAX_WITNESSES as u32,
            });
        }
        Ok(())
    }
}

/// One Mycelix-retained candidate at an explicit object/revision.
///
/// This is transport data and is not trusted until [`verify_independently_witnessed_checkpoint`]
/// recomputes the byte digest and composes it with Xenia verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MycelixCheckpointCandidate {
    pub entry_schema: String,
    pub source_protocol: String,
    pub checkpoint_action_ref: String,
    pub anchor_fingerprint: [u8; 32],
    pub checkpoint_digest: [u8; 32],
    pub checkpoint_bytes: Vec<u8>,
    pub witness_ids: Vec<String>,
}

/// Fail-closed witness status for one explicit object/revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MycelixRevisionWitnessStatus {
    pub namespace: String,
    pub object_fingerprint: [u8; 32],
    pub revision: u64,
    pub fork_detected: bool,
    pub candidates: Vec<MycelixCheckpointCandidate>,
}

/// Minimal result that a deployment-specific Xenia verifier may return after verifying the exact
/// checkpoint bytes against its trusted signer/key-transition/freshness policy and the supplied
/// expected namespace/object ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedXeniaCheckpointView {
    pub target_fingerprint: [u8; 32],
    pub revision: u64,
    pub anchor_fingerprint: [u8; 32],
    pub checkpoint_timestamp_unix_secs: u64,
}

/// Boundary to Xenia's real checkpoint parser and cryptographic verifier.
///
/// Implementations MUST:
/// - parse exactly `checkpoint_bytes` as the supported Xenia checkpoint schema;
/// - verify its signature and trusted signer/key-transition policy;
/// - verify freshness according to deployment policy;
/// - verify its target fingerprint corresponds to `expected_namespace` + `expected_object_id`;
/// - return only after all of those checks succeed.
pub trait VerifiedXeniaCheckpointBytesVerifier {
    type Error: StdError + Send + Sync + 'static;

    fn verify_checkpoint_bytes(
        &self,
        checkpoint_bytes: &[u8],
        expected_namespace: &str,
        expected_object_id: &str,
    ) -> Result<VerifiedXeniaCheckpointView, Self::Error>;
}

/// Evidence emitted only after the Mycelix and Xenia views agree on one exact checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependentlyWitnessedCheckpoint {
    pub object_id: String,
    pub revision: u64,
    pub object_fingerprint: [u8; 32],
    pub anchor_fingerprint: [u8; 32],
    pub checkpoint_digest: Sha256Digest,
    pub checkpoint_action_ref: String,
    pub distinct_witness_ids: Vec<String>,
    pub xenia_checkpoint_timestamp_unix_secs: u64,
}

/// Recompute the exact digest used by Mycelix continuity-witness entries.
pub fn mycelix_checkpoint_digest(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(MYCELIX_CHECKPOINT_DIGEST_DOMAIN);
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    hasher.finalize()
}

/// Compose independent Mycelix retention with Xenia cryptographic checkpoint verification.
///
/// This function never chooses among forks. `fork_detected == true` **or** more than one visible
/// candidate makes the revision unusable regardless of witness counts.
pub fn verify_independently_witnessed_checkpoint<V>(
    expected_object_id: &str,
    expected_revision: u64,
    status: &MycelixRevisionWitnessStatus,
    policy: MycelixWitnessPolicy,
    xenia: &V,
) -> Result<IndependentlyWitnessedCheckpoint, WitnessVerificationError<V::Error>>
where
    V: VerifiedXeniaCheckpointBytesVerifier,
{
    policy.validate()?;
    validate_text("expected_object_id", expected_object_id, MAX_TARGET_ID_BYTES)?;
    if expected_revision == 0 {
        return Err(WitnessVerificationError::ZeroRevision);
    }
    if status.namespace != XENIA_SYMTHAEA_CONTINUITY_NAMESPACE {
        return Err(WitnessVerificationError::NamespaceMismatch);
    }
    if status.revision != expected_revision {
        return Err(WitnessVerificationError::RevisionMismatch {
            expected: expected_revision,
            actual: status.revision,
        });
    }
    if status.object_fingerprint == [0; 32] {
        return Err(WitnessVerificationError::ZeroObjectFingerprint);
    }
    if status.fork_detected || status.candidates.len() > 1 {
        return Err(WitnessVerificationError::ForkDetected {
            candidates: status.candidates.len(),
        });
    }
    let candidate = status
        .candidates
        .first()
        .ok_or(WitnessVerificationError::MissingCandidate)?;
    validate_candidate_shape(candidate)?;

    let actual_digest = mycelix_checkpoint_digest(&candidate.checkpoint_bytes);
    if actual_digest.0 != candidate.checkpoint_digest {
        return Err(WitnessVerificationError::CheckpointDigestMismatch);
    }

    let distinct_witness_ids = distinct_valid_witness_ids(&candidate.witness_ids)?;
    if distinct_witness_ids.len() < policy.minimum_distinct_witnesses as usize {
        return Err(WitnessVerificationError::InsufficientDistinctWitnesses {
            required: policy.minimum_distinct_witnesses,
            actual: distinct_witness_ids.len() as u32,
        });
    }

    let verified = xenia
        .verify_checkpoint_bytes(
            &candidate.checkpoint_bytes,
            XENIA_SYMTHAEA_CONTINUITY_NAMESPACE,
            expected_object_id,
        )
        .map_err(WitnessVerificationError::Xenia)?;
    if verified.target_fingerprint != status.object_fingerprint {
        return Err(WitnessVerificationError::TargetFingerprintMismatch);
    }
    if verified.revision != expected_revision {
        return Err(WitnessVerificationError::XeniaRevisionMismatch {
            expected: expected_revision,
            actual: verified.revision,
        });
    }
    if verified.anchor_fingerprint != candidate.anchor_fingerprint {
        return Err(WitnessVerificationError::AnchorFingerprintMismatch);
    }

    Ok(IndependentlyWitnessedCheckpoint {
        object_id: expected_object_id.to_string(),
        revision: expected_revision,
        object_fingerprint: status.object_fingerprint,
        anchor_fingerprint: candidate.anchor_fingerprint,
        checkpoint_digest: actual_digest,
        checkpoint_action_ref: candidate.checkpoint_action_ref.clone(),
        distinct_witness_ids,
        xenia_checkpoint_timestamp_unix_secs: verified.checkpoint_timestamp_unix_secs,
    })
}

fn validate_candidate_shape<E>(
    candidate: &MycelixCheckpointCandidate,
) -> Result<(), WitnessVerificationError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    if candidate.entry_schema != MYCELIX_CONTINUITY_CHECKPOINT_SCHEMA {
        return Err(WitnessVerificationError::MycelixSchemaMismatch);
    }
    if candidate.source_protocol != XENIA_STATE_ANCHOR_CHECKPOINT_SCHEMA {
        return Err(WitnessVerificationError::SourceProtocolMismatch);
    }
    validate_text(
        "checkpoint_action_ref",
        &candidate.checkpoint_action_ref,
        MAX_CHECKPOINT_ACTION_REF_BYTES,
    )?;
    if candidate.anchor_fingerprint == [0; 32] {
        return Err(WitnessVerificationError::ZeroAnchorFingerprint);
    }
    if candidate.checkpoint_digest == [0; 32] {
        return Err(WitnessVerificationError::ZeroCheckpointDigest);
    }
    if candidate.checkpoint_bytes.is_empty() {
        return Err(WitnessVerificationError::EmptyCheckpointBytes);
    }
    if candidate.checkpoint_bytes.len() > MAX_CHECKPOINT_BYTES {
        return Err(WitnessVerificationError::CheckpointTooLarge {
            actual: candidate.checkpoint_bytes.len(),
            max: MAX_CHECKPOINT_BYTES,
        });
    }
    if candidate.witness_ids.len() > MAX_WITNESSES {
        return Err(WitnessVerificationError::TooManyWitnessIds {
            actual: candidate.witness_ids.len(),
            max: MAX_WITNESSES,
        });
    }
    Ok(())
}

fn distinct_valid_witness_ids<E>(
    witness_ids: &[String],
) -> Result<Vec<String>, WitnessVerificationError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    let mut distinct = BTreeSet::new();
    for witness_id in witness_ids {
        validate_text("witness_id", witness_id, MAX_WITNESS_ID_BYTES)?;
        distinct.insert(witness_id.clone());
    }
    Ok(distinct.into_iter().collect())
}

fn validate_text<E>(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), WitnessVerificationError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(WitnessVerificationError::InvalidText { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WitnessPolicyError {
    #[error("Mycelix witness threshold must be non-zero")]
    ZeroWitnessThreshold,
    #[error("Mycelix witness threshold too large: actual={actual}, max={max}")]
    WitnessThresholdTooLarge { actual: u32, max: u32 },
}

#[derive(Debug, Error)]
pub enum WitnessVerificationError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Policy(#[from] WitnessPolicyError),
    #[error("invalid witness text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("requested continuity revision must be non-zero")]
    ZeroRevision,
    #[error("Mycelix witness namespace does not match the Symthaea/Xenia continuity profile")]
    NamespaceMismatch,
    #[error("Mycelix revision mismatch: expected={expected}, actual={actual}")]
    RevisionMismatch { expected: u64, actual: u64 },
    #[error("Mycelix object fingerprint must not be zero")]
    ZeroObjectFingerprint,
    #[error("same-revision continuity fork detected across {candidates} visible candidates")]
    ForkDetected { candidates: usize },
    #[error("no Mycelix checkpoint candidate is visible for the requested revision")]
    MissingCandidate,
    #[error("Mycelix checkpoint entry schema mismatch")]
    MycelixSchemaMismatch,
    #[error("Mycelix retained checkpoint source protocol is not the supported Xenia checkpoint profile")]
    SourceProtocolMismatch,
    #[error("Mycelix retained anchor fingerprint must not be zero")]
    ZeroAnchorFingerprint,
    #[error("Mycelix retained checkpoint digest must not be zero")]
    ZeroCheckpointDigest,
    #[error("Mycelix retained checkpoint bytes are empty")]
    EmptyCheckpointBytes,
    #[error("Mycelix retained checkpoint too large: actual={actual}, max={max}")]
    CheckpointTooLarge { actual: usize, max: usize },
    #[error("too many witness IDs: actual={actual}, max={max}")]
    TooManyWitnessIds { actual: usize, max: usize },
    #[error("Mycelix checkpoint byte digest mismatch")]
    CheckpointDigestMismatch,
    #[error("insufficient distinct Mycelix witnesses: required={required}, actual={actual}")]
    InsufficientDistinctWitnesses { required: u32, actual: u32 },
    #[error("Xenia checkpoint verification failed: {0}")]
    Xenia(E),
    #[error("Xenia verified target fingerprint disagrees with Mycelix")]
    TargetFingerprintMismatch,
    #[error("Xenia verified revision mismatch: expected={expected}, actual={actual}")]
    XeniaRevisionMismatch { expected: u64, actual: u64 },
    #[error("Xenia verified anchor fingerprint disagrees with Mycelix")]
    AnchorFingerprintMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    const OBJECT: &str = "symthaea:self:episodic-memory";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MockXeniaError;

    impl fmt::Display for MockXeniaError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "mock xenia verification failure")
        }
    }

    impl StdError for MockXeniaError {}

    struct MockVerifier {
        expected_bytes: Vec<u8>,
        view: VerifiedXeniaCheckpointView,
        fail: bool,
    }

    impl VerifiedXeniaCheckpointBytesVerifier for MockVerifier {
        type Error = MockXeniaError;

        fn verify_checkpoint_bytes(
            &self,
            checkpoint_bytes: &[u8],
            expected_namespace: &str,
            expected_object_id: &str,
        ) -> Result<VerifiedXeniaCheckpointView, Self::Error> {
            if self.fail
                || checkpoint_bytes != self.expected_bytes
                || expected_namespace != XENIA_SYMTHAEA_CONTINUITY_NAMESPACE
                || expected_object_id != OBJECT
            {
                return Err(MockXeniaError);
            }
            Ok(self.view)
        }
    }

    fn fixture() -> (
        MycelixRevisionWitnessStatus,
        MycelixWitnessPolicy,
        MockVerifier,
    ) {
        let checkpoint_bytes = b"verified-xenia-checkpoint-v1".to_vec();
        let digest = mycelix_checkpoint_digest(&checkpoint_bytes).0;
        let object_fingerprint = [0x11; 32];
        let anchor_fingerprint = [0x22; 32];
        let status = MycelixRevisionWitnessStatus {
            namespace: XENIA_SYMTHAEA_CONTINUITY_NAMESPACE.into(),
            object_fingerprint,
            revision: 7,
            fork_detected: false,
            candidates: vec![MycelixCheckpointCandidate {
                entry_schema: MYCELIX_CONTINUITY_CHECKPOINT_SCHEMA.into(),
                source_protocol: XENIA_STATE_ANCHOR_CHECKPOINT_SCHEMA.into(),
                checkpoint_action_ref: "uhCk-example-action".into(),
                anchor_fingerprint,
                checkpoint_digest: digest,
                checkpoint_bytes: checkpoint_bytes.clone(),
                witness_ids: vec!["agent-a".into(), "agent-b".into()],
            }],
        };
        let policy = MycelixWitnessPolicy {
            minimum_distinct_witnesses: 2,
        };
        let verifier = MockVerifier {
            expected_bytes: checkpoint_bytes,
            view: VerifiedXeniaCheckpointView {
                target_fingerprint: object_fingerprint,
                revision: 7,
                anchor_fingerprint,
                checkpoint_timestamp_unix_secs: 1_800_000_007,
            },
            fail: false,
        };
        (status, policy, verifier)
    }

    #[test]
    fn unique_threshold_met_checkpoint_is_accepted() {
        let (status, policy, verifier) = fixture();
        let verified = verify_independently_witnessed_checkpoint(
            OBJECT,
            7,
            &status,
            policy,
            &verifier,
        )
        .unwrap();
        assert_eq!(verified.revision, 7);
        assert_eq!(verified.distinct_witness_ids, vec!["agent-a", "agent-b"]);
        assert_eq!(verified.anchor_fingerprint, [0x22; 32]);
    }

    #[test]
    fn fork_is_never_resolved_by_more_witnesses() {
        let (mut status, policy, verifier) = fixture();
        let mut fork = status.candidates[0].clone();
        fork.anchor_fingerprint = [0x33; 32];
        fork.witness_ids = vec![
            "agent-c".into(),
            "agent-d".into(),
            "agent-e".into(),
            "agent-f".into(),
        ];
        status.candidates.push(fork);
        status.fork_detected = false; // verifier must not trust this flag alone.
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::ForkDetected { candidates: 2 })
        ));
    }

    #[test]
    fn duplicate_witness_identity_does_not_count_twice() {
        let (mut status, policy, verifier) = fixture();
        status.candidates[0].witness_ids = vec!["agent-a".into(), "agent-a".into()];
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::InsufficientDistinctWitnesses {
                required: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn tampered_retained_bytes_fail_before_xenia_acceptance() {
        let (mut status, policy, verifier) = fixture();
        status.candidates[0].checkpoint_bytes.push(0xff);
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::CheckpointDigestMismatch)
        ));
    }

    #[test]
    fn xenia_target_disagreement_fails_closed() {
        let (status, policy, mut verifier) = fixture();
        verifier.view.target_fingerprint = [0x99; 32];
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::TargetFingerprintMismatch)
        ));
    }

    #[test]
    fn xenia_anchor_fingerprint_disagreement_fails_closed() {
        let (status, policy, mut verifier) = fixture();
        verifier.view.anchor_fingerprint = [0x99; 32];
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::AnchorFingerprintMismatch)
        ));
    }

    #[test]
    fn xenia_verifier_failure_is_not_downgraded() {
        let (status, policy, mut verifier) = fixture();
        verifier.fail = true;
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::Xenia(MockXeniaError))
        ));
    }

    #[test]
    fn malformed_or_zero_policy_is_rejected() {
        let (status, _, verifier) = fixture();
        let policy = MycelixWitnessPolicy {
            minimum_distinct_witnesses: 0,
        };
        assert!(matches!(
            verify_independently_witnessed_checkpoint(OBJECT, 7, &status, policy, &verifier),
            Err(WitnessVerificationError::Policy(
                WitnessPolicyError::ZeroWitnessThreshold
            ))
        ));
    }
}
