// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic tamper-evident checkpoints for effect-bound CogSec evidence.
//!
//! This layer binds ordinary portable evidence into a hash chain. It does not
//! authenticate the host or turn a locally computed hash into trusted evidence.
//! External signers/witnesses should bind the checkpoint signing message/root.
//! Disclosure of a checkpoint root is also not automatically privacy-safe: a
//! root over low-entropy private metadata may still be sensitive under the
//! evidence release policy.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use symthaea_cogsec::Digest32;
use thiserror::Error;

use crate::{EffectBoundEvidenceSnapshot, ObservedEffectBinding};

/// Schema version for deterministic effect-bound evidence checkpoints.
pub const EVIDENCE_CHECKPOINT_SCHEMA_V1: u16 = 1;

const SNAPSHOT_DOMAIN: &[u8] = b"symthaea:cogsec:effect-bound-snapshot:v1";
const CHECKPOINT_DOMAIN: &[u8] = b"symthaea:cogsec:evidence-checkpoint:v1";
const SIGNING_DOMAIN: &[u8] = b"symthaea:cogsec:evidence-checkpoint-signing:v1";

/// One deterministic checkpoint over a portable effect-bound evidence snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCheckpoint {
    /// Checkpoint schema version.
    pub schema_version: u16,
    /// Monotonic checkpoint index within this evidence lineage.
    pub checkpoint_index: u64,
    /// Root of the immediately preceding checkpoint, absent only at lineage genesis.
    pub previous_checkpoint_root: Option<Digest32>,
    /// Canonical commitment to the complete effect-bound snapshot.
    pub snapshot_root: Digest32,
    /// Ledger epoch summarized by this checkpoint.
    pub ledger_epoch: u64,
    /// Last event sequence allocated by the ledger at checkpoint time.
    pub last_assigned_sequence: u64,
    /// Number of retained typed events in the checkpointed snapshot.
    pub retained_event_count: u64,
    /// Number of exact-effect sidecar bindings in the snapshot.
    pub effect_binding_count: u64,
    /// Domain-separated root committing to this checkpoint and its predecessor.
    pub checkpoint_root: Digest32,
}

impl EvidenceCheckpoint {
    /// Message external signature/witness adapters should bind.
    ///
    /// This is deliberately domain-separated from the raw checkpoint root so a
    /// signature obtained for another protocol cannot be silently repurposed.
    pub fn signing_message(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SIGNING_DOMAIN.len() + 8 + 32);
        push_bytes(&mut out, SIGNING_DOMAIN);
        out.extend_from_slice(&self.checkpoint_root.0);
        out
    }
}

/// Portable evidence plus the deterministic checkpoint that commits to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointedEffectBoundEvidence {
    /// Portable typed-event + exact-effect evidence.
    pub snapshot: EffectBoundEvidenceSnapshot,
    /// Deterministic hash-chain checkpoint over `snapshot`.
    pub checkpoint: EvidenceCheckpoint,
}

/// Failure while deterministically constructing a checkpoint.
#[derive(Debug, Error)]
pub enum CheckpointBuildError {
    /// Portable evidence could not be represented in the canonical value model.
    #[error("failed to canonicalize CogSec evidence: {0}")]
    Canonicalization(#[from] serde_json::Error),
    /// Checkpoint index cannot advance without wrapping.
    #[error("CogSec evidence checkpoint index exhausted")]
    CheckpointIndexExhausted,
    /// Same-ledger history moved backwards relative to its anchored predecessor.
    #[error(
        "CogSec evidence ledger sequence rolled back within epoch {ledger_epoch}: {before} -> {after}"
    )]
    LedgerRollbackWithinEpoch {
        /// Ledger epoch whose sequence regressed.
        ledger_epoch: u64,
        /// Sequence in the predecessor checkpoint.
        before: u64,
        /// Sequence in the proposed successor snapshot.
        after: u64,
    },
    /// Snapshot changed while claiming the exact same same-epoch sequence frontier.
    #[error("CogSec evidence changed without advancing the ledger sequence frontier")]
    SameSequenceDifferentSnapshot,
}

/// Deterministic contradiction found while verifying a checkpoint chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointViolation {
    /// Checkpoint schema is unsupported.
    UnsupportedSchema {
        /// Checkpoint index containing the unsupported schema.
        checkpoint_index: u64,
        /// Version found.
        found: u16,
    },
    /// Checkpoint index does not continue from the supplied anchor/predecessor.
    CheckpointIndexMismatch {
        /// Index found in the checkpoint.
        found: u64,
        /// Index required by chain continuity.
        expected: u64,
    },
    /// Previous-root link does not match the anchor/predecessor.
    PreviousRootMismatch {
        /// Checkpoint index containing the bad link.
        checkpoint_index: u64,
        /// Link found in the checkpoint.
        found: Option<Digest32>,
        /// Link required by chain continuity.
        expected: Option<Digest32>,
    },
    /// Checkpoint ledger epoch summary disagrees with the embedded snapshot.
    LedgerEpochMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Checkpoint sequence summary disagrees with the embedded snapshot.
    LastSequenceMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Checkpoint retained-event count disagrees with the embedded snapshot.
    RetainedEventCountMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Checkpoint sidecar count disagrees with the embedded snapshot.
    EffectBindingCountMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Canonical snapshot root does not match the embedded snapshot.
    SnapshotRootMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Checkpoint root does not match its canonical fields.
    CheckpointRootMismatch {
        /// Checkpoint index.
        checkpoint_index: u64,
    },
    /// Same-ledger sequence regressed relative to the predecessor/anchor.
    LedgerRollbackWithinEpoch {
        /// Checkpoint index containing the rollback.
        checkpoint_index: u64,
        /// Ledger epoch.
        ledger_epoch: u64,
        /// Previous sequence frontier.
        before: u64,
        /// Current sequence frontier.
        after: u64,
    },
    /// Snapshot changed without advancing the same-ledger sequence frontier.
    SameSequenceDifferentSnapshot {
        /// Checkpoint index containing the inconsistent successor.
        checkpoint_index: u64,
    },
}

/// Result of deterministic checkpoint-chain verification.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointVerificationReport {
    /// Hard structural/hash-chain contradictions.
    pub violations: Vec<CheckpointViolation>,
}

impl CheckpointVerificationReport {
    /// Whether the supplied snapshots and links form one internally consistent chain.
    ///
    /// This is an integrity/continuity claim, not an authenticity claim.
    pub fn chain_is_consistent(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Evidence that two distinct checkpoint roots claim the same predecessor/index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointFork {
    /// Child checkpoint index at which histories diverge.
    pub checkpoint_index: u64,
    /// Shared predecessor root.
    pub previous_checkpoint_root: Option<Digest32>,
    /// Distinct child roots observed for the same predecessor/index.
    pub competing_roots: Vec<Digest32>,
}

/// Compute the deterministic canonical root of one effect-bound evidence snapshot.
pub fn effect_bound_snapshot_root(
    snapshot: &EffectBoundEvidenceSnapshot,
) -> Result<Digest32, CheckpointBuildError> {
    let normalized = normalized_snapshot(snapshot);
    let value = serde_json::to_value(normalized)?;
    let mut canonical = Vec::new();
    encode_canonical_value(&value, &mut canonical);
    Ok(domain_hash(SNAPSHOT_DOMAIN, &canonical))
}

/// Create the next checkpoint in an evidence lineage.
///
/// Passing a previous externally anchored checkpoint extends that exact lineage.
/// A ledger epoch change is allowed across restart/recovery, but same-epoch
/// sequence rollback is rejected.
pub fn checkpoint_effect_bound_snapshot(
    snapshot: EffectBoundEvidenceSnapshot,
    previous: Option<&EvidenceCheckpoint>,
) -> Result<CheckpointedEffectBoundEvidence, CheckpointBuildError> {
    let snapshot_root = effect_bound_snapshot_root(&snapshot)?;
    let checkpoint_index = match previous {
        Some(previous) => previous
            .checkpoint_index
            .checked_add(1)
            .ok_or(CheckpointBuildError::CheckpointIndexExhausted)?,
        None => 0,
    };

    if let Some(previous) = previous {
        if previous.ledger_epoch == snapshot.base.ledger_epoch {
            if snapshot.base.last_assigned_sequence < previous.last_assigned_sequence {
                return Err(CheckpointBuildError::LedgerRollbackWithinEpoch {
                    ledger_epoch: snapshot.base.ledger_epoch,
                    before: previous.last_assigned_sequence,
                    after: snapshot.base.last_assigned_sequence,
                });
            }
            if snapshot.base.last_assigned_sequence == previous.last_assigned_sequence
                && snapshot_root != previous.snapshot_root
            {
                return Err(CheckpointBuildError::SameSequenceDifferentSnapshot);
            }
        }
    }

    let mut checkpoint = EvidenceCheckpoint {
        schema_version: EVIDENCE_CHECKPOINT_SCHEMA_V1,
        checkpoint_index,
        previous_checkpoint_root: previous.map(|checkpoint| checkpoint.checkpoint_root),
        snapshot_root,
        ledger_epoch: snapshot.base.ledger_epoch,
        last_assigned_sequence: snapshot.base.last_assigned_sequence,
        retained_event_count: snapshot.base.events.len() as u64,
        effect_binding_count: snapshot.observed_effects.len() as u64,
        checkpoint_root: Digest32([0; 32]),
    };
    checkpoint.checkpoint_root = compute_checkpoint_root(&checkpoint);

    Ok(CheckpointedEffectBoundEvidence {
        snapshot,
        checkpoint,
    })
}

/// Verify a complete or externally anchored sequence of checkpointed snapshots.
///
/// With `anchor = None`, the first item must be genesis (`index = 0`, no previous
/// root). With an anchor, the first item must be its exact successor. External
/// signature/witness verification of the anchor is intentionally out of scope.
pub fn verify_checkpoint_chain(
    items: &[CheckpointedEffectBoundEvidence],
    anchor: Option<&EvidenceCheckpoint>,
) -> Result<CheckpointVerificationReport, CheckpointBuildError> {
    let mut report = CheckpointVerificationReport::default();
    let mut previous = anchor.cloned();

    for item in items {
        let checkpoint = &item.checkpoint;
        let expected_index = match &previous {
            Some(previous) => previous
                .checkpoint_index
                .checked_add(1)
                .ok_or(CheckpointBuildError::CheckpointIndexExhausted)?,
            None => 0,
        };
        let expected_previous_root = previous.as_ref().map(|value| value.checkpoint_root);

        if checkpoint.schema_version != EVIDENCE_CHECKPOINT_SCHEMA_V1 {
            report
                .violations
                .push(CheckpointViolation::UnsupportedSchema {
                    checkpoint_index: checkpoint.checkpoint_index,
                    found: checkpoint.schema_version,
                });
        }
        if checkpoint.checkpoint_index != expected_index {
            report
                .violations
                .push(CheckpointViolation::CheckpointIndexMismatch {
                    found: checkpoint.checkpoint_index,
                    expected: expected_index,
                });
        }
        if checkpoint.previous_checkpoint_root != expected_previous_root {
            report
                .violations
                .push(CheckpointViolation::PreviousRootMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                    found: checkpoint.previous_checkpoint_root,
                    expected: expected_previous_root,
                });
        }
        if checkpoint.ledger_epoch != item.snapshot.base.ledger_epoch {
            report
                .violations
                .push(CheckpointViolation::LedgerEpochMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }
        if checkpoint.last_assigned_sequence != item.snapshot.base.last_assigned_sequence {
            report
                .violations
                .push(CheckpointViolation::LastSequenceMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }
        if checkpoint.retained_event_count != item.snapshot.base.events.len() as u64 {
            report
                .violations
                .push(CheckpointViolation::RetainedEventCountMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }
        if checkpoint.effect_binding_count != item.snapshot.observed_effects.len() as u64 {
            report
                .violations
                .push(CheckpointViolation::EffectBindingCountMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }

        let snapshot_root = effect_bound_snapshot_root(&item.snapshot)?;
        if checkpoint.snapshot_root != snapshot_root {
            report
                .violations
                .push(CheckpointViolation::SnapshotRootMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }
        if checkpoint.checkpoint_root != compute_checkpoint_root(checkpoint) {
            report
                .violations
                .push(CheckpointViolation::CheckpointRootMismatch {
                    checkpoint_index: checkpoint.checkpoint_index,
                });
        }

        if let Some(previous) = &previous {
            if previous.ledger_epoch == checkpoint.ledger_epoch {
                if checkpoint.last_assigned_sequence < previous.last_assigned_sequence {
                    report
                        .violations
                        .push(CheckpointViolation::LedgerRollbackWithinEpoch {
                            checkpoint_index: checkpoint.checkpoint_index,
                            ledger_epoch: checkpoint.ledger_epoch,
                            before: previous.last_assigned_sequence,
                            after: checkpoint.last_assigned_sequence,
                        });
                }
                if checkpoint.last_assigned_sequence == previous.last_assigned_sequence
                    && checkpoint.snapshot_root != previous.snapshot_root
                {
                    report
                        .violations
                        .push(CheckpointViolation::SameSequenceDifferentSnapshot {
                            checkpoint_index: checkpoint.checkpoint_index,
                        });
                }
            }
        }

        previous = Some(checkpoint.clone());
    }

    Ok(report)
}

/// Detect competing successor checkpoints over the same predecessor/index.
///
/// Two different roots with the same predecessor and child index are direct
/// fork evidence. A witness need only retain checkpoint metadata/roots to run
/// this comparison; it need not retain private cognitive event contents.
pub fn detect_checkpoint_forks(checkpoints: &[EvidenceCheckpoint]) -> Vec<CheckpointFork> {
    let mut groups = BTreeMap::<(u64, Option<[u8; 32]>), Vec<Digest32>>::new();
    for checkpoint in checkpoints {
        groups
            .entry((
                checkpoint.checkpoint_index,
                checkpoint.previous_checkpoint_root.map(|root| root.0),
            ))
            .or_default()
            .push(checkpoint.checkpoint_root);
    }

    groups
        .into_iter()
        .filter_map(|((checkpoint_index, previous), roots)| {
            let mut unique = roots;
            unique.sort_by_key(|root| root.0);
            unique.dedup_by_key(|root| root.0);
            (unique.len() > 1).then_some(CheckpointFork {
                checkpoint_index,
                previous_checkpoint_root: previous.map(Digest32),
                competing_roots: unique,
            })
        })
        .collect()
}

fn normalized_snapshot(snapshot: &EffectBoundEvidenceSnapshot) -> EffectBoundEvidenceSnapshot {
    let mut normalized = snapshot.clone();
    normalized.base.events.sort_by_key(|event| event.event_id);
    normalized.observed_effects.sort_by(|left, right| {
        left.observed_event_id
            .cmp(&right.observed_event_id)
            .then_with(|| left.effect_digest.0.cmp(&right.effect_digest.0))
    });
    normalized
}

fn compute_checkpoint_root(checkpoint: &EvidenceCheckpoint) -> Digest32 {
    let mut payload = Vec::with_capacity(2 + (8 * 5) + (32 * 3));
    payload.extend_from_slice(&checkpoint.schema_version.to_be_bytes());
    payload.extend_from_slice(&checkpoint.checkpoint_index.to_be_bytes());
    match checkpoint.previous_checkpoint_root {
        Some(root) => {
            payload.push(1);
            payload.extend_from_slice(&root.0);
        }
        None => payload.push(0),
    }
    payload.extend_from_slice(&checkpoint.snapshot_root.0);
    payload.extend_from_slice(&checkpoint.ledger_epoch.to_be_bytes());
    payload.extend_from_slice(&checkpoint.last_assigned_sequence.to_be_bytes());
    payload.extend_from_slice(&checkpoint.retained_event_count.to_be_bytes());
    payload.extend_from_slice(&checkpoint.effect_binding_count.to_be_bytes());
    domain_hash(CHECKPOINT_DOMAIN, &payload)
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    Digest32(hasher.finalize().into())
}

fn encode_canonical_value(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Null => out.push(0),
        Value::Bool(false) => out.push(1),
        Value::Bool(true) => out.push(2),
        Value::Number(number) => {
            out.push(3);
            push_bytes(out, number.to_string().as_bytes());
        }
        Value::String(value) => {
            out.push(4);
            push_bytes(out, value.as_bytes());
        }
        Value::Array(values) => {
            out.push(5);
            out.extend_from_slice(&(values.len() as u64).to_be_bytes());
            for value in values {
                encode_canonical_value(value, out);
            }
        }
        Value::Object(values) => {
            out.push(6);
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            out.extend_from_slice(&(keys.len() as u64).to_be_bytes());
            for key in keys {
                push_bytes(out, key.as_bytes());
                encode_canonical_value(&values[key], out);
            }
        }
    }
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        EvidenceCompleteness, EvidenceLedgerSnapshot, LedgerStats, QualificationManifest,
        SHADOW_EVENT_SCHEMA_V1,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn snapshot(ledger_epoch: u64, last_sequence: u64) -> EffectBoundEvidenceSnapshot {
        EffectBoundEvidenceSnapshot::new(
            EvidenceLedgerSnapshot {
                schema_version: SHADOW_EVENT_SCHEMA_V1,
                ledger_epoch,
                last_assigned_sequence: last_sequence,
                manifest: QualificationManifest::new([], []),
                completeness: EvidenceCompleteness::Complete,
                stats: LedgerStats {
                    assigned_sequences: last_sequence,
                    ..LedgerStats::default()
                },
                events: Vec::new(),
            },
            Vec::new(),
        )
    }

    #[test]
    fn canonical_snapshot_root_is_deterministic() {
        let snapshot = snapshot(1, 0);
        assert_eq!(
            effect_bound_snapshot_root(&snapshot).unwrap(),
            effect_bound_snapshot_root(&snapshot.clone()).unwrap()
        );
    }

    #[test]
    fn effect_binding_changes_snapshot_root() {
        let first = snapshot(1, 1);
        let mut second = first.clone();
        second.observed_effects.push(ObservedEffectBinding {
            observed_event_id: crate::EventId {
                ledger_epoch: 1,
                sequence: 1,
            },
            effect_digest: d(9),
        });
        assert_ne!(
            effect_bound_snapshot_root(&first).unwrap(),
            effect_bound_snapshot_root(&second).unwrap()
        );
    }

    #[test]
    fn anchored_checkpoint_chain_verifies() {
        let first = checkpoint_effect_bound_snapshot(snapshot(1, 0), None).unwrap();
        let second = checkpoint_effect_bound_snapshot(
            snapshot(1, 1),
            Some(&first.checkpoint),
        )
        .unwrap();
        let third = checkpoint_effect_bound_snapshot(
            snapshot(2, 0),
            Some(&second.checkpoint),
        )
        .unwrap();

        let report = verify_checkpoint_chain(
            &[second.clone(), third],
            Some(&first.checkpoint),
        )
        .unwrap();
        assert!(report.chain_is_consistent());
    }

    #[test]
    fn removing_middle_checkpoint_breaks_chain() {
        let first = checkpoint_effect_bound_snapshot(snapshot(1, 0), None).unwrap();
        let second = checkpoint_effect_bound_snapshot(
            snapshot(1, 1),
            Some(&first.checkpoint),
        )
        .unwrap();
        let third = checkpoint_effect_bound_snapshot(
            snapshot(1, 2),
            Some(&second.checkpoint),
        )
        .unwrap();

        let report = verify_checkpoint_chain(&[first, third], None).unwrap();
        assert!(!report.chain_is_consistent());
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            CheckpointViolation::CheckpointIndexMismatch { .. }
                | CheckpointViolation::PreviousRootMismatch { .. }
        )));
    }

    #[test]
    fn same_epoch_rollback_is_rejected_at_construction() {
        let first = checkpoint_effect_bound_snapshot(snapshot(7, 3), None).unwrap();
        let result = checkpoint_effect_bound_snapshot(
            snapshot(7, 2),
            Some(&first.checkpoint),
        );
        assert!(matches!(
            result,
            Err(CheckpointBuildError::LedgerRollbackWithinEpoch { .. })
        ));
    }

    #[test]
    fn competing_successors_are_detected_as_fork_evidence() {
        let first = checkpoint_effect_bound_snapshot(snapshot(1, 0), None).unwrap();
        let left = checkpoint_effect_bound_snapshot(
            snapshot(1, 1),
            Some(&first.checkpoint),
        )
        .unwrap();

        let mut right_snapshot = snapshot(1, 1);
        right_snapshot.observed_effects.push(ObservedEffectBinding {
            observed_event_id: crate::EventId {
                ledger_epoch: 1,
                sequence: 1,
            },
            effect_digest: d(77),
        });
        let right = checkpoint_effect_bound_snapshot(
            right_snapshot,
            Some(&first.checkpoint),
        )
        .unwrap();

        let forks = detect_checkpoint_forks(&[
            first.checkpoint,
            left.checkpoint,
            right.checkpoint,
        ]);
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].checkpoint_index, 1);
        assert_eq!(forks[0].competing_roots.len(), 2);
    }

    #[test]
    fn signing_message_is_domain_separated_and_root_bound() {
        let first = checkpoint_effect_bound_snapshot(snapshot(1, 0), None).unwrap();
        let message = first.checkpoint.signing_message();
        assert!(message.windows(32).any(|window| window == first.checkpoint.checkpoint_root.0));
        assert_ne!(message.as_slice(), first.checkpoint.checkpoint_root.0.as_slice());
    }
}
