// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proof-carrying, source-bound symbolic score patch receipts.
//!
//! v1 is intentionally limited to same-cardinality, in-place note edits. Event
//! identity is derived from the immutable source-score digest and the event's
//! original vector index. Insertions, deletions, reordering, score-metadata
//! changes and out-of-scope edits fail closed until they have a richer lineage
//! model.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::Score;

pub const SCORE_PATCH_SCHEMA_V1: &str = "mel-score-patch-receipt-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScorePatchErrorV1 {
    SerializationFailed,
    EmptyTargetScope,
    DuplicateTargetIndex {
        index: usize,
    },
    TargetIndexOutOfRange {
        index: usize,
        event_count: usize,
    },
    EventCountChanged {
        source_count: usize,
        result_count: usize,
    },
    ScoreMetadataChanged,
    ProtectedEventChanged {
        index: usize,
    },
    ReceiptMismatch,
}

/// Stable identity for one event in an immutable source-score revision.
///
/// This identity remains stable across v1 pitch/velocity/annotation changes as
/// long as the source score and original event index remain fixed. It does not
/// claim cross-revision identity after event insertion, deletion or reordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventIdentityV1 {
    pub source_score_sha256: String,
    pub source_index: usize,
    pub event_id_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventChangeV1 {
    pub event: ScoreEventIdentityV1,
    pub before_note_sha256: String,
    pub after_note_sha256: String,
}

/// A deterministic, independently recomputable receipt for a score mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScorePatchReceiptV1 {
    pub schema: String,
    pub source_score_sha256: String,
    pub result_score_sha256: String,
    pub source_event_count: usize,
    /// Sorted, unique source event indices authorized for this transaction.
    pub target_event_indices: Vec<usize>,
    /// Only events whose serialized ScoreNote actually changed.
    pub changed_events: Vec<ScoreEventChangeV1>,
    /// Digest of every non-target event, including its original position.
    pub protected_complement_sha256: String,
    /// Digest over every receipt field other than this digest.
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ReceiptHashPayloadV1<'a> {
    schema: &'a str,
    source_score_sha256: &'a str,
    result_score_sha256: &'a str,
    source_event_count: usize,
    target_event_indices: &'a [usize],
    changed_events: &'a [ScoreEventChangeV1],
    protected_complement_sha256: &'a str,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn json_sha256<T: Serialize>(value: &T) -> Result<String, ScorePatchErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|_| ScorePatchErrorV1::SerializationFailed)?;
    Ok(sha256(&bytes))
}

fn event_id_sha256(source_score_sha256: &str, source_index: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics:mel-score-event-id:v1\0");
    hasher.update(source_score_sha256.as_bytes());
    hasher.update((source_index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn score_metadata_equal(source: &Score, result: &Score) -> bool {
    source.key == result.key
        && source.tempo_bpm == result.tempo_bpm
        && source.meter == result.meter
        && source.total_beats == result.total_beats
}

fn receipt_hash(receipt: &ScorePatchReceiptV1) -> Result<String, ScorePatchErrorV1> {
    let payload = ReceiptHashPayloadV1 {
        schema: &receipt.schema,
        source_score_sha256: &receipt.source_score_sha256,
        result_score_sha256: &receipt.result_score_sha256,
        source_event_count: receipt.source_event_count,
        target_event_indices: &receipt.target_event_indices,
        changed_events: &receipt.changed_events,
        protected_complement_sha256: &receipt.protected_complement_sha256,
    };
    json_sha256(&payload)
}

/// Build a receipt for an in-place score edit over an explicit source-index
/// target set. The function rejects every change outside that set.
pub fn create_score_patch_receipt(
    source: &Score,
    result: &Score,
    target_event_indices: &[usize],
) -> Result<ScorePatchReceiptV1, ScorePatchErrorV1> {
    if target_event_indices.is_empty() {
        return Err(ScorePatchErrorV1::EmptyTargetScope);
    }
    if source.notes.len() != result.notes.len() {
        return Err(ScorePatchErrorV1::EventCountChanged {
            source_count: source.notes.len(),
            result_count: result.notes.len(),
        });
    }
    if !score_metadata_equal(source, result) {
        return Err(ScorePatchErrorV1::ScoreMetadataChanged);
    }

    let mut targets = target_event_indices.to_vec();
    targets.sort_unstable();
    for pair in targets.windows(2) {
        if pair[0] == pair[1] {
            return Err(ScorePatchErrorV1::DuplicateTargetIndex { index: pair[0] });
        }
    }
    for index in &targets {
        if *index >= source.notes.len() {
            return Err(ScorePatchErrorV1::TargetIndexOutOfRange {
                index: *index,
                event_count: source.notes.len(),
            });
        }
    }

    let target_mask: Vec<bool> = (0..source.notes.len())
        .map(|index| targets.binary_search(&index).is_ok())
        .collect();

    let mut protected_complement = Vec::new();
    for (index, (before, after)) in source.notes.iter().zip(&result.notes).enumerate() {
        if target_mask[index] {
            continue;
        }
        if before != after {
            return Err(ScorePatchErrorV1::ProtectedEventChanged { index });
        }
        protected_complement.push((index, *before));
    }

    let source_score_sha256 = json_sha256(source)?;
    let result_score_sha256 = json_sha256(result)?;
    let protected_complement_sha256 = json_sha256(&protected_complement)?;

    let mut changed_events = Vec::new();
    for index in &targets {
        let before = source.notes[*index];
        let after = result.notes[*index];
        if before == after {
            continue;
        }
        let event = ScoreEventIdentityV1 {
            source_score_sha256: source_score_sha256.clone(),
            source_index: *index,
            event_id_sha256: event_id_sha256(&source_score_sha256, *index),
        };
        changed_events.push(ScoreEventChangeV1 {
            event,
            before_note_sha256: json_sha256(&before)?,
            after_note_sha256: json_sha256(&after)?,
        });
    }

    let mut receipt = ScorePatchReceiptV1 {
        schema: SCORE_PATCH_SCHEMA_V1.to_owned(),
        source_score_sha256,
        result_score_sha256,
        source_event_count: source.notes.len(),
        target_event_indices: targets,
        changed_events,
        protected_complement_sha256,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_hash(&receipt)?;
    Ok(receipt)
}

/// Independently recompute the complete transaction receipt from source and
/// result Scores. A stale or modified receipt is rejected.
pub fn verify_score_patch_receipt(
    source: &Score,
    result: &Score,
    receipt: &ScorePatchReceiptV1,
) -> Result<(), ScorePatchErrorV1> {
    if receipt.schema != SCORE_PATCH_SCHEMA_V1 {
        return Err(ScorePatchErrorV1::ReceiptMismatch);
    }
    let recomputed = create_score_patch_receipt(source, result, &receipt.target_event_indices)?;
    if &recomputed == receipt {
        Ok(())
    } else {
        Err(ScorePatchErrorV1::ReceiptMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        Duration, Emphasis, Key, Pitch, PitchClass, Score, ScoreNote, VoiceRole, score::PartId,
    };

    fn fixture_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (index, midi) in [60_u8, 64, 67].into_iter().enumerate() {
            score.push(ScoreNote {
                part: PartId::UNASSIGNED,
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(index as i64, 1),
                duration: Duration::quarter(),
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    #[test]
    fn one_target_pitch_change_produces_a_replayable_receipt() {
        let source = fixture_score();
        let mut result = source.clone();
        result.notes[1].pitch = result.notes[1].pitch.transpose(2);

        let receipt = create_score_patch_receipt(&source, &result, &[1]).unwrap();
        verify_score_patch_receipt(&source, &result, &receipt).unwrap();

        assert_eq!(receipt.source_event_count, 3);
        assert_eq!(receipt.target_event_indices, vec![1]);
        assert_eq!(receipt.changed_events.len(), 1);
        assert_eq!(receipt.changed_events[0].event.source_index, 1);
        assert_eq!(
            receipt.changed_events[0].event.event_id_sha256,
            event_id_sha256(&receipt.source_score_sha256, 1)
        );
        assert_ne!(
            receipt.changed_events[0].before_note_sha256,
            receipt.changed_events[0].after_note_sha256
        );
    }

    #[test]
    fn protected_complement_change_is_rejected() {
        let source = fixture_score();
        let mut result = source.clone();
        result.notes[0].pitch = result.notes[0].pitch.transpose(1);

        assert_eq!(
            create_score_patch_receipt(&source, &result, &[1]),
            Err(ScorePatchErrorV1::ProtectedEventChanged { index: 0 })
        );
    }

    #[test]
    fn note_insertion_or_deletion_is_rejected_until_lineage_support_exists() {
        let source = fixture_score();
        let mut result = source.clone();
        result.notes.pop();

        assert_eq!(
            create_score_patch_receipt(&source, &result, &[1]),
            Err(ScorePatchErrorV1::EventCountChanged {
                source_count: 3,
                result_count: 2,
            })
        );
    }

    #[test]
    fn score_level_tempo_or_form_change_is_rejected_by_note_only_patch() {
        let source = fixture_score();
        let mut result = source.clone();
        result.tempo_bpm = 132.0;

        assert_eq!(
            create_score_patch_receipt(&source, &result, &[1]),
            Err(ScorePatchErrorV1::ScoreMetadataChanged)
        );
    }

    #[test]
    fn empty_duplicate_and_out_of_range_scopes_are_rejected() {
        let source = fixture_score();
        assert_eq!(
            create_score_patch_receipt(&source, &source, &[]),
            Err(ScorePatchErrorV1::EmptyTargetScope)
        );
        assert_eq!(
            create_score_patch_receipt(&source, &source, &[1, 1]),
            Err(ScorePatchErrorV1::DuplicateTargetIndex { index: 1 })
        );
        assert_eq!(
            create_score_patch_receipt(&source, &source, &[3]),
            Err(ScorePatchErrorV1::TargetIndexOutOfRange {
                index: 3,
                event_count: 3,
            })
        );
    }

    #[test]
    fn no_op_patch_is_explicit_and_deterministic() {
        let source = fixture_score();
        let first = create_score_patch_receipt(&source, &source, &[1]).unwrap();
        let second = create_score_patch_receipt(&source, &source, &[1]).unwrap();

        assert_eq!(first, second);
        assert!(first.changed_events.is_empty());
        verify_score_patch_receipt(&source, &source, &first).unwrap();
    }

    #[test]
    fn altered_receipt_cannot_verify_against_unchanged_scores() {
        let source = fixture_score();
        let mut result = source.clone();
        result.notes[1].pitch = result.notes[1].pitch.transpose(2);
        let mut receipt = create_score_patch_receipt(&source, &result, &[1]).unwrap();
        receipt.protected_complement_sha256 = "0".repeat(64);

        assert_eq!(
            verify_score_patch_receipt(&source, &result, &receipt),
            Err(ScorePatchErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn target_indices_are_canonicalized_before_receipt_hashing() {
        let source = fixture_score();
        let mut result = source.clone();
        result.notes[1].pitch = result.notes[1].pitch.transpose(2);
        result.notes[2].pitch = result.notes[2].pitch.transpose(-1);

        let unsorted = create_score_patch_receipt(&source, &result, &[2, 1]).unwrap();
        let sorted = create_score_patch_receipt(&source, &result, &[1, 2]).unwrap();
        assert_eq!(unsorted, sorted);
        assert_eq!(unsorted.target_event_indices, vec![1, 2]);
    }
}
