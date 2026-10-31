// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent post-plan verification of frozen edit contracts against Scores.
//!
//! This module verifies pitch-at and whole-score transpose operations against
//! concrete source/target Score objects. It does not certify rendered audio,
//! acoustic preservation, renderer configuration, or perceptual acceptance.
//! Those obligations remain explicitly incomplete until independently proven.

use crate::edit_contract::{
    AttributeValue, EditAttribute, EditContractError, FrozenEditContract, ObligationRequirement,
    ObligationScope, SupportedEditOperation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::harmony::Key;
use symthaea_music_theory::pitch::{Pitch, PitchClass};
use symthaea_music_theory::score::ScoreNote;
use symthaea_music_theory::Score;

const SCORE_DOMAIN: &[u8] = b"luminous-dynamics-muse-symbolic-score-v1\0";
const RECEIPT_DOMAIN: &[u8] = b"luminous-dynamics-muse-symbolic-edit-verification-v1\0";
const VERIFICATION_VERSION: &str = "muse-symbolic-edit-verification-v1";
const MICROS_PER_UNIT: i64 = 1_000_000;
const SCORE_EVENT_PREFIX: &str = "score_event_index:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicEditVerdict {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicFailureKind {
    NoOpRequestedChange,
    RequestedPitchNotRealized,
    ProtectedNoteModified,
    EditedNoteAttributesModified,
    NoteCountChanged,
    ScoreMetadataChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicVerificationFailure {
    pub kind: SymbolicFailureKind,
    pub note_indices: Vec<usize>,
    pub obligation_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicEditVerification {
    pub verification_version: String,
    pub contract_sha256: String,
    pub operation_id: String,
    pub source_score_sha256: String,
    pub target_score_sha256: String,
    pub symbolic_verdict: SymbolicEditVerdict,
    pub overall_verdict: SymbolicEditVerdict,
    pub verified_obligation_ids: Vec<String>,
    pub failed_obligation_ids: Vec<String>,
    pub pending_external_obligation_ids: Vec<String>,
    pub failures: Vec<SymbolicVerificationFailure>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicEditVerificationError {
    Contract(EditContractError),
    InvalidContractReceipt,
    InvalidScoreMetadata,
    ScoreSerializationFailure,
    SourceScoreDigestMismatch,
    TargetScoreDigestMismatch,
}

impl std::fmt::Display for SymbolicEditVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "frozen edit contract rejected: {error}"),
            Self::InvalidContractReceipt => write!(f, "frozen edit contract receipt did not verify"),
            Self::InvalidScoreMetadata => write!(f, "source or target Score contains invalid metadata"),
            Self::ScoreSerializationFailure => write!(f, "symbolic Score could not be canonically serialized"),
            Self::SourceScoreDigestMismatch => write!(f, "source Score digest does not match the frozen contract"),
            Self::TargetScoreDigestMismatch => write!(f, "target Score digest does not match the frozen contract"),
        }
    }
}

impl std::error::Error for SymbolicEditVerificationError {}

impl From<EditContractError> for SymbolicEditVerificationError {
    fn from(value: EditContractError) -> Self {
        Self::Contract(value)
    }
}

fn length_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Canonical, domain-separated Score digest used by frozen contract bindings.
pub fn symbolic_score_sha256(
    score: &Score,
) -> Result<String, SymbolicEditVerificationError> {
    let bytes = serde_json::to_vec(score)
        .map_err(|_| SymbolicEditVerificationError::ScoreSerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(SCORE_DOMAIN);
    length_prefixed(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn receipt_sha256(
    report: &SymbolicEditVerification,
) -> Result<String, SymbolicEditVerificationError> {
    let mut unsigned = report.clone();
    unsigned.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| SymbolicEditVerificationError::ScoreSerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DOMAIN);
    length_prefixed(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_score_event_index(event_id: &str) -> Option<usize> {
    let suffix = event_id.strip_prefix(SCORE_EVENT_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index.to_string() == suffix).then_some(index)
}

fn obligation_ids_for(
    contract: &FrozenEditContract,
    attribute: EditAttribute,
) -> Vec<String> {
    contract
        .obligations
        .iter()
        .filter(|obligation| obligation.attribute == attribute)
        .map(|obligation| obligation.obligation_id.clone())
        .collect()
}

fn push_failure(
    failures: &mut Vec<SymbolicVerificationFailure>,
    kind: SymbolicFailureKind,
    mut note_indices: Vec<usize>,
    mut obligation_ids: Vec<String>,
) {
    note_indices.sort_unstable();
    note_indices.dedup();
    obligation_ids.sort();
    obligation_ids.dedup();
    failures.push(SymbolicVerificationFailure {
        kind,
        note_indices,
        obligation_ids,
    });
}

fn non_pitch_note_equal(source: ScoreNote, target: ScoreNote) -> bool {
    let mut expected = target;
    expected.pitch = source.pitch;
    expected == source
}

fn score_metadata_valid(score: &Score) -> bool {
    score.tempo_bpm.is_finite()
        && score.tempo_bpm > 0.0
        && score.total_beats.beats().is_finite()
        && score.total_beats.beats() >= 0.0
        && score.notes.iter().all(|note| {
            note.velocity.is_finite()
                && note.section_intensity.is_finite()
                && note.onset.beats().is_finite()
                && note.duration.beats().is_finite()
                && note.onset.beats() >= 0.0
                && note.duration.beats() > 0.0
        })
}

fn score_metadata_failure_ids(
    source: &Score,
    target: &Score,
    contract: &FrozenEditContract,
    key_is_expected: bool,
) -> Vec<String> {
    let mut affected = Vec::new();
    if source.tempo_bpm.to_bits() != target.tempo_bpm.to_bits() {
        affected.extend(obligation_ids_for(contract, EditAttribute::Onset));
    }
    if source.meter != target.meter || source.total_beats != target.total_beats {
        affected.extend(obligation_ids_for(contract, EditAttribute::Duration));
    }
    if !key_is_expected {
        affected.extend(obligation_ids_for(contract, EditAttribute::Pitch));
        affected.extend(obligation_ids_for(contract, EditAttribute::NeighboringContent));
    }
    affected.sort();
    affected.dedup();
    affected
}

fn expected_pitch_at(contract: &FrozenEditContract) -> Option<(usize, u8)> {
    let obligation = contract.obligations.iter().find(|obligation| {
        obligation.attribute == EditAttribute::Pitch
    })?;
    let ObligationScope::Event { event_id } = &obligation.scope else {
        return None;
    };
    let index = parse_score_event_index(event_id)?;
    let ObligationRequirement::ChangeTo {
        expected: AttributeValue::Numeric {
            value_microunits,
            unit,
        },
    } = &obligation.requirement else {
        return None;
    };
    if unit != "midi_note" || *value_microunits % MICROS_PER_UNIT != 0 {
        return None;
    }
    let midi = *value_microunits / MICROS_PER_UNIT;
    (0..=127).contains(&midi).then_some((index, midi as u8))
}

fn expected_transpose_delta(contract: &FrozenEditContract) -> Option<i32> {
    let obligation = contract.obligations.iter().find(|obligation| {
        obligation.attribute == EditAttribute::Pitch
    })?;
    if !matches!(obligation.scope, ObligationScope::ScoreWide) {
        return None;
    }
    let ObligationRequirement::ChangeBy {
        delta: AttributeValue::Numeric {
            value_microunits,
            unit,
        },
    } = &obligation.requirement else {
        return None;
    };
    if unit != "semitone" || *value_microunits == 0 || *value_microunits % MICROS_PER_UNIT != 0 {
        return None;
    }
    i32::try_from(*value_microunits / MICROS_PER_UNIT).ok()
}

fn verify_pitch_at(
    source: &Score,
    target: &Score,
    contract: &FrozenEditContract,
    failures: &mut Vec<SymbolicVerificationFailure>,
) {
    let pitch_ids = obligation_ids_for(contract, EditAttribute::Pitch);
    let onset_ids = obligation_ids_for(contract, EditAttribute::Onset);
    let duration_ids = obligation_ids_for(contract, EditAttribute::Duration);
    let neighbor_ids = obligation_ids_for(contract, EditAttribute::NeighboringContent);

    let metadata_failures = score_metadata_failure_ids(source, target, contract, source.key == target.key);
    if !metadata_failures.is_empty() {
        push_failure(
            failures,
            SymbolicFailureKind::ScoreMetadataChanged,
            Vec::new(),
            metadata_failures,
        );
    }

    let source_events = source.events();
    let target_events = target.events();
    if source_events.len() != target_events.len() {
        let mut affected = pitch_ids.clone();
        affected.extend(onset_ids.clone());
        affected.extend(duration_ids.clone());
        affected.extend(neighbor_ids.clone());
        push_failure(
            failures,
            SymbolicFailureKind::NoteCountChanged,
            Vec::new(),
            affected,
        );
        return;
    }

    let Some((selected_index, expected_midi)) = expected_pitch_at(contract) else {
        push_failure(
            failures,
            SymbolicFailureKind::RequestedPitchNotRealized,
            Vec::new(),
            pitch_ids,
        );
        return;
    };
    if selected_index >= source_events.len() {
        push_failure(
            failures,
            SymbolicFailureKind::RequestedPitchNotRealized,
            vec![selected_index],
            pitch_ids,
        );
        return;
    }

    let source_pitch = source_events[selected_index].pitch.midi();
    let actual_pitch = target_events[selected_index].pitch.midi();
    if source_pitch == expected_midi {
        push_failure(
            failures,
            SymbolicFailureKind::NoOpRequestedChange,
            vec![selected_index],
            pitch_ids.clone(),
        );
    } else if actual_pitch != expected_midi {
        push_failure(
            failures,
            SymbolicFailureKind::RequestedPitchNotRealized,
            vec![selected_index],
            pitch_ids.clone(),
        );
    }

    for index in 0..source_events.len() {
        let before = source_events[index];
        let after = target_events[index];
        if index == selected_index {
            if !non_pitch_note_equal(before, after) {
                let mut affected = pitch_ids.clone();
                affected.extend(onset_ids.clone());
                affected.extend(duration_ids.clone());
                affected.extend(obligation_ids_for(contract, EditAttribute::RmsEnvelope));
                push_failure(
                    failures,
                    SymbolicFailureKind::EditedNoteAttributesModified,
                    vec![index],
                    affected,
                );
            }
        } else if before != after {
            let mut affected = neighbor_ids.clone();
            if before.onset != after.onset {
                affected.extend(onset_ids.clone());
            }
            if before.duration != after.duration {
                affected.extend(duration_ids.clone());
            }
            push_failure(
                failures,
                SymbolicFailureKind::ProtectedNoteModified,
                vec![index],
                affected,
            );
        }
    }
}

fn verify_transpose(
    source: &Score,
    target: &Score,
    contract: &FrozenEditContract,
    failures: &mut Vec<SymbolicVerificationFailure>,
) {
    let pitch_ids = obligation_ids_for(contract, EditAttribute::Pitch);
    let onset_ids = obligation_ids_for(contract, EditAttribute::Onset);
    let duration_ids = obligation_ids_for(contract, EditAttribute::Duration);
    let neighbor_ids = obligation_ids_for(contract, EditAttribute::NeighboringContent);
    let Some(delta) = expected_transpose_delta(contract) else {
        push_failure(
            failures,
            SymbolicFailureKind::RequestedPitchNotRealized,
            Vec::new(),
            pitch_ids,
        );
        return;
    };

    let expected_tonic = (source.key.tonic.value() as i32 + delta).rem_euclid(12) as u8;
    let key_matches = target.key.tonic.value() == expected_tonic
        && target.key.tonality == source.key.tonality;
    let metadata_failures = score_metadata_failure_ids(source, target, contract, key_matches);
    if !metadata_failures.is_empty() {
        push_failure(
            failures,
            SymbolicFailureKind::ScoreMetadataChanged,
            Vec::new(),
            metadata_failures,
        );
    }

    let source_events = source.events();
    let target_events = target.events();
    if source_events.len() != target_events.len() {
        let mut affected = pitch_ids.clone();
        affected.extend(onset_ids.clone());
        affected.extend(duration_ids.clone());
        affected.extend(neighbor_ids.clone());
        push_failure(
            failures,
            SymbolicFailureKind::NoteCountChanged,
            Vec::new(),
            affected,
        );
        return;
    }

    for (index, (before, after)) in source_events.iter().zip(target_events.iter()).enumerate() {
        let expected_raw = before.pitch.midi() as i32 + delta;
        let pitch_matches = (0..=127).contains(&expected_raw)
            && after.pitch == Pitch::from_midi(expected_raw as u8);
        if !pitch_matches {
            push_failure(
                failures,
                SymbolicFailureKind::RequestedPitchNotRealized,
                vec![index],
                pitch_ids.clone(),
            );
        }
        if !non_pitch_note_equal(*before, *after) {
            let mut affected = neighbor_ids.clone();
            if before.onset != after.onset {
                affected.extend(onset_ids.clone());
            }
            if before.duration != after.duration {
                affected.extend(duration_ids.clone());
            }
            push_failure(
                failures,
                SymbolicFailureKind::ProtectedNoteModified,
                vec![index],
                affected,
            );
        }
    }
}

fn build_report(
    contract: &FrozenEditContract,
    source_digest: String,
    target_digest: String,
    mut failures: Vec<SymbolicVerificationFailure>,
) -> Result<SymbolicEditVerification, SymbolicEditVerificationError> {
    failures.sort_by(|left, right| {
        (left.kind, &left.note_indices, &left.obligation_ids)
            .cmp(&(right.kind, &right.note_indices, &right.obligation_ids))
    });
    let mut failed_ids = failures
        .iter()
        .flat_map(|failure| failure.obligation_ids.iter().cloned())
        .collect::<Vec<_>>();
    failed_ids.sort();
    failed_ids.dedup();

    let mut verified_ids = Vec::new();
    for attribute in [EditAttribute::Pitch, EditAttribute::Onset, EditAttribute::Duration] {
        for id in obligation_ids_for(contract, attribute) {
            if !failed_ids.contains(&id) {
                verified_ids.push(id);
            }
        }
    }
    verified_ids.sort();
    verified_ids.dedup();

    let mut pending_ids = Vec::new();
    for attribute in [
        EditAttribute::RmsEnvelope,
        EditAttribute::NeighboringContent,
        EditAttribute::RendererConfig,
    ] {
        for id in obligation_ids_for(contract, attribute) {
            if !failed_ids.contains(&id) {
                pending_ids.push(id);
            }
        }
    }
    pending_ids.sort();
    pending_ids.dedup();

    let symbolic_verdict = if failed_ids.is_empty() {
        SymbolicEditVerdict::Pass
    } else {
        SymbolicEditVerdict::Fail
    };
    let overall_verdict = if symbolic_verdict == SymbolicEditVerdict::Fail {
        SymbolicEditVerdict::Fail
    } else if !pending_ids.is_empty() {
        SymbolicEditVerdict::Incomplete
    } else {
        SymbolicEditVerdict::Pass
    };
    let mut report = SymbolicEditVerification {
        verification_version: VERIFICATION_VERSION.to_owned(),
        contract_sha256: contract.contract_sha256.clone(),
        operation_id: contract.operation_id.clone(),
        source_score_sha256: source_digest,
        target_score_sha256: target_digest,
        symbolic_verdict,
        overall_verdict,
        verified_obligation_ids: verified_ids,
        failed_obligation_ids: failed_ids,
        pending_external_obligation_ids: pending_ids,
        failures,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = receipt_sha256(&report)?;
    Ok(report)
}

/// Verify concrete source/target Scores against a sealed, supported edit contract.
/// Score digests are recomputed; planners cannot choose them by fiat.
pub fn verify_symbolic_edit_contract(
    source: &Score,
    target: &Score,
    contract: &FrozenEditContract,
) -> Result<SymbolicEditVerification, SymbolicEditVerificationError> {
    if !contract.verify()? {
        return Err(SymbolicEditVerificationError::InvalidContractReceipt);
    }
    let operation = contract.validate_supported_operation()?;
    if !score_metadata_valid(source) || !score_metadata_valid(target) {
        return Err(SymbolicEditVerificationError::InvalidScoreMetadata);
    }
    let source_digest = symbolic_score_sha256(source)?;
    let target_digest = symbolic_score_sha256(target)?;
    if source_digest != contract.source_score_sha256 {
        return Err(SymbolicEditVerificationError::SourceScoreDigestMismatch);
    }
    if target_digest != contract.target_score_sha256 {
        return Err(SymbolicEditVerificationError::TargetScoreDigestMismatch);
    }

    let mut failures = Vec::new();
    match operation {
        SupportedEditOperation::PitchAt => verify_pitch_at(source, target, contract, &mut failures),
        SupportedEditOperation::Transpose => verify_transpose(source, target, contract, &mut failures),
    }
    build_report(contract, source_digest, target_digest, failures)
}

/// Replay verification binds the report to the original source, target and contract.
pub fn verify_symbolic_edit_verification(
    source: &Score,
    target: &Score,
    contract: &FrozenEditContract,
    supplied: &SymbolicEditVerification,
) -> bool {
    verify_symbolic_edit_contract(source, target, contract)
        .is_ok_and(|expected| expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit_contract::{EditObligation, QuantizedTolerance};
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::rhythm::Duration;
    use symthaea_music_theory::score::{Emphasis, PartId, ScoreNote, VoiceRole};

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn score(pitches: &[u8]) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (index, midi) in pitches.iter().copied().enumerate() {
            score.notes.push(ScoreNote {
                part: PartId(index as u16),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(index as i64, 1),
                duration: Duration::quarter(),
                velocity: 0.7 + index as f32 * 0.05,
                role: if index == 0 { VoiceRole::Melody } else { VoiceRole::Harmony },
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score.total_beats = Duration::new(pitches.len() as i64 + 1, 1);
        score
    }

    fn preservation(
        id: &str,
        attribute: EditAttribute,
        scope: ObligationScope,
        comparator: &str,
        unit: &str,
        tolerance: u64,
    ) -> EditObligation {
        EditObligation {
            obligation_id: id.to_owned(),
            attribute,
            scope,
            requirement: ObligationRequirement::PreserveWithin {
                comparator_id: comparator.to_owned(),
                tolerance: QuantizedTolerance {
                    value_microunits: tolerance,
                    unit: unit.to_owned(),
                },
            },
            critical: true,
        }
    }

    fn common_obligations(selected_index: Option<usize>) -> Vec<EditObligation> {
        let envelope_scope = selected_index.map_or(ObligationScope::ScoreWide, |index| {
            ObligationScope::Event {
                event_id: format!("{SCORE_EVENT_PREFIX}{index}"),
            }
        });
        vec![
            preservation(
                "preserve-onsets",
                EditAttribute::Onset,
                ObligationScope::ScoreWide,
                "onset-preservation-v1",
                "seconds",
                10_000,
            ),
            preservation(
                "preserve-durations",
                EditAttribute::Duration,
                ObligationScope::ScoreWide,
                "score-duration-exact-v1",
                "beats",
                0,
            ),
            preservation(
                "preserve-edited-envelope",
                EditAttribute::RmsEnvelope,
                envelope_scope,
                "rms-envelope-delta-v1",
                "db",
                500_000,
            ),
            preservation(
                "preserve-protected-rms",
                EditAttribute::NeighboringContent,
                ObligationScope::ProtectedRemainder,
                "protected-region-rms-v1",
                "db",
                500_000,
            ),
            preservation(
                "preserve-protected-spectrum",
                EditAttribute::NeighboringContent,
                ObligationScope::ProtectedRemainder,
                "protected-region-spectral-v1",
                "normalized_distance",
                80_000,
            ),
            EditObligation {
                obligation_id: "bind-renderer".to_owned(),
                attribute: EditAttribute::RendererConfig,
                scope: ObligationScope::ScoreWide,
                requirement: ObligationRequirement::BindDigest {
                    expected_sha256: digest('e'),
                },
                critical: true,
            },
        ]
    }

    fn pitch_at_contract(
        source: &Score,
        target: &Score,
        index: usize,
        midi: u8,
    ) -> FrozenEditContract {
        let mut obligations = common_obligations(Some(index));
        obligations.push(EditObligation {
            obligation_id: "change-selected-pitch".to_owned(),
            attribute: EditAttribute::Pitch,
            scope: ObligationScope::Event {
                event_id: format!("{SCORE_EVENT_PREFIX}{index}"),
            },
            requirement: ObligationRequirement::ChangeTo {
                expected: AttributeValue::Numeric {
                    value_microunits: midi as i64 * MICROS_PER_UNIT,
                    unit: "midi_note".to_owned(),
                },
            },
            critical: true,
        });
        FrozenEditContract::new(
            "task-pitch-at".to_owned(),
            "pitch-at".to_owned(),
            digest('a'),
            symbolic_score_sha256(source).unwrap(),
            symbolic_score_sha256(target).unwrap(),
            "symbolic-profile-v1".to_owned(),
            digest('d'),
            obligations,
        )
        .seal()
        .unwrap()
    }

    fn transpose_contract(source: &Score, target: &Score, semitones: i32) -> FrozenEditContract {
        let mut obligations = common_obligations(None);
        obligations.push(EditObligation {
            obligation_id: "transpose-whole-score".to_owned(),
            attribute: EditAttribute::Pitch,
            scope: ObligationScope::ScoreWide,
            requirement: ObligationRequirement::ChangeBy {
                delta: AttributeValue::Numeric {
                    value_microunits: semitones as i64 * MICROS_PER_UNIT,
                    unit: "semitone".to_owned(),
                },
            },
            critical: true,
        });
        FrozenEditContract::new(
            "task-transpose".to_owned(),
            "transpose".to_owned(),
            digest('a'),
            symbolic_score_sha256(source).unwrap(),
            symbolic_score_sha256(target).unwrap(),
            "symbolic-profile-v1".to_owned(),
            digest('d'),
            obligations,
        )
        .seal()
        .unwrap()
    }

    #[test]
    fn valid_pitch_at_is_symbolically_verified_but_audio_remains_incomplete() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.symbolic_verdict, SymbolicEditVerdict::Pass);
        assert_eq!(report.overall_verdict, SymbolicEditVerdict::Incomplete);
        assert!(report.failed_obligation_ids.is_empty());
        assert_eq!(report.pending_external_obligation_ids.len(), 4);
        assert!(verify_symbolic_edit_verification(&source, &target, &contract, &report));
    }


    #[test]
    fn pitch_at_event_index_uses_onset_sorted_render_event_stream() {
        let mut source = score(&[60, 64, 67]);
        source.notes.swap(0, 2);
        let mut target = source.clone();
        let event_to_edit = target
            .notes
            .iter_mut()
            .find(|note| note.onset == Duration::new(1, 1))
            .unwrap();
        event_to_edit.pitch = Pitch::from_midi(69);
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Incomplete);
        assert!(report.failed_obligation_ids.is_empty());
    }

    #[test]
    fn no_op_pitch_at_is_rejected_even_when_target_hash_matches() {
        let source = score(&[60, 64, 67]);
        let target = source.clone();
        let contract = pitch_at_contract(&source, &target, 1, 64);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.symbolic_verdict, SymbolicEditVerdict::Fail);
        assert_eq!(report.overall_verdict, SymbolicEditVerdict::Fail);
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::NoOpRequestedChange));
    }

    #[test]
    fn wrong_pitch_and_wrong_event_are_rejected() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[2].pitch = Pitch::from_midi(69);
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Fail);
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::RequestedPitchNotRealized));
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::ProtectedNoteModified));
    }

    #[test]
    fn collateral_note_edit_is_rejected() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        target.notes[0].velocity = 0.99;
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Fail);
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::ProtectedNoteModified));
    }

    #[test]
    fn selected_note_timing_change_and_score_metadata_drift_fail() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        target.notes[1].duration = Duration::half();
        target.tempo_bpm = 121.0;
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Fail);
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::EditedNoteAttributesModified));
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::ScoreMetadataChanged));
    }

    #[test]
    fn valid_whole_score_transpose_updates_every_pitch_and_key() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        for note in &mut target.notes {
            note.pitch = Pitch::from_midi(note.pitch.midi() + 2);
        }
        target.key = Key::major(PitchClass::new(2));
        let contract = transpose_contract(&source, &target, 2);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Incomplete);
        assert!(report.failed_obligation_ids.is_empty());
    }

    #[test]
    fn partial_transpose_and_pitch_range_clamping_fail() {
        let source = score(&[60, 64, 127]);
        let mut target = source.clone();
        target.notes[0].pitch = Pitch::from_midi(62);
        target.notes[1].pitch = Pitch::from_midi(66);
        target.notes[2].pitch = Pitch::from_midi(127);
        target.key = Key::major(PitchClass::new(2));
        let contract = transpose_contract(&source, &target, 2);
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Fail);
        assert!(report.failures.iter().any(|failure| failure.kind == SymbolicFailureKind::RequestedPitchNotRealized));
    }

    #[test]
    fn source_and_target_score_digest_mismatches_are_rejected() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let mut changed_source = source.clone();
        changed_source.tempo_bpm = 121.0;
        assert_eq!(
            verify_symbolic_edit_contract(&changed_source, &target, &contract),
            Err(SymbolicEditVerificationError::SourceScoreDigestMismatch),
        );
        let mut changed_target = target.clone();
        changed_target.tempo_bpm = 121.0;
        assert_eq!(
            verify_symbolic_edit_contract(&source, &changed_target, &contract),
            Err(SymbolicEditVerificationError::TargetScoreDigestMismatch),
        );
    }

    #[test]
    fn replay_verification_rejects_tampered_symbolic_receipt() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        let contract = pitch_at_contract(&source, &target, 1, 69);
        let mut report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert!(verify_symbolic_edit_verification(&source, &target, &contract, &report));
        report.pending_external_obligation_ids.clear();
        assert!(!verify_symbolic_edit_verification(&source, &target, &contract, &report));
    }

    #[test]
    fn noncanonical_event_ids_are_rejected_fail_closed() {
        let source = score(&[60, 64, 67]);
        let mut target = source.clone();
        target.notes[1].pitch = Pitch::from_midi(69);
        let mut contract = pitch_at_contract(&source, &target, 1, 69);
        if let ObligationScope::Event { event_id } = &mut contract.obligations
            .iter_mut()
            .find(|obligation| obligation.attribute == EditAttribute::Pitch)
            .unwrap()
            .scope
        {
            *event_id = "score_event_index:01".to_owned();
        }
        contract = contract.seal().unwrap();
        let report = verify_symbolic_edit_contract(&source, &target, &contract).unwrap();
        assert_eq!(report.verdict, SymbolicEditVerdict::Fail);
    }
}
