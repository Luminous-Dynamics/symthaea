// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compositional, auditable verification of symbolic motif transformations.
//!
//! Unlike the legacy single-label motif-return API, this API does not choose
//! one transformation label. A claim describes pitch orientation, transposition,
//! onset scaling, duration scaling, and a source window independently. All
//! required axes must pass; good pitch similarity cannot compensate for a
//! failed timing contract.
//!
//! This first version uses bounded monotonic edit alignment. It is a symbolic
//! reference implementation, not an audio/perception metric and not a claim
//! that arbitrary ornamentation is understood.

use serde::{Deserialize, Serialize};

/// Versioned contract for compositional motif transformation verification.
pub const COMPOSITIONAL_MOTIF_CONTRACT_VERSION: &str = "compositional-motif-v1";

/// Hard cap on dynamic-programming cells to bound untrusted-input work and memory.
pub const MAX_ALIGNMENT_CELLS: usize = 1_000_000;

/// Whether melodic intervals are preserved or sign-inverted around the first
/// selected pitch, before the declared semitone translation is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PitchOrientation {
    Preserve,
    Invert,
}

/// Which source events are claimed to participate in the return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    Full,
    ContiguousFragment,
}

/// Explicit source-window selection. A fragment must be a proper contiguous
/// source subsequence of at least three events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSelection {
    pub mode: SelectionMode,
    pub source_start_index: usize,
    pub note_count: usize,
}

/// A product of independently verifiable symbolic transformations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositionalTransform {
    pub pitch_orientation: PitchOrientation,
    pub transpose_semitones: i32,
    pub onset_scale: f64,
    pub duration_scale: f64,
    pub selection: SourceSelection,
}

/// Monophonic symbolic event represented in MIDI semitones and quarter-note beats.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SymbolicEvent {
    pub pitch_midi: i32,
    pub onset_beats: f64,
    pub duration_beats: f64,
}

/// Explicit tolerances and edit budget. Defaults are conservative initial values,
/// not empirically calibrated thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositionalVerificationConfig {
    pub pitch_tolerance_semitones: f64,
    pub onset_tolerance_beats: f64,
    /// Absolute log ratio: ln(target_duration / expected_duration).
    pub duration_log_ratio_tolerance: f64,
    pub minimum_coverage: f64,
    pub maximum_gap_fraction: f64,
}

impl Default for CompositionalVerificationConfig {
    fn default() -> Self {
        Self {
            pitch_tolerance_semitones: 0.25,
            onset_tolerance_beats: 0.05,
            duration_log_ratio_tolerance: 0.05,
            minimum_coverage: 0.90,
            maximum_gap_fraction: 0.10,
        }
    }
}

/// A verifier-input error is different from a negative verdict: the submitted
/// claim is malformed or outside this contract, so it cannot be certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionalVerificationError {
    InvalidInput(&'static str),
    UnsupportedTransform(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxisEvidence {
    pub max_error: f64,
    pub mean_error: f64,
    pub tolerance: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositionalMotifEvidence {
    pub contract_version: &'static str,
    pub matched_events: usize,
    pub deleted_source_events: usize,
    pub inserted_target_events: usize,
    pub selected_source_events: usize,
    pub target_events: usize,
    pub coverage: f64,
    pub gap_fraction: f64,
    pub pitch: AxisEvidence,
    pub onset: AxisEvidence,
    pub duration: AxisEvidence,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

/// Verify a compositional transformation using conservative default tolerances.
pub fn verify_compositional_transform(
    source: &[SymbolicEvent],
    target: &[SymbolicEvent],
    target_start_offset_beats: f64,
    claim: &CompositionalTransform,
) -> Result<CompositionalMotifEvidence, CompositionalVerificationError> {
    verify_compositional_transform_with_config(
        source,
        target,
        target_start_offset_beats,
        claim,
        &CompositionalVerificationConfig::default(),
    )
}

/// Verify each declared transformation axis independently, using bounded
/// monotonic alignment for insertions/deletions. A claim only passes when all
/// axis residuals, coverage, and the explicit gap budget pass.
pub fn verify_compositional_transform_with_config(
    source: &[SymbolicEvent],
    target: &[SymbolicEvent],
    target_start_offset_beats: f64,
    claim: &CompositionalTransform,
    config: &CompositionalVerificationConfig,
) -> Result<CompositionalMotifEvidence, CompositionalVerificationError> {
    validate_config(config)?;
    validate_events(source, "source")?;
    validate_events(target, "target")?;
    if !target_start_offset_beats.is_finite() || target_start_offset_beats < 0.0 {
        return Err(CompositionalVerificationError::InvalidInput(
            "target start offset must be finite and non-negative",
        ));
    }
    if source.is_empty() || target.is_empty() {
        return Err(CompositionalVerificationError::InvalidInput(
            "source and target must both contain events",
        ));
    }
    if !claim.onset_scale.is_finite()
        || claim.onset_scale <= 0.0
        || claim.onset_scale > 8.0
        || !claim.duration_scale.is_finite()
        || claim.duration_scale <= 0.0
        || claim.duration_scale > 8.0
    {
        return Err(CompositionalVerificationError::UnsupportedTransform(
            "onset and duration scales must be finite and in (0, 8]",
        ));
    }
    if !(-24..=24).contains(&claim.transpose_semitones) {
        return Err(CompositionalVerificationError::UnsupportedTransform(
            "transposition must be between -24 and +24 semitones",
        ));
    }

    let selection = claim.selection;
    let end = selection
        .source_start_index
        .checked_add(selection.note_count)
        .ok_or(CompositionalVerificationError::InvalidInput(
            "source selection index overflow",
        ))?;
    if end > source.len() || selection.note_count < 3 {
        return Err(CompositionalVerificationError::InvalidInput(
            "selection must be in bounds and contain at least three events",
        ));
    }
    match selection.mode {
        SelectionMode::Full
            if selection.source_start_index != 0 || selection.note_count != source.len() =>
        {
            return Err(CompositionalVerificationError::InvalidInput(
                "full selection must cover the complete source sequence",
            ));
        }
        SelectionMode::ContiguousFragment if selection.note_count >= source.len() => {
            return Err(CompositionalVerificationError::InvalidInput(
                "fragment selection must be a proper source subsequence",
            ));
        }
        _ => {}
    }

    let selected = &source[selection.source_start_index..end];
    let cells = selected
        .len()
        .checked_add(1)
        .and_then(|rows| target.len().checked_add(1).and_then(|cols| rows.checked_mul(cols)));
    if cells.is_none_or(|count| count > MAX_ALIGNMENT_CELLS) {
        return Err(CompositionalVerificationError::UnsupportedTransform(
            "alignment exceeds the configured computation-cell budget",
        ));
    }
    let anchor = selected[0];
    for event in selected {
        let (pitch, onset, duration) =
            expected_values(*event, anchor, claim, target_start_offset_beats);
        if !(0..=127).contains(&pitch) {
            return Err(CompositionalVerificationError::UnsupportedTransform(
                "declared transformation produces a pitch outside MIDI range",
            ));
        }
        if !onset.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(CompositionalVerificationError::InvalidInput(
                "transformed event values must remain finite with positive duration",
            ));
        }
    }

    let (matches, deleted_source_events, inserted_target_events) =
        align_monotonically(selected, target, anchor, claim, target_start_offset_beats, config);
    let pitch_errors: Vec<f64> = matches
        .iter()
        .map(|(i, j)| {
            let expected = expected_values(selected[*i], anchor, claim, target_start_offset_beats);
            (target[*j].pitch_midi - expected.0).abs() as f64
        })
        .collect();
    let onset_errors: Vec<f64> = matches
        .iter()
        .map(|(i, j)| {
            let expected = expected_values(selected[*i], anchor, claim, target_start_offset_beats);
            (target[*j].onset_beats - expected.1).abs()
        })
        .collect();
    let duration_errors: Vec<f64> = matches
        .iter()
        .map(|(i, j)| {
            let expected = expected_values(selected[*i], anchor, claim, target_start_offset_beats);
            (target[*j].duration_beats / expected.2).ln().abs()
        })
        .collect();

    let pitch = make_axis_evidence(&pitch_errors, config.pitch_tolerance_semitones);
    let onset = make_axis_evidence(&onset_errors, config.onset_tolerance_beats);
    let duration = make_axis_evidence(&duration_errors, config.duration_log_ratio_tolerance);
    let denominator = (selected.len() + target.len()) as f64;
    let coverage = matches.len() as f64 / selected.len().max(target.len()) as f64;
    let gap_fraction =
        (deleted_source_events + inserted_target_events) as f64 / denominator;
    let mut failure_reasons = Vec::new();
    if !pitch.passed {
        failure_reasons.push("pitch axis exceeds its independent tolerance".to_owned());
    }
    if !onset.passed {
        failure_reasons.push("onset axis exceeds its independent tolerance".to_owned());
    }
    if !duration.passed {
        failure_reasons.push("duration axis exceeds its independent tolerance".to_owned());
    }
    if coverage < config.minimum_coverage {
        failure_reasons.push("matched-event coverage is below the required floor".to_owned());
    }
    if gap_fraction > config.maximum_gap_fraction {
        failure_reasons.push("insertions/deletions exceed the explicit edit budget".to_owned());
    }
    Ok(CompositionalMotifEvidence {
        contract_version: COMPOSITIONAL_MOTIF_CONTRACT_VERSION,
        matched_events: matches.len(),
        deleted_source_events,
        inserted_target_events,
        selected_source_events: selected.len(),
        target_events: target.len(),
        coverage,
        gap_fraction,
        pitch,
        onset,
        duration,
        passed: failure_reasons.is_empty(),
        failure_reasons,
    })
}

fn validate_config(
    config: &CompositionalVerificationConfig,
) -> Result<(), CompositionalVerificationError> {
    let tolerances = [
        config.pitch_tolerance_semitones,
        config.onset_tolerance_beats,
        config.duration_log_ratio_tolerance,
    ];
    if tolerances.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(CompositionalVerificationError::InvalidInput(
            "all axis tolerances must be finite and positive",
        ));
    }
    if !config.minimum_coverage.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_coverage)
        || !config.maximum_gap_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_gap_fraction)
    {
        return Err(CompositionalVerificationError::InvalidInput(
            "coverage and gap budgets must be within [0, 1]",
        ));
    }
    Ok(())
}

fn validate_events(
    events: &[SymbolicEvent],
    label: &'static str,
) -> Result<(), CompositionalVerificationError> {
    let mut previous_onset: Option<f64> = None;
    for event in events {
        if !(0..=127).contains(&event.pitch_midi) {
            return Err(CompositionalVerificationError::InvalidInput(
                "pitch must be a MIDI semitone in [0, 127]",
            ));
        }
        if !event.onset_beats.is_finite()
            || event.onset_beats < 0.0
            || !event.duration_beats.is_finite()
            || event.duration_beats <= 0.0
        {
            return Err(CompositionalVerificationError::InvalidInput(
                "onsets must be finite/non-negative and durations finite/positive",
            ));
        }
        if previous_onset.is_some_and(|previous| event.onset_beats <= previous) {
            return Err(match label {
                "source" => CompositionalVerificationError::InvalidInput(
                    "source events must be strictly ordered as a monophonic sequence",
                ),
                _ => CompositionalVerificationError::InvalidInput(
                    "target events must be strictly ordered as a monophonic sequence",
                ),
            });
        }
        previous_onset = Some(event.onset_beats);
    }
    Ok(())
}

fn expected_values(
    event: SymbolicEvent,
    anchor: SymbolicEvent,
    claim: &CompositionalTransform,
    target_start_offset_beats: f64,
) -> (i32, f64, f64) {
    let direction = match claim.pitch_orientation {
        PitchOrientation::Preserve => 1,
        PitchOrientation::Invert => -1,
    };
    (
        anchor.pitch_midi
            + claim.transpose_semitones
            + direction * (event.pitch_midi - anchor.pitch_midi),
        target_start_offset_beats
            + claim.onset_scale * (event.onset_beats - anchor.onset_beats),
        event.duration_beats * claim.duration_scale,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Start,
    Match,
    DeleteSource,
    InsertTarget,
}

fn align_monotonically(
    source: &[SymbolicEvent],
    target: &[SymbolicEvent],
    anchor: SymbolicEvent,
    claim: &CompositionalTransform,
    target_start_offset_beats: f64,
    config: &CompositionalVerificationConfig,
) -> (Vec<(usize, usize)>, usize, usize) {
    let n = source.len();
    let m = target.len();
    const GAP_COST: f64 = 1.5;
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    let mut step = vec![vec![Step::Start; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n {
        cost[i][0] = cost[i - 1][0] + GAP_COST;
        step[i][0] = Step::DeleteSource;
    }
    for j in 1..=m {
        cost[0][j] = cost[0][j - 1] + GAP_COST;
        step[0][j] = Step::InsertTarget;
    }

    for i in 1..=n {
        for j in 1..=m {
            let match_cost = event_match_cost(
                source[i - 1],
                target[j - 1],
                anchor,
                claim,
                target_start_offset_beats,
                config,
            );
            let mut best_cost = cost[i - 1][j - 1] + match_cost;
            let mut best_step = Step::Match;
            let delete_cost = cost[i - 1][j] + GAP_COST;
            if delete_cost < best_cost {
                best_cost = delete_cost;
                best_step = Step::DeleteSource;
            }
            let insert_cost = cost[i][j - 1] + GAP_COST;
            if insert_cost < best_cost {
                best_cost = insert_cost;
                best_step = Step::InsertTarget;
            }
            cost[i][j] = best_cost;
            step[i][j] = best_step;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut matches = Vec::new();
    let mut deleted = 0;
    let mut inserted = 0;
    while i > 0 || j > 0 {
        match step[i][j] {
            Step::Match if i > 0 && j > 0 => {
                matches.push((i - 1, j - 1));
                i -= 1;
                j -= 1;
            }
            Step::DeleteSource if i > 0 => {
                deleted += 1;
                i -= 1;
            }
            Step::InsertTarget if j > 0 => {
                inserted += 1;
                j -= 1;
            }
            _ => break,
        }
    }
    matches.reverse();
    (matches, deleted, inserted)
}

fn event_match_cost(
    source: SymbolicEvent,
    target: SymbolicEvent,
    anchor: SymbolicEvent,
    claim: &CompositionalTransform,
    target_start_offset_beats: f64,
    config: &CompositionalVerificationConfig,
) -> f64 {
    let expected = expected_values(source, anchor, claim, target_start_offset_beats);
    let pitch_error = (target.pitch_midi - expected.0).abs() as f64;
    let onset_error = (target.onset_beats - expected.1).abs();
    let duration_error = (target.duration_beats / expected.2).ln().abs();
    normalized_cost(pitch_error, config.pitch_tolerance_semitones)
        + normalized_cost(onset_error, config.onset_tolerance_beats)
        + normalized_cost(duration_error, config.duration_log_ratio_tolerance)
}

fn normalized_cost(error: f64, tolerance: f64) -> f64 {
    (error / tolerance).min(4.0)
}

fn make_axis_evidence(errors: &[f64], tolerance: f64) -> AxisEvidence {
    if errors.is_empty() {
        return AxisEvidence {
            max_error: f64::INFINITY,
            mean_error: f64::INFINITY,
            tolerance,
            passed: false,
        };
    }
    let max_error = errors.iter().copied().fold(0.0_f64, f64::max);
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    AxisEvidence {
        max_error,
        mean_error,
        tolerance,
        passed: max_error.is_finite() && max_error <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SMOKE_CASES: &str =
        include_str!("../mel-mr-eval-003-compositional-smoke-cases-v1.jsonl");

    fn descriptor(
        orientation: PitchOrientation,
        transpose: i32,
        onset_scale: f64,
        duration_scale: f64,
        mode: SelectionMode,
        start: usize,
        count: usize,
    ) -> CompositionalTransform {
        CompositionalTransform {
            pitch_orientation: orientation,
            transpose_semitones: transpose,
            onset_scale,
            duration_scale,
            selection: SourceSelection {
                mode,
                source_start_index: start,
                note_count: count,
            },
        }
    }

    fn simple_source() -> Vec<SymbolicEvent> {
        vec![
            SymbolicEvent { pitch_midi: 60, onset_beats: 0.0, duration_beats: 0.5 },
            SymbolicEvent { pitch_midi: 62, onset_beats: 1.0, duration_beats: 0.5 },
            SymbolicEvent { pitch_midi: 65, onset_beats: 2.0, duration_beats: 1.0 },
            SymbolicEvent { pitch_midi: 64, onset_beats: 3.0, duration_beats: 0.5 },
            SymbolicEvent { pitch_midi: 67, onset_beats: 4.0, duration_beats: 1.0 },
        ]
    }

    fn transform_sequence(
        source: &[SymbolicEvent],
        transform: CompositionalTransform,
        target_start: f64,
    ) -> Vec<SymbolicEvent> {
        let sel = transform.selection;
        let selected = &source[sel.source_start_index..sel.source_start_index + sel.note_count];
        let anchor = selected[0];
        selected
            .iter()
            .map(|event| {
                let (pitch_midi, onset_beats, duration_beats) =
                    expected_values(*event, anchor, &transform, target_start);
                SymbolicEvent { pitch_midi, onset_beats, duration_beats }
            })
            .collect()
    }

    #[test]
    fn verifies_compound_inversion_transposition_and_temporal_scaling() {
        let source = simple_source();
        let claim = descriptor(
            PitchOrientation::Invert,
            5,
            2.0,
            2.0,
            SelectionMode::Full,
            0,
            source.len(),
        );
        let target = transform_sequence(&source, claim, 8.0);
        let evidence = verify_compositional_transform(&source, &target, 8.0, &claim).unwrap();
        assert!(evidence.passed, "{evidence:?}");
        assert!(evidence.pitch.passed && evidence.onset.passed && evidence.duration.passed);
        assert_eq!(evidence.matched_events, source.len());
        assert_eq!(evidence.gap_fraction, 0.0);
    }

    #[test]
    fn rejects_single_axis_violations_without_averaging_them_away() {
        let source = simple_source();
        let claim = descriptor(
            PitchOrientation::Preserve,
            5,
            2.0,
            2.0,
            SelectionMode::Full,
            0,
            source.len(),
        );
        let mut target = transform_sequence(&source, claim, 3.0);
        target[2].pitch_midi += 2;
        let pitch_evidence = verify_compositional_transform(&source, &target, 3.0, &claim).unwrap();
        assert!(!pitch_evidence.passed);
        assert!(
            !pitch_evidence.pitch.passed
                || pitch_evidence.coverage < 0.90
                || pitch_evidence.gap_fraction > 0.10
        );

        let mut target = transform_sequence(&source, claim, 3.0);
        target[2].onset_beats += 0.25;
        let onset_evidence = verify_compositional_transform(&source, &target, 3.0, &claim).unwrap();
        assert!(!onset_evidence.passed);
        assert!(
            !onset_evidence.onset.passed
                || onset_evidence.coverage < 0.90
                || onset_evidence.gap_fraction > 0.10
        );
    }

    #[test]
    fn alignment_accepts_one_bounded_insert_but_reports_it() {
        let mut source = simple_source();
        for i in 5..10 {
            source.push(SymbolicEvent {
                pitch_midi: 60 + (i % 5) * 2,
                onset_beats: i as f64,
                duration_beats: 0.5,
            });
        }
        let claim = descriptor(
            PitchOrientation::Preserve,
            0,
            1.0,
            1.0,
            SelectionMode::Full,
            0,
            source.len(),
        );
        let mut target = transform_sequence(&source, claim, 4.0);
        target.insert(
            2,
            SymbolicEvent { pitch_midi: 66, onset_beats: 5.5, duration_beats: 0.25 },
        );
        let evidence = verify_compositional_transform(&source, &target, 4.0, &claim).unwrap();
        assert!(evidence.passed, "{evidence:?}");
        assert_eq!(evidence.inserted_target_events, 1);
    }

    #[test]
    fn rejects_malformed_and_unsupported_claims_fail_closed() {
        let source = simple_source();
        let invalid_selection = descriptor(
            PitchOrientation::Preserve,
            0,
            1.0,
            1.0,
            SelectionMode::Full,
            1,
            3,
        );
        assert!(verify_compositional_transform(&source, &source, 0.0, &invalid_selection).is_err());

        let invalid_scale = descriptor(
            PitchOrientation::Preserve,
            0,
            f64::NAN,
            1.0,
            SelectionMode::Full,
            0,
            source.len(),
        );
        assert!(verify_compositional_transform(&source, &source, 0.0, &invalid_scale).is_err());

        let source_high: Vec<_> = source
            .iter()
            .map(|event| SymbolicEvent {
                pitch_midi: event.pitch_midi + 50,
                ..*event
            })
            .collect();
        let out_of_range = descriptor(
            PitchOrientation::Preserve,
            24,
            1.0,
            1.0,
            SelectionMode::Full,
            0,
            source_high.len(),
        );
        assert!(
            verify_compositional_transform(&source_high, &source_high, 0.0, &out_of_range).is_err()
        );
    }

    #[test]
    fn rejects_alignment_work_above_the_hard_budget() {
        let source: Vec<_> = (0..1_000)
            .map(|index| SymbolicEvent {
                pitch_midi: 60 + (index % 12) as i32,
                onset_beats: index as f64,
                duration_beats: 0.5,
            })
            .collect();
        let claim = descriptor(
            PitchOrientation::Preserve,
            0,
            1.0,
            1.0,
            SelectionMode::Full,
            0,
            source.len(),
        );
        assert!((source.len() + 1) * (source.len() + 1) > MAX_ALIGNMENT_CELLS);
        assert!(
            verify_compositional_transform(&source, &source, 0.0, &claim).is_err(),
            "quadratic work must be bounded before allocating DP matrices"
        );
    }

    #[test]
    fn committed_smoke_corpus_replays_against_rust_verifier() {
        let mut seen = std::collections::HashSet::new();
        let mut accepted = 0;
        let mut rejected = 0;
        for line in SMOKE_CASES.lines().filter(|line| !line.trim().is_empty()) {
            let value: Value = serde_json::from_str(line).expect("valid fixture JSON");
            let case_id = value["case_id"].as_str().expect("case id");
            assert!(seen.insert(case_id.to_owned()), "duplicate case id {case_id}");
            let source: Vec<SymbolicEvent> =
                serde_json::from_value(value["source_events"].clone()).expect("source events");
            let target: Vec<SymbolicEvent> =
                serde_json::from_value(value["target_events"].clone()).expect("target events");
            let claim: CompositionalTransform =
                serde_json::from_value(value["claimed_transform"].clone()).expect("claim descriptor");
            let start = value["target_start_offset_beats"]
                .as_f64()
                .expect("target start offset");
            let evidence =
                verify_compositional_transform(&source, &target, start, &claim).expect("valid contract");
            let expected = value["expected_verdict"].as_str().expect("expected verdict");
            match expected {
                "accept" => {
                    accepted += 1;
                    assert!(evidence.passed, "{case_id} should pass: {evidence:?}");
                }
                "reject" => {
                    rejected += 1;
                    assert!(!evidence.passed, "{case_id} should fail: {evidence:?}");
                }
                other => panic!("unhandled fixture verdict {other} for {case_id}"),
            }
        }
        assert_eq!(seen.len(), 108);
        assert_eq!(accepted, 60);
        assert_eq!(rejected, 48);
    }
}
