// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dependency-free scoring for supplied singing phoneme alignments.
//!
//! This crate evaluates an aligner's event sequence against an independently
//! prepared reference. It does not infer phonemes from waveform bytes, perform
//! forced alignment, estimate F0, or decide whether a performance is beautiful.
//! Its purpose is to keep pronunciation/alignment evidence separate from
//! score-conditioned pitch diagnostics and human listening judgments.

#![forbid(unsafe_code)]

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneEvent {
    /// One canonical phone token (normally an explicitly normalized IPA token).
    pub symbol: String,
    /// Monotonic word identity within this utterance.
    pub word_index: u32,
    /// Monotonic syllable identity within this utterance.
    pub syllable_index: u32,
    /// Half-open interval on the source audio sample-frame clock.
    pub start_frame: u64,
    pub end_frame: u64,
    pub is_vowel: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictedPhoneEvent {
    /// One canonical phone token. Unknown phones must remain explicit errors
    /// upstream; do not silently map them to schwa or another fallback.
    pub symbol: String,
    pub word_index: u32,
    pub syllable_index: u32,
    /// Half-open interval on the same sample-frame clock as the reference.
    pub start_frame: u64,
    pub end_frame: u64,
    pub is_vowel: bool,
    /// Optional aligner confidence. Missing confidence is not treated as zero.
    pub confidence: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MelismaSpan {
    /// Reference syllable and vowel event that should carry the sustained vowel.
    pub syllable_index: u32,
    pub reference_phone_index: usize,
    /// Union of the musical note intervals belonging to this melisma.
    pub start_frame: u64,
    pub end_frame: u64,
    pub note_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditCounts {
    pub exact_matches: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
}

impl EditCounts {
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimingMetrics {
    /// Errors are computed only for exact phone identity matches. Substituted
    /// phones do not get to contribute timing accuracy for the wrong sound.
    pub exact_phone_pairs: usize,
    pub onset_mae_ms: Option<f64>,
    pub onset_p95_ms: Option<f64>,
    pub offset_mae_ms: Option<f64>,
    pub offset_p95_ms: Option<f64>,
    pub duration_mae_ms: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MelismaMetrics {
    pub expected_spans: usize,
    pub covered_spans: usize,
    /// None means no melismas were present; it is not a perfect score.
    pub coverage_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignmentMetrics {
    pub sample_rate_hz: u32,
    pub total_frames: u64,
    pub reference_phone_count: usize,
    pub predicted_phone_count: usize,
    pub edits: EditCounts,
    /// (S + D + I) / reference phone count. May exceed 1.0 due to insertions.
    pub phone_error_rate: f64,
    /// Exact matches / reference phone count.
    pub reference_exact_match_fraction: f64,
    /// Exact matches / predicted phone count.
    pub prediction_exact_precision: Option<f64>,
    pub timing: TimingMetrics,
    pub melisma: MelismaMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Edit {
    Exact { reference: usize, predicted: usize },
    Substitute { reference: usize, predicted: usize },
    Delete { reference: usize },
    Insert { predicted: usize },
}

/// Score a supplied phoneme alignment against an independently authored
/// reference. All intervals are half-open sample-frame ranges.
///
/// The reference must be non-empty. Empty predictions are valid and score as
/// deletions, not as a successful zero-error case. Timing measurements are
/// returned only for exact phone identity matches. Melisma spans are checked
/// against the single vowel event that should carry them; repeated vowel events
/// produced for individual notes are counted as phone insertions.
pub fn score_alignment(
    sample_rate_hz: u32,
    total_frames: u64,
    reference: &[PhoneEvent],
    predicted: &[PredictedPhoneEvent],
    melismas: &[MelismaSpan],
) -> Result<AlignmentMetrics, String> {
    validate_header(sample_rate_hz, total_frames, reference)?;
    validate_reference(total_frames, reference)?;
    validate_predictions(total_frames, predicted)?;
    validate_melismas(total_frames, reference, melismas)?;

    let edits = align_phone_sequences(reference, predicted);
    let mut counts = EditCounts {
        exact_matches: 0,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    let mut exact_prediction_for_reference = vec![None; reference.len()];
    let mut onset_errors_ms = Vec::new();
    let mut offset_errors_ms = Vec::new();
    let mut duration_errors_ms = Vec::new();

    let frames_to_ms = 1000.0 / f64::from(sample_rate_hz);
    for edit in edits {
        match edit {
            Edit::Exact { reference: r, predicted: p } => {
                counts.exact_matches += 1;
                exact_prediction_for_reference[r] = Some(p);
                let expected = &reference[r];
                let observed = &predicted[p];
                onset_errors_ms.push(
                    expected.start_frame.abs_diff(observed.start_frame) as f64 * frames_to_ms,
                );
                offset_errors_ms.push(
                    expected.end_frame.abs_diff(observed.end_frame) as f64 * frames_to_ms,
                );
                let expected_duration = expected.end_frame - expected.start_frame;
                let observed_duration = observed.end_frame - observed.start_frame;
                duration_errors_ms.push(
                    expected_duration.abs_diff(observed_duration) as f64 * frames_to_ms,
                );
            }
            Edit::Substitute { .. } => counts.substitutions += 1,
            Edit::Delete { .. } => counts.deletions += 1,
            Edit::Insert { .. } => counts.insertions += 1,
        }
    }

    let reference_count = reference.len();
    let predicted_count = predicted.len();
    let phone_error_rate = counts.errors() as f64 / reference_count as f64;
    let reference_exact_match_fraction = counts.exact_matches as f64 / reference_count as f64;
    let prediction_exact_precision = (predicted_count > 0)
        .then(|| counts.exact_matches as f64 / predicted_count as f64);

    let mut covered_melismas = 0usize;
    for span in melismas {
        if let Some(predicted_index) = exact_prediction_for_reference[span.reference_phone_index] {
            let phone = &predicted[predicted_index];
            if phone.is_vowel
                && phone.syllable_index == span.syllable_index
                && phone.start_frame <= span.start_frame
                && phone.end_frame >= span.end_frame
            {
                covered_melismas += 1;
            }
        }
    }
    let expected_melismas = melismas.len();

    Ok(AlignmentMetrics {
        sample_rate_hz,
        total_frames,
        reference_phone_count: reference_count,
        predicted_phone_count: predicted_count,
        edits: counts.clone(),
        phone_error_rate,
        reference_exact_match_fraction,
        prediction_exact_precision,
        timing: TimingMetrics {
            exact_phone_pairs: counts.exact_matches,
            onset_mae_ms: mean(&onset_errors_ms),
            onset_p95_ms: percentile(&onset_errors_ms, 0.95),
            offset_mae_ms: mean(&offset_errors_ms),
            offset_p95_ms: percentile(&offset_errors_ms, 0.95),
            duration_mae_ms: mean(&duration_errors_ms),
        },
        melisma: MelismaMetrics {
            expected_spans: expected_melismas,
            covered_spans: covered_melismas,
            coverage_fraction: (expected_melismas > 0)
                .then(|| covered_melismas as f64 / expected_melismas as f64),
        },
    })
}

fn validate_header(
    sample_rate_hz: u32,
    total_frames: u64,
    reference: &[PhoneEvent],
) -> Result<(), String> {
    if sample_rate_hz == 0 {
        return Err("sample_rate_hz must be positive".into());
    }
    if total_frames == 0 {
        return Err("total_frames must be positive".into());
    }
    if reference.is_empty() {
        return Err("reference must contain at least one phone event".into());
    }
    Ok(())
}

fn validate_reference(total_frames: u64, events: &[PhoneEvent]) -> Result<(), String> {
    let mut previous_start = 0u64;
    let mut previous_word = 0u32;
    let mut previous_syllable = 0u32;
    for (index, event) in events.iter().enumerate() {
        if event.symbol.trim().is_empty() {
            return Err(format!("reference phone {index} has an empty symbol"));
        }
        validate_interval(
            total_frames,
            event.start_frame,
            event.end_frame,
            "reference",
            index,
        )?;
        if index > 0
            && (event.start_frame < previous_start
                || event.word_index < previous_word
                || event.syllable_index < previous_syllable)
        {
            return Err(format!("reference phone {index} is not chronologically ordered"));
        }
        previous_start = event.start_frame;
        previous_word = event.word_index;
        previous_syllable = event.syllable_index;
    }
    Ok(())
}

fn validate_predictions(total_frames: u64, events: &[PredictedPhoneEvent]) -> Result<(), String> {
    let mut previous_start = 0u64;
    let mut previous_word = 0u32;
    let mut previous_syllable = 0u32;
    for (index, event) in events.iter().enumerate() {
        if event.symbol.trim().is_empty() {
            return Err(format!("predicted phone {index} has an empty symbol"));
        }
        validate_interval(
            total_frames,
            event.start_frame,
            event.end_frame,
            "predicted",
            index,
        )?;
        if index > 0
            && (event.start_frame < previous_start
                || event.word_index < previous_word
                || event.syllable_index < previous_syllable)
        {
            return Err(format!("predicted phone {index} is not chronologically ordered"));
        }
        if let Some(confidence) = event.confidence
            && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            return Err(format!("predicted phone {index} has invalid confidence"));
        }
        previous_start = event.start_frame;
        previous_word = event.word_index;
        previous_syllable = event.syllable_index;
    }
    Ok(())
}

fn validate_interval(
    total_frames: u64,
    start_frame: u64,
    end_frame: u64,
    label: &str,
    index: usize,
) -> Result<(), String> {
    if start_frame >= end_frame || end_frame > total_frames {
        return Err(format!(
            "{label} phone {index} has an invalid half-open frame interval"
        ));
    }
    Ok(())
}

fn validate_melismas(
    total_frames: u64,
    reference: &[PhoneEvent],
    melismas: &[MelismaSpan],
) -> Result<(), String> {
    for (index, span) in melismas.iter().enumerate() {
        if span.note_count < 2
            || span.start_frame >= span.end_frame
            || span.end_frame > total_frames
        {
            return Err(format!(
                "melisma span {index} has invalid note count or frame interval"
            ));
        }
        let Some(phone) = reference.get(span.reference_phone_index) else {
            return Err(format!("melisma span {index} references a missing phone"));
        };
        if !phone.is_vowel
            || phone.syllable_index != span.syllable_index
            || phone.start_frame > span.start_frame
            || phone.end_frame < span.end_frame
        {
            return Err(format!(
                "melisma span {index} must be contained by its reference vowel event"
            ));
        }
    }
    Ok(())
}

fn align_phone_sequences(
    reference: &[PhoneEvent],
    predicted: &[PredictedPhoneEvent],
) -> Vec<Edit> {
    let n = reference.len();
    let m = predicted.len();
    let mut costs = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        costs[i][0] = i;
    }
    for j in 0..=m {
        costs[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let substitution_cost = if reference[i - 1].symbol == predicted[j - 1].symbol {
                0
            } else {
                1
            };
            let diagonal = costs[i - 1][j - 1] + substitution_cost;
            let deletion = costs[i - 1][j] + 1;
            let insertion = costs[i][j - 1] + 1;
            costs[i][j] = diagonal.min(deletion).min(insertion);
        }
    }

    let mut reversed = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        // Deterministic tie-breaking: diagonal, then deletion, then insertion.
        if i > 0 && j > 0 {
            let cost = if reference[i - 1].symbol == predicted[j - 1].symbol {
                0
            } else {
                1
            };
            if costs[i][j] == costs[i - 1][j - 1] + cost {
                if cost == 0 {
                    reversed.push(Edit::Exact {
                        reference: i - 1,
                        predicted: j - 1,
                    });
                } else {
                    reversed.push(Edit::Substitute {
                        reference: i - 1,
                        predicted: j - 1,
                    });
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && costs[i][j] == costs[i - 1][j] + 1 {
            reversed.push(Edit::Delete { reference: i - 1 });
            i -= 1;
        } else {
            reversed.push(Edit::Insert { predicted: j - 1 });
            j -= 1;
        }
    }
    reversed.reverse();
    reversed
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Nearest-rank percentile. Empty support is reported as None, never zero.
fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        symbol: &str,
        syllable: u32,
        start: u64,
        end: u64,
        is_vowel: bool,
    ) -> PhoneEvent {
        PhoneEvent {
            symbol: symbol.into(),
            word_index: 0,
            syllable_index: syllable,
            start_frame: start,
            end_frame: end,
            is_vowel,
        }
    }

    fn prediction(
        symbol: &str,
        syllable: u32,
        start: u64,
        end: u64,
        is_vowel: bool,
    ) -> PredictedPhoneEvent {
        PredictedPhoneEvent {
            symbol: symbol.into(),
            word_index: 0,
            syllable_index: syllable,
            start_frame: start,
            end_frame: end,
            is_vowel,
            confidence: Some(0.95),
        }
    }

    #[test]
    fn exact_alignment_has_zero_phone_and_timing_error() {
        let reference = vec![
            reference("h", 0, 0, 400, false),
            reference("ə", 0, 400, 1800, true),
            reference("l", 0, 1800, 2200, false),
        ];
        let predicted = vec![
            prediction("h", 0, 0, 400, false),
            prediction("ə", 0, 400, 1800, true),
            prediction("l", 0, 1800, 2200, false),
        ];
        let result = score_alignment(44_100, 3_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.exact_matches, 3);
        assert_eq!(result.phone_error_rate, 0.0);
        assert_eq!(result.timing.onset_mae_ms, Some(0.0));
        assert_eq!(result.timing.offset_p95_ms, Some(0.0));
    }

    #[test]
    fn edit_counts_report_substitution_deletion_and_insertion() {
        let reference = vec![
            reference("a", 0, 0, 400, true),
            reference("b", 1, 500, 800, false),
            reference("c", 2, 900, 1200, false),
        ];
        let predicted = vec![
            prediction("a", 0, 0, 400, true),
            prediction("x", 1, 500, 800, false),
            prediction("c", 2, 900, 1200, false),
            prediction("d", 3, 1300, 1500, false),
        ];
        let result = score_alignment(24_000, 2_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.substitutions, 1);
        assert_eq!(result.edits.deletions, 0);
        assert_eq!(result.edits.insertions, 1);
        assert!((result.phone_error_rate - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(result.timing.exact_phone_pairs, 2);
    }

    #[test]
    fn timing_uses_only_exact_phone_identity_matches() {
        let reference = vec![
            reference("a", 0, 100, 500, true),
            reference("s", 1, 600, 900, false),
        ];
        let predicted = vec![
            prediction("a", 0, 541, 900, true),
            prediction("z", 1, 620, 910, false),
        ];
        let result = score_alignment(44_100, 2_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.timing.exact_phone_pairs, 1);
        assert!((result.timing.onset_mae_ms.unwrap() - 10.0).abs() < 1e-10);
        assert_eq!(result.edits.substitutions, 1);
    }

    #[test]
    fn empty_prediction_is_a_full_deletion_not_a_vacuous_pass() {
        let reference = vec![reference("a", 0, 0, 100, true)];
        let result = score_alignment(16_000, 200, &reference, &[], &[]).unwrap();
        assert_eq!(result.edits.deletions, 1);
        assert_eq!(result.phone_error_rate, 1.0);
        assert_eq!(result.timing.onset_mae_ms, None);
        assert_eq!(result.prediction_exact_precision, None);
    }

    #[test]
    fn melisma_is_one_sustained_vowel_not_repeated_phone_tokens() {
        let reference = vec![
            reference("m", 0, 0, 300, false),
            reference("a", 0, 300, 2_500, true),
        ];
        let predicted = vec![
            prediction("m", 0, 0, 300, false),
            prediction("a", 0, 300, 2_500, true),
        ];
        let melisma = MelismaSpan {
            syllable_index: 0,
            reference_phone_index: 1,
            start_frame: 700,
            end_frame: 2_300,
            note_count: 4,
        };
        let result =
            score_alignment(24_000, 3_000, &reference, &predicted, &[melisma]).unwrap();
        assert_eq!(result.edits.exact_matches, 2);
        assert_eq!(result.edits.insertions, 0);
        assert_eq!(result.melisma.covered_spans, 1);
        assert_eq!(result.melisma.coverage_fraction, Some(1.0));
    }

    #[test]
    fn repeated_vowels_per_melisma_note_are_insertions() {
        let reference = vec![reference("a", 0, 0, 2_500, true)];
        let predicted = vec![
            prediction("a", 0, 0, 800, true),
            prediction("a", 0, 800, 1_600, true),
            prediction("a", 0, 1_600, 2_500, true),
        ];
        let result = score_alignment(24_000, 3_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.exact_matches, 1);
        assert_eq!(result.edits.insertions, 2);
        assert_eq!(result.phone_error_rate, 2.0);
    }

    #[test]
    fn a_truncated_melisma_vowel_fails_coverage_without_hiding_phone_match() {
        let reference = vec![reference("a", 0, 0, 2_500, true)];
        let predicted = vec![prediction("a", 0, 100, 1_900, true)];
        let melisma = MelismaSpan {
            syllable_index: 0,
            reference_phone_index: 0,
            start_frame: 700,
            end_frame: 2_300,
            note_count: 4,
        };
        let result = score_alignment(24_000, 3_000, &reference, &predicted, &[melisma]).unwrap();
        assert_eq!(result.edits.exact_matches, 1);
        assert_eq!(result.melisma.covered_spans, 0);
        assert_eq!(result.melisma.coverage_fraction, Some(0.0));
    }

    #[test]
    fn unsupported_missing_reference_and_out_of_range_inputs_fail_closed() {
        assert!(score_alignment(0, 1, &[reference("a", 0, 0, 1, true)], &[], &[]).is_err());
        assert!(score_alignment(16_000, 0, &[reference("a", 0, 0, 1, true)], &[], &[]).is_err());
        assert!(score_alignment(16_000, 1, &[], &[], &[]).is_err());
        assert!(
            score_alignment(16_000, 100, &[reference("a", 0, 0, 200, true)], &[], &[]).is_err()
        );
    }

    #[test]
    fn invalid_confidence_is_rejected_not_clamped() {
        let reference = vec![reference("a", 0, 0, 100, true)];
        let mut predicted = vec![prediction("a", 0, 0, 100, true)];
        predicted[0].confidence = Some(f32::NAN);
        assert!(score_alignment(16_000, 200, &reference, &predicted, &[]).is_err());
    }

    #[test]
    fn undefined_timing_and_melisma_are_not_reported_as_zero_error() {
        let reference = vec![reference("a", 0, 0, 100, true)];
        let result = score_alignment(16_000, 200, &reference, &[], &[]).unwrap();
        assert_eq!(result.timing.onset_p95_ms, None);
        assert_eq!(result.timing.offset_mae_ms, None);
        assert_eq!(result.melisma.coverage_fraction, None);
    }
}
