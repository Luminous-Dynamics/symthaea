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
    pub onset_median_ms: Option<f64>,
    pub onset_p95_ms: Option<f64>,
    /// Fraction of strict identity pairs whose absolute onset error is <= 20 ms.
    pub onset_within_20ms_fraction: Option<f64>,
    /// Jointly correct phone identity and onset <= 20 ms / all reference phones.
    /// Unlike conditional timing fractions, missed or wrongly named phones remain
    /// in the denominator and cannot disappear from the score.
    pub joint_onset_within_20ms_fraction: f64,
    /// Fraction of strict identity pairs whose absolute onset error is <= 50 ms.
    pub onset_within_50ms_fraction: Option<f64>,
    /// Jointly correct phone identity and onset <= 50 ms / all reference phones.
    pub joint_onset_within_50ms_fraction: f64,
    /// Fraction of exact vowel-identity pairs whose onset error is <= 50 ms.
    pub vowel_onset_within_50ms_fraction: Option<f64>,
    /// Fraction of strict identity pairs whose absolute onset error is <= 100 ms.
    pub onset_within_100ms_fraction: Option<f64>,
    /// Jointly correct phone identity and onset <= 100 ms / all reference phones.
    pub joint_onset_within_100ms_fraction: f64,
    pub offset_mae_ms: Option<f64>,
    pub offset_p95_ms: Option<f64>,
    pub duration_mae_ms: Option<f64>,
    /// Sum(predicted exact-phone durations) / sum(reference exact-phone durations).
    /// A value near 1.0 means aligned exact phones have conserved duration overall.
    pub duration_ratio: Option<f64>,
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
    /// Canonical phone-symbol matches / reference phone count. This is the
    /// complement of PER and does not include lexical slot/boundary correctness.
    pub reference_phone_match_fraction: f64,
    /// Strict identity matches / reference phone count. Requires symbol,
    /// word index, syllable index, and vowel/consonant class to agree.
    pub reference_identity_match_fraction: f64,
    /// Canonical phone-symbol matches / predicted phone count.
    pub prediction_phone_precision: Option<f64>,
    pub timing: TimingMetrics,
    pub melisma: MelismaMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Edit {
    Exact { reference: usize, predicted: usize },
    Substitute,
    Delete,
    Insert,
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

    let phone_edits = align_phone_sequences(reference, predicted, false);
    let identity_edits = align_phone_sequences(reference, predicted, true);
    let mut counts = EditCounts {
        exact_matches: 0,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    let mut identity_prediction_for_reference = vec![None; reference.len()];
    let mut onset_errors_ms = Vec::new();
    let mut joint_onset_hits_20 = 0usize;
    let mut joint_onset_hits_50 = 0usize;
    let mut joint_onset_hits_100 = 0usize;
    let mut offset_errors_ms = Vec::new();
    let mut duration_errors_ms = Vec::new();
    let mut vowel_onset_errors_ms = Vec::new();
    let mut exact_reference_duration_frames = 0u128;
    let mut exact_predicted_duration_frames = 0u128;

    let frames_to_ms = 1000.0 / f64::from(sample_rate_hz);
    for edit in phone_edits {
        match edit {
            Edit::Exact { .. } => counts.exact_matches += 1,
            Edit::Substitute => counts.substitutions += 1,
            Edit::Delete => counts.deletions += 1,
            Edit::Insert => counts.insertions += 1,
        }
    }

    // A second alignment pass enforces lexical position and syllable identity.
    // Its timing denominator is deliberately stricter than canonical phone PER.
    for edit in identity_edits {
        if let Edit::Exact {
            reference: r,
            predicted: p,
        } = edit
        {
            identity_prediction_for_reference[r] = Some(p);
            let expected = &reference[r];
            let observed = &predicted[p];
            let onset_error_ms =
                expected.start_frame.abs_diff(observed.start_frame) as f64 * frames_to_ms;
            onset_errors_ms.push(onset_error_ms);
            if onset_error_ms <= 20.0 {
                joint_onset_hits_20 += 1;
            }
            if onset_error_ms <= 50.0 {
                joint_onset_hits_50 += 1;
            }
            if onset_error_ms <= 100.0 {
                joint_onset_hits_100 += 1;
            }
            if expected.is_vowel {
                vowel_onset_errors_ms.push(onset_error_ms);
            }
            offset_errors_ms
                .push(expected.end_frame.abs_diff(observed.end_frame) as f64 * frames_to_ms);
            let expected_duration = expected.end_frame - expected.start_frame;
            let observed_duration = observed.end_frame - observed.start_frame;
            exact_reference_duration_frames += u128::from(expected_duration);
            exact_predicted_duration_frames += u128::from(observed_duration);
            duration_errors_ms
                .push(expected_duration.abs_diff(observed_duration) as f64 * frames_to_ms);
        }
    }

    let reference_count = reference.len();
    let predicted_count = predicted.len();
    let phone_error_rate = counts.errors() as f64 / reference_count as f64;
    let reference_phone_match_fraction = counts.exact_matches as f64 / reference_count as f64;
    let identity_match_count = identity_prediction_for_reference
        .iter()
        .filter(|prediction_index| prediction_index.is_some())
        .count();
    let reference_identity_match_fraction = identity_match_count as f64 / reference_count as f64;
    let prediction_phone_precision =
        (predicted_count > 0).then(|| counts.exact_matches as f64 / predicted_count as f64);

    let mut covered_melismas = 0usize;
    for span in melismas {
        if let Some(predicted_index) = identity_prediction_for_reference[span.reference_phone_index]
        {
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
        reference_phone_match_fraction,
        reference_identity_match_fraction,
        prediction_phone_precision,
        timing: TimingMetrics {
            exact_phone_pairs: identity_match_count,
            onset_mae_ms: mean(&onset_errors_ms),
            onset_median_ms: median(&onset_errors_ms),
            onset_p95_ms: percentile(&onset_errors_ms, 0.95),
            onset_within_20ms_fraction: fraction_at_or_below(&onset_errors_ms, 20.0),
            joint_onset_within_20ms_fraction: joint_onset_hits_20 as f64 / reference_count as f64,
            onset_within_50ms_fraction: fraction_at_or_below(&onset_errors_ms, 50.0),
            joint_onset_within_50ms_fraction: joint_onset_hits_50 as f64 / reference_count as f64,
            vowel_onset_within_50ms_fraction: fraction_at_or_below(&vowel_onset_errors_ms, 50.0),
            onset_within_100ms_fraction: fraction_at_or_below(&onset_errors_ms, 100.0),
            joint_onset_within_100ms_fraction: joint_onset_hits_100 as f64 / reference_count as f64,
            offset_mae_ms: mean(&offset_errors_ms),
            offset_p95_ms: percentile(&offset_errors_ms, 0.95),
            duration_mae_ms: mean(&duration_errors_ms),
            duration_ratio: (exact_reference_duration_frames > 0).then(|| {
                exact_predicted_duration_frames as f64 / exact_reference_duration_frames as f64
            }),
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
            return Err(format!(
                "reference phone {index} is not chronologically ordered"
            ));
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
            return Err(format!(
                "predicted phone {index} is not chronologically ordered"
            ));
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

/// A phone is an exact match only when its symbol, lexical position,
/// syllable assignment, and vowel/consonant class all agree. Timing metrics
/// must not reward a correctly spelled phone attached to the wrong lyric slot.
fn same_phone_identity(reference: &PhoneEvent, predicted: &PredictedPhoneEvent) -> bool {
    reference.symbol == predicted.symbol
        && reference.word_index == predicted.word_index
        && reference.syllable_index == predicted.syllable_index
        && reference.is_vowel == predicted.is_vowel
}

fn align_phone_sequences(
    reference: &[PhoneEvent],
    predicted: &[PredictedPhoneEvent],
    require_lexical_identity: bool,
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
            let matches = if require_lexical_identity {
                same_phone_identity(&reference[i - 1], &predicted[j - 1])
            } else {
                reference[i - 1].symbol == predicted[j - 1].symbol
            };
            let substitution_cost = usize::from(!matches);
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
            let matches = if require_lexical_identity {
                same_phone_identity(&reference[i - 1], &predicted[j - 1])
            } else {
                reference[i - 1].symbol == predicted[j - 1].symbol
            };
            let cost = usize::from(!matches);
            if costs[i][j] == costs[i - 1][j - 1] + cost {
                if cost == 0 {
                    reversed.push(Edit::Exact {
                        reference: i - 1,
                        predicted: j - 1,
                    });
                } else {
                    reversed.push(Edit::Substitute);
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && costs[i][j] == costs[i - 1][j] + 1 {
            reversed.push(Edit::Delete);
            i -= 1;
        } else {
            reversed.push(Edit::Insert);
            j -= 1;
        }
    }
    reversed.reverse();
    reversed
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Conventional median: arithmetic mean of the two center values for even n.
/// Empty timing support remains None rather than a synthetic zero.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        Some(sorted[middle])
    }
}

/// Empirical fraction of errors at or below a declared timing threshold.
fn fraction_at_or_below(values: &[f64], threshold_ms: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let within = values
        .iter()
        .filter(|value| **value <= threshold_ms)
        .count();
    Some(within as f64 / values.len() as f64)
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

    fn reference(symbol: &str, syllable: u32, start: u64, end: u64, is_vowel: bool) -> PhoneEvent {
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
    fn onset_timing_reports_median_and_published_threshold_coverage() {
        let reference = vec![
            reference("a", 0, 100, 200, true),
            reference("b", 1, 300, 400, false),
            reference("c", 2, 500, 600, false),
            reference("d", 3, 700, 800, true),
        ];
        let predicted = vec![
            prediction("a", 0, 110, 210, true),  // 10 ms
            prediction("b", 1, 360, 460, false), // 60 ms
            prediction("c", 2, 590, 690, false), // 90 ms
            prediction("d", 3, 850, 950, true),  // 150 ms
        ];
        let result = score_alignment(1_000, 2_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.timing.onset_median_ms, Some(75.0));
        assert_eq!(result.timing.onset_within_50ms_fraction, Some(0.25));
        assert_eq!(result.timing.onset_within_100ms_fraction, Some(0.75));
        assert_eq!(result.timing.onset_p95_ms, Some(150.0));
    }

    #[test]
    fn onset_metrics_include_20ms_and_vowel_50ms_thresholds_and_duration_ratio() {
        let reference = vec![
            reference("a", 0, 100, 200, true),
            reference("b", 1, 300, 400, false),
            reference("e", 2, 500, 600, true),
            reference("s", 3, 700, 800, false),
        ];
        let predicted = vec![
            prediction("a", 0, 110, 220, true),  // 10 ms, duration ratio 1.1
            prediction("b", 1, 340, 440, false), // 40 ms, ratio 1.0
            prediction("e", 2, 560, 680, true),  // 60 ms, ratio 1.2
            prediction("s", 3, 830, 930, false), // 130 ms, ratio 1.0
        ];
        let result = score_alignment(1_000, 2_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.timing.onset_within_20ms_fraction, Some(0.25));
        assert_eq!(result.timing.onset_within_50ms_fraction, Some(0.5));
        assert_eq!(result.timing.vowel_onset_within_50ms_fraction, Some(0.5));
        assert_eq!(result.timing.onset_within_100ms_fraction, Some(0.75));
        assert!((result.timing.duration_ratio.unwrap() - 1.075).abs() < 1e-12);
    }

    #[test]
    fn median_and_threshold_coverage_use_exact_phone_pairs_only() {
        let reference = vec![
            reference("a", 0, 100, 200, true),
            reference("b", 1, 300, 400, false),
        ];
        let predicted = vec![
            prediction("a", 0, 120, 220, true),
            prediction("x", 1, 300, 400, false),
        ];
        let result = score_alignment(1_000, 1_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.timing.exact_phone_pairs, 1);
        assert_eq!(result.timing.onset_median_ms, Some(20.0));
        assert_eq!(result.timing.onset_within_50ms_fraction, Some(1.0));
        assert_eq!(result.timing.onset_within_100ms_fraction, Some(1.0));
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
    fn same_ipa_in_wrong_lexical_slot_is_not_an_exact_match() {
        let reference = vec![reference("a", 0, 100, 500, true)];
        let mut predicted = vec![prediction("a", 0, 100, 500, true)];
        predicted[0].word_index = 1;

        let result = score_alignment(16_000, 1_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.exact_matches, 1);
        assert_eq!(result.edits.substitutions, 0);
        assert_eq!(result.phone_error_rate, 0.0);
        assert_eq!(result.reference_phone_match_fraction, 1.0);
        assert_eq!(result.reference_identity_match_fraction, 0.0);
        assert_eq!(result.timing.joint_onset_within_20ms_fraction, 0.0);
        assert_eq!(result.timing.joint_onset_within_50ms_fraction, 0.0);
        assert_eq!(result.timing.joint_onset_within_100ms_fraction, 0.0);
        assert_eq!(result.timing.exact_phone_pairs, 0);
        assert_eq!(result.timing.onset_mae_ms, None);
    }

    #[test]
    fn same_ipa_in_wrong_syllable_is_not_an_exact_match() {
        let reference = vec![reference("a", 0, 100, 500, true)];
        let mut predicted = vec![prediction("a", 0, 100, 500, true)];
        predicted[0].syllable_index = 1;

        let result = score_alignment(16_000, 1_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.exact_matches, 1);
        assert_eq!(result.edits.substitutions, 0);
        assert_eq!(result.phone_error_rate, 0.0);
        assert_eq!(result.reference_identity_match_fraction, 0.0);
        assert_eq!(result.timing.onset_mae_ms, None);
    }

    #[test]
    fn same_ipa_with_wrong_vowel_class_matches_phone_symbol_but_not_strict_identity() {
        let reference = vec![reference("m", 0, 100, 500, false)];
        let predicted = vec![prediction("m", 0, 100, 500, true)];

        let result = score_alignment(16_000, 1_000, &reference, &predicted, &[]).unwrap();
        assert_eq!(result.edits.exact_matches, 1);
        assert_eq!(result.edits.substitutions, 0);
        assert_eq!(result.phone_error_rate, 0.0);
        assert_eq!(result.reference_phone_match_fraction, 1.0);
        assert_eq!(result.reference_identity_match_fraction, 0.0);
        assert_eq!(result.timing.exact_phone_pairs, 0);
        assert_eq!(result.timing.offset_mae_ms, None);
    }

    #[test]
    fn empty_prediction_is_a_full_deletion_not_a_vacuous_pass() {
        let reference = vec![reference("a", 0, 0, 100, true)];
        let result = score_alignment(16_000, 200, &reference, &[], &[]).unwrap();
        assert_eq!(result.edits.deletions, 1);
        assert_eq!(result.phone_error_rate, 1.0);
        assert_eq!(result.timing.onset_mae_ms, None);
        assert_eq!(result.timing.onset_median_ms, None);
        assert_eq!(result.timing.onset_within_20ms_fraction, None);
        assert_eq!(result.timing.onset_within_50ms_fraction, None);
        assert_eq!(result.timing.vowel_onset_within_50ms_fraction, None);
        assert_eq!(result.timing.onset_within_100ms_fraction, None);
        assert_eq!(result.timing.duration_ratio, None);
        assert_eq!(result.prediction_phone_precision, None);
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
        let result = score_alignment(24_000, 3_000, &reference, &predicted, &[melisma]).unwrap();
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
        assert_eq!(result.timing.onset_median_ms, None);
        assert_eq!(result.timing.onset_within_50ms_fraction, None);
        assert_eq!(result.timing.onset_within_100ms_fraction, None);
        assert_eq!(result.timing.offset_mae_ms, None);
        assert_eq!(result.melisma.coverage_fraction, None);
    }
}
