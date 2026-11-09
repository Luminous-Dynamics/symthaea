// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Objective diagnostics for dry vocal stems. Human ratings remain mandatory.

use serde::{Deserialize, Serialize};

use super::singing_engine::{VocalPerformance, VocalStem};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalQualityMetrics {
    pub voiced_frame_fraction: f32,
    pub voiced_unvoiced_error_rate: f32,
    pub median_pitch_error_cents: f32,
    pub p95_pitch_error_cents: f32,
    pub stable_pitch_p95_cents: f32,
    pub transition_pitch_p95_cents: f32,
    /// Score-independent 70-500 Hz autocorrelation cross-check. Diagnostic
    /// only until calibrated on human-sung reference recordings.
    pub score_independent_pitch_median_error_cents: f32,
    pub score_independent_pitch_p95_error_cents: f32,
    pub score_independent_pitch_valid_fraction: f32,
    /// Frames where the score-conditioned estimate appears reassuring while
    /// the global estimate finds a large pitch mismatch.
    pub pitch_prior_mismatch_frames: usize,
    /// None means the score-independent estimate did not have enough voiced
    /// support to qualify this diagnostic.
    pub score_independent_pitch_audit_pass: Option<bool>,
    pub median_onset_error_ms: f32,
    pub duration_error_ms: f32,
    pub rest_interval_count: usize,
    /// Rest RMS relative to voiced-note RMS in dB; None when the score has no rest.
    pub rest_leakage_db: Option<f32>,
    /// None means no rest interval was available to evaluate.
    pub rest_leakage_pass: Option<bool>,
    pub peak_sample_delta: f32,
    /// Peak first difference divided by whole-signal RMS. Reported for
    /// cross-level diagnostics; contextual click decisions remain local.
    pub normalized_peak_sample_delta: f32,
    pub contextual_click_events: usize,
    pub max_contextual_click_score: f32,
    /// RMS(first difference) / RMS(signal), a cheap broadband/fizz proxy.
    pub normalized_first_difference_rms: f32,
    pub clipped_sample_fraction: f32,
    pub dc_offset: f32,
    pub rms_dbfs: f32,
    pub finite: bool,
    pub pitch_tracking_pass: bool,
    pub transition_pitch_pass: bool,
    pub timing_pass: bool,
    pub physical_stability_pass: bool,
    pub render_cleanliness_pass: bool,
    pub objective_pass: bool,
}

/// Level-aware render-cleanliness diagnostics reusable outside a complete
/// scored [`VocalPerformance`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderCleanlinessMetrics {
    pub peak_sample_delta: f32,
    pub normalized_peak_sample_delta: f32,
    pub contextual_click_events: usize,
    pub max_contextual_click_score: f32,
    pub normalized_first_difference_rms: f32,
    pub rms_dbfs: f32,
    pub finite: bool,
    pub render_cleanliness_pass: bool,
}

/// Analyze waveform continuity without requiring notes, lyrics, or a target
/// performance. This uses the same frozen cleanliness bounds as the singing
/// quality report; callers must not reinterpret it as a naturalness gate.
pub fn analyze_render_cleanliness(samples: &[f32], sample_rate: u32) -> RenderCleanlinessMetrics {
    let peak_sample_delta = samples
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .fold(0.0, f32::max);
    let (contextual_click_events, max_contextual_click_score) =
        contextual_clicks(samples, sample_rate);
    let finite = samples.iter().all(|sample| sample.is_finite());
    let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len().max(1) as f32)
        .sqrt();
    let difference_rms = (samples
        .windows(2)
        .map(|window| (window[1] - window[0]).powi(2))
        .sum::<f32>()
        / samples.len().saturating_sub(1).max(1) as f32)
        .sqrt();
    let normalized_peak_sample_delta = peak_sample_delta / rms.max(1e-9);
    let normalized_first_difference_rms = difference_rms / rms.max(1e-9);
    let render_cleanliness_pass = finite
        && peak_sample_delta <= 0.12
        && contextual_click_events == 0
        && normalized_first_difference_rms <= 0.75;
    RenderCleanlinessMetrics {
        peak_sample_delta,
        normalized_peak_sample_delta,
        contextual_click_events,
        max_contextual_click_score,
        normalized_first_difference_rms,
        rms_dbfs: 20.0 * rms.max(1e-9).log10(),
        finite,
        render_cleanliness_pass,
    }
}

pub fn analyze_vocal_stem(performance: &VocalPerformance, stem: &VocalStem) -> VocalQualityMetrics {
    let rate = stem.sample_rate;
    let frame = (rate as f32 * 0.05) as usize;
    let hop = (rate as f32 * 0.02) as usize;
    let mut errors = Vec::new();
    let mut stable_errors = Vec::new();
    let mut transition_errors = Vec::new();
    let mut score_independent_errors = Vec::new();
    let mut pitch_prior_mismatch_frames = 0usize;
    let mut expected_voiced_frames = 0usize;
    let mut attempted = 0usize;
    let mut vuv_errors = 0usize;
    let mut all_frames = 0usize;
    if stem.samples.len() >= frame {
        for start in (0..=stem.samples.len() - frame).step_by(hop.max(1)) {
            let time = (start + frame / 2) as f32 / rate as f32;
            let expected_voiced = performance
                .syllables
                .iter()
                .flat_map(|s| s.notes())
                .any(|note| time >= note.start_time && time < note.start_time + note.duration);
            let expected_pitch = expected_pitch_at(performance, time);
            let window = &stem.samples[start..start + frame];

            // Keep the score-conditioned tracker for local note/transition
            // diagnostics, but audit it with a second estimator that does not
            // search around the expected pitch.
            let actual = if expected_voiced {
                expected_pitch.and_then(|pitch| estimate_f0_near(window, rate, pitch))
            } else {
                estimate_f0(window, rate)
            };
            all_frames += 1;
            if expected_voiced != actual.is_some() {
                vuv_errors += 1;
            }
            if expected_voiced {
                attempted += 1;
                expected_voiced_frames += 1;

                let score_independent = estimate_f0(window, rate);
                if let (Some(expected), Some(independent)) = (expected_pitch, score_independent) {
                    let independent_error =
                        (1200.0 * (independent / expected).log2()).abs();
                    score_independent_errors.push(independent_error);
                    if let Some(constrained) = actual {
                        let constrained_error =
                            (1200.0 * (constrained / expected).log2()).abs();
                        if independent_error > 400.0
                            && constrained_error < 350.0
                            && independent_error - constrained_error > 180.0
                        {
                            pitch_prior_mismatch_frames += 1;
                        }
                    }
                }

                if let (Some(expected), Some(actual)) = (expected_pitch, actual) {
                    let error = (1200.0 * (actual / expected).log2()).abs();
                    errors.push(error);
                    let stable = performance
                        .syllables
                        .iter()
                        .flat_map(|syllable| syllable.notes())
                        .find(|note| {
                            time >= note.start_time && time < note.start_time + note.duration
                        })
                        .is_some_and(|note| {
                            time >= note.start_time + 0.08
                                && time <= note.start_time + note.duration - 0.04
                        });
                    if stable {
                        stable_errors.push(error);
                    } else {
                        transition_errors.push(error);
                    }
                }
            }
        }
    }
    errors.sort_by(f32::total_cmp);
    let median_pitch_error_cents = percentile(&errors, 0.5);
    let p95_pitch_error_cents = percentile(&errors, 0.95);
    stable_errors.sort_by(f32::total_cmp);
    transition_errors.sort_by(f32::total_cmp);
    score_independent_errors.sort_by(f32::total_cmp);
    let stable_pitch_p95_cents = percentile(&stable_errors, 0.95);
    let transition_pitch_p95_cents = percentile(&transition_errors, 0.95);
    let score_independent_pitch_median_error_cents =
        percentile(&score_independent_errors, 0.5);
    let score_independent_pitch_p95_error_cents =
        percentile(&score_independent_errors, 0.95);
    let score_independent_pitch_valid_fraction =
        score_independent_errors.len() as f32 / expected_voiced_frames.max(1) as f32;
    let score_independent_pitch_audit_pass =
        (score_independent_pitch_valid_fraction >= 0.8)
            .then_some(pitch_prior_mismatch_frames == 0);
    // An onset-from-silence metric is meaningful for the first note and after
    // an actual gap, not at a connected legato boundary where audio continues.
    let mut previous_end = None;
    let mut onset_errors = Vec::new();
    for syllable in &performance.syllables {
        let start = syllable.note.start_time;
        let follows_gap = previous_end.is_none_or(|end: f32| start - end >= 0.04);
        if follows_gap && let Some(error) = onset_error_ms(&stem.samples, rate, start) {
            onset_errors.push(error);
        }
        previous_end = Some(syllable.end_time_s());
    }
    onset_errors.sort_by(f32::total_cmp);
    let median_onset_error_ms = percentile(&onset_errors, 0.5);
    let (rest_interval_count, rest_leakage_db, rest_leakage_pass) =
        rest_leakage_metrics(performance, &stem.samples, rate);
    let cleanliness = analyze_render_cleanliness(&stem.samples, rate);
    let peak_sample_delta = cleanliness.peak_sample_delta;
    let normalized_peak_sample_delta = cleanliness.normalized_peak_sample_delta;
    let contextual_click_events = cleanliness.contextual_click_events;
    let max_contextual_click_score = cleanliness.max_contextual_click_score;
    let finite = stem.samples.iter().all(|x| x.is_finite());
    let voiced_frame_fraction = errors.len() as f32 / attempted.max(1) as f32;
    let voiced_unvoiced_error_rate = vuv_errors as f32 / all_frames.max(1) as f32;
    let duration_error_ms =
        ((stem.samples.len() as f32 / rate as f32) - performance.duration_s()).abs() * 1000.0;
    let clipped_sample_fraction = stem.samples.iter().filter(|x| x.abs() >= 0.999).count() as f32
        / stem.samples.len().max(1) as f32;
    let dc_offset = stem.samples.iter().copied().sum::<f32>() / stem.samples.len().max(1) as f32;
    let rms =
        (stem.samples.iter().map(|x| x * x).sum::<f32>() / stem.samples.len().max(1) as f32).sqrt();
    let rms_dbfs = 20.0 * rms.max(1e-9).log10();
    let normalized_first_difference_rms = cleanliness.normalized_first_difference_rms;
    let pitch_tracking_pass = voiced_frame_fraction >= 0.8
        && voiced_unvoiced_error_rate <= 0.15
        && median_pitch_error_cents <= 25.0
        && stable_pitch_p95_cents <= 40.0;
    // Transitions follow the performance's intended pitch curve but permit
    // portamento/onset acquisition unavailable to a steady-state gate. Keep a
    // separate, explicit bound rather than hiding them inside whole-phrase p95.
    let transition_pitch_pass = transition_pitch_p95_cents <= 150.0;
    let timing_pass = median_onset_error_ms <= 30.0 && duration_error_ms <= 80.0;
    let physical_stability_pass =
        finite && clipped_sample_fraction <= 0.0001 && dc_offset.abs() <= 0.01;
    // A large isolated jump is a click even when no sample clips. The
    // contextual detector additionally requires a robust local outlier, so
    // sustained frication is not rejected merely for being noisy.
    let render_cleanliness_pass = cleanliness.render_cleanliness_pass;
    let objective_pass = pitch_tracking_pass
        && transition_pitch_pass
        && timing_pass
        && physical_stability_pass
        && render_cleanliness_pass
        && rest_leakage_pass != Some(false)
        // Raw physical renders intentionally retain diagnostic headroom. The
        // gallery emits a separate loudness-normalized listening copy, so a
        // clean quiet source must not fail the synthesis-quality gate merely
        // for staying below a mix/master target.
        && rms_dbfs >= -45.0
        && rms_dbfs <= -8.0;
    VocalQualityMetrics {
        voiced_frame_fraction,
        voiced_unvoiced_error_rate,
        median_pitch_error_cents,
        p95_pitch_error_cents,
        stable_pitch_p95_cents,
        transition_pitch_p95_cents,
        score_independent_pitch_median_error_cents,
        score_independent_pitch_p95_error_cents,
        score_independent_pitch_valid_fraction,
        pitch_prior_mismatch_frames,
        score_independent_pitch_audit_pass,
        median_onset_error_ms,
        duration_error_ms,
        rest_interval_count,
        rest_leakage_db,
        rest_leakage_pass,
        peak_sample_delta,
        normalized_peak_sample_delta,
        contextual_click_events,
        max_contextual_click_score,
        normalized_first_difference_rms,
        clipped_sample_fraction,
        dc_offset,
        rms_dbfs,
        finite,
        pitch_tracking_pass,
        transition_pitch_pass,
        timing_pass,
        physical_stability_pass,
        render_cleanliness_pass,
        objective_pass,
    }
}

fn rest_leakage_metrics(
    performance: &VocalPerformance,
    samples: &[f32],
    sample_rate: u32,
) -> (usize, Option<f32>, Option<bool>) {
    let notes: Vec<_> = performance
        .syllables
        .iter()
        .flat_map(|syllable| syllable.notes())
        .collect();
    if notes.is_empty() || sample_rate == 0 {
        return (0, None, None);
    }

    let mut rest_intervals = Vec::new();
    let first_start = notes[0].start_time.max(0.0);
    if first_start > 0.005 {
        rest_intervals.push((0.0, first_start));
    }
    for pair in notes.windows(2) {
        let previous_end = pair[0].start_time + pair[0].duration;
        let gap_start = previous_end.max(0.0);
        let gap_end = pair[1].start_time.max(0.0);
        if gap_end - gap_start > 0.005 {
            rest_intervals.push((gap_start, gap_end));
        }
    }
    if rest_intervals.is_empty() {
        return (0, None, None);
    }

    let mut rest_energy = 0.0f64;
    let mut rest_count = 0usize;
    for (start_s, end_s) in &rest_intervals {
        let start = ((*start_s * sample_rate as f32).floor() as usize).min(samples.len());
        let end = ((*end_s * sample_rate as f32).ceil() as usize).min(samples.len());
        for sample in &samples[start..end.max(start)] {
            rest_energy += (*sample as f64) * (*sample as f64);
            rest_count += 1;
        }
    }

    let mut voiced_energy = 0.0f64;
    let mut voiced_count = 0usize;
    for note in notes {
        let start =
            ((note.start_time.max(0.0) * sample_rate as f32).floor() as usize).min(samples.len());
        let end = (((note.start_time + note.duration).max(0.0) * sample_rate as f32).ceil()
            as usize)
            .min(samples.len());
        for sample in &samples[start..end.max(start)] {
            voiced_energy += (*sample as f64) * (*sample as f64);
            voiced_count += 1;
        }
    }

    if rest_count == 0 || voiced_count == 0 {
        return (rest_intervals.len(), None, None);
    }
    let rest_rms = (rest_energy / rest_count as f64).sqrt();
    let voiced_rms = (voiced_energy / voiced_count as f64).sqrt();
    let leakage_db = (20.0 * (rest_rms / voiced_rms.max(1e-12)).max(1e-12).log10()) as f32;
    let pass = leakage_db <= -35.0;
    (rest_intervals.len(), Some(leakage_db), Some(pass))
}

fn contextual_clicks(samples: &[f32], sample_rate: u32) -> (usize, f32) {
    if samples.len() < 3 {
        return (0, 0.0);
    }
    let differences: Vec<_> = samples
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .collect();
    let block = ((sample_rate as f32 * 0.01).round() as usize).max(32);
    let mut events = 0;
    let mut max_score = 0.0f32;
    let mut last_event = None;
    for (block_index, chunk) in differences.chunks(block).enumerate() {
        let mut ordered = chunk.to_vec();
        ordered.sort_by(f32::total_cmp);
        let median = ordered[ordered.len() / 2];
        let mut deviations: Vec<_> = chunk.iter().map(|value| (value - median).abs()).collect();
        deviations.sort_by(f32::total_cmp);
        let mad = deviations[deviations.len() / 2];
        let start = block_index * block;
        let local_rms = (samples[start..(start + chunk.len() + 1).min(samples.len())]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / (chunk.len() + 1) as f32)
            .sqrt();
        for (offset, &delta) in chunk.iter().enumerate() {
            let score = (delta - median).max(0.0) / (1.4826 * mad + 1e-4);
            max_score = max_score.max(score);
            if delta > 0.035 && delta > local_rms * 4.0 && score > 12.0 {
                let index = start + offset;
                if last_event.is_none_or(|previous| index - previous > 16) {
                    events += 1;
                }
                last_event = Some(index);
            }
        }
    }
    (events, max_score)
}

fn estimate_f0_near(x: &[f32], rate: u32, expected: f32) -> Option<f32> {
    let rms = (x.iter().map(|value| value * value).sum::<f32>() / x.len().max(1) as f32).sqrt();
    if rms < 0.003 || !expected.is_finite() || expected <= 0.0 {
        return None;
    }
    let ratio = 2.0f32.powf(300.0 / 1200.0); // ±300 cents
    let min_lag = (rate as f32 / (expected * ratio)).floor().max(1.0) as usize;
    let max_lag = (rate as f32 / (expected / ratio)).ceil() as usize;
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in min_lag..=max_lag.min(x.len() / 2) {
        let (dot, energy_a, energy_b) = x[..x.len() - lag].iter().zip(&x[lag..]).fold(
            (0.0f64, 0.0f64, 0.0f64),
            |(dot, a2, b2), (&a, &b)| {
                (
                    dot + (a * b) as f64,
                    a2 + (a * a) as f64,
                    b2 + (b * b) as f64,
                )
            },
        );
        let score = dot / (energy_a * energy_b).sqrt().max(1e-12);
        if score > best.1 {
            best = (lag, score);
        }
    }
    (best.0 > 0 && best.1 >= 0.12).then(|| rate as f32 / best.0 as f32)
}

fn expected_pitch_at(performance: &VocalPerformance, time: f32) -> Option<f32> {
    performance.target_pitch_at(time)
}

fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    values[((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize]
}

fn onset_error_ms(samples: &[f32], rate: u32, expected_s: f32) -> Option<f32> {
    let expected = (expected_s * rate as f32) as usize;
    let radius = (0.12 * rate as f32) as usize;
    let from = expected.saturating_sub(radius);
    let to = (expected + radius).min(samples.len());
    if to <= from {
        return None;
    }
    let peak = samples[from..to]
        .iter()
        .fold(0.0f32, |p, &x| p.max(x.abs()));
    let threshold = (peak * 0.12).max(0.002);
    let actual = samples[from..to]
        .iter()
        .position(|x| x.abs() >= threshold)?
        + from;
    Some((actual as f32 - expected as f32).abs() * 1000.0 / rate as f32)
}

fn estimate_f0(x: &[f32], rate: u32) -> Option<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    if rms < 0.003 {
        return None;
    }
    // Singing fundamentals can fall well outside speech-oriented 70-500 Hz
    // ranges. Keep the score-independent audit broad enough for low and high
    // registers while leaving the score-conditioned tracker as a separate
    // measurement path.
    let min_lag = (rate as f32 / 1_200.0).floor().max(1.0) as usize;
    let max_lag = (rate as usize / 40).min(x.len() / 2);
    if max_lag < min_lag {
        return None;
    }
    let mut scores = Vec::with_capacity(max_lag - min_lag + 1);
    let mut best_score = 0.0f64;
    for lag in min_lag..=max_lag {
        let (dot, ea, eb) = x[..x.len() - lag]
            .iter()
            .zip(&x[lag..])
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, a2, b2), (&a, &b)| {
                (d + (a * b) as f64, a2 + (a * a) as f64, b2 + (b * b) as f64)
            });
        let score = dot / (ea * eb).sqrt().max(1e-12);
        best_score = best_score.max(score);
        scores.push((lag, score));
    }
    if best_score < 0.25 {
        return None;
    }
    // A periodic source has strong peaks at integer multiples of its period.
    // Select the earliest credible local peak rather than the global peak,
    // which otherwise tends to report octave/sub-octave errors.
    let threshold = (best_score * 0.85).max(0.25);
    let selected = scores
        .windows(3)
        .find(|window| {
            window[1].1 >= threshold && window[1].1 >= window[0].1 && window[1].1 >= window[2].1
        })
        .map(|window| window[1].0)
        .or_else(|| {
            scores
                .iter()
                .find(|(_, score)| *score >= threshold)
                .map(|(lag, _)| *lag)
        })?;
    Some(rate as f32 / selected as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_cleanliness_accepts_continuous_tone_and_rejects_impulse() {
        let rate = 48_000;
        let mut samples: Vec<_> = (0..rate)
            .map(|index| (std::f32::consts::TAU * 220.0 * index as f32 / rate as f32).sin() * 0.1)
            .collect();
        let clean = analyze_render_cleanliness(&samples, rate);
        assert!(clean.render_cleanliness_pass, "{clean:?}");
        assert_eq!(clean.contextual_click_events, 0);

        samples[rate as usize / 2] += 0.8;
        let clicked = analyze_render_cleanliness(&samples, rate);
        assert!(!clicked.render_cleanliness_pass, "{clicked:?}");
        assert!(clicked.contextual_click_events > 0);
    }

    #[test]
    fn rest_leakage_is_measured_relative_to_voiced_material() {
        let notes = [
            symthaea_muse::Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.8,
            },
            symthaea_muse::Note {
                frequency: 220.0,
                start_time: 0.75,
                duration: 0.5,
                velocity: 0.8,
            },
        ];
        let performance = VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
        let rate = 24_000;
        let samples: Vec<_> = (0..rate * 5 / 4)
            .map(|index| {
                let time = index as f32 / rate as f32;
                if (time < 0.5) || (0.75..1.25).contains(&time) {
                    (std::f32::consts::TAU * 220.0 * time).sin() * 0.2
                } else {
                    0.0
                }
            })
            .collect();
        let (intervals, leakage_db, pass) = rest_leakage_metrics(&performance, &samples, rate);
        assert_eq!(intervals, 1);
        assert!(leakage_db.unwrap() <= -35.0);
        assert_eq!(pass, Some(true));

        let leaked: Vec<_> = (0..rate * 5 / 4)
            .map(|index| {
                let time = index as f32 / rate as f32;
                if (time < 0.5) || (0.75..1.25).contains(&time) {
                    (std::f32::consts::TAU * 220.0 * time).sin() * 0.2
                } else {
                    (std::f32::consts::TAU * 220.0 * time).sin() * 0.02
                }
            })
            .collect();
        let (_, leaked_db, leaked_pass) = rest_leakage_metrics(&performance, &leaked, rate);
        assert!(leaked_db.unwrap() > -35.0);
        assert_eq!(leaked_pass, Some(false));
    }

    #[test]
    fn no_rest_has_undefined_rest_leakage_not_a_fake_zero() {
        let note = symthaea_muse::Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let (intervals, leakage_db, pass) =
            rest_leakage_metrics(&performance, &[0.0; 12_000], 24_000);
        assert_eq!(intervals, 0);
        assert_eq!(leakage_db, None);
        assert_eq!(pass, None);
    }

    #[test]
    fn perfect_sine_tracks_note() {
        let note = symthaea_muse::Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let rate = 24_000;
        let samples = (0..rate / 2)
            .map(|i| (std::f32::consts::TAU * 220.0 * i as f32 / rate as f32).sin() * 0.2)
            .collect();
        let stem = VocalStem {
            samples,
            sample_rate: rate,
            backend: "test".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        assert!(metrics.median_pitch_error_cents < 15.0, "{:?}", metrics);
    }

    /// A score prior can make an incorrect pitch look close by constraining
    #[test]
    fn score_independent_pitch_audit_detects_prior_induced_false_reassurance() {
        let note = symthaea_muse::Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let rate = 24_000;
        let actual_hz: f32 = 500.0;
        let samples = (0..rate / 2)
            .map(|i| (std::f32::consts::TAU * actual_hz * i as f32 / rate as f32).sin() * 0.2)
            .collect();
        let stem = VocalStem {
            samples,
            sample_rate: rate,
            backend: "test".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        let true_error_cents = 1200.0f32 * (actual_hz / note.frequency).log2();

        assert!(
            metrics.median_pitch_error_cents < 400.0,
            "the constrained estimate still illustrates the old prior-limited result: {metrics:?}"
        );
        assert!(
            (metrics.score_independent_pitch_median_error_cents - true_error_cents).abs() < 5.0,
            "the global estimate should recover the large error: {metrics:?}"
        );
        assert!(
            metrics.pitch_prior_mismatch_frames > 0,
            "the evaluator must explicitly flag false reassurance: {metrics:?}"
        );
        assert_eq!(metrics.score_independent_pitch_audit_pass, Some(false));
        assert!(
            metrics.score_independent_pitch_valid_fraction >= 0.8,
            "the score-independent diagnostic should have adequate frame coverage: {metrics:?}"
        );
    }
    #[test]
    fn unconstrained_pitch_estimator_covers_low_and_high_singing_registers() {
        let rate = 24_000;
        for expected_hz in [80.0f32, 800.0f32] {
            let samples: Vec<_> = (0..rate / 20)
                .map(|index| {
                    (std::f32::consts::TAU * expected_hz * index as f32 / rate as f32)
                        .sin()
                        * 0.2
                })
                .collect();
            let measured = estimate_f0(&samples, rate).expect("periodic tone must be detected");
            assert!(
                (measured - expected_hz).abs() < 2.0,
                "expected {expected_hz} Hz, observed {measured} Hz"
            );
        }
    }

}
