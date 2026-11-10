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
    /// Error from a broad 40-1200 Hz search that does not use the score pitch.
    pub score_independent_median_pitch_error_cents: f32,
    pub score_independent_p95_pitch_error_cents: f32,
    /// Independent F0 support divided by frames accepted by the score-conditioned tracker.
    pub score_independent_valid_fraction: f32,
    /// Count of matched frames whose broad-search pitch differs by more than 300 cents.
    pub prior_mismatch_frames: usize,
    pub prior_mismatch_fraction: f32,
    /// None means insufficient independent coverage; Some(false) means a material mismatch.
    pub score_independent_pitch_audit_pass: Option<bool>,
    /// Second-family YIN difference-function estimate; not conditioned on score pitch.
    pub yin_median_pitch_error_cents: f32,
    pub yin_p95_pitch_error_cents: f32,
    pub yin_valid_fraction: f32,
    pub yin_median_confidence: f32,
    pub yin_pitch_audit_pass: Option<bool>,
    pub median_onset_error_ms: f32,
    pub duration_error_ms: f32,
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
    let mut attempted = 0usize;
    let mut vuv_errors = 0usize;
    let mut all_frames = 0usize;
    let mut score_conditioned_matches = 0usize;
    let mut score_independent_matches = 0usize;
    let mut score_independent_errors = Vec::new();
    let mut prior_mismatch_frames = 0usize;
    let mut yin_matches = 0usize;
    let mut yin_errors = Vec::new();
    let mut yin_confidences = Vec::new();
    let mut yin_mismatch_frames = 0usize;
    if stem.samples.len() >= frame {
        for start in (0..=stem.samples.len() - frame).step_by(hop.max(1)) {
            let time = (start + frame / 2) as f32 / rate as f32;
            let expected_voiced = performance
                .syllables
                .iter()
                .flat_map(|s| s.notes())
                .any(|note| time >= note.start_time && time < note.start_time + note.duration);
            let expected_pitch = expected_pitch_at(performance, time);
            // Conditional singing evaluation has the score available. Use it
            // as the octave/continuity prior, as production pitch trackers do,
            // while still estimating the actual local autocorrelation peak.
            let actual = if expected_voiced {
                expected_pitch.and_then(|pitch| {
                    estimate_f0_near(&stem.samples[start..start + frame], rate, pitch)
                })
            } else {
                estimate_f0(&stem.samples[start..start + frame], rate)
            };
            all_frames += 1;
            if expected_voiced != actual.is_some() {
                vuv_errors += 1;
            }
            if expected_voiced {
                attempted += 1;
                if let (Some(expected), Some(actual)) = (expected_pitch, actual) {
                    let error = (1200.0 * (actual / expected).log2()).abs();
                    errors.push(error);
                    score_conditioned_matches += 1;

                    // This search deliberately ignores the expected score pitch.
                    // It is still autocorrelation-based, so it is a broad-prior
                    // cross-check rather than a fully independent algorithm family.
                    let frame_samples = &stem.samples[start..start + frame];
                    if let Some(independent) = estimate_f0_broad(frame_samples, rate) {
                        score_independent_matches += 1;
                        let independent_error =
                            (1200.0 * (independent / expected).log2()).abs();
                        score_independent_errors.push(independent_error);
                        if independent_error > 300.0 {
                            prior_mismatch_frames += 1;
                        }
                    }
                    // YIN uses a normalized difference function rather than the
                    // autocorrelation peak used by both score-conditioned and
                    // broad-range estimators.
                    if let Some((yin_hz, confidence)) = estimate_f0_yin(frame_samples, rate) {
                        yin_matches += 1;
                        let yin_error = (1200.0 * (yin_hz / expected).log2()).abs();
                        yin_errors.push(yin_error);
                        yin_confidences.push(confidence);
                        if yin_error > 300.0 {
                            yin_mismatch_frames += 1;
                        }
                    }

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
    let stable_pitch_p95_cents = percentile(&stable_errors, 0.95);
    let transition_pitch_p95_cents = percentile(&transition_errors, 0.95);
    score_independent_errors.sort_by(f32::total_cmp);
    let score_independent_median_pitch_error_cents = percentile(&score_independent_errors, 0.5);
    let score_independent_p95_pitch_error_cents = percentile(&score_independent_errors, 0.95);
    let score_independent_valid_fraction = score_independent_matches as f32
        / score_conditioned_matches.max(1) as f32;
    let prior_mismatch_fraction = prior_mismatch_frames as f32
        / score_independent_matches.max(1) as f32;
    // Insufficient independent support is undefined, never a zero-error pass.
    // 80% coverage and a 10% gross-mismatch limit are provisional engineering
    // guards that require calibration on held-out human-sung references.
    let score_independent_pitch_audit_pass =
        if score_independent_valid_fraction < 0.8 || score_independent_matches == 0 {
            None
        } else {
            Some(prior_mismatch_fraction <= 0.10)
        };
    yin_errors.sort_by(f32::total_cmp);
    yin_confidences.sort_by(f32::total_cmp);
    let yin_median_pitch_error_cents = percentile(&yin_errors, 0.5);
    let yin_p95_pitch_error_cents = percentile(&yin_errors, 0.95);
    let yin_valid_fraction = yin_matches as f32 / score_conditioned_matches.max(1) as f32;
    let yin_median_confidence = percentile(&yin_confidences, 0.5);
    let yin_mismatch_fraction = yin_mismatch_frames as f32 / yin_matches.max(1) as f32;
    let yin_pitch_audit_pass = if yin_valid_fraction < 0.8 || yin_matches == 0 {
        None
    } else {
        Some(yin_mismatch_fraction <= 0.10)
    };
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
        && score_independent_pitch_audit_pass == Some(true)
        && yin_pitch_audit_pass == Some(true)
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
        score_independent_median_pitch_error_cents,
        score_independent_p95_pitch_error_cents,
        score_independent_valid_fraction,
        prior_mismatch_frames,
        prior_mismatch_fraction,
        score_independent_pitch_audit_pass,
        yin_median_pitch_error_cents,
        yin_p95_pitch_error_cents,
        yin_valid_fraction,
        yin_median_confidence,
        yin_pitch_audit_pass,
        median_onset_error_ms,
        duration_error_ms,
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

/// Three-tap binomial low-pass followed by 2:1 decimation. This is used
/// before broad pitch searches to reduce lag work while keeping 1200 Hz well
/// below the decimated Nyquist limit at normal audio sample rates.
fn lowpass_decimate_by_two(x: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(x.len().div_ceil(2));
    for center in (0..x.len()).step_by(2) {
        let left = if center == 0 { x[0] } else { x[center - 1] };
        let middle = x[center];
        let right = x.get(center + 1).copied().unwrap_or(middle);
        output.push((left + 2.0 * middle + right) * 0.25);
    }
    output
}

/// Broad-range autocorrelation cross-check that deliberately does not use
/// the expected note. The 40-1200 Hz range covers common low/high singing
/// fundamentals and permits an octave-error detector to observe a true pitch
/// outside the score-conditioned estimator's +/-300-cent search window.
///
/// This shares the autocorrelation algorithm family with the score-conditioned
/// estimator. It addresses prior-window bias, not correlated estimator error.
fn estimate_f0_broad(x: &[f32], rate: u32) -> Option<f32> {
    if x.len() < 8 || rate < 80 {
        return None;
    }
    let downsampled = lowpass_decimate_by_two(x);
    let effective_rate = rate as f64 / 2.0;
    let rms = (downsampled.iter().map(|value| value * value).sum::<f32>()
        / downsampled.len().max(1) as f32)
        .sqrt();
    if rms < 0.003 {
        return None;
    }
    let min_lag = (effective_rate / 1200.0).floor().max(1.0) as usize;
    let max_lag = ((effective_rate / 40.0).ceil() as usize).min(downsampled.len() / 2);
    if min_lag > max_lag {
        return None;
    }

    let mut scores = Vec::with_capacity(max_lag - min_lag + 1);
    let mut best_score = f64::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let (dot, energy_a, energy_b) = downsampled[..downsampled.len() - lag]
            .iter()
            .zip(&downsampled[lag..])
            .fold((0.0f64, 0.0f64, 0.0f64), |(dot, a2, b2), (&a, &b)| {
                (
                    dot + (a * b) as f64,
                    a2 + (a * a) as f64,
                    b2 + (b * b) as f64,
                )
            });
        let score = dot / (energy_a * energy_b).sqrt().max(1e-12);
        best_score = best_score.max(score);
        scores.push((lag, score));
    }
    if best_score < 0.25 {
        return None;
    }

    let threshold = (best_score * 0.85).max(0.25);
    let selected = scores
        .windows(3)
        .find(|window| {
            window[1].1 >= threshold
                && window[1].1 >= window[0].1
                && window[1].1 >= window[2].1
        })
        .map(|window| window[1].0)
        .or_else(|| {
            scores
                .iter()
                .find(|(_, score)| *score >= threshold)
                .map(|(lag, _)| *lag)
        })?;
    Some((effective_rate / selected as f64) as f32)
}

/// YIN-style F0 estimate using the difference function and cumulative
/// mean normalized difference (CMNDF). Returns (frequency_hz, confidence).
///
/// Two cascaded 3-tap smoothing/2:1 stages bound evaluation cost while
/// retaining the 40-1200 Hz fundamental range. Confidence is
/// 1-CMNDF at the selected trough. This is a second estimator family relative
/// to the autocorrelation peak search, not an independent reference signal.
fn estimate_f0_yin(x: &[f32], rate: u32) -> Option<(f32, f32)> {
    if x.len() < 8 || rate < 80 {
        return None;
    }
    // Cascaded anti-aliased 2:1 stages yield a 4:1 reduction overall.
    // Two stages preserve the target range while keeping this O(N * lag)
    // estimator cheap enough for the offline comparison gate.
    let first_stage = lowpass_decimate_by_two(x);
    let downsampled = lowpass_decimate_by_two(&first_stage);
    let effective_rate = rate as f64 / 4.0;
    let rms = (downsampled.iter().map(|value| value * value).sum::<f32>()
        / downsampled.len().max(1) as f32)
        .sqrt();
    if rms < 0.003 {
        return None;
    }

    let min_lag = (effective_rate / 1200.0).floor().max(1.0) as usize;
    let max_lag = ((effective_rate / 40.0).ceil() as usize).min(downsampled.len() - 1);
    if min_lag > max_lag {
        return None;
    }

    let mut difference = vec![0.0f64; max_lag + 1];
    for lag in 1..=max_lag {
        let mut sum = 0.0f64;
        for index in 0..downsampled.len() - lag {
            let delta = (downsampled[index] - downsampled[index + lag]) as f64;
            sum += delta * delta;
        }
        difference[lag] = sum;
    }

    let mut cmndf = vec![1.0f64; max_lag + 1];
    let mut cumulative = 0.0f64;
    for lag in 1..=max_lag {
        cumulative += difference[lag];
        if cumulative > f64::EPSILON {
            cmndf[lag] = difference[lag] * lag as f64 / cumulative;
        }
    }

    const CMNDF_THRESHOLD: f64 = 0.10;
    let mut selected = None;
    let mut lag = min_lag;
    while lag <= max_lag {
        if cmndf[lag] < CMNDF_THRESHOLD {
            while lag < max_lag && cmndf[lag + 1] < cmndf[lag] {
                lag += 1;
            }
            selected = Some(lag);
            break;
        }
        lag += 1;
    }
    let selected = selected.or_else(|| {
        let candidate =
            (min_lag..=max_lag).min_by(|&left, &right| cmndf[left].total_cmp(&cmndf[right]))?;
        (cmndf[candidate] <= 0.25).then_some(candidate)
    })?;

    let mut refined_lag = selected as f64;
    if selected > min_lag && selected < max_lag {
        let left = cmndf[selected - 1];
        let centre = cmndf[selected];
        let right = cmndf[selected + 1];
        let denominator = left - 2.0 * centre + right;
        if denominator.abs() > 1e-12 {
            let offset = (0.5 * (left - right) / denominator).clamp(-1.0, 1.0);
            refined_lag += offset;
        }
    }
    if refined_lag <= 0.0 {
        return None;
    }
    let frequency = (effective_rate / refined_lag) as f32;
    let confidence = (1.0 - cmndf[selected]).clamp(0.0, 1.0) as f32;
    (frequency.is_finite() && (40.0..=1200.0).contains(&frequency))
        .then_some((frequency, confidence))
}

fn expected_pitch_at(performance: &VocalPerformance, time: f32) -> Option<f32> {
    let points = &performance.pitch_curve;
    if points.is_empty() {
        return None;
    }
    for pair in points.windows(2) {
        if time >= pair[0].time_s && time <= pair[1].time_s {
            let span = pair[1].time_s - pair[0].time_s;
            if span <= f32::EPSILON {
                return Some(pair[1].frequency_hz);
            }
            let t = ((time - pair[0].time_s) / span).clamp(0.0, 1.0);
            return Some(pair[0].frequency_hz + (pair[1].frequency_hz - pair[0].frequency_hz) * t);
        }
    }
    performance
        .syllables
        .iter()
        .flat_map(|s| s.notes())
        .find(|note| time >= note.start_time && time < note.start_time + note.duration)
        .map(|note| note.frequency)
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
    let min_lag = rate as usize / 500;
    let max_lag = (rate as usize / 70).min(x.len() / 2);
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

    /// `estimate_f0_near`'s octave-safety window is fixed at +/-300 cents
    /// (line: `let ratio = 2.0f32.powf(300.0 / 1200.0)`). When the actual
    /// pitch is further from the expected note than that, the search can
    /// only find its best match at the window's edge -- so the reported
    /// error saturates near the window boundary rather than reflecting the
    /// true (potentially much larger) mistracking. This reproduces that
    /// saturation with a sine wave synthesized nearly an octave away from
    /// the note it is meant to represent, and confirms the reported error
    /// sits near the window edge (~300-320 cents), not the true ~1421 cents.
    #[test]
    fn score_independent_pitch_detects_octave_subharmonic_false_reassurance() {
        let note = symthaea_muse::Note {
            frequency: 240.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let rate = 24_000;
        let actual_hz = 480.0;
        let samples: Vec<_> = (0..rate / 2)
            .map(|index| {
                (std::f32::consts::TAU * actual_hz * index as f32 / rate as f32).sin() * 0.2
            })
            .collect();
        let stem = VocalStem {
            samples,
            sample_rate: rate,
            backend: "octave-negative-control".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        assert!(
            metrics.median_pitch_error_cents < 100.0,
            "score-conditioned estimate should demonstrate the prior-window false reassurance: {metrics:?}"
        );
        assert!(
            metrics.score_independent_median_pitch_error_cents > 1100.0,
            "broad search should recover the 480 Hz fundamental: {metrics:?}"
        );
        assert!(metrics.prior_mismatch_frames > 0, "{metrics:?}");
        assert!(metrics.prior_mismatch_fraction > 0.1, "{metrics:?}");
        assert_eq!(metrics.score_independent_pitch_audit_pass, Some(false));
        assert!(!metrics.objective_pass);
    }

    #[test]
    fn yin_crosscheck_detects_octave_subharmonic_false_reassurance() {
        let note = symthaea_muse::Note {
            frequency: 240.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let rate = 24_000;
        let actual_hz = 480.0;
        let samples: Vec<_> = (0..rate / 2)
            .map(|index| {
                (std::f32::consts::TAU * actual_hz * index as f32 / rate as f32).sin() * 0.2
            })
            .collect();
        let stem = VocalStem {
            samples,
            sample_rate: rate,
            backend: "yin-octave-negative-control".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        assert!(
            metrics.yin_median_pitch_error_cents > 1100.0,
            "YIN should identify the true 480 Hz fundamental: {metrics:?}"
        );
        assert!(metrics.yin_median_confidence > 0.9, "{metrics:?}");
        assert_eq!(metrics.yin_pitch_audit_pass, Some(false));
        assert!(!metrics.objective_pass);
    }

    #[test]
    fn yin_retains_fundamental_when_upper_harmonics_dominate() {
        let rate = 24_000;
        let fundamental = 220.0f32;
        let samples: Vec<_> = (0..rate / 20)
            .map(|index| {
                let time = index as f32 / rate as f32;
                0.02 * (std::f32::consts::TAU * fundamental * time).sin()
                    + 0.20 * (std::f32::consts::TAU * fundamental * 2.0 * time).sin()
                    + 0.12 * (std::f32::consts::TAU * fundamental * 3.0 * time).sin()
                    + 0.04 * (std::f32::consts::TAU * fundamental * 4.0 * time).sin()
            })
            .collect();
        let (estimated, confidence) = estimate_f0_yin(&samples, rate).unwrap();
        let error_cents = (1200.0 * (estimated / fundamental).log2()).abs();
        assert!(error_cents < 25.0, "estimate={estimated}, error={error_cents}");
        assert!(confidence > 0.9, "confidence={confidence}");
    }

    #[test]
    fn score_independent_pitch_audit_is_undefined_without_acoustic_support() {
        let note = symthaea_muse::Note {
            frequency: 240.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let stem = VocalStem {
            samples: vec![0.0; 12_000],
            sample_rate: 24_000,
            backend: "silence-negative-control".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        assert_eq!(metrics.score_independent_valid_fraction, 0.0);
        assert_eq!(metrics.score_independent_pitch_audit_pass, None);
        assert!(!metrics.objective_pass);
    }

    #[test]
    fn broad_pitch_search_covers_low_and_high_singing_registers() {
        let rate = 24_000;
        for target_hz in [80.0f32, 800.0] {
            let samples: Vec<_> = (0..rate / 2)
                .map(|index| {
                    (std::f32::consts::TAU * target_hz * index as f32 / rate as f32).sin() * 0.2
                })
                .collect();
            let estimated = estimate_f0_broad(&samples, rate).unwrap();
            let error_cents = (1200.0 * (estimated / target_hz).log2()).abs();
            assert!(
                error_cents < 20.0,
                "target={target_hz} estimated={estimated} error={error_cents}"
            );
        }
    }

    #[test]
    fn pitch_error_saturates_at_octave_safety_window_edge() {
        let note = symthaea_muse::Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let rate = 24_000;
        let actual_hz: f32 = 500.0; // ~1421 cents away from the 220 Hz target
        let samples = (0..rate / 2)
            .map(|i| (std::f32::consts::TAU * actual_hz * i as f32 / rate as f32).sin() * 0.2)
            .collect();
        let stem = VocalStem {
            samples,
            sample_rate: rate,
            backend: "test".into(),
        };
        let metrics = analyze_vocal_stem(&performance, &stem);
        let true_error_cents = 1200.0f64 * (actual_hz as f64 / note.frequency as f64).log2();
        assert!(
            metrics.median_pitch_error_cents < 400.0
                && metrics.median_pitch_error_cents < true_error_cents as f32 - 500.0,
            "expected the reported error to saturate well below the true \
             {true_error_cents:.1} cent mistracking, got {:?}",
            metrics
        );
    }
}
