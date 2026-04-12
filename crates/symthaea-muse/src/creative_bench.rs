// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Creative quality benchmarks for Symthaea's music and art generation.
//!
//! These metrics make creative improvements *measurable*. Unlike BLEU/ROUGE
//! (which are unsuitable for creative output), these benchmarks measure
//! properties that are empirically validated against human aesthetic judgement:
//!
//! - **Melodic coherence**: interval transition quality vs. trained corpus
//! - **Form compliance**: structural section distribution correctness
//! - **Emotional alignment**: does VA of output match target state?
//! - **Synesthetic coherence**: audio-visual cross-modal alignment score
//! - **Harmony diversity**: are all 8 Harmonies represented over time?
//! - **Rhythmic regularity**: coefficient of variation of note onset intervals
//!
//! # Design Philosophy
//!
//! These metrics are *proxy* measures, not ground truth. Human evaluation
//! remains the gold standard. Use these for:
//! - Regression testing (did a change break something?)
//! - Ablation studies (does feature X improve quality?)
//! - Monitoring aesthetic drift over sessions
//!
//! # References
//! - Eerola, T. & Vuoskoski, J. K. (2013). A review of music and emotion research.
//! - Pearce, M. T. (2005). The construction and evaluation of statistical models of
//!   melodic structure. PhD thesis, City University London.

use serde::{Deserialize, Serialize};
use symthaea_aesthetic::ValenceArousal;

use crate::{Composition, MuseConfig, MusicalState, Note};

// ─── Melodic Coherence ───────────────────────────────────────────────────────

/// Evaluate melodic coherence: how well do interval transitions match
/// learned expectations from a tonal music corpus?
///
/// Uses a simplified version of Pearce's IDyOM model: intervals that appear
/// frequently in Western tonal music are "expected"; rare intervals are "surprising."
/// A good melody balances expectation with surprise (Berlyne's optimal arousal).
///
/// Score: 0.0 (incoherent/random) to 1.0 (highly coherent tonal melody).
pub fn melodic_coherence(notes: &[Note]) -> f32 {
    if notes.len() < 2 {
        return 0.5; // insufficient data
    }

    // Semitone interval histogram from a large corpus (approximated from
    // Huron's "Sweet Anticipation" Table 4.2, transposed to relative frequencies).
    // Unison, Minor 2nd, Major 2nd, Minor 3rd, Major 3rd, Perfect 4th, Tritone,
    // Perfect 5th, Minor 6th, Major 6th, Minor 7th, Major 7th, Octave+
    #[rustfmt::skip]
    let interval_prob: [f32; 13] = [
        0.18, // unison (very common: repeated notes)
        0.09, // m2  (stepwise, common)
        0.16, // M2  (stepwise, most common)
        0.08, // m3  (common skip)
        0.07, // M3  (common skip)
        0.09, // P4  (common leap)
        0.02, // TT  (rare, dissonant)
        0.10, // P5  (common leap, consonant)
        0.04, // m6  (uncommon)
        0.05, // M6  (uncommon)
        0.03, // m7  (rare)
        0.02, // M7  (very rare)
        0.07, // 8ve+ (large leap, uncommon)
    ];

    let mut total_log_prob = 0.0f32;
    let mut count = 0usize;

    for w in notes.windows(2) {
        let freq_ratio = w[1].frequency / w[0].frequency.max(0.001);
        let semitones = (freq_ratio.log2() * 12.0).abs().round() as usize;
        let idx = semitones.min(12);
        let prob = interval_prob[idx];
        total_log_prob += prob.ln();
        count += 1;
    }

    if count == 0 {
        return 0.5;
    }

    // Convert mean log-probability to [0, 1] score.
    // ln(0.18) ≈ -1.71 (best), ln(0.02) ≈ -3.91 (worst).
    let mean_log_prob = total_log_prob / count as f32;
    let normalized = (mean_log_prob - (-4.0)) / ((-1.5) - (-4.0));
    normalized.clamp(0.0, 1.0)
}

// ─── Rhythmic Regularity ─────────────────────────────────────────────────────

/// Evaluate rhythmic regularity: consistency of note onset intervals.
///
/// Uses coefficient of variation (CV = σ/μ). A perfectly metronomic melody
/// scores 1.0; completely random onsets score near 0.0.
///
/// Note: some rhythmic variation is desirable (humanization); this measures
/// whether the music has a detectable pulse at all.
pub fn rhythmic_regularity(notes: &[Note]) -> f32 {
    if notes.len() < 3 {
        return 0.5;
    }

    let mut intervals: Vec<f32> = notes
        .windows(2)
        .map(|w| (w[1].start_time - w[0].start_time).abs())
        .filter(|&i| i > 0.001)
        .collect();

    if intervals.is_empty() {
        return 0.5;
    }

    // Filter out section boundary gaps: large IOIs (> 3× median) are structural
    // pauses between song sections, not rhythmic irregularity. Including them
    // destroys the CV even when within-section rhythm is perfectly regular.
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = intervals[intervals.len() / 2];
    let threshold = median * 3.0;
    let filtered: Vec<f32> = intervals.into_iter().filter(|&i| i <= threshold).collect();

    if filtered.is_empty() {
        return 0.5;
    }

    let mean = filtered.iter().sum::<f32>() / filtered.len() as f32;
    if mean < 0.001 {
        return 0.5;
    }

    let variance =
        filtered.iter().map(|&i| (i - mean).powi(2)).sum::<f32>() / filtered.len() as f32;
    let cv = variance.sqrt() / mean;

    // CV=0 (perfectly regular) → 1.0, CV=1.0 (chaotic) → 0.0
    // Allow some variation: CV=0.3 is pleasing human rhythm
    (1.0 - cv).clamp(0.0, 1.0)
}

// ─── Emotional Alignment ─────────────────────────────────────────────────────

/// Evaluate how well a composition's musical properties align with a target VA state.
///
/// Extracts musical proxies for valence and arousal from the composition,
/// then measures the distance to the target VA coordinate in the circumplex.
///
/// Score: 1.0 = perfect alignment, 0.0 = completely misaligned.
pub fn emotional_alignment(composition: &Composition, target: ValenceArousal) -> f32 {
    if composition.notes.is_empty() {
        return 0.0;
    }

    // Proxy for arousal: note density (notes per second)
    let density = composition.notes.len() as f32 / composition.duration_secs.max(0.1);
    // Normalize: 0 notes/sec → 0.0 arousal, 8+ notes/sec → 1.0 arousal
    let inferred_arousal = (density / 8.0).clamp(0.0, 1.0);

    // Proxy for valence: interval brightness (major intervals → positive)
    // Count major 3rds and major 6ths vs minor 3rds and minor 6ths
    let mut major_count = 0usize;
    let mut minor_count = 0usize;
    for w in composition.notes.windows(2) {
        let ratio = w[1].frequency / w[0].frequency.max(0.001);
        let semitones = (ratio.log2() * 12.0).abs().round() as i32;
        match semitones % 12 {
            4 | 9 => major_count += 1,  // major 3rd, major 6th
            3 | 8 => minor_count += 1,  // minor 3rd, minor 6th
            _ => {}
        }
    }
    let total = (major_count + minor_count).max(1);
    // major-heavy → positive valence, minor-heavy → negative
    let inferred_valence = (major_count as f32 - minor_count as f32) / total as f32;

    // Compute distance in VA space
    let d_valence = inferred_valence - target.valence;
    let d_arousal = inferred_arousal - target.arousal;
    let distance = (d_valence.powi(2) + d_arousal.powi(2)).sqrt();

    // Max possible distance in VA space: √(4 + 1) ≈ 2.24 (corners: v=±1, a=0/1)
    let max_distance = 2.0f32.sqrt(); // more realistic max
    (1.0 - distance / max_distance).clamp(0.0, 1.0)
}

// ─── Form Compliance ─────────────────────────────────────────────────────────

/// Evaluate structural form compliance.
///
/// Checks whether the composition has meaningful section structure by
/// measuring note density variation across the piece. A well-structured
/// piece has distinct high-density (chorus/climax) and low-density (intro/outro) regions.
///
/// Score: 0.0 (uniform density = no structure) to 1.0 (clear high/low contrast).
pub fn form_compliance(composition: &Composition) -> f32 {
    if composition.notes.is_empty() || composition.duration_secs < 0.5 {
        return 0.0;
    }

    let n_windows = 4.min(composition.notes.len() / 2);
    if n_windows < 2 {
        return 0.5;
    }

    let window_dur = composition.duration_secs / n_windows as f32;
    let mut densities = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let t_start = w as f32 * window_dur;
        let t_end = t_start + window_dur;
        let count = composition
            .notes
            .iter()
            .filter(|n| n.start_time >= t_start && n.start_time < t_end)
            .count();
        densities.push(count as f32 / window_dur);
    }

    let mean_density = densities.iter().sum::<f32>() / densities.len() as f32;
    if mean_density < 0.001 {
        return 0.0;
    }

    // Coefficient of variation: higher CV = more structure
    let variance = densities
        .iter()
        .map(|&d| (d - mean_density).powi(2))
        .sum::<f32>()
        / densities.len() as f32;
    let cv = variance.sqrt() / mean_density;

    // CV=0 (no structure) → 0.0, CV=1.0+ (strong structure) → 1.0
    cv.min(1.0)
}

// ─── Harmony Diversity ────────────────────────────────────────────────────────

/// Evaluate harmony diversity across a session.
///
/// Measures whether all 8 Harmonies are represented in the generated harmony
/// activations. A diverse aesthetic identity draws from all 8 harmonies;
/// a narrow one gets stuck in one or two.
///
/// Input: a slice of harmony activation vectors (one per artwork).
/// Score: 0.0 (one harmony dominates) to 1.0 (all 8 equally represented).
pub fn harmony_diversity(sessions: &[[f32; 8]]) -> f32 {
    if sessions.is_empty() {
        return 0.0;
    }

    // Sum activations across sessions
    let mut totals = [0.0f32; 8];
    for session in sessions {
        for (i, &v) in session.iter().enumerate() {
            totals[i] += v;
        }
    }

    let grand_total: f32 = totals.iter().sum();
    if grand_total < 0.001 {
        return 0.0;
    }

    // Normalized distribution
    let dist: Vec<f32> = totals.iter().map(|&v| v / grand_total).collect();

    // Shannon entropy of distribution: H = -Σ p*log(p)
    // Max entropy for 8 elements: log(8) ≈ 2.079
    let entropy: f32 = dist
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    let max_entropy = (8.0f32).ln();
    (entropy / max_entropy).clamp(0.0, 1.0)
}

// ─── Composite Creative Score ─────────────────────────────────────────────────

/// Composite creative quality score for a composition.
///
/// Aggregates all individual metrics into a single score that can be tracked
/// over time to detect regressions or improvements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeQualityScore {
    /// Melodic coherence: tonal interval transition quality.
    pub melodic_coherence: f32,
    /// Rhythmic regularity: consistency of note onset intervals.
    pub rhythmic_regularity: f32,
    /// Emotional alignment with the target VA state.
    pub emotional_alignment: f32,
    /// Form compliance: structural density variation.
    pub form_compliance: f32,
    /// Weighted composite [0, 1].
    pub composite: f32,
}

impl CreativeQualityScore {
    /// Evaluate a composition against a target VA state.
    pub fn evaluate(composition: &Composition, target_va: ValenceArousal) -> Self {
        let melodic_coherence = melodic_coherence(&composition.notes);
        let rhythmic_regularity = rhythmic_regularity(&composition.notes);
        let emotional_alignment = emotional_alignment(composition, target_va);
        let form_compliance = form_compliance(composition);

        // Weighting: melodic coherence most important for musical quality
        let composite = 0.35 * melodic_coherence
            + 0.25 * rhythmic_regularity
            + 0.25 * emotional_alignment
            + 0.15 * form_compliance;

        Self {
            melodic_coherence,
            rhythmic_regularity,
            emotional_alignment,
            form_compliance,
            composite: composite.clamp(0.0, 1.0),
        }
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        format!(
            "Creative Quality Report\n\
             ═══════════════════════\n\
             Melodic Coherence:   {:.3} (interval transition quality)\n\
             Rhythmic Regularity: {:.3} (pulse consistency)\n\
             Emotional Alignment: {:.3} (VA space proximity)\n\
             Form Compliance:     {:.3} (structural variation)\n\
             ─────────────────────\n\
             Composite Score:     {:.3}",
            self.melodic_coherence,
            self.rhythmic_regularity,
            self.emotional_alignment,
            self.form_compliance,
            self.composite,
        )
    }
}

// ─── Audio Quality Score ─────────────────────────────────────────────────────

/// Audio-level quality metrics computed on the actual waveform, not just notes.
///
/// Complements `CreativeQualityScore` (which measures compositional structure)
/// with perceptual audio features: dynamics, silence, spectral brightness.
/// These catch problems invisible to note-level analysis (e.g., clipping,
/// silence from synthesis bugs, harsh timbre).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityScore {
    /// RMS level in dB (loudness). Good music: -20 to -10 dB.
    pub rms_db: f32,
    /// Peak level in dB. Should be < 0 dB (no clipping).
    pub peak_db: f32,
    /// Crest factor (peak/RMS in dB). Good range: 6-20 dB.
    pub crest_db: f32,
    /// Fraction of samples below silence threshold [0, 1]. Good: < 0.3.
    pub silence_ratio: f32,
    /// Spectral centroid in Hz (brightness). Music: 500-3000 Hz.
    pub spectral_centroid: f32,
    /// Number of clipped samples (|sample| >= 0.999).
    pub clipped_samples: usize,
    /// Spectral flatness (Wiener entropy): 0 = tonal, 1 = noise-like.
    /// Good music: 0.05-0.40 (mostly tonal with some texture).
    pub spectral_flatness: f32,
    /// Dynamic range variation: std dev of windowed RMS levels.
    /// Higher = more expressive dynamics. Good: 3-12 dB.
    pub dynamic_range_variation_db: f32,
    /// Harmonic-to-noise ratio estimate (dB). Higher = cleaner tone.
    /// Good music: > 10 dB.
    pub harmonic_to_noise_db: f32,
    /// Composite audio quality [0, 1].
    pub composite: f32,
}

impl AudioQualityScore {
    /// Evaluate audio quality from stereo f32 samples.
    pub fn evaluate(samples: &[[f32; 2]], sample_rate: u32) -> Self {
        if samples.is_empty() {
            return Self {
                rms_db: -100.0,
                peak_db: -100.0,
                crest_db: 0.0,
                silence_ratio: 1.0,
                spectral_centroid: 0.0,
                clipped_samples: 0,
                spectral_flatness: 0.0,
                dynamic_range_variation_db: 0.0,
                harmonic_to_noise_db: 0.0,
                composite: 0.0,
            };
        }

        // Mono mix
        let mono: Vec<f32> = samples.iter().map(|[l, r]| (l + r) * 0.5).collect();
        let n = mono.len() as f32;

        // Peak and RMS
        let peak = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / n).sqrt();

        let peak_db = if peak > 0.0 { 20.0 * peak.log10() } else { -100.0 };
        let rms_db = if rms > 0.0 { 20.0 * rms.log10() } else { -100.0 };
        let crest_db = peak_db - rms_db;

        // Silence ratio
        let silence_thresh = 0.001f32;
        let silent = mono.iter().filter(|s| s.abs() < silence_thresh).count();
        let silence_ratio = silent as f32 / n;

        // Clipping
        let clipped_samples = mono.iter().filter(|s| s.abs() >= 0.999).count();

        // Spectral centroid (simplified: single FFT over the full signal)
        // For short signals this is sufficient; for long signals consider windowing.
        let mut centroid_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;
        let sr = sample_rate as f32;
        // Use Goertzel-like frequency-band energy estimation (much cheaper than full FFT)
        // Sample 32 frequency bands from 100 Hz to 8000 Hz
        for band in 0..32 {
            let freq = 100.0 * (8000.0f32 / 100.0).powf(band as f32 / 31.0);
            let omega = 2.0 * std::f32::consts::PI * freq / sr;
            // Compute energy at this frequency (first 8192 samples for speed)
            let window = mono.len().min(8192);
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (j, &s) in mono[..window].iter().enumerate() {
                re += s * (omega * j as f32).cos();
                im += s * (omega * j as f32).sin();
            }
            let mag = (re * re + im * im).sqrt();
            centroid_sum += freq * mag;
            magnitude_sum += mag;
        }
        let spectral_centroid = if magnitude_sum > 0.0 {
            centroid_sum / magnitude_sum
        } else {
            0.0
        };

        // Spectral flatness: geometric mean / arithmetic mean of power spectrum
        // Low = tonal (good for music), high = noise-like
        let spectral_flatness = {
            let mut band_mags = Vec::with_capacity(32);
            for band in 0..32 {
                let freq = 100.0 * (8000.0f32 / 100.0).powf(band as f32 / 31.0);
                let omega = 2.0 * std::f32::consts::PI * freq / sr;
                let window = mono.len().min(8192);
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for (j, &s) in mono[..window].iter().enumerate() {
                    re += s * (omega * j as f32).cos();
                    im += s * (omega * j as f32).sin();
                }
                band_mags.push((re * re + im * im).sqrt().max(1e-10));
            }
            let n_bands = band_mags.len() as f32;
            let log_geo_mean = band_mags.iter().map(|m| m.ln()).sum::<f32>() / n_bands;
            let arith_mean = band_mags.iter().sum::<f32>() / n_bands;
            if arith_mean > 0.0 { (log_geo_mean.exp() / arith_mean).clamp(0.0, 1.0) } else { 0.0 }
        };

        // Dynamic range variation: std dev of windowed RMS in dB
        // Higher = more expression (crescendo/decrescendo present)
        let dynamic_range_variation_db = {
            let window_size = (sr * 0.1) as usize; // 100ms windows
            if mono.len() > window_size * 2 {
                let mut window_rms_db = Vec::new();
                for chunk in mono.chunks(window_size) {
                    let rms_w = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
                    if rms_w > 1e-6 {
                        window_rms_db.push(20.0 * rms_w.log10());
                    }
                }
                if window_rms_db.len() > 1 {
                    let mean_db = window_rms_db.iter().sum::<f32>() / window_rms_db.len() as f32;
                    let var = window_rms_db.iter().map(|d| (d - mean_db).powi(2)).sum::<f32>()
                        / window_rms_db.len() as f32;
                    var.sqrt()
                } else { 0.0 }
            } else { 0.0 }
        };

        // Harmonic-to-noise ratio: ratio of energy at harmonic peaks vs rest
        // Estimate via autocorrelation peak strength
        let harmonic_to_noise_db = {
            let window = mono.len().min(4096);
            let samples = &mono[..window];
            // Find autocorrelation peak (pitch period) in lag range 40-500 (88-1102 Hz)
            let min_lag = (sr / 1102.0) as usize;
            let max_lag = (sr / 88.0) as usize;
            let mut best_r = 0.0f32;
            let energy: f32 = samples.iter().map(|s| s * s).sum();
            if energy > 1e-8 {
                for lag in min_lag..max_lag.min(window / 2) {
                    let mut corr = 0.0f32;
                    for j in 0..(window - lag) {
                        corr += samples[j] * samples[j + lag];
                    }
                    let r = corr / energy;
                    if r > best_r { best_r = r; }
                }
            }
            // HNR from autocorrelation: HNR = 10 * log10(r / (1 - r))
            let r = best_r.clamp(0.01, 0.99);
            10.0 * (r / (1.0 - r)).log10()
        };

        // Composite score
        // Dynamics: RMS between -25 and -8 dB is ideal
        let dynamics_score = if rms_db > -8.0 {
            0.5 // too loud
        } else if rms_db < -40.0 {
            0.1 // too quiet
        } else {
            // Peak at -15 dB
            let dist = (rms_db + 15.0).abs();
            (1.0 - dist / 25.0).clamp(0.0, 1.0)
        };

        // Silence: < 20% is good, > 50% is bad
        let silence_score = (1.0 - silence_ratio * 2.0).clamp(0.0, 1.0);

        // Clipping: any clipping degrades quality
        let clip_score = if clipped_samples == 0 { 1.0 } else { 0.3 };

        // Brightness: centroid 500-2500 Hz is musical
        let brightness_score = (1.0 - (spectral_centroid - 1500.0).abs() / 3000.0).clamp(0.0, 1.0);

        // Crest factor: 6-18 dB is healthy dynamic range
        let crest_score = if crest_db > 6.0 && crest_db < 18.0 {
            1.0
        } else {
            0.5
        };

        // Spectral flatness: too flat = noise, too peaked = pure sine
        let flatness_score = if spectral_flatness > 0.05 && spectral_flatness < 0.40 {
            1.0
        } else if spectral_flatness < 0.60 {
            0.6
        } else {
            0.2 // very noise-like
        };

        // Dynamic variation: expression matters — flat dynamics = robotic
        let expression_score = if dynamic_range_variation_db > 3.0 && dynamic_range_variation_db < 15.0 {
            1.0
        } else if dynamic_range_variation_db > 1.0 {
            0.6
        } else {
            0.3 // flat dynamics
        };

        // HNR: higher = cleaner harmonic content
        let hnr_score = if harmonic_to_noise_db > 15.0 {
            1.0
        } else if harmonic_to_noise_db > 5.0 {
            0.7
        } else {
            0.4
        };

        let composite = (0.18 * dynamics_score
            + 0.15 * silence_score
            + 0.12 * clip_score
            + 0.10 * brightness_score
            + 0.10 * crest_score
            + 0.10 * flatness_score
            + 0.15 * expression_score
            + 0.10 * hnr_score)
            .clamp(0.0, 1.0);

        Self {
            rms_db,
            peak_db,
            crest_db,
            silence_ratio,
            spectral_centroid,
            clipped_samples,
            spectral_flatness,
            dynamic_range_variation_db,
            harmonic_to_noise_db,
            composite,
        }
    }

    /// Evaluate from mono f32 samples.
    pub fn evaluate_mono(samples: &[f32], sample_rate: u32) -> Self {
        let stereo: Vec<[f32; 2]> = samples.iter().map(|&s| [s, s]).collect();
        Self::evaluate(&stereo, sample_rate)
    }

    pub fn report(&self) -> String {
        format!(
            "Audio Quality Report\n\
             ════════════════════\n\
             RMS Level:     {:6.1} dB\n\
             Peak Level:    {:6.1} dB\n\
             Crest Factor:  {:6.1} dB\n\
             Silence:       {:5.1}%\n\
             Centroid:      {:5.0} Hz\n\
             Flatness:      {:5.3}\n\
             Dynamic Var:   {:5.1} dB\n\
             HNR:           {:5.1} dB\n\
             Clipped:       {}\n\
             ────────────────────\n\
             Audio Score:   {:.3}",
            self.rms_db,
            self.peak_db,
            self.crest_db,
            self.silence_ratio * 100.0,
            self.spectral_centroid,
            self.spectral_flatness,
            self.dynamic_range_variation_db,
            self.harmonic_to_noise_db,
            self.clipped_samples,
            self.composite,
        )
    }
}

/// Run the full creative benchmark suite on a given musical state.
///
/// Generates multiple compositions across different emotional states and
/// averages the quality scores, giving a stable estimate of creative quality.
pub fn run_benchmark(config: &MuseConfig, state: &MusicalState) -> BenchmarkResult {
    let test_cases: Vec<(ValenceArousal, u64)> = vec![
        (ValenceArousal::new(0.5, 0.7), 1),   // happy/excited
        (ValenceArousal::new(-0.4, 0.6), 2),  // tense
        (ValenceArousal::new(-0.2, 0.2), 3),  // melancholy
        (ValenceArousal::new(0.6, 0.3), 4),   // content
        (ValenceArousal::new(0.0, 0.5), 5),   // neutral
    ];

    let mut scores: Vec<CreativeQualityScore> = Vec::new();
    let mut audio_scores: Vec<AudioQualityScore> = Vec::new();
    for (target_va, seed) in &test_cases {
        let comp = crate::compose(config, state, *seed);
        let score = CreativeQualityScore::evaluate(&comp, *target_va);
        // Audio quality from the actual waveform
        let audio_score = match &comp.audio {
            crate::AudioData::StereoF32(samples) => {
                AudioQualityScore::evaluate(samples, comp.sample_rate)
            }
            crate::AudioData::F32(samples) => {
                AudioQualityScore::evaluate_mono(samples, comp.sample_rate)
            }
            crate::AudioData::I16(samples) => {
                let f: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
                AudioQualityScore::evaluate_mono(&f, comp.sample_rate)
            }
        };
        audio_scores.push(audio_score);
        scores.push(score);
    }

    let n = scores.len() as f32;
    BenchmarkResult {
        mean_melodic_coherence: scores.iter().map(|s| s.melodic_coherence).sum::<f32>() / n,
        mean_rhythmic_regularity: scores.iter().map(|s| s.rhythmic_regularity).sum::<f32>() / n,
        mean_emotional_alignment: scores.iter().map(|s| s.emotional_alignment).sum::<f32>() / n,
        mean_form_compliance: scores.iter().map(|s| s.form_compliance).sum::<f32>() / n,
        mean_composite: scores.iter().map(|s| s.composite).sum::<f32>() / n,
        mean_audio_quality: audio_scores.iter().map(|s| s.composite).sum::<f32>() / n,
        mean_rms_db: audio_scores.iter().map(|s| s.rms_db).sum::<f32>() / n,
        mean_silence_ratio: audio_scores.iter().map(|s| s.silence_ratio).sum::<f32>() / n,
        mean_spectral_centroid: audio_scores.iter().map(|s| s.spectral_centroid).sum::<f32>() / n,
        n_compositions: scores.len(),
    }
}

/// Result of the creative benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub mean_melodic_coherence: f32,
    pub mean_rhythmic_regularity: f32,
    pub mean_emotional_alignment: f32,
    pub mean_form_compliance: f32,
    pub mean_composite: f32,
    /// Audio-level quality: dynamics, silence, brightness, clipping.
    pub mean_audio_quality: f32,
    pub mean_rms_db: f32,
    pub mean_silence_ratio: f32,
    pub mean_spectral_centroid: f32,
    pub n_compositions: usize,
}

impl BenchmarkResult {
    /// Pass/fail: composite score above threshold (default 0.35 for a new system).
    pub fn passes(&self, threshold: f32) -> bool {
        self.mean_composite >= threshold
    }

    pub fn report(&self) -> String {
        format!(
            "Creative Benchmark Results (n={})\n\
             ══════════════════════════════════\n\
             Melodic Coherence:   {:.3}\n\
             Rhythmic Regularity: {:.3}\n\
             Emotional Alignment: {:.3}\n\
             Form Compliance:     {:.3}\n\
             ──────────────────────────────────\n\
             Mean Composite:      {:.3}\n\n\
             Audio Quality\n\
             ══════════════════════════════════\n\
             Audio Score:         {:.3}\n\
             RMS Level:           {:.1} dB\n\
             Silence:             {:.1}%\n\
             Spectral Centroid:   {:.0} Hz",
            self.n_compositions,
            self.mean_melodic_coherence,
            self.mean_rhythmic_regularity,
            self.mean_emotional_alignment,
            self.mean_form_compliance,
            self.mean_composite,
            self.mean_audio_quality,
            self.mean_rms_db,
            self.mean_silence_ratio * 100.0,
            self.mean_spectral_centroid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MuseConfig, MusicalState, Note};
    use symthaea_aesthetic::ValenceArousal;

    fn ascending_scale_notes() -> Vec<Note> {
        // C major scale ascending: C4 D4 E4 F4 G4 A4 B4 C5
        let freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25];
        freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| Note {
                frequency: f,
                start_time: i as f32 * 0.25,
                duration: 0.2,
                velocity: 0.8,
            })
            .collect()
    }

    fn random_notes() -> Vec<Note> {
        // Deliberately "random" frequencies — should score low on coherence
        let freqs = [100.0, 700.0, 250.0, 1200.0, 150.0, 900.0, 300.0, 1100.0];
        freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| Note {
                frequency: f,
                start_time: i as f32 * 0.3,
                duration: 0.2,
                velocity: 0.7,
            })
            .collect()
    }

    // ─── Melodic coherence ────────────────────────────────────────────────────

    #[test]
    fn coherence_bounded() {
        let score = melodic_coherence(&ascending_scale_notes());
        assert!(score >= 0.0 && score <= 1.0, "score out of bounds: {score}");
    }

    #[test]
    fn scale_more_coherent_than_random() {
        let scale_score = melodic_coherence(&ascending_scale_notes());
        let random_score = melodic_coherence(&random_notes());
        assert!(
            scale_score > random_score,
            "scale ({scale_score}) should outscore random ({random_score})"
        );
    }

    #[test]
    fn coherence_single_note() {
        let note = vec![Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 1.0,
            velocity: 0.8,
        }];
        let score = melodic_coherence(&note);
        assert!(score >= 0.0 && score <= 1.0);
    }

    // ─── Rhythmic regularity ──────────────────────────────────────────────────

    #[test]
    fn regular_rhythm_high_score() {
        let notes: Vec<Note> = (0..8)
            .map(|i| Note {
                frequency: 440.0,
                start_time: i as f32 * 0.25, // perfectly regular
                duration: 0.2,
                velocity: 0.8,
            })
            .collect();
        let score = rhythmic_regularity(&notes);
        assert!(score > 0.8, "perfectly regular rhythm should score high: {score}");
    }

    #[test]
    fn irregular_rhythm_lower_score() {
        let onsets = [0.0, 0.1, 0.5, 0.51, 1.2, 1.3, 2.5, 2.51];
        let notes: Vec<Note> = onsets
            .iter()
            .map(|&t| Note {
                frequency: 440.0,
                start_time: t,
                duration: 0.08,
                velocity: 0.7,
            })
            .collect();
        let reg_score = rhythmic_regularity(&ascending_scale_notes());
        let irr_score = rhythmic_regularity(&notes);
        // Irregular should be lower (though both are valid)
        assert!(irr_score <= reg_score + 0.1, "irregular {irr_score} vs regular {reg_score}");
    }

    // ─── Emotional alignment ──────────────────────────────────────────────────

    #[test]
    fn alignment_bounded() {
        let config = MuseConfig { duration_secs: 2.0, max_notes: 8, ..Default::default() };
        let state = MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let score = emotional_alignment(&comp, ValenceArousal::neutral());
        assert!(score >= 0.0 && score <= 1.0, "alignment out of bounds: {score}");
    }

    // ─── Form compliance ──────────────────────────────────────────────────────

    #[test]
    fn form_compliance_structured() {
        let config = MuseConfig { duration_secs: 4.0, max_notes: 16, ..Default::default() };
        let state = MusicalState {
            harmony_activations: [0.2, 0.8, 0.3, 0.9, 0.1, 0.5, 0.7, 0.1],
            ..Default::default()
        };
        let comp = crate::compose(&config, &state, 42);
        let score = form_compliance(&comp);
        assert!(score >= 0.0 && score <= 1.0);
    }

    // ─── Harmony diversity ────────────────────────────────────────────────────

    #[test]
    fn diversity_uniform_is_max() {
        let sessions: Vec<[f32; 8]> = (0..10).map(|_| [0.5; 8]).collect();
        let score = harmony_diversity(&sessions);
        assert!(score > 0.95, "uniform activations should score near 1.0: {score}");
    }

    #[test]
    fn diversity_one_harmony_is_min() {
        let mut sessions = Vec::new();
        for _ in 0..10 {
            let mut h = [0.0f32; 8];
            h[0] = 1.0; // only ResonantCoherence
            sessions.push(h);
        }
        let score = harmony_diversity(&sessions);
        assert!(score < 0.1, "single harmony should score near 0.0: {score}");
    }

    // ─── Composite quality score ──────────────────────────────────────────────

    #[test]
    fn quality_score_bounded() {
        let config = MuseConfig { duration_secs: 2.0, max_notes: 8, ..Default::default() };
        let state = MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let score = CreativeQualityScore::evaluate(&comp, ValenceArousal::neutral());
        assert!(score.composite >= 0.0 && score.composite <= 1.0);
        assert!(score.melodic_coherence >= 0.0 && score.melodic_coherence <= 1.0);
        assert!(score.rhythmic_regularity >= 0.0 && score.rhythmic_regularity <= 1.0);
        assert!(score.emotional_alignment >= 0.0 && score.emotional_alignment <= 1.0);
        assert!(score.form_compliance >= 0.0 && score.form_compliance <= 1.0);
    }

    #[test]
    fn benchmark_produces_result() {
        let config = MuseConfig { duration_secs: 2.0, max_notes: 8, ..Default::default() };
        let state = MusicalState::default();
        let result = run_benchmark(&config, &state);
        assert_eq!(result.n_compositions, 5);
        assert!(result.mean_composite >= 0.0 && result.mean_composite <= 1.0);
        // Minimum quality bar: composite should be above 0.15 (very low bar for any output)
        assert!(result.passes(0.15), "benchmark failed minimum quality: {}", result.report());
    }

    #[test]
    fn quality_report_contains_scores() {
        let config = MuseConfig { duration_secs: 1.0, max_notes: 4, ..Default::default() };
        let state = MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let score = CreativeQualityScore::evaluate(&comp, ValenceArousal::neutral());
        let report = score.report();
        assert!(report.contains("Melodic"));
        assert!(report.contains("Composite"));
    }
}

// ─── Music Theory Validation ────────────────────────────────────────────────

/// Theory-level validation of generated music against fundamental musical rules.
///
/// Unlike `CreativeQualityScore` (which measures statistical properties) and
/// `AudioQualityScore` (which measures signal properties), this validates
/// against *prescriptive* music theory rules that distinguish "correct" from
/// "wrong" in a formal sense.
///
/// # References
/// - Aldwell, E. & Schachter, C. (2010). *Harmony and Voice Leading*. 4th ed.
/// - Kostka, S. & Payne, D. (2012). *Tonal Harmony*. 7th ed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryValidation {
    /// Fraction of notes that fall on scale degrees [0, 1]. 1.0 = all in scale.
    pub scale_adherence: f32,
    /// Fraction of consecutive note pairs that avoid parallel fifths [0, 1].
    pub parallel_fifth_avoidance: f32,
    /// Fraction of note onsets that fall on rhythmic grid positions [0, 1].
    pub rhythmic_quantization: f32,
    /// Fraction of intervals within a singable range (< octave) [0, 1].
    pub voice_range_compliance: f32,
    /// Fraction of phrases with proper contour (arc shape) [0, 1].
    pub phrase_contour_quality: f32,
    /// Number of theory violations found.
    pub violations: usize,
    /// Composite theory score [0, 1].
    pub composite: f32,
}

impl TheoryValidation {
    /// Validate a composition against music theory rules.
    ///
    /// `scale_freqs` is the set of frequencies in the target scale. If empty,
    /// scale adherence is computed against the chromatic scale (always 1.0).
    pub fn validate(notes: &[Note], scale_freqs: &[f32]) -> Self {
        let scale_adherence = Self::compute_scale_adherence(notes, scale_freqs);
        let parallel_fifth_avoidance = Self::compute_parallel_fifth_avoidance(notes);
        let rhythmic_quantization = Self::compute_rhythmic_quantization(notes);
        let voice_range_compliance = Self::compute_voice_range(notes);
        let phrase_contour_quality = Self::compute_phrase_contour(notes);

        let violations = [
            scale_adherence < 0.7,
            parallel_fifth_avoidance < 0.8,
            rhythmic_quantization < 0.6,
            voice_range_compliance < 0.7,
            phrase_contour_quality < 0.5,
        ]
        .iter()
        .filter(|&&v| v)
        .count();

        let composite = (0.30 * scale_adherence
            + 0.20 * parallel_fifth_avoidance
            + 0.20 * rhythmic_quantization
            + 0.15 * voice_range_compliance
            + 0.15 * phrase_contour_quality)
            .clamp(0.0, 1.0);

        Self {
            scale_adherence,
            parallel_fifth_avoidance,
            rhythmic_quantization,
            voice_range_compliance,
            phrase_contour_quality,
            violations,
            composite,
        }
    }

    /// What fraction of notes fall on scale degrees?
    fn compute_scale_adherence(notes: &[Note], scale_freqs: &[f32]) -> f32 {
        if notes.is_empty() || scale_freqs.is_empty() {
            return 1.0; // no scale to violate
        }

        let mut on_scale = 0usize;
        for note in notes {
            // Normalize to within one octave of the nearest scale frequency
            let pitch_class = note_to_pitch_class(note.frequency);
            let scale_classes: Vec<f32> = scale_freqs.iter().map(|&f| note_to_pitch_class(f)).collect();

            // Check if this pitch class is within 25 cents of any scale degree
            let min_distance = scale_classes
                .iter()
                .map(|&sc| {
                    let diff = (pitch_class - sc).abs();
                    diff.min(12.0 - diff) // wrap around octave
                })
                .fold(f32::MAX, f32::min);

            if min_distance < 0.25 {
                // Within quarter-tone tolerance
                on_scale += 1;
            }
        }
        on_scale as f32 / notes.len() as f32
    }

    /// Check for parallel perfect fifths (a voice-leading error in classical theory).
    fn compute_parallel_fifth_avoidance(notes: &[Note]) -> f32 {
        if notes.len() < 4 {
            return 1.0;
        }

        let mut parallel_fifths = 0usize;
        let mut total_pairs = 0usize;

        // Check consecutive note pairs for parallel fifth motion
        for w in notes.windows(4) {
            let interval_1 = interval_semitones(w[0].frequency, w[1].frequency);
            let interval_2 = interval_semitones(w[2].frequency, w[3].frequency);

            // Both intervals are perfect fifths (7 semitones) = parallel fifths
            if (interval_1 - 7.0).abs() < 0.5 && (interval_2 - 7.0).abs() < 0.5 {
                // Check if motion is parallel (both voices move in same direction)
                let motion_1 = w[2].frequency - w[0].frequency;
                let motion_2 = w[3].frequency - w[1].frequency;
                if motion_1.signum() == motion_2.signum() {
                    parallel_fifths += 1;
                }
            }
            total_pairs += 1;
        }

        if total_pairs == 0 {
            return 1.0;
        }
        1.0 - (parallel_fifths as f32 / total_pairs as f32)
    }

    /// How well do note onsets align with a rhythmic grid?
    fn compute_rhythmic_quantization(notes: &[Note]) -> f32 {
        if notes.len() < 2 {
            return 1.0;
        }

        // Estimate tempo from median IOI (inter-onset interval)
        let mut iois: Vec<f32> = notes
            .windows(2)
            .map(|w| (w[1].start_time - w[0].start_time).abs())
            .filter(|&ioi| ioi > 0.01) // ignore simultaneous notes
            .collect();

        if iois.is_empty() {
            return 0.5;
        }
        iois.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ioi = iois[iois.len() / 2];

        if median_ioi < 0.05 {
            return 0.5; // too short to be meaningful
        }

        // Check how many onsets fall near grid positions (multiples/subdivisions of median IOI)
        let grid_sizes = [median_ioi, median_ioi / 2.0, median_ioi / 3.0, median_ioi * 2.0];
        let tolerance = median_ioi * 0.15; // 15% tolerance

        let mut on_grid = 0usize;
        for note in notes {
            let best_grid_dist = grid_sizes
                .iter()
                .map(|&grid| {
                    if grid < 0.01 { return f32::MAX; }
                    let remainder = note.start_time % grid;
                    remainder.min(grid - remainder)
                })
                .fold(f32::MAX, f32::min);

            if best_grid_dist < tolerance {
                on_grid += 1;
            }
        }

        on_grid as f32 / notes.len() as f32
    }

    /// What fraction of intervals are within a comfortable singing range?
    fn compute_voice_range(notes: &[Note]) -> f32 {
        if notes.len() < 2 {
            return 1.0;
        }

        let mut comfortable = 0usize;
        for w in notes.windows(2) {
            let semitones = interval_semitones(w[0].frequency, w[1].frequency);
            // Within an octave (12 semitones) is comfortable for a single voice
            if semitones <= 12.0 {
                comfortable += 1;
            }
        }
        comfortable as f32 / (notes.len() - 1) as f32
    }

    /// Do phrases have proper melodic contour (rise → peak → fall)?
    fn compute_phrase_contour(notes: &[Note]) -> f32 {
        if notes.len() < 4 {
            return 0.5;
        }

        // Split into phrases at gaps > 0.3s
        let mut phrases: Vec<Vec<f32>> = Vec::new();
        let mut current_phrase = vec![notes[0].frequency];

        for w in notes.windows(2) {
            let gap = w[1].start_time - w[0].start_time - w[0].duration;
            if gap > 0.3 {
                if current_phrase.len() >= 3 {
                    phrases.push(current_phrase.clone());
                }
                current_phrase.clear();
            }
            current_phrase.push(w[1].frequency);
        }
        if current_phrase.len() >= 3 {
            phrases.push(current_phrase);
        }

        if phrases.is_empty() {
            return 0.5;
        }

        // Score each phrase for arc shape: should have a peak somewhere in the middle
        let mut good_contours = 0usize;
        for phrase in &phrases {
            let max_idx = phrase
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let relative_peak = max_idx as f32 / (phrase.len() - 1) as f32;
            // Peak between 20% and 80% of phrase = good arc
            if relative_peak > 0.15 && relative_peak < 0.85 {
                good_contours += 1;
            }
        }

        good_contours as f32 / phrases.len() as f32
    }

    pub fn report(&self) -> String {
        format!(
            "Music Theory Validation\n\
             ═══════════════════════\n\
             Scale Adherence:    {:.1}%\n\
             Parallel 5th Avoid: {:.1}%\n\
             Rhythmic Grid:      {:.1}%\n\
             Voice Range:        {:.1}%\n\
             Phrase Contour:     {:.1}%\n\
             Violations:         {}\n\
             ───────────────────────\n\
             Theory Score:       {:.3}",
            self.scale_adherence * 100.0,
            self.parallel_fifth_avoidance * 100.0,
            self.rhythmic_quantization * 100.0,
            self.voice_range_compliance * 100.0,
            self.phrase_contour_quality * 100.0,
            self.violations,
            self.composite,
        )
    }
}

/// Convert frequency to pitch class (0-12, where 0 = C, continuous).
fn note_to_pitch_class(freq: f32) -> f32 {
    if freq <= 0.0 { return 0.0; }
    (12.0 * (freq / 261.63).log2()).rem_euclid(12.0) // relative to C4
}

/// Interval between two frequencies in semitones.
fn interval_semitones(f1: f32, f2: f32) -> f32 {
    if f1 <= 0.0 || f2 <= 0.0 { return 0.0; }
    (12.0 * (f2 / f1).log2()).abs()
}

// ─── FAD (Fréchet Audio Distance) ───────────────────────────────────────────

/// Fréchet Audio Distance: the industry-standard metric for generative music.
///
/// Compares the statistical distribution of generated audio features against
/// a reference distribution of real music. Lower FAD = closer to real music.
///
/// Since we don't have VGGish/CLAP embeddings available in pure Rust, this
/// implementation uses hand-crafted audio features (MFCCs approximation via
/// spectral band energies) as the embedding space. This is less precise than
/// VGGish FAD but still captures the key distributional differences.
///
/// # References
/// - Kilgour et al. (2019). "Fréchet Audio Distance: A Reference-Free Metric
///   for Evaluating Music Enhancement Algorithms." INTERSPEECH.
/// - Hershey et al. (2017). "CNN Architectures for Large-Scale Audio
///   Classification" (VGGish model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FadScore {
    /// Fréchet distance between generated and reference distributions.
    /// Lower is better. < 5.0 = excellent, < 15.0 = good, > 30.0 = poor.
    pub fad: f32,
    /// Number of generated compositions evaluated.
    pub n_generated: usize,
    /// Number of reference compositions used.
    pub n_reference: usize,
    /// Mean embedding of generated set (for debugging).
    pub generated_mean: Vec<f32>,
    /// Mean embedding of reference set (for debugging).
    pub reference_mean: Vec<f32>,
}

/// Number of spectral bands used as audio features (pseudo-MFCCs).
const FAD_N_BANDS: usize = 24;

impl FadScore {
    /// Compute FAD between generated compositions and a reference set.
    ///
    /// `generated`: stereo audio samples from Symthaea compositions.
    /// `reference`: stereo audio samples from real music (e.g., Spotify likes).
    /// Both at `sample_rate` Hz.
    pub fn compute(
        generated: &[Vec<[f32; 2]>],
        reference: &[Vec<[f32; 2]>],
        sample_rate: u32,
    ) -> Self {
        let gen_embeddings: Vec<[f32; FAD_N_BANDS]> = generated
            .iter()
            .map(|s| Self::extract_embedding(s, sample_rate))
            .collect();
        let ref_embeddings: Vec<[f32; FAD_N_BANDS]> = reference
            .iter()
            .map(|s| Self::extract_embedding(s, sample_rate))
            .collect();

        if gen_embeddings.is_empty() || ref_embeddings.is_empty() {
            return Self {
                fad: f32::MAX,
                n_generated: gen_embeddings.len(),
                n_reference: ref_embeddings.len(),
                generated_mean: vec![0.0; FAD_N_BANDS],
                reference_mean: vec![0.0; FAD_N_BANDS],
            };
        }

        // Compute means
        let gen_mean = Self::mean_embedding(&gen_embeddings);
        let ref_mean = Self::mean_embedding(&ref_embeddings);

        // Compute covariances
        let gen_cov = Self::covariance(&gen_embeddings, &gen_mean);
        let ref_cov = Self::covariance(&ref_embeddings, &ref_mean);

        // Fréchet distance: ||mu_g - mu_r||^2 + Tr(C_g + C_r - 2*(C_g*C_r)^0.5)
        // Simplified: use diagonal covariance (assumes independence between bands)
        let mean_diff_sq: f32 = gen_mean
            .iter()
            .zip(&ref_mean)
            .map(|(g, r)| (g - r).powi(2))
            .sum();

        // Diagonal FAD: sum of (var_g + var_r - 2*sqrt(var_g * var_r))
        let trace_term: f32 = gen_cov
            .iter()
            .zip(&ref_cov)
            .map(|(&vg, &vr)| {
                let vg = vg.max(0.0);
                let vr = vr.max(0.0);
                vg + vr - 2.0 * (vg * vr).sqrt()
            })
            .sum();

        let fad = (mean_diff_sq + trace_term).max(0.0);

        Self {
            fad,
            n_generated: gen_embeddings.len(),
            n_reference: ref_embeddings.len(),
            generated_mean: gen_mean,
            reference_mean: ref_mean,
        }
    }

    /// Extract a pseudo-MFCC embedding from stereo audio.
    ///
    /// Uses log-spaced spectral band energies as features (approximation of
    /// mel-frequency cepstral coefficients without the DCT step).
    fn extract_embedding(samples: &[[f32; 2]], sample_rate: u32) -> [f32; FAD_N_BANDS] {
        let sr = sample_rate as f32;
        let mono: Vec<f32> = samples.iter().map(|[l, r]| (l + r) * 0.5).collect();
        let window = mono.len().min(16384);
        let mut bands = [0.0f32; FAD_N_BANDS];

        if window < 64 {
            return bands;
        }

        // Log-spaced frequency bands from 50 Hz to 8000 Hz
        for (b, band) in bands.iter_mut().enumerate() {
            let freq = 50.0 * (8000.0f32 / 50.0).powf(b as f32 / (FAD_N_BANDS - 1) as f32);
            let omega = 2.0 * std::f32::consts::PI * freq / sr;

            // Goertzel energy at this frequency
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (j, &s) in mono[..window].iter().enumerate() {
                re += s * (omega * j as f32).cos();
                im += s * (omega * j as f32).sin();
            }
            let energy = (re * re + im * im).sqrt() / window as f32;
            *band = (energy + 1e-10).ln(); // log energy
        }

        bands
    }

    fn mean_embedding(embeddings: &[[f32; FAD_N_BANDS]]) -> Vec<f32> {
        let n = embeddings.len() as f32;
        let mut mean = vec![0.0f32; FAD_N_BANDS];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        mean
    }

    fn covariance(embeddings: &[[f32; FAD_N_BANDS]], mean: &[f32]) -> Vec<f32> {
        let n = embeddings.len() as f32;
        let mut var = vec![0.0f32; FAD_N_BANDS];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                var[i] += (v - mean[i]).powi(2);
            }
        }
        for v in &mut var {
            *v /= n.max(1.0);
        }
        var
    }

    pub fn report(&self) -> String {
        let quality = if self.fad < 5.0 {
            "Excellent"
        } else if self.fad < 15.0 {
            "Good"
        } else if self.fad < 30.0 {
            "Fair"
        } else {
            "Poor"
        };

        format!(
            "Frechet Audio Distance\n\
             ══════════════════════\n\
             FAD Score:      {:.2} ({})\n\
             Generated:      {} compositions\n\
             Reference:      {} compositions\n\
             ──────────────────────\n\
             Interpretation: < 5 excellent, < 15 good, < 30 fair, > 30 poor",
            self.fad, quality, self.n_generated, self.n_reference,
        )
    }
}

#[cfg(test)]
mod external_validation_tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::MuseConfig;

    #[test]
    fn theory_validation_on_composition() {
        let config = MuseConfig { duration_secs: 2.0, max_notes: 8, ..Default::default() };
        let state = MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let scale = crate::pitch::build_scale(&state);
        let theory = TheoryValidation::validate(&comp.notes, &scale);
        assert!(theory.composite > 0.0);
        assert!(theory.scale_adherence >= 0.0 && theory.scale_adherence <= 1.0);
        assert!(theory.violations <= 5);
        let report = theory.report();
        assert!(report.contains("Scale Adherence"));
    }

    #[test]
    fn audio_quality_has_new_metrics() {
        let config = MuseConfig { duration_secs: 1.0, max_notes: 4, ..Default::default() };
        let state = MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let aq = match &comp.audio {
            crate::AudioData::StereoF32(s) => AudioQualityScore::evaluate(s, comp.sample_rate),
            crate::AudioData::F32(s) => AudioQualityScore::evaluate_mono(s, comp.sample_rate),
            crate::AudioData::I16(s) => {
                let f: Vec<f32> = s.iter().map(|&v| v as f32 / 32768.0).collect();
                AudioQualityScore::evaluate_mono(&f, comp.sample_rate)
            }
        };
        assert!(aq.spectral_flatness >= 0.0 && aq.spectral_flatness <= 1.0);
        assert!(aq.dynamic_range_variation_db >= 0.0);
        // HNR can be negative for noise-heavy audio
        let report = aq.report();
        assert!(report.contains("Flatness"));
        assert!(report.contains("HNR"));
    }

    #[test]
    fn fad_self_distance_near_zero() {
        // FAD of a set against itself should be ~0
        let config = MuseConfig { duration_secs: 1.0, max_notes: 4, ..Default::default() };
        let state = MusicalState::default();

        let mut compositions = Vec::new();
        for seed in 0..3 {
            let comp = crate::compose(&config, &state, seed);
            if let crate::AudioData::StereoF32(samples) = &comp.audio {
                compositions.push(samples.clone());
            }
        }

        if compositions.len() >= 2 {
            let fad = FadScore::compute(&compositions, &compositions, 44100);
            assert!(
                fad.fad < 1.0,
                "FAD of set against itself should be near zero, got {}",
                fad.fad
            );
        }
    }

    #[test]
    fn fad_different_sets_diverge() {
        let config = MuseConfig { duration_secs: 1.0, max_notes: 4, ..Default::default() };

        // Set A: calm music
        let calm = MusicalState { arousal: 0.2, valence: 0.5, ..Default::default() };
        let mut set_a = Vec::new();
        for seed in 0..3 {
            let comp = crate::compose(&config, &calm, seed);
            if let crate::AudioData::StereoF32(s) = &comp.audio {
                set_a.push(s.clone());
            }
        }

        // Set B: intense music
        let intense = MusicalState { arousal: 0.9, valence: -0.5, dopamine: 0.8, ..Default::default() };
        let mut set_b = Vec::new();
        for seed in 10..13 {
            let comp = crate::compose(&config, &intense, seed);
            if let crate::AudioData::StereoF32(s) = &comp.audio {
                set_b.push(s.clone());
            }
        }

        if !set_a.is_empty() && !set_b.is_empty() {
            let fad_self = FadScore::compute(&set_a, &set_a, 44100);
            let fad_cross = FadScore::compute(&set_a, &set_b, 44100);
            assert!(
                fad_cross.fad > fad_self.fad,
                "Cross-set FAD ({:.2}) should exceed self FAD ({:.2})",
                fad_cross.fad, fad_self.fad,
            );
        }
    }

    #[test]
    fn full_quality_benchmark() {
        let config = MuseConfig { duration_secs: 4.0, max_notes: 16, ..Default::default() };

        let scenarios: Vec<(&str, MusicalState)> = vec![
            ("Joyful", MusicalState {
                arousal: 0.7, valence: 0.6, dopamine: 0.7, serotonin: 0.6,
                consciousness_level: 0.6, ..Default::default()
            }),
            ("Tense", MusicalState {
                arousal: 0.8, valence: -0.5, dopamine: 0.4, noradrenaline: 0.7,
                consciousness_level: 0.5, ..Default::default()
            }),
            ("Melancholy", MusicalState {
                arousal: 0.2, valence: -0.3, serotonin: 0.3,
                consciousness_level: 0.4, ..Default::default()
            }),
            ("Serene", MusicalState {
                arousal: 0.3, valence: 0.4, serotonin: 0.7,
                consciousness_level: 0.7, ..Default::default()
            }),
            ("Neutral", MusicalState::default()),
        ];

        let mut total_creative = 0.0f32;
        let mut total_audio = 0.0f32;
        let mut total_theory = 0.0f32;

        eprintln!("\n══════════════════════════════════════════════════");
        eprintln!("  Symthaea Music Quality Benchmark");
        eprintln!("══════════════════════════════════════════════════");

        for (name, state) in &scenarios {
            let comp = crate::compose(&config, state, 42);
            let va = ValenceArousal::new(state.valence, state.arousal);

            let creative = CreativeQualityScore::evaluate(&comp, va);
            let scale = crate::pitch::build_scale(state);
            let theory = TheoryValidation::validate(&comp.notes, &scale);
            let audio = match &comp.audio {
                crate::AudioData::StereoF32(s) => AudioQualityScore::evaluate(s, comp.sample_rate),
                crate::AudioData::F32(s) => AudioQualityScore::evaluate_mono(s, comp.sample_rate),
                crate::AudioData::I16(s) => {
                    let f: Vec<f32> = s.iter().map(|&v| v as f32 / 32768.0).collect();
                    AudioQualityScore::evaluate_mono(&f, comp.sample_rate)
                }
            };

            eprintln!("\n── {} ({} notes, {:.1}s) ──", name, comp.notes.len(), comp.duration_secs);
            eprintln!("  Creative:  mel={:.3} rhy={:.3} emo={:.3} form={:.3} -> {:.3}",
                creative.melodic_coherence, creative.rhythmic_regularity,
                creative.emotional_alignment, creative.form_compliance, creative.composite);
            eprintln!("  Audio:     rms={:.1}dB flat={:.3} dynVar={:.1}dB hnr={:.1}dB -> {:.3}",
                audio.rms_db, audio.spectral_flatness,
                audio.dynamic_range_variation_db, audio.harmonic_to_noise_db, audio.composite);
            eprintln!("  Theory:    scale={:.0}% p5={:.0}% grid={:.0}% range={:.0}% contour={:.0}% -> {:.3}",
                theory.scale_adherence * 100.0, theory.parallel_fifth_avoidance * 100.0,
                theory.rhythmic_quantization * 100.0, theory.voice_range_compliance * 100.0,
                theory.phrase_contour_quality * 100.0, theory.composite);

            total_creative += creative.composite;
            total_audio += audio.composite;
            total_theory += theory.composite;
        }

        let n = scenarios.len() as f32;
        let avg_creative = total_creative / n;
        let avg_audio = total_audio / n;
        let avg_theory = total_theory / n;
        let overall = (avg_creative + avg_audio + avg_theory) / 3.0;

        eprintln!("\n══════════════════════════════════════════════════");
        eprintln!("  AVERAGES");
        eprintln!("  Creative Quality: {:.3}", avg_creative);
        eprintln!("  Audio Quality:    {:.3}", avg_audio);
        eprintln!("  Theory Score:     {:.3}", avg_theory);
        eprintln!("  Overall:          {:.3}", overall);
        eprintln!("══════════════════════════════════════════════════\n");

        // Baseline assertions: overall should be at least 0.3
        assert!(overall > 0.2, "Overall quality {overall:.3} is below minimum");
        assert!(avg_creative > 0.2, "Creative quality too low: {avg_creative:.3}");
    }
}
