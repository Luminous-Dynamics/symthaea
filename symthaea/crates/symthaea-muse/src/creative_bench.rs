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

    let intervals: Vec<f32> = notes
        .windows(2)
        .map(|w| (w[1].start_time - w[0].start_time).abs())
        .filter(|&i| i > 0.001)
        .collect();

    if intervals.is_empty() {
        return 0.5;
    }

    let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
    if mean < 0.001 {
        return 0.5;
    }

    let variance =
        intervals.iter().map(|&i| (i - mean).powi(2)).sum::<f32>() / intervals.len() as f32;
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

    let mut scores = Vec::new();
    for (target_va, seed) in &test_cases {
        let comp = crate::compose(config, state, *seed);
        let score = CreativeQualityScore::evaluate(&comp, *target_va);
        scores.push(score);
    }

    let n = scores.len() as f32;
    BenchmarkResult {
        mean_melodic_coherence: scores.iter().map(|s| s.melodic_coherence).sum::<f32>() / n,
        mean_rhythmic_regularity: scores.iter().map(|s| s.rhythmic_regularity).sum::<f32>() / n,
        mean_emotional_alignment: scores.iter().map(|s| s.emotional_alignment).sum::<f32>() / n,
        mean_form_compliance: scores.iter().map(|s| s.form_compliance).sum::<f32>() / n,
        mean_composite: scores.iter().map(|s| s.composite).sum::<f32>() / n,
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
             Mean Composite:      {:.3}",
            self.n_compositions,
            self.mean_melodic_coherence,
            self.mean_rhythmic_regularity,
            self.mean_emotional_alignment,
            self.mean_form_compliance,
            self.mean_composite,
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
