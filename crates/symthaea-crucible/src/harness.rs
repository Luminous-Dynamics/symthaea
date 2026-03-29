// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-cycle simulation harness for running thought experiments.
//!
//! The harness feeds scenario steps through the harmony basis pipeline,
//! collecting telemetry at each step and checking invariants.

use symthaea_core::hdc::ContinuousHV;
use symthaea_harmonies::{AlignmentResult, EightHarmonies};
use symthaea_types::N_HARMONIES;

use crate::invariants;
use crate::scenarios::ScenarioStep;
use crate::telemetry::{AnomalyFlags, CrucibleReport, CycleTelemetry};

/// Configuration for a crucible run.
#[derive(Debug, Clone)]
pub struct CrucibleConfig {
    /// HDC dimension for encoding.
    pub dim: usize,
    /// N-gram size for character encoding.
    pub ngram_size: usize,
}

impl Default for CrucibleConfig {
    fn default() -> Self {
        Self {
            dim: 4096,
            ngram_size: 3,
        }
    }
}

/// The crucible engine: encodes scenario text, evaluates against the Eight
/// Harmonies via HDC similarity, and tracks moral trajectory.
pub struct CrucibleEngine {
    config: CrucibleConfig,
    harmonies: EightHarmonies,
    /// Per-harmony basis vectors (encoded from keyword sets).
    basis_vectors: Vec<ContinuousHV>,
    /// Running harmony coordinate history (sliding window).
    trajectory: Vec<[f64; N_HARMONIES]>,
    /// Previous harmony coordinates (for drift detection).
    prev_coords: Option<[f64; N_HARMONIES]>,
    /// Running mean of harmony coordinates (moral prior).
    prior: [f64; N_HARMONIES],
    prior_count: u64,
}

impl CrucibleEngine {
    /// Create a new crucible engine with default config.
    pub fn new() -> Self {
        Self::with_config(CrucibleConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: CrucibleConfig) -> Self {
        // Build harmony basis vectors from keyword sets using N-gram encoding
        let harmony_keywords = [
            "integrate harmonize unify coherent order luminous resonant balance alignment wholeness",
            "help support care benefit serve protect nurture compassion flourishing kindness love",
            "learn understand wisdom knowledge insight intelligence truth awareness knowing embodied",
            "create explore play discover experiment joy creativity generativity novelty imagination",
            "connect share collaborate together community unity empathy resonance interconnected belonging",
            "give share contribute reciprocate exchange generous flow mutual upliftment trust",
            "grow evolve improve progress develop transcend becoming evolution advancement consciousness",
            "stillness silence rest surrender void release contemplation emptiness peace presence apophatic",
        ];

        let basis_vectors: Vec<ContinuousHV> = harmony_keywords
            .iter()
            .map(|kw| ngram_encode(kw, config.dim, config.ngram_size))
            .collect();

        Self {
            config,
            harmonies: EightHarmonies::new(),
            basis_vectors,
            trajectory: Vec::new(),
            prev_coords: None,
            prior: [0.0; N_HARMONIES],
            prior_count: 0,
        }
    }

    /// Encode text into an HDC vector via N-gram bundling.
    pub fn encode(&self, text: &str) -> ContinuousHV {
        ngram_encode(text, self.config.dim, self.config.ngram_size)
    }

    /// Project an action HV onto the 8D harmony manifold.
    pub fn project(&self, hv: &ContinuousHV) -> [f64; N_HARMONIES] {
        let mut coords = [0.0f64; N_HARMONIES];
        for (i, basis) in self.basis_vectors.iter().enumerate() {
            coords[i] = hv.similarity(basis) as f64;
        }
        coords
    }

    /// Evaluate a single scenario step. Returns telemetry + alignment result.
    pub fn evaluate_step(
        &mut self,
        step: &ScenarioStep,
        cycle: u64,
    ) -> (CycleTelemetry, AlignmentResult) {
        let action_hv = self.encode(&step.text);
        let coords = self.project(&action_hv);

        // HDC-based harmony evaluation
        let alignment = self.harmonies.evaluate_hdc(&action_hv, &self.basis_vectors);

        // Compute moral free energy: F = KL(q || p) + H(q)
        let q = softmax_8(&coords);
        let p = if self.prior_count > 0 {
            softmax_8(&self.prior)
        } else {
            [1.0 / N_HARMONIES as f64; N_HARMONIES]
        };
        let kl = kl_divergence(&q, &p);
        let entropy = shannon_entropy(&q);
        let free_energy = kl + entropy;

        // Update prior via EMA
        let alpha = if self.prior_count == 0 { 1.0 } else { 0.1 };
        for (i, &c) in coords.iter().enumerate().take(N_HARMONIES) {
            self.prior[i] = self.prior[i] * (1.0 - alpha) + c * alpha;
        }
        self.prior_count += 1;

        // Drift detection
        let drift = if let Some(prev) = &self.prev_coords {
            l2_distance(&coords, prev)
        } else {
            0.0
        };
        self.prev_coords = Some(coords);

        // Simple anomaly detection
        let value_inversion = self.trajectory.len() >= 4 && {
            let prev_dominant = dominant_axis(&self.trajectory[self.trajectory.len() - 1]);
            let curr_dominant = dominant_axis(&coords);
            prev_dominant != curr_dominant
        };
        let fe_spike = self.trajectory.len() >= 5 && free_energy > 2.0;

        let anomaly_flags = AnomalyFlags {
            value_inversion,
            fe_spike,
            drift_alert: drift > 0.3,
            ..Default::default()
        };
        let anomaly_score = (value_inversion as u8 as f64 * 0.3
            + fe_spike as u8 as f64 * 0.3
            + anomaly_flags.drift_alert as u8 as f64 * 0.2)
            .min(1.0);

        // Determine moral verdict from harmony alignment
        let moral_score = alignment.overall_score;
        let text_lower = step.text.to_lowercase();
        let moral_verdict = if text_lower.contains("without consent")
            || (text_lower.contains("force") && text_lower.contains("against"))
        {
            "ConsentViolation".to_string()
        } else if moral_score > 0.3 {
            "Good".to_string()
        } else if moral_score < -0.3 {
            "Bad".to_string()
        } else {
            "Neutral".to_string()
        };

        let consent_violation = moral_verdict == "ConsentViolation";

        self.trajectory.push(coords);

        let telemetry = CycleTelemetry {
            cycle,
            moral_score,
            moral_verdict,
            consent_violation,
            harmony_coordinates: coords,
            moral_free_energy: free_energy,
            anomaly_score,
            anomaly_flags,
        };

        (telemetry, alignment)
    }

    /// Run a complete scenario and produce a report.
    pub fn run_scenario(&mut self, name: &str, steps: &[ScenarioStep]) -> CrucibleReport {
        let mut report = CrucibleReport::new(name);

        for (i, step) in steps.iter().enumerate() {
            let cycle = (i as u64 + 1) * 7;
            let (telemetry, _alignment) = self.evaluate_step(step, cycle);

            // Check invariants
            let violations = invariants::check_universal(&telemetry);
            report.invariant_violations.extend(violations);

            report
                .harmony_trajectory
                .push(telemetry.harmony_coordinates);
            report.free_energy_trace.push(telemetry.moral_free_energy);
            report.cycle_telemetry.push(telemetry);
        }

        // Check trajectory invariants
        let trajectory_violations = invariants::check_trajectory(&report.cycle_telemetry);
        report.invariant_violations.extend(trajectory_violations);

        report.total_cycles = steps.len() as u64;
        report.finalize();
        report
    }

    /// Reset engine state for a fresh scenario run.
    pub fn reset(&mut self) {
        self.trajectory.clear();
        self.prev_coords = None;
        self.prior = [0.0; N_HARMONIES];
        self.prior_count = 0;
    }

    /// Access the harmony trajectory.
    pub fn trajectory(&self) -> &[[f64; N_HARMONIES]] {
        &self.trajectory
    }

    /// Get the HDC dimension.
    pub fn dim(&self) -> usize {
        self.config.dim
    }
}

impl Default for CrucibleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── N-gram text encoder ──────────────────────────────────────────────────
//
// Encodes text as the bundle (element-wise sum) of character N-gram HVs.
// Each N-gram is deterministically hashed to a pseudo-random ContinuousHV.
// This mirrors the dual-channel approach in the main crate's TextHdcEncoder.

fn ngram_encode(text: &str, dim: usize, ngram_size: usize) -> ContinuousHV {
    let text = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();

    if chars.is_empty() {
        return ContinuousHV::zero(dim);
    }

    let mut accumulator = vec![0.0f32; dim];
    let mut count = 0u32;

    // Character N-grams
    for window in chars.windows(ngram_size.max(1)) {
        let ngram: String = window.iter().collect();
        let hv = hash_to_hv(&ngram, dim);
        for (a, h) in accumulator.iter_mut().zip(hv.iter()) {
            *a += h;
        }
        count += 1;
    }

    // Word-level encoding (bundled on top of N-grams)
    for word in text.split_whitespace() {
        if !word.is_empty() {
            let hv = hash_to_hv(word, dim);
            for (a, h) in accumulator.iter_mut().zip(hv.iter()) {
                *a += h;
            }
            count += 1;
        }
    }

    // Normalize
    if count > 0 {
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for a in &mut accumulator {
                *a /= norm;
            }
        }
    }

    ContinuousHV::from_slice(&accumulator)
}

/// Deterministic hash of a string to a pseudo-random f32 vector.
fn hash_to_hv(s: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut result = vec![0.0f32; dim];
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    let mut seed = hasher.finish();

    for val in &mut result {
        // Simple xorshift64 PRNG seeded from the hash
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Map to [-1, 1]
        *val = ((seed as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
    }

    result
}

// ── Math helpers ──────────────────────────────────────────────────────────

fn softmax_8(x: &[f64; N_HARMONIES]) -> [f64; N_HARMONIES] {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut result = [0.0; N_HARMONIES];
    let mut sum = 0.0;
    for i in 0..N_HARMONIES {
        result[i] = (x[i] - max).exp();
        sum += result[i];
    }
    let sum = sum.max(1e-12);
    for v in &mut result {
        *v /= sum;
    }
    result
}

fn kl_divergence(q: &[f64; N_HARMONIES], p: &[f64; N_HARMONIES]) -> f64 {
    let mut kl = 0.0;
    for i in 0..N_HARMONIES {
        let qi = q[i].max(1e-12);
        let pi = p[i].max(1e-12);
        kl += qi * (qi / pi).ln();
    }
    kl.max(0.0)
}

fn shannon_entropy(q: &[f64; N_HARMONIES]) -> f64 {
    let mut h = 0.0;
    for &qi in q {
        let qi = qi.max(1e-12);
        h -= qi * qi.ln();
    }
    h.max(0.0)
}

fn l2_distance(a: &[f64; N_HARMONIES], b: &[f64; N_HARMONIES]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dominant_axis(coords: &[f64; N_HARMONIES]) -> usize {
    coords
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = CrucibleEngine::new();
        assert_eq!(engine.dim(), 4096);
        assert_eq!(engine.basis_vectors.len(), N_HARMONIES);
    }

    #[test]
    fn test_encode_produces_finite() {
        let engine = CrucibleEngine::new();
        let hv = engine.encode("help the community grow");
        let coords = engine.project(&hv);
        for c in &coords {
            assert!(c.is_finite(), "coordinate should be finite: {c}");
        }
    }

    #[test]
    fn test_different_texts_different_encodings() {
        let engine = CrucibleEngine::new();
        let hv1 = engine.encode("help the community grow");
        let hv2 = engine.encode("destroy and harm everything");
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.95,
            "Different texts should produce different encodings (sim={sim})"
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = [0.1, 0.5, -0.2, 0.3, 0.1, 0.0, 0.4, -0.1];
        let s = softmax_8(&x);
        let sum: f64 = s.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_kl_divergence_zero_for_same() {
        let p = [0.125; N_HARMONIES];
        let kl = kl_divergence(&p, &p);
        assert!(kl < 1e-10, "KL(p||p) should be ~0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_positive() {
        let q = [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05];
        let p = [0.125; N_HARMONIES];
        let kl = kl_divergence(&q, &p);
        assert!(kl > 0.0, "KL(q||p) should be positive, got {kl}");
    }

    #[test]
    fn test_run_simple_scenario() {
        let mut engine = CrucibleEngine::new();
        let steps = vec![
            ScenarioStep::new("help others learn and grow together"),
            ScenarioStep::new("share knowledge with the community"),
            ScenarioStep::new("create something beautiful and new"),
        ];
        let report = engine.run_scenario("simple_positive", &steps);
        assert_eq!(report.total_cycles, 3);
        assert!(
            report.passed(),
            "Simple positive scenario should pass: {:?}",
            report.invariant_violations
        );
    }

    #[test]
    fn test_empty_input_no_panic() {
        let mut engine = CrucibleEngine::new();
        let steps = vec![ScenarioStep::new("")];
        let report = engine.run_scenario("empty", &steps);
        assert!(
            report.passed(),
            "Empty input should pass: {:?}",
            report.invariant_violations
        );
    }
}
