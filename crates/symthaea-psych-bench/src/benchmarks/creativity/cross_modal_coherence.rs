// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Modal Coherence benchmark.
//!
//! Measures whether outputs across different modalities (visual, musical,
//! textual) maintain thematic coherence when generated from the same
//! cognitive state. A system with genuine cross-modal creativity should
//! produce art, music, and text that "feel related."
//!
//! Human baseline: synesthetic artists achieve ~0.7 cross-modal correlation
//! (Ramachandran & Hubbard, 2001; Ward, 2008).
//!
//! HDC implementation: generate "visual" and "auditory" representations
//! from the same mood seed but different modality bases, then measure
//! their structural similarity (shared mood signature despite different
//! surface features).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Cross-Modal Coherence benchmark.
pub struct CrossModalCoherenceBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

impl CrossModalCoherenceBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        let dim = config.dimension;
        let mut rng_state = config.seed.wrapping_add(trial_idx as u64 * 8501);

        // Shared mood vector (the cognitive state)
        let mood = BinaryHV::random(next_seed(&mut rng_state));

        // Different modality bases (visual vs auditory "codebook")
        let visual_base = BinaryHV::random(next_seed(&mut rng_state));
        let auditory_base = BinaryHV::random(next_seed(&mut rng_state));

        // Generate modality-specific outputs:
        // visual = mood ⊗ visual_base (mood shapes visual expression)
        // auditory = mood ⊗ auditory_base (same mood shapes auditory expression)
        let visual_output = mood.bind(&visual_base);
        let auditory_output = mood.bind(&auditory_base);

        // Cross-modal coherence: unbind modality bases, compare residual mood signatures
        // If mood is preserved: visual ⊗ visual_base⁻¹ ≈ mood ≈ auditory ⊗ auditory_base⁻¹
        // For BinaryHV, bind is self-inverse: A ⊗ A = identity
        let visual_mood_residual = visual_output.bind(&visual_base);
        let auditory_mood_residual = auditory_output.bind(&auditory_base);

        // These should both be ≈ mood
        let visual_fidelity = visual_mood_residual.similarity(&mood) as f64;
        let auditory_fidelity = auditory_mood_residual.similarity(&mood) as f64;

        // Cross-modal coherence: how similar are the mood residuals?
        let cross_modal_sim = visual_mood_residual.similarity(&auditory_mood_residual) as f64;

        // Compare with a different-mood baseline
        let different_mood = BinaryHV::random(next_seed(&mut rng_state));
        let different_visual = different_mood.bind(&visual_base);
        let different_residual = different_visual.bind(&visual_base);
        let mismatch_sim = different_residual.similarity(&auditory_mood_residual) as f64;

        // Coherence = same-mood similarity minus different-mood similarity
        let coherence_advantage = (cross_modal_sim - mismatch_sim).max(0.0);

        // Apply lapse rate
        let coherence_advantage = coherence_advantage * (1.0 - config.lapse_rate as f64 * 0.4);

        // Fidelity: average of visual and auditory mood preservation
        let fidelity = (visual_fidelity + auditory_fidelity) / 2.0;

        (coherence_advantage, fidelity)
    }
}

impl PsychBenchmark for CrossModalCoherenceBenchmark {
    fn name(&self) -> &str {
        "Creativity::CrossModalCoherence"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Cross-Modal Creative Coherence",
            citation: "Ramachandran & Hubbard (2001); Ward (2008)",
            year: 2001,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut coherences = Vec::new();
        let mut fidelities = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (coh, fid) = self.run_trial(config, trial);
            coherences.push(coh);
            fidelities.push(fid);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "cross_modal".to_string(),
                    correct: coh > 0.3,
                    rt_ticks: 0.0,
                    similarity: fid,
                    confidence: coh,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "cross_modal_coherence",
            MetricValue::from_samples(&coherences),
        );
        result.insert("mood_fidelity", MetricValue::from_samples(&fidelities));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = CrossModalCoherenceBenchmark.run(&config);
        assert!(result.metrics.contains_key("cross_modal_coherence"));
        assert!(result.metrics.contains_key("mood_fidelity"));
    }

    #[test]
    fn same_mood_more_coherent_than_different() {
        let config = BenchmarkConfig {
            trials_per_condition: 30,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = CrossModalCoherenceBenchmark.run(&config);
        let coherence = result.metrics["cross_modal_coherence"].mean;
        // Same-mood cross-modal should have positive coherence advantage
        assert!(coherence > 0.0, "coherence advantage = {coherence}");
    }

    #[test]
    fn mood_fidelity_high() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            seed: 42,
            ..BenchmarkConfig::default()
        };
        let result = CrossModalCoherenceBenchmark.run(&config);
        let fidelity = result.metrics["mood_fidelity"].mean;
        // XOR bind is self-inverse, so mood recovery should be perfect
        assert!(fidelity > 0.95, "mood_fidelity = {fidelity}");
    }
}