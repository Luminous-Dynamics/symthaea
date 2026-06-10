// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice Onset Time (VOT) Continuum benchmark.
//!
//! Tests categorical perception along the VOT dimension — the time between
//! lip release and voicing onset that distinguishes /ba/ from /pa/.
//! Stimuli are HDC interpolations along a 10-step continuum. The system
//! must identify each stimulus as voiced (/ba/) or voiceless (/pa/).
//!
//! A sharp identification boundary (step function, not linear) is the
//! hallmark of categorical perception (Lisker & Abramson, 1964).
//!
//! Human baselines (Lisker & Abramson, 1964; Eimas et al., 1971):
//! - boundary_width: 2.0 steps (SD 0.8) — narrower = sharper boundary
//! - identification_accuracy: 0.92 (SD 0.06) — at continuum endpoints
//! - slope_at_boundary: 0.35 (SD 0.10) — logistic slope at 50% crossover

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// VOT Continuum benchmark.
pub struct VotContinuumBenchmark;

struct TrialResult {
    boundary_width: f64,
    identification_accuracy: f64,
    slope_at_boundary: f64,
}

impl VotContinuumBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("speech", "vot_continuum", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let n_steps = 10usize;
        let noise = config.effective_noise();

        // Category prototypes: /ba/ (voiced) and /pa/ (voiceless)
        let ba_proto = ContinuousHV::random(dim, seed.wrapping_add(100));
        let pa_proto = ContinuousHV::random(dim, seed.wrapping_add(200));

        // Generate continuum by weighted interpolation
        let continuum: Vec<ContinuousHV> = (0..n_steps)
            .map(|step| {
                let t = step as f32 / (n_steps - 1) as f32;
                ContinuousHV::weighted_bundle(&[&ba_proto, &pa_proto], &[1.0 - t, t])
            })
            .collect();

        // Add encoding noise to each stimulus
        let noise_frac = 0.03 + noise as f32 * 0.08;

        let trials_per_step = 20;
        let mut identifications = vec![0.0f64; n_steps]; // proportion "pa" responses

        for step in 0..n_steps {
            let mut pa_count = 0u32;
            for trial in 0..trials_per_step {
                xor_shift(&mut rng);
                let noise_hv =
                    ContinuousHV::random(dim, rng.wrapping_add(step as u64 * 100 + trial as u64));
                let stimulus = ContinuousHV::weighted_bundle(
                    &[&continuum[step], &noise_hv],
                    &[1.0 - noise_frac, noise_frac],
                );

                // Per-comparison encoding noise: individual differences in
                // phoneme boundary sharpness (Pisoni & Tash, 1974).
                let ba_noise = {
                    let ns = rng.wrapping_add(7000 + step as u64 * 11 + trial as u64 * 3);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                let pa_noise = {
                    let ns = rng.wrapping_add(7001 + step as u64 * 11 + trial as u64 * 3);
                    ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                };
                let sim_ba = stimulus.similarity(&ba_proto) + ba_noise * noise as f32 * 0.12;
                let sim_pa = stimulus.similarity(&pa_proto) + pa_noise * noise as f32 * 0.12;

                let vot_trial_idx = step * trials_per_step + trial;
                let chose_pa = if config.should_lapse("vot_continuum", vot_trial_idx) {
                    // Attention lapse: random 50/50 guess
                    (rng >> 63) == 1
                } else {
                    sim_pa > sim_ba
                };
                if chose_pa {
                    pa_count += 1;
                }
            }
            identifications[step] = pa_count as f64 / trials_per_step as f64;
        }

        // Identification accuracy: mean of endpoint clarity
        // Steps 0-1 should be ~0% "pa", steps 8-9 should be ~100% "pa"
        let endpoint_acc = ((1.0 - identifications[0])
            + (1.0 - identifications[1])
            + identifications[n_steps - 2]
            + identifications[n_steps - 1])
            / 4.0;

        // Find boundary: step where identification crosses 0.5
        let mut boundary_step = 4.5f64; // default midpoint
        for i in 0..n_steps - 1 {
            if identifications[i] < 0.5 && identifications[i + 1] >= 0.5 {
                // Linear interpolation
                let frac = (0.5 - identifications[i])
                    / (identifications[i + 1] - identifications[i]).max(0.001);
                boundary_step = i as f64 + frac;
                break;
            }
        }

        // Boundary width: count steps within 0.25-0.75 identification range
        let transition_steps = identifications
            .iter()
            .filter(|&&id| id > 0.25 && id < 0.75)
            .count();
        let boundary_width = transition_steps as f64;

        // Slope at boundary: approximate logistic slope
        // Use the two steps flanking the boundary
        let b_low = (boundary_step as usize).min(n_steps - 2);
        let b_high = (b_low + 1).min(n_steps - 1);
        let slope = (identifications[b_high] - identifications[b_low]).abs();

        TrialResult {
            boundary_width,
            identification_accuracy: endpoint_acc,
            slope_at_boundary: slope,
        }
    }
}

impl PsychBenchmark for VotContinuumBenchmark {
    fn name(&self) -> &str {
        "Speech::VotContinuum"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Voice Onset Time Categorical Perception",
            citation: "Lisker & Abramson (1964)",
            year: 1964,
            doi: Some("10.1121/1.1918616"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut widths = Vec::new();
        let mut accs = Vec::new();
        let mut slopes = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            widths.push(r.boundary_width);
            accs.push(r.identification_accuracy);
            slopes.push(r.slope_at_boundary);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "vot_continuum".to_string(),
                    correct: r.identification_accuracy > 0.8,
                    rt_ticks: 0.0,
                    similarity: r.identification_accuracy,
                    confidence: r.slope_at_boundary,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("boundary_width", MetricValue::from_samples(&widths));
        result.insert("identification_accuracy", MetricValue::from_samples(&accs));
        result.insert("slope_at_boundary", MetricValue::from_samples(&slopes));

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
    fn test_vot_continuum_runs() {
        let config = BenchmarkConfig::default();
        let result = VotContinuumBenchmark.run(&config);
        assert!(result.metrics.contains_key("boundary_width"));
        assert!(result.metrics.contains_key("identification_accuracy"));
        assert!(result.metrics.contains_key("slope_at_boundary"));
    }

    #[test]
    fn test_vot_continuum_finite() {
        let config = BenchmarkConfig::default();
        let result = VotContinuumBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_vot_endpoint_accuracy_high() {
        let config = BenchmarkConfig::default();
        let result = VotContinuumBenchmark.run(&config);
        let acc = result.metrics["identification_accuracy"].mean;
        assert!(acc > 0.7, "endpoint accuracy {acc} too low");
    }
}
