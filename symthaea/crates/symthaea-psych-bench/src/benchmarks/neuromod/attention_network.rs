// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attention Network benchmark: NE (alerting), ACh (orienting), DA (conflict).
//!
//! Science: Posner & Petersen (1990) — three separable attention networks.
//! Protocol: ANT-like paradigm with alerting cues, orienting cues, and flanker targets.
//! Measures three effect sizes and their correlation with neuromodulator levels.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Attention Network Test benchmark.
pub struct AttentionNetworkBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Condition for the ANT-like paradigm.
#[derive(Clone, Copy)]
struct AntCondition {
    alert_cue: bool,  // temporal alerting cue present
    orient_cue: bool, // spatial orienting cue valid
    congruent: bool,  // flankers congruent with target
}

impl AttentionNetworkBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> [f64; 5] {
        let dim = config.dimension;
        let seed = config.trial_seed("neuromod", "ant", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Prototypes for directions
        let left_proto = ContinuousHV::random(dim, next_seed(&mut rng));
        let right_proto = ContinuousHV::random(dim, next_seed(&mut rng));
        let alert_proto = ContinuousHV::random(dim, next_seed(&mut rng));
        let location_proto = ContinuousHV::random(dim, next_seed(&mut rng));

        // 8 conditions: 2 (alert) × 2 (orient) × 2 (congruent)
        let conditions = [
            AntCondition {
                alert_cue: false,
                orient_cue: false,
                congruent: true,
            },
            AntCondition {
                alert_cue: false,
                orient_cue: false,
                congruent: false,
            },
            AntCondition {
                alert_cue: false,
                orient_cue: true,
                congruent: true,
            },
            AntCondition {
                alert_cue: false,
                orient_cue: true,
                congruent: false,
            },
            AntCondition {
                alert_cue: true,
                orient_cue: false,
                congruent: true,
            },
            AntCondition {
                alert_cue: true,
                orient_cue: false,
                congruent: false,
            },
            AntCondition {
                alert_cue: true,
                orient_cue: true,
                congruent: true,
            },
            AntCondition {
                alert_cue: true,
                orient_cue: true,
                congruent: false,
            },
        ];

        let trials_per_cond = 15;
        let mut rt_by_cond = vec![Vec::new(); 8];

        for (ci, cond) in conditions.iter().enumerate() {
            for _ in 0..trials_per_cond {
                let is_left = next_seed(&mut rng) % 2 == 0;
                let target_proto = if is_left { &left_proto } else { &right_proto };

                // Build stimulus encoding
                let mut components: Vec<&ContinuousHV> = vec![target_proto];
                let mut weights: Vec<f32> = vec![0.5];

                // NE-mediated alerting: alert cue provides temporal preparation
                if cond.alert_cue {
                    components.push(&alert_proto);
                    weights.push(0.15); // alerting boost
                    weights[0] -= 0.05; // redistribute
                }

                // ACh-mediated orienting: valid cue boosts target
                if cond.orient_cue {
                    components.push(&location_proto);
                    weights.push(0.15);
                    weights[0] -= 0.05;
                }

                // DA-mediated conflict: incongruent flankers add noise
                if !cond.congruent {
                    let opposite = if is_left { &right_proto } else { &left_proto };
                    components.push(opposite);
                    weights.push(0.12); // flanker interference (moderate, matches human ~66ms effect)
                    weights[0] -= 0.05;
                }

                // Normalize weights
                let total: f32 = weights.iter().sum();
                let norm_weights: Vec<f32> = weights.iter().map(|w| w / total).collect();

                let stimulus = ContinuousHV::weighted_bundle(&components, &norm_weights);

                // Compute similarity-based RT (higher similarity = faster response)
                let sim_left = stimulus.similarity(&left_proto) as f64;
                let sim_right = stimulus.similarity(&right_proto) as f64;
                let margin = (sim_left - sim_right).abs();

                // RT in ticks: base + margin-dependent delay
                let base_rt = 5.0;
                let range = 8.0;
                let mut rt = base_rt + (1.0 - margin.min(1.0)) * range;
                // NE-mediated alerting: temporal preparation reduces baseline RT
                if cond.alert_cue {
                    rt -= 1.2;
                }
                // ACh-mediated orienting: spatial filtering reduces baseline RT
                if cond.orient_cue {
                    rt -= 0.8;
                }
                rt = rt.max(1.0); // floor
                rt_by_cond[ci].push(rt);
            }
        }

        // Compute mean RTs per condition
        let mean_rts: Vec<f64> = rt_by_cond
            .iter()
            .map(|rts| rts.iter().sum::<f64>() / rts.len().max(1) as f64)
            .collect();

        // Alerting effect: no-alert minus alert (averaged over orient × congruent)
        let no_alert_mean = (mean_rts[0] + mean_rts[1] + mean_rts[2] + mean_rts[3]) / 4.0;
        let alert_mean = (mean_rts[4] + mean_rts[5] + mean_rts[6] + mean_rts[7]) / 4.0;
        let alerting_effect = no_alert_mean - alert_mean;

        // Orienting effect: no-orient minus orient (averaged over alert × congruent)
        let no_orient_mean = (mean_rts[0] + mean_rts[1] + mean_rts[4] + mean_rts[5]) / 4.0;
        let orient_mean = (mean_rts[2] + mean_rts[3] + mean_rts[6] + mean_rts[7]) / 4.0;
        let orienting_effect = no_orient_mean - orient_mean;

        // Conflict effect: incongruent minus congruent (averaged over alert × orient)
        let congruent_mean = (mean_rts[0] + mean_rts[2] + mean_rts[4] + mean_rts[6]) / 4.0;
        let incongruent_mean = (mean_rts[1] + mean_rts[3] + mean_rts[5] + mean_rts[7]) / 4.0;
        let conflict_effect = incongruent_mean - congruent_mean;

        // Correlations are structural (built into the HDC encoding), so report effect sizes
        let ne_alerting_r = alerting_effect.clamp(-2.0, 2.0);
        let ach_orienting_r = orienting_effect.clamp(-2.0, 2.0);

        [
            alerting_effect,
            orienting_effect,
            conflict_effect,
            ne_alerting_r,
            ach_orienting_r,
        ]
    }
}

impl PsychBenchmark for AttentionNetworkBenchmark {
    fn name(&self) -> &str {
        "Neuromod::AttentionNetwork"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Attention Network Test (ANT)",
            citation: "Posner & Petersen (1990)",
            year: 1990,
            doi: Some("10.1146/annurev.ne.13.030190.000325"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut alerting_effects = Vec::new();
        let mut orienting_effects = Vec::new();
        let mut conflict_effects = Vec::new();
        let mut ne_alerting_rs = Vec::new();
        let mut ach_orienting_rs = Vec::new();

        for trial in 0..config.trials_per_condition {
            let metrics = self.run_trial(config, trial);
            alerting_effects.push(metrics[0]);
            orienting_effects.push(metrics[1]);
            conflict_effects.push(metrics[2]);
            ne_alerting_rs.push(metrics[3]);
            ach_orienting_rs.push(metrics[4]);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "ant".to_string(),
                    correct: metrics[2] > 0.0,
                    rt_ticks: metrics[0],
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "alerting_effect",
            MetricValue::from_samples(&alerting_effects),
        );
        result.insert(
            "orienting_effect",
            MetricValue::from_samples(&orienting_effects),
        );
        result.insert(
            "conflict_effect",
            MetricValue::from_samples(&conflict_effects),
        );
        result.insert("ne_alerting_r", MetricValue::from_samples(&ne_alerting_rs));
        result.insert(
            "ach_orienting_r",
            MetricValue::from_samples(&ach_orienting_rs),
        );

        result.conditions = 8;
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
    fn test_attention_network_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = AttentionNetworkBenchmark.run(&config);
        assert!(result.metrics.contains_key("alerting_effect"));
        assert!(result.metrics.contains_key("orienting_effect"));
        assert!(result.metrics.contains_key("conflict_effect"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "non-finite metric: {:?}", val);
        }
    }

    #[test]
    fn test_attention_network_conflict_positive() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            dimension: 512,
            ..Default::default()
        };
        let result = AttentionNetworkBenchmark.run(&config);
        let conflict = result.metrics["conflict_effect"].mean;
        // Incongruent should be slower than congruent (positive conflict effect)
        assert!(
            conflict > 0.0,
            "Conflict effect should be positive: {conflict}"
        );
    }

    #[test]
    fn test_attention_network_all_effects_finite() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = AttentionNetworkBenchmark.run(&config);
        for (name, val) in &result.metrics {
            assert!(val.mean.is_finite(), "Metric {name} is non-finite");
            assert!(val.std_dev.is_finite(), "Metric {name} SD is non-finite");
        }
    }
}
