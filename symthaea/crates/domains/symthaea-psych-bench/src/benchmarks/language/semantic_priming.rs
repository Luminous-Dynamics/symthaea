// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Priming Task.
//!
//! Related words (e.g., "doctor"→"nurse") are recognized faster than unrelated
//! pairs. SOA (stimulus onset asynchrony) modulates the priming effect: short
//! SOA produces automatic priming; long SOA adds strategic processing.
//!
//! HDC implementation: 5 semantic clusters of 4 HVs each, generated via
//! weighted_bundle with a cluster center. Related pairs come from the same
//! cluster; unrelated pairs from different clusters. SOA is modeled as
//! SSM temporal steps: 1 step (short/automatic) vs 3 steps (long/strategic).
//!
//! Human baselines (Neely, 1977; McNamara, 2005):
//! - priming_effect: 0.10 (SD~0.04) — accuracy boost for related vs unrelated
//! - related_accuracy: 0.90 (SD~0.06)
//! - unrelated_accuracy: 0.80 (SD~0.08)
//! - soa_modulation: 0.05 (SD~0.03) — priming difference between short and long SOA

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Semantic Priming benchmark.
pub struct SemanticPrimingBenchmark;

struct TrialResult {
    priming_effect: f64,
    related_accuracy: f64,
    unrelated_accuracy: f64,
    soa_modulation: f64,
    rt_ticks: f64,
}

impl SemanticPrimingBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("language", "semantic_priming", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // 5 semantic clusters, 4 words each
        let n_clusters = 5;
        let words_per_cluster = 4;
        let mut clusters: Vec<Vec<ContinuousHV>> = Vec::with_capacity(n_clusters);

        for c in 0..n_clusters {
            let center = ContinuousHV::random(dim, seed.wrapping_add(c as u64 * 200 + 1));
            let mut cluster = Vec::with_capacity(words_per_cluster);
            for w in 0..words_per_cluster {
                let noise = ContinuousHV::random(
                    dim,
                    seed.wrapping_add(c as u64 * 200 + w as u64 * 50 + 100),
                );
                // Cluster members share 80% of cluster center (stronger semantic structure)
                let word = ContinuousHV::weighted_bundle(&[&center, &noise], &[0.80, 0.20]);
                cluster.push(word);
            }
            clusters.push(cluster);
        }

        // Time pressure noise
        let noise_level: f32 = (0.20 + config.time_pressure as f32 * 0.15)
            * diff_model.interference_multiplier(config.difficulty) as f32;

        let n_pairs = 20; // 10 related + 10 unrelated
        let mut related_correct_short = 0u32;
        let mut related_total_short = 0u32;
        let mut related_correct_long = 0u32;
        let mut related_total_long = 0u32;
        let mut unrelated_correct_short = 0u32;
        let mut unrelated_total_short = 0u32;
        let mut unrelated_correct_long = 0u32;
        let mut unrelated_total_long = 0u32;
        let mut rt_sum = 0.0f64;

        for pair_idx in 0..n_pairs {
            let is_related = pair_idx < n_pairs / 2;
            xor_shift(&mut rng);
            let is_short_soa = (rng % 2) == 0;

            // Select prime cluster and word
            xor_shift(&mut rng);
            let prime_cluster = (rng % n_clusters as u64) as usize;
            xor_shift(&mut rng);
            let prime_word_idx = (rng % words_per_cluster as u64) as usize;
            let prime = &clusters[prime_cluster][prime_word_idx];

            // Select target
            let (target_cluster, target_word_idx) = if is_related {
                // Same cluster, different word
                xor_shift(&mut rng);
                let tw = ((prime_word_idx + 1) + (rng % (words_per_cluster as u64 - 1)) as usize)
                    % words_per_cluster;
                (prime_cluster, tw)
            } else {
                // Different cluster
                xor_shift(&mut rng);
                let tc =
                    ((prime_cluster + 1) + (rng % (n_clusters as u64 - 1)) as usize) % n_clusters;
                xor_shift(&mut rng);
                let tw = (rng % words_per_cluster as u64) as usize;
                (tc, tw)
            };
            let target = &clusters[target_cluster][target_word_idx];

            // SOA processing: prime activation decays exponentially over SOA steps.
            // Science: Neely (1977) — automatic priming decays with SOA;
            // McNamara (2005) — spreading activation with temporal decay.
            let soa_steps = if is_short_soa { 1 } else { 3 };
            let decay = config.language_priming_decay;
            let mut activation = prime.similarity(target).max(0.0) as f64;
            for _ in 0..soa_steps {
                activation *= decay; // Exponential decay per SOA step
            }
            // Also run through SSM for backward compatibility with ssm_backend
            let mut ssm = SsmTemporalBackend::new(-0.10, 4);
            let prime_activation = prime.similarity(target);
            for _ in 0..soa_steps {
                ssm.step(prime_activation.max(0.0));
            }
            let ssm_boost = ssm.step(0.0);
            // Blend: temporal decay + SSM (decay dominates for better SOA modulation)
            let priming_boost = (activation as f32 * 0.7 + ssm_boost * 0.3).max(0.0);

            // 4-AFC with priming-modulated probe quality.
            // Science: priming pre-activates the target's semantic neighborhood,
            // enhancing the internal recognition template (Neely 1977). When the
            // prime is related, the probe is less noisy (better recognition).
            // When unrelated, no pre-activation benefit — noisier probe.
            // FEP predictive coding amplifies the pre-activation benefit.
            let mut candidates = vec![(target_cluster, target_word_idx)];
            while candidates.len() < 4 {
                xor_shift(&mut rng);
                let cc = (rng % n_clusters as u64) as usize;
                xor_shift(&mut rng);
                let cw = (rng % words_per_cluster as u64) as usize;
                if !candidates.contains(&(cc, cw)) {
                    candidates.push((cc, cw));
                }
            }

            // Prime pre-activation modulates probe quality:
            // - Related prime: reduces effective noise (better recognition)
            // - Unrelated prime: no benefit (full noise)
            let base_noise = 0.55 + config.encoding_noise as f32 * 0.20;
            let fep_scale = if config.enable_fep { 1.0 } else { 0.4 };
            let prime_benefit = if is_related {
                priming_boost * fep_scale * 0.70
            } else {
                0.0
            };
            let effective_noise = (base_noise - prime_benefit).clamp(0.15, 0.80);
            let probe_noise =
                ContinuousHV::random(dim, seed.wrapping_add(pair_idx as u64 * 777 + 42));
            let probe = ContinuousHV::weighted_bundle(
                &[target, &probe_noise],
                &[1.0 - effective_noise, effective_noise],
            );

            let mut best_sim = f32::NEG_INFINITY;
            let mut best_is_target = false;
            for (ci, wi) in &candidates {
                let candidate = &clusters[*ci][*wi];
                let sim = probe.similarity(candidate);
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                let total = sim + noise;
                if total > best_sim {
                    best_sim = total;
                    best_is_target = *ci == target_cluster && *wi == target_word_idx;
                }
            }

            // Lapse model: attention lapses can flip the correctness outcome.
            let lapse_trial_idx = trial_idx * n_pairs + pair_idx;
            let best_is_target =
                config.check_correct(best_is_target, "semantic_priming", lapse_trial_idx);

            if best_is_target {
                if is_related {
                    if is_short_soa {
                        related_correct_short += 1;
                    } else {
                        related_correct_long += 1;
                    }
                } else if is_short_soa {
                    unrelated_correct_short += 1;
                } else {
                    unrelated_correct_long += 1;
                }
            }

            if is_related {
                if is_short_soa {
                    related_total_short += 1;
                } else {
                    related_total_long += 1;
                }
            } else if is_short_soa {
                unrelated_total_short += 1;
            } else {
                unrelated_total_long += 1;
            }

            // RT: related pairs faster, short SOA faster
            let base_rt = if is_related { 7.0 } else { 9.0 };
            let soa_cost = if is_short_soa { 0.0 } else { 1.0 };
            let tp_speedup = config.time_pressure * 1.5;
            let pair_rt = (base_rt + soa_cost - tp_speedup).max(1.0);
            rt_sum += pair_rt;

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: *global_trial_idx,
                    condition: if is_related {
                        "related".to_string()
                    } else {
                        "unrelated".to_string()
                    },
                    correct: best_is_target,
                    rt_ticks: pair_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                *global_trial_idx += 1;
            }
        }

        let related_total = related_total_short + related_total_long;
        let unrelated_total = unrelated_total_short + unrelated_total_long;
        let related_correct = related_correct_short + related_correct_long;
        let unrelated_correct = unrelated_correct_short + unrelated_correct_long;

        let related_acc = if related_total > 0 {
            related_correct as f64 / related_total as f64
        } else {
            0.0
        };
        let unrelated_acc = if unrelated_total > 0 {
            unrelated_correct as f64 / unrelated_total as f64
        } else {
            0.0
        };

        // SOA modulation: difference in priming effect between short and long SOA
        let related_acc_short = if related_total_short > 0 {
            related_correct_short as f64 / related_total_short as f64
        } else {
            0.0
        };
        let unrelated_acc_short = if unrelated_total_short > 0 {
            unrelated_correct_short as f64 / unrelated_total_short as f64
        } else {
            0.0
        };
        let related_acc_long = if related_total_long > 0 {
            related_correct_long as f64 / related_total_long as f64
        } else {
            0.0
        };
        let unrelated_acc_long = if unrelated_total_long > 0 {
            unrelated_correct_long as f64 / unrelated_total_long as f64
        } else {
            0.0
        };

        let priming_short = (related_acc_short - unrelated_acc_short).max(0.0);
        let priming_long = (related_acc_long - unrelated_acc_long).max(0.0);

        TrialResult {
            priming_effect: (related_acc - unrelated_acc).max(0.0),
            related_accuracy: related_acc,
            unrelated_accuracy: unrelated_acc,
            soa_modulation: (priming_long - priming_short).abs(),
            rt_ticks: rt_sum / n_pairs as f64,
        }
    }
}

impl PsychBenchmark for SemanticPrimingBenchmark {
    fn name(&self) -> &str {
        "Language::SemanticPriming"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Semantic Priming",
            citation: "Neely (1977)",
            year: 1977,
            doi: Some("10.1037/0278-7393.3.3.346"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut priming_effects = Vec::new();
        let mut related_accs = Vec::new();
        let mut unrelated_accs = Vec::new();
        let mut soa_mods = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial, &mut trace, &mut global_trial_idx);
            priming_effects.push(r.priming_effect);
            related_accs.push(r.related_accuracy);
            unrelated_accs.push(r.unrelated_accuracy);
            soa_mods.push(r.soa_modulation);
            rts.push(r.rt_ticks);
        }

        result.insert(
            "priming_effect",
            MetricValue::from_samples(&priming_effects),
        );
        result.insert("related_accuracy", MetricValue::from_samples(&related_accs));
        result.insert(
            "unrelated_accuracy",
            MetricValue::from_samples(&unrelated_accs),
        );
        result.insert("soa_modulation", MetricValue::from_samples(&soa_mods));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 4; // related×short, related×long, unrelated×short, unrelated×long
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_priming_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SemanticPrimingBenchmark.run(&config);
        assert!(result.metrics.contains_key("priming_effect"));
        assert!(result.metrics.contains_key("related_accuracy"));
        assert!(result.metrics.contains_key("unrelated_accuracy"));
        assert!(result.metrics.contains_key("soa_modulation"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_semantic_priming_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SemanticPrimingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_semantic_priming_effect_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SemanticPrimingBenchmark.run(&config);
        let effect = result.metrics["priming_effect"].mean;
        assert!(
            effect >= 0.0,
            "priming effect ({:.3}) should be >= 0",
            effect
        );
    }

    #[test]
    fn test_semantic_priming_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SemanticPrimingBenchmark.run(&config);
        let acc = result.metrics["related_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "related accuracy ({:.3}) out of bounds",
            acc
        );
    }

    #[test]
    fn test_semantic_priming_effect_positive() {
        // With degraded probe, priming should produce a measurable effect
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 30,
            ..Default::default()
        };
        let result = SemanticPrimingBenchmark.run(&config);
        let effect = result.metrics["priming_effect"].mean;
        let related = result.metrics["related_accuracy"].mean;
        let unrelated = result.metrics["unrelated_accuracy"].mean;
        eprintln!(
            "priming_effect={:.3}, related={:.3}, unrelated={:.3}",
            effect, related, unrelated
        );
        assert!(
            effect > 0.02,
            "priming effect ({:.3}) should be > 0.02 with degraded probe",
            effect
        );
    }

    #[test]
    fn test_semantic_priming_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = SemanticPrimingBenchmark.run(&base);
        let r_press = SemanticPrimingBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
