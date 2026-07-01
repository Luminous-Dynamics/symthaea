// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Coherence in Text Generation.
//!
//! Measures the ability to maintain semantic coherence across a generated
//! token sequence — testing topic drift, coherence recovery after disruption,
//! and the relationship between semantic complexity and coherence decay.
//!
//! HDC implementation: A "topic" HV is established. Each "generated token" is
//! bound with a positional HV and bundled into a running context. Coherence is
//! measured as cosine similarity between context and topic. Periodically, a
//! distractor topic is injected, and recovery is measured.
//!
//! Human baselines (Graesser et al., 2004; McNamara et al., 2014):
//! - coherence_mean: 0.72 (SD≈0.08) — average topic coherence
//! - coherence_decay: 0.15 (SD≈0.05) — coherence drop over sequence length
//! - recovery_speed: 0.80 (SD≈0.10) — recovery after topic disruption
//! - complexity_penalty: 0.20 (SD≈0.06) — coherence reduction for complex topics

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Semantic Coherence benchmark.
pub struct SemanticCoherenceBenchmark;

struct TrialResult {
    coherence_mean: f64,
    coherence_decay: f64,
    recovery_speed: f64,
    complexity_penalty: f64,
    rt_ticks: f64,
}

impl SemanticCoherenceBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("language", "semantic_coherence", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Generate topic HVs (simple and complex)
        let n_topics = 4;
        let topics: Vec<ContinuousHV> = (0..n_topics)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i as u64 * 50)).normalize())
            .collect();

        // Sequence length for topic tracking (matches human reading studies)
        let n_positions = 30;

        // Vocabulary: on-topic words (close to topic) and off-topic words
        let n_on_topic = 20;
        let n_off_topic = 10;

        // Time pressure: increases topic drift (McDonald et al., 2006)
        let drift_noise: f32 = 0.15 + config.time_pressure as f32 * 0.10;

        let mut coherences = Vec::new();
        let mut decays = Vec::new();
        let mut recoveries = Vec::new();
        let mut complexities = Vec::new();
        let mut rt_sum = 0.0f64;

        // Run sequences for each topic
        for topic_idx in 0..n_topics {
            let topic = &topics[topic_idx];
            // Complex topics are created by bundling multiple sub-topics
            let is_complex = topic_idx >= n_topics / 2;

            let effective_topic = if is_complex {
                // Complex: blend two topics → harder to maintain coherence
                let other_idx = (topic_idx + 1) % n_topics;
                ContinuousHV::weighted_bundle(&[topic, &topics[other_idx]], &[0.6, 0.4]).normalize()
            } else {
                topic.clone()
            };

            // Generate on-topic word HVs (blend of topic + random)
            let on_topic_words: Vec<ContinuousHV> = (0..n_on_topic)
                .map(|i| {
                    let rand_hv = ContinuousHV::random(
                        dim,
                        seed.wrapping_add(1000 + topic_idx as u64 * 200 + i as u64 * 10),
                    );
                    ContinuousHV::weighted_bundle(&[&effective_topic, &rand_hv], &[0.6, 0.4])
                        .normalize()
                })
                .collect();

            // Off-topic words (random, unrelated to topic)
            let off_topic_words: Vec<ContinuousHV> = (0..n_off_topic)
                .map(|i| {
                    ContinuousHV::random(
                        dim,
                        seed.wrapping_add(3000 + topic_idx as u64 * 200 + i as u64 * 10),
                    )
                    .normalize()
                })
                .collect();

            let sequence_len = n_positions;
            let distractor_pos = sequence_len / 2; // Inject disruption halfway
            let mut context = ContinuousHV::zero(dim);
            let mut token_coherences = Vec::new();

            for pos in 0..sequence_len {
                xor_shift(&mut rng);

                // Choose token: mostly on-topic, with noise
                let use_off_topic = if pos == distractor_pos {
                    true // Force disruption
                } else {
                    let noise = (rng % 10000) as f32 / 10000.0;
                    noise < drift_noise
                };

                xor_shift(&mut rng);
                let word = if use_off_topic {
                    let idx = (rng as usize) % n_off_topic;
                    &off_topic_words[idx]
                } else {
                    let idx = (rng as usize) % n_on_topic;
                    &on_topic_words[idx]
                };

                // Topic coherence via semantic EMA (Graesser et al., 2004).
                //
                // We track two signals:
                // 1. **Semantic context** (unbound words) — for measuring topic
                //    coherence. Position-binding would rotate word HVs orthogonal
                //    to the topic, destroying the semantic signal.
                // 2. **Positional ordering** is implicitly captured by the EMA
                //    weighting (recent tokens dominate), matching human recency
                //    effects in discourse comprehension.
                let alpha = config.language_coherence_alpha as f32;
                context = ContinuousHV::weighted_bundle(&[&context, word], &[1.0 - alpha, alpha])
                    .normalize();

                // Measure coherence with topic via EMA tracking
                let raw_coh = effective_topic.similarity(&context).clamp(-1.0, 1.0) as f64;
                token_coherences.push(raw_coh);
            }

            // Metrics from this sequence
            let mean_coh: f64 = token_coherences.iter().sum::<f64>() / sequence_len as f64;
            coherences.push(mean_coh);

            // Decay: coherence of last quarter minus first quarter
            let q1_len = sequence_len / 4;
            let first_q: f64 = token_coherences[..q1_len].iter().sum::<f64>() / q1_len as f64;
            let last_q: f64 = token_coherences[sequence_len - q1_len..]
                .iter()
                .sum::<f64>()
                / q1_len as f64;
            let decay = (first_q - last_q).max(0.0);
            decays.push(decay);

            // Recovery: measure how much coherence rebounds after the off-topic
            // injection. Compare coherence at distractor vs 3 tokens later.
            // A disruption is detected when coherence drops below the pre-disruption
            // mean (tokens before the distractor position).
            let post_recovery = if distractor_pos + 3 < sequence_len && distractor_pos > 0 {
                let pre_mean: f64 =
                    token_coherences[..distractor_pos].iter().sum::<f64>() / distractor_pos as f64;
                let at_disrupt = token_coherences[distractor_pos];
                let after_disrupt = token_coherences[distractor_pos + 3];
                if at_disrupt < pre_mean * 0.9 {
                    // Disruption detected: measure fractional recovery toward pre-mean
                    let drop = pre_mean - at_disrupt;
                    let rebound = after_disrupt - at_disrupt;
                    (rebound / drop.max(0.01)).clamp(0.0, 1.0)
                } else {
                    1.0 // No significant disruption occurred
                }
            } else {
                0.5
            };
            recoveries.push(post_recovery);

            // Complexity penalty: simple vs complex
            if is_complex {
                let simple_coh = coherences.first().copied().unwrap_or(0.5);
                let penalty = (simple_coh - mean_coh).max(0.0);
                complexities.push(penalty);
            }

            // RT: complex topics take longer
            let base_rt = 5.0 + if is_complex { 2.0 } else { 0.0 };
            let tp_speedup = config.time_pressure * 1.5;
            rt_sum += (base_rt - tp_speedup).max(2.0);
        }

        let mean_coherence = coherences.iter().sum::<f64>() / coherences.len().max(1) as f64;
        let mean_decay = decays.iter().sum::<f64>() / decays.len().max(1) as f64;
        let mean_recovery = recoveries.iter().sum::<f64>() / recoveries.len().max(1) as f64;
        let mean_complexity = if complexities.is_empty() {
            0.0
        } else {
            complexities.iter().sum::<f64>() / complexities.len() as f64
        };
        let mean_rt = rt_sum / n_topics as f64;

        TrialResult {
            coherence_mean: mean_coherence,
            coherence_decay: mean_decay,
            recovery_speed: mean_recovery,
            complexity_penalty: mean_complexity,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for SemanticCoherenceBenchmark {
    fn name(&self) -> &str {
        "Language::SemanticCoherence"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Semantic Coherence in Text Comprehension",
            citation: "Graesser et al. (2004)",
            year: 2004,
            doi: Some("10.1207/s15516709cog2805_2"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut coherences = Vec::new();
        let mut decays = Vec::new();
        let mut recoveries = Vec::new();
        let mut complexities = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            coherences.push(r.coherence_mean);
            decays.push(r.coherence_decay);
            recoveries.push(r.recovery_speed);
            complexities.push(r.complexity_penalty);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "coherence".to_string(),
                    correct: r.coherence_mean > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("coherence_mean", MetricValue::from_samples(&coherences));
        result.insert("coherence_decay", MetricValue::from_samples(&decays));
        result.insert("recovery_speed", MetricValue::from_samples(&recoveries));
        result.insert(
            "complexity_penalty",
            MetricValue::from_samples(&complexities),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // simple vs complex topics
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
    fn test_semantic_coherence_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SemanticCoherenceBenchmark.run(&config);
        assert!(result.metrics.contains_key("coherence_mean"));
        assert!(result.metrics.contains_key("coherence_decay"));
        assert!(result.metrics.contains_key("recovery_speed"));
        assert!(result.metrics.contains_key("complexity_penalty"));
    }

    #[test]
    fn test_semantic_coherence_finite() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SemanticCoherenceBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_semantic_coherence_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SemanticCoherenceBenchmark.run(&config);
        let coh = result.metrics["coherence_mean"].mean;
        assert!(coh > 0.0, "coherence ({:.3}) should be > 0", coh);
    }

    #[test]
    fn test_semantic_coherence_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 0.8,
            ..base.clone()
        };

        let base_result = SemanticCoherenceBenchmark.run(&base);
        let pressed_result = SemanticCoherenceBenchmark.run(&pressed);

        let base_rt = base_result.metrics["rt_ticks"].mean;
        let pressed_rt = pressed_result.metrics["rt_ticks"].mean;
        assert!(
            pressed_rt <= base_rt,
            "Time pressure should reduce RT: base={base_rt:.2}, pressed={pressed_rt:.2}"
        );
    }
}
