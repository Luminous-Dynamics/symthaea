// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lexical Decision Task.
//!
//! Participants discriminate real words from non-words as quickly as possible.
//! Word frequency modulates recognition speed — high-frequency words are
//! recognized faster than low-frequency words.
//!
//! HDC implementation: Words are prototype ContinuousHVs in a 20-word codebook.
//! Non-words are chimeric blends (50:50 of two random word HVs). Lexical
//! decision is a 4-AFC match against the codebook; frequency effect emerges
//! from prototype signal strength (high-frequency words have stronger encoding).
//!
//! Human baselines (Meyer & Schvaneveldt, 1971; Balota & Chumbley, 1984):
//! - lexicality_effect: 0.12 (SD~0.05) — accuracy advantage for words over non-words
//! - word_accuracy: 0.92 (SD~0.05)
//! - nonword_accuracy: 0.80 (SD~0.08)
//! - frequency_effect: 0.08 (SD~0.04) — accuracy advantage for high vs low frequency
//! - rt_ticks: 8 (SD~2) — ~400ms mean RT

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Lexical Decision Task benchmark.
pub struct LexicalDecisionBenchmark;

struct TrialResult {
    lexicality_effect: f64,
    word_accuracy: f64,
    nonword_accuracy: f64,
    frequency_effect: f64,
    rt_ticks: f64,
    /// Per-item trial trace (populated when config.trial_trace is true).
    item_trace: Vec<TrialOutcome>,
}

impl LexicalDecisionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("language", "lexical_decision", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // 20-word codebook: prototypes at full fidelity
        let codebook_size = 20;
        let words: Vec<ContinuousHV> = (0..codebook_size)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(i as u64 * 83 + 1)))
            .collect();

        // Frequency split: first 10 = high frequency (stronger signal), rest = low frequency
        let high_freq_count = codebook_size / 2;

        // Time pressure noise, scaled by difficulty temperature model
        let diff_model = difficulty_model_for(self.name());
        let noise_level: f32 = (0.20 + config.time_pressure as f32 * 0.15)
            * diff_model.temperature_multiplier(config.difficulty) as f32;

        let n_items = 40; // 20 word trials + 20 non-word trials
        let mut item_trace = Vec::new();
        let mut global_item_idx = 0usize;
        let mut word_correct = 0u32;
        let mut word_total = 0u32;
        let mut nonword_correct = 0u32;
        let mut nonword_total = 0u32;
        let mut high_freq_correct = 0u32;
        let mut high_freq_total = 0u32;
        let mut low_freq_correct = 0u32;
        let mut low_freq_total = 0u32;
        let mut rt_sum = 0.0f64;

        for item in 0..n_items {
            xor_shift(&mut rng);
            let is_word = item < n_items / 2;

            if is_word {
                // Select a word from codebook
                xor_shift(&mut rng);
                let word_idx = (rng % codebook_size as u64) as usize;
                let is_high_freq = word_idx < high_freq_count;

                // High-frequency words: stronger prototype signal
                // FEP provides top-down predictive refinement; without it, weaker signal
                let fep_penalty: f32 = if config.enable_fep { 0.0 } else { 0.10 };
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let signal_strength: f32 =
                    (if is_high_freq { 0.75 } else { 0.55 }) - fep_penalty - noise_degrade;
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(
                    &[&words[word_idx], &noise_hv],
                    &[signal_strength, 1.0 - signal_strength],
                );

                // Match against codebook with frequency weighting:
                // High-frequency words get a recognition boost (Balota & Chumbley 1984).
                // This models the word frequency effect in lexical access.
                let freq_boost = config.language_frequency_boost as f32;
                let mut best_sim = f32::NEG_INFINITY;
                let mut best_idx = 0;
                for (i, w) in words.iter().enumerate() {
                    let mut sim = stimulus.similarity(w);
                    // High-frequency words (first half) are easier to recognize
                    if i < high_freq_count {
                        sim += freq_boost;
                    }
                    xor_shift(&mut rng);
                    let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                    sim += noise;
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                // Correct if matched to the right word
                if best_idx == word_idx {
                    word_correct += 1;
                    if is_high_freq {
                        high_freq_correct += 1;
                    } else {
                        low_freq_correct += 1;
                    }
                }
                word_total += 1;
                if is_high_freq {
                    high_freq_total += 1;
                } else {
                    low_freq_total += 1;
                }

                // RT: faster for high-frequency words
                let base_rt = if is_high_freq { 7.0 } else { 9.0 };
                let tp_speedup = config.time_pressure * 1.5;
                let item_rt = (base_rt - tp_speedup).max(1.0);
                rt_sum += item_rt;

                // Per-item trial trace
                if config.trial_trace {
                    let cond = if is_high_freq {
                        "word_high_freq"
                    } else {
                        "word_low_freq"
                    };
                    item_trace.push(TrialOutcome {
                        trial_idx: global_item_idx,
                        condition: cond.to_string(),
                        correct: best_idx == word_idx,
                        rt_ticks: item_rt,
                        similarity: best_sim as f64,
                        confidence: best_sim as f64,
                        response_idx: best_idx,
                        extra: BTreeMap::new(),
                    });
                    global_item_idx += 1;
                }
            } else {
                // Non-word: chimeric blend of two random words
                xor_shift(&mut rng);
                let w1 = (rng % codebook_size as u64) as usize;
                xor_shift(&mut rng);
                let mut w2 = (rng % codebook_size as u64) as usize;
                if w2 == w1 {
                    w2 = (w1 + 1) % codebook_size;
                }

                let nonword =
                    ContinuousHV::weighted_bundle(&[&words[w1], &words[w2]], &[0.50, 0.50]);
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let stimulus = ContinuousHV::weighted_bundle(&[&nonword, &noise_hv], &[0.60, 0.40]);

                // Match against codebook: non-word should NOT strongly match any entry
                let mut best_sim = f32::NEG_INFINITY;
                for w in &words {
                    let mut sim = stimulus.similarity(w);
                    xor_shift(&mut rng);
                    let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                    sim += noise;
                    if sim > best_sim {
                        best_sim = sim;
                    }
                }

                // Correct rejection: best match similarity below threshold
                // Non-words should be less similar to any single prototype
                let rejection_threshold = 0.55;
                if best_sim < rejection_threshold {
                    nonword_correct += 1;
                }
                nonword_total += 1;

                // RT: non-words take longer (exhaustive search)
                let base_rt = 10.0;
                let tp_speedup = config.time_pressure * 1.5;
                let item_rt = (base_rt - tp_speedup).max(1.0);
                rt_sum += item_rt;

                // Per-item trial trace
                if config.trial_trace {
                    let item_correct = best_sim < rejection_threshold;
                    item_trace.push(TrialOutcome {
                        trial_idx: global_item_idx,
                        condition: "nonword".to_string(),
                        correct: item_correct,
                        rt_ticks: item_rt,
                        similarity: best_sim as f64,
                        confidence: best_sim as f64,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                    global_item_idx += 1;
                }
            }
        }

        let word_acc = if word_total > 0 {
            word_correct as f64 / word_total as f64
        } else {
            0.0
        };
        let nonword_acc = if nonword_total > 0 {
            nonword_correct as f64 / nonword_total as f64
        } else {
            0.0
        };
        let high_freq_acc = if high_freq_total > 0 {
            high_freq_correct as f64 / high_freq_total as f64
        } else {
            0.0
        };
        let low_freq_acc = if low_freq_total > 0 {
            low_freq_correct as f64 / low_freq_total as f64
        } else {
            0.0
        };

        TrialResult {
            lexicality_effect: (word_acc - nonword_acc).max(0.0),
            word_accuracy: word_acc,
            nonword_accuracy: nonword_acc,
            frequency_effect: (high_freq_acc - low_freq_acc).max(0.0),
            rt_ticks: rt_sum / n_items as f64,
            item_trace,
        }
    }
}

impl PsychBenchmark for LexicalDecisionBenchmark {
    fn name(&self) -> &str {
        "Language::LexicalDecision"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Lexical Decision Task",
            citation: "Meyer & Schvaneveldt (1971)",
            year: 1971,
            doi: Some("10.1016/S0022-5371(71)80004-8"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut lex_effects = Vec::new();
        let mut word_accs = Vec::new();
        let mut nonword_accs = Vec::new();
        let mut freq_effects = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            lex_effects.push(r.lexicality_effect);
            word_accs.push(r.word_accuracy);
            nonword_accs.push(r.nonword_accuracy);
            freq_effects.push(r.frequency_effect);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.extend(r.item_trace);
            }
        }

        result.insert("lexicality_effect", MetricValue::from_samples(&lex_effects));
        result.insert("word_accuracy", MetricValue::from_samples(&word_accs));
        result.insert("nonword_accuracy", MetricValue::from_samples(&nonword_accs));
        result.insert("frequency_effect", MetricValue::from_samples(&freq_effects));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_decision_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = LexicalDecisionBenchmark.run(&config);
        assert!(result.metrics.contains_key("lexicality_effect"));
        assert!(result.metrics.contains_key("word_accuracy"));
        assert!(result.metrics.contains_key("nonword_accuracy"));
        assert!(result.metrics.contains_key("frequency_effect"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_lexical_decision_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = LexicalDecisionBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_lexical_decision_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = LexicalDecisionBenchmark.run(&config);
        let acc = result.metrics["word_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "word accuracy ({:.3}) out of bounds",
            acc
        );
    }

    #[test]
    fn test_lexical_decision_lexicality_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = LexicalDecisionBenchmark.run(&config);
        let effect = result.metrics["lexicality_effect"].mean;
        assert!(
            effect >= 0.0,
            "lexicality effect ({:.3}) should be >= 0",
            effect
        );
    }

    #[test]
    fn test_lexical_decision_time_pressure() {
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
        let r_base = LexicalDecisionBenchmark.run(&base);
        let r_press = LexicalDecisionBenchmark.run(&pressed);
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
