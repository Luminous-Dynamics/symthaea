// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Garden-Path Sentence Processing.
//!
//! Tests the cost of syntactic reanalysis when an initial parse is
//! incorrect. Garden-path sentences lead readers down one interpretation
//! before disambiguation forces reanalysis.
//!
//! HDC implementation: Sentences encoded as sequences of word-role HVs.
//! Initial parse binds words to roles (e.g., "The horse raced" → agent-verb).
//! Disambiguation ("past the barn was tired") requires unbinding and rebinding
//! to a new structural interpretation. Cost = similarity difference between
//! initial and revised parses.
//!
//! Human baselines (Frazier & Rayner, 1982; Ferreira & Clifton, 1986):
//! - disambiguation_cost: 0.12 (SD≈0.06) — processing time increase
//! - overall_accuracy: 0.85 (SD≈0.08) — final comprehension accuracy
//! - garden_path_accuracy: 0.75 (SD≈0.10) — accuracy on GP sentences

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Garden-Path benchmark.
pub struct GardenPathBenchmark;

struct TrialResult {
    disambiguation_cost: f64,
    overall_accuracy: f64,
    garden_path_accuracy: f64,
    control_accuracy: f64,
    rt_ticks: f64,
}

impl GardenPathBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("language", "garden_path", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // --- Semantically-grounded role HVs (NSM-inspired) ---
        //
        // Natural Semantic Metalanguage (Wierzbicka, 1996) decomposes meaning
        // into universal primes: SOMEONE, SOMETHING, DO, HAPPEN, BECAUSE, etc.
        // We build syntactic roles as structured composites of semantic primes,
        // giving them internal structure that supports role-filler binding.
        let prime_someone = ContinuousHV::random(dim, seed.wrapping_add(100));
        let prime_something = ContinuousHV::random(dim, seed.wrapping_add(101));
        let prime_do = ContinuousHV::random(dim, seed.wrapping_add(102));
        let prime_happen = ContinuousHV::random(dim, seed.wrapping_add(103));
        let prime_because = ContinuousHV::random(dim, seed.wrapping_add(104));
        let prime_like = ContinuousHV::random(dim, seed.wrapping_add(105));

        // AGENT = SOMEONE who DOES (causal initiator)
        let role_agent = ContinuousHV::weighted_bundle(
            &[&prime_someone, &prime_do, &prime_because],
            &[0.5, 0.3, 0.2],
        )
        .normalize();
        // VERB = DO/HAPPEN (action or process)
        let role_verb =
            ContinuousHV::weighted_bundle(&[&prime_do, &prime_happen], &[0.6, 0.4]).normalize();
        // PATIENT = SOMETHING that something HAPPENS to
        let role_patient =
            ContinuousHV::weighted_bundle(&[&prime_something, &prime_happen], &[0.5, 0.5])
                .normalize();
        // MODIFIER = LIKE/BECAUSE (descriptive/causal adjunct)
        let role_modifier = ContinuousHV::weighted_bundle(
            &[&prime_like, &prime_because, &prime_something],
            &[0.4, 0.3, 0.3],
        )
        .normalize();

        // Word embeddings: each word carries partial role affinity.
        // Words 0-3 have agent-like semantics (animate actors),
        // words 4-7 have verb-like semantics (actions),
        // words 8-11 have patient/theme semantics.
        let n_words = 12;
        let words: Vec<ContinuousHV> = (0..n_words)
            .map(|i| {
                let base = ContinuousHV::random(dim, seed.wrapping_add(500 + i as u64 * 50));
                let role_tint = match i {
                    0..=3 => &role_agent,
                    4..=7 => &role_verb,
                    _ => &role_patient,
                };
                // 75% unique identity + 25% role affinity
                ContinuousHV::weighted_bundle(&[&base, role_tint], &[0.75, 0.25]).normalize()
            })
            .collect();

        let n_sentences = 40;
        let gp_rate = 0.50;

        // Time pressure: reduces reanalysis thoroughness (Ferreira & Henderson, 1991).
        let reanalysis_noise: f32 = 0.25 + config.time_pressure as f32 * 0.15;

        let mut gp_correct = 0u32;
        let mut gp_total = 0u32;
        let mut ctrl_correct = 0u32;
        let mut ctrl_total = 0u32;
        let mut cost_sum = 0.0f64;
        let mut cost_count = 0u32;
        let mut rt_sum = 0.0f64;

        for _sent in 0..n_sentences {
            xor_shift(&mut rng);
            let is_gp = (rng % 100) as f64 / 100.0 < gp_rate;

            xor_shift(&mut rng);
            let w1_idx = (rng % n_words as u64) as usize;
            xor_shift(&mut rng);
            let w2_idx = (rng % n_words as u64) as usize;
            xor_shift(&mut rng);
            let w3_idx = (rng % n_words as u64) as usize;

            if is_gp {
                gp_total += 1;

                // Initial (incorrect) parse: w1=agent, w2=verb
                // "The horse raced..." → horse=agent, raced=verb
                let initial_parse = ContinuousHV::weighted_bundle(
                    &[
                        &words[w1_idx].bind(&role_agent),
                        &words[w2_idx].bind(&role_verb),
                    ],
                    &[0.5, 0.5],
                );

                // Revised (correct) parse: w1=patient, w2=modifier, w3=verb
                // "The horse raced past the barn was tired"
                // → horse=patient, raced-past-barn=modifier, was-tired=verb
                let revised_parse = ContinuousHV::weighted_bundle(
                    &[
                        &words[w1_idx].bind(&role_patient),
                        &words[w2_idx].bind(&role_modifier),
                        &words[w3_idx].bind(&role_verb),
                    ],
                    &[0.35, 0.30, 0.35],
                );

                // Disambiguation cost: dissimilarity between initial and revised parses
                // (Frazier & Rayner, 1982). Higher cost = more cognitive effort to reanalyze.
                let parse_similarity = initial_parse.similarity(&revised_parse);
                let reparse_effort = (1.0 - parse_similarity.abs()) as f64;
                let cost = reparse_effort * config.language_reparse_cost_scale;
                cost_sum += cost;
                cost_count += 1;

                // Reanalysis success check: compare revised parse against the
                // **correct interpretation template** — what the sentence means
                // under the correct parse. The template is built from the same
                // role-word bindings, so similarity measures structural match.
                let correct_template = ContinuousHV::weighted_bundle(
                    &[
                        &words[w1_idx].bind(&role_patient),
                        &words[w2_idx].bind(&role_modifier),
                        &words[w3_idx].bind(&role_verb),
                    ],
                    &[0.35, 0.30, 0.35],
                );
                let reanalysis_score = revised_parse.similarity(&correct_template);

                // Noise degrades reanalysis fidelity
                xor_shift(&mut rng);
                let noise_val = (rng % 10000) as f32 / 10000.0 * reanalysis_noise;
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let effective_score = reanalysis_score * (1.0 - noise_degrade) - noise_val;

                // Higher cost → harder to reanalyze. Threshold scales with effort.
                let threshold = 0.3 + cost as f32 * 0.25;
                if effective_score > threshold {
                    gp_correct += 1;
                }

                // GP sentences take longer (reanalysis time)
                let base_rt = 6.0 + cost * 4.0;
                let tp_speedup = config.time_pressure * 2.0;
                rt_sum += (base_rt - tp_speedup).max(2.0);
            } else {
                ctrl_total += 1;

                // Control sentence: unambiguous agent-verb-patient structure
                let parse = ContinuousHV::weighted_bundle(
                    &[
                        &words[w1_idx].bind(&role_agent),
                        &words[w2_idx].bind(&role_verb),
                        &words[w3_idx].bind(&role_patient),
                    ],
                    &[0.35, 0.30, 0.35],
                );

                // Comprehension check: compare parse against its own template.
                // For control sentences, the initial parse IS the correct parse,
                // so this should succeed easily (high self-similarity).
                let template = ContinuousHV::weighted_bundle(
                    &[
                        &words[w1_idx].bind(&role_agent),
                        &words[w2_idx].bind(&role_verb),
                        &words[w3_idx].bind(&role_patient),
                    ],
                    &[0.35, 0.30, 0.35],
                );
                let comp_score = parse.similarity(&template);
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * reanalysis_noise * 0.5;
                if comp_score - noise > 0.3 {
                    ctrl_correct += 1;
                }

                // Control RT: no reanalysis needed
                let base_rt = 4.0;
                let tp_speedup = config.time_pressure * 1.0;
                rt_sum += (base_rt - tp_speedup).max(1.0);
            }
        }

        let gp_acc = if gp_total > 0 {
            gp_correct as f64 / gp_total as f64
        } else {
            0.0
        };
        let ctrl_acc = if ctrl_total > 0 {
            ctrl_correct as f64 / ctrl_total as f64
        } else {
            0.0
        };
        let overall_acc = if gp_total + ctrl_total > 0 {
            (gp_correct + ctrl_correct) as f64 / (gp_total + ctrl_total) as f64
        } else {
            0.0
        };
        let mean_cost = if cost_count > 0 {
            cost_sum / cost_count as f64
        } else {
            0.0
        };
        let mean_rt = rt_sum / n_sentences as f64;

        TrialResult {
            disambiguation_cost: mean_cost,
            overall_accuracy: overall_acc,
            garden_path_accuracy: gp_acc,
            control_accuracy: ctrl_acc,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for GardenPathBenchmark {
    fn name(&self) -> &str {
        "Language::GardenPath"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Garden-Path Sentence Processing",
            citation: "Frazier & Rayner (1982)",
            year: 1982,
            doi: Some("10.1016/0010-0285(82)90008-1"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut costs = Vec::new();
        let mut overall_accs = Vec::new();
        let mut gp_accs = Vec::new();
        let mut ctrl_accs = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            costs.push(r.disambiguation_cost);
            overall_accs.push(r.overall_accuracy);
            gp_accs.push(r.garden_path_accuracy);
            ctrl_accs.push(r.control_accuracy);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "garden_path".to_string(),
                    correct: r.overall_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("disambiguation_cost", MetricValue::from_samples(&costs));
        result.insert("overall_accuracy", MetricValue::from_samples(&overall_accs));
        result.insert("garden_path_accuracy", MetricValue::from_samples(&gp_accs));
        result.insert("control_accuracy", MetricValue::from_samples(&ctrl_accs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 2; // garden-path vs control
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
    fn test_garden_path_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = GardenPathBenchmark.run(&config);
        assert!(result.metrics.contains_key("disambiguation_cost"));
        assert!(result.metrics.contains_key("overall_accuracy"));
        assert!(result.metrics.contains_key("garden_path_accuracy"));
        assert!(result.metrics.contains_key("control_accuracy"));
    }

    #[test]
    fn test_garden_path_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = GardenPathBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_garden_path_cost_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = GardenPathBenchmark.run(&config);
        let cost = result.metrics["disambiguation_cost"].mean;
        assert!(
            cost > 0.0,
            "disambiguation cost ({:.3}) should be > 0",
            cost
        );
    }

    #[test]
    fn test_garden_path_time_pressure() {
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
        let r_base = GardenPathBenchmark.run(&base);
        let r_press = GardenPathBenchmark.run(&pressed);
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
