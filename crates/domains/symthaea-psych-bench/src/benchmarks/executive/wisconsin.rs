// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wisconsin Card Sorting Test (WCST).
//!
//! Tests executive function: rule learning, set-shifting, and perseveration
//! detection. The agent must sort cards by an unknown rule (color, shape, or
//! number) that changes after 10 consecutive correct sorts.
//!
//! Human baselines (Kohli & Kaur, 2006):
//! - categories_completed: 5.62
//! - perseverative_errors: 8.29
//! - trials_to_first_category: 12.17

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Card features.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    Red,
    Blue,
    Green,
    Yellow,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Triangle,
    Circle,
    Square,
    Star,
}

/// A stimulus card with 3 features.
#[derive(Clone, Copy)]
struct Card {
    color: Color,
    shape: Shape,
    number: u8, // 1-4
}

/// Sorting rule dimension.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Color,
    Shape,
    Number,
}

impl Rule {
    fn cycle(self) -> Self {
        match self {
            Rule::Color => Rule::Shape,
            Rule::Shape => Rule::Number,
            Rule::Number => Rule::Color,
        }
    }
}

/// Wisconsin Card Sorting Test benchmark.
pub struct WisconsinCardSortingBenchmark;

impl WisconsinCardSortingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("executive", "wcst", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;
        let mut card_trace = Vec::new();
        let mut global_card_idx = 0usize;

        // 4 target cards with unique feature combos
        let targets = [
            Card {
                color: Color::Red,
                shape: Shape::Triangle,
                number: 1,
            },
            Card {
                color: Color::Blue,
                shape: Shape::Circle,
                number: 2,
            },
            Card {
                color: Color::Green,
                shape: Shape::Square,
                number: 3,
            },
            Card {
                color: Color::Yellow,
                shape: Shape::Star,
                number: 4,
            },
        ];

        // State tracking
        let mut current_rule = Rule::Color;
        let mut consecutive_correct = 0u32;
        let mut categories_completed = 0u32;
        let mut perseverative_errors = 0u32;
        let mut non_perseverative_errors = 0u32;
        let mut total_errors = 0u32;
        let mut trials_to_first: Option<u32> = None;
        let mut prev_rule: Option<Rule> = None;
        let mut rt_ticks = Vec::new();

        // Explicit hypothesis testing: 3 rule confidences [color, shape, number]
        // Mild color prior: color is perceptually most salient and most subjects
        // try it first (Milner, 1963; Grant & Berg, 1948).
        let mut rule_confidence = [0.50f64, 0.25, 0.25];
        // Encoding noise and time pressure degrade learning rates (impaired feedback)
        let noise = config.effective_noise();
        let lr_scale = 1.0 - noise * 0.6;
        let lr_correct = 0.2 * lr_scale; // Moderate reinforcement (lower cap = less needed)
        let lr_error = 0.8 * lr_scale; // Stronger error penalty for faster set-shifting

        let all_colors = [Color::Red, Color::Blue, Color::Green, Color::Yellow];
        let all_shapes = [Shape::Triangle, Shape::Circle, Shape::Square, Shape::Star];
        let max_trials = 128;

        for trial in 0..max_trials {
            if categories_completed >= 6 {
                break;
            }

            // Generate a random response card
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let rc = all_colors[(rng % 4) as usize];
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let rs = all_shapes[(rng % 4) as usize];
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let rn = (rng % 4) as u8 + 1;
            let response_card = Card {
                color: rc,
                shape: rs,
                number: rn,
            };

            // For each hypothesis, compute which target the response matches
            let match_by_color = targets
                .iter()
                .position(|t| t.color == response_card.color)
                .unwrap_or(0);
            let match_by_shape = targets
                .iter()
                .position(|t| t.shape == response_card.shape)
                .unwrap_or(0);
            let match_by_number = targets
                .iter()
                .position(|t| t.number == response_card.number)
                .unwrap_or(0);

            // Pick action by highest-confidence rule
            let candidates = [match_by_color, match_by_shape, match_by_number];

            // Softmax over rule confidences for stochastic selection.
            // Gain 1.8: explicit hypothesis testing (Ashby & Maddox, 2005 COVIS)
            // exploits rule confidence more decisively than implicit learning,
            // producing faster category completion with fewer total errors.
            // Time pressure: -1.2/unit flattens toward chance (Heitz, 2014).
            // Noise: -0.8/unit degrades rule discrimination.
            let diff_model = difficulty_model_for("Executive::WCST");
            let softmax_gain = (1.8 - config.time_pressure * 1.2 - noise * 0.8).max(0.1)
                * diff_model.temperature_multiplier(config.difficulty);
            let max_conf = rule_confidence
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_conf: Vec<f64> = rule_confidence
                .iter()
                .map(|c| ((c - max_conf) * softmax_gain).exp())
                .collect();
            let exp_sum: f64 = exp_conf.iter().sum();

            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let r = (rng % 10000) as f64 / 10000.0;
            let mut cumsum = 0.0;
            let mut chosen_rule_idx = 0;
            for (i, e) in exp_conf.iter().enumerate() {
                cumsum += e / exp_sum;
                if r < cumsum {
                    chosen_rule_idx = i;
                    break;
                }
            }
            let chosen_target = candidates[chosen_rule_idx];

            // RT proxy: deliberation ticks based on confidence spread
            let max_conf = rule_confidence
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let second_conf = rule_confidence
                .iter()
                .cloned()
                .filter(|&c| (c - max_conf).abs() > 1e-10)
                .fold(f64::NEG_INFINITY, f64::max);
            let conf_margin = if second_conf.is_finite() {
                max_conf - second_conf
            } else {
                max_conf
            };
            let ticks = 4.0 + (1.0 - conf_margin.clamp(0.0, 1.0)) * 6.0;
            rt_ticks.push(ticks);

            // Determine correct target by current rule
            let correct_target = match current_rule {
                Rule::Color => match_by_color,
                Rule::Shape => match_by_shape,
                Rule::Number => match_by_number,
            };

            let is_correct = chosen_target == correct_target;

            if is_correct {
                consecutive_correct += 1;
                // Reinforce the chosen hypothesis
                rule_confidence[chosen_rule_idx] += lr_correct;
                // Mild decay on others
                for (i, c) in rule_confidence.iter_mut().enumerate() {
                    if i != chosen_rule_idx {
                        *c *= 0.95;
                    }
                }
            } else {
                total_errors += 1;
                // Check if perseverative (using the old rule after a switch)
                if let Some(pr) = prev_rule {
                    let old_rule_idx = match pr {
                        Rule::Color => 0,
                        Rule::Shape => 1,
                        Rule::Number => 2,
                    };
                    if chosen_rule_idx == old_rule_idx {
                        perseverative_errors += 1;
                    } else {
                        non_perseverative_errors += 1;
                    }
                } else {
                    non_perseverative_errors += 1;
                }
                consecutive_correct = 0;

                // Multiplicative decay + additive penalty for faster switching.
                // Halving the wrong hypothesis' confidence is faster than subtracting
                // a fixed amount when confidence is high.
                rule_confidence[chosen_rule_idx] *= 0.25; // Rapid decay
                rule_confidence[chosen_rule_idx] -= lr_error;
                for (i, c) in rule_confidence.iter_mut().enumerate() {
                    if i != chosen_rule_idx {
                        *c += lr_error * 0.5;
                    }
                }
            }

            // Clamp confidences to [0, 2.5] — lower cap = less momentum to overcome
            for c in &mut rule_confidence {
                *c = c.clamp(0.0, 2.5);
            }

            // Per-card trial trace
            if config.trial_trace {
                let current_rule_name = match current_rule {
                    Rule::Color => "color",
                    Rule::Shape => "shape",
                    Rule::Number => "number",
                };
                let is_perseverative = !is_correct
                    && prev_rule.is_some_and(|pr| {
                        let old_rule_idx = match pr {
                            Rule::Color => 0,
                            Rule::Shape => 1,
                            Rule::Number => 2,
                        };
                        chosen_rule_idx == old_rule_idx
                    });
                card_trace.push(TrialOutcome {
                    trial_idx: global_card_idx,
                    condition: format!("rule_{}", current_rule_name),
                    correct: is_correct,
                    rt_ticks: ticks,
                    similarity: conf_margin,
                    confidence: rule_confidence[chosen_rule_idx],
                    response_idx: chosen_target,
                    extra: {
                        let mut m = BTreeMap::new();
                        m.insert(
                            "perseverative".to_string(),
                            if is_perseverative { 1.0 } else { 0.0 },
                        );
                        m
                    },
                });
                global_card_idx += 1;
            }

            // Rule switch after 10 consecutive correct
            if consecutive_correct >= 10 {
                categories_completed += 1;
                if trials_to_first.is_none() {
                    trials_to_first = Some(trial as u32 + 1);
                }
                prev_rule = Some(current_rule);
                current_rule = current_rule.cycle();
                consecutive_correct = 0;
                // Strongly dampen all confidences after category completion.
                // Agent "expects" rule may change, reducing perseveration.
                for c in &mut rule_confidence {
                    *c *= 0.25;
                }
            }
        }

        TrialResult {
            categories_completed,
            perseverative_errors,
            non_perseverative_errors,
            total_errors,
            trials_to_first: trials_to_first.unwrap_or(max_trials as u32),
            rt_ticks,
            card_trace,
        }
    }
}

struct TrialResult {
    categories_completed: u32,
    perseverative_errors: u32,
    non_perseverative_errors: u32,
    total_errors: u32,
    trials_to_first: u32,
    rt_ticks: Vec<f64>,
    /// Per-card trial trace (populated when config.trial_trace is true).
    card_trace: Vec<TrialOutcome>,
}

impl PsychBenchmark for WisconsinCardSortingBenchmark {
    fn name(&self) -> &str {
        "Executive::WCST"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Wisconsin Card Sorting Test",
            citation: "Berg (1948)",
            year: 1948,
            doi: Some("10.1085/jgp.32.2.163"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut categories = Vec::new();
        let mut perseverative = Vec::new();
        let mut non_perseverative = Vec::new();
        let mut total_errs = Vec::new();
        let mut trials_first = Vec::new();
        let mut all_rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            categories.push(r.categories_completed as f64);
            perseverative.push(r.perseverative_errors as f64);
            non_perseverative.push(r.non_perseverative_errors as f64);
            total_errs.push(r.total_errors as f64);
            trials_first.push(r.trials_to_first as f64);
            all_rts.extend_from_slice(&r.rt_ticks);

            if config.trial_trace {
                trace.extend(r.card_trace);
            }
        }

        result.insert(
            "categories_completed",
            MetricValue::from_samples(&categories),
        );
        result.insert(
            "perseverative_errors",
            MetricValue::from_samples(&perseverative),
        );
        result.insert(
            "non_perseverative_errors",
            MetricValue::from_samples(&non_perseverative),
        );
        result.insert("total_errors", MetricValue::from_samples(&total_errs));
        result.insert(
            "trials_to_first_category",
            MetricValue::from_samples(&trials_first),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&all_rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wcst_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = WisconsinCardSortingBenchmark.run(&config);
        assert!(result.metrics.contains_key("categories_completed"));
        assert!(result.metrics.contains_key("perseverative_errors"));
        assert!(result.metrics.contains_key("trials_to_first_category"));
        let cats = result.metrics["categories_completed"].mean;
        assert!(cats.is_finite());
    }

    #[test]
    fn test_wcst_produces_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = WisconsinCardSortingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }
}
