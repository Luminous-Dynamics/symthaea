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
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use symthaea_core::hdc::ContinuousHV;

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
        let dim = config.dimension;
        let seed = config.trial_seed("executive", "wcst", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Create deterministic atomic HVs for features
        let color_hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i)))
            .collect();
        let shape_hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i)))
            .collect();
        let number_hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(300 + i)))
            .collect();

        // Role HVs for binding
        let role_color = ContinuousHV::random(dim, seed.wrapping_add(400));
        let role_shape = ContinuousHV::random(dim, seed.wrapping_add(401));
        let role_number = ContinuousHV::random(dim, seed.wrapping_add(402));

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

        let encode_card = |card: &Card| -> ContinuousHV {
            let c = role_color.bind(&color_hvs[card.color as usize]);
            let s = role_shape.bind(&shape_hvs[card.shape as usize]);
            let n = role_number.bind(&number_hvs[card.number as usize - 1]);
            ContinuousHV::bundle(&[&c, &s, &n])
        };

        let _target_hvs: Vec<ContinuousHV> = targets.iter().map(encode_card).collect();

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
        let mut rule_confidence = [1.0f64, 0.0, 0.0]; // Start believing color
        let lr_correct = 0.2; // Moderate reinforcement (lower cap = less needed)
        let lr_error = 0.8; // Stronger error penalty for faster set-shifting

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

            // Pick action by highest-confidence rule; HDC similarity as tiebreaker
            let candidates = [match_by_color, match_by_shape, match_by_number];
            let _response_hv = encode_card(&response_card);

            // Softmax over rule confidences for stochastic selection.
            // Time pressure reduces the softmax gain (more random rule selection).
            let softmax_gain = 1.5 - config.time_pressure * 0.8;
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
            let max_conf = rule_confidence.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let second_conf = rule_confidence.iter().cloned()
                .filter(|&c| (c - max_conf).abs() > 1e-10)
                .fold(f64::NEG_INFINITY, f64::max);
            let conf_margin = if second_conf.is_finite() { max_conf - second_conf } else { max_conf };
            let ticks = 4.0 + (1.0 - conf_margin.min(1.0).max(0.0)) * 6.0;
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
                rule_confidence[chosen_rule_idx] *= 0.4; // Rapid decay
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
}

impl PsychBenchmark for WisconsinCardSortingBenchmark {
    fn name(&self) -> &str {
        "Executive::WCST"
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

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            categories.push(r.categories_completed as f64);
            perseverative.push(r.perseverative_errors as f64);
            non_perseverative.push(r.non_perseverative_errors as f64);
            total_errs.push(r.total_errors as f64);
            trials_first.push(r.trials_to_first as f64);
            all_rts.extend_from_slice(&r.rt_ticks);
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
