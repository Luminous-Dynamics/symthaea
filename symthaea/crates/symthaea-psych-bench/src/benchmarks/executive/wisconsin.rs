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
enum Color { Red, Blue, Green, Yellow }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape { Triangle, Circle, Square, Star }

/// A stimulus card with 3 features.
#[derive(Clone, Copy)]
struct Card {
    color: Color,
    shape: Shape,
    number: u8, // 1-4
}

/// Sorting rule dimension.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule { Color, Shape, Number }

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
            Card { color: Color::Red, shape: Shape::Triangle, number: 1 },
            Card { color: Color::Blue, shape: Shape::Circle, number: 2 },
            Card { color: Color::Green, shape: Shape::Square, number: 3 },
            Card { color: Color::Yellow, shape: Shape::Star, number: 4 },
        ];

        let encode_card = |card: &Card| -> ContinuousHV {
            let c = role_color.bind(&color_hvs[card.color as usize]);
            let s = role_shape.bind(&shape_hvs[card.shape as usize]);
            let n = role_number.bind(&number_hvs[card.number as usize - 1]);
            ContinuousHV::bundle(&[&c, &s, &n])
        };

        let target_hvs: Vec<ContinuousHV> = targets.iter().map(|t| encode_card(t)).collect();

        // State tracking
        let mut current_rule = Rule::Color;
        let mut consecutive_correct = 0u32;
        let mut categories_completed = 0u32;
        let mut perseverative_errors = 0u32;
        let mut non_perseverative_errors = 0u32;
        let mut total_errors = 0u32;
        let mut trials_to_first: Option<u32> = None;
        let mut prev_rule: Option<Rule> = None;

        // Agent's rule belief: weighted average of role HVs
        let mut belief = role_color.clone();
        let learning_rate = 0.4f32;

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
            let response_card = Card { color: rc, shape: rs, number: rn };

            // Agent sorts: unbind belief from response card to get feature,
            // then find most similar target
            let response_hv = encode_card(&response_card);
            let feature_hv = belief.bind(&response_hv);

            let chosen_target = target_hvs
                .iter()
                .enumerate()
                .map(|(i, t)| (i, feature_hv.similarity(t)))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Determine correct target by current rule
            let correct_target = match current_rule {
                Rule::Color => targets.iter().position(|t| t.color == response_card.color),
                Rule::Shape => targets.iter().position(|t| t.shape == response_card.shape),
                Rule::Number => targets.iter().position(|t| t.number == response_card.number),
            }.unwrap_or(0);

            let is_correct = chosen_target == correct_target;

            if is_correct {
                consecutive_correct += 1;
                // Reinforce current belief
                let role_hv = match current_rule {
                    Rule::Color => &role_color,
                    Rule::Shape => &role_shape,
                    Rule::Number => &role_number,
                };
                belief = ContinuousHV::weighted_bundle(
                    &[&belief, role_hv],
                    &[1.0 - learning_rate, learning_rate],
                ).normalize();
            } else {
                total_errors += 1;
                // Check if perseverative (still using the old rule)
                if let Some(pr) = prev_rule {
                    let would_be_correct_old = match pr {
                        Rule::Color => targets.iter().position(|t| t.color == response_card.color),
                        Rule::Shape => targets.iter().position(|t| t.shape == response_card.shape),
                        Rule::Number => targets.iter().position(|t| t.number == response_card.number),
                    }.unwrap_or(usize::MAX);
                    if chosen_target == would_be_correct_old {
                        perseverative_errors += 1;
                    } else {
                        non_perseverative_errors += 1;
                    }
                } else {
                    non_perseverative_errors += 1;
                }
                consecutive_correct = 0;

                // Update belief toward correct rule (error-driven learning)
                let correct_role = match current_rule {
                    Rule::Color => &role_color,
                    Rule::Shape => &role_shape,
                    Rule::Number => &role_number,
                };
                belief = ContinuousHV::weighted_bundle(
                    &[&belief, correct_role],
                    &[1.0 - learning_rate * 0.5, learning_rate * 0.5],
                ).normalize();
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

                // Partially reset belief to force relearning
                let new_role = match current_rule {
                    Rule::Color => &role_color,
                    Rule::Shape => &role_shape,
                    Rule::Number => &role_number,
                };
                belief = ContinuousHV::weighted_bundle(
                    &[&belief, new_role],
                    &[0.6, 0.4],
                ).normalize();
            }
        }

        TrialResult {
            categories_completed,
            perseverative_errors,
            non_perseverative_errors,
            total_errors,
            trials_to_first: trials_to_first.unwrap_or(max_trials as u32),
        }
    }
}

struct TrialResult {
    categories_completed: u32,
    perseverative_errors: u32,
    non_perseverative_errors: u32,
    total_errors: u32,
    trials_to_first: u32,
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

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            categories.push(r.categories_completed as f64);
            perseverative.push(r.perseverative_errors as f64);
            non_perseverative.push(r.non_perseverative_errors as f64);
            total_errs.push(r.total_errors as f64);
            trials_first.push(r.trials_to_first as f64);
        }

        result.insert("categories_completed", MetricValue::from_samples(&categories));
        result.insert("perseverative_errors", MetricValue::from_samples(&perseverative));
        result.insert("non_perseverative_errors", MetricValue::from_samples(&non_perseverative));
        result.insert("total_errors", MetricValue::from_samples(&total_errs));
        result.insert("trials_to_first_category", MetricValue::from_samples(&trials_first));

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
