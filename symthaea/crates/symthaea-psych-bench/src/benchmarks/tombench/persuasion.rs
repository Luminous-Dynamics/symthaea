//! Persuasion story task.
//!
//! Tests intention tracking: one agent tries to change another's mind.
//! The system should detect the intent to persuade by bundling the
//! scenario context and measuring similarity to persuasion vs neutral
//! intent markers. Uses HDC bundling for accumulated context encoding.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use symthaea_core::hdc::ContinuousHV;

/// Persuasion story benchmark.
pub struct PersuasionBenchmark;

struct PersuasionScenario {
    setup: Vec<&'static str>,
    /// Whether persuasion intent is present.
    has_persuasion: bool,
}

impl PersuasionBenchmark {
    fn scenarios() -> Vec<PersuasionScenario> {
        vec![
            PersuasionScenario {
                setup: vec![
                    "Alice wants Bob to come to her party",
                    "Alice tells Bob that his favorite band will be playing",
                    "Alice says everyone from work will be there",
                    "Bob initially said he was too tired to go",
                ],
                has_persuasion: true,
            },
            PersuasionScenario {
                setup: vec![
                    "Carol tells Dave about the weather forecast",
                    "Carol mentions it will rain tomorrow",
                    "Dave thanks Carol for the information",
                    "They continue eating lunch together",
                ],
                has_persuasion: false,
            },
            PersuasionScenario {
                setup: vec![
                    "The manager wants the team to work overtime",
                    "The manager mentions the bonus for completing early",
                    "The manager says the client is very important",
                    "The team was planning to leave on time today",
                ],
                has_persuasion: true,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // Bundle all scenario sentences into accumulated context
        let context_hvs: Vec<ContinuousHV> = scenario
            .setup
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        // Intent markers for detection
        let persuasion_marker = adapter.encode(
            &Scenario::new("wants convince persuade influence change mind"),
            dim,
        );
        let neutral_marker = adapter.encode(
            &Scenario::new("inform share tell describe mention"),
            dim,
        );

        // Measure context affinity to each intent marker
        let persuasion_sim = context_bundle.similarity(&persuasion_marker);
        let neutral_sim = context_bundle.similarity(&neutral_marker);

        let detected_persuasion = persuasion_sim > neutral_sim;

        if detected_persuasion == scenario.has_persuasion { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for PersuasionBenchmark {
    fn name(&self) -> &str {
        "ToMBench::Persuasion"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        for trial in 0..config.trials_per_condition {
            accuracies.push(self.run_trial(config, trial));
        }

        result.insert("persuasion_detection", MetricValue::from_samples(&accuracies));

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
    fn test_persuasion_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = PersuasionBenchmark.run(&config);
        assert!(result.metrics.contains_key("persuasion_detection"));
    }
}
