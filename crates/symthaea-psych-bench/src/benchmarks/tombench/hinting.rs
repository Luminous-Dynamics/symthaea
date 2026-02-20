//! Hinting task.
//!
//! Tests desire inference from indirect cues: the system must infer
//! what a character wants without being told directly, by detecting
//! implicit desires in their hints via WM similarity.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Hinting task benchmark.
pub struct HintingBenchmark;

struct HintingScenario {
    context: Vec<&'static str>,
    correct_inference: &'static str,
    wrong_inference: &'static str,
}

impl HintingBenchmark {
    fn scenarios() -> Vec<HintingScenario> {
        vec![
            HintingScenario {
                context: vec![
                    "George walks past the shop and sees a coat in the window",
                    "George says that coat looks really warm",
                    "George shivers and pulls his thin jacket tighter",
                ],
                correct_inference: "George wants someone to buy him the coat",
                wrong_inference: "George is commenting on coat design",
            },
            HintingScenario {
                context: vec![
                    "Maria is tired after working all day",
                    "Maria says the dishes are piling up in the kitchen",
                    "Maria yawns and sits down on the sofa",
                ],
                correct_inference: "Maria wants her partner to do the dishes",
                wrong_inference: "Maria is describing the state of the kitchen",
            },
            HintingScenario {
                context: vec![
                    "Paul and his friend are at a restaurant",
                    "Paul forgot his wallet at home",
                    "Paul says this meal is more expensive than I expected",
                ],
                correct_inference: "Paul wants his friend to help pay for the meal",
                wrong_inference: "Paul is commenting on restaurant prices",
            },
            HintingScenario {
                context: vec![
                    "Emma is at a boring party with her friend",
                    "Emma looks at her watch and says it is getting late",
                    "Emma yawns and mentions the long drive home",
                ],
                correct_inference: "Emma wants to leave the party",
                wrong_inference: "Emma is telling the time",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        for sentence in &scenario.context {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            wm.perceive(hv);
            wm.tick();
        }

        let contents = wm.contents();
        let correct_hv = adapter.encode(&Scenario::new(scenario.correct_inference), dim);
        let wrong_hv = adapter.encode(&Scenario::new(scenario.wrong_inference), dim);

        let correct_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&correct_hv))
            .fold(0.0f32, f32::max);
        let wrong_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&wrong_hv))
            .fold(0.0f32, f32::max);

        if correct_sim > wrong_sim { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for HintingBenchmark {
    fn name(&self) -> &str {
        "ToMBench::Hinting"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        for trial in 0..config.trials_per_condition {
            accuracies.push(self.run_trial(config, trial));
        }

        result.insert("hinting_accuracy", MetricValue::from_samples(&accuracies));

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
    fn test_hinting_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 4,
            dimension: 256,
            ..Default::default()
        };
        let result = HintingBenchmark.run(&config);
        assert!(result.metrics.contains_key("hinting_accuracy"));
    }
}
