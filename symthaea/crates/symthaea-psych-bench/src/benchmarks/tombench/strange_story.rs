//! Strange story task.
//!
//! Tests understanding of non-literal language: irony, white lies,
//! misunderstanding, double bluff. The system must maintain dual
//! encodings (literal vs contextual meaning).

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Strange story benchmark for non-literal language comprehension.
pub struct StrangeStoryBenchmark;

struct StrangeStoryScenario {
    context: Vec<&'static str>,
    literal_meaning: &'static str,
    intended_meaning: &'static str,
    story_type: &'static str,
}

impl StrangeStoryBenchmark {
    fn scenarios() -> Vec<StrangeStoryScenario> {
        vec![
            StrangeStoryScenario {
                context: vec![
                    "It is pouring rain outside",
                    "Tom looks out the window at the heavy rain",
                    "Tom says what lovely weather we are having",
                ],
                literal_meaning: "Tom thinks the weather is good",
                intended_meaning: "Tom is being sarcastic about the bad weather",
                story_type: "irony",
            },
            StrangeStoryScenario {
                context: vec![
                    "Mary baked a cake for the office party",
                    "The cake turned out dry and tasteless",
                    "John takes a bite and says this is delicious Mary",
                ],
                literal_meaning: "John thinks the cake tastes good",
                intended_meaning: "John is telling a white lie to be polite",
                story_type: "white_lie",
            },
            StrangeStoryScenario {
                context: vec![
                    "Peter tells his mom he has no homework tonight",
                    "Peter actually has a math assignment due tomorrow",
                    "Peter wants to play video games instead",
                ],
                literal_meaning: "Peter has no homework",
                intended_meaning: "Peter is lying to avoid doing homework",
                story_type: "deception",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, &'static str) {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
        });

        // Present context
        for sentence in &scenario.context {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            wm.perceive(hv);
            wm.tick();
        }

        let contents = wm.contents();

        // Encode both interpretations
        let literal_hv = adapter.encode(&Scenario::new(scenario.literal_meaning), dim);
        let intended_hv = adapter.encode(&Scenario::new(scenario.intended_meaning), dim);

        // WM should be more similar to the intended meaning (non-literal)
        let literal_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&literal_hv))
            .fold(0.0f32, f32::max);
        let intended_sim: f32 = contents
            .iter()
            .map(|item| item.similarity(&intended_hv))
            .fold(0.0f32, f32::max);

        // Correct if intended meaning scores higher
        let correct = if intended_sim > literal_sim { 1.0 } else { 0.0 };
        (correct, scenario.story_type)
    }
}

impl PsychBenchmark for StrangeStoryBenchmark {
    fn name(&self) -> &str {
        "ToMBench::StrangeStory"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        let mut type_accs: std::collections::HashMap<&str, Vec<f64>> =
            std::collections::HashMap::new();

        for trial in 0..config.trials_per_condition {
            let (acc, stype) = self.run_trial(config, trial);
            accuracies.push(acc);
            type_accs.entry(stype).or_default().push(acc);
        }

        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        for (stype, accs) in &type_accs {
            result.insert(
                format!("{}::accuracy", stype),
                MetricValue::from_samples(accs),
            );
        }

        result.conditions = type_accs.len();
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strange_story_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = StrangeStoryBenchmark.run(&config);
        assert!(result.metrics.contains_key("overall_accuracy"));
    }
}
