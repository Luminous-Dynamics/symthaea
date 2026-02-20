//! Faux-pas recognition task.
//!
//! Tests whether the system detects when a speaker makes an unintentional
//! social blunder by modeling the divergence between speaker intent and
//! listener emotional response via WM similarity to emotional markers.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Faux-pas recognition benchmark.
pub struct FauxPasBenchmark;

struct FauxPasScenario {
    context: Vec<&'static str>,
    is_faux_pas: bool,
}

impl FauxPasBenchmark {
    fn scenarios() -> Vec<FauxPasScenario> {
        vec![
            FauxPasScenario {
                context: vec![
                    "Lisa just got a new haircut that she loves",
                    "Her friend Amy says I liked your old hair better",
                    "Lisa looks disappointed and touches her hair",
                ],
                is_faux_pas: true,
            },
            FauxPasScenario {
                context: vec![
                    "James cooked dinner for his friends",
                    "His friend Mark says this tastes just like my mom used to make",
                    "James smiles and serves more food",
                ],
                is_faux_pas: false,
            },
            FauxPasScenario {
                context: vec![
                    "Sarah spent weeks painting a portrait for the art show",
                    "Tom walks in and says did a child paint that",
                    "Sarah feels embarrassed in front of the other artists",
                ],
                is_faux_pas: true,
            },
            FauxPasScenario {
                context: vec![
                    "Mike shows his friends the new house he bought",
                    "David says the garden looks beautiful and well maintained",
                    "Mike thanks David and shows more of the house",
                ],
                is_faux_pas: false,
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

        // Present scenario sentences
        for sentence in &scenario.context {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Detect faux pas: look for emotional divergence in WM
        // A faux pas creates a mismatch between the speaker's intent (neutral/positive)
        // and the listener's emotional reaction (negative/embarrassed)
        let contents = wm.contents();

        // Encode positive and negative emotional markers
        let positive_marker = adapter.encode(&Scenario::new("happy pleased satisfied"), dim);
        let negative_marker = adapter.encode(&Scenario::new("disappointed embarrassed hurt"), dim);

        // Check if the last WM items (reaction) lean negative
        let last_items = if contents.len() >= 2 { &contents[contents.len() - 2..] } else { contents };
        let neg_sim: f32 = last_items
            .iter()
            .map(|item| item.similarity(&negative_marker))
            .sum::<f32>()
            / last_items.len().max(1) as f32;
        let pos_sim: f32 = last_items
            .iter()
            .map(|item| item.similarity(&positive_marker))
            .sum::<f32>()
            / last_items.len().max(1) as f32;

        // System detects faux pas if negative > positive for recent items
        let detected_faux_pas = neg_sim > pos_sim;

        if detected_faux_pas == scenario.is_faux_pas { 1.0 } else { 0.0 }
    }
}

impl PsychBenchmark for FauxPasBenchmark {
    fn name(&self) -> &str {
        "ToMBench::FauxPas"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        for trial in 0..config.trials_per_condition {
            accuracies.push(self.run_trial(config, trial));
        }

        result.insert("faux_pas_accuracy", MetricValue::from_samples(&accuracies));

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
    fn test_faux_pas_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 4,
            dimension: 256,
            ..Default::default()
        };
        let result = FauxPasBenchmark.run(&config);
        assert!(result.metrics.contains_key("faux_pas_accuracy"));
    }
}
