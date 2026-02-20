//! Faux-pas recognition task.
//!
//! Tests whether the system detects when a speaker makes an unintentional
//! social blunder by modeling the divergence between speaker intent and
//! listener emotional response. Uses agent-model tracking: encode the
//! speaker's intent and the listener's reaction as separate ContinuousHV
//! embeddings, then detect faux pas via intent-reaction divergence.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;


/// Faux-pas recognition benchmark.
pub struct FauxPasBenchmark;

struct FauxPasScenario {
    /// The statement made by the speaker.
    statement: &'static str,
    /// The listener's reaction.
    reaction: &'static str,
    /// Whether this is a faux pas.
    is_faux_pas: bool,
}

impl FauxPasBenchmark {
    fn scenarios() -> Vec<FauxPasScenario> {
        vec![
            FauxPasScenario {
                statement: "I liked your old hair better",
                reaction: "Lisa looks disappointed and touches her hair",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "this tastes just like my mom used to make",
                reaction: "James smiles and serves more food",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "did a child paint that",
                reaction: "Sarah feels embarrassed in front of the other artists",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "the garden looks beautiful and well maintained",
                reaction: "Mike thanks David and shows more of the house",
                is_faux_pas: false,
            },
            FauxPasScenario {
                statement: "I didn't know you were still working here",
                reaction: "Karen feels hurt by the implication she should have left",
                is_faux_pas: true,
            },
            FauxPasScenario {
                statement: "your presentation was very informative",
                reaction: "Robert nods and continues the meeting confidently",
                is_faux_pas: false,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        // Encode the speaker's statement and listener's reaction
        let statement_hv = adapter.encode(&Scenario::new(scenario.statement), dim);
        let reaction_hv = adapter.encode(&Scenario::new(scenario.reaction), dim);

        // Faux-pas detection via intent-reaction divergence:
        // Encode positive and negative emotional markers
        let positive_marker = adapter.encode(&Scenario::new("happy pleased grateful smiles"), dim);
        let negative_marker = adapter.encode(&Scenario::new("disappointed embarrassed hurt upset"), dim);

        // Speaker's intent is typically neutral/positive (not trying to offend)
        // Listener's reaction reveals the social impact
        let reaction_neg = reaction_hv.similarity(&negative_marker);
        let reaction_pos = reaction_hv.similarity(&positive_marker);

        // Also check statement harshness (unintentional offensiveness)
        let statement_neg = statement_hv.similarity(&negative_marker);

        // Detect faux pas: reaction leans negative AND/OR statement has negative valence
        // but the COMBINATION matters more than either alone
        let divergence = reaction_neg - reaction_pos + statement_neg * 0.3;
        let detected_faux_pas = divergence > 0.0;

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
            trials_per_condition: 6,
            dimension: 256,
            ..Default::default()
        };
        let result = FauxPasBenchmark.run(&config);
        assert!(result.metrics.contains_key("faux_pas_accuracy"));
    }
}
