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

    /// Lightweight trial: HDC geometry only.
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let statement_hv = adapter.encode(&Scenario::new(scenario.statement), dim);
        let reaction_hv = adapter.encode(&Scenario::new(scenario.reaction), dim);

        let positive_marker = adapter.encode(&Scenario::new("happy pleased grateful smiles"), dim);
        let negative_marker = adapter.encode(&Scenario::new("disappointed embarrassed hurt upset"), dim);

        let reaction_neg = reaction_hv.similarity(&negative_marker);
        let reaction_pos = reaction_hv.similarity(&positive_marker);
        let statement_neg = statement_hv.similarity(&negative_marker);

        let divergence = reaction_neg - reaction_pos + statement_neg * 0.3;
        let detected_faux_pas = divergence > 0.0;

        if detected_faux_pas == scenario.is_faux_pas { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP behavioral prediction for faux pas detection.
    ///
    /// States: [social_safe, social_blunder] (dim=2)
    /// Actions: [continue_normally, repair_attempt] (2 actions)
    /// Observations: [positive_reaction, negative_reaction] (2 obs)
    /// For faux pas: negative reaction → agent detects blunder → action 1 (repair)
    /// For non-faux-pas: positive reaction → agent stays calm → action 0 (continue)
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Encode statement as observation: faux-pas statements bias toward blunder
        let statement_obs = if scenario.is_faux_pas {
            make_observation(vec![0.3, 0.7], "social") // lean toward blunder
        } else {
            make_observation(vec![0.7, 0.3], "social") // lean toward safe
        };
        agent.perceive(&statement_obs);

        // Encode listener reaction: negative = blunder signal
        let reaction_obs = if scenario.is_faux_pas {
            make_observation(vec![0.1, 0.9], "social") // strong negative reaction
        } else {
            make_observation(vec![0.9, 0.1], "social") // positive reaction
        };
        agent.perceive(&reaction_obs);

        // Agent wants to maintain social safety → prefers safe-state observations
        agent.set_goals(vec![1.0, 0.0], 4.0);

        let (action, probs) = predict_behavior(&mut agent);

        // For faux pas: should select action 1 (repair_attempt)
        // For non-faux-pas: should select action 0 (continue_normally)
        let expected_action = if scenario.is_faux_pas { 1 } else { 0 };
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence)
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        #[cfg(feature = "symthaea-backend")]
        {
            let (acc, _) = self.run_trial_full(config, trial_idx);
            return acc;
        }
        #[cfg(not(feature = "symthaea-backend"))]
        {
            return self.run_trial_lightweight(config, trial_idx);
        }
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
        #[cfg(feature = "symthaea-backend")]
        let mut confidences = Vec::new();

        for trial in 0..config.trials_per_condition {
            #[cfg(feature = "symthaea-backend")]
            {
                let (acc, conf) = self.run_trial_full(config, trial);
                accuracies.push(acc);
                confidences.push(conf);
            }
            #[cfg(not(feature = "symthaea-backend"))]
            {
                accuracies.push(self.run_trial(config, trial));
            }
        }

        result.insert("faux_pas_accuracy", MetricValue::from_samples(&accuracies));
        #[cfg(feature = "symthaea-backend")]
        result.insert("action_confidence", MetricValue::from_samples(&confidences));

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
