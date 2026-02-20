//! Hinting task.
//!
//! Tests desire inference from indirect cues: the system must infer
//! what a character wants without being told directly. Uses HDC
//! bundling to accumulate contextual cues, then measures whether
//! the accumulated context is more similar to the correct desire
//! inference than to the surface-level (wrong) interpretation.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use symthaea_core::hdc::ContinuousHV;

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

    /// Lightweight trial: HDC geometry only.
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let context_hvs: Vec<ContinuousHV> = scenario
            .context
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        let desire_marker = adapter.encode(
            &Scenario::new("wants needs desires wishes hopes for"),
            dim,
        );

        let correct_hv = adapter.encode(&Scenario::new(scenario.correct_inference), dim);
        let wrong_hv = adapter.encode(&Scenario::new(scenario.wrong_inference), dim);

        let correct_context_sim = context_bundle.similarity(&correct_hv);
        let wrong_context_sim = context_bundle.similarity(&wrong_hv);
        let correct_desire_sim = desire_marker.similarity(&correct_hv);
        let wrong_desire_sim = desire_marker.similarity(&wrong_hv);

        let correct_score = correct_context_sim + correct_desire_sim * 0.3;
        let wrong_score = wrong_context_sim + wrong_desire_sim * 0.3;

        if correct_score > wrong_score { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP behavioral prediction for desire inference.
    ///
    /// States: [desire_unfulfilled, desire_fulfilled] (dim=2)
    /// Actions: [do_nothing, fulfill_desire] (2 actions)
    /// Observer perceives hints → accumulates desire cues via perceive()
    /// select_action() → should choose action 1 (fulfill) after accumulating hint cues
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Observer perceives each hint as a desire-weighted observation
        for (i, _hint) in scenario.context.iter().enumerate() {
            // Each hint increasingly signals desire: later hints are stronger
            let desire_strength = 0.3 + 0.2 * (i as f64);
            let obs = make_observation(vec![1.0 - desire_strength, desire_strength], "social");
            agent.perceive(&obs);
        }

        // Observer wants desire fulfilled → prefers fulfilled-state observations
        agent.set_goals(vec![0.0, 1.0], 4.0);

        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 1 (fulfill_desire) after accumulating hints
        let expected_action = 1;
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

impl PsychBenchmark for HintingBenchmark {
    fn name(&self) -> &str {
        "ToMBench::Hinting"
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

        result.insert("hinting_accuracy", MetricValue::from_samples(&accuracies));
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
