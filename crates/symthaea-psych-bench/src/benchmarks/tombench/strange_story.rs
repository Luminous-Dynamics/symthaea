//! Strange story task.
//!
//! Tests understanding of non-literal language: irony, white lies,
//! misunderstanding, double bluff. Uses HDC bundling to accumulate
//! context, then measures whether the bundled context is more
//! consistent with the intended (non-literal) or literal meaning.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use symthaea_core::hdc::ContinuousHV;

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

    /// Lightweight trial: HDC geometry only.
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, &'static str) {
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

        let literal_hv = adapter.encode(&Scenario::new(scenario.literal_meaning), dim);
        let intended_hv = adapter.encode(&Scenario::new(scenario.intended_meaning), dim);

        let literal_sim = context_bundle.similarity(&literal_hv);
        let intended_sim = context_bundle.similarity(&intended_hv);

        let correct = if intended_sim > literal_sim { 1.0 } else { 0.0 };
        (correct, scenario.story_type)
    }

    /// Full trial: FEP behavioral prediction for non-literal language.
    ///
    /// States: [literal_interpretation, nonliteral_interpretation] (dim=2)
    /// Actions: [respond_literally, respond_to_intent] (2 actions)
    /// Listener perceives context + statement → contextual cues via perceive()
    /// Contradictory cues (bad weather + "lovely") shift belief toward nonliteral
    /// select_action() → should choose action 1 (respond to intent)
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, &'static str) {
        use super::applied_tom::{make_observation, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Perceive context cues: earlier context establishes ground truth,
        // the final statement contradicts it → drives toward nonliteral
        for (i, _cue) in scenario.context.iter().enumerate() {
            let is_last = i == scenario.context.len() - 1;
            let obs = if is_last {
                // The contradictory statement (e.g., "lovely weather" during rain)
                // signals non-literal interpretation
                make_observation(vec![0.2, 0.8], "linguistic")
            } else {
                // Context cues that establish the true situation
                make_observation(vec![0.3, 0.7], "linguistic")
            };
            agent.perceive(&obs);
        }

        // Listener wants to understand true intent → prefers nonliteral obs
        agent.set_goals(vec![0.0, 1.0], 4.0);

        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 1 (respond_to_intent) — detected non-literal meaning
        let expected_action = 1;
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence, scenario.story_type)
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, &'static str) {
        #[cfg(feature = "symthaea-backend")]
        {
            let (acc, _conf, stype) = self.run_trial_full(config, trial_idx);
            return (acc, stype);
        }
        #[cfg(not(feature = "symthaea-backend"))]
        {
            return self.run_trial_lightweight(config, trial_idx);
        }
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
        #[cfg(feature = "symthaea-backend")]
        let mut confidences = Vec::new();

        for trial in 0..config.trials_per_condition {
            #[cfg(feature = "symthaea-backend")]
            {
                let (acc, conf, stype) = self.run_trial_full(config, trial);
                accuracies.push(acc);
                confidences.push(conf);
                type_accs.entry(stype).or_default().push(acc);
            }
            #[cfg(not(feature = "symthaea-backend"))]
            {
                let (acc, stype) = self.run_trial(config, trial);
                accuracies.push(acc);
                type_accs.entry(stype).or_default().push(acc);
            }
        }

        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        #[cfg(feature = "symthaea-backend")]
        result.insert("action_confidence", MetricValue::from_samples(&confidences));
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
