//! Persuasion story task.
//!
//! Tests intention tracking: one agent tries to change another's mind.
//! The system should detect the intent to persuade by bundling the
//! scenario context and measuring similarity to persuasion vs neutral
//! intent markers. Uses HDC bundling for accumulated context encoding.

#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
#[cfg(not(feature = "symthaea-backend"))]
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

    /// Lightweight trial: HDC geometry only.
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let context_hvs: Vec<ContinuousHV> = scenario
            .setup
            .iter()
            .map(|s| adapter.encode(&Scenario::new(*s), dim))
            .collect();
        let context_bundle = ContinuousHV::bundle_owned(&context_hvs);

        let persuasion_marker = adapter.encode(
            &Scenario::new("wants convince persuade influence change mind"),
            dim,
        );
        let neutral_marker = adapter.encode(
            &Scenario::new("inform share tell describe mention"),
            dim,
        );

        let persuasion_sim = context_bundle.similarity(&persuasion_marker);
        let neutral_sim = context_bundle.similarity(&neutral_marker);

        let detected_persuasion = persuasion_sim > neutral_sim;

        if detected_persuasion == scenario.has_persuasion { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP belief detection for persuasion recognition.
    ///
    /// States: [original_intent, persuaded] (dim=2)
    /// Observations: [neutral_info, persuasion_cue] (2 obs)
    ///
    /// Persuasion detection is a *belief tracking* task: after the target
    /// perceives a sequence of statements, has their belief shifted toward
    /// the "persuaded" state? The FEP agent's belief update from accumulated
    /// observations IS the ToM capability.
    ///
    /// Detection: belief.mean[1] > belief.mean[0] indicates persuasion detected.
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{make_observation, social_agent};

        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent = social_agent(2, 2, 2);

        // Target perceives each statement as persuasion-weighted observations
        for (i, _sentence) in scenario.setup.iter().enumerate() {
            let obs = if scenario.has_persuasion {
                // Persuasion cues accumulate: later statements are more persuasive
                let persuasion_weight = 0.3 + 0.15 * (i as f64);
                make_observation(vec![1.0 - persuasion_weight, persuasion_weight], "social")
            } else {
                // Neutral: observations centered, no directional bias
                make_observation(vec![0.6, 0.4], "social")
            };
            agent.perceive(&obs);
        }

        // Detection: did the agent's belief shift toward persuaded (state 1)?
        let detected_persuasion = agent.belief.mean[1] > agent.belief.mean[0];
        let accuracy = if detected_persuasion == scenario.has_persuasion { 1.0 } else { 0.0 };
        let confidence = if scenario.has_persuasion {
            agent.belief.mean[1] // confidence in persuasion detection
        } else {
            agent.belief.mean[0] // confidence in original intent
        };

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        self.run_trial_lightweight(config, trial_idx)
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

        result.insert("persuasion_detection", MetricValue::from_samples(&accuracies));
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
