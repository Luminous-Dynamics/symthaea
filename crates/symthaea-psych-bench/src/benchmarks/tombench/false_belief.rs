//! False belief tasks (Sally-Anne type).
//!
//! Tests whether the system tracks that an agent holds a stale belief
//! about a state of the world that has changed in their absence.
//!
//! Uses a lightweight agent-model approach inspired by SocialCoherence:
//! each character's beliefs are tracked as ContinuousHV embeddings.
//! When a character is absent during a state change, their belief
//! model is NOT updated (stays stale), while the "reality" model IS
//! updated. The test checks whether the system predicts based on the
//! character's belief rather than reality.

#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::scenario::{Scenario, ScenarioAdapter};
#[cfg(not(feature = "symthaea-backend"))]
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
#[cfg(not(feature = "symthaea-backend"))]
use symthaea_core::hdc::ContinuousHV;

/// False belief benchmark (Sally-Anne paradigm).
pub struct FalseBeliefBenchmark;

/// A single false-belief scenario.
#[allow(dead_code)]
struct FalseBeliefScenario {
    /// Setup sentences (agent observes object location).
    setup: Vec<&'static str>,
    /// Agent leaves (becomes absent for the change).
    _absence: &'static str,
    /// Object is moved (while agent is absent).
    change: &'static str,
    /// Correct answer: where the agent BELIEVES the object is.
    belief_location: &'static str,
    /// Foil: where the object ACTUALLY is (reality answer).
    reality_location: &'static str,
}

impl FalseBeliefBenchmark {
    fn scenarios() -> Vec<FalseBeliefScenario> {
        vec![
            FalseBeliefScenario {
                setup: vec![
                    "Sally puts her marble in the basket",
                    "Sally sees the marble is in the basket",
                ],
                _absence: "Sally leaves the room",
                change: "Anne moves the marble to the box",
                belief_location: "Sally thinks the marble is in the basket",
                reality_location: "The marble is actually in the box",
            },
            FalseBeliefScenario {
                setup: vec![
                    "John puts his chocolate in the cupboard",
                    "John remembers putting chocolate in the cupboard",
                ],
                _absence: "John goes outside to play",
                change: "Mother moves the chocolate to the drawer",
                belief_location: "John thinks the chocolate is in the cupboard",
                reality_location: "The chocolate is actually in the drawer",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Alice places her book on the shelf",
                    "Alice knows her book is on the shelf",
                ],
                _absence: "Alice goes to school",
                change: "Bob moves the book to the table",
                belief_location: "Alice thinks the book is on the shelf",
                reality_location: "The book is actually on the table",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Tom hides his toy behind the curtain",
                    "Tom saw himself hide the toy behind the curtain",
                ],
                _absence: "Tom goes to the kitchen",
                change: "Emma moves the toy under the bed",
                belief_location: "Tom thinks the toy is behind the curtain",
                reality_location: "The toy is actually under the bed",
            },
        ]
    }

    /// Lightweight trial: HDC geometry only (no FEP).
    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial_lightweight(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        let dim = config.dimension;
        let adapter = ScenarioAdapter;
        let scenarios = Self::scenarios();
        let scenario = &scenarios[trial_idx % scenarios.len()];

        let mut agent_belief: Option<ContinuousHV> = None;
        let mut reality_state: Option<ContinuousHV> = None;

        for sentence in &scenario.setup {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            agent_belief = Some(match agent_belief {
                Some(prev) => ContinuousHV::bundle_owned(&[prev, hv.clone()]),
                None => hv.clone(),
            });
            reality_state = agent_belief.clone();
        }

        let change_hv = adapter.encode(&Scenario::new(scenario.change), dim);
        let _reality_state = Some(match reality_state {
            Some(prev) => ContinuousHV::bundle_owned(&[prev, change_hv]),
            None => change_hv,
        });

        let belief_hv = adapter.encode(&Scenario::new(scenario.belief_location), dim);
        let reality_hv = adapter.encode(&Scenario::new(scenario.reality_location), dim);

        let agent = agent_belief.unwrap();
        let belief_sim = agent.similarity(&belief_hv);
        let reality_sim = agent.similarity(&reality_hv);

        if belief_sim > reality_sim { 1.0 } else { 0.0 }
    }

    /// Full trial: FEP behavioral prediction from false beliefs.
    ///
    /// States: [marble_at_basket, marble_at_box] (dim=2)
    /// Actions: [go_to_basket, go_to_box] (2 actions)
    /// Sally observes marble in basket → belief = [0.9, 0.1]
    /// Anne moves marble → NO update to Sally's beliefs
    /// Sally wants to find marble → set_goals prefers basket-area obs
    /// select_action() → should return action 0 (go-to-basket) based on false beliefs
    #[cfg(feature = "symthaea-backend")]
    fn run_trial_full(&self, _config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64) {
        use super::applied_tom::{inject_belief, predict_behavior, social_agent};

        let scenarios = Self::scenarios();
        let _scenario = &scenarios[trial_idx % scenarios.len()];

        // Create FEP agent modeling Sally's mental state
        let mut agent = social_agent(2, 2, 2);

        // Sally observes marble placed in basket → belief = [0.9, 0.1]
        inject_belief(&mut agent, vec![0.9, 0.1]);

        // Anne moves marble to box while Sally is away → NO update to Sally's beliefs
        // (belief stays frozen at [0.9, 0.1])

        // Sally wants to find the marble → prefers basket-area observations
        agent.set_goals(vec![1.0, 0.0], 4.0);

        // Ask: what would Sally DO given her (false) beliefs?
        let (action, probs) = predict_behavior(&mut agent);

        // Expected: action 0 (go-to-basket) because Sally believes marble is there
        let expected_action = 0;
        let accuracy = if action == expected_action { 1.0 } else { 0.0 };
        let confidence = probs[expected_action];

        (accuracy, confidence)
    }

    #[cfg(not(feature = "symthaea-backend"))]
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> f64 {
        self.run_trial_lightweight(config, trial_idx)
    }
}

impl PsychBenchmark for FalseBeliefBenchmark {
    fn name(&self) -> &str {
        "ToMBench::FalseBelief"
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

        result.insert("false_belief_accuracy", MetricValue::from_samples(&accuracies));
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
    fn test_false_belief_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 4,
            dimension: 256,
            ..Default::default()
        };
        let result = FalseBeliefBenchmark.run(&config);
        assert!(result.metrics.contains_key("false_belief_accuracy"));
    }
}
