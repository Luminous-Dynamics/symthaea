//! False belief tasks (Sally-Anne type).
//!
//! Tests whether the system tracks that an agent holds a stale belief
//! about a state of the world that has changed in their absence.
//! Uses WM similarity to compare belief vs reality encoding proximity.

use crate::adapter::scenario::{Scenario, ScenarioAdapter};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// False belief benchmark (Sally-Anne paradigm).
pub struct FalseBeliefBenchmark;

/// A single false-belief scenario.
struct FalseBeliefScenario {
    /// Setup sentences (agent observes object location).
    setup: Vec<&'static str>,
    /// Agent leaves (becomes absent for the change).
    absence: &'static str,
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
                absence: "Sally leaves the room",
                change: "Anne moves the marble to the box",
                belief_location: "Sally thinks the marble is in the basket",
                reality_location: "The marble is actually in the box",
            },
            FalseBeliefScenario {
                setup: vec![
                    "John puts his chocolate in the cupboard",
                    "John remembers putting chocolate in the cupboard",
                ],
                absence: "John goes outside to play",
                change: "Mother moves the chocolate to the drawer",
                belief_location: "John thinks the chocolate is in the cupboard",
                reality_location: "The chocolate is actually in the drawer",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Alice places her book on the shelf",
                    "Alice knows her book is on the shelf",
                ],
                absence: "Alice goes to school",
                change: "Bob moves the book to the table",
                belief_location: "Alice thinks the book is on the shelf",
                reality_location: "The book is actually on the table",
            },
            FalseBeliefScenario {
                setup: vec![
                    "Tom hides his toy behind the curtain",
                    "Tom saw himself hide the toy behind the curtain",
                ],
                absence: "Tom goes to the kitchen",
                change: "Emma moves the toy under the bed",
                belief_location: "Tom thinks the toy is behind the curtain",
                reality_location: "The toy is actually under the bed",
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

        // Present setup: agent observes the initial state
        for sentence in &scenario.setup {
            let hv = adapter.encode(&Scenario::new(*sentence), dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Agent leaves
        let absence_hv = adapter.encode(&Scenario::new(scenario.absence), dim);
        wm.perceive(absence_hv);
        wm.tick();

        // Object is moved (agent doesn't see this)
        let change_hv = adapter.encode(&Scenario::new(scenario.change), dim);
        wm.perceive(change_hv);
        wm.tick();

        // Test: where does the ABSENT agent believe the object is?
        let belief_hv = adapter.encode(&Scenario::new(scenario.belief_location), dim);
        let reality_hv = adapter.encode(&Scenario::new(scenario.reality_location), dim);

        // Check WM: the correct answer should be the BELIEF (stale) location
        let contents = wm.contents();
        let belief_sim = contents.iter().map(|item| item.similarity(&belief_hv)).fold(0.0f32, f32::max);
        let reality_sim = contents.iter().map(|item| item.similarity(&reality_hv)).fold(0.0f32, f32::max);

        // Correct if belief location scores higher than reality location
        // (system understands the agent has a FALSE belief, not reality)
        if belief_sim > reality_sim { 1.0 } else { 0.0 }
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
        for trial in 0..config.trials_per_condition {
            let acc = self.run_trial(config, trial);
            accuracies.push(acc);
        }

        result.insert("false_belief_accuracy", MetricValue::from_samples(&accuracies));

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
