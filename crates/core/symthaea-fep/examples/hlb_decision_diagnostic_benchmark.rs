// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic, source-bound synthetic comparison of diagnostic policies.
//!
//! This runner deliberately separates model EIG, optimal-decision information
//! gain, and EVSI. Its result is a synthetic fixture artifact, not a live
//! ActiveInferenceAgent evaluation or an empirical deployment study.

use serde_json::json;
use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use symthaea_fep::decision_diagnostic_benchmark::{
    DiagnosticScenario, run_diagnostic_policy_benchmark,
};
use symthaea_fep::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionHypothesis, DiagnosticAction,
};
use symthaea_fep::sequential_diagnostic_benchmark::run_sequential_diagnostic_policy_benchmark;
use symthaea_fep::sequential_diagnostic_planning::SequentialDiagnosticConfig;

const EPISODES: usize = 10_000;
const SEED: u64 = 0x5EED;

fn scenario() -> DiagnosticScenario {
    let mut hypotheses = Vec::with_capacity(4);
    let mut truth = Vec::with_capacity(4);

    for index in 0..4 {
        let fault_a = index < 2;
        let nuisance_zero = index % 2 == 0;
        let nuisance_distribution = if nuisance_zero {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        let fault_distribution = if fault_a {
            vec![0.7, 0.3]
        } else {
            vec![0.3, 0.7]
        };

        hypotheses.push(DecisionHypothesis {
            id: format!("model-{index}"),
            prior_probability: 0.25,
            outcome_probabilities_by_action: vec![
                nuisance_distribution.clone(),
                fault_distribution.clone(),
            ],
            loss_by_final_decision: if fault_a {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            },
        });
        truth.push(vec![nuisance_distribution, fault_distribution]);
    }

    DiagnosticScenario {
        scenario_id: "irrelevant-high-eig-vs-relevant-low-eig".to_owned(),
        hypotheses,
        actions: vec![
            DiagnosticAction {
                id: "inspect-irrelevant-nuisance".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "inspect-decision-relevant-fault".into(),
                observation_cost: 0.05,
                safety_risk: 0.0,
            },
        ],
        true_outcome_probabilities_by_hypothesis_action: truth,
    }
}

fn sequential_scenario() -> DiagnosticScenario {
    let mut hypotheses = Vec::with_capacity(8);
    let mut truth = Vec::with_capacity(8);

    for nuisance in 0..2 {
        for first_bit in 0..2 {
            for second_bit in 0..2 {
                let xor = first_bit ^ second_bit;
                let nuisance_outcomes = if nuisance == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                let first_outcomes = if first_bit == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                let second_outcomes = if second_bit == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                let action_outcomes = vec![
                    nuisance_outcomes.clone(),
                    first_outcomes.clone(),
                    second_outcomes.clone(),
                ];

                hypotheses.push(DecisionHypothesis {
                    id: format!("fault-{nuisance}-{first_bit}-{second_bit}"),
                    prior_probability: 0.125,
                    outcome_probabilities_by_action: action_outcomes,
                    loss_by_final_decision: if xor == 0 {
                        vec![0.0, 1.0]
                    } else {
                        vec![1.0, 0.0]
                    },
                });
                truth.push(vec![nuisance_outcomes, first_outcomes, second_outcomes]);
            }
        }
    }

    DiagnosticScenario {
        scenario_id: "complementary-probes-with-irrelevant-high-eig-nuisance".to_owned(),
        hypotheses,
        actions: vec![
            DiagnosticAction {
                id: "inspect-irrelevant-nuisance".into(),
                observation_cost: 0.05,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "reveal-first-fault-bit".into(),
                observation_cost: 0.05,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "reveal-second-fault-bit".into(),
                observation_cost: 0.05,
                safety_risk: 0.0,
            },
        ],
        true_outcome_probabilities_by_hypothesis_action: truth,
    }
}

fn required_source_sha() -> Result<String, Box<dyn Error>> {
    let source_sha = env::var("SYMTHAEA_SOURCE_SHA")?;
    if source_sha.len() != 40 || !source_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal Git SHA",
        )
        .into());
    }
    Ok(source_sha)
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = required_source_sha()?;
    let scenario = scenario();
    let config = DecisionDiagnosticConfig {
        observation_cost_weight: 1.0,
        safety_risk_weight: 0.0,
        maximum_safety_risk: 1.0,
    };
    let result = run_diagnostic_policy_benchmark(&scenario, EPISODES, SEED, config)?;

    let sequential_scenario = sequential_scenario();
    let sequential_episodes = 4096;
    let sequential_seed = 0x5EED_2;
    let sequential_planning = SequentialDiagnosticConfig {
        horizon: 2,
        max_policy_expansions: 4096,
    };
    let sequential_result = run_sequential_diagnostic_policy_benchmark(
        &sequential_scenario,
        sequential_episodes,
        sequential_seed,
        config,
        sequential_planning,
    )?;

    let report = json!({
        "schema_version": "hlb-decision-diagnostic-result-v1",
        "status": "synthetic_fixture_result_not_environment_result",
        "claim_ceiling": "A deterministic finite-hypothesis policy comparison only; no live ActiveInferenceAgent, trained RL, field, or deployment result is established.",
        "source_sha": source_sha,
        "protocol": {
            "task_id": scenario.scenario_id.clone(),
            "episodes": EPISODES,
            "seed": SEED,
            "seed_hex": format!("0x{SEED:016X}"),
            "paired_episode_seeds": true,
            "confidence_interval_method": "paired_normal_approximation_95",
            "policy_order": [
                "NoProbe",
                "PureModelInformationGain",
                "OptimalDecisionInformationGain",
                "CostAwareEvsi",
                "AlwaysInspect"
            ]
        },
        "decision_diagnostic_config": config,
        "scenario": scenario,
        "result": result,
        "sequential_protocol": {
            "task_id": sequential_scenario.scenario_id.clone(),
            "episodes": sequential_episodes,
            "seed": sequential_seed,
            "seed_hex": format!("0x{sequential_seed:016X}"),
            "paired_episode_seeds": true,
            "confidence_interval_method": "paired_normal_approximation_95",
            "policy_order": ["NoProbe", "GreedyEvsi", "SequentialEvsi", "AlwaysInspect"]
        },
        "sequential_decision_diagnostic_config": config,
        "sequential_planning_config": sequential_planning,
        "sequential_scenario": sequential_scenario,
        "sequential_result": sequential_result
    });

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
