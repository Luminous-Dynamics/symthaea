// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use civbench_social_learning_core::{
    run_history, AgentState, LearningRule, Network, Population,
};

fn main() {
    let network = Network::from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        .expect("valid line network");
    let population = Population::new(vec![
        AgentState::informed(true, 10_000, 10, 10_000),
        AgentState::uninformed(0, 1_000),
        AgentState::uninformed(0, 1_000),
        AgentState::uninformed(0, 1_000),
        AgentState::uninformed(0, 1_000),
        AgentState::uninformed(0, 1_000),
        AgentState::uninformed(0, 1_000),
    ]);

    println!("step,coverage,accuracy_among_informed,consensus,high_confidence_error");
    for (step, metrics) in run_history(
        &population,
        &network,
        LearningRule::MajorityImitation,
        true,
        6,
        8_000,
    )
    .expect("matching population and graph")
    .into_iter()
    .enumerate()
    {
        println!(
            "{step},{:.3},{},{},{:?}",
            metrics.coverage.unwrap_or(0.0),
            metrics
                .accuracy_among_informed
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "undefined".to_string()),
            metrics
                .consensus_fraction
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "undefined".to_string()),
            metrics.high_confidence_error_fraction
        );
    }
}
