// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Scoring Model Comparison — Multi-Model Governance Evaluation
//!
//! Runs the multiworld simulator to completion, then evaluates ALL registered
//! scoring models (4D, 8D, 3D) on the final population to compare governance
//! outcomes across different model architectures.
//!
//! ## Output
//!
//! Structured comparison report to stderr + CSV-compatible summary to stdout.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --bin scoring_model_comparison
//! cargo run --release --bin scoring_model_comparison -- --years 50 --seed 42
//! cargo run --release --bin scoring_model_comparison -- --seed 123 --years 300
//! ```

use mycelix_multiworld_sim::{
    config::SimulationConfig,
    scoring_bridge::{builtin_scoring_models, ScoringModelGovernance},
    MultiWorldSimulator,
};

use mycelix_bridge_common::sovereign_gate::CivicTier;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let years = parse_arg(&args, "--years", 150u32);
    let seed = parse_arg(&args, "--seed", 42u64);

    eprintln!("=== SCORING MODEL COMPARISON ===");
    eprintln!("Seed: {seed}, Duration: {years} years\n");

    // Run simulation
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.total_ticks = years * 12;
    config.policy.trust_weighted_governance = true;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    let final_tick = years * 12;

    eprintln!(
        "Simulation complete: CVS={:.4}, Pop={}, Worlds={}, Disasters={}",
        report.final_cvs, report.final_population, report.final_worlds, report.total_disasters
    );

    // Collect living agents
    let agents: Vec<&mycelix_multiworld_sim::agent::CivAgent> = sim
        .worlds
        .iter()
        .flat_map(|w| w.agents.iter())
        .filter(|a| a.is_alive())
        .collect();

    let total = agents.len();
    eprintln!(
        "\nEvaluating {} living agents across {} scoring models...\n",
        total, 3
    );

    if total == 0 {
        eprintln!("No surviving agents — cannot evaluate.");
        return;
    }

    // Load scoring models
    let models = builtin_scoring_models();

    // CSV header
    println!(
        "seed,years,model_id,mean_score,mean_weight,\
         observer_pct,participant_pct,citizen_pct,steward_pct,guardian_pct,\
         voting_agents,total_agents,gini_weight,cvs"
    );

    // Evaluate each model
    let mut results = Vec::new();
    for model in &models {
        let pop = evaluate_population(model, &agents, final_tick);

        println!(
            "{seed},{years},{},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.4},{:.4}",
            model.descriptor.model_id,
            pop.mean_score,
            pop.mean_weight,
            pop.observer_pct,
            pop.participant_pct,
            pop.citizen_pct,
            pop.steward_pct,
            pop.guardian_pct,
            pop.voting_agents,
            pop.total_agents,
            pop.gini_weight,
            report.final_cvs,
        );

        results.push((model.descriptor.model_id.clone(), pop));
    }

    // Comparison report
    eprintln!("\n=== MODEL COMPARISON ===\n");
    eprintln!(
        "{:<25} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "Model", "Score", "Weight", "Citizen+%", "Steward+%", "Voting%", "Gini"
    );
    eprintln!("{}", "-".repeat(85));

    for (model_id, pop) in &results {
        let citizen_plus = pop.citizen_pct + pop.steward_pct + pop.guardian_pct;
        let steward_plus = pop.steward_pct + pop.guardian_pct;
        let voting_pct = pop.voting_agents as f64 / pop.total_agents.max(1) as f64;

        eprintln!(
            "{:<25} {:>8.4} {:>8.4} {:>9.1}% {:>9.1}% {:>9.1}% {:>8.4}",
            model_id,
            pop.mean_score,
            pop.mean_weight,
            citizen_plus * 100.0,
            steward_plus * 100.0,
            voting_pct * 100.0,
            pop.gini_weight,
        );
    }

    // Divergence analysis
    eprintln!("\n=== DIVERGENCE ANALYSIS ===\n");

    if results.len() >= 2 {
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let (id_a, pop_a) = &results[i];
                let (id_b, pop_b) = &results[j];

                let score_diff = (pop_a.mean_score - pop_b.mean_score).abs();
                let weight_diff = (pop_a.mean_weight - pop_b.mean_weight).abs();

                // Check if models would produce different governance outcomes
                let a_citizen_plus = pop_a.citizen_pct + pop_a.steward_pct + pop_a.guardian_pct;
                let b_citizen_plus = pop_b.citizen_pct + pop_b.steward_pct + pop_b.guardian_pct;
                let tier_diff = (a_citizen_plus - b_citizen_plus).abs();

                eprintln!("{} vs {}:", id_a, id_b);
                eprintln!("  Score diff:    {:.4}", score_diff);
                eprintln!("  Weight diff:   {:.4}", weight_diff);
                eprintln!("  Citizen+ diff: {:.1}%", tier_diff * 100.0);
                eprintln!(
                    "  Gini diff:     {:.4}",
                    (pop_a.gini_weight - pop_b.gini_weight).abs()
                );
                eprintln!();
            }
        }
    }
}

/// Population-level scoring results.
struct PopulationResults {
    mean_score: f64,
    mean_weight: f64,
    observer_pct: f64,
    participant_pct: f64,
    citizen_pct: f64,
    steward_pct: f64,
    guardian_pct: f64,
    total_agents: usize,
    voting_agents: usize,
    gini_weight: f64,
}

/// Evaluate a scoring model across all living agents.
fn evaluate_population(
    model: &ScoringModelGovernance,
    agents: &[&mycelix_multiworld_sim::agent::CivAgent],
    current_tick: u32,
) -> PopulationResults {
    let mut scores = Vec::with_capacity(agents.len());
    let mut weights = Vec::with_capacity(agents.len());
    let mut tier_counts = [0usize; 5];
    let mut voting = 0usize;

    for agent in agents {
        let result = model.evaluate_agent(agent, current_tick, 0.0);
        scores.push(result.decayed_score);
        weights.push(result.vote_weight);

        let tier_idx = match result.tier {
            CivicTier::Observer => 0,
            CivicTier::Participant => 1,
            CivicTier::Citizen => 2,
            CivicTier::Steward => 3,
            CivicTier::Guardian => 4,
        };
        tier_counts[tier_idx] += 1;

        if result.vote_weight > 0.0 {
            voting += 1;
        }
    }

    let n = agents.len() as f64;
    let mean_score = if n > 0.0 {
        scores.iter().sum::<f64>() / n
    } else {
        0.0
    };
    let mean_weight = if n > 0.0 {
        weights.iter().sum::<f64>() / n
    } else {
        0.0
    };

    PopulationResults {
        mean_score,
        mean_weight,
        observer_pct: tier_counts[0] as f64 / n.max(1.0),
        participant_pct: tier_counts[1] as f64 / n.max(1.0),
        citizen_pct: tier_counts[2] as f64 / n.max(1.0),
        steward_pct: tier_counts[3] as f64 / n.max(1.0),
        guardian_pct: tier_counts[4] as f64 / n.max(1.0),
        total_agents: agents.len(),
        voting_agents: voting,
        gini_weight: gini_coefficient(&weights),
    }
}

/// Compute the Gini coefficient of voting weights.
///
/// 0.0 = perfect equality (everyone has same weight).
/// 1.0 = perfect inequality (one person has all weight).
fn gini_coefficient(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sum: f64 = sorted.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let mut numerator = 0.0f64;
    for (i, &v) in sorted.iter().enumerate() {
        numerator += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * v;
    }

    (numerator / (n as f64 * sum)).clamp(0.0, 1.0)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
