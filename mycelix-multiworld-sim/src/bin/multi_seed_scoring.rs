// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Multi-Seed Scoring Model Comparison — Statistical Governance Analysis
//!
//! Runs the multiworld simulator across N seeds with all 3 scoring models,
//! producing aggregate statistics with confidence intervals and effect sizes.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --bin multi_seed_scoring -- --seeds 10 --years 50
//! cargo run --release --bin multi_seed_scoring -- --seeds 50 --years 150
//! ```

use mycelix_bridge_common::sovereign_gate::CivicTier;
use mycelix_multiworld_sim::{
    config::SimulationConfig,
    scoring_bridge::{builtin_scoring_models, ScoringModelGovernance},
    MultiWorldSimulator,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds = parse_arg(&args, "--seeds", 10u64);
    let years = parse_arg(&args, "--years", 50u32);

    eprintln!("=== MULTI-SEED SCORING MODEL COMPARISON ===");
    eprintln!("Seeds: {n_seeds}, Duration: {years} years, Models: 3\n");

    let models = builtin_scoring_models();

    // Collect per-seed, per-model results
    let mut all_results: Vec<Vec<ModelStats>> = Vec::new(); // [seed][model]

    for seed in 0..n_seeds {
        let actual_seed = seed * 7 + 42; // spread seeds
        eprint!("  Seed {}/{} ({})", seed + 1, n_seeds, actual_seed);

        let mut config = SimulationConfig::default_150_year();
        config.seed = actual_seed;
        config.total_ticks = years * 12;
        config.policy.trust_weighted_governance = true;

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        let final_tick = years * 12;

        let agents: Vec<&mycelix_multiworld_sim::agent::CivAgent> = sim
            .worlds
            .iter()
            .flat_map(|w| w.agents.iter())
            .filter(|a| a.is_alive())
            .collect();

        let mut seed_results = Vec::new();
        for model in &models {
            let stats = evaluate_model_stats(model, &agents, final_tick, report.final_cvs);
            seed_results.push(stats);
        }

        eprintln!(" — CVS: {:.4}, Pop: {}", report.final_cvs, agents.len());
        all_results.push(seed_results);
    }

    // Aggregate statistics per model
    eprintln!(
        "\n=== AGGREGATE RESULTS ({} seeds × {} years) ===\n",
        n_seeds, years
    );

    let model_names: Vec<&str> = models
        .iter()
        .map(|m| m.descriptor.model_id.as_str())
        .collect();

    eprintln!(
        "{:<25} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Model", "Score", "Steward+%", "Gini", "CVS", "Voting%"
    );
    eprintln!("{}", "-".repeat(80));

    let mut model_aggregates: Vec<ModelAggregate> = Vec::new();

    for (m_idx, name) in model_names.iter().enumerate() {
        let scores: Vec<f64> = all_results.iter().map(|r| r[m_idx].mean_score).collect();
        let stewards: Vec<f64> = all_results
            .iter()
            .map(|r| r[m_idx].steward_plus_pct)
            .collect();
        let ginis: Vec<f64> = all_results.iter().map(|r| r[m_idx].gini).collect();
        let cvss: Vec<f64> = all_results.iter().map(|r| r[m_idx].cvs).collect();
        let votings: Vec<f64> = all_results.iter().map(|r| r[m_idx].voting_pct).collect();

        let agg = ModelAggregate {
            name: name.to_string(),
            score: aggregate(&scores),
            steward_plus: aggregate(&stewards),
            gini: aggregate(&ginis),
            cvs: aggregate(&cvss),
            voting: aggregate(&votings),
        };

        eprintln!(
            "{:<25} {:>10} {:>10} {:>12} {:>12} {:>10}",
            name,
            format!("{:.3}±{:.3}", agg.score.mean, agg.score.ci95),
            format!(
                "{:.1}±{:.1}",
                agg.steward_plus.mean * 100.0,
                agg.steward_plus.ci95 * 100.0
            ),
            format!("{:.3}±{:.3}", agg.gini.mean, agg.gini.ci95),
            format!("{:.3}±{:.3}", agg.cvs.mean, agg.cvs.ci95),
            format!(
                "{:.1}±{:.1}",
                agg.voting.mean * 100.0,
                agg.voting.ci95 * 100.0
            ),
        );

        model_aggregates.push(agg);
    }

    // Pairwise comparisons with effect sizes
    eprintln!("\n=== PAIRWISE COMPARISONS (Cohen's d) ===\n");

    for i in 0..model_names.len() {
        for j in (i + 1)..model_names.len() {
            let scores_a: Vec<f64> = all_results.iter().map(|r| r[i].mean_score).collect();
            let scores_b: Vec<f64> = all_results.iter().map(|r| r[j].mean_score).collect();

            let stewards_a: Vec<f64> = all_results.iter().map(|r| r[i].steward_plus_pct).collect();
            let stewards_b: Vec<f64> = all_results.iter().map(|r| r[j].steward_plus_pct).collect();

            let ginis_a: Vec<f64> = all_results.iter().map(|r| r[i].gini).collect();
            let ginis_b: Vec<f64> = all_results.iter().map(|r| r[j].gini).collect();

            let d_score = cohens_d(&scores_a, &scores_b);
            let d_steward = cohens_d(&stewards_a, &stewards_b);
            let d_gini = cohens_d(&ginis_a, &ginis_b);

            eprintln!("{} vs {}:", model_names[i], model_names[j]);
            eprintln!(
                "  Score:    d = {:+.3} ({})",
                d_score,
                effect_label(d_score)
            );
            eprintln!(
                "  Steward+: d = {:+.3} ({})",
                d_steward,
                effect_label(d_steward)
            );
            eprintln!("  Gini:     d = {:+.3} ({})", d_gini, effect_label(d_gini));
            eprintln!();
        }
    }

    // CSV output to stdout
    println!(
        "seed,model_id,mean_score,steward_plus_pct,citizen_plus_pct,gini,cvs,voting_pct,population"
    );
    for (seed_idx, seed_results) in all_results.iter().enumerate() {
        let actual_seed = seed_idx as u64 * 7 + 42;
        for stats in seed_results {
            println!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                actual_seed,
                stats.model_id,
                stats.mean_score,
                stats.steward_plus_pct,
                stats.citizen_plus_pct,
                stats.gini,
                stats.cvs,
                stats.voting_pct,
                stats.population,
            );
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

struct ModelStats {
    model_id: String,
    mean_score: f64,
    steward_plus_pct: f64,
    citizen_plus_pct: f64,
    gini: f64,
    cvs: f64,
    voting_pct: f64,
    population: usize,
}

struct AggStat {
    mean: f64,
    stddev: f64,
    ci95: f64,
}

struct ModelAggregate {
    name: String,
    score: AggStat,
    steward_plus: AggStat,
    gini: AggStat,
    cvs: AggStat,
    voting: AggStat,
}

fn aggregate(values: &[f64]) -> AggStat {
    let n = values.len() as f64;
    if n == 0.0 {
        return AggStat {
            mean: 0.0,
            stddev: 0.0,
            ci95: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    let ci95 = 1.96 * stddev / n.sqrt(); // 95% confidence interval
    AggStat { mean, stddev, ci95 }
}

fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let agg_a = aggregate(a);
    let agg_b = aggregate(b);
    let pooled_sd = ((agg_a.stddev.powi(2) + agg_b.stddev.powi(2)) / 2.0).sqrt();
    if pooled_sd < 1e-10 {
        return 0.0;
    }
    (agg_a.mean - agg_b.mean) / pooled_sd
}

fn effect_label(d: f64) -> &'static str {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "negligible"
    } else if abs_d < 0.5 {
        "small"
    } else if abs_d < 0.8 {
        "medium"
    } else {
        "LARGE"
    }
}

fn evaluate_model_stats(
    model: &ScoringModelGovernance,
    agents: &[&mycelix_multiworld_sim::agent::CivAgent],
    current_tick: u32,
    cvs: f64,
) -> ModelStats {
    let mut total_score = 0.0f64;
    let mut weights = Vec::new();
    let mut tier_counts = [0usize; 5];
    let mut voting = 0usize;

    for agent in agents {
        let result = model.evaluate_agent(agent, current_tick, 0.0);
        total_score += result.decayed_score;
        weights.push(result.vote_weight);

        let idx = match result.tier {
            CivicTier::Observer => 0,
            CivicTier::Participant => 1,
            CivicTier::Citizen => 2,
            CivicTier::Steward => 3,
            CivicTier::Guardian => 4,
        };
        tier_counts[idx] += 1;
        if result.vote_weight > 0.0 {
            voting += 1;
        }
    }

    let n = agents.len() as f64;
    let citizen_plus = (tier_counts[2] + tier_counts[3] + tier_counts[4]) as f64 / n.max(1.0);
    let steward_plus = (tier_counts[3] + tier_counts[4]) as f64 / n.max(1.0);

    ModelStats {
        model_id: model.descriptor.model_id.clone(),
        mean_score: if n > 0.0 { total_score / n } else { 0.0 },
        steward_plus_pct: steward_plus,
        citizen_plus_pct: citizen_plus,
        gini: gini_coefficient(&weights),
        cvs,
        voting_pct: voting as f64 / n.max(1.0),
        population: agents.len(),
    }
}

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
    let mut num = 0.0f64;
    for (i, &v) in sorted.iter().enumerate() {
        num += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * v;
    }
    (num / (n as f64 * sum)).clamp(0.0, 1.0)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
