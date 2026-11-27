// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reproducible sweep for the Czaplicka-Baumann-Rahwan cultural mediation
//! model. Default mode is a small smoke experiment. Use --paper-grid for the
//! source runner's N=100, Q=200, p_ER and r grids. Use --extension-grid for
//! the new exposure-aware recommender treatment.

use civbench_social_learning_core::czaplicka_2025::{
    simulate, Config, RecommendationPolicy, Run, AUTHOR_POPULATION_SIZE,
    AUTHOR_REPLICATIONS, AUTHOR_STEPS_PER_AGENT,
};
use std::env;

#[derive(Clone)]
struct Summary {
    mean_payoff: f64,
    mean_level: f64,
    mean_budget: f64,
    max_achievable: f64,
    active_unique_traits: f64,
    active_lineage_hhi: f64,
    innovations: f64,
    peer_copies: f64,
    recommendations: f64,
}

fn summarize_run(run: &Run) -> Summary {
    let tail_count = run.steps.len().min(200).max(1);
    let tail = &run.steps[run.steps.len() - tail_count..];
    let denominator = tail.len() as f64;
    let mean = |f: fn(&civbench_social_learning_core::czaplicka_2025::StepMetrics) -> f64| {
        tail.iter().map(f).sum::<f64>() / denominator
    };
    let final_step = run.steps.last().expect("run contains at least step zero");
    Summary {
        mean_payoff: mean(|step| step.mean_payoff),
        mean_level: mean(|step| step.mean_level),
        mean_budget: mean(|step| step.mean_remaining_budget),
        max_achievable: mean(|step| step.legacy_max_achievable_payoff_mean),
        active_unique_traits: mean(|step| step.active_unique_trait_count as f64),
        active_lineage_hhi: mean(|step| step.active_lineage_hhi),
        innovations: final_step.cumulative_unique_innovations as f64,
        peer_copies: final_step.cumulative_peer_copies as f64,
        recommendations: final_step.cumulative_recommendations as f64,
    }
}

#[derive(Default)]
struct OnlineStats {
    count: u64,
    sum: f64,
    sum_squared: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squared += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let centered = (self.sum_squared - self.sum * self.sum / n).max(0.0);
        let sd = (centered / (n - 1.0)).sqrt();
        Some(1.96 * sd / n.sqrt())
    }
}

#[derive(Default)]
struct SummaryStats {
    payoff: OnlineStats,
    level: OnlineStats,
    budget: OnlineStats,
    max_achievable: OnlineStats,
    active_traits: OnlineStats,
    lineage_hhi: OnlineStats,
    innovations: OnlineStats,
    peer_copies: OnlineStats,
    recommendations: OnlineStats,
}

impl SummaryStats {
    fn push(&mut self, s: Summary) {
        self.payoff.push(s.mean_payoff);
        self.level.push(s.mean_level);
        self.budget.push(s.mean_budget);
        self.max_achievable.push(s.max_achievable);
        self.active_traits.push(s.active_unique_traits);
        self.lineage_hhi.push(s.active_lineage_hhi);
        self.innovations.push(s.innovations);
        self.peer_copies.push(s.peer_copies);
        self.recommendations.push(s.recommendations);
    }
}

fn fmt(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_default()
}

fn run_condition(
    population_size: usize,
    steps_per_agent: usize,
    replications: usize,
    p_er: f64,
    recommendation_probability: f64,
    policy: RecommendationPolicy,
    seed_base: u64,
) -> SummaryStats {
    let mut stats = SummaryStats::default();

    for replication in 0..replications {
        let config = Config {
            population_size,
            er_probability: p_er,
            recommendation_probability,
            steps_per_agent,
            recommendation_policy: policy,
            ..Config::default()
        };
        let seed = seed_base.wrapping_add(replication as u64);
        let run = simulate(config, seed).expect("predeclared sweep configuration must validate");
        stats.push(summarize_run(&run));
    }

    stats
}

fn emit_row(
    label: &str,
    mode: &str,
    population_size: usize,
    steps_per_agent: usize,
    replications: usize,
    p_er: f64,
    recommendation_probability: f64,
    policy_name: &str,
    stats: SummaryStats,
) {
    println!(
        "{label},{mode},{population_size},{steps_per_agent},{replications},{p_er:.4},{recommendation_probability:.2},{policy_name},{},{},{},{},{},{},{},{},{},{},{},{}",
        fmt(stats.payoff.mean()),
        fmt(stats.payoff.ci95_half_width()),
        fmt(stats.level.mean()),
        fmt(stats.level.ci95_half_width()),
        fmt(stats.budget.mean()),
        fmt(stats.max_achievable.mean()),
        fmt(stats.active_traits.mean()),
        fmt(stats.lineage_hhi.mean()),
        fmt(stats.innovations.mean()),
        fmt(stats.peer_copies.mean()),
        fmt(stats.recommendations.mean()),
        fmt(stats.recommendations.ci95_half_width()),
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let paper_grid = args.iter().any(|arg| arg == "--paper-grid");
    let extension_grid = args.iter().any(|arg| arg == "--extension-grid");
    let quick = !paper_grid && !extension_grid;

    let (population_size, steps_per_agent, replications, densities, ratios) = if paper_grid {
        (
            AUTHOR_POPULATION_SIZE,
            AUTHOR_STEPS_PER_AGENT,
            AUTHOR_REPLICATIONS,
            vec![
                0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
            ],
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
    } else if extension_grid {
        (
            AUTHOR_POPULATION_SIZE,
            AUTHOR_STEPS_PER_AGENT,
            50,
            vec![0.01, 0.05, 0.10, 0.20, 0.50, 1.00],
            vec![0.0, 0.3, 0.5, 0.7, 1.0],
        )
    } else {
        (
            20,
            2,
            3,
            vec![0.0, 0.1, 0.5, 1.0],
            vec![0.0, 0.5, 1.0],
        )
    };

    println!(
        "condition,mode,population_size,steps_per_agent,replications,p_er,r,policy,mean_payoff,mean_payoff_ci95,mean_level,mean_level_ci95,mean_budget,mean_max_achievable_payoff,active_unique_traits,active_lineage_hhi,cumulative_innovations,cumulative_peer_copies,cumulative_recommendations,cumulative_recommendations_ci95"
    );

    for (p_index, &p_er) in densities.iter().enumerate() {
        for (r_index, &recommendation_probability) in ratios.iter().enumerate() {
            let condition_index = (p_index * ratios.len() + r_index) as u64;
            let seed_base = 0xC2_A5_2025_u64.wrapping_add(condition_index << 24);

            let reference = run_condition(
                population_size,
                steps_per_agent,
                replications,
                p_er,
                recommendation_probability,
                RecommendationPolicy::GlobalBestByLevel,
                seed_base,
            );
            emit_row(
                "social_learning_plus_algorithmic_mediation",
                if paper_grid { "paper_grid" } else if extension_grid { "extension_grid" } else { "quick" },
                population_size,
                steps_per_agent,
                replications,
                p_er,
                recommendation_probability,
                "global_best",
                reference,
            );

            if !paper_grid {
                for weight in if quick { vec![5.0] } else { vec![0.5, 1.0, 2.0, 5.0] } {
                    let ucb = run_condition(
                        population_size,
                        steps_per_agent,
                        replications,
                        p_er,
                        recommendation_probability,
                        RecommendationPolicy::ExposureAwareUcb {
                            exploration_weight: weight,
                        },
                        seed_base,
                    );
                    emit_row(
                        "social_learning_plus_algorithmic_mediation",
                        if quick { "quick" } else { "extension_grid" },
                        population_size,
                        steps_per_agent,
                        replications,
                        p_er,
                        recommendation_probability,
                        &format!("exposure_ucb_{weight:.1}"),
                        ucb,
                    );
                }
            }
        }
    }
}
