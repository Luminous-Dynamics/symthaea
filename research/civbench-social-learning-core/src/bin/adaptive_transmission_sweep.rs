// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Matched-seed benchmark for state-aware content transmission over recipe DAGs.
//!
//! This is a synthetic mechanism benchmark, not a reproduction of Collective
//! LittleAlchemy or the LLM-guided adaptive-transmission paper.

use civbench_social_learning_core::adaptive_transmission::{
    simulate, Config, InnovationTask, TaskFamily, TransmissionPolicy,
};

const REPLICATES: u64 = 128;
const POPULATION_SIZES: [usize; 3] = [8, 16, 32];
const ATTENTION_CAPACITIES: [usize; 3] = [1, 2, 4];
const SHORTCUT_PROBABILITIES: [f64; 2] = [0.0, 0.2];
const POLICIES: [TransmissionPolicy; 7] = [
    TransmissionPolicy::NoSocial,
    TransmissionPolicy::FixedRing,
    TransmissionPolicy::FixedRingContentShuffled,
    TransmissionPolicy::RandomNetworkContent,
    TransmissionPolicy::StateAwareFrontier,
    TransmissionPolicy::StateAwareRandomContent,
    TransmissionPolicy::StateAwareRecipientShuffled,
];

#[derive(Default)]
struct OnlineStats {
    n: u64,
    sum: f64,
    sum_squares: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let centered = (self.sum_squares - self.sum * self.sum / n).max(0.0);
        let sample_sd = (centered / (n - 1.0)).sqrt();
        Some(1.96 * sample_sd / n.sqrt())
    }
}

#[derive(Default)]
struct ConditionStats {
    final_result_coverage: OnlineStats,
    final_max_depth: OnlineStats,
    final_mean_results_per_agent: OnlineStats,
    final_mean_recipes_per_agent: OnlineStats,
    individual_discoveries: OnlineStats,
    social_recipe_acquisitions: OnlineStats,
    new_collective_results: OnlineStats,
    transmissions_scheduled: OnlineStats,
    redundant_transmissions: OnlineStats,
    attention_used: OnlineStats,
    transmission_efficiency: OnlineStats,
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn policies_name(policy: TransmissionPolicy) -> &'static str {
    policy.name()
}

fn accumulate(stats: &mut ConditionStats, family: TaskFamily, config: Config) {
    let task = InnovationTask::family_task(family);
    for seed in 0..REPLICATES {
        let run = simulate(&task, config, seed)
            .expect("predeclared adaptive-transmission condition must validate");
        let Some(final_metrics) = run.final_metrics() else {
            continue;
        };

        let cumulative_individual: usize = run.history.iter()
            .skip(1)
            .map(|metrics| metrics.individual_recipe_discoveries)
            .sum();
        let cumulative_acquisitions: usize = run.history.iter()
            .map(|metrics| metrics.successful_recipe_acquisitions)
            .sum();
        let cumulative_new_results: usize = run.history.iter()
            .map(|metrics| metrics.new_collective_results)
            .sum();
        let cumulative_transmissions: usize = run.history.iter()
            .map(|metrics| metrics.transmissions_scheduled)
            .sum();
        let cumulative_redundant: usize = run.history.iter()
            .map(|metrics| metrics.redundant_transmissions)
            .sum();
        let cumulative_attention: usize = run.history.iter()
            .map(|metrics| metrics.attention_capacity_used)
            .sum();
        let efficiency = if cumulative_transmissions == 0 {
            None
        } else {
            Some(cumulative_new_results as f64 / cumulative_transmissions as f64)
        };

        stats.final_result_coverage.push(final_metrics.result_coverage());
        stats.final_max_depth.push(f64::from(final_metrics.max_discovered_depth));
        stats.final_mean_results_per_agent.push(
            final_metrics.mean_results_per_agent_milli as f64 / 1000.0
        );
        stats.final_mean_recipes_per_agent.push(
            final_metrics.mean_recipes_known_per_agent_milli as f64 / 1000.0
        );
        stats.individual_discoveries.push(cumulative_individual as f64);
        stats.social_recipe_acquisitions.push(cumulative_acquisitions as f64);
        stats.new_collective_results.push(cumulative_new_results as f64);
        stats.transmissions_scheduled.push(cumulative_transmissions as f64);
        stats.redundant_transmissions.push(cumulative_redundant as f64);
        stats.attention_used.push(cumulative_attention as f64);
        if let Some(value) = efficiency {
            stats.transmission_efficiency.push(value);
        }
    }
}

fn emit_condition(
    family: TaskFamily,
    population_size: usize,
    shortcut_probability: f64,
    attention_capacity: usize,
    policy: TransmissionPolicy,
) {
    let config = Config {
        population_size,
        steps: 12,
        network_shortcut_probability: shortcut_probability,
        initial_seed_agents: 1,
        initial_seed_recipes_per_agent: 3,
        individual_attempts_per_agent_step: 2,
        individual_discovery_probability: 0.04,
        transmission_budget_per_step: population_size,
        attention_capacity_per_recipient: attention_capacity,
        acquisition_probability: 0.9,
        policy,
    };
    let mut stats = ConditionStats::default();
    accumulate(&mut stats, family, config);

    println!(
        "{},{},{:.3},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        family.name(),
        population_size,
        shortcut_probability,
        attention_capacity,
        policies_name(policy),
        REPLICATES,
        12,
        fmt(stats.final_result_coverage.mean()),
        fmt(stats.final_result_coverage.ci95_half_width()),
        fmt(stats.final_max_depth.mean()),
        fmt(stats.final_mean_results_per_agent.mean()),
        fmt(stats.final_mean_recipes_per_agent.mean()),
        fmt(stats.individual_discoveries.mean()),
        fmt(stats.social_recipe_acquisitions.mean()),
        fmt(stats.new_collective_results.mean()),
        fmt(stats.transmissions_scheduled.mean()),
        fmt(stats.redundant_transmissions.mean()),
        fmt(stats.attention_used.mean()),
        fmt(stats.transmission_efficiency.mean()),
        "same_seed_schedule"
    );
}

fn main() {
    println!(
        "task_family,population_size,shortcut_probability,attention_capacity_per_recipient,policy,replicates,steps,final_result_coverage_mean,final_result_coverage_ci95,max_discovered_depth_mean,mean_results_per_agent,mean_recipes_per_agent,individual_discoveries_per_run,social_recipe_acquisitions_per_run,new_collective_results_per_run,transmissions_per_run,redundant_transmissions_per_run,attention_uses_per_run,new_results_per_transmission,seed_design"
    );

    for family in [TaskFamily::Layered, TaskFamily::Branching, TaskFamily::Convergent] {
        for population_size in POPULATION_SIZES {
            for shortcut_probability in SHORTCUT_PROBABILITIES {
                for attention_capacity in ATTENTION_CAPACITIES {
                    for policy in POLICIES {
                        emit_condition(
                            family,
                            population_size,
                            shortcut_probability,
                            attention_capacity,
                            policy,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_grid_is_fully_declared_and_has_648_conditions() {
        let conditions = 3
            * POPULATION_SIZES.len()
            * SHORTCUT_PROBABILITIES.len()
            * ATTENTION_CAPACITIES.len()
            * POLICIES.len();
        assert_eq!(conditions, 378);
        assert_eq!(REPLICATES, 128);
    }

    #[test]
    fn state_shuffled_and_state_aware_policies_are_both_in_the_grid() {
        assert!(POLICIES.contains(&TransmissionPolicy::StateAwareFrontier));
        assert!(POLICIES.contains(&TransmissionPolicy::StateAwareRandomContent));
        assert!(POLICIES.contains(&TransmissionPolicy::StateAwareRecipientShuffled));
    }
}
