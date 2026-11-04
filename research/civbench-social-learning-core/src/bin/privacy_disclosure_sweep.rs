// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compare centralized full-inventory routing with a minimized recipient
//! readiness signal under matched tasks, seeds, message and attention budgets.
//!
//! This is a synthetic disclosure/performance benchmark, not a cryptographic
//! privacy proof or a deployed federated protocol.

use civbench_social_learning_core::adaptive_transmission::{
    simulate, simulate_with_program_and_disclosure, Config, InnovationTask,
    RecipientStateDisclosure, TaskFamily, TransmissionPolicy, TransmissionProgram,
    TransmissionScoreWeights,
};

const REPLICATES: u64 = 256;
const TASK_FAMILIES: [TaskFamily; 3] = [
    TaskFamily::Layered,
    TaskFamily::Branching,
    TaskFamily::Convergent,
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
        let variance = ((self.sum_squares - self.sum * self.sum / n) / (n - 1.0)).max(0.0);
        // Conservative 95% Student-t upper bound for df >= 120.
        Some(1.98 * variance.sqrt() / n.sqrt())
    }
}

#[derive(Default)]
struct OutcomeStats {
    coverage: OnlineStats,
    depth: OnlineStats,
    messages: OnlineStats,
    attention: OnlineStats,
    acquisitions: OnlineStats,
    redundancies: OnlineStats,
    individual_discoveries: OnlineStats,
}

impl OutcomeStats {
    fn observe(&mut self, run: &civbench_social_learning_core::adaptive_transmission::Run) {
        let final_metrics = run.final_metrics().expect("run includes initial metrics");
        self.coverage.push(final_metrics.result_coverage());
        self.depth.push(f64::from(final_metrics.max_discovered_depth));
        self.messages.push(
            run.history.iter().map(|step| step.transmissions_scheduled).sum::<usize>() as f64,
        );
        self.attention.push(
            run.history.iter().map(|step| step.attention_capacity_used).sum::<usize>() as f64,
        );
        self.acquisitions.push(
            run.history.iter().map(|step| step.successful_recipe_acquisitions).sum::<usize>() as f64,
        );
        self.redundancies.push(
            run.history.iter().map(|step| step.redundant_transmissions).sum::<usize>() as f64,
        );
        self.individual_discoveries.push(
            run.history.iter().map(|step| step.individual_recipe_discoveries).sum::<usize>() as f64,
        );
    }
}

fn config() -> Config {
    Config {
        population_size: 12,
        steps: 12,
        network_shortcut_probability: 0.08,
        initial_seed_agents: 1,
        initial_seed_recipes_per_agent: 3,
        individual_attempts_per_agent_step: 2,
        individual_discovery_probability: 0.04,
        transmission_budget_per_step: 12,
        attention_capacity_per_recipient: 2,
        acquisition_probability: 0.9,
        policy: TransmissionPolicy::NoSocial,
    }
}

fn program(config: Config) -> TransmissionProgram {
    TransmissionProgram {
        program_id: "frontier-coarse-disclosure-v1".to_owned(),
        version: 1,
        minimum_prerequisite_match_milli: 0,
        require_collective_novelty: false,
        max_messages_per_step: config.transmission_budget_per_step,
        max_messages_per_sender: config.transmission_budget_per_step,
        weights: TransmissionScoreWeights {
            immediate_executability: 100_000,
            prerequisite_match: 10_000,
            collective_novelty: 0,
            downstream_unlocks: 1_000,
            sender_knowledge: 1,
        },
    }
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn main() {
    println!(
        "task_family,disclosure_mode,replicates,coverage_mean,coverage_ci95_t,max_depth_mean,messages_per_run,attention_uses_per_run,acquisitions_per_run,redundant_messages_per_run,individual_discoveries_per_run,paired_delta_vs_central_ci95_t"
    );

    for family in TASK_FAMILIES {
        let task = InnovationTask::family_task(family);
        let run_config = config();
        let program = program(run_config);

        let mut central_stats = OutcomeStats::default();
        let mut coarse_stats = OutcomeStats::default();
        let mut no_social_stats = OutcomeStats::default();
        let mut coarse_vs_central = OnlineStats::default();
        let mut no_social_vs_central = OnlineStats::default();

        for seed in 0..REPLICATES {
            let central = simulate_with_program_and_disclosure(
                &task,
                run_config,
                seed,
                &program,
                RecipientStateDisclosure::CentralFullInventory,
            )
            .expect("central full-inventory condition must validate");
            let coarse = simulate_with_program_and_disclosure(
                &task,
                run_config,
                seed,
                &program,
                RecipientStateDisclosure::CoarseRecipeReadiness,
            )
            .expect("coarse readiness condition must validate");
            let no_social = simulate(
                &task,
                run_config,
                seed,
            )
            .expect("no-social control must validate");

            let central_coverage = central.final_metrics().unwrap().result_coverage();
            let coarse_coverage = coarse.final_metrics().unwrap().result_coverage();
            let no_social_coverage = no_social.final_metrics().unwrap().result_coverage();

            coarse_vs_central.push(coarse_coverage - central_coverage);
            no_social_vs_central.push(no_social_coverage - central_coverage);
            central_stats.observe(&central);
            coarse_stats.observe(&coarse);
            no_social_stats.observe(&no_social);
        }

        for (name, stats, delta) in [
            ("central_full_inventory", &central_stats, None),
            (
                "coarse_recipe_readiness",
                &coarse_stats,
                coarse_vs_central.ci95_half_width(),
            ),
            (
                "no_social_control",
                &no_social_stats,
                no_social_vs_central.ci95_half_width(),
            ),
        ] {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                family.name(),
                name,
                REPLICATES,
                fmt(stats.coverage.mean()),
                fmt(stats.coverage.ci95_half_width()),
                fmt(stats.depth.mean()),
                fmt(stats.messages.mean()),
                fmt(stats.attention.mean()),
                fmt(stats.acquisitions.mean()),
                fmt(stats.redundancies.mean()),
                fmt(stats.individual_discoveries.mean()),
                fmt(delta),
            );
        }

        println!(
            "# paired_deltas,{},coarse_minus_central_mean={},coarse_minus_central_ci95_t={},no_social_minus_central_mean={},no_social_minus_central_ci95_t={}",
            family.name(),
            fmt(coarse_vs_central.mean()),
            fmt(coarse_vs_central.ci95_half_width()),
            fmt(no_social_vs_central.mean()),
            fmt(no_social_vs_central.ci95_half_width()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disclosure_runner_uses_same_task_budget_and_seed_range() {
        let config = config();
        let task = InnovationTask::family_task(TaskFamily::Branching);
        let program = program(config);
        for seed in 0..8 {
            let full = simulate_with_program_and_disclosure(
                &task,
                config,
                seed,
                &program,
                RecipientStateDisclosure::CentralFullInventory,
            )
            .unwrap();
            let coarse = simulate_with_program_and_disclosure(
                &task,
                config,
                seed,
                &program,
                RecipientStateDisclosure::CoarseRecipeReadiness,
            )
            .unwrap();
            assert_eq!(full.graph, coarse.graph);
            assert!(full.history.iter().all(|step| {
                step.transmissions_scheduled <= config.transmission_budget_per_step
            }));
            assert!(coarse.history.iter().all(|step| {
                step.transmissions_scheduled <= config.transmission_budget_per_step
            }));
        }
    }
}
