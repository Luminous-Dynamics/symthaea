// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sweep recipient-side coarse-need disclosure quotas and report the
//! communication/disclosure/utility frontier under common task seeds.
//!
//! This is a synthetic mechanism experiment. The quota changes the set of
//! recipient need signals visible to the planner; realized message volume may
//! consequently change. The output reports, rather than hides, that cost.
//! Field counts are a disclosure-volume proxy, not bits or a privacy metric.

use civbench_social_learning_core::adaptive_transmission::{
    simulate, simulate_with_program_and_disclosure_budget, Config, InnovationTask,
    RecipientStateDisclosure, TaskFamily, TransmissionPolicy, TransmissionProgram,
    TransmissionScoreWeights,
};

const REPLICATES: u64 = 256;
const SIGNAL_QUOTAS: [usize; 4] = [1, 2, 4, 8];
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
    disclosure_units: OnlineStats,
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
        self.disclosure_units.push(run.recipient_state_disclosure_units as f64);
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
        program_id: "frontier-coarse-disclosure-quota-sweep-v1".to_owned(),
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

fn print_row(
    family: TaskFamily,
    mode: &str,
    quota: Option<usize>,
    stats: &OutcomeStats,
    paired_delta: Option<&OnlineStats>,
) {
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        family.name(),
        mode,
        quota.map(|value| value.to_string()).unwrap_or_default(),
        REPLICATES,
        fmt(stats.coverage.mean()),
        fmt(stats.coverage.ci95_half_width()),
        fmt(stats.depth.mean()),
        fmt(stats.messages.mean()),
        fmt(stats.attention.mean()),
        fmt(stats.disclosure_units.mean()),
        fmt(stats.acquisitions.mean()),
        fmt(stats.redundancies.mean()),
        fmt(stats.individual_discoveries.mean()),
        fmt(paired_delta.and_then(OnlineStats::mean)),
        fmt(paired_delta.and_then(OnlineStats::ci95_half_width)),
    );
}

fn main() {
    println!(
        "task_family,disclosure_mode,signal_quota_per_recipient,replicates,coverage_mean,coverage_ci95_t,max_depth_mean,messages_per_run,attention_uses_per_run,disclosure_field_units_per_run,acquisitions_per_run,redundant_messages_per_run,individual_discoveries_per_run,paired_delta_vs_central_mean,paired_delta_vs_central_ci95_t"
    );

    for family in TASK_FAMILIES {
        let task = InnovationTask::family_task(family);
        let run_config = config();
        let program = program(run_config);
        let mut central_stats = OutcomeStats::default();
        let mut no_social_stats = OutcomeStats::default();
        let mut quota_stats: Vec<OutcomeStats> =
            SIGNAL_QUOTAS.iter().map(|_| OutcomeStats::default()).collect();
        let mut paired_deltas: Vec<OnlineStats> =
            SIGNAL_QUOTAS.iter().map(|_| OnlineStats::default()).collect();

        for seed in 0..REPLICATES {
            let central = simulate_with_program_and_disclosure_budget(
                &task,
                run_config,
                seed,
                &program,
                RecipientStateDisclosure::CentralFullInventory,
                SIGNAL_QUOTAS[0],
            )
            .expect("central full-inventory condition must validate");
            let central_coverage = central.final_metrics().unwrap().result_coverage();
            central_stats.observe(&central);

            for (idx, quota) in SIGNAL_QUOTAS.iter().copied().enumerate() {
                let coarse = simulate_with_program_and_disclosure_budget(
                    &task,
                    run_config,
                    seed,
                    &program,
                    RecipientStateDisclosure::CoarseRecipeReadiness,
                    quota,
                )
                .expect("coarse quota condition must validate");
                let coarse_coverage = coarse.final_metrics().unwrap().result_coverage();
                paired_deltas[idx].push(coarse_coverage - central_coverage);
                quota_stats[idx].observe(&coarse);
            }

            let no_social = simulate(&task, run_config, seed)
                .expect("no-social control must validate");
            no_social_stats.observe(&no_social);
        }

        print_row(family, "central_full_inventory", None, &central_stats, None);
        for (idx, quota) in SIGNAL_QUOTAS.iter().copied().enumerate() {
            print_row(
                family,
                "coarse_recipe_readiness",
                Some(quota),
                &quota_stats[idx],
                Some(&paired_deltas[idx]),
            );
        }
        print_row(family, "no_social_control", None, &no_social_stats, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_sweep_respects_per_recipient_signal_caps_and_common_graph() {
        let task = InnovationTask::family_task(TaskFamily::Branching);
        let config = config();
        let program = program(config);
        let mut reference_graph = None;

        for quota in SIGNAL_QUOTAS {
            let run = simulate_with_program_and_disclosure_budget(
                &task,
                config,
                707,
                &program,
                RecipientStateDisclosure::CoarseRecipeReadiness,
                quota,
            )
            .unwrap();

            assert!(run.history.iter().all(|step| {
                step.transmissions_scheduled <= config.transmission_budget_per_step
            }));
            assert!(run.history.iter().all(|step| {
                step.attention_capacity_used
                    <= config.population_size * config.attention_capacity_per_recipient
            }));
            assert!(run.recipient_state_disclosure_units > 0);

            if let Some(graph) = &reference_graph {
                assert_eq!(&run.graph, graph);
            } else {
                reference_graph = Some(run.graph.clone());
            }
        }
    }

    #[test]
    fn quota_sweep_includes_central_and_no_social_controls() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = config();
        let program = program(config);
        let central = simulate_with_program_and_disclosure_budget(
            &task,
            config,
            811,
            &program,
            RecipientStateDisclosure::CentralFullInventory,
            SIGNAL_QUOTAS[0],
        )
        .unwrap();
        let no_social = simulate(&task, config, 811).unwrap();
        let central_coverage = central.final_metrics().unwrap().result_coverage();
        assert!(central_coverage.is_finite());
        assert!((0.0..=1.0).contains(&central_coverage));
        assert_eq!(
            no_social.history.iter().map(|s| s.transmissions_scheduled).sum::<usize>(),
            0
        );
    }
}
