// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded data-program grid search with train/validation/held-out task families.
//!
//! This is an interpretable parameter search over TransmissionProgram values,
//! not LLM-guided evolutionary search and not a reproduction of Colas et al.
//! (2026). All candidates compile to the same bounded Rust program IR.
//!
//! The simulator uses domain-separated graph/discovery/schedule/acquisition RNG
//! streams. Same master seeds reuse the exact graph and subsystem seed schedule;
//! endogenous candidate sets still prevent perfect per-action random pairing.

use civbench_social_learning_core::adaptive_transmission::{
    simulate, simulate_with_program, Config, InnovationTask, TaskFamily, TransmissionPolicy,
    TransmissionProgram, TransmissionScoreWeights,
};

const TRAIN_REPLICATES: u64 = 16;
const VALIDATION_REPLICATES: u64 = 12;
const TEST_REPLICATES: u64 = 64;
const TOP_K: usize = 8;

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
        let standard_deviation = (centered / (n - 1.0)).sqrt();
        Some(1.96 * standard_deviation / n.sqrt())
    }
}

#[derive(Default)]
struct OutcomeStats {
    coverage: OnlineStats,
    max_depth: OnlineStats,
    messages: OnlineStats,
    individual_discoveries: OnlineStats,
    social_acquisitions: OnlineStats,
}

impl OutcomeStats {
    fn observe_run(&mut self, run: &civbench_social_learning_core::adaptive_transmission::Run) {
        let metrics = run.final_metrics().expect("run includes initial and final metrics");
        self.coverage.push(metrics.result_coverage());
        self.max_depth.push(f64::from(metrics.max_discovered_depth));
        self.messages.push(
            run.history.iter().map(|step| step.transmissions_scheduled).sum::<usize>() as f64,
        );
        self.individual_discoveries.push(
            run.history.iter().map(|step| step.individual_recipe_discoveries).sum::<usize>() as f64,
        );
        self.social_acquisitions.push(
            run.history.iter().map(|step| step.successful_recipe_acquisitions).sum::<usize>() as f64,
        );
    }
}

struct Candidate {
    program: TransmissionProgram,
    training_mean: f64,
    validation_mean: Option<f64>,
}

fn base_config() -> Config {
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
        policy: TransmissionPolicy::StateAwareFrontier,
    }
}

fn weight_presets() -> [(&'static str, TransmissionScoreWeights); 6] {
    [
        ("frontier", TransmissionScoreWeights {
            immediate_executability: 100_000,
            prerequisite_match: 10_000,
            collective_novelty: 5_000,
            downstream_unlocks: 1_000,
            sender_knowledge: 1,
        }),
        ("execute_first", TransmissionScoreWeights {
            immediate_executability: 150_000,
            prerequisite_match: 0,
            collective_novelty: 0,
            downstream_unlocks: 0,
            sender_knowledge: 0,
        }),
        ("prerequisite_first", TransmissionScoreWeights {
            immediate_executability: 0,
            prerequisite_match: 30_000,
            collective_novelty: 0,
            downstream_unlocks: 0,
            sender_knowledge: 0,
        }),
        ("novelty_first", TransmissionScoreWeights {
            immediate_executability: 0,
            prerequisite_match: 0,
            collective_novelty: 30_000,
            downstream_unlocks: 0,
            sender_knowledge: 0,
        }),
        ("unlock_first", TransmissionScoreWeights {
            immediate_executability: 0,
            prerequisite_match: 0,
            collective_novelty: 0,
            downstream_unlocks: 10_000,
            sender_knowledge: 0,
        }),
        ("balanced", TransmissionScoreWeights {
            immediate_executability: 60_000,
            prerequisite_match: 15_000,
            collective_novelty: 8_000,
            downstream_unlocks: 4_000,
            sender_knowledge: 1,
        }),
    ]
}

fn candidate_programs(config: Config) -> Vec<TransmissionProgram> {
    let thresholds = [0_u16, 500, 1_000];
    let novelty_filters = [false, true];
    let sender_caps = [1_usize, 2, 4];

    let mut programs = Vec::new();
    for (preset_name, weights) in weight_presets() {
        for threshold in thresholds {
            for require_novelty in novelty_filters {
                for sender_cap in sender_caps {
                    programs.push(TransmissionProgram {
                        program_id: format!(
                            "{preset_name}-prereq{threshold}-novel{}-sender{sender_cap}",
                            u8::from(require_novelty)
                        ),
                        version: 1,
                        minimum_prerequisite_match_milli: threshold,
                        require_collective_novelty: require_novelty,
                        max_messages_per_step: config.transmission_budget_per_step,
                        max_messages_per_sender: sender_cap,
                        weights,
                    });
                }
            }
        }
    }
    programs
}

fn evaluate_program(
    program: &TransmissionProgram,
    families: &[TaskFamily],
    seed_start: u64,
    replicates_per_family: u64,
) -> OutcomeStats {
    let config = base_config();
    let mut stats = OutcomeStats::default();
    for family in families {
        let task = InnovationTask::family_task(*family);
        for offset in 0..replicates_per_family {
            let seed = seed_start + offset;
            let run = simulate_with_program(&task, config, seed, program)
                .expect("predeclared program and task must validate");
            stats.observe_run(&run);
        }
    }
    stats
}

fn print_row(
    stage: &str,
    family: &str,
    program_id: &str,
    stats: &OutcomeStats,
    paired_delta: Option<f64>,
) {
    println!(
        "{},{},{},{},{},{},{},{},{},{}",
        stage,
        family,
        program_id,
        stats.coverage.n,
        fmt(stats.coverage.mean()),
        fmt(stats.coverage.ci95_half_width()),
        fmt(stats.max_depth.mean()),
        fmt(stats.messages.mean()),
        fmt(stats.individual_discoveries.mean()),
        fmt(paired_delta),
    );
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn main() {
    let config = base_config();
    let candidates = candidate_programs(config);
    assert_eq!(candidates.len(), 108);

    println!(
        "stage,task_family,program_id,replicates,coverage_mean,coverage_ci95,max_depth_mean,messages_per_run,individual_discoveries_per_run,paired_coverage_delta"
    );

    // Training selection uses Layered and Branching only, with a locked seed range.
    let training_families = [TaskFamily::Layered, TaskFamily::Branching];
    let mut evaluated: Vec<Candidate> = candidates.into_iter().map(|program| {
        let stats = evaluate_program(&program, &training_families, 0, TRAIN_REPLICATES);
        let mean = stats.coverage.mean().unwrap_or(0.0);
        print_row("train", "layered+branching", &program.program_id, &stats, None);
        Candidate {
            program,
            training_mean: mean,
            validation_mean: None,
        }
    }).collect();

    evaluated.sort_by(|left, right| {
        right.training_mean.total_cmp(&left.training_mean)
            .then_with(|| left.program.program_id.cmp(&right.program.program_id))
    });
    evaluated.truncate(TOP_K);

    // Validation uses a disjoint seed range and the same training task families.
    for candidate in &mut evaluated {
        let stats = evaluate_program(
            &candidate.program,
            &training_families,
            1_000,
            VALIDATION_REPLICATES,
        );
        candidate.validation_mean = stats.coverage.mean();
        print_row(
            "validation",
            "layered+branching",
            &candidate.program.program_id,
            &stats,
            None,
        );
    }

    evaluated.sort_by(|left, right| {
        right.validation_mean.unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&left.validation_mean.unwrap_or(f64::NEG_INFINITY))
            .then_with(|| right.training_mean.total_cmp(&left.training_mean))
            .then_with(|| left.program.program_id.cmp(&right.program.program_id))
    });
    let selected = &evaluated[0].program;

    // The final test is the held-out Convergent DAG family and a new seed range.
    let test_family = TaskFamily::Convergent;
    let test_task = InnovationTask::family_task(test_family);
    let test_config = base_config();
    let mut selected_stats = OutcomeStats::default();
    let mut baseline_stats: Vec<(TransmissionPolicy, OutcomeStats, OnlineStats)> = [
        TransmissionPolicy::StateAwareFrontier,
        TransmissionPolicy::NoSocial,
        TransmissionPolicy::FixedRing,
        TransmissionPolicy::RandomNetworkContent,
        TransmissionPolicy::StateAwareRandomContent,
        TransmissionPolicy::StateAwareRecipientShuffled,
    ].into_iter().map(|policy| (policy, OutcomeStats::default(), OnlineStats::default())).collect();

    for offset in 0..TEST_REPLICATES {
        let seed = 2_000 + offset;
        let selected_run = simulate_with_program(&test_task, test_config, seed, selected)
            .expect("selected program must validate on the held-out family");
        let selected_coverage = selected_run.final_metrics().unwrap().result_coverage();
        selected_stats.observe_run(&selected_run);

        for (policy, stats, delta) in &mut baseline_stats {
            let run = simulate(
                &test_task,
                Config { policy: *policy, ..test_config },
                seed,
            ).expect("baseline policy must validate on held-out family");
            let coverage = run.final_metrics().unwrap().result_coverage();
            stats.observe_run(&run);
            delta.push(selected_coverage - coverage);
        }
    }

    print_row("test_selected_program", test_family.name(), &selected.program_id, &selected_stats, None);
    for (policy, stats, delta) in &baseline_stats {
        print_row(
            "test_baseline",
            test_family.name(),
            policy.name(),
            stats,
            None,
        );
        println!(
            "test_paired_delta,{},{},{},{},{},{},{},{},{}",
            test_family.name(),
            selected.program_id,
            delta.n,
            fmt(delta.mean()),
            fmt(delta.ci95_half_width()),
            "",
            "",
            "",
            ""
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_grid_has_108_bounded_programs() {
        let programs = candidate_programs(base_config());
        assert_eq!(programs.len(), 108);
        assert!(programs.iter().all(|program| program.validate(base_config()).is_ok()));
    }

    #[test]
    fn training_validation_and_test_seed_ranges_are_disjoint() {
        assert!(0 + TRAIN_REPLICATES <= 1_000);
        assert!(1_000 + VALIDATION_REPLICATES <= 2_000);
        assert!(2_000 + TEST_REPLICATES > 2_000);
    }
}
