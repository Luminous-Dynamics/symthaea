// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-002: falsification experiment for value-estimator robustness.
//!
//! Standalone, dependency-free evaluator/controller fixture. The producer-side
//! policy receives only ProducerInput. EvaluatorTrial separately owns paired
//! task value, actual resource charge, OOD stratum, and critical-tail truth.
//!
//! This is a synthetic mechanism test, not robotics-performance evidence.
//!
//! Run:
//!   rustc --edition=2024 --test tools/vision-dvpc-estimator-robustness.rs -o /tmp/vis-dvpc-002-tests
//!   /tmp/vis-dvpc-002-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-estimator-robustness.rs -o /tmp/vis-dvpc-002
//!   /tmp/vis-dvpc-002 > vis-dvpc-002.tsv

#![deny(unsafe_code)]

use std::collections::BTreeMap;

const TRIALS_PER_SEED: usize = 256;
const SEEDS: [u64; 8] = [11, 23, 37, 41, 59, 71, 83, 97];
const BUDGET_PERCENTAGES: [u64; 4] = [10, 20, 40, 60];
const BASE_TASK_LOSS: f64 = 10.0;
const VALUE_THRESHOLD: f64 = 0.25;
const NEUTRAL_VALUE: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Regime {
    CalibratedIid,
    OverconfidentErrors,
    SignReversalOnOod,
    RareTailHidden,
    RareTailObservableCue,
    CostUnderestimate,
}

impl Regime {
    const ALL: [Self; 6] = [
        Self::CalibratedIid,
        Self::OverconfidentErrors,
        Self::SignReversalOnOod,
        Self::RareTailHidden,
        Self::RareTailObservableCue,
        Self::CostUnderestimate,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::CalibratedIid => "calibrated_iid",
            Self::OverconfidentErrors => "overconfident_errors",
            Self::SignReversalOnOod => "sign_reversal_on_ood",
            Self::RareTailHidden => "rare_tail_hidden",
            Self::RareTailObservableCue => "rare_tail_observable_cue",
            Self::CostUnderestimate => "cost_underestimate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Policy {
    NoEscalation,
    AlwaysIfAffordable,
    PointEstimate,
    ConservativeEstimate,
    BudgetRandom,
    ObservableCueGuard,
}

impl Policy {
    const ALL: [Self; 6] = [
        Self::NoEscalation,
        Self::AlwaysIfAffordable,
        Self::PointEstimate,
        Self::ConservativeEstimate,
        Self::BudgetRandom,
        Self::ObservableCueGuard,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::NoEscalation => "no_escalation",
            Self::AlwaysIfAffordable => "always_if_affordable",
            Self::PointEstimate => "point_estimate",
            Self::ConservativeEstimate => "conservative_estimate",
            Self::BudgetRandom => "budget_constrained_random",
            Self::ObservableCueGuard => "observable_cue_guard",
        }
    }
}

/// Only pre-escalation information visible to an online policy.
/// No paired losses, hidden regime, OOD label, future frame or oracle value.
#[derive(Debug, Clone, PartialEq)]
struct ProducerInput {
    trial_id: u64,
    estimated_net_benefit: f64,
    reported_uncertainty: f64,
    declared_cost_units: Option<u64>,
    /// A deliberately weak but genuinely producer-visible cue from the cheap path.
    /// It is not a ground-truth critical-event label.
    weak_critical_cue: bool,
}

/// Evaluator-private fields are kept outside ProducerInput and never passed to choose().
#[derive(Debug, Clone)]
struct EvaluatorTrial {
    producer: ProducerInput,
    true_dvpc: f64,
    actual_cost_units: u64,
    is_ood: bool,
    is_critical_tail: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    KeepCheap,
    Escalate,
}

#[derive(Debug, Clone)]
struct Commit {
    choice: Choice,
    declared_cost: Option<u64>,
}

#[derive(Debug, Clone)]
struct RunResult {
    regime: Regime,
    policy: Policy,
    budget_percent: u64,
    seed: u64,
    trials: usize,
    escalations: usize,
    beneficial_escalations: usize,
    harmful_escalations: usize,
    missed_beneficial: usize,
    critical_misses: usize,
    ood_trials: usize,
    ood_escalations: usize,
    ood_harmful: usize,
    total_benefit: f64,
    task_loss: f64,
    charged_units: u64,
    budget_units: u64,
    reservation_overruns: usize,
    run_budget_overruns: usize,
    oracle_benefit: f64,
    regret: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SummaryStats {
    mean: f64,
    lower_95: f64,
    upper_95: f64,
}

#[derive(Debug, Clone)]
struct SummaryResult {
    regime: Regime,
    policy: Policy,
    budget_percent: u64,
    seeds: usize,
    mean_task_loss: SummaryStats,
    mean_total_benefit: SummaryStats,
    mean_regret: SummaryStats,
    mean_harmful_escalations: SummaryStats,
    mean_missed_beneficial: SummaryStats,
    mean_charged_units: SummaryStats,
    budget_violation_runs: usize,
    paired_loss_delta_vs_random: SummaryStats,
}

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn symmetric(&mut self, magnitude: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * magnitude
    }

    fn integer_inclusive(&mut self, low: u64, high: u64) -> u64 {
        low + self.next_u64() % (high - low + 1)
    }
}

fn base_value(rng: &mut SplitMix64) -> f64 {
    let draw = rng.unit();
    if draw < 0.28 {
        -2.0 - rng.unit()
    } else if draw < 0.42 {
        -0.2 + rng.unit() * 0.3
    } else if draw < 0.82 {
        0.5 + rng.unit() * 1.5
    } else {
        2.0 + rng.unit() * 2.0
    }
}

/// Generate paired evaluator and producer records from one deterministic latent world.
/// Regime/holdout/critical truth is not copied into the producer input.
fn make_trials(regime: Regime, seed: u64) -> Vec<EvaluatorTrial> {
    let mut rng = SplitMix64::new(seed ^ 0xD7A1_CE02_4F19_B385);
    let mut trials = Vec::with_capacity(TRIALS_PER_SEED);

    for index in 0..TRIALS_PER_SEED {
        let trial_id = index as u64 + 1;
        let actual_cost_units = rng.integer_inclusive(2, 5);
        let mut true_dvpc = base_value(&mut rng);
        let is_ood = regime == Regime::SignReversalOnOod && rng.unit() < 0.35;
        let is_critical_tail = matches!(
            regime,
            Regime::RareTailHidden | Regime::RareTailObservableCue
        ) && rng.unit() < 0.04;

        let false_cue_rate = if regime == Regime::RareTailObservableCue {
            0.025
        } else {
            0.01
        };
        let mut weak_critical_cue = rng.unit() < false_cue_rate;
        let mut estimated_net_benefit;
        let mut reported_uncertainty;
        let mut declared_cost_units = Some(actual_cost_units);

        if is_critical_tail {
            true_dvpc = 6.0;
            weak_critical_cue = regime == Regime::RareTailObservableCue;
            // The value estimator misses a high-value tail even when a weak cue exists.
            estimated_net_benefit = -0.15;
            reported_uncertainty = 0.05;
        } else {
            match regime {
                Regime::CalibratedIid | Regime::RareTailHidden | Regime::RareTailObservableCue => {
                    estimated_net_benefit = true_dvpc + rng.symmetric(0.85);
                    reported_uncertainty = 0.75;
                }
                Regime::OverconfidentErrors => {
                    if true_dvpc < 0.0 && rng.unit() < 0.45 {
                        estimated_net_benefit = true_dvpc.abs() + 0.7 + rng.symmetric(0.15);
                    } else {
                        estimated_net_benefit = true_dvpc + rng.symmetric(1.25);
                    }
                    reported_uncertainty = 0.05;
                }
                Regime::SignReversalOnOod => {
                    estimated_net_benefit = if is_ood {
                        -true_dvpc + rng.symmetric(0.25)
                    } else {
                        true_dvpc + rng.symmetric(0.60)
                    };
                    reported_uncertainty = 0.20;
                }
                Regime::CostUnderestimate => {
                    estimated_net_benefit = true_dvpc + rng.symmetric(0.70);
                    reported_uncertainty = 0.60;
                    if rng.unit() < 0.75 {
                        declared_cost_units = Some(actual_cost_units.saturating_sub(1).max(1));
                    }
                    if index % 31 == 0 {
                        declared_cost_units = None;
                    }
                }
            }
        }

        if regime == Regime::CostUnderestimate && !is_critical_tail {
            // The intervention is applied irrespective of the value estimator.
            if rng.unit() < 0.75 {
                declared_cost_units = Some(actual_cost_units.saturating_sub(1).max(1));
            }
        }

        // Keep model outputs finite by construction; the evaluator still validates below.
        if !estimated_net_benefit.is_finite() {
            estimated_net_benefit = 0.0;
        }
        if !reported_uncertainty.is_finite() || reported_uncertainty < 0.0 {
            reported_uncertainty = 0.0;
        }

        trials.push(EvaluatorTrial {
            producer: ProducerInput {
                trial_id,
                estimated_net_benefit,
                reported_uncertainty,
                declared_cost_units,
                weak_critical_cue,
            },
            true_dvpc,
            actual_cost_units,
            is_ood,
            is_critical_tail,
        });
    }

    trials
}

/// Producer-side policy. Deliberately accepts ProducerInput only, not EvaluatorTrial.
fn choose(policy: Policy, input: &ProducerInput, deterministic_seed: u64) -> Choice {
    let escalate = match policy {
        Policy::NoEscalation => false,
        Policy::AlwaysIfAffordable => true,
        Policy::PointEstimate => input.estimated_net_benefit > VALUE_THRESHOLD,
        Policy::ConservativeEstimate => {
            input.estimated_net_benefit - 1.25 * input.reported_uncertainty > VALUE_THRESHOLD
        }
        Policy::BudgetRandom => {
            let x = mix64(deterministic_seed ^ input.trial_id.wrapping_mul(0x9E3779B97F4A7C15));
            x % 100 < 50
        }
        Policy::ObservableCueGuard => {
            input.estimated_net_benefit > VALUE_THRESHOLD || input.weak_critical_cue
        }
    };
    if escalate {
        Choice::Escalate
    } else {
        Choice::KeepCheap
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn budget_capacity(trials: &[EvaluatorTrial], percentage: u64) -> u64 {
    let full_cost = trials
        .iter()
        .map(|trial| trial.actual_cost_units as u128)
        .sum::<u128>();
    ((full_cost * percentage as u128) / 100).min(u64::MAX as u128) as u64
}

/// Exact 0/1 knapsack oracle. It uses hidden paired values and actual costs and must
/// remain evaluator-only; it is not an eligible producer policy or deployable baseline.
fn evaluator_oracle_benefit(trials: &[EvaluatorTrial], budget: u64) -> f64 {
    let cap = budget as usize;
    let mut best = vec![0.0_f64; cap + 1];
    for trial in trials {
        if trial.true_dvpc <= NEUTRAL_VALUE || trial.actual_cost_units as usize > cap {
            continue;
        }
        let cost = trial.actual_cost_units as usize;
        for used in (cost..=cap).rev() {
            let candidate = best[used - cost] + trial.true_dvpc;
            if candidate > best[used] {
                best[used] = candidate;
            }
        }
    }
    best.into_iter().fold(0.0_f64, f64::max)
}

fn run_policy(
    regime: Regime,
    policy: Policy,
    budget_percent: u64,
    seed: u64,
    trials: &[EvaluatorTrial],
) -> RunResult {
    let budget_units = budget_capacity(trials, budget_percent);
    let oracle_benefit = evaluator_oracle_benefit(trials, budget_units);
    let mut charged_units = 0_u64;
    let mut escalations = 0;
    let mut beneficial_escalations = 0;
    let mut harmful_escalations = 0;
    let mut missed_beneficial = 0;
    let mut critical_misses = 0;
    let mut ood_trials = 0;
    let mut ood_escalations = 0;
    let mut ood_harmful = 0;
    let mut total_benefit = 0.0_f64;
    let mut reservation_overruns = 0;
    let mut run_budget_overruns = 0;
    let mut stop_after_accounting_failure = false;

    for trial in trials {
        if trial.is_ood {
            ood_trials += 1;
        }
        let candidate = choose(policy, &trial.producer, seed ^ 0xA11C_A710);
        let declared_cost = trial.producer.declared_cost_units;
        let has_budget_for_reservation = match declared_cost {
            Some(cost) if cost > 0 && !stop_after_accounting_failure => {
                charged_units
                    .checked_add(cost)
                    .is_some_and(|next| next <= budget_units)
            }
            _ => false,
        };
        let choice = if candidate == Choice::Escalate && has_budget_for_reservation {
            Choice::Escalate
        } else {
            Choice::KeepCheap
        };

        if choice == Choice::Escalate {
            escalations += 1;
            total_benefit += trial.true_dvpc;
            charged_units = charged_units.saturating_add(trial.actual_cost_units);
            if trial.true_dvpc > NEUTRAL_VALUE {
                beneficial_escalations += 1;
            } else if trial.true_dvpc < -NEUTRAL_VALUE {
                harmful_escalations += 1;
            }
            if trial.is_ood {
                ood_escalations += 1;
                if trial.true_dvpc < -NEUTRAL_VALUE {
                    ood_harmful += 1;
                }
            }
            if let Some(reserved) = declared_cost {
                if trial.actual_cost_units > reserved {
                    reservation_overruns += 1;
                    // The conservative comparison policy halts after an unbudgeted overrun.
                    stop_after_accounting_failure = true;
                }
            } else {
                // Unknown cost should never execute. Guard in case policy behavior changes.
                stop_after_accounting_failure = true;
            }
            if charged_units > budget_units {
                run_budget_overruns += 1;
                stop_after_accounting_failure = true;
            }
        } else if trial.true_dvpc > NEUTRAL_VALUE {
            missed_beneficial += 1;
            if trial.is_critical_tail {
                critical_misses += 1;
            }
        }
    }

    let task_loss = BASE_TASK_LOSS * trials.len() as f64 - total_benefit;
    RunResult {
        regime,
        policy,
        budget_percent,
        seed,
        trials: trials.len(),
        escalations,
        beneficial_escalations,
        harmful_escalations,
        missed_beneficial,
        critical_misses,
        ood_trials,
        ood_escalations,
        ood_harmful,
        total_benefit,
        task_loss,
        charged_units,
        budget_units,
        reservation_overruns,
        run_budget_overruns,
        oracle_benefit,
        regret: oracle_benefit - total_benefit,
    }
}

fn mean_ci(values: &[f64]) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats { mean: 0.0, lower_95: 0.0, upper_95: 0.0 };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return SummaryStats { mean, lower_95: mean, upper_95: mean };
    }
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = 1.96 * (variance / values.len() as f64).sqrt();
    SummaryStats { mean, lower_95: mean - half_width, upper_95: mean + half_width }
}

fn summarize(results: &[RunResult]) -> Vec<SummaryResult> {
    let mut output = Vec::new();
    for regime in Regime::ALL {
        for budget_percent in BUDGET_PERCENTAGES {
            let random_by_seed: BTreeMap<u64, f64> = results
                .iter()
                .filter(|row| row.regime == regime
                    && row.budget_percent == budget_percent
                    && row.policy == Policy::BudgetRandom)
                .map(|row| (row.seed, row.task_loss))
                .collect();

            for policy in Policy::ALL {
                let rows: Vec<&RunResult> = results
                    .iter()
                    .filter(|row| row.regime == regime
                        && row.budget_percent == budget_percent
                        && row.policy == policy)
                    .collect();
                let task_loss: Vec<f64> = rows.iter().map(|row| row.task_loss).collect();
                let benefit: Vec<f64> = rows.iter().map(|row| row.total_benefit).collect();
                let regret: Vec<f64> = rows.iter().map(|row| row.regret).collect();
                let harmful: Vec<f64> = rows.iter().map(|row| row.harmful_escalations as f64).collect();
                let missed: Vec<f64> = rows.iter().map(|row| row.missed_beneficial as f64).collect();
                let charged: Vec<f64> = rows.iter().map(|row| row.charged_units as f64).collect();
                let paired_delta: Vec<f64> = rows.iter().filter_map(|row| {
                    random_by_seed.get(&row.seed).map(|random_loss| row.task_loss - random_loss)
                }).collect();
                output.push(SummaryResult {
                    regime,
                    policy,
                    budget_percent,
                    seeds: rows.len(),
                    mean_task_loss: mean_ci(&task_loss),
                    mean_total_benefit: mean_ci(&benefit),
                    mean_regret: mean_ci(&regret),
                    mean_harmful_escalations: mean_ci(&harmful),
                    mean_missed_beneficial: mean_ci(&missed),
                    mean_charged_units: mean_ci(&charged),
                    budget_violation_runs: rows.iter().filter(|row| row.run_budget_overruns > 0).count(),
                    paired_loss_delta_vs_random: mean_ci(&paired_delta),
                });
            }
        }
    }
    output
}

fn fmt_stats(stats: SummaryStats) -> String {
    format!("{:.6}\t{:.6}\t{:.6}", stats.mean, stats.lower_95, stats.upper_95)
}

fn render_report() -> String {
    let mut results = Vec::new();
    for regime in Regime::ALL {
        for seed in SEEDS {
            let trials = make_trials(regime, seed);
            for budget_percent in BUDGET_PERCENTAGES {
                for policy in Policy::ALL {
                    results.push(run_policy(regime, policy, budget_percent, seed, &trials));
                }
            }
        }
    }

    let mut report = String::new();
    report.push_str("META\tformat\tVIS-DVPC-002-v0.1\n");
    report.push_str("META\tclaim_class\tsynthetic_estimator_robustness_diagnostic_not_native_robotics_or_sota\n");
    report.push_str("META\tproducer_boundary\tpre_escalation_estimate_uncertainty_declared_cost_and_weak_observable_cue_only\n");
    report.push_str("META\toracle_boundary\tevaluator_only_exact_knapsack_from_hidden_paired_values_and_actual_costs\n");
    report.push_str("META\tresource_units\tabstract_integer_cost_units_not_latency_flops_watts_or_energy\n");
    report.push_str("META\tseed_count\t8\n");
    report.push_str("META\ttrials_per_regime_seed\t256\n");
    report.push_str("META\tinterval\tnormal_approximation_1.96_sd_over_sqrt_n_descriptive_not_confirmatory\n");
    report.push_str("RUN_HEADER\tregime\tpolicy\tbudget_percent\tseed\ttrial_count\tescalations\tbeneficial_escalations\tharmful_escalations\tmissed_beneficial\tcritical_misses\tood_trials\tood_escalations\tood_harmful\ttotal_benefit\ttask_loss\tcharged_units\tbudget_units\treservation_overruns\trun_budget_overruns\toracle_benefit\tregret\n");
    for row in &results {
        report.push_str(&format!(
            "RUN\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\n",
            row.regime.label(), row.policy.label(), row.budget_percent, row.seed,
            row.trials, row.escalations, row.beneficial_escalations, row.harmful_escalations,
            row.missed_beneficial, row.critical_misses, row.ood_trials, row.ood_escalations,
            row.ood_harmful, row.total_benefit, row.task_loss, row.charged_units,
            row.budget_units, row.reservation_overruns, row.run_budget_overruns,
            row.oracle_benefit, row.regret
        ));
    }

    report.push_str("SUMMARY_HEADER\tregime\tpolicy\tbudget_percent\tseed_count\tmean_task_loss\ttask_loss_ci95_low\ttask_loss_ci95_high\tmean_total_benefit\tbenefit_ci95_low\tbenefit_ci95_high\tmean_regret\tregret_ci95_low\tregret_ci95_high\tmean_harmful_escalations\tharmful_ci95_low\tharmful_ci95_high\tmean_missed_beneficial\tmissed_ci95_low\tmissed_ci95_high\tmean_charged_units\tcharged_ci95_low\tcharged_ci95_high\tbudget_violation_runs\tpaired_loss_delta_vs_random\tdelta_ci95_low\tdelta_ci95_high\n");
    for row in summarize(&results) {
        report.push_str(&format!(
            "SUMMARY\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.regime.label(), row.policy.label(), row.budget_percent, row.seeds,
            fmt_stats(row.mean_task_loss), fmt_stats(row.mean_total_benefit),
            fmt_stats(row.mean_regret), fmt_stats(row.mean_harmful_escalations),
            fmt_stats(row.mean_missed_beneficial), fmt_stats(row.mean_charged_units),
            row.budget_violation_runs, fmt_stats(row.paired_loss_delta_vs_random)
        ));
    }
    report
}

fn main() {
    let report = render_report();
    print!("{report}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_policy_signature_uses_only_producer_input() {
        // Compile-time contract is represented by choose(Policy, &ProducerInput, seed).
        // Paired losses, hidden regime, OOD label and actual costs live only in EvaluatorTrial.
        let input = ProducerInput {
            trial_id: 7,
            estimated_net_benefit: 1.0,
            reported_uncertainty: 0.1,
            declared_cost_units: Some(2),
            weak_critical_cue: false,
        };
        assert_eq!(choose(Policy::PointEstimate, &input, 13), Choice::Escalate);
        assert_eq!(choose(Policy::NoEscalation, &input, 13), Choice::KeepCheap);
    }

    #[test]
    fn conservative_policy_rejects_uncertainty_that_erases_value() {
        let input = ProducerInput {
            trial_id: 1,
            estimated_net_benefit: 0.8,
            reported_uncertainty: 0.6,
            declared_cost_units: Some(2),
            weak_critical_cue: false,
        };
        assert_eq!(choose(Policy::PointEstimate, &input, 0), Choice::Escalate);
        assert_eq!(choose(Policy::ConservativeEstimate, &input, 0), Choice::KeepCheap);
    }

    #[test]
    fn cue_guard_uses_a_producer_visible_signal_not_hidden_tail_label() {
        let input = ProducerInput {
            trial_id: 9,
            estimated_net_benefit: -0.15,
            reported_uncertainty: 0.05,
            declared_cost_units: Some(2),
            weak_critical_cue: true,
        };
        assert_eq!(choose(Policy::PointEstimate, &input, 0), Choice::KeepCheap);
        assert_eq!(choose(Policy::ObservableCueGuard, &input, 0), Choice::Escalate);
    }

    #[test]
    fn unknown_cost_never_passes_reservation_gate() {
        let trials = vec![EvaluatorTrial {
            producer: ProducerInput {
                trial_id: 1,
                estimated_net_benefit: 5.0,
                reported_uncertainty: 0.1,
                declared_cost_units: None,
                weak_critical_cue: true,
            },
            true_dvpc: 6.0,
            actual_cost_units: 2,
            is_ood: false,
            is_critical_tail: true,
        }];
        let row = run_policy(Regime::RareTailObservableCue, Policy::ObservableCueGuard, 60, 11, &trials);
        assert_eq!(row.escalations, 0);
    }

    #[test]
    fn cost_underestimate_is_detected_and_stops_future_escalations() {
        let trials = vec![
            EvaluatorTrial {
                producer: ProducerInput {
                    trial_id: 1,
                    estimated_net_benefit: 3.0,
                    reported_uncertainty: 0.1,
                    declared_cost_units: Some(1),
                    weak_critical_cue: false,
                },
                true_dvpc: 2.0,
                actual_cost_units: 4,
                is_ood: false,
                is_critical_tail: false,
            },
            EvaluatorTrial {
                producer: ProducerInput {
                    trial_id: 2,
                    estimated_net_benefit: 3.0,
                    reported_uncertainty: 0.1,
                    declared_cost_units: Some(2),
                    weak_critical_cue: false,
                },
                true_dvpc: 2.0,
                actual_cost_units: 2,
                is_ood: false,
                is_critical_tail: false,
            },
        ];
        let row = run_policy(Regime::CostUnderestimate, Policy::AlwaysIfAffordable, 60, 1, &trials);
        assert_eq!(row.escalations, 1);
        assert_eq!(row.reservation_overruns, 1);
        assert_eq!(row.run_budget_overruns, 1);
    }

    #[test]
    fn hindsight_oracle_respects_the_actual_cost_budget() {
        let trials = vec![
            EvaluatorTrial {
                producer: ProducerInput {
                    trial_id: 1, estimated_net_benefit: 0.0, reported_uncertainty: 0.0,
                    declared_cost_units: Some(2), weak_critical_cue: false,
                },
                true_dvpc: 5.0, actual_cost_units: 2, is_ood: false, is_critical_tail: false,
            },
            EvaluatorTrial {
                producer: ProducerInput {
                    trial_id: 2, estimated_net_benefit: 0.0, reported_uncertainty: 0.0,
                    declared_cost_units: Some(2), weak_critical_cue: false,
                },
                true_dvpc: 4.0, actual_cost_units: 2, is_ood: false, is_critical_tail: false,
            },
            EvaluatorTrial {
                producer: ProducerInput {
                    trial_id: 3, estimated_net_benefit: 0.0, reported_uncertainty: 0.0,
                    declared_cost_units: Some(2), weak_critical_cue: false,
                },
                true_dvpc: -9.0, actual_cost_units: 2, is_ood: false, is_critical_tail: false,
            },
        ];
        assert_eq!(evaluator_oracle_benefit(&trials, 2), 5.0);
        assert_eq!(evaluator_oracle_benefit(&trials, 4), 9.0);
    }

    #[test]
    fn generated_regimes_include_separate_ood_and_critical_cases() {
        let sign_shift = make_trials(Regime::SignReversalOnOod, 11);
        assert!(sign_shift.iter().any(|trial| trial.is_ood));
        assert!(sign_shift.iter().any(|trial| !trial.is_ood));
        let hidden_tail = make_trials(Regime::RareTailHidden, 11);
        let visible_tail = make_trials(Regime::RareTailObservableCue, 11);
        assert!(hidden_tail.iter().any(|trial| trial.is_critical_tail));
        assert!(visible_tail.iter().any(|trial| trial.is_critical_tail && trial.producer.weak_critical_cue));
    }

    #[test]
    fn report_is_deterministic_and_separates_runs_from_summaries() {
        let first = render_report();
        let second = render_report();
        assert_eq!(first, second);
        assert!(first.contains("VIS-DVPC-002-v0.1"));
        assert!(first.contains("sign_reversal_on_ood"));
        assert!(first.contains("rare_tail_observable_cue"));
        assert!(first.contains("paired_loss_delta_vs_random"));
        assert!(!first.contains("future_frame_payload"));
        assert_eq!(first.lines().filter(|line| line.starts_with("RUN\t")).count(), 6 * 8 * 4 * 6);
        assert_eq!(first.lines().filter(|line| line.starts_with("SUMMARY\t")).count(), 6 * 4 * 6);
    }
}
