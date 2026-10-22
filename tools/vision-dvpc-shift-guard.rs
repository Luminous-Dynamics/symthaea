// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-003: observable shift-alarm admission guard for perception compute.
//!
//! This is a deterministic synthetic mechanism benchmark, not robotics evidence.
//! Producer-side selection only accepts ProducerInput. EvaluatorTrial carries hidden
//! shift truth and downstream value and is never passed to a policy.
//!
//! Run:
//!   rustc --edition=2024 --test tools/vision-dvpc-shift-guard.rs -o /tmp/vis-dvpc-003-tests
//!   /tmp/vis-dvpc-003-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-shift-guard.rs -o /tmp/vis-dvpc-003
//!   /tmp/vis-dvpc-003 > vis-dvpc-003.tsv

#![deny(unsafe_code)]

const TRIALS_PER_SEED: usize = 256;
const SEEDS: [u64; 8] = [11, 23, 37, 41, 59, 71, 83, 97];
const BUDGET_PERCENTAGES: [u64; 3] = [20, 40, 60];
const FULL_FIDELITY_REFERENCE_COST: u64 = 4;
const BASE_TASK_LOSS: f64 = 10.0;
const VALUE_THRESHOLD: f64 = 0.25;
const NEUTRAL_VALUE: f64 = 0.10;
const LCB_MULTIPLIER: f64 = 1.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Regime {
    StableCalibrated,
    OverconfidentErrors,
    SignReversalShift,
}

impl Regime {
    const ALL: [Self; 3] = [
        Self::StableCalibrated,
        Self::OverconfidentErrors,
        Self::SignReversalShift,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::StableCalibrated => "stable_calibrated",
            Self::OverconfidentErrors => "overconfident_errors",
            Self::SignReversalShift => "sign_reversal_shift",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Policy {
    NoEscalation,
    PointEstimate,
    ConservativeLcb,
    ShiftGuardedLcb,
    BudgetRandom,
}

impl Policy {
    const ALL: [Self; 5] = [
        Self::NoEscalation,
        Self::PointEstimate,
        Self::ConservativeLcb,
        Self::ShiftGuardedLcb,
        Self::BudgetRandom,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::NoEscalation => "no_escalation",
            Self::PointEstimate => "point_estimate",
            Self::ConservativeLcb => "conservative_lcb",
            Self::ShiftGuardedLcb => "shift_guarded_lcb",
            Self::BudgetRandom => "budget_constrained_random",
        }
    }
}

/// Only information that the online selector is permitted to observe.
/// In particular, there is no shift/OOD label or true downstream value.
#[derive(Debug, Clone)]
struct ProducerInput {
    trial_id: u64,
    estimated_net_benefit: f64,
    reported_uncertainty: f64,
    declared_cost_units: Option<u64>,
    observable_shift_alarm: bool,
}

/// Private evaluator record. Never pass this type to choose().
#[derive(Debug, Clone)]
struct EvaluatorTrial {
    producer: ProducerInput,
    true_dvpc: f64,
    actual_cost_units: u64,
    is_shifted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    KeepCheap,
    Escalate,
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
    shifted_trials: usize,
    shifted_escalations: usize,
    shifted_harmful: usize,
    shift_alarms: usize,
    false_alarms: usize,
    false_alarm_candidate_blocks: usize,
    false_alarm_beneficial_blocks: usize,
    total_benefit: f64,
    task_loss: f64,
    cvar95_incremental_loss: f64,
    charged_units: u64,
    budget_units: u64,
    reservation_overruns: usize,
    run_budget_overruns: usize,
    oracle_benefit: f64,
    regret: Option<f64>,
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

/// The alarm is a noisy observable detector output: 80% sensitivity and 10%
/// false-positive rate by construction in the shifted regime. It is not truth.
fn make_trials(regime: Regime, seed: u64) -> Vec<EvaluatorTrial> {
    let mut rng = SplitMix64::new(seed ^ 0xD7A1_CE02_4F19_B385);
    let mut trials = Vec::with_capacity(TRIALS_PER_SEED);

    for index in 0..TRIALS_PER_SEED {
        let trial_id = index as u64 + 1;
        let actual_cost_units = 2 + (rng.next_u64() % 4);
        let is_shifted = regime == Regime::SignReversalShift && rng.unit() < 0.35;
        let true_dvpc = base_value(&mut rng);

        let (estimated_net_benefit, reported_uncertainty) = match regime {
            Regime::StableCalibrated => (true_dvpc + rng.symmetric(0.75), 0.75),
            Regime::OverconfidentErrors => {
                if true_dvpc < 0.0 && rng.unit() < 0.45 {
                    (true_dvpc.abs() + 0.7 + rng.symmetric(0.15), 0.05)
                } else {
                    (true_dvpc + rng.symmetric(1.25), 0.05)
                }
            }
            Regime::SignReversalShift => {
                let estimate = if is_shifted {
                    -true_dvpc + rng.symmetric(0.15)
                } else {
                    true_dvpc + rng.symmetric(0.15)
                };
                (estimate, 0.20)
            }
        };

        let observable_shift_alarm = if is_shifted {
            rng.unit() < 0.80
        } else {
            rng.unit() < 0.10
        };
        // Unknown-cost and underestimated-cost controls are observable to the
        // producer only through the declared cost. The evaluator retains actual cost.
        let declared_cost_units = if index % 47 == 0 {
            None
        } else if index % 29 == 0 {
            Some(actual_cost_units.saturating_sub(1).max(1))
        } else {
            Some(actual_cost_units)
        };

        trials.push(EvaluatorTrial {
            producer: ProducerInput {
                trial_id,
                estimated_net_benefit,
                reported_uncertainty,
                declared_cost_units,
                observable_shift_alarm,
            },
            true_dvpc,
            actual_cost_units,
            is_shifted,
        });
    }

    trials
}

/// Producer policy boundary: evaluator truth is not an argument.
fn choose(policy: Policy, input: &ProducerInput, seed: u64) -> Choice {
    let lcb = input.estimated_net_benefit - LCB_MULTIPLIER * input.reported_uncertainty;
    let should_escalate = match policy {
        Policy::NoEscalation => false,
        Policy::PointEstimate => input.estimated_net_benefit > VALUE_THRESHOLD,
        Policy::ConservativeLcb => lcb > VALUE_THRESHOLD,
        Policy::ShiftGuardedLcb => lcb > VALUE_THRESHOLD && !input.observable_shift_alarm,
        Policy::BudgetRandom => {
            mix64(seed ^ input.trial_id.wrapping_mul(0x9E3779B97F4A7C15)) % 100 < 50
        }
    };

    if should_escalate {
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

fn budget_capacity(trials: usize, percentage: u64) -> u64 {
    ((trials as u128)
        .saturating_mul(FULL_FIDELITY_REFERENCE_COST as u128)
        .saturating_mul(percentage as u128)
        / 100)
        .min(u64::MAX as u128) as u64
}

/// Exact small-instance 0/1-knapsack oracle. Evaluator-only and scored only
/// after every producer commitment. It is not an online or deployable policy.
fn evaluator_oracle_benefit(trials: &[EvaluatorTrial], budget: u64) -> f64 {
    let cap = budget as usize;
    let mut best = vec![0.0_f64; cap + 1];
    for trial in trials {
        if trial.true_dvpc <= 0.0 || trial.actual_cost_units as usize > cap {
            continue;
        }
        let cost = trial.actual_cost_units as usize;
        for used in (cost..=cap).rev() {
            best[used] = best[used].max(best[used - cost] + trial.true_dvpc);
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
    let budget_units = budget_capacity(trials.len(), budget_percent);
    let mut charged_units = 0_u64;
    let mut escalations = 0;
    let mut beneficial_escalations = 0;
    let mut harmful_escalations = 0;
    let mut missed_beneficial = 0;
    let mut shifted_trials = 0;
    let mut shifted_escalations = 0;
    let mut shifted_harmful = 0;
    let mut shift_alarms = 0;
    let mut false_alarms = 0;
    let mut false_alarm_candidate_blocks = 0;
    let mut false_alarm_beneficial_blocks = 0;
    let mut total_benefit = 0.0;
    let mut incremental_losses = Vec::with_capacity(trials.len());
    let mut reservation_overruns = 0;
    let mut run_budget_overruns = 0;
    let mut stop_after_accounting_failure = false;

    for trial in trials {
        if trial.is_shifted {
            shifted_trials += 1;
        }
        if trial.producer.observable_shift_alarm {
            shift_alarms += 1;
            if !trial.is_shifted {
                false_alarms += 1;
            }
        }

        let candidate = choose(policy, &trial.producer, seed ^ 0xA11C_A710);
        let reservation = trial.producer.declared_cost_units;
        let can_reserve = match reservation {
            Some(cost) if cost > 0 && !stop_after_accounting_failure => {
                charged_units.checked_add(cost).is_some_and(|next| next <= budget_units)
            }
            _ => false,
        };
        if policy == Policy::ShiftGuardedLcb
            && can_reserve
            && trial.producer.observable_shift_alarm
            && !trial.is_shifted
            && choose(Policy::ConservativeLcb, &trial.producer, seed) == Choice::Escalate
        {
            false_alarm_candidate_blocks += 1;
            if trial.true_dvpc > NEUTRAL_VALUE {
                false_alarm_beneficial_blocks += 1;
            }
        }

        let choice = if candidate == Choice::Escalate && can_reserve {
            Choice::Escalate
        } else {
            Choice::KeepCheap
        };
        incremental_losses.push(if choice == Choice::Escalate {
            -trial.true_dvpc
        } else {
            0.0
        });

        if choice == Choice::Escalate {
            escalations += 1;
            total_benefit += trial.true_dvpc;
            charged_units = charged_units.saturating_add(trial.actual_cost_units);
            if trial.true_dvpc > NEUTRAL_VALUE {
                beneficial_escalations += 1;
            } else if trial.true_dvpc < -NEUTRAL_VALUE {
                harmful_escalations += 1;
            }
            if trial.is_shifted {
                shifted_escalations += 1;
                if trial.true_dvpc < -NEUTRAL_VALUE {
                    shifted_harmful += 1;
                }
            }
            if let Some(reserved) = reservation {
                if trial.actual_cost_units > reserved {
                    reservation_overruns += 1;
                    stop_after_accounting_failure = true;
                }
            } else {
                // Defensive invariant: unknown cost should never execute.
                reservation_overruns += 1;
                stop_after_accounting_failure = true;
            }
            if charged_units > budget_units {
                run_budget_overruns += 1;
                stop_after_accounting_failure = true;
            }
        } else if trial.true_dvpc > NEUTRAL_VALUE {
            missed_beneficial += 1;
        }
    }

    // Score the evaluator-only oracle only after every policy decision is committed.
    let oracle_benefit = evaluator_oracle_benefit(trials, budget_units);
    let task_loss = BASE_TASK_LOSS * trials.len() as f64 - total_benefit;
    let cvar95_incremental_loss = cvar95_upper_tail(&incremental_losses);
    let regret_is_valid = reservation_overruns == 0 && run_budget_overruns == 0;

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
        shifted_trials,
        shifted_escalations,
        shifted_harmful,
        shift_alarms,
        false_alarms,
        false_alarm_candidate_blocks,
        false_alarm_beneficial_blocks,
        total_benefit,
        task_loss,
        cvar95_incremental_loss,
        charged_units,
        budget_units,
        reservation_overruns,
        run_budget_overruns,
        oracle_benefit,
        regret: regret_is_valid.then_some(oracle_benefit - total_benefit),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Mean of the worst ceil(5% * n) per-trial incremental losses (positive means harm).
fn cvar95_upper_tail(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| right.total_cmp(left));
    let tail_count = ((ordered.len() + 19) / 20).max(1);
    ordered.iter().take(tail_count).sum::<f64>() / tail_count as f64
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
    report.push_str("META\tformat\tVIS-DVPC-003-v0.1\n");
    report.push_str("META\tclaim_class\tsynthetic_shift_guard_mechanism_not_robotics_performance_or_sota\n");
    report.push_str("META\tproducer_boundary\testimate_uncertainty_declared_cost_noisy_observable_shift_alarm_only\n");
    report.push_str("META\tevaluator_boundary\thidden_shift_truth_true_dvpc_actual_cost_and_oracle_after_commit\n");
    report.push_str("META\tbudget_reference\t4_fixed_full_fidelity_units_per_trial\n");
    report.push_str("META\talarm_generation\t80_percent_sensitivity_10_percent_false_positive_rate_in_generator\n");
    report.push_str("META\tregret_validity\tINVALID_IF_RESERVATION_OVERRUN_OR_RUN_BUDGET_OVERRUN\n");
    report.push_str("RUN_HEADER\tregime\tpolicy\tbudget_percent\tseed\ttrial_count\tescalations\tbeneficial_escalations\tharmful_escalations\tmissed_beneficial\tshifted_trials\tshifted_escalations\tshifted_harmful\tshift_alarms\tfalse_alarms\tfalse_alarm_candidate_blocks\tfalse_alarm_beneficial_blocks\ttotal_benefit\ttask_loss\tcvar95_incremental_loss\tcharged_units\tbudget_units\treservation_overruns\trun_budget_overruns\toracle_benefit\tregret\n");
    for row in &results {
        report.push_str(&format!(
            "RUN\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{:.6}\t{}\n",
            row.regime.label(), row.policy.label(), row.budget_percent, row.seed, row.trials,
            row.escalations, row.beneficial_escalations, row.harmful_escalations,
            row.missed_beneficial, row.shifted_trials, row.shifted_escalations, row.shifted_harmful,
            row.shift_alarms, row.false_alarms, row.false_alarm_candidate_blocks,
            row.false_alarm_beneficial_blocks, row.total_benefit, row.task_loss,
            row.cvar95_incremental_loss, row.charged_units,
            row.budget_units, row.reservation_overruns, row.run_budget_overruns, row.oracle_benefit,
            row.regret
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "INVALID_BUDGET".to_owned())
        ));
    }

    report.push_str("SUMMARY_HEADER\tregime\tpolicy\tbudget_percent\tseed_count\tmean_task_loss\tmean_cvar95_incremental_loss\tmean_total_benefit\tmean_harmful_escalations\tmean_missed_beneficial\tmean_shifted_harmful\tmean_false_alarm_beneficial_blocks\tmean_charged_units\tbudget_violation_runs\tvalid_regret_seeds\tmean_regret\n");
    for regime in Regime::ALL {
        for budget_percent in BUDGET_PERCENTAGES {
            for policy in Policy::ALL {
                let rows: Vec<&RunResult> = results.iter()
                    .filter(|row| {
                        row.regime == regime
                            && row.budget_percent == budget_percent
                            && row.policy == policy
                    })
                    .collect();
                let loss: Vec<f64> = rows.iter().map(|row| row.task_loss).collect();
                let cvar95: Vec<f64> = rows.iter().map(|row| row.cvar95_incremental_loss).collect();
                let benefit: Vec<f64> = rows.iter().map(|row| row.total_benefit).collect();
                let harmful: Vec<f64> = rows
                    .iter()
                    .map(|row| row.harmful_escalations as f64)
                    .collect();
                let missed: Vec<f64> = rows
                    .iter()
                    .map(|row| row.missed_beneficial as f64)
                    .collect();
                let shifted_harmful: Vec<f64> = rows
                    .iter()
                    .map(|row| row.shifted_harmful as f64)
                    .collect();
                let false_alarm_beneficial: Vec<f64> = rows
                    .iter()
                    .map(|row| row.false_alarm_beneficial_blocks as f64)
                    .collect();
                let charged: Vec<f64> = rows.iter().map(|row| row.charged_units as f64).collect();
                let valid_regret: Vec<f64> = rows.iter().filter_map(|row| row.regret).collect();
                let violations = rows
                    .iter()
                    .filter(|row| row.reservation_overruns > 0 || row.run_budget_overruns > 0)
                    .count();
                report.push_str(&format!(
                    "SUMMARY\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    regime.label(), policy.label(), budget_percent, rows.len(),
                    mean(&loss).map(|v| format!("{v:.6}")).unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&cvar95).map(|v| format!("{v:.6}")).unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&benefit)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&harmful)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&missed)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&shifted_harmful)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&false_alarm_beneficial)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    mean(&charged)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned()),
                    violations, valid_regret.len(),
                    mean(&valid_regret)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "UNKNOWN".to_owned())
                ));
            }
        }
    }
    report
}

fn main() {
    print!("{}", render_report());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(estimate: f64, uncertainty: f64, cost: Option<u64>, alarm: bool) -> ProducerInput {
        ProducerInput {
            trial_id: 1,
            estimated_net_benefit: estimate,
            reported_uncertainty: uncertainty,
            declared_cost_units: cost,
            observable_shift_alarm: alarm,
        }
    }

    #[test]
    fn cvar95_uses_the_worst_five_percent_of_trial_losses() {
        let values: Vec<f64> = (0..19).map(|_| 0.0).chain([10.0]).collect();
        assert_eq!(cvar95_upper_tail(&values), 10.0);
    }

    #[test]
    fn shift_guard_blocks_a_high_lcb_when_alarm_is_on() {
        let producer = input(2.0, 0.1, Some(3), true);
        assert_eq!(choose(Policy::ConservativeLcb, &producer, 7), Choice::Escalate);
        assert_eq!(choose(Policy::ShiftGuardedLcb, &producer, 7), Choice::KeepCheap);
    }

    #[test]
    fn no_shift_alarm_does_not_change_lcb_policy() {
        let producer = input(2.0, 0.1, Some(3), false);
        assert_eq!(choose(Policy::ConservativeLcb, &producer, 7), Choice::Escalate);
        assert_eq!(choose(Policy::ShiftGuardedLcb, &producer, 7), Choice::Escalate);
    }

    #[test]
    fn unknown_cost_fails_closed_for_every_policy() {
        let trial = EvaluatorTrial {
            producer: input(3.0, 0.0, None, false),
            true_dvpc: 3.0,
            actual_cost_units: 3,
            is_shifted: false,
        };
        for policy in Policy::ALL {
            let row = run_policy(Regime::StableCalibrated, policy, 60, 11, &[trial.clone()]);
            assert_eq!(row.escalations, 0, "policy {}", policy.label());
            assert_eq!(row.charged_units, 0, "policy {}", policy.label());
        }
    }

    #[test]
    fn regret_is_invalidated_by_cost_reservation_overrun() {
        let trial = EvaluatorTrial {
            producer: input(3.0, 0.0, Some(1), false),
            true_dvpc: 3.0,
            actual_cost_units: 3,
            is_shifted: false,
        };
        let row = run_policy(Regime::StableCalibrated, Policy::PointEstimate, 60, 11, &[trial]);
        assert_eq!(row.reservation_overruns, 1);
        assert!(row.regret.is_none());
    }

    #[test]
    fn output_replays_byte_identically_and_has_expected_rows() {
        let first = render_report();
        let second = render_report();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("RUN\t")).count(), 360);
        assert_eq!(first.lines().filter(|line| line.starts_with("SUMMARY\t")).count(), 45);
        assert!(!first.contains("is_shifted\t"));
        assert!(!first.contains("true_dvpc\t"));
    }
}
