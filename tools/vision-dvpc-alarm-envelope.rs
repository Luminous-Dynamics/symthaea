// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-004: map the alarm-gated DVPC operating envelope.
//! Deterministic synthetic mechanism test only. Online policy selection receives
//! ProducerInput and visible alarm history; evaluator truth is a separate type.
//!
//! Run:
//!   rustc --edition=2024 --test tools/vision-dvpc-alarm-envelope.rs -o /tmp/vis-dvpc-004-tests
//!   /tmp/vis-dvpc-004-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-alarm-envelope.rs -o /tmp/vis-dvpc-004
//!   /tmp/vis-dvpc-004 > vis-dvpc-004.tsv

#![deny(unsafe_code)]

const TRIALS_PER_SEED: usize = 256;
const SEEDS: [u64; 8] = [11, 23, 37, 41, 59, 71, 83, 97];
const BUDGET_PERCENTAGES: [u64; 3] = [20, 40, 60];
const FULL_FIDELITY_REFERENCE_COST: u64 = 400;
const DETECTOR_COST_UNITS_PER_OBSERVATION: u64 = 20;
const BASE_TASK_LOSS: f64 = 10.0;
const VALUE_THRESHOLD: f64 = 0.25;
const NEUTRAL_VALUE: f64 = 0.10;
const LCB_MULTIPLIER: f64 = 1.25;
const SHIFT_TEMPORAL_PERSISTENCE: f64 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Policy {
    LcbOnly,
    InstantGuard,
    TwoOfThreeGuard,
}

impl Policy {
    const ALL: [Self; 3] = [Self::LcbOnly, Self::InstantGuard, Self::TwoOfThreeGuard];

    fn label(self) -> &'static str {
        match self {
            Self::LcbOnly => "lcb_only",
            Self::InstantGuard => "instant_guard",
            Self::TwoOfThreeGuard => "two_of_three_guard",
        }
    }

    fn pays_for_detector(self) -> bool {
        self != Self::LcbOnly
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScenarioKind {
    ShiftSweep,
    NoShiftControl,
    OverconfidentControl,
}

#[derive(Debug, Clone)]
struct Scenario {
    name: String,
    kind: ScenarioKind,
    prevalence_percent: u8,
    sensitivity_percent: u8,
    false_positive_percent: u8,
    alarm_persistence_percent: u8,
}

fn scenarios() -> Vec<Scenario> {
    let mut rows = Vec::new();
    for prevalence in [5, 15, 35, 60] {
        for sensitivity in [50, 70, 90] {
            for false_positive in [1, 5, 20] {
                for persistence in [0, 70] {
                    rows.push(Scenario {
                        name: format!(
                            "shift_p{prevalence:02}_s{sensitivity:02}_f{false_positive:02}_r{persistence:02}"
                        ),
                        kind: ScenarioKind::ShiftSweep,
                        prevalence_percent: prevalence,
                        sensitivity_percent: sensitivity,
                        false_positive_percent: false_positive,
                        alarm_persistence_percent: persistence,
                    });
                }
            }
        }
    }
    rows.push(Scenario {
        name: "control_no_shift".to_owned(),
        kind: ScenarioKind::NoShiftControl,
        prevalence_percent: 0,
        sensitivity_percent: 70,
        false_positive_percent: 5,
        alarm_persistence_percent: 0,
    });
    rows.push(Scenario {
        name: "control_overconfident".to_owned(),
        kind: ScenarioKind::OverconfidentControl,
        prevalence_percent: 0,
        sensitivity_percent: 70,
        false_positive_percent: 5,
        alarm_persistence_percent: 0,
    });
    rows
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

/// Only pre-escalation fields and alarms computed within the budget are visible.
#[derive(Debug, Clone)]
struct ProducerInput {
    estimated_net_benefit: f64,
    reported_uncertainty: f64,
    declared_action_cost_units: Option<u64>,
}

/// Evaluator-private truth; never passed to choose().
#[derive(Debug, Clone)]
struct EvaluatorTrial {
    producer: ProducerInput,
    alarm_draw: bool,
    true_dvpc: f64,
    actual_action_cost_units: u64,
    is_shifted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    KeepCheap,
    Escalate,
}

#[derive(Debug, Clone)]
struct RunResult {
    scenario: String,
    policy: Policy,
    budget_percent: u64,
    seed: u64,
    trial_count: usize,
    lcb_candidates: usize,
    escalations: usize,
    beneficial_escalations: usize,
    harmful_escalations: usize,
    missed_beneficial: usize,
    shifted_trials: usize,
    shifted_escalations: usize,
    shifted_harmful: usize,
    alarm_observations: usize,
    alarm_true_positive: usize,
    alarm_false_positive: usize,
    alarm_false_negative: usize,
    shift_episodes: usize,
    detected_shift_episodes: usize,
    undetected_shift_episodes: usize,
    mean_detection_delay_trials: Option<f64>,
    empirical_sensitivity: Option<f64>,
    empirical_false_positive_rate: Option<f64>,
    false_alarm_blocks: usize,
    false_alarm_beneficial_blocks: usize,
    total_benefit: f64,
    task_loss: f64,
    cvar95_incremental_loss: f64,
    charged_units: u64,
    budget_units: u64,
    detector_cost_units: u64,
    reservation_overruns: usize,
    run_budget_overruns: usize,
}

#[derive(Debug, Clone)]
struct ObservedAlarm {
    value: bool,
    is_shifted_for_evaluator_only: bool,
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

/// A Markov alarm with target marginal probability p and persistence rho.
/// At rho=0 draws are independent; positive rho produces bursts.
fn correlated_alarm_probability(target_p: f64, persistence: f64, previous_alarm: bool) -> f64 {
    if previous_alarm {
        target_p + persistence * (1.0 - target_p)
    } else {
        target_p * (1.0 - persistence)
    }
}

fn make_trials(scenario: &Scenario, seed: u64) -> Vec<EvaluatorTrial> {
    let scenario_seed = scenario.name.bytes().fold(0_u64, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(byte as u64)
    });
    let mut rng = SplitMix64::new(seed ^ scenario_seed);
    let mut trials = Vec::with_capacity(TRIALS_PER_SEED);
    let mut previous_shift = false;
    let mut previous_alarm = false;

    for index in 0..TRIALS_PER_SEED {
        let prevalence = scenario.prevalence_percent as f64 / 100.0;
        let shifted = if scenario.kind != ScenarioKind::ShiftSweep || prevalence == 0.0 {
            false
        } else {
            let transition = if previous_shift {
                prevalence + SHIFT_TEMPORAL_PERSISTENCE * (1.0 - prevalence)
            } else {
                prevalence * (1.0 - SHIFT_TEMPORAL_PERSISTENCE)
            };
            rng.unit() < transition
        };
        previous_shift = shifted;

        let true_dvpc = base_value(&mut rng);
        let (estimated_net_benefit, reported_uncertainty) = match scenario.kind {
            ScenarioKind::ShiftSweep if shifted => (-true_dvpc + rng.symmetric(0.15), 0.20),
            ScenarioKind::ShiftSweep => (true_dvpc + rng.symmetric(0.15), 0.20),
            ScenarioKind::NoShiftControl => (true_dvpc + rng.symmetric(0.75), 0.75),
            ScenarioKind::OverconfidentControl => {
                if true_dvpc < 0.0 && rng.unit() < 0.45 {
                    (true_dvpc.abs() + 0.7 + rng.symmetric(0.15), 0.05)
                } else {
                    (true_dvpc + rng.symmetric(1.25), 0.05)
                }
            }
        };

        let target_alarm_probability =
            if scenario.kind == ScenarioKind::ShiftSweep && shifted {
                scenario.sensitivity_percent as f64 / 100.0
            } else {
                scenario.false_positive_percent as f64 / 100.0
            };
        let persistence = scenario.alarm_persistence_percent as f64 / 100.0;
        let alarm_probability =
            correlated_alarm_probability(target_alarm_probability, persistence, previous_alarm);
        let alarm_draw = rng.unit() < alarm_probability;
        previous_alarm = alarm_draw;

        let actual_action_cost_units = 200 + (rng.next_u64() % 301);
        let declared_action_cost_units = if index % 47 == 0 {
            None
        } else if index % 29 == 0 {
            Some(actual_action_cost_units.saturating_sub(100).max(1))
        } else {
            Some(actual_action_cost_units)
        };

        trials.push(EvaluatorTrial {
            producer: ProducerInput {
                estimated_net_benefit,
                reported_uncertainty,
                declared_action_cost_units,
            },
            alarm_draw,
            true_dvpc,
            actual_action_cost_units,
            is_shifted: shifted,
        });
    }
    trials
}

/// No hidden shift label or true DVPC is passed across this boundary.
fn choose(
    policy: Policy,
    input: &ProducerInput,
    current_alarm: Option<bool>,
    observed_history: &[bool],
) -> Choice {
    let lower_bound =
        input.estimated_net_benefit - LCB_MULTIPLIER * input.reported_uncertainty;
    if lower_bound <= VALUE_THRESHOLD {
        return Choice::KeepCheap;
    }

    let permit = match policy {
        Policy::LcbOnly => true,
        Policy::InstantGuard => current_alarm.is_some_and(|alarm| !alarm),
        Policy::TwoOfThreeGuard => {
            observed_history.len() >= 3
                && observed_history.iter().rev().take(3).filter(|alarm| **alarm).count() < 2
        }
    };
    if permit {
        Choice::Escalate
    } else {
        Choice::KeepCheap
    }
}

fn budget_capacity(trials: usize, percentage: u64) -> u64 {
    ((trials as u128)
        .saturating_mul(FULL_FIDELITY_REFERENCE_COST as u128)
        .saturating_mul(percentage as u128)
        / 100)
        .min(u64::MAX as u128) as u64
}

/// Mean of the worst ceil(5% * n) incremental decision losses; positive means harm.
fn cvar95_upper_tail(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| right.total_cmp(left));
    let tail_count = ((ordered.len() + 19) / 20).max(1);
    ordered.iter().take(tail_count).sum::<f64>() / tail_count as f64
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn run_policy(
    scenario: &Scenario,
    policy: Policy,
    budget_percent: u64,
    seed: u64,
    trials: &[EvaluatorTrial],
) -> RunResult {
    let budget_units = budget_capacity(trials.len(), budget_percent);
    let mut charged_units = 0_u64;
    let mut detector_cost_units = 0_u64;
    let mut lcb_candidates = 0;
    let mut escalations = 0;
    let mut beneficial_escalations = 0;
    let mut harmful_escalations = 0;
    let mut missed_beneficial = 0;
    let mut shifted_trials = 0;
    let mut shifted_escalations = 0;
    let mut shifted_harmful = 0;
    let mut alarm_observations = 0;
    let mut alarm_true_positive = 0;
    let mut alarm_false_positive = 0;
    let mut alarm_false_negative = 0;
    let mut false_alarm_blocks = 0;
    let mut false_alarm_beneficial_blocks = 0;
    let mut reservation_overruns = 0;
    let mut run_budget_overruns = 0;
    let mut detector_budget_exhausted = false;
    let mut stop_after_accounting_failure = false;
    let mut alarm_history: Vec<ObservedAlarm> = Vec::new();
    let mut gate_trigger_timeline = Vec::with_capacity(trials.len());
    let mut incremental_losses = Vec::with_capacity(trials.len());

    for trial in trials {
        if trial.is_shifted {
            shifted_trials += 1;
        }

        let current_alarm = if policy.pays_for_detector()
            && !detector_budget_exhausted
            && !stop_after_accounting_failure
        {
            let can_reserve_detector = charged_units
                .checked_add(DETECTOR_COST_UNITS_PER_OBSERVATION)
                .is_some_and(|next| next <= budget_units);
            if can_reserve_detector {
                charged_units += DETECTOR_COST_UNITS_PER_OBSERVATION;
                detector_cost_units += DETECTOR_COST_UNITS_PER_OBSERVATION;
                alarm_observations += 1;
                if trial.alarm_draw && trial.is_shifted {
                    alarm_true_positive += 1;
                } else if trial.alarm_draw && !trial.is_shifted {
                    alarm_false_positive += 1;
                } else if !trial.alarm_draw && trial.is_shifted {
                    alarm_false_negative += 1;
                }
                alarm_history.push(ObservedAlarm {
                    value: trial.alarm_draw,
                    is_shifted_for_evaluator_only: trial.is_shifted,
                });
                if alarm_history.len() > 3 {
                    alarm_history.remove(0);
                }
                Some(trial.alarm_draw)
            } else {
                detector_budget_exhausted = true;
                None
            }
        } else {
            None
        };

        let lcb_candidate = trial.producer.estimated_net_benefit
            - LCB_MULTIPLIER * trial.producer.reported_uncertainty
            > VALUE_THRESHOLD;
        if lcb_candidate {
            lcb_candidates += 1;
        }

        let observed_values: Vec<bool> = alarm_history.iter().map(|entry| entry.value).collect();
        let gate_triggered = match policy {
            Policy::LcbOnly => false,
            Policy::InstantGuard => current_alarm == Some(true) && trial.is_shifted,
            Policy::TwoOfThreeGuard => {
                let recent = alarm_history.iter().rev().take(3);
                current_alarm.is_some()
                    && observed_values.len() >= 3
                    && observed_values.iter().rev().take(3).filter(|alarm| **alarm).count() >= 2
                    && recent.filter(|entry| {
                        entry.value && entry.is_shifted_for_evaluator_only
                    }).count() >= 2
            }
        };
        gate_trigger_timeline.push(gate_triggered);
        let candidate = if stop_after_accounting_failure
            || (policy.pays_for_detector() && current_alarm.is_none())
        {
            Choice::KeepCheap
        } else {
            choose(policy, &trial.producer, current_alarm, &observed_values)
        };

        let declared_cost = trial.producer.declared_action_cost_units;
        let would_fit_budget = match declared_cost {
            Some(cost) if cost > 0 && !stop_after_accounting_failure => charged_units
                .checked_add(cost)
                .is_some_and(|next| next <= budget_units),
            _ => false,
        };

        if candidate == Choice::KeepCheap
            && lcb_candidate
            && would_fit_budget
            && policy != Policy::LcbOnly
        {
            let guard_blocked = match policy {
                Policy::InstantGuard => current_alarm == Some(true),
                Policy::TwoOfThreeGuard => {
                    observed_values.len() >= 3
                        && observed_values.iter().rev().take(3).filter(|alarm| **alarm).count() >= 2
                }
                Policy::LcbOnly => false,
            };
            if guard_blocked {
                let recent: Vec<&ObservedAlarm> = alarm_history.iter().rev().take(3).collect();
                let recent_alarm_count = recent.iter().filter(|entry| entry.value).count();
                let recent_true_positive_count = recent
                    .iter()
                    .filter(|entry| entry.value && entry.is_shifted_for_evaluator_only)
                    .count();
                // A false alarm is counted as causal only if removing false positives
                // from this window would leave fewer than two positive alarms.
                let false_alarm_was_necessary = match policy {
                    Policy::InstantGuard => current_alarm == Some(true)
                        && recent.first().is_some_and(|entry| {
                            entry.value && !entry.is_shifted_for_evaluator_only
                        }),
                    Policy::TwoOfThreeGuard => {
                        recent_alarm_count >= 2
                            && recent_true_positive_count < 2
                    }
                    Policy::LcbOnly => false,
                };
                if false_alarm_was_necessary {
                    false_alarm_blocks += 1;
                    if trial.true_dvpc > NEUTRAL_VALUE {
                        false_alarm_beneficial_blocks += 1;
                    }
                }
            }
        }

        let can_reserve_action = would_fit_budget;
        let choice = if candidate == Choice::Escalate && can_reserve_action {
            Choice::Escalate
        } else {
            Choice::KeepCheap
        };

        if choice == Choice::Escalate {
            escalations += 1;
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
            let declared = declared_cost.expect("reserved action must have declared cost");
            charged_units = charged_units.saturating_add(trial.actual_action_cost_units);
            if trial.actual_action_cost_units > declared {
                reservation_overruns += 1;
                stop_after_accounting_failure = true;
            }
            if charged_units > budget_units {
                run_budget_overruns += 1;
                stop_after_accounting_failure = true;
            }
            incremental_losses.push(-trial.true_dvpc);
        } else {
            if trial.true_dvpc > NEUTRAL_VALUE {
                missed_beneficial += 1;
            }
            incremental_losses.push(0.0);
        }
    }

    let total_benefit = -incremental_losses.iter().sum::<f64>();
    let task_loss = BASE_TASK_LOSS * trials.len() as f64 - total_benefit;
    let mut shift_episodes = 0;
    let mut detected_shift_episodes = 0;
    let mut undetected_shift_episodes = 0;
    let mut detection_delays = Vec::new();
    let mut episode_start: Option<usize> = None;
    for index in 0..=trials.len() {
        let in_shift = index < trials.len() && trials[index].is_shifted;
        if in_shift {
            if episode_start.is_none() {
                episode_start = Some(index);
            }
        } else if let Some(start) = episode_start.take() {
            shift_episodes += 1;
            if let Some(detected_at) = (start..index).find(|&at| gate_trigger_timeline[at]) {
                detected_shift_episodes += 1;
                detection_delays.push((detected_at - start) as f64);
            } else {
                undetected_shift_episodes += 1;
            }
        }
    }
    let mean_detection_delay_trials = mean(&detection_delays);
    let shifted_alarm_denominator = alarm_true_positive + alarm_false_negative;
    let unshifted_alarm_denominator =
        alarm_observations.saturating_sub(shifted_alarm_denominator);

    RunResult {
        scenario: scenario.name.clone(),
        policy,
        budget_percent,
        seed,
        trial_count: trials.len(),
        lcb_candidates,
        escalations,
        beneficial_escalations,
        harmful_escalations,
        missed_beneficial,
        shifted_trials,
        shifted_escalations,
        shifted_harmful,
        alarm_observations,
        alarm_true_positive,
        alarm_false_positive,
        alarm_false_negative,
        shift_episodes,
        detected_shift_episodes,
        undetected_shift_episodes,
        mean_detection_delay_trials,
        empirical_sensitivity: ratio(alarm_true_positive, shifted_alarm_denominator),
        empirical_false_positive_rate: ratio(alarm_false_positive, unshifted_alarm_denominator),
        false_alarm_blocks,
        false_alarm_beneficial_blocks,
        total_benefit,
        task_loss,
        cvar95_incremental_loss: cvar95_upper_tail(&incremental_losses),
        charged_units,
        budget_units,
        detector_cost_units,
        reservation_overruns,
        run_budget_overruns,
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_else(|| "NA".to_owned())
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn render_report() -> String {
    let scenario_rows = scenarios();
    let mut results = Vec::new();
    for scenario in &scenario_rows {
        for seed in SEEDS {
            let trials = make_trials(scenario, seed);
            for budget_percent in BUDGET_PERCENTAGES {
                for policy in Policy::ALL {
                    results.push(run_policy(scenario, policy, budget_percent, seed, &trials));
                }
            }
        }
    }

    let mut report = String::new();
    report.push_str("META\tformat\tVIS-DVPC-004-v0.1\n");
    report.push_str("META\tclaim_class\tsynthetic_alarm_operating_envelope_not_detector_validation_or_sota\n");
    report.push_str("META\tsweep_cells\t72\n");
    report.push_str("META\tcontrol_cells\t2\n");
    report.push_str("META\ttrials_per_seed\t256\n");
    report.push_str("META\tseeds\t8\n");
    report.push_str("META\tpolicies\tlcb_only;instant_guard;two_of_three_guard\n");
    report.push_str("META\tfull_fidelity_reference_cost\t400_units_per_trial\n");
    report.push_str("META\tguard_detector_cost\t20_units_per_observation_included_in_same_budget\n");
    report.push_str("META\tshift_episode_persistence\t0.75\n");
    report.push_str("META\tbudget_percentages\t20;40;60\n");
    report.push_str("META\tevaluator_boundary\thidden_shift_true_dvpc_actual_cost_post_commit_only\n");
    report.push_str("RUN_HEADER\tscenario\tpolicy\tbudget_percent\tseed\ttrial_count\tlcb_candidates\tescalations\tbeneficial_escalations\tharmful_escalations\tmissed_beneficial\tshifted_trials\tshifted_escalations\tshifted_harmful\talarm_observations\talarm_true_positive\talarm_false_positive\talarm_false_negative\tshift_episodes\tdetected_shift_episodes\tundetected_shift_episodes\tmean_detection_delay_trials\tempirical_sensitivity\tempirical_false_positive_rate\tfalse_alarm_blocks\tfalse_alarm_beneficial_blocks\ttotal_benefit\ttask_loss\tcvar95_incremental_loss\tcharged_units\tbudget_units\tdetector_cost_units\treservation_overruns\trun_budget_overruns\n");

    for row in &results {
        report.push_str(&format!(
            "RUN\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\n",
            row.scenario,
            row.policy.label(),
            row.budget_percent,
            row.seed,
            row.trial_count,
            row.lcb_candidates,
            row.escalations,
            row.beneficial_escalations,
            row.harmful_escalations,
            row.missed_beneficial,
            row.shifted_trials,
            row.shifted_escalations,
            row.shifted_harmful,
            row.alarm_observations,
            row.alarm_true_positive,
            row.alarm_false_positive,
            row.alarm_false_negative,
            row.shift_episodes,
            row.detected_shift_episodes,
            row.undetected_shift_episodes,
            fmt_opt(row.mean_detection_delay_trials),
            fmt_opt(row.empirical_sensitivity),
            fmt_opt(row.empirical_false_positive_rate),
            row.false_alarm_blocks,
            row.false_alarm_beneficial_blocks,
            row.total_benefit,
            row.task_loss,
            row.cvar95_incremental_loss,
            row.charged_units,
            row.budget_units,
            row.detector_cost_units,
            row.reservation_overruns,
            row.run_budget_overruns
        ));
    }

    report.push_str("SUMMARY_HEADER\tscenario\tpolicy\tbudget_percent\tseed_count\tmean_escalations\tmean_harmful_escalations\tmean_missed_beneficial\tmean_shifted_harmful\tmean_detected_shift_episodes\tmean_undetected_shift_episodes\tmean_detection_delay_trials\tmean_sensitivity\tmean_false_positive_rate\tmean_false_alarm_beneficial_blocks\tmean_total_benefit\tmean_task_loss\tmean_cvar95_incremental_loss\tmean_charged_units\tmean_detector_cost_units\tbudget_violation_runs\n");
    for scenario in &scenario_rows {
        for budget_percent in BUDGET_PERCENTAGES {
            for policy in Policy::ALL {
                let rows: Vec<&RunResult> = results
                    .iter()
                    .filter(|row| {
                        row.scenario == scenario.name
                            && row.budget_percent == budget_percent
                            && row.policy == policy
                    })
                    .collect();
                let collect = |get: fn(&RunResult) -> f64| {
                    rows.iter().map(|row| get(row)).collect::<Vec<_>>()
                };
                let sensitivities: Vec<f64> = rows
                    .iter()
                    .filter_map(|row| row.empirical_sensitivity)
                    .collect();
                let false_positive_rates: Vec<f64> = rows
                    .iter()
                    .filter_map(|row| row.empirical_false_positive_rate)
                    .collect();
                let detection_delays: Vec<f64> = rows
                    .iter()
                    .filter_map(|row| row.mean_detection_delay_trials)
                    .collect();
                let violations = rows
                    .iter()
                    .filter(|row| row.reservation_overruns > 0 || row.run_budget_overruns > 0)
                    .count();

                report.push_str(&format!(
                    "SUMMARY\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    scenario.name,
                    policy.label(),
                    budget_percent,
                    rows.len(),
                    fmt_opt(mean(&collect(|row| row.escalations as f64))),
                    fmt_opt(mean(&collect(|row| row.harmful_escalations as f64))),
                    fmt_opt(mean(&collect(|row| row.missed_beneficial as f64))),
                    fmt_opt(mean(&collect(|row| row.shifted_harmful as f64))),
                    fmt_opt(mean(&collect(|row| row.detected_shift_episodes as f64))),
                    fmt_opt(mean(&collect(|row| row.undetected_shift_episodes as f64))),
                    fmt_opt(mean(&detection_delays)),
                    fmt_opt(mean(&sensitivities)),
                    fmt_opt(mean(&false_positive_rates)),
                    fmt_opt(mean(&collect(|row| row.false_alarm_beneficial_blocks as f64))),
                    fmt_opt(mean(&collect(|row| row.total_benefit))),
                    fmt_opt(mean(&collect(|row| row.task_loss))),
                    fmt_opt(mean(&collect(|row| row.cvar95_incremental_loss))),
                    fmt_opt(mean(&collect(|row| row.charged_units as f64))),
                    fmt_opt(mean(&collect(|row| row.detector_cost_units as f64))),
                    violations
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

    #[test]
    fn scenario_grid_has_72_operating_cells_and_two_controls() {
        let rows = scenarios();
        assert_eq!(rows.len(), 74);
        assert_eq!(rows.iter().filter(|row| row.kind == ScenarioKind::ShiftSweep).count(), 72);
        assert!(rows.iter().any(|row| row.kind == ScenarioKind::NoShiftControl));
        assert!(rows.iter().any(|row| row.kind == ScenarioKind::OverconfidentControl));
    }

    #[test]
    fn alarm_guard_blocks_only_after_alarm_or_confirmation() {
        let input = ProducerInput {
            trial_id: 1,
            estimated_net_benefit: 2.0,
            reported_uncertainty: 0.1,
            declared_action_cost_units: Some(250),
        };
        assert_eq!(choose(Policy::LcbOnly, &input, Some(true), &[true]), Choice::Escalate);
        assert_eq!(choose(Policy::InstantGuard, &input, Some(true), &[true]), Choice::KeepCheap);
        assert_eq!(choose(Policy::InstantGuard, &input, Some(false), &[false]), Choice::Escalate);
        assert_eq!(choose(Policy::TwoOfThreeGuard, &input, Some(false), &[true, false]), Choice::KeepCheap);
        assert_eq!(choose(Policy::TwoOfThreeGuard, &input, Some(false), &[true, false, false]), Choice::Escalate);
        assert_eq!(choose(Policy::TwoOfThreeGuard, &input, Some(false), &[true, true, false]), Choice::KeepCheap);
    }

    #[test]
    fn unknown_action_cost_fails_closed() {
        let scenario = Scenario {
            name: "test".to_owned(),
            kind: ScenarioKind::NoShiftControl,
            prevalence_percent: 0,
            sensitivity_percent: 70,
            false_positive_percent: 5,
            alarm_persistence_percent: 0,
        };
        let mut trials = make_trials(&scenario, 11);
        for trial in &mut trials {
            trial.producer.declared_action_cost_units = None;
            trial.alarm_draw = false;
        }
        let row = run_policy(&scenario, Policy::LcbOnly, 60, 11, &trials);
        assert_eq!(row.escalations, 0);
        assert_eq!(row.charged_units, 0);
    }

    #[test]
    fn detector_compute_is_charged_and_obeys_shared_budget() {
        let scenario = Scenario {
            name: "test".to_owned(),
            kind: ScenarioKind::NoShiftControl,
            prevalence_percent: 0,
            sensitivity_percent: 70,
            false_positive_percent: 5,
            alarm_persistence_percent: 0,
        };
        let trials = make_trials(&scenario, 11);
        let row = run_policy(&scenario, Policy::InstantGuard, 0, 11, &trials);
        assert_eq!(row.alarm_observations, 0);
        assert_eq!(row.detector_cost_units, 0);
        assert_eq!(row.escalations, 0);
    }

    #[test]
    fn cvar95_reports_worst_five_percent_loss() {
        let values: Vec<f64> = (0..19).map(|_| 0.0).chain([10.0]).collect();
        assert_eq!(cvar95_upper_tail(&values), 10.0);
    }

    #[test]
    fn report_replays_byte_identically_and_has_frozen_row_counts() {
        let first = render_report();
        let second = render_report();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("RUN\t")).count(), 74 * 8 * 3 * 3);
        assert_eq!(first.lines().filter(|line| line.starts_with("SUMMARY\t")).count(), 74 * 3 * 3);
        assert!(!first.contains("is_shifted\t"));
        assert!(!first.contains("true_dvpc\t"));
    }
}
