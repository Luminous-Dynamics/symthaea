use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::BTreeMap;

use crate::cropfoundry_water_policy::{
    ScenarioResult, robustness_scenarios, simulate_policy_suite,
};

const TAIL_FRACTION_95: f64 = 0.05;

#[derive(Debug, Clone, Copy)]
struct Preference {
    id: &'static str,
    name: &'static str,
    mean_stress: f64,
    cvar95_stress: f64,
    worst_zone_stress: f64,
    water_use: f64,
    allocation_dispersion: f64,
}

fn preferences() -> [Preference; 4] {
    [
        Preference {
            id: "balanced",
            name: "balanced mean-tail-equity",
            mean_stress: 0.40,
            cvar95_stress: 0.30,
            worst_zone_stress: 0.20,
            water_use: 0.05,
            allocation_dispersion: 0.05,
        },
        Preference {
            id: "tail_guard",
            name: "upper-tail protection",
            mean_stress: 0.20,
            cvar95_stress: 0.45,
            worst_zone_stress: 0.30,
            water_use: 0.025,
            allocation_dispersion: 0.025,
        },
        Preference {
            id: "water_scarcity",
            name: "water scarcity",
            mean_stress: 0.30,
            cvar95_stress: 0.25,
            worst_zone_stress: 0.20,
            water_use: 0.20,
            allocation_dispersion: 0.05,
        },
        Preference {
            id: "distribution_first",
            name: "distributional protection",
            mean_stress: 0.25,
            cvar95_stress: 0.20,
            worst_zone_stress: 0.30,
            water_use: 0.05,
            allocation_dispersion: 0.20,
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
struct PolicyRobustnessScore {
    policy: String,
    scenario_count: usize,
    tail_fraction: f64,
    tail_sample_count: usize,
    upper_tail_scenario_ids: Vec<String>,
    mean_stress_exposure_proxy: f64,
    median_stress_exposure_proxy: f64,
    p95_stress_exposure_proxy_nearest_rank: f64,
    cvar95_stress_exposure_proxy: f64,
    max_worst_zone_stress_exposure_proxy: f64,
    mean_worst_zone_stress_exposure_proxy: f64,
    mean_water_applied_plot_mm: f64,
    total_water_applied_plot_mm: f64,
    mean_plot_allocation_dispersion_plot_mm: f64,
    mean_regret_vs_per_scenario_best_stress_proxy: f64,
    total_water_budget_violation_days: usize,
    total_per_plot_limit_violation_days: usize,
    pareto_front_member: bool,
    dominated_by_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ScenarioPolicyDiagnostic {
    scenario_id: String,
    policy: String,
    stress_exposure_proxy: f64,
    max_zone_stress_exposure_proxy: f64,
    water_applied_plot_mm: f64,
    plot_allocation_dispersion_plot_mm: f64,
    water_budget_violation_days: usize,
    per_plot_limit_violation_days: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PreferenceRanking {
    preference_id: String,
    preference_name: String,
    weights: BTreeMap<String, f64>,
    top_ranked_policy: String,
    top_policy_score: f64,
    ranking: Vec<PreferenceCandidateScore>,
}

#[derive(Debug, Clone, Serialize)]
struct PreferenceCandidateScore {
    policy: String,
    normalized_score: f64,
    rank: usize,
}

#[derive(Debug, Serialize)]
struct RobustPolicyReceipt {
    receipt_schema: String,
    benchmark_id: String,
    execution_mode: String,
    synthetic_only: bool,
    calibrated_probability_claim: bool,
    biological_performance_claim: bool,
    training_authorized: bool,
    scenario_count: usize,
    policy_count: usize,
    robustness_objectives: Vec<String>,
    preference_count: usize,
    distinct_top_ranked_policy_count: usize,
    hard_constraint_summary: String,
    scenario_policy_diagnostics_count: usize,
    scenario_policy_diagnostics: Vec<ScenarioPolicyDiagnostic>,
    policies: Vec<PolicyRobustnessScore>,
    preference_sweep: Vec<PreferenceRanking>,
    caveats: Vec<String>,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(sorted: &[f64]) -> f64 {
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}

fn nearest_rank(sorted: &[f64], quantile: f64) -> f64 {
    let rank = ((sorted.len() as f64 * quantile).ceil() as usize).max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn cvar_upper_tail(sorted: &[f64], tail_fraction: f64) -> (f64, usize) {
    let tail_count = ((sorted.len() as f64 * tail_fraction).ceil() as usize).max(1);
    let start = sorted.len().saturating_sub(tail_count);
    (mean(&sorted[start..]), sorted.len() - start)
}

fn normalize(value: f64, maximum: f64) -> f64 {
    if maximum > 0.0 { value / maximum } else { 0.0 }
}

fn dominates(left: &PolicyRobustnessScore, right: &PolicyRobustnessScore) -> bool {
    let left_values = [
        left.mean_stress_exposure_proxy,
        left.cvar95_stress_exposure_proxy,
        left.max_worst_zone_stress_exposure_proxy,
        left.mean_water_applied_plot_mm,
        left.mean_plot_allocation_dispersion_plot_mm,
    ];
    let right_values = [
        right.mean_stress_exposure_proxy,
        right.cvar95_stress_exposure_proxy,
        right.max_worst_zone_stress_exposure_proxy,
        right.mean_water_applied_plot_mm,
        right.mean_plot_allocation_dispersion_plot_mm,
    ];
    let no_worse = left_values
        .iter()
        .zip(right_values)
        .all(|(left_value, right_value)| *left_value <= right_value + 1e-9);
    let strictly_better = left_values
        .iter()
        .zip(right_values)
        .any(|(left_value, right_value)| *left_value < right_value - 1e-9);
    no_worse && strictly_better
}

fn build_policy_scores(results: &[ScenarioResult]) -> Result<Vec<PolicyRobustnessScore>> {
    if results.is_empty() {
        return Err(anyhow!(
            "robust-policy evaluator received no scenario results"
        ));
    }
    let mut grouped = BTreeMap::<String, Vec<&ScenarioResult>>::new();
    let mut minimum_stress_by_scenario = BTreeMap::<String, f64>::new();
    for result in results {
        grouped
            .entry(result.policy.clone())
            .or_default()
            .push(result);
        minimum_stress_by_scenario
            .entry(result.scenario_id.clone())
            .and_modify(|best| *best = best.min(result.stress_exposure_proxy))
            .or_insert(result.stress_exposure_proxy);
    }
    let expected_scenario_ids = grouped
        .values()
        .next()
        .map(|items| {
            items
                .iter()
                .map(|item| item.scenario_id.clone())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let expected_scenario_count = expected_scenario_ids.len();
    if expected_scenario_count == 0 {
        return Err(anyhow!("robust-policy evaluator has no declared scenarios"));
    }

    let mut scores = Vec::new();
    for (policy, items) in grouped {
        let observed_ids = items
            .iter()
            .map(|item| item.scenario_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if items.len() != expected_scenario_count
            || observed_ids.len() != items.len()
            || observed_ids != expected_scenario_ids
        {
            return Err(anyhow!(
                "policy {policy} scenario IDs do not match the common evaluation suite"
            ));
        }

        let mut stress_by_scenario = items
            .iter()
            .map(|item| (item.scenario_id.clone(), item.stress_exposure_proxy))
            .collect::<Vec<_>>();
        stress_by_scenario.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let stress = stress_by_scenario
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        let (cvar95, tail_count) = cvar_upper_tail(&stress, TAIL_FRACTION_95);
        let tail_start = stress_by_scenario.len().saturating_sub(tail_count);
        let mut upper_tail_scenario_ids = stress_by_scenario[tail_start..]
            .iter()
            .map(|(scenario_id, _)| scenario_id.clone())
            .collect::<Vec<_>>();
        upper_tail_scenario_ids.sort();
        let worst_zone = items
            .iter()
            .map(|item| item.max_zone_stress_exposure_proxy)
            .collect::<Vec<_>>();
        let regrets = items
            .iter()
            .map(|item| {
                let best = minimum_stress_by_scenario
                    .get(&item.scenario_id)
                    .copied()
                    .unwrap_or(item.stress_exposure_proxy);
                item.stress_exposure_proxy - best
            })
            .collect::<Vec<_>>();
        let water = items
            .iter()
            .map(|item| item.total_irrigation_applied_plot_mm)
            .collect::<Vec<_>>();
        let dispersions = items
            .iter()
            .map(|item| item.plot_allocation_dispersion_plot_mm)
            .collect::<Vec<_>>();
        let water_budget_violations = items
            .iter()
            .map(|item| item.water_budget_violation_days)
            .sum::<usize>();
        let actuator_violations = items
            .iter()
            .map(|item| item.per_plot_limit_violation_days)
            .sum::<usize>();
        if water_budget_violations != 0 || actuator_violations != 0 {
            return Err(anyhow!(
                "robust-policy feasibility gate failed for {policy}: {water_budget_violations} water-cap violations, {actuator_violations} actuator-limit violations"
            ));
        }
        scores.push(PolicyRobustnessScore {
            policy,
            scenario_count: items.len(),
            tail_fraction: TAIL_FRACTION_95,
            tail_sample_count: tail_count,
            upper_tail_scenario_ids,
            mean_stress_exposure_proxy: mean(&stress),
            median_stress_exposure_proxy: median(&stress),
            p95_stress_exposure_proxy_nearest_rank: nearest_rank(&stress, 0.95),
            cvar95_stress_exposure_proxy: cvar95,
            max_worst_zone_stress_exposure_proxy: worst_zone
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .unwrap_or(0.0),
            mean_worst_zone_stress_exposure_proxy: mean(&worst_zone),
            mean_water_applied_plot_mm: mean(&water),
            total_water_applied_plot_mm: water.iter().sum(),
            mean_plot_allocation_dispersion_plot_mm: mean(&dispersions),
            mean_regret_vs_per_scenario_best_stress_proxy: mean(&regrets),
            total_water_budget_violation_days: water_budget_violations,
            total_per_plot_limit_violation_days: actuator_violations,
            pareto_front_member: false,
            dominated_by_policies: Vec::new(),
        });
    }

    for right_index in 0..scores.len() {
        let dominators = (0..scores.len())
            .filter(|left_index| {
                *left_index != right_index && dominates(&scores[*left_index], &scores[right_index])
            })
            .map(|left_index| scores[left_index].policy.clone())
            .collect::<Vec<_>>();
        scores[right_index].pareto_front_member = dominators.is_empty();
        scores[right_index].dominated_by_policies = dominators;
    }
    scores.sort_by(|left, right| left.policy.cmp(&right.policy));
    Ok(scores)
}

fn build_preference_sweep(scores: &[PolicyRobustnessScore]) -> Vec<PreferenceRanking> {
    let maxima = [
        scores
            .iter()
            .map(|score| score.mean_stress_exposure_proxy)
            .fold(0.0_f64, f64::max),
        scores
            .iter()
            .map(|score| score.cvar95_stress_exposure_proxy)
            .fold(0.0_f64, f64::max),
        scores
            .iter()
            .map(|score| score.max_worst_zone_stress_exposure_proxy)
            .fold(0.0_f64, f64::max),
        scores
            .iter()
            .map(|score| score.mean_water_applied_plot_mm)
            .fold(0.0_f64, f64::max),
        scores
            .iter()
            .map(|score| score.mean_plot_allocation_dispersion_plot_mm)
            .fold(0.0_f64, f64::max),
    ];
    let mut sweep = Vec::new();
    for preference in preferences() {
        let mut ranking = scores
            .iter()
            .map(|score| {
                let normalized_score = preference.mean_stress
                    * normalize(score.mean_stress_exposure_proxy, maxima[0])
                    + preference.cvar95_stress
                        * normalize(score.cvar95_stress_exposure_proxy, maxima[1])
                    + preference.worst_zone_stress
                        * normalize(score.max_worst_zone_stress_exposure_proxy, maxima[2])
                    + preference.water_use * normalize(score.mean_water_applied_plot_mm, maxima[3])
                    + preference.allocation_dispersion
                        * normalize(score.mean_plot_allocation_dispersion_plot_mm, maxima[4]);
                PreferenceCandidateScore {
                    policy: score.policy.clone(),
                    normalized_score,
                    rank: 0,
                }
            })
            .collect::<Vec<_>>();
        ranking.sort_by(|left, right| {
            left.normalized_score
                .total_cmp(&right.normalized_score)
                .then_with(|| left.policy.cmp(&right.policy))
        });
        for (index, candidate) in ranking.iter_mut().enumerate() {
            candidate.rank = index + 1;
        }
        let top = ranking.first().expect("nonempty policy list");
        let weights = BTreeMap::from([
            ("mean_stress".to_owned(), preference.mean_stress),
            ("cvar95_stress".to_owned(), preference.cvar95_stress),
            ("worst_zone_stress".to_owned(), preference.worst_zone_stress),
            ("water_use".to_owned(), preference.water_use),
            (
                "allocation_dispersion".to_owned(),
                preference.allocation_dispersion,
            ),
        ]);
        sweep.push(PreferenceRanking {
            preference_id: preference.id.to_owned(),
            preference_name: preference.name.to_owned(),
            weights,
            top_ranked_policy: top.policy.clone(),
            top_policy_score: top.normalized_score,
            ranking,
        });
    }
    sweep
}

fn build_receipt() -> Result<RobustPolicyReceipt> {
    let scenarios = robustness_scenarios();
    if scenarios.len() != 30 {
        return Err(anyhow!(
            "robust-policy evaluator expected 30 declared sensitivity cases, received {}",
            scenarios.len()
        ));
    }
    let results = simulate_policy_suite(&scenarios)?;
    let policy_count = results
        .iter()
        .map(|result| result.policy.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if policy_count != 5 || results.len() != scenarios.len() * policy_count {
        return Err(anyhow!(
            "robust-policy evaluator expected 30 scenarios x 5 policies, received {} rows across {policy_count} policies",
            results.len()
        ));
    }
    let scores = build_policy_scores(&results)?;
    if scores.len() != policy_count
        || scores
            .iter()
            .any(|score| score.scenario_count != scenarios.len())
    {
        return Err(anyhow!(
            "robust-policy aggregation did not preserve the sensitivity-suite cardinalities"
        ));
    }
    let mut scenario_policy_diagnostics = results
        .iter()
        .map(|result| ScenarioPolicyDiagnostic {
            scenario_id: result.scenario_id.clone(),
            policy: result.policy.clone(),
            stress_exposure_proxy: result.stress_exposure_proxy,
            max_zone_stress_exposure_proxy: result.max_zone_stress_exposure_proxy,
            water_applied_plot_mm: result.total_irrigation_applied_plot_mm,
            plot_allocation_dispersion_plot_mm: result.plot_allocation_dispersion_plot_mm,
            water_budget_violation_days: result.water_budget_violation_days,
            per_plot_limit_violation_days: result.per_plot_limit_violation_days,
        })
        .collect::<Vec<_>>();
    scenario_policy_diagnostics.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then_with(|| left.policy.cmp(&right.policy))
    });
    if scenario_policy_diagnostics.len() != scenarios.len() * policy_count {
        return Err(anyhow!(
            "scenario-policy diagnostics are incomplete: expected {}, got {}",
            scenarios.len() * policy_count,
            scenario_policy_diagnostics.len()
        ));
    }
    let preference_sweep = build_preference_sweep(&scores);
    let distinct_top_ranked_policy_count = preference_sweep
        .iter()
        .map(|item| item.top_ranked_policy.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if scores.iter().any(|score| {
        [
            score.mean_stress_exposure_proxy,
            score.median_stress_exposure_proxy,
            score.p95_stress_exposure_proxy_nearest_rank,
            score.cvar95_stress_exposure_proxy,
            score.max_worst_zone_stress_exposure_proxy,
            score.mean_worst_zone_stress_exposure_proxy,
            score.mean_water_applied_plot_mm,
            score.total_water_applied_plot_mm,
            score.mean_plot_allocation_dispersion_plot_mm,
            score.mean_regret_vs_per_scenario_best_stress_proxy,
        ]
        .iter()
        .any(|value| !value.is_finite())
    }) {
        return Err(anyhow!("robust-policy receipt contains non-finite metrics"));
    }
    Ok(RobustPolicyReceipt {
        receipt_schema: "cropfoundry-robust-policy-receipt-v1".to_owned(),
        benchmark_id: "CF-025".to_owned(),
        execution_mode: "deterministic_synthetic_30_case_sensitivity_lattice".to_owned(),
        synthetic_only: true,
        calibrated_probability_claim: false,
        biological_performance_claim: false,
        training_authorized: false,
        scenario_count: scenarios.len(),
        policy_count,
        robustness_objectives: vec![
            "mean synthetic stress exposure".to_owned(),
            "nearest-rank 95th percentile and upper-tail mean".to_owned(),
            "maximum per-zone synthetic stress burden".to_owned(),
            "irrigation use in synthetic plot-mm".to_owned(),
            "per-zone irrigation allocation dispersion".to_owned(),
        ],
        preference_count: preference_sweep.len(),
        distinct_top_ranked_policy_count,
        hard_constraint_summary: "All 30 sensitivity scenarios x 5 policies were re-simulated; every result has zero daily water-budget and per-plot actuator-limit violations. This is a code-level synthetic invariant only.".to_owned(),
        scenario_policy_diagnostics_count: scenario_policy_diagnostics.len(),
        scenario_policy_diagnostics,
        policies: scores,
        preference_sweep,
        caveats: vec![
            "All scenario parameters and objective values are synthetic; no plant, soil or farm is modelled.".to_owned(),
            "The fixed 30-case lattice gives the scenario-set CVaR95 two upper-tail sensitivity cases. These are designed points, not a random or calibrated probability sample; p95/CVaR95 remain diagnostics, not estimates of field-outcome probabilities.".to_owned(),
            "The per-scenario best policy used for regret is a hindsight oracle diagnostic and is not available prospectively.".to_owned(),
            "Normalized weighted rankings are a declared preference-sensitivity demonstration, not an objectively correct recommendation.".to_owned(),
            "Pareto dominance only applies to these five declared synthetic objectives; omitted constraints or objectives can change conclusions.".to_owned(),
            "Zero synthetic violations do not validate real actuator limits, safety, water rights, yield, nutrition, economics or field transfer.".to_owned(),
            "No training or external dataset ingestion occurs in this command.".to_owned(),
        ],
    })
}

/// Run robust-policy diagnostics over the existing synthetic irrigation suite; no new simulation model is introduced.
pub fn run(out: Option<std::path::PathBuf>) -> Result<()> {
    let receipt = build_receipt()?;
    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| {
                anyhow!(
                    "cannot create output directory {}: {error}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(&path, format!("{rendered}\n")).map_err(|error| {
            anyhow!(
                "cannot write robust-policy receipt {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robustness_lattice_is_unique_and_sweeps_declared_failure_modes() {
        let scenarios = robustness_scenarios();
        let ids = scenarios
            .iter()
            .map(|scenario| scenario.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(scenarios.len(), 30);
        assert_eq!(ids.len(), 30);
        assert!(scenarios.iter().any(|scenario| scenario.forecast_bias == -0.30));
        assert!(scenarios.iter().any(|scenario| scenario.forecast_bias == 0.20));
        assert!(scenarios.iter().any(|scenario| scenario.sensor_dropout_period == Some(3)));
        assert!(scenarios.iter().any(|scenario| {
            scenario.heat_start_day.is_some() && scenario.drought_start_day.is_some()
        }));
        assert!(scenarios.iter().any(|scenario| scenario.daily_water_budget_plot_mm == 3.5));
    }

    #[test]
    fn robust_policy_receipt_is_deterministic_and_synthetic_only() {
        let first = serde_json::to_string(&build_receipt().unwrap()).unwrap();
        let second = serde_json::to_string(&build_receipt().unwrap()).unwrap();
        assert_eq!(first, second);
        let receipt = build_receipt().unwrap();
        assert_eq!(receipt.receipt_schema, "cropfoundry-robust-policy-receipt-v1");
        assert_eq!(receipt.benchmark_id, "CF-025");
        assert_eq!(receipt.scenario_count, 30);
        assert_eq!(receipt.execution_mode, "deterministic_synthetic_30_case_sensitivity_lattice");
        assert_eq!(receipt.policy_count, 5);
        assert_eq!(receipt.preference_count, 4);
        assert!(!receipt.calibrated_probability_claim);
        assert!(!receipt.biological_performance_claim);
        assert!(!receipt.training_authorized);
    }

    #[test]
    fn robust_policy_tail_sample_count_uses_two_of_thirty_sensitivity_cases() {
        let receipt = build_receipt().unwrap();
        assert!(
            receipt
                .policies
                .iter()
                .all(|policy| policy.tail_sample_count == 2)
        );
        assert!(receipt.policies.iter().all(|policy| {
            policy.cvar95_stress_exposure_proxy.is_finite()
                && policy.p95_stress_exposure_proxy_nearest_rank.is_finite()
                && policy.cvar95_stress_exposure_proxy
                    >= policy.p95_stress_exposure_proxy_nearest_rank - 1e-9
        }));
    }

    #[test]
    fn robust_policy_rejects_different_scenario_sets_even_when_row_counts_match() {
        let scenarios = robustness_scenarios();
        let mut results = simulate_policy_suite(&scenarios).unwrap();
        let policy = results[0].policy.clone();
        let row = results.iter_mut().find(|row| row.policy == policy).unwrap();
        row.scenario_id = "unexpected-replacement-scenario".to_owned();
        assert!(build_policy_scores(&results).is_err());
    }

    #[test]
    fn robust_policy_receipt_exposes_each_case_and_exact_cvar_tail() {
        let receipt = build_receipt().unwrap();
        assert_eq!(receipt.scenario_policy_diagnostics_count, 150);
        assert_eq!(receipt.scenario_policy_diagnostics.len(), 150);
        let keys = receipt
            .scenario_policy_diagnostics
            .iter()
            .map(|row| (row.policy.clone(), row.scenario_id.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(keys.len(), 150);
        for score in &receipt.policies {
            let mut values = receipt
                .scenario_policy_diagnostics
                .iter()
                .filter(|row| row.policy == score.policy)
                .map(|row| (row.stress_exposure_proxy, row.scenario_id.clone()))
                .collect::<Vec<_>>();
            values.sort_by(|left, right| {
                left.0.total_cmp(&right.0).then_with(|| left.1.cmp(&right.1))
            });
            let tail = &values[values.len() - 2..];
            let expected_ids = tail
                .iter()
                .map(|(_, id)| id.clone())
                .collect::<std::collections::BTreeSet<_>>();
            let actual_ids = score
                .upper_tail_scenario_ids
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual_ids, expected_ids);
            let expected_cvar = (tail[0].0 + tail[1].0) / 2.0;
            assert!((score.cvar95_stress_exposure_proxy - expected_cvar).abs() <= 1e-9);
        }
    }

    #[test]
    fn robust_policy_pareto_and_preference_sweep_preserve_all_policies() {
        let receipt = build_receipt().unwrap();
        assert!(
            receipt
                .policies
                .iter()
                .any(|policy| policy.pareto_front_member)
        );
        assert!(
            receipt
                .preference_sweep
                .iter()
                .all(|item| item.ranking.len() == 5)
        );
        assert!(receipt.preference_sweep.iter().all(|item| {
            item.ranking
                .iter()
                .map(|score| score.rank)
                .collect::<Vec<_>>()
                == (1..=5).collect::<Vec<_>>()
        }));
        assert!(
            receipt
                .preference_sweep
                .iter()
                .all(|item| { (item.weights.values().sum::<f64>() - 1.0).abs() <= 1e-9 })
        );
        assert!(receipt.policies.iter().all(|policy| {
            policy.pareto_front_member == policy.dominated_by_policies.is_empty()
        }));
        assert!(receipt.policies.iter().all(|policy| {
            policy.dominated_by_policies.iter().all(|dominator| {
                receipt.policies.iter().any(|candidate| {
                    candidate.policy == *dominator && candidate.pareto_front_member
                })
            })
        }));
    }

    #[test]
    fn policy_scenario_pairing_and_hard_constraints_are_required() {
        let scenarios = robustness_scenarios();
        let results = simulate_policy_suite(&scenarios).unwrap();
        let scores = build_policy_scores(&results).unwrap();
        assert_eq!(scores.len(), 5);
        assert!(scores.iter().all(|score| score.scenario_count == 30));
        assert!(scores.iter().all(|score| {
            score.total_water_budget_violation_days == 0
                && score.total_per_plot_limit_violation_days == 0
        }));
    }
}
