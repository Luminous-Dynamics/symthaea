// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired, budget-censored statistical summaries for the synthetic campaign harness.
//!
//! This analysis module consumes evaluator reports only. It does not inspect oracle
//! implementation, call policy selection, or infer physical manufacturing effects.
use crate::{BENCHMARK_ID, EvaluationReport, Policy, ScenarioFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const NORMAL_95_Z: f64 = 1.96;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignEvaluationSample {
    pub paired_seed_index: u32,
    pub budget_units: u32,
    pub evaluation: EvaluationReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAggregate {
    pub scenario: ScenarioFamily,
    pub policy: Policy,
    pub trials: u32,
    pub discoveries: u32,
    pub discovery_rate: f64,
    /// Conditional on successful discovery; never reported without its denominator.
    pub mean_discovery_cost_when_found: Option<f64>,
    /// Found => observed cost; NotFound => preregistered budget horizon.
    pub restricted_mean_cost_to_discovery_units: f64,
    /// Normal-approximation interval for the restricted mean across independent paired seeds.
    pub restricted_mean_cost_ci95_lower: Option<f64>,
    pub restricted_mean_cost_ci95_upper: Option<f64>,
    /// Best utility averaged over all trials; a no-discovery trial contributes zero.
    pub mean_conforming_utility_per_trial: f64,
    /// Mean best utility conditional on having discovered a conforming result.
    pub mean_best_utility_when_found: Option<f64>,
    /// Best conforming throughput averaged over all trials; no discovery contributes zero.
    pub mean_conforming_throughput_per_trial: f64,
    pub mean_resource_units_per_trial: f64,
    pub full_rank_design_rate: f64,
    /// Baseline restricted mean / this policy's restricted mean; >1 means lower discovery cost.
    pub acceleration_factor_vs_fixed_design: Option<f64>,
    /// Utility per trial / fixed-design utility per trial, independent of discovery rate.
    pub enhancement_factor_vs_fixed_design: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedPolicyComparison {
    pub scenario: ScenarioFamily,
    pub baseline_policy: Policy,
    pub candidate_policy: Policy,
    pub paired_seeds: u32,
    /// Candidate minus baseline; negative means the candidate used less budget before discovery.
    pub mean_paired_delta_restricted_cost_units: f64,
    pub paired_delta_cost_ci95_lower: Option<f64>,
    pub paired_delta_cost_ci95_upper: Option<f64>,
    /// Candidate minus baseline utility per trial; positive favors candidate.
    pub mean_paired_delta_utility_per_trial: f64,
    pub paired_delta_utility_ci95_lower: Option<f64>,
    pub paired_delta_utility_ci95_upper: Option<f64>,
    pub interval_excludes_zero_for_cost: Option<bool>,
    pub acceleration_factor: Option<f64>,
    pub enhancement_factor: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub benchmark_id: String,
    pub analysis_version: String,
    pub policy_aggregates: Vec<PolicyAggregate>,
    pub paired_comparisons_vs_fixed_design: Vec<PairedPolicyComparison>,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct TrialMetric {
    seed: u32,
    budget: u32,
    restricted_cost: f64,
    successful_discovery_cost: Option<f64>,
    utility_per_trial: f64,
    utility_if_found: Option<f64>,
    throughput_per_trial: f64,
    resource_units: f64,
    full_rank: bool,
}

pub fn summarize_paired_campaigns(
    samples: &[CampaignEvaluationSample],
) -> Result<BenchmarkSummary, String> {
    if samples.is_empty() {
        return Err("cannot summarize an empty campaign set".into());
    }

    let mut groups: BTreeMap<(ScenarioFamily, Policy), Vec<TrialMetric>> = BTreeMap::new();
    let mut paired_inputs: BTreeMap<(ScenarioFamily, u32), (String, u32, BTreeSet<Policy>)> =
        BTreeMap::new();

    for sample in samples {
        let report = &sample.evaluation;
        if report.benchmark_id != BENCHMARK_ID {
            return Err("evaluation report has an unexpected benchmark identity".into());
        }
        if report.manifest_digest.trim().is_empty() {
            return Err("evaluation report is missing a manifest digest".into());
        }
        if sample.budget_units == 0 || report.total_cost_units > sample.budget_units {
            return Err("evaluation report exceeds or lacks the declared budget".into());
        }
        if report
            .first_conforming_cost_units
            .is_some_and(|cost| cost == 0 || cost > report.total_cost_units)
        {
            return Err("discovery cost exceeds total campaign cost".into());
        }
        let has_discovery = report.first_conforming_cost_units.is_some();
        if report.best_conforming_utility.is_some() != has_discovery
            || report.max_conforming_throughput_units.is_some() != has_discovery
        {
            return Err("conforming utility/throughput does not match discovery disposition".into());
        }

        let paired_key = (report.scenario, sample.paired_seed_index);
        let entry = paired_inputs.entry(paired_key).or_insert_with(|| {
            (report.manifest_digest.clone(), sample.budget_units, BTreeSet::new())
        });
        if entry.0 != report.manifest_digest || entry.1 != sample.budget_units {
            return Err("paired policy arms have different manifest identity or budget".into());
        }
        if !entry.2.insert(report.policy) {
            return Err("duplicate policy result for the same scenario and paired seed".into());
        }

        let success_cost = report.first_conforming_cost_units.map(f64::from);
        let restricted_cost = success_cost.unwrap_or(f64::from(sample.budget_units));
        let utility_if_found = report.best_conforming_utility.map(|value| value as f64);
        let utility_per_trial = utility_if_found.unwrap_or(0.0);
        let throughput_per_trial = report
            .max_conforming_throughput_units
            .map(f64::from)
            .unwrap_or(0.0);
        groups
            .entry((report.scenario, report.policy))
            .or_default()
            .push(TrialMetric {
                seed: sample.paired_seed_index,
                budget: sample.budget_units,
                restricted_cost,
                successful_discovery_cost: success_cost,
                utility_per_trial,
                utility_if_found,
                throughput_per_trial,
                resource_units: f64::from(report.total_resource_units),
                full_rank: report.factor_design_full_rank,
            });
    }

    let scenarios: BTreeSet<ScenarioFamily> =
        samples.iter().map(|sample| sample.evaluation.scenario).collect();
    for scenario in &scenarios {
        let expected_seeds: BTreeSet<u32> = paired_inputs
            .iter()
            .filter(|((seen_scenario, _), _)| seen_scenario == scenario)
            .map(|((_, seed), _)| *seed)
            .collect();
        for policy in Policy::ALL {
            let actual = groups
                .get(&(*scenario, policy))
                .ok_or_else(|| format!("missing policy arm {policy:?} for {scenario:?}"))?;
            let actual_seeds: BTreeSet<u32> = actual.iter().map(|trial| trial.seed).collect();
            if actual_seeds != expected_seeds {
                return Err(format!(
                    "policy arm {policy:?} has unmatched paired seeds for {scenario:?}"
                ));
            }
        }
        for seed in &expected_seeds {
            let policies = &paired_inputs[&(*scenario, *seed)].2;
            if Policy::ALL.iter().any(|policy| !policies.contains(policy)) {
                return Err(format!("incomplete policy set for {scenario:?}, seed {seed}"));
            }
        }
    }

    let mut policy_aggregates = Vec::new();
    for scenario in &scenarios {
        let baseline_trials = &groups[&(*scenario, Policy::FixedSpaceFilling)];
        let baseline_costs: Vec<f64> = baseline_trials
            .iter()
            .map(|trial| trial.restricted_cost)
            .collect();
        let baseline_utilities: Vec<f64> = baseline_trials
            .iter()
            .map(|trial| trial.utility_per_trial)
            .collect();
        let baseline_mean_cost = mean(&baseline_costs);
        let baseline_mean_utility = mean(&baseline_utilities);

        for policy in Policy::ALL {
            let trials = &groups[&(*scenario, policy)];
            let costs: Vec<f64> = trials.iter().map(|trial| trial.restricted_cost).collect();
            let found_costs: Vec<f64> = trials
                .iter()
                .filter_map(|trial| trial.successful_discovery_cost)
                .collect();
            let utilities_all: Vec<f64> = trials
                .iter()
                .map(|trial| trial.utility_per_trial)
                .collect();
            let utilities_found: Vec<f64> = trials
                .iter()
                .filter_map(|trial| trial.utility_if_found)
                .collect();
            let throughput_all: Vec<f64> = trials
                .iter()
                .map(|trial| trial.throughput_per_trial)
                .collect();
            let resources: Vec<f64> = trials.iter().map(|trial| trial.resource_units).collect();
            let full_rank_count = trials.iter().filter(|trial| trial.full_rank).count() as f64;
            let max_budget = trials
                .iter()
                .map(|trial| f64::from(trial.budget))
                .fold(0.0, f64::max);
            let restricted_cost_interval = mean_interval(&costs)
                .map(|(lower, upper)| (lower.max(0.0), upper.min(max_budget)));
            let mean_cost = mean(&costs);
            let mean_utility = mean(&utilities_all);
            policy_aggregates.push(PolicyAggregate {
                scenario: *scenario,
                policy,
                trials: trials.len() as u32,
                discoveries: found_costs.len() as u32,
                discovery_rate: found_costs.len() as f64 / trials.len() as f64,
                mean_discovery_cost_when_found: if found_costs.is_empty() {
                    None
                } else {
                    Some(mean(&found_costs))
                },
                restricted_mean_cost_to_discovery_units: mean_cost,
                restricted_mean_cost_ci95_lower: restricted_cost_interval.map(|interval| interval.0),
                restricted_mean_cost_ci95_upper: restricted_cost_interval.map(|interval| interval.1),
                mean_conforming_utility_per_trial: mean_utility,
                mean_best_utility_when_found: if utilities_found.is_empty() {
                    None
                } else {
                    Some(mean(&utilities_found))
                },
                mean_conforming_throughput_per_trial: mean(&throughput_all),
                mean_resource_units_per_trial: mean(&resources),
                full_rank_design_rate: full_rank_count / trials.len() as f64,
                acceleration_factor_vs_fixed_design: if mean_cost > 0.0 {
                    Some(baseline_mean_cost / mean_cost)
                } else {
                    None
                },
                enhancement_factor_vs_fixed_design: if baseline_mean_utility > 0.0 {
                    Some(mean_utility / baseline_mean_utility)
                } else {
                    None
                },
            });
        }
    }

    let mut paired_comparisons = Vec::new();
    for scenario in &scenarios {
        let baseline_trials = trials_by_seed(&groups[&(*scenario, Policy::FixedSpaceFilling)]);
        for policy in Policy::ALL {
            let candidate_trials = trials_by_seed(&groups[&(*scenario, policy)]);
            let seeds: Vec<u32> = baseline_trials.keys().copied().collect();
            let cost_deltas: Vec<f64> = seeds
                .iter()
                .map(|seed| {
                    candidate_trials[seed].restricted_cost - baseline_trials[seed].restricted_cost
                })
                .collect();
            let utility_deltas: Vec<f64> = seeds
                .iter()
                .map(|seed| {
                    candidate_trials[seed].utility_per_trial
                        - baseline_trials[seed].utility_per_trial
                })
                .collect();
            let mean_cost_delta = mean(&cost_deltas);
            let cost_delta_interval = mean_interval(&cost_deltas);
            let mean_utility_delta = mean(&utility_deltas);
            let utility_delta_interval = mean_interval(&utility_deltas);
            let baseline_mean_cost = mean(
                &baseline_trials
                    .values()
                    .map(|trial| trial.restricted_cost)
                    .collect::<Vec<_>>(),
            );
            let candidate_mean_cost = mean(
                &candidate_trials
                    .values()
                    .map(|trial| trial.restricted_cost)
                    .collect::<Vec<_>>(),
            );
            let baseline_mean_utility = mean(
                &baseline_trials
                    .values()
                    .map(|trial| trial.utility_per_trial)
                    .collect::<Vec<_>>(),
            );
            let candidate_mean_utility = mean(
                &candidate_trials
                    .values()
                    .map(|trial| trial.utility_per_trial)
                    .collect::<Vec<_>>(),
            );

            paired_comparisons.push(PairedPolicyComparison {
                scenario: *scenario,
                baseline_policy: Policy::FixedSpaceFilling,
                candidate_policy: policy,
                paired_seeds: seeds.len() as u32,
                mean_paired_delta_restricted_cost_units: mean_cost_delta,
                paired_delta_cost_ci95_lower: cost_delta_interval.map(|interval| interval.0),
                paired_delta_cost_ci95_upper: cost_delta_interval.map(|interval| interval.1),
                mean_paired_delta_utility_per_trial: mean_utility_delta,
                paired_delta_utility_ci95_lower: utility_delta_interval.map(|interval| interval.0),
                paired_delta_utility_ci95_upper: utility_delta_interval.map(|interval| interval.1),
                interval_excludes_zero_for_cost: cost_delta_interval
                    .map(|(lower, upper)| lower > 0.0 || upper < 0.0),
                acceleration_factor: if candidate_mean_cost > 0.0 {
                    Some(baseline_mean_cost / candidate_mean_cost)
                } else {
                    None
                },
                enhancement_factor: if baseline_mean_utility > 0.0 {
                    Some(candidate_mean_utility / baseline_mean_utility)
                } else {
                    None
                },
            });
        }
    }

    Ok(BenchmarkSummary {
        benchmark_id: BENCHMARK_ID.into(),
        analysis_version: "paired-censor-aware-v1".into(),
        policy_aggregates,
        paired_comparisons_vs_fixed_design: paired_comparisons,
        assumptions: vec![
            "A campaign without a conforming discovery contributes the predeclared budget horizon to restricted mean cost-to-discovery; it remains a failure in the separate discovery-rate metric.".into(),
            "Cost and utility intervals are 95% normal-approximation intervals across paired seeds; they are descriptive and are not a substitute for preregistration, model checks, or domain-specific inference.".into(),
            "Acceleration factor = fixed-design restricted mean cost-to-discovery / candidate restricted mean cost-to-discovery; values above 1 favor the candidate on this bounded synthetic metric.".into(),
            "Enhancement factor = candidate mean conforming utility per trial / fixed-design mean conforming utility per trial; no-discovery trials contribute zero utility. Discovery rate, throughput, resources and guardrails remain separately visible.".into(),
            "All quantities are synthetic benchmark units. No result establishes physical manufacturing performance, safety, or cost savings.".into(),
        ],
    })
}

fn trials_by_seed(trials: &[TrialMetric]) -> BTreeMap<u32, TrialMetric> {
    trials.iter().map(|trial| (trial.seed, *trial)).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_interval(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let center = mean(values);
    let sample_variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let standard_error = (sample_variance / values.len() as f64).sqrt();
    let half_width = NORMAL_95_Z * standard_error;
    Some((center - half_width, center + half_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(
        scenario: ScenarioFamily,
        policy: Policy,
        seed: u32,
        discovery_cost: Option<u32>,
        utility: Option<i64>,
        total_cost: u32,
    ) -> CampaignEvaluationSample {
        CampaignEvaluationSample {
            paired_seed_index: seed,
            budget_units: 10,
            evaluation: EvaluationReport {
                benchmark_id: BENCHMARK_ID.into(),
                scenario,
                policy,
                manifest_digest: format!("{scenario:?}-seed-{seed}"),
                experiments: 1,
                total_cost_units: total_cost,
                first_conforming_cost_units: discovery_cost,
                best_conforming_utility: utility,
                max_conforming_throughput_units: utility.map(|_| 5),
                total_resource_units: 2,
                blocked_catalog_candidates: 2,
                unknown_catalog_candidates: 1,
                factor_design_full_rank: true,
            },
        }
    }

    fn complete_pair_set(
        scenario: ScenarioFamily,
        seed: u32,
        fixed: (Option<u32>, Option<i64>, u32),
        candidate: (Option<u32>, Option<i64>, u32),
    ) -> Vec<CampaignEvaluationSample> {
        Policy::ALL
            .into_iter()
            .map(|policy| {
                let (discovery, utility, cost) = if policy == Policy::FixedSpaceFilling {
                    fixed
                } else {
                    candidate
                };
                report(scenario, policy, seed, discovery, utility, cost)
            })
            .collect()
    }

    #[test]
    fn not_found_trials_are_censored_at_budget_and_count_as_no_discovery() {
        let samples = complete_pair_set(
            ScenarioFamily::SmoothResponse,
            0,
            (None, None, 10),
            (Some(4), Some(1200), 4),
        );
        let summary = summarize_paired_campaigns(&samples).unwrap();
        let fixed = summary.policy_aggregates.iter()
            .find(|row| row.policy == Policy::FixedSpaceFilling).unwrap();
        assert_eq!(fixed.discovery_rate, 0.0);
        assert_eq!(fixed.restricted_mean_cost_to_discovery_units, 10.0);
        assert_eq!(fixed.mean_conforming_utility_per_trial, 0.0);
    }

    #[test]
    fn paired_delta_uses_same_seed_and_budget_censoring() {
        let mut samples = complete_pair_set(
            ScenarioFamily::SmoothResponse,
            0,
            (Some(8), Some(1000), 8),
            (Some(4), Some(1100), 4),
        );
        samples.extend(complete_pair_set(
            ScenarioFamily::SmoothResponse,
            1,
            (None, None, 10),
            (Some(6), Some(1050), 6),
        ));
        let summary = summarize_paired_campaigns(&samples).unwrap();
        let comparison = summary.paired_comparisons_vs_fixed_design.iter()
            .find(|row| row.candidate_policy == Policy::DecisionAware).unwrap();
        assert_eq!(comparison.paired_seeds, 2);
        // Deltas are 4-8 = -4 and 6-10 = -4.
        assert_eq!(comparison.mean_paired_delta_restricted_cost_units, -4.0);
        assert_eq!(comparison.paired_delta_cost_ci95_lower, Some(-4.0));
        assert_eq!(comparison.paired_delta_cost_ci95_upper, Some(-4.0));
    }

    #[test]
    fn one_seed_does_not_emit_a_degenerate_confidence_interval() {
        let samples = complete_pair_set(
            ScenarioFamily::SmoothResponse,
            0,
            (Some(5), Some(1000), 5),
            (Some(4), Some(1100), 4),
        );
        let summary = summarize_paired_campaigns(&samples).unwrap();
        let fixed = summary
            .policy_aggregates
            .iter()
            .find(|row| row.policy == Policy::FixedSpaceFilling)
            .unwrap();
        assert_eq!(fixed.restricted_mean_cost_ci95_lower, None);
        assert_eq!(fixed.restricted_mean_cost_ci95_upper, None);

        let paired = summary
            .paired_comparisons_vs_fixed_design
            .iter()
            .find(|row| row.candidate_policy == Policy::DecisionAware)
            .unwrap();
        assert_eq!(paired.paired_delta_cost_ci95_lower, None);
        assert_eq!(paired.paired_delta_cost_ci95_upper, None);
        assert_eq!(paired.interval_excludes_zero_for_cost, None);
    }

    #[test]
    fn duplicate_arm_for_seed_is_rejected() {
        let mut samples = complete_pair_set(
            ScenarioFamily::SmoothResponse,
            0,
            (Some(5), Some(1000), 5),
            (Some(4), Some(1100), 4),
        );
        samples.push(samples[0].clone());
        assert!(summarize_paired_campaigns(&samples).unwrap_err().contains("duplicate policy"));
    }

    #[test]
    fn missing_policy_arm_is_rejected() {
        let mut samples = complete_pair_set(
            ScenarioFamily::SmoothResponse,
            0,
            (Some(5), Some(1000), 5),
            (Some(4), Some(1100), 4),
        );
        samples.pop();
        assert!(summarize_paired_campaigns(&samples).unwrap_err().contains("missing policy arm"));
    }
}
