//! Oracle-backed, deterministic synthetic benchmarks for experiment selectors.
//!
//! These scenarios are designed to test selector behavior when the correct
//! answer is known by construction. They do not model a real energy converter,
//! validate a physical solver, or support physical-superiority claims.

use crate::experiment_selection::{
    rank_experiments, DecisionRelevance, ExperimentProposal, RejectedExperiment,
    ResourceBasis, SafetyReview, SelectionError, SelectionPolicy,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// A synthetic experiment with separate proposer estimates and oracle-known value.
///
/// `oracle_information_gain` is interpreted as the probability that this
/// experiment independently resolves the benchmark's hidden decision. This is
/// deliberately a toy oracle model, not a thermodynamic quantity or calibrated
/// estimate of real-world information gain.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticExperiment {
    pub proposal: ExperimentProposal,
    /// One-dimensional normalized design-space coordinate in [0, 1].
    pub design_coordinate: f64,
    /// Oracle-known resolution probability in [0, 1].
    pub oracle_information_gain: f64,
}

/// One known-answer synthetic decision task with a fixed resource budget.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticScenario {
    pub scenario_id: String,
    pub resource_budget: f64,
    /// Decision is considered resolved once the independent-information utility
    /// reaches this threshold. It is an evaluation rule, not a physical claim.
    pub resolution_threshold: f64,
    pub experiments: Vec<SyntheticExperiment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExperimentSelectionStrategy {
    PhaseA,
    Random { seed: u64 },
    CheapestFirst,
    SpaceFilling,
    OracleUpperBound,
}

impl ExperimentSelectionStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::PhaseA => "phase_a",
            Self::Random { .. } => "random",
            Self::CheapestFirst => "cheapest_first",
            Self::SpaceFilling => "space_filling",
            Self::OracleUpperBound => "oracle_upper_bound",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioBenchmarkResult {
    pub scenario_id: String,
    pub strategy: ExperimentSelectionStrategy,
    pub selected_experiment_ids: Vec<String>,
    pub rejected_experiments: Vec<RejectedExperiment>,
    pub resource_budget: f64,
    pub resource_used: f64,
    pub budget_used_fraction: f64,
    pub achieved_utility: f64,
    pub oracle_utility: f64,
    /// Exact minimum cost of any admissible subset reaching the registered
    /// resolution threshold, if such a subset fits the budget.
    pub oracle_minimum_cost_to_resolve: Option<f64>,
    pub decision_regret: f64,
    pub resolved: bool,
    pub oracle_resolved: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategyBenchmarkSummary {
    pub strategy: ExperimentSelectionStrategy,
    pub scenario_count: usize,
    pub mean_decision_regret: f64,
    pub median_decision_regret: f64,
    pub maximum_decision_regret: f64,
    pub resolution_rate: f64,
    pub oracle_resolution_rate: f64,
    pub mean_budget_used_fraction: f64,
    /// Mean resource cost among scenarios the selector actually resolved.
    pub mean_cost_when_resolved: Option<f64>,
    /// Median resource cost among scenarios the selector actually resolved.
    pub median_cost_when_resolved: Option<f64>,
    /// Oracle lower bound on cost to resolution, averaged over resolvable scenarios.
    pub mean_oracle_minimum_cost_to_resolve: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkSuiteReport {
    pub scenarios: usize,
    pub results: Vec<ScenarioBenchmarkResult>,
    pub summaries: Vec<StrategyBenchmarkSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkError {
    pub reason: String,
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for BenchmarkError {}

impl From<SelectionError> for BenchmarkError {
    fn from(value: SelectionError) -> Self {
        Self { reason: value.reason }
    }
}

/// Run each strategy on each scenario using exactly the same admissible proposals.
///
/// The oracle strategy exhaustively evaluates every subset for scenarios with at
/// most 16 admissible experiments. It is an upper bound only, not a deployable
/// selector. Strategies that cannot afford a proposal skip it and continue to
/// the next feasible one; all resource use is charged against the same budget.
pub fn run_selection_benchmark(
    policy: &SelectionPolicy,
    scenarios: &[SyntheticScenario],
    strategies: &[ExperimentSelectionStrategy],
) -> Result<BenchmarkSuiteReport, BenchmarkError> {
    if scenarios.is_empty() {
        return Err(BenchmarkError {
            reason: "benchmark suite must contain at least one scenario".into(),
        });
    }
    if strategies.is_empty() {
        return Err(BenchmarkError {
            reason: "benchmark suite must contain at least one strategy".into(),
        });
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(BenchmarkError {
                reason: format!("duplicate scenario ID {:?}", scenario.scenario_id),
            });
        }
        validate_scenario(scenario)?;
    }

    let mut results = Vec::with_capacity(scenarios.len() * strategies.len());
    for scenario in scenarios {
        let oracle = evaluate_scenario(policy, scenario, ExperimentSelectionStrategy::OracleUpperBound)?;
        for strategy in strategies {
            let mut result = evaluate_scenario(policy, scenario, *strategy)?;
            result.oracle_utility = oracle.achieved_utility;
            result.oracle_resolved = oracle.resolved;
            result.decision_regret = (oracle.achieved_utility - result.achieved_utility).max(0.0);
            results.push(result);
        }
    }

    let mut summaries = Vec::with_capacity(strategies.len());
    for strategy in strategies {
        let strategy_results: Vec<&ScenarioBenchmarkResult> =
            results.iter().filter(|result| result.strategy == *strategy).collect();
        summaries.push(summarize_strategy(*strategy, &strategy_results)?);
    }

    Ok(BenchmarkSuiteReport {
        scenarios: scenarios.len(),
        results,
        summaries,
    })
}

/// Generate reproducible scenario families for automated selector regression tests.
///
/// Every fifth scenario intentionally reverses the claimed uncertainty-reduction
/// proxy relative to oracle value, stress-testing proxy exploitation. Other
/// scenarios mix informative, noisy, and uninformative proposer attributes.
/// Generated cases are synthetic and must not be presented as physical evidence.
pub fn generate_synthetic_scenarios(
    count: usize,
    seed: u64,
    policy: &SelectionPolicy,
) -> Vec<SyntheticScenario> {
    let mut rng = SplitMix64::new(seed);
    let mut scenarios = Vec::with_capacity(count);

    for scenario_index in 0..count {
        let scenario_id = format!("synthetic-{seed:016x}-{scenario_index:06}");
        let mut raw_values: Vec<f64> = (0..6).map(|_| 0.05 + 0.90 * rng.unit()).collect();
        if scenario_index % 5 == 0 {
            raw_values.reverse();
        }

        let mut experiments = Vec::with_capacity(6);
        for index in 0..6 {
            let oracle_gain = raw_values[index];
            let noise = rng.unit();
            let declared_reduction = match scenario_index % 4 {
                0 => (0.75 * oracle_gain + 0.25 * noise).clamp(0.01, 1.0),
                1 => (0.75 * (1.0 - oracle_gain) + 0.25 * noise).clamp(0.01, 1.0),
                _ => (0.15 + 0.85 * noise).clamp(0.01, 1.0),
            };
            let relevance_bucket = match scenario_index % 4 {
                0 => (oracle_gain * 4.999) as u8,
                1 => ((1.0 - oracle_gain) * 4.999) as u8,
                2 => (rng.unit() * 4.999) as u8,
                _ => ((0.5 * oracle_gain + 0.5 * rng.unit()) * 4.999) as u8,
            }
            .min(4);
            let decision_relevance = match relevance_bucket {
                0 => DecisionRelevance::Exploratory,
                1 => DecisionRelevance::CharacterizesNamedUncertainty,
                2 => DecisionRelevance::DiscriminatesCompetingModels,
                3 => DecisionRelevance::CanFalsifyFavoredHypothesis,
                _ => DecisionRelevance::CanChangeFeasibilityOrDesignDecision,
            };
            let cost = 1.0 + (rng.next_u64() % 8) as f64;
            let proposal = ExperimentProposal {
                experiment_id: format!("{scenario_id}-e{index:02}"),
                decision_question: format!("Which model family governs decision {scenario_index}?"),
                measurable_observable: "normalized net useful energy response".into(),
                competing_hypotheses: vec![
                    "loss is dominated by thermal rejection".into(),
                    "loss is dominated by auxiliary demand".into(),
                ],
                decision_relevance,
                expected_uncertainty_reduction_fraction: declared_reduction,
                estimated_resource_cost: crate::experiment_selection::ResourceCost {
                    basis: policy.resource_basis,
                    unit_id: policy.resource_unit_id.clone(),
                    amount: cost,
                },
                reproducibility: match index % 4 {
                    0 => crate::experiment_selection::Reproducibility::IndependentReplicationPlanned,
                    1 => crate::experiment_selection::Reproducibility::IndependentlyCrossCheckable,
                    2 => crate::experiment_selection::Reproducibility::ReplayableWithDeclaredArtifacts,
                    _ => crate::experiment_selection::Reproducibility::OneOff,
                },
                registered_before_observation: true,
                safety_review: SafetyReview::Approved,
                operating_envelope_registered: true,
                abort_criteria_registered: true,
                provenance_plan_complete: true,
            };
            experiments.push(SyntheticExperiment {
                proposal,
                design_coordinate: (index as f64 + 0.5 * rng.unit()) / 6.0,
                oracle_information_gain: oracle_gain,
            });
        }

        let total_cost: f64 = experiments
            .iter()
            .map(|experiment| experiment.proposal.estimated_resource_cost.amount)
            .sum();
        let budget_fraction = 0.30 + 0.40 * rng.unit();
        scenarios.push(SyntheticScenario {
            scenario_id,
            resource_budget: (total_cost * budget_fraction).max(1.0),
            resolution_threshold: 0.80,
            experiments,
        });
    }

    scenarios
}

/// Evaluate a single strategy against the oracle for one scenario.
pub fn evaluate_scenario(
    policy: &SelectionPolicy,
    scenario: &SyntheticScenario,
    strategy: ExperimentSelectionStrategy,
) -> Result<ScenarioBenchmarkResult, BenchmarkError> {
    validate_scenario(scenario)?;
    let proposals: Vec<ExperimentProposal> =
        scenario.experiments.iter().map(|experiment| experiment.proposal.clone()).collect();
    let ranking = rank_experiments(policy, &proposals)?;
    let eligible_ids: BTreeSet<String> =
        ranking.eligible.iter().map(|item| item.proposal.experiment_id.clone()).collect();
    let eligible: Vec<&SyntheticExperiment> = scenario
        .experiments
        .iter()
        .filter(|experiment| eligible_ids.contains(&experiment.proposal.experiment_id))
        .collect();
    if eligible.is_empty() {
        return Err(BenchmarkError {
            reason: format!("scenario {:?} has no eligible experiments", scenario.scenario_id),
        });
    }

    let oracle_utility = oracle_subset(&eligible, scenario.resource_budget).1;
    let oracle_minimum_cost_to_resolve = minimum_cost_to_resolve(
        &eligible,
        scenario.resource_budget,
        scenario.resolution_threshold,
    );
    let selected = match strategy {
        ExperimentSelectionStrategy::OracleUpperBound => {
            oracle_subset(&eligible, scenario.resource_budget).0
        }
        ExperimentSelectionStrategy::PhaseA
        | ExperimentSelectionStrategy::Random { .. }
        | ExperimentSelectionStrategy::CheapestFirst
        | ExperimentSelectionStrategy::SpaceFilling => {
            select_greedy(&eligible, &ranking, scenario, strategy)
        }
    };

    let mut selected_ids: Vec<String> = selected
        .iter()
        .map(|experiment| experiment.proposal.experiment_id.clone())
        .collect();
    selected_ids.sort();

    let resource_used: f64 = selected
        .iter()
        .map(|experiment| experiment.proposal.estimated_resource_cost.amount)
        .sum();
    let achieved_utility = utility(&selected);
    let budget_used_fraction = resource_used / scenario.resource_budget;
    let regret = (oracle_utility - achieved_utility).max(0.0);
    let resolved = achieved_utility >= scenario.resolution_threshold;
    let oracle_resolved = oracle_utility >= scenario.resolution_threshold;

    Ok(ScenarioBenchmarkResult {
        scenario_id: scenario.scenario_id.clone(),
        strategy,
        selected_experiment_ids: selected_ids,
        rejected_experiments: ranking.rejected,
        resource_budget: scenario.resource_budget,
        resource_used,
        budget_used_fraction,
        achieved_utility,
        oracle_utility,
        oracle_minimum_cost_to_resolve,
        decision_regret: regret,
        resolved,
        oracle_resolved,
    })
}

fn validate_scenario(scenario: &SyntheticScenario) -> Result<(), BenchmarkError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(BenchmarkError { reason: "scenario ID must be non-empty".into() });
    }
    if scenario.experiments.is_empty() {
        return Err(BenchmarkError { reason: "scenario must have at least one experiment".into() });
    }
    if !scenario.resource_budget.is_finite() || scenario.resource_budget <= 0.0 {
        return Err(BenchmarkError { reason: "resource budget must be finite and positive".into() });
    }
    if !scenario.resolution_threshold.is_finite()
        || !(0.0..=1.0).contains(&scenario.resolution_threshold)
    {
        return Err(BenchmarkError { reason: "resolution threshold must be in [0, 1]".into() });
    }
    if scenario.experiments.len() > 16 {
        return Err(BenchmarkError {
            reason: "oracle subset enumeration supports at most 16 experiments per scenario".into(),
        });
    }
    let mut ids = BTreeSet::new();
    for experiment in &scenario.experiments {
        if !ids.insert(experiment.proposal.experiment_id.as_str()) {
            return Err(BenchmarkError {
                reason: format!("duplicate experiment ID {:?}", experiment.proposal.experiment_id),
            });
        }
        if !experiment.design_coordinate.is_finite()
            || !(0.0..=1.0).contains(&experiment.design_coordinate)
        {
            return Err(BenchmarkError {
                reason: format!(
                    "experiment {:?} design coordinate must be finite and in [0, 1]",
                    experiment.proposal.experiment_id
                ),
            });
        }
        if !experiment.oracle_information_gain.is_finite()
            || !(0.0..=1.0).contains(&experiment.oracle_information_gain)
        {
            return Err(BenchmarkError {
                reason: format!(
                    "experiment {:?} oracle value must be finite and in [0, 1]",
                    experiment.proposal.experiment_id
                ),
            });
        }
    }
    Ok(())
}

fn select_greedy<'a>(
    eligible: &[&'a SyntheticExperiment],
    ranking: &crate::experiment_selection::RankingReport,
    scenario: &SyntheticScenario,
    strategy: ExperimentSelectionStrategy,
) -> Vec<&'a SyntheticExperiment> {
    let rank_position: BTreeMap<&str, usize> = ranking
        .eligible
        .iter()
        .enumerate()
        .map(|(index, ranked)| (ranked.proposal.experiment_id.as_str(), index))
        .collect();
    let mut remaining = eligible.to_vec();
    let mut selected: Vec<&SyntheticExperiment> = Vec::new();
    let mut remaining_budget = scenario.resource_budget;
    let mut rng = match strategy {
        ExperimentSelectionStrategy::Random { seed } => {
            Some(SplitMix64::new(seed ^ stable_hash(&scenario.scenario_id)))
        }
        _ => None,
    };

    loop {
        // Real selectors should stop paying when the registered decision rule
        // is satisfied. The exhaustive oracle upper-bound strategy is separate.
        if utility(&selected) >= scenario.resolution_threshold {
            break;
        }
        let affordable: Vec<usize> = remaining
            .iter()
            .enumerate()
            .filter(|(_, experiment)| {
                experiment.proposal.estimated_resource_cost.amount <= remaining_budget
            })
            .map(|(index, _)| index)
            .collect();
        if affordable.is_empty() {
            break;
        }

        let chosen_index = match strategy {
            ExperimentSelectionStrategy::PhaseA => *affordable
                .iter()
                .min_by_key(|index| {
                    rank_position
                        .get(remaining[**index].proposal.experiment_id.as_str())
                        .copied()
                        .unwrap_or(usize::MAX)
                })
                .unwrap(),
            ExperimentSelectionStrategy::Random { .. } => {
                affordable[rng.as_mut().unwrap().next_usize(affordable.len())]
            }
            ExperimentSelectionStrategy::CheapestFirst => *affordable
                .iter()
                .min_by(|left, right| {
                    remaining[**left]
                        .proposal
                        .estimated_resource_cost
                        .amount
                        .total_cmp(&remaining[**right].proposal.estimated_resource_cost.amount)
                        .then_with(|| {
                            remaining[**left]
                                .proposal
                                .experiment_id
                                .cmp(&remaining[**right].proposal.experiment_id)
                        })
                })
                .unwrap(),
            ExperimentSelectionStrategy::SpaceFilling => *affordable
                .iter()
                .max_by(|left, right| {
                    let left_distance = space_filling_distance(remaining[**left], &selected);
                    let right_distance = space_filling_distance(remaining[**right], &selected);
                    left_distance
                        .total_cmp(&right_distance)
                        // max_by chooses the right on Equal; reverse IDs so smaller ID wins.
                        .then_with(|| {
                            remaining[**right]
                                .proposal
                                .experiment_id
                                .cmp(&remaining[**left].proposal.experiment_id)
                        })
                })
                .unwrap(),
            ExperimentSelectionStrategy::OracleUpperBound => unreachable!("oracle uses exhaustive search"),
        };
        let chosen = remaining.remove(chosen_index);
        remaining_budget -= chosen.proposal.estimated_resource_cost.amount;
        selected.push(chosen);
    }
    selected
}

fn space_filling_distance(
    candidate: &SyntheticExperiment,
    selected: &[&SyntheticExperiment],
) -> f64 {
    if selected.is_empty() {
        return (candidate.design_coordinate - 0.5).abs();
    }
    selected
        .iter()
        .map(|prior| (candidate.design_coordinate - prior.design_coordinate).abs())
        .fold(f64::INFINITY, f64::min)
}

fn oracle_subset<'a>(
    eligible: &[&'a SyntheticExperiment],
    budget: f64,
) -> (Vec<&'a SyntheticExperiment>, f64) {
    let mut candidates = eligible.to_vec();
    candidates.sort_by(|left, right| {
        left.proposal.experiment_id.cmp(&right.proposal.experiment_id)
    });

    let subset_count = 1usize << candidates.len();
    let mut best = Vec::new();
    let mut best_utility = 0.0;
    let mut best_cost = 0.0;
    let mut best_ids = Vec::new();

    for mask in 0..subset_count {
        let mut subset = Vec::new();
        let mut cost = 0.0;
        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1usize << index) != 0 {
                subset.push(*candidate);
                cost += candidate.proposal.estimated_resource_cost.amount;
            }
        }
        if cost > budget {
            continue;
        }
        let score = utility(&subset);
        let ids: Vec<&str> = subset
            .iter()
            .map(|experiment| experiment.proposal.experiment_id.as_str())
            .collect();
        let is_better = score > best_utility + 1e-12
            || ((score - best_utility).abs() <= 1e-12
                && (cost < best_cost - 1e-12
                    || ((cost - best_cost).abs() <= 1e-12 && ids.as_slice() < best_ids.as_slice())));
        if is_better {
            best = subset;
            best_utility = score;
            best_cost = cost;
            best_ids = ids;
        }
    }
    (best, best_utility)
}

/// Exact lower bound on the resource cost needed to reach the threshold, if feasible.
fn minimum_cost_to_resolve(
    eligible: &[&SyntheticExperiment],
    budget: f64,
    threshold: f64,
) -> Option<f64> {
    let candidates = sorted_by_id(eligible);
    let mut best_cost: Option<f64> = None;
    let mut best_utility = f64::NEG_INFINITY;

    for mask in 0..(1usize << candidates.len()) {
        let mut subset = Vec::new();
        let mut cost = 0.0;
        for (index, candidate) in candidates.iter().enumerate() {
            if mask & (1usize << index) != 0 {
                subset.push(*candidate);
                cost += candidate.proposal.estimated_resource_cost.amount;
            }
        }
        if cost > budget {
            continue;
        }
        let score = utility(&subset);
        if score + 1e-12 < threshold {
            continue;
        }
        let better = best_cost.is_none_or(|current| cost < current - 1e-12)
            || (best_cost.is_some_and(|current| (cost - current).abs() <= 1e-12)
                && score > best_utility + 1e-12);
        if better {
            best_cost = Some(cost);
            best_utility = score;
        }
    }
    best_cost
}

fn sorted_by_id<'a>(eligible: &[&'a SyntheticExperiment]) -> Vec<&'a SyntheticExperiment> {
    let mut candidates = eligible.to_vec();
    candidates.sort_by(|left, right| {
        left.proposal.experiment_id.cmp(&right.proposal.experiment_id)
    });
    candidates
}

fn optional_mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn optional_median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

/// Independent evidence accumulates as 1 - product(1 - gain_i).
fn utility(experiments: &[&SyntheticExperiment]) -> f64 {
    1.0 - experiments
        .iter()
        .map(|experiment| 1.0 - experiment.oracle_information_gain)
        .product::<f64>()
}

fn summarize_strategy(
    strategy: ExperimentSelectionStrategy,
    results: &[&ScenarioBenchmarkResult],
) -> Result<StrategyBenchmarkSummary, BenchmarkError> {
    if results.is_empty() {
        return Err(BenchmarkError {
            reason: format!("no results available for strategy {}", strategy.label()),
        });
    }
    let mut regrets: Vec<f64> = results.iter().map(|result| result.decision_regret).collect();
    regrets.sort_by(f64::total_cmp);
    let median = if regrets.len() % 2 == 0 {
        (regrets[regrets.len() / 2 - 1] + regrets[regrets.len() / 2]) / 2.0
    } else {
        regrets[regrets.len() / 2]
    };
    let n = results.len() as f64;
    Ok(StrategyBenchmarkSummary {
        strategy,
        scenario_count: results.len(),
        mean_decision_regret: regrets.iter().sum::<f64>() / n,
        median_decision_regret: median,
        maximum_decision_regret: regrets.last().copied().unwrap_or(0.0),
        resolution_rate: results.iter().filter(|result| result.resolved).count() as f64 / n,
        oracle_resolution_rate: results.iter().filter(|result| result.oracle_resolved).count() as f64 / n,
        mean_budget_used_fraction: results.iter().map(|result| result.budget_used_fraction).sum::<f64>() / n,
        mean_cost_when_resolved: optional_mean(
            results.iter().filter(|result| result.resolved).map(|result| result.resource_used),
        ),
        median_cost_when_resolved: optional_median(
            results.iter().filter(|result| result.resolved).map(|result| result.resource_used).collect(),
        ),
        mean_oracle_minimum_cost_to_resolve: optional_mean(
            results.iter().filter_map(|result| result.oracle_minimum_cost_to_resolve),
        ),
    })
}

fn stable_hash(text: &str) -> u64 {
    text.as_bytes().iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

/// Generate the same scenario suite for the same seed and registered policy.
pub fn replay_synthetic_suite(count: usize, seed: u64, policy: &SelectionPolicy) -> Vec<SyntheticScenario> {
    generate_synthetic_scenarios(count, seed, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_selection::ResourceBasis;

    fn policy() -> SelectionPolicy {
        SelectionPolicy {
            policy_id: "synthetic-selector-benchmark-v1".into(),
            registered_before_queue_generation: true,
            resource_basis: ResourceBasis::ComputeTime,
            resource_unit_id: "s".into(),
        }
    }

    const STRATEGIES: [ExperimentSelectionStrategy; 5] = [
        ExperimentSelectionStrategy::PhaseA,
        ExperimentSelectionStrategy::Random { seed: 7 },
        ExperimentSelectionStrategy::CheapestFirst,
        ExperimentSelectionStrategy::SpaceFilling,
        ExperimentSelectionStrategy::OracleUpperBound,
    ];

    #[test]
    fn oracle_upper_bound_has_zero_regret_and_respects_budget() {
        let scenarios = generate_synthetic_scenarios(24, 0x5eed, &policy());
        let report = run_selection_benchmark(&policy(), &scenarios, &STRATEGIES).unwrap();
        let oracle: Vec<&ScenarioBenchmarkResult> = report
            .results
            .iter()
            .filter(|result| result.strategy == ExperimentSelectionStrategy::OracleUpperBound)
            .collect();

        assert_eq!(oracle.len(), 24);
        for result in oracle {
            assert!(result.decision_regret.abs() <= 1e-12);
            assert!(result.resource_used <= result.resource_budget + 1e-12);
        }
        assert_eq!(report.summaries.len(), STRATEGIES.len());
    }

    #[test]
    fn suite_generation_and_execution_are_replayable() {
        let first = generate_synthetic_scenarios(12, 12345, &policy());
        let second = replay_synthetic_suite(12, 12345, &policy());
        assert_eq!(first, second);

        let first_report = run_selection_benchmark(&policy(), &first, &STRATEGIES).unwrap();
        let second_report = run_selection_benchmark(&policy(), &second, &STRATEGIES).unwrap();
        assert_eq!(first_report, second_report);
    }

    #[test]
    fn deceptive_proxy_scenarios_expose_nonzero_phase_a_regret() {
        let scenarios = generate_synthetic_scenarios(100, 0xdec0, &policy());
        let report = run_selection_benchmark(
            &policy(),
            &scenarios,
            &[ExperimentSelectionStrategy::PhaseA, ExperimentSelectionStrategy::OracleUpperBound],
        )
        .unwrap();
        let phase_a: Vec<&ScenarioBenchmarkResult> = report
            .results
            .iter()
            .filter(|result| result.strategy == ExperimentSelectionStrategy::PhaseA)
            .collect();
        assert!(phase_a.iter().any(|result| result.decision_regret > 1e-9));
    }

    #[test]
    fn all_strategies_use_same_registered_budget_and_admissible_set() {
        let scenarios = generate_synthetic_scenarios(32, 0xabcd, &policy());
        let report = run_selection_benchmark(&policy(), &scenarios, &STRATEGIES).unwrap();
        for result in &report.results {
            assert!(result.resource_used <= result.resource_budget + 1e-12);
            assert!(result.budget_used_fraction.is_finite());
            assert!((0.0..=1.0).contains(&result.achieved_utility));
            assert!((0.0..=1.0).contains(&result.oracle_utility));
        }
    }

    #[test]
    fn phase_a_stops_spending_when_registered_resolution_threshold_is_reached() {
        let mut scenario = generate_synthetic_scenarios(1, 0x1234, &policy()).remove(0);
        scenario.resource_budget = 10.0;
        scenario.resolution_threshold = 0.80;
        for experiment in &mut scenario.experiments {
            experiment.proposal.decision_relevance = DecisionRelevance::Exploratory;
            experiment.proposal.expected_uncertainty_reduction_fraction = 0.10;
            experiment.proposal.estimated_resource_cost.amount = 8.0;
            experiment.oracle_information_gain = 0.05;
        }
        let first = &mut scenario.experiments[0];
        first.proposal.decision_relevance =
            DecisionRelevance::CanChangeFeasibilityOrDesignDecision;
        first.proposal.expected_uncertainty_reduction_fraction = 0.99;
        first.proposal.estimated_resource_cost.amount = 1.0;
        first.oracle_information_gain = 0.90;

        let result = evaluate_scenario(
            &policy(),
            &scenario,
            ExperimentSelectionStrategy::PhaseA,
        )
        .unwrap();

        assert!(result.resolved);
        assert_eq!(result.selected_experiment_ids, vec![first.proposal.experiment_id.clone()]);
        assert!((result.resource_used - 1.0).abs() <= 1e-12);
        assert_eq!(result.oracle_minimum_cost_to_resolve, Some(1.0));
    }

    #[test]
    fn reference_suite_runs_one_thousand_known_answer_scenarios() {
        let scenarios = generate_synthetic_scenarios(1_000, 0x51ec7, &policy());
        let report = run_selection_benchmark(&policy(), &scenarios, &STRATEGIES).unwrap();

        assert_eq!(report.scenarios, 1_000);
        assert_eq!(report.results.len(), 1_000 * STRATEGIES.len());
        assert_eq!(report.summaries.len(), STRATEGIES.len());
        for summary in &report.summaries {
            assert_eq!(summary.scenario_count, 1_000);
            assert!(summary.mean_decision_regret.is_finite());
            assert!((0.0..=1.0).contains(&summary.resolution_rate));
        }
    }

    #[test]
    fn rejects_oversized_or_duplicate_scenarios() {
        let mut scenarios = generate_synthetic_scenarios(1, 2, &policy());
        scenarios.push(scenarios[0].clone());
        assert!(run_selection_benchmark(&policy(), &scenarios, &STRATEGIES).is_err());

        let mut too_many = generate_synthetic_scenarios(1, 2, &policy()).remove(0);
        too_many.experiments.push(too_many.experiments[0].clone());
        too_many.experiments[6].proposal.experiment_id = "extra".into();
        assert!(validate_scenario(&too_many).is_err());
    }
}
