//! Canonical analytical functions for external-validity checks of multi-fidelity search.
//!
//! This layer complements the architecture-shaped synthetic benchmark. It uses four standard
//! continuous minimization objectives with known global optimum value zero: Sphere, Rosenbrock,
//! Rastrigin, and Ackley. Fidelity-response transforms are explicitly synthetic and are not
//! claims about real training runs. The objective oracle is only used post hoc for scoring and
//! benchmark generation; promotion receives only the current rung's observed score and SE.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::multi_fidelity::{
    plan_promotions, FidelityLevel, FidelityObservation, PromotionPolicy,
};

const CANDIDATE_COUNT: usize = 256;
const LOW_COST: u64 = 1;
const MID_COST: u64 = 4;
const HIGH_COST: u64 = 16;
const FULL_COST: u64 = 64;
const PROMOTION_FRACTION: f64 = 0.25;
const EXPLORATION_SHARE: f64 = 0.25;

/// Canonical analytical objective. All functions are minimized and have global optimum 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticalFunction {
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
}

impl AnalyticalFunction {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "sphere" => Some(Self::Sphere),
            "rosenbrock" => Some(Self::Rosenbrock),
            "rastrigin" => Some(Self::Rastrigin),
            "ackley" => Some(Self::Ackley),
            _ => None,
        }
    }

    pub fn bounds(self) -> (f64, f64) {
        match self {
            Self::Sphere | Self::Rastrigin => (-5.12, 5.12),
            Self::Rosenbrock => (-2.0, 2.0),
            Self::Ackley => (-5.0, 5.0),
        }
    }

    pub fn objective(self, x: &[f64]) -> Result<f64, String> {
        if x.len() < 2 || x.len() > 64 {
            return Err("dimension must be in 2..=64".to_owned());
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err("coordinates must be finite".to_owned());
        }

        let dimension = x.len() as f64;
        let value = match self {
            Self::Sphere => x.iter().map(|value| value * value).sum(),
            Self::Rosenbrock => x
                .windows(2)
                .map(|pair| {
                    let first = pair[0];
                    let second = pair[1];
                    100.0 * (second - first * first).powi(2) + (1.0 - first).powi(2)
                })
                .sum(),
            Self::Rastrigin => {
                10.0 * dimension
                    + x.iter()
                        .map(|value| value * value - 10.0 * (2.0 * std::f64::consts::PI * value).cos())
                        .sum::<f64>()
            }
            Self::Ackley => {
                let sum_squares = x.iter().map(|value| value * value).sum::<f64>() / dimension;
                let sum_cosines = x
                    .iter()
                    .map(|value| (2.0 * std::f64::consts::PI * value).cos())
                    .sum::<f64>()
                    / dimension;
                -20.0 * (-0.2 * sum_squares.sqrt()).exp()
                    - sum_cosines.exp()
                    + 20.0
                    + std::f64::consts::E
            }
        };

        if value.is_finite() {
            Ok(value.max(0.0))
        } else {
            Err("objective evaluation produced non-finite output".to_owned())
        }
    }

    /// A coordinate vector attaining the known global minimum for this function.
    pub fn optimum(self, dimension: usize) -> Result<Vec<f64>, String> {
        if !(2..=64).contains(&dimension) {
            return Err("dimension must be in 2..=64".to_owned());
        }
        let coordinate = if self == Self::Rosenbrock { 1.0 } else { 0.0 };
        Ok(vec![coordinate; dimension])
    }
}

/// Known fidelity response. Rank reversal is intentional to measure pruning risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityScenario {
    RankPreserving,
    RankReversal,
    LateEmergingQuality,
    MiscalibratedUncertainty,
}

impl FidelityScenario {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "rank_preserving" => Some(Self::RankPreserving),
            "rank_reversal" => Some(Self::RankReversal),
            "late_emerging_quality" => Some(Self::LateEmergingQuality),
            "miscalibrated_uncertainty" => Some(Self::MiscalibratedUncertainty),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyticalCandidate {
    pub id: String,
    pub coordinates: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticalArm {
    MultiFidelityPromotion,
    FullFidelityRandom,
}

/// Post-hoc event log. Oracle objective values must never be provided to the promotion policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyticalTraceEvent {
    pub arm: AnalyticalArm,
    pub candidate_id: String,
    pub fidelity_index: usize,
    pub fidelity_id: String,
    pub observed_score: f64,
    pub expected_score: f64,
    pub reported_standard_error: f64,
    pub perturbation_scale: f64,
    pub oracle_objective: f64,
    pub incremental_cost_units: u64,
    pub promoted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyticalRun {
    pub function: AnalyticalFunction,
    pub scenario: FidelityScenario,
    pub dimension: usize,
    pub seed: u64,
    pub total_cost_budget_units: u64,
    pub promotion_cost_units: u64,
    pub random_cost_units: u64,
    pub promoted_full_fidelity_count: usize,
    pub random_full_fidelity_count: usize,
    pub promotion_best_candidate_id: String,
    pub random_best_candidate_id: String,
    pub promotion_best_objective: f64,
    pub random_best_objective: f64,
    pub promotion_regret: f64,
    pub random_regret: f64,
    /// Positive means full-fidelity random search had worse objective than the promotion arm.
    pub paired_regret_improvement: f64,
    pub promotion_top_10_recall: f64,
    pub random_top_10_recall: f64,
    pub f0_to_oracle_spearman: f64,
    pub f0_nominal_95_coverage: f64,
    pub f0_mean_absolute_standardized_residual: f64,
    pub trace: Vec<AnalyticalTraceEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyticalAggregate {
    pub function: AnalyticalFunction,
    pub scenario: FidelityScenario,
    pub dimension: usize,
    pub seed_count: usize,
    pub total_cost_budget_units: u64,
    pub mean_promotion_regret: f64,
    pub mean_random_regret: f64,
    pub mean_paired_regret_improvement: f64,
    pub paired_regret_improvement_sample_sd: f64,
    pub promotion_top_10_recall_mean: f64,
    pub random_top_10_recall_mean: f64,
    pub mean_f0_to_oracle_spearman: f64,
    pub mean_f0_nominal_95_coverage: f64,
    pub mean_f0_standardized_residual: f64,
}

/// Cumulative cost levels for resumable synthetic fidelity stages.
pub fn analytical_fidelity_ladder() -> Vec<FidelityLevel> {
    vec![
        FidelityLevel { id: "f0_low".to_owned(), cumulative_cost_units: LOW_COST },
        FidelityLevel { id: "f1_medium".to_owned(), cumulative_cost_units: MID_COST },
        FidelityLevel { id: "f2_high".to_owned(), cumulative_cost_units: HIGH_COST },
        FidelityLevel { id: "f3_full".to_owned(), cumulative_cost_units: FULL_COST },
    ]
}

pub fn run_analytical_trial(
    function: AnalyticalFunction,
    dimension: usize,
    scenario: FidelityScenario,
    seed: u64,
) -> Result<AnalyticalRun, String> {
    if !(2..=64).contains(&dimension) {
        return Err("dimension must be in 2..=64".to_owned());
    }
    let pool = generate_candidates(function, dimension, seed)?;
    let ladder = analytical_fidelity_ladder();
    let costs: HashMap<String, u64> = ladder
        .iter()
        .map(|level| (level.id.clone(), level.cumulative_cost_units))
        .collect();
    let by_id: HashMap<String, AnalyticalCandidate> = pool
        .iter()
        .cloned()
        .map(|candidate| (candidate.id.clone(), candidate))
        .collect();
    let mut active = pool.clone();
    let mut trace = Vec::with_capacity(CANDIDATE_COUNT + 128);
    let mut total_cost = (active.len() as u64)
        .checked_mul(LOW_COST)
        .ok_or_else(|| "initial analytical cost overflow".to_owned())?;

    let oracle_objectives = pool
        .iter()
        .map(|candidate| function.objective(&candidate.coordinates))
        .collect::<Result<Vec<_>, _>>()?;
    let mut oracle_ranked: Vec<(&AnalyticalCandidate, f64)> = pool
        .iter()
        .zip(oracle_objectives.iter().copied())
        .map(|(candidate, objective)| (candidate, objective))
        .collect();
    oracle_ranked.sort_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.id.cmp(&right.0.id)));
    let top_10: HashSet<&str> = oracle_ranked.iter().take(10).map(|(candidate, _)| candidate.id.as_str()).collect();

    let mut f0_scores = Vec::with_capacity(pool.len());
    let mut f0_expected = Vec::with_capacity(pool.len());
    let mut f0_errors = Vec::with_capacity(pool.len());

    for (index, candidate) in pool.iter().enumerate() {
        let (expected, observed, standard_error, perturbation_scale) =
            fidelity_score(function, scenario, candidate, 0, seed)?;
        f0_scores.push(observed);
        f0_expected.push(expected);
        f0_errors.push(standard_error);
        trace.push(AnalyticalTraceEvent {
            arm: AnalyticalArm::MultiFidelityPromotion,
            candidate_id: candidate.id.clone(),
            fidelity_index: 0,
            fidelity_id: ladder[0].id.clone(),
            observed_score: observed,
            expected_score: expected,
            reported_standard_error: standard_error,
            perturbation_scale,
            oracle_objective: oracle_objectives[index],
            incremental_cost_units: LOW_COST,
            promoted: false,
        });
    }

    for fidelity_index in 0..(ladder.len() - 1) {
        let observations = active
            .iter()
            .map(|candidate| {
                let (expected, observed, standard_error, _) =
                    fidelity_score(function, scenario, candidate, fidelity_index, seed)?;
                let _ = expected;
                Ok(FidelityObservation {
                    candidate_id: candidate.id.clone(),
                    fidelity_index,
                    mean_score: observed,
                    standard_error,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let decision = plan_promotions(
            &ladder,
            &observations,
            fidelity_index,
            &PromotionPolicy {
                promotion_fraction: 0.25,
                exploration_share: 0.25,
                confidence_multiplier: 1.0,
                seed: seed ^ ((fidelity_index as u64 + 1) * 0x9e37_79b9_7f4a_7c15),
            },
        )
        .map_err(|error| format!("analytical fidelity decision rejected: {error}"))?;

        let promoted_ids: HashSet<&str> = decision.promoted_candidate_ids.iter().map(String::as_str).collect();
        let rung_cost = if fidelity_index == 0 {
            LOW_COST
        } else {
            ladder[fidelity_index].cumulative_cost_units - ladder[fidelity_index - 1].cumulative_cost_units
        };

        // F0 trace entries were emitted above; update promotion flags in place.
        for event in trace.iter_mut().filter(|event| {
            event.arm == AnalyticalArm::MultiFidelityPromotion
                && event.fidelity_index == fidelity_index
        }) {
            event.promoted = promoted_ids.contains(event.candidate_id.as_str());
        }

        // Later rungs are recorded now. F0 is intentionally not duplicated.
        if fidelity_index > 0 {
            for candidate in &active {
                let (expected, observed, standard_error, perturbation_scale) =
                    fidelity_score(function, scenario, candidate, fidelity_index, seed)?;
                let oracle_objective = function.objective(&candidate.coordinates)?;
                trace.push(AnalyticalTraceEvent {
                    arm: AnalyticalArm::MultiFidelityPromotion,
                    candidate_id: candidate.id.clone(),
                    fidelity_index,
                    fidelity_id: ladder[fidelity_index].id.clone(),
                    observed_score: observed,
                    expected_score: expected,
                    reported_standard_error: standard_error,
                    perturbation_scale,
                    oracle_objective,
                    incremental_cost_units: rung_cost,
                    promoted: promoted_ids.contains(candidate.id.as_str()),
                });
            }
        }

        total_cost = total_cost
            .checked_add(decision.estimated_incremental_cost_units)
            .ok_or_else(|| "analytical promotion cost overflow".to_owned())?;
        active = decision.promoted_candidate_ids
            .iter()
            .map(|id| by_id.get(id).cloned().ok_or_else(|| format!("unknown promoted candidate {id:?}")))
            .collect::<Result<Vec<_>, _>>()?;
    }

    // Final rung is charged and recorded for each survivor.
    let final_increment = FULL_COST - HIGH_COST;
    for candidate in &active {
        let objective = function.objective(&candidate.coordinates)?;
        trace.push(AnalyticalTraceEvent {
            arm: AnalyticalArm::MultiFidelityPromotion,
            candidate_id: candidate.id.clone(),
            fidelity_index: 3,
            fidelity_id: ladder[3].id.clone(),
            observed_score: -objective,
            expected_score: -objective,
            reported_standard_error: 0.0,
            perturbation_scale: 0.0,
            oracle_objective: objective,
            incremental_cost_units: final_increment,
            promoted: false,
        });
    }
    if total_cost != 832 {
        return Err(format!("unexpected promotion cost: expected 832 units, got {total_cost}"));
    }

    let random_count = (total_cost / FULL_COST) as usize;
    let mut random_pool = pool.clone();
    shuffle_candidates(&mut random_pool, seed ^ 0xa076_1d64_78bd_642f);
    random_pool.truncate(random_count);
    for candidate in &random_pool {
        let objective = function.objective(&candidate.coordinates)?;
        trace.push(AnalyticalTraceEvent {
            arm: AnalyticalArm::FullFidelityRandom,
            candidate_id: candidate.id.clone(),
            fidelity_index: 3,
            fidelity_id: ladder[3].id.clone(),
            observed_score: -objective,
            expected_score: -objective,
            reported_standard_error: 0.0,
            perturbation_scale: 0.0,
            oracle_objective: objective,
            incremental_cost_units: FULL_COST,
            promoted: false,
        });
    }

    let promotion_best = best_by_objective(&active, function)?;
    let random_best = best_by_objective(&random_pool, function)?;
    let promotion_objective = function.objective(&promotion_best.coordinates)?;
    let random_objective = function.objective(&random_best.coordinates)?;
    let f0_nominal_95_coverage = f0_scores.iter().zip(&f0_expected).zip(&f0_errors)
        .filter(|((observed, expected), se)| (observed - expected).abs() <= 1.96 * **se)
        .count() as f64 / f0_scores.len() as f64;
    let f0_mean_absolute_standardized_residual = f0_scores.iter().zip(&f0_expected).zip(&f0_errors)
        .map(|((observed, expected), se)| (observed - expected).abs() / se.max(f64::MIN_POSITIVE))
        .sum::<f64>() / f0_scores.len() as f64;
    let oracle_score = |candidate: &AnalyticalCandidate| -> Result<f64, String> {
        Ok(-function.objective(&candidate.coordinates)?)
    };
    let promotion_top_10_recall = active.iter().filter(|candidate| top_10.contains(candidate.id.as_str())).count() as f64 / top_10.len() as f64;
    let random_top_10_recall = random_pool.iter().filter(|candidate| top_10.contains(candidate.id.as_str())).count() as f64 / top_10.len() as f64;
    let f0_to_oracle_spearman = spearman(
        &f0_scores,
        &pool.iter().map(oracle_score).collect::<Result<Vec<_>, _>>()?,
    );

    // Reconcile from trace, independently of the scheduler's budget calculation.
    let promotion_trace_cost: u64 = trace.iter()
        .filter(|event| event.arm == AnalyticalArm::MultiFidelityPromotion)
        .map(|event| event.incremental_cost_units)
        .sum();
    let random_trace_cost: u64 = trace.iter()
        .filter(|event| event.arm == AnalyticalArm::FullFidelityRandom)
        .map(|event| event.incremental_cost_units)
        .sum();
    if promotion_trace_cost != total_cost || random_trace_cost != total_cost {
        return Err(format!(
            "trace cost mismatch: policy={promotion_trace_cost}, random={random_trace_cost}, budget={total_cost}"
        ));
    }

    Ok(AnalyticalRun {
        function,
        scenario,
        dimension,
        seed,
        total_cost_budget_units: total_cost,
        promotion_cost_units: promotion_trace_cost,
        random_cost_units: random_trace_cost,
        promoted_full_fidelity_count: active.len(),
        random_full_fidelity_count: random_pool.len(),
        promotion_best_candidate_id: promotion_best.id.clone(),
        random_best_candidate_id: random_best.id.clone(),
        promotion_best_objective: promotion_objective,
        random_best_objective: random_objective,
        promotion_regret: promotion_objective,
        random_regret: random_objective,
        paired_regret_improvement: random_objective - promotion_objective,
        promotion_top_10_recall,
        random_top_10_recall,
        f0_to_oracle_spearman,
        f0_nominal_95_coverage,
        f0_mean_absolute_standardized_residual,
        trace,
    })
}

pub fn run_analytical_sweep(
    function: AnalyticalFunction,
    dimension: usize,
    scenario: FidelityScenario,
    seed_count: usize,
) -> Result<(Vec<AnalyticalRun>, AnalyticalAggregate), String> {
    if !(2..=10_000).contains(&seed_count) {
        return Err("seed_count must be in 2..=10000".to_owned());
    }
    let mut runs = Vec::with_capacity(seed_count);
    for seed in 1..=seed_count as u64 {
        runs.push(run_analytical_trial(
            function,
            dimension,
            scenario,
            seed.wrapping_add(0x414e_414c_5954_4943),
        )?);
    }
    let budget = runs.first().ok_or_else(|| "empty analytical sweep".to_owned())?.total_cost_budget_units;
    if runs.iter().any(|run| run.function != function || run.dimension != dimension || run.scenario != scenario || run.total_cost_budget_units != budget) {
        return Err("aggregate contains mixed function/scenario/dimension/budget".to_owned());
    }
    let improvements: Vec<f64> = runs.iter().map(|run| run.paired_regret_improvement).collect();
    let mean_of = |values: Vec<f64>| -> f64 {
        if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
    };
    let aggregate = AnalyticalAggregate {
        function,
        scenario,
        dimension,
        seed_count: runs.len(),
        total_cost_budget_units: budget,
        mean_promotion_regret: mean_of(runs.iter().map(|run| run.promotion_regret).collect()),
        mean_random_regret: mean_of(runs.iter().map(|run| run.random_regret).collect()),
        mean_paired_regret_improvement: mean_of(improvements.clone()),
        paired_regret_improvement_sample_sd: sample_sd(&improvements),
        promotion_top_10_recall_mean: mean_of(runs.iter().map(|run| run.promotion_top_10_recall).collect()),
        random_top_10_recall_mean: mean_of(runs.iter().map(|run| run.random_top_10_recall).collect()),
        mean_f0_to_oracle_spearman: mean_of(runs.iter().map(|run| run.f0_to_oracle_spearman).collect()),
        mean_f0_nominal_95_coverage: mean_of(runs.iter().map(|run| run.f0_nominal_95_coverage).collect()),
        mean_f0_standardized_residual: mean_of(runs.iter().map(|run| run.f0_mean_absolute_standardized_residual).collect()),
    };
    Ok((runs, aggregate))
}

fn generate_candidates(
    function: AnalyticalFunction,
    dimension: usize,
    seed: u64,
) -> Result<Vec<AnalyticalCandidate>, String> {
    if !(2..=64).contains(&dimension) {
        return Err("dimension must be in 2..=64".to_owned());
    }
    let (low, high) = function.bounds();
    let mut state = seed;
    let mut candidates = Vec::with_capacity(CANDIDATE_COUNT);
    for index in 0..CANDIDATE_COUNT {
        let coordinates = (0..dimension)
            .map(|_| low + (high - low) * uniform(&mut state))
            .collect();
        candidates.push(AnalyticalCandidate { id: format!("x{index:04}"), coordinates });
    }
    Ok(candidates)
}

fn fidelity_score(
    function: AnalyticalFunction,
    scenario: FidelityScenario,
    candidate: &AnalyticalCandidate,
    fidelity_index: usize,
    seed: u64,
) -> Result<(f64, f64, f64, f64), String> {
    let objective = function.objective(&candidate.coordinates)?;
    let wave = candidate.coordinates.iter().enumerate()
        .map(|(index, value)| ((index as f64 + 1.0) * 1.7 * value).sin())
        .sum::<f64>() / candidate.coordinates.len() as f64;
    let (expected_score, noise_scale, reported_scale) = match scenario {
        FidelityScenario::RankPreserving => (
            -objective + 0.01 * (1.0 + objective.abs()) * wave,
            0.005 * (1.0 + objective.abs()),
            0.01 * (1.0 + objective.abs()),
        ),
        FidelityScenario::RankReversal => match fidelity_index {
            0 => (objective, 0.005 * (1.0 + objective.abs()), 0.01 * (1.0 + objective.abs())),
            1 => (-objective, 0.005 * (1.0 + objective.abs()), 0.01 * (1.0 + objective.abs())),
            2 => (-objective, 0.002 * (1.0 + objective.abs()), 0.004 * (1.0 + objective.abs())),
            _ => (-objective, 0.0, 0.0),
        },
        FidelityScenario::LateEmergingQuality => match fidelity_index {
            0 => (objective, 0.005 * (1.0 + objective.abs()), 0.01 * (1.0 + objective.abs())),
            1 => (0.5 * objective, 0.005 * (1.0 + objective.abs()), 0.01 * (1.0 + objective.abs())),
            2 => (-0.25 * objective, 0.002 * (1.0 + objective.abs()), 0.004 * (1.0 + objective.abs())),
            _ => (-objective, 0.0, 0.0),
        },
        FidelityScenario::MiscalibratedUncertainty => (
            -objective + 0.02 * (1.0 + objective.abs()) * wave,
            0.03 * (1.0 + objective.abs()),
            0.0005 * (1.0 + objective.abs()),
        ),
    };
    let candidate_key = candidate.id.as_bytes().iter()
        .fold(0u64, |acc, byte| acc.wrapping_mul(257).wrapping_add(*byte as u64));
    let noise = if noise_scale == 0.0 { 0.0 } else {
        deterministic_noise(
            seed ^ candidate_key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (fidelity_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        ) * noise_scale
    };
    let observed = expected_score + noise;
    let reported_se = if fidelity_index == 3 { 0.0 } else { reported_scale };
    if !observed.is_finite() || !reported_se.is_finite() || reported_se < 0.0 {
        return Err("invalid synthetic fidelity measurement".to_owned());
    }
    Ok((expected_score, observed, reported_se, noise_scale))
}

fn best_by_objective(
    candidates: &[AnalyticalCandidate],
    function: AnalyticalFunction,
) -> Result<AnalyticalCandidate, String> {
    candidates.iter().try_fold(None, |best: Option<(AnalyticalCandidate, f64)>, candidate| {
        let objective = function.objective(&candidate.coordinates)?;
        match best {
            None => Ok(Some((candidate.clone(), objective))),
            Some((current, current_objective)) if objective < current_objective
                || (objective == current_objective && candidate.id < current.id) =>
            {
                Ok(Some((candidate.clone(), objective)))
            }
            other => Ok(other),
        }
    })?
    .map(|(candidate, _)| candidate)
    .ok_or_else(|| "cannot select best from empty candidate set".to_owned())
}

fn shuffle_candidates(candidates: &mut [AnalyticalCandidate], mut state: u64) {
    for index in (1..candidates.len()).rev() {
        state = splitmix64(state);
        let selected = (state as usize) % (index + 1);
        candidates.swap(index, selected);
    }
}

fn uniform(state: &mut u64) -> f64 {
    let bits = splitmix64(*state);
    *state = bits;
    ((bits >> 11) as f64) / ((1u64 << 53) as f64)
}

fn deterministic_noise(seed: u64) -> f64 {
    2.0 * uniform(&mut seed.clone()) - 1.0
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]).then_with(|| left.cmp(right)));
    let mut result = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]].total_cmp(&values[order[end]]).is_eq() {
            end += 1;
        }
        let average = ((start + 1) as f64 + end as f64) / 2.0;
        for index in start..end { result[order[index]] = average; }
        start = end;
    }
    result
}

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() { return 0.0; }
    pearson(&ranks(left), &ranks(right))
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() { return 0.0; }
    let count = left.len() as f64;
    let mean_left = left.iter().sum::<f64>() / count;
    let mean_right = right.iter().sum::<f64>() / count;
    let covariance = left.iter().zip(right).map(|(a, b)| (a - mean_left) * (b - mean_right)).sum::<f64>();
    let variance_left = left.iter().map(|value| (value - mean_left).powi(2)).sum::<f64>();
    let variance_right = right.iter().map(|value| (value - mean_right).powi(2)).sum::<f64>();
    let denominator = (variance_left * variance_right).sqrt();
    if denominator <= f64::EPSILON { 0.0 } else { (covariance / denominator).clamp(-1.0, 1.0) }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 { return 0.0; }
    let average = mean(values);
    (values.iter().map(|value| (value - average).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_functions_have_known_zero_minima() {
        for function in [AnalyticalFunction::Sphere, AnalyticalFunction::Rosenbrock, AnalyticalFunction::Rastrigin, AnalyticalFunction::Ackley] {
            let optimum = function.optimum(4).unwrap();
            assert!(function.objective(&optimum).unwrap().abs() < 1e-8, "{function:?}");
        }
    }

    #[test]
    fn rejects_invalid_dimensions_and_non_finite_coordinates() {
        assert!(AnalyticalFunction::Sphere.objective(&[0.0]).is_err());
        assert!(AnalyticalFunction::Rosenbrock.objective(&[0.0; 65]).is_err());
        assert!(AnalyticalFunction::Ackley.objective(&[f64::NAN, 0.0]).is_err());
    }

    #[test]
    fn identical_seed_reproduces_full_trace_and_summary() {
        let a = run_analytical_trial(AnalyticalFunction::Rastrigin, 4, FidelityScenario::RankPreserving, 81).unwrap();
        let b = run_analytical_trial(AnalyticalFunction::Rastrigin, 4, FidelityScenario::RankPreserving, 81).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn promotion_and_random_baseline_spend_the_same_cost() {
        let run = run_analytical_trial(AnalyticalFunction::Sphere, 3, FidelityScenario::RankPreserving, 13).unwrap();
        assert_eq!(run.total_cost_budget_units, 832);
        assert_eq!(run.promotion_cost_units, 832);
        assert_eq!(run.random_cost_units, 832);
        assert_eq!(run.promoted_full_fidelity_count, 4);
        assert_eq!(run.random_full_fidelity_count, 13);
        assert!(run.trace.iter().all(|event| event.incremental_cost_units > 0));
    }

    #[test]
    fn rank_reversal_control_is_detected_by_f0_rank_correlation() {
        let run = run_analytical_trial(AnalyticalFunction::Rastrigin, 3, FidelityScenario::RankReversal, 29).unwrap();
        assert!(run.f0_to_oracle_spearman < -0.9);
    }

    #[test]
    fn reported_uncertainty_miscalibration_is_detectable() {
        let run = run_analytical_trial(AnalyticalFunction::Sphere, 3, FidelityScenario::MiscalibratedUncertainty, 17).unwrap();
        assert!(run.f0_nominal_95_coverage < 0.5);
        assert!(run.f0_mean_absolute_standardized_residual > 1.0);
    }

    #[test]
    fn sweep_rejects_invalid_seed_counts() {
        assert!(run_analytical_sweep(AnalyticalFunction::Sphere, 3, FidelityScenario::RankPreserving, 1).is_err());
        assert!(run_analytical_sweep(AnalyticalFunction::Sphere, 3, FidelityScenario::RankPreserving, 10_001).is_err());
    }
}
