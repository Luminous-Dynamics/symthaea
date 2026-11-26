// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-horizon empirical balanced projection primitives.
//!
//! Constructs biorthogonal trial/test bases from empirical reachability and
//! observability factors. This is a linearized reduction primitive only: it
//! does not derive solver tangents/adjoints, execute Vlasov dynamics, or certify
//! nonlinear echo preservation.

use super::EmpiricalGramian;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RbkrError {
    #[error("RBKR configuration is invalid")]
    InvalidConfiguration,
    #[error("reachability and observability state dimensions differ")]
    StateDimensionMismatch,
    #[error("reachability and observability Gramians must both contain factors")]
    EmptyGramian,
    #[error("cross-Gramian factor-pair budget exceeded")]
    CrossGramianBudgetExceeded,
    #[error("cross-Gramian scalar-operation budget exceeded")]
    CrossGramianWorkBudgetExceeded,
    #[error("factor-space SVD storage budget exceeded")]
    SvdStorageBudgetExceeded,
    #[error("Jacobi SVD work budget exceeded")]
    JacobiWorkBudgetExceeded,
    #[error("basis construction work or storage budget exceeded")]
    BasisBudgetExceeded,
    #[error("cross-Gramian arithmetic is non-finite")]
    NonFiniteArithmetic,
    #[error("one-sided Jacobi SVD did not converge")]
    SvdDidNotConverge,
    #[error("requested order exceeds the numerically supported balanced rank")]
    InsufficientBalancedRank,
    #[error("trial and test bases are not biorthogonal within tolerance")]
    BiorthogonalityFailure,
    #[error("state vector dimension mismatch")]
    StateVectorDimensionMismatch,
    #[error("reduced coordinate dimension mismatch or non-finite coordinate")]
    ReducedVectorDimensionMismatch,
    #[error("linear A matrix must be finite, square, and match the state dimension")]
    InvalidAOperator,
    #[error(
        "linear B matrix must be finite, non-empty, and have the state dimension as its row count"
    )]
    InvalidBOperator,
    #[error(
        "linear C matrix must be finite, non-empty, and have the state dimension as its column count"
    )]
    InvalidCOperator,
    #[error("response Gramian scenario set or coordinate contract is invalid")]
    InvalidScenarioSet,
    #[error("response Gramian scenarios use different task contracts")]
    MixedScenarioContract,
    #[error("response Gramian scenarios use different checkpoints")]
    MixedScenarioCheckpoint,
    #[error("response Gramian scenarios use different state metrics")]
    MixedScenarioStateMetric,
    #[error("response Gramian scenarios use different state coordinate systems")]
    MixedScenarioCoordinates,
    #[error("positive-weight scenario is missing reachability or observability factors")]
    EmptyScenarioGramian,
    #[error("response Gramian scenario weight must be finite and non-negative")]
    InvalidScenarioWeight,
    #[error("response Gramian aggregation factor-storage budget exceeded")]
    ScenarioFactorStorageBudgetExceeded,
    #[error("response-residual CVaR level must be finite and in [0, 1)")]
    InvalidCvarLevel,
    #[error("scenario has no non-zero reachable-to-observable reference response")]
    NoReferenceResponse,
}

/// Errors from finite-horizon linearized Gramian construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FiniteHorizonGramianError {
    #[error("finite-horizon trajectory axes are inconsistent")]
    InvalidTrajectoryAxes,
    #[error(
        "finite-horizon state, input, or output matrix has an invalid shape or non-finite value"
    )]
    InvalidTrajectoryMatrix,
    #[error("input cost must be finite and strictly positive")]
    InvalidInputCost,
    #[error("QoI weight must be finite and non-negative")]
    InvalidQoiWeight,
    #[error("finite-horizon factor storage budget exceeded")]
    FactorStorageBudgetExceeded,
    #[error("finite-horizon propagation work budget exceeded")]
    PropagationWorkBudgetExceeded,
    #[error("finite-horizon Gramian construction produced invalid factors")]
    InvalidGramianFactor,
}

/// One frozen physical trajectory / pulse scenario contributing to a task Gramian.
///
/// All scenarios being aggregated must share the same task contract,
/// checkpoint state, coordinate ordering, and state metric. The scenario
/// weight is normalized across the set and scales both Gramians. The caller
/// should construct each pair with build_finite_horizon_gramians using the
/// same checkpoint digest, then aggregate future-response factors across the
/// registered intervention family.
#[derive(Clone, Copy, Debug)]
pub struct ResponseGramianScenario<'a> {
    pub scenario_id: &'a str,
    /// SHA-256 of the frozen full-order trajectory and pulse path.
    pub trajectory_sha256: &'a str,
    pub task_contract_id: &'a str,
    pub checkpoint_id: &'a str,
    /// SHA-256 of the exact checkpoint state shared by all scenarios.
    pub checkpoint_sha256: &'a str,
    /// Coordinate ordering / layout fingerprint, separate from the metric.
    pub state_coordinate_id: &'a str,
    pub state_metric_id: &'a str,
    pub reachability: &'a EmpiricalGramian,
    pub observability: &'a EmpiricalGramian,
    pub scenario_weight: f64,
}

/// Linearized response residual for one immutable pulse scenario.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioResponseResidual {
    pub scenario_id: String,
    pub trajectory_sha256: String,
    pub normalized_weight: f64,
    pub response_residual: f64,
}

/// Distribution summary of scenario-wise response residuals.
///
/// The worst-case value includes every supplied scenario, including scenarios
/// with zero statistical weight. Weighted mean and CVaR use normalized
/// non-negative scenario weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseResidualReport {
    pub scenarios: Vec<ScenarioResponseResidual>,
    pub weighted_mean: f64,
    pub worst_case: f64,
    pub cvar_alpha: f64,
    pub cvar: f64,
}

/// Peak live-factor storage limit while aggregating factorized Gramian families.
///
/// The budget includes both the input scenario factors and the newly allocated
/// aggregate factors, because aggregation does not consume or mutate inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseGramianAggregationConfig {
    pub maximum_live_factor_scalars: usize,
}

impl ResponseGramianAggregationConfig {
    pub fn new(maximum_live_factor_scalars: usize) -> Self {
        Self {
            maximum_live_factor_scalars,
        }
    }
}

impl Default for ResponseGramianAggregationConfig {
    fn default() -> Self {
        Self::new(32_000_000)
    }
}

/// Evaluate a balanced basis against individual scenario Gramians and summarize
/// the weighted mean, worst-case residual, and upper-tail CVaR.
///
/// This function is intended for validation / held-out challenge cohorts. It
/// does not modify the basis and does not promote held-out data into training.
pub fn evaluate_response_scenario_residuals(
    basis: &ResponseBalancedBasis,
    scenarios: &[ResponseGramianScenario<'_>],
    cvar_alpha: f64,
) -> Result<ResponseResidualReport, RbkrError> {
    if !cvar_alpha.is_finite() || !(0.0..1.0).contains(&cvar_alpha) {
        return Err(RbkrError::InvalidCvarLevel);
    }
    let first = scenarios.first().ok_or(RbkrError::InvalidScenarioSet)?;
    if first.task_contract_id.trim().is_empty()
        || first.checkpoint_id.trim().is_empty()
        || !is_sha256_hex(first.checkpoint_sha256)
        || first.state_coordinate_id.trim().is_empty()
        || first.state_metric_id.trim().is_empty()
    {
        return Err(RbkrError::InvalidScenarioSet);
    }

    let mut scenario_ids = HashSet::with_capacity(scenarios.len());
    let mut trajectory_digests = HashSet::with_capacity(scenarios.len());
    let mut maximum_weight = 0.0_f64;
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id)
            || !is_sha256_hex(scenario.trajectory_sha256)
            || !trajectory_digests.insert(scenario.trajectory_sha256)
        {
            return Err(RbkrError::InvalidScenarioSet);
        }
        if scenario.task_contract_id != first.task_contract_id {
            return Err(RbkrError::MixedScenarioContract);
        }
        if scenario.checkpoint_id != first.checkpoint_id
            || !is_sha256_hex(scenario.checkpoint_sha256)
            || scenario.checkpoint_sha256 != first.checkpoint_sha256
        {
            return Err(RbkrError::MixedScenarioCheckpoint);
        }
        if scenario.state_coordinate_id != first.state_coordinate_id {
            return Err(RbkrError::MixedScenarioCoordinates);
        }
        if scenario.state_metric_id != first.state_metric_id {
            return Err(RbkrError::MixedScenarioStateMetric);
        }
        if !scenario.scenario_weight.is_finite() || scenario.scenario_weight < 0.0 {
            return Err(RbkrError::InvalidScenarioWeight);
        }
        if scenario.reachability.dimension != basis.state_dimension
            || scenario.observability.dimension != basis.state_dimension
        {
            return Err(RbkrError::StateDimensionMismatch);
        }
        if scenario.reachability.weighted_factors.is_empty()
            || scenario.observability.weighted_factors.is_empty()
        {
            return Err(RbkrError::EmptyScenarioGramian);
        }
        maximum_weight = maximum_weight.max(scenario.scenario_weight);
    }
    if maximum_weight <= 0.0 {
        return Err(RbkrError::InvalidScenarioWeight);
    }

    let mut scaled_total = 0.0_f64;
    for scenario in scenarios {
        scaled_total += scenario.scenario_weight / maximum_weight;
    }
    if !scaled_total.is_finite() || scaled_total <= 0.0 {
        return Err(RbkrError::InvalidScenarioWeight);
    }

    let mut reports = Vec::with_capacity(scenarios.len());
    let mut weighted_mean = 0.0_f64;
    let mut worst_case = 0.0_f64;
    for scenario in scenarios {
        let normalized_weight = (scenario.scenario_weight / maximum_weight) / scaled_total;
        let residual = basis.linearized_response_residual(
            scenario.reachability,
            scenario.observability,
        )?;
        weighted_mean += normalized_weight * residual;
        if !weighted_mean.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        worst_case = worst_case.max(residual);
        reports.push(ScenarioResponseResidual {
            scenario_id: scenario.scenario_id.to_owned(),
            trajectory_sha256: scenario.trajectory_sha256.to_owned(),
            normalized_weight,
            response_residual: residual,
        });
    }

    let tail_mass = 1.0 - cvar_alpha;
    let mut tail_remaining = tail_mass;
    let mut tail_loss = 0.0_f64;
    let mut ordered = reports.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .response_residual
            .total_cmp(&left.response_residual)
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });
    for scenario in ordered {
        if tail_remaining <= 0.0 {
            break;
        }
        let included_weight = scenario.normalized_weight.min(tail_remaining);
        tail_loss += included_weight * scenario.response_residual;
        if !tail_loss.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        tail_remaining -= included_weight;
    }
    if tail_remaining > 1.0e-10 {
        return Err(RbkrError::InvalidScenarioWeight);
    }
    let cvar = tail_loss / tail_mass;
    if !cvar.is_finite() {
        return Err(RbkrError::NonFiniteArithmetic);
    }

    Ok(ResponseResidualReport {
        scenarios: reports,
        weighted_mean,
        worst_case,
        cvar_alpha,
        cvar,
    })
}

/// Aggregate checkpoint-aligned Gramians over a registered future-response family.
///
/// Each scenario carries its own linearized A/B/C response trajectory but must
/// share the exact checkpoint, state coordinates/metric, and task contract.
/// This forms the weighted sums
///
/// Wc = sum_s p_s Wc_s and Wo = sum_s p_s Wo_s,
///
/// without materializing either dense matrix. Scenario weights are normalized
/// to sum to one. This is an empirical family average, not a worst-case bound:
/// adversarial/worst-case pulses must remain in a separate challenge set and
/// be evaluated independently.
pub fn aggregate_response_gramians(
    scenarios: &[ResponseGramianScenario<'_>],
    config: ResponseGramianAggregationConfig,
) -> Result<(EmpiricalGramian, EmpiricalGramian), RbkrError> {
    let first = scenarios.first().ok_or(RbkrError::InvalidScenarioSet)?;
    if config.maximum_live_factor_scalars == 0
        || first.task_contract_id.trim().is_empty()
        || first.checkpoint_id.trim().is_empty()
        || !is_sha256_hex(first.checkpoint_sha256)
        || first.state_coordinate_id.trim().is_empty()
        || first.state_metric_id.trim().is_empty()
    {
        return Err(RbkrError::InvalidScenarioSet);
    }

    let dimension = first.reachability.dimension;
    if dimension == 0 || first.observability.dimension != dimension {
        return Err(RbkrError::StateDimensionMismatch);
    }

    let mut scenario_ids = HashSet::with_capacity(scenarios.len());
    let mut trajectory_digests = HashSet::with_capacity(scenarios.len());
    let mut maximum_weight = 0.0_f64;
    let mut scaled_weight_sum = 0.0_f64;
    let mut input_factor_count = 0_usize;
    let mut aggregate_factor_count = 0_usize;
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id)
        {
            return Err(RbkrError::InvalidScenarioSet);
        }
        if !is_sha256_hex(scenario.trajectory_sha256)
            || !trajectory_digests.insert(scenario.trajectory_sha256)
        {
            return Err(RbkrError::InvalidScenarioSet);
        }
        if scenario.task_contract_id != first.task_contract_id {
            return Err(RbkrError::MixedScenarioContract);
        }
        if scenario.checkpoint_id != first.checkpoint_id
            || !is_sha256_hex(scenario.checkpoint_sha256)
            || scenario.checkpoint_sha256 != first.checkpoint_sha256
        {
            return Err(RbkrError::MixedScenarioCheckpoint);
        }
        if scenario.state_coordinate_id != first.state_coordinate_id {
            return Err(RbkrError::MixedScenarioCoordinates);
        }
        if scenario.state_metric_id != first.state_metric_id {
            return Err(RbkrError::MixedScenarioStateMetric);
        }
        if scenario.reachability.dimension != dimension
            || scenario.observability.dimension != dimension
        {
            return Err(RbkrError::StateDimensionMismatch);
        }
        if !scenario.scenario_weight.is_finite() || scenario.scenario_weight < 0.0 {
            return Err(RbkrError::InvalidScenarioWeight);
        }
        maximum_weight = maximum_weight.max(scenario.scenario_weight);
        let scenario_factor_count = scenario
            .reachability
            .weighted_factors
            .len()
            .checked_add(scenario.observability.weighted_factors.len())
            .ok_or(RbkrError::ScenarioFactorStorageBudgetExceeded)?;
        input_factor_count = input_factor_count
            .checked_add(scenario_factor_count)
            .ok_or(RbkrError::ScenarioFactorStorageBudgetExceeded)?;
        if scenario.scenario_weight == 0.0 {
            continue;
        }
        if scenario.reachability.weighted_factors.is_empty()
            || scenario.observability.weighted_factors.is_empty()
        {
            return Err(RbkrError::EmptyScenarioGramian);
        }
        aggregate_factor_count = aggregate_factor_count
            .checked_add(scenario_factor_count)
            .ok_or(RbkrError::ScenarioFactorStorageBudgetExceeded)?;
    }
    if maximum_weight <= 0.0 || !maximum_weight.is_finite() {
        return Err(RbkrError::InvalidScenarioWeight);
    }
    // Scale before summation to avoid overflow when individually valid
    // scenario weights are near f64::MAX.
    for scenario in scenarios {
        scaled_weight_sum += scenario.scenario_weight / maximum_weight;
    }
    if !scaled_weight_sum.is_finite() || scaled_weight_sum <= 0.0 {
        return Err(RbkrError::InvalidScenarioWeight);
    }

    // All borrowed source factors remain resident, including zero-weight
    // scenarios; only positive-weight factors are copied into the aggregate.
    let peak_live_factor_scalars = input_factor_count
        .checked_add(aggregate_factor_count)
        .and_then(|count| count.checked_mul(dimension))
        .ok_or(RbkrError::ScenarioFactorStorageBudgetExceeded)?;
    if peak_live_factor_scalars > config.maximum_live_factor_scalars {
        return Err(RbkrError::ScenarioFactorStorageBudgetExceeded);
    }

    let mut reachability =
        EmpiricalGramian::new(dimension).map_err(|_| RbkrError::StateDimensionMismatch)?;
    let mut observability =
        EmpiricalGramian::new(dimension).map_err(|_| RbkrError::StateDimensionMismatch)?;
    for scenario in scenarios {
        if scenario.scenario_weight == 0.0 {
            continue;
        }
        let normalized_weight = (scenario.scenario_weight / maximum_weight) / scaled_weight_sum;
        for factor in &scenario.reachability.weighted_factors {
            validate_factor(factor, dimension)?;
            reachability
                .add_direction(factor, normalized_weight)
                .map_err(|_| RbkrError::NonFiniteArithmetic)?;
        }
        for factor in &scenario.observability.weighted_factors {
            validate_factor(factor, dimension)?;
            observability
                .add_direction(factor, normalized_weight)
                .map_err(|_| RbkrError::NonFiniteArithmetic)?;
        }
    }
    if reachability.weighted_factors.is_empty() || observability.weighted_factors.is_empty() {
        return Err(RbkrError::EmptyGramian);
    }
    Ok((reachability, observability))
}

/// Resource limits for exact finite-horizon LTV Gramian factor construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteHorizonGramianConfig {
    pub maximum_factor_scalars: usize,
    pub maximum_propagation_scalar_operations: usize,
}

impl FiniteHorizonGramianConfig {
    pub fn new(
        maximum_factor_scalars: usize,
        maximum_propagation_scalar_operations: usize,
    ) -> Self {
        Self {
            maximum_factor_scalars,
            maximum_propagation_scalar_operations,
        }
    }
}

impl Default for FiniteHorizonGramianConfig {
    fn default() -> Self {
        Self::new(8_000_000, 100_000_000)
    }
}

/// Construct finite-horizon, checkpoint-centred Gramians for a frozen
/// time-varying discrete linearization.
///
/// The past system obeys x[k+1] = A_past[k] x[k] + B_past[k] u[k].
/// Each diagonal input-cost entry R[k,j] is positive. The reachability
/// factors are Phi(checkpoint,k+1) B_past[k][:,j] / sqrt(R[k,j]).
///
/// The future system starts at the checkpoint. A_future[k] maps sample k to
/// k+1; C_future[k] observes sample k, including the checkpoint and terminal
/// sample. Each non-negative QoI weight Q[k,j] scales one output sensitivity.
/// The observability factors are Phi(k,checkpoint)^T C_future[k][j,:]^T
/// multiplied by sqrt(Q[k,j]).
///
/// Both Gramians therefore use the same checkpoint-state coordinates:
/// reachability is propagated from past interventions to the checkpoint,
/// while future QoI sensitivities are pulled back to the checkpoint. Input
/// and output channel counts may vary by time sample.
///
/// Operators must come from one frozen discrete solver revision, or be
/// explicitly labelled finite-difference estimates. This routine does not
/// derive physical derivatives, execute Vlasov dynamics, or certify nonlinear
/// response preservation.
pub fn build_finite_horizon_gramians(
    past_a: &[Vec<Vec<f64>>],
    past_b: &[Vec<Vec<f64>>],
    input_costs: &[Vec<f64>],
    future_a: &[Vec<Vec<f64>>],
    future_c: &[Vec<Vec<f64>>],
    qoi_weights: &[Vec<f64>],
    config: FiniteHorizonGramianConfig,
) -> Result<(EmpiricalGramian, EmpiricalGramian), FiniteHorizonGramianError> {
    let dimension = past_a
        .first()
        .or_else(|| future_a.first())
        .map(Vec::len)
        .ok_or(FiniteHorizonGramianError::InvalidTrajectoryAxes)?;
    if dimension == 0
        || past_a.is_empty()
        || past_a.len() != past_b.len()
        || past_a.len() != input_costs.len()
        || future_c.len() != future_a.len().saturating_add(1)
        || future_c.len() != qoi_weights.len()
        || config.maximum_factor_scalars == 0
        || config.maximum_propagation_scalar_operations == 0
    {
        return Err(FiniteHorizonGramianError::InvalidTrajectoryAxes);
    }

    validate_square_sequence(past_a, dimension)?;
    validate_square_sequence(future_a, dimension)?;

    let mut factor_count = 0_usize;
    let mut propagation_count = 0_usize;
    for (step, (matrix, costs)) in past_b.iter().zip(input_costs).enumerate() {
        let input_count = matrix.first().map(Vec::len).unwrap_or(0);
        if matrix.len() != dimension
            || input_count == 0
            || matrix
                .iter()
                .any(|row| row.len() != input_count || row.iter().any(|value| !value.is_finite()))
            || costs.len() != input_count
        {
            return Err(FiniteHorizonGramianError::InvalidTrajectoryMatrix);
        }
        for cost in costs {
            if !cost.is_finite() || *cost <= 0.0 || !(1.0 / cost).is_finite() {
                return Err(FiniteHorizonGramianError::InvalidInputCost);
            }
        }
        factor_count = factor_count
            .checked_add(input_count)
            .ok_or(FiniteHorizonGramianError::FactorStorageBudgetExceeded)?;
        propagation_count = propagation_count
            .checked_add(
                input_count
                    .checked_mul(past_a.len() - step - 1)
                    .ok_or(FiniteHorizonGramianError::PropagationWorkBudgetExceeded)?,
            )
            .ok_or(FiniteHorizonGramianError::PropagationWorkBudgetExceeded)?;
    }

    for (sample, (matrix, weights)) in future_c.iter().zip(qoi_weights).enumerate() {
        if matrix
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
            || weights.len() != matrix.len()
        {
            return Err(FiniteHorizonGramianError::InvalidTrajectoryMatrix);
        }
        for weight in weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(FiniteHorizonGramianError::InvalidQoiWeight);
            }
        }
        let active_outputs = weights.iter().filter(|weight| **weight > 0.0).count();
        factor_count = factor_count
            .checked_add(active_outputs)
            .ok_or(FiniteHorizonGramianError::FactorStorageBudgetExceeded)?;
        propagation_count = propagation_count
            .checked_add(
                active_outputs
                    .checked_mul(sample)
                    .ok_or(FiniteHorizonGramianError::PropagationWorkBudgetExceeded)?,
            )
            .ok_or(FiniteHorizonGramianError::PropagationWorkBudgetExceeded)?;
    }

    let factor_scalars = factor_count
        .checked_mul(dimension)
        .ok_or(FiniteHorizonGramianError::FactorStorageBudgetExceeded)?;
    if factor_scalars > config.maximum_factor_scalars {
        return Err(FiniteHorizonGramianError::FactorStorageBudgetExceeded);
    }
    let scalar_work = propagation_count
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(dimension))
        .and_then(|value| value.checked_mul(2))
        .ok_or(FiniteHorizonGramianError::PropagationWorkBudgetExceeded)?;
    if scalar_work > config.maximum_propagation_scalar_operations {
        return Err(FiniteHorizonGramianError::PropagationWorkBudgetExceeded);
    }

    let mut reachability = EmpiricalGramian::new(dimension)
        .map_err(|_| FiniteHorizonGramianError::InvalidTrajectoryAxes)?;
    for (step, (matrix, costs)) in past_b.iter().zip(input_costs).enumerate() {
        for input in 0..costs.len() {
            let mut direction = matrix.iter().map(|row| row[input]).collect::<Vec<_>>();
            for transition in past_a.iter().skip(step + 1) {
                direction = matrix_vector(transition, &direction)?;
            }
            reachability
                .add_direction(&direction, 1.0 / costs[input])
                .map_err(|_| FiniteHorizonGramianError::InvalidGramianFactor)?;
        }
    }

    let mut observability = EmpiricalGramian::new(dimension)
        .map_err(|_| FiniteHorizonGramianError::InvalidTrajectoryAxes)?;
    for (sample, (matrix, weights)) in future_c.iter().zip(qoi_weights).enumerate() {
        for (qoi, weight) in weights.iter().enumerate() {
            if *weight == 0.0 {
                continue;
            }
            let mut direction = matrix[qoi].clone();
            for transition in (0..sample).rev() {
                direction = transpose_matrix_vector(&future_a[transition], &direction)?;
            }
            observability
                .add_direction(&direction, *weight)
                .map_err(|_| FiniteHorizonGramianError::InvalidGramianFactor)?;
        }
    }

    Ok((reachability, observability))
}

fn validate_square_sequence(
    matrices: &[Vec<Vec<f64>>],
    dimension: usize,
) -> Result<(), FiniteHorizonGramianError> {
    if matrices.iter().any(|matrix| {
        matrix.len() != dimension
            || matrix
                .iter()
                .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
    }) {
        return Err(FiniteHorizonGramianError::InvalidTrajectoryMatrix);
    }
    Ok(())
}

fn matrix_vector(
    matrix: &[Vec<f64>],
    vector: &[f64],
) -> Result<Vec<f64>, FiniteHorizonGramianError> {
    matrix
        .iter()
        .map(|row| {
            let mut sum = 0.0;
            for (left, right) in row.iter().zip(vector) {
                sum += left * right;
                if !sum.is_finite() {
                    return Err(FiniteHorizonGramianError::InvalidTrajectoryMatrix);
                }
            }
            Ok(sum)
        })
        .collect()
}

fn transpose_matrix_vector(
    matrix: &[Vec<f64>],
    vector: &[f64],
) -> Result<Vec<f64>, FiniteHorizonGramianError> {
    let mut result = vec![0.0; vector.len()];
    for (row_index, row) in matrix.iter().enumerate() {
        for column_index in 0..vector.len() {
            result[column_index] += row[column_index] * vector[row_index];
            if !result[column_index].is_finite() {
                return Err(FiniteHorizonGramianError::InvalidTrajectoryMatrix);
            }
        }
    }
    Ok(result)
}

/// Numerical and resource controls for deterministic RBKR construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RbkrConfig {
    pub retained_order: usize,
    pub relative_singular_value_cutoff: f64,
    pub jacobi_tolerance: f64,
    pub maximum_jacobi_sweeps: usize,
    pub maximum_cross_gramian_entries: usize,
    pub maximum_cross_gramian_scalar_operations: usize,
    pub maximum_svd_storage_entries: usize,
    pub maximum_jacobi_scalar_operations: usize,
    pub maximum_basis_scalars: usize,
    pub maximum_basis_scalar_operations: usize,
    pub biorthogonality_tolerance: f64,
}

impl RbkrConfig {
    pub fn new(retained_order: usize) -> Self {
        Self {
            retained_order,
            relative_singular_value_cutoff: 1.0e-12,
            jacobi_tolerance: 1.0e-12,
            maximum_jacobi_sweeps: 64,
            maximum_cross_gramian_entries: 1_000_000,
            maximum_cross_gramian_scalar_operations: 100_000_000,
            maximum_svd_storage_entries: 2_000_000,
            maximum_jacobi_scalar_operations: 100_000_000,
            maximum_basis_scalars: 8_000_000,
            maximum_basis_scalar_operations: 100_000_000,
            biorthogonality_tolerance: 1.0e-8,
        }
    }

    fn validate(self) -> Result<(), RbkrError> {
        if self.retained_order == 0
            || !self.relative_singular_value_cutoff.is_finite()
            || !(0.0..1.0).contains(&self.relative_singular_value_cutoff)
            || !self.jacobi_tolerance.is_finite()
            || self.jacobi_tolerance <= 0.0
            || self.jacobi_tolerance >= 1.0
            || self.maximum_jacobi_sweeps == 0
            || self.maximum_cross_gramian_entries == 0
            || self.maximum_cross_gramian_scalar_operations == 0
            || self.maximum_svd_storage_entries == 0
            || self.maximum_jacobi_scalar_operations == 0
            || self.maximum_basis_scalars == 0
            || self.maximum_basis_scalar_operations == 0
            || !self.biorthogonality_tolerance.is_finite()
            || self.biorthogonality_tolerance <= 0.0
            || self.biorthogonality_tolerance >= 1.0
        {
            return Err(RbkrError::InvalidConfiguration);
        }
        Ok(())
    }
}

impl Default for RbkrConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Finite-dimensional empirical balanced trial/test basis.
///
/// The trial basis reconstructs states; the test basis extracts reduced
/// coordinates. Their cross-pairing is checked against the identity. The
/// singular values are empirical, finite-horizon quantities for the supplied
/// intervention and QoI weights, not nonlinear Vlasov error bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseBalancedBasis {
    state_dimension: usize,
    trial_basis: Vec<Vec<f64>>,
    test_basis: Vec<Vec<f64>>,
    balanced_singular_values: Vec<f64>,
    biorthogonality_defect: f64,
}

impl ResponseBalancedBasis {
    /// Build the bases from Wc = X X^T and Wo = Y Y^T.
    ///
    /// The caller supplies factors in one shared state coordinate system and
    /// state metric. The method forms H = Y^T X, computes its SVD, and returns
    /// Vstate = X V Sigma^(-1/2) and Wstate = Y U Sigma^(-1/2), so that
    /// Wstate^T Vstate is identity to the reported numerical tolerance.
    pub fn from_empirical_gramians(
        reachability: &EmpiricalGramian,
        observability: &EmpiricalGramian,
        config: RbkrConfig,
    ) -> Result<Self, RbkrError> {
        config.validate()?;
        if reachability.dimension != observability.dimension {
            return Err(RbkrError::StateDimensionMismatch);
        }
        let dimension = reachability.dimension;
        let reach_count = reachability.weighted_factors.len();
        let observe_count = observability.weighted_factors.len();
        if reach_count == 0 || observe_count == 0 {
            return Err(RbkrError::EmptyGramian);
        }

        let cross_entries = observe_count
            .checked_mul(reach_count)
            .ok_or(RbkrError::CrossGramianBudgetExceeded)?;
        if cross_entries > config.maximum_cross_gramian_entries {
            return Err(RbkrError::CrossGramianBudgetExceeded);
        }
        let cross_work = cross_entries
            .checked_mul(dimension)
            .ok_or(RbkrError::CrossGramianWorkBudgetExceeded)?;
        if cross_work > config.maximum_cross_gramian_scalar_operations {
            return Err(RbkrError::CrossGramianWorkBudgetExceeded);
        }
        let right_entries = reach_count
            .checked_mul(reach_count)
            .ok_or(RbkrError::SvdStorageBudgetExceeded)?;
        let svd_entries = cross_entries
            .checked_add(right_entries)
            .ok_or(RbkrError::SvdStorageBudgetExceeded)?;
        if svd_entries > config.maximum_svd_storage_entries {
            return Err(RbkrError::SvdStorageBudgetExceeded);
        }

        let pairs = reach_count
            .checked_mul(reach_count.saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .ok_or(RbkrError::JacobiWorkBudgetExceeded)?;
        // Per column-pair sweep: norms and inner product (up to 6m
        // operations), matrix rotations (up to 6m), right-vector rotations
        // (up to 6n), plus scalar rotation setup. The padded 16m + 8n + 64
        // estimate is intentionally conservative for m observation factors
        // and n reachability factors.
        let per_pair_work = observe_count
            .checked_mul(16)
            .and_then(|value| {
                reach_count
                    .checked_mul(8)
                    .and_then(|right| value.checked_add(right))
            })
            .and_then(|value| value.checked_add(64))
            .ok_or(RbkrError::JacobiWorkBudgetExceeded)?;
        let jacobi_work = pairs
            .checked_mul(per_pair_work)
            .and_then(|value| value.checked_mul(config.maximum_jacobi_sweeps))
            .ok_or(RbkrError::JacobiWorkBudgetExceeded)?;
        if jacobi_work > config.maximum_jacobi_scalar_operations {
            return Err(RbkrError::JacobiWorkBudgetExceeded);
        }

        let basis_scalars = dimension
            .checked_mul(config.retained_order)
            .and_then(|value| value.checked_mul(2))
            .ok_or(RbkrError::BasisBudgetExceeded)?;
        let basis_work = dimension
            .checked_mul(
                reach_count
                    .checked_add(observe_count)
                    .ok_or(RbkrError::BasisBudgetExceeded)?,
            )
            .and_then(|value| value.checked_mul(config.retained_order))
            .ok_or(RbkrError::BasisBudgetExceeded)?;
        if basis_scalars > config.maximum_basis_scalars
            || basis_work > config.maximum_basis_scalar_operations
        {
            return Err(RbkrError::BasisBudgetExceeded);
        }

        let mut cross = vec![vec![0.0; reach_count]; observe_count];
        for (row, y_factor) in observability.weighted_factors.iter().enumerate() {
            validate_factor(y_factor, dimension)?;
            for (column, x_factor) in reachability.weighted_factors.iter().enumerate() {
                validate_factor(x_factor, dimension)?;
                cross[row][column] = checked_dot(y_factor, x_factor)?;
            }
        }

        let (rotated_cross, right_vectors) =
            one_sided_jacobi_svd(cross, config.maximum_jacobi_sweeps, config.jacobi_tolerance)?;
        let unsorted_singular_values = (0..reach_count)
            .map(|index| column_squared_norm(&rotated_cross, index).map(f64::sqrt))
            .collect::<Result<Vec<_>, _>>()?;
        if unsorted_singular_values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        let mut order = (0..reach_count).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            unsorted_singular_values[*right].total_cmp(&unsorted_singular_values[*left])
        });
        let singular_values = order
            .iter()
            .map(|index| unsorted_singular_values[*index])
            .collect::<Vec<_>>();
        let maximum = singular_values.first().copied().unwrap_or(0.0);
        if maximum <= 0.0 || !maximum.is_finite() {
            return Err(RbkrError::InsufficientBalancedRank);
        }
        let threshold = maximum * config.relative_singular_value_cutoff;
        let available = singular_values
            .iter()
            .take_while(|value| **value > threshold)
            .count();
        if config.retained_order > available {
            return Err(RbkrError::InsufficientBalancedRank);
        }

        let retained = config.retained_order;
        let mut trial_basis = vec![vec![0.0; dimension]; retained];
        let mut test_basis = vec![vec![0.0; dimension]; retained];

        for reduced_index in 0..retained {
            let source_column = order[reduced_index];
            let sigma = singular_values[reduced_index];
            let inverse_sqrt = 1.0 / sigma.sqrt();
            if !inverse_sqrt.is_finite() {
                return Err(RbkrError::NonFiniteArithmetic);
            }

            // Trial vectors: X V Sigma^(-1/2).
            for (factor_index, factor) in reachability.weighted_factors.iter().enumerate() {
                let coefficient = right_vectors[factor_index][source_column] * inverse_sqrt;
                for (state_index, value) in factor.iter().enumerate() {
                    trial_basis[reduced_index][state_index] += coefficient * value;
                }
            }

            // Left singular vectors are the normalized columns of the rotated H.
            // Test vectors: Y U Sigma^(-1/2).
            let left_vector = (0..observe_count)
                .map(|row| rotated_cross[row][source_column] / sigma)
                .collect::<Vec<_>>();
            for (factor_index, factor) in observability.weighted_factors.iter().enumerate() {
                let coefficient = left_vector[factor_index] * inverse_sqrt;
                for (state_index, value) in factor.iter().enumerate() {
                    test_basis[reduced_index][state_index] += coefficient * value;
                }
            }
        }

        if trial_basis.iter().flatten().any(|value| !value.is_finite())
            || test_basis.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        let defect = pairing_defect(&trial_basis, &test_basis)?;
        if defect > config.biorthogonality_tolerance {
            return Err(RbkrError::BiorthogonalityFailure);
        }

        Ok(Self {
            state_dimension: dimension,
            trial_basis,
            test_basis,
            balanced_singular_values: singular_values,
            biorthogonality_defect: defect,
        })
    }

    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    pub fn reduced_order(&self) -> usize {
        self.trial_basis.len()
    }

    pub fn trial_basis(&self) -> &[Vec<f64>] {
        &self.trial_basis
    }

    pub fn test_basis(&self) -> &[Vec<f64>] {
        &self.test_basis
    }

    /// Descending empirical balanced singular values, including discarded modes.
    pub fn balanced_singular_values(&self) -> &[f64] {
        &self.balanced_singular_values
    }

    pub fn biorthogonality_defect(&self) -> f64 {
        self.biorthogonality_defect
    }

    /// Compute reduced coordinates z = W^T x.
    pub fn reduce_state(&self, state: &[f64]) -> Result<Vec<f64>, RbkrError> {
        if state.len() != self.state_dimension {
            return Err(RbkrError::StateVectorDimensionMismatch);
        }
        self.test_basis
            .iter()
            .map(|basis| checked_dot(basis, state))
            .collect()
    }

    /// Reconstruct the trial-space approximation x_hat = V z.
    pub fn reconstruct_state(&self, coordinates: &[f64]) -> Result<Vec<f64>, RbkrError> {
        if coordinates.len() != self.reduced_order()
            || coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(RbkrError::ReducedVectorDimensionMismatch);
        }
        let mut state = vec![0.0; self.state_dimension];
        for (coordinate, basis) in coordinates.iter().zip(&self.trial_basis) {
            for (slot, value) in state.iter_mut().zip(basis) {
                *slot += coordinate * value;
                if !slot.is_finite() {
                    return Err(RbkrError::NonFiniteArithmetic);
                }
            }
        }
        Ok(state)
    }

    /// Project a frozen discrete linearized system: Ar = W^T A V,
    /// Br = W^T B, and Cr = C V. A/B/C must be derived independently from the
    /// actual discrete solver, intervention map, and registered QoI map.
    pub fn project_linear_system(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        c: &[Vec<f64>],
    ) -> Result<ReducedLinearSystem, RbkrError> {
        let n = self.state_dimension;
        if a.len() != n
            || a.iter()
                .any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        {
            return Err(RbkrError::InvalidAOperator);
        }
        let input_count = b.first().map(Vec::len).unwrap_or(0);
        if b.len() != n
            || input_count == 0
            || b.iter()
                .any(|row| row.len() != input_count || row.iter().any(|value| !value.is_finite()))
        {
            return Err(RbkrError::InvalidBOperator);
        }
        if c.is_empty()
            || c.iter()
                .any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        {
            return Err(RbkrError::InvalidCOperator);
        }

        let order = self.reduced_order();
        let output_count = c.len();
        let mut a_trial = vec![vec![0.0; order]; n];
        for state_row in 0..n {
            for (basis_index, trial) in self.trial_basis.iter().enumerate() {
                a_trial[state_row][basis_index] = checked_dot(&a[state_row], trial)?;
            }
        }
        let mut reduced_a = vec![vec![0.0; order]; order];
        for (test, output_row) in self.test_basis.iter().zip(reduced_a.iter_mut()) {
            for (reduced_column, output) in output_row.iter_mut().enumerate() {
                *output = checked_dot(test, &matrix_column(&a_trial, reduced_column))?;
            }
        }
        let mut reduced_b = vec![vec![0.0; input_count]; order];
        for (test, output_row) in self.test_basis.iter().zip(reduced_b.iter_mut()) {
            for (input_index, output) in output_row.iter_mut().enumerate() {
                *output = checked_dot(test, &matrix_column(b, input_index))?;
            }
        }
        let mut reduced_c = vec![vec![0.0; order]; output_count];
        for output_index in 0..output_count {
            for (column, trial) in self.trial_basis.iter().enumerate() {
                reduced_c[output_index][column] = checked_dot(&c[output_index], trial)?;
            }
        }
        Ok(ReducedLinearSystem {
            a: reduced_a,
            b: reduced_b,
            c: reduced_c,
        })
    }

    /// Measure the response lost from a future discrete pulse injection.
    ///
    /// pulse_directions are columns of the derivative of the exact discrete
    /// pulse map with respect to its pulse parameters, evaluated at the pulse
    /// application state. post_pulse_observability contains QoI sensitivities
    /// pulled back to that same post-pulse state. The reported quantity is
    ///
    /// ||Y_p^T (I - V W^T) B_p||_F / ||Y_p^T B_p||_F.
    ///
    /// This is a fixed-basis, local linearized control-to-output diagnostic.
    /// A time-varying atlas must instead use the stage-local basis and qualify
    /// the inter-basis transfer separately.
    pub fn linearized_pulse_response_residual(
        &self,
        pulse_directions: &[Vec<f64>],
        post_pulse_observability: &EmpiricalGramian,
    ) -> Result<f64, RbkrError> {
        if post_pulse_observability.dimension != self.state_dimension {
            return Err(RbkrError::StateDimensionMismatch);
        }
        if pulse_directions.is_empty() || post_pulse_observability.weighted_factors.is_empty() {
            return Err(RbkrError::EmptyGramian);
        }

        let mut residual_norm = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        for direction in pulse_directions {
            validate_factor(direction, self.state_dimension)?;
            let reduced = self.reduce_state(direction)?;
            let projected = self.reconstruct_state(&reduced)?;
            let residual = direction
                .iter()
                .zip(&projected)
                .map(|(full, retained)| {
                    let value = full - retained;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(RbkrError::NonFiniteArithmetic)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            for output_factor in &post_pulse_observability.weighted_factors {
                validate_factor(output_factor, self.state_dimension)?;
                let reference = checked_dot(output_factor, direction)?;
                let lost = checked_dot(output_factor, &residual)?;
                reference_norm = reference_norm.hypot(reference);
                residual_norm = residual_norm.hypot(lost);
                if !reference_norm.is_finite() || !residual_norm.is_finite() {
                    return Err(RbkrError::NonFiniteArithmetic);
                }
            }
        }
        if reference_norm == 0.0 {
            return Err(RbkrError::NoReferenceResponse);
        }
        let ratio = residual_norm / reference_norm;
        if !ratio.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        Ok(ratio)
    }

    /// Measure the local linearized input-to-output response lost by this
    /// oblique projection for one frozen scenario.
    ///
    /// The residual is the Frobenius norm ratio of
    /// Y^T (I - V W^T) X to Y^T X, where X/Y are weighted reachability /
    /// observability factors. This is a finite-horizon linear diagnostic,
    /// not a nonlinear echo-error bound.
    pub fn linearized_response_residual(
        &self,
        reachability: &EmpiricalGramian,
        observability: &EmpiricalGramian,
    ) -> Result<f64, RbkrError> {
        if reachability.dimension != self.state_dimension
            || observability.dimension != self.state_dimension
        {
            return Err(RbkrError::StateDimensionMismatch);
        }
        if reachability.weighted_factors.is_empty()
            || observability.weighted_factors.is_empty()
        {
            return Err(RbkrError::EmptyGramian);
        }

        let mut residual_norm = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        for input_factor in &reachability.weighted_factors {
            validate_factor(input_factor, self.state_dimension)?;
            let reduced = self.reduce_state(input_factor)?;
            let reconstructed = self.reconstruct_state(&reduced)?;
            let residual = input_factor
                .iter()
                .zip(&reconstructed)
                .map(|(full, projected)| {
                    let value = full - projected;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(RbkrError::NonFiniteArithmetic)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            for output_factor in &observability.weighted_factors {
                validate_factor(output_factor, self.state_dimension)?;
                let reference_response = checked_dot(output_factor, input_factor)?;
                let residual_response = checked_dot(output_factor, &residual)?;
                reference_norm = reference_norm.hypot(reference_response);
                residual_norm = residual_norm.hypot(residual_response);
                if !reference_norm.is_finite() || !residual_norm.is_finite() {
                    return Err(RbkrError::NonFiniteArithmetic);
                }
            }
        }
        if reference_norm == 0.0 {
            return Err(RbkrError::NoReferenceResponse);
        }
        let ratio = residual_norm / reference_norm;
        if !ratio.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
        Ok(ratio)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedLinearSystem {
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub c: Vec<Vec<f64>>,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_factor(factor: &[f64], dimension: usize) -> Result<(), RbkrError> {
    if factor.len() != dimension {
        return Err(RbkrError::StateDimensionMismatch);
    }
    if factor.iter().any(|value| !value.is_finite()) {
        return Err(RbkrError::NonFiniteArithmetic);
    }
    Ok(())
}

fn checked_dot(left: &[f64], right: &[f64]) -> Result<f64, RbkrError> {
    if left.len() != right.len() {
        return Err(RbkrError::StateDimensionMismatch);
    }
    let mut sum = 0.0;
    for (a, b) in left.iter().zip(right) {
        sum += a * b;
        if !sum.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
    }
    Ok(sum)
}

fn column_squared_norm(matrix: &[Vec<f64>], column_index: usize) -> Result<f64, RbkrError> {
    let mut sum = 0.0;
    for row in matrix {
        let value = *row
            .get(column_index)
            .ok_or(RbkrError::NonFiniteArithmetic)?;
        sum += value * value;
        if !sum.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
    }
    Ok(sum)
}

fn matrix_column(matrix: &[Vec<f64>], index: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[index]).collect()
}

fn pairing_defect(trial: &[Vec<f64>], test: &[Vec<f64>]) -> Result<f64, RbkrError> {
    let mut defect = 0.0_f64;
    for (row, test_vector) in test.iter().enumerate() {
        for (column, trial_vector) in trial.iter().enumerate() {
            let expected = if row == column { 1.0 } else { 0.0 };
            defect = defect.max((checked_dot(test_vector, trial_vector)? - expected).abs());
        }
    }
    Ok(defect)
}

fn column_inner_product(matrix: &[Vec<f64>], left: usize, right: usize) -> Result<f64, RbkrError> {
    let mut sum = 0.0;
    for row in matrix {
        sum += row[left] * row[right];
        if !sum.is_finite() {
            return Err(RbkrError::NonFiniteArithmetic);
        }
    }
    Ok(sum)
}

/// One-sided Jacobi SVD: columns of the returned matrix are orthogonal and
/// right_vectors contains the matching right singular vectors.
type JacobiSvdResult = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn one_sided_jacobi_svd(
    mut matrix: Vec<Vec<f64>>,
    maximum_sweeps: usize,
    tolerance: f64,
) -> Result<JacobiSvdResult, RbkrError> {
    let rows = matrix.len();
    let columns = matrix.first().map(Vec::len).unwrap_or(0);
    if rows == 0 || columns == 0 || matrix.iter().any(|row| row.len() != columns) {
        return Err(RbkrError::EmptyGramian);
    }
    let mut right_vectors = vec![vec![0.0; columns]; columns];
    for (index, row) in right_vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    let mut converged = columns <= 1;
    for _ in 0..maximum_sweeps {
        if converged {
            break;
        }
        let mut rotated_any = false;
        for left in 0..columns {
            for right in (left + 1)..columns {
                let alpha = column_squared_norm(&matrix, left)?;
                let beta = column_squared_norm(&matrix, right)?;
                let gamma = column_inner_product(&matrix, left, right)?;
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                let scale = alpha.sqrt() * beta.sqrt();
                if gamma.abs() <= tolerance * scale {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                if !zeta.is_finite() {
                    return Err(RbkrError::NonFiniteArithmetic);
                }
                let denominator = zeta.abs() + zeta.hypot(1.0);
                let tangent = if zeta >= 0.0 {
                    1.0 / denominator
                } else {
                    -1.0 / denominator
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;

                for row in &mut matrix {
                    let a = row[left];
                    let b = row[right];
                    row[left] = cosine * a - sine * b;
                    row[right] = sine * a + cosine * b;
                }
                for row in &mut right_vectors {
                    let a = row[left];
                    let b = row[right];
                    row[left] = cosine * a - sine * b;
                    row[right] = sine * a + cosine * b;
                }
                rotated_any = true;
            }
        }
        if !rotated_any {
            converged = true;
        }
    }

    if !converged {
        for left in 0..columns {
            for right in (left + 1)..columns {
                let alpha = column_squared_norm(&matrix, left)?;
                let beta = column_squared_norm(&matrix, right)?;
                let gamma = column_inner_product(&matrix, left, right)?;
                if alpha > 0.0
                    && beta > 0.0
                    && gamma.abs() > 10.0 * tolerance * alpha.sqrt() * beta.sqrt()
                {
                    return Err(RbkrError::SvdDidNotConverge);
                }
            }
        }
    }
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || right_vectors
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(RbkrError::NonFiniteArithmetic);
    }
    Ok((matrix, right_vectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmpiricalGramian;

    fn gramian(dimension: usize, factors: &[Vec<f64>]) -> EmpiricalGramian {
        let mut result = EmpiricalGramian::new(dimension).unwrap();
        for factor in factors {
            result.add_direction(factor, 1.0).unwrap();
        }
        result
    }

    #[test]
    fn finite_horizon_ltv_gramians_pull_future_qoi_to_shared_checkpoint() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let future_transition = vec![vec![1.0, 0.0], vec![0.0, 10.0]];
        let past_b = vec![vec![1.0, 0.0], vec![0.0, 0.01]];
        let output_at_checkpoint = vec![vec![1.0, 0.0]];
        let output_at_horizon = vec![vec![0.0, 1.0]];

        let (reach, observe) = build_finite_horizon_gramians(
            &[identity],
            &[past_b],
            &[vec![1.0, 1.0]],
            &[future_transition],
            &[output_at_checkpoint, output_at_horizon],
            &[vec![0.0], vec![100.0]],
            FiniteHorizonGramianConfig::default(),
        )
        .unwrap();

        let reachable = reach.apply(&[1.0, 0.0]).unwrap();
        let observable = observe.apply(&[0.0, 1.0]).unwrap();
        assert!((reachable[0] - 1.0).abs() < 1.0e-12);
        assert!(reachable[1].abs() < 1.0e-12);
        assert!((observable[1] - 10_000.0).abs() < 1.0e-8);

        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1))
                .unwrap();
        assert!(basis.trial_basis()[0][0].abs() < 1.0e-12);
        assert!((basis.trial_basis()[0][1] - 0.01).abs() < 1.0e-10);
        assert!(basis.biorthogonality_defect() < 1.0e-8);
    }

    #[test]
    fn finite_horizon_builder_propagates_past_inputs_and_future_adjoint_to_checkpoint() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let past_growth = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let future_growth = vec![vec![1.0, 0.0], vec![0.0, 10.0]];
        let past_input = vec![vec![1.0, 0.0], vec![0.0, 0.005]];
        let no_late_input = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let first_output = vec![vec![1.0, 0.0]];
        let terminal_output = vec![vec![0.0, 1.0]];

        let (reach, observe) = build_finite_horizon_gramians(
            &[identity.clone(), past_growth],
            &[past_input, no_late_input],
            &[vec![1.0, 1.0], vec![1.0, 1.0]],
            &[identity, future_growth],
            &[first_output.clone(), first_output, terminal_output],
            &[vec![0.0], vec![0.0], vec![100.0]],
            FiniteHorizonGramianConfig::default(),
        )
        .unwrap();

        let reachable = reach.apply(&[0.0, 1.0]).unwrap();
        let observable = observe.apply(&[0.0, 1.0]).unwrap();
        assert!((reachable[1] - 1.0e-4).abs() < 1.0e-12);
        assert!((observable[1] - 10_000.0).abs() < 1.0e-8);

        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1))
                .unwrap();
        assert!(basis.trial_basis()[0][0].abs() < 1.0e-12);
        assert!((basis.trial_basis()[0][1] - 0.01).abs() < 1.0e-10);
    }

    #[test]
    fn finite_horizon_gramian_builder_refuses_invalid_axes_and_budgets() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let input = vec![vec![1.0], vec![0.0]];
        let output = vec![vec![1.0, 0.0]];
        assert_eq!(
            build_finite_horizon_gramians(
                std::slice::from_ref(&identity),
                std::slice::from_ref(&input),
                &[vec![1.0]],
                &[],
                &[],
                &[],
                FiniteHorizonGramianConfig::default(),
            ),
            Err(FiniteHorizonGramianError::InvalidTrajectoryAxes)
        );
        assert_eq!(
            build_finite_horizon_gramians(
                std::slice::from_ref(&identity),
                std::slice::from_ref(&input),
                &[vec![0.0]],
                &[],
                std::slice::from_ref(&output),
                &[vec![1.0]],
                FiniteHorizonGramianConfig::default(),
            ),
            Err(FiniteHorizonGramianError::InvalidInputCost)
        );
        assert_eq!(
            build_finite_horizon_gramians(
                &[identity],
                &[input],
                &[vec![1.0]],
                &[],
                &[output],
                &[vec![1.0]],
                FiniteHorizonGramianConfig::new(1, 100),
            ),
            Err(FiniteHorizonGramianError::FactorStorageBudgetExceeded)
        );
    }

    #[test]
    fn pulse_injection_response_is_measured_separately_from_checkpoint_state_response() {
        let reach = gramian(2, &[vec![1.0, 0.0]]);
        let checkpoint_observe = gramian(2, &[vec![1.0, 0.0]]);
        let basis = ResponseBalancedBasis::from_empirical_gramians(
            &reach,
            &checkpoint_observe,
            RbkrConfig::new(1),
        )
        .unwrap();
        let post_pulse_observe = gramian(2, &[vec![0.0, 1.0]]);

        let preserved = basis
            .linearized_pulse_response_residual(
                &[vec![1.0, 0.0]],
                &checkpoint_observe,
            )
            .unwrap();
        let lost = basis
            .linearized_pulse_response_residual(
                &[vec![0.0, 1.0]],
                &post_pulse_observe,
            )
            .unwrap();

        assert!(preserved < 1.0e-12);
        assert!((lost - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn average_balancing_cannot_hide_worst_case_scenario_response_loss() {
        let reach_a = gramian(2, &[vec![1.0, 0.0]]);
        let observe_a = gramian(2, &[vec![1.0, 0.0]]);
        let reach_b = gramian(2, &[vec![0.0, 1.0]]);
        let observe_b = gramian(2, &[vec![0.0, 1.0]]);
        let scenarios = [
            ResponseGramianScenario {
                scenario_id: "dominant-common-pulse",
                trajectory_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &reach_a,
                observability: &observe_a,
                scenario_weight: 0.51,
            },
            ResponseGramianScenario {
                scenario_id: "rare-adversarial-pulse",
                trajectory_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &reach_b,
                observability: &observe_b,
                scenario_weight: 0.49,
            },
        ];
        let (reach, observe) = aggregate_response_gramians(
            &scenarios,
            ResponseGramianAggregationConfig::default(),
        )
        .unwrap();
        let basis = ResponseBalancedBasis::from_empirical_gramians(
            &reach,
            &observe,
            RbkrConfig::new(1),
        )
        .unwrap();
        let report = evaluate_response_scenario_residuals(&basis, &scenarios, 0.9).unwrap();

        assert!(report.scenarios[0].response_residual < 1.0e-10);
        assert!((report.scenarios[1].response_residual - 1.0).abs() < 1.0e-10);
        assert!((report.weighted_mean - 0.49).abs() < 1.0e-10);
        assert!((report.worst_case - 1.0).abs() < 1.0e-10);
        assert!((report.cvar - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn response_scenario_aggregation_weights_families_at_one_checkpoint() {
        let reach_a = gramian(2, &[vec![1.0, 0.0]]);
        let observe_a = gramian(2, &[vec![0.5, 0.0]]);
        let reach_b = gramian(2, &[vec![0.0, 1.0]]);
        let observe_b = gramian(2, &[vec![0.0, 2.0]]);
        let scenarios = [
            ResponseGramianScenario {
                scenario_id: "scenario-a",
                trajectory_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-sha256-abc",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "nondimensional-state-metric-v1",
                reachability: &reach_a,
                observability: &observe_a,
                scenario_weight: 0.25,
            },
            ResponseGramianScenario {
                scenario_id: "scenario-b",
                trajectory_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-sha256-abc",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "nondimensional-state-metric-v1",
                reachability: &reach_b,
                observability: &observe_b,
                scenario_weight: 0.75,
            },
        ];

        let (reach, observe) =
            aggregate_response_gramians(&scenarios, ResponseGramianAggregationConfig::default())
                .unwrap();
        assert!(
            (checked_dot(&[1.0, 0.0], &reach.apply(&[1.0, 0.0]).unwrap()).unwrap() - 0.25).abs()
                < 1.0e-12
        );
        assert!(
            (checked_dot(&[0.0, 1.0], &reach.apply(&[0.0, 1.0]).unwrap()).unwrap() - 0.75).abs()
                < 1.0e-12
        );
        assert!(
            (checked_dot(&[1.0, 0.0], &observe.apply(&[1.0, 0.0]).unwrap()).unwrap() - 0.0625).abs()
                < 1.0e-12
        );
        assert!(
            (checked_dot(&[0.0, 1.0], &observe.apply(&[0.0, 1.0]).unwrap()).unwrap() - 3.0).abs()
                < 1.0e-12
        );

        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1))
                .unwrap();
        assert!(basis.biorthogonality_defect() < 1.0e-8);
    }

    #[test]
    fn response_scenario_aggregation_counts_zero_weight_source_storage() {
        let positive_reach = gramian(2, &[vec![1.0, 0.0]]);
        let positive_observe = gramian(2, &[vec![1.0, 0.0]]);
        let zero_weight_reach =
            gramian(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let zero_weight_observe =
            gramian(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let scenarios = [
            ResponseGramianScenario {
                scenario_id: "positive",
                trajectory_sha256: "1111111111111111111111111111111111111111111111111111111111111111",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &positive_reach,
                observability: &positive_observe,
                scenario_weight: 1.0,
            },
            ResponseGramianScenario {
                scenario_id: "zero-weight",
                trajectory_sha256: "2222222222222222222222222222222222222222222222222222222222222222",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &zero_weight_reach,
                observability: &zero_weight_observe,
                scenario_weight: 0.0,
            },
        ];
        // Source storage = 6 factors * 2 coordinates = 12 scalars;
        // aggregate storage = 2 factors * 2 coordinates = 4 scalars.
        // A budget of 12 must fail because peak live storage is 16.
        assert_eq!(
            aggregate_response_gramians(
                &scenarios,
                ResponseGramianAggregationConfig::new(12),
            ),
            Err(RbkrError::ScenarioFactorStorageBudgetExceeded)
        );
    }

    #[test]
    fn response_scenario_aggregation_is_scale_safe_and_rejects_duplicate_digests() {
        let reach_a = gramian(2, &[vec![1.0, 0.0]]);
        let observe_a = gramian(2, &[vec![1.0, 0.0]]);
        let reach_b = gramian(2, &[vec![0.0, 1.0]]);
        let observe_b = gramian(2, &[vec![0.0, 1.0]]);
        let scenarios = [
            ResponseGramianScenario {
                scenario_id: "large-weight-a",
                trajectory_sha256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &reach_a,
                observability: &observe_a,
                scenario_weight: 1.0e308,
            },
            ResponseGramianScenario {
                scenario_id: "large-weight-b",
                trajectory_sha256: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                task_contract_id: "kresp1-contract-v1",
                checkpoint_id: "checkpoint-a",
                checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
                state_metric_id: "metric-v1",
                reachability: &reach_b,
                observability: &observe_b,
                scenario_weight: 1.0e308,
            },
        ];
        let (reach, _) =
            aggregate_response_gramians(&scenarios, ResponseGramianAggregationConfig::default())
                .unwrap();
        let e1 = checked_dot(&[1.0, 0.0], &reach.apply(&[1.0, 0.0]).unwrap()).unwrap();
        let e2 = checked_dot(&[0.0, 1.0], &reach.apply(&[0.0, 1.0]).unwrap()).unwrap();
        assert!((e1 - 0.5).abs() < 1.0e-12);
        assert!((e2 - 0.5).abs() < 1.0e-12);

        let duplicate_digest = ResponseGramianScenario {
            trajectory_sha256: scenarios[0].trajectory_sha256,
            ..scenarios[1]
        };
        assert_eq!(
            aggregate_response_gramians(
                &[scenarios[0], duplicate_digest],
                ResponseGramianAggregationConfig::default(),
            ),
            Err(RbkrError::InvalidScenarioSet)
        );
    }

    #[test]
    fn response_scenario_aggregation_rejects_missing_positive_weight_evidence() {
        let empty_reach = EmpiricalGramian::new(2).unwrap();
        let observe = gramian(2, &[vec![1.0, 0.0]]);
        let scenario = ResponseGramianScenario {
            scenario_id: "scenario-empty",
            trajectory_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            task_contract_id: "kresp1-contract-v1",
            checkpoint_id: "checkpoint-a",
            checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
            state_metric_id: "metric-v1",
            reachability: &empty_reach,
            observability: &observe,
            scenario_weight: 1.0,
        };
        assert_eq!(
            aggregate_response_gramians(&[scenario], ResponseGramianAggregationConfig::default(),),
            Err(RbkrError::EmptyScenarioGramian)
        );
    }

    #[test]
    fn response_scenario_aggregation_rejects_coordinate_mixing_and_budget_overrun() {
        let reach = gramian(2, &[vec![1.0, 0.0]]);
        let observe = gramian(2, &[vec![1.0, 0.0]]);
        let base = ResponseGramianScenario {
            scenario_id: "scenario-base",
            trajectory_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            task_contract_id: "kresp1-contract-v1",
            checkpoint_id: "checkpoint-a",
            checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
            state_metric_id: "metric-v1",
            reachability: &reach,
            observability: &observe,
            scenario_weight: 1.0,
        };
        let other_checkpoint = ResponseGramianScenario {
            checkpoint_id: "checkpoint-b",
            checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            state_coordinate_id: "vlasov-1d1v-phase-space-layout-v1",
            ..base
        };
        assert_eq!(
            aggregate_response_gramians(
                &[base, other_checkpoint],
                ResponseGramianAggregationConfig::default(),
            ),
            Err(RbkrError::MixedScenarioCheckpoint)
        );
        let other_checkpoint_digest = ResponseGramianScenario {
            checkpoint_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ..base
        };
        assert_eq!(
            aggregate_response_gramians(
                &[base, other_checkpoint_digest],
                ResponseGramianAggregationConfig::default(),
            ),
            Err(RbkrError::MixedScenarioCheckpoint)
        );
        let other_coordinates = ResponseGramianScenario {
            state_coordinate_id: "vlasov-1d1v-permuted-layout-v1",
            ..base
        };
        assert_eq!(
            aggregate_response_gramians(
                &[base, other_coordinates],
                ResponseGramianAggregationConfig::default(),
            ),
            Err(RbkrError::MixedScenarioCoordinates)
        );
        assert_eq!(
            aggregate_response_gramians(&[base], ResponseGramianAggregationConfig::new(1),),
            Err(RbkrError::ScenarioFactorStorageBudgetExceeded)
        );
    }

    #[test]
    fn low_energy_direction_is_selected_when_future_observability_is_high() {
        // POD of the reachability Gramian alone selects e1 (energy 1 versus
        // 1e-4). The declared future output makes e2's balanced value larger.
        let reach = gramian(2, &[vec![1.0, 0.0], vec![0.0, 0.01]]);
        let observe = gramian(2, &[vec![0.1, 0.0], vec![0.0, 100.0]]);
        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1))
                .unwrap();

        assert_eq!(basis.reduced_order(), 1);
        assert!((basis.balanced_singular_values()[0] - 1.0).abs() < 1.0e-12);
        assert!((basis.balanced_singular_values()[1] - 0.1).abs() < 1.0e-12);
        assert!(basis.trial_basis()[0][0].abs() < 1.0e-12);
        assert!((basis.trial_basis()[0][1] - 0.01).abs() < 1.0e-12);
        assert!(basis.test_basis()[0][0].abs() < 1.0e-12);
        assert!((basis.test_basis()[0][1] - 100.0).abs() < 1.0e-10);
        assert!(basis.biorthogonality_defect() < 1.0e-10);

        let reduced = basis.reduce_state(&[0.0, 3.0]).unwrap();
        let restored = basis.reconstruct_state(&reduced).unwrap();
        assert!(restored[0].abs() < 1.0e-12);
        assert!((restored[1] - 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn full_order_bases_are_biorthogonal_for_non_aligned_factors() {
        let reach = gramian(
            3,
            &[
                vec![1.0, 0.5, 0.0],
                vec![0.0, 0.7, 0.2],
                vec![0.1, 0.0, 0.8],
            ],
        );
        let observe = gramian(
            3,
            &[
                vec![0.5, 0.1, 0.0],
                vec![0.0, 1.3, 0.4],
                vec![0.2, 0.0, 0.9],
            ],
        );
        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(3))
                .unwrap();
        assert!(basis.biorthogonality_defect() < 1.0e-8);

        for scale in [1.0, 2.0, -0.5] {
            let state = vec![scale, scale * 0.25, -scale * 2.0];
            let restored = basis
                .reconstruct_state(&basis.reduce_state(&state).unwrap())
                .unwrap();
            for (actual, expected) in restored.iter().zip(&state) {
                assert!((actual - expected).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn rectangular_factor_spaces_preserve_supported_state_rank() {
        // Three reachability factors and two observability factors create a
        // 2x3 cross-factor matrix. Its rank is two, the full state dimension.
        let reach = gramian(2, &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let observe = gramian(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(2))
                .unwrap();

        assert_eq!(basis.reduced_order(), 2);
        assert!(basis.biorthogonality_defect() < 1.0e-8);
        let state = vec![0.37, -1.25];
        let restored = basis
            .reconstruct_state(&basis.reduce_state(&state).unwrap())
            .unwrap();
        for (actual, expected) in restored.iter().zip(&state) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn reduced_abc_operators_match_full_dynamics_in_reduced_coordinates() {
        let reach = gramian(2, &[vec![1.0, 0.2], vec![0.1, 1.0]]);
        let observe = gramian(2, &[vec![0.7, 0.1], vec![0.2, 1.1]]);
        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(2))
                .unwrap();
        let a = vec![vec![0.8, 0.3], vec![-0.1, 0.9]];
        let b = vec![vec![1.0], vec![0.25]];
        let c = vec![vec![0.5, -0.2]];
        let reduced = basis.project_linear_system(&a, &b, &c).unwrap();

        let z = vec![0.4, -0.7];
        let full = basis.reconstruct_state(&z).unwrap();
        let full_next = a
            .iter()
            .map(|row| checked_dot(row, &full).unwrap())
            .collect::<Vec<_>>();
        let expected_next = basis.reduce_state(&full_next).unwrap();
        let actual_next = reduced
            .a
            .iter()
            .map(|row| checked_dot(row, &z).unwrap())
            .collect::<Vec<_>>();
        for (actual, expected) in actual_next.iter().zip(expected_next) {
            assert!((actual - expected).abs() < 1.0e-8);
        }

        let input = b.iter().map(|row| row[0]).collect::<Vec<_>>();
        let expected_input = basis.reduce_state(&input).unwrap();
        let actual_input = reduced.b.iter().map(|row| row[0]).collect::<Vec<_>>();
        for (actual, expected) in actual_input.iter().zip(expected_input) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        let expected_output = checked_dot(&c[0], &full).unwrap();
        let actual_output = checked_dot(&reduced.c[0], &z).unwrap();
        assert!((actual_output - expected_output).abs() < 1.0e-8);
    }

    #[test]
    fn empty_gramians_and_dimension_mismatches_are_rejected() {
        let empty = EmpiricalGramian::new(2).unwrap();
        let populated = gramian(2, &[vec![1.0, 0.0]]);
        assert_eq!(
            ResponseBalancedBasis::from_empirical_gramians(&empty, &populated, RbkrConfig::new(1),),
            Err(RbkrError::EmptyGramian)
        );
        let wrong = gramian(3, &[vec![1.0, 0.0, 0.0]]);
        assert_eq!(
            ResponseBalancedBasis::from_empirical_gramians(&populated, &wrong, RbkrConfig::new(1),),
            Err(RbkrError::StateDimensionMismatch)
        );
    }

    #[test]
    fn jacobi_svd_work_budget_is_checked_before_factor_rotations() {
        let reach = gramian(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let observe = gramian(2, &[vec![0.5, 0.0], vec![0.0, 0.5]]);
        let mut config = RbkrConfig::new(1);
        config.maximum_jacobi_scalar_operations = 1;
        assert_eq!(
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, config),
            Err(RbkrError::JacobiWorkBudgetExceeded)
        );
    }

    #[test]
    fn zero_cross_gramian_and_budget_exhaustion_fail_closed() {
        let reach = gramian(2, &[vec![1.0, 0.0]]);
        let observe = gramian(2, &[vec![0.0, 1.0]]);
        assert_eq!(
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1),),
            Err(RbkrError::InsufficientBalancedRank)
        );

        let mut config = RbkrConfig::new(1);
        config.maximum_cross_gramian_scalar_operations = 1;
        assert_eq!(
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, config),
            Err(RbkrError::CrossGramianWorkBudgetExceeded)
        );
    }

    #[test]
    fn non_finite_state_and_invalid_operator_shapes_are_rejected() {
        let reach = gramian(2, &[vec![1.0, 0.2], vec![0.1, 1.0]]);
        let observe = gramian(2, &[vec![0.7, 0.1], vec![0.2, 1.1]]);
        let basis =
            ResponseBalancedBasis::from_empirical_gramians(&reach, &observe, RbkrConfig::new(1))
                .unwrap();
        assert_eq!(
            basis.reconstruct_state(&[f64::NAN]),
            Err(RbkrError::ReducedVectorDimensionMismatch)
        );
        assert_eq!(
            basis.project_linear_system(&[vec![1.0]], &[vec![1.0], vec![1.0]], &[vec![1.0, 0.0]],),
            Err(RbkrError::InvalidAOperator)
        );
        assert_eq!(
            basis.project_linear_system(
                &[vec![1.0, 0.0], vec![0.0, 1.0]],
                &[vec![1.0], vec![f64::INFINITY]],
                &[vec![1.0, 0.0]],
            ),
            Err(RbkrError::InvalidBOperator)
        );
    }
}
