//! Mixed-strategy decision-observability bounds for finite hidden-world classes.
//!
//! This is evaluator-only. A candidate supplies one explicit probability
//! distribution over admissible actions for a fixed candidate-visible view.
//! The evaluator checks that the view, policy identity and distribution are
//! identical across hidden worlds, then compares the candidate with a
//! distributionally robust mixed-strategy minimax-regret policy.
//!
//! The optimizer uses full-information exponential weights on the finite
//! regret matrix. It is approximate; the returned primal/dual bounds are
//! computed from quantized valid strategies using integer arithmetic. Their
//! duality gap, not the iteration count, determines whether the configured
//! tolerance was met. This does not solve sequential stochastic policies or
//! general partial-observation games.
//!
//! Do not link this module into candidate code. Hidden-world IDs, digests and
//! action losses remain evaluator-owned.

use super::decision_observability::{
    ActionLoss, CandidateViewIdentity, DECISION_OBSERVABILITY_PPM,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const PPM: u128 = DECISION_OBSERVABILITY_PPM as u128;
const MAX_ALTERNATIVES: usize = 128;
const MAX_ACTIONS: usize = 64;
const MAX_ITERATIONS: u32 = 8_192;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionProbability {
    pub action_id: String,
    /// Probability in parts per million. All actions in the distribution sum
    /// to exactly 1_000_000 PPM.
    pub probability_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedStrategyCounterfactualWorld {
    pub evaluator_world_id: u64,
    pub hidden_world_sha256: [u8; 32],
    pub candidate_view: CandidateViewIdentity,
    /// Exact identity of the candidate policy implementation/profile.
    pub candidate_policy_sha256: [u8; 32],
    /// The candidate's probability kernel for this view, not a sampled action.
    pub candidate_action_distribution: Vec<ActionProbability>,
    /// Evaluator-only loss for each common admissible action.
    pub action_losses: Vec<ActionLoss>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedStrategyEquivalenceClass {
    pub equivalence_class_id: u64,
    pub alternatives: Vec<MixedStrategyCounterfactualWorld>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MixedSolverConfig {
    /// Fixed evaluator-side iteration budget. This must be frozen in the
    /// evaluator profile before confirmatory runs.
    pub iterations: u32,
    /// Exponential-weights learning rate in PPM, strictly between 0 and 1.
    pub learning_rate_ppm: u32,
    /// Maximum acceptable primal/dual gap in PPM.
    pub tolerance_ppm: u32,
}

impl Default for MixedSolverConfig {
    fn default() -> Self {
        Self {
            iterations: 4_096,
            learning_rate_ppm: 50_000,
            tolerance_ppm: 10_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedStrategyDecisionObservabilityCertificate {
    pub equivalence_class_id: u64,
    pub candidate_view: CandidateViewIdentity,
    pub candidate_policy_sha256: [u8; 32],
    pub alternative_world_count: u64,
    pub admissible_action_count: u64,
    pub candidate_action_distribution: Vec<ActionProbability>,
    pub candidate_probability_support_size: u64,
    /// Outward-rounded bounds on the candidate's worst-case expected regret.
    pub candidate_worst_case_expected_regret_lower_bound_ppm: u32,
    pub candidate_worst_case_expected_regret_upper_bound_ppm: u32,
    /// Best pure action's worst-case regret. This is not a lower bound for
    /// randomized policies; it is retained as an ablation/comparison.
    pub deterministic_minimax_regret_ppm: u32,
    /// Valid lower and upper bounds on the mixed-strategy minimax-regret value.
    pub mixed_minimax_lower_bound_ppm: u32,
    pub mixed_minimax_upper_bound_ppm: u32,
    pub mixed_solver_duality_gap_ppm: u32,
    /// Approximate candidate-side minimax strategy attaining the upper bound.
    pub mixed_minimax_strategy: Vec<ActionProbability>,
    /// Candidate regret above the unknown exact mixed minimax value lies in
    /// [avoidable_regret_lower_bound_ppm, avoidable_regret_upper_bound_ppm].
    pub avoidable_regret_lower_bound_ppm: u32,
    pub avoidable_regret_upper_bound_ppm: u32,
    pub solver_iterations: u32,
    pub solver_learning_rate_ppm: u32,
    pub solver_tolerance_ppm: u32,
    pub solver_tolerance_met: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedStrategyError {
    EmptyEquivalenceClass,
    TooManyAlternatives,
    TooManyActions,
    EmptyActionDistribution,
    CandidateViewsDiffer,
    CandidatePoliciesDiffer,
    CandidateDistributionsDiffer,
    DuplicateWorldId { world_id: u64 },
    DuplicateHiddenWorldDigest,
    EmptyActionId { world_id: u64 },
    DuplicateActionId { world_id: u64, action_id: String },
    ActionSetMismatch { world_id: u64 },
    InvalidProbability { action_id: String, probability_ppm: u32 },
    ProbabilitySumMismatch { sum_ppm: u64 },
    LossOutOfRange { world_id: u64, action_id: String, loss_ppm: u32 },
    InvalidSolverConfig,
    SolverNumericalFailure,
    ArithmeticOverflow,
}

impl fmt::Display for MixedStrategyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEquivalenceClass => write!(formatter, "equivalence class requires at least one world"),
            Self::TooManyAlternatives => write!(formatter, "equivalence class exceeds the mixed-strategy world limit"),
            Self::TooManyActions => write!(formatter, "action set exceeds the mixed-strategy action limit"),
            Self::EmptyActionDistribution => write!(formatter, "candidate action distribution cannot be empty"),
            Self::CandidateViewsDiffer => write!(formatter, "hidden worlds do not share an identical candidate-visible view"),
            Self::CandidatePoliciesDiffer => write!(formatter, "hidden worlds do not share an identical candidate policy identity"),
            Self::CandidateDistributionsDiffer => write!(formatter, "candidate action distributions differ across indistinguishable worlds"),
            Self::DuplicateWorldId { world_id } => write!(formatter, "duplicate evaluator world ID {world_id}"),
            Self::DuplicateHiddenWorldDigest => write!(formatter, "duplicate hidden-world digest in one equivalence class"),
            Self::EmptyActionId { world_id } => write!(formatter, "world {world_id} contains an empty action ID"),
            Self::DuplicateActionId { world_id, action_id } => write!(formatter, "world {world_id} repeats action ID {action_id}"),
            Self::ActionSetMismatch { world_id } => write!(formatter, "world {world_id} has a different admissible action set"),
            Self::InvalidProbability { action_id, probability_ppm } => write!(formatter, "action {action_id} has invalid probability {probability_ppm} PPM"),
            Self::ProbabilitySumMismatch { sum_ppm } => write!(formatter, "candidate action probabilities sum to {sum_ppm} PPM instead of 1_000_000"),
            Self::LossOutOfRange { world_id, action_id, loss_ppm } => write!(formatter, "world {world_id}, action {action_id} has out-of-range loss {loss_ppm} PPM"),
            Self::InvalidSolverConfig => write!(formatter, "mixed-strategy solver configuration is outside its registered bounds"),
            Self::SolverNumericalFailure => write!(formatter, "mixed-strategy solver produced non-finite or degenerate weights"),
            Self::ArithmeticOverflow => write!(formatter, "mixed-strategy evaluator arithmetic overflow"),
        }
    }
}

impl std::error::Error for MixedStrategyError {}

#[derive(Clone, Debug)]
struct PreparedWorld {
    regrets_ppm: Vec<u32>,
}

fn softmax(log_weights: &[f64]) -> Result<Vec<f64>, MixedStrategyError> {
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(MixedStrategyError::SolverNumericalFailure);
    }
    let mut weights: Vec<f64> = log_weights
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(MixedStrategyError::SolverNumericalFailure);
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

/// Quantize a probability vector to PPM using largest remainders. The result
/// sums exactly to PPM and breaks equal remainders by the original stable order.
fn quantize_probabilities(values: &[f64]) -> Result<Vec<u32>, MixedStrategyError> {
    if values.is_empty() {
        return Err(MixedStrategyError::SolverNumericalFailure);
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(MixedStrategyError::SolverNumericalFailure);
    }

    let mut quantized = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    let mut assigned = 0_u64;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(MixedStrategyError::SolverNumericalFailure);
        }
        let scaled = (*value / total) * DECISION_OBSERVABILITY_PPM as f64;
        let floor = scaled.floor();
        let base = u32::try_from(floor as u64)
            .map_err(|_| MixedStrategyError::ArithmeticOverflow)?;
        assigned = assigned
            .checked_add(u64::from(base))
            .ok_or(MixedStrategyError::ArithmeticOverflow)?;
        quantized.push(base);
        remainders.push((index, scaled - floor));
    }

    let remaining = u64::from(DECISION_OBSERVABILITY_PPM)
        .checked_sub(assigned)
        .ok_or(MixedStrategyError::ArithmeticOverflow)?;
    remainders.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    if remaining > remainders.len() as u64 {
        return Err(MixedStrategyError::ArithmeticOverflow);
    }
    for (index, _) in remainders.into_iter().take(remaining as usize) {
        quantized[index] = quantized[index]
            .checked_add(1)
            .ok_or(MixedStrategyError::ArithmeticOverflow)?;
    }
    Ok(quantized)
}

fn ceil_weighted_regret(
    regrets_ppm: &[u32],
    probabilities_ppm: &[u32],
) -> Result<u32, MixedStrategyError> {
    if regrets_ppm.len() != probabilities_ppm.len() {
        return Err(MixedStrategyError::ArithmeticOverflow);
    }
    let numerator = regrets_ppm
        .iter()
        .zip(probabilities_ppm)
        .try_fold(0_u128, |sum, (regret, probability)| {
            sum.checked_add(u128::from(*regret) * u128::from(*probability))
        })
        .ok_or(MixedStrategyError::ArithmeticOverflow)?;
    let rounded_up = numerator
        .checked_add(PPM - 1)
        .ok_or(MixedStrategyError::ArithmeticOverflow)?
        / PPM;
    u32::try_from(rounded_up).map_err(|_| MixedStrategyError::ArithmeticOverflow)
}

fn floor_weighted_regret(
    regrets_ppm: &[u32],
    probabilities_ppm: &[u32],
) -> Result<u32, MixedStrategyError> {
    if regrets_ppm.len() != probabilities_ppm.len() {
        return Err(MixedStrategyError::ArithmeticOverflow);
    }
    let numerator = regrets_ppm
        .iter()
        .zip(probabilities_ppm)
        .try_fold(0_u128, |sum, (regret, probability)| {
            sum.checked_add(u128::from(*regret) * u128::from(*probability))
        })
        .ok_or(MixedStrategyError::ArithmeticOverflow)?;
    u32::try_from(numerator / PPM).map_err(|_| MixedStrategyError::ArithmeticOverflow)
}

fn validate_solver_config(config: MixedSolverConfig) -> Result<(), MixedStrategyError> {
    if config.iterations == 0
        || config.iterations > MAX_ITERATIONS
        || config.learning_rate_ppm == 0
        || config.learning_rate_ppm > 250_000
        || config.tolerance_ppm > DECISION_OBSERVABILITY_PPM
    {
        return Err(MixedStrategyError::InvalidSolverConfig);
    }
    Ok(())
}

/// Evaluate a candidate probability kernel over a finite observational
/// equivalence class and approximate the mixed-strategy minimax-regret value.
///
/// Matrix rows are hidden worlds; columns are common admissible actions.
/// The candidate minimizes worst-case expected regret, while the evaluator's
/// dual mixture maximizes the corresponding lower bound. The exact minimax
/// value lies between the returned lower and upper bounds.
pub fn evaluate_mixed_strategy_decision_observability(
    class: &MixedStrategyEquivalenceClass,
    solver: MixedSolverConfig,
) -> Result<MixedStrategyDecisionObservabilityCertificate, MixedStrategyError> {
    validate_solver_config(solver)?;
    if class.alternatives.is_empty() {
        return Err(MixedStrategyError::EmptyEquivalenceClass);
    }
    if class.alternatives.len() > MAX_ALTERNATIVES {
        return Err(MixedStrategyError::TooManyAlternatives);
    }

    let first = &class.alternatives[0];
    if first.candidate_action_distribution.is_empty() {
        return Err(MixedStrategyError::EmptyActionDistribution);
    }
    if first.candidate_action_distribution.len() > MAX_ACTIONS {
        return Err(MixedStrategyError::TooManyActions);
    }

    let reference_view = &first.candidate_view;
    let reference_policy = first.candidate_policy_sha256;
    let reference_distribution = &first.candidate_action_distribution;

    let mut action_ids = Vec::with_capacity(reference_distribution.len());
    let mut distribution_sum = 0_u64;
    let mut candidate_probabilities = Vec::with_capacity(reference_distribution.len());
    let mut seen_actions = BTreeSet::new();
    for probability in reference_distribution {
        if probability.action_id.trim().is_empty() {
            return Err(MixedStrategyError::EmptyActionDistribution);
        }
        if !seen_actions.insert(probability.action_id.clone()) {
            return Err(MixedStrategyError::DuplicateActionId {
                world_id: first.evaluator_world_id,
                action_id: probability.action_id.clone(),
            });
        }
        if probability.probability_ppm > DECISION_OBSERVABILITY_PPM {
            return Err(MixedStrategyError::InvalidProbability {
                action_id: probability.action_id.clone(),
                probability_ppm: probability.probability_ppm,
            });
        }
        distribution_sum = distribution_sum
            .checked_add(u64::from(probability.probability_ppm))
            .ok_or(MixedStrategyError::ArithmeticOverflow)?;
        action_ids.push(probability.action_id.clone());
        candidate_probabilities.push(probability.probability_ppm);
    }
    if distribution_sum != u64::from(DECISION_OBSERVABILITY_PPM) {
        return Err(MixedStrategyError::ProbabilitySumMismatch {
            sum_ppm: distribution_sum,
        });
    }

    let mut world_ids = BTreeSet::new();
    let mut hidden_digests = BTreeSet::new();
    let mut prepared = Vec::with_capacity(class.alternatives.len());

    for world in &class.alternatives {
        if &world.candidate_view != reference_view {
            return Err(MixedStrategyError::CandidateViewsDiffer);
        }
        if world.candidate_policy_sha256 != reference_policy {
            return Err(MixedStrategyError::CandidatePoliciesDiffer);
        }
        if &world.candidate_action_distribution != reference_distribution {
            return Err(MixedStrategyError::CandidateDistributionsDiffer);
        }
        if !world_ids.insert(world.evaluator_world_id) {
            return Err(MixedStrategyError::DuplicateWorldId {
                world_id: world.evaluator_world_id,
            });
        }
        if !hidden_digests.insert(world.hidden_world_sha256) {
            return Err(MixedStrategyError::DuplicateHiddenWorldDigest);
        }
        if world.action_losses.len() != action_ids.len() {
            return Err(MixedStrategyError::ActionSetMismatch {
                world_id: world.evaluator_world_id,
            });
        }

        let mut losses = BTreeMap::new();
        for item in &world.action_losses {
            if item.action_id.trim().is_empty() {
                return Err(MixedStrategyError::EmptyActionId {
                    world_id: world.evaluator_world_id,
                });
            }
            if item.loss_ppm > DECISION_OBSERVABILITY_PPM {
                return Err(MixedStrategyError::LossOutOfRange {
                    world_id: world.evaluator_world_id,
                    action_id: item.action_id.clone(),
                    loss_ppm: item.loss_ppm,
                });
            }
            if losses.insert(item.action_id.clone(), item.loss_ppm).is_some() {
                return Err(MixedStrategyError::DuplicateActionId {
                    world_id: world.evaluator_world_id,
                    action_id: item.action_id.clone(),
                });
            }
        }
        if losses.len() != action_ids.len()
            || action_ids.iter().any(|action_id| !losses.contains_key(action_id))
        {
            return Err(MixedStrategyError::ActionSetMismatch {
                world_id: world.evaluator_world_id,
            });
        }

        let best_loss = losses.values().copied().min().ok_or(
            MixedStrategyError::ActionSetMismatch {
                world_id: world.evaluator_world_id,
            },
        )?;
        let regrets_ppm = action_ids
            .iter()
            .map(|action_id| {
                losses
                    .get(action_id)
                    .copied()
                    .map(|loss| loss - best_loss)
                    .ok_or(MixedStrategyError::ActionSetMismatch {
                        world_id: world.evaluator_world_id,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        prepared.push(PreparedWorld { regrets_ppm });
    }

    // Canonical world order prevents input serialization order from changing
    // floating-point summation or the iterative solver trajectory.
    prepared.sort_by_key(|world| world.world_id);

    let mut deterministic_minimax_regret = DECISION_OBSERVABILITY_PPM;
    for action_index in 0..action_ids.len() {
        let worst = prepared
            .iter()
            .map(|world| world.regrets_ppm[action_index])
            .max()
            .ok_or(MixedStrategyError::EmptyEquivalenceClass)?;
        deterministic_minimax_regret = deterministic_minimax_regret.min(worst);
    }

    let mut candidate_worst_case_lower = 0_u32;
    let mut candidate_worst_case_upper = 0_u32;
    for world in &prepared {
        candidate_worst_case_lower = candidate_worst_case_lower.max(floor_weighted_regret(
            &world.regrets_ppm,
            &candidate_probabilities,
        )?);
        candidate_worst_case_upper = candidate_worst_case_upper.max(ceil_weighted_regret(
            &world.regrets_ppm,
            &candidate_probabilities,
        )?);
    }

    let action_count = action_ids.len();
    let world_count = prepared.len();
    // If a single action is optimal in every observationally equivalent
    // world, the mixed minimax-regret value is exactly zero. Use that action
    // directly rather than reporting the finite-iteration optimizer's small
    // residual probability on dominated actions as an approximate bound.
    let common_optimal_action_index = (0..action_count).find(|action_index| {
        prepared
            .iter()
            .all(|world| world.regrets_ppm[*action_index] == 0)
    });

    let (minimax_probabilities, adversarial_probabilities, solver_iterations_used) =
        if let Some(common_index) = common_optimal_action_index {
            let mut minimax_probabilities = vec![0_u32; action_count];
            minimax_probabilities[common_index] = DECISION_OBSERVABILITY_PPM;
            let mut adversarial_probabilities = vec![0_u32; world_count];
            adversarial_probabilities[0] = DECISION_OBSERVABILITY_PPM;
            (minimax_probabilities, adversarial_probabilities, 0)
        } else {
            let mut log_action_weights = vec![0.0_f64; action_count];
            let mut log_world_weights = vec![0.0_f64; world_count];
            let mut average_action = vec![0.0_f64; action_count];
            let mut average_world = vec![0.0_f64; world_count];
            let learning_rate =
                f64::from(solver.learning_rate_ppm) / f64::from(DECISION_OBSERVABILITY_PPM);
            let iterations = f64::from(solver.iterations);

            for _ in 0..solver.iterations {
                let action_distribution = softmax(&log_action_weights)?;
                let world_distribution = softmax(&log_world_weights)?;

                for (average, probability) in average_action.iter_mut().zip(&action_distribution) {
                    *average += *probability;
                }
                for (average, probability) in average_world.iter_mut().zip(&world_distribution) {
                    *average += *probability;
                }

                let mut expected_loss_by_action = vec![0.0_f64; action_count];
                let mut expected_loss_by_world = vec![0.0_f64; world_count];
                for (world_index, world) in prepared.iter().enumerate() {
                    let adversarial_probability = world_distribution[world_index];
                    for action_index in 0..action_count {
                        let regret = f64::from(world.regrets_ppm[action_index])
                            / f64::from(DECISION_OBSERVABILITY_PPM);
                        expected_loss_by_action[action_index] += adversarial_probability * regret;
                        expected_loss_by_world[world_index] +=
                            action_distribution[action_index] * regret;
                    }
                }

                for (weight, expected_loss) in
                    log_action_weights.iter_mut().zip(expected_loss_by_action)
                {
                    *weight -= learning_rate * expected_loss;
                }
                for (weight, expected_loss) in
                    log_world_weights.iter_mut().zip(expected_loss_by_world)
                {
                    *weight += learning_rate * expected_loss;
                }
            }

            for value in &mut average_action {
                *value /= iterations;
            }
            for value in &mut average_world {
                *value /= iterations;
            }

            (
                quantize_probabilities(&average_action)?,
                quantize_probabilities(&average_world)?,
                solver.iterations,
            )
        };

    let mut mixed_upper_bound = 0_u32;
    for world in &prepared {
        mixed_upper_bound = mixed_upper_bound.max(ceil_weighted_regret(
            &world.regrets_ppm,
            &minimax_probabilities,
        )?);
    }

    let mut mixed_lower_bound = DECISION_OBSERVABILITY_PPM;
    for action_index in 0..action_count {
        let action_regrets: Vec<u32> = prepared
            .iter()
            .map(|world| world.regrets_ppm[action_index])
            .collect();
        mixed_lower_bound = mixed_lower_bound.min(floor_weighted_regret(
            &action_regrets,
            &adversarial_probabilities,
        )?);
    }

    let duality_gap = mixed_upper_bound
        .checked_sub(mixed_lower_bound)
        .ok_or(MixedStrategyError::ArithmeticOverflow)?;
    // The candidate value can be fractional in PPM units. Use outward-rounded
    // endpoints for the avoidable-regret interval as well.
    let avoidable_lower = candidate_worst_case_lower.saturating_sub(mixed_upper_bound);
    let avoidable_upper = candidate_worst_case_upper.saturating_sub(mixed_lower_bound);

    let mixed_strategy = action_ids
        .iter()
        .cloned()
        .zip(minimax_probabilities.iter().copied())
        .map(|(action_id, probability_ppm)| ActionProbability {
            action_id,
            probability_ppm,
        })
        .collect();
    let support_size = candidate_probabilities
        .iter()
        .filter(|probability| **probability > 0)
        .count() as u64;

    Ok(MixedStrategyDecisionObservabilityCertificate {
        equivalence_class_id: class.equivalence_class_id,
        candidate_view: reference_view.clone(),
        candidate_policy_sha256: reference_policy,
        alternative_world_count: u64::try_from(world_count)
            .map_err(|_| MixedStrategyError::ArithmeticOverflow)?,
        admissible_action_count: u64::try_from(action_count)
            .map_err(|_| MixedStrategyError::ArithmeticOverflow)?,
        candidate_action_distribution: reference_distribution.clone(),
        candidate_probability_support_size: support_size,
        candidate_worst_case_expected_regret_lower_bound_ppm: candidate_worst_case_lower,
        candidate_worst_case_expected_regret_upper_bound_ppm: candidate_worst_case_upper,
        deterministic_minimax_regret_ppm: deterministic_minimax_regret,
        mixed_minimax_lower_bound_ppm: mixed_lower_bound,
        mixed_minimax_upper_bound_ppm: mixed_upper_bound,
        mixed_solver_duality_gap_ppm: duality_gap,
        mixed_minimax_strategy,
        avoidable_regret_lower_bound_ppm: avoidable_lower,
        avoidable_regret_upper_bound_ppm: avoidable_upper,
        solver_iterations: solver_iterations_used,
        solver_learning_rate_ppm: solver.learning_rate_ppm,
        solver_tolerance_ppm: solver.tolerance_ppm,
        solver_tolerance_met: duality_gap <= solver.tolerance_ppm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn view() -> CandidateViewIdentity {
        CandidateViewIdentity {
            observation_history_sha256: digest(1),
            recurrent_state_sha256: digest(2),
            model_configuration_sha256: digest(3),
            policy_seed: 17,
        }
    }

    fn distribution(left_ppm: u32, right_ppm: u32) -> Vec<ActionProbability> {
        vec![
            ActionProbability {
                action_id: "left".to_owned(),
                probability_ppm: left_ppm,
            },
            ActionProbability {
                action_id: "right".to_owned(),
                probability_ppm: right_ppm,
            },
        ]
    }

    fn world(
        world_id: u64,
        hidden_digest: u8,
        candidate_distribution: Vec<ActionProbability>,
        losses: &[(&str, u32)],
    ) -> MixedStrategyCounterfactualWorld {
        MixedStrategyCounterfactualWorld {
            evaluator_world_id: world_id,
            hidden_world_sha256: digest(hidden_digest),
            candidate_view: view(),
            candidate_policy_sha256: digest(4),
            candidate_action_distribution: candidate_distribution,
            action_losses: losses
                .iter()
                .map(|(action_id, loss_ppm)| ActionLoss {
                    action_id: (*action_id).to_owned(),
                    loss_ppm: *loss_ppm,
                })
                .collect(),
        }
    }

    fn opposed_worlds(candidate_distribution: Vec<ActionProbability>) -> MixedStrategyEquivalenceClass {
        MixedStrategyEquivalenceClass {
            equivalence_class_id: 7,
            alternatives: vec![
                world(1, 10, candidate_distribution.clone(), &[("left", 0), ("right", 1_000_000)]),
                world(2, 11, candidate_distribution, &[("left", 1_000_000), ("right", 0)]),
            ],
        }
    }

    #[test]
    fn mixed_strategy_improves_the_worst_case_ambiguity_floor() {
        let class = opposed_worlds(distribution(500_000, 500_000));
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert_eq!(result.deterministic_minimax_regret_ppm, 1_000_000);
        assert_eq!(result.candidate_worst_case_expected_regret_lower_bound_ppm, 500_000);
        assert_eq!(result.candidate_worst_case_expected_regret_upper_bound_ppm, 500_000);
        assert_eq!(result.mixed_minimax_lower_bound_ppm, 500_000);
        assert_eq!(result.mixed_minimax_upper_bound_ppm, 500_000);
        assert_eq!(result.mixed_solver_duality_gap_ppm, 0);
        assert_eq!(result.avoidable_regret_lower_bound_ppm, 0);
        assert_eq!(result.avoidable_regret_upper_bound_ppm, 0);
        assert!(result.solver_tolerance_met);
    }

    #[test]
    fn candidate_policy_excess_is_bounded_above_mixed_minimax_value() {
        let class = opposed_worlds(distribution(1_000_000, 0));
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert_eq!(result.candidate_worst_case_expected_regret_lower_bound_ppm, 1_000_000);
        assert_eq!(result.candidate_worst_case_expected_regret_upper_bound_ppm, 1_000_000);
        assert!(result.avoidable_regret_lower_bound_ppm >= 490_000);
        assert!(result.avoidable_regret_upper_bound_ppm >= result.avoidable_regret_lower_bound_ppm);
        assert!(result.solver_tolerance_met);
    }

    #[test]
    fn common_optimal_action_produces_zero_mixed_floor() {
        let class = MixedStrategyEquivalenceClass {
            equivalence_class_id: 8,
            alternatives: vec![
                world(1, 10, distribution(1_000_000, 0), &[("left", 0), ("right", 100_000)]),
                world(2, 11, distribution(1_000_000, 0), &[("left", 0), ("right", 200_000)]),
            ],
        };
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert_eq!(result.mixed_minimax_lower_bound_ppm, 0);
        assert_eq!(result.mixed_minimax_upper_bound_ppm, 0);
        assert_eq!(result.candidate_worst_case_expected_regret_lower_bound_ppm, 0);
        assert_eq!(result.candidate_worst_case_expected_regret_upper_bound_ppm, 0);
        assert_eq!(result.mixed_minimax_strategy, distribution(1_000_000, 0));
        assert_eq!(result.solver_iterations, 0);
    }

    #[test]
    fn candidate_distributions_must_match_across_hidden_worlds() {
        let mut class = opposed_worlds(distribution(500_000, 500_000));
        class.alternatives[1].candidate_action_distribution = distribution(600_000, 400_000);
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, MixedSolverConfig::default()),
            Err(MixedStrategyError::CandidateDistributionsDiffer)
        );
    }

    #[test]
    fn candidate_view_and_policy_identity_must_match() {
        let mut class = opposed_worlds(distribution(500_000, 500_000));
        class.alternatives[1].candidate_view.policy_seed = 18;
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, MixedSolverConfig::default()),
            Err(MixedStrategyError::CandidateViewsDiffer)
        );

        let mut class = opposed_worlds(distribution(500_000, 500_000));
        class.alternatives[1].candidate_policy_sha256 = digest(9);
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, MixedSolverConfig::default()),
            Err(MixedStrategyError::CandidatePoliciesDiffer)
        );
    }

    #[test]
    fn probability_mass_must_sum_to_one_million_ppm() {
        let class = opposed_worlds(distribution(400_000, 400_000));
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, MixedSolverConfig::default()),
            Err(MixedStrategyError::ProbabilitySumMismatch { sum_ppm: 800_000 })
        );
    }

    #[test]
    fn action_universe_must_be_identical_and_complete() {
        let mut class = opposed_worlds(distribution(500_000, 500_000));
        class.alternatives[1].action_losses = vec![
            ActionLoss { action_id: "left".to_owned(), loss_ppm: 0 },
            ActionLoss { action_id: "hold".to_owned(), loss_ppm: 0 },
        ];
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, MixedSolverConfig::default()),
            Err(MixedStrategyError::ActionSetMismatch { world_id: 2 })
        );
    }

    #[test]
    fn deterministic_point_mass_is_a_valid_mixed_distribution() {
        let class = opposed_worlds(distribution(1_000_000, 0));
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert_eq!(result.candidate_probability_support_size, 1);
        assert_eq!(result.candidate_worst_case_expected_regret_lower_bound_ppm, 1_000_000);
        assert_eq!(result.candidate_worst_case_expected_regret_upper_bound_ppm, 1_000_000);
        assert!(result.mixed_minimax_upper_bound_ppm <= result.deterministic_minimax_regret_ppm);
    }

    #[test]
    fn primal_dual_bounds_are_ordered_and_gap_is_exactly_reported() {
        let class = MixedStrategyEquivalenceClass {
            equivalence_class_id: 9,
            alternatives: vec![
                world(1, 10, distribution(700_000, 300_000), &[("left", 100_000), ("right", 900_000)]),
                world(2, 11, distribution(700_000, 300_000), &[("left", 800_000), ("right", 200_000)]),
                world(3, 12, distribution(700_000, 300_000), &[("left", 400_000), ("right", 600_000)]),
            ],
        };
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert!(result.mixed_minimax_lower_bound_ppm <= result.mixed_minimax_upper_bound_ppm);
        assert_eq!(
            result.mixed_solver_duality_gap_ppm,
            result.mixed_minimax_upper_bound_ppm - result.mixed_minimax_lower_bound_ppm
        );
        assert!(result.mixed_minimax_upper_bound_ppm <= 1_000_000);
    }

    #[test]
    fn fractional_candidate_regret_uses_outward_rounded_avoidable_bounds() {
        let class = MixedStrategyEquivalenceClass {
            equivalence_class_id: 10,
            alternatives: vec![
                world(1, 10, distribution(500_000, 500_000), &[("left", 0), ("right", 1)]),
                world(2, 11, distribution(500_000, 500_000), &[("left", 0), ("right", 1)]),
            ],
        };
        let result = evaluate_mixed_strategy_decision_observability(
            &class,
            MixedSolverConfig::default(),
        )
        .unwrap();
        assert_eq!(result.candidate_worst_case_expected_regret_lower_bound_ppm, 0);
        assert_eq!(result.candidate_worst_case_expected_regret_upper_bound_ppm, 1);
        assert_eq!(result.mixed_minimax_lower_bound_ppm, 0);
        assert_eq!(result.mixed_minimax_upper_bound_ppm, 0);
        assert_eq!(result.avoidable_regret_lower_bound_ppm, 0);
        assert_eq!(result.avoidable_regret_upper_bound_ppm, 1);
    }

    #[test]
    fn solver_rejects_unbounded_or_invalid_profiles() {
        let class = opposed_worlds(distribution(500_000, 500_000));
        let config = MixedSolverConfig {
            iterations: MAX_ITERATIONS + 1,
            ..MixedSolverConfig::default()
        };
        assert_eq!(
            evaluate_mixed_strategy_decision_observability(&class, config),
            Err(MixedStrategyError::InvalidSolverConfig)
        );
    }
}
