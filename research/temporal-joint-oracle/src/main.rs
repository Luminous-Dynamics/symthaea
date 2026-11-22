//! Evaluator-only exact oracle for small joint temporal-control episodes.
//!
//! This binary accepts hidden future outcomes and measures exact feasible
//! alternatives. It must never be linked into or executed inside candidate
//! processes. The candidate crate has no dependency on this package.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_temporal_joint_control::{
    ActionKind, JointBudget, JointPlan, ResourceDimension, ResourceVector,
};

pub const MAX_EXACT_FACTORS: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeFingerprint {
    pub episode_id: u64,
    pub task_sha256: [u8; 32],
    pub future_stream_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub initial_state_sha256: [u8; 32],
    pub candidate_feature_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    pub total_exposure_us: u64,
    pub budget: JointBudget,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExposureDebt {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

impl ExposureDebt {
    fn validate(self, total_exposure_us: u64) -> Result<(), OracleError> {
        let total = self.wrong_belief_us
            .checked_add(self.abstention_us)
            .and_then(|value| value.checked_add(self.missing_output_us))
            .ok_or(OracleError::ArithmeticOverflow)?;
        if total > total_exposure_us {
            return Err(OracleError::ExposureExceedsEpisode {
                total,
                maximum: total_exposure_us,
            });
        }
        Ok(())
    }

    fn values(self) -> [u64; 3] {
        [self.wrong_belief_us, self.abstention_us, self.missing_output_us]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatedAlternative {
    pub run_id: u64,
    pub context: EpisodeFingerprint,
    pub plan: JointPlan,
    /// Measured values, not candidate estimates. Every dimension is explicit;
    /// adapters must reject missing or uncalibrated measurements before intake.
    pub measured_cost: ResourceVector,
    pub observed_maximum_gap_us: u64,
    /// Evaluator-only exposure measured against the committed hidden future.
    pub realized_debt: ExposureDebt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactJointOracleInput {
    /// Action IDs whose on/off combinations define the exact small problem.
    /// Four factors can represent sensing, memory, compute and verification.
    pub factor_action_ids: Vec<u64>,
    pub candidate_plan_id: u64,
    pub alternatives: Vec<EvaluatedAlternative>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExposureRegret {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactJointOracleReport {
    pub evaluated_alternatives: usize,
    pub candidate_plan_id: u64,
    pub candidate_debt: ExposureDebt,
    pub best_plan_ids_by_axis: [u64; 3],
    pub componentwise_regret: ExposureRegret,
    pub pareto_plan_ids: Vec<u64>,
}

/// One exact factorial (Moebius) interaction term for a subset of joint actions.
///
/// Exposure dividends are expressed as reductions in debt relative to the
/// empty-action baseline: positive values are synergy, negative values are
/// antagonism. Resource dividends are incremental measured costs relative to
/// that same baseline: positive values are additional cost, negative values
/// are resource savings. Resource array order is ResourceDimension::ALL and
/// ResourceVector::values().
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorialInteractionTerm {
    pub subset_mask: u64,
    pub interaction_order: u32,
    /// Wrong-belief, abstention, missing-output exposure benefit dividends.
    pub exposure_benefit_dividend_us: [i128; 3],
    /// Twelve resource-cost dividends in ResourceDimension::ALL order.
    pub resource_cost_dividend: [i128; 12],
}

/// Net higher-order contribution aggregated by interaction order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionOrderSummary {
    pub interaction_order: u32,
    pub term_count: usize,
    pub exposure_benefit_dividend_us: [i128; 3],
    /// Resource-cost dividends in ResourceDimension::ALL order.
    pub resource_cost_dividend: [i128; 12],
}

/// Exact decomposition of a complete binary factorial over evaluator-measured
/// joint outcomes. The sum of all non-empty subset dividends reconstructs the
/// full-joint effect relative to the empty-action baseline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorialInteractionReport {
    pub factor_action_ids: Vec<u64>,
    pub factor_action_kinds: Vec<ActionKind>,
    pub evaluated_subset_count: usize,
    pub baseline_plan_id: u64,
    pub full_joint_plan_id: u64,
    pub baseline_debt: ExposureDebt,
    pub full_joint_debt: ExposureDebt,
    /// Baseline debt minus full-joint debt, by exposure axis.
    pub total_exposure_benefit_us: [i128; 3],
    /// Full-joint measured cost minus baseline measured cost, in resource-axis order.
    pub total_resource_cost_delta: [i128; 12],
    /// Includes first-order main effects and every higher-order interaction.
    pub interaction_terms: Vec<FactorialInteractionTerm>,
    pub interaction_order_summaries: Vec<InteractionOrderSummary>,
    /// Common denominator for every exact Shapley attribution.
    pub shapley_common_denominator: u64,
    pub factor_shapley_attributions: Vec<FactorShapleyAttribution>,
}

/// Shapley allocation of the complete joint effect to one binary action factor.
///
/// Every numerator is expressed over the report's shared
/// `shapley_common_denominator`. Divide by that denominator to recover the
/// exact rational microseconds or resource units attributed to this factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorShapleyAttribution {
    pub action_id: u64,
    pub action_kind: ActionKind,
    /// Exact rational exposure-benefit attribution numerator.
    pub exposure_benefit_scaled_numerator_us: [i128; 3],
    /// Exact rational resource-cost attribution numerator in
    /// ResourceDimension::ALL / ResourceVector::values() order.
    pub resource_cost_scaled_numerator: [i128; 12],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OracleError {
    EmptyAlternatives,
    TooManyFactors { requested: usize, maximum: usize },
    DuplicateOrUnsortedFactors,
    ContextMismatch,
    DuplicateRunId { run_id: u64 },
    DuplicatePlanId { plan_id: u64 },
    DuplicateOrUnsortedActions { plan_id: u64 },
    UnknownFactorAction { action_id: u64 },
    ActionKindChanged { action_id: u64 },
    DuplicateSubset { mask: u64 },
    MissingSubset { mask: u64 },
    CandidatePlanMissing { plan_id: u64 },
    ResourceOverrun { run_id: u64, dimensions: Vec<ResourceDimension> },
    ObservationGapOverrun { run_id: u64, observed_us: u64, maximum_us: u64 },
    ExposureExceedsEpisode { total: u64, maximum: u64 },
    InteractionReconstructionMismatch,
    ArithmeticOverflow,

}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlternatives => write!(f, "exact oracle requires at least one evaluated alternative"),
            Self::TooManyFactors { requested, maximum } => write!(f, "exact enumeration requested {requested} factors; maximum is {maximum}"),
            Self::DuplicateOrUnsortedFactors => write!(f, "factor action IDs must be unique and strictly increasing"),
            Self::ContextMismatch => write!(f, "paired alternatives do not share an identical committed episode context"),
            Self::DuplicateRunId { run_id } => write!(f, "duplicate evaluator run ID {run_id}"),
            Self::DuplicatePlanId { plan_id } => write!(f, "duplicate joint plan ID {plan_id}"),
            Self::DuplicateOrUnsortedActions { plan_id } => write!(f, "plan {plan_id} action IDs are not strictly increasing"),
            Self::UnknownFactorAction { action_id } => write!(f, "unknown factor action ID {action_id}"),
            Self::ActionKindChanged { action_id } => write!(f, "factor action {action_id} changes kind between alternatives"),
            Self::DuplicateSubset { mask } => write!(f, "factor subset {mask:#x} was evaluated more than once"),
            Self::MissingSubset { mask } => write!(f, "factor subset {mask:#x} is missing from exact enumeration"),
            Self::CandidatePlanMissing { plan_id } => write!(f, "candidate plan {plan_id} is absent from evaluator alternatives"),
            Self::ResourceOverrun { run_id, dimensions } => write!(f, "run {run_id} exceeds hard resource ceilings in {dimensions:?}"),
            Self::ObservationGapOverrun { run_id, observed_us, maximum_us } => write!(f, "run {run_id} observation gap {observed_us} exceeds {maximum_us}"),
            Self::ExposureExceedsEpisode { total, maximum } => write!(f, "exposure debt {total} exceeds episode duration {maximum}"),
            Self::InteractionReconstructionMismatch => write!(f, "factorial interaction dividends do not reconstruct the full-joint contrast"),
            Self::ArithmeticOverflow => write!(f, "exact joint oracle arithmetic overflowed"),
        }
    }
}

impl std::error::Error for OracleError {}

/// Independently enumerate all 2^n assignments, validate paired context and
/// measured hard budgets, then report per-axis optima and the full non-dominated
/// frontier. There is no scalarized leaderboard score.
pub fn qualify_exact_joint_oracle(
    input: &ExactJointOracleInput,
) -> Result<ExactJointOracleReport, OracleError> {
    if input.alternatives.is_empty() {
        return Err(OracleError::EmptyAlternatives);
    }
    if input.factor_action_ids.len() > MAX_EXACT_FACTORS {
        return Err(OracleError::TooManyFactors {
            requested: input.factor_action_ids.len(),
            maximum: MAX_EXACT_FACTORS,
        });
    }
    if input.factor_action_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(OracleError::DuplicateOrUnsortedFactors);
    }

    let factor_index: BTreeMap<u64, usize> = input.factor_action_ids
        .iter()
        .enumerate()
        .map(|(index, action_id)| (*action_id, index))
        .collect();
    let expected_count = 1_usize
        .checked_shl(u32::try_from(input.factor_action_ids.len())
            .map_err(|_| OracleError::ArithmeticOverflow)?)
        .ok_or(OracleError::ArithmeticOverflow)?;

    let reference_context = &input.alternatives[0].context;
    let mut run_ids = BTreeSet::new();
    let mut plan_ids = BTreeSet::new();
    let mut action_kinds = BTreeMap::<u64, ActionKind>::new();
    let mut by_mask = BTreeMap::<u64, &EvaluatedAlternative>::new();

    for alternative in &input.alternatives {
        if &alternative.context != reference_context {
            return Err(OracleError::ContextMismatch);
        }
        if !run_ids.insert(alternative.run_id) {
            return Err(OracleError::DuplicateRunId { run_id: alternative.run_id });
        }
        if !plan_ids.insert(alternative.plan.plan_id) {
            return Err(OracleError::DuplicatePlanId { plan_id: alternative.plan.plan_id });
        }
        if alternative.plan.actions.windows(2)
            .any(|pair| pair[0].action_id >= pair[1].action_id)
        {
            return Err(OracleError::DuplicateOrUnsortedActions {
                plan_id: alternative.plan.plan_id,
            });
        }

        let mut mask = 0_u64;
        for action in &alternative.plan.actions {
            let Some(index) = factor_index.get(&action.action_id) else {
                return Err(OracleError::UnknownFactorAction { action_id: action.action_id });
            };
            if let Some(previous_kind) = action_kinds.get(&action.action_id) {
                if previous_kind != &action.kind {
                    return Err(OracleError::ActionKindChanged { action_id: action.action_id });
                }
            } else {
                action_kinds.insert(action.action_id, action.kind);
            }
            mask |= 1_u64 << *index;
        }

        let overruns = alternative.measured_cost.exceeds(alternative.context.budget.resources);
        if !overruns.is_empty() {
            return Err(OracleError::ResourceOverrun {
                run_id: alternative.run_id,
                dimensions: overruns,
            });
        }
        if alternative.observed_maximum_gap_us
            > alternative.context.budget.maximum_observation_gap_us
        {
            return Err(OracleError::ObservationGapOverrun {
                run_id: alternative.run_id,
                observed_us: alternative.observed_maximum_gap_us,
                maximum_us: alternative.context.budget.maximum_observation_gap_us,
            });
        }
        alternative.realized_debt.validate(alternative.context.total_exposure_us)?;

        if by_mask.insert(mask, alternative).is_some() {
            return Err(OracleError::DuplicateSubset { mask });
        }
    }

    if by_mask.len() != expected_count {
        for mask in 0..u64::try_from(expected_count).map_err(|_| OracleError::ArithmeticOverflow)? {
            if !by_mask.contains_key(&mask) {
                return Err(OracleError::MissingSubset { mask });
            }
        }
        return Err(OracleError::ArithmeticOverflow);
    }
    for mask in 0..u64::try_from(expected_count).map_err(|_| OracleError::ArithmeticOverflow)? {
        if !by_mask.contains_key(&mask) {
            return Err(OracleError::MissingSubset { mask });
        }
    }

    let candidate = by_mask.values()
        .find(|alternative| alternative.plan.plan_id == input.candidate_plan_id)
        .copied()
        .ok_or(OracleError::CandidatePlanMissing { plan_id: input.candidate_plan_id })?;

    let alternatives = by_mask.values().copied().collect::<Vec<_>>();
    let best_wrong = best_by_axis(&alternatives, 0);
    let best_abstention = best_by_axis(&alternatives, 1);
    let best_missing = best_by_axis(&alternatives, 2);
    let pareto_plan_ids = pareto_frontier(&alternatives);

    Ok(ExactJointOracleReport {
        evaluated_alternatives: alternatives.len(),
        candidate_plan_id: candidate.plan.plan_id,
        candidate_debt: candidate.realized_debt,
        best_plan_ids_by_axis: [
            best_wrong.plan.plan_id,
            best_abstention.plan.plan_id,
            best_missing.plan.plan_id,
        ],
        componentwise_regret: ExposureRegret {
            wrong_belief_us: candidate.realized_debt.wrong_belief_us
                .saturating_sub(best_wrong.realized_debt.wrong_belief_us),
            abstention_us: candidate.realized_debt.abstention_us
                .saturating_sub(best_abstention.realized_debt.abstention_us),
            missing_output_us: candidate.realized_debt.missing_output_us
                .saturating_sub(best_missing.realized_debt.missing_output_us),
        },
        pareto_plan_ids,
    })
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Decompose a fully enumerated joint-action experiment into exact
/// main-effect and higher-order interaction dividends.
///
/// The input is first passed through the existing exact-oracle qualification:
/// all 2^n subsets must be present exactly once, share an identical episode
/// commitment, stay within measured resource ceilings, and have valid outcome
/// debt. The decomposition then applies a subset-lattice Moebius transform to
/// measured outcomes. It never consumes candidate-predicted benefits/costs.
///
/// For exposures, the value function is baseline debt minus subset debt, so
/// positive dividends mean reduced temporal error debt. For resources, the
/// value function is subset measured cost minus baseline measured cost, so
/// positive dividends mean additional cost. All arithmetic is checked i128.
pub fn decompose_exact_joint_interactions(
    input: &ExactJointOracleInput,
) -> Result<FactorialInteractionReport, OracleError> {
    // Reuse the authoritative exact-oracle validation rather than weakening
    // or duplicating its context, subset, budget and outcome checks.
    qualify_exact_joint_oracle(input)?;

    let factor_count = input.factor_action_ids.len();
    let expected_count = 1_usize
        .checked_shl(u32::try_from(factor_count).map_err(|_| OracleError::ArithmeticOverflow)?)
        .ok_or(OracleError::ArithmeticOverflow)?;

    // Canonicalize outcomes by factor mask. Input row order must not affect the
    // reported interaction terms or their exact integer values.
    let mut by_subset = vec![None; expected_count];
    for alternative in &input.alternatives {
        let mut mask = 0_usize;
        for action in &alternative.plan.actions {
            let index = input
                .factor_action_ids
                .binary_search(&action.action_id)
                .map_err(|_| OracleError::UnknownFactorAction {
                    action_id: action.action_id,
                })?;
            let bit = 1_usize
                .checked_shl(u32::try_from(index).map_err(|_| OracleError::ArithmeticOverflow)?)
                .ok_or(OracleError::ArithmeticOverflow)?;
            mask |= bit;
        }
        if mask >= expected_count || by_subset[mask].replace(alternative).is_some() {
            return Err(OracleError::DuplicateSubset {
                mask: u64::try_from(mask).map_err(|_| OracleError::ArithmeticOverflow)?,
            });
        }
    }
    let by_subset = by_subset
        .into_iter()
        .enumerate()
        .map(|(mask, alternative)| {
            alternative.ok_or(OracleError::MissingSubset {
                mask: u64::try_from(mask).map_err(|_| OracleError::ArithmeticOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let baseline = by_subset[0];
    let full_mask = expected_count - 1;
    let full = by_subset[full_mask];
    let baseline_debt = baseline.realized_debt;
    let full_joint_debt = full.realized_debt;
    let baseline_debt_values = baseline_debt.values();
    let baseline_cost_values = baseline.measured_cost.values();
    let full_debt_values = full_joint_debt.values();
    let full_cost_values = full.measured_cost.values();

    // The first three dimensions are exposure benefit; the following twelve
    // are incremental measured resource costs in ResourceVector::values order.
    const AXIS_COUNT: usize = 15;
    let mut lattice = vec![[0_i128; AXIS_COUNT]; expected_count];
    for (mask, alternative) in by_subset.iter().enumerate() {
        let debt = alternative.realized_debt.values();
        let cost = alternative.measured_cost.values();
        for axis in 0..3 {
            lattice[mask][axis] =
                i128::from(baseline_debt_values[axis]) - i128::from(debt[axis]);
        }
        for axis in 0..12 {
            lattice[mask][3 + axis] =
                i128::from(cost[axis]) - i128::from(baseline_cost_values[axis]);
        }
    }

    // In-place subset Moebius transform:
    // dividend[S] = value[S] - sum(dividend[T] for T proper subset of S).
    for bit_index in 0..factor_count {
        let bit = 1_usize
            .checked_shl(u32::try_from(bit_index).map_err(|_| OracleError::ArithmeticOverflow)?)
            .ok_or(OracleError::ArithmeticOverflow)?;
        for mask in 0..expected_count {
            if mask & bit != 0 {
                let prior = lattice[mask ^ bit];
                for axis in 0..AXIS_COUNT {
                    lattice[mask][axis] = lattice[mask][axis]
                        .checked_sub(prior[axis])
                        .ok_or(OracleError::ArithmeticOverflow)?;
                }
            }
        }
    }

    let mut interaction_terms = Vec::with_capacity(expected_count.saturating_sub(1));
    let mut interaction_order_summaries = (1..=factor_count)
        .map(|order| InteractionOrderSummary {
            interaction_order: u32::try_from(order).unwrap_or(u32::MAX),
            term_count: 0,
            exposure_benefit_dividend_us: [0; 3],
            resource_cost_dividend: [0; 12],
        })
        .collect::<Vec<_>>();

    for mask in 1..expected_count {
        let order = mask.count_ones();
        let summary_index = usize::try_from(order - 1).map_err(|_| OracleError::ArithmeticOverflow)?;
        let summary = interaction_order_summaries
            .get_mut(summary_index)
            .ok_or(OracleError::ArithmeticOverflow)?;
        summary.term_count = summary
            .term_count
            .checked_add(1)
            .ok_or(OracleError::ArithmeticOverflow)?;

        let mut exposure = [0_i128; 3];
        let mut resources = [0_i128; 12];
        exposure.copy_from_slice(&lattice[mask][0..3]);
        resources.copy_from_slice(&lattice[mask][3..15]);

        for axis in 0..3 {
            summary.exposure_benefit_dividend_us[axis] = summary.exposure_benefit_dividend_us[axis]
                .checked_add(exposure[axis])
                .ok_or(OracleError::ArithmeticOverflow)?;
        }
        for axis in 0..12 {
            summary.resource_cost_dividend[axis] = summary.resource_cost_dividend[axis]
                .checked_add(resources[axis])
                .ok_or(OracleError::ArithmeticOverflow)?;
        }

        interaction_terms.push(FactorialInteractionTerm {
            subset_mask: u64::try_from(mask).map_err(|_| OracleError::ArithmeticOverflow)?,
            interaction_order: order,
            exposure_benefit_dividend_us: exposure,
            resource_cost_dividend: resources,
        });
    }

    let total_exposure_benefit_us = std::array::from_fn(|axis| {
        i128::from(baseline_debt_values[axis]) - i128::from(full_debt_values[axis])
    });
    let total_resource_cost_delta = std::array::from_fn(|axis| {
        i128::from(full_cost_values[axis]) - i128::from(baseline_cost_values[axis])
    });

    // Internal reconstruction check: all main effects and interaction
    // dividends together must exactly reconstruct the full-joint contrast.
    let mut reconstructed_exposure = [0_i128; 3];
    let mut reconstructed_resources = [0_i128; 12];
    for term in &interaction_terms {
        for axis in 0..3 {
            reconstructed_exposure[axis] = reconstructed_exposure[axis]
                .checked_add(term.exposure_benefit_dividend_us[axis])
                .ok_or(OracleError::ArithmeticOverflow)?;
        }
        for axis in 0..12 {
            reconstructed_resources[axis] = reconstructed_resources[axis]
                .checked_add(term.resource_cost_dividend[axis])
                .ok_or(OracleError::ArithmeticOverflow)?;
        }
    }
    if reconstructed_exposure != total_exposure_benefit_us
        || reconstructed_resources != total_resource_cost_delta
    {
        return Err(OracleError::InteractionReconstructionMismatch);
    }

    // Standard Shapley allocation derived from the exact Harsanyi dividends:
    // each coalition dividend is shared equally among its participating factors.
    // Use one common LCM denominator for all interaction orders so fractional
    // shares remain exact and can be reconstructed without floating point.
    let mut shapley_common_denominator = 1_u64;
    for order in 1..=factor_count {
        let order_u64 = u64::try_from(order).map_err(|_| OracleError::ArithmeticOverflow)?;
        let divisor = gcd_u64(shapley_common_denominator, order_u64);
        shapley_common_denominator = (shapley_common_denominator / divisor)
            .checked_mul(order_u64)
            .ok_or(OracleError::ArithmeticOverflow)?;
    }

    let mut shapley_lattice = vec![[0_i128; AXIS_COUNT]; factor_count];
    for term in &interaction_terms {
        let mask = usize::try_from(term.subset_mask).map_err(|_| OracleError::ArithmeticOverflow)?;
        let scale = shapley_common_denominator
            .checked_div(u64::from(term.interaction_order))
            .ok_or(OracleError::ArithmeticOverflow)?;
        let scale = i128::from(scale);
        for factor_index in 0..factor_count {
            let bit = 1_usize
                .checked_shl(u32::try_from(factor_index).map_err(|_| OracleError::ArithmeticOverflow)?)
                .ok_or(OracleError::ArithmeticOverflow)?;
            if mask & bit == 0 {
                continue;
            }
            for axis in 0..3 {
                let contribution = term.exposure_benefit_dividend_us[axis]
                    .checked_mul(scale)
                    .ok_or(OracleError::ArithmeticOverflow)?;
                shapley_lattice[factor_index][axis] = shapley_lattice[factor_index][axis]
                    .checked_add(contribution)
                    .ok_or(OracleError::ArithmeticOverflow)?;
            }
            for axis in 0..12 {
                let contribution = term.resource_cost_dividend[axis]
                    .checked_mul(scale)
                    .ok_or(OracleError::ArithmeticOverflow)?;
                shapley_lattice[factor_index][3 + axis] =
                    shapley_lattice[factor_index][3 + axis]
                        .checked_add(contribution)
                        .ok_or(OracleError::ArithmeticOverflow)?;
            }
        }
    }

    // Efficiency axiom check in exact rational arithmetic:
    // the Shapley numerators across factors must sum to the full effect
    // multiplied by the common denominator.
    let denominator_i128 = i128::from(shapley_common_denominator);
    for axis in 0..AXIS_COUNT {
        let allocated = shapley_lattice.iter().try_fold(0_i128, |sum, factor| {
            sum.checked_add(factor[axis]).ok_or(OracleError::ArithmeticOverflow)
        })?;
        let full_value = if axis < 3 {
            total_exposure_benefit_us[axis]
        } else {
            total_resource_cost_delta[axis - 3]
        };
        let expected = full_value
            .checked_mul(denominator_i128)
            .ok_or(OracleError::ArithmeticOverflow)?;
        if allocated != expected {
            return Err(OracleError::InteractionReconstructionMismatch);
        }
    }

    let factor_action_kinds: Vec<ActionKind> =
        full.plan.actions.iter().map(|action| action.kind).collect();
    let factor_shapley_attributions = (0..factor_count)
        .map(|factor_index| {
            let mut exposure = [0_i128; 3];
            let mut resources = [0_i128; 12];
            exposure.copy_from_slice(&shapley_lattice[factor_index][0..3]);
            resources.copy_from_slice(&shapley_lattice[factor_index][3..15]);
            FactorShapleyAttribution {
                action_id: input.factor_action_ids[factor_index],
                action_kind: factor_action_kinds[factor_index],
                exposure_benefit_scaled_numerator_us: exposure,
                resource_cost_scaled_numerator: resources,
            }
        })
        .collect();

    Ok(FactorialInteractionReport {
        factor_action_ids: input.factor_action_ids.clone(),
        factor_action_kinds,
        evaluated_subset_count: expected_count,
        baseline_plan_id: baseline.plan.plan_id,
        full_joint_plan_id: full.plan.plan_id,
        baseline_debt,
        full_joint_debt,
        total_exposure_benefit_us,
        total_resource_cost_delta,
        interaction_terms,
        interaction_order_summaries,
        shapley_common_denominator,
        factor_shapley_attributions,
    })
}

fn best_by_axis<'a>(
    alternatives: &[&'a EvaluatedAlternative],
    axis: usize,
) -> &'a EvaluatedAlternative {
    alternatives.iter().copied().min_by(|left, right| {
        left.realized_debt.values()[axis]
            .cmp(&right.realized_debt.values()[axis])
            .then_with(|| left.measured_cost.values().cmp(&right.measured_cost.values()))
            .then_with(|| left.plan.plan_id.cmp(&right.plan.plan_id))
    }).expect("validated exact enumeration is non-empty")
}

fn pareto_frontier(alternatives: &[&EvaluatedAlternative]) -> Vec<u64> {
    let mut ids = alternatives.iter().copied()
        .filter(|candidate| {
            !alternatives.iter().copied().any(|other| {
                other.plan.plan_id != candidate.plan.plan_id && dominates(other, candidate)
            })
        })
        .map(|alternative| alternative.plan.plan_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn dominates(left: &EvaluatedAlternative, right: &EvaluatedAlternative) -> bool {
    let left_debt = left.realized_debt.values();
    let right_debt = right.realized_debt.values();
    let left_cost = left.measured_cost.values();
    let right_cost = right.measured_cost.values();
    let debt_no_worse = left_debt.iter().zip(right_debt).all(|(a, b)| *a <= b);
    let cost_no_worse = left_cost.iter().zip(right_cost).all(|(a, b)| *a <= b);
    let gap_no_worse = left.observed_maximum_gap_us <= right.observed_maximum_gap_us;
    if !debt_no_worse || !cost_no_worse || !gap_no_worse {
        return false;
    }
    let strictly_better = left_debt.iter().zip(right_debt).any(|(a, b)| *a < b)
        || left_cost.iter().zip(right_cost).any(|(a, b)| *a < b)
        || left.observed_maximum_gap_us < right.observed_maximum_gap_us;
    strictly_better || left.plan.plan_id < right.plan.plan_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_temporal_joint_control::{
        ActionKind, JointAction, PredictedExposureReduction,
    };

    fn context() -> EpisodeFingerprint {
        EpisodeFingerprint {
            episode_id: 77,
            task_sha256: [1; 32],
            future_stream_sha256: [2; 32],
            model_sha256: [3; 32],
            initial_state_sha256: [4; 32],
            candidate_feature_sha256: [5; 32],
            evaluator_profile_sha256: [6; 32],
            total_exposure_us: 1_000,
            budget: JointBudget {
                resources: ResourceVector {
                    persistent_bytes: 128,
                    retained_byte_us: 10_000,
                    read_bytes: 10_000,
                    write_bytes: 10_000,
                    compute_us: 10_000,
                    refresh_ops: 100,
                    observations: 100,
                    sensor_active_us: 10_000,
                    latency_us: 10_000,
                    energy_nj: 100_000,
                    verification_ops: 100,
                    receipt_bytes: 10_000,
                },
                maximum_observation_gap_us: 1_000,
            },
        }
    }

    fn kind(action_id: u64) -> ActionKind {
        match action_id {
            10 => ActionKind::Observe,
            20 => ActionKind::Retain,
            30 => ActionKind::Compute,
            40 => ActionKind::Verify,
            _ => ActionKind::Defer,
        }
    }

    fn full_enumeration() -> Vec<EvaluatedAlternative> {
        let ids = [10_u64, 20, 30, 40];
        (0_u64..16).map(|mask| {
            let actions = ids.iter().enumerate()
                .filter(|(index, _)| (mask & (1_u64 << *index)) != 0)
                .map(|(_, action_id)| JointAction {
                    action_id: *action_id,
                    kind: kind(*action_id),
                })
                .collect::<Vec<_>>();
            let count = mask.count_ones() as u64;
            let (wrong, abstention, missing) = if mask == 15 {
                // The four-way interaction is deliberately non-additive:
                // no single subsystem reproduces the combined benefit.
                (20, 15, 12)
            } else {
                (100 - count * 2, 80 - count, 70 - if mask & 1 != 0 { 5 } else { 0 })
            };
            let measured = ResourceVector {
                persistent_bytes: count * 8,
                retained_byte_us: count * 100,
                read_bytes: count * 32,
                write_bytes: count * 16,
                compute_us: count * 20,
                refresh_ops: count,
                observations: if mask & 1 != 0 { 1 } else { 0 },
                sensor_active_us: (if mask & 1 != 0 { 1 } else { 0 }) * 10,
                latency_us: count * 5,
                energy_nj: count * 100,
                verification_ops: if mask & 8 != 0 { 1 } else { 0 },
                receipt_bytes: (if mask & 8 != 0 { 1 } else { 0 }) * 128,
            };
            EvaluatedAlternative {
                run_id: mask + 1,
                context: context(),
                plan: JointPlan {
                    plan_id: mask + 100,
                    actions,
                    predicted_reduction: PredictedExposureReduction::default(),
                    estimated_cost: measured,
                    predicted_maximum_observation_gap_us: 1_000 - (mask & 1) * 10,
                },
                measured_cost: measured,
                observed_maximum_gap_us: 1_000 - (mask & 1) * 10,
                realized_debt: ExposureDebt {
                    wrong_belief_us: wrong,
                    abstention_us: abstention,
                    missing_output_us: missing,
                },
            }
        }).collect()
    }

    fn input() -> ExactJointOracleInput {
        ExactJointOracleInput {
            factor_action_ids: vec![10, 20, 30, 40],
            candidate_plan_id: 115,
            alternatives: full_enumeration(),
        }
    }

    /// Independent small-fixture reference using direct inclusion-exclusion
    /// over all sub-subsets rather than the production in-place transform.
    fn direct_inclusion_exclusion_reference(
        input: &ExactJointOracleInput,
    ) -> Vec<(u64, [i128; 3], [i128; 12])> {
        let expected_count = 1_usize << input.factor_action_ids.len();
        let mut by_mask = vec![None; expected_count];
        for alternative in &input.alternatives {
            let mut mask = 0_usize;
            for action in &alternative.plan.actions {
                let index = input.factor_action_ids.binary_search(&action.action_id).unwrap();
                mask |= 1_usize << index;
            }
            by_mask[mask] = Some(alternative);
        }
        let by_mask = by_mask
            .into_iter()
            .map(|alternative| alternative.unwrap())
            .collect::<Vec<_>>();
        let baseline_debt = by_mask[0].realized_debt.values();
        let baseline_cost = by_mask[0].measured_cost.values();
        let mut result = Vec::with_capacity(expected_count.saturating_sub(1));

        for mask in 1..expected_count {
            let mut exposure = [0_i128; 3];
            let mut resources = [0_i128; 12];
            for subset in 0..expected_count {
                if subset & !mask != 0 {
                    continue;
                }
                let distance = mask.count_ones() - subset.count_ones();
                let sign = if distance % 2 == 0 { 1_i128 } else { -1_i128 };
                let debt = by_mask[subset].realized_debt.values();
                let cost = by_mask[subset].measured_cost.values();
                for axis in 0..3 {
                    exposure[axis] += sign
                        * (i128::from(baseline_debt[axis]) - i128::from(debt[axis]));
                }
                for axis in 0..12 {
                    resources[axis] += sign
                        * (i128::from(cost[axis]) - i128::from(baseline_cost[axis]));
                }
            }
            result.push((
                u64::try_from(mask).unwrap(),
                exposure,
                resources,
            ));
        }
        result
    }

    fn three_factor_interaction_input() -> ExactJointOracleInput {
        let ids = [10_u64, 20, 30];
        let alternatives = (0_u64..8)
            .map(|mask| {
                let a = if mask & 1 != 0 { 1_u64 } else { 0_u64 };
                let b = if mask & 2 != 0 { 1_u64 } else { 0_u64 };
                let c = if mask & 4 != 0 { 1_u64 } else { 0_u64 };
                let ab = a * b;
                let abc = a * b * c;

                let actions = ids
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| (mask & (1_u64 << *index)) != 0)
                    .map(|(_, action_id)| JointAction {
                        action_id: *action_id,
                        kind: kind(*action_id),
                    })
                    .collect::<Vec<_>>();

                // Known ground-truth response surface:
                // main effects: A=10, B=20, C=30;
                // pairwise synergy: AB=+7;
                // three-way antagonism: ABC=-5.
                let wrong_benefit = 10 * a + 20 * b + 30 * c + 7 * ab - 5 * abc;
                let abstention_benefit = 2 * a + 3 * b + 4 * c;
                let missing_benefit = a + 2 * b + 3 * c;

                // Resource cost has a distinct AB cost interaction (+11)
                // and a three-way resource-saving interaction (-3).
                let measured_cost = ResourceVector {
                    persistent_bytes: 4 + 2 * a + 3 * b + 4 * c + 5 * ab - 2 * abc,
                    energy_nj: 100 + 10 * a + 20 * b + 30 * c + 11 * ab - 3 * abc,
                    ..ResourceVector::default()
                };
                EvaluatedAlternative {
                    run_id: mask + 1,
                    context: context(),
                    plan: JointPlan {
                        plan_id: mask + 500,
                        actions,
                        predicted_reduction: PredictedExposureReduction::default(),
                        estimated_cost: measured_cost,
                        predicted_maximum_observation_gap_us: 1_000,
                    },
                    measured_cost,
                    observed_maximum_gap_us: 1_000,
                    realized_debt: ExposureDebt {
                        wrong_belief_us: 500 - wrong_benefit,
                        abstention_us: 300 - abstention_benefit,
                        missing_output_us: 100 - missing_benefit,
                    },
                }
            })
            .collect();

        ExactJointOracleInput {
            factor_action_ids: ids.to_vec(),
            candidate_plan_id: 507,
            alternatives,
        }
    }

    #[test]
    fn factorial_transform_matches_independent_inclusion_exclusion_reference() {
        for fixture_case in [three_factor_interaction_input(), input()] {
            let report = decompose_exact_joint_interactions(&fixture_case).unwrap();
            let reference = direct_inclusion_exclusion_reference(&fixture_case);
            assert_eq!(report.interaction_terms.len(), reference.len());

            for (term, (mask, exposure, resources)) in
                report.interaction_terms.iter().zip(reference.iter())
            {
                assert_eq!(term.subset_mask, *mask);
                assert_eq!(term.exposure_benefit_dividend_us, *exposure);
                assert_eq!(term.resource_cost_dividend, *resources);
            }
        }
    }

    #[test]
    fn factorial_decomposition_recovers_pairwise_synergy_and_three_way_antagonism() {
        let report = decompose_exact_joint_interactions(&three_factor_interaction_input()).unwrap();

        let term = |mask: u64| {
            report
                .interaction_terms
                .iter()
                .find(|item| item.subset_mask == mask)
                .unwrap()
        };

        // First-order effects are measured independently.
        assert_eq!(term(1).exposure_benefit_dividend_us[0], 10);
        assert_eq!(term(2).exposure_benefit_dividend_us[0], 20);
        assert_eq!(term(4).exposure_benefit_dividend_us[0], 30);

        // AB works better together than the sum of its independent effects.
        assert_eq!(term(3).interaction_order, 2);
        assert_eq!(term(3).exposure_benefit_dividend_us[0], 7);

        // Adding C changes the AB synergy: the three-way term is antagonistic.
        assert_eq!(term(7).interaction_order, 3);
        assert_eq!(term(7).exposure_benefit_dividend_us[0], -5);

        // The same decomposition identifies resource cost and resource savings.
        assert_eq!(term(3).resource_cost_dividend[9], 11);
        assert_eq!(term(7).resource_cost_dividend[9], -3);

        assert_eq!(report.total_exposure_benefit_us, [62, 9, 6]);
        assert_eq!(report.total_resource_cost_delta[9], 68);
        assert_eq!(report.factor_action_kinds, vec![
            ActionKind::Observe,
            ActionKind::Retain,
            ActionKind::Compute,
        ]);
    }

    #[test]
    fn four_way_interaction_is_separated_from_all_lower_order_terms() {
        let report = decompose_exact_joint_interactions(&input()).unwrap();
        let fourth_order = report
            .interaction_terms
            .iter()
            .find(|term| term.subset_mask == 0b1111)
            .unwrap();

        assert_eq!(fourth_order.interaction_order, 4);
        // The 4-way fixture is additive for every proper subset. Only the
        // complete sensing+memory+compute+verification assignment contributes
        // these additional measured exposure reductions.
        assert_eq!(fourth_order.exposure_benefit_dividend_us, [72, 61, 53]);
        assert!(report.interaction_terms
            .iter()
            .filter(|term| term.subset_mask != 0b1111)
            .all(|term| term.interaction_order < 4));
        assert_eq!(report.interaction_order_summaries[3].term_count, 1);
    }

    #[test]
    fn exact_shapley_attributions_reconstruct_joint_effect_with_rational_shares() {
        let report = decompose_exact_joint_interactions(&three_factor_interaction_input()).unwrap();
        assert_eq!(report.shapley_common_denominator, 6);
        assert_eq!(report.factor_shapley_attributions.len(), 3);

        let first = &report.factor_shapley_attributions[0];
        let second = &report.factor_shapley_attributions[1];
        let third = &report.factor_shapley_attributions[2];

        // Shapley shares split each coalition dividend equally among members,
        // then retain the exact rational values using denominator 6.
        assert_eq!(first.action_id, 10);
        assert_eq!(second.action_id, 20);
        assert_eq!(third.action_id, 30);
        assert_eq!(first.exposure_benefit_scaled_numerator_us[0], 71);
        assert_eq!(second.exposure_benefit_scaled_numerator_us[0], 131);
        assert_eq!(third.exposure_benefit_scaled_numerator_us[0], 170);

        assert_eq!(first.resource_cost_scaled_numerator[9], 87);
        assert_eq!(second.resource_cost_scaled_numerator[9], 147);
        assert_eq!(third.resource_cost_scaled_numerator[9], 174);

        let total_wrong_belief_numerator =
            first.exposure_benefit_scaled_numerator_us[0]
                + second.exposure_benefit_scaled_numerator_us[0]
                + third.exposure_benefit_scaled_numerator_us[0];
        assert_eq!(
            total_wrong_belief_numerator,
            report.total_exposure_benefit_us[0] * i128::from(report.shapley_common_denominator)
        );

        let total_energy_numerator = first.resource_cost_scaled_numerator[9]
            + second.resource_cost_scaled_numerator[9]
            + third.resource_cost_scaled_numerator[9];
        assert_eq!(
            total_energy_numerator,
            report.total_resource_cost_delta[9]
                * i128::from(report.shapley_common_denominator)
        );
    }

    #[test]
    fn four_factor_shapley_uses_lcm_denominator_and_splits_four_way_term_equally() {
        let report = decompose_exact_joint_interactions(&input()).unwrap();
        assert_eq!(report.shapley_common_denominator, 12);
        assert_eq!(report.factor_shapley_attributions.len(), 4);

        // The fixture has additive main effects and one pure four-way term.
        // Shapley therefore gives each factor its main effect plus exactly
        // one quarter of the four-way dividend.
        let expected_wrong = [240_i128, 240, 240, 240];
        let expected_abstention = [195_i128, 195, 195, 195];
        let expected_missing = [219_i128, 159, 159, 159];

        for index in 0..4 {
            assert_eq!(
                report.factor_shapley_attributions[index]
                    .exposure_benefit_scaled_numerator_us[0],
                expected_wrong[index]
            );
            assert_eq!(
                report.factor_shapley_attributions[index]
                    .exposure_benefit_scaled_numerator_us[1],
                expected_abstention[index]
            );
            assert_eq!(
                report.factor_shapley_attributions[index]
                    .exposure_benefit_scaled_numerator_us[2],
                expected_missing[index]
            );
        }

        for axis in 0..3 {
            let allocated: i128 = report
                .factor_shapley_attributions
                .iter()
                .map(|attribution| attribution.exposure_benefit_scaled_numerator_us[axis])
                .sum();
            assert_eq!(
                allocated,
                report.total_exposure_benefit_us[axis]
                    * i128::from(report.shapley_common_denominator)
            );
        }
    }

    #[test]
    fn interaction_order_summaries_reconstruct_total_joint_effect() {
        let report = decompose_exact_joint_interactions(&three_factor_interaction_input()).unwrap();
        assert_eq!(report.interaction_order_summaries.len(), 3);

        assert_eq!(report.interaction_order_summaries[0].term_count, 3);
        assert_eq!(
            report.interaction_order_summaries[0].exposure_benefit_dividend_us[0],
            60
        );
        assert_eq!(report.interaction_order_summaries[1].term_count, 3);
        assert_eq!(
            report.interaction_order_summaries[1].exposure_benefit_dividend_us[0],
            7
        );
        assert_eq!(report.interaction_order_summaries[2].term_count, 1);
        assert_eq!(
            report.interaction_order_summaries[2].exposure_benefit_dividend_us[0],
            -5
        );

        let reconstructed: i128 = report
            .interaction_order_summaries
            .iter()
            .map(|summary| summary.exposure_benefit_dividend_us[0])
            .sum();
        assert_eq!(reconstructed, report.total_exposure_benefit_us[0]);
    }

    #[test]
    fn factorial_interaction_output_is_independent_of_input_row_order() {
        let input = three_factor_interaction_input();
        let forward = decompose_exact_joint_interactions(&input).unwrap();
        let mut reversed = input;
        reversed.alternatives.reverse();
        let backward = decompose_exact_joint_interactions(&reversed).unwrap();
        assert_eq!(forward, backward);
    }

    #[test]
    fn factorial_interaction_rejects_incomplete_factorial_design() {
        let mut input = three_factor_interaction_input();
        input.alternatives.pop();
        assert!(matches!(
            decompose_exact_joint_interactions(&input),
            Err(OracleError::MissingSubset { .. })
        ));
    }

    #[test]
    fn exact_oracle_captures_four_way_sensing_memory_compute_verification_interaction() {
        let result = qualify_exact_joint_oracle(&input()).unwrap();
        assert_eq!(result.evaluated_alternatives, 16);
        assert_eq!(result.best_plan_ids_by_axis, [115, 115, 115]);
        assert_eq!(result.componentwise_regret, ExposureRegret::default());
        assert!(result.pareto_plan_ids.contains(&115));
    }

    #[test]
    fn incomplete_joint_enumeration_fails_closed() {
        let mut value = input();
        value.alternatives.pop();
        assert!(matches!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::MissingSubset { .. })
        ));
    }

    #[test]
    fn paired_context_mismatch_is_rejected() {
        let mut value = input();
        value.alternatives[3].context.future_stream_sha256 = [99; 32];
        assert_eq!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::ContextMismatch)
        );
    }

    #[test]
    fn measured_resource_overrun_is_rejected_even_when_prediction_is_feasible() {
        let mut value = input();
        value.alternatives[0].measured_cost.energy_nj = u64::MAX;
        assert!(matches!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::ResourceOverrun { run_id: 1, .. })
        ));
    }

    #[test]
    fn exposure_components_cannot_sum_beyond_episode_duration() {
        let mut value = input();
        value.alternatives[0].realized_debt = ExposureDebt {
            wrong_belief_us: 700,
            abstention_us: 200,
            missing_output_us: 101,
        };
        assert_eq!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::ExposureExceedsEpisode { total: 1_001, maximum: 1_000 })
        );
    }

    #[test]
    fn duplicate_plan_and_run_ids_are_rejected() {
        let mut duplicate_plan = input();
        duplicate_plan.alternatives[1].plan.plan_id = duplicate_plan.alternatives[0].plan.plan_id;
        assert!(matches!(
            qualify_exact_joint_oracle(&duplicate_plan),
            Err(OracleError::DuplicatePlanId { .. })
        ));

        let mut duplicate_run = input();
        duplicate_run.alternatives[1].run_id = duplicate_run.alternatives[0].run_id;
        assert!(matches!(
            qualify_exact_joint_oracle(&duplicate_run),
            Err(OracleError::DuplicateRunId { .. })
        ));
    }

    #[test]
    fn unknown_factor_action_is_rejected() {
        let mut value = input();
        value.alternatives[0].plan.actions.push(JointAction {
            action_id: 99,
            kind: ActionKind::Defer,
        });
        assert!(matches!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::UnknownFactorAction { action_id: 99 })
        ));
    }

    #[test]
    fn missing_candidate_assignment_is_rejected() {
        let mut value = input();
        value.candidate_plan_id = 999;
        assert_eq!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::CandidatePlanMissing { plan_id: 999 })
        );
    }

    #[test]
    fn exact_enumeration_has_a_hard_complexity_ceiling() {
        let mut value = input();
        value.factor_action_ids = (1..=17).collect();
        assert_eq!(
            qualify_exact_joint_oracle(&value),
            Err(OracleError::TooManyFactors { requested: 17, maximum: 16 })
        );
    }
}
