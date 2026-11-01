use crate::{ResourceVector, ShiftStatus};

const RESOURCE_DIMENSIONS: usize = 6;

/// Common identity and hard-budget context for a set of operation forecasts.
/// All candidate forecasts must be generated from this exact decision state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetedActionContext {
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub candidate_pool_digest: [u8; 32],
    /// Sorted, unique operation-family IDs considered admissible by the planner.
    pub admissible_operation_families: Vec<u32>,
    pub remaining_budget: ResourceVector,
    /// Minimum number of completed, independently verified examples supporting
    /// each candidate forecast. Insufficient candidates abstain.
    pub minimum_support: u64,
    pub model_digest: [u8; 32],
    pub policy_digest: [u8; 32],
}

/// One operation-family forecast, conditioned on the common decision state.
/// Quantiles are marginal by resource dimension; their conjunction is not a
/// simultaneous six-dimensional coverage guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateOperationForecast {
    pub operation_family: u32,
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub candidate_pool_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    /// Expected independently verifiable proof progress, in typed progress units.
    pub predicted_proof_progress: i128,
    pub remaining_q50: ResourceVector,
    pub remaining_q90: ResourceVector,
    pub remaining_q95: ResourceVector,
    pub mandatory_verification_reserve: ResourceVector,
    pub support_count: u64,
    pub shift_status: ShiftStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateDisposition {
    /// q90 plus mandatory verification reserve fits the remaining hard budget
    /// in every dimension. This is a budget gate, not a joint probability claim.
    Feasible {
        reserved_q90: ResourceVector,
    },
    /// At least one q90-plus-reserve resource component exceeds the hard budget.
    BudgetInfeasible,
    /// The shift detector does not support a forecast-based decision.
    ShiftAbstention,
    /// The forecast lacks the configured minimum completed-proof support.
    InsufficientSupport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateAssessment {
    pub operation_family: u32,
    pub disposition: CandidateDisposition,
    pub predicted_proof_progress: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetedParetoSelection {
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub candidate_pool_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    /// Every candidate, in canonical operation-family order.
    pub assessments: Vec<CandidateAssessment>,
    /// Feasible candidates not dominated in any resource/proof-progress objective.
    pub pareto_operation_families: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetedParetoError {
    MissingPreStateDigest,
    MissingFrontierDigest,
    MissingCandidatePoolDigest,
    MissingModelDigest,
    MissingPolicyDigest,
    EmptyCandidatePool,
    CandidatePoolNotCanonical,
    ZeroMinimumSupport,
    ForecastCountMismatch,
    DuplicateForecast,
    UnknownForecast,
    MissingForecast,
    PreStateMismatch,
    FrontierMismatch,
    CandidatePoolMismatch,
    ModelMismatch,
    PolicyMismatch,
    NonMonotoneQuantiles,
    ResourceArithmeticOverflow,
}

impl std::fmt::Display for BudgetedParetoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for BudgetedParetoError {}

impl BudgetedActionContext {
    pub fn validate(&self) -> Result<(), BudgetedParetoError> {
        if is_zero(self.pre_state_digest) {
            return Err(BudgetedParetoError::MissingPreStateDigest);
        }
        if is_zero(self.frontier_digest) {
            return Err(BudgetedParetoError::MissingFrontierDigest);
        }
        if is_zero(self.candidate_pool_digest) {
            return Err(BudgetedParetoError::MissingCandidatePoolDigest);
        }
        if is_zero(self.model_digest) {
            return Err(BudgetedParetoError::MissingModelDigest);
        }
        if is_zero(self.policy_digest) {
            return Err(BudgetedParetoError::MissingPolicyDigest);
        }
        if self.admissible_operation_families.is_empty() {
            return Err(BudgetedParetoError::EmptyCandidatePool);
        }
        if self
            .admissible_operation_families
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(BudgetedParetoError::CandidatePoolNotCanonical);
        }
        if self.minimum_support == 0 {
            return Err(BudgetedParetoError::ZeroMinimumSupport);
        }
        Ok(())
    }
}

/// Filter by hard budget and forecast validity, then return the complete
/// multi-objective Pareto set. This function never commits an operation.
///
/// It deliberately does not collapse resource dimensions into a scalar score.
/// A later explicit policy may choose among the retained Pareto alternatives;
/// the independent budget ledger and verifier remain authoritative.
pub fn select_budget_feasible_pareto(
    context: &BudgetedActionContext,
    forecasts: &[CandidateOperationForecast],
) -> Result<BudgetedParetoSelection, BudgetedParetoError> {
    context.validate()?;
    if forecasts.len() != context.admissible_operation_families.len() {
        return Err(BudgetedParetoError::ForecastCountMismatch);
    }

    let mut ordered = forecasts.to_vec();
    ordered.sort_by_key(|forecast| forecast.operation_family);
    for pair in ordered.windows(2) {
        if pair[0].operation_family == pair[1].operation_family {
            return Err(BudgetedParetoError::DuplicateForecast);
        }
    }

    for forecast in &ordered {
        if !context
            .admissible_operation_families
            .binary_search(&forecast.operation_family)
            .is_ok()
        {
            return Err(BudgetedParetoError::UnknownForecast);
        }
        if forecast.pre_state_digest != context.pre_state_digest {
            return Err(BudgetedParetoError::PreStateMismatch);
        }
        if forecast.frontier_digest != context.frontier_digest {
            return Err(BudgetedParetoError::FrontierMismatch);
        }
        if forecast.candidate_pool_digest != context.candidate_pool_digest {
            return Err(BudgetedParetoError::CandidatePoolMismatch);
        }
        if forecast.model_digest != context.model_digest {
            return Err(BudgetedParetoError::ModelMismatch);
        }
        if forecast.policy_digest != context.policy_digest {
            return Err(BudgetedParetoError::PolicyMismatch);
        }
        if !quantiles_are_monotone(*forecast) {
            return Err(BudgetedParetoError::NonMonotoneQuantiles);
        }
    }

    for (expected, actual) in context
        .admissible_operation_families
        .iter()
        .zip(ordered.iter())
    {
        if *expected != actual.operation_family {
            return Err(BudgetedParetoError::MissingForecast);
        }
    }

    let mut assessments = Vec::with_capacity(ordered.len());
    for forecast in &ordered {
        let disposition = if forecast.shift_status != ShiftStatus::InDomain {
            CandidateDisposition::ShiftAbstention
        } else if forecast.support_count < context.minimum_support {
            CandidateDisposition::InsufficientSupport
        } else {
            let reserved_q90 = forecast
                .remaining_q90
                .checked_add(forecast.mandatory_verification_reserve)
                .ok_or(BudgetedParetoError::ResourceArithmeticOverflow)?;
            if reserved_q90.fits_within(context.remaining_budget) {
                CandidateDisposition::Feasible { reserved_q90 }
            } else {
                CandidateDisposition::BudgetInfeasible
            }
        };
        assessments.push(CandidateAssessment {
            operation_family: forecast.operation_family,
            disposition,
            predicted_proof_progress: forecast.predicted_proof_progress,
        });
    }

    let feasible = assessments
        .iter()
        .filter(|assessment| matches!(assessment.disposition, CandidateDisposition::Feasible { .. }))
        .collect::<Vec<_>>();
    let mut pareto_operation_families = Vec::new();

    for candidate in &feasible {
        let CandidateDisposition::Feasible {
            reserved_q90: candidate_cost,
        } = candidate.disposition
        else {
            continue;
        };
        let dominated = feasible.iter().any(|other| {
            if other.operation_family == candidate.operation_family {
                return false;
            }
            let CandidateDisposition::Feasible {
                reserved_q90: other_cost,
            } = other.disposition
            else {
                return false;
            };
            dominates(
                other_cost,
                other.predicted_proof_progress,
                candidate_cost,
                candidate.predicted_proof_progress,
            )
        });
        if !dominated {
            pareto_operation_families.push(candidate.operation_family);
        }
    }

    Ok(BudgetedParetoSelection {
        pre_state_digest: context.pre_state_digest,
        frontier_digest: context.frontier_digest,
        candidate_pool_digest: context.candidate_pool_digest,
        model_digest: context.model_digest,
        policy_digest: context.policy_digest,
        assessments,
        pareto_operation_families,
    })
}

fn quantiles_are_monotone(forecast: CandidateOperationForecast) -> bool {
    let q50 = resource_values(forecast.remaining_q50);
    let q90 = resource_values(forecast.remaining_q90);
    let q95 = resource_values(forecast.remaining_q95);
    (0..RESOURCE_DIMENSIONS).all(|index| q50[index] <= q90[index] && q90[index] <= q95[index])
}

fn dominates(
    left_cost: ResourceVector,
    left_progress: i128,
    right_cost: ResourceVector,
    right_progress: i128,
) -> bool {
    let left = resource_values(left_cost);
    let right = resource_values(right_cost);
    let no_worse_on_cost = (0..RESOURCE_DIMENSIONS).all(|index| left[index] <= right[index]);
    let no_worse_on_progress = left_progress >= right_progress;
    let strictly_better = (0..RESOURCE_DIMENSIONS).any(|index| left[index] < right[index])
        || left_progress > right_progress;
    no_worse_on_cost && no_worse_on_progress && strictly_better
}

fn resource_values(vector: ResourceVector) -> [u128; RESOURCE_DIMENSIONS] {
    [
        vector.tokens,
        vector.tool_calls,
        vector.cpu_ms,
        vector.gpu_ms,
        vector.wall_ms,
        vector.micro_usd,
    ]
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> BudgetedActionContext {
        BudgetedActionContext {
            pre_state_digest: [1; 32],
            frontier_digest: [2; 32],
            candidate_pool_digest: [3; 32],
            admissible_operation_families: vec![2, 3, 4, 5],
            remaining_budget: ResourceVector::new(100, 10, 20, 10, 40, 50),
            minimum_support: 8,
            model_digest: [4; 32],
            policy_digest: [5; 32],
        }
    }

    fn forecast(
        family: u32,
        q50: ResourceVector,
        q90: ResourceVector,
        q95: ResourceVector,
        reserve: ResourceVector,
        progress: i128,
    ) -> CandidateOperationForecast {
        CandidateOperationForecast {
            operation_family: family,
            pre_state_digest: [1; 32],
            frontier_digest: [2; 32],
            candidate_pool_digest: [3; 32],
            model_digest: [4; 32],
            policy_digest: [5; 32],
            predicted_proof_progress: progress,
            remaining_q50: q50,
            remaining_q90: q90,
            remaining_q95: q95,
            mandatory_verification_reserve: reserve,
            support_count: 16,
            shift_status: ShiftStatus::InDomain,
        }
    }

    fn forecasts() -> Vec<CandidateOperationForecast> {
        vec![
            forecast(
                2,
                ResourceVector::new(20, 2, 3, 1, 5, 5),
                ResourceVector::new(30, 3, 4, 2, 8, 8),
                ResourceVector::new(40, 4, 5, 3, 10, 10),
                ResourceVector::new(5, 1, 1, 0, 1, 1),
                1,
            ),
            forecast(
                3,
                ResourceVector::new(24, 2, 3, 1, 5, 5),
                ResourceVector::new(35, 3, 4, 2, 8, 8),
                ResourceVector::new(45, 4, 5, 3, 10, 10),
                ResourceVector::new(5, 1, 1, 0, 1, 1),
                3,
            ),
            forecast(
                4,
                ResourceVector::new(10, 1, 2, 0, 4, 4),
                ResourceVector::new(110, 2, 3, 1, 5, 5),
                ResourceVector::new(120, 3, 4, 2, 7, 7),
                ResourceVector::ZERO,
                9,
            ),
            forecast(
                5,
                ResourceVector::new(5, 1, 1, 0, 2, 2),
                ResourceVector::new(7, 2, 2, 4, 3, 3),
                ResourceVector::new(9, 3, 3, 5, 4, 4),
                ResourceVector::ZERO,
                2,
            ),
        ]
    }

    #[test]
    fn selector_returns_budget_feasible_multiobjective_frontier() {
        let selection = select_budget_feasible_pareto(&context(), &forecasts()).unwrap();
        assert_eq!(selection.pareto_operation_families, vec![2, 3, 5]);
        assert!(matches!(
            selection.assessments[2].disposition,
            CandidateDisposition::BudgetInfeasible
        ));
        assert!(matches!(
            selection.assessments[0].disposition,
            CandidateDisposition::Feasible { .. }
        ));
        assert!(matches!(
            selection.assessments[1].disposition,
            CandidateDisposition::Feasible { .. }
        ));
        assert!(matches!(
            selection.assessments[3].disposition,
            CandidateDisposition::Feasible { .. }
        ));
    }

    #[test]
    fn shifted_and_under_supported_forecasts_abstain() {
        let mut data = forecasts();
        data[0].shift_status = ShiftStatus::SuspectedShift;
        data[1].support_count = 2;
        let selection = select_budget_feasible_pareto(&context(), &data).unwrap();
        assert_eq!(
            selection.assessments[0].disposition,
            CandidateDisposition::ShiftAbstention
        );
        assert_eq!(
            selection.assessments[1].disposition,
            CandidateDisposition::InsufficientSupport
        );
        assert_eq!(selection.pareto_operation_families, vec![5]);
    }

    #[test]
    fn forecasts_must_share_exact_context_and_cover_candidate_pool() {
        let mut data = forecasts();
        data[0].pre_state_digest = [99; 32];
        assert_eq!(
            select_budget_feasible_pareto(&context(), &data),
            Err(BudgetedParetoError::PreStateMismatch)
        );

        let mut data = forecasts();
        data.pop();
        assert_eq!(
            select_budget_feasible_pareto(&context(), &data),
            Err(BudgetedParetoError::ForecastCountMismatch)
        );

        let mut data = forecasts();
        data[0].candidate_pool_digest = [99; 32];
        assert_eq!(
            select_budget_feasible_pareto(&context(), &data),
            Err(BudgetedParetoError::CandidatePoolMismatch)
        );
    }

    #[test]
    fn duplicate_or_unknown_forecasts_are_rejected() {
        let mut duplicate = forecasts();
        duplicate[1].operation_family = 2;
        assert_eq!(
            select_budget_feasible_pareto(&context(), &duplicate),
            Err(BudgetedParetoError::DuplicateForecast)
        );

        let mut unknown = forecasts();
        unknown[3].operation_family = 9;
        assert_eq!(
            select_budget_feasible_pareto(&context(), &unknown),
            Err(BudgetedParetoError::UnknownForecast)
        );
    }

    #[test]
    fn quantile_order_and_all_six_budget_dimensions_are_enforced() {
        let mut nonmonotone = forecasts();
        nonmonotone[0].remaining_q90.tokens = 1;
        assert_eq!(
            select_budget_feasible_pareto(&context(), &nonmonotone),
            Err(BudgetedParetoError::NonMonotoneQuantiles)
        );

        let mut budget = context();
        budget.remaining_budget = ResourceVector::new(100, 10, 20, 0, 40, 50);
        let selection = select_budget_feasible_pareto(&budget, &forecasts()).unwrap();
        assert!(matches!(
            selection.assessments[0].disposition,
            CandidateDisposition::BudgetInfeasible
        ));
    }

    #[test]
    fn q90_and_verifier_reserve_are_checked_for_overflow() {
        let mut data = forecasts();
        data[0].remaining_q90.tokens = u128::MAX;
        data[0].remaining_q95.tokens = u128::MAX;
        data[0].mandatory_verification_reserve.tokens = 1;
        assert_eq!(
            select_budget_feasible_pareto(&context(), &data),
            Err(BudgetedParetoError::ResourceArithmeticOverflow)
        );
    }
}
