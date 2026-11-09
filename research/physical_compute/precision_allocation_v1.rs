//! Deterministic certificate-sensitive precision allocation reference.
//!
//! This module consumes externally justified upper bounds; it does not infer
//! physical noise laws, Krylov convergence bounds, or hardware costs. Missing
//! sensitivity evidence fails closed. Synthetic assumptions may be explored,
//! but any plan using them is explicitly exploratory and cannot be treated as
//! a qualified execution plan.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundKind {
    /// A deterministic analytical upper bound with a separately reviewable derivation.
    AnalyticDeterministic,
    /// An assumed/synthetic upper bound used only for exploratory experiments.
    SyntheticModelled,
}

#[derive(Debug, Clone)]
pub struct SensitivityBound {
    pub upper: f64,
    pub kind: BoundKind,
    pub evidence_digest: String,
}

#[derive(Debug, Clone)]
pub struct PrecisionOption {
    pub reads: u32,
    pub error_upper: f64,
    /// Conservative complete local cost: reads, settling, conversion and local overhead.
    pub cost_upper: f64,
    pub kind: BoundKind,
    pub evidence_digest: String,
}

#[derive(Debug, Clone)]
pub struct OperationBudget {
    pub operation_id: String,
    pub sensitivity: Option<SensitivityBound>,
    /// Options are ordered by nondecreasing reads and cost, and nonincreasing error.
    pub options: Vec<PrecisionOption>,
}

#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub task_error_budget: f64,
    /// Hard cap on exhaustive Cartesian enumeration; no heuristic result on overflow.
    pub max_combinations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAbstention {
    MissingSensitivityEvidence,
    NoFeasibleAllocation,
    SearchBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanQualification {
    AnalyticBoundOnly,
    ExploratoryModelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedPrecision {
    pub operation_id: String,
    pub reads: u32,
    pub error_upper: f64,
    pub propagated_error_upper: f64,
    pub cost_upper: f64,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllocationPlan {
    pub selections: Vec<SelectedPrecision>,
    pub total_propagated_error_upper: f64,
    pub total_cost_upper: f64,
    pub total_reads: u64,
    pub qualification: PlanQualification,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AllocationOutcome {
    Plan(AllocationPlan),
    Abstain(AllocationAbstention),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    InvalidTaskErrorBudget,
    InvalidSearchBudget,
    EmptyOperationSet,
    EmptyOperationId,
    DuplicateOperationId,
    EmptyOptionSet,
    EmptyEvidenceDigest,
    NonFiniteOrNegativeBound,
    InvalidOptionOrdering,
    CombinationCountOverflow,
    NonFiniteAggregate,
}

#[derive(Debug, Clone)]
struct ValidatedOperation<'a> {
    operation: &'a OperationBudget,
    sensitivity: &'a SensitivityBound,
}

#[derive(Debug, Clone)]
struct BestCandidate {
    option_indices: Vec<usize>,
    propagated_error: f64,
    cost: f64,
    reads: u64,
    qualification: PlanQualification,
}

/// Minimize total option cost subject to an additive propagated-error bound:
///
/// sum_k sensitivity_upper[k] * output_error_upper[k] <= task_error_budget.
///
/// This is valid only when the caller independently justifies each sensitivity
/// bound and the additive decomposition for the selected algorithm. This
/// function does not derive those premises. All arithmetic is rounded upward
/// one representable float so accumulated bounds do not round down.
pub fn allocate_precision(
    operations: &[OperationBudget],
    request: &AllocationRequest,
) -> Result<AllocationOutcome, AllocationError> {
    if !request.task_error_budget.is_finite() || request.task_error_budget < 0.0 {
        return Err(AllocationError::InvalidTaskErrorBudget);
    }
    if request.max_combinations == 0 {
        return Err(AllocationError::InvalidSearchBudget);
    }
    if operations.is_empty() {
        return Err(AllocationError::EmptyOperationSet);
    }

    let mut seen_ids = std::collections::BTreeSet::new();
    let mut validated = Vec::with_capacity(operations.len());
    let mut combinations = 1_u64;
    for operation in operations {
        if operation.operation_id.trim().is_empty() {
            return Err(AllocationError::EmptyOperationId);
        }
        if !seen_ids.insert(operation.operation_id.as_str()) {
            return Err(AllocationError::DuplicateOperationId);
        }
        if operation.options.is_empty() {
            return Err(AllocationError::EmptyOptionSet);
        }

        let Some(sensitivity) = operation.sensitivity.as_ref() else {
            return Ok(AllocationOutcome::Abstain(
                AllocationAbstention::MissingSensitivityEvidence,
            ));
        };
        validate_bound(sensitivity.upper)?;
        validate_digest(&sensitivity.evidence_digest)?;

        let mut previous: Option<&PrecisionOption> = None;
        for option in &operation.options {
            validate_bound(option.error_upper)?;
            validate_bound(option.cost_upper)?;
            validate_digest(&option.evidence_digest)?;
            if let Some(prior) = previous {
                if option.reads <= prior.reads
                    || option.cost_upper < prior.cost_upper
                    || option.error_upper > prior.error_upper
                {
                    return Err(AllocationError::InvalidOptionOrdering);
                }
            }
            previous = Some(option);
        }

        combinations = combinations
            .checked_mul(operation.options.len() as u64)
            .ok_or(AllocationError::CombinationCountOverflow)?;
        validated.push(ValidatedOperation {
            operation,
            sensitivity,
        });
    }

    if combinations > request.max_combinations {
        return Ok(AllocationOutcome::Abstain(
            AllocationAbstention::SearchBudgetExceeded,
        ));
    }

    let mut best: Option<BestCandidate> = None;
    let mut option_indices = Vec::with_capacity(validated.len());
    search_combinations(
        &validated,
        0,
        request.task_error_budget,
        0.0,
        0.0,
        0,
        PlanQualification::AnalyticBoundOnly,
        &mut option_indices,
        &mut best,
    )?;

    let Some(best) = best else {
        return Ok(AllocationOutcome::Abstain(
            AllocationAbstention::NoFeasibleAllocation,
        ));
    };

    let mut selections = Vec::with_capacity(validated.len());
    for (operation, option_index) in validated.iter().zip(best.option_indices.iter()) {
        let option = &operation.operation.options[*option_index];
        let propagated_error_upper =
            upward_product(operation.sensitivity.upper, option.error_upper)?;
        selections.push(SelectedPrecision {
            operation_id: operation.operation.operation_id.clone(),
            reads: option.reads,
            error_upper: option.error_upper,
            propagated_error_upper,
            cost_upper: option.cost_upper,
            evidence_digest: option.evidence_digest.clone(),
        });
    }

    Ok(AllocationOutcome::Plan(AllocationPlan {
        selections,
        total_propagated_error_upper: best.propagated_error,
        total_cost_upper: best.cost,
        total_reads: best.reads,
        qualification: best.qualification,
    }))
}

fn validate_bound(value: f64) -> Result<(), AllocationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(AllocationError::NonFiniteOrNegativeBound);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), AllocationError> {
    if value.trim().is_empty() {
        return Err(AllocationError::EmptyEvidenceDigest);
    }
    Ok(())
}

fn next_up_nonnegative(value: f64) -> f64 {
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(value.to_bits() + 1)
}

fn upward_product(left: f64, right: f64) -> Result<f64, AllocationError> {
    if left == 0.0 || right == 0.0 {
        return Ok(0.0);
    }
    let product = left * right;
    if !product.is_finite() {
        return Err(AllocationError::NonFiniteAggregate);
    }
    let upper = next_up_nonnegative(product);
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(AllocationError::NonFiniteAggregate)
    }
}

fn upward_sum(left: f64, right: f64) -> Result<f64, AllocationError> {
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    let sum = left + right;
    if !sum.is_finite() {
        return Err(AllocationError::NonFiniteAggregate);
    }
    let upper = next_up_nonnegative(sum);
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(AllocationError::NonFiniteAggregate)
    }
}

#[allow(clippy::too_many_arguments)]
fn search_combinations(
    operations: &[ValidatedOperation<'_>],
    index: usize,
    task_error_budget: f64,
    current_error: f64,
    current_cost: f64,
    current_reads: u64,
    qualification: PlanQualification,
    option_indices: &mut Vec<usize>,
    best: &mut Option<BestCandidate>,
) -> Result<(), AllocationError> {
    if current_error > task_error_budget {
        return Ok(());
    }
    if let Some(candidate) = best.as_ref() {
        if current_cost > candidate.cost {
            return Ok(());
        }
    }
    if index == operations.len() {
        let replace = match best.as_ref() {
            None => true,
            Some(candidate) => {
                current_cost < candidate.cost
                    || (current_cost == candidate.cost && current_reads < candidate.reads)
                    || (current_cost == candidate.cost
                        && current_reads == candidate.reads
                        && option_indices.as_slice() < candidate.option_indices.as_slice())
            }
        };
        if replace {
            *best = Some(BestCandidate {
                option_indices: option_indices.clone(),
                propagated_error: current_error,
                cost: current_cost,
                reads: current_reads,
                qualification,
            });
        }
        return Ok(());
    }

    let operation = &operations[index];
    for (option_index, option) in operation.operation.options.iter().enumerate() {
        let contribution = upward_product(operation.sensitivity.upper, option.error_upper)?;
        let next_error = upward_sum(current_error, contribution)?;
        let next_cost = upward_sum(current_cost, option.cost_upper)?;
        let next_reads = current_reads
            .checked_add(option.reads as u64)
            .ok_or(AllocationError::NonFiniteAggregate)?;
        if next_error > task_error_budget {
            continue;
        }

        let next_qualification = if qualification == PlanQualification::ExploratoryModelled
            || option.kind == BoundKind::SyntheticModelled
            || operation.sensitivity.kind == BoundKind::SyntheticModelled
        {
            PlanQualification::ExploratoryModelled
        } else {
            PlanQualification::AnalyticBoundOnly
        };

        option_indices.push(option_index);
        search_combinations(
            operations,
            index + 1,
            task_error_budget,
            next_error,
            next_cost,
            next_reads,
            next_qualification,
            option_indices,
            best,
        )?;
        option_indices.pop();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensitivity(upper: f64) -> Option<SensitivityBound> {
        Some(SensitivityBound {
            upper,
            kind: BoundKind::AnalyticDeterministic,
            evidence_digest: "sensitivity-fixture-v1".to_string(),
        })
    }

    fn option(reads: u32, error_upper: f64, cost_upper: f64) -> PrecisionOption {
        PrecisionOption {
            reads,
            error_upper,
            cost_upper,
            kind: BoundKind::AnalyticDeterministic,
            evidence_digest: format!("precision-{reads}-{error_upper}"),
        }
    }

    fn request(task_error_budget: f64) -> AllocationRequest {
        AllocationRequest {
            task_error_budget,
            max_combinations: 10_000,
        }
    }

    #[test]
    fn chooses_globally_minimum_cost_feasible_combination() {
        let operations = vec![
            OperationBudget {
                operation_id: "op-a".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![option(4, 0.6, 1.0), option(8, 0.0, 5.0)],
            },
            OperationBudget {
                operation_id: "op-b".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![
                    option(4, 0.6, 0.5),
                    option(8, 0.3, 1.0),
                    option(16, 0.0, 1.5),
                ],
            },
        ];
        let outcome = allocate_precision(&operations, &request(0.61)).unwrap();
        let AllocationOutcome::Plan(plan) = outcome else {
            panic!("expected feasible plan");
        };
        assert!((plan.total_cost_upper - 2.5).abs() < 1.0e-12);
        assert!(plan.total_propagated_error_upper <= 0.61);
        assert!(plan.total_propagated_error_upper >= 0.6);
        assert_eq!(plan.total_reads, 20);
        assert_eq!(plan.selections[0].reads, 4);
        assert_eq!(plan.selections[1].reads, 16);
        assert_eq!(plan.qualification, PlanQualification::AnalyticBoundOnly);
    }

    #[test]
    fn missing_sensitivity_fails_closed() {
        let operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: None,
            options: vec![option(4, 0.1, 1.0)],
        }];
        assert_eq!(
            allocate_precision(&operations, &request(0.1)).unwrap(),
            AllocationOutcome::Abstain(AllocationAbstention::MissingSensitivityEvidence)
        );
    }

    #[test]
    fn synthetic_bounds_produce_exploratory_not_analytic_plan() {
        let mut operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.1, 1.0)],
        }];
        operations[0].options[0].kind = BoundKind::SyntheticModelled;
        let outcome = allocate_precision(&operations, &request(0.11)).unwrap();
        let AllocationOutcome::Plan(plan) = outcome else {
            panic!("expected exploratory plan");
        };
        assert_eq!(plan.qualification, PlanQualification::ExploratoryModelled);
    }

    #[test]
    fn impossible_error_budget_abstains() {
        let operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.5, 1.0), option(8, 0.2, 2.0)],
        }];
        assert_eq!(
            allocate_precision(&operations, &request(0.1)).unwrap(),
            AllocationOutcome::Abstain(AllocationAbstention::NoFeasibleAllocation)
        );
    }

    #[test]
    fn search_limit_abstains_instead_of_returning_heuristic_plan() {
        let operations = vec![
            OperationBudget {
                operation_id: "op-a".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![option(4, 0.5, 1.0), option(8, 0.1, 2.0)],
            },
            OperationBudget {
                operation_id: "op-b".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![option(4, 0.5, 1.0), option(8, 0.1, 2.0)],
            },
        ];
        let mut req = request(0.5);
        req.max_combinations = 3;
        assert_eq!(
            allocate_precision(&operations, &req).unwrap(),
            AllocationOutcome::Abstain(AllocationAbstention::SearchBudgetExceeded)
        );
    }

    #[test]
    fn rejects_non_monotone_precision_profile() {
        let operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.1, 1.0), option(8, 0.2, 2.0)],
        }];
        assert_eq!(
            allocate_precision(&operations, &request(0.5)),
            Err(AllocationError::InvalidOptionOrdering)
        );
    }

    #[test]
    fn rejects_duplicate_operation_ids() {
        let operations = vec![
            OperationBudget {
                operation_id: "same".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![option(4, 0.1, 1.0)],
            },
            OperationBudget {
                operation_id: "same".to_string(),
                sensitivity: sensitivity(1.0),
                options: vec![option(4, 0.1, 1.0)],
            },
        ];
        assert_eq!(
            allocate_precision(&operations, &request(0.5)),
            Err(AllocationError::DuplicateOperationId)
        );
    }

    #[test]
    fn rejects_empty_evidence_digest() {
        let mut operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.1, 1.0)],
        }];
        operations[0].options[0].evidence_digest.clear();
        assert_eq!(
            allocate_precision(&operations, &request(0.5)),
            Err(AllocationError::EmptyEvidenceDigest)
        );
    }


    #[test]
    fn zero_budget_accepts_only_zero_error_option() {
        let operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.0, 2.0), option(8, 0.0, 3.0)],
        }];
        let outcome = allocate_precision(&operations, &request(0.0)).unwrap();
        let AllocationOutcome::Plan(plan) = outcome else {
            panic!("expected zero-error plan");
        };
        assert_eq!(plan.selections[0].reads, 4);
        assert_eq!(plan.total_propagated_error_upper, 0.0);
    }

    #[test]
    fn upward_product_covers_positive_subnormal_underflow() {
        let smallest = f64::from_bits(1);
        let upper = upward_product(smallest, smallest).unwrap();
        assert_eq!(upper.to_bits(), 1);
        assert!(upper > 0.0);
    }

    #[test]
    fn zero_sensitivity_selects_cheapest_precision() {
        let operations = vec![OperationBudget {
            operation_id: "insensitive-op".to_string(),
            sensitivity: sensitivity(0.0),
            options: vec![option(4, 1.0, 1.0), option(8, 0.0, 2.0)],
        }];
        let outcome = allocate_precision(&operations, &request(0.0)).unwrap();
        let AllocationOutcome::Plan(plan) = outcome else {
            panic!("expected plan");
        };
        assert_eq!(plan.selections[0].reads, 4);
        assert_eq!(plan.total_propagated_error_upper, 0.0);
    }
    #[test]
    fn rejects_non_finite_task_budget() {
        let operations = vec![OperationBudget {
            operation_id: "op-a".to_string(),
            sensitivity: sensitivity(1.0),
            options: vec![option(4, 0.1, 1.0)],
        }];
        assert_eq!(
            allocate_precision(&operations, &request(f64::INFINITY)),
            Err(AllocationError::InvalidTaskErrorBudget)
        );
    }
}
