//! Fixed-budget allocation of pruning-audit sentinels.
//!
//! This planner chooses audit sample counts before target-fidelity audit outcomes exist.
//! It solves a bounded multiple-choice knapsack exactly for the stated design objective:
//! minimize the weighted sum of zero-miss finite-population upper bounds, subject to a
//! hard total audit-cost budget.
//!
//! This is a schedule-design primitive, not an online learner. Its zero-miss risk objective
//! is optimistic; the realized audit must still pass the observed-outcome gate in
//! pruning_audit::audit_pruning. Every audit cost is charged, and the planner may leave
//! budget unspent where an additional sample cannot improve the integer risk bound.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use super::pruning_audit::zero_miss_upper_bound_false_pruned_count;

/// Hard work limits prevent an accidentally unbounded planning request.
const MAX_BUDGET_UNITS: u64 = 50_000;
const MAX_TRANSITION_CELLS: u64 = 25_000_000;

/// One independent population that may be audited at a particular pruning stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditAllocationStage {
    pub stage_id: String,
    pub pruned_population_size: usize,
    /// Actual incremental cost of one sentinel evaluation at the target fidelity.
    pub cost_per_sentinel_units: u64,
    /// Pre-registered non-zero importance multiplier for this stage's missed-candidate bound.
    /// This is an engineering priority weight, not an estimated regret or probability.
    pub risk_weight: u64,
}

/// Planner constraints. The total budget applies only to sentinel evaluations; the caller
/// must separately account for promoted candidates, training, failed runs and fallbacks.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetedAuditConfig {
    pub total_budget_units: u64,
    pub confidence_level: f64,
}

/// A stage's precommitted sample allocation and optimistic (zero-miss) risk bound.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditStageAllocation {
    pub stage_id: String,
    pub pruned_population_size: usize,
    pub sample_count: usize,
    pub cost_per_sentinel_units: u64,
    pub allocated_cost_units: u64,
    pub zero_miss_upper_bound_false_pruned_count: usize,
    pub zero_miss_upper_bound_false_pruned_rate: f64,
    pub risk_weight: u64,
}

/// Exact result for the stated integer planning objective.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetedAuditPlan {
    pub total_budget_units: u64,
    pub allocated_cost_units: u64,
    pub unallocated_budget_units: u64,
    /// Sum of risk_weight * zero_miss_upper_bound_false_pruned_count.
    pub weighted_zero_miss_risk_bound: u128,
    pub estimated_transition_cells: u64,
    /// Sorted by stage_id, independent of input stage order.
    pub stages: Vec<AuditStageAllocation>,
}

/// Invalid inputs or a request that exceeds the planner's deterministic work limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedAuditError {
    pub field: String,
    pub message: String,
}

impl BudgetedAuditError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for BudgetedAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for BudgetedAuditError {}

#[derive(Debug, Clone)]
struct DpState {
    weighted_risk: u128,
    sample_counts: Vec<usize>,
}

/// Allocate a fixed sentinel budget using exact dynamic programming.
///
/// For each stage j and pre-outcome sample count n_j, let U_j(n_j) be the exact one-sided
/// upper bound on false-pruned population count under the optimistic assumption x=0. The
/// planner minimizes:
///
///     sum_j risk_weight_j * U_j(n_j)
///
/// subject to:
///
///     sum_j n_j * cost_per_sentinel_j <= total_budget.
///
/// This objective is transparent and reproducible. It is not expected regret. The weights
/// are preregistered engineering priorities, not values inferred from hidden outcomes.
/// Because the schedule depends only on public stage sizes/costs/weights and the fixed
/// confidence level, it can be frozen before any sentinel target-fidelity scores are read.
pub fn plan_budgeted_audits(
    stages: &[AuditAllocationStage],
    config: &BudgetedAuditConfig,
) -> Result<BudgetedAuditPlan, BudgetedAuditError> {
    if stages.is_empty() {
        return Err(BudgetedAuditError::new("stages", "at least one stage is required"));
    }
    if config.total_budget_units > MAX_BUDGET_UNITS {
        return Err(BudgetedAuditError::new(
            "total_budget_units",
            format!("must not exceed the deterministic planner limit of {MAX_BUDGET_UNITS}"),
        ));
    }
    if !config.confidence_level.is_finite()
        || config.confidence_level <= 0.0
        || config.confidence_level >= 1.0
    {
        return Err(BudgetedAuditError::new(
            "confidence_level",
            "must be finite and in (0, 1)",
        ));
    }

    let budget = usize::try_from(config.total_budget_units).map_err(|_| {
        BudgetedAuditError::new("total_budget_units", "does not fit this platform's usize")
    })?;

    let mut canonical_stages = stages.to_vec();
    canonical_stages.sort_by(|left, right| left.stage_id.cmp(&right.stage_id));

    let mut seen_ids = HashSet::with_capacity(canonical_stages.len());
    let mut estimated_transition_cells = 0u64;
    for (index, stage) in canonical_stages.iter().enumerate() {
        let field = format!("stages[{index}]");
        if stage.stage_id.trim().is_empty() {
            return Err(BudgetedAuditError::new(
                format!("{field}.stage_id"),
                "must be a non-empty string",
            ));
        }
        if !seen_ids.insert(stage.stage_id.as_str()) {
            return Err(BudgetedAuditError::new(
                format!("{field}.stage_id"),
                format!("duplicate stage ID {:?}", stage.stage_id),
            ));
        }
        if stage.pruned_population_size == 0 {
            return Err(BudgetedAuditError::new(
                format!("{field}.pruned_population_size"),
                "must be positive",
            ));
        }
        if stage.cost_per_sentinel_units == 0 {
            return Err(BudgetedAuditError::new(
                format!("{field}.cost_per_sentinel_units"),
                "must be positive",
            ));
        }
        if stage.risk_weight == 0 {
            return Err(BudgetedAuditError::new(
                format!("{field}.risk_weight"),
                "must be positive",
            ));
        }

        let max_samples = stage
            .pruned_population_size
            .min(budget / usize::try_from(stage.cost_per_sentinel_units).unwrap_or(usize::MAX));
        let option_count = u64::try_from(max_samples)
            .unwrap_or(u64::MAX)
            .checked_add(1)
            .ok_or_else(|| BudgetedAuditError::new("work_limit", "sample option count overflow"))?;
        let cells = u64::try_from(budget)
            .unwrap_or(u64::MAX)
            .checked_add(1)
            .and_then(|value| value.checked_mul(option_count))
            .ok_or_else(|| BudgetedAuditError::new("work_limit", "transition estimate overflow"))?;
        estimated_transition_cells = estimated_transition_cells
            .checked_add(cells)
            .ok_or_else(|| BudgetedAuditError::new("work_limit", "transition estimate overflow"))?;
    }
    if estimated_transition_cells > MAX_TRANSITION_CELLS {
        return Err(BudgetedAuditError::new(
            "work_limit",
            format!(
                "estimated {estimated_transition_cells} DP transitions exceeds the deterministic limit of {MAX_TRANSITION_CELLS}"
            ),
        ));
    }

    // One option per possible sample count is precomputed before any DP transitions.
    let mut stage_options: Vec<Vec<(u64, usize, u128)>> =
        Vec::with_capacity(canonical_stages.len());
    for stage in &canonical_stages {
        let max_samples = stage.pruned_population_size.min(
            budget / usize::try_from(stage.cost_per_sentinel_units).unwrap_or(usize::MAX),
        );
        let mut options = Vec::with_capacity(max_samples + 1);
        for sample_count in 0..=max_samples {
            let cost = u64::try_from(sample_count)
                .ok()
                .and_then(|count| count.checked_mul(stage.cost_per_sentinel_units))
                .ok_or_else(|| {
                    BudgetedAuditError::new("stage_cost", "u64 overflow for audit stage cost")
                })?;
            let upper_bound = zero_miss_upper_bound_false_pruned_count(
                stage.pruned_population_size,
                sample_count,
                config.confidence_level,
            )
            .map_err(|error| {
                BudgetedAuditError::new(
                    format!("{}.{}", stage.stage_id, error.field),
                    error.message,
                )
            })?;
            let weighted_risk = (stage.risk_weight as u128)
                .checked_mul(upper_bound as u128)
                .ok_or_else(|| {
                    BudgetedAuditError::new("weighted_risk", "u128 multiplication overflow")
                })?;
            options.push((cost, sample_count, weighted_risk));
        }
        stage_options.push(options);
    }

    let mut dp: Vec<Option<DpState>> = vec![None; budget + 1];
    dp[0] = Some(DpState {
        weighted_risk: 0,
        sample_counts: Vec::with_capacity(canonical_stages.len()),
    });

    for options in &stage_options {
        let mut next: Vec<Option<DpState>> = vec![None; budget + 1];
        for (used_cost, state) in dp.iter().enumerate() {
            let Some(state) = state else {
                continue;
            };
            let remaining = budget - used_cost;
            for (incremental_cost, sample_count, stage_risk) in options {
                let incremental_cost = usize::try_from(*incremental_cost).map_err(|_| {
                    BudgetedAuditError::new("stage_cost", "does not fit usize")
                })?;
                if incremental_cost > remaining {
                    continue;
                }
                let new_cost = used_cost + incremental_cost;
                let new_risk = state
                    .weighted_risk
                    .checked_add(*stage_risk)
                    .ok_or_else(|| {
                        BudgetedAuditError::new("weighted_risk", "u128 addition overflow")
                    })?;
                let mut counts = state.sample_counts.clone();
                counts.push(*sample_count);
                let replace = match &next[new_cost] {
                    None => true,
                    Some(existing) => {
                        new_risk < existing.weighted_risk
                            || (new_risk == existing.weighted_risk
                                && counts < existing.sample_counts)
                    }
                };
                if replace {
                    next[new_cost] = Some(DpState {
                        weighted_risk: new_risk,
                        sample_counts: counts,
                    });
                }
            }
        }
        dp = next;
    }

    let (allocated_cost, best_state) = dp
        .iter()
        .enumerate()
        .filter_map(|(cost, state)| state.as_ref().map(|state| (cost, state)))
        .min_by(|(cost_a, state_a), (cost_b, state_b)| {
            state_a
                .weighted_risk
                .cmp(&state_b.weighted_risk)
                .then_with(|| cost_a.cmp(cost_b))
                .then_with(|| state_a.sample_counts.cmp(&state_b.sample_counts))
        })
        .ok_or_else(|| BudgetedAuditError::new("plan", "no feasible allocation found"))?;

    let mut allocations = Vec::with_capacity(canonical_stages.len());
    for (stage, sample_count) in canonical_stages.iter().zip(&best_state.sample_counts) {
        let allocated_cost_units = u64::try_from(*sample_count)
            .ok()
            .and_then(|count| count.checked_mul(stage.cost_per_sentinel_units))
            .ok_or_else(|| BudgetedAuditError::new("stage_cost", "u64 overflow"))?;
        let upper_bound = zero_miss_upper_bound_false_pruned_count(
            stage.pruned_population_size,
            *sample_count,
            config.confidence_level,
        )
        .map_err(|error| {
            BudgetedAuditError::new(
                format!("{}.{}", stage.stage_id, error.field),
                error.message,
            )
        })?;
        allocations.push(AuditStageAllocation {
            stage_id: stage.stage_id.clone(),
            pruned_population_size: stage.pruned_population_size,
            sample_count: *sample_count,
            cost_per_sentinel_units: stage.cost_per_sentinel_units,
            allocated_cost_units,
            zero_miss_upper_bound_false_pruned_count: upper_bound,
            zero_miss_upper_bound_false_pruned_rate: upper_bound as f64
                / stage.pruned_population_size as f64,
            risk_weight: stage.risk_weight,
        });
    }

    let allocated_cost_units = u64::try_from(allocated_cost).map_err(|_| {
        BudgetedAuditError::new("allocated_cost_units", "does not fit u64")
    })?;
    Ok(BudgetedAuditPlan {
        total_budget_units: config.total_budget_units,
        allocated_cost_units,
        unallocated_budget_units: config.total_budget_units - allocated_cost_units,
        weighted_zero_miss_risk_bound: best_state.weighted_risk,
        estimated_transition_cells,
        stages: allocations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stages() -> Vec<AuditAllocationStage> {
        vec![
            AuditAllocationStage {
                stage_id: "early".into(),
                pruned_population_size: 192,
                cost_per_sentinel_units: 3,
                risk_weight: 1,
            },
            AuditAllocationStage {
                stage_id: "middle".into(),
                pruned_population_size: 48,
                cost_per_sentinel_units: 12,
                risk_weight: 1,
            },
            AuditAllocationStage {
                stage_id: "late".into(),
                pruned_population_size: 12,
                cost_per_sentinel_units: 48,
                risk_weight: 1,
            },
        ]
    }

    fn config(budget: u64) -> BudgetedAuditConfig {
        BudgetedAuditConfig {
            total_budget_units: budget,
            confidence_level: 0.95,
        }
    }

    fn allocation<'a>(plan: &'a BudgetedAuditPlan, id: &str) -> &'a AuditStageAllocation {
        plan.stages.iter().find(|stage| stage.stage_id == id).unwrap()
    }

    #[test]
    fn zero_budget_produces_no_audits_and_trivial_bounds() {
        let plan = plan_budgeted_audits(&stages(), &config(0)).unwrap();
        assert_eq!(plan.allocated_cost_units, 0);
        assert_eq!(plan.unallocated_budget_units, 0);
        assert!(plan.stages.iter().all(|stage| stage.sample_count == 0));
        assert_eq!(plan.weighted_zero_miss_risk_bound, 252);
    }

    #[test]
    fn fixed_budget_is_never_exceeded_and_all_costs_reconcile() {
        let plan = plan_budgeted_audits(&stages(), &config(166)).unwrap();
        assert!(plan.allocated_cost_units <= 166);
        assert_eq!(
            plan.allocated_cost_units,
            plan.stages.iter().map(|stage| stage.allocated_cost_units).sum()
        );
        assert_eq!(
            plan.unallocated_budget_units,
            plan.total_budget_units - plan.allocated_cost_units
        );
    }

    #[test]
    fn known_schedule_matches_independent_reference_fixture() {
        let plan = plan_budgeted_audits(&stages(), &config(1_083)).unwrap();
        assert_eq!(plan.allocated_cost_units, 1_059);
        assert_eq!(plan.weighted_zero_miss_risk_bound, 6);
        assert_eq!(allocation(&plan, "early").sample_count, 121);
        assert_eq!(allocation(&plan, "middle").sample_count, 30);
        assert_eq!(allocation(&plan, "late").sample_count, 7);
        assert_eq!(allocation(&plan, "early").zero_miss_upper_bound_false_pruned_count, 2);
        assert_eq!(allocation(&plan, "middle").zero_miss_upper_bound_false_pruned_count, 2);
        assert_eq!(allocation(&plan, "late").zero_miss_upper_bound_false_pruned_count, 2);
    }

    #[test]
    fn stage_input_order_does_not_change_the_plan() {
        let mut reversed = stages();
        reversed.reverse();
        assert_eq!(
            plan_budgeted_audits(&stages(), &config(500)).unwrap(),
            plan_budgeted_audits(&reversed, &config(500)).unwrap()
        );
    }

    #[test]
    fn stage_risk_weight_changes_where_budget_is_allocated() {
        let mut weighted = stages();
        weighted[2].risk_weight = 20;
        let equal = plan_budgeted_audits(&stages(), &config(300)).unwrap();
        let prioritized = plan_budgeted_audits(&weighted, &config(300)).unwrap();
        assert!(
            allocation(&prioritized, "late").sample_count
                >= allocation(&equal, "late").sample_count
        );
    }

    #[test]
    fn a_census_removes_the_optimistic_false_prune_bound() {
        let one = vec![AuditAllocationStage {
            stage_id: "only".into(),
            pruned_population_size: 8,
            cost_per_sentinel_units: 2,
            risk_weight: 1,
        }];
        let plan = plan_budgeted_audits(&one, &config(16)).unwrap();
        assert_eq!(plan.stages[0].sample_count, 8);
        assert_eq!(plan.stages[0].zero_miss_upper_bound_false_pruned_count, 0);
        assert_eq!(plan.weighted_zero_miss_risk_bound, 0);
        assert_eq!(plan.allocated_cost_units, 16);
    }

    #[test]
    fn extra_budget_is_not_wasted_after_risk_bound_stops_improving() {
        let one = vec![AuditAllocationStage {
            stage_id: "only".into(),
            pruned_population_size: 2,
            cost_per_sentinel_units: 10,
            risk_weight: 1,
        }];
        let plan = plan_budgeted_audits(&one, &config(100)).unwrap();
        assert_eq!(plan.stages[0].sample_count, 2);
        assert_eq!(plan.allocated_cost_units, 20);
        assert_eq!(plan.unallocated_budget_units, 80);
    }

    #[test]
    fn planner_rejects_empty_duplicate_invalid_and_zero_weight_stages() {
        assert!(plan_budgeted_audits(&[], &config(10)).is_err());
        let mut invalid = stages();
        invalid[0].stage_id = " ".into();
        assert!(plan_budgeted_audits(&invalid, &config(10)).is_err());
        let mut duplicate = stages();
        duplicate[1].stage_id = duplicate[0].stage_id.clone();
        assert!(plan_budgeted_audits(&duplicate, &config(10)).is_err());
        let mut zero_population = stages();
        zero_population[0].pruned_population_size = 0;
        assert!(plan_budgeted_audits(&zero_population, &config(10)).is_err());
        let mut zero_cost = stages();
        zero_cost[0].cost_per_sentinel_units = 0;
        assert!(plan_budgeted_audits(&zero_cost, &config(10)).is_err());
        let mut zero_weight = stages();
        zero_weight[0].risk_weight = 0;
        assert!(plan_budgeted_audits(&zero_weight, &config(10)).is_err());
    }

    #[test]
    fn planner_rejects_invalid_confidence_and_unbounded_budget() {
        let mut invalid = config(10);
        invalid.confidence_level = f64::NAN;
        assert!(plan_budgeted_audits(&stages(), &invalid).is_err());
        invalid.confidence_level = 1.0;
        assert!(plan_budgeted_audits(&stages(), &invalid).is_err());
        assert!(plan_budgeted_audits(&stages(), &config(MAX_BUDGET_UNITS + 1)).is_err());
    }

    #[test]
    fn planner_enforces_transition_work_limit() {
        let many = vec![AuditAllocationStage {
            stage_id: "many".into(),
            pruned_population_size: 1_000,
            cost_per_sentinel_units: 1,
            risk_weight: 1,
        }];
        assert!(plan_budgeted_audits(&many, &config(30_000)).is_err());
    }

    #[test]
    fn exact_dp_matches_brute_force_for_small_stage_sets() {
        let small = vec![
            AuditAllocationStage {
                stage_id: "a".into(),
                pruned_population_size: 5,
                cost_per_sentinel_units: 3,
                risk_weight: 2,
            },
            AuditAllocationStage {
                stage_id: "b".into(),
                pruned_population_size: 4,
                cost_per_sentinel_units: 5,
                risk_weight: 1,
            },
        ];
        let budget = 24u64;
        let plan = plan_budgeted_audits(&small, &config(budget)).unwrap();
        let mut best: Option<(u128, u64, Vec<usize>)> = None;
        for a in 0..=5usize {
            for b in 0..=4usize {
                let cost = (a as u64) * 3 + (b as u64) * 5;
                if cost > budget {
                    continue;
                }
                let risk = 2u128
                    * zero_miss_upper_bound_false_pruned_count(5, a, 0.95).unwrap() as u128
                    + zero_miss_upper_bound_false_pruned_count(4, b, 0.95).unwrap() as u128;
                let candidate = (risk, cost, vec![a, b]);
                if best.as_ref().map_or(true, |current| &candidate < current) {
                    best = Some(candidate);
                }
            }
        }
        let best = best.unwrap();
        assert_eq!(plan.weighted_zero_miss_risk_bound, best.0);
        assert_eq!(plan.allocated_cost_units, best.1);
        assert_eq!(
            plan.stages.iter().map(|stage| stage.sample_count).collect::<Vec<_>>(),
            best.2
        );
    }

    #[test]
    fn weighted_risk_arithmetic_and_costs_are_checked() {
        let huge_weight = vec![AuditAllocationStage {
            stage_id: "huge".into(),
            pruned_population_size: 2,
            cost_per_sentinel_units: 1,
            risk_weight: u64::MAX,
        }];
        let plan = plan_budgeted_audits(&huge_weight, &config(0)).unwrap();
        assert_eq!(plan.weighted_zero_miss_risk_bound, (u64::MAX as u128) * 2);
    }

    #[test]
    fn planner_is_outcome_blind_and_returns_optimistic_bounds_only() {
        let plan = plan_budgeted_audits(&stages(), &config(200)).unwrap();
        assert!(plan.stages.iter().all(|stage| {
            stage.zero_miss_upper_bound_false_pruned_rate
                == stage.zero_miss_upper_bound_false_pruned_count as f64
                    / stage.pruned_population_size as f64
        }));
        assert!(plan.stages.iter().all(|stage| stage.sample_count <= stage.pruned_population_size));
    }
}
