//! Candidate-only joint allocation of sensing, memory, computation and verification.
//!
//! This crate accepts candidate predictions and typed resource estimates only.
//! It deliberately contains no evaluator outcomes, hidden future streams,
//! benchmark seeds, oracle code, signing keys or receipt-verification logic.
//! An empty action list is the explicit no-op baseline.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ActionKind {
    Observe,
    Retain,
    Evict,
    Refresh,
    Compute,
    Verify,
    Abstain,
    Defer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct JointAction {
    pub action_id: u64,
    pub kind: ActionKind,
}

/// Positive values mean predicted reductions in the corresponding exposure.
/// Negative values mean the candidate predicts an increase. These are model
/// estimates, never evaluator measurements.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PredictedExposureReduction {
    pub wrong_belief_us: i64,
    pub abstention_us: i64,
    pub missing_output_us: i64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceVector {
    pub persistent_bytes: u64,
    pub retained_byte_us: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub compute_us: u64,
    pub refresh_ops: u64,
    pub observations: u64,
    pub sensor_active_us: u64,
    pub latency_us: u64,
    pub energy_nj: u64,
    pub verification_ops: u64,
    pub receipt_bytes: u64,
}

impl ResourceVector {
    pub const fn values(self) -> [u64; 12] {
        [
            self.persistent_bytes,
            self.retained_byte_us,
            self.read_bytes,
            self.write_bytes,
            self.compute_us,
            self.refresh_ops,
            self.observations,
            self.sensor_active_us,
            self.latency_us,
            self.energy_nj,
            self.verification_ops,
            self.receipt_bytes,
        ]
    }

    pub fn exceeds(self, limits: Self) -> Vec<ResourceDimension> {
        let actual = self.values();
        let maximum = limits.values();
        ResourceDimension::ALL
            .into_iter()
            .enumerate()
            .filter_map(|(index, dimension)| {
                (actual[index] > maximum[index]).then_some(dimension)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointBudget {
    pub resources: ResourceVector,
    pub maximum_observation_gap_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointPlan {
    pub plan_id: u64,
    /// Action IDs must be strictly increasing. The same action ID must retain
    /// the same ActionKind across every alternative in an episode.
    pub actions: Vec<JointAction>,
    pub predicted_reduction: PredictedExposureReduction,
    /// Complete-plan costs, including interactions between subsystems.
    pub estimated_cost: ResourceVector,
    pub predicted_maximum_observation_gap_us: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ExposureAxis {
    WrongBelief,
    Abstention,
    MissingOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceDimension {
    PersistentBytes,
    RetainedByteMicroseconds,
    ReadBytes,
    WriteBytes,
    ComputeMicroseconds,
    RefreshOperations,
    Observations,
    SensorActiveMicroseconds,
    LatencyMicroseconds,
    EnergyNanojoules,
    VerificationOperations,
    ReceiptBytes,
}

impl ResourceDimension {
    pub const ALL: [Self; 12] = [
        Self::PersistentBytes,
        Self::RetainedByteMicroseconds,
        Self::ReadBytes,
        Self::WriteBytes,
        Self::ComputeMicroseconds,
        Self::RefreshOperations,
        Self::Observations,
        Self::SensorActiveMicroseconds,
        Self::LatencyMicroseconds,
        Self::EnergyNanojoules,
        Self::VerificationOperations,
        Self::ReceiptBytes,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedPlan {
    pub plan_id: u64,
    pub resource_overruns: Vec<ResourceDimension>,
    pub observation_gap_overrun_us: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionReport {
    pub selected_plan_id: u64,
    pub feasible_plan_ids: Vec<u64>,
    pub pareto_plan_ids: Vec<u64>,
    pub rejected: Vec<RejectedPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    EmptyPlanSet,
    DuplicatePlanId { plan_id: u64 },
    DuplicateOrUnsortedActionIds { plan_id: u64 },
    ActionKindChanged { action_id: u64 },
    InvalidExposurePriority,
    NoFeasiblePlan { rejected: Vec<RejectedPlan> },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlanSet => write!(f, "joint planner requires at least one complete plan"),
            Self::DuplicatePlanId { plan_id } => write!(f, "duplicate joint plan ID {plan_id}"),
            Self::DuplicateOrUnsortedActionIds { plan_id } => {
                write!(f, "plan {plan_id} action IDs must be unique and strictly increasing")
            }
            Self::ActionKindChanged { action_id } => {
                write!(f, "action {action_id} changes kind across joint-plan alternatives")
            }
            Self::InvalidExposurePriority => {
                write!(f, "exposure priority must contain each of the three axes exactly once")
            }
            Self::NoFeasiblePlan { rejected } => {
                write!(f, "no complete plan satisfies every hard resource and observation-gap ceiling ({} rejected)", rejected.len())
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Filter by every hard ceiling, retain the complete cost/benefit Pareto set,
/// then select deterministically using an explicit non-compensatory exposure
/// priority. Hard budgets are never converted into a scalar penalty.
pub fn select_joint_plan(
    plans: &[JointPlan],
    budget: JointBudget,
    exposure_priority: [ExposureAxis; 3],
) -> Result<SelectionReport, PlanError> {
    if plans.is_empty() {
        return Err(PlanError::EmptyPlanSet);
    }
    let priorities: BTreeSet<_> = exposure_priority.into_iter().collect();
    if priorities.len() != 3 {
        return Err(PlanError::InvalidExposurePriority);
    }

    let mut seen_plans = BTreeSet::new();
    let mut action_kinds = std::collections::BTreeMap::<u64, ActionKind>::new();
    for plan in plans {
        if !seen_plans.insert(plan.plan_id) {
            return Err(PlanError::DuplicatePlanId { plan_id: plan.plan_id });
        }
        if plan.actions.windows(2).any(|pair| pair[0].action_id >= pair[1].action_id) {
            return Err(PlanError::DuplicateOrUnsortedActionIds { plan_id: plan.plan_id });
        }
        for action in &plan.actions {
            if let Some(previous_kind) = action_kinds.get(&action.action_id) {
                if previous_kind != &action.kind {
                    return Err(PlanError::ActionKindChanged { action_id: action.action_id });
                }
            } else {
                action_kinds.insert(action.action_id, action.kind);
            }
        }
    }

    let mut feasible = Vec::new();
    let mut rejected = Vec::new();
    for plan in plans {
        let resource_overruns = plan.estimated_cost.exceeds(budget.resources);
        let gap_overrun = (plan.predicted_maximum_observation_gap_us
            > budget.maximum_observation_gap_us)
            .then_some(plan.predicted_maximum_observation_gap_us);
        if resource_overruns.is_empty() && gap_overrun.is_none() {
            feasible.push(plan.clone());
        } else {
            rejected.push(RejectedPlan {
                plan_id: plan.plan_id,
                resource_overruns,
                observation_gap_overrun_us: gap_overrun,
            });
        }
    }

    if feasible.is_empty() {
        rejected.sort_by_key(|row| row.plan_id);
        return Err(PlanError::NoFeasiblePlan { rejected });
    }

    let mut pareto = feasible
        .iter()
        .filter(|candidate| {
            !feasible.iter().any(|other| {
                other.plan_id != candidate.plan_id && dominates(other, candidate)
            })
        })
        .cloned()
        .collect::<Vec<_>>();

    pareto.sort_by(|left, right| compare_selection(left, right, exposure_priority));
    let selected_plan_id = pareto[0].plan_id;

    let mut feasible_plan_ids = feasible.iter().map(|plan| plan.plan_id).collect::<Vec<_>>();
    feasible_plan_ids.sort_unstable();
    let mut pareto_plan_ids = pareto.iter().map(|plan| plan.plan_id).collect::<Vec<_>>();
    pareto_plan_ids.sort_unstable();
    rejected.sort_by_key(|row| row.plan_id);

    Ok(SelectionReport {
        selected_plan_id,
        feasible_plan_ids,
        pareto_plan_ids,
        rejected,
    })
}

fn dominates(left: &JointPlan, right: &JointPlan) -> bool {
    let left_benefit = [
        left.predicted_reduction.wrong_belief_us,
        left.predicted_reduction.abstention_us,
        left.predicted_reduction.missing_output_us,
    ];
    let right_benefit = [
        right.predicted_reduction.wrong_belief_us,
        right.predicted_reduction.abstention_us,
        right.predicted_reduction.missing_output_us,
    ];
    let left_cost = left.estimated_cost.values();
    let right_cost = right.estimated_cost.values();

    let benefit_no_worse = left_benefit.iter().zip(right_benefit).all(|(a, b)| *a >= b);
    let cost_no_worse = left_cost.iter().zip(right_cost).all(|(a, b)| *a <= b);
    let gap_no_worse = left.predicted_maximum_observation_gap_us
        <= right.predicted_maximum_observation_gap_us;
    if !benefit_no_worse || !cost_no_worse || !gap_no_worse {
        return false;
    }

    let strictly_better = left_benefit.iter().zip(right_benefit).any(|(a, b)| *a > b)
        || left_cost.iter().zip(right_cost).any(|(a, b)| *a < b)
        || left.predicted_maximum_observation_gap_us
            < right.predicted_maximum_observation_gap_us;

    // Exact duplicates collapse to the lower stable plan ID.
    strictly_better || left.plan_id < right.plan_id
}

fn compare_selection(
    left: &JointPlan,
    right: &JointPlan,
    exposure_priority: [ExposureAxis; 3],
) -> Ordering {
    for axis in exposure_priority {
        let left_value = exposure_value(left.predicted_reduction, axis);
        let right_value = exposure_value(right.predicted_reduction, axis);
        let ordering = right_value.cmp(&left_value);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.estimated_cost
        .values()
        .cmp(&right.estimated_cost.values())
        .then_with(|| left.predicted_maximum_observation_gap_us
            .cmp(&right.predicted_maximum_observation_gap_us))
        .then_with(|| left.plan_id.cmp(&right.plan_id))
}

fn exposure_value(value: PredictedExposureReduction, axis: ExposureAxis) -> i64 {
    match axis {
        ExposureAxis::WrongBelief => value.wrong_belief_us,
        ExposureAxis::Abstention => value.abstention_us,
        ExposureAxis::MissingOutput => value.missing_output_us,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIORITY: [ExposureAxis; 3] = [
        ExposureAxis::WrongBelief,
        ExposureAxis::Abstention,
        ExposureAxis::MissingOutput,
    ];

    fn costs(persistent_bytes: u64, observations: u64, compute_us: u64) -> ResourceVector {
        ResourceVector {
            persistent_bytes,
            observations,
            compute_us,
            ..ResourceVector::default()
        }
    }

    fn plan(
        id: u64,
        action_ids: &[u64],
        benefit: [i64; 3],
        cost: ResourceVector,
        gap_us: u64,
    ) -> JointPlan {
        JointPlan {
            plan_id: id,
            actions: action_ids.iter().map(|action_id| JointAction {
                action_id: *action_id,
                kind: match *action_id {
                    10 => ActionKind::Observe,
                    20 => ActionKind::Retain,
                    30 => ActionKind::Compute,
                    40 => ActionKind::Verify,
                    _ => ActionKind::Defer,
                },
            }).collect(),
            predicted_reduction: PredictedExposureReduction {
                wrong_belief_us: benefit[0],
                abstention_us: benefit[1],
                missing_output_us: benefit[2],
            },
            estimated_cost: cost,
            predicted_maximum_observation_gap_us: gap_us,
        }
    }

    fn budget() -> JointBudget {
        JointBudget {
            resources: costs(64, 4, 100),
            maximum_observation_gap_us: 50,
        }
    }

    #[test]
    fn joint_plan_can_capture_sensing_memory_compute_verification_synergy() {
        let alternatives = [
            plan(1, &[10], [3, 0, 0], costs(0, 2, 0), 40),
            plan(2, &[20], [2, 0, 0], costs(32, 0, 0), 50),
            plan(3, &[30], [1, 1, 0], costs(0, 0, 60), 50),
            plan(4, &[10, 20, 30, 40], [12, 5, 4], costs(32, 2, 60), 20),
        ];
        let report = select_joint_plan(&alternatives, budget(), PRIORITY).unwrap();
        assert_eq!(report.selected_plan_id, 4);
        assert!(report.pareto_plan_ids.contains(&4));
    }

    #[test]
    fn pareto_frontier_preserves_non_compensatory_exposure_tradeoffs() {
        let alternatives = [
            plan(1, &[10], [9, 1, 1], costs(8, 1, 10), 30),
            plan(2, &[20], [1, 9, 1], costs(8, 1, 10), 30),
            plan(3, &[30], [8, 1, 1], costs(16, 1, 10), 30),
        ];
        let report = select_joint_plan(&alternatives, budget(), PRIORITY).unwrap();
        assert_eq!(report.pareto_plan_ids, vec![1, 2]);
        assert_eq!(report.selected_plan_id, 1);
    }

    #[test]
    fn very_high_predicted_benefit_cannot_buy_a_hard_budget_violation() {
        let alternatives = [
            plan(1, &[10], [i64::MAX, i64::MAX, i64::MAX], costs(65, 1, 1), 1),
            plan(2, &[], [1, 1, 1], costs(1, 1, 1), 1),
        ];
        let report = select_joint_plan(&alternatives, budget(), PRIORITY).unwrap();
        assert_eq!(report.selected_plan_id, 2);
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].resource_overruns, vec![ResourceDimension::PersistentBytes]);
    }

    #[test]
    fn no_feasible_plan_fails_closed_and_reports_each_overrun() {
        let only = [plan(7, &[10], [100, 0, 0], ResourceVector {
            energy_nj: 9,
            verification_ops: 8,
            receipt_bytes: 7,
            ..ResourceVector::default()
        }, 1)];
        let mut limits = ResourceVector::default();
        limits.energy_nj = 8;
        limits.verification_ops = 7;
        limits.receipt_bytes = 6;
        let error = select_joint_plan(&only, JointBudget {
            resources: limits,
            maximum_observation_gap_us: 0,
        }, PRIORITY).unwrap_err();
        match error {
            PlanError::NoFeasiblePlan { rejected } => {
                assert_eq!(rejected.len(), 1);
                assert_eq!(rejected[0].resource_overruns, vec![
                    ResourceDimension::EnergyNanojoules,
                    ResourceDimension::VerificationOperations,
                    ResourceDimension::ReceiptBytes,
                ]);
                assert_eq!(rejected[0].observation_gap_overrun_us, Some(1));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn observation_gap_is_a_hard_constraint_not_a_soft_penalty() {
        let alternatives = [
            plan(1, &[10], [i64::MAX, 0, 0], costs(1, 1, 1), 51),
            plan(2, &[20], [1, 0, 0], costs(1, 1, 1), 50),
        ];
        let report = select_joint_plan(&alternatives, budget(), PRIORITY).unwrap();
        assert_eq!(report.selected_plan_id, 2);
        assert_eq!(report.rejected[0].observation_gap_overrun_us, Some(51));
    }

    #[test]
    fn exact_duplicate_alternatives_collapse_to_smallest_plan_id() {
        let a = plan(8, &[10], [4, 4, 4], costs(4, 1, 1), 20);
        let b = plan(3, &[20], [4, 4, 4], costs(4, 1, 1), 20);
        let report = select_joint_plan(&[a, b], budget(), PRIORITY).unwrap();
        assert_eq!(report.pareto_plan_ids, vec![3]);
        assert_eq!(report.selected_plan_id, 3);
    }

    #[test]
    fn malformed_action_and_plan_ids_are_rejected() {
        let duplicate_actions = [plan(1, &[10, 10], [1, 1, 1], costs(1, 1, 1), 1)];
        assert!(matches!(
            select_joint_plan(&duplicate_actions, budget(), PRIORITY),
            Err(PlanError::DuplicateOrUnsortedActionIds { plan_id: 1 })
        ));
        let duplicate_plans = [
            plan(2, &[10], [1, 1, 1], costs(1, 1, 1), 1),
            plan(2, &[20], [2, 2, 2], costs(1, 1, 1), 1),
        ];
        assert!(matches!(
            select_joint_plan(&duplicate_plans, budget(), PRIORITY),
            Err(PlanError::DuplicatePlanId { plan_id: 2 })
        ));
    }

    #[test]
    fn action_kind_is_stable_across_complete_plan_alternatives() {
        let first = plan(1, &[10], [2, 0, 0], costs(1, 1, 1), 1);
        let mut second = plan(2, &[10], [3, 0, 0], costs(1, 1, 1), 1);
        second.actions[0].kind = ActionKind::Verify;
        assert_eq!(
            select_joint_plan(&[first, second], budget(), PRIORITY),
            Err(PlanError::ActionKindChanged { action_id: 10 })
        );
    }

    #[test]
    fn no_op_baseline_is_a_valid_candidate_plan() {
        let noop = plan(0, &[], [0, 0, 0], ResourceVector::default(), 50);
        let report = select_joint_plan(&[noop], budget(), PRIORITY).unwrap();
        assert_eq!(report.selected_plan_id, 0);
        assert_eq!(report.feasible_plan_ids, vec![0]);
    }

    #[test]
    fn invalid_priority_that_omits_an_exposure_axis_is_rejected() {
        let duplicate_priority = [
            ExposureAxis::WrongBelief,
            ExposureAxis::WrongBelief,
            ExposureAxis::MissingOutput,
        ];
        let candidate = [plan(1, &[], [1, 1, 1], ResourceVector::default(), 0)];
        assert_eq!(
            select_joint_plan(&candidate, budget(), duplicate_priority),
            Err(PlanError::InvalidExposurePriority)
        );
    }
}
