//! Decision-focused value-of-information planning for active sensor acquisition.
//!
//! The planner estimates the reduction in expected task decision loss from each possible
//! next observation. It deliberately does not rank actions by entropy reduction alone:
//! an observation that reveals a lot about a decision-irrelevant variable has zero task VoI.
//!
//! Sensor energy, wake energy, compute cycles, latency, transmit bytes and memory writes remain
//! a resource vector. The planner returns a non-dominated action frontier; it does not invent
//! a scalar exchange rate between physically unlike resources. A product may apply an explicit,
//! preregistered utility function to choose one action from that frontier.
//!
//! This is a decision-support planner, not a physical measurement. Resource estimates must be
//! calibrated/validated by the system track before an energy or battery-life claim.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const MANIFEST_SCHEMA: &str = "symthaea.temporal.active-sensing-voi-plan";
const VERSION: &str = "0.1.0";
const PROBABILITY_SCALE: u64 = 1_000_000;
const MAX_ACTIONS: usize = 100_000;
const MAX_OUTCOMES_PER_ACTION: usize = 10_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingError {
    pub field: String,
    pub message: String,
}

impl ActiveSensingError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for ActiveSensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for ActiveSensingError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum ActiveSensingActionKind {
    SampleNow = 0,
    ActivateModality = 1,
    IncreaseSamplingRate = 2,
    DecreaseSamplingRate = 3,
    TransmitSummary = 4,
    WaitOrSleep = 5,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ActiveSensingResourceCost {
    pub sensor_energy_nj: u64,
    pub wake_energy_nj: u64,
    pub compute_cycles: u64,
    pub latency_us: u64,
    pub transmit_bytes: u64,
    pub memory_write_bytes: u64,
}

impl ActiveSensingResourceCost {
    fn fits(self, budget: Self) -> bool {
        self.sensor_energy_nj <= budget.sensor_energy_nj
            && self.wake_energy_nj <= budget.wake_energy_nj
            && self.compute_cycles <= budget.compute_cycles
            && self.latency_us <= budget.latency_us
            && self.transmit_bytes <= budget.transmit_bytes
            && self.memory_write_bytes <= budget.memory_write_bytes
    }

    fn consumes_anything(self) -> bool {
        self.sensor_energy_nj > 0
            || self.wake_energy_nj > 0
            || self.compute_cycles > 0
            || self.latency_us > 0
            || self.transmit_bytes > 0
            || self.memory_write_bytes > 0
    }

    fn componentwise_no_more_than(self, other: Self) -> bool {
        self.sensor_energy_nj <= other.sensor_energy_nj
            && self.wake_energy_nj <= other.wake_energy_nj
            && self.compute_cycles <= other.compute_cycles
            && self.latency_us <= other.latency_us
            && self.transmit_bytes <= other.transmit_bytes
            && self.memory_write_bytes <= other.memory_write_bytes
    }

    fn componentwise_strictly_less(self, other: Self) -> bool {
        self.sensor_energy_nj < other.sensor_energy_nj
            || self.wake_energy_nj < other.wake_energy_nj
            || self.compute_cycles < other.compute_cycles
            || self.latency_us < other.latency_us
            || self.transmit_bytes < other.transmit_bytes
            || self.memory_write_bytes < other.memory_write_bytes
    }

    fn hash_into(self, hasher: &mut CanonicalHasher) {
        hasher.u64(self.sensor_energy_nj);
        hasher.u64(self.wake_energy_nj);
        hasher.u64(self.compute_cycles);
        hasher.u64(self.latency_us);
        hasher.u64(self.transmit_bytes);
        hasher.u64(self.memory_write_bytes);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationOutcomeEstimate {
    /// Probability that this observation outcome occurs, in parts per million.
    pub probability_ppm: u32,
    /// Optimal task loss after updating the decision using this outcome, in ppm.
    pub posterior_decision_loss_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingActionEstimate {
    pub action_id: String,
    pub kind: ActiveSensingActionKind,
    pub resource_cost: ActiveSensingResourceCost,
    /// Model-estimated entropy reduction is diagnostic only; it is never used to rank actions.
    pub claimed_entropy_reduction_ppm: u32,
    pub outcomes: Vec<ObservationOutcomeEstimate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveSensingDecisionContext {
    /// Minimum expected task decision loss before observing, in ppm.
    pub current_expected_decision_loss_ppm: u32,
    /// Maximum permitted incremental resource use for the next action.
    pub remaining_resource_budget: ActiveSensingResourceCost,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingActionValue {
    pub action_id: String,
    pub kind: ActiveSensingActionKind,
    pub resource_cost: ActiveSensingResourceCost,
    pub claimed_entropy_reduction_ppm: u32,
    /// Forecast outcome distribution used to reproduce the expected task-loss estimate.
    pub outcome_estimates: Vec<ObservationOutcomeEstimate>,
    pub expected_loss_without_observation_ppm: u32,
    pub expected_loss_after_observation_ppm: u32,
    /// Positive is beneficial, zero is decision-irrelevant, negative is harmful.
    pub expected_task_loss_reduction_ppm: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActiveSensingRejectionReason {
    ExceedsResourceBudget,
    NoExpectedTaskLossReduction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedActiveSensingAction {
    pub action_id: String,
    pub reason: ActiveSensingRejectionReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveSensingPlanReceipt {
    pub schema: String,
    pub version: String,
    pub current_expected_decision_loss_ppm: u32,
    pub remaining_resource_budget: ActiveSensingResourceCost,
    /// Feasible, positive-VoI and non-dominated actions.
    pub pareto_frontier: Vec<ActiveSensingActionValue>,
    pub rejected_actions: Vec<RejectedActiveSensingAction>,
    pub plan_sha256: [u8; 32],
}

fn compute_expected_loss_ppm(
    outcomes: &[ObservationOutcomeEstimate],
) -> Result<u32, ActiveSensingError> {
    if outcomes.is_empty() || outcomes.len() > MAX_OUTCOMES_PER_ACTION {
        return Err(ActiveSensingError::new(
            "action.outcomes",
            format!("outcome count must be in 1..={MAX_OUTCOMES_PER_ACTION}"),
        ));
    }
    let mut probability_sum = 0_u64;
    let mut weighted_loss = 0_u128;
    for (index, outcome) in outcomes.iter().enumerate() {
        if outcome.probability_ppm > PROBABILITY_SCALE as u32
            || outcome.posterior_decision_loss_ppm > PROBABILITY_SCALE as u32
        {
            return Err(ActiveSensingError::new(
                format!("action.outcomes[{index}]"),
                "probability and decision loss must be in [0, 1,000,000] ppm",
            ));
        }
        probability_sum = probability_sum
            .checked_add(u64::from(outcome.probability_ppm))
            .ok_or_else(|| ActiveSensingError::new("action.outcomes", "probability sum overflow"))?;
        weighted_loss = weighted_loss
            .checked_add(
                u128::from(outcome.probability_ppm)
                    * u128::from(outcome.posterior_decision_loss_ppm),
            )
            .ok_or_else(|| ActiveSensingError::new("action.outcomes", "weighted loss overflow"))?;
    }
    if probability_sum != PROBABILITY_SCALE {
        return Err(ActiveSensingError::new(
            "action.outcomes.probability_ppm",
            format!("outcome probabilities must sum to {PROBABILITY_SCALE}, got {probability_sum}"),
        ));
    }
    let rounded = weighted_loss
        .checked_add(u128::from(PROBABILITY_SCALE / 2))
        .ok_or_else(|| ActiveSensingError::new("action.outcomes", "rounding overflow"))?
        / u128::from(PROBABILITY_SCALE);
    u32::try_from(rounded)
        .map_err(|_| ActiveSensingError::new("action.outcomes", "expected loss does not fit in u32"))
}

fn dominates(left: &ActiveSensingActionValue, right: &ActiveSensingActionValue) -> bool {
    let benefit_no_worse =
        left.expected_task_loss_reduction_ppm >= right.expected_task_loss_reduction_ppm;
    let resources_no_worse = left.resource_cost.componentwise_no_more_than(right.resource_cost);
    let strict = left.expected_task_loss_reduction_ppm > right.expected_task_loss_reduction_ppm
        || left.resource_cost.componentwise_strictly_less(right.resource_cost);
    benefit_no_worse && resources_no_worse && strict
}

fn hash_text(hasher: &mut CanonicalHasher, value: &str) {
    hasher.bytes(value.as_bytes());
}

pub fn compute_active_sensing_plan_sha256(receipt: &ActiveSensingPlanReceipt) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.active-sensing-voi-plan.v0.1");
    hash_text(&mut hasher, &receipt.schema);
    hash_text(&mut hasher, &receipt.version);
    hasher.u32(receipt.current_expected_decision_loss_ppm);
    receipt.remaining_resource_budget.hash_into(&mut hasher);
    hasher.usize(receipt.pareto_frontier.len());
    for action in &receipt.pareto_frontier {
        hash_text(&mut hasher, &action.action_id);
        hasher.u8(action.kind as u8);
        action.resource_cost.hash_into(&mut hasher);
        hasher.u32(action.claimed_entropy_reduction_ppm);
        hasher.usize(action.outcome_estimates.len());
        for outcome in &action.outcome_estimates {
            hasher.u32(outcome.probability_ppm);
            hasher.u32(outcome.posterior_decision_loss_ppm);
        }
        hasher.u32(action.expected_loss_without_observation_ppm);
        hasher.u32(action.expected_loss_after_observation_ppm);
        hasher.u64(action.expected_task_loss_reduction_ppm as u64);
    }
    hasher.usize(receipt.rejected_actions.len());
    for rejected in &receipt.rejected_actions {
        hash_text(&mut hasher, &rejected.action_id);
        hasher.u8(rejected.reason as u8);
    }
    hasher.finish()
}

/// Evaluate candidate next-observation actions and return the Pareto frontier.
///
/// Expected task loss is computed from a finite, declared distribution of observation outcomes.
/// Actions with high entropy reduction but zero/negative task-loss reduction are rejected.
/// All resource constraints are checked component-wise. No action is selected by hidden scalar
/// weights; the caller's application policy chooses among the returned non-dominated options.
pub fn plan_active_sensing_actions(
    context: ActiveSensingDecisionContext,
    actions: &[ActiveSensingActionEstimate],
) -> Result<ActiveSensingPlanReceipt, ActiveSensingError> {
    if context.current_expected_decision_loss_ppm > PROBABILITY_SCALE as u32 {
        return Err(ActiveSensingError::new(
            "context.current_expected_decision_loss_ppm",
            "must be in [0, 1,000,000] ppm",
        ));
    }
    if actions.is_empty() || actions.len() > MAX_ACTIONS {
        return Err(ActiveSensingError::new(
            "actions",
            format!("action count must be in 1..={MAX_ACTIONS}"),
        ));
    }
    if actions.windows(2).any(|pair| pair[0].action_id >= pair[1].action_id) {
        return Err(ActiveSensingError::new(
            "actions.action_id",
            "actions must be sorted by unique action_id",
        ));
    }

    let mut values = Vec::with_capacity(actions.len());
    let mut rejected = Vec::new();
    let mut seen = HashSet::with_capacity(actions.len());

    for (index, action) in actions.iter().enumerate() {
        if action.action_id.trim().is_empty()
            || !seen.insert(action.action_id.as_str())
            || action.claimed_entropy_reduction_ppm > PROBABILITY_SCALE as u32
        {
            return Err(ActiveSensingError::new(
                format!("actions[{index}]"),
                "action ID must be non-empty/unique and claimed entropy reduction in [0, 1,000,000] ppm",
            ));
        }
        if !action.resource_cost.consumes_anything() {
            return Err(ActiveSensingError::new(
                format!("actions[{index}].resource_cost"),
                "action must declare at least one non-zero incremental resource cost",
            ));
        }
        let expected_after = compute_expected_loss_ppm(&action.outcomes)?;
        let value = ActiveSensingActionValue {
            action_id: action.action_id.clone(),
            kind: action.kind,
            resource_cost: action.resource_cost,
            claimed_entropy_reduction_ppm: action.claimed_entropy_reduction_ppm,
            outcome_estimates: action.outcomes.clone(),
            expected_loss_without_observation_ppm: context.current_expected_decision_loss_ppm,
            expected_loss_after_observation_ppm: expected_after,
            expected_task_loss_reduction_ppm:
                i64::from(context.current_expected_decision_loss_ppm) - i64::from(expected_after),
        };

        if !action.resource_cost.fits(context.remaining_resource_budget) {
            rejected.push(RejectedActiveSensingAction {
                action_id: action.action_id.clone(),
                reason: ActiveSensingRejectionReason::ExceedsResourceBudget,
            });
        } else if value.expected_task_loss_reduction_ppm <= 0 {
            rejected.push(RejectedActiveSensingAction {
                action_id: action.action_id.clone(),
                reason: ActiveSensingRejectionReason::NoExpectedTaskLossReduction,
            });
        } else {
            values.push(value);
        }
    }

    let mut frontier = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let dominated = values.iter().enumerate().any(|(other_index, other)| {
            other_index != index && dominates(other, value)
        });
        if !dominated {
            frontier.push(value.clone());
        }
    }
    frontier.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    rejected.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    let mut receipt = ActiveSensingPlanReceipt {
        schema: MANIFEST_SCHEMA.into(),
        version: VERSION.into(),
        current_expected_decision_loss_ppm: context.current_expected_decision_loss_ppm,
        remaining_resource_budget: context.remaining_resource_budget,
        pareto_frontier: frontier,
        rejected_actions: rejected,
        plan_sha256: [0; 32],
    };
    receipt.plan_sha256 = compute_active_sensing_plan_sha256(&receipt);
    Ok(receipt)
}

pub fn validate_active_sensing_plan_receipt(
    receipt: &ActiveSensingPlanReceipt,
) -> Result<(), ActiveSensingError> {
    if receipt.schema != MANIFEST_SCHEMA || receipt.version != VERSION {
        return Err(ActiveSensingError::new("schema_version", "unsupported plan schema/version"));
    }
    if receipt.current_expected_decision_loss_ppm > PROBABILITY_SCALE as u32 {
        return Err(ActiveSensingError::new("current_expected_decision_loss_ppm", "must be in [0, 1,000,000] ppm"));
    }
    if receipt.pareto_frontier.windows(2).any(|pair| pair[0].action_id >= pair[1].action_id)
        || receipt.rejected_actions.windows(2).any(|pair| pair[0].action_id >= pair[1].action_id)
    {
        return Err(ActiveSensingError::new("plan.order", "frontier and rejected lists must be sorted by unique action ID"));
    }
    let mut all_action_ids = HashSet::with_capacity(
        receipt.pareto_frontier.len() + receipt.rejected_actions.len()
    );
    for action in &receipt.pareto_frontier {
        if !all_action_ids.insert(action.action_id.as_str()) {
            return Err(ActiveSensingError::new("plan.action_ids", "duplicate action ID across plan receipt"));
        }
    }
    for rejected in &receipt.rejected_actions {
        if rejected.action_id.trim().is_empty()
            || !all_action_ids.insert(rejected.action_id.as_str())
        {
            return Err(ActiveSensingError::new("plan.rejected_actions", "rejected action IDs must be non-empty and unique across frontier/rejections"));
        }
    }
    for action in &receipt.pareto_frontier {
        let reproduced_loss = compute_expected_loss_ppm(&action.outcome_estimates)?;
        if action.action_id.trim().is_empty()
            || !action.resource_cost.consumes_anything()
            || action.claimed_entropy_reduction_ppm > PROBABILITY_SCALE as u32
            || action.outcome_estimates.is_empty()
            || action.expected_task_loss_reduction_ppm <= 0
            || action.expected_loss_without_observation_ppm != receipt.current_expected_decision_loss_ppm
            || action.expected_loss_after_observation_ppm > PROBABILITY_SCALE as u32
            || action.expected_loss_after_observation_ppm != reproduced_loss
            || i64::from(action.expected_loss_without_observation_ppm)
                - i64::from(action.expected_loss_after_observation_ppm)
                != action.expected_task_loss_reduction_ppm
            || !action.resource_cost.fits(receipt.remaining_resource_budget)
        {
            return Err(ActiveSensingError::new("plan.frontier", "frontier action metrics/resource cost are inconsistent"));
        }
    }
    for (index, action) in receipt.pareto_frontier.iter().enumerate() {
        if receipt.pareto_frontier.iter().enumerate().any(|(other_index, other)| {
            other_index != index && dominates(other, action)
        }) {
            return Err(ActiveSensingError::new("plan.frontier", "frontier contains a dominated action"));
        }
    }
    if compute_active_sensing_plan_sha256(receipt) != receipt.plan_sha256 {
        return Err(ActiveSensingError::new("plan_sha256", "does not match canonical plan receipt"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(
        sensor: u64,
        wake: u64,
        compute: u64,
        latency: u64,
        tx: u64,
        writes: u64,
    ) -> ActiveSensingResourceCost {
        ActiveSensingResourceCost {
            sensor_energy_nj: sensor,
            wake_energy_nj: wake,
            compute_cycles: compute,
            latency_us: latency,
            transmit_bytes: tx,
            memory_write_bytes: writes,
        }
    }

    fn action(
        id: &str,
        kind: ActiveSensingActionKind,
        resource_cost: ActiveSensingResourceCost,
        entropy: u32,
        outcomes: Vec<ObservationOutcomeEstimate>,
    ) -> ActiveSensingActionEstimate {
        ActiveSensingActionEstimate {
            action_id: id.into(),
            kind,
            resource_cost,
            claimed_entropy_reduction_ppm: entropy,
            outcomes,
        }
    }

    fn context() -> ActiveSensingDecisionContext {
        ActiveSensingDecisionContext {
            current_expected_decision_loss_ppm: 300_000,
            remaining_resource_budget: cost(100_000, 50_000, 2_000_000, 10_000, 1_024, 4_096),
        }
    }

    #[test]
    fn decision_irrelevant_high_entropy_observation_is_rejected() {
        // The sensor claims a large entropy reduction, but neither outcome changes task loss.
        let irrelevant = action(
            "irrelevant",
            ActiveSensingActionKind::ActivateModality,
            cost(100, 10, 100, 10, 0, 0),
            900_000,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 300_000 },
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 300_000 },
            ],
        );
        let plan = plan_active_sensing_actions(context(), &[irrelevant]).unwrap();
        assert!(plan.pareto_frontier.is_empty());
        assert_eq!(plan.rejected_actions[0].reason, ActiveSensingRejectionReason::NoExpectedTaskLossReduction);
    }

    #[test]
    fn decision_relevant_observation_enters_frontier() {
        let useful = action(
            "useful",
            ActiveSensingActionKind::SampleNow,
            cost(100, 10, 100, 10, 0, 0),
            100,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 100_000 },
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 200_000 },
            ],
        );
        let plan = plan_active_sensing_actions(context(), &[useful]).unwrap();
        assert_eq!(plan.pareto_frontier.len(), 1);
        assert_eq!(plan.pareto_frontier[0].expected_task_loss_reduction_ppm, 150_000);
    }

    #[test]
    fn resource_tradeoffs_are_preserved_as_a_pareto_frontier() {
        let cheaper_sensor = action(
            "cheap_sensor",
            ActiveSensingActionKind::SampleNow,
            cost(50, 10, 100, 10, 0, 0),
            100,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 150_000 },
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 250_000 },
            ],
        );
        let lower_compute = action(
            "low_compute",
            ActiveSensingActionKind::ActivateModality,
            cost(100, 20, 50, 10, 32, 0),
            200,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 100_000 },
                ObservationOutcomeEstimate { probability_ppm: 500_000, posterior_decision_loss_ppm: 150_000 },
            ],
        );
        let plan = plan_active_sensing_actions(context(), &[cheaper_sensor, lower_compute]).unwrap();
        assert_eq!(plan.pareto_frontier.len(), 2);
    }

    #[test]
    fn budget_is_checked_componentwise_and_over_budget_action_is_rejected() {
        let expensive = action(
            "over_budget",
            ActiveSensingActionKind::TransmitSummary,
            cost(100, 10, 100, 10, 2_048, 0),
            100,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 1_000_000, posterior_decision_loss_ppm: 100_000 },
            ],
        );
        let plan = plan_active_sensing_actions(context(), &[expensive]).unwrap();
        assert!(plan.pareto_frontier.is_empty());
        assert_eq!(plan.rejected_actions[0].reason, ActiveSensingRejectionReason::ExceedsResourceBudget);
    }

    #[test]
    fn invalid_outcome_probabilities_and_unordered_actions_fail_closed() {
        let invalid = action(
            "invalid",
            ActiveSensingActionKind::SampleNow,
            cost(100, 10, 100, 10, 0, 0),
            100,
            vec![ObservationOutcomeEstimate { probability_ppm: 999_999, posterior_decision_loss_ppm: 100_000 }],
        );
        assert!(plan_active_sensing_actions(context(), &[invalid]).is_err());

        let a = action(
            "z",
            ActiveSensingActionKind::SampleNow,
            cost(100, 10, 100, 10, 0, 0),
            100,
            vec![ObservationOutcomeEstimate { probability_ppm: 1_000_000, posterior_decision_loss_ppm: 100_000 }],
        );
        let b = action(
            "a",
            ActiveSensingActionKind::SampleNow,
            cost(100, 10, 100, 10, 0, 0),
            100,
            vec![ObservationOutcomeEstimate { probability_ppm: 1_000_000, posterior_decision_loss_ppm: 100_000 }],
        );
        assert!(plan_active_sensing_actions(context(), &[a, b]).is_err());
    }

    #[test]
    fn receipt_digest_and_validator_reject_tampering() {
        let useful = action(
            "useful",
            ActiveSensingActionKind::SampleNow,
            cost(100, 10, 100, 10, 0, 0),
            100,
            vec![
                ObservationOutcomeEstimate { probability_ppm: 1_000_000, posterior_decision_loss_ppm: 100_000 },
            ],
        );
        let mut plan = plan_active_sensing_actions(context(), &[useful]).unwrap();
        validate_active_sensing_plan_receipt(&plan).unwrap();
        plan.pareto_frontier[0].resource_cost.sensor_energy_nj += 1;
        assert!(validate_active_sensing_plan_receipt(&plan).is_err());
    }
}
