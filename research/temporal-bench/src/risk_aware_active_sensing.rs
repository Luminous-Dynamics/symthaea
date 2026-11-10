//! Risk-aware goal-oriented evaluation for decision-focused active sensing.
//!
//! This additive layer consumes the same finite observation-outcome distributions as
//! `active_sensing`. It preserves the expected task-loss reduction, upper-tail CVaR,
//! worst-case posterior task loss, and the six-dimensional resource vector as separate
//! decision axes. It does not collapse unlike risks/resources into an implicit scalar.
//!
//! CVaR is computed over the discrete distribution of posterior task loss across possible
//! observation outcomes. A caller may impose a preregistered CVaR ceiling and/or worst-case
//! loss ceiling; otherwise the planner returns a non-dominated frontier over benefit, both
//! risk measures, and the complete resource vector.
//!
//! This is a finite-outcome risk-aware adapter inspired by risk-aware goal-oriented OED.
//! It is not an exact reproduction of the nested risk functional in Jakeman et al. (2026),
//! not a calibrated probability model, and not physical sensing or safety authority.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use super::active_sensing::{
    ActiveSensingActionEstimate, ActiveSensingActionKind, ActiveSensingDecisionContext,
    ActiveSensingResourceCost, ObservationOutcomeEstimate,
};
use super::digest::CanonicalHasher;

const PROBABILITY_SCALE: u64 = 1_000_000;
const MAX_ACTIONS: usize = 100_000;
const MAX_OUTCOMES_PER_ACTION: usize = 10_000;

/// Risk-aware planner input or receipt validation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RiskAwareSensingError {
    /// Field or nested input path that failed validation.
    pub field: String,
    /// Human-readable failure reason.
    pub message: String,
}

impl RiskAwareSensingError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for RiskAwareSensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for RiskAwareSensingError {}

/// Explicit tail-risk policy for posterior task loss.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RiskAwareSensingPolicy {
    /// Upper-tail probability mass used for CVaR, in parts per million.
    /// For example, 50_000 evaluates the worst 5% of posterior-loss outcomes.
    pub tail_probability_ppm: u32,
    /// Optional maximum permitted upper-tail CVaR, in normalized loss ppm.
    pub maximum_posterior_cvar_ppm: Option<u32>,
    /// Optional maximum permitted worst-case posterior loss, in normalized loss ppm.
    pub maximum_worst_case_loss_ppm: Option<u32>,
}

/// Why an action was excluded from the risk-aware frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RiskAwareRejectionReason {
    /// At least one incremental resource cost exceeds its corresponding budget.
    ExceedsResourceBudget = 0,
    /// The expected task-loss reduction is not strictly positive.
    NoExpectedTaskLossReduction = 1,
    /// The action's upper-tail CVaR exceeds the preregistered ceiling.
    ExceedsPosteriorCvarCeiling = 2,
    /// The action's worst-case posterior loss exceeds the preregistered ceiling.
    ExceedsWorstCaseLossCeiling = 3,
}

/// Rejected action identity and reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedRiskAwareAction {
    /// Stable action identity from the input estimate.
    pub action_id: String,
    /// Exact hard-gate reason for rejection.
    pub reason: RiskAwareRejectionReason,
}

/// Risk-aware value for one feasible action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RiskAwareActionValue {
    /// Stable action identity.
    pub action_id: String,
    /// Type of sensing/communication action.
    pub kind: ActiveSensingActionKind,
    /// Full incremental resource vector.
    pub resource_cost: ActiveSensingResourceCost,
    /// Digest of the cost model used to estimate this vector.
    pub resource_cost_model_sha256: [u8; 32],
    /// Original finite outcome distribution, retained for independent replay.
    pub outcome_estimates: Vec<ObservationOutcomeEstimate>,
    /// Expected posterior task decision loss, in normalized ppm.
    pub expected_loss_after_observation_ppm: u32,
    /// Positive is beneficial, zero is decision-irrelevant, negative is harmful.
    pub expected_task_loss_reduction_ppm: i64,
    /// Mean posterior task loss over the worst registered tail probability mass.
    pub posterior_cvar_ppm: u32,
    /// Maximum posterior task loss among positive-probability outcomes.
    pub posterior_worst_case_loss_ppm: u32,
}

/// Complete risk-aware active-sensing plan receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RiskAwareActiveSensingReceipt {
    /// SHA-256 identity of current public state/history.
    pub public_state_sha256: [u8; 32],
    /// SHA-256 identity of the forecasting model.
    pub forecast_model_sha256: [u8; 32],
    /// Calibration receipt for forecast probabilities/posterior losses.
    pub calibration_receipt_sha256: [u8; 32],
    /// Frozen task decision-loss and action semantics.
    pub decision_policy_sha256: [u8; 32],
    /// Baseline expected decision loss before the next observation.
    pub current_expected_decision_loss_ppm: u32,
    /// Component-wise budget for the next observation action.
    pub remaining_resource_budget: ActiveSensingResourceCost,
    /// Frozen CVaR tail probability and optional risk ceilings.
    pub policy: RiskAwareSensingPolicy,
    /// Non-dominated action frontier in benefit × risk × resource space.
    pub pareto_frontier: Vec<RiskAwareActionValue>,
    /// Actions rejected by hard feasibility/adequacy constraints.
    pub rejected_actions: Vec<RejectedRiskAwareAction>,
    /// Canonical digest binding the complete plan.
    pub plan_sha256: [u8; 32],
}

fn validate_policy(policy: RiskAwareSensingPolicy) -> Result<(), RiskAwareSensingError> {
    if policy.tail_probability_ppm == 0
        || u64::from(policy.tail_probability_ppm) > PROBABILITY_SCALE
    {
        return Err(RiskAwareSensingError::new(
            "policy.tail_probability_ppm",
            "tail probability must lie in 1..=1,000,000 ppm",
        ));
    }
    for (field, value) in [
        ("policy.maximum_posterior_cvar_ppm", policy.maximum_posterior_cvar_ppm),
        ("policy.maximum_worst_case_loss_ppm", policy.maximum_worst_case_loss_ppm),
    ] {
        if value.is_some_and(|value| u64::from(value) > PROBABILITY_SCALE) {
            return Err(RiskAwareSensingError::new(
                field,
                "risk ceiling must lie in [0, 1,000,000] ppm",
            ));
        }
    }
    Ok(())
}

fn digest_is_nonzero(digest: &[u8; 32]) -> bool {
    digest.iter().any(|byte| *byte != 0)
}

fn resource_cost_fits(
    cost: ActiveSensingResourceCost,
    budget: ActiveSensingResourceCost,
) -> bool {
    cost.sensor_energy_nj <= budget.sensor_energy_nj
        && cost.wake_energy_nj <= budget.wake_energy_nj
        && cost.compute_cycles <= budget.compute_cycles
        && cost.latency_us <= budget.latency_us
        && cost.transmit_bytes <= budget.transmit_bytes
        && cost.memory_write_bytes <= budget.memory_write_bytes
}

fn expected_loss_ppm(
    outcomes: &[ObservationOutcomeEstimate],
) -> Result<u32, RiskAwareSensingError> {
    if outcomes.is_empty() || outcomes.len() > MAX_OUTCOMES_PER_ACTION {
        return Err(RiskAwareSensingError::new(
            "action.outcomes",
            format!("outcome count must be in 1..={MAX_OUTCOMES_PER_ACTION}"),
        ));
    }
    let mut probability_sum = 0_u64;
    let mut weighted_loss = 0_u128;
    for (index, outcome) in outcomes.iter().enumerate() {
        if u64::from(outcome.probability_ppm) > PROBABILITY_SCALE
            || u64::from(outcome.posterior_decision_loss_ppm) > PROBABILITY_SCALE
        {
            return Err(RiskAwareSensingError::new(
                format!("action.outcomes[{index}]"),
                "probability and posterior loss must be in [0, 1,000,000] ppm",
            ));
        }
        probability_sum = probability_sum
            .checked_add(u64::from(outcome.probability_ppm))
            .ok_or_else(|| {
                RiskAwareSensingError::new("action.outcomes", "probability sum overflow")
            })?;
        weighted_loss = weighted_loss
            .checked_add(
                u128::from(outcome.probability_ppm)
                    * u128::from(outcome.posterior_decision_loss_ppm),
            )
            .ok_or_else(|| {
                RiskAwareSensingError::new("action.outcomes", "weighted loss overflow")
            })?;
    }
    if probability_sum != PROBABILITY_SCALE {
        return Err(RiskAwareSensingError::new(
            "action.outcomes.probability_ppm",
            format!(
                "outcome probabilities must sum to {PROBABILITY_SCALE}, got {probability_sum}"
            ),
        ));
    }
    let rounded = weighted_loss
        .checked_add(u128::from(PROBABILITY_SCALE / 2))
        .ok_or_else(|| RiskAwareSensingError::new("action.outcomes", "rounding overflow"))?
        / u128::from(PROBABILITY_SCALE);
    u32::try_from(rounded)
        .map_err(|_| RiskAwareSensingError::new("action.outcomes", "expected loss overflow"))
}

/// Compute discrete upper-tail CVaR and worst-case loss for posterior outcomes.
///
/// The tail is formed by sorting positive-probability outcomes from highest to
/// lowest loss and consuming exactly `tail_probability_ppm` probability mass.
/// The final outcome may contribute only a fraction of its probability mass.
pub fn posterior_loss_tail_risk(
    outcomes: &[ObservationOutcomeEstimate],
    tail_probability_ppm: u32,
) -> Result<(u32, u32), RiskAwareSensingError> {
    if tail_probability_ppm == 0 || u64::from(tail_probability_ppm) > PROBABILITY_SCALE {
        return Err(RiskAwareSensingError::new(
            "tail_probability_ppm",
            "tail probability must lie in 1..=1,000,000 ppm",
        ));
    }
    // Reuse the distribution validator so CVaR and expected loss share the same contract.
    let _ = expected_loss_ppm(outcomes)?;
    let mut sorted = outcomes
        .iter()
        .filter(|outcome| outcome.probability_ppm > 0)
        .map(|outcome| (outcome.posterior_decision_loss_ppm, outcome.probability_ppm))
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.0.cmp(&left.0));

    let mut remaining_tail = tail_probability_ppm;
    let mut weighted_tail_loss = 0_u128;
    let mut worst_case = 0_u32;
    for (loss, probability) in sorted {
        worst_case = worst_case.max(loss);
        if remaining_tail == 0 {
            break;
        }
        let consumed = probability.min(remaining_tail);
        weighted_tail_loss += u128::from(consumed) * u128::from(loss);
        remaining_tail -= consumed;
    }
    if remaining_tail != 0 {
        return Err(RiskAwareSensingError::new(
            "outcomes",
            "positive-probability mass is smaller than the requested CVaR tail",
        ));
    }
    let rounded = weighted_tail_loss
        .checked_add(u128::from(tail_probability_ppm / 2))
        .ok_or_else(|| RiskAwareSensingError::new("outcomes", "CVaR rounding overflow"))?
        / u128::from(tail_probability_ppm);
    let cvar = u32::try_from(rounded)
        .map_err(|_| RiskAwareSensingError::new("outcomes", "CVaR overflow"))?;
    Ok((cvar, worst_case))
}

fn componentwise_no_more_than(
    left: ActiveSensingResourceCost,
    right: ActiveSensingResourceCost,
) -> bool {
    left.sensor_energy_nj <= right.sensor_energy_nj
        && left.wake_energy_nj <= right.wake_energy_nj
        && left.compute_cycles <= right.compute_cycles
        && left.latency_us <= right.latency_us
        && left.transmit_bytes <= right.transmit_bytes
        && left.memory_write_bytes <= right.memory_write_bytes
}

fn componentwise_strictly_less(
    left: ActiveSensingResourceCost,
    right: ActiveSensingResourceCost,
) -> bool {
    left.sensor_energy_nj < right.sensor_energy_nj
        || left.wake_energy_nj < right.wake_energy_nj
        || left.compute_cycles < right.compute_cycles
        || left.latency_us < right.latency_us
        || left.transmit_bytes < right.transmit_bytes
        || left.memory_write_bytes < right.memory_write_bytes
}

fn dominates(left: &RiskAwareActionValue, right: &RiskAwareActionValue) -> bool {
    let benefit_no_worse =
        left.expected_task_loss_reduction_ppm >= right.expected_task_loss_reduction_ppm;
    let cvar_no_worse = left.posterior_cvar_ppm <= right.posterior_cvar_ppm;
    let worst_case_no_worse =
        left.posterior_worst_case_loss_ppm <= right.posterior_worst_case_loss_ppm;
    let resources_no_worse = componentwise_no_more_than(left.resource_cost, right.resource_cost);
    let strict = left.expected_task_loss_reduction_ppm > right.expected_task_loss_reduction_ppm
        || left.posterior_cvar_ppm < right.posterior_cvar_ppm
        || left.posterior_worst_case_loss_ppm < right.posterior_worst_case_loss_ppm
        || componentwise_strictly_less(left.resource_cost, right.resource_cost);
    benefit_no_worse && cvar_no_worse && worst_case_no_worse && resources_no_worse && strict
}

fn hash_cost(hasher: &mut CanonicalHasher, cost: ActiveSensingResourceCost) {
    hasher.u64(cost.sensor_energy_nj);
    hasher.u64(cost.wake_energy_nj);
    hasher.u64(cost.compute_cycles);
    hasher.u64(cost.latency_us);
    hasher.u64(cost.transmit_bytes);
    hasher.u64(cost.memory_write_bytes);
}

fn hash_action(hasher: &mut CanonicalHasher, action: &RiskAwareActionValue) {
    hasher.bytes(action.action_id.as_bytes());
    hasher.u8(action.kind as u8);
    hash_cost(hasher, action.resource_cost);
    hasher.bytes(&action.resource_cost_model_sha256);
    hasher.usize(action.outcome_estimates.len());
    for outcome in &action.outcome_estimates {
        hasher.u32(outcome.probability_ppm);
        hasher.u32(outcome.posterior_decision_loss_ppm);
    }
    hasher.u32(action.expected_loss_after_observation_ppm);
    hasher.u64(action.expected_task_loss_reduction_ppm as u64);
    hasher.u32(action.posterior_cvar_ppm);
    hasher.u32(action.posterior_worst_case_loss_ppm);
}

fn compute_plan_digest(receipt: &RiskAwareActiveSensingReceipt) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.risk-aware-active-sensing.v0.1");
    hasher.bytes(&receipt.public_state_sha256);
    hasher.bytes(&receipt.forecast_model_sha256);
    hasher.bytes(&receipt.calibration_receipt_sha256);
    hasher.bytes(&receipt.decision_policy_sha256);
    hasher.u32(receipt.current_expected_decision_loss_ppm);
    hash_cost(&mut hasher, receipt.remaining_resource_budget);
    hasher.u32(receipt.policy.tail_probability_ppm);
    hasher.u32(receipt.policy.maximum_posterior_cvar_ppm.unwrap_or(u32::MAX));
    hasher.u32(receipt.policy.maximum_worst_case_loss_ppm.unwrap_or(u32::MAX));
    hasher.usize(receipt.pareto_frontier.len());
    for action in &receipt.pareto_frontier {
        hash_action(&mut hasher, action);
    }
    hasher.usize(receipt.rejected_actions.len());
    for rejected in &receipt.rejected_actions {
        hasher.bytes(rejected.action_id.as_bytes());
        hasher.u8(rejected.reason as u8);
    }
    hasher.finish()
}

/// Evaluate action estimates and return a risk-aware non-dominated frontier.
///
/// Actions must improve expected task loss, fit every resource budget, and meet
/// any preregistered CVaR/worst-case ceilings. Feasible actions are Pareto-ranked
/// across expected benefit, posterior CVaR, worst-case loss, and all six resources.
pub fn plan_risk_aware_active_sensing_actions(
    context: ActiveSensingDecisionContext,
    policy: RiskAwareSensingPolicy,
    actions: &[ActiveSensingActionEstimate],
) -> Result<RiskAwareActiveSensingReceipt, RiskAwareSensingError> {
    validate_policy(policy)?;
    if actions.len() > MAX_ACTIONS {
        return Err(RiskAwareSensingError::new(
            "actions",
            format!("action count exceeds {MAX_ACTIONS}"),
        ));
    }
    for (field, digest) in [
        ("public_state_sha256", &context.public_state_sha256),
        ("forecast_model_sha256", &context.forecast_model_sha256),
        ("calibration_receipt_sha256", &context.calibration_receipt_sha256),
        ("decision_policy_sha256", &context.decision_policy_sha256),
    ] {
        if !digest_is_nonzero(digest) {
            return Err(RiskAwareSensingError::new(
                field,
                "digest must not be all zero",
            ));
        }
    }
    if u64::from(context.current_expected_decision_loss_ppm) > PROBABILITY_SCALE {
        return Err(RiskAwareSensingError::new(
            "current_expected_decision_loss_ppm",
            "current task loss must lie in [0, 1,000,000] ppm",
        ));
    }

    let mut seen_ids = HashSet::with_capacity(actions.len());
    let mut evaluated = Vec::with_capacity(actions.len());
    let mut rejected_actions = Vec::new();

    for action in actions {
        if action.action_id.trim().is_empty() {
            return Err(RiskAwareSensingError::new("action.action_id", "must not be empty"));
        }
        if !seen_ids.insert(action.action_id.as_str()) {
            return Err(RiskAwareSensingError::new(
                "actions.action_id",
                format!("duplicate action ID: {}", action.action_id),
            ));
        }
        if !digest_is_nonzero(&action.resource_cost_model_sha256) {
            return Err(RiskAwareSensingError::new(
                format!("actions[{}].resource_cost_model_sha256", action.action_id),
                "digest must not be all zero",
            ));
        }
        if !resource_cost_fits(action.resource_cost, context.remaining_resource_budget) {
            rejected_actions.push(RejectedRiskAwareAction {
                action_id: action.action_id.clone(),
                reason: RiskAwareRejectionReason::ExceedsResourceBudget,
            });
            continue;
        }
        let expected_after = expected_loss_ppm(&action.outcomes)?;
        let reduction =
            i64::from(context.current_expected_decision_loss_ppm) - i64::from(expected_after);
        if reduction <= 0 {
            rejected_actions.push(RejectedRiskAwareAction {
                action_id: action.action_id.clone(),
                reason: RiskAwareRejectionReason::NoExpectedTaskLossReduction,
            });
            continue;
        }
        let (cvar, worst_case) =
            posterior_loss_tail_risk(&action.outcomes, policy.tail_probability_ppm)?;
        if policy
            .maximum_posterior_cvar_ppm
            .is_some_and(|ceiling| cvar > ceiling)
        {
            rejected_actions.push(RejectedRiskAwareAction {
                action_id: action.action_id.clone(),
                reason: RiskAwareRejectionReason::ExceedsPosteriorCvarCeiling,
            });
            continue;
        }
        if policy
            .maximum_worst_case_loss_ppm
            .is_some_and(|ceiling| worst_case > ceiling)
        {
            rejected_actions.push(RejectedRiskAwareAction {
                action_id: action.action_id.clone(),
                reason: RiskAwareRejectionReason::ExceedsWorstCaseLossCeiling,
            });
            continue;
        }

        evaluated.push(RiskAwareActionValue {
            action_id: action.action_id.clone(),
            kind: action.kind,
            resource_cost: action.resource_cost,
            resource_cost_model_sha256: action.resource_cost_model_sha256,
            outcome_estimates: action.outcomes.clone(),
            expected_loss_after_observation_ppm: expected_after,
            expected_task_loss_reduction_ppm: reduction,
            posterior_cvar_ppm: cvar,
            posterior_worst_case_loss_ppm: worst_case,
        });
    }

    let mut frontier = evaluated
        .iter()
        .filter(|candidate| !evaluated.iter().any(|other| dominates(other, candidate)))
        .cloned()
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    rejected_actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    let mut receipt = RiskAwareActiveSensingReceipt {
        public_state_sha256: context.public_state_sha256,
        forecast_model_sha256: context.forecast_model_sha256,
        calibration_receipt_sha256: context.calibration_receipt_sha256,
        decision_policy_sha256: context.decision_policy_sha256,
        current_expected_decision_loss_ppm: context.current_expected_decision_loss_ppm,
        remaining_resource_budget: context.remaining_resource_budget,
        policy,
        pareto_frontier: frontier,
        rejected_actions,
        plan_sha256: [0; 32],
    };
    receipt.plan_sha256 = compute_plan_digest(&receipt);
    Ok(receipt)
}

/// Validate an internally consistent, content-addressed risk-aware plan receipt.
pub fn validate_risk_aware_active_sensing_receipt(
    receipt: &RiskAwareActiveSensingReceipt,
) -> Result<(), RiskAwareSensingError> {
    validate_policy(receipt.policy)?;
    for (field, digest) in [
        ("public_state_sha256", &receipt.public_state_sha256),
        ("forecast_model_sha256", &receipt.forecast_model_sha256),
        ("calibration_receipt_sha256", &receipt.calibration_receipt_sha256),
        ("decision_policy_sha256", &receipt.decision_policy_sha256),
    ] {
        if !digest_is_nonzero(digest) {
            return Err(RiskAwareSensingError::new(field, "digest must not be all zero"));
        }
    }
    if u64::from(receipt.current_expected_decision_loss_ppm) > PROBABILITY_SCALE {
        return Err(RiskAwareSensingError::new(
            "current_expected_decision_loss_ppm",
            "current task loss must lie in [0, 1,000,000] ppm",
        ));
    }
    if receipt
        .pareto_frontier
        .windows(2)
        .any(|pair| pair[0].action_id >= pair[1].action_id)
        || receipt
            .rejected_actions
            .windows(2)
            .any(|pair| pair[0].action_id >= pair[1].action_id)
    {
        return Err(RiskAwareSensingError::new(
            "receipt.action_order",
            "frontier and rejected action IDs must be sorted and unique",
        ));
    }
    let mut all_ids = HashSet::new();
    for action in &receipt.pareto_frontier {
        if action.action_id.trim().is_empty() || !all_ids.insert(action.action_id.as_str()) {
            return Err(RiskAwareSensingError::new(
                "receipt.action_id",
                "empty or duplicate action ID",
            ));
        }
        if !digest_is_nonzero(&action.resource_cost_model_sha256)
            || !resource_cost_fits(action.resource_cost, receipt.remaining_resource_budget)
        {
            return Err(RiskAwareSensingError::new(
                format!("receipt.frontier[{}]", action.action_id),
                "resource-cost provenance or budget is invalid",
            ));
        }
        let expected_after = expected_loss_ppm(&action.outcome_estimates)?;
        let reduction =
            i64::from(receipt.current_expected_decision_loss_ppm) - i64::from(expected_after);
        let (cvar, worst_case) =
            posterior_loss_tail_risk(&action.outcome_estimates, receipt.policy.tail_probability_ppm)?;
        if expected_after != action.expected_loss_after_observation_ppm
            || reduction != action.expected_task_loss_reduction_ppm
            || reduction <= 0
            || cvar != action.posterior_cvar_ppm
            || worst_case != action.posterior_worst_case_loss_ppm
        {
            return Err(RiskAwareSensingError::new(
                format!("receipt.frontier[{}]", action.action_id),
                "stored expected-loss or risk values do not reproduce from outcome distribution",
            ));
        }
        if receipt.policy.maximum_posterior_cvar_ppm.is_some_and(|ceiling| cvar > ceiling)
            || receipt
                .policy
                .maximum_worst_case_loss_ppm
                .is_some_and(|ceiling| worst_case > ceiling)
        {
            return Err(RiskAwareSensingError::new(
                format!("receipt.frontier[{}]", action.action_id),
                "accepted action violates a registered risk ceiling",
            ));
        }
    }
    for rejected in &receipt.rejected_actions {
        if rejected.action_id.trim().is_empty() || !all_ids.insert(rejected.action_id.as_str()) {
            return Err(RiskAwareSensingError::new(
                "receipt.rejected_action_id",
                "empty or duplicate action ID",
            ));
        }
    }
    for left in &receipt.pareto_frontier {
        for right in &receipt.pareto_frontier {
            if left.action_id != right.action_id && dominates(left, right) {
                return Err(RiskAwareSensingError::new(
                    "receipt.pareto_frontier",
                    "a frontier action is dominated",
                ));
            }
        }
    }
    if compute_plan_digest(receipt) != receipt.plan_sha256 {
        return Err(RiskAwareSensingError::new(
            "receipt.plan_sha256",
            "canonical receipt digest mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn context() -> ActiveSensingDecisionContext {
        ActiveSensingDecisionContext {
            public_state_sha256: digest(1),
            forecast_model_sha256: digest(2),
            calibration_receipt_sha256: digest(3),
            decision_policy_sha256: digest(4),
            current_expected_decision_loss_ppm: 200_000,
            remaining_resource_budget: ActiveSensingResourceCost {
                sensor_energy_nj: 10_000,
                wake_energy_nj: 10_000,
                compute_cycles: 10_000,
                latency_us: 10_000,
                transmit_bytes: 10_000,
                memory_write_bytes: 10_000,
            },
        }
    }

    fn action(
        id: &str,
        outcomes: Vec<ObservationOutcomeEstimate>,
        sensor_energy_nj: u64,
    ) -> ActiveSensingActionEstimate {
        ActiveSensingActionEstimate {
            action_id: id.to_owned(),
            kind: ActiveSensingActionKind::SampleNow,
            resource_cost: ActiveSensingResourceCost {
                sensor_energy_nj,
                wake_energy_nj: 100,
                compute_cycles: 200,
                latency_us: 50,
                transmit_bytes: 32,
                memory_write_bytes: 16,
            },
            resource_cost_model_sha256: digest(9),
            claimed_entropy_reduction_ppm: 800_000,
            outcomes,
        }
    }

    fn policy() -> RiskAwareSensingPolicy {
        RiskAwareSensingPolicy {
            tail_probability_ppm: 50_000,
            maximum_posterior_cvar_ppm: None,
            maximum_worst_case_loss_ppm: None,
        }
    }

    #[test]
    fn cvar_uses_fractional_probability_mass_from_the_worst_tail() {
        let outcomes = [
            ObservationOutcomeEstimate {
                probability_ppm: 10_000,
                posterior_decision_loss_ppm: 1_000_000,
            },
            ObservationOutcomeEstimate {
                probability_ppm: 40_000,
                posterior_decision_loss_ppm: 500_000,
            },
            ObservationOutcomeEstimate {
                probability_ppm: 950_000,
                posterior_decision_loss_ppm: 100_000,
            },
        ];
        let (cvar, worst) = posterior_loss_tail_risk(&outcomes, 50_000).unwrap();
        assert_eq!(cvar, 600_000);
        assert_eq!(worst, 1_000_000);
    }

    #[test]
    fn risk_ceiling_rejects_rare_catastrophic_outcome_despite_better_mean() {
        let risky = action(
            "rare-catastrophe",
            vec![
                ObservationOutcomeEstimate {
                    probability_ppm: 990_000,
                    posterior_decision_loss_ppm: 100_000,
                },
                ObservationOutcomeEstimate {
                    probability_ppm: 10_000,
                    posterior_decision_loss_ppm: 1_000_000,
                },
            ],
            100,
        );
        let stable = action(
            "stable-moderate",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 150_000,
            }],
            200,
        );
        let mut risk_policy = policy();
        risk_policy.maximum_posterior_cvar_ppm = Some(200_000);
        risk_policy.maximum_worst_case_loss_ppm = Some(300_000);
        let receipt = plan_risk_aware_active_sensing_actions(
            context(),
            risk_policy,
            &[risky, stable],
        )
        .unwrap();
        assert_eq!(receipt.pareto_frontier.len(), 1);
        assert_eq!(receipt.pareto_frontier[0].action_id, "stable-moderate");
        assert_eq!(receipt.rejected_actions.len(), 1);
        assert_eq!(
            receipt.rejected_actions[0].reason,
            RiskAwareRejectionReason::ExceedsPosteriorCvarCeiling
        );
    }

    #[test]
    fn mean_benefit_and_tail_risk_remain_separate_pareto_dimensions() {
        let high_mean_high_tail = action(
            "high-mean-high-tail",
            vec![
                ObservationOutcomeEstimate {
                    probability_ppm: 990_000,
                    posterior_decision_loss_ppm: 50_000,
                },
                ObservationOutcomeEstimate {
                    probability_ppm: 10_000,
                    posterior_decision_loss_ppm: 1_000_000,
                },
            ],
            100,
        );
        let lower_mean_lower_tail = action(
            "lower-mean-lower-tail",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 150_000,
            }],
            200,
        );
        let receipt = plan_risk_aware_active_sensing_actions(
            context(),
            policy(),
            &[high_mean_high_tail, lower_mean_lower_tail],
        )
        .unwrap();
        assert_eq!(receipt.pareto_frontier.len(), 2);
        validate_risk_aware_active_sensing_receipt(&receipt).unwrap();
    }

    #[test]
    fn risk_policy_applies_worst_case_limit_independently_of_cvar() {
        let action = action(
            "bounded-tail-but-high-max",
            vec![
                ObservationOutcomeEstimate {
                    probability_ppm: 999_000,
                    posterior_decision_loss_ppm: 100_000,
                },
                ObservationOutcomeEstimate {
                    probability_ppm: 1_000,
                    posterior_decision_loss_ppm: 900_000,
                },
            ],
            100,
        );
        let mut risk_policy = policy();
        risk_policy.maximum_worst_case_loss_ppm = Some(500_000);
        let receipt =
            plan_risk_aware_active_sensing_actions(context(), risk_policy, &[action]).unwrap();
        assert!(receipt.pareto_frontier.is_empty());
        assert_eq!(
            receipt.rejected_actions[0].reason,
            RiskAwareRejectionReason::ExceedsWorstCaseLossCeiling
        );
    }

    #[test]
    fn componentwise_resource_budgets_are_preserved_with_risk_constraints() {
        let feasible = action(
            "feasible",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 150_000,
            }],
            100,
        );
        let mut expensive = action(
            "too-expensive",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 100_000,
            }],
            100,
        );
        expensive.resource_cost = ActiveSensingResourceCost {
            sensor_energy_nj: 100,
            wake_energy_nj: 100,
            compute_cycles: 100,
            latency_us: 100,
            transmit_bytes: 100,
            memory_write_bytes: 100,
        };
        let mut ctx = context();
        ctx.remaining_resource_budget.compute_cycles = 50;
        let receipt =
            plan_risk_aware_active_sensing_actions(ctx, policy(), &[feasible, expensive]).unwrap();
        assert_eq!(receipt.pareto_frontier.len(), 1);
        assert_eq!(receipt.rejected_actions[0].reason, RiskAwareRejectionReason::ExceedsResourceBudget);
    }

    #[test]
    fn malformed_outcomes_policy_and_receipt_tampering_fail_closed() {
        let malformed = action(
            "bad-probabilities",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 100,
                posterior_decision_loss_ppm: 100_000,
            }],
            100,
        );
        assert!(plan_risk_aware_active_sensing_actions(context(), policy(), &[malformed]).is_err());

        let mut bad_policy = policy();
        bad_policy.tail_probability_ppm = 0;
        assert!(plan_risk_aware_active_sensing_actions(context(), bad_policy, &[]).is_err());

        let good = action(
            "good",
            vec![ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 100_000,
            }],
            100,
        );
        let mut receipt =
            plan_risk_aware_active_sensing_actions(context(), policy(), &[good]).unwrap();
        validate_risk_aware_active_sensing_receipt(&receipt).unwrap();
        receipt.pareto_frontier[0].posterior_cvar_ppm ^= 1;
        assert!(validate_risk_aware_active_sensing_receipt(&receipt).is_err());
    }

    #[test]
    fn zero_probability_outcomes_do_not_inflate_worst_case_or_cvar() {
        let outcomes = [
            ObservationOutcomeEstimate {
                probability_ppm: 1_000_000,
                posterior_decision_loss_ppm: 100_000,
            },
            ObservationOutcomeEstimate {
                probability_ppm: 0,
                posterior_decision_loss_ppm: 1_000_000,
            },
        ];
        let (cvar, worst) = posterior_loss_tail_risk(&outcomes, 10_000).unwrap();
        assert_eq!(cvar, 100_000);
        assert_eq!(worst, 100_000);
    }
}
