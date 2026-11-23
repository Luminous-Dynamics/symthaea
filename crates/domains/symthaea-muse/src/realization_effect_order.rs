//! Counterbalanced arm order and session-carryover qualification for
//! renderer-control effect probes.
//!
//! The four-arm evaluator establishes a target effect against sham, reversal
//! and collateral controls. This companion contract addresses a separate
//! confound: the order in which those arms execute and latent state carried
//! from one arm into the next. It assigns every permutation once per
//! 24-probe block and requires a digest-bound reset/washout receipt before
//! each arm can be scored.

use crate::realization_effect::{
    ControlEffectArmReceiptV1, ControlEffectIssueV1, ControlEffectMetricV1,
    ControlEffectProbeSpecV1, ControlEffectReportV1, ControlProbeArmKindV1,
    evaluate_control_effect_probe_v1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1: u32 = 1;
pub const CONTROL_PROBE_ARM_PERMUTATIONS_V1: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterbalancedOrderPlanPayloadV1 {
    pub schema_version: u32,
    pub probe_id: String,
    pub block_id: String,
    pub block_index: u8,
    pub schedule_seed_sha256: String,
    pub arm_order: [ControlProbeArmKindV1; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterbalancedOrderPlanV1 {
    pub payload: CounterbalancedOrderPlanPayloadV1,
    pub payload_sha256: String,
}

impl CounterbalancedOrderPlanV1 {
    pub fn from_payload(payload: CounterbalancedOrderPlanPayloadV1) -> Result<Self, String> {
        validate_order_plan_payload_v1(&payload)?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(Self {
            payload,
            payload_sha256: sha256_hex(&bytes),
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_order_plan_payload_v1(&self.payload)?;
        let bytes = serde_json::to_vec(&self.payload).map_err(|error| error.to_string())?;
        if self.payload_sha256 != sha256_hex(&bytes) {
            return Err("counterbalanced order-plan digest does not match its payload".to_owned());
        }
        Ok(())
    }
}

fn validate_order_plan_payload_v1(payload: &CounterbalancedOrderPlanPayloadV1) -> Result<(), String> {
    if payload.schema_version != REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1 {
        return Err("unsupported control-effect order-plan schema version".to_owned());
    }
    if payload.probe_id.trim().is_empty() || payload.block_id.trim().is_empty() {
        return Err("order plan requires non-empty probe and block IDs".to_owned());
    }
    if usize::from(payload.block_index) >= CONTROL_PROBE_ARM_PERMUTATIONS_V1 {
        return Err("counterbalanced block index must be in 0..24".to_owned());
    }
    require_sha256(&payload.schedule_seed_sha256, "order schedule seed digest")?;
    let unique: BTreeSet<_> = payload.arm_order.iter().copied().collect();
    if unique.len() != 4 {
        return Err("arm order must contain each arm exactly once".to_owned());
    }
    Ok(())
}

/// Generate a deterministic, hash-ranked randomized order for each probe in
/// a complete 24-probe block. Each of the 24 arm permutations occurs exactly
/// once. Probe IDs are kept in the caller's preregistered block order.
pub fn build_counterbalanced_order_block_v1(
    block_id: &str,
    schedule_seed_sha256: &str,
    probe_ids: &[String],
) -> Result<Vec<CounterbalancedOrderPlanV1>, String> {
    if block_id.trim().is_empty() {
        return Err("counterbalanced block ID must not be empty".to_owned());
    }
    require_sha256(schedule_seed_sha256, "order schedule seed digest")?;
    if probe_ids.len() != CONTROL_PROBE_ARM_PERMUTATIONS_V1 {
        return Err("counterbalanced order block must contain exactly 24 probe IDs".to_owned());
    }
    let unique: BTreeSet<_> = probe_ids.iter().map(String::as_str).collect();
    if unique.len() != probe_ids.len() || probe_ids.iter().any(|id| id.trim().is_empty()) {
        return Err("counterbalanced probe IDs must be non-empty and unique".to_owned());
    }
    let mut permutations = all_arm_permutations_v1();
    let mut ranked = Vec::with_capacity(permutations.len());
    for order in permutations.drain(..) {
        let bytes = serde_json::to_vec(&(block_id, schedule_seed_sha256, order))
            .map_err(|error| error.to_string())?;
        ranked.push((sha256_hex(&bytes), order));
    }
    ranked.sort_by(|left, right| left.0.cmp(&right.0));

    ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, arm_order))| {
            CounterbalancedOrderPlanV1::from_payload(CounterbalancedOrderPlanPayloadV1 {
                schema_version: REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1,
                probe_id: probe_ids[index].clone(),
                block_id: block_id.to_owned(),
                block_index: index as u8,
                schedule_seed_sha256: schedule_seed_sha256.to_owned(),
                arm_order,
            })
        })
        .collect()
}

/// Verify that a collected block contains all 24 unique permutations and that
/// each probe was assigned the deterministic preregistered order.
pub fn validate_counterbalanced_order_plan_assignment_v1(
    plan: &CounterbalancedOrderPlanV1,
) -> Result<(), String> {
    plan.validate()?;
    let probe_ids: Vec<_> = (0..CONTROL_PROBE_ARM_PERMUTATIONS_V1)
        .map(|index| format!("assignment-check-{index:02}"))
        .collect();
    let expected = build_counterbalanced_order_block_v1(
        &plan.payload.block_id,
        &plan.payload.schedule_seed_sha256,
        &probe_ids,
    )?;
    let assigned = expected
        .get(usize::from(plan.payload.block_index))
        .ok_or_else(|| "order plan block index is out of range".to_owned())?;
    if assigned.payload.arm_order != plan.payload.arm_order {
        return Err("order plan is not the preregistered hash-ranked order for its block index".to_owned());
    }
    Ok(())
}

/// Verify that a collected block contains all 24 unique permutations and that
/// each probe was assigned the deterministic preregistered order.
pub fn validate_counterbalanced_order_block_v1(
    plans: &[CounterbalancedOrderPlanV1],
) -> Result<(), String> {
    if plans.len() != CONTROL_PROBE_ARM_PERMUTATIONS_V1 {
        return Err("counterbalanced order block is incomplete".to_owned());
    }
    for plan in plans {
        plan.validate()?;
    }
    let first = &plans[0].payload;
    let mut by_index = vec![None; CONTROL_PROBE_ARM_PERMUTATIONS_V1];
    for plan in plans {
        if plan.payload.block_id != first.block_id
            || plan.payload.schedule_seed_sha256 != first.schedule_seed_sha256
        {
            return Err("counterbalanced block mixes block IDs or schedule seeds".to_owned());
        }
        let slot = &mut by_index[usize::from(plan.payload.block_index)];
        if slot.replace(plan).is_some() {
            return Err("counterbalanced block contains a duplicate block index".to_owned());
        }
    }
    let ordered: Vec<CounterbalancedOrderPlanV1> = by_index
        .into_iter()
        .map(|plan| {
            plan.cloned()
                .ok_or_else(|| "counterbalanced block index is missing".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let probe_ids: Vec<_> = ordered
        .iter()
        .map(|plan: &CounterbalancedOrderPlanV1| plan.payload.probe_id.clone())
        .collect();
    let expected = build_counterbalanced_order_block_v1(
        &first.block_id,
        &first.schedule_seed_sha256,
        &probe_ids,
    )?;
    if expected != ordered {
        return Err("counterbalanced block does not match its preregistered hash-ranked schedule".to_owned());
    }
    let permutations: BTreeSet<_> = ordered
        .iter()
        .map(|plan| plan.payload.arm_order)
        .collect();
    if permutations.len() != CONTROL_PROBE_ARM_PERMUTATIONS_V1 {
        return Err("counterbalanced block does not contain all 24 arm permutations".to_owned());
    }
    Ok(())
}

fn all_arm_permutations_v1() -> Vec<[ControlProbeArmKindV1; 4]> {
    fn visit(
        prefix: &mut Vec<ControlProbeArmKindV1>,
        remaining: &mut Vec<ControlProbeArmKindV1>,
        output: &mut Vec<[ControlProbeArmKindV1; 4]>,
    ) {
        if remaining.is_empty() {
            if let Ok(order) = <[ControlProbeArmKindV1; 4]>::try_from(prefix.as_slice()) {
                output.push(order);
            }
            return;
        }
        for index in 0..remaining.len() {
            let next = remaining.remove(index);
            prefix.push(next);
            visit(prefix, remaining, output);
            prefix.pop();
            remaining.insert(index, next);
        }
    }

    let mut prefix = Vec::with_capacity(4);
    let mut remaining = vec![
        ControlProbeArmKindV1::Baseline,
        ControlProbeArmKindV1::Sham,
        ControlProbeArmKindV1::Active,
        ControlProbeArmKindV1::Reversal,
    ];
    let mut output = Vec::with_capacity(CONTROL_PROBE_ARM_PERMUTATIONS_V1);
    visit(&mut prefix, &mut remaining, &mut output);
    output
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionResetPolicyV1 {
    pub require_explicit_reset_before_each_arm: bool,
    pub minimum_washout_duration_ms: u64,
    pub maximum_pre_arm_baseline_drift_milli: u64,
    pub minimum_state_epoch_increment: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionTransitionReceiptPayloadV1 {
    pub schema_version: u32,
    pub session_id: String,
    pub from_arm: Option<ControlProbeArmKindV1>,
    pub to_arm: ControlProbeArmKindV1,
    pub from_state_epoch: Option<u64>,
    pub to_state_epoch: u64,
    pub previous_output_artifact_sha256: Option<String>,
    pub reset_applied: bool,
    pub reset_event_sha256: Option<String>,
    pub reset_observer_id: Option<String>,
    pub reset_observer_identity_sha256: Option<String>,
    pub washout_duration_ms: u64,
    pub pre_arm_target_metric: ControlEffectMetricV1,
    pub pre_arm_target_value_milli: i64,
    pub baseline_observation_sha256: String,
    pub baseline_observer_id: String,
    pub baseline_observer_identity_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionTransitionReceiptV1 {
    pub payload: SessionTransitionReceiptPayloadV1,
    pub payload_sha256: String,
}

impl SessionTransitionReceiptV1 {
    pub fn from_payload(payload: SessionTransitionReceiptPayloadV1) -> Result<Self, String> {
        validate_transition_payload_v1(&payload)?;
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(Self {
            payload,
            payload_sha256: sha256_hex(&bytes),
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_transition_payload_v1(&self.payload)?;
        let bytes = serde_json::to_vec(&self.payload).map_err(|error| error.to_string())?;
        if self.payload_sha256 != sha256_hex(&bytes) {
            return Err("session transition receipt digest does not match its payload".to_owned());
        }
        Ok(())
    }
}

fn validate_transition_payload_v1(payload: &SessionTransitionReceiptPayloadV1) -> Result<(), String> {
    if payload.schema_version != REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1 {
        return Err("unsupported session transition receipt schema version".to_owned());
    }
    if payload.session_id.trim().is_empty()
        || payload.baseline_observer_id.trim().is_empty()
    {
        return Err("session transition requires session and baseline observer IDs".to_owned());
    }
    require_sha256(&payload.baseline_observation_sha256, "pre-arm baseline observation digest")?;
    require_sha256(
        &payload.baseline_observer_identity_sha256,
        "baseline observer identity digest",
    )?;
    match (
        payload.from_arm,
        payload.from_state_epoch,
        payload.previous_output_artifact_sha256.as_deref(),
    ) {
        (None, None, None) => {}
        (Some(_), Some(_), Some(digest)) => {
            require_sha256(digest, "previous output artifact digest")?;
        }
        _ => return Err("transition predecessor arm, epoch and output digest must be bound together".to_owned()),
    }
    if payload.reset_applied {
        if payload.reset_event_sha256.is_none()
            || payload.reset_observer_id.as_deref().unwrap_or("").trim().is_empty()
            || payload.reset_observer_identity_sha256.is_none()
        {
            return Err("applied session reset requires an event digest and independent reset observer identity".to_owned());
        }
        require_sha256(payload.reset_event_sha256.as_deref().unwrap_or(""), "session reset event digest")?;
        require_sha256(
            payload.reset_observer_identity_sha256.as_deref().unwrap_or(""),
            "session reset observer identity digest",
        )?;
    } else if payload.reset_event_sha256.is_some()
        || payload.reset_observer_id.is_some()
        || payload.reset_observer_identity_sha256.is_some()
    {
        return Err("non-applied reset must not carry a reset event or observer receipt".to_owned());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderedControlEffectProbeSpecV1 {
    pub schema_version: u32,
    pub base_probe_spec: ControlEffectProbeSpecV1,
    pub order_plan: CounterbalancedOrderPlanV1,
    pub reset_policy: SessionResetPolicyV1,
    pub expected_pre_arm_target_metric_value_milli: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderedControlEffectArmReceiptV1 {
    pub base_arm_receipt: ControlEffectArmReceiptV1,
    pub arm_sequence_index: u8,
    pub session_state_epoch: u64,
    pub transition_receipt: SessionTransitionReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectOrderIssueV1 {
    InvalidSpec,
    InvalidOrderPlan(String),
    WrongArmCount { expected: usize, found: usize },
    ArmOrderMismatch {
        index: u8,
        expected: ControlProbeArmKindV1,
        found: ControlProbeArmKindV1,
    },
    SequenceIndexMismatch { expected: u8, found: u8 },
    TransitionReceiptInvalid { arm: ControlProbeArmKindV1, detail: String },
    TransitionTargetMismatch { arm: ControlProbeArmKindV1 },
    SessionIdentityMismatch { arm: ControlProbeArmKindV1 },
    PreviousArmMismatch { arm: ControlProbeArmKindV1 },
    PreviousOutputMismatch { arm: ControlProbeArmKindV1 },
    StateEpochMismatch { arm: ControlProbeArmKindV1 },
    StateEpochNotIncreasing { arm: ControlProbeArmKindV1 },
    ExplicitResetMissing { arm: ControlProbeArmKindV1 },
    WashoutTooShort { arm: ControlProbeArmKindV1 },
    PreArmCarryoverDetected {
        arm: ControlProbeArmKindV1,
        drift_milli: u128,
        maximum_drift_milli: u64,
    },
    BaselineObserverIsProducer { arm: ControlProbeArmKindV1 },
    ResetObserverIsProducer { arm: ControlProbeArmKindV1 },
    BaseProbeEvaluation(Vec<ControlEffectIssueV1>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderedControlEffectReportV1 {
    pub schema_version: u32,
    pub probe_id: String,
    pub order_plan_sha256: String,
    pub session_id: String,
    pub arm_order: [ControlProbeArmKindV1; 4],
    pub state_epochs: [u64; 4],
    pub transition_receipt_sha256: [String; 4],
    pub maximum_pre_arm_baseline_drift_milli: u128,
    pub effect_report: ControlEffectReportV1,
}

pub fn evaluate_ordered_control_effect_probe_v1(
    spec: &OrderedControlEffectProbeSpecV1,
    arms: &[OrderedControlEffectArmReceiptV1],
) -> Result<OrderedControlEffectReportV1, Vec<ControlEffectOrderIssueV1>> {
    if spec.schema_version != REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1
        || spec.base_probe_spec.probe_id != spec.order_plan.payload.probe_id
        || spec.reset_policy.minimum_state_epoch_increment == 0
    {
        return Err(vec![ControlEffectOrderIssueV1::InvalidSpec]);
    }
    if let Err(error) = validate_counterbalanced_order_plan_assignment_v1(&spec.order_plan) {
        return Err(vec![ControlEffectOrderIssueV1::InvalidOrderPlan(error)]);
    }
    if arms.len() != 4 {
        return Err(vec![ControlEffectOrderIssueV1::WrongArmCount {
            expected: 4,
            found: arms.len(),
        }]);
    }

    let mut issues = Vec::new();
    let mut max_drift = 0u128;
    let expected_baseline = spec.expected_pre_arm_target_metric_value_milli;
    let mut session_id: Option<&str> = None;
    let mut state_epochs = [0u64; 4];
    let mut transition_digests: [String; 4] = std::array::from_fn(|_| String::new());

    for (index, arm) in arms.iter().enumerate() {
        let expected_arm = spec.order_plan.payload.arm_order[index];
        if arm.base_arm_receipt.arm != expected_arm {
            issues.push(ControlEffectOrderIssueV1::ArmOrderMismatch {
                index: index as u8,
                expected: expected_arm,
                found: arm.base_arm_receipt.arm,
            });
        }
        if usize::from(arm.arm_sequence_index) != index {
            issues.push(ControlEffectOrderIssueV1::SequenceIndexMismatch {
                expected: index as u8,
                found: arm.arm_sequence_index,
            });
        }
        let transition = &arm.transition_receipt;
        if let Err(error) = transition.validate() {
            issues.push(ControlEffectOrderIssueV1::TransitionReceiptInvalid {
                arm: arm.base_arm_receipt.arm,
                detail: error,
            });
            continue;
        }
        let receipt = &transition.payload;
        if receipt.to_arm != arm.base_arm_receipt.arm
            || receipt.pre_arm_target_metric != spec.base_probe_spec.target_metric
        {
            issues.push(ControlEffectOrderIssueV1::TransitionTargetMismatch {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if receipt.session_id != arm.base_arm_receipt.probe_session_id {
            issues.push(ControlEffectOrderIssueV1::SessionIdentityMismatch {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if let Some(expected_session) = session_id {
            if receipt.session_id != expected_session {
                issues.push(ControlEffectOrderIssueV1::SessionIdentityMismatch {
                    arm: arm.base_arm_receipt.arm,
                });
            }
        } else {
            session_id = Some(&receipt.session_id);
        }
        if receipt.to_state_epoch != arm.session_state_epoch {
            issues.push(ControlEffectOrderIssueV1::StateEpochMismatch {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if spec.reset_policy.require_explicit_reset_before_each_arm && !receipt.reset_applied {
            issues.push(ControlEffectOrderIssueV1::ExplicitResetMissing {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if receipt.washout_duration_ms < spec.reset_policy.minimum_washout_duration_ms {
            issues.push(ControlEffectOrderIssueV1::WashoutTooShort {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if receipt.baseline_observer_id == arm.base_arm_receipt.producer_id {
            issues.push(ControlEffectOrderIssueV1::BaselineObserverIsProducer {
                arm: arm.base_arm_receipt.arm,
            });
        }
        if receipt.reset_applied
            && receipt.reset_observer_id.as_deref()
                == Some(arm.base_arm_receipt.producer_id.as_str())
        {
            issues.push(ControlEffectOrderIssueV1::ResetObserverIsProducer {
                arm: arm.base_arm_receipt.arm,
            });
        }
        let drift = abs_diff(receipt.pre_arm_target_value_milli, expected_baseline);
        max_drift = max_drift.max(drift);
        if drift > u128::from(spec.reset_policy.maximum_pre_arm_baseline_drift_milli) {
            issues.push(ControlEffectOrderIssueV1::PreArmCarryoverDetected {
                arm: arm.base_arm_receipt.arm,
                drift_milli: drift,
                maximum_drift_milli: spec.reset_policy.maximum_pre_arm_baseline_drift_milli,
            });
        }
        state_epochs[index] = receipt.to_state_epoch;
        transition_digests[index] = transition.payload_sha256.clone();

        if index == 0 {
            if receipt.from_arm.is_some()
                || receipt.from_state_epoch.is_some()
                || receipt.previous_output_artifact_sha256.is_some()
            {
                issues.push(ControlEffectOrderIssueV1::PreviousArmMismatch {
                    arm: arm.base_arm_receipt.arm,
                });
            }
        } else {
            let previous = &arms[index - 1];
            if receipt.from_arm != Some(previous.base_arm_receipt.arm)
                || receipt.from_state_epoch != Some(previous.session_state_epoch)
            {
                issues.push(ControlEffectOrderIssueV1::PreviousArmMismatch {
                    arm: arm.base_arm_receipt.arm,
                });
            }
            if receipt.previous_output_artifact_sha256.as_deref()
                != Some(previous.base_arm_receipt.output_artifact_sha256.as_str())
            {
                issues.push(ControlEffectOrderIssueV1::PreviousOutputMismatch {
                    arm: arm.base_arm_receipt.arm,
                });
            }
            let minimum_epoch = previous
                .session_state_epoch
                .saturating_add(spec.reset_policy.minimum_state_epoch_increment);
            if receipt.to_state_epoch < minimum_epoch {
                issues.push(ControlEffectOrderIssueV1::StateEpochNotIncreasing {
                    arm: arm.base_arm_receipt.arm,
                });
            }
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    let base_arms: Vec<_> = arms
        .iter()
        .map(|arm| arm.base_arm_receipt.clone())
        .collect();
    let effect_report = evaluate_control_effect_probe_v1(&spec.base_probe_spec, &base_arms)
        .map_err(|error| vec![ControlEffectOrderIssueV1::BaseProbeEvaluation(error)])?;
    let session_id = session_id.unwrap_or_default().to_owned();
    Ok(OrderedControlEffectReportV1 {
        schema_version: REALIZATION_EFFECT_ORDER_SCHEMA_VERSION_V1,
        probe_id: spec.base_probe_spec.probe_id.clone(),
        order_plan_sha256: spec.order_plan.payload_sha256.clone(),
        session_id,
        arm_order: spec.order_plan.payload.arm_order,
        state_epochs,
        transition_receipt_sha256: transition_digests,
        maximum_pre_arm_baseline_drift_milli: max_drift,
        effect_report,
    })
}

fn abs_diff(left: i64, right: i64) -> u128 {
    (i128::from(left) - i128::from(right)).unsigned_abs()
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !valid_sha256(value) {
        return Err(format!("{label} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realization_effect::{
        CollateralMetricV1, ControlApplicationV1, ControlEffectDirectionV1,
        ControlEffectMetricV1, ControlEffectProbeSpecV1,
        ControlObservationEligibilityV1,
    };
    use crate::realization_contract::RealizationControlV1;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn effect_spec(probe_id: &str) -> ControlEffectProbeSpecV1 {
        ControlEffectProbeSpecV1 {
            schema_version: 1,
            probe_id: probe_id.to_owned(),
            control: RealizationControlV1::Tempo,
            target_metric: ControlEffectMetricV1::TempoMilliBpm,
            expected_direction: ControlEffectDirectionV1::Increase,
            baseline_control_value_milli: 100_000,
            treatment_control_value_milli: 120_000,
            minimum_effect_milli: 10_000,
            sham_equivalence_margin_milli: 2_000,
            reversal_return_margin_milli: 2_000,
            maximum_collateral_drift_milli: 5_000,
        }
    }

    fn plans() -> Vec<CounterbalancedOrderPlanV1> {
        let probe_ids: Vec<_> = (0..24).map(|index| format!("probe-{index:02}")).collect();
        build_counterbalanced_order_block_v1("block-a", &digest(1), &probe_ids).unwrap()
    }

    fn base_arm(kind: ControlProbeArmKindV1, observed: i64, collateral: i64, output: u8) -> ControlEffectArmReceiptV1 {
        let (application, requested, accepted, effective) = match kind {
            ControlProbeArmKindV1::Baseline => (
                ControlApplicationV1::NoIntervention,
                None,
                false,
                100_000,
            ),
            ControlProbeArmKindV1::Sham => (
                ControlApplicationV1::ShamNoop,
                None,
                false,
                100_000,
            ),
            ControlProbeArmKindV1::Active => (
                ControlApplicationV1::Applied,
                Some(120_000),
                true,
                120_000,
            ),
            ControlProbeArmKindV1::Reversal => (
                ControlApplicationV1::Applied,
                Some(100_000),
                true,
                100_000,
            ),
        };
        ControlEffectArmReceiptV1 {
            arm_id: format!("arm-{kind:?}"),
            probe_session_id: "session-order-1".to_owned(),
            arm: kind,
            application,
            backend_id: "stream-backend".to_owned(),
            backend_version: "v1".to_owned(),
            adapter_version: "a1".to_owned(),
            capability_manifest_sha256: digest(2),
            source_artifact_sha256: digest(3),
            session_seed: 42,
            requested_control_value_milli: requested,
            reported_effective_control_value_milli: Some(effective),
            accepted_by_backend: accepted,
            ignored_by_backend: false,
            target_metric: ControlEffectMetricV1::TempoMilliBpm,
            observed_target_value_milli: observed,
            collateral_metrics: vec![CollateralMetricV1 {
                metric: ControlEffectMetricV1::IntegratedLoudnessMilliLufs,
                value_milli: collateral,
            }],
            output_artifact_sha256: digest(output),
            realization_receipt_sha256: digest(4),
            capture_payload_sha256: Some(digest(5)),
            capture_closure_receipt_sha256: Some(digest(6)),
            event_observer_manifest_sha256: Some(digest(7)),
            capture_observer_id: "loopback".to_owned(),
            event_observer_id: "detector".to_owned(),
            producer_id: "backend".to_owned(),
            observation_eligibility: ControlObservationEligibilityV1::StructurallyEligible,
        }
    }

    fn ordered_arms(plan: &CounterbalancedOrderPlanV1) -> Vec<OrderedControlEffectArmReceiptV1> {
        let observed_values = [
            (100_000, -16_000, 10),
            (101_000, -16_100, 11),
            (120_000, -17_000, 12),
            (101_000, -16_200, 13),
        ];
        let mut prior: Option<(ControlProbeArmKindV1, u64, String)> = None;
        plan.payload.arm_order
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let (observed, collateral, output) = match kind {
                    ControlProbeArmKindV1::Baseline => observed_values[0],
                    ControlProbeArmKindV1::Sham => observed_values[1],
                    ControlProbeArmKindV1::Active => observed_values[2],
                    ControlProbeArmKindV1::Reversal => observed_values[3],
                };
                let base = base_arm(*kind, observed, collateral, output);
                let epoch = 100 + index as u64;
                let previous = prior.clone();
                let transition = SessionTransitionReceiptV1::from_payload(
                    SessionTransitionReceiptPayloadV1 {
                        schema_version: 1,
                        session_id: base.probe_session_id.clone(),
                        from_arm: previous.as_ref().map(|value| value.0),
                        to_arm: *kind,
                        from_state_epoch: previous.as_ref().map(|value| value.1),
                        to_state_epoch: epoch,
                        previous_output_artifact_sha256: previous.map(|value| value.2),
                        reset_applied: true,
                        reset_event_sha256: Some(digest(8 + index as u8)),
                        reset_observer_id: Some("reset-observer".to_owned()),
                        reset_observer_identity_sha256: Some(digest(9)),
                        washout_duration_ms: 2_000,
                        pre_arm_target_metric: ControlEffectMetricV1::TempoMilliBpm,
                        pre_arm_target_value_milli: 100_000,
                        baseline_observation_sha256: digest(10 + index as u8),
                        baseline_observer_id: "baseline-observer".to_owned(),
                        baseline_observer_identity_sha256: digest(11),
                    },
                ).unwrap();
                prior = Some((*kind, epoch, base.output_artifact_sha256.clone()));
                OrderedControlEffectArmReceiptV1 {
                    base_arm_receipt: base,
                    arm_sequence_index: index as u8,
                    session_state_epoch: epoch,
                    transition_receipt: transition,
                }
            })
            .collect()
    }

    fn ordered_spec(plan: CounterbalancedOrderPlanV1) -> OrderedControlEffectProbeSpecV1 {
        OrderedControlEffectProbeSpecV1 {
            schema_version: 1,
            base_probe_spec: effect_spec(&plan.payload.probe_id),
            order_plan: plan,
            reset_policy: SessionResetPolicyV1 {
                require_explicit_reset_before_each_arm: true,
                minimum_washout_duration_ms: 1_000,
                maximum_pre_arm_baseline_drift_milli: 1_500,
                minimum_state_epoch_increment: 1,
            },
            expected_pre_arm_target_metric_value_milli: 100_000,
        }
    }

    #[test]
    fn counterbalanced_order_block_contains_every_permutation_exactly_once() {
        let first = plans();
        let second = plans();
        assert_eq!(first, second);
        assert_eq!(first.len(), 24);
        validate_counterbalanced_order_block_v1(&first).unwrap();
        let orders: BTreeSet<_> = first.iter().map(|plan| plan.payload.arm_order).collect();
        assert_eq!(orders.len(), 24);
    }

    #[test]
    fn missing_or_duplicated_order_block_member_is_rejected() {
        let mut missing = plans();
        missing.pop();
        assert!(validate_counterbalanced_order_block_v1(&missing).is_err());

        let mut duplicated = plans();
        duplicated[1] = duplicated[0].clone();
        assert!(validate_counterbalanced_order_block_v1(&duplicated).is_err());
    }

    #[test]
    fn ordered_probe_qualifies_with_counterbalanced_order_and_reset_receipts() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let arms = ordered_arms(&plan);
        let report = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap();
        assert_eq!(
            report.effect_report.verdict,
            crate::realization_effect::ControlEffectVerdictV1::QualifiedObservedEffect
        );
        assert_eq!(report.arm_order, plan.payload.arm_order);
        assert_eq!(report.maximum_pre_arm_baseline_drift_milli, 0);
        assert_eq!(report.state_epochs, [100, 101, 102, 103]);
    }

    #[test]
    fn arm_sequence_must_match_preregistered_permutation() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms.swap(0, 1);
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ControlEffectOrderIssueV1::ArmOrderMismatch { .. }
                | ControlEffectOrderIssueV1::SequenceIndexMismatch { .. }
        )));
    }

    #[test]
    fn explicit_reset_is_required_when_declared_by_policy() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms[2].transition_receipt.payload.reset_applied = false;
        arms[2].transition_receipt.payload.reset_event_sha256 = None;
        arms[2].transition_receipt.payload.reset_observer_id = None;
        arms[2].transition_receipt.payload.reset_observer_identity_sha256 = None;
        let payload = arms[2].transition_receipt.payload.clone();
        arms[2].transition_receipt = SessionTransitionReceiptV1::from_payload(payload).unwrap();
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.contains(&ControlEffectOrderIssueV1::ExplicitResetMissing {
            arm: arms[2].base_arm_receipt.arm
        }));
    }

    #[test]
    fn too_short_washout_is_not_scored_as_a_clean_probe() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms[1].transition_receipt.payload.washout_duration_ms = 10;
        let payload = arms[1].transition_receipt.payload.clone();
        arms[1].transition_receipt = SessionTransitionReceiptV1::from_payload(payload).unwrap();
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.contains(&ControlEffectOrderIssueV1::WashoutTooShort {
            arm: arms[1].base_arm_receipt.arm
        }));
    }

    #[test]
    fn pre_arm_baseline_drift_detects_session_carryover() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms[3].transition_receipt.payload.pre_arm_target_value_milli = 103_000;
        let payload = arms[3].transition_receipt.payload.clone();
        arms[3].transition_receipt = SessionTransitionReceiptV1::from_payload(payload).unwrap();
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ControlEffectOrderIssueV1::PreArmCarryoverDetected {
                arm: ControlProbeArmKindV1::Reversal,
                ..
            }
        )));
    }

    #[test]
    fn transition_chain_must_bind_previous_arm_epoch_and_exact_output() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms[2].transition_receipt.payload.previous_output_artifact_sha256 = Some(digest(55));
        let payload = arms[2].transition_receipt.payload.clone();
        arms[2].transition_receipt = SessionTransitionReceiptV1::from_payload(payload).unwrap();
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.contains(&ControlEffectOrderIssueV1::PreviousOutputMismatch {
            arm: arms[2].base_arm_receipt.arm
        }));
    }

    #[test]
    fn reset_observer_must_not_be_the_output_producer() {
        let plan = plans().into_iter().next().unwrap();
        let spec = ordered_spec(plan.clone());
        let mut arms = ordered_arms(&plan);
        arms[1].transition_receipt.payload.reset_observer_id = Some("backend".to_owned());
        let payload = arms[1].transition_receipt.payload.clone();
        arms[1].transition_receipt = SessionTransitionReceiptV1::from_payload(payload).unwrap();
        let issues = evaluate_ordered_control_effect_probe_v1(&spec, &arms).unwrap_err();
        assert!(issues.contains(&ControlEffectOrderIssueV1::ResetObserverIsProducer {
            arm: arms[1].base_arm_receipt.arm
        }));
    }
}
