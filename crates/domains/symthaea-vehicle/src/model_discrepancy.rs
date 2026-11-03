// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline horizon-conditioned model discrepancy and planner-exploitation audit.
//! Positive signed residual is predicted_value - realized_value. Interpretation
//! depends on the named metric. Decision utilities are higher-is-better.
//! This module owns neither a predictor nor a simulator; caller-supplied sequence
//! numbers are audit assertions, not cryptographic proof of outcome blinding.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};

pub const VEHICLE_MODEL_DISCREPANCY_SCHEMA: &str = "symthaea.vehicle.model-discrepancy.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscrepancyFeasibility { Feasible, Infeasible, Indeterminate }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceOutcomeStatus { Observed, Missing, Censored, TimedOut, Unsupported }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAuditStatus { Comparable, IncompleteReference, IndeterminateFeasibility, SelectedActionInfeasible, NoFeasibleAction }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResidualObservation {
    pub scenario_id: String,
    pub failure_family_id: String,
    pub actor_policy_id: String,
    pub split_id: String,
    pub shift_regime_id: String,
    pub initial_state_digest: String,
    pub environment_digest: String,
    pub dynamics_digest: String,
    pub action_history_digest: String,
    pub observation_availability_digest: String,
    pub forecast_issued_at_ms: u64,
    pub horizon_ms: u64,
    pub metric_id: String,
    pub metric_unit_id: String,
    /// Present when the predictor emitted a forecast at this horizon.
    pub predicted_value: Option<f64>,
    /// Present only when the reference outcome is fully observed.
    pub realized_value: Option<f64>,
    /// Present only for a fully observed reference outcome.
    pub prediction_set_contains_realized: Option<bool>,
    /// Every expected scenario × actor × action-history × metric × horizon slot
    /// must appear, including missing/censored/timed-out/unsupported outcomes.
    pub reference_status: ReferenceOutcomeStatus,
}

/// A frozen schedule slot that is expected to produce exactly one residual outcome row.
/// The schedule is supplied by the benchmark harness before reference execution and is
/// retained independently of the observed/missing outcome data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedResidualSlot {
    pub scenario_id: String,
    pub failure_family_id: String,
    pub actor_policy_id: String,
    pub split_id: String,
    pub shift_regime_id: String,
    pub initial_state_digest: String,
    pub environment_digest: String,
    pub dynamics_digest: String,
    pub action_history_digest: String,
    pub observation_availability_digest: String,
    pub forecast_issued_at_ms: u64,
    pub horizon_ms: u64,
    pub metric_id: String,
    pub metric_unit_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscrepancyActionOutcome {
    pub action_id: String,
    pub command_digest: String,
    pub feasibility: DiscrepancyFeasibility,
    pub reference_status: ReferenceOutcomeStatus,
    pub predicted_utility: f64,
    /// Present only when a feasible action has a complete reference outcome.
    pub realized_utility: Option<f64>,
    pub execution_receipt_id: Option<String>,
    pub execution_digest: Option<String>,
    pub evaluator_receipt_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDecisionObservation {
    pub decision_id: String,
    pub scenario_id: String,
    pub failure_family_id: String,
    pub actor_policy_id: String,
    pub split_id: String,
    pub shift_regime_id: String,
    pub initial_state_digest: String,
    pub environment_digest: String,
    pub dynamics_digest: String,
    pub action_history_digest: String,
    pub observation_availability_digest: String,
    pub forecast_issued_at_ms: u64,
    pub decision_context_digest: String,
    pub action_set_digest: String,
    pub utility_id: String,
    pub utility_unit_id: String,
    pub tie_tolerance: f64,
    pub selected_action_id: String,
    pub candidate_set_complete: bool,
    pub selection_commit_sequence: u64,
    pub reference_outcome_reveal_sequence: u64,
    pub selection_receipt_id: String,
    pub actions: Vec<DiscrepancyActionOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VehicleModelDiscrepancyInput {
    pub predictor_model_id: String,
    pub predictor_artifact_digest: String,
    pub reference_model_id: String,
    pub reference_model_artifact_digest: String,
    pub evaluator_id: String,
    pub evaluator_artifact_digest: String,
    pub planner_id: String,
    pub planner_artifact_digest: String,
    pub sampling_profile_digest: String,
    pub protocol_id: String,
    pub protocol_digest: String,
    /// Frozen expected slots; each must have exactly one typed outcome row below.
    pub expected_residual_slots: Vec<ExpectedResidualSlot>,
    pub residual_observations: Vec<ModelResidualObservation>,
    pub decisions: Vec<ModelDecisionObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VehicleModelDiscrepancyError {
    EmptyIdentity, InvalidDigest, EmptyResidualSet, InvalidHorizon, InvalidMetric,
    InvalidResidual, DuplicateResidualKey, MismatchedResidualContext,
    EmptyResidualSchedule, DuplicateExpectedResidualSlot, MissingResidualOutcomeSlot,
    UnexpectedResidualOutcomeSlot,
    InvalidDecisionIdentity, InvalidDecisionContextDigest, InvalidActionSetDigest,
    InvalidTieTolerance, TooFewActions, DuplicateDecisionId, DuplicateActionId,
    DuplicateReceiptId, MissingSelectedAction, OutcomeLeakage, InvalidUtility,
    InvalidActionReceipt, InconsistentReferenceOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonResidualSummary {
    pub horizon_ms: u64,
    /// Number of expected outcome rows supplied for this horizon.
    pub outcome_count: usize,
    pub observed_count: usize,
    pub missing_count: usize,
    pub censored_count: usize,
    pub timed_out_count: usize,
    pub unsupported_count: usize,
    pub failure_family_count: usize,
    pub mean_signed_residual: Option<f64>,
    pub median_signed_residual: Option<f64>,
    pub median_absolute_residual: Option<f64>,
    pub p90_absolute_residual: Option<f64>,
    pub p95_absolute_residual: Option<f64>,
    pub worst_family_p95_absolute_residual: Option<f64>,
    pub prediction_set_membership_count: usize,
    pub prediction_set_covered_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyHorizonResidualSummary {
    pub horizon_ms: u64,
    pub failure_family_id: String,
    /// Number of expected outcome rows supplied for this family/horizon.
    pub outcome_count: usize,
    pub observed_count: usize,
    pub missing_count: usize,
    pub censored_count: usize,
    pub timed_out_count: usize,
    pub unsupported_count: usize,
    pub mean_signed_residual: Option<f64>,
    pub p95_absolute_residual: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedHorizonResidualGrowth {
    pub scenario_id: String,
    pub actor_policy_id: String,
    pub action_history_digest: String,
    pub metric_id: String,
    pub from_horizon_ms: u64,
    pub to_horizon_ms: u64,
    pub from_signed_residual: f64,
    pub to_signed_residual: f64,
    pub absolute_residual_growth: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionAuditResult {
    pub decision_id: String,
    pub scenario_id: String,
    pub failure_family_id: String,
    pub selected_action_id: String,
    pub status: DecisionAuditStatus,
    pub candidate_set_complete: bool,
    pub predicted_best_action_id: Option<String>,
    pub realized_best_action_id: Option<String>,
    pub ranking_reversed: Option<bool>,
    pub selected_action_regret: Option<f64>,
    pub optimism_gap: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionAuditSummary {
    pub decision_count: usize,
    pub comparable_decision_count: usize,
    pub incomplete_reference_decision_count: usize,
    pub indeterminate_decision_count: usize,
    pub selected_action_infeasible_count: usize,
    pub no_feasible_action_count: usize,
    pub ranking_reversal_count: usize,
    pub ranking_reversal_rate: Option<f64>,
    pub mean_selected_action_regret: Option<f64>,
    pub p95_selected_action_regret: Option<f64>,
    pub mean_optimism_gap: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VehicleModelDiscrepancyReport {
    pub schema_version: String,
    pub input: VehicleModelDiscrepancyInput,
    pub horizon_summaries: Vec<HorizonResidualSummary>,
    pub family_horizon_summaries: Vec<FamilyHorizonResidualSummary>,
    pub paired_horizon_growth: Vec<PairedHorizonResidualGrowth>,
    pub decision_results: Vec<DecisionAuditResult>,
    pub decision_summary: DecisionAuditSummary,
    pub report_digest: String,
}

impl VehicleModelDiscrepancyReport {
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != VEHICLE_MODEL_DISCREPANCY_SCHEMA || !valid_digest(&self.report_digest) { return false; }
        let Ok(expected) = derive_report(&self.input) else { return false; };
        let mut actual = self.clone();
        actual.report_digest.clear();
        let (Ok(a), Ok(e)) = (serde_json::to_vec(&actual), serde_json::to_vec(&expected)) else { return false; };
        a == e && blake3::hash(&a).to_hex().to_string() == self.report_digest
    }
}

/// Compute the canonical digest of a finite action set from sorted action IDs and command digests.
/// This binds the declared candidate payload identities; it does not authenticate the commands.
pub fn model_discrepancy_action_set_digest(
    actions: &[DiscrepancyActionOutcome],
) -> Result<String, VehicleModelDiscrepancyError> {
    if actions.len() < 2 {
        return Err(VehicleModelDiscrepancyError::TooFewActions);
    }
    let mut pairs = actions
        .iter()
        .map(|action| (action.action_id.as_str(), action.command_digest.as_str()))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.cmp(right.0));
    if pairs.iter().any(|(id, digest)| id.trim().is_empty() || !valid_digest(digest)) {
        return Err(VehicleModelDiscrepancyError::InvalidActionSetDigest);
    }
    if pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(VehicleModelDiscrepancyError::DuplicateActionId);
    }
    hash_json(&pairs)
}

/// Compute the context digest required by a decision observation.
/// Call this after setting the decision fields, including action_set_digest.
pub fn model_discrepancy_decision_context_digest(
    decision: &ModelDecisionObservation,
) -> Result<String, VehicleModelDiscrepancyError> {
    decision_context_digest(decision)
}

pub fn summarize_vehicle_model_discrepancy(
    input: &VehicleModelDiscrepancyInput,
) -> Result<VehicleModelDiscrepancyReport, VehicleModelDiscrepancyError> {
    let mut report = derive_report(input)?;
    let bytes = serde_json::to_vec(&report).map_err(|_| VehicleModelDiscrepancyError::InvalidResidual)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

fn derive_report(input: &VehicleModelDiscrepancyInput) -> Result<VehicleModelDiscrepancyReport, VehicleModelDiscrepancyError> {
    if [input.predictor_model_id.as_str(), input.reference_model_id.as_str(), input.evaluator_id.as_str(), input.planner_id.as_str(), input.protocol_id.as_str()].iter().any(|s| s.trim().is_empty()) {
        return Err(VehicleModelDiscrepancyError::EmptyIdentity);
    }
    if [input.predictor_artifact_digest.as_str(), input.reference_model_artifact_digest.as_str(), input.evaluator_artifact_digest.as_str(), input.planner_artifact_digest.as_str(), input.sampling_profile_digest.as_str(), input.protocol_digest.as_str()].iter().any(|d| !valid_digest(d)) {
        return Err(VehicleModelDiscrepancyError::InvalidDigest);
    }
    if input.residual_observations.is_empty() { return Err(VehicleModelDiscrepancyError::EmptyResidualSet); }
    if input.expected_residual_slots.is_empty() {
        return Err(VehicleModelDiscrepancyError::EmptyResidualSchedule);
    }

    let mut canonical = input.clone();
    canonical.expected_residual_slots.sort_by(|a,b| a.scenario_id.cmp(&b.scenario_id)
        .then_with(|| a.actor_policy_id.cmp(&b.actor_policy_id))
        .then_with(|| a.action_history_digest.cmp(&b.action_history_digest))
        .then_with(|| a.metric_id.cmp(&b.metric_id)).then_with(|| a.horizon_ms.cmp(&b.horizon_ms)));
    canonical.residual_observations.sort_by(|a,b| a.scenario_id.cmp(&b.scenario_id)
        .then_with(|| a.actor_policy_id.cmp(&b.actor_policy_id))
        .then_with(|| a.action_history_digest.cmp(&b.action_history_digest))
        .then_with(|| a.metric_id.cmp(&b.metric_id)).then_with(|| a.horizon_ms.cmp(&b.horizon_ms)));
    canonical.decisions.sort_by(|a,b| a.decision_id.cmp(&b.decision_id));
    for d in &mut canonical.decisions { d.actions.sort_by(|a,b| a.action_id.cmp(&b.action_id)); }

    let mut expected_contexts = BTreeMap::<(String,String,String,String,u64), String>::new();
    for slot in &canonical.expected_residual_slots {
        if [
            slot.scenario_id.as_str(), slot.failure_family_id.as_str(),
            slot.actor_policy_id.as_str(), slot.split_id.as_str(), slot.shift_regime_id.as_str(),
            slot.metric_id.as_str(), slot.metric_unit_id.as_str(),
        ].iter().any(|value| value.trim().is_empty()) {
            return Err(VehicleModelDiscrepancyError::InvalidMetric);
        }
        if slot.horizon_ms == 0 {
            return Err(VehicleModelDiscrepancyError::InvalidHorizon);
        }
        if [
            slot.initial_state_digest.as_str(), slot.environment_digest.as_str(),
            slot.dynamics_digest.as_str(), slot.action_history_digest.as_str(),
            slot.observation_availability_digest.as_str(),
        ].iter().any(|digest| !valid_digest(digest)) {
            return Err(VehicleModelDiscrepancyError::InvalidDigest);
        }
        let key = (
            slot.scenario_id.clone(), slot.actor_policy_id.clone(),
            slot.action_history_digest.clone(), slot.metric_id.clone(), slot.horizon_ms,
        );
        if expected_contexts.insert(key, expected_slot_context_digest(slot)?).is_some() {
            return Err(VehicleModelDiscrepancyError::DuplicateExpectedResidualSlot);
        }
    }

    let mut contexts = BTreeMap::<(String,String,String,String), String>::new();
    let mut residual_keys = BTreeSet::new();
    for row in &canonical.residual_observations {
        if [row.scenario_id.as_str(),row.failure_family_id.as_str(),row.actor_policy_id.as_str(),row.split_id.as_str(),row.shift_regime_id.as_str(),row.metric_id.as_str(),row.metric_unit_id.as_str()].iter().any(|s| s.trim().is_empty()) {
            return Err(VehicleModelDiscrepancyError::InvalidMetric);
        }
        if row.horizon_ms == 0 { return Err(VehicleModelDiscrepancyError::InvalidHorizon); }
        if [row.initial_state_digest.as_str(),row.environment_digest.as_str(),row.dynamics_digest.as_str(),row.action_history_digest.as_str(),row.observation_availability_digest.as_str()].iter().any(|d| !valid_digest(d)) {
            return Err(VehicleModelDiscrepancyError::InvalidDigest);
        }
        match row.reference_status {
            ReferenceOutcomeStatus::Observed => {
                let (Some(predicted), Some(realized)) = (row.predicted_value, row.realized_value)
                else { return Err(VehicleModelDiscrepancyError::InvalidResidual); };
                if !predicted.is_finite() || !realized.is_finite()
                    || !(predicted - realized).is_finite()
                {
                    return Err(VehicleModelDiscrepancyError::InvalidResidual);
                }
            }
            ReferenceOutcomeStatus::Missing | ReferenceOutcomeStatus::Censored
            | ReferenceOutcomeStatus::TimedOut | ReferenceOutcomeStatus::Unsupported => {
                if row.realized_value.is_some() || row.prediction_set_contains_realized.is_some()
                    || row.predicted_value.is_some_and(|value| !value.is_finite())
                {
                    return Err(VehicleModelDiscrepancyError::InvalidResidual);
                }
            }
        }
        let key = (row.scenario_id.clone(),row.actor_policy_id.clone(),row.action_history_digest.clone(),row.metric_id.clone());
        let ctx = residual_context_digest(row)?;
        if let Some(old) = contexts.get(&key) {
            if old != &ctx { return Err(VehicleModelDiscrepancyError::MismatchedResidualContext); }
        } else { contexts.insert(key.clone(),ctx); }
        let slot_key = (key.0.clone(), key.1.clone(), key.2.clone(), key.3.clone(), row.horizon_ms);
        let expected_context = expected_contexts
            .get(&slot_key)
            .ok_or(VehicleModelDiscrepancyError::UnexpectedResidualOutcomeSlot)?;
        if expected_context != &ctx {
            return Err(VehicleModelDiscrepancyError::MismatchedResidualContext);
        }
        if !residual_keys.insert((key.0,key.1,key.2,key.3,row.horizon_ms)) {
            return Err(VehicleModelDiscrepancyError::DuplicateResidualKey);
        }
    }
    if expected_contexts.keys().any(|key| !residual_keys.contains(key)) {
        return Err(VehicleModelDiscrepancyError::MissingResidualOutcomeSlot);
    }

    let mut horizons = BTreeMap::<u64,Vec<&ModelResidualObservation>>::new();
    let mut families = BTreeMap::<(u64,String),Vec<&ModelResidualObservation>>::new();
    let mut paired = BTreeMap::<(String,String,String,String),Vec<&ModelResidualObservation>>::new();
    for row in &canonical.residual_observations {
        horizons.entry(row.horizon_ms).or_default().push(row);
        families.entry((row.horizon_ms,row.failure_family_id.clone())).or_default().push(row);
        paired.entry((row.scenario_id.clone(),row.actor_policy_id.clone(),row.action_history_digest.clone(),row.metric_id.clone())).or_default().push(row);
    }

    let family_horizon_summaries = families.iter().map(|((h,f),rows)| {
        let observed = rows.iter().filter(|r| r.reference_status == ReferenceOutcomeStatus::Observed).collect::<Vec<_>>();
        let signed = observed.iter().map(|r| r.predicted_value.expect("observed prediction") - r.realized_value.expect("observed reference")).collect::<Vec<_>>();
        let absolute = signed.iter().map(|v| v.abs()).collect::<Vec<_>>();
        FamilyHorizonResidualSummary {
            horizon_ms:*h,
            failure_family_id:f.clone(),
            outcome_count:rows.len(),
            observed_count:status_count(rows, ReferenceOutcomeStatus::Observed),
            missing_count:status_count(rows, ReferenceOutcomeStatus::Missing),
            censored_count:status_count(rows, ReferenceOutcomeStatus::Censored),
            timed_out_count:status_count(rows, ReferenceOutcomeStatus::TimedOut),
            unsupported_count:status_count(rows, ReferenceOutcomeStatus::Unsupported),
            mean_signed_residual:optional_mean(&signed),
            p95_absolute_residual:optional_quantile(&absolute,0.95),
        }
    }).collect::<Vec<_>>();

    let mut horizon_summaries = Vec::new();
    for (h,rows) in horizons {
        let observed = rows.iter().filter(|r| r.reference_status == ReferenceOutcomeStatus::Observed).collect::<Vec<_>>();
        let signed = observed.iter().map(|r| r.predicted_value.expect("observed prediction") - r.realized_value.expect("observed reference")).collect::<Vec<_>>();
        let absolute = signed.iter().map(|v| v.abs()).collect::<Vec<_>>();
        let family_p95 = family_horizon_summaries.iter()
            .filter(|family| family.horizon_ms == h)
            .filter_map(|family| family.p95_absolute_residual)
            .collect::<Vec<_>>();
        let worst = family_p95.iter().copied().reduce(f64::max);
        let memberships = observed.iter().filter(|r| r.prediction_set_contains_realized.is_some()).collect::<Vec<_>>();
        let covered = memberships.iter().filter(|r| r.prediction_set_contains_realized==Some(true)).count();
        horizon_summaries.push(HorizonResidualSummary {
            horizon_ms:h,
            outcome_count:rows.len(),
            observed_count:status_count(&rows, ReferenceOutcomeStatus::Observed),
            missing_count:status_count(&rows, ReferenceOutcomeStatus::Missing),
            censored_count:status_count(&rows, ReferenceOutcomeStatus::Censored),
            timed_out_count:status_count(&rows, ReferenceOutcomeStatus::TimedOut),
            unsupported_count:status_count(&rows, ReferenceOutcomeStatus::Unsupported),
            failure_family_count:rows.iter().map(|r|r.failure_family_id.as_str()).collect::<BTreeSet<_>>().len(),
            mean_signed_residual:optional_mean(&signed),
            median_signed_residual:optional_quantile(&signed,0.5),
            median_absolute_residual:optional_quantile(&absolute,0.5),
            p90_absolute_residual:optional_quantile(&absolute,0.9),
            p95_absolute_residual:optional_quantile(&absolute,0.95),
            worst_family_p95_absolute_residual:worst,
            prediction_set_membership_count:memberships.len(),
            prediction_set_covered_count:covered
        });
    }

    let mut paired_horizon_growth = Vec::new();
    for ((scenario,actor,action,metric),rows) in paired {
        let mut ordered=rows; ordered.sort_by_key(|r|r.horizon_ms);
        for w in ordered.windows(2) {
            // Do not bridge an explicitly missing/censored/timed-out horizon.
            if w[0].reference_status != ReferenceOutcomeStatus::Observed
                || w[1].reference_status != ReferenceOutcomeStatus::Observed
            {
                continue;
            }
            let from=w[0].predicted_value.expect("observed prediction")-w[0].realized_value.expect("observed reference");
            let to=w[1].predicted_value.expect("observed prediction")-w[1].realized_value.expect("observed reference");
            paired_horizon_growth.push(PairedHorizonResidualGrowth {
                scenario_id:scenario.clone(),actor_policy_id:actor.clone(),action_history_digest:action.clone(),metric_id:metric.clone(),
                from_horizon_ms:w[0].horizon_ms,to_horizon_ms:w[1].horizon_ms,from_signed_residual:from,to_signed_residual:to,
                absolute_residual_growth:to.abs()-from.abs()
            });
        }
    }
    paired_horizon_growth.sort_by(|a,b|a.scenario_id.cmp(&b.scenario_id).then_with(||a.actor_policy_id.cmp(&b.actor_policy_id))
        .then_with(||a.action_history_digest.cmp(&b.action_history_digest)).then_with(||a.metric_id.cmp(&b.metric_id)).then_with(||a.from_horizon_ms.cmp(&b.from_horizon_ms)));

    let mut decision_ids=BTreeSet::new(); let mut receipt_ids=BTreeSet::new(); let mut decision_results=Vec::new();
    for d in &canonical.decisions {
        if [d.decision_id.as_str(),d.scenario_id.as_str(),d.failure_family_id.as_str(),d.actor_policy_id.as_str(),d.split_id.as_str(),d.shift_regime_id.as_str(),d.utility_id.as_str(),d.utility_unit_id.as_str(),d.selected_action_id.as_str(),d.selection_receipt_id.as_str()].iter().any(|s|s.trim().is_empty()) {
            return Err(VehicleModelDiscrepancyError::InvalidDecisionIdentity);
        }
        if !decision_ids.insert(d.decision_id.as_str()) || !receipt_ids.insert(d.selection_receipt_id.as_str()) { return Err(VehicleModelDiscrepancyError::DuplicateDecisionId); }
        if [d.initial_state_digest.as_str(),d.environment_digest.as_str(),d.dynamics_digest.as_str(),d.action_history_digest.as_str(),d.observation_availability_digest.as_str(),d.action_set_digest.as_str()].iter().any(|x|!valid_digest(x)) { return Err(VehicleModelDiscrepancyError::InvalidDigest); }
        if !d.tie_tolerance.is_finite() || d.tie_tolerance<0.0 { return Err(VehicleModelDiscrepancyError::InvalidTieTolerance); }
        if d.selection_commit_sequence>=d.reference_outcome_reveal_sequence { return Err(VehicleModelDiscrepancyError::OutcomeLeakage); }
        if decision_context_digest(d)? != d.decision_context_digest { return Err(VehicleModelDiscrepancyError::InvalidDecisionContextDigest); }
        if d.actions.len()<2 { return Err(VehicleModelDiscrepancyError::TooFewActions); }
        let mut action_ids=BTreeSet::new(); let mut action_pairs=Vec::new();
        let predicted_min = d.actions.iter().map(|a| a.predicted_utility).fold(f64::INFINITY, f64::min);
        let predicted_max = d.actions.iter().map(|a| a.predicted_utility).fold(f64::NEG_INFINITY, f64::max);
        let realized_values = d.actions.iter().filter_map(|a| a.realized_utility).collect::<Vec<_>>();
        let realized_min = realized_values.iter().copied().fold(f64::INFINITY, f64::min);
        let realized_max = realized_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let predicted_range = predicted_max - predicted_min;
        let realized_range = if realized_values.is_empty() { 0.0 } else { realized_max - realized_min };
        if !predicted_range.is_finite() || !realized_range.is_finite()
            || !(predicted_range + realized_range).is_finite()
        {
            return Err(VehicleModelDiscrepancyError::InvalidUtility);
        }
        for a in &d.actions {
            if a.action_id.trim().is_empty() { return Err(VehicleModelDiscrepancyError::InvalidActionReceipt); }
            if !action_ids.insert(a.action_id.as_str()) { return Err(VehicleModelDiscrepancyError::DuplicateActionId); }
            if !valid_digest(&a.command_digest) || !a.predicted_utility.is_finite() {
                return Err(VehicleModelDiscrepancyError::InvalidUtility);
            }
            match a.reference_status {
                ReferenceOutcomeStatus::Observed => {
                    let (Some(exec_receipt), Some(exec_digest), Some(eval_receipt)) =
                        (a.execution_receipt_id.as_deref(), a.execution_digest.as_deref(), a.evaluator_receipt_id.as_deref())
                    else { return Err(VehicleModelDiscrepancyError::InvalidActionReceipt); };
                    if exec_receipt.trim().is_empty() || eval_receipt.trim().is_empty() {
                        return Err(VehicleModelDiscrepancyError::InvalidActionReceipt);
                    }
                    if !valid_digest(exec_digest) { return Err(VehicleModelDiscrepancyError::InvalidDigest); }
                    if !receipt_ids.insert(exec_receipt) || !receipt_ids.insert(eval_receipt) {
                        return Err(VehicleModelDiscrepancyError::DuplicateReceiptId);
                    }
                    match (a.feasibility, a.realized_utility) {
                        (DiscrepancyFeasibility::Feasible, Some(value)) if value.is_finite() => {}
                        (DiscrepancyFeasibility::Infeasible, None) => {}
                        (DiscrepancyFeasibility::Indeterminate, None) => {}
                        _ => return Err(VehicleModelDiscrepancyError::InconsistentReferenceOutcome),
                    }
                }
                ReferenceOutcomeStatus::Missing | ReferenceOutcomeStatus::Censored
                | ReferenceOutcomeStatus::TimedOut | ReferenceOutcomeStatus::Unsupported => {
                    if a.feasibility != DiscrepancyFeasibility::Indeterminate
                        || a.realized_utility.is_some()
                        || a.execution_receipt_id.is_some()
                        || a.execution_digest.is_some()
                        || a.evaluator_receipt_id.is_some()
                    {
                        return Err(VehicleModelDiscrepancyError::InconsistentReferenceOutcome);
                    }
                }
            }
            action_pairs.push((a.action_id.as_str(),a.command_digest.as_str()));
        }
        if !action_ids.contains(d.selected_action_id.as_str()) { return Err(VehicleModelDiscrepancyError::MissingSelectedAction); }
        if hash_json(&action_pairs)? != d.action_set_digest { return Err(VehicleModelDiscrepancyError::InvalidActionSetDigest); }
        decision_results.push(evaluate_decision(d));
    }
    let decision_summary=summarize_decisions(&decision_results);
    Ok(VehicleModelDiscrepancyReport { schema_version:VEHICLE_MODEL_DISCREPANCY_SCHEMA.into(),input:canonical,horizon_summaries,family_horizon_summaries,paired_horizon_growth,decision_results,decision_summary,report_digest:String::new() })
}

fn evaluate_decision(d:&ModelDecisionObservation)->DecisionAuditResult {
    let selected=d.actions.iter().find(|a|a.action_id==d.selected_action_id).expect("validated selected action");
    let base=|status,predicted_best,realized_best,reversed,regret,optimism| DecisionAuditResult {
        decision_id:d.decision_id.clone(),scenario_id:d.scenario_id.clone(),failure_family_id:d.failure_family_id.clone(),selected_action_id:d.selected_action_id.clone(),
        status,candidate_set_complete:d.candidate_set_complete,predicted_best_action_id:predicted_best,realized_best_action_id:realized_best,ranking_reversed:reversed,selected_action_regret:regret,optimism_gap:optimism
    };
    if d.actions.iter().any(|a|a.reference_status!=ReferenceOutcomeStatus::Observed) {
        return base(DecisionAuditStatus::IncompleteReference,None,None,None,None,None);
    }
    if d.actions.iter().any(|a|a.feasibility==DiscrepancyFeasibility::Indeterminate) { return base(DecisionAuditStatus::IndeterminateFeasibility,None,None,None,None,None); }
    let feasible=d.actions.iter().filter(|a|a.feasibility==DiscrepancyFeasibility::Feasible).collect::<Vec<_>>();
    if feasible.is_empty() { return base(DecisionAuditStatus::NoFeasibleAction,None,None,None,None,None); }
    let p=best_action(&feasible,|a|a.predicted_utility,d.tie_tolerance);
    let r=best_action(&feasible,|a|a.realized_utility.expect("feasible observed action has a realized utility"),d.tie_tolerance);
    if selected.feasibility!=DiscrepancyFeasibility::Feasible { return base(DecisionAuditStatus::SelectedActionInfeasible,Some(p.action_id.clone()),Some(r.action_id.clone()),Some(p.action_id!=r.action_id),None,None); }
    let selected_realized=selected.realized_utility.expect("feasible observed selected action has a realized utility");
    let realized_best=r.realized_utility.expect("feasible observed best action has a realized utility");
    let regret=(realized_best-selected_realized).max(0.0);
    let alternatives=feasible.iter().copied().filter(|a|a.action_id!=selected.action_id).collect::<Vec<_>>();
    let optimism=if alternatives.is_empty(){None}else{
        let predicted_alternative=best_action(&alternatives,|a|a.predicted_utility,d.tie_tolerance);
        let realized_alternative=best_action(&alternatives,|a|a.realized_utility.expect("feasible observed alternative has a realized utility"),d.tie_tolerance);
        let predicted_advantage=selected.predicted_utility-predicted_alternative.predicted_utility;
        let realized_advantage=selected_realized-realized_alternative.realized_utility.expect("feasible observed alternative has a realized utility");
        Some(predicted_advantage-realized_advantage)
    };
    base(DecisionAuditStatus::Comparable,Some(p.action_id.clone()),Some(r.action_id.clone()),Some(p.action_id!=r.action_id),Some(regret),optimism)
}

fn best_action<'a,F:Fn(&DiscrepancyActionOutcome)->f64>(actions:&[&'a DiscrepancyActionOutcome],value:F,tol:f64)->&'a DiscrepancyActionOutcome {
    let mut best=actions[0];
    for candidate in actions.iter().skip(1) {
        let delta=value(candidate)-value(best);
        if delta>tol || (delta.abs()<=tol && candidate.action_id.as_str()<best.action_id.as_str()) { best=*candidate; }
    }
    best
}

fn summarize_decisions(rows:&[DecisionAuditResult])->DecisionAuditSummary {
    let comparable=rows.iter().filter(|r|r.status==DecisionAuditStatus::Comparable).collect::<Vec<_>>();
    let regrets=comparable.iter().filter_map(|r|r.selected_action_regret).collect::<Vec<_>>();
    let optimism=comparable.iter().filter_map(|r|r.optimism_gap).collect::<Vec<_>>();
    let reversals=comparable.iter().filter(|r|r.ranking_reversed==Some(true)).count();
    DecisionAuditSummary {
        decision_count:rows.len(),comparable_decision_count:comparable.len(),
        incomplete_reference_decision_count:rows.iter().filter(|r|r.status==DecisionAuditStatus::IncompleteReference).count(),
        indeterminate_decision_count:rows.iter().filter(|r|r.status==DecisionAuditStatus::IndeterminateFeasibility).count(),
        selected_action_infeasible_count:rows.iter().filter(|r|r.status==DecisionAuditStatus::SelectedActionInfeasible).count(),
        no_feasible_action_count:rows.iter().filter(|r|r.status==DecisionAuditStatus::NoFeasibleAction).count(),
        ranking_reversal_count:reversals,ranking_reversal_rate:if comparable.is_empty(){None}else{Some(reversals as f64/comparable.len() as f64)},
        mean_selected_action_regret:optional_mean(&regrets),p95_selected_action_regret:optional_quantile(&regrets,0.95),mean_optimism_gap:optional_mean(&optimism)
    }
}

fn expected_slot_context_digest(slot:&ExpectedResidualSlot)->Result<String,VehicleModelDiscrepancyError> {
    hash_json(&(
        slot.failure_family_id.as_str(), slot.actor_policy_id.as_str(),
        slot.split_id.as_str(), slot.shift_regime_id.as_str(),
        slot.initial_state_digest.as_str(), slot.environment_digest.as_str(),
        slot.dynamics_digest.as_str(), slot.action_history_digest.as_str(),
        slot.observation_availability_digest.as_str(), slot.forecast_issued_at_ms,
        slot.metric_id.as_str(), slot.metric_unit_id.as_str(),
    ))
}
fn residual_context_digest(r:&ModelResidualObservation)->Result<String,VehicleModelDiscrepancyError> {
    hash_json(&(r.failure_family_id.as_str(),r.actor_policy_id.as_str(),r.split_id.as_str(),r.shift_regime_id.as_str(),
        r.initial_state_digest.as_str(),r.environment_digest.as_str(),r.dynamics_digest.as_str(),r.action_history_digest.as_str(),
        r.observation_availability_digest.as_str(),r.forecast_issued_at_ms,r.metric_id.as_str(),r.metric_unit_id.as_str()))
}
fn decision_context_digest(d:&ModelDecisionObservation)->Result<String,VehicleModelDiscrepancyError> {
    hash_json(&(d.decision_id.as_str(),d.scenario_id.as_str(),d.failure_family_id.as_str(),d.actor_policy_id.as_str(),d.split_id.as_str(),d.shift_regime_id.as_str(),
        d.initial_state_digest.as_str(),d.environment_digest.as_str(),d.dynamics_digest.as_str(),d.action_history_digest.as_str(),d.observation_availability_digest.as_str(),
        d.forecast_issued_at_ms,d.utility_id.as_str(),d.utility_unit_id.as_str(),d.tie_tolerance,d.selected_action_id.as_str(),d.selection_commit_sequence,d.reference_outcome_reveal_sequence,d.selection_receipt_id.as_str(),d.action_set_digest.as_str(),d.candidate_set_complete))
}
fn hash_json<T:Serialize>(value:&T)->Result<String,VehicleModelDiscrepancyError> {
    let bytes=serde_json::to_vec(value).map_err(|_|VehicleModelDiscrepancyError::InvalidDecisionContextDigest)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}
fn valid_digest(value:&str)->bool { value.len()==64 && value.bytes().all(|b|b.is_ascii_hexdigit()) }
fn status_count(rows: &[&ModelResidualObservation], status: ReferenceOutcomeStatus) -> usize {
    rows.iter().filter(|row| row.reference_status == status).count()
}

fn mean(values:&[f64])->f64 {
    values.iter().enumerate().fold(0.0, |average, (index, value)| {
        let count = (index + 1) as f64;
        if index == 0 { *value } else {
            average * ((count - 1.0) / count) + *value / count
        }
    })
}
fn quantile(values:&[f64],p:f64)->f64 { let mut v=values.to_vec();v.sort_by(f64::total_cmp);let rank=((v.len() as f64)*p).ceil() as usize;v[rank.saturating_sub(1).min(v.len()-1)] }
fn optional_mean(v:&[f64])->Option<f64>{if v.is_empty(){None}else{Some(mean(v))}}
fn optional_quantile(v:&[f64],p:f64)->Option<f64>{if v.is_empty(){None}else{Some(quantile(v,p))}}

#[cfg(test)]
mod tests {
    use super::*;
    fn h(c:char)->String{c.to_string().repeat(64)}
    fn residual(t:u64,p:f64,r:f64)->ModelResidualObservation{ModelResidualObservation{
        scenario_id:"scenario-1".into(),failure_family_id:"crossing".into(),actor_policy_id:"ped-yield-v1".into(),split_id:"heldout".into(),shift_regime_id:"nominal".into(),
        initial_state_digest:h('a'),environment_digest:h('b'),dynamics_digest:h('c'),action_history_digest:h('d'),observation_availability_digest:h('e'),
        forecast_issued_at_ms:1000,horizon_ms:t,metric_id:"position-error".into(),metric_unit_id:"meters".into(),predicted_value:Some(p),realized_value:Some(r),prediction_set_contains_realized:None,reference_status:ReferenceOutcomeStatus::Observed}}
    fn incomplete_residual(t:u64,status:ReferenceOutcomeStatus)->ModelResidualObservation {
        let mut row=residual(t,0.0,0.0);
        row.reference_status=status;
        row.realized_value=None;
        row.prediction_set_contains_realized=None;
        row
    }
    fn decision()->ModelDecisionObservation{
        let mut d=ModelDecisionObservation{
            decision_id:"decision-1".into(),scenario_id:"scenario-1".into(),failure_family_id:"crossing".into(),actor_policy_id:"ped-yield-v1".into(),split_id:"heldout".into(),shift_regime_id:"nominal".into(),
            initial_state_digest:h('a'),environment_digest:h('b'),dynamics_digest:h('c'),action_history_digest:h('d'),observation_availability_digest:h('e'),forecast_issued_at_ms:1000,
            decision_context_digest:String::new(),action_set_digest:String::new(),utility_id:"progress-risk".into(),utility_unit_id:"utility".into(),tie_tolerance:1e-9,selected_action_id:"yield".into(),
            candidate_set_complete:true,selection_commit_sequence:10,reference_outcome_reveal_sequence:11,selection_receipt_id:"select-1".into(),actions:vec![
                DiscrepancyActionOutcome{action_id:"yield".into(),command_digest:h('f'),feasibility:DiscrepancyFeasibility::Feasible,reference_status:ReferenceOutcomeStatus::Observed,predicted_utility:10.0,realized_utility:Some(2.0),execution_receipt_id:Some("exec-y".into()),execution_digest:Some(h('1')),evaluator_receipt_id:Some("eval-y".into())},
                DiscrepancyActionOutcome{action_id:"continue".into(),command_digest:h('0'),feasibility:DiscrepancyFeasibility::Feasible,reference_status:ReferenceOutcomeStatus::Observed,predicted_utility:7.0,realized_utility:Some(8.0),execution_receipt_id:Some("exec-c".into()),execution_digest:Some(h('2')),evaluator_receipt_id:Some("eval-c".into())}
            ]};
        d.actions.sort_by(|a,b|a.action_id.cmp(&b.action_id));
        d.action_set_digest=model_discrepancy_action_set_digest(&d.actions).unwrap();
        d.decision_context_digest=model_discrepancy_decision_context_digest(&d).unwrap();d
    }
    fn input()->VehicleModelDiscrepancyInput{VehicleModelDiscrepancyInput{
        predictor_model_id:"predictor-v0".into(),predictor_artifact_digest:h('3'),reference_model_id:"reference-v0".into(),reference_model_artifact_digest:h('4'),
        evaluator_id:"evaluator-v0".into(),evaluator_artifact_digest:h('5'),planner_id:"planner-v0".into(),planner_artifact_digest:h('6'),sampling_profile_digest:h('7'),
        protocol_id:"SAM-MODEL-RESIDUAL-001".into(),protocol_digest:h('8'),residual_observations:vec![residual(1000,1.0,0.0),residual(2000,4.0,0.0)],decisions:vec![decision()]}}
    #[test]
    fn analytic_linear_motion_fixture_has_known_horizon_growth() {
        let mut sample = input();
        let speed_prediction = 5.0_f64;
        let speed_reference = 5.2_f64;
        sample.residual_observations = [1_000_u64, 2_000_u64].into_iter().map(|horizon_ms| {
            let seconds = horizon_ms as f64 / 1_000.0;
            residual(
                horizon_ms,
                speed_prediction * seconds,
                speed_reference * seconds,
            )
        }).collect();
        let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
        assert!((report.paired_horizon_growth[0].absolute_residual_growth - 0.2).abs() < 1e-12);
    }

    #[test]
    fn perfect_model_fixture_has_zero_residual_at_every_horizon() {
        let mut sample = input();
        sample.residual_observations = vec![residual(1_000, 5.0, 5.0), residual(2_000, 10.0, 10.0)];
        let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
        assert_eq!(report.horizon_summaries[0].p95_absolute_residual, Some(0.0));
        assert_eq!(report.horizon_summaries[1].p95_absolute_residual, Some(0.0));
        assert_eq!(report.paired_horizon_growth[0].absolute_residual_growth, 0.0);
    }

    #[test]
    fn preserves_signed_bias_direction() {
        let mut sample = input();
        sample.residual_observations = vec![residual(1_000, 3.0, 4.0)];
        let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
        assert_eq!(report.horizon_summaries[0].mean_signed_residual, Some(-1.0));
    }

    #[test]
    fn rejects_overflowing_signed_residual() {
        let mut sample = input();
        sample.residual_observations[0].predicted_value = Some(f64::MAX);
        sample.residual_observations[0].realized_value = Some(-f64::MAX);
        assert_eq!(
            summarize_vehicle_model_discrepancy(&sample).unwrap_err(),
            VehicleModelDiscrepancyError::InvalidResidual
        );
    }

    #[test]
    fn rejects_overflowing_utility_range() {
        let mut sample = input();
        sample.decisions[0].actions[0].predicted_utility = f64::MAX;
        sample.decisions[0].actions[1].predicted_utility = -f64::MAX;
        assert_eq!(
            summarize_vehicle_model_discrepancy(&sample).unwrap_err(),
            VehicleModelDiscrepancyError::InvalidUtility
        );
    }

    #[test]
    fn horizon_denominators_retain_each_incomplete_outcome_status() {
        let mut sample = input();
        sample.residual_observations = vec![
            residual(1_000, 2.0, 1.0),
            incomplete_residual(2_000, ReferenceOutcomeStatus::Missing),
            incomplete_residual(3_000, ReferenceOutcomeStatus::Censored),
            incomplete_residual(4_000, ReferenceOutcomeStatus::TimedOut),
            incomplete_residual(5_000, ReferenceOutcomeStatus::Unsupported),
        ];
        let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
        assert_eq!(report.horizon_summaries.len(), 5);
        assert_eq!(report.horizon_summaries[0].outcome_count, 1);
        assert_eq!(report.horizon_summaries[0].observed_count, 1);
        assert_eq!(report.horizon_summaries[1].missing_count, 1);
        assert_eq!(report.horizon_summaries[2].censored_count, 1);
        assert_eq!(report.horizon_summaries[3].timed_out_count, 1);
        assert_eq!(report.horizon_summaries[4].unsupported_count, 1);
        assert_eq!(report.horizon_summaries[1].p95_absolute_residual, None);
    }

    #[test]
    fn missing_intermediate_horizon_does_not_bridge_paired_growth() {
        let mut sample = input();
        sample.residual_observations = vec![
            residual(1_000, 1.0, 0.0),
            incomplete_residual(2_000, ReferenceOutcomeStatus::Missing),
            residual(3_000, 5.0, 0.0),
        ];
        let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
        assert!(report.paired_horizon_growth.is_empty());
        assert_eq!(report.horizon_summaries[1].outcome_count, 1);
        assert_eq!(report.horizon_summaries[1].observed_count, 0);
        assert_eq!(report.horizon_summaries[1].mean_signed_residual, None);
    }

    #[test] fn reports_paired_horizon_error_growth(){let r=summarize_vehicle_model_discrepancy(&input()).unwrap();assert_eq!(r.paired_horizon_growth.len(),1);assert_eq!(r.paired_horizon_growth[0].absolute_residual_growth,3.0);}
    #[test] fn rejects_mismatched_context_pair(){let mut x=input();x.residual_observations[1].environment_digest=h('9');assert_eq!(summarize_vehicle_model_discrepancy(&x).unwrap_err(),VehicleModelDiscrepancyError::MismatchedResidualContext);}
    #[test] fn rejects_duplicate_scenario_actor_action_horizon(){let mut x=input();x.residual_observations.push(residual(1000,2.0,0.0));assert_eq!(summarize_vehicle_model_discrepancy(&x).unwrap_err(),VehicleModelDiscrepancyError::DuplicateResidualKey);}
    #[test] fn missing_horizons_are_not_imputed(){let mut x=input();x.residual_observations.truncate(1);assert!(summarize_vehicle_model_discrepancy(&x).unwrap().paired_horizon_growth.is_empty());}
    #[test] fn detects_planner_ranking_reversal_and_regret(){let r=summarize_vehicle_model_discrepancy(&input()).unwrap();let d=&r.decision_results[0];assert_eq!(d.predicted_best_action_id.as_deref(),Some("yield"));assert_eq!(d.realized_best_action_id.as_deref(),Some("continue"));assert_eq!(d.ranking_reversed,Some(true));assert_eq!(d.selected_action_regret,Some(6.0));assert_eq!(d.optimism_gap,Some(9.0));}
    #[test] fn rejects_outcome_leakage(){let mut x=input();x.decisions[0].reference_outcome_reveal_sequence=9;x.decisions[0].decision_context_digest=model_discrepancy_decision_context_digest(&x.decisions[0]).unwrap();assert_eq!(summarize_vehicle_model_discrepancy(&x).unwrap_err(),VehicleModelDiscrepancyError::OutcomeLeakage);}
    #[test] fn unknown_feasibility_makes_decision_indeterminate(){let mut x=input();x.decisions[0].actions[0].feasibility=DiscrepancyFeasibility::Indeterminate;x.decisions[0].actions[0].realized_utility=None;let r=summarize_vehicle_model_discrepancy(&x).unwrap();assert_eq!(r.decision_results[0].status,DecisionAuditStatus::IndeterminateFeasibility);}
    #[test]
    fn missing_or_censored_reference_outcomes_make_decision_incomplete() {
        for status in [
            ReferenceOutcomeStatus::Missing,
            ReferenceOutcomeStatus::Censored,
            ReferenceOutcomeStatus::TimedOut,
            ReferenceOutcomeStatus::Unsupported,
        ] {
            let mut sample = input();
            let action = &mut sample.decisions[0].actions[0];
            action.reference_status = status;
            action.feasibility = DiscrepancyFeasibility::Indeterminate;
            action.realized_utility = None;
            action.execution_receipt_id = None;
            action.execution_digest = None;
            action.evaluator_receipt_id = None;
            let report = summarize_vehicle_model_discrepancy(&sample).unwrap();
            assert_eq!(report.decision_results[0].status, DecisionAuditStatus::IncompleteReference);
            assert_eq!(report.decision_summary.incomplete_reference_decision_count, 1);
            assert_eq!(report.decision_results[0].ranking_reversed, None);
            assert_eq!(report.decision_results[0].selected_action_regret, None);
        }
    }

    #[test] fn rejects_action_set_digest_mismatch(){let mut x=input();x.decisions[0].action_set_digest=h('9');x.decisions[0].decision_context_digest=model_discrepancy_decision_context_digest(&x.decisions[0]).unwrap();assert_eq!(summarize_vehicle_model_discrepancy(&x).unwrap_err(),VehicleModelDiscrepancyError::InvalidActionSetDigest);}
    #[test] fn canonicalizes_row_and_action_order(){let x=input();let a=summarize_vehicle_model_discrepancy(&x).unwrap();let mut y=x;y.residual_observations.reverse();y.decisions[0].actions.reverse();let b=summarize_vehicle_model_discrepancy(&y).unwrap();assert_eq!(a.report_digest,b.report_digest);}
    #[test] fn report_digest_detects_tampering(){let mut r=summarize_vehicle_model_discrepancy(&input()).unwrap();assert!(r.identity_is_valid());r.horizon_summaries[0].p95_absolute_residual=Some(r.horizon_summaries[0].p95_absolute_residual.unwrap()+1.0);assert!(!r.identity_is_valid());}
}
