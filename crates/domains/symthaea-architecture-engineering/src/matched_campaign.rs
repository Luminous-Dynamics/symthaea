//! Cross-policy reconciliation for mechanism-matched AR-5B4 campaigns.
//!
//! This module validates that per-policy tapes share one frozen experiment contract and
//! that any repeated candidate/proposal identity produces the same evaluator/verifier outcome.
//! It does not prove that the evaluator-owned offer set was fully materialized; that remains
//! a separate campaign-input artifact and is an explicit qualification limitation.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::event_tape::{
    ActionIntent, ActionOutcome, BudgetModel, ComparisonMode, EventTape, EventTapeSummary,
    MeterValue, ResourceCostWeights, ResourceReceipt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedCampaignError {
    pub path: String,
    pub message: String,
}

impl MatchedCampaignError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MatchedCampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for MatchedCampaignError {}

/// Result of reconciling per-policy event tapes for one campaign/seed/budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedCampaignSummary {
    pub campaign_id: String,
    pub common_contract_sha256: String,
    pub policy_ids: Vec<String>,
    pub oracle_policy_ids: Vec<String>,
    pub event_counts_by_policy: BTreeMap<String, usize>,
    /// None means at least one action had an explicitly unknown normalized charge.
    pub charged_budget_by_policy_microunits: BTreeMap<String, Option<u64>>,
    pub shared_candidate_identity_count: usize,
    pub candidate_identity_shared_by_all_policies: usize,
    pub shared_proposal_identity_count: usize,
    pub proposal_identity_shared_by_all_policies: usize,
    pub all_execution_schedules_complete: bool,
    /// Full per-dimension aggregate receipts, with None for any unknown/overflowed dimension.
    pub resource_totals_by_policy: BTreeMap<String, ResourceTotals>,
    /// True only if all policies spend exactly the same fully measured normalized budget.
    pub normalized_budget_spend_matched: bool,
    /// True only if every tape passes strict multi-resource qualification.
    pub resource_budget_qualified: bool,
    /// This is a necessary but not sufficient condition for full scientific comparison.
    pub matched_contract_reconciled: bool,
}

/// Per-policy resource totals. Each field is None if any action had an unknown
/// measurement or if summation overflowed. These vectors are intentionally not collapsed
/// into one scalar so Pareto trade-offs remain inspectable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceTotals {
    pub host_cpu_millis: Option<u64>,
    /// Sum of action wall durations (worker-time); not elapsed campaign latency.
    pub sum_action_wall_millis: Option<u64>,
    pub accelerator_millis: Option<u64>,
    /// Maximum per-action host peak; not concurrent campaign-wide memory.
    pub max_action_peak_memory_bytes: Option<u64>,
    /// Maximum per-action accelerator peak; not concurrent campaign-wide memory.
    pub max_action_accelerator_peak_memory_bytes: Option<u64>,
    pub energy_millijoules: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub verifier_calls: Option<u64>,
    pub charged_budget_microunits: Option<u64>,
}

impl Default for ResourceTotals {
    fn default() -> Self {
        Self {
            host_cpu_millis: Some(0),
            sum_action_wall_millis: Some(0),
            accelerator_millis: Some(0),
            max_action_peak_memory_bytes: Some(0),
            max_action_accelerator_peak_memory_bytes: Some(0),
            energy_millijoules: Some(0),
            input_tokens: Some(0),
            output_tokens: Some(0),
            tool_calls: Some(0),
            evaluator_calls: Some(0),
            verifier_calls: Some(0),
            charged_budget_microunits: Some(0),
        }
    }
}

impl ResourceTotals {
    fn add_receipt(
        &mut self,
        receipt: &ResourceReceipt,
        path: &str,
    ) -> Result<(), MatchedCampaignError> {
        add_meter(&mut self.host_cpu_millis, &receipt.host_cpu_millis, &format!("{path}.host_cpu_millis"))?;
        add_meter(&mut self.sum_action_wall_millis, &receipt.wall_millis, &format!("{path}.wall_millis"))?;
        add_meter(&mut self.accelerator_millis, &receipt.accelerator_millis, &format!("{path}.accelerator_millis"))?;
        add_max_meter(&mut self.max_action_peak_memory_bytes, &receipt.peak_memory_bytes, &format!("{path}.peak_memory_bytes"))?;
        add_max_meter(&mut self.max_action_accelerator_peak_memory_bytes, &receipt.accelerator_peak_memory_bytes, &format!("{path}.accelerator_peak_memory_bytes"))?;
        add_meter(&mut self.energy_millijoules, &receipt.energy_millijoules, &format!("{path}.energy_millijoules"))?;
        add_meter(&mut self.input_tokens, &receipt.input_tokens, &format!("{path}.input_tokens"))?;
        add_meter(&mut self.output_tokens, &receipt.output_tokens, &format!("{path}.output_tokens"))?;
        add_meter(&mut self.tool_calls, &receipt.tool_calls, &format!("{path}.tool_calls"))?;
        add_meter(&mut self.evaluator_calls, &receipt.evaluator_calls, &format!("{path}.evaluator_calls"))?;
        add_meter(&mut self.verifier_calls, &receipt.verifier_calls, &format!("{path}.verifier_calls"))?;
        add_meter(&mut self.charged_budget_microunits, &receipt.charged_budget_microunits, &format!("{path}.charged_budget_microunits"))?;
        Ok(())
    }
}

fn add_max_meter(
    total: &mut Option<u64>,
    value: &MeterValue,
    path: &str,
) -> Result<(), MatchedCampaignError> {
    match value {
        MeterValue::Unknown { .. } => *total = None,
        MeterValue::Measured { value } => {
            if let Some(current) = total {
                *current = (*current).max(*value);
            }
        }
    }
    Ok(())
}

fn add_meter(
    total: &mut Option<u64>,
    value: &MeterValue,
    path: &str,
) -> Result<(), MatchedCampaignError> {
    match value {
        MeterValue::Unknown { .. } => *total = None,
        MeterValue::Measured { value } => {
            if let Some(current) = total {
                *current = current.checked_add(*value).ok_or_else(|| {
                    MatchedCampaignError::new(path, "resource total overflow")
                })?;
            }
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct CommonContract<'a> {
    campaign_id: &'a str,
    source_commit: &'a str,
    execution_environment_sha256: &'a str,
    comparison_mode: ComparisonMode,
    proposal_source_id: &'a str,
    proposal_stream_sha256: Option<&'a str>,
    task_recipe_sha256: &'a str,
    comparison_profile_sha256: &'a str,
    evaluator_sha256: &'a str,
    verifier_sha256: &'a str,
    budget_model_id: &'a str,
    budget_model: &'a BudgetModel,
    budget_model_sha256: &'a str,
    budget_limit_microunits: u64,
    initial_direction_ids: &'a [String],
    execution: &'a super::event_tape::ExecutionContract,
}

type CandidateRecord = (
    String,
    Option<String>,
    String,
    ActionOutcome,
    BTreeSet<String>,
);

type ProposalRecord = (
    Option<String>,
    String,
    ActionOutcome,
    BTreeSet<String>,
);

/// Reconcile a set of policy tapes intended to share one evaluator-owned proposal stream.
///
/// Policy snapshots may differ; all other experiment-contract fields must match. When two
/// policies evaluate the same candidate digest or submit the same proposal descriptor digest,
/// their direction metadata and full evaluator/verifier outcome must also agree.
pub fn reconcile_mechanism_matched_campaign(
    tapes: &[EventTape],
) -> Result<MatchedCampaignSummary, MatchedCampaignError> {
    if tapes.len() < 2 {
        return Err(MatchedCampaignError::new(
            "tapes",
            "at least two distinct policy tapes are required",
        ));
    }

    let mut policy_ids = BTreeSet::new();
    let mut oracle_policy_ids = Vec::new();
    let mut event_counts_by_policy = BTreeMap::new();
    let mut charged_budget_by_policy_microunits = BTreeMap::new();
    let mut resource_totals_by_policy = BTreeMap::new();
    let mut candidate_records = BTreeMap::<String, CandidateRecord>::new();
    let mut proposal_records = BTreeMap::<String, ProposalRecord>::new();
    let mut common_contract_sha256: Option<String> = None;
    let mut campaign_id: Option<String> = None;
    let mut all_execution_schedules_complete = true;
    let mut resource_budget_qualified = true;

    for (tape_index, tape) in tapes.iter().enumerate() {
        let header = tape.header();
        if header.comparison_mode != ComparisonMode::MechanismMatched {
            return Err(MatchedCampaignError::new(
                format!("tapes[{tape_index}].comparison_mode"),
                "best-practical-system tapes cannot be mixed into a mechanism-matched campaign",
            ));
        }

        let summary: EventTapeSummary = tape
            .validate_integrity()
            .map_err(|error| {
                MatchedCampaignError::new(
                    format!("tapes[{tape_index}].integrity"),
                    error.to_string(),
                )
            })?;
        if summary.event_count == 0 {
            return Err(MatchedCampaignError::new(
                format!("tapes[{tape_index}].events"),
                "an empty policy tape cannot be part of an executed campaign",
            ));
        }
        if !summary.execution_schedule_complete {
            all_execution_schedules_complete = false;
        }
        if tape.validate().is_err() {
            resource_budget_qualified = false;
        }

        let policy_id = header.policy.policy_id.clone();
        if !policy_ids.insert(policy_id.clone()) {
            return Err(MatchedCampaignError::new(
                format!("tapes[{tape_index}].policy.policy_id"),
                format!("duplicate policy tape {policy_id:?}"),
            ));
        }
        if header.policy.has_hidden_oracle_access {
            oracle_policy_ids.push(policy_id.clone());
        }
        event_counts_by_policy.insert(policy_id.clone(), summary.event_count);
        charged_budget_by_policy_microunits
            .insert(policy_id.clone(), summary.total_charged_budget_microunits);
        let mut resource_totals = ResourceTotals::default();
        for (event_index, envelope) in tape.events().iter().enumerate() {
            resource_totals.add_receipt(
                &envelope.payload.resources,
                &format!("tapes[{tape_index}].events[{event_index}].resources"),
            )?;
        }
        if resource_totals.charged_budget_microunits
            != summary.total_charged_budget_microunits
        {
            return Err(MatchedCampaignError::new(
                format!("tapes[{tape_index}].resources.charged_budget_microunits"),
                "independently summed receipt charges do not reconcile to the event-tape summary",
            ));
        }
        resource_totals_by_policy.insert(policy_id.clone(), resource_totals);

        if let Some(expected_campaign) = &campaign_id {
            if expected_campaign != &header.campaign_id {
                return Err(MatchedCampaignError::new(
                    format!("tapes[{tape_index}].campaign_id"),
                    "campaign ID differs from the paired policy tapes",
                ));
            }
        } else {
            campaign_id = Some(header.campaign_id.clone());
        }

        let contract = CommonContract {
            campaign_id: &header.campaign_id,
            source_commit: &header.source_commit,
            execution_environment_sha256: &header.execution_environment_sha256,
            comparison_mode: header.comparison_mode,
            proposal_source_id: &header.proposal_source_id,
            proposal_stream_sha256: header.proposal_stream_sha256.as_deref(),
            task_recipe_sha256: &header.task_recipe_sha256,
            comparison_profile_sha256: &header.comparison_profile_sha256,
            evaluator_sha256: &header.evaluator_sha256,
            verifier_sha256: &header.verifier_sha256,
            budget_model_id: &header.budget_model_id,
            budget_model: &header.budget_model,
            budget_model_sha256: &header.budget_model_sha256,
            budget_limit_microunits: header.budget_limit_microunits,
            initial_direction_ids: &header.initial_direction_ids,
            execution: &header.execution,
        };
        let contract_sha = digest_serializable(&contract)?;
        if let Some(expected_sha) = &common_contract_sha256 {
            if expected_sha != &contract_sha {
                return Err(MatchedCampaignError::new(
                    format!("tapes[{tape_index}].header"),
                    "source, environment, evaluator, proposal stream, budget model, budget ceiling or execution contract differs",
                ));
            }
        } else {
            common_contract_sha256 = Some(contract_sha);
        }

        for (event_index, envelope) in tape.events().iter().enumerate() {
            let event = &envelope.payload;
            match (&event.intent, &event.outcome) {
                (
                    ActionIntent::EvaluateCandidate {
                        candidate_sha256,
                        parent_candidate_sha256,
                        semantic_delta_sha256,
                        direction_id,
                    },
                    outcome @ ActionOutcome::CandidateEvaluation { .. },
                ) => {
                    let record = candidate_records.entry(candidate_sha256.clone()).or_insert_with(|| {
                        (
                            direction_id.clone(),
                            parent_candidate_sha256.clone(),
                            semantic_delta_sha256.clone(),
                            (*outcome).clone(),
                            BTreeSet::new(),
                        )
                    });
                    if record.0.as_str() != direction_id.as_str()
                        || record.1.as_ref() != parent_candidate_sha256.as_ref()
                        || record.2.as_str() != semantic_delta_sha256.as_str()
                        || !same_outcome_semantics(&record.3, outcome)
                    {
                        return Err(MatchedCampaignError::new(
                            format!("tapes[{tape_index}].events[{event_index}]"),
                            "same candidate digest has conflicting direction, semantic delta, parent or evaluator/verifier outcome",
                        ));
                    }
                    record.4.insert(policy_id.clone());
                }
                (
                    ActionIntent::ProposeDirection {
                        descriptor_sha256,
                        parent_direction_id,
                        semantic_delta_sha256,
                    },
                    outcome @ ActionOutcome::DirectionVerification { .. },
                ) => {
                    let record = proposal_records.entry(descriptor_sha256.clone()).or_insert_with(|| {
                        (
                            parent_direction_id.clone(),
                            semantic_delta_sha256.clone(),
                            (*outcome).clone(),
                            BTreeSet::new(),
                        )
                    });
                    if record.0.as_ref() != parent_direction_id.as_ref()
                        || record.1.as_str() != semantic_delta_sha256.as_str()
                        || !same_outcome_semantics(&record.2, outcome)
                    {
                        return Err(MatchedCampaignError::new(
                            format!("tapes[{tape_index}].events[{event_index}]"),
                            "same proposal descriptor has conflicting parent, semantic delta or verifier outcome",
                        ));
                    }
                    record.3.insert(policy_id.clone());
                }
                _ => {
                    return Err(MatchedCampaignError::new(
                        format!("tapes[{tape_index}].events[{event_index}]"),
                        "intent and outcome kinds disagree",
                    ));
                }
            }
        }
    }

    let policy_count = policy_ids.len();
    let normalized_budget_spend_matched = {
        let complete_spends: BTreeSet<u64> = charged_budget_by_policy_microunits
            .values()
            .filter_map(|value| *value)
            .collect();
        charged_budget_by_policy_microunits
            .values()
            .all(Option::is_some)
            && complete_spends.len() == 1
            && charged_budget_by_policy_microunits.len() == policy_count
    };
    let shared_candidate_identity_count = candidate_records
        .values()
        .filter(|record| record.4.len() >= 2)
        .count();
    let candidate_identity_shared_by_all_policies = candidate_records
        .values()
        .filter(|record| record.4.len() == policy_count)
        .count();
    let shared_proposal_identity_count = proposal_records
        .values()
        .filter(|record| record.3.len() >= 2)
        .count();
    let proposal_identity_shared_by_all_policies = proposal_records
        .values()
        .filter(|record| record.3.len() == policy_count)
        .count();

    let mut policy_ids: Vec<String> = policy_ids.into_iter().collect();
    policy_ids.sort_unstable();
    oracle_policy_ids.sort_unstable();

    Ok(MatchedCampaignSummary {
        campaign_id: campaign_id.unwrap_or_default(),
        common_contract_sha256: common_contract_sha256.ok_or_else(|| {
            MatchedCampaignError::new("header", "common campaign contract was not initialized")
        })?,
        policy_ids,
        oracle_policy_ids,
        event_counts_by_policy,
        charged_budget_by_policy_microunits,
        resource_totals_by_policy,
        normalized_budget_spend_matched,
        shared_candidate_identity_count,
        candidate_identity_shared_by_all_policies,
        shared_proposal_identity_count,
        proposal_identity_shared_by_all_policies,
        all_execution_schedules_complete,
        resource_budget_qualified,
        matched_contract_reconciled: true,
    })
}


/// Compare scientific outcomes while allowing per-execution receipt envelopes to differ.
/// Evaluator/verifier identities and the evidence artifact digest remain part of the comparison;
/// only the receipt digest itself may be unique to a policy execution.
fn same_outcome_semantics(left: &ActionOutcome, right: &ActionOutcome) -> bool {
    match (left, right) {
        (
            ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: left_task,
                comparison_profile_sha256: left_profile,
                evaluator_sha256: left_evaluator,
                verifier_sha256: left_verifier,
                disposition: left_disposition,
                score: left_score,
                uncertainty: left_uncertainty,
                evidence_sha256: left_evidence,
                ..
            },
            ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: right_task,
                comparison_profile_sha256: right_profile,
                evaluator_sha256: right_evaluator,
                verifier_sha256: right_verifier,
                disposition: right_disposition,
                score: right_score,
                uncertainty: right_uncertainty,
                evidence_sha256: right_evidence,
                ..
            },
        ) => {
            left_task == right_task
                && left_profile == right_profile
                && left_evaluator == right_evaluator
                && left_verifier == right_verifier
                && left_disposition == right_disposition
                && left_score == right_score
                && left_uncertainty == right_uncertainty
                && left_evidence == right_evidence
        }
        (
            ActionOutcome::DirectionVerification {
                task_recipe_sha256: left_task,
                comparison_profile_sha256: left_profile,
                verifier_sha256: left_verifier,
                disposition: left_disposition,
                ..
            },
            ActionOutcome::DirectionVerification {
                task_recipe_sha256: right_task,
                comparison_profile_sha256: right_profile,
                verifier_sha256: right_verifier,
                disposition: right_disposition,
                ..
            },
        ) => {
            left_task == right_task
                && left_profile == right_profile
                && left_verifier == right_verifier
                && left_disposition == right_disposition
        }
        _ => false,
    }
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, MatchedCampaignError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        MatchedCampaignError::new(
            "serialization",
            format!("deterministic campaign-contract encoding failed: {error}"),
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::event_tape::{
        ActionEvent, CandidateDisposition, EventTapeHeader, ExecutionContract, ExecutionMode,
        MeterValue, PolicySnapshot, ResourceReceipt,
    };

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn digest<T: Serialize>(value: &T) -> String {
        let bytes = serde_json::to_vec(value).unwrap();
        let hash = Sha256::digest(bytes);
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn meter(value: u64) -> MeterValue {
        MeterValue::Measured { value }
    }

    fn policy_tape(
        policy_id: &str,
        algorithm_id: &str,
        config_value: serde_json::Value,
        score: f64,
        unknown_wall_time: bool,
    ) -> EventTape {
        policy_tape_with_budget_model(
            policy_id,
            algorithm_id,
            config_value,
            score,
            unknown_wall_time,
            1,
            100,
        )
    }

    fn policy_tape_with_budget_model(
        policy_id: &str,
        algorithm_id: &str,
        config_value: serde_json::Value,
        score: f64,
        unknown_wall_time: bool,
        budget_model_weight: u64,
        charge: u64,
    ) -> EventTape {
        let config_sha256 = digest(&config_value);
        let budget_model = BudgetModel::WeightedResources {
            weights: ResourceCostWeights {
                host_cpu_millis: budget_model_weight,
                wall_millis: 0,
                accelerator_millis: 0,
                host_memory_byte_millis: 0,
                accelerator_memory_byte_millis: 0,
                energy_millijoules: 0,
                input_tokens: 0,
                output_tokens: 0,
                tool_calls: 0,
                evaluator_calls: 0,
                verifier_calls: 0,
            },
        };
        let header = EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id: "campaign-paired-1".to_owned(),
            source_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            execution_environment_sha256: H1.to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "frozen-proposals-v1".to_owned(),
            proposal_stream_sha256: Some(H2.to_owned()),
            task_recipe_sha256: H3.to_owned(),
            comparison_profile_sha256: H4.to_owned(),
            evaluator_sha256: H5.to_owned(),
            verifier_sha256: H1.to_owned(),
            budget_model_id: "microcost-v1".to_owned(),
            budget_model_sha256: digest(&budget_model),
            budget_model,
            budget_limit_microunits: 1000,
            initial_direction_ids: vec!["A".to_owned()],
            execution: ExecutionContract {
                mode: ExecutionMode::Sequential,
                worker_count: 1,
                seats_per_round: 1,
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: policy_id.to_owned(),
                algorithm_id: algorithm_id.to_owned(),
                config_sha256,
                config: config_value,
                has_hidden_oracle_access: policy_id == "oracle",
            },
        };
        let mut tape = EventTape::new(header).unwrap();
        let unknown = |reason: &str| MeterValue::Unknown {
            reason: reason.to_owned(),
        };
        if budget_model_weight == 0 || charge % budget_model_weight != 0 {
            panic!("test fixture charge must be divisible by its positive budget weight");
        }
        let mut resources = ResourceReceipt {
            host_cpu_millis: meter(charge / budget_model_weight),
            wall_millis: meter(5),
            accelerator_millis: meter(0),
            peak_memory_bytes: meter(1024),
            accelerator_peak_memory_bytes: meter(0),
            energy_millijoules: meter(1),
            input_tokens: meter(0),
            output_tokens: meter(0),
            tool_calls: meter(0),
            evaluator_calls: meter(1),
            verifier_calls: meter(1),
            charged_budget_microunits: meter(charge),
        };
        if unknown_wall_time {
            resources.wall_millis = unknown("test runner did not provide wall-time telemetry");
        }
        tape.append(ActionEvent {
            event_id: "paired-event-1".to_owned(),
            policy_id: policy_id.to_owned(),
            worker_id: 0,
            round_id: 1,
            batch_id: None,
            observed_through_sequence: 0,
            visible_history_sha256: String::new(),
            retry_of_event_id: None,
            intent: ActionIntent::EvaluateCandidate {
                candidate_sha256: H3.to_owned(),
                parent_candidate_sha256: None,
                semantic_delta_sha256: H4.to_owned(),
                direction_id: "A".to_owned(),
            },
            outcome: ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H3.to_owned(),
                comparison_profile_sha256: H4.to_owned(),
                evaluator_sha256: H5.to_owned(),
                verifier_sha256: H1.to_owned(),
                verifier_receipt_sha256: H2.to_owned(),
                disposition: CandidateDisposition::Verified,
                score: Some(score),
                uncertainty: Some(0.0),
                evidence_sha256: Some(H4.to_owned()),
            },
            resources,
        })
        .unwrap();
        tape
    }

    #[test]
    fn matching_policies_reconcile_common_contract_and_candidate_outcome() {
        let left = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, false);
        let right = policy_tape("thompson", "thompson-build-1", json!({"prior": 1.0}), 0.91, false);
        let summary = reconcile_mechanism_matched_campaign(&[left, right]).unwrap();
        assert_eq!(summary.policy_ids, vec!["board".to_owned(), "thompson".to_owned()]);
        assert_eq!(summary.shared_candidate_identity_count, 1);
        assert_eq!(summary.candidate_identity_shared_by_all_policies, 1);
        assert!(summary.resource_budget_qualified);
        assert!(summary.normalized_budget_spend_matched);
        assert_eq!(
            summary.resource_totals_by_policy["board"].host_cpu_millis,
            Some(100)
        );
        assert_eq!(
            summary.resource_totals_by_policy["board"].max_action_peak_memory_bytes,
            Some(1024),
            "peak memory is a max, never a sum of action peaks"
        );
        assert_eq!(
            summary.resource_totals_by_policy["board"].sum_action_wall_millis,
            Some(5),
            "summed wall durations are worker-time, not campaign latency"
        );
        assert!(summary.all_execution_schedules_complete);
    }

    #[test]
    fn candidate_identity_cannot_hide_conflicting_evaluator_results() {
        let left = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, false);
        let right = policy_tape("thompson", "thompson-build-1", json!({"prior": 1.0}), 0.87, false);
        assert!(reconcile_mechanism_matched_campaign(&[left, right]).is_err());
    }

    #[test]
    fn unknown_resource_receipts_reconcile_but_do_not_qualify_budget_claims() {
        let left = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, true);
        let right = policy_tape("thompson", "thompson-build-1", json!({"prior": 1.0}), 0.91, false);
        let summary = reconcile_mechanism_matched_campaign(&[left, right]).unwrap();
        assert!(!summary.resource_budget_qualified);
        assert!(summary.normalized_budget_spend_matched);
        assert_eq!(
            summary.charged_budget_by_policy_microunits.get("board"),
            Some(&Some(100))
        );
        assert_eq!(summary.resource_totals_by_policy["board"].sum_action_wall_millis, None);
    }

    #[test]
    fn oracle_policy_is_reported_separately_not_silently_hidden() {
        let baseline = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, false);
        let oracle = policy_tape("oracle", "oracle-build-1", json!({"peek": true}), 0.99, false);
        let summary = reconcile_mechanism_matched_campaign(&[baseline, oracle]).unwrap();
        assert_eq!(summary.oracle_policy_ids, vec!["oracle".to_owned()]);
    }

    #[test]
    fn mismatched_budget_model_is_rejected() {
        let left = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, false);
        let right = policy_tape_with_budget_model(
            "thompson",
            "thompson-build-1",
            json!({"prior": 1.0}),
            0.91,
            false,
            2,
            100,
        );
        assert!(reconcile_mechanism_matched_campaign(&[left, right]).is_err());
    }
    #[test]
    fn unequal_normalized_spend_is_reported_as_not_matched() {
        let left = policy_tape("board", "board-build-1", json!({"alpha": 0.5}), 0.91, false);
        let right = policy_tape_with_budget_model(
            "thompson",
            "thompson-build-1",
            json!({"prior": 1.0}),
            0.91,
            false,
            H2,
            120,
        );
        let summary = reconcile_mechanism_matched_campaign(&[left, right]).unwrap();
        assert!(!summary.normalized_budget_spend_matched);
        assert!(summary.resource_budget_qualified);
    }

}
