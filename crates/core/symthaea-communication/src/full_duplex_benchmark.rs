// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic benchmark harness for full-duplex turn-taking control.
//!
//! This module measures interaction-control behavior from the VOICE-FDX-001A
//! controller. Trace validity, semantic correctness, and latency budgets remain
//! separate evidence dimensions and are never collapsed into one score.

use crate::full_duplex_turn_controller::{
    FullDuplexAssistantOutputStateV1, FullDuplexControlActionV1,
    FullDuplexTurnControllerV1, FullDuplexTurnErrorV1, FullDuplexUserFloorStateV1,
};
use serde::{Deserialize, Serialize};

pub const FULL_DUPLEX_BENCHMARK_SCHEMA_V1: &str =
    "symthaea.communication.full-duplex-benchmark.v1";
const CASE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea:communication:full-duplex-benchmark-case:v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexBenchmarkScenarioV1 {
    InterruptionYield,
    ExplicitStop,
    BackchannelContinuity,
    DeescalationAcknowledgement,
    ResponseLatency,
    MixedInteraction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexBenchmarkEventV1 {
    AssistantBegin { turn_id: String, at_ns: u64 },
    AssistantEnd { turn_id: String, at_ns: u64 },
    UserSpeechStart { utterance_id: String, at_ns: u64 },
    UserSpeechEnd { utterance_id: String, at_ns: u64 },
    UserBackchannel { event_id: String, at_ns: u64 },
    UserSlowDown { event_id: String, at_ns: u64 },
    AcknowledgeDeescalation { plan_ref: String, at_ns: u64 },
    UserStop { event_id: String, at_ns: u64 },
    AcknowledgeOutputStopped { turn_id: String, at_ns: u64 },
}

impl FullDuplexBenchmarkEventV1 {
    pub const fn at_ns(&self) -> u64 {
        match self {
            Self::AssistantBegin { at_ns, .. }
            | Self::AssistantEnd { at_ns, .. }
            | Self::UserSpeechStart { at_ns, .. }
            | Self::UserSpeechEnd { at_ns, .. }
            | Self::UserBackchannel { at_ns, .. }
            | Self::UserSlowDown { at_ns, .. }
            | Self::AcknowledgeDeescalation { at_ns, .. }
            | Self::UserStop { at_ns, .. }
            | Self::AcknowledgeOutputStopped { at_ns, .. } => *at_ns,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexLatencyBudgetV1 {
    pub max_interruption_yield_latency_ns: Option<u64>,
    pub max_stop_latency_ns: Option<u64>,
    pub max_deescalation_ack_latency_ns: Option<u64>,
    pub max_response_latency_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexBenchmarkCaseV1 {
    pub case_id: String,
    pub session_epoch: u64,
    pub scenario: FullDuplexBenchmarkScenarioV1,
    pub events: Vec<FullDuplexBenchmarkEventV1>,
    pub latency_budget: FullDuplexLatencyBudgetV1,
}

impl FullDuplexBenchmarkCaseV1 {
    pub fn new(
        case_id: impl Into<String>,
        session_epoch: u64,
        scenario: FullDuplexBenchmarkScenarioV1,
        events: Vec<FullDuplexBenchmarkEventV1>,
        latency_budget: FullDuplexLatencyBudgetV1,
    ) -> Result<Self, FullDuplexBenchmarkDefinitionErrorV1> {
        let case_id = canonical_case_id(case_id.into())?;
        if session_epoch == 0 {
            return Err(FullDuplexBenchmarkDefinitionErrorV1::InvalidSessionEpoch);
        }
        if events.is_empty() || events.len() > 4096 {
            return Err(FullDuplexBenchmarkDefinitionErrorV1::InvalidEventCount);
        }

        let mut previous_at_ns = None;
        for event in &events {
            if previous_at_ns.is_some_and(|previous| event.at_ns() < previous) {
                return Err(FullDuplexBenchmarkDefinitionErrorV1::NonMonotonicEventTime);
            }
            previous_at_ns = Some(event.at_ns());
            validate_event_identity(event)?;
        }

        Ok(Self {
            case_id,
            session_epoch,
            scenario,
            events,
            latency_budget,
        })
    }

    pub fn commitment_v1(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CASE_COMMITMENT_DOMAIN_V1);
        hash_str(&mut hasher, FULL_DUPLEX_BENCHMARK_SCHEMA_V1);
        hash_str(&mut hasher, &self.case_id);
        hasher.update(&self.session_epoch.to_le_bytes());
        hasher.update(&[scenario_tag(self.scenario)]);
        hasher.update(&(self.events.len() as u32).to_le_bytes());
        for event in &self.events {
            hash_event(&mut hasher, event);
        }
        hash_optional_u64(
            &mut hasher,
            self.latency_budget.max_interruption_yield_latency_ns,
        );
        hash_optional_u64(&mut hasher, self.latency_budget.max_stop_latency_ns);
        hash_optional_u64(
            &mut hasher,
            self.latency_budget.max_deescalation_ack_latency_ns,
        );
        hash_optional_u64(&mut hasher, self.latency_budget.max_response_latency_ns);
        format!("full-duplex-benchmark-case:{}", hasher.finalize().to_hex())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexTraceStatusV1 {
    Complete,
    RuntimeRejected,
    SemanticallyIncomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexSemanticStatusV1 {
    Satisfied,
    Violated,
    NotEstablished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullDuplexLatencyBudgetStatusV1 {
    /// No latency limits were requested for this case.
    NotEvaluated,
    /// At least one requested latency measurement was absent.
    NotEstablished,
    /// All requested latency measurements were present and within their limits.
    WithinBudget,
    /// All requested measurements were present and at least one exceeded its limit.
    Exceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FullDuplexSemanticViolationV1 {
    MissingInterruptionReceipt,
    MissingStopReceipt,
    MissingDeescalationReceipt,
    MissingResponseLatency,
    BackchannelStoleFloor,
    AssistantStillOutputtingAfterYieldOrStop,
    StopNotLatched,
    DeescalationStillLatchedAfterAcknowledgement,
    UserFloorStillOccupied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FullDuplexLatencyViolationV1 {
    InterruptionYield,
    ExplicitStop,
    DeescalationAcknowledgement,
    ResponseStart,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexLatencyObservationV1 {
    pub interruption_yield_latency_ns: Option<u64>,
    pub stop_latency_ns: Option<u64>,
    pub deescalation_ack_latency_ns: Option<u64>,
    pub response_latency_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullDuplexBenchmarkCaseReceiptV1 {
    pub schema: String,
    pub case_id: String,
    pub case_commitment: String,
    pub scenario: FullDuplexBenchmarkScenarioV1,
    pub session_epoch: u64,
    pub trace_status: FullDuplexTraceStatusV1,
    pub semantic_status: FullDuplexSemanticStatusV1,
    pub latency_budget_status: FullDuplexLatencyBudgetStatusV1,
    pub accepted_events: u64,
    pub rejected_step_index: Option<u64>,
    pub rejected_error: Option<String>,
    pub latency: FullDuplexLatencyObservationV1,
    pub semantic_violations: Vec<FullDuplexSemanticViolationV1>,
    pub latency_violations: Vec<FullDuplexLatencyViolationV1>,
    pub backchannel_count: u64,
    pub final_user_floor_silent: bool,
    pub final_assistant_output_silent: bool,
    pub final_stop_latched: bool,
    pub final_deescalation_required: bool,
}

pub fn run_full_duplex_benchmark_case_v1(
    case: &FullDuplexBenchmarkCaseV1,
) -> FullDuplexBenchmarkCaseReceiptV1 {
    let case_commitment = case.commitment_v1();
    let mut controller = match FullDuplexTurnControllerV1::new(case.session_epoch) {
        Ok(controller) => controller,
        Err(error) => {
            return rejected_receipt(case, case_commitment, 0, format!("{error:?}"));
        }
    };

    let mut accepted_events = 0_u64;
    let mut response_latency_ns = None;
    for (index, event) in case.events.iter().enumerate() {
        let actions = match apply_event(&mut controller, event) {
            Ok(actions) => actions,
            Err(error) => {
                return rejected_receipt(
                    case,
                    case_commitment,
                    index as u64,
                    format!("{error:?}"),
                );
            }
        };
        accepted_events += 1;
        for action in actions {
            if let FullDuplexControlActionV1::AssistantSpeechPermitted(permit) = action {
                if let Some(latency) = permit.response_latency_ns {
                    response_latency_ns = Some(latency);
                }
            }
        }
    }

    let latency = FullDuplexLatencyObservationV1 {
        interruption_yield_latency_ns: controller
            .last_interruption_receipt()
            .map(|receipt| receipt.latency_ns),
        stop_latency_ns: controller.last_stop_receipt().map(|receipt| receipt.latency_ns),
        deescalation_ack_latency_ns: controller
            .last_deescalation_receipt()
            .map(|receipt| receipt.latency_ns),
        response_latency_ns,
    };

    let mut semantic_violations = semantic_violations_for(case.scenario, &controller, &latency);
    semantic_violations.sort();
    semantic_violations.dedup();
    let semantic_status = semantic_status_for(&semantic_violations);
    let (latency_budget_status, latency_violations) =
        evaluate_latency_budget(&case.latency_budget, &latency);

    let trace_status = if semantic_status == FullDuplexSemanticStatusV1::NotEstablished {
        FullDuplexTraceStatusV1::SemanticallyIncomplete
    } else {
        FullDuplexTraceStatusV1::Complete
    };

    FullDuplexBenchmarkCaseReceiptV1 {
        schema: FULL_DUPLEX_BENCHMARK_SCHEMA_V1.into(),
        case_id: case.case_id.clone(),
        case_commitment,
        scenario: case.scenario,
        session_epoch: case.session_epoch,
        trace_status,
        semantic_status,
        latency_budget_status,
        accepted_events,
        rejected_step_index: None,
        rejected_error: None,
        latency,
        semantic_violations,
        latency_violations,
        backchannel_count: controller.backchannel_receipts().len() as u64,
        final_user_floor_silent: matches!(
            controller.user_floor(),
            FullDuplexUserFloorStateV1::Silent
        ),
        final_assistant_output_silent: matches!(
            controller.assistant_output(),
            FullDuplexAssistantOutputStateV1::Silent
        ),
        final_stop_latched: controller.stop_latch().is_some(),
        final_deescalation_required: controller.deescalation_required(),
    }
}

fn apply_event(
    controller: &mut FullDuplexTurnControllerV1,
    event: &FullDuplexBenchmarkEventV1,
) -> Result<Vec<FullDuplexControlActionV1>, FullDuplexTurnErrorV1> {
    match event {
        FullDuplexBenchmarkEventV1::AssistantBegin { turn_id, at_ns } => controller
            .assistant_begin(turn_id.clone(), *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::AssistantEnd { turn_id, at_ns } => controller
            .assistant_end(turn_id, *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::UserSpeechStart {
            utterance_id,
            at_ns,
        } => controller.user_speech_start(utterance_id.clone(), *at_ns),
        FullDuplexBenchmarkEventV1::UserSpeechEnd {
            utterance_id,
            at_ns,
        } => controller
            .user_speech_end(utterance_id, *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::UserBackchannel { event_id, at_ns } => controller
            .user_backchannel(event_id.clone(), *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::UserSlowDown { event_id, at_ns } => controller
            .user_slow_down(event_id.clone(), *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::AcknowledgeDeescalation { plan_ref, at_ns } => controller
            .acknowledge_deescalation(plan_ref.clone(), *at_ns)
            .map(|action| vec![action]),
        FullDuplexBenchmarkEventV1::UserStop { event_id, at_ns } => {
            controller.user_stop(event_id.clone(), *at_ns)
        }
        FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped { turn_id, at_ns } => controller
            .acknowledge_output_stopped(turn_id, *at_ns)
            .map(|action| vec![action]),
    }
}

fn semantic_status_for(
    violations: &[FullDuplexSemanticViolationV1],
) -> FullDuplexSemanticStatusV1 {
    if violations.is_empty() {
        return FullDuplexSemanticStatusV1::Satisfied;
    }
    if violations.iter().any(|violation| {
        matches!(
            violation,
            FullDuplexSemanticViolationV1::MissingInterruptionReceipt
                | FullDuplexSemanticViolationV1::MissingStopReceipt
                | FullDuplexSemanticViolationV1::MissingDeescalationReceipt
                | FullDuplexSemanticViolationV1::MissingResponseLatency
        )
    }) {
        FullDuplexSemanticStatusV1::NotEstablished
    } else {
        FullDuplexSemanticStatusV1::Violated
    }
}

fn semantic_violations_for(
    scenario: FullDuplexBenchmarkScenarioV1,
    controller: &FullDuplexTurnControllerV1,
    latency: &FullDuplexLatencyObservationV1,
) -> Vec<FullDuplexSemanticViolationV1> {
    let mut violations = Vec::new();
    match scenario {
        FullDuplexBenchmarkScenarioV1::InterruptionYield => {
            if latency.interruption_yield_latency_ns.is_none() {
                violations.push(FullDuplexSemanticViolationV1::MissingInterruptionReceipt);
            }
            if !assistant_is_silent(controller) {
                violations.push(
                    FullDuplexSemanticViolationV1::AssistantStillOutputtingAfterYieldOrStop,
                );
            }
        }
        FullDuplexBenchmarkScenarioV1::ExplicitStop => {
            if latency.stop_latency_ns.is_none() {
                violations.push(FullDuplexSemanticViolationV1::MissingStopReceipt);
            }
            if controller.stop_latch().is_none() {
                violations.push(FullDuplexSemanticViolationV1::StopNotLatched);
            }
            if !assistant_is_silent(controller) {
                violations.push(
                    FullDuplexSemanticViolationV1::AssistantStillOutputtingAfterYieldOrStop,
                );
            }
        }
        FullDuplexBenchmarkScenarioV1::BackchannelContinuity => {
            if controller.backchannel_receipts().is_empty()
                || matches!(
                    controller.assistant_output(),
                    FullDuplexAssistantOutputStateV1::Silent
                        | FullDuplexAssistantOutputStateV1::YieldRequested { .. }
                )
            {
                violations.push(FullDuplexSemanticViolationV1::BackchannelStoleFloor);
            }
        }
        FullDuplexBenchmarkScenarioV1::DeescalationAcknowledgement => {
            if latency.deescalation_ack_latency_ns.is_none() {
                violations.push(FullDuplexSemanticViolationV1::MissingDeescalationReceipt);
            }
            if controller.deescalation_required() {
                violations.push(
                    FullDuplexSemanticViolationV1::DeescalationStillLatchedAfterAcknowledgement,
                );
            }
        }
        FullDuplexBenchmarkScenarioV1::ResponseLatency => {
            if latency.response_latency_ns.is_none() {
                violations.push(FullDuplexSemanticViolationV1::MissingResponseLatency);
            }
            if !matches!(controller.user_floor(), FullDuplexUserFloorStateV1::Silent) {
                violations.push(FullDuplexSemanticViolationV1::UserFloorStillOccupied);
            }
        }
        FullDuplexBenchmarkScenarioV1::MixedInteraction => {
            if !matches!(controller.user_floor(), FullDuplexUserFloorStateV1::Silent) {
                violations.push(FullDuplexSemanticViolationV1::UserFloorStillOccupied);
            }
            if controller.stop_latch().is_some() && !assistant_is_silent(controller) {
                violations.push(
                    FullDuplexSemanticViolationV1::AssistantStillOutputtingAfterYieldOrStop,
                );
            }
        }
    }
    violations
}

fn evaluate_latency_budget(
    budget: &FullDuplexLatencyBudgetV1,
    latency: &FullDuplexLatencyObservationV1,
) -> (
    FullDuplexLatencyBudgetStatusV1,
    Vec<FullDuplexLatencyViolationV1>,
) {
    let requested = [
        (
            budget.max_interruption_yield_latency_ns,
            latency.interruption_yield_latency_ns,
            FullDuplexLatencyViolationV1::InterruptionYield,
        ),
        (
            budget.max_stop_latency_ns,
            latency.stop_latency_ns,
            FullDuplexLatencyViolationV1::ExplicitStop,
        ),
        (
            budget.max_deescalation_ack_latency_ns,
            latency.deescalation_ack_latency_ns,
            FullDuplexLatencyViolationV1::DeescalationAcknowledgement,
        ),
        (
            budget.max_response_latency_ns,
            latency.response_latency_ns,
            FullDuplexLatencyViolationV1::ResponseStart,
        ),
    ];

    if requested.iter().all(|(limit, _, _)| limit.is_none()) {
        return (FullDuplexLatencyBudgetStatusV1::NotEvaluated, Vec::new());
    }

    if requested
        .iter()
        .any(|(limit, observed, _)| limit.is_some() && observed.is_none())
    {
        return (FullDuplexLatencyBudgetStatusV1::NotEstablished, Vec::new());
    }

    let violations = requested
        .iter()
        .filter_map(|(limit, observed, violation)| match (limit, observed) {
            (Some(limit), Some(observed)) if observed > limit => Some(*violation),
            _ => None,
        })
        .collect::<Vec<_>>();

    if violations.is_empty() {
        (FullDuplexLatencyBudgetStatusV1::WithinBudget, violations)
    } else {
        (FullDuplexLatencyBudgetStatusV1::Exceeded, violations)
    }
}

fn assistant_is_silent(controller: &FullDuplexTurnControllerV1) -> bool {
    matches!(
        controller.assistant_output(),
        FullDuplexAssistantOutputStateV1::Silent
    )
}

fn rejected_receipt(
    case: &FullDuplexBenchmarkCaseV1,
    case_commitment: String,
    rejected_step_index: u64,
    rejected_error: String,
) -> FullDuplexBenchmarkCaseReceiptV1 {
    FullDuplexBenchmarkCaseReceiptV1 {
        schema: FULL_DUPLEX_BENCHMARK_SCHEMA_V1.into(),
        case_id: case.case_id.clone(),
        case_commitment,
        scenario: case.scenario,
        session_epoch: case.session_epoch,
        trace_status: FullDuplexTraceStatusV1::RuntimeRejected,
        semantic_status: FullDuplexSemanticStatusV1::NotEstablished,
        latency_budget_status: FullDuplexLatencyBudgetStatusV1::NotEvaluated,
        accepted_events: rejected_step_index,
        rejected_step_index: Some(rejected_step_index),
        rejected_error: Some(rejected_error),
        latency: FullDuplexLatencyObservationV1 {
            interruption_yield_latency_ns: None,
            stop_latency_ns: None,
            deescalation_ack_latency_ns: None,
            response_latency_ns: None,
        },
        semantic_violations: Vec::new(),
        latency_violations: Vec::new(),
        backchannel_count: 0,
        final_user_floor_silent: false,
        final_assistant_output_silent: false,
        final_stop_latched: false,
        final_deescalation_required: false,
    }
}

fn validate_event_identity(
    event: &FullDuplexBenchmarkEventV1,
) -> Result<(), FullDuplexBenchmarkDefinitionErrorV1> {
    let value = match event {
        FullDuplexBenchmarkEventV1::AssistantBegin { turn_id, .. }
        | FullDuplexBenchmarkEventV1::AssistantEnd { turn_id, .. }
        | FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped { turn_id, .. } => turn_id,
        FullDuplexBenchmarkEventV1::UserSpeechStart { utterance_id, .. }
        | FullDuplexBenchmarkEventV1::UserSpeechEnd { utterance_id, .. } => utterance_id,
        FullDuplexBenchmarkEventV1::UserBackchannel { event_id, .. }
        | FullDuplexBenchmarkEventV1::UserSlowDown { event_id, .. }
        | FullDuplexBenchmarkEventV1::UserStop { event_id, .. } => event_id,
        FullDuplexBenchmarkEventV1::AcknowledgeDeescalation { plan_ref, .. } => plan_ref,
    };
    if value.trim().is_empty() || value.len() > 1024 {
        return Err(FullDuplexBenchmarkDefinitionErrorV1::InvalidEventIdentity);
    }
    Ok(())
}

fn canonical_case_id(value: String) -> Result<String, FullDuplexBenchmarkDefinitionErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(FullDuplexBenchmarkDefinitionErrorV1::InvalidCaseId);
    }
    Ok(value)
}

fn hash_event(hasher: &mut blake3::Hasher, event: &FullDuplexBenchmarkEventV1) {
    match event {
        FullDuplexBenchmarkEventV1::AssistantBegin { turn_id, at_ns } => {
            hasher.update(&[0]);
            hash_str(hasher, turn_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::AssistantEnd { turn_id, at_ns } => {
            hasher.update(&[1]);
            hash_str(hasher, turn_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::UserSpeechStart {
            utterance_id,
            at_ns,
        } => {
            hasher.update(&[2]);
            hash_str(hasher, utterance_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::UserSpeechEnd {
            utterance_id,
            at_ns,
        } => {
            hasher.update(&[3]);
            hash_str(hasher, utterance_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::UserBackchannel { event_id, at_ns } => {
            hasher.update(&[4]);
            hash_str(hasher, event_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::UserSlowDown { event_id, at_ns } => {
            hasher.update(&[5]);
            hash_str(hasher, event_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::AcknowledgeDeescalation { plan_ref, at_ns } => {
            hasher.update(&[6]);
            hash_str(hasher, plan_ref);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::UserStop { event_id, at_ns } => {
            hasher.update(&[7]);
            hash_str(hasher, event_id);
            hasher.update(&at_ns.to_le_bytes());
        }
        FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped { turn_id, at_ns } => {
            hasher.update(&[8]);
            hash_str(hasher, turn_id);
            hasher.update(&at_ns.to_le_bytes());
        }
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

fn hash_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    if let Some(value) = value {
        hasher.update(&[1]);
        hasher.update(&value.to_le_bytes());
    } else {
        hasher.update(&[0]);
    }
}

const fn scenario_tag(scenario: FullDuplexBenchmarkScenarioV1) -> u8 {
    match scenario {
        FullDuplexBenchmarkScenarioV1::InterruptionYield => 0,
        FullDuplexBenchmarkScenarioV1::ExplicitStop => 1,
        FullDuplexBenchmarkScenarioV1::BackchannelContinuity => 2,
        FullDuplexBenchmarkScenarioV1::DeescalationAcknowledgement => 3,
        FullDuplexBenchmarkScenarioV1::ResponseLatency => 4,
        FullDuplexBenchmarkScenarioV1::MixedInteraction => 5,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullDuplexBenchmarkDefinitionErrorV1 {
    InvalidCaseId,
    InvalidSessionEpoch,
    InvalidEventCount,
    InvalidEventIdentity,
    NonMonotonicEventTime,
}
