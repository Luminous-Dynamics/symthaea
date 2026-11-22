#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA_VERSION: &str = "mel-predictive-forecast-integrity-v1";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputObservedV1 {
    pub input_id: String,
    pub input_artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetWindowV1 {
    pub start_ns: u64,
    pub end_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastIssuedV1 {
    pub forecast_id: String,
    pub input_event_id: String,
    pub policy_sha256: String,
    pub prediction_set_sha256: String,
    pub predicted_event_count: u32,
    pub target_window: TargetWindowV1,
    pub expires_at_ns: u64,
    pub supersedes_forecast_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastDisplayedV1 {
    pub forecast_id: String,
    pub display_receipt_sha256: String,
    pub display_channel: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartnerDispositionV1 {
    Accepted,
    Steered,
    Vetoed,
    Branched,
    Ignored,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartnerDispositionEventV1 {
    pub forecast_id: String,
    pub actor_id: String,
    pub disposition: PartnerDispositionV1,
    pub disposition_receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidationReasonV1 {
    Superseded,
    CancelledByCollaborator,
    InputChanged,
    Expired,
    PolicyChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastInvalidatedV1 {
    pub forecast_id: String,
    pub reason: InvalidationReasonV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizationStartedV1 {
    pub forecast_id: String,
    pub realized_event_set_sha256: String,
    pub realized_event_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizationCompletedV1 {
    pub forecast_id: String,
    pub realized_artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventKindV1 {
    InputObserved(InputObservedV1),
    ForecastIssued(ForecastIssuedV1),
    ForecastDisplayed(ForecastDisplayedV1),
    PartnerDisposition(PartnerDispositionEventV1),
    ForecastInvalidated(ForecastInvalidatedV1),
    RealizationStarted(RealizationStartedV1),
    RealizationCompleted(RealizationCompletedV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceEventV1 {
    pub event_id: String,
    pub sequence: u64,
    pub monotonic_time_ns: u64,
    pub previous_event_sha256: String,
    pub event_sha256: String,
    pub kind: EventKindV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastTraceV1 {
    pub schema_version: String,
    pub session_id: String,
    pub clock_domain_id: String,
    pub clock_origin_id: String,
    pub events: Vec<TraceEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssueV1 {
    pub code: String,
    pub event_id: Option<String>,
    pub forecast_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastResultV1 {
    pub forecast_id: String,
    pub lifecycle: String,
    pub displayed: bool,
    pub partner_disposition: Option<PartnerDispositionV1>,
    pub realized: bool,
    pub human_visible_realization: bool,
    pub input_to_issue_latency_ns: Option<u64>,
    pub display_lead_time_ns: Option<u64>,
    pub input_to_realization_ns: Option<u64>,
    pub issue_to_realization_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastReportV1 {
    pub report_version: String,
    pub session_id: String,
    pub valid: bool,
    pub event_count: usize,
    pub forecast_count: usize,
    pub displayed_count: usize,
    pub disposition_count: usize,
    pub realized_count: usize,
    pub human_visible_realization_count: usize,
    pub invalidated_count: usize,
    pub forecasts: Vec<ForecastResultV1>,
    pub issues: Vec<ValidationIssueV1>,
}

#[derive(Serialize)]
struct EventHashInput<'a> {
    event_id: &'a str,
    sequence: u64,
    monotonic_time_ns: u64,
    previous_event_sha256: &'a str,
    kind: &'a EventKindV1,
}

#[derive(Debug, Clone)]
struct InputStateV1 {
    event_id: String,
    time_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleV1 {
    Pending,
    Invalidated,
    Realized,
}

#[derive(Debug, Clone)]
struct ForecastStateV1 {
    issued: ForecastIssuedV1,
    issued_at_ns: u64,
    input_time_ns: u64,
    displayed_at_ns: Option<u64>,
    disposition: Option<PartnerDispositionV1>,
    realization_started_at_ns: Option<u64>,
    lifecycle: LifecycleV1,
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn event_sha256(event: &TraceEventV1) -> Result<String, String> {
    digest_serializable(&EventHashInput {
        event_id: &event.event_id,
        sequence: event.sequence,
        monotonic_time_ns: event.monotonic_time_ns,
        previous_event_sha256: &event.previous_event_sha256,
        kind: &event.kind,
    })
}

/// Seals a trace for deterministic fixtures and trusted trace producers.
/// Sealing establishes internal consistency, not actor authority or display proof.
pub fn seal_trace(trace: &mut ForecastTraceV1) -> Result<(), String> {
    let mut previous = ZERO_SHA256.to_owned();
    for event in &mut trace.events {
        event.previous_event_sha256 = previous.clone();
        event.event_sha256 = event_sha256(event)?;
        previous = event.event_sha256.clone();
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn push_issue(
    issues: &mut Vec<ValidationIssueV1>,
    code: &str,
    event_id: Option<&str>,
    forecast_id: Option<&str>,
    detail: &str,
) {
    issues.push(ValidationIssueV1 {
        code: code.to_owned(),
        event_id: event_id.map(str::to_owned),
        forecast_id: forecast_id.map(str::to_owned),
        detail: detail.to_owned(),
    });
}

fn require_digest(
    issues: &mut Vec<ValidationIssueV1>,
    event_id: &str,
    forecast_id: Option<&str>,
    field: &str,
    value: &str,
) {
    if !is_sha256(value) {
        push_issue(
            issues,
            "invalid_sha256",
            Some(event_id),
            forecast_id,
            &format!("{field} must be 64 lowercase hexadecimal characters"),
        );
    }
}

/// Replays forecast lifecycle events and checks provenance references and temporal ordering.
///
/// This evaluator does not inspect prediction or realization bytes, authenticate actors,
/// prove that a UI displayed a preview, score musical accuracy, or infer human anticipation.
pub fn validate_trace(trace: &ForecastTraceV1) -> ForecastReportV1 {
    let mut issues = Vec::new();
    let mut event_ids = BTreeSet::new();
    let mut inputs: BTreeMap<String, InputStateV1> = BTreeMap::new();
    let mut forecasts: BTreeMap<String, ForecastStateV1> = BTreeMap::new();
    let mut previous_event_sha256 = ZERO_SHA256.to_owned();
    let mut previous_time_ns: Option<u64> = None;

    if trace.schema_version != SCHEMA_VERSION {
        push_issue(
            &mut issues,
            "schema_version_mismatch",
            None,
            None,
            "unsupported schema version",
        );
    }
    if !non_empty(&trace.session_id)
        || !non_empty(&trace.clock_domain_id)
        || !non_empty(&trace.clock_origin_id)
    {
        push_issue(
            &mut issues,
            "missing_trace_identity",
            None,
            None,
            "session, clock-domain and clock-origin identifiers are required",
        );
    }
    if trace.events.is_empty() {
        push_issue(&mut issues, "empty_trace", None, None, "trace must contain events");
    }

    for (index, event) in trace.events.iter().enumerate() {
        if !non_empty(&event.event_id) {
            push_issue(
                &mut issues,
                "empty_event_id",
                Some(&event.event_id),
                None,
                "event identifier must not be empty",
            );
        }
        if !event_ids.insert(event.event_id.clone()) {
            push_issue(
                &mut issues,
                "duplicate_event_id",
                Some(&event.event_id),
                None,
                "event identifiers must be unique",
            );
        }
        let expected_sequence = (index as u64) + 1;
        if event.sequence != expected_sequence {
            push_issue(
                &mut issues,
                "non_contiguous_sequence",
                Some(&event.event_id),
                None,
                &format!("expected sequence {expected_sequence}, got {}", event.sequence),
            );
        }
        if previous_time_ns.is_some_and(|previous| event.monotonic_time_ns <= previous) {
            push_issue(
                &mut issues,
                "non_monotonic_event_time",
                Some(&event.event_id),
                None,
                "event monotonic time must increase strictly",
            );
        }
        previous_time_ns = Some(event.monotonic_time_ns);

        if event.previous_event_sha256 != previous_event_sha256 {
            push_issue(
                &mut issues,
                "previous_hash_mismatch",
                Some(&event.event_id),
                None,
                "event does not reference the immediately preceding event digest",
            );
        }
        require_digest(
            &mut issues,
            &event.event_id,
            None,
            "previous_event_sha256",
            &event.previous_event_sha256,
        );
        require_digest(
            &mut issues,
            &event.event_id,
            None,
            "event_sha256",
            &event.event_sha256,
        );
        match event_sha256(event) {
            Ok(actual) if actual != event.event_sha256 => push_issue(
                &mut issues,
                "event_hash_mismatch",
                Some(&event.event_id),
                None,
                "event digest does not match its canonical payload",
            ),
            Err(error) => push_issue(
                &mut issues,
                "event_hash_serialization_failed",
                Some(&event.event_id),
                None,
                &error,
            ),
            _ => {}
        }
        previous_event_sha256 = event.event_sha256.clone();

        match &event.kind {
            EventKindV1::InputObserved(payload) => {
                if !non_empty(&payload.input_id) {
                    push_issue(
                        &mut issues,
                        "empty_input_id",
                        Some(&event.event_id),
                        None,
                        "input identifier must not be empty",
                    );
                }
                require_digest(
                    &mut issues,
                    &event.event_id,
                    None,
                    "input_artifact_sha256",
                    &payload.input_artifact_sha256,
                );
                if inputs.contains_key(&payload.input_id) {
                    push_issue(
                        &mut issues,
                        "duplicate_input_id",
                        Some(&event.event_id),
                        None,
                        "input identifiers must be unique within a session",
                    );
                } else {
                    inputs.insert(
                        payload.input_id.clone(),
                        InputStateV1 {
                            event_id: event.event_id.clone(),
                            time_ns: event.monotonic_time_ns,
                        },
                    );
                }
            }
            EventKindV1::ForecastIssued(payload) => {
                let forecast_id = payload.forecast_id.as_str();
                if !non_empty(forecast_id) {
                    push_issue(
                        &mut issues,
                        "empty_forecast_id",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast identifier must not be empty",
                    );
                }
                for (field, digest) in [
                    ("policy_sha256", payload.policy_sha256.as_str()),
                    ("prediction_set_sha256", payload.prediction_set_sha256.as_str()),
                ] {
                    require_digest(
                        &mut issues,
                        &event.event_id,
                        Some(forecast_id),
                        field,
                        digest,
                    );
                }
                if payload.predicted_event_count == 0 {
                    push_issue(
                        &mut issues,
                        "empty_prediction_set",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast must contain at least one predicted event",
                    );
                }
                if payload.target_window.end_ns <= payload.target_window.start_ns {
                    push_issue(
                        &mut issues,
                        "invalid_target_window",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "target window end must be later than its start",
                    );
                }
                if payload.target_window.start_ns <= event.monotonic_time_ns {
                    push_issue(
                        &mut issues,
                        "forecast_not_precommitted",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast must be issued strictly before its target window begins",
                    );
                }
                if payload.expires_at_ns <= event.monotonic_time_ns {
                    push_issue(
                        &mut issues,
                        "invalid_forecast_expiry",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast expiry must follow issue time",
                    );
                }
                let input = inputs
                    .values()
                    .find(|input| input.event_id == payload.input_event_id);
                let Some(input) = input else {
                    push_issue(
                        &mut issues,
                        "forecast_input_unresolved",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast must reference an earlier observed-input event",
                    );
                    continue;
                };
                if input.time_ns >= event.monotonic_time_ns {
                    push_issue(
                        &mut issues,
                        "forecast_input_not_prior",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast input must precede forecast issue",
                    );
                }
                if let Some(existing) = forecasts.get(forecast_id) {
                    let _ = existing;
                    push_issue(
                        &mut issues,
                        "duplicate_forecast_id",
                        Some(&event.event_id),
                        Some(forecast_id),
                        "forecast identifiers must be unique",
                    );
                    continue;
                }
                if let Some(previous_id) = payload.supersedes_forecast_id.as_deref() {
                    match forecasts.get_mut(previous_id) {
                        Some(previous) if previous.lifecycle == LifecycleV1::Pending => {
                            previous.lifecycle = LifecycleV1::Invalidated;
                        }
                        Some(_) => push_issue(
                            &mut issues,
                            "supersession_target_not_pending",
                            Some(&event.event_id),
                            Some(forecast_id),
                            "only a pending forecast can be superseded",
                        ),
                        None => push_issue(
                            &mut issues,
                            "supersession_target_unresolved",
                            Some(&event.event_id),
                            Some(forecast_id),
                            "superseded forecast must already exist in this trace",
                        ),
                    }
                }
                forecasts.insert(
                    payload.forecast_id.clone(),
                    ForecastStateV1 {
                        issued: payload.clone(),
                        issued_at_ns: event.monotonic_time_ns,
                        input_time_ns: input.time_ns,
                        displayed_at_ns: None,
                        disposition: None,
                        realization_started_at_ns: None,
                        lifecycle: LifecycleV1::Pending,
                    },
                );
            }
            EventKindV1::ForecastDisplayed(payload) => {
                let Some(forecast) = forecasts.get_mut(&payload.forecast_id) else {
                    push_issue(
                        &mut issues,
                        "display_forecast_unresolved",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "display must reference an earlier forecast",
                    );
                    continue;
                };
                require_digest(
                    &mut issues,
                    &event.event_id,
                    Some(&payload.forecast_id),
                    "display_receipt_sha256",
                    &payload.display_receipt_sha256,
                );
                if !non_empty(&payload.display_channel) {
                    push_issue(
                        &mut issues,
                        "missing_display_channel",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "display channel identifier is required",
                    );
                }
                if forecast.lifecycle != LifecycleV1::Pending {
                    push_issue(
                        &mut issues,
                        "display_after_invalidation_or_realization",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "invalidated or realized forecasts cannot be newly displayed",
                    );
                }
                if event.monotonic_time_ns < forecast.issued_at_ns
                    || event.monotonic_time_ns >= forecast.issued.target_window.start_ns
                {
                    push_issue(
                        &mut issues,
                        "display_not_pre_output",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "display must occur after issue and before the target window",
                    );
                }
                if event.monotonic_time_ns > forecast.issued.expires_at_ns {
                    push_issue(
                        &mut issues,
                        "display_after_expiry",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "expired forecasts must not be presented as current",
                    );
                }
                if forecast.displayed_at_ns.replace(event.monotonic_time_ns).is_some() {
                    push_issue(
                        &mut issues,
                        "duplicate_display",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "v1 allows one display receipt per forecast",
                    );
                }
            }
            EventKindV1::PartnerDisposition(payload) => {
                let Some(forecast) = forecasts.get_mut(&payload.forecast_id) else {
                    push_issue(
                        &mut issues,
                        "disposition_forecast_unresolved",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "partner disposition must reference an earlier forecast",
                    );
                    continue;
                };
                require_digest(
                    &mut issues,
                    &event.event_id,
                    Some(&payload.forecast_id),
                    "disposition_receipt_sha256",
                    &payload.disposition_receipt_sha256,
                );
                if !non_empty(&payload.actor_id) {
                    push_issue(
                        &mut issues,
                        "missing_disposition_actor",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "actor identifier is required",
                    );
                }
                if forecast.displayed_at_ns.is_none() {
                    push_issue(
                        &mut issues,
                        "disposition_without_display",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "a disposition requires a prior display event",
                    );
                }
                if forecast.lifecycle != LifecycleV1::Pending {
                    push_issue(
                        &mut issues,
                        "disposition_after_invalidation_or_realization",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "disposition cannot modify an invalidated or realized forecast",
                    );
                }
                if event.monotonic_time_ns >= forecast.issued.target_window.start_ns
                    || event.monotonic_time_ns > forecast.issued.expires_at_ns
                {
                    push_issue(
                        &mut issues,
                        "disposition_too_late",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "partner disposition must precede execution and expiry",
                    );
                }
                if forecast.disposition.replace(payload.disposition).is_some() {
                    push_issue(
                        &mut issues,
                        "duplicate_disposition",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "v1 allows one disposition per forecast",
                    );
                }
                if payload.disposition == PartnerDispositionV1::Vetoed {
                    forecast.lifecycle = LifecycleV1::Invalidated;
                }
            }
            EventKindV1::ForecastInvalidated(payload) => {
                let Some(forecast) = forecasts.get_mut(&payload.forecast_id) else {
                    push_issue(
                        &mut issues,
                        "invalidation_forecast_unresolved",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "invalidation must reference an earlier forecast",
                    );
                    continue;
                };
                if forecast.lifecycle != LifecycleV1::Pending {
                    push_issue(
                        &mut issues,
                        "invalid_invalidation_transition",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "only a pending forecast can be invalidated",
                    );
                }
                if event.monotonic_time_ns >= forecast.issued.target_window.start_ns {
                    push_issue(
                        &mut issues,
                        "invalidation_after_target_start",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "invalidation must happen before target execution begins",
                    );
                }
                forecast.lifecycle = LifecycleV1::Invalidated;
            }
            EventKindV1::RealizationStarted(payload) => {
                let Some(forecast) = forecasts.get_mut(&payload.forecast_id) else {
                    push_issue(
                        &mut issues,
                        "realization_forecast_unresolved",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "realization must bind an earlier forecast",
                    );
                    continue;
                };
                require_digest(
                    &mut issues,
                    &event.event_id,
                    Some(&payload.forecast_id),
                    "realized_event_set_sha256",
                    &payload.realized_event_set_sha256,
                );
                if payload.realized_event_count == 0 {
                    push_issue(
                        &mut issues,
                        "empty_realization",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "realization must contain at least one event",
                    );
                }
                if forecast.lifecycle != LifecycleV1::Pending {
                    push_issue(
                        &mut issues,
                        "realization_of_inactive_forecast",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "invalidated or already realized forecasts cannot start realization",
                    );
                }
                if event.monotonic_time_ns < forecast.issued.target_window.start_ns
                    || event.monotonic_time_ns > forecast.issued.target_window.end_ns
                    || event.monotonic_time_ns > forecast.issued.expires_at_ns
                {
                    push_issue(
                        &mut issues,
                        "realization_outside_valid_window",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "realization start must lie inside the forecast target and expiry windows",
                    );
                }
                if forecast.realization_started_at_ns.replace(event.monotonic_time_ns).is_some() {
                    push_issue(
                        &mut issues,
                        "duplicate_realization_start",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "v1 allows one realization-start event per forecast",
                    );
                }
            }
            EventKindV1::RealizationCompleted(payload) => {
                let Some(forecast) = forecasts.get_mut(&payload.forecast_id) else {
                    push_issue(
                        &mut issues,
                        "completion_forecast_unresolved",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "completion must reference an earlier forecast",
                    );
                    continue;
                };
                require_digest(
                    &mut issues,
                    &event.event_id,
                    Some(&payload.forecast_id),
                    "realized_artifact_sha256",
                    &payload.realized_artifact_sha256,
                );
                let Some(started_at) = forecast.realization_started_at_ns else {
                    push_issue(
                        &mut issues,
                        "completion_without_start",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "realization completion requires a prior realization-start event",
                    );
                    continue;
                };
                if event.monotonic_time_ns < started_at {
                    push_issue(
                        &mut issues,
                        "completion_before_start",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "realization completion must not precede realization start",
                    );
                }
                if forecast.lifecycle != LifecycleV1::Pending {
                    push_issue(
                        &mut issues,
                        "completion_of_inactive_forecast",
                        Some(&event.event_id),
                        Some(&payload.forecast_id),
                        "only a pending forecast can be completed",
                    );
                }
                forecast.lifecycle = LifecycleV1::Realized;
            }
        }
    }

    let forecast_results: Vec<ForecastResultV1> = forecasts
        .iter()
        .map(|(forecast_id, state)| {
            let realized = state.lifecycle == LifecycleV1::Realized;
            let displayed = state.displayed_at_ns.is_some();
            let issue_latency = state.issued_at_ns.checked_sub(state.input_time_ns);
            let display_lead = state
                .displayed_at_ns
                .and_then(|displayed_at| state.issued.target_window.start_ns.checked_sub(displayed_at));
            let input_to_realization = state
                .realization_started_at_ns
                .and_then(|realized_at| realized_at.checked_sub(state.input_time_ns));
            let issue_to_realization = state
                .realization_started_at_ns
                .and_then(|realized_at| realized_at.checked_sub(state.issued_at_ns));
            ForecastResultV1 {
                forecast_id: forecast_id.clone(),
                lifecycle: match state.lifecycle {
                    LifecycleV1::Pending => "pending",
                    LifecycleV1::Invalidated => "invalidated",
                    LifecycleV1::Realized => "realized",
                }
                .to_owned(),
                displayed,
                partner_disposition: state.disposition,
                realized,
                human_visible_realization: displayed && realized,
                input_to_issue_latency_ns: issue_latency,
                display_lead_time_ns: display_lead,
                input_to_realization_ns: input_to_realization,
                issue_to_realization_ns: issue_to_realization,
            }
        })
        .collect();

    let displayed_count = forecast_results.iter().filter(|f| f.displayed).count();
    let disposition_count = forecast_results
        .iter()
        .filter(|f| f.partner_disposition.is_some())
        .count();
    let realized_count = forecast_results.iter().filter(|f| f.realized).count();
    let human_visible_realization_count = forecast_results
        .iter()
        .filter(|f| f.human_visible_realization)
        .count();
    let invalidated_count = forecast_results
        .iter()
        .filter(|f| f.lifecycle == "invalidated")
        .count();

    ForecastReportV1 {
        report_version: "mel-predictive-forecast-report-v1".to_owned(),
        session_id: trace.session_id.clone(),
        valid: issues.is_empty(),
        event_count: trace.events.len(),
        forecast_count: forecast_results.len(),
        displayed_count,
        disposition_count,
        realized_count,
        human_visible_realization_count,
        invalidated_count,
        forecasts: forecast_results,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn event(id: &str, sequence: u64, time: u64, kind: EventKindV1) -> TraceEventV1 {
        TraceEventV1 {
            event_id: id.to_owned(),
            sequence,
            monotonic_time_ns: time,
            previous_event_sha256: ZERO_SHA256.to_owned(),
            event_sha256: ZERO_SHA256.to_owned(),
            kind,
        }
    }

    fn valid_trace() -> ForecastTraceV1 {
        let mut trace = ForecastTraceV1 {
            schema_version: SCHEMA_VERSION.to_owned(),
            session_id: "session-001".to_owned(),
            clock_domain_id: "runner-monotonic-v1".to_owned(),
            clock_origin_id: "origin-001".to_owned(),
            events: vec![
                event("input-evt", 1, 1_000, EventKindV1::InputObserved(InputObservedV1 {
                    input_id: "input-001".to_owned(),
                    input_artifact_sha256: digest('a'),
                })),
                event("forecast-evt", 2, 2_000, EventKindV1::ForecastIssued(ForecastIssuedV1 {
                    forecast_id: "forecast-001".to_owned(),
                    input_event_id: "input-evt".to_owned(),
                    policy_sha256: digest('b'),
                    prediction_set_sha256: digest('c'),
                    predicted_event_count: 4,
                    target_window: TargetWindowV1 { start_ns: 5_000, end_ns: 15_000 },
                    expires_at_ns: 15_000,
                    supersedes_forecast_id: None,
                })),
                event("display-evt", 3, 3_000, EventKindV1::ForecastDisplayed(ForecastDisplayedV1 {
                    forecast_id: "forecast-001".to_owned(),
                    display_receipt_sha256: digest('d'),
                    display_channel: "piano-roll-preview".to_owned(),
                })),
                event("disposition-evt", 4, 4_000, EventKindV1::PartnerDisposition(PartnerDispositionEventV1 {
                    forecast_id: "forecast-001".to_owned(),
                    actor_id: "performer-01".to_owned(),
                    disposition: PartnerDispositionV1::Accepted,
                    disposition_receipt_sha256: digest('e'),
                })),
                event("realization-start", 5, 6_000, EventKindV1::RealizationStarted(RealizationStartedV1 {
                    forecast_id: "forecast-001".to_owned(),
                    realized_event_set_sha256: digest('f'),
                    realized_event_count: 4,
                })),
                event("realization-done", 6, 8_000, EventKindV1::RealizationCompleted(RealizationCompletedV1 {
                    forecast_id: "forecast-001".to_owned(),
                    realized_artifact_sha256: digest('1'),
                })),
            ],
        };
        seal_trace(&mut trace).unwrap();
        trace
    }

    #[test]
    fn precommitted_displayed_forecast_can_be_realized() {
        let report = validate_trace(&valid_trace());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.forecast_count, 1);
        assert_eq!(report.displayed_count, 1);
        assert_eq!(report.disposition_count, 1);
        assert_eq!(report.realized_count, 1);
        assert_eq!(report.human_visible_realization_count, 1);
    }

    #[test]
    fn issue_latency_and_preview_lead_time_are_separate() {
        let report = validate_trace(&valid_trace());
        let forecast = &report.forecasts[0];
        assert_eq!(forecast.input_to_issue_latency_ns, Some(1_000));
        assert_eq!(forecast.display_lead_time_ns, Some(2_000));
        assert_eq!(forecast.input_to_realization_ns, Some(5_000));
        assert_eq!(forecast.issue_to_realization_ns, Some(4_000));
    }

    #[test]
    fn post_hoc_display_is_rejected() {
        let mut trace = valid_trace();
        trace.events[2].monotonic_time_ns = 7_000;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "display_not_pre_output"));
    }

    #[test]
    fn forecast_issued_after_target_start_is_rejected() {
        let mut trace = valid_trace();
        if let EventKindV1::ForecastIssued(payload) = &mut trace.events[1].kind {
            payload.target_window.start_ns = 1_500;
            payload.target_window.end_ns = 15_000;
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "forecast_not_precommitted"));
    }

    #[test]
    fn realization_without_a_forecast_is_rejected() {
        let mut trace = valid_trace();
        trace.events.remove(1);
        trace.events[1].sequence = 2;
        trace.events[1].kind = EventKindV1::RealizationStarted(RealizationStartedV1 {
            forecast_id: "unknown-forecast".to_owned(),
            realized_event_set_sha256: digest('f'),
            realized_event_count: 4,
        });
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "realization_forecast_unresolved"));
    }

    #[test]
    fn superseded_forecast_is_retained_but_cannot_be_realized() {
        let mut trace = valid_trace();
        trace.events.truncate(3);
        trace.events.push(event("input-evt-2", 4, 3_500, EventKindV1::InputObserved(InputObservedV1 {
            input_id: "input-002".to_owned(),
            input_artifact_sha256: digest('2'),
        })));
        trace.events.push(event("forecast-evt-2", 5, 4_000, EventKindV1::ForecastIssued(ForecastIssuedV1 {
            forecast_id: "forecast-002".to_owned(),
            input_event_id: "input-evt-2".to_owned(),
            policy_sha256: digest('b'),
            prediction_set_sha256: digest('3'),
            predicted_event_count: 3,
            target_window: TargetWindowV1 { start_ns: 7_000, end_ns: 16_000 },
            expires_at_ns: 16_000,
            supersedes_forecast_id: Some("forecast-001".to_owned()),
        })));
        trace.events.push(event("stale-realization", 6, 6_000, EventKindV1::RealizationStarted(RealizationStartedV1 {
            forecast_id: "forecast-001".to_owned(),
            realized_event_set_sha256: digest('f'),
            realized_event_count: 4,
        })));
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "realization_of_inactive_forecast"));
        let old = report.forecasts.iter().find(|f| f.forecast_id == "forecast-001").unwrap();
        assert_eq!(old.lifecycle, "invalidated");
    }

    #[test]
    fn disposition_requires_a_prior_display() {
        let mut trace = valid_trace();
        trace.events.remove(2);
        trace.events[2].sequence = 3;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "disposition_without_display"));
    }

    #[test]
    fn realization_outside_window_is_rejected() {
        let mut trace = valid_trace();
        trace.events[4].monotonic_time_ns = 16_000;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "realization_outside_valid_window"));
    }

    #[test]
    fn invalidated_forecast_cannot_be_realized() {
        let mut trace = valid_trace();
        trace.events.truncate(4);
        trace.events.push(event("cancel-evt", 5, 4_500, EventKindV1::ForecastInvalidated(ForecastInvalidatedV1 {
            forecast_id: "forecast-001".to_owned(),
            reason: InvalidationReasonV1::CancelledByCollaborator,
        })));
        trace.events.push(event("realization-start", 6, 6_000, EventKindV1::RealizationStarted(RealizationStartedV1 {
            forecast_id: "forecast-001".to_owned(),
            realized_event_set_sha256: digest('f'),
            realized_event_count: 4,
        })));
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "realization_of_inactive_forecast"));
    }

    #[test]
    fn event_chain_tampering_is_detected() {
        let mut trace = valid_trace();
        if let EventKindV1::ForecastDisplayed(payload) = &mut trace.events[2].kind {
            payload.display_channel = "different-channel".to_owned();
        }
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "event_hash_mismatch"));
    }

    #[test]
    fn duplicate_event_ids_and_non_monotonic_time_are_rejected() {
        let mut trace = valid_trace();
        trace.events[2].event_id = trace.events[1].event_id.clone();
        trace.events[3].monotonic_time_ns = trace.events[2].monotonic_time_ns;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "duplicate_event_id"));
        assert!(report.issues.iter().any(|issue| issue.code == "non_monotonic_event_time"));
    }

    #[test]
    fn vetoed_forecast_cannot_be_realized() {
        let mut trace = valid_trace();
        if let EventKindV1::PartnerDisposition(payload) = &mut trace.events[3].kind {
            payload.disposition = PartnerDispositionV1::Vetoed;
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "realization_of_inactive_forecast"));
        let forecast = &report.forecasts[0];
        assert_eq!(forecast.lifecycle, "invalidated");
        assert!(!forecast.human_visible_realization);
    }

    #[test]
    fn no_display_is_not_counted_as_human_visible() {
        let mut trace = valid_trace();
        trace.events.remove(2);
        trace.events[2].sequence = 3;
        trace.events[2].monotonic_time_ns = 4_000;
        trace.events[3].sequence = 4;
        trace.events[3].monotonic_time_ns = 6_000;
        trace.events[4].sequence = 5;
        trace.events[4].monotonic_time_ns = 8_000;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.realized_count, 1);
        assert_eq!(report.human_visible_realization_count, 0);
    }
}
