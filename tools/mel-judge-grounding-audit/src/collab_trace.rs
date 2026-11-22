use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const TRACE_SCHEMA: &str = "mel-collab-interaction-trace-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractionTraceV1 {
    schema_version: String,
    session_id: String,
    case_id: String,
    track: TrackKind,
    clock_domain_id: String,
    hardware_profile_id: String,
    events: Vec<EventRecordV1>,
    rest_windows: Vec<RestWindowV1>,
    loopback_measurement: Option<LoopbackMeasurementV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    Midi,
    RenderedAudio,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventRecordV1 {
    event_id: String,
    sequence: u64,
    monotonic_ns: u64,
    kind: String,
    input_chunk: Option<InputChunkV1>,
    decision: Option<DecisionV1>,
    output: Option<OutputV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputChunkV1 {
    input_sequence: u64,
    context_start_ns: u64,
    context_end_ns: u64,
    available_at_ns: u64,
    media_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionV1 {
    input_sequence: u64,
    state: DecisionState,
    reason_code: String,
    confidence: f64,
    deadline_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionState {
    Wait,
    Respond,
    Continue,
    Yield,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputV1 {
    decision_event_id: String,
    input_sequence: u64,
    output_sequence: u64,
    output_sha256: String,
    first_emitted_ns: u64,
    first_audible_onset_ns: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RestWindowV1 {
    window_id: String,
    start_ns: u64,
    end_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoopbackMeasurementV1 {
    measurement_id: String,
    clock_domain_id: String,
    receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencySummary {
    pub sample_count: usize,
    pub p50_ns: Option<u64>,
    pub p95_ns: Option<u64>,
    pub p99_ns: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceAuditReport {
    pub schema_version: String,
    pub session_id: String,
    pub case_id: String,
    pub track: TrackKind,
    pub event_count: usize,
    pub input_chunk_count: usize,
    pub decision_count: usize,
    pub wait_count: usize,
    pub respond_count: usize,
    pub continue_count: usize,
    pub yield_count: usize,
    pub output_count: usize,
    pub response_decisions_without_output: usize,
    pub late_output_count: usize,
    pub active_decision_during_rest_count: usize,
    pub false_interruption_count: usize,
    pub rest_windows_without_wait_or_yield: usize,
    pub loopback_audible_latency_sample_count: usize,
    pub emission_latency_from_input_available: LatencySummary,
    pub audible_latency_from_input_available: LatencySummary,
    pub qualified_for_declared_trace_contract: bool,
    pub qualification_findings: Vec<String>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Default)]
struct DecisionRecord {
    input_sequence: u64,
    state: Option<DecisionState>,
    timestamp_ns: u64,
    deadline_ns: Option<u64>,
    output_count: usize,
}

pub fn parse_and_audit(input: &str) -> Result<TraceAuditReport, String> {
    let trace: InteractionTraceV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid interaction trace JSON: {error}"))?;
    audit_trace(&trace)
}

fn audit_trace(trace: &InteractionTraceV1) -> Result<TraceAuditReport, String> {
    validate_header(trace)?;

    let mut event_ids = BTreeSet::new();
    let mut input_chunks = BTreeMap::<u64, (u64, u64)>::new();
    let mut decisions = BTreeMap::<String, DecisionRecord>::new();
    let mut previous_output_sequence = 0u64;
    let mut emission_latencies = Vec::new();
    let mut audible_latencies = Vec::new();
    let mut qualification_findings = Vec::new();

    let mut previous_sequence = 0u64;
    let mut previous_time = 0u64;
    let mut input_chunk_count = 0usize;
    let mut decision_count = 0usize;
    let mut wait_count = 0usize;
    let mut respond_count = 0usize;
    let mut continue_count = 0usize;
    let mut yield_count = 0usize;
    let mut output_count = 0usize;
    let mut late_output_count = 0usize;
    let mut active_decision_during_rest_count = 0usize;
    let mut false_interruption_count = 0usize;

    for event in &trace.events {
        require_text(&event.event_id, "event_id")?;
        if !event_ids.insert(event.event_id.as_str()) {
            return Err(format!("duplicate event_id: {}", event.event_id));
        }
        if event.sequence != previous_sequence + 1 {
            return Err(format!(
                "event sequence must be contiguous: expected {}, received {}",
                previous_sequence + 1,
                event.sequence
            ));
        }
        if event.monotonic_ns < previous_time {
            return Err(format!(
                "non-monotonic event time at {}: {} < {}",
                event.event_id, event.monotonic_ns, previous_time
            ));
        }
        previous_sequence = event.sequence;
        previous_time = event.monotonic_ns;

        let payload_count = event.input_chunk.is_some() as usize
            + event.decision.is_some() as usize
            + event.output.is_some() as usize;
        if payload_count != 1 {
            return Err(format!(
                "event {} must contain exactly one typed payload",
                event.event_id
            ));
        }

        match event.kind.as_str() {
            "input_chunk" => {
                let chunk = event.input_chunk.as_ref().ok_or_else(|| {
                    format!("event {} kind/payload mismatch for input_chunk", event.event_id)
                })?;
                validate_input_chunk(chunk, event.monotonic_ns)?;
                if chunk.input_sequence != input_chunks.len() as u64 + 1 {
                    return Err(format!(
                        "input_sequence must be contiguous: expected {}, received {}",
                        input_chunks.len() as u64 + 1,
                        chunk.input_sequence
                    ));
                }
                input_chunks.insert(
                    chunk.input_sequence,
                    (chunk.context_end_ns, chunk.available_at_ns),
                );
                input_chunk_count += 1;
            }
            "decision" => {
                let decision = event.decision.as_ref().ok_or_else(|| {
                    format!("event {} kind/payload mismatch for decision", event.event_id)
                })?;
                validate_decision(decision, event.monotonic_ns, &input_chunks)?;
                if decisions.contains_key(&event.event_id) {
                    return Err(format!("duplicate decision event_id: {}", event.event_id));
                }
                match decision.state {
                    DecisionState::Wait => wait_count += 1,
                    DecisionState::Respond => respond_count += 1,
                    DecisionState::Continue => continue_count += 1,
                    DecisionState::Yield => yield_count += 1,
                }
                decisions.insert(
                    event.event_id.clone(),
                    DecisionRecord {
                        input_sequence: decision.input_sequence,
                        state: Some(decision.state),
                        timestamp_ns: event.monotonic_ns,
                        deadline_ns: decision.deadline_ns,
                        output_count: 0,
                    },
                );
                decision_count += 1;
                if matches!(decision.state, DecisionState::Respond | DecisionState::Continue)
                    && is_inside_rest(event.monotonic_ns, &trace.rest_windows)
                {
                    active_decision_during_rest_count += 1;
                    qualification_findings.push(format!(
                        "active decision {} occurred inside a protected rest window",
                        event.event_id
                    ));
                }
            }
            "output" => {
                let output = event.output.as_ref().ok_or_else(|| {
                    format!("event {} kind/payload mismatch for output", event.event_id)
                })?;
                validate_output(
                    output,
                    event.monotonic_ns,
                    &trace.clock_domain_id,
                    trace.loopback_measurement.as_ref(),
                    &mut decisions,
                    &input_chunks,
                    &mut previous_output_sequence,
                )?;
                let decision = decisions
                    .get_mut(&output.decision_event_id)
                    .expect("validate_output checked the decision reference");
                decision.output_count += 1;
                let (_, available_at_ns) = input_chunks
                    .get(&output.input_sequence)
                    .copied()
                    .expect("validate_output checked the input reference");
                emission_latencies.push(output.first_emitted_ns - available_at_ns);
                if let Some(audible_ns) = output.first_audible_onset_ns {
                    audible_latencies.push(audible_ns - available_at_ns);
                }
                if output.first_emitted_ns > decision.deadline_ns.unwrap_or(u64::MAX) {
                    late_output_count += 1;
                    qualification_findings.push(format!(
                        "output {} missed its declared response deadline",
                        event.event_id
                    ));
                }
                let rest_time = match trace.track {
                    TrackKind::Midi => Some(output.first_emitted_ns),
                    TrackKind::RenderedAudio => output.first_audible_onset_ns,
                };
                if rest_time.is_some_and(|time| is_inside_rest(time, &trace.rest_windows)) {
                    false_interruption_count += 1;
                    qualification_findings.push(format!(
                        "output {} interrupted a protected rest window",
                        event.event_id
                    ));
                }
                output_count += 1;
            }
            other => {
                return Err(format!("unsupported event kind {other}"));
            }
        }
    }

    for (decision_id, decision) in &decisions {
        match decision.state {
            Some(DecisionState::Respond | DecisionState::Continue) if decision.output_count == 0 => {
                qualification_findings.push(format!(
                    "response decision {decision_id} has no corresponding output"
                ));
            }
            Some(DecisionState::Wait | DecisionState::Yield) if decision.output_count > 0 => {
                return Err(format!(
                    "WAIT/YIELD decision {decision_id} must not produce output"
                ));
            }
            _ => {}
        }
    }

    let response_decisions_without_output = decisions
        .values()
        .filter(|decision| {
            matches!(decision.state, Some(DecisionState::Respond | DecisionState::Continue))
                && decision.output_count == 0
        })
        .count();

    let mut rest_windows_without_wait_or_yield = 0usize;
    for window in &trace.rest_windows {
        let has_rest_decision = trace.events.iter().any(|event| {
            if event.monotonic_ns < window.start_ns || event.monotonic_ns >= window.end_ns {
                return false;
            }
            decisions.get(&event.event_id).is_some_and(|decision| {
                matches!(decision.state, Some(DecisionState::Wait | DecisionState::Yield))
            })
        });
        if !has_rest_decision {
            rest_windows_without_wait_or_yield += 1;
            qualification_findings.push(format!(
                "rest window {} has no WAIT or YIELD decision",
                window.window_id
            ));
        }
    }

    if trace.track == TrackKind::RenderedAudio
        && audible_latencies.len() != output_count
    {
        qualification_findings.push(format!(
            "rendered-audio trace has {} of {} outputs with independent loopback audible-onset timestamps",
            audible_latencies.len(),
            output_count
        ));
    }

    Ok(TraceAuditReport {
        schema_version: "mel-collab-interaction-trace-report-v1".to_owned(),
        session_id: trace.session_id.clone(),
        case_id: trace.case_id.clone(),
        track: trace.track,
        event_count: trace.events.len(),
        input_chunk_count,
        decision_count,
        wait_count,
        respond_count,
        continue_count,
        yield_count,
        output_count,
        response_decisions_without_output,
        late_output_count,
        active_decision_during_rest_count,
        false_interruption_count,
        rest_windows_without_wait_or_yield,
        loopback_audible_latency_sample_count: audible_latencies.len(),
        emission_latency_from_input_available: summarize_latency(emission_latencies),
        audible_latency_from_input_available: summarize_latency(audible_latencies),
        qualified_for_declared_trace_contract: qualification_findings.is_empty(),
        qualification_findings,
        claim_ceiling: trace.claim_ceiling.clone(),
    })
}

fn validate_header(trace: &InteractionTraceV1) -> Result<(), String> {
    if trace.schema_version != TRACE_SCHEMA {
        return Err("unsupported interaction trace schema_version".to_owned());
    }
    require_text(&trace.session_id, "session_id")?;
    require_text(&trace.case_id, "case_id")?;
    require_text(&trace.clock_domain_id, "clock_domain_id")?;
    require_text(&trace.hardware_profile_id, "hardware_profile_id")?;
    if trace.events.is_empty() {
        return Err("events must be non-empty".to_owned());
    }
    if trace.claim_ceiling.is_empty()
        || trace.claim_ceiling.iter().any(|claim| claim.trim().is_empty())
    {
        return Err("claim_ceiling must contain non-empty limits".to_owned());
    }

    let mut previous_end = 0u64;
    let mut window_ids = BTreeSet::new();
    for window in &trace.rest_windows {
        require_text(&window.window_id, "rest window_id")?;
        if !window_ids.insert(window.window_id.as_str()) {
            return Err(format!("duplicate rest window_id: {}", window.window_id));
        }
        if window.start_ns >= window.end_ns || window.start_ns < previous_end {
            return Err("rest windows must be positive-duration and ordered without overlap".to_owned());
        }
        previous_end = window.end_ns;
    }

    match &trace.loopback_measurement {
        Some(measurement) => {
            require_text(&measurement.measurement_id, "loopback measurement_id")?;
            require_digest(&measurement.receipt_sha256, "loopback receipt digest")?;
            if measurement.clock_domain_id != trace.clock_domain_id {
                return Err("loopback clock domain must match the trace monotonic clock".to_owned());
            }
        }
        None => {
            if trace.events.iter().any(|event| {
                event
                    .output
                    .as_ref()
                    .is_some_and(|output| output.first_audible_onset_ns.is_some())
            }) {
                return Err("audible onset timestamps require a loopback measurement receipt".to_owned());
            }
        }
    }
    Ok(())
}

fn validate_input_chunk(chunk: &InputChunkV1, event_time_ns: u64) -> Result<(), String> {
    if chunk.context_start_ns > chunk.context_end_ns
        || chunk.context_end_ns > chunk.available_at_ns
        || chunk.available_at_ns > event_time_ns
    {
        return Err("input chunk context/availability timestamps violate causal ordering".to_owned());
    }
    require_digest(&chunk.media_sha256, "input media digest")
}

fn validate_decision(
    decision: &DecisionV1,
    event_time_ns: u64,
    inputs: &BTreeMap<u64, (u64, u64)>,
) -> Result<(), String> {
    let (_, available_at_ns) = inputs
        .get(&decision.input_sequence)
        .copied()
        .ok_or_else(|| "decision references an input chunk that is not yet available".to_owned())?;
    if event_time_ns < available_at_ns {
        return Err("decision occurs before referenced input availability".to_owned());
    }
    require_text(&decision.reason_code, "decision reason_code")?;
    if !decision.confidence.is_finite() || !(0.0..=1.0).contains(&decision.confidence) {
        return Err("decision confidence must be finite and within [0,1]".to_owned());
    }
    match decision.state {
        DecisionState::Respond | DecisionState::Continue => {
            let deadline = decision
                .deadline_ns
                .ok_or_else(|| "RESPOND/CONTINUE decisions require deadline_ns".to_owned())?;
            if deadline < event_time_ns {
                return Err("response deadline cannot precede its decision".to_owned());
            }
        }
        DecisionState::Wait | DecisionState::Yield => {
            if decision.deadline_ns.is_some() {
                return Err("WAIT/YIELD decisions must not declare a response deadline".to_owned());
            }
        }
    }
    Ok(())
}

fn validate_output(
    output: &OutputV1,
    event_time_ns: u64,
    clock_domain_id: &str,
    loopback: Option<&LoopbackMeasurementV1>,
    decisions: &mut BTreeMap<String, DecisionRecord>,
    inputs: &BTreeMap<u64, (u64, u64)>,
    previous_output_sequence: &mut u64,
) -> Result<(), String> {
    require_text(&output.decision_event_id, "output decision_event_id")?;
    require_digest(&output.output_sha256, "output digest")?;
    if output.output_sequence != *previous_output_sequence + 1 {
        return Err(format!(
            "output_sequence must be contiguous: expected {}, received {}",
            *previous_output_sequence + 1,
            output.output_sequence
        ));
    }
    *previous_output_sequence = output.output_sequence;
    let decision = decisions
        .get(&output.decision_event_id)
        .ok_or_else(|| "output references a decision that does not precede it".to_owned())?;
    if !matches!(decision.state, Some(DecisionState::Respond | DecisionState::Continue)) {
        return Err("output may only follow RESPOND or CONTINUE".to_owned());
    }
    if decision.input_sequence != output.input_sequence {
        return Err("output input_sequence must match its decision".to_owned());
    }
    let (_, available_at_ns) = inputs
        .get(&output.input_sequence)
        .copied()
        .ok_or_else(|| "output references unknown input sequence".to_owned())?;
    if output.first_emitted_ns < decision.timestamp_ns
        || output.first_emitted_ns < available_at_ns
        || output.first_emitted_ns > event_time_ns
    {
        return Err("output emission timestamp violates causal ordering".to_owned());
    }
    if let Some(audible_ns) = output.first_audible_onset_ns {
        let measurement = loopback.ok_or_else(|| {
            "audible onset timestamp has no independent loopback receipt".to_owned()
        })?;
        if measurement.clock_domain_id != clock_domain_id {
            return Err("audible onset and trace clocks are not in the same clock domain".to_owned());
        }
        if audible_ns < output.first_emitted_ns || audible_ns > event_time_ns {
            return Err("audible onset timestamp violates output ordering".to_owned());
        }
    }
    Ok(())
}

fn summarize_latency(mut values: Vec<u64>) -> LatencySummary {
    values.sort_unstable();
    LatencySummary {
        sample_count: values.len(),
        p50_ns: percentile(&values, 50),
        p95_ns: percentile(&values, 95),
        p99_ns: percentile(&values, 99),
    }
}

fn percentile(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let rank = (values.len() * percentile).div_ceil(100).max(1);
    values.get(rank - 1).copied()
}

fn is_inside_rest(timestamp_ns: u64, windows: &[RestWindowV1]) -> bool {
    windows
        .iter()
        .any(|window| timestamp_ns >= window.start_ns && timestamp_ns < window.end_ns)
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!("{name} must be a 64-character lowercase SHA-256 digest"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-collab-trace-v1.json");

    fn trace() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer trace fixture JSON")
    }

    fn audit(value: &Value) -> Result<TraceAuditReport, String> {
        parse_and_audit(&value.to_string())
    }

    #[test]
    fn valid_trace_reports_causal_emission_latency_and_rest_respect() {
        let report = parse_and_audit(FIXTURE).expect("trace should be valid");
        assert!(report.qualified_for_declared_trace_contract);
        assert_eq!(report.event_count, 5);
        assert_eq!(report.input_chunk_count, 2);
        assert_eq!(report.wait_count, 1);
        assert_eq!(report.respond_count, 1);
        assert_eq!(report.output_count, 1);
        assert_eq!(report.emission_latency_from_input_available.p50_ns, Some(40));
        assert_eq!(report.false_interruption_count, 0);
    }

    #[test]
    fn non_monotonic_event_time_is_rejected() {
        let mut value = trace();
        value["events"][1]["monotonic_ns"] = serde_json::json!(99);
        let error = audit(&value).expect_err("backdated decision must fail");
        assert!(error.contains("non-monotonic"));
    }

    #[test]
    fn future_input_reference_is_rejected() {
        let mut value = trace();
        value["events"][1]["decision"]["input_sequence"] = serde_json::json!(2);
        let error = audit(&value).expect_err("decision cannot see future chunk");
        assert!(error.contains("not yet available"));
    }

    #[test]
    fn input_context_cannot_extend_past_availability() {
        let mut value = trace();
        value["events"][0]["input_chunk"]["context_end_ns"] = serde_json::json!(101);
        let error = audit(&value).expect_err("future context must fail");
        assert!(error.contains("causal ordering"));
    }

    #[test]
    fn response_inside_rest_is_ineligible_not_silently_accepted() {
        let mut value = trace();
        value["events"][4]["monotonic_ns"] = serde_json::json!(220);
        value["events"][4]["decision"]["state"] = serde_json::json!("respond");
        value["events"][4]["decision"]["deadline_ns"] = serde_json::json!(280);
        let report = audit(&value).expect("trace structure is still parseable");
        assert!(!report.qualified_for_declared_trace_contract);
        assert!(report.active_decision_during_rest_count > 0);
    }

    #[test]
    fn rest_without_wait_or_yield_is_reported_as_unqualified() {
        let mut value = trace();
        value["events"][4]["decision"]["state"] = serde_json::json!("continue");
        value["events"][4]["decision"]["deadline_ns"] = serde_json::json!(280);
        let report = audit(&value).expect("trace structure is parseable");
        assert!(!report.qualified_for_declared_trace_contract);
        assert_eq!(report.rest_windows_without_wait_or_yield, 1);
    }

    #[test]
    fn wait_must_not_emit_audio_or_midi() {
        let mut value = trace();
        value["events"][4]["decision"]["state"] = serde_json::json!("wait");
        value["events"][4]["decision"]["deadline_ns"] = serde_json::Value::Null;
        let mut output = value["events"][2].clone();
        output["event_id"] = serde_json::json!("illegal-output");
        output["sequence"] = serde_json::json!(6);
        output["monotonic_ns"] = serde_json::json!(260);
        output["output"]["decision_event_id"] = serde_json::json!("decision-rest");
        output["output"]["input_sequence"] = serde_json::json!(2);
        output["output"]["output_sequence"] = serde_json::json!(2);
        output["output"]["first_emitted_ns"] = serde_json::json!(250);
        value["events"].as_array_mut().unwrap().push(output);
        let error = audit(&value).expect_err("WAIT cannot emit");
        assert!(error.contains("output may only follow RESPOND or CONTINUE"));
    }

    #[test]
    fn duplicate_output_sequence_is_rejected() {
        let mut value = trace();
        let mut duplicate = value["events"][2].clone();
        duplicate["event_id"] = serde_json::json!("duplicate-output");
        duplicate["sequence"] = serde_json::json!(4);
        duplicate["monotonic_ns"] = serde_json::json!(160);
        value["events"].as_array_mut().unwrap().insert(3, duplicate);
        value["events"][4]["sequence"] = serde_json::json!(5);
        value["events"][5]["sequence"] = serde_json::json!(6);
        let error = audit(&value).expect_err("duplicate event sequence/output sequence must fail");
        assert!(error.contains("sequence") || error.contains("output_sequence"));
    }

    #[test]
    fn output_after_deadline_is_reported_as_late() {
        let mut value = trace();
        value["events"][1]["decision"]["deadline_ns"] = serde_json::json!(130);
        let report = audit(&value).expect("late output remains auditable");
        assert_eq!(report.late_output_count, 1);
        assert!(!report.qualified_for_declared_trace_contract);
    }

    #[test]
    fn audible_latency_requires_independent_loopback_receipt() {
        let mut value = trace();
        value["events"][2]["output"]["first_audible_onset_ns"] = serde_json::json!(145);
        let error = audit(&value).expect_err("audible time cannot be self-attested");
        assert!(error.contains("loopback measurement"));
    }

    #[test]
    fn loopback_clock_mismatch_is_rejected() {
        let mut value = trace();
        value["loopback_measurement"] = serde_json::json!({
            "measurement_id":"loopback-1",
            "clock_domain_id":"different-clock",
            "receipt_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        });
        value["events"][2]["output"]["first_audible_onset_ns"] = serde_json::json!(145);
        let error = audit(&value).expect_err("clock domains must match");
        assert!(error.contains("clock domain"));
    }

    #[test]
    fn incomplete_response_is_not_counted_as_pass() {
        let mut value = trace();
        value["events"].as_array_mut().unwrap().remove(2);
        value["events"][2]["sequence"] = serde_json::json!(3);
        value["events"][2]["monotonic_ns"] = serde_json::json!(220);
        let report = audit(&value).expect("missing output is an evaluation failure");
        assert_eq!(report.response_decisions_without_output, 1);
        assert!(!report.qualified_for_declared_trace_contract);
    }
}
