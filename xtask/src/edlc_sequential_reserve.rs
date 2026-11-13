//! Sequential EDLC reserve-service replay audit.
//!
//! A device passes only when every ordered call and each inter-call recovery
//! gate passes. Synthetic fixtures validate arithmetic and data contracts only.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const FIXTURE_PATH: &str =
    "fixtures/energy/electrochemical-capacitor/sequential_reserve_known_answer_v0_1.json";
const SCHEMA_PATH: &str = "schemas/energy/edlc_sequential_reserve_v0_1.schema.json";
const EXPECTED_FIXTURE_ID: &str = "EDLC-SEQUENTIAL-RESERVE-KNOWN-ANSWER-001";
const EXPECTED_EVIDENCE_CLASS: &str = "synthetic_sequential_reserve_replay";
const EXPECTED_CLAIM_CEILING: &str =
    "SYNTHETIC_SEQUENCE_AND_RECOVERY_GATE_ARITHMETIC_ONLY_NOT_PHYSICAL_DEVICE_RELIABILITY";
const EXPECTED_METHOD: &str =
    "non_compensatory_all_events_and_recovery_gates_per_device_sequence";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    fixture_id: String,
    evidence_class: String,
    claim_ceiling: String,
    assumptions: Vec<String>,
    protocol: Protocol,
    requirements: Requirements,
    device_sequences: Vec<DeviceSequence>,
    expected: Expected,
    negative_controls: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    protocol_id: String,
    sequence_profile_id: String,
    standby_mode: String,
    physical_boundary: String,
    initial_standby_hours: f64,
    event_order_frozen_before_evaluation: bool,
    external_replenishment_boundary: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Requirements {
    minimum_delivered_energy_wh: f64,
    minimum_sustained_power_w: f64,
    maximum_response_latency_s: f64,
    minimum_application_voltage_v: f64,
    maximum_temperature_rise_c: f64,
    maximum_standby_auxiliary_energy_wh: f64,
    maximum_external_replenishment_energy_wh_per_interval: f64,
    maximum_time_to_restored_readiness_hours: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceSequence {
    physical_device_serial: String,
    status: String,
    failure_reason: Option<String>,
    events: Vec<Event>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    event_index: usize,
    event_id: String,
    load_profile_id: String,
    readiness_probe_profile_id: Option<String>,
    preceding_idle_window_hours: f64,
    readiness_probe_status: String,
    time_to_restored_readiness_hours: Option<f64>,
    external_replenishment_energy_wh: f64,
    standby_auxiliary_energy_wh: f64,
    delivered_energy_wh: f64,
    minimum_sustained_power_w: f64,
    response_latency_s: f64,
    minimum_application_voltage_v: f64,
    temperature_rise_c: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    sequence_pass_rate: f64,
    sequence_passes: usize,
    valid_sequences: usize,
    false_ready_events: usize,
    method: String,
}

#[derive(Debug, Clone, PartialEq)]
struct EventReport {
    event_index: usize,
    event_id: String,
    observed_sequences: usize,
    service_passes: usize,
    interval_budget_passes: usize,
    recovery_applicable_sequences: usize,
    recovery_passes: usize,
    event_gate_passes: usize,
    false_ready_events: usize,
}

#[derive(Debug, PartialEq)]
struct AuditReport {
    sequence_passes: usize,
    valid_sequences: usize,
    invalid_sequences: usize,
    censored_sequences: usize,
    sequence_pass_rate: f64,
    false_ready_events: usize,
    events: Vec<EventReport>,
}

fn nonempty(value: &str, context: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{context} must not be empty");
    }
    Ok(())
}

fn finite_nonnegative(value: f64, context: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("{context} must be finite and non-negative");
    }
    Ok(())
}

fn close_enough(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}

fn validate_requirements(requirements: &Requirements) -> Result<()> {
    for (name, value) in [
        ("minimum_delivered_energy_wh", requirements.minimum_delivered_energy_wh),
        ("minimum_sustained_power_w", requirements.minimum_sustained_power_w),
        ("maximum_response_latency_s", requirements.maximum_response_latency_s),
        ("minimum_application_voltage_v", requirements.minimum_application_voltage_v),
        ("maximum_standby_auxiliary_energy_wh", requirements.maximum_standby_auxiliary_energy_wh),
        (
            "maximum_external_replenishment_energy_wh_per_interval",
            requirements.maximum_external_replenishment_energy_wh_per_interval,
        ),
        (
            "maximum_time_to_restored_readiness_hours",
            requirements.maximum_time_to_restored_readiness_hours,
        ),
    ] {
        finite_nonnegative(value, name)?;
    }
    if !requirements.maximum_temperature_rise_c.is_finite() {
        bail!("maximum_temperature_rise_c must be finite");
    }
    Ok(())
}

fn validate_event_values(event: &Event, context: &str) -> Result<()> {
    nonempty(&event.event_id, &format!("{context}.event_id"))?;
    nonempty(&event.load_profile_id, &format!("{context}.load_profile_id"))?;
    for (name, value) in [
        ("preceding_idle_window_hours", event.preceding_idle_window_hours),
        ("external_replenishment_energy_wh", event.external_replenishment_energy_wh),
        ("standby_auxiliary_energy_wh", event.standby_auxiliary_energy_wh),
        ("delivered_energy_wh", event.delivered_energy_wh),
        ("minimum_sustained_power_w", event.minimum_sustained_power_w),
        ("response_latency_s", event.response_latency_s),
        ("minimum_application_voltage_v", event.minimum_application_voltage_v),
    ] {
        finite_nonnegative(value, &format!("{context}.{name}"))?;
    }
    if event.preceding_idle_window_hours <= 0.0 {
        bail!("{context}.preceding_idle_window_hours must be positive");
    }
    if !event.temperature_rise_c.is_finite() {
        bail!("{context}.temperature_rise_c must be finite");
    }
    if let Some(profile_id) = event.readiness_probe_profile_id.as_deref() {
        nonempty(profile_id, &format!("{context}.readiness_probe_profile_id"))?;
    }
    if let Some(hours) = event.time_to_restored_readiness_hours {
        finite_nonnegative(hours, &format!("{context}.time_to_restored_readiness_hours"))?;
    }
    Ok(())
}

fn service_passes(event: &Event, requirements: &Requirements) -> bool {
    event.delivered_energy_wh >= requirements.minimum_delivered_energy_wh
        && event.minimum_sustained_power_w >= requirements.minimum_sustained_power_w
        && event.response_latency_s <= requirements.maximum_response_latency_s
        && event.minimum_application_voltage_v >= requirements.minimum_application_voltage_v
        && event.temperature_rise_c <= requirements.maximum_temperature_rise_c
}

fn audit_fixture(fixture: &Fixture) -> Result<AuditReport> {
    if fixture.schema_version != "0.1.0" {
        bail!("unsupported sequential-reserve fixture schema_version");
    }
    if fixture.fixture_id != EXPECTED_FIXTURE_ID {
        bail!("unexpected sequential-reserve fixture_id");
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS {
        bail!("sequential-reserve known-answer fixture must remain synthetic");
    }
    if fixture.claim_ceiling != EXPECTED_CLAIM_CEILING {
        bail!("sequential-reserve claim ceiling was weakened or changed");
    }
    if fixture.assumptions.is_empty() || fixture.negative_controls.is_empty() {
        bail!("assumptions and negative_controls must both be non-empty");
    }
    for (index, assumption) in fixture.assumptions.iter().enumerate() {
        nonempty(assumption, &format!("assumptions[{index}]"))?;
    }
    for (index, control) in fixture.negative_controls.iter().enumerate() {
        nonempty(control, &format!("negative_controls[{index}]"))?;
    }

    let protocol = &fixture.protocol;
    nonempty(&protocol.protocol_id, "protocol.protocol_id")?;
    nonempty(&protocol.sequence_profile_id, "protocol.sequence_profile_id")?;
    nonempty(
        &protocol.external_replenishment_boundary,
        "protocol.external_replenishment_boundary",
    )?;
    if !matches!(protocol.standby_mode.as_str(), "open_circuit" | "regulated_voltage_hold") {
        bail!("unsupported standby_mode; separate operating modes into separate fixtures");
    }
    if !matches!(
        protocol.physical_boundary.as_str(),
        "cell" | "complete_device" | "installed_system"
    ) {
        bail!("unsupported physical_boundary");
    }
    finite_nonnegative(protocol.initial_standby_hours, "protocol.initial_standby_hours")?;
    if protocol.initial_standby_hours <= 0.0 {
        bail!("protocol.initial_standby_hours must be positive");
    }
    if !protocol.event_order_frozen_before_evaluation {
        bail!("event order must be frozen before evaluation");
    }
    validate_requirements(&fixture.requirements)?;

    if fixture.device_sequences.is_empty() {
        bail!("at least one device sequence is required");
    }

    let mut serials = BTreeSet::new();
    let mut reference_signature: Option<Vec<(usize, String, String, Option<String>, f64)>> = None;
    let mut sequence_passes_count = 0_usize;
    let mut valid_sequences = 0_usize;
    let mut invalid_sequences = 0_usize;
    let mut censored_sequences = 0_usize;
    let mut false_ready_events = 0_usize;
    let mut event_reports: Vec<EventReport> = Vec::new();

    for sequence in &fixture.device_sequences {
        nonempty(&sequence.physical_device_serial, "physical_device_serial")?;
        if !serials.insert(sequence.physical_device_serial.clone()) {
            bail!(
                "duplicate physical device serial {}",
                sequence.physical_device_serial
            );
        }

        match sequence.status.as_str() {
            "invalid" => {
                let reason = sequence.failure_reason.as_deref().context(
                    "invalid sequence requires failure_reason",
                )?;
                nonempty(reason, "sequence.failure_reason")?;
                if !sequence.events.is_empty() {
                    bail!("invalid sequence must not carry scored events");
                }
                invalid_sequences += 1;
                continue;
            }
            "censored" => {
                let reason = sequence.failure_reason.as_deref().context(
                    "censored sequence requires failure_reason",
                )?;
                nonempty(reason, "sequence.failure_reason")?;
                if !sequence.events.is_empty() {
                    bail!("censored sequence must not carry scored events");
                }
                censored_sequences += 1;
                continue;
            }
            "failed" => {
                let reason = sequence.failure_reason.as_deref().context(
                    "failed sequence requires failure_reason",
                )?;
                nonempty(reason, "sequence.failure_reason")?;
                if !sequence.events.is_empty() {
                    bail!("failed sequence must not carry scored events; use measured events for traceable failures");
                }
                valid_sequences += 1;
                continue;
            }
            "measured" => {
                if sequence
                    .failure_reason
                    .as_ref()
                    .is_some_and(|reason| !reason.trim().is_empty())
                {
                    bail!("measured sequence must not carry failure_reason");
                }
            }
            other => bail!("unsupported device sequence status {other}"),
        }

        valid_sequences += 1;
        if sequence.events.len() < 2 {
            bail!(
                "measured sequence {} must contain at least two ordered calls",
                sequence.physical_device_serial
            );
        }

        let mut signature = Vec::with_capacity(sequence.events.len());
        let mut sequence_passes = true;

        for (position, event) in sequence.events.iter().enumerate() {
            let context = format!(
                "sequence.{}.event.{}",
                sequence.physical_device_serial, event.event_index
            );
            validate_event_values(event, &context)?;

            if event.event_index != position + 1 {
                bail!("{context}: event indexes must be contiguous and start at one");
            }

            if protocol.standby_mode == "open_circuit"
                && event.external_replenishment_energy_wh > 1.0e-12
            {
                bail!(
                    "{context}: open-circuit standby must have zero external replenishment energy"
                );
            }

            signature.push((
                event.event_index,
                event.event_id.clone(),
                event.load_profile_id.clone(),
                event.readiness_probe_profile_id.clone(),
                event.preceding_idle_window_hours,
            ));

            let first_event = position == 0;
            if first_event {
                if !close_enough(
                    event.preceding_idle_window_hours,
                    protocol.initial_standby_hours,
                ) {
                    bail!("{context}: first event idle window must equal initial_standby_hours");
                }
                if event.readiness_probe_status != "not_applicable"
                    || event.readiness_probe_profile_id.is_some()
                    || event.time_to_restored_readiness_hours.is_some()
                {
                    bail!("{context}: first event must use not_applicable readiness, null probe profile and null recovery time");
                }
            } else if !matches!(
                event.readiness_probe_status.as_str(),
                "passed" | "failed"
            ) || event.readiness_probe_profile_id.is_none()
                || event.time_to_restored_readiness_hours.is_none()
            {
                bail!("{context}: later events require passed/failed readiness probe, a probe profile ID and measured recovery time");
            }

            let service_pass = service_passes(event, &fixture.requirements);
            let interval_budget_pass = event.standby_auxiliary_energy_wh
                <= fixture.requirements.maximum_standby_auxiliary_energy_wh
                && event.external_replenishment_energy_wh
                    <= fixture.requirements.maximum_external_replenishment_energy_wh_per_interval;
            let recovery_pass = if first_event {
                false
            } else {
                let recovered_hours = event
                    .time_to_restored_readiness_hours
                    .context("later event recovery time unexpectedly missing")?;
                event.readiness_probe_status == "passed"
                    && recovered_hours
                        <= fixture.requirements.maximum_time_to_restored_readiness_hours
                    && recovered_hours <= event.preceding_idle_window_hours
            };

            // The first call has no preceding inter-call recovery gate. It
            // still must satisfy the initial-standby energy budget.
            let event_gate_pass =
                service_pass && interval_budget_pass && (first_event || recovery_pass);
            if !first_event && event.readiness_probe_status == "passed" && !service_pass {
                false_ready_events += 1;
            }
            if !event_gate_pass {
                sequence_passes = false;
            }

            if event_reports.len() < sequence.events.len() {
                event_reports.push(EventReport {
                    event_index: event.event_index,
                    event_id: event.event_id.clone(),
                    observed_sequences: 1,
                    service_passes: if service_pass { 1 } else { 0 },
                    interval_budget_passes: if interval_budget_pass { 1 } else { 0 },
                    recovery_applicable_sequences: if first_event { 0 } else { 1 },
                    recovery_passes: if !first_event && recovery_pass { 1 } else { 0 },
                    event_gate_passes: if event_gate_pass { 1 } else { 0 },
                    false_ready_events: if !first_event && event.readiness_probe_status == "passed" && !service_pass { 1 } else { 0 },
                });
            } else {
                let report = event_reports
                    .get_mut(position)
                    .context("event report position is missing")?;
                if report.event_id != event.event_id || report.event_index != event.event_index {
                    bail!("event order or event identity differs across measured sequences");
                }
                report.observed_sequences += 1;
                report.service_passes += if service_pass { 1 } else { 0 };
                report.interval_budget_passes += if interval_budget_pass { 1 } else { 0 };
                report.recovery_applicable_sequences += if first_event { 0 } else { 1 };
                report.recovery_passes += if !first_event && recovery_pass { 1 } else { 0 };
                report.event_gate_passes += if event_gate_pass { 1 } else { 0 };
                report.false_ready_events += if !first_event && event.readiness_probe_status == "passed" && !service_pass { 1 } else { 0 };
            }
        }

        if let Some(reference) = &reference_signature {
            if reference.len() != signature.len()
                || reference.iter().zip(signature.iter()).any(|(left, right)| {
                    left.0 != right.0
                        || left.1 != right.1
                        || left.2 != right.2
                        || left.3 != right.3
                        || !close_enough(left.4, right.4)
                })
            {
                bail!("measured device sequences do not share identical event order, load-profile IDs, readiness-probe IDs and idle windows");
            }
        } else {
            reference_signature = Some(signature);
        }

        if let Some(reference) = &reference_signature {
            if reference.len() != sequence.events.len() {
                bail!("measured device sequences have inconsistent event counts");
            }
        }

        if sequence_passes {
            sequence_passes_count += 1;
        }
    }

    if valid_sequences == 0 {
        bail!("no valid measured or failed device sequences remain");
    }
    if reference_signature.is_none() {
        bail!("at least one fully measured sequence is required to define the frozen event profile");
    }

    let sequence_pass_rate = sequence_passes_count as f64 / valid_sequences as f64;
    let expected = &fixture.expected;
    if expected.method != EXPECTED_METHOD {
        bail!("unsupported sequential-reserve scoring method");
    }
    finite_nonnegative(expected.sequence_pass_rate, "expected.sequence_pass_rate")?;
    if expected.sequence_pass_rate > 1.0 {
        bail!("expected.sequence_pass_rate must not exceed 1");
    }
    if expected.valid_sequences != valid_sequences
        || expected.sequence_passes != sequence_passes_count
        || expected.false_ready_events != false_ready_events
        || !close_enough(expected.sequence_pass_rate, sequence_pass_rate)
    {
        bail!(
            "sequential-reserve known-answer mismatch: pass={sequence_passes_count}/{valid_sequences}, rate={sequence_pass_rate:.12}, false-ready={false_ready_events}"
        );
    }

    Ok(AuditReport {
        sequence_passes: sequence_passes_count,
        valid_sequences,
        invalid_sequences,
        censored_sequences,
        sequence_pass_rate,
        false_ready_events,
        events: event_reports,
    })
}

fn audit_schema(root: &Path) -> Result<()> {
    let path = root.join(SCHEMA_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read sequential-reserve schema at {}", path.display()))?;
    let schema: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("sequential-reserve schema at {} is not valid JSON", path.display()))?;
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        bail!("sequential-reserve schema must declare JSON Schema draft 2020-12");
    }
    if schema["properties"]["device_sequences"]["items"]["$ref"].as_str()
        != Some("#/$defs/deviceSequence")
    {
        bail!("device_sequences must reference the canonical deviceSequence schema");
    }
    if schema["$defs"]["event"]["properties"]["load_profile_id"]["type"].as_str()
        != Some("string")
    {
        bail!("event schema must require a load_profile_id field");
    }
    let top_required = schema["required"]
        .as_array()
        .context("sequential-reserve schema has no top-level required fields")?;
    for field in [
        "schema_version",
        "fixture_id",
        "evidence_class",
        "claim_ceiling",
        "assumptions",
        "protocol",
        "requirements",
        "device_sequences",
        "expected",
        "negative_controls",
    ] {
        if !top_required.iter().any(|entry| entry.as_str() == Some(field)) {
            bail!("sequential-reserve schema must require top-level field {field}");
        }
    }
    let event_required = schema["$defs"]["event"]["required"]
        .as_array()
        .context("event schema has no required-field list")?;
    for field in ["event_index", "event_id", "load_profile_id", "readiness_probe_profile_id", "readiness_probe_status", "time_to_restored_readiness_hours"] {
        if !event_required.iter().any(|entry| entry.as_str() == Some(field)) {
            bail!("event schema must require {field}");
        }
    }
    Ok(())
}

fn audit_value(value: serde_json::Value) -> Result<AuditReport> {
    let fixture: Fixture = serde_json::from_value(value)
        .context("sequential-reserve fixture does not match its strict data contract")?;
    audit_fixture(&fixture)
}

pub fn audit(root: &Path) -> Result<()> {
    audit_schema(root)?;
    let path = root.join(FIXTURE_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read sequential-reserve fixture at {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("sequential-reserve fixture at {} is not valid JSON", path.display()))?;
    let report = audit_value(value)?;

    println!("edlc-sequential-reserve-audit: PASS (synthetic sequence semantics only)");
    println!(
        "complete-device sequence pass rate (descriptive point estimate only): {}/{} = {:.6}",
        report.sequence_passes, report.valid_sequences, report.sequence_pass_rate
    );
    println!(
        "invalid sequences={}, censored sequences={}, false-ready events={}",
        report.invalid_sequences, report.censored_sequences, report.false_ready_events
    );
    for event in &report.events {
        let recovery = if event.recovery_applicable_sequences == 0 {
            "n/a".to_string()
        } else {
            format!(
                "{}/{}",
                event.recovery_passes, event.recovery_applicable_sequences
            )
        };
        println!(
            "event {} {}: service={}/{}, interval-budget={}/{}, recovery={}, joint-gate={}/{}, false-ready={}",
            event.event_index,
            event.event_id,
            event.service_passes,
            event.observed_sequences,
            event.interval_budget_passes,
            event.observed_sequences,
            recovery,
            event.event_gate_passes,
            event.observed_sequences,
            event.false_ready_events
        );
    }
    println!(
        "claim ceiling: synthetic sequence and recovery-gate arithmetic only; no physical device reliability"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::{Value, json};

    fn fixture_value() -> Value {
        json!({
            "schema_version":"0.1.0",
            "fixture_id":EXPECTED_FIXTURE_ID,
            "evidence_class":EXPECTED_EVIDENCE_CLASS,
            "claim_ceiling":EXPECTED_CLAIM_CEILING,
            "assumptions":["synthetic only"],
            "protocol":{
                "protocol_id":"TEST-SEQ-001",
                "sequence_profile_id":"TEST-THREE-CALLS",
                "standby_mode":"open_circuit",
                "physical_boundary":"complete_device",
                "initial_standby_hours":168.0,
                "event_order_frozen_before_evaluation":true,
                "external_replenishment_boundary":"source side"
            },
            "requirements":{
                "minimum_delivered_energy_wh":0.8,
                "minimum_sustained_power_w":1.0,
                "maximum_response_latency_s":0.8,
                "minimum_application_voltage_v":1.0,
                "maximum_temperature_rise_c":10.0,
                "maximum_standby_auxiliary_energy_wh":0.1,
                "maximum_external_replenishment_energy_wh_per_interval":0.02,
                "maximum_time_to_restored_readiness_hours":4.0
            },
            "device_sequences":[
                {"physical_device_serial":"A","status":"measured","events":[
                    {"event_index":1,"event_id":"E1","load_profile_id":"L1","preceding_idle_window_hours":168.0,"readiness_probe_status":"not_applicable","readiness_probe_profile_id":null,"time_to_restored_readiness_hours":null,"external_replenishment_energy_wh":0.0,"standby_auxiliary_energy_wh":0.02,"delivered_energy_wh":1.0,"minimum_sustained_power_w":1.2,"response_latency_s":0.4,"minimum_application_voltage_v":1.1,"temperature_rise_c":2.0},
                    {"event_index":2,"event_id":"E2","load_profile_id":"L2","preceding_idle_window_hours":6.0,"readiness_probe_status":"passed","readiness_probe_profile_id":"TEST-READINESS-PROBE","time_to_restored_readiness_hours":2.0,"external_replenishment_energy_wh":0.0,"standby_auxiliary_energy_wh":0.02,"delivered_energy_wh":0.9,"minimum_sustained_power_w":1.2,"response_latency_s":0.4,"minimum_application_voltage_v":1.1,"temperature_rise_c":2.0}
                ]},
                {"physical_device_serial":"B","status":"measured","events":[
                    {"event_index":1,"event_id":"E1","load_profile_id":"L1","preceding_idle_window_hours":168.0,"readiness_probe_status":"not_applicable","readiness_probe_profile_id":null,"time_to_restored_readiness_hours":null,"external_replenishment_energy_wh":0.0,"standby_auxiliary_energy_wh":0.02,"delivered_energy_wh":1.0,"minimum_sustained_power_w":1.2,"response_latency_s":0.4,"minimum_application_voltage_v":1.1,"temperature_rise_c":2.0},
                    {"event_index":2,"event_id":"E2","load_profile_id":"L2","preceding_idle_window_hours":6.0,"readiness_probe_status":"passed","readiness_probe_profile_id":"TEST-READINESS-PROBE","time_to_restored_readiness_hours":5.0,"external_replenishment_energy_wh":0.0,"standby_auxiliary_energy_wh":0.02,"delivered_energy_wh":0.9,"minimum_sustained_power_w":1.2,"response_latency_s":0.4,"minimum_application_voltage_v":1.1,"temperature_rise_c":2.0}
                ]}
            ],
            "expected":{"sequence_pass_rate":0.5,"sequence_passes":1,"valid_sequences":2,"false_ready_events":0,"method":EXPECTED_METHOD},
            "negative_controls":["later failures cannot be averaged away"]
        })
    }

    #[test]
    fn known_answer_detects_late_recovery_and_sequence_failure() {
        let fixture: Fixture = serde_json::from_value(fixture_value()).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.sequence_passes, 1);
        assert_eq!(report.valid_sequences, 2);
        assert_eq!(report.sequence_pass_rate, 0.5);
        assert_eq!(report.events[0].recovery_applicable_sequences, 0);
        assert_eq!(report.events[0].recovery_passes, 0);
        assert_eq!(report.events[0].interval_budget_passes, 2);
        assert_eq!(report.events[1].recovery_applicable_sequences, 2);
        assert_eq!(report.events[1].recovery_passes, 1);
        assert_eq!(report.events[1].event_gate_passes, 1);
    }

    #[test]
    fn sequence_requires_at_least_two_calls() {
        let mut value = fixture_value();
        let _ = value["device_sequences"][0]["events"].as_array_mut().unwrap().pop();
        let _ = value["device_sequences"][1]["events"].as_array_mut().unwrap().pop();
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("at least two ordered calls"));
    }

    #[test]
    fn initial_standby_auxiliary_budget_is_non_compensatory() {
        let mut value = fixture_value();
        value["device_sequences"][0]["events"][0]["standby_auxiliary_energy_wh"] = json!(0.2);
        value["expected"]["sequence_pass_rate"] = json!(0.0);
        value["expected"]["sequence_passes"] = json!(0);
        let fixture: Fixture = serde_json::from_value(value).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.sequence_passes, 0);
        assert_eq!(report.events[0].interval_budget_passes, 1);
    }

    #[test]
    fn first_event_requires_initial_standby_not_applicable_probe() {
        let mut value = fixture_value();
        value["device_sequences"][0]["events"][0]["readiness_probe_status"] = json!("passed");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("first event"));
    }

    #[test]
    fn open_circuit_replenishment_is_a_protocol_integrity_error() {
        let mut value = fixture_value();
        value["device_sequences"][0]["events"][1]["external_replenishment_energy_wh"] = json!(0.001);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("open-circuit"));
    }

    #[test]
    fn duplicate_serials_are_rejected() {
        let mut value = fixture_value();
        value["device_sequences"][1]["physical_device_serial"] = json!("A");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("duplicate physical device serial"));
    }

    #[test]
    fn inconsistent_load_profile_identity_is_rejected() {
        let mut value = fixture_value();
        value["device_sequences"][1]["events"][1]["load_profile_id"] = json!("OTHER");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("identical event order"));
    }

    #[test]
    fn inconsistent_readiness_probe_identity_is_rejected() {
        let mut value = fixture_value();
        value["device_sequences"][1]["events"][1]["readiness_probe_profile_id"] = json!("OTHER-PROBE");
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("readiness-probe IDs"));
    }

    #[test]
    fn event_order_must_be_contiguous() {
        let mut value = fixture_value();
        value["device_sequences"][0]["events"][1]["event_index"] = json!(3);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("contiguous"));
    }

    #[test]
    fn failed_sequences_count_as_valid_service_failures() {
        let mut value = fixture_value();
        value["device_sequences"][1]["status"] = json!("failed");
        value["device_sequences"][1]["failure_reason"] = json!("device failed during sequence");
        value["device_sequences"][1]["events"] = json!([]);
        value["expected"]["sequence_pass_rate"] = json!(0.5);
        let fixture: Fixture = serde_json::from_value(value).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.valid_sequences, 2);
        assert_eq!(report.sequence_passes, 1);
    }

    #[test]
    fn invalid_and_censored_sequences_are_excluded_but_visible() {
        let mut value = fixture_value();
        value["device_sequences"][1]["status"] = json!("invalid");
        value["device_sequences"][1]["failure_reason"] = json!("instrument sync lost");
        value["device_sequences"][1]["events"] = json!([]);
        value["expected"]["sequence_pass_rate"] = json!(1.0);
        value["expected"]["sequence_passes"] = json!(1);
        value["expected"]["valid_sequences"] = json!(1);
        let fixture: Fixture = serde_json::from_value(value).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.invalid_sequences, 1);
        assert_eq!(report.valid_sequences, 1);
        assert_eq!(report.sequence_pass_rate, 1.0);
    }

    #[test]
    fn readiness_probe_false_ready_is_counted() {
        let mut value = fixture_value();
        value["device_sequences"][0]["events"][1]["delivered_energy_wh"] = json!(0.2);
        value["device_sequences"][0]["events"][1]["readiness_probe_status"] = json!("passed");
        value["expected"]["sequence_pass_rate"] = json!(0.0);
        value["expected"]["sequence_passes"] = json!(0);
        value["expected"]["false_ready_events"] = json!(1);
        let fixture: Fixture = serde_json::from_value(value).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.false_ready_events, 1);
        assert_eq!(report.sequence_passes, 0);
    }

    #[test]
    fn replenishment_budget_is_non_compensatory_for_regulated_hold() {
        let mut value = fixture_value();
        value["protocol"]["standby_mode"] = json!("regulated_voltage_hold");
        value["device_sequences"][0]["events"][1]["external_replenishment_energy_wh"] = json!(0.03);
        value["expected"]["sequence_pass_rate"] = json!(0.0);
        value["expected"]["sequence_passes"] = json!(0);
        let fixture: Fixture = serde_json::from_value(value).unwrap();
        let report = audit_fixture(&fixture).unwrap();
        assert_eq!(report.sequence_passes, 0);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value = fixture_value();
        value["unreviewed_field"] = json!(true);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("strict data contract"));
    }

    #[test]
    fn expected_result_cannot_be_weakened_or_misreported() {
        let mut value = fixture_value();
        value["expected"]["sequence_pass_rate"] = json!(0.9);
        let error = audit_value(value).unwrap_err().to_string();
        assert!(error.contains("known-answer mismatch"));
    }
}
