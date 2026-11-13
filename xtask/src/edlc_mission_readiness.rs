//! Mission-weighted EDLC standby-to-service contract audit.
//!
//! Validates declared mission profiles and non-compensatory device-level service
//! outcomes. Wilson bounds are per-stratum one-sided 95% lower bounds, not
//! simultaneous confidence bounds across all mission strata.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const FIXTURE_PATH: &str =
    "fixtures/energy/electrochemical-capacitor/mission_readiness_known_answer_v0_1.json";
const EXPECTED_FIXTURE_ID: &str = "EDLC-MISSION-READINESS-KNOWN-ANSWER-001";
const EXPECTED_EVIDENCE_CLASS: &str = "synthetic_mission_readiness_arithmetic";
const EXPECTED_CLAIM_CEILING: &str =
    "SYNTHETIC_MISSION_PROFILE_WEIGHTING_AND_PASS_RULES_ONLY_NOT_PHYSICAL_DEVICE_READINESS_OR_RELIABILITY";
const WEIGHTED_METHOD: &str = "sum_of_event_weight_times_stratum_service_pass_fraction";
const WILSON_Z_ONE_SIDED_95: f64 = 1.644_853_626_951_472_2;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    fixture_id: String,
    evidence_class: String,
    claim_ceiling: String,
    assumptions: Vec<String>,
    protocol: Protocol,
    strata: Vec<Stratum>,
    expected: Expected,
    negative_controls: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    protocol_id: String,
    standby_mode: String,
    physical_boundary: String,
    event_weight_source: String,
    event_weight_source_reference: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stratum {
    stratum_id: String,
    standby_duration_hours: f64,
    event_arrival_weight: f64,
    requirements: Requirements,
    observations: Vec<Observation>,
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
    maximum_external_replenishment_energy_wh: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    physical_device_serial: String,
    status: String,
    failure_reason: Option<String>,
    delivered_energy_wh: Option<f64>,
    minimum_sustained_power_w: Option<f64>,
    response_latency_s: Option<f64>,
    minimum_application_voltage_v: Option<f64>,
    temperature_rise_c: Option<f64>,
    standby_auxiliary_energy_wh: Option<f64>,
    external_replenishment_energy_wh: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    mission_weighted_service_pass_rate: f64,
    method: String,
}

#[derive(Debug, Clone, PartialEq)]
struct StratumReport {
    stratum_id: String,
    event_arrival_weight: f64,
    valid_devices: usize,
    measured_devices: usize,
    device_failures: usize,
    invalid_devices: usize,
    censored_devices: usize,
    service_passes: usize,
    service_pass_rate: f64,
    wilson_lower_95_one_sided: f64,
}

#[derive(Debug, PartialEq)]
struct AuditReport {
    strata: Vec<StratumReport>,
    mission_weighted_service_pass_rate: f64,
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

fn wilson_lower_one_sided_95(successes: usize, trials: usize) -> Result<f64> {
    if trials == 0 || successes > trials {
        bail!("Wilson interval requires 0 <= successes <= positive trials");
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = WILSON_Z_ONE_SIDED_95;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    Ok((center - half_width).clamp(0.0, 1.0))
}

fn measured_value(value: Option<f64>, context: &str) -> Result<f64> {
    let number =
        value.with_context(|| format!("{context} is required for measured observations"))?;
    finite_nonnegative(number, context)?;
    Ok(number)
}

fn service_passes(observation: &Observation, requirements: &Requirements) -> Result<bool> {
    let energy = measured_value(observation.delivered_energy_wh, "delivered_energy_wh")?;
    let power = measured_value(
        observation.minimum_sustained_power_w,
        "minimum_sustained_power_w",
    )?;
    let latency = measured_value(observation.response_latency_s, "response_latency_s")?;
    let voltage = measured_value(
        observation.minimum_application_voltage_v,
        "minimum_application_voltage_v",
    )?;
    let temperature = observation
        .temperature_rise_c
        .context("temperature_rise_c is required for measured observations")?;
    if !temperature.is_finite() {
        bail!("temperature_rise_c must be finite");
    }
    let auxiliary = measured_value(
        observation.standby_auxiliary_energy_wh,
        "standby_auxiliary_energy_wh",
    )?;
    let replenishment = measured_value(
        observation.external_replenishment_energy_wh,
        "external_replenishment_energy_wh",
    )?;

    Ok(energy >= requirements.minimum_delivered_energy_wh
        && power >= requirements.minimum_sustained_power_w
        && latency <= requirements.maximum_response_latency_s
        && voltage >= requirements.minimum_application_voltage_v
        && temperature <= requirements.maximum_temperature_rise_c
        && auxiliary <= requirements.maximum_standby_auxiliary_energy_wh
        && replenishment <= requirements.maximum_external_replenishment_energy_wh)
}

fn validate_requirements(requirements: &Requirements, context: &str) -> Result<()> {
    for (name, value) in [
        (
            "minimum_delivered_energy_wh",
            requirements.minimum_delivered_energy_wh,
        ),
        (
            "minimum_sustained_power_w",
            requirements.minimum_sustained_power_w,
        ),
        (
            "maximum_response_latency_s",
            requirements.maximum_response_latency_s,
        ),
        (
            "minimum_application_voltage_v",
            requirements.minimum_application_voltage_v,
        ),
        (
            "maximum_temperature_rise_c",
            requirements.maximum_temperature_rise_c,
        ),
        (
            "maximum_standby_auxiliary_energy_wh",
            requirements.maximum_standby_auxiliary_energy_wh,
        ),
        (
            "maximum_external_replenishment_energy_wh",
            requirements.maximum_external_replenishment_energy_wh,
        ),
    ] {
        finite_nonnegative(value, &format!("{context}.{name}"))?;
    }
    Ok(())
}

fn audit_fixture(fixture: &Fixture) -> Result<AuditReport> {
    if fixture.schema_version != "0.1.0" {
        bail!("unsupported mission-readiness fixture schema_version");
    }
    if fixture.fixture_id != EXPECTED_FIXTURE_ID {
        bail!("unexpected mission-readiness fixture_id");
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS {
        bail!("mission-readiness known-answer fixture must remain synthetic");
    }
    if fixture.claim_ceiling != EXPECTED_CLAIM_CEILING {
        bail!("mission-readiness fixture claim ceiling was weakened or changed");
    }
    if fixture.assumptions.is_empty() || fixture.negative_controls.is_empty() {
        bail!("fixture assumptions and negative_controls must both be non-empty");
    }
    for (index, value) in fixture.assumptions.iter().enumerate() {
        nonempty(value, &format!("assumptions[{index}]"))?;
    }
    for (index, value) in fixture.negative_controls.iter().enumerate() {
        nonempty(value, &format!("negative_controls[{index}]"))?;
    }

    let protocol = &fixture.protocol;
    nonempty(&protocol.protocol_id, "protocol.protocol_id")?;
    nonempty(
        &protocol.event_weight_source_reference,
        "protocol.event_weight_source_reference",
    )?;
    if !matches!(
        protocol.standby_mode.as_str(),
        "open_circuit" | "regulated_voltage_hold"
    ) {
        bail!("unsupported standby_mode; separate operating modes into separate profiles");
    }
    if !matches!(
        protocol.physical_boundary.as_str(),
        "complete_device" | "module" | "installed_system"
    ) {
        bail!("unsupported physical_boundary");
    }
    if !matches!(
        protocol.event_weight_source.as_str(),
        "observed_application_log"
            | "contractual_profile"
            | "engineering_assumption"
            | "synthetic_demo_only"
    ) {
        bail!("unsupported event_weight_source");
    }
    if protocol.event_weight_source == "synthetic_demo_only"
        && fixture.evidence_class != EXPECTED_EVIDENCE_CLASS
    {
        bail!("synthetic event weights cannot be represented as a physical mission profile");
    }

    if fixture.strata.is_empty() {
        bail!("mission-readiness fixture must contain at least one stratum");
    }
    let mut stratum_ids = BTreeSet::new();
    let mut physical_serials = BTreeSet::new();
    let mut weight_sum = 0.0_f64;
    let mut weighted_pass_rate = 0.0_f64;
    let mut reports = Vec::with_capacity(fixture.strata.len());

    for stratum in &fixture.strata {
        let context = format!("stratum.{}", stratum.stratum_id);
        nonempty(&stratum.stratum_id, "stratum.stratum_id")?;
        if !stratum_ids.insert(stratum.stratum_id.clone()) {
            bail!("duplicate mission stratum id {}", stratum.stratum_id);
        }
        finite_nonnegative(
            stratum.standby_duration_hours,
            &format!("{context}.standby_duration_hours"),
        )?;
        if stratum.standby_duration_hours <= 0.0 {
            bail!("{context}.standby_duration_hours must be positive");
        }
        finite_nonnegative(
            stratum.event_arrival_weight,
            &format!("{context}.event_arrival_weight"),
        )?;
        if stratum.event_arrival_weight <= 0.0 {
            bail!("{context}.event_arrival_weight must be positive");
        }
        weight_sum += stratum.event_arrival_weight;
        validate_requirements(&stratum.requirements, &context)?;

        if stratum.observations.is_empty() {
            bail!("{context} must contain observations");
        }
        let mut measured_devices = 0_usize;
        let mut device_failures = 0_usize;
        let mut invalid_devices = 0_usize;
        let mut censored_devices = 0_usize;
        let mut service_passes = 0_usize;

        for observation in &stratum.observations {
            nonempty(
                &observation.physical_device_serial,
                "observation.physical_device_serial",
            )?;
            if !physical_serials.insert(observation.physical_device_serial.clone()) {
                bail!(
                    "duplicate physical device serial {} across mission strata",
                    observation.physical_device_serial
                );
            }

            match observation.status.as_str() {
                "measured" => {
                    if observation
                        .failure_reason
                        .as_ref()
                        .is_some_and(|reason| !reason.trim().is_empty())
                    {
                        bail!("measured observation must not carry a failure_reason");
                    }
                    if protocol.standby_mode == "open_circuit"
                        && observation
                            .external_replenishment_energy_wh
                            .unwrap_or(f64::NAN)
                            > 1.0e-12
                    {
                        bail!(
                            "{}: open-circuit standby must have zero external replenishment energy",
                            observation.physical_device_serial
                        );
                    }
                    measured_devices += 1;
                    if service_passes(observation, &stratum.requirements)? {
                        service_passes += 1;
                    }
                }
                "failed" => {
                    let reason = observation
                        .failure_reason
                        .as_deref()
                        .context("failed observation requires failure_reason")?;
                    nonempty(reason, "observation.failure_reason")?;
                    device_failures += 1;
                }
                "invalid" => {
                    let reason = observation
                        .failure_reason
                        .as_deref()
                        .context("invalid observation requires failure_reason")?;
                    nonempty(reason, "observation.failure_reason")?;
                    invalid_devices += 1;
                }
                "censored" => {
                    let reason = observation
                        .failure_reason
                        .as_deref()
                        .context("censored observation requires failure_reason")?;
                    nonempty(reason, "observation.failure_reason")?;
                    censored_devices += 1;
                }
                other => bail!("{context}: unsupported observation status {other}"),
            }
        }

        let valid_devices = measured_devices + device_failures;
        if valid_devices == 0 {
            bail!("{context} has no valid measured or failed device outcomes");
        }
        let service_pass_rate = service_passes as f64 / valid_devices as f64;
        let lower_bound = wilson_lower_one_sided_95(service_passes, valid_devices)?;
        weighted_pass_rate += stratum.event_arrival_weight * service_pass_rate;

        reports.push(StratumReport {
            stratum_id: stratum.stratum_id.clone(),
            event_arrival_weight: stratum.event_arrival_weight,
            valid_devices,
            measured_devices,
            device_failures,
            invalid_devices,
            censored_devices,
            service_passes,
            service_pass_rate,
            wilson_lower_95_one_sided: lower_bound,
        });
    }

    if (weight_sum - 1.0).abs() > 1.0e-9 {
        bail!("event-arrival weights must sum to 1.0 (found {weight_sum:.12})");
    }
    if fixture.expected.method != WEIGHTED_METHOD {
        bail!("unsupported mission-weighting method");
    }
    finite_nonnegative(
        fixture.expected.mission_weighted_service_pass_rate,
        "expected.mission_weighted_service_pass_rate",
    )?;
    if fixture.expected.mission_weighted_service_pass_rate > 1.0 {
        bail!("expected mission-weighted service pass rate must not exceed 1");
    }
    let expected = fixture.expected.mission_weighted_service_pass_rate;
    let tolerance = 1.0e-12 + 1.0e-9 * weighted_pass_rate.abs().max(expected.abs());
    if (weighted_pass_rate - expected).abs() > tolerance {
        bail!(
            "mission-weighted service pass rate mismatch: calculated {weighted_pass_rate:.12}, expected {expected:.12}"
        );
    }

    Ok(AuditReport {
        strata: reports,
        mission_weighted_service_pass_rate: weighted_pass_rate,
    })
}

fn audit_value(value: serde_json::Value) -> Result<AuditReport> {
    let fixture: Fixture = serde_json::from_value(value)
        .context("mission-readiness fixture does not match its strict data contract")?;
    audit_fixture(&fixture)
}

pub fn audit(root: &Path) -> Result<()> {
    let path = root.join(FIXTURE_PATH);
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "could not read mission-readiness fixture at {}",
            path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "mission-readiness fixture at {} is not valid JSON",
            path.display()
        )
    })?;
    let report = audit_value(value)?;

    println!("edlc-mission-readiness-audit: PASS (synthetic mission semantics only)");
    for stratum in &report.strata {
        println!(
            "{}: weight={:.3}, pass={}/{}, rate={:.3}, one-sided-95%-Wilson-LB={:.3}, invalid={}, censored={}",
            stratum.stratum_id,
            stratum.event_arrival_weight,
            stratum.service_passes,
            stratum.valid_devices,
            stratum.service_pass_rate,
            stratum.wilson_lower_95_one_sided,
            stratum.invalid_devices,
            stratum.censored_devices
        );
    }
    println!(
        "mission-weighted service pass rate (descriptive point estimate only): {:.6}",
        report.mission_weighted_service_pass_rate
    );
    println!(
        "claim ceiling: synthetic profile weighting and conjunctive pass arithmetic only; no physical device readiness or reliability"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is one level below the workspace root")
            .to_path_buf()
    }

    fn fixture_value() -> Value {
        let bytes = std::fs::read(root().join(FIXTURE_PATH))
            .expect("checked-in mission-readiness fixture must be readable");
        serde_json::from_slice(&bytes)
            .expect("checked-in mission-readiness fixture must be valid JSON")
    }

    #[test]
    fn checked_in_fixture_matches_weighted_known_answer() {
        let report = audit_value(fixture_value()).expect("mission-readiness fixture should pass");
        assert!((report.mission_weighted_service_pass_rate - 0.54).abs() < 1.0e-12);
        assert_eq!(report.strata.len(), 3);
        assert_eq!(report.strata[0].service_passes, 4);
        assert_eq!(report.strata[1].service_passes, 3);
        assert_eq!(report.strata[2].service_passes, 2);
    }

    #[test]
    fn event_weights_must_sum_to_one() {
        let mut value = fixture_value();
        value["strata"][0]["event_arrival_weight"] = json!(0.5);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn a_physical_serial_cannot_be_reused_across_dwell_strata() {
        let mut value = fixture_value();
        let serial = value["strata"][0]["observations"][0]["physical_device_serial"].clone();
        value["strata"][1]["observations"][0]["physical_device_serial"] = serial;
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn open_circuit_mode_rejects_external_replenishment() {
        let mut value = fixture_value();
        value["strata"][0]["observations"][0]["external_replenishment_energy_wh"] =
            json!(0.001);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn service_pass_is_conjunctive_not_a_compensating_score() {
        let mut value = fixture_value();
        value["strata"][2]["observations"][0]["delivered_energy_wh"] = json!(100.0);
        value["strata"][2]["observations"][0]["minimum_sustained_power_w"] = json!(0.1);
        // One previously passing 30 d device now fails; the weighted rate drops by 0.5 * 0.2.
        value["expected"]["mission_weighted_service_pass_rate"] = json!(0.49);
        assert!(audit_value(value).is_ok());
    }

    #[test]
    fn a_missing_measured_metric_is_rejected() {
        let mut value = fixture_value();
        value["strata"][0]["observations"][0]
            .as_object_mut()
            .unwrap()
            .remove("minimum_sustained_power_w");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn invalid_and_censored_outcomes_are_preserved_but_not_counted_as_passes() {
        let mut value = fixture_value();
        value["strata"][0]["observations"].as_array_mut().unwrap().push(json!({
            "physical_device_serial": "SYNTH-24H-INVALID-001",
            "status": "invalid",
            "failure_reason": "Synthetic invalid-control"
        }));
        let report = audit_value(value).expect("invalid observation should be retained");
        assert_eq!(report.strata[0].invalid_devices, 1);
        assert_eq!(report.strata[0].valid_devices, 5);
    }

    #[test]
    fn wilson_bound_is_zero_with_no_successes() {
        assert_eq!(wilson_lower_one_sided_95(0, 5).unwrap(), 0.0);
    }

    #[test]
    fn wilson_lower_bound_increases_with_more_successes() {
        assert!(
            wilson_lower_one_sided_95(4, 5).unwrap()
                > wilson_lower_one_sided_95(3, 5).unwrap()
        );
    }

    #[test]
    fn wrong_event_weight_provenance_is_rejected() {
        let mut value = fixture_value();
        value["protocol"]["event_weight_source"] = json!("universal_default");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut value = fixture_value();
        value["protocol"]["invented_probability"] = json!(0.5);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn weakened_claim_ceiling_is_rejected() {
        let mut value = fixture_value();
        value["claim_ceiling"] = json!("PHYSICAL_READINESS_CONFIRMED");
        assert!(audit_value(value).is_err());
    }
}
