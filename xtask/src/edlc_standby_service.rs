//! Executable known-answer audit for EDLC standby-service trace arithmetic.
//!
//! This audit checks units, sample integrity, trapezoidal leakage-charge integration,
//! terminal V×I energy integration, and the fixture's explicit synthetic-evidence ceiling.
//! It is not a device model, lifetime predictor, laboratory controller, or physical test.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

const FIXTURE_PATH: &str =
    "fixtures/energy/electrochemical-capacitor/standby_service_known_answer_v0_1.json";
const EXPECTED_FIXTURE_ID: &str = "EDLC-STANDBY-SERVICE-KNOWN-ANSWER-001";
const EXPECTED_EVIDENCE_CLASS: &str = "synthetic_known_answer_arithmetic";
const EXPECTED_CLAIM_CEILING: &str =
    "TRACE_INTEGRATION_AND_UNIT_ARITHMETIC_ONLY_NOT_A_PHYSICAL_EDLC_MEASUREMENT_OR_LIFETIME_MODEL";
const LEAKAGE_METHOD: &str = "trapezoidal_integral_current_over_time";
const DISCHARGE_METHOD: &str =
    "trapezoidal_integral_of_simultaneous_terminal_voltage_times_current";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    fixture_id: String,
    evidence_class: String,
    claim_ceiling: String,
    assumptions: Vec<String>,
    leakage_trace: LeakageTrace,
    post_standby_discharge_trace: DischargeTrace,
    negative_controls: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LeakageTrace {
    quantity: String,
    unit: String,
    time_unit: String,
    samples: Vec<LeakageSample>,
    expected: LeakageExpected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LeakageSample {
    t: f64,
    current: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LeakageExpected {
    #[serde(rename = "integrated_charge_C")]
    integrated_charge_c: f64,
    method: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DischargeTrace {
    quantity: String,
    voltage_unit: String,
    current_unit: String,
    time_unit: String,
    samples: Vec<DischargeSample>,
    expected: DischargeExpected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DischargeSample {
    t: f64,
    voltage: f64,
    current: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DischargeExpected {
    #[serde(rename = "delivered_energy_J")]
    delivered_energy_j: f64,
    #[serde(rename = "delivered_energy_Wh")]
    delivered_energy_wh: f64,
    #[serde(rename = "peak_sampled_power_W")]
    peak_sampled_power_w: f64,
    method: String,
}

#[derive(Debug, PartialEq)]
struct AuditReport {
    integrated_leakage_charge_c: f64,
    delivered_energy_j: f64,
    delivered_energy_wh: f64,
    peak_sampled_power_w: f64,
}

fn require_nonempty(value: &str, context: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{context} must not be empty");
    }
    Ok(())
}

fn validate_samples(times_and_values: &[(f64, f64)], context: &str) -> Result<()> {
    if times_and_values.len() < 2 {
        bail!("{context} requires at least two samples");
    }

    let mut previous_time = None;
    for (index, (time, value)) in times_and_values.iter().copied().enumerate() {
        if !time.is_finite() || time < 0.0 {
            bail!("{context}[{index}].t must be finite and non-negative");
        }
        if !value.is_finite() || value < 0.0 {
            bail!("{context}[{index}] measurement must be finite and non-negative");
        }
        if previous_time.is_some_and(|previous| time <= previous) {
            bail!("{context} timestamps must be strictly increasing");
        }
        previous_time = Some(time);
    }
    Ok(())
}

fn trapezoidal_integral(times_and_values: &[(f64, f64)], context: &str) -> Result<f64> {
    validate_samples(times_and_values, context)?;
    let mut total = 0.0_f64;
    for pair in times_and_values.windows(2) {
        let (t0, y0) = pair[0];
        let (t1, y1) = pair[1];
        let area = (t1 - t0) * (y0 + y1) * 0.5;
        if !area.is_finite() || area < 0.0 {
            bail!("{context} produced a non-finite or negative trapezoid");
        }
        total += area;
        if !total.is_finite() {
            bail!("{context} integral overflowed");
        }
    }
    Ok(total)
}

fn require_close(name: &str, actual: f64, expected: f64) -> Result<()> {
    if !expected.is_finite() || expected < 0.0 {
        bail!("{name} expected value must be finite and non-negative");
    }
    if !actual.is_finite() || actual < 0.0 {
        bail!("{name} calculated value must be finite and non-negative");
    }

    // Absolute floor protects values near zero; relative term scales for larger traces.
    let tolerance = 1.0e-12 + 1.0e-9 * actual.abs().max(expected.abs());
    if (actual - expected).abs() > tolerance {
        bail!(
            "{name} mismatch: calculated {actual:.15}, expected {expected:.15}, tolerance {tolerance:.3e}"
        );
    }
    Ok(())
}

fn audit_fixture(fixture: &Fixture) -> Result<AuditReport> {
    if fixture.schema_version != "0.1.0" {
        bail!("unsupported standby-service fixture schema_version");
    }
    if fixture.fixture_id != EXPECTED_FIXTURE_ID {
        bail!("unexpected standby-service fixture_id");
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS {
        bail!("standby-service known-answer fixture must remain synthetic");
    }
    if fixture.claim_ceiling != EXPECTED_CLAIM_CEILING {
        bail!("standby-service fixture claim ceiling was weakened or changed");
    }
    if fixture.assumptions.is_empty() || fixture.negative_controls.is_empty() {
        bail!("fixture assumptions and negative_controls must both be non-empty");
    }
    for (index, assumption) in fixture.assumptions.iter().enumerate() {
        require_nonempty(assumption, &format!("assumptions[{index}]"))?;
    }
    for (index, control) in fixture.negative_controls.iter().enumerate() {
        require_nonempty(control, &format!("negative_controls[{index}]"))?;
    }

    let leakage = &fixture.leakage_trace;
    if leakage.quantity != "terminal_leakage_current"
        || leakage.unit != "A"
        || leakage.time_unit != "s"
    {
        bail!("leakage trace must be terminal_leakage_current in A over seconds");
    }
    if leakage.expected.method != LEAKAGE_METHOD {
        bail!("unsupported leakage integration method");
    }
    let leakage_samples: Vec<(f64, f64)> = leakage
        .samples
        .iter()
        .map(|sample| (sample.t, sample.current))
        .collect();
    let integrated_leakage_charge_c =
        trapezoidal_integral(&leakage_samples, "leakage_trace.samples")?;
    require_close(
        "integrated leakage charge (C)",
        integrated_leakage_charge_c,
        leakage.expected.integrated_charge_c,
    )?;

    let discharge = &fixture.post_standby_discharge_trace;
    if discharge.quantity != "terminal_voltage_and_discharge_current"
        || discharge.voltage_unit != "V"
        || discharge.current_unit != "A"
        || discharge.time_unit != "s"
    {
        bail!("discharge trace must contain simultaneous terminal V (V) and I (A) over seconds");
    }
    if discharge.expected.method != DISCHARGE_METHOD {
        bail!("unsupported discharge-energy integration method");
    }

    let mut power_samples = Vec::with_capacity(discharge.samples.len());
    let mut peak_sampled_power_w = 0.0_f64;
    for (index, sample) in discharge.samples.iter().enumerate() {
        if !sample.voltage.is_finite() || sample.voltage < 0.0 {
            bail!("post_standby_discharge_trace.samples[{index}].voltage must be finite and non-negative");
        }
        if !sample.current.is_finite() || sample.current < 0.0 {
            bail!("post_standby_discharge_trace.samples[{index}].current must be finite and non-negative");
        }
        let power = sample.voltage * sample.current;
        if !power.is_finite() || power < 0.0 {
            bail!("post-standby sampled terminal power is invalid");
        }
        peak_sampled_power_w = peak_sampled_power_w.max(power);
        power_samples.push((sample.t, power));
    }

    let delivered_energy_j =
        trapezoidal_integral(&power_samples, "post_standby_discharge_trace.samples")?;
    let delivered_energy_wh = delivered_energy_j / 3_600.0;
    require_close(
        "delivered discharge energy (J)",
        delivered_energy_j,
        discharge.expected.delivered_energy_j,
    )?;
    require_close(
        "delivered discharge energy (Wh)",
        delivered_energy_wh,
        discharge.expected.delivered_energy_wh,
    )?;
    require_close(
        "peak sampled terminal power (W)",
        peak_sampled_power_w,
        discharge.expected.peak_sampled_power_w,
    )?;

    Ok(AuditReport {
        integrated_leakage_charge_c,
        delivered_energy_j,
        delivered_energy_wh,
        peak_sampled_power_w,
    })
}

fn audit_value(value: serde_json::Value) -> Result<AuditReport> {
    let fixture: Fixture =
        serde_json::from_value(value).context("standby-service fixture does not match its strict data contract")?;
    audit_fixture(&fixture)
}

pub fn audit(root: &Path) -> Result<()> {
    let path = root.join(FIXTURE_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read standby-service fixture at {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("standby-service fixture at {} is not valid JSON", path.display()))?;
    let report = audit_value(value)?;

    println!("edlc-standby-service-audit: PASS (known-answer trace arithmetic)");
    println!(
        "integrated leakage charge: {:.12} C",
        report.integrated_leakage_charge_c
    );
    println!(
        "post-standby delivered energy: {:.12} J ({:.12} Wh)",
        report.delivered_energy_j, report.delivered_energy_wh
    );
    println!(
        "peak sampled terminal power: {:.12} W",
        report.peak_sampled_power_w
    );
    println!(
        "claim ceiling: synthetic trace integration and unit arithmetic only; no physical measurement or lifetime inference"
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
            .expect("xtask is one level below workspace root")
            .to_path_buf()
    }

    fn fixture_value() -> Value {
        let bytes = std::fs::read(root().join(FIXTURE_PATH))
            .expect("checked-in standby-service fixture must be readable");
        serde_json::from_slice(&bytes).expect("checked-in fixture must be valid JSON")
    }

    #[test]
    fn checked_in_trace_matches_all_known_answers() {
        let report = audit_value(fixture_value()).expect("known-answer fixture should pass");
        assert!((report.integrated_leakage_charge_c - 0.065).abs() < 1.0e-12);
        assert!((report.delivered_energy_j - 3.95).abs() < 1.0e-12);
        assert!((report.delivered_energy_wh - (3.95 / 3_600.0)).abs() < 1.0e-12);
        assert!((report.peak_sampled_power_w - 2.7).abs() < 1.0e-12);
    }

    #[test]
    fn leakage_trace_rejects_non_monotonic_time() {
        let mut value = fixture_value();
        value["leakage_trace"]["samples"][1]["t"] = json!(0.0);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn discharge_trace_rejects_non_monotonic_time() {
        let mut value = fixture_value();
        value["post_standby_discharge_trace"]["samples"][2]["t"] = json!(1.0);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn discharge_trace_requires_simultaneous_voltage_and_current() {
        let mut value = fixture_value();
        value["post_standby_discharge_trace"]["samples"][1]
            .as_object_mut()
            .unwrap()
            .remove("current");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn wrong_units_are_rejected() {
        let mut value = fixture_value();
        value["post_standby_discharge_trace"]["voltage_unit"] = json!("mV");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn synthetic_claim_ceiling_cannot_be_relabelled_as_physical() {
        let mut value = fixture_value();
        value["evidence_class"] = json!("physical_measurement");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn claim_ceiling_cannot_be_weakened() {
        let mut value = fixture_value();
        value["claim_ceiling"] = json!("MEASUREMENT_VALIDATED");
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn incorrect_expected_energy_is_rejected() {
        let mut value = fixture_value();
        value["post_standby_discharge_trace"]["expected"]["delivered_energy_J"] = json!(4.0);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn negative_leakage_current_is_rejected_under_fixture_sign_convention() {
        let mut value = fixture_value();
        value["leakage_trace"]["samples"][0]["current"] = json!(-0.001);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn negative_discharge_current_is_rejected_under_fixture_sign_convention() {
        let mut value = fixture_value();
        value["post_standby_discharge_trace"]["samples"][0]["current"] = json!(-1.0);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut value = fixture_value();
        value["unexpected_promotion_flag"] = json!(true);
        assert!(audit_value(value).is_err());
    }

    #[test]
    fn trapezoidal_integration_handles_nonuniform_sampling() {
        let samples = [(0.0, 1.0), (2.0, 2.0), (5.0, 0.0)];
        let result = trapezoidal_integral(&samples, "test").expect("valid samples");
        assert!((result - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn too_few_samples_are_rejected() {
        let samples = [(0.0, 1.0)];
        assert!(trapezoidal_integral(&samples, "test").is_err());
    }
}
