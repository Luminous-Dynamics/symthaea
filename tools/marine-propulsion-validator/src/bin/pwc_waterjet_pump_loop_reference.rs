use chrono::{DateTime, Utc};
use jsonschema::draft202012;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::f64::consts::PI;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const STANDARD_GRAVITY_M_S2: f64 = 9.80665;
const RPM_TO_REV_PER_SECOND: f64 = 1.0 / 60.0;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    value: f64,
    lower: f64,
    upper: f64,
    unit: String,
    observed_at_utc: String,
    available_at_utc: String,
    calibration_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Method {
    document: String,
    revision: String,
    effective_year: u16,
    test_type: String,
    scope_note: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestBoundary {
    drive_power_boundary: String,
    torque_source: String,
    friction_correction_method: String,
    electrical_conversion_losses_included: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasurementWindow {
    start_utc: String,
    end_utc: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fluid {
    density: Measurement,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PumpLoop {
    volume_flow_rate: Measurement,
    pump_total_head_rise: Measurement,
    shaft_speed: Measurement,
    measured_shaft_torque: Measurement,
    shaft_friction_torque: Measurement,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema_version: String,
    record_id: String,
    record_status: String,
    method: Method,
    test_boundary: TestBoundary,
    measurement_window: MeasurementWindow,
    fluid: Fluid,
    pump_loop: PumpLoop,
    limitations: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ScalarEstimate {
    value: f64,
    lower: f64,
    upper: f64,
    unit: &'static str,
}

#[derive(Debug, Serialize)]
struct SystemLevelMetrics {
    installed_pump_efficiency: Option<f64>,
    ducting_efficiency: Option<f64>,
    ideal_efficiency: Option<f64>,
    interaction_efficiency: Option<f64>,
    free_stream_system_efficiency: Option<f64>,
    overall_hull_waterjet_efficiency: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    record_id: String,
    record_status: String,
    disposition: &'static str,
    arithmetic_status: &'static str,
    reason_code: &'static str,
    corrected_impeller_torque: Option<ScalarEstimate>,
    impeller_shaft_power: Option<ScalarEstimate>,
    effective_pump_power: Option<ScalarEstimate>,
    free_stream_pump_loop_efficiency_eta_p0: Option<ScalarEstimate>,
    system_level_metrics: SystemLevelMetrics,
    diagnostics: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct Findings {
    invalid: bool,
    blocked: bool,
    incomplete: bool,
    diagnostics: Vec<String>,
}

impl Findings {
    fn invalid(&mut self, path: &str, message: impl Into<String>) {
        self.invalid = true;
        self.diagnostics.push(format!("{path}: {}", message.into()));
    }

    fn blocked(&mut self, path: &str, message: impl Into<String>) {
        self.blocked = true;
        self.diagnostics.push(format!("{path}: {}", message.into()));
    }

    fn incomplete(&mut self, path: &str, message: impl Into<String>) {
        self.incomplete = true;
        self.diagnostics.push(format!("{path}: {}", message.into()));
    }

    fn disposition(&self) -> &'static str {
        if self.invalid {
            "INVALID"
        } else if self.blocked {
            "BLOCKED"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "ARITHMETIC_VALID_SYNTHETIC"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn parse_time(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|date| date.with_timezone(&Utc))
}

fn validate_schema(schema: &Value, instance: &Value) -> Result<(), String> {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(schema) {
        return Err("schema does not validate against the Draft 2020-12 meta-schema".to_owned());
    }
    let validator = draft202012::options()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot compile schema: {error}"))?;
    let errors = validator
        .iter_errors(instance)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("instance schema violations: {}", errors.join("; ")))
    }
}

fn validate_measurement(
    label: &str,
    measurement: &Measurement,
    expected_unit: &str,
    window_start: Option<&DateTime<Utc>>,
    window_end: Option<&DateTime<Utc>>,
    findings: &mut Findings,
) {
    if measurement.unit != expected_unit {
        findings.invalid(
            label,
            format!("expected unit {expected_unit}, found {}", measurement.unit),
        );
    }
    if !measurement.value.is_finite()
        || !measurement.lower.is_finite()
        || !measurement.upper.is_finite()
    {
        findings.invalid(label, "measurement and bounds must be finite");
    } else if measurement.lower > measurement.value || measurement.value > measurement.upper {
        findings.invalid(label, "bounds must satisfy lower <= value <= upper");
    } else if measurement.lower < 0.0 {
        findings.invalid(label, "this v0.1 physical magnitude cannot have a negative lower bound");
    }

    if measurement.calibration_status != "synthetic" {
        findings.incomplete(label, "v0.1 fixture measurements must remain explicitly synthetic");
    }

    let observed = parse_time(&measurement.observed_at_utc);
    let available = parse_time(&measurement.available_at_utc);
    if observed.is_none() || available.is_none() {
        findings.invalid(label, "observation and availability times must be RFC3339 timestamps");
        return;
    }
    let observed = observed.expect("checked above");
    let available = available.expect("checked above");
    if available < observed {
        findings.invalid(label, "availability time cannot precede observation time");
    }
    if let (Some(start), Some(end)) = (window_start, window_end) {
        if observed < *start || observed > *end {
            findings.invalid(label, "observation time must lie inside the declared measurement window");
        }
    }
}

fn estimate(value: f64, lower: f64, upper: f64, unit: &'static str) -> ScalarEstimate {
    ScalarEstimate { value, lower, upper, unit }
}

fn report_without_calculation(
    record_id: String,
    record_status: String,
    disposition: &'static str,
    arithmetic_status: &'static str,
    reason_code: &'static str,
    diagnostics: Vec<String>,
) -> AuditReport {
    AuditReport {
        auditor: "pwc-waterjet-pump-loop-reference",
        auditor_version: "0.1.0",
        record_id,
        record_status,
        disposition,
        arithmetic_status,
        reason_code,
        corrected_impeller_torque: None,
        impeller_shaft_power: None,
        effective_pump_power: None,
        free_stream_pump_loop_efficiency_eta_p0: None,
        system_level_metrics: SystemLevelMetrics {
            installed_pump_efficiency: None,
            ducting_efficiency: None,
            ideal_efficiency: None,
            interaction_efficiency: None,
            free_stream_system_efficiency: None,
            overall_hull_waterjet_efficiency: None,
        },
        diagnostics,
        claim_ceiling: "Synthetic arithmetic reference only; no measured PWC, waterjet-system, hydrodynamic, cavitation, safety, regulatory, or product-performance claim.",
    }
}

fn audit(record: &Record) -> AuditReport {
    let mut findings = Findings::default();

    if record.schema_version != "0.1.0" {
        findings.invalid("/schema_version", "unsupported record schema version");
    }
    if record.record_id.trim().is_empty() {
        findings.invalid("/record_id", "record_id must be non-empty");
    }
    if record.record_status != "synthetic" {
        findings.incomplete("/record_status", "only synthetic reference records are supported in v0.1.0");
    }

    if record.method.document != "ITTC 7.5-02-05-03.2"
        || record.method.revision != "02"
        || record.method.effective_year != 2017
        || record.method.test_type != "pump_loop"
        || record.method.scope_note.trim().is_empty()
    {
        findings.invalid("/method", "method identity or pump-loop scope does not match the v0.1 contract");
    }

    if record.test_boundary.drive_power_boundary != "impeller_shaft"
        || record.test_boundary.torque_source != "shaft_dynamometer"
        || record.test_boundary.friction_correction_method != "separately_measured_and_subtracted"
        || record.test_boundary.electrical_conversion_losses_included
    {
        findings.invalid("/test_boundary", "drive-power/torque boundary is not the declared impeller-shaft boundary");
    }

    let window_start = parse_time(&record.measurement_window.start_utc);
    let window_end = parse_time(&record.measurement_window.end_utc);
    if window_start.is_none() || window_end.is_none() {
        findings.invalid("/measurement_window", "window bounds must be RFC3339 timestamps");
    } else if window_start >= window_end {
        findings.invalid("/measurement_window", "measurement window must have positive duration");
    }

    validate_measurement(
        "/fluid/density",
        &record.fluid.density,
        "kg/m3",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );
    validate_measurement(
        "/pump_loop/volume_flow_rate",
        &record.pump_loop.volume_flow_rate,
        "m3/s",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );
    validate_measurement(
        "/pump_loop/pump_total_head_rise",
        &record.pump_loop.pump_total_head_rise,
        "m",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );
    validate_measurement(
        "/pump_loop/shaft_speed",
        &record.pump_loop.shaft_speed,
        "rpm",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );
    validate_measurement(
        "/pump_loop/measured_shaft_torque",
        &record.pump_loop.measured_shaft_torque,
        "N*m",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );
    validate_measurement(
        "/pump_loop/shaft_friction_torque",
        &record.pump_loop.shaft_friction_torque,
        "N*m",
        window_start.as_ref(),
        window_end.as_ref(),
        &mut findings,
    );

    if !findings.invalid {
        let density = &record.fluid.density;
        let flow = &record.pump_loop.volume_flow_rate;
        let head = &record.pump_loop.pump_total_head_rise;
        let rpm = &record.pump_loop.shaft_speed;
        let measured_torque = &record.pump_loop.measured_shaft_torque;
        let friction_torque = &record.pump_loop.shaft_friction_torque;

        let net_torque = measured_torque.value - friction_torque.value;
        let net_torque_lower = measured_torque.lower - friction_torque.upper;
        let net_torque_upper = measured_torque.upper - friction_torque.lower;

        if net_torque < 0.0 || net_torque_upper < 0.0 {
            findings.invalid("/pump_loop/measured_shaft_torque", "friction-corrected impeller torque is negative");
        } else if net_torque_lower <= 0.0 && net_torque > 0.0 {
            findings.incomplete(
                "/pump_loop/measured_shaft_torque",
                "uncertainty interval reaches zero after friction correction; pump-loop efficiency denominator is not bounded away from zero",
            );
        } else {
            let all_zero = flow.value == 0.0
                && head.value == 0.0
                && rpm.value == 0.0
                && measured_torque.value == 0.0
                && friction_torque.value == 0.0;
            if all_zero {
                findings.incomplete(
                    "/pump_loop",
                    "IDLE_NOT_EVALUABLE: zero-flow/zero-head/zero-speed idle is not an efficiency operating point",
                );
                return report_without_calculation(
                    record.record_id.clone(),
                    record.record_status.clone(),
                    findings.disposition(),
                    "NOT_EVALUABLE",
                    "IDLE_NOT_EVALUABLE",
                    findings.diagnostics,
                );
            }

            if density.value <= 0.0
                || flow.value <= 0.0
                || head.value <= 0.0
                || rpm.value <= 0.0
                || net_torque <= 0.0
            {
                findings.incomplete(
                    "/pump_loop",
                    "required density, flow, head, speed and corrected torque must all be positive for this pump-loop calculation",
                );
                return report_without_calculation(
                    record.record_id.clone(),
                    record.record_status.clone(),
                    findings.disposition(),
                    "NOT_EVALUABLE",
                    "NON_POSITIVE_OPERATING_POINT",
                    findings.diagnostics,
                );
            }

            let p_drive = 2.0 * PI * (rpm.value * RPM_TO_REV_PER_SECOND) * net_torque;
            let p_drive_lower =
                2.0 * PI * (rpm.lower * RPM_TO_REV_PER_SECOND) * net_torque_lower;
            let p_drive_upper =
                2.0 * PI * (rpm.upper * RPM_TO_REV_PER_SECOND) * net_torque_upper;

            let p_effective =
                density.value * STANDARD_GRAVITY_M_S2 * flow.value * head.value;
            let p_effective_lower = density.lower
                * STANDARD_GRAVITY_M_S2
                * flow.lower
                * head.lower;
            let p_effective_upper = density.upper
                * STANDARD_GRAVITY_M_S2
                * flow.upper
                * head.upper;

            if p_drive_lower <= 0.0 || p_effective_lower < 0.0 {
                findings.incomplete(
                    "/pump_loop",
                    "uncertainty bounds do not support a positive, finite efficiency ratio",
                );
                return report_without_calculation(
                    record.record_id.clone(),
                    record.record_status.clone(),
                    findings.disposition(),
                    "NOT_EVALUABLE",
                    "UNBOUNDED_RATIO",
                    findings.diagnostics,
                );
            }

            let eta = p_effective / p_drive;
            let eta_lower = p_effective_lower / p_drive_upper;
            let eta_upper = p_effective_upper / p_drive_lower;

            if !p_drive.is_finite()
                || !p_effective.is_finite()
                || !eta.is_finite()
                || !p_drive_lower.is_finite()
                || !p_drive_upper.is_finite()
                || !p_effective_lower.is_finite()
                || !p_effective_upper.is_finite()
                || !eta_lower.is_finite()
                || !eta_upper.is_finite()
            {
                findings.invalid("/pump_loop", "calculation overflowed or produced a non-finite result");
            } else if eta > 1.0 {
                findings.blocked(
                    "/pump_loop/eta_p0",
                    format!("free-stream pump-loop efficiency {eta:.9} exceeds unity; check units, power boundary, friction correction, or measurements; do not clip"),
                );
            } else if eta_upper > 1.0 {
                findings.incomplete(
                    "/pump_loop/eta_p0",
                    format!("upper uncertainty bound {eta_upper:.9} exceeds unity; measurement bounds do not establish a physically admissible efficiency interval"),
                );
            }

            // The fixture is synthetic by contract, so even valid arithmetic never
            // becomes measured performance or physical qualification.
            findings.incomplete(
                "/record_status",
                "synthetic fixture validates arithmetic only and cannot qualify measured pump performance",
            );

            let report = AuditReport {
                auditor: "pwc-waterjet-pump-loop-reference",
                auditor_version: "0.1.0",
                record_id: record.record_id.clone(),
                record_status: record.record_status.clone(),
                disposition: findings.disposition(),
                arithmetic_status: "VALID",
                reason_code: if findings.blocked {
                    "PHYSICALLY_INADMISSIBLE_EFFICIENCY"
                } else if findings.incomplete && eta_upper > 1.0 {
                    "UNCERTAINTY_CROSSES_PHYSICAL_LIMIT"
                } else {
                    "SYNTHETIC_ONLY"
                },
                corrected_impeller_torque: Some(estimate(
                    net_torque,
                    net_torque_lower,
                    net_torque_upper,
                    "N*m",
                )),
                impeller_shaft_power: Some(estimate(
                    p_drive,
                    p_drive_lower,
                    p_drive_upper,
                    "W",
                )),
                effective_pump_power: Some(estimate(
                    p_effective,
                    p_effective_lower,
                    p_effective_upper,
                    "W",
                )),
                free_stream_pump_loop_efficiency_eta_p0: Some(estimate(
                    eta,
                    eta_lower,
                    eta_upper,
                    "1",
                )),
                system_level_metrics: SystemLevelMetrics {
                    installed_pump_efficiency: None,
                    ducting_efficiency: None,
                    ideal_efficiency: None,
                    interaction_efficiency: None,
                    free_stream_system_efficiency: None,
                    overall_hull_waterjet_efficiency: None,
                },
                diagnostics: findings.diagnostics,
                claim_ceiling: "Synthetic arithmetic reference only; no measured PWC, waterjet-system, hydrodynamic, cavitation, safety, regulatory, or product-performance claim.",
            };
            return report;
        }
    }

    report_without_calculation(
        record.record_id.clone(),
        record.record_status.clone(),
        findings.disposition(),
        "NOT_EVALUABLE",
        if findings.invalid {
            "INVALID_INPUT"
        } else if findings.blocked {
            "BLOCKED_PHYSICAL_BOUNDARY"
        } else {
            "INCOMPLETE_INPUT"
        },
        findings.diagnostics,
    )
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: pwc-waterjet-pump-loop-reference <schema.json> <record.json>");
        std::process::exit(64);
    }

    let schema = match read_json(&args[0]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };
    let instance = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };
    if let Err(error) = validate_schema(&schema, &instance) {
        eprintln!("{error}");
        std::process::exit(65);
    }
    let record: Record = match serde_json::from_value(instance) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("cannot decode record: {error}");
            std::process::exit(65);
        }
    };
    let report = audit(&record);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_else(|error| {
            format!("{{\"disposition\":\"INVALID\",\"error\":\"{error}\"}}")
        })
    );
    if report.disposition != "ARITHMETIC_VALID_SYNTHETIC" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = include_str!("../../../../docs/research/marine/pwc-waterjet-pump-loop-reference-v0.1.0.schema.json");
    const POSITIVE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-v0.1.0.json");
    const IDLE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-idle-v0.1.0.json");
    const NEGATIVE_FLOW: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-negative-flow-v0.1.0.json");
    const IMPOSSIBLE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-impossible-efficiency-v0.1.0.json");
    const UNCERTAINTY_CROSSES_UNITY: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-uncertainty-crosses-unity-v0.1.0.json");
    const TORQUE_BOUNDARY: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-pump-loop-torque-boundary-v0.1.0.json");

    fn record(json: &str) -> Record {
        let value: Value = serde_json::from_str(json).unwrap();
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn known_answer_matches_independent_pump_loop_vector() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let instance: Value = serde_json::from_str(POSITIVE).unwrap();
        validate_schema(&schema, &instance).unwrap();
        let report = audit(&record(POSITIVE));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.arithmetic_status, "VALID");
        let p_drive = report.impeller_shaft_power.unwrap();
        let p_effective = report.effective_pump_power.unwrap();
        let eta = report.free_stream_pump_loop_efficiency_eta_p0.unwrap();
        assert!((p_drive.value - 3141.592653589793).abs() < 1e-9);
        assert!((p_effective.value - 1957.40734).abs() < 1e-8);
        assert!((eta.value - 0.6230621076107163).abs() < 1e-12);
    }

    #[test]
    fn friction_torque_is_subtracted_before_shaft_power() {
        let report = audit(&record(POSITIVE));
        let corrected = report.corrected_impeller_torque.unwrap();
        assert!((corrected.value - 5.0).abs() < 1e-12);
        assert!((corrected.lower - 4.97).abs() < 1e-12);
        assert!((corrected.upper - 5.03).abs() < 1e-12);
    }

    #[test]
    fn interval_arithmetic_is_conservative() {
        let report = audit(&record(POSITIVE));
        let drive = report.impeller_shaft_power.unwrap();
        let pump = report.effective_pump_power.unwrap();
        let eta = report.free_stream_pump_loop_efficiency_eta_p0.unwrap();
        assert!(drive.lower < drive.value && drive.value < drive.upper);
        assert!(pump.lower < pump.value && pump.value < pump.upper);
        assert!((eta.lower - pump.lower / drive.upper).abs() < 1e-12);
        assert!((eta.upper - pump.upper / drive.lower).abs() < 1e-12);
    }

    #[test]
    fn idle_boundary_is_not_misreported_as_perfect_efficiency() {
        let report = audit(&record(IDLE));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.reason_code, "IDLE_NOT_EVALUABLE");
        assert!(report.free_stream_pump_loop_efficiency_eta_p0.is_none());
    }

    #[test]
    fn negative_flow_is_rejected() {
        let report = audit(&record(NEGATIVE_FLOW));
        assert_eq!(report.disposition, "INVALID");
        assert_eq!(report.reason_code, "INVALID_INPUT");
    }

    #[test]
    fn efficiency_above_unity_is_blocked_not_clipped() {
        let report = audit(&record(IMPOSSIBLE));
        assert_eq!(report.disposition, "BLOCKED");
        assert_eq!(report.reason_code, "PHYSICALLY_INADMISSIBLE_EFFICIENCY");
        assert!(report.free_stream_pump_loop_efficiency_eta_p0.unwrap().value > 1.0);
    }

    #[test]
    fn uncertainty_interval_crossing_unity_is_incomplete_not_clipped() {
        let report = audit(&record(UNCERTAINTY_CROSSES_UNITY));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.reason_code, "UNCERTAINTY_CROSSES_PHYSICAL_LIMIT");
        let eta = report.free_stream_pump_loop_efficiency_eta_p0.unwrap();
        assert!(eta.value < 1.0);
        assert!(eta.upper > 1.0);
    }

    #[test]
    fn friction_corrected_torque_interval_must_be_bounded_away_from_zero() {
        let report = audit(&record(TORQUE_BOUNDARY));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.reason_code, "INCOMPLETE_INPUT");
        assert!(report.impeller_shaft_power.is_none());
        assert!(report.free_stream_pump_loop_efficiency_eta_p0.is_none());
    }

    #[test]
    fn pump_loop_never_claims_system_level_efficiencies() {
        let report = audit(&record(POSITIVE));
        assert!(report.system_level_metrics.installed_pump_efficiency.is_none());
        assert!(report.system_level_metrics.ducting_efficiency.is_none());
        assert!(report.system_level_metrics.ideal_efficiency.is_none());
        assert!(report.system_level_metrics.interaction_efficiency.is_none());
        assert!(report.system_level_metrics.free_stream_system_efficiency.is_none());
        assert!(report.system_level_metrics.overall_hull_waterjet_efficiency.is_none());
    }

    #[test]
    fn rpm_is_converted_to_revolutions_per_second() {
        assert!((6000.0 * RPM_TO_REV_PER_SECOND - 100.0).abs() < f64::EPSILON);
        assert!((2.0 * PI * 100.0 * 5.0 - 3141.592653589793).abs() < 1e-9);
    }
}
