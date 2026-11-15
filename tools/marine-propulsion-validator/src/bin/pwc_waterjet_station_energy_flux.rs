use chrono::{DateTime, Utc};
use jsonschema::draft202012;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const GRAVITY_M_S2: f64 = 9.80665;

#[derive(Debug, Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn new(lo: f64, hi: f64) -> Self { Self { lo, hi } }
    fn point(value: f64) -> Self { Self { lo: value, hi: value } }
    fn add(self, other: Self) -> Self { Self::new(self.lo + other.lo, self.hi + other.hi) }
    fn sub(self, other: Self) -> Self { Self::new(self.lo - other.hi, self.hi - other.lo) }
    fn mul(self, other: Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo, hi)
    }
    fn square(self) -> Self {
        let lo = if self.lo <= 0.0 && self.hi >= 0.0 { 0.0 } else { (self.lo * self.lo).min(self.hi * self.hi) };
        let hi = (self.lo * self.lo).max(self.hi * self.hi);
        Self::new(lo, hi)
    }
    fn scale(self, factor: f64) -> Self {
        if factor >= 0.0 { Self::new(self.lo * factor, self.hi * factor) }
        else { Self::new(self.hi * factor, self.lo * factor) }
    }
    fn is_finite(self) -> bool { self.lo.is_finite() && self.hi.is_finite() }
    fn max_abs(self) -> f64 { self.lo.abs().max(self.hi.abs()) }
}

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
impl Measurement {
    fn interval(&self) -> Interval { Interval::new(self.lower, self.upper) }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Method {
    document: String,
    revision: String,
    effective_year: u16,
    calculation: String,
    axis_definition: String,
    scope_note: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasurementWindow { start_utc: String, end_utc: String }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fluid { density: Measurement }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PressureReference { kind: String, reference_id: String }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ElevationDatum { datum_id: String }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlowBalanceTolerance { absolute_m3_s: f64, relative_fraction: f64 }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    cell_id: String,
    axial_velocity: Measurement,
    normal_x: Measurement,
    static_pressure: Measurement,
    elevation: Measurement,
    area: Measurement,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Station {
    station_number: u8,
    station_id: String,
    pressure_reference_id: String,
    elevation_datum_id: String,
    cells: Vec<Cell>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema_version: String,
    record_id: String,
    record_status: String,
    method: Method,
    measurement_window: MeasurementWindow,
    evaluation_time_utc: String,
    fluid: Fluid,
    pressure_reference: PressureReference,
    elevation_datum: ElevationDatum,
    flow_balance_tolerance: FlowBalanceTolerance,
    station_1: Station,
    station_7: Station,
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Scalar {
    value: f64,
    lower: f64,
    upper: f64,
    unit: &'static str,
}
fn scalar(value: f64, interval: Interval, unit: &'static str) -> Scalar {
    Scalar { value, lower: interval.lo, upper: interval.hi, unit }
}

#[derive(Debug, Serialize)]
struct StationResult {
    station_number: u8,
    station_id: String,
    integrated_volume_flow: Scalar,
    total_axial_energy_flux: Scalar,
    integrated_area_m2: f64,
    cell_count: usize,
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
    station_1: Option<StationResult>,
    station_7: Option<StationResult>,
    station_flow_difference: Option<Scalar>,
    flow_balance_tolerance_m3_s: Option<f64>,
    flow_balance_valid: bool,
    effective_jet_system_power_p_jse: Option<Scalar>,
    station_6_shortcut_used: bool,
    diagnostics: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct Findings { invalid: bool, blocked: bool, incomparable: bool, incomplete: bool, diagnostics: Vec<String> }
impl Findings {
    fn invalid(&mut self, path: &str, msg: impl Into<String>) { self.invalid = true; self.diagnostics.push(format!("{path}: {}", msg.into())); }
    fn blocked(&mut self, path: &str, msg: impl Into<String>) { self.blocked = true; self.diagnostics.push(format!("{path}: {}", msg.into())); }
    fn incomparable(&mut self, path: &str, msg: impl Into<String>) { self.incomparable = true; self.diagnostics.push(format!("{path}: {}", msg.into())); }
    fn incomplete(&mut self, path: &str, msg: impl Into<String>) { self.incomplete = true; self.diagnostics.push(format!("{path}: {}", msg.into())); }
    fn disposition(&self) -> &'static str {
        if self.invalid { "INVALID" } else if self.blocked { "BLOCKED" } else if self.incomparable { "INCOMPARABLE" } else if self.incomplete { "INCOMPLETE" } else { "STATION_FLUX_VALID_SYNTHETIC" }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON in {path}: {e}"))
}
fn parse_time(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value).ok().map(|t| t.with_timezone(&Utc))
}
fn validate_schema(schema: &Value, instance: &Value) -> Result<(), String> {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(schema) { return Err("schema does not validate against Draft 2020-12 meta-schema".to_owned()); }
    let validator = draft202012::options().should_validate_formats(true).build(schema)
        .map_err(|e| format!("cannot compile schema: {e}"))?;
    let errors = validator.iter_errors(instance)
        .map(|e| format!("{}: {e}", e.instance_path())).collect::<Vec<_>>();
    if errors.is_empty() { Ok(()) } else { Err(format!("instance schema violations: {}", errors.join("; "))) }
}

fn validate_measurement(
    path: &str,
    m: &Measurement,
    unit: &str,
    start: Option<&DateTime<Utc>>,
    end: Option<&DateTime<Utc>>,
    evaluation: Option<&DateTime<Utc>>,
    findings: &mut Findings,
) {
    if m.unit != unit { findings.invalid(path, format!("expected unit {unit}, got {}", m.unit)); }
    if !m.value.is_finite() || !m.lower.is_finite() || !m.upper.is_finite() {
        findings.invalid(path, "value and bounds must be finite");
    } else if m.lower > m.value || m.value > m.upper {
        findings.invalid(path, "bounds must satisfy lower <= value <= upper");
    }
    if m.calibration_status != "synthetic" { findings.incomplete(path, "v0.1 supports synthetic-only measurement records"); }
    let observed = parse_time(&m.observed_at_utc);
    let available = parse_time(&m.available_at_utc);
    let (Some(observed), Some(available)) = (observed, available) else {
        findings.invalid(path, "observation and availability timestamps must be RFC3339");
        return;
    };
    if available < observed { findings.invalid(path, "availability cannot precede observation"); }
    if let Some(start) = start { if &observed < start { findings.invalid(path, "observation precedes the measurement window"); } }
    if let Some(end) = end { if &observed > end { findings.invalid(path, "observation follows the measurement window"); } }
    if let Some(evaluation) = evaluation { if &available > evaluation { findings.invalid(path, "measurement was not available at evaluation time (future-data leakage)"); } }
}

fn validate_cell(
    path: &str,
    cell: &Cell,
    start: Option<&DateTime<Utc>>,
    end: Option<&DateTime<Utc>>,
    evaluation: Option<&DateTime<Utc>>,
    findings: &mut Findings,
) {
    if cell.cell_id.trim().is_empty() { findings.invalid(path, "cell_id must be non-empty"); }
    validate_measurement(&format!("{path}/axial_velocity"), &cell.axial_velocity, "m/s", start, end, evaluation, findings);
    validate_measurement(&format!("{path}/normal_x"), &cell.normal_x, "1", start, end, evaluation, findings);
    validate_measurement(&format!("{path}/static_pressure"), &cell.static_pressure, "Pa", start, end, evaluation, findings);
    validate_measurement(&format!("{path}/elevation"), &cell.elevation, "m", start, end, evaluation, findings);
    validate_measurement(&format!("{path}/area"), &cell.area, "m2", start, end, evaluation, findings);

    if cell.normal_x.lower < -1.0 || cell.normal_x.upper > 1.0 {
        findings.invalid(&format!("{path}/normal_x"), "normal_x projection and bounds must lie in [-1, 1]");
    }
    if cell.area.lower <= 0.0 { findings.invalid(&format!("{path}/area"), "differential area must be strictly positive"); }
}

fn validate_station(
    path: &str,
    station: &Station,
    expected_number: u8,
    record: &Record,
    start: Option<&DateTime<Utc>>,
    end: Option<&DateTime<Utc>>,
    evaluation: Option<&DateTime<Utc>>,
    findings: &mut Findings,
) {
    if station.station_number != expected_number { findings.invalid(path, format!("expected station {expected_number}")); }
    if station.station_id.trim().is_empty() { findings.invalid(path, "station_id must be non-empty"); }
    if station.pressure_reference_id != record.pressure_reference.reference_id {
        findings.invalid(path, "station pressure reference does not match the record-wide pressure datum");
    }
    if station.elevation_datum_id != record.elevation_datum.datum_id {
        findings.invalid(path, "station elevation datum does not match the record-wide elevation datum");
    }
    if station.cells.is_empty() { findings.invalid(path, "station requires at least one quadrature cell"); }
    let mut ids = BTreeSet::new();
    for (index, cell) in station.cells.iter().enumerate() {
        if !ids.insert(cell.cell_id.clone()) { findings.invalid(&format!("{path}/cells/{index}/cell_id"), "duplicate cell identity"); }
        validate_cell(&format!("{path}/cells/{index}"), cell, start, end, evaluation, findings);
    }
}

fn integrate_station(station: &Station, density: &Measurement) -> (Interval, Interval, f64) {
    let rho = density.interval();
    let mut volume_flow = Interval::point(0.0);
    let mut energy_flux = Interval::point(0.0);
    let mut area_sum = 0.0;
    for cell in &station.cells {
        let u = cell.axial_velocity.interval();
        let nx = cell.normal_x.interval();
        let pressure = cell.static_pressure.interval();
        let elevation = cell.elevation.interval();
        let area = cell.area.interval();

        // ITTC 7.5-02-05-03.1 Eq. (11):
        // E_sξ = ∫ ρ(1/2 u_ξ² + p/ρ - g z_s)(u_ξ n_ξ) dA.
        // Rewritten as (1/2 ρu² + p - ρgz) * (u n_x dA).
        let axial_volume_flux = u.mul(nx).mul(area);
        let kinetic_pressure = rho.mul(u.square()).scale(0.5);
        let hydrostatic_pressure = rho.scale(GRAVITY_M_S2).mul(elevation);
        let specific_energy_pressure = kinetic_pressure.add(pressure).sub(hydrostatic_pressure);
        let local_energy_flux = specific_energy_pressure.mul(axial_volume_flux);

        volume_flow = volume_flow.add(axial_volume_flux);
        energy_flux = energy_flux.add(local_energy_flux);
        area_sum += cell.area.value;
    }
    (volume_flow, energy_flux, area_sum)
}

fn station_result(station: &Station, q_nominal: f64, q: Interval, energy_nominal: f64, energy: Interval, area: f64) -> StationResult {
    StationResult {
        station_number: station.station_number,
        station_id: station.station_id.clone(),
        integrated_volume_flow: scalar(q_nominal, q, "m3/s"),
        total_axial_energy_flux: scalar(energy_nominal, energy, "W"),
        integrated_area_m2: area,
        cell_count: station.cells.len(),
    }
}

fn empty_report(record: &Record, findings: Findings, arithmetic: &'static str, reason: &'static str) -> AuditReport {
    AuditReport {
        auditor: "pwc-waterjet-station-energy-flux",
        auditor_version: "0.1.0",
        record_id: record.record_id.clone(),
        record_status: record.record_status.clone(),
        disposition: findings.disposition(),
        arithmetic_status: arithmetic,
        reason_code: reason,
        station_1: None,
        station_7: None,
        station_flow_difference: None,
        flow_balance_tolerance_m3_s: None,
        flow_balance_valid: false,
        effective_jet_system_power_p_jse: None,
        station_6_shortcut_used: false,
        diagnostics: findings.diagnostics,
        claim_ceiling: "Synthetic station-energy-flux quadrature only; no PWC, waterjet-system, cavitation, hull-interaction, propulsion, safety, regulatory, or product-performance claim.",
    }
}

fn audit(record: &Record) -> AuditReport {
    let mut findings = Findings::default();
    if record.schema_version != "0.1.0" { findings.invalid("/schema_version", "unsupported version"); }
    if record.record_id.trim().is_empty() { findings.invalid("/record_id", "record_id must be non-empty"); }
    if record.record_status != "synthetic" { findings.incomplete("/record_status", "v0.1 is synthetic-only"); }
    if record.limitations.is_empty() || record.limitations.iter().any(|s| s.trim().is_empty()) {
        findings.invalid("/limitations", "at least one non-empty limitation is required");
    }
    if record.method.document != "ITTC 7.5-02-05-03.1"
        || record.method.revision != "02"
        || record.method.effective_year != 2011
        || record.method.calculation != "direct_station_energy_flux"
        || record.method.axis_definition != "local_axial_xi"
        || record.method.scope_note.trim().is_empty()
    { findings.invalid("/method", "method identity or energy-flux calculation profile is unsupported"); }
    if record.pressure_reference.reference_id.trim().is_empty() || !matches!(record.pressure_reference.kind.as_str(), "absolute" | "gauge") {
        findings.invalid("/pressure_reference", "pressure datum must be explicit and either absolute or gauge");
    }
    if record.elevation_datum.datum_id.trim().is_empty() { findings.invalid("/elevation_datum", "elevation datum must be declared"); }
    if !record.flow_balance_tolerance.absolute_m3_s.is_finite()
        || !record.flow_balance_tolerance.relative_fraction.is_finite()
        || record.flow_balance_tolerance.absolute_m3_s < 0.0
        || !(0.0..=0.1).contains(&record.flow_balance_tolerance.relative_fraction)
    { findings.invalid("/flow_balance_tolerance", "flow tolerances must be finite, non-negative and relative fraction <= 0.1"); }

    let start = parse_time(&record.measurement_window.start_utc);
    let end = parse_time(&record.measurement_window.end_utc);
    let evaluation = parse_time(&record.evaluation_time_utc);
    if start.is_none() || end.is_none() || evaluation.is_none() {
        findings.invalid("/measurement_window", "window and evaluation timestamps must be RFC3339");
    }
    if let (Some(start), Some(end)) = (start.as_ref(), end.as_ref()) {
        if start >= end { findings.invalid("/measurement_window", "measurement window must have positive duration"); }
    }
    if let (Some(evaluation), Some(end)) = (evaluation.as_ref(), end.as_ref()) {
        if evaluation < end { findings.invalid("/evaluation_time_utc", "evaluation time cannot precede the end of the declared measurement window"); }
    }

    validate_measurement("/fluid/density", &record.fluid.density, "kg/m3", start.as_ref(), end.as_ref(), evaluation.as_ref(), &mut findings);
    if record.fluid.density.lower <= 0.0 { findings.invalid("/fluid/density", "fluid density must be strictly positive"); }

    validate_station("/station_1", &record.station_1, 1, record, start.as_ref(), end.as_ref(), evaluation.as_ref(), &mut findings);
    validate_station("/station_7", &record.station_7, 7, record, start.as_ref(), end.as_ref(), evaluation.as_ref(), &mut findings);
    if record.station_1.station_id == record.station_7.station_id { findings.invalid("/station_7/station_id", "station identities must differ"); }

    if findings.invalid || findings.blocked {
        let reason = if findings.invalid { "INVALID_INPUT" } else { "BLOCKED_INPUT" };
        return empty_report(record, findings, "NOT_EVALUATED", reason);
    }

    let (q1, e1, area1) = integrate_station(&record.station_1, &record.fluid.density);
    let (q7, e7, area7) = integrate_station(&record.station_7, &record.fluid.density);
    if !q1.is_finite() || !q7.is_finite() || !e1.is_finite() || !e7.is_finite() {
        findings.invalid("/stations", "station flux integration produced non-finite values");
        return empty_report(record, findings, "NOT_EVALUATED", "NON_FINITE_INTEGRATION");
    }

    let q1_nominal = record.station_1.cells.iter().map(|c| c.axial_velocity.value * c.normal_x.value * c.area.value).sum::<f64>();
    let q7_nominal = record.station_7.cells.iter().map(|c| c.axial_velocity.value * c.normal_x.value * c.area.value).sum::<f64>();
    let e1_nominal = integrate_nominal(&record.station_1, &record.fluid.density);
    let e7_nominal = integrate_nominal(&record.station_7, &record.fluid.density);
    let flow_difference = q7.sub(q1);
    let nominal_flow_difference = q7_nominal - q1_nominal;
    let tolerance = record.flow_balance_tolerance.absolute_m3_s
        + record.flow_balance_tolerance.relative_fraction * q1_nominal.abs().max(q7_nominal.abs());
    let flow_uncertainty_within_tolerance = flow_difference.max_abs() <= tolerance;
    let nominal_flow_within_tolerance = nominal_flow_difference.abs() <= tolerance;

    let station_1 = station_result(&record.station_1, q1_nominal, q1, e1_nominal, e1, area1);
    let station_7 = station_result(&record.station_7, q7_nominal, q7, e7_nominal, e7, area7);

    if q1_nominal <= 0.0 || q7_nominal <= 0.0 {
        findings.incomparable("/station_flow", "integrated flow is non-positive; check local axial velocity sign and station normal orientation");
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, false, None, "STATION_FLOW_DIRECTION_INVALID");
    }
    if !nominal_flow_within_tolerance {
        findings.incomparable("/station_flow", format!("station volume-flow mismatch {:.9} m3/s exceeds tolerance {:.9} m3/s", nominal_flow_difference, tolerance));
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, false, None, "STATION_FLOW_MISMATCH");
    }
    if !flow_uncertainty_within_tolerance {
        findings.incomplete("/station_flow", "station-flow uncertainty intervals do not establish continuity within the declared tolerance");
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, false, None, "FLOW_BALANCE_UNCERTAIN");
    }

    let p_jse = e7.sub(e1);
    let p_nominal = e7_nominal - e1_nominal;
    if !p_jse.is_finite() || !p_nominal.is_finite() {
        findings.invalid("/p_jse", "effective jet-system power is non-finite");
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, false, None, "NON_FINITE_P_JSE");
    }
    if p_nominal <= 0.0 {
        findings.blocked("/p_jse", "station-7 energy flux must exceed station-1 energy flux for this forward-propulsion reference profile");
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, true, None, "NON_POSITIVE_P_JSE");
    }
    if p_jse.lo <= 0.0 {
        findings.incomplete("/p_jse", "energy-flux difference uncertainty interval reaches zero; positive effective power is not established");
        return report_with_stations(record, findings, station_1, station_7, flow_difference, tolerance, false, None, "P_JSE_UNCERTAINTY_CROSSES_ZERO");
    }

    findings.incomplete("/record_status", "synthetic station data can validate quadrature arithmetic only, not propulsion performance");
    report_with_stations(
        record,
        findings,
        station_1,
        station_7,
        flow_difference,
        tolerance,
        true,
        Some(scalar(p_nominal, p_jse, "W")),
        "SYNTHETIC_ONLY",
    )
}

fn integrate_nominal(station: &Station, density: &Measurement) -> f64 {
    let rho = density.value;
    station.cells.iter().map(|cell| {
        let u = cell.axial_velocity.value;
        let n = cell.normal_x.value;
        let p = cell.static_pressure.value;
        let z = cell.elevation.value;
        let area = cell.area.value;
        (0.5 * rho * u * u + p - rho * GRAVITY_M_S2 * z) * (u * n * area)
    }).sum()
}

fn report_with_stations(
    record: &Record,
    findings: Findings,
    station_1: StationResult,
    station_7: StationResult,
    flow_difference: Interval,
    tolerance: f64,
    flow_valid: bool,
    p_jse: Option<Scalar>,
    reason: &'static str,
) -> AuditReport {
    let nominal_flow_difference =
        station_7.integrated_volume_flow.value - station_1.integrated_volume_flow.value;
    AuditReport {
        auditor: "pwc-waterjet-station-energy-flux",
        auditor_version: "0.1.0",
        record_id: record.record_id.clone(),
        record_status: record.record_status.clone(),
        disposition: findings.disposition(),
        arithmetic_status: if p_jse.is_some() || reason == "SYNTHETIC_ONLY" { "VALID" } else { "PARTIAL" },
        reason_code: reason,
        station_1: Some(station_1),
        station_7: Some(station_7),
        station_flow_difference: Some(scalar(nominal_flow_difference, flow_difference, "m3/s")),
        flow_balance_tolerance_m3_s: Some(tolerance),
        flow_balance_valid: flow_valid,
        effective_jet_system_power_p_jse: p_jse,
        station_6_shortcut_used: false,
        diagnostics: findings.diagnostics,
        claim_ceiling: "This synthetic report validates direct station-1/station-7 energy-flux quadrature and a flow-continuity guard only; it does not validate any physical waterjet, PWC, hull interaction, cavitation, overall efficiency, safety, regulatory compliance, or product performance.",
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: pwc-waterjet-station-energy-flux <schema.json> <record.json>");
        std::process::exit(64);
    }
    let schema = match read_json(&args[0]) { Ok(v) => v, Err(e) => { eprintln!("{e}"); std::process::exit(64); } };
    let instance = match read_json(&args[1]) { Ok(v) => v, Err(e) => { eprintln!("{e}"); std::process::exit(64); } };
    if let Err(e) = validate_schema(&schema, &instance) { eprintln!("{e}"); std::process::exit(65); }
    let record: Record = match serde_json::from_value(instance) { Ok(v) => v, Err(e) => { eprintln!("cannot decode record: {e}"); std::process::exit(65); } };
    let report = audit(&record);
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_else(|e| format!("{{\"disposition\":\"INVALID\",\"error\":\"{e}\"}}")));
    if report.disposition != "INCOMPLETE" { std::process::exit(2); }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = include_str!("../../../../docs/research/marine/pwc-waterjet-station-energy-flux-v0.1.0.schema.json");
    const POSITIVE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-station-flux-v0.1.0.json");
    const MISMATCH: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-station-flux-mismatch-v0.1.0.json");
    const REVERSED: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-station-flux-reversed-normal-v0.1.0.json");

    fn instance(json: &str) -> (Value, Record) {
        let value: Value = serde_json::from_str(json).unwrap();
        let record: Record = serde_json::from_value(value.clone()).unwrap();
        (value, record)
    }

    #[test]
    fn station_flux_known_answer_matches_independent_vector() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let (value, record) = instance(POSITIVE);
        validate_schema(&schema, &value).unwrap();
        let report = audit(&record);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.arithmetic_status, "VALID");
        let s1 = report.station_1.unwrap();
        let s7 = report.station_7.unwrap();
        let p = report.effective_jet_system_power_p_jse.unwrap();
        assert!((s1.integrated_volume_flow.value - 2.0).abs() < 1e-12);
        assert!((s7.integrated_volume_flow.value - 2.0).abs() < 1e-12);
        assert!((s1.total_axial_energy_flux.value - 209650.0).abs() < 1e-8);
        assert!((s7.total_axial_energy_flux.value - 306650.0).abs() < 1e-8);
        assert!((p.value - 97000.0).abs() < 1e-8);
        assert!(!report.station_6_shortcut_used);
    }

    #[test]
    fn nonuniform_velocity_cells_are_integrated_without_bulk_mean_substitution() {
        let record = instance(POSITIVE).1;
        let p = audit(&record).effective_jet_system_power_p_jse.unwrap();
        assert!((p.value - 97000.0).abs() < 1e-8);
        // Replacing each station by a single bulk-mean velocity would not preserve
        // the velocity-cubed kinetic-energy flux term represented by Eq. (11).
    }

    #[test]
    fn station_flow_mismatch_blocks_effective_power() {
        let report = audit(&instance(MISMATCH).1);
        assert_eq!(report.disposition, "INCOMPARABLE");
        assert_eq!(report.reason_code, "STATION_FLOW_MISMATCH");
        assert!(report.effective_jet_system_power_p_jse.is_none());
    }

    #[test]
    fn reversed_station_normal_is_not_silently_absorbed() {
        let report = audit(&instance(REVERSED).1);
        assert_eq!(report.disposition, "INCOMPARABLE");
        assert_eq!(report.reason_code, "STATION_FLOW_DIRECTION_INVALID");
        assert!(report.effective_jet_system_power_p_jse.is_none());
    }

    #[test]
    fn energy_flux_uses_signed_axial_normal_projection() {
        let record = instance(REVERSED).1;
        let (q, _, _) = integrate_station(&record.station_7, &record.fluid.density);
        assert!((q.lo + 2.0).abs() < 1e-12);
        assert!((q.hi + 2.0).abs() < 1e-12);
    }

    #[test]
    fn pressure_reference_mismatch_is_invalid() {
        let mut record = instance(POSITIVE).1;
        record.station_7.pressure_reference_id = "OTHER-PRESSURE-DATUM".to_owned();
        let report = audit(&record);
        assert_eq!(report.disposition, "INVALID");
        assert_eq!(report.reason_code, "INVALID_INPUT");
    }

    #[test]
    fn elevation_datum_mismatch_is_invalid() {
        let mut record = instance(POSITIVE).1;
        record.station_7.elevation_datum_id = "OTHER-HEIGHT-DATUM".to_owned();
        let report = audit(&record);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn future_measurement_availability_is_rejected() {
        let mut record = instance(POSITIVE).1;
        record.station_7.cells[0].axial_velocity.available_at_utc = "2026-11-15T00:02:00Z".to_owned();
        let report = audit(&record);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn interval_product_bounds_are_conservative_for_signed_velocity() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(4.0, 5.0);
        let result = a.mul(b);
        assert_eq!(result.lo, -10.0);
        assert_eq!(result.hi, 15.0);
        let sq = a.square();
        assert_eq!(sq.lo, 0.0);
        assert_eq!(sq.hi, 9.0);
    }

    #[test]
    fn pressure_and_elevation_terms_affect_the_integrated_energy_flux() {
        let mut record = instance(POSITIVE).1;
        let before = integrate_nominal(&record.station_7, &record.fluid.density);
        for cell in &mut record.station_7.cells {
            cell.elevation.value = 1.0;
            cell.elevation.lower = 1.0;
            cell.elevation.upper = 1.0;
        }
        let after = integrate_nominal(&record.station_7, &record.fluid.density);
        // ΔE = -ρ g z Q for Q=2 m3/s at z=1 m.
        assert!((after - (before - 1000.0 * GRAVITY_M_S2 * 2.0)).abs() < 1e-7);
    }
}
