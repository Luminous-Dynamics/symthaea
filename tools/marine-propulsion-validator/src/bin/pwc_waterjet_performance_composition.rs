use jsonschema::draft202012;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const GRAVITY_M_S2: f64 = 9.80665;

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(value: f64) -> Self { Self { lo: value, hi: value } }

    fn new(lo: f64, hi: f64) -> Self { Self { lo, hi } }

    fn add(self, other: Self) -> Self {
        Self::new(self.lo + other.lo, self.hi + other.hi)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.lo - other.hi, self.hi - other.lo)
    }

    fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        Self::new(
            products.iter().copied().fold(f64::INFINITY, f64::min),
            products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return None;
        }
        let reciprocal = Self::new(1.0 / other.hi, 1.0 / other.lo);
        Some(self.mul(reciprocal))
    }

    fn scale(self, factor: f64) -> Self {
        if factor >= 0.0 {
            Self::new(self.lo * factor, self.hi * factor)
        } else {
            Self::new(self.hi * factor, self.lo * factor)
        }
    }

    fn is_finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    fn gap(self, other: Self) -> f64 {
        if self.hi < other.lo {
            other.lo - self.hi
        } else if other.hi < self.lo {
            self.lo - other.hi
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Estimate {
    nominal: f64,
    bounds: Interval,
}
impl Estimate {
    fn add(self, other: Self) -> Self {
        Self {
            nominal: self.nominal + other.nominal,
            bounds: self.bounds.add(other.bounds),
        }
    }
    fn sub(self, other: Self) -> Self {
        Self {
            nominal: self.nominal - other.nominal,
            bounds: self.bounds.sub(other.bounds),
        }
    }
}

#[derive(Debug, Serialize)]
struct Scalar {
    value: f64,
    lower: f64,
    upper: f64,
    unit: &'static str,
}

fn scalar(value: f64, bounds: Interval, unit: &'static str) -> Scalar {
    Scalar { value, lower: bounds.lo, upper: bounds.hi, unit }
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    case_id: String,
    test_path: Option<String>,
    disposition: &'static str,
    arithmetic_status: &'static str,
    reason_code: &'static str,
    source_record_ids: Vec<String>,
    inlet_loss_coefficient_zeta13: Option<Scalar>,
    nozzle_loss_coefficient_zeta57: Option<Scalar>,
    effective_jet_system_power_p_jse: Option<Scalar>,
    effective_pump_power_p_pe: Option<Scalar>,
    impeller_shaft_power_p_d: Option<Scalar>,
    free_stream_pump_efficiency_eta_p0: Option<Scalar>,
    installation_efficiency_eta_inst: Option<Scalar>,
    pump_efficiency_eta_p: Option<Scalar>,
    ducting_efficiency_eta_duct: Option<Scalar>,
    ideal_efficiency_eta_i: Option<Scalar>,
    free_stream_efficiency_eta_0: Option<Scalar>,
    interaction_efficiency_eta_int: Option<Scalar>,
    overall_efficiency_eta_oa: Option<Scalar>,
    diagnostics: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct Findings {
    invalid: bool,
    blocked: bool,
    incomparable: bool,
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

    fn incomparable(&mut self, path: &str, message: impl Into<String>) {
        self.incomparable = true;
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
        } else if self.incomparable {
            "INCOMPARABLE"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "COMPOSITION_VALID_SYNTHETIC"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
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
        .map_err(|error| format!("cannot compile composition schema: {error}"))?;
    let errors = validator
        .iter_errors(instance)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("composition schema violations: {}", errors.join("; ")))
    }
}

fn at<'a>(root: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    root.pointer(pointer).ok_or_else(|| format!("missing {pointer}"))
}

fn string_at(root: &Value, pointer: &str) -> Result<String, String> {
    at(root, pointer)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{pointer} must be a string"))
}

fn number_at(root: &Value, pointer: &str) -> Result<f64, String> {
    at(root, pointer)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{pointer} must be a finite number"))
}

fn measurement_at(root: &Value, pointer: &str, expected_unit: &str) -> Result<Estimate, String> {
    let value = number_at(root, &format!("{pointer}/value"))?;
    let lower = number_at(root, &format!("{pointer}/lower"))?;
    let upper = number_at(root, &format!("{pointer}/upper"))?;
    let unit = string_at(root, &format!("{pointer}/unit"))?;
    if unit != expected_unit {
        return Err(format!("{pointer}: expected unit {expected_unit}, got {unit}"));
    }
    if lower > value || value > upper {
        return Err(format!("{pointer}: bounds must satisfy lower <= value <= upper"));
    }
    Ok(Estimate { nominal: value, bounds: Interval::new(lower, upper) })
}

fn ratio(numerator: Estimate, denominator: Estimate, path: &str) -> Result<Estimate, String> {
    if denominator.nominal <= 0.0 || denominator.bounds.lo <= 0.0 {
        return Err(format!("{path}: denominator must be bounded strictly above zero"));
    }
    let bounds = numerator
        .bounds
        .div(denominator.bounds)
        .ok_or_else(|| format!("{path}: interval denominator crosses zero"))?;
    Ok(Estimate {
        nominal: numerator.nominal / denominator.nominal,
        bounds,
    })
}

fn multiply(left: Estimate, right: Estimate) -> Estimate {
    Estimate {
        nominal: left.nominal * right.nominal,
        bounds: left.bounds.mul(right.bounds),
    }
}

fn multiply_three(a: Estimate, b: Estimate, c: Estimate) -> Estimate {
    multiply(multiply(a, b), c)
}

fn unit_interval(value: f64) -> Estimate {
    Estimate { nominal: value, bounds: Interval::point(value) }
}

fn match_measurements(
    base: &Value,
    other: &Value,
    base_pointer: &str,
    other_pointer: &str,
    tolerance: f64,
    unit: &str,
    label: &str,
) -> Result<(), String> {
    let left = measurement_at(base, base_pointer, unit)?;
    let right = measurement_at(other, other_pointer, unit)?;
    if (left.nominal - right.nominal).abs() > tolerance
        || left.bounds.gap(right.bounds) > tolerance
    {
        return Err(format!(
            "{label} mismatch: nominal delta {:.9} exceeds declared tolerance {:.9} or uncertainty intervals do not overlap within tolerance",
            (left.nominal - right.nominal).abs(),
            tolerance
        ));
    }
    Ok(())
}

fn match_conditions(
    propulsion: &Value,
    stage: &Value,
    tolerances: &Value,
    stage_name: &str,
) -> Result<(), String> {
    let base_path = "/propulsion_test/condition";
    let stage_path = "/condition";

    for field in ["craft_configuration_sha256", "pump_configuration_sha256", "scale_context_id"] {
        let expected = string_at(propulsion, &format!("{base_path}/{field}"))?;
        let actual = string_at(stage, &format!("{stage_path}/{field}"))?;
        if expected != actual {
            return Err(format!("{stage_name}: {field} differs from propulsion test"));
        }
    }

    let pairs = [
        ("craft_speed_mps", "m/s", "craft_speed_abs_m_s", "craft speed"),
        ("flow_rate_m3_s", "m3/s", "flow_rate_abs_m3_s", "flow rate"),
        ("nvr", "1", "nvr_abs", "NVR"),
        ("water_density_kg_m3", "kg/m3", "density_abs_kg_m3", "water density"),
        ("water_temperature_c", "degC", "temperature_abs_c", "water temperature"),
    ];
    for (field, unit, tolerance_field, label) in pairs {
        let tolerance = number_at(tolerances, &format!("/{tolerance_field}"))?;
        match_measurements(
            propulsion,
            stage,
            &format!("{base_path}/{field}"),
            &format!("{stage_path}/{field}"),
            tolerance,
            unit,
            &format!("{stage_name} {label}"),
        )?;
    }
    Ok(())
}

fn source_ids(root: &Value, test_path: &str) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();
    for name in ["propulsion_test", "station_flux_test", "inlet_duct_test", "pump_loop_test", "waterjet_system_test"] {
        if root.get(name).is_some() {
            ids.push(string_at(root, &format!("/{name}/source_record_id"))?);
        }
    }
    let unique = ids.iter().cloned().collect::<BTreeSet<_>>();
    if unique.len() != ids.len() {
        return Err("source_record_id values must be unique within a composition case".to_owned());
    }
    if test_path == "composite_independent_tests"
        && !["propulsion_test", "station_flux_test", "inlet_duct_test", "pump_loop_test", "waterjet_system_test"]
            .iter().all(|name| root.get(name).is_some())
    {
        return Err("composite_independent_tests requires all five source records".to_owned());
    }
    if test_path == "integrated_waterjet_system_test"
        && ["inlet_duct_test", "pump_loop_test"].iter().any(|name| root.get(name).is_some())
    {
        return Err("integrated_waterjet_system_test must not silently combine optional separate inlet/pump-loop records".to_owned());
    }
    Ok(ids)
}

fn validate_efficiency(name: &str, estimate: Estimate, findings: &mut Findings) {
    if estimate.nominal < 0.0 || estimate.bounds.lo < 0.0 {
        findings.blocked(name, "efficiency value/bounds are negative");
    } else if estimate.nominal > 1.0 {
        findings.blocked(name, format!("nominal efficiency {:.9} exceeds unity; do not clip", estimate.nominal));
    } else if estimate.bounds.hi > 1.0 {
        findings.incomplete(name, format!("upper uncertainty bound {:.9} exceeds unity", estimate.bounds.hi));
    }
}

fn make_empty_report(
    case_id: String,
    test_path: Option<String>,
    findings: Findings,
    reason: &'static str,
) -> AuditReport {
    AuditReport {
        auditor: "pwc-waterjet-performance-composition",
        auditor_version: "0.1.0",
        case_id,
        test_path,
        disposition: findings.disposition(),
        arithmetic_status: "NOT_EVALUATED",
        reason_code: reason,
        source_record_ids: Vec::new(),
        inlet_loss_coefficient_zeta13: None,
        nozzle_loss_coefficient_zeta57: None,
        effective_jet_system_power_p_jse: None,
        effective_pump_power_p_pe: None,
        impeller_shaft_power_p_d: None,
        free_stream_pump_efficiency_eta_p0: None,
        installation_efficiency_eta_inst: None,
        pump_efficiency_eta_p: None,
        ducting_efficiency_eta_duct: None,
        ideal_efficiency_eta_i: None,
        free_stream_efficiency_eta_0: None,
        interaction_efficiency_eta_int: None,
        overall_efficiency_eta_oa: None,
        diagnostics: findings.diagnostics,
        claim_ceiling: "Synthetic evidence composition only; no real PWC, pump, inlet, propulsion, hydrodynamic, cavitation, safety, regulatory, or product-performance claim.",
    }
}

fn audit(root: &Value) -> AuditReport {
    let case_id = root.get("case_id").and_then(Value::as_str).unwrap_or("UNKNOWN").to_owned();
    let test_path = root.get("test_path").and_then(Value::as_str).map(str::to_owned);
    let path_name = test_path.as_deref().unwrap_or("unknown");
    let mut findings = Findings::default();

    if root.get("record_status").and_then(Value::as_str) != Some("synthetic") {
        findings.incomplete("/record_status", "v0.1 is a synthetic-only composition contract");
    }
    if let Some(path) = &test_path {
        if path != "composite_independent_tests" && path != "integrated_waterjet_system_test" {
            findings.invalid("/test_path", "unsupported composition path");
        }
    }

    let ids = match source_ids(root, path_name) {
        Ok(ids) => ids,
        Err(error) => {
            findings.invalid("/source_record_ids", error);
            return make_empty_report(case_id, test_path, findings, "INVALID_SOURCE_SET");
        }
    };

    let propulsion = match root.get("propulsion_test") {
        Some(value) => value,
        None => {
            findings.invalid("/propulsion_test", "propulsion test record is required");
            return make_empty_report(case_id, test_path, findings, "MISSING_PROPULSION_TEST");
        }
    };
    let station = match root.get("station_flux_test") {
        Some(value) => value,
        None => {
            findings.invalid("/station_flux_test", "station-flux record is required");
            return make_empty_report(case_id, test_path, findings, "MISSING_STATION_FLUX_TEST");
        }
    };
    let system = match root.get("waterjet_system_test") {
        Some(value) => value,
        None => {
            findings.invalid("/waterjet_system_test", "waterjet-system test record is required");
            return make_empty_report(case_id, test_path, findings, "MISSING_SYSTEM_TEST");
        }
    };
    let tolerances = match root.get("comparison_tolerances") {
        Some(value) => value,
        None => {
            findings.invalid("/comparison_tolerances", "comparison tolerances are required");
            return make_empty_report(case_id, test_path, findings, "MISSING_COMPARISON_TOLERANCES");
        }
    };

    let stages = [
        ("station_flux_test", station),
        ("waterjet_system_test", system),
    ];
    for (name, stage) in stages {
        if let Err(error) = match_conditions(propulsion, stage, tolerances, name) {
            findings.incomparable(&format!("/{name}/condition"), error);
        }
    }
    if findings.incomparable {
        let mut report = make_empty_report(case_id, test_path, findings, "TEST_STAGES_INCOMPARABLE");
        report.source_record_ids = ids;
        return report;
    }
    if path_name == "composite_independent_tests" {
        for name in ["inlet_duct_test", "pump_loop_test"] {
            if let Some(stage) = root.get(name) {
                if let Err(error) = match_conditions(propulsion, stage, tolerances, name) {
                    findings.incomparable(&format!("/{name}/condition"), error);
                }
            }
        }
        if findings.incomparable {
            let mut report = make_empty_report(case_id, test_path, findings, "TEST_STAGES_INCOMPARABLE");
            report.source_record_ids = ids;
            return report;
        }
    }

    let q_prop = measurement_at(propulsion, "/condition/flow_rate_m3_s", "m3/s");
    let q1 = measurement_at(station, "/integrated_station_1_flow_m3_s", "m3/s");
    let q7 = measurement_at(station, "/integrated_station_7_flow_m3_s", "m3/s");
    let e1_station = measurement_at(station, "/energy_flux_e1_w", "W");
    let e7_station = measurement_at(station, "/energy_flux_e7_w", "W");
    let p_jse_recorded = measurement_at(station, "/effective_jet_system_power_p_jse_w", "W");
    let p_d = measurement_at(system, "/impeller_shaft_power_w", "W");
    let p_pe_system = measurement_at(system, "/effective_pump_power_w", "W");

    let (q_prop, q1, q7, e1, e7, p_jse_recorded, p_d, p_pe_system) =
        match (q_prop, q1, q7, e1_station, e7_station, p_jse_recorded, p_d, p_pe_system) {
            (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f), Ok(g), Ok(h)) => (a, b, c, d, e, f, g, h),
            (values) => {
                findings.invalid("/measurements", format!("missing or invalid required measurement in composition inputs: {}", values_error(values)));
                return make_empty_report(case_id, test_path, findings, "INVALID_REQUIRED_MEASUREMENT");
            }
        };

    let flow_tolerance = number_at(tolerances, "/flow_rate_abs_m3_s").unwrap_or(0.0);
    for (label, flow) in [("station_1", q1), ("station_7", q7)] {
        if (flow.nominal - q_prop.nominal).abs() > flow_tolerance {
            findings.incomparable(&format!("/station_flux_test/{label}"), "integrated station flow does not match propulsion-test flow");
        }
    }
    if (q1.nominal - q7.nominal).abs() > flow_tolerance {
        findings.incomparable("/station_flux_test", "station-1 and station-7 volume flows do not balance");
    }

    let p_jse_from_flux = e7.bounds.sub(e1.bounds);
    let p_jse_nominal = e7.nominal - e1.nominal;
    let p_power_tolerance = number_at(tolerances, "/power_abs_w").unwrap_or(0.0);
    if (p_jse_nominal - p_jse_recorded.nominal).abs() > p_power_tolerance
        || p_jse_from_flux.gap(p_jse_recorded.bounds) > p_power_tolerance
    {
        findings.incomparable("/station_flux_test/effective_jet_system_power_p_jse_w", "recorded P_JSE does not agree with E7-E1 within the declared power tolerance");
    }
    let p_jse = Estimate { nominal: p_jse_nominal, bounds: p_jse_from_flux };
    if findings.incomparable || p_jse.nominal <= 0.0 || p_jse.bounds.lo <= 0.0 {
        if p_jse.nominal <= 0.0 || p_jse.bounds.lo <= 0.0 {
            findings.blocked("/station_flux_test/p_jse", "P_JSE must be bounded strictly above zero");
        }
        let mut report = make_empty_report(case_id, test_path, findings, "STATION_FLUX_NOT_COMPOSABLE");
        report.source_record_ids = ids;
        return report;
    }

    let nvr = match measurement_at(propulsion, "/condition/nvr", "1") {
        Ok(value) => value,
        Err(error) => {
            findings.invalid("/propulsion_test/condition/nvr", error);
            return make_empty_report(case_id, test_path, findings, "INVALID_NVR");
        }
    };
    if nvr.nominal <= 0.0 || nvr.bounds.lo <= 0.0 {
        findings.incomplete("/propulsion_test/condition/nvr", "NVR must be bounded strictly above zero for ideal efficiency");
        return make_empty_report(case_id, test_path, findings, "NVR_NOT_EVALUABLE");
    }
    let eta_i = Estimate {
        nominal: 2.0 / (1.0 + nvr.nominal),
        bounds: Interval::new(
            2.0 / (1.0 + nvr.bounds.hi),
            2.0 / (1.0 + nvr.bounds.lo),
        ),
    };

    let interaction = match propulsion.get("interaction_terms") {
        Some(value) => value,
        None => {
            findings.invalid("/propulsion_test/interaction_terms", "interaction terms must be supplied by the propulsion test");
            return make_empty_report(case_id, test_path, findings, "MISSING_INTERACTION_TERMS");
        }
    };
    let one_minus_t = measurement_at(interaction, "/one_minus_thrust_deduction", "1");
    let eta_ei = measurement_at(interaction, "/eta_eI", "1");
    let eta_mi = measurement_at(interaction, "/eta_mI", "1");
    let (one_minus_t, eta_ei, eta_mi) = match (one_minus_t, eta_ei, eta_mi) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            findings.invalid("/propulsion_test/interaction_terms", "invalid or missing interaction measurement");
            return make_empty_report(case_id, test_path, findings, "INVALID_INTERACTION_TERMS");
        }
    };
    if eta_mi.nominal <= 0.0 || eta_mi.bounds.lo <= 0.0 {
        findings.incomplete("/propulsion_test/interaction_terms/eta_mI", "momentum interaction coefficient must be bounded above zero");
        return make_empty_report(case_id, test_path, findings, "INTERACTION_DENOMINATOR_NOT_EVALUABLE");
    }
    let eta_int = match ratio(multiply(one_minus_t, eta_ei), eta_mi, "eta_INT") {
        Ok(value) => value,
        Err(error) => {
            findings.invalid("/propulsion_test/interaction_terms", error);
            return make_empty_report(case_id, test_path, findings, "INVALID_INTERACTION_RATIO");
        }
    };

    let mut zeta13_out: Option<Estimate> = None;
    let mut zeta57_out: Option<Estimate> = None;
    let mut eta_p0_out: Option<Estimate> = None;
    let mut eta_inst_out: Option<Estimate> = None;
    let eta_p: Estimate;
    let eta_duct: Estimate;
    let p_pe: Estimate;

    if path_name == "composite_independent_tests" {
        let inlet = match root.get("inlet_duct_test") {
            Some(value) => value,
            None => {
                findings.invalid("/inlet_duct_test", "independent-test path requires inlet-duct test");
                return make_empty_report(case_id, test_path, findings, "MISSING_INLET_TEST");
            }
        };
        let pump = match root.get("pump_loop_test") {
            Some(value) => value,
            None => {
                findings.invalid("/pump_loop_test", "independent-test path requires pump-loop test");
                return make_empty_report(case_id, test_path, findings, "MISSING_PUMP_LOOP_TEST");
            }
        };
        let e0 = measurement_at(inlet, "/energy_flux_e0_w", "W");
        let e1_inlet = measurement_at(inlet, "/energy_flux_e1_w", "W");
        let e3 = measurement_at(inlet, "/energy_flux_e3_w", "W");
        let (e0, e1_inlet, e3) = match (e0, e1_inlet, e3) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => {
                findings.invalid("/inlet_duct_test", "missing or invalid inlet energy-flux term");
                return make_empty_report(case_id, test_path, findings, "INVALID_INLET_ENERGY_FLUX");
            }
        };
        let zeta13 = match ratio(e1_inlet.bounds.sub(e3.bounds).into_estimate(e1_inlet.nominal - e3.nominal), e0, "zeta13") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/inlet_duct_test", error);
                return make_empty_report(case_id, test_path, findings, "INVALID_ZETA13");
            }
        };
        if zeta13.nominal < 0.0 || zeta13.bounds.lo < 0.0 {
            findings.blocked("/inlet_duct_test/zeta13", "negative inlet-duct loss coefficient is inconsistent with this declared passive-loss test contract");
        }
        let zeta57 = match measurement_at(system, "/nozzle_loss_coefficient_zeta57", "1") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test/nozzle_loss_coefficient_zeta57", error);
                return make_empty_report(case_id, test_path, findings, "MISSING_ZETA57");
            }
        };
        if zeta57.nominal < 0.0 || zeta57.bounds.lo < 0.0 {
            findings.blocked("/waterjet_system_test/nozzle_loss_coefficient_zeta57", "negative nozzle loss coefficient is physically inadmissible for this contract");
        }
        let eta_p0 = match measurement_at(pump, "/free_stream_pump_efficiency_eta_p0", "1") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/pump_loop_test", error);
                return make_empty_report(case_id, test_path, findings, "MISSING_ETA_P0");
            }
        };
        let eta_inst = match measurement_at(system, "/installation_efficiency_eta_inst", "1") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                return make_empty_report(case_id, test_path, findings, "MISSING_ETA_INST");
            }
        };
        let h_js = match measurement_at(system, "/pump_total_head_rise_m", "m") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                return make_empty_report(case_id, test_path, findings, "MISSING_SYSTEM_HEAD");
            }
        };
        let rho = match measurement_at(propulsion, "/condition/water_density_kg_m3", "kg/m3") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/propulsion_test/condition/water_density_kg_m3", error);
                return make_empty_report(case_id, test_path, findings, "INVALID_DENSITY");
            }
        };
        let p_pe_loss = p_jse
            .add(multiply(zeta13, e0))
            .add(multiply(zeta57, e7));
        let p_pe_head = multiply_three(rho, unit_interval(GRAVITY_M_S2), multiply(q_prop, h_js));
        let ppe_comparison = measurement_at(system, "/effective_pump_power_w", "W");
        let ppe_comparison = match ppe_comparison {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test/effective_pump_power_w", error);
                return make_empty_report(case_id, test_path, findings, "INVALID_P_PE");
            }
        };
        if (p_pe_loss.nominal - p_pe_head.nominal).abs() > p_power_tolerance
            || p_pe_loss.bounds.gap(p_pe_head.bounds) > p_power_tolerance
            || (p_pe_loss.nominal - ppe_comparison.nominal).abs() > p_power_tolerance
            || p_pe_loss.bounds.gap(ppe_comparison.bounds) > p_power_tolerance
        {
            findings.incomparable("/waterjet_system_test/effective_pump_power_w", "P_PE from P_JSE + inlet/nozzle loss terms, rho*g*Q*H_JS and system-test measurement do not agree within declared power tolerance");
        }
        let eta_p_from_terms = multiply(eta_p0, eta_inst);
        let eta_p_from_power = match ratio(ppe_comparison, p_d, "eta_P") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                let mut report = make_empty_report(case_id, test_path, findings, "INVALID_ETA_P");
                report.source_record_ids = ids;
                return report;
            }
        };
        let eff_tol = number_at(tolerances, "/efficiency_abs").unwrap_or(0.0);
        if (eta_p_from_terms.nominal - eta_p_from_power.nominal).abs() > eff_tol
            || eta_p_from_terms.bounds.gap(eta_p_from_power.bounds) > eff_tol
        {
            findings.incomparable("/waterjet_system_test/eta_P", "eta_P0*eta_inst does not agree with P_PE/P_D within the declared efficiency tolerance");
        }
        zeta13_out = Some(zeta13);
        zeta57_out = Some(zeta57);
        eta_p0_out = Some(eta_p0);
        eta_inst_out = Some(eta_inst);
        eta_p = eta_p_from_power;
        eta_duct = match ratio(p_jse, ppe_comparison, "eta_duct") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                let mut report = make_empty_report(case_id, test_path, findings, "INVALID_ETA_DUCT");
                report.source_record_ids = ids;
                return report;
            }
        };
        p_pe = ppe_comparison;
    } else {
        eta_p = match ratio(p_pe_system, p_d, "eta_P") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                let mut report = make_empty_report(case_id, test_path, findings, "INVALID_ETA_P");
                report.source_record_ids = ids;
                return report;
            }
        };
        eta_duct = match ratio(p_jse, p_pe_system, "eta_duct") {
            Ok(value) => value,
            Err(error) => {
                findings.invalid("/waterjet_system_test", error);
                let mut report = make_empty_report(case_id, test_path, findings, "INVALID_ETA_DUCT");
                report.source_record_ids = ids;
                return report;
            }
        };
        p_pe = p_pe_system;
    }

    if findings.incomparable {
        let mut report = make_empty_report(case_id, test_path, findings, "TEST_POWER_BALANCE_INCOMPARABLE");
        report.source_record_ids = ids;
        return report;
    }

    let eta_0 = multiply_three(eta_i, eta_p, eta_duct);
    let eta_oa = multiply(eta_0, eta_int);

    validate_efficiency("eta_I", eta_i, &mut findings);
    validate_efficiency("eta_P", eta_p, &mut findings);
    validate_efficiency("eta_duct", eta_duct, &mut findings);
    validate_efficiency("eta_0", eta_0, &mut findings);
    validate_efficiency("eta_INT", eta_int, &mut findings);
    validate_efficiency("eta_OA", eta_oa, &mut findings);
    if let Some(value) = eta_p0_out { validate_efficiency("eta_P0", value, &mut findings); }
    if let Some(value) = eta_inst_out { validate_efficiency("eta_inst", value, &mut findings); }

    if p_jse.nominal <= 0.0 || p_pe.nominal <= 0.0 || p_d.nominal <= 0.0 {
        findings.blocked("/powers", "P_JSE, P_PE and P_D must be strictly positive");
    }

    // Valid arithmetic on a synthetic case is deliberately not a physical qualification.
    findings.incomplete("/record_status", "all input stages are synthetic; valid composition arithmetic does not qualify a real PWC or waterjet system");

    let report = AuditReport {
        auditor: "pwc-waterjet-performance-composition",
        auditor_version: "0.1.0",
        case_id,
        test_path,
        disposition: findings.disposition(),
        arithmetic_status: "VALID",
        reason_code: if findings.blocked {
            "PHYSICAL_BOUNDARY_BLOCKED"
        } else if findings.incomparable {
            "TEST_STAGES_INCOMPARABLE"
        } else {
            "SYNTHETIC_ONLY"
        },
        source_record_ids: ids,
        inlet_loss_coefficient_zeta13: zeta13_out.map(|value| scalar(value.nominal, value.bounds, "1")),
        nozzle_loss_coefficient_zeta57: zeta57_out.map(|value| scalar(value.nominal, value.bounds, "1")),
        effective_jet_system_power_p_jse: Some(scalar(p_jse.nominal, p_jse.bounds, "W")),
        effective_pump_power_p_pe: Some(scalar(p_pe.nominal, p_pe.bounds, "W")),
        impeller_shaft_power_p_d: Some(scalar(p_d.nominal, p_d.bounds, "W")),
        free_stream_pump_efficiency_eta_p0: eta_p0_out.map(|value| scalar(value.nominal, value.bounds, "1")),
        installation_efficiency_eta_inst: eta_inst_out.map(|value| scalar(value.nominal, value.bounds, "1")),
        pump_efficiency_eta_p: Some(scalar(eta_p.nominal, eta_p.bounds, "1")),
        ducting_efficiency_eta_duct: Some(scalar(eta_duct.nominal, eta_duct.bounds, "1")),
        ideal_efficiency_eta_i: Some(scalar(eta_i.nominal, eta_i.bounds, "1")),
        free_stream_efficiency_eta_0: Some(scalar(eta_0.nominal, eta_0.bounds, "1")),
        interaction_efficiency_eta_int: Some(scalar(eta_int.nominal, eta_int.bounds, "1")),
        overall_efficiency_eta_oa: Some(scalar(eta_oa.nominal, eta_oa.bounds, "1")),
        diagnostics: findings.diagnostics,
        claim_ceiling: "This report validates only a declared synthetic test-stage composition and arithmetic contract. It does not establish real-vessel propulsion efficiency, hydrodynamic validity, pump performance, energy savings, safety, class approval, or operational permission.",
    };
    report
}

trait IntervalEstimateExt {
    fn into_estimate(self, nominal: f64) -> Estimate;
}
impl IntervalEstimateExt for Interval {
    fn into_estimate(self, nominal: f64) -> Estimate { Estimate { nominal, bounds: self } }
}

fn values_error<T>(_values: T) -> &'static str {
    "one or more required inputs were absent, invalid, or used an unexpected unit"
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: pwc-waterjet-performance-composition <schema.json> <case.json>");
        std::process::exit(64);
    }
    let schema = match read_json(&args[0]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); std::process::exit(64); }
    };
    let instance = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); std::process::exit(64); }
    };
    if let Err(error) = validate_schema(&schema, &instance) {
        eprintln!("{error}");
        std::process::exit(65);
    }
    let report = audit(&instance);
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_else(|error| format!("{{\"disposition\":\"INVALID\",\"error\":\"{error}\"}}")));
    if report.disposition != "INCOMPLETE" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = include_str!("../../../../docs/research/marine/pwc-waterjet-performance-composition-v0.1.0.schema.json");
    const INDEPENDENT: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-composition-independent-v0.1.0.json");
    const INTEGRATED: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-composition-integrated-v0.1.0.json");
    const MISMATCH: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-composition-mismatch-v0.1.0.json");
    const ZERO_NVR: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-composition-zero-nvr-v0.1.0.json");
    const IMPOSSIBLE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-pwc-waterjet-composition-impossible-efficiency-v0.1.0.json");

    fn load(json: &str) -> Value { serde_json::from_str(json).unwrap() }

    #[test]
    fn schema_accepts_both_declared_test_paths() {
        let schema = load(SCHEMA);
        validate_schema(&schema, &load(INDEPENDENT)).unwrap();
        validate_schema(&schema, &load(INTEGRATED)).unwrap();
    }

    #[test]
    fn independent_test_path_composes_all_itcc_efficiency_terms() {
        let report = audit(&load(INDEPENDENT));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.arithmetic_status, "VALID");
        assert!((report.inlet_loss_coefficient_zeta13.unwrap().value - 0.05).abs() < 1e-12);
        assert!((report.nozzle_loss_coefficient_zeta57.unwrap().value - 0.05).abs() < 1e-12);
        assert!((report.effective_jet_system_power_p_jse.unwrap().value - 1000.0).abs() < 1e-9);
        assert!((report.effective_pump_power_p_pe.unwrap().value - 1400.0).abs() < 1e-8);
        assert!((report.pump_efficiency_eta_p.unwrap().value - 0.7).abs() < 1e-12);
        assert!((report.ducting_efficiency_eta_duct.unwrap().value - (5.0/7.0)).abs() < 1e-12);
        assert!((report.ideal_efficiency_eta_i.unwrap().value - (2.0/3.0)).abs() < 1e-12);
        assert!((report.interaction_efficiency_eta_int.unwrap().value - 0.72).abs() < 1e-12);
        assert!((report.overall_efficiency_eta_oa.unwrap().value - 0.24).abs() < 1e-12);
    }

    #[test]
    fn integrated_system_path_does_not_fabricate_pump_loop_or_installation_terms() {
        let report = audit(&load(INTEGRATED));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!((report.pump_efficiency_eta_p.unwrap().value - 0.7).abs() < 1e-12);
        assert!((report.ducting_efficiency_eta_duct.unwrap().value - (5.0/7.0)).abs() < 1e-12);
        assert!(report.free_stream_pump_efficiency_eta_p0.is_none());
        assert!(report.installation_efficiency_eta_inst.is_none());
        assert!(report.nozzle_loss_coefficient_zeta57.is_none());
        assert!((report.overall_efficiency_eta_oa.unwrap().value - 0.24).abs() < 1e-12);
    }

    #[test]
    fn pump_revision_mismatch_is_incomparable() {
        let report = audit(&load(MISMATCH));
        assert_eq!(report.disposition, "INCOMPARABLE");
        assert!(report.diagnostics.iter().any(|d| d.contains("pump_configuration_sha256")));
    }

    #[test]
    fn zero_nvr_is_not_converted_into_an_ideal_efficiency() {
        let report = audit(&load(ZERO_NVR));
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.reason_code, "NVR_NOT_EVALUABLE");
        assert!(report.ideal_efficiency_eta_i.is_none());
        assert!(report.overall_efficiency_eta_oa.is_none());
    }

    #[test]
    fn impossible_pump_efficiency_is_blocked_not_clipped() {
        let report = audit(&load(IMPOSSIBLE));
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.pump_efficiency_eta_p.unwrap().value > 1.0);
    }

    #[test]
    fn ratio_refuses_uncertainty_interval_crossing_zero() {
        let n = Estimate { nominal: 1.0, bounds: Interval::new(0.9, 1.1) };
        let d = Estimate { nominal: 0.1, bounds: Interval::new(-0.1, 0.2) };
        assert!(ratio(n, d, "test").is_err());
    }

    #[test]
    fn condition_comparison_rejects_revision_and_operating_point_mismatch() {
        let root = load(INDEPENDENT);
        let mut stage = root["pump_loop_test"].clone();
        stage["condition"]["craft_speed_mps"]["value"] = serde_json::json!(5.2);
        let result = match_conditions(&root["propulsion_test"], &stage, &root["comparison_tolerances"], "pump_loop_test");
        assert!(result.is_err());
    }

    #[test]
    fn power_and_efficiency_equations_use_independent_cross_checks() {
        let report = audit(&load(INDEPENDENT));
        assert!((report.effective_pump_power_p_pe.unwrap().value - 1400.0).abs() < 1e-8);
        assert!((report.impeller_shaft_power_p_d.unwrap().value - 2000.0).abs() < 1e-8);
        assert!((report.free_stream_efficiency_eta_0.unwrap().value - (1.0/3.0)).abs() < 1e-12);
    }
}
