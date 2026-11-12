//! Offline audit for the planetary-energy orbital screening corpus.
//!
//! This is an evidence/consistency checker, not an ephemeris, irradiance,
//! photovoltaic, storage, thermal, or power-system solver.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const EXPECTED_SCHEMA: &str = "planetary-energy-service-orbit-reference-v0.1";
const EXPECTED_STATUS: &str = "analytic_screening_only";
const EXPECTED_EVIDENCE_CLASS: &str = "analytic_derived";
const EXPECTED_EQUATION: &str = "S(r) = S_1AU / r^2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: String,
    status: String,
    created_for_issue: String,
    reference_model: ReferenceModel,
    cases: Vec<OrbitCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceModel {
    solar_irradiance_at_1_au_w_m2: f64,
    equation: String,
    distance_unit: String,
    irradiance_unit: String,
    assumptions: Vec<String>,
    rounding_tolerance_w_m2: f64,
    rounding_policy: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbitCase {
    case_id: String,
    body_or_region: String,
    heliocentric_distance_au: f64,
    normal_incidence_top_of_atmosphere_irradiance_w_m2: f64,
    evidence_class: String,
    required_downstream_inputs: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct AuditReport {
    pub report_schema: &'static str,
    pub source_sha256: String,
    pub status: &'static str,
    pub claim_ceiling: &'static str,
    pub case_count: usize,
    pub passed_count: usize,
    pub blocked_count: usize,
    pub corpus_findings: Vec<Finding>,
    pub cases: Vec<CaseResult>,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Finding {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub status: &'static str,
    pub computed_irradiance_w_m2: Option<f64>,
    pub reference_flux_fraction_of_1au: Option<f64>,
    pub ideal_area_multiplier_for_equal_incident_power: Option<f64>,
    pub reported_irradiance_w_m2: f64,
    pub absolute_error_w_m2: Option<f64>,
    pub findings: Vec<Finding>,
}

pub fn run(input: &Path, output: Option<&Path>) -> Result<()> {
    let bytes = fs::read(input)
        .with_context(|| format!("reading planetary-energy corpus {}", input.display()))?;
    let report = audit_corpus(&bytes)?;
    let rendered = serde_json::to_string_pretty(&report)?;

    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating report directory {}", parent.display()))?;
            }
        }
        fs::write(path, &rendered)
            .with_context(|| format!("writing planetary-energy report {}", path.display()))?;
    }
    println!("{rendered}");

    if report.status != "PASS" {
        bail!(
            "planetary-energy audit BLOCKED: {} corpus finding(s), {} blocked case(s)",
            report.corpus_findings.len(),
            report.blocked_count
        );
    }
    Ok(())
}

pub fn audit_corpus(bytes: &[u8]) -> Result<AuditReport> {
    let corpus: Corpus = serde_json::from_slice(bytes).context("parsing orbit reference corpus")?;
    let source_sha256 = sha256_hex(bytes);
    let mut corpus_findings = Vec::new();

    if corpus.schema_version != EXPECTED_SCHEMA {
        corpus_findings.push(finding(
            "SCHEMA_VERSION",
            format!(
                "expected {EXPECTED_SCHEMA}, received {}",
                corpus.schema_version
            ),
        ));
    }
    if corpus.status != EXPECTED_STATUS {
        corpus_findings.push(finding(
            "CORPUS_STATUS",
            format!("expected {EXPECTED_STATUS}, received {}", corpus.status),
        ));
    }
    if !corpus.created_for_issue.ends_with("/issues/11756") {
        corpus_findings.push(finding(
            "ISSUE_LINEAGE",
            "created_for_issue must identify Symthaea issue #11756".to_string(),
        ));
    }

    let model = &corpus.reference_model;
    if !model.solar_irradiance_at_1_au_w_m2.is_finite()
        || model.solar_irradiance_at_1_au_w_m2 <= 0.0
    {
        corpus_findings.push(finding(
            "INVALID_REFERENCE_FLUX",
            "1-AU reference irradiance must be finite and positive".to_string(),
        ));
    }
    if model.equation != EXPECTED_EQUATION {
        corpus_findings.push(finding(
            "UNSUPPORTED_EQUATION",
            format!("expected equation {EXPECTED_EQUATION}"),
        ));
    }
    if model.distance_unit != "AU" || model.irradiance_unit != "W/m^2" {
        corpus_findings.push(finding(
            "UNIT_MISMATCH",
            "distance must be AU and irradiance must be W/m^2".to_string(),
        ));
    }
    if model.assumptions.is_empty()
        || model.assumptions.iter().any(|value| value.trim().is_empty())
    {
        corpus_findings.push(finding(
            "ASSUMPTIONS_INCOMPLETE",
            "at least one non-empty screening assumption is required".to_string(),
        ));
    }
    if !model.rounding_tolerance_w_m2.is_finite()
        || model.rounding_tolerance_w_m2 <= 0.0
        || model.rounding_tolerance_w_m2 > 0.01
    {
        corpus_findings.push(finding(
            "INVALID_ROUNDING_TOLERANCE",
            "rounding tolerance must be finite, positive and no greater than 0.01 W/m^2"
                .to_string(),
        ));
    }
    if model.rounding_policy != "nearest_0.001_w_m2" {
        corpus_findings.push(finding(
            "ROUNDING_POLICY",
            "v0.1 requires nearest_0.001_w_m2".to_string(),
        ));
    }

    if corpus.cases.is_empty() {
        corpus_findings.push(finding(
            "EMPTY_CORPUS",
            "at least one orbit reference case is required".to_string(),
        ));
    }

    let mut seen_case_ids = BTreeSet::new();
    let mut cases = Vec::with_capacity(corpus.cases.len());

    for case in corpus.cases {
        let mut findings = Vec::new();

        if case.case_id.trim().is_empty() {
            findings.push(finding("EMPTY_CASE_ID", "case_id is required".to_string()));
        } else if !seen_case_ids.insert(case.case_id.clone()) {
            findings.push(finding(
                "DUPLICATE_CASE_ID",
                format!("duplicate case_id {}", case.case_id),
            ));
        }
        if case.body_or_region.trim().is_empty() {
            findings.push(finding(
                "EMPTY_BODY_OR_REGION",
                "body_or_region is required".to_string(),
            ));
        }
        if !case.heliocentric_distance_au.is_finite()
            || case.heliocentric_distance_au <= 0.0
        {
            findings.push(finding(
                "INVALID_ORBIT_DISTANCE",
                "heliocentric_distance_au must be finite and positive".to_string(),
            ));
        }
        if !case
            .normal_incidence_top_of_atmosphere_irradiance_w_m2
            .is_finite()
            || case.normal_incidence_top_of_atmosphere_irradiance_w_m2 <= 0.0
        {
            findings.push(finding(
                "INVALID_REPORTED_IRRADIANCE",
                "reported irradiance must be finite and positive".to_string(),
            ));
        }
        if case.evidence_class != EXPECTED_EVIDENCE_CLASS {
            findings.push(finding(
                "EVIDENCE_CLASS_PROMOTION",
                format!(
                    "expected {EXPECTED_EVIDENCE_CLASS}; received {}",
                    case.evidence_class
                ),
            ));
        }
        if case.required_downstream_inputs.is_empty() {
            findings.push(finding(
                "DOWNSTREAM_INPUTS_MISSING",
                "screening case must state its required downstream inputs".to_string(),
            ));
        }
        let mut seen_inputs = BTreeSet::new();
        for input in &case.required_downstream_inputs {
            if input.trim().is_empty() || !seen_inputs.insert(input) {
                findings.push(finding(
                    "INVALID_DOWNSTREAM_INPUT",
                    "downstream inputs must be non-empty and unique".to_string(),
                ));
            }
        }

        let distance_squared = if case.heliocentric_distance_au.is_finite()
            && case.heliocentric_distance_au > 0.0
        {
            let value = case.heliocentric_distance_au.powi(2);
            if value.is_finite() && value > 0.0 {
                Some(value)
            } else {
                None
            }
        } else {
            None
        };
        let ideal_area_multiplier = distance_squared;
        let reference_flux_fraction = distance_squared.and_then(|value| {
            let fraction = 1.0 / value;
            if fraction.is_finite() && fraction > 0.0 {
                Some(fraction)
            } else {
                None
            }
        });
        let computed = reference_flux_fraction.and_then(|fraction| {
            if model.solar_irradiance_at_1_au_w_m2.is_finite()
                && model.solar_irradiance_at_1_au_w_m2 > 0.0
            {
                Some(model.solar_irradiance_at_1_au_w_m2 * fraction)
            } else {
                None
            }
        });

        if ideal_area_multiplier.is_none() || reference_flux_fraction.is_none() {
            findings.push(finding(
                "AREA_MULTIPLIER_NOT_COMPUTABLE",
                "distance cannot produce a finite positive equal-flux area multiplier".to_string(),
            ));
        }

        let absolute_error = computed.map(|value| {
            (value - case.normal_incidence_top_of_atmosphere_irradiance_w_m2).abs()
        });

        if let Some(error) = absolute_error {
            if !error.is_finite() || error > model.rounding_tolerance_w_m2 {
                findings.push(finding(
                    "IRRADIANCE_MISMATCH",
                    format!(
                        "inverse-square result differs by {:.9} W/m^2 (allowed {:.9} W/m^2)",
                        error, model.rounding_tolerance_w_m2
                    ),
                ));
            }
        } else {
            findings.push(finding(
                "IRRADIANCE_NOT_COMPUTABLE",
                "invalid distance or reference flux prevents calculation".to_string(),
            ));
        }

        findings.sort();
        let status = if findings.is_empty() {
            "PASS_ANALYTIC_REFERENCE_ONLY"
        } else {
            "BLOCKED"
        };

        cases.push(CaseResult {
            case_id: case.case_id,
            status,
            computed_irradiance_w_m2: computed.map(round_to_six_decimals),
            reference_flux_fraction_of_1au: reference_flux_fraction.map(round_to_six_decimals),
            ideal_area_multiplier_for_equal_incident_power: ideal_area_multiplier
                .map(round_to_six_decimals),
            reported_irradiance_w_m2: case
                .normal_incidence_top_of_atmosphere_irradiance_w_m2,
            absolute_error_w_m2: absolute_error.map(round_to_six_decimals),
            findings,
        });
    }

    cases.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    corpus_findings.sort();

    let passed_count = cases
        .iter()
        .filter(|case| case.status == "PASS_ANALYTIC_REFERENCE_ONLY")
        .count();
    let blocked_count = cases.len() - passed_count;
    let status = if corpus_findings.is_empty() && blocked_count == 0 {
        "PASS"
    } else {
        "BLOCKED"
    };

    Ok(AuditReport {
        report_schema: "planetary-energy-service-audit-report-v0.1",
        source_sha256,
        status,
        claim_ceiling: "analytic_inverse_square_screening_consistency_only",
        case_count: cases.len(),
        passed_count,
        blocked_count,
        corpus_findings,
        cases,
    })
}

fn finding(code: &str, message: String) -> Finding {
    Finding {
        code: code.to_string(),
        message,
    }
}

fn round_to_six_decimals(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect::<Vec<_>>().join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../benchmarks/planetary-energy-service/v0.1/orbit-reference-cases.json");

    fn value_from_fixture() -> serde_json::Value {
        serde_json::from_str(FIXTURE).expect("checked-in reference JSON must parse")
    }

    #[test]
    fn checked_in_orbit_reference_cases_pass() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("audit parses");
        assert_eq!(report.status, "PASS");
        assert_eq!(report.case_count, 8);
        assert_eq!(report.passed_count, 8);
        assert_eq!(report.blocked_count, 0);
        assert!(report.cases.iter().all(|case| {
            case.status == "PASS_ANALYTIC_REFERENCE_ONLY"
                && case.computed_irradiance_w_m2.is_some()
        }));
    }

    #[test]
    fn mismatched_flux_blocks_only_the_affected_case() {
        let mut value = value_from_fixture();
        value["cases"][0]["normal_incidence_top_of_atmosphere_irradiance_w_m2"] =
            serde_json::json!(999.0);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_corpus(&bytes).expect("audit parses");
        assert_eq!(report.status, "BLOCKED");
        assert_eq!(report.blocked_count, 1);
        assert!(report.cases.iter().any(|case| {
            case.findings
                .iter()
                .any(|finding| finding.code == "IRRADIANCE_MISMATCH")
        }));
    }

    #[test]
    fn invalid_distance_cannot_become_favourable_zero_flux() {
        let mut value = value_from_fixture();
        value["cases"][0]["heliocentric_distance_au"] = serde_json::json!(0.0);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_corpus(&bytes).expect("audit parses");
        assert_eq!(report.status, "BLOCKED");
        assert!(report.cases[0]
            .findings
            .iter()
            .any(|finding| finding.code == "INVALID_ORBIT_DISTANCE"));
    }

    #[test]
    fn source_evidence_cannot_be_promoted_to_physical_measurement() {
        let mut value = value_from_fixture();
        value["cases"][0]["evidence_class"] = serde_json::json!("physical_measurement");
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_corpus(&bytes).expect("audit parses");
        assert_eq!(report.status, "BLOCKED");
        assert!(report.cases[0]
            .findings
            .iter()
            .any(|finding| finding.code == "EVIDENCE_CLASS_PROMOTION"));
    }

    #[test]
    fn duplicate_case_and_downstream_input_identities_are_blocked() {
        let mut value = value_from_fixture();
        value["cases"][1]["case_id"] = value["cases"][0]["case_id"].clone();
        value["cases"][0]["required_downstream_inputs"] = serde_json::json!([
            "panel_orientation",
            "panel_orientation"
        ]);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_corpus(&bytes).expect("audit parses");
        assert_eq!(report.status, "BLOCKED");
        assert!(report.cases.iter().any(|case| {
            case.findings
                .iter()
                .any(|finding| finding.code == "DUPLICATE_CASE_ID")
        }));
        assert!(report.cases.iter().any(|case| {
            case.findings
                .iter()
                .any(|finding| finding.code == "INVALID_DOWNSTREAM_INPUT")
        }));
    }

    #[test]
    fn reports_are_order_independent_except_for_source_digest() {
        let original = audit_corpus(FIXTURE.as_bytes()).expect("audit parses");
        let mut value = value_from_fixture();
        value["cases"].as_array_mut().unwrap().reverse();
        let reversed = serde_json::to_vec(&value).expect("fixture serializes");
        let permuted = audit_corpus(&reversed).expect("audit parses");

        assert_eq!(original.status, permuted.status);
        assert_eq!(original.cases, permuted.cases);
        assert_eq!(original.passed_count, permuted.passed_count);
        assert_ne!(original.source_sha256, permuted.source_sha256);
    }

    #[test]
    fn unknown_fields_are_rejected_instead_of_silently_ignored() {
        let mut value = value_from_fixture();
        value["reference_model"]["surprise_efficiency_factor"] = serde_json::json!(1.5);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        assert!(audit_corpus(&bytes).is_err());
    }


    #[test]
    fn equal_incident_power_area_multiplier_scales_with_distance_squared() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("audit parses");
        let belt = report
            .cases
            .iter()
            .find(|case| case.case_id == "main-belt-reference-screen")
            .expect("main-belt reference exists");
        assert!((belt.ideal_area_multiplier_for_equal_incident_power.unwrap() - 7.29).abs() < 1e-6);
        let saturn = report
            .cases
            .iter()
            .find(|case| case.case_id == "saturn-distance-control")
            .expect("Saturn control exists");
        assert!(
            (saturn
                .ideal_area_multiplier_for_equal_incident_power
                .unwrap()
                - 90.25)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn empty_case_corpus_is_blocked() {
        let mut value = value_from_fixture();
        value["cases"] = serde_json::json!([]);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_corpus(&bytes).expect("audit parses");
        assert_eq!(report.status, "BLOCKED");
        assert!(report
            .corpus_findings
            .iter()
            .any(|finding| finding.code == "EMPTY_CORPUS"));
    }
}


#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceCorpus {
    schema_version: String,
    status: String,
    created_for_issue: String,
    scenarios: Vec<ServiceScenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceScenario {
    scenario_id: String,
    service_contract_id: String,
    evidence_class: String,
    mission_duration_h: f64,
    intervals: Vec<ServiceInterval>,
    expected_disposition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceInterval {
    interval_id: String,
    start_hour: f64,
    duration_hours: f64,
    capability_receipt_id: String,
    electricity: ElectricService,
    thermal_services: Vec<ThermalService>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ElectricService {
    required_energy_kwh: f64,
    deliverable_energy_after_reserve_kwh: Option<f64>,
    required_peak_power_kw: f64,
    deliverable_peak_power_kw: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThermalService {
    service_id: String,
    capability_receipt_id: String,
    required_useful_heat_kwh: f64,
    deliverable_useful_heat_kwh: Option<f64>,
    required_supply_temperature_c: f64,
    deliverable_supply_temperature_c: Option<f64>,
    required_peak_thermal_power_kw: f64,
    deliverable_peak_thermal_power_kw: Option<f64>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ServiceAuditReport {
    pub report_schema: &'static str,
    pub source_sha256: String,
    pub status: &'static str,
    pub claim_ceiling: &'static str,
    pub scenario_count: usize,
    pub passed_count: usize,
    pub infeasible_count: usize,
    pub blocked_unknown_count: usize,
    pub invalid_count: usize,
    pub expectation_mismatch_count: usize,
    pub corpus_findings: Vec<Finding>,
    pub scenarios: Vec<ServiceScenarioResult>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ServiceScenarioResult {
    pub scenario_id: String,
    pub actual_disposition: &'static str,
    pub expected_disposition: String,
    pub expectation_matches: bool,
    pub findings: Vec<Finding>,
}

/// Audit upstream, time-indexed service-capability receipts against one frozen
/// mission contract. This checks completeness and stated service sufficiency;
/// it does not derive generation, storage dispatch, or thermal performance.
pub fn audit_service_corpus(bytes: &[u8]) -> Result<ServiceAuditReport> {
    const SCHEMA: &str = "planetary-energy-service-envelope-fixtures-v0.1";
    const STATUS: &str = "synthetic_known_truth";
    const EPSILON: f64 = 1.0e-9;
    const ALLOWED: [&str; 4] = [
        "PASS_SYNTHETIC_SERVICE",
        "INFEASIBLE",
        "BLOCKED_UNKNOWN",
        "INVALID",
    ];

    let corpus: ServiceCorpus =
        serde_json::from_slice(bytes).context("parsing service-envelope scenario corpus")?;
    let source_sha256 = sha256_hex(bytes);
    let mut corpus_findings = Vec::new();

    if corpus.schema_version != SCHEMA {
        corpus_findings.push(finding(
            "SERVICE_SCHEMA_VERSION",
            format!("expected {SCHEMA}, received {}", corpus.schema_version),
        ));
    }
    if corpus.status != STATUS {
        corpus_findings.push(finding(
            "SERVICE_FIXTURE_STATUS",
            format!("expected {STATUS}, received {}", corpus.status),
        ));
    }
    if !corpus.created_for_issue.ends_with("/issues/11756") {
        corpus_findings.push(finding(
            "SERVICE_ISSUE_LINEAGE",
            "created_for_issue must identify Symthaea issue #11756".to_string(),
        ));
    }
    if corpus.scenarios.is_empty() {
        corpus_findings.push(finding(
            "SERVICE_EMPTY_CORPUS",
            "at least one service scenario is required".to_string(),
        ));
    }

    let mut seen_scenario_ids = BTreeSet::new();
    let mut scenarios = Vec::with_capacity(corpus.scenarios.len());

    for scenario in corpus.scenarios {
        let mut findings = Vec::new();
        let mut invalid = false;
        let mut unknown = false;
        let mut infeasible = false;

        if scenario.scenario_id.trim().is_empty() {
            invalid = true;
            findings.push(finding(
                "EMPTY_SCENARIO_ID",
                "scenario_id is required".to_string(),
            ));
        } else if !seen_scenario_ids.insert(scenario.scenario_id.clone()) {
            invalid = true;
            findings.push(finding(
                "DUPLICATE_SCENARIO_ID",
                format!("duplicate scenario_id {}", scenario.scenario_id),
            ));
        }
        if scenario.service_contract_id.trim().is_empty() {
            invalid = true;
            findings.push(finding(
                "MISSING_SERVICE_CONTRACT",
                "service_contract_id is required".to_string(),
            ));
        }
        if scenario.evidence_class != "synthetic_known_truth" {
            invalid = true;
            findings.push(finding(
                "SERVICE_EVIDENCE_CLASS",
                "v0.1 service fixtures must remain synthetic_known_truth".to_string(),
            ));
        }
        if !scenario.mission_duration_h.is_finite() || scenario.mission_duration_h <= 0.0 {
            invalid = true;
            findings.push(finding(
                "INVALID_MISSION_DURATION",
                "mission_duration_h must be finite and positive".to_string(),
            ));
        }
        if !ALLOWED.contains(&scenario.expected_disposition.as_str()) {
            invalid = true;
            findings.push(finding(
                "INVALID_EXPECTED_DISPOSITION",
                "expected_disposition is not a supported disposition".to_string(),
            ));
        }
        if scenario.intervals.is_empty() {
            invalid = true;
            findings.push(finding(
                "NO_SERVICE_INTERVALS",
                "at least one service interval is required".to_string(),
            ));
        }

        let mut interval_ids = BTreeSet::new();
        let mut ordered: Vec<&ServiceInterval> = scenario.intervals.iter().collect();
        ordered.sort_by(|left, right| {
            left.start_hour
                .partial_cmp(&right.start_hour)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.interval_id.cmp(&right.interval_id))
        });

        let mut expected_start = 0.0_f64;
        for interval in ordered {
            if interval.interval_id.trim().is_empty() {
                invalid = true;
                findings.push(finding(
                    "EMPTY_INTERVAL_ID",
                    "interval_id is required".to_string(),
                ));
            } else if !interval_ids.insert(interval.interval_id.clone()) {
                invalid = true;
                findings.push(finding(
                    "DUPLICATE_INTERVAL_ID",
                    format!("duplicate interval_id {}", interval.interval_id),
                ));
            }

            if !interval.start_hour.is_finite()
                || !interval.duration_hours.is_finite()
                || interval.start_hour < 0.0
                || interval.duration_hours <= 0.0
            {
                invalid = true;
                findings.push(finding(
                    "INVALID_INTERVAL_TIME",
                    format!("interval {} has invalid start/duration", interval.interval_id),
                ));
            } else {
                if interval.start_hour > expected_start + EPSILON {
                    unknown = true;
                    findings.push(finding(
                        "INCOMPLETE_TIME_COVERAGE",
                        format!(
                            "gap from hour {:.9} to {:.9} before interval {}",
                            expected_start, interval.start_hour, interval.interval_id
                        ),
                    ));
                } else if interval.start_hour < expected_start - EPSILON {
                    invalid = true;
                    findings.push(finding(
                        "OVERLAPPING_TIME_INTERVAL",
                        format!("interval {} overlaps its predecessor", interval.interval_id),
                    ));
                }
                let end_hour = interval.start_hour + interval.duration_hours;
                if end_hour > scenario.mission_duration_h + EPSILON {
                    invalid = true;
                    findings.push(finding(
                        "INTERVAL_EXCEEDS_MISSION",
                        format!("interval {} ends after mission horizon", interval.interval_id),
                    ));
                }
                expected_start = end_hour;
            }

            if interval.capability_receipt_id.trim().is_empty() {
                invalid = true;
                findings.push(finding(
                    "MISSING_CAPABILITY_RECEIPT",
                    format!("interval {} has no capability receipt identity", interval.interval_id),
                ));
            }

            validate_required_nonnegative(
                "required_energy_kwh",
                interval.electricity.required_energy_kwh,
                &interval.interval_id,
                &mut findings,
                &mut invalid,
            );
            validate_optional_capacity(
                "deliverable_energy_after_reserve_kwh",
                interval.electricity.deliverable_energy_after_reserve_kwh,
                interval.electricity.required_energy_kwh,
                &interval.interval_id,
                &mut findings,
                &mut invalid,
                &mut unknown,
                &mut infeasible,
            );
            validate_required_nonnegative(
                "required_peak_power_kw",
                interval.electricity.required_peak_power_kw,
                &interval.interval_id,
                &mut findings,
                &mut invalid,
            );
            validate_optional_capacity(
                "deliverable_peak_power_kw",
                interval.electricity.deliverable_peak_power_kw,
                interval.electricity.required_peak_power_kw,
                &interval.interval_id,
                &mut findings,
                &mut invalid,
                &mut unknown,
                &mut infeasible,
            );

            let mut thermal_ids = BTreeSet::new();
            for thermal in &interval.thermal_services {
                if thermal.service_id.trim().is_empty() {
                    invalid = true;
                    findings.push(finding(
                        "EMPTY_THERMAL_SERVICE_ID",
                        format!(
                            "interval {} has an empty thermal service id",
                            interval.interval_id
                        ),
                    ));
                } else if !thermal_ids.insert(thermal.service_id.clone()) {
                    invalid = true;
                    findings.push(finding(
                        "DUPLICATE_THERMAL_SERVICE_ID",
                        format!(
                            "interval {} repeats thermal service {}",
                            interval.interval_id, thermal.service_id
                        ),
                    ));
                }
                if thermal.capability_receipt_id.trim().is_empty() {
                    invalid = true;
                    findings.push(finding(
                        "MISSING_THERMAL_CAPABILITY_RECEIPT",
                        format!(
                            "thermal service {} has no capability receipt",
                            thermal.service_id
                        ),
                    ));
                }

                validate_required_nonnegative(
                    "required_useful_heat_kwh",
                    thermal.required_useful_heat_kwh,
                    &thermal.service_id,
                    &mut findings,
                    &mut invalid,
                );
                validate_optional_capacity(
                    "deliverable_useful_heat_kwh",
                    thermal.deliverable_useful_heat_kwh,
                    thermal.required_useful_heat_kwh,
                    &thermal.service_id,
                    &mut findings,
                    &mut invalid,
                    &mut unknown,
                    &mut infeasible,
                );
                validate_required_finite(
                    "required_supply_temperature_c",
                    thermal.required_supply_temperature_c,
                    &thermal.service_id,
                    &mut findings,
                    &mut invalid,
                );
                match thermal.deliverable_supply_temperature_c {
                    None => {
                        unknown = true;
                        findings.push(finding(
                            "UNKNOWN_THERMAL_SUPPLY_TEMPERATURE",
                            format!(
                                "thermal service {} lacks a delivered temperature",
                                thermal.service_id
                            ),
                        ));
                    }
                    Some(value) if !value.is_finite() => {
                        invalid = true;
                        findings.push(finding(
                            "INVALID_THERMAL_SUPPLY_TEMPERATURE",
                            format!(
                                "thermal service {} has a non-finite delivered temperature",
                                thermal.service_id
                            ),
                        ));
                    }
                    Some(value)
                        if value + EPSILON < thermal.required_supply_temperature_c =>
                    {
                        infeasible = true;
                        findings.push(finding(
                            "THERMAL_TEMPERATURE_SHORTFALL",
                            format!(
                                "thermal service {} requires {:.6} C but declares {:.6} C",
                                thermal.service_id, thermal.required_supply_temperature_c, value
                            ),
                        ));
                    }
                    Some(_) => {}
                }
                validate_required_nonnegative(
                    "required_peak_thermal_power_kw",
                    thermal.required_peak_thermal_power_kw,
                    &thermal.service_id,
                    &mut findings,
                    &mut invalid,
                );
                validate_optional_capacity(
                    "deliverable_peak_thermal_power_kw",
                    thermal.deliverable_peak_thermal_power_kw,
                    thermal.required_peak_thermal_power_kw,
                    &thermal.service_id,
                    &mut findings,
                    &mut invalid,
                    &mut unknown,
                    &mut infeasible,
                );
            }
        }

        if expected_start < scenario.mission_duration_h - EPSILON {
            unknown = true;
            findings.push(finding(
                "INCOMPLETE_TIME_COVERAGE",
                format!(
                    "service trace ends at hour {:.9}, before mission horizon {:.9}",
                    expected_start, scenario.mission_duration_h
                ),
            ));
        } else if expected_start > scenario.mission_duration_h + EPSILON {
            invalid = true;
            findings.push(finding(
                "TRACE_EXCEEDS_MISSION",
                "service trace extends beyond mission horizon".to_string(),
            ));
        }

        findings.sort();
        let actual_disposition = if invalid {
            "INVALID"
        } else if infeasible {
            "INFEASIBLE"
        } else if unknown {
            "BLOCKED_UNKNOWN"
        } else {
            "PASS_SYNTHETIC_SERVICE"
        };
        let expectation_matches = actual_disposition == scenario.expected_disposition.as_str();
        if !expectation_matches {
            findings.push(finding(
                "EXPECTED_DISPOSITION_MISMATCH",
                format!(
                    "expected {}, computed {}",
                    scenario.expected_disposition, actual_disposition
                ),
            ));
            findings.sort();
        }

        scenarios.push(ServiceScenarioResult {
            scenario_id: scenario.scenario_id,
            actual_disposition,
            expected_disposition: scenario.expected_disposition,
            expectation_matches,
            findings,
        });
    }

    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    corpus_findings.sort();

    let passed_count = scenarios
        .iter()
        .filter(|scenario| scenario.actual_disposition == "PASS_SYNTHETIC_SERVICE")
        .count();
    let infeasible_count = scenarios
        .iter()
        .filter(|scenario| scenario.actual_disposition == "INFEASIBLE")
        .count();
    let blocked_unknown_count = scenarios
        .iter()
        .filter(|scenario| scenario.actual_disposition == "BLOCKED_UNKNOWN")
        .count();
    let invalid_count = scenarios
        .iter()
        .filter(|scenario| scenario.actual_disposition == "INVALID")
        .count();
    let expectation_mismatch_count = scenarios
        .iter()
        .filter(|scenario| !scenario.expectation_matches)
        .count();

    let status = if corpus_findings.is_empty() && expectation_mismatch_count == 0 {
        "PASS"
    } else {
        "BLOCKED"
    };

    Ok(ServiceAuditReport {
        report_schema: "planetary-energy-service-audit-report-v0.1",
        source_sha256,
        status,
        claim_ceiling: "synthetic_receipt_identity_presence_and_declared_service_comparison_only",
        scenario_count: scenarios.len(),
        passed_count,
        infeasible_count,
        blocked_unknown_count,
        invalid_count,
        expectation_mismatch_count,
        corpus_findings,
        scenarios,
    })
}

fn validate_required_nonnegative(
    field: &str,
    value: f64,
    subject: &str,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
) {
    if !value.is_finite() || value < 0.0 {
        *invalid = true;
        findings.push(finding(
            "INVALID_REQUIRED_SERVICE_VALUE",
            format!("{subject}: {field} must be finite and non-negative"),
        ));
    }
}

fn validate_required_finite(
    field: &str,
    value: f64,
    subject: &str,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
) {
    if !value.is_finite() {
        *invalid = true;
        findings.push(finding(
            "INVALID_REQUIRED_SERVICE_VALUE",
            format!("{subject}: {field} must be finite"),
        ));
    }
}

fn validate_optional_capacity(
    field: &str,
    available: Option<f64>,
    required: f64,
    subject: &str,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
    unknown: &mut bool,
    infeasible: &mut bool,
) {
    match available {
        None => {
            *unknown = true;
            findings.push(finding(
                "UNKNOWN_SERVICE_CAPABILITY",
                format!("{subject}: {field} is unknown, not zero"),
            ));
        }
        Some(value) if !value.is_finite() || value < 0.0 => {
            *invalid = true;
            findings.push(finding(
                "INVALID_DECLARED_CAPABILITY",
                format!("{subject}: {field} must be finite and non-negative"),
            ));
        }
        Some(value) if value + 1.0e-9 < required => {
            *infeasible = true;
            findings.push(finding(
                "SERVICE_CAPACITY_SHORTFALL",
                format!("{subject}: {field} {:.6} is below required {:.6}", value, required),
            ));
        }
        Some(_) => {}
    }
}

pub fn run_service_audit(input: &Path, output: Option<&Path>) -> Result<()> {
    let bytes = fs::read(input)
        .with_context(|| format!("reading service-envelope corpus {}", input.display()))?;
    let report = audit_service_corpus(&bytes)?;
    let rendered = serde_json::to_string_pretty(&report)?;

    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating report directory {}", parent.display()))?;
            }
        }
        fs::write(path, &rendered)
            .with_context(|| format!("writing service-envelope report {}", path.display()))?;
    }
    println!("{rendered}");

    if report.status != "PASS" {
        bail!(
            concat!(
                "service-envelope audit BLOCKED: {} corpus finding(s), ",
                "{} expected-disposition mismatch(es)"
            ),
            report.corpus_findings.len(),
            report.expectation_mismatch_count
        );
    }
    Ok(())
}


#[cfg(test)]
mod service_tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../benchmarks/planetary-energy-service/v0.1/service-envelope-cases.json");

    fn value_from_fixture() -> serde_json::Value {
        serde_json::from_str(FIXTURE).expect("checked-in service fixture must parse")
    }

    fn scenario<'a>(report: &'a ServiceAuditReport, id: &str) -> &'a ServiceScenarioResult {
        report
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == id)
            .expect("scenario exists")
    }

    #[test]
    fn checked_in_service_fixture_matches_all_expected_dispositions() {
        let report = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        assert_eq!(report.status, "PASS");
        assert_eq!(report.scenario_count, 5);
        assert_eq!(report.passed_count, 1);
        assert_eq!(report.infeasible_count, 2);
        assert_eq!(report.blocked_unknown_count, 2);
        assert_eq!(report.invalid_count, 0);
        assert_eq!(report.expectation_mismatch_count, 0);
    }

    #[test]
    fn every_interval_must_meet_energy_and_peak_power_after_reserve() {
        let report = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        let moon = scenario(&report, "moon-long-darkness-reserve-shortfall");
        assert_eq!(moon.actual_disposition, "INFEASIBLE");
        assert!(moon
            .findings
            .iter()
            .any(|finding| finding.code == "SERVICE_CAPACITY_SHORTFALL"));
    }

    #[test]
    fn unknown_source_capability_is_not_interpreted_as_zero_or_as_pass() {
        let report = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        let mars = scenario(&report, "mars-dust-storm-unknown-generation");
        assert_eq!(mars.actual_disposition, "BLOCKED_UNKNOWN");
        assert!(mars.findings.iter().any(|finding| {
            finding.code == "UNKNOWN_SERVICE_CAPABILITY"
        }));
    }

    #[test]
    fn thermal_service_requires_the_requested_temperature_grade() {
        let report = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        let belt = scenario(&report, "main-belt-thermal-grade-shortfall");
        assert_eq!(belt.actual_disposition, "INFEASIBLE");
        assert!(belt.findings.iter().any(|finding| {
            finding.code == "THERMAL_TEMPERATURE_SHORTFALL"
        }));
    }

    #[test]
    fn missing_time_buckets_block_service_continuity() {
        let report = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        let outer = scenario(&report, "outer-system-incomplete-time-coverage");
        assert_eq!(outer.actual_disposition, "BLOCKED_UNKNOWN");
        assert!(outer.findings.iter().any(|finding| {
            finding.code == "INCOMPLETE_TIME_COVERAGE"
        }));
    }

    #[test]
    fn changed_expected_disposition_blocks_the_fixture_audit() {
        let mut value = value_from_fixture();
        value["scenarios"][1]["expected_disposition"] =
            serde_json::json!("PASS_SYNTHETIC_SERVICE");
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_service_corpus(&bytes).expect("service corpus parses");
        assert_eq!(report.status, "BLOCKED");
        assert_eq!(report.expectation_mismatch_count, 1);
        assert!(scenario(&report, "moon-long-darkness-reserve-shortfall")
            .findings
            .iter()
            .any(|finding| finding.code == "EXPECTED_DISPOSITION_MISMATCH"));
    }

    #[test]
    fn overlapping_intervals_are_invalid_not_silently_merged() {
        let mut value = value_from_fixture();
        value["scenarios"][0]["intervals"][1]["start_hour"] = serde_json::json!(11.0);
        value["scenarios"][0]["expected_disposition"] = serde_json::json!("INVALID");
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_service_corpus(&bytes).expect("service corpus parses");
        assert_eq!(report.status, "PASS");
        assert_eq!(
            scenario(&report, "earth-clear-day-night-service").actual_disposition,
            "INVALID"
        );
    }

    #[test]
    fn changing_interval_input_order_does_not_change_dispositions() {
        let original = audit_service_corpus(FIXTURE.as_bytes()).expect("service corpus parses");
        let mut value = value_from_fixture();
        value["scenarios"][0]["intervals"]
            .as_array_mut()
            .expect("interval array")
            .reverse();
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let permuted = audit_service_corpus(&bytes).expect("service corpus parses");
        assert_eq!(original.status, permuted.status);
        assert_eq!(original.scenarios, permuted.scenarios);
    }


    #[test]
    fn thermal_service_list_must_be_explicit_even_when_empty() {
        let mut value = value_from_fixture();
        value["scenarios"][1]["intervals"][0]
            .as_object_mut()
            .expect("interval object")
            .remove("thermal_services");
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        assert!(audit_service_corpus(&bytes).is_err());
    }

    #[test]
    fn physical_evidence_class_cannot_be_fabricated_by_synthetic_fixture() {
        let mut value = value_from_fixture();
        value["scenarios"][0]["evidence_class"] =
            serde_json::json!("physical_measurement");
        value["scenarios"][0]["expected_disposition"] = serde_json::json!("INVALID");
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let report = audit_service_corpus(&bytes).expect("service corpus parses");
        assert_eq!(report.status, "PASS");
        assert_eq!(
            scenario(&report, "earth-clear-day-night-service").actual_disposition,
            "INVALID"
        );
        assert!(scenario(&report, "earth-clear-day-night-service")
            .findings
            .iter()
            .any(|finding| finding.code == "SERVICE_EVIDENCE_CLASS"));
    }
}
