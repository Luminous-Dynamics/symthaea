//! Offline contract audit for low-optical-impact energy-surface evidence.
//!
//! This validates evidence completeness, observer/surface metadata, simple
//! declared thresholds and required laser-safety state. It does not calculate
//! BRDF, ray paths, radiative transfer, detector response, biology or laser
//! hazard distances, and it never authorizes hardware actuation.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpticalCorpus {
    schema_version: String,
    status: String,
    created_for_issue: String,
    scenarios: Vec<OpticalScenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpticalScenario {
    scenario_id: String,
    surface_role: String,
    observer: Observer,
    surface: Surface,
    limits: Limits,
    safety: SafetyState,
    expected_disposition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observer {
    observer_id: String,
    instrument_type: String,
    band_id: String,
    wavelength_min_um: f64,
    wavelength_max_um: f64,
    detector_response_receipt_id: String,
    integration_time_s: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Surface {
    surface_id: String,
    configuration_id: String,
    material_lot_id: String,
    measurement_temperature_c: f64,
    spectral_property_receipt_id: Option<String>,
    brdf_measurement_receipt_id: Option<String>,
    angular_coverage_min_deg: f64,
    angular_coverage_max_deg: f64,
    angle_coverage_complete: bool,
    solar_absorptance: Option<f64>,
    infrared_emissivity: Option<f64>,
    observer_band_radiance_w_m2_sr: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    maximum_observer_band_radiance_w_m2_sr: Option<f64>,
    minimum_ir_emissivity: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetyState {
    optical_emitter_present: bool,
    independent_interlock_verified: Option<bool>,
    fail_safe_off_on_unknown: Option<bool>,
    emergency_indicator_available: Option<bool>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct OpticalAuditReport {
    pub report_schema: &'static str,
    pub source_sha256: String,
    pub status: &'static str,
    pub claim_ceiling: &'static str,
    pub scenario_count: usize,
    pub contract_only_pass_count: usize,
    pub impact_exceeded_count: usize,
    pub infeasible_thermal_count: usize,
    pub blocked_unknown_count: usize,
    pub safety_blocked_count: usize,
    pub invalid_count: usize,
    pub expectation_mismatch_count: usize,
    pub scenarios: Vec<OpticalScenarioResult>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct OpticalScenarioResult {
    pub scenario_id: String,
    pub actual_disposition: &'static str,
    pub expected_disposition: String,
    pub expectation_matches: bool,
    pub findings: Vec<OpticalFinding>,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpticalFinding {
    pub code: String,
    pub message: String,
}

const ALLOWED_DISPOSITIONS: [&str; 6] = [
    "PASS_CONTRACT_ONLY",
    "IMPACT_EXCEEDS_BUDGET",
    "INFEASIBLE",
    "BLOCKED_UNKNOWN",
    "SAFETY_BLOCKED",
    "INVALID",
];

pub fn run(input: &Path, output: Option<&Path>) -> Result<()> {
    let bytes = fs::read(input)
        .with_context(|| format!("reading optical-impact corpus {}", input.display()))?;
    let report = audit_corpus(&bytes)?;
    let rendered = serde_json::to_string_pretty(&report)?;

    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| {
                        format!(
                            "creating optical-impact report directory {}",
                            parent.display()
                        )
                    })?;
            }
        }
        fs::write(path, &rendered)
            .with_context(|| format!("writing optical-impact report {}", path.display()))?;
    }
    println!("{rendered}");

    if report.status != "PASS" {
        bail!(
            "optical-impact audit BLOCKED: {} expected-disposition mismatch(es)",
            report.expectation_mismatch_count
        );
    }
    Ok(())
}

/// Evaluate metadata completeness and declared synthetic constraints only.
/// PASS_CONTRACT_ONLY never means that a physical surface is safe or qualified.
pub fn audit_corpus(bytes: &[u8]) -> Result<OpticalAuditReport> {
    const SCHEMA: &str = "planetary-optical-impact-fixtures-v0.1";
    const STATUS: &str = "synthetic_known_truth";
    const EPSILON: f64 = 1.0e-12;

    let corpus: OpticalCorpus =
        serde_json::from_slice(bytes).context("parsing planetary optical-impact corpus")?;
    let source_sha256 = sha256_hex(bytes);
    let mut corpus_invalid = Vec::new();

    if corpus.schema_version != SCHEMA {
        corpus_invalid.push("unexpected optical-impact schema version");
    }
    if corpus.status != STATUS {
        corpus_invalid.push("optical-impact fixtures must retain synthetic_known_truth status");
    }
    if !corpus.created_for_issue.ends_with("/issues/11796") {
        corpus_invalid.push("created_for_issue must identify issue #11796");
    }
    if corpus.scenarios.is_empty() {
        corpus_invalid.push("at least one optical-impact scenario is required");
    }
    if !corpus_invalid.is_empty() {
        bail!("invalid optical-impact corpus: {}", corpus_invalid.join("; "));
    }

    let mut scenario_ids = BTreeSet::new();
    let mut results = Vec::with_capacity(corpus.scenarios.len());

    for scenario in corpus.scenarios {
        let mut findings = Vec::new();
        let mut invalid = false;
        let mut unknown = false;
        let mut impact_exceeded = false;
        let mut thermal_infeasible = false;
        let mut safety_blocked = false;

        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.clone())
        {
            invalid = true;
            findings.push(optical_finding(
                "SCENARIO_IDENTITY",
                "scenario_id must be non-empty and unique",
            ));
        }

        let supported_role = matches!(
            scenario.surface_role.as_str(),
            "solar_collector" | "radiator" | "structure" | "beacon" | "optical_transmitter"
        );
        if !supported_role {
            invalid = true;
            findings.push(optical_finding(
                "UNSUPPORTED_SURFACE_ROLE",
                "surface_role is not supported by v0.1",
            ));
        }

        let observer = &scenario.observer;
        if observer.observer_id.trim().is_empty()
            || observer.instrument_type.trim().is_empty()
            || observer.band_id.trim().is_empty()
            || observer.detector_response_receipt_id.trim().is_empty()
        {
            unknown = true;
            findings.push(optical_finding(
                "OBSERVER_DEFINITION_INCOMPLETE",
                "observer identity, instrument, band and detector-response receipt are required",
            ));
        }
        if !observer.wavelength_min_um.is_finite()
            || !observer.wavelength_max_um.is_finite()
            || observer.wavelength_min_um <= 0.0
            || observer.wavelength_max_um <= observer.wavelength_min_um
            || !observer.integration_time_s.is_finite()
            || observer.integration_time_s <= 0.0
        {
            invalid = true;
            findings.push(optical_finding(
                "OBSERVER_BAND_INVALID",
                concat!(
                    "observer wavelength bounds and integration time must be finite ",
                    "and physically ordered"
                ),
            ));
        }

        let surface = &scenario.surface;
        if surface.surface_id.trim().is_empty()
            || surface.configuration_id.trim().is_empty()
            || surface.material_lot_id.trim().is_empty()
        {
            invalid = true;
            findings.push(optical_finding(
                "SURFACE_IDENTITY_INCOMPLETE",
                "surface, as-built configuration and material-lot identities are required",
            ));
        }
        if !surface.measurement_temperature_c.is_finite()
            || !surface.angular_coverage_min_deg.is_finite()
            || !surface.angular_coverage_max_deg.is_finite()
            || surface.angular_coverage_min_deg < 0.0
            || surface.angular_coverage_max_deg > 180.0
            || surface.angular_coverage_max_deg <= surface.angular_coverage_min_deg
        {
            invalid = true;
            findings.push(optical_finding(
                "SURFACE_MEASUREMENT_DOMAIN_INVALID",
                "temperature and angular measurement coverage must be finite and ordered",
            ));
        }
        if surface
            .spectral_property_receipt_id
            .as_deref()
            .is_none_or(str::is_empty)
        {
            unknown = true;
            findings.push(optical_finding(
                "SPECTRAL_PROPERTY_EVIDENCE_MISSING",
                "spectral property receipt is required",
            ));
        }
        if surface.brdf_measurement_receipt_id.as_deref().is_none_or(str::is_empty)
            || !surface.angle_coverage_complete
        {
            unknown = true;
            findings.push(optical_finding(
                "BRDF_EVIDENCE_INCOMPLETE",
                concat!(
                    "BRDF receipt and complete angular coverage are required; ",
                    "missing angles are not zero reflectance"
                ),
            ));
        }

        for (name, value) in [
            ("solar_absorptance", surface.solar_absorptance),
            ("infrared_emissivity", surface.infrared_emissivity),
        ] {
            if value.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
                invalid = true;
                findings.push(optical_finding(
                    "SURFACE_PROPERTY_OUT_OF_RANGE",
                    format!("{name} must be finite and in [0,1]"),
                ));
            }
        }

        if matches!(scenario.surface_role.as_str(), "solar_collector" | "radiator")
            && surface.solar_absorptance.is_none()
        {
            unknown = true;
            findings.push(optical_finding(
                "SOLAR_ABSORPTANCE_UNKNOWN",
                "solar collector/radiator requires a declared solar absorptance",
            ));
        }
        if scenario.surface_role == "radiator" {
            match (
                surface.infrared_emissivity,
                scenario.limits.minimum_ir_emissivity,
            ) {
                (Some(measured), Some(required))
                    if required.is_finite()
                        && (0.0..=1.0).contains(&required)
                        && measured + EPSILON < required =>
                {
                    thermal_infeasible = true;
                    findings.push(optical_finding(
                        "RADIATOR_EMISSIVITY_BELOW_REQUIREMENT",
                        format!(
                            "declared IR emissivity {:.6} is below synthetic requirement {:.6}",
                            measured, required
                        ),
                    ));
                }
                (Some(_), Some(required))
                    if !required.is_finite() || !(0.0..=1.0).contains(&required) =>
                {
                    invalid = true;
                    findings.push(optical_finding(
                        "RADIATOR_EMISSIVITY_LIMIT_INVALID",
                        "minimum_ir_emissivity must be finite and in [0,1]",
                    ));
                }
                (None, _) | (_, None) => {
                    unknown = true;
                    findings.push(optical_finding(
                        "RADIATOR_EMISSIVITY_EVIDENCE_MISSING",
                        "radiator IR emissivity and its required minimum must both be declared",
                    ));
                }
                _ => {}
            }
        } else if scenario
            .limits
            .minimum_ir_emissivity
            .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            invalid = true;
            findings.push(optical_finding(
                "IR_EMISSIVITY_LIMIT_INVALID",
                "minimum_ir_emissivity must be finite and in [0,1]",
            ));
        }

        match (
            surface.observer_band_radiance_w_m2_sr,
            scenario.limits.maximum_observer_band_radiance_w_m2_sr,
        ) {
            (Some(measured), Some(limit))
                if measured.is_finite()
                    && measured >= 0.0
                    && limit.is_finite()
                    && limit >= 0.0
                    && measured > limit + EPSILON =>
            {
                impact_exceeded = true;
                findings.push(optical_finding(
                    "OBSERVER_RADIANCE_EXCEEDS_BUDGET",
                    format!(
                        concat!(
                            "declared observer-band radiance {:.9} W/(m^2 sr) ",
                            "exceeds synthetic budget {:.9}"
                        ),
                        measured, limit
                    ),
                ));
            }
            (Some(measured), Some(limit))
                if !measured.is_finite()
                    || measured < 0.0
                    || !limit.is_finite()
                    || limit < 0.0 =>
            {
                invalid = true;
                findings.push(optical_finding(
                    "OBSERVER_RADIANCE_INVALID",
                    concat!(
                        "observer-band radiance must be finite/non-negative ",
                        "and its limit finite/non-negative"
                    ),
                ));
            }
            (None, _) | (_, None) => {
                unknown = true;
                findings.push(optical_finding(
                    "OBSERVER_RADIANCE_OR_LIMIT_UNKNOWN",
                    "measured observer-band radiance and allowable limit are both required",
                ));
            }
            _ => {}
        }

        let safety = &scenario.safety;
        if scenario.surface_role == "beacon"
            && safety.emergency_indicator_available != Some(true)
        {
            safety_blocked = true;
            findings.push(optical_finding(
                "EMERGENCY_INDICATOR_UNAVAILABLE",
                "a low-impact design may not suppress required navigation or emergency indication",
            ));
        }
        if scenario.surface_role == "optical_transmitter" && !safety.optical_emitter_present {
            invalid = true;
            findings.push(optical_finding(
                "TRANSMITTER_ROLE_WITHOUT_EMITTER",
                "optical_transmitter role requires an explicitly declared optical emitter",
            ));
        }
        if safety.optical_emitter_present {
            if safety.independent_interlock_verified != Some(true)
                || safety.fail_safe_off_on_unknown != Some(true)
            {
                safety_blocked = true;
                findings.push(optical_finding(
                    "OPTICAL_EMITTER_SAFETY_NOT_VERIFIED",
                    "independent interlock and fail-safe-off-on-unknown must both be verified true",
                ));
            }
        }

        if !ALLOWED_DISPOSITIONS.contains(&scenario.expected_disposition.as_str()) {
            invalid = true;
            findings.push(optical_finding(
                "EXPECTED_DISPOSITION_INVALID",
                "expected_disposition is not a supported v0.1 disposition",
            ));
        }
        findings.sort();

        let actual_disposition = if invalid {
            "INVALID"
        } else if safety_blocked {
            "SAFETY_BLOCKED"
        } else if thermal_infeasible {
            "INFEASIBLE"
        } else if impact_exceeded {
            "IMPACT_EXCEEDS_BUDGET"
        } else if unknown {
            "BLOCKED_UNKNOWN"
        } else {
            "PASS_CONTRACT_ONLY"
        };

        let expectation_matches =
            actual_disposition == scenario.expected_disposition.as_str();
        if !expectation_matches {
            findings.push(optical_finding(
                "EXPECTED_DISPOSITION_MISMATCH",
                format!(
                    "expected {}, computed {}",
                    scenario.expected_disposition, actual_disposition
                ),
            ));
        }
        findings.sort();

        results.push(OpticalScenarioResult {
            scenario_id: scenario.scenario_id,
            actual_disposition,
            expected_disposition: scenario.expected_disposition,
            expectation_matches,
            findings,
        });
    }

    results.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let contract_only_pass_count = results
        .iter()
        .filter(|r| r.actual_disposition == "PASS_CONTRACT_ONLY")
        .count();
    let impact_exceeded_count = results
        .iter()
        .filter(|r| r.actual_disposition == "IMPACT_EXCEEDS_BUDGET")
        .count();
    let infeasible_thermal_count = results
        .iter()
        .filter(|r| r.actual_disposition == "INFEASIBLE")
        .count();
    let blocked_unknown_count = results
        .iter()
        .filter(|r| r.actual_disposition == "BLOCKED_UNKNOWN")
        .count();
    let safety_blocked_count = results
        .iter()
        .filter(|r| r.actual_disposition == "SAFETY_BLOCKED")
        .count();
    let invalid_count = results
        .iter()
        .filter(|r| r.actual_disposition == "INVALID")
        .count();
    let expectation_mismatch_count =
        results.iter().filter(|r| !r.expectation_matches).count();

    Ok(OpticalAuditReport {
        report_schema: "planetary-optical-impact-audit-report-v0.1",
        source_sha256,
        status: if expectation_mismatch_count == 0 {
            "PASS"
        } else {
            "BLOCKED"
        },
        claim_ceiling: "synthetic_surface_evidence_and_declared_observer_budget_contract_only",
        scenario_count: results.len(),
        contract_only_pass_count,
        impact_exceeded_count,
        infeasible_thermal_count,
        blocked_unknown_count,
        safety_blocked_count,
        invalid_count,
        expectation_mismatch_count,
        scenarios: results,
    })
}

fn optical_finding(code: &str, message: impl Into<String>) -> OpticalFinding {
    OpticalFinding {
        code: code.to_string(),
        message: message.into(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha2::Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../benchmarks/planetary-energy-service/v0.1/optical-impact-cases.json");

    fn fixture_value() -> serde_json::Value {
        serde_json::from_str(FIXTURE).expect("optical-impact fixture parses")
    }

    fn scenario<'a>(report: &'a OpticalAuditReport, id: &str) -> &'a OpticalScenarioResult {
        report
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == id)
            .expect("scenario exists")
    }

    #[test]
    fn fixture_reproduces_all_expected_optical_dispositions() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(report.status, "PASS");
        assert_eq!(report.scenario_count, 8);
        assert_eq!(report.contract_only_pass_count, 1);
        assert_eq!(report.impact_exceeded_count, 1);
        assert_eq!(report.infeasible_thermal_count, 1);
        assert_eq!(report.blocked_unknown_count, 3);
        assert_eq!(report.safety_blocked_count, 2);
        assert_eq!(report.invalid_count, 0);
        assert_eq!(report.expectation_mismatch_count, 0);
    }

    #[test]
    fn missing_brdf_blocks_a_low_glare_claim() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "solar-collector-missing-brdf").actual_disposition,
            "BLOCKED_UNKNOWN"
        );
    }

    #[test]
    fn visible_surface_darkness_does_not_override_radiator_emissivity() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "radiator-insufficient-emissivity").actual_disposition,
            "INFEASIBLE"
        );
    }

    #[test]
    fn optical_transmitter_fails_closed_without_independent_interlock() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "optical-transmitter-interlock-unknown").actual_disposition,
            "SAFETY_BLOCKED"
        );
    }

    #[test]
    fn required_emergency_beacon_cannot_be_suppressed() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "emergency-beacon-unavailable").actual_disposition,
            "SAFETY_BLOCKED"
        );
    }

    #[test]
    fn observer_radiance_over_declared_budget_is_not_passed() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "radiator-over-observer-budget").actual_disposition,
            "IMPACT_EXCEEDS_BUDGET"
        );
    }

    #[test]
    fn absent_observer_limit_blocks_a_low_impact_claim() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture audit parses");
        assert_eq!(
            scenario(&report, "observer-limit-unknown").actual_disposition,
            "BLOCKED_UNKNOWN"
        );
    }

    #[test]
    fn fixture_input_order_does_not_change_scenario_results() {
        let original = audit_corpus(FIXTURE.as_bytes()).expect("fixture parses");
        let mut value = fixture_value();
        value["scenarios"].as_array_mut().expect("scenario array").reverse();
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        let reordered = audit_corpus(&bytes).expect("reordered fixture parses");
        assert_eq!(original.scenarios, reordered.scenarios);
        assert_eq!(original.status, reordered.status);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value = fixture_value();
        value["scenarios"][0]["surface"]["covert_detection_score"] = serde_json::json!(0.99);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        assert!(audit_corpus(&bytes).is_err());
    }

    #[test]
    fn contract_pass_is_never_labelled_physical_qualification() {
        let report = audit_corpus(FIXTURE.as_bytes()).expect("fixture parses");
        assert_eq!(
            report.claim_ceiling,
            "synthetic_surface_evidence_and_declared_observer_budget_contract_only"
        );
        assert_eq!(
            scenario(&report, "solar-collector-reference").actual_disposition,
            "PASS_CONTRACT_ONLY"
        );
    }
}
