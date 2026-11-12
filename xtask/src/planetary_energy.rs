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

        let computed = if case.heliocentric_distance_au.is_finite()
            && case.heliocentric_distance_au > 0.0
            && model.solar_irradiance_at_1_au_w_m2.is_finite()
            && model.solar_irradiance_at_1_au_w_m2 > 0.0
        {
            Some(model.solar_irradiance_at_1_au_w_m2
                / case.heliocentric_distance_au.powi(2))
        } else {
            None
        };

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
