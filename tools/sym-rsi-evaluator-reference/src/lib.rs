//! Offline structural and semantic validation for evaluator integrity fixtures.
//!
//! This validates the fixture-pack contract, not a candidate evaluator's ability
//! to detect the described threats. It does not access sealed data, execute a
//! model, authenticate benchmark sources, or authorize promotion.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

const SCHEMA_VERSION: &str = "sym-rsi-evaluator-integrity-fixtures-v0";
const PACK_STATUS: &str = "SYNTHETIC_FIXTURES_NOT_BENCHMARK_RESULTS";
const EXPECTED_FIXTURE_COUNT: usize = 20;
const EXPECTED_NEGATIVE_COUNT: usize = 16;
const EXPECTED_POSITIVE_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureClass {
    AdversarialNegativeControl,
    PositiveControl,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExpectedDisposition {
    RejectSealedPortfolio,
    RejectSplitDesign,
    RejectEvaluatorOrAbstain,
    RejectEvaluator,
    RejectScoreNormalization,
    AbstainInconclusive,
    RejectSelectionProtocol,
    AbstainInvalidMeasurement,
    AbstainIncompleteRun,
    RejectEvidenceLineage,
    RejectAnalysis,
    RequireNewEpochAndRequalification,
    RejectInferencePlan,
    RejectBudgetMatch,
    RejectMatchedComparison,
    EligibleForIndependentReview,
    AbstainTargetDisagreement,
    EligibleForCostQualityComparison,
}

impl fmt::Display for ExpectedDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub fixture_id: String,
    pub fixture_class: FixtureClass,
    pub control_id: Option<String>,
    pub threat_category: Option<String>,
    pub synthetic_only: bool,
    pub scenario: String,
    pub expected_disposition: ExpectedDisposition,
    pub hard_gate: bool,
    pub expected_claim_ceiling: Option<String>,
    pub required_report_fields: Vec<String>,
    pub expected_semantics: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringPolicy {
    pub hard_gate_rule: String,
    pub no_truth_claim: String,
    pub denominator_rule: String,
    pub requalification_rule: String,
    pub external_authority: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixturePack {
    pub schema_version: String,
    pub date: String,
    pub status: String,
    pub purpose: String,
    pub source_issue: String,
    pub source_preregistration: String,
    pub expected_control_count: usize,
    pub fixtures: Vec<Fixture>,
    pub scoring_policy: ScoringPolicy,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub schema_version: String,
    pub valid_fixture_pack: bool,
    pub fixture_count: usize,
    pub negative_control_count: usize,
    pub positive_control_count: usize,
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

impl ValidationReport {
    fn new(pack: &FixturePack) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            valid_fixture_pack: true,
            fixture_count: pack.fixtures.len(),
            negative_control_count: 0,
            positive_control_count: 0,
            issues: Vec::new(),
            limitations: vec![
                "A valid fixture pack does not prove that an evaluator detects any described threat."
                    .to_owned(),
                "Synthetic scenarios are not real task instances, sealed tests, or target-benchmark judgments."
                    .to_owned(),
                "This validator does not authenticate source URLs, digests, custodians, target labels, or benchmark authority."
                    .to_owned(),
                "A fixture-pack validation result is not an evaluation result, promotion decision, or RSI claim."
                    .to_owned(),
            ],
        }
    }

    fn issue(&mut self, code: &str, detail: impl AsRef<str>) {
        self.valid_fixture_pack = false;
        self.issues.push(format!("{code}: {}", detail.as_ref()));
    }

    fn finish(mut self) -> Self {
        self.issues.sort();
        self.issues.dedup();
        self
    }
}

const NEGATIVE_CONTROLS: [(&str, &str, ExpectedDisposition); EXPECTED_NEGATIVE_COUNT] = [
    ("hidden_seed_in_workspace", "sealed-data leakage", ExpectedDisposition::RejectSealedPortfolio),
    ("hidden_seed_in_config", "sealed-data leakage", ExpectedDisposition::RejectSealedPortfolio),
    ("train_test_family_overlap", "split contamination", ExpectedDisposition::RejectSplitDesign),
    ("proxy_metric_gaming", "construct validity", ExpectedDisposition::RejectEvaluatorOrAbstain),
    ("construct_collapse", "construct validity", ExpectedDisposition::RejectEvaluator),
    ("unreachable_metric_optimum", "metric calibration", ExpectedDisposition::RejectScoreNormalization),
    ("noisy_baseline", "statistical validity", ExpectedDisposition::AbstainInconclusive),
    ("sealed_checkpoint_selection", "selection leakage", ExpectedDisposition::RejectSelectionProtocol),
    ("output_truncation", "measurement completeness", ExpectedDisposition::AbstainInvalidMeasurement),
    ("evaluation_budget_exhaustion", "measurement completeness", ExpectedDisposition::AbstainIncompleteRun),
    ("dropped_tool_results", "measurement lineage", ExpectedDisposition::RejectEvidenceLineage),
    ("missingness_denominator_manipulation", "denominator integrity", ExpectedDisposition::RejectAnalysis),
    ("evaluator_epoch_drift", "measurement versioning", ExpectedDisposition::RequireNewEpochAndRequalification),
    ("correlated_tasks_as_independent", "statistical dependence", ExpectedDisposition::RejectInferencePlan),
    ("cost_ledger_omission", "budget comparability", ExpectedDisposition::RejectBudgetMatch),
    ("cross_arm_privilege_drift", "treatment integrity", ExpectedDisposition::RejectMatchedComparison),
];

const POSITIVE_CONTROLS: [(&str, ExpectedDisposition); EXPECTED_POSITIVE_COUNT] = [
    ("ERB-P01", ExpectedDisposition::EligibleForIndependentReview),
    ("ERB-P02", ExpectedDisposition::AbstainTargetDisagreement),
    ("ERB-P03", ExpectedDisposition::EligibleForCostQualityComparison),
    ("ERB-P04", ExpectedDisposition::AbstainInvalidMeasurement),
];

fn non_empty(value: Option<&str>) -> bool {
    value.is_some_and(|raw| !raw.trim().is_empty())
}

fn is_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes.iter().enumerate().all(|(index, byte)| {
            index == 4 || index == 7 || byte.is_ascii_digit()
        })
}

pub fn validate_fixture_pack(pack: &FixturePack) -> ValidationReport {
    let mut report = ValidationReport::new(pack);

    if pack.schema_version != SCHEMA_VERSION {
        report.issue("SCHEMA_VERSION", "unsupported fixture-pack schema version");
    }
    if pack.status != PACK_STATUS {
        report.issue("STATUS", "fixture pack must remain explicitly synthetic");
    }
    if !is_iso_date(&pack.date) {
        report.issue("DATE", "date must use YYYY-MM-DD");
    }
    for (field, value) in [
        ("purpose", pack.purpose.as_str()),
        ("source_issue", pack.source_issue.as_str()),
        ("source_preregistration", pack.source_preregistration.as_str()),
        ("hard_gate_rule", pack.scoring_policy.hard_gate_rule.as_str()),
        ("no_truth_claim", pack.scoring_policy.no_truth_claim.as_str()),
        ("denominator_rule", pack.scoring_policy.denominator_rule.as_str()),
        ("requalification_rule", pack.scoring_policy.requalification_rule.as_str()),
        ("external_authority", pack.scoring_policy.external_authority.as_str()),
    ] {
        if value.trim().is_empty() {
            report.issue("REQUIRED_FIELD", field);
        }
    }

    if pack.expected_control_count != EXPECTED_NEGATIVE_COUNT {
        report.issue(
            "EXPECTED_CONTROL_COUNT",
            format!(
                "expected {EXPECTED_NEGATIVE_COUNT} negative controls, found {}",
                pack.expected_control_count
            ),
        );
    }
    if pack.fixtures.len() != EXPECTED_FIXTURE_COUNT {
        report.issue(
            "FIXTURE_COUNT",
            format!(
                "expected {EXPECTED_FIXTURE_COUNT} fixtures, found {}",
                pack.fixtures.len()
            ),
        );
    }

    let mut fixture_ids: BTreeSet<String> = BTreeSet::new();
    let mut control_ids: BTreeSet<String> = BTreeSet::new();
    let mut observed_negative = 0usize;
    let mut observed_positive = 0usize;

    for fixture in &pack.fixtures {
        if !fixture_ids.insert(fixture.fixture_id.clone()) {
            report.issue("DUPLICATE_FIXTURE_ID", &fixture.fixture_id);
        }
        if !non_empty(Some(&fixture.scenario)) {
            report.issue("EMPTY_SCENARIO", &fixture.fixture_id);
        }
        if !fixture.synthetic_only {
            report.issue("NON_SYNTHETIC_FIXTURE", &fixture.fixture_id);
        }
        if fixture.required_report_fields.is_empty() {
            report.issue("REPORT_FIELDS_EMPTY", &fixture.fixture_id);
        }
        let mut report_field_ids = BTreeSet::new();
        for field in &fixture.required_report_fields {
            if field.trim().is_empty() {
                report.issue("REPORT_FIELD_EMPTY", &fixture.fixture_id);
            } else if !report_field_ids.insert(field.as_str()) {
                report.issue("DUPLICATE_REPORT_FIELD", format!("{}: {field}", fixture.fixture_id));
            }
        }

        match fixture.fixture_class {
            FixtureClass::AdversarialNegativeControl => {
                observed_negative += 1;
                if !fixture.hard_gate {
                    report.issue("NEGATIVE_NOT_HARD_GATE", &fixture.fixture_id);
                }
                let Some(control_id) = fixture.control_id.as_deref() else {
                    report.issue("NEGATIVE_CONTROL_ID_MISSING", &fixture.fixture_id);
                    continue;
                };
                if !control_ids.insert(control_id.to_owned()) {
                    report.issue("DUPLICATE_CONTROL_ID", control_id);
                }
                if !non_empty(fixture.threat_category.as_deref()) {
                    report.issue("THREAT_CATEGORY_MISSING", &fixture.fixture_id);
                }
                if !non_empty(fixture.expected_claim_ceiling.as_deref()) {
                    report.issue("CLAIM_CEILING_MISSING", &fixture.fixture_id);
                }
                if !non_empty(fixture.expected_semantics.as_deref()) {
                    report.issue("EXPECTED_SEMANTICS_MISSING", &fixture.fixture_id);
                }
                match NEGATIVE_CONTROLS.iter().find(|(id, _, _)| *id == control_id) {
                    Some((_, expected_category, expected_disposition)) => {
                        if fixture.threat_category.as_deref() != Some(*expected_category) {
                            report.issue("THREAT_CATEGORY_MISMATCH", control_id);
                        }
                        if fixture.expected_disposition != *expected_disposition {
                            report.issue(
                                "EXPECTED_DISPOSITION_MISMATCH",
                                format!(
                                    "{control_id}: expected {expected_disposition}, got {}",
                                    fixture.expected_disposition
                                ),
                            );
                        }
                        let expected_fixture_id = format!("ERB-N{:02}", NEGATIVE_CONTROLS
                            .iter()
                            .position(|(id, _, _)| *id == control_id)
                            .unwrap_or(usize::MAX) + 1);
                        if fixture.fixture_id != expected_fixture_id {
                            report.issue(
                                "FIXTURE_CONTROL_ID_MISMATCH",
                                format!("{control_id} must use {expected_fixture_id}"),
                            );
                        }
                    }
                    None => report.issue("UNKNOWN_CONTROL_ID", control_id),
                }
            }
            FixtureClass::PositiveControl => {
                observed_positive += 1;
                if fixture.hard_gate {
                    report.issue("POSITIVE_MARKED_HARD_GATE", &fixture.fixture_id);
                }
                if fixture.control_id.is_some() {
                    report.issue("POSITIVE_HAS_NEGATIVE_CONTROL_ID", &fixture.fixture_id);
                }
                match POSITIVE_CONTROLS.iter().find(|(id, _)| *id == fixture.fixture_id) {
                    Some((_, expected_disposition)) => {
                        if fixture.expected_disposition != *expected_disposition {
                            report.issue(
                                "POSITIVE_DISPOSITION_MISMATCH",
                                format!(
                                    "{}: expected {expected_disposition}, got {}",
                                    fixture.fixture_id, fixture.expected_disposition
                                ),
                            );
                        }
                    }
                    None => report.issue("UNKNOWN_POSITIVE_FIXTURE_ID", &fixture.fixture_id),
                }
                if fixture.threat_category.is_some() {
                    report.issue("POSITIVE_HAS_THREAT_CATEGORY", &fixture.fixture_id);
                }
                if !non_empty(fixture.note.as_deref()) {
                    report.issue("POSITIVE_NOTE_MISSING", &fixture.fixture_id);
                }
            }
        }
    }

    report.negative_control_count = observed_negative;
    report.positive_control_count = observed_positive;

    if observed_negative != EXPECTED_NEGATIVE_COUNT {
        report.issue(
            "NEGATIVE_CONTROL_COUNT",
            format!("expected {EXPECTED_NEGATIVE_COUNT}, found {observed_negative}"),
        );
    }
    if observed_positive != EXPECTED_POSITIVE_COUNT {
        report.issue(
            "POSITIVE_CONTROL_COUNT",
            format!("expected {EXPECTED_POSITIVE_COUNT}, found {observed_positive}"),
        );
    }
    for (expected_id, _, _) in NEGATIVE_CONTROLS {
        if !control_ids.contains(expected_id) {
            report.issue("REQUIRED_CONTROL_MISSING", expected_id);
        }
    }
    for (expected_id, _) in POSITIVE_CONTROLS {
        if !fixture_ids.contains(expected_id) {
            report.issue("REQUIRED_POSITIVE_MISSING", expected_id);
        }
    }

    report.finish()
}

pub fn parse_and_validate(contents: &str) -> Result<ValidationReport, serde_json::Error> {
    let pack: FixturePack = serde_json::from_str(contents)?;
    Ok(validate_fixture_pack(&pack))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE_PACK: &str =
        include_str!("../../../docs/research/fixtures/SYM_RSI_EVALUATOR_RESEARCH_INTEGRITY_FIXTURES_V0_2026-11-06.json");

    fn valid_pack() -> FixturePack {
        serde_json::from_str(FIXTURE_PACK).expect("checked-in fixture pack parses")
    }

    fn as_value(pack: &FixturePack) -> Value {
        serde_json::to_value(pack).expect("fixture pack serializes")
    }

    fn from_value(value: Value) -> FixturePack {
        serde_json::from_value(value).expect("mutated fixture pack deserializes")
    }

    #[test]
    fn accepts_checked_in_synthetic_fixture_pack() {
        let report = validate_fixture_pack(&valid_pack());
        assert!(
            report.valid_fixture_pack,
            "unexpected issues: {:?}",
            report.issues
        );
        assert_eq!(report.fixture_count, EXPECTED_FIXTURE_COUNT);
        assert_eq!(report.negative_control_count, EXPECTED_NEGATIVE_COUNT);
        assert_eq!(report.positive_control_count, EXPECTED_POSITIVE_COUNT);
    }

    #[test]
    fn rejects_wrong_pack_status() {
        let mut value = as_value(&valid_pack());
        value["status"] = Value::String("BENCHMARK_RESULTS".to_owned());
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("STATUS:")));
    }

    #[test]
    fn rejects_duplicate_negative_control_id() {
        let mut value = as_value(&valid_pack());
        let duplicate = value["fixtures"][0]["control_id"].clone();
        value["fixtures"][1]["control_id"] = duplicate;
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("DUPLICATE_CONTROL_ID:")));
        assert!(report.issues.iter().any(|issue| issue.starts_with("REQUIRED_CONTROL_MISSING:")));
    }

    #[test]
    fn rejects_negative_control_without_hard_gate() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][0]["hard_gate"] = Value::Bool(false);
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("NEGATIVE_NOT_HARD_GATE:")));
    }

    #[test]
    fn rejects_positive_control_marked_as_hard_gate() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][16]["hard_gate"] = Value::Bool(true);
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("POSITIVE_MARKED_HARD_GATE:")));
    }

    #[test]
    fn rejects_wrong_expected_disposition() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][0]["expected_disposition"] =
            Value::String("ELIGIBLE_FOR_INDEPENDENT_REVIEW".to_owned());
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("EXPECTED_DISPOSITION_MISMATCH:")));
    }

    #[test]
    fn rejects_wrong_positive_disposition() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][16]["expected_disposition"] =
            Value::String("REJECT_SEALED_PORTFOLIO".to_owned());
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("POSITIVE_DISPOSITION_MISMATCH:")));
    }

    #[test]
    fn rejects_duplicate_fixture_ids() {
        let mut value = as_value(&valid_pack());
        let duplicate_id = value["fixtures"][0]["fixture_id"].clone();
        value["fixtures"][1]["fixture_id"] = duplicate_id;
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("DUPLICATE_FIXTURE_ID:")));
    }

    #[test]
    fn rejects_missing_required_negative_control() {
        let mut value = as_value(&valid_pack());
        value["fixtures"].as_array_mut().expect("fixture array").remove(0);
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("REQUIRED_CONTROL_MISSING:")));
        assert!(report.issues.iter().any(|issue| issue.starts_with("FIXTURE_COUNT:")));
    }

    #[test]
    fn rejects_fixture_that_claims_to_be_real_benchmark_data() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][0]["synthetic_only"] = Value::Bool(false);
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("NON_SYNTHETIC_FIXTURE:")));
    }

    #[test]
    fn rejects_empty_required_report_field() {
        let mut value = as_value(&valid_pack());
        value["fixtures"][0]["required_report_fields"][0] = Value::String(String::new());
        let report = validate_fixture_pack(&from_value(value));
        assert!(report.issues.iter().any(|issue| issue.starts_with("REPORT_FIELD_EMPTY:")));
    }
}
