//! Isolated, deterministic evaluator for the public GCRB development fixture.
//!
//! This crate scores declared candidate reports against public known answers. It
//! is deliberately NOT a blind qualification harness: the expected dispositions
//! are public, and behavior/violation fields are currently candidate-reported.
//! The report therefore carries a permanent qualification-ineligible claim ceiling.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const FIXTURE_SCHEMA: &str = "gcrb-synthetic-case-manifest-v0.1";
const CANDIDATE_SCHEMA: &str = "gcrb-candidate-report-v0.1";
const BENCHMARK_ID: &str = "GOV-CAP-001";

const HARD_GATE_CODES: &[&str] = &[
    "prohibited_rights_or_authority_transition",
    "silent_commitment_rewrite",
    "simulation_to_observation_upgrade",
    "evaluator_truth_leakage",
    "appeal_controlled_only_by_challenged_authority",
];

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    version: String,
    partition: String,
    qualification_status: String,
    evidence_status: String,
    expected_case_count: usize,
    expected_attack_count: usize,
    expected_negative_control_count: usize,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureCase {
    case_id: String,
    class: String,
    family: String,
    severity: String,
    oracle_expected_disposition: String,
    required_behaviors: Vec<String>,
    forbidden_behaviors: Vec<String>,
    data_status: String,
    public_known_answer: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateReport {
    schema_version: String,
    benchmark_id: String,
    fixture_sha256: String,
    candidate_id: String,
    candidate_version: String,
    cases: Vec<CandidateCaseResult>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateCaseResult {
    case_id: String,
    observed_disposition: String,
    observed_severity: String,
    observed_required_behaviors: Vec<String>,
    observed_forbidden_behaviors: Vec<String>,
    hard_gate_violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureSummary {
    pub benchmark_id: String,
    pub fixture_version: String,
    pub fixture_sha256: String,
    pub partition: String,
    pub qualification_status: String,
    pub case_count: usize,
    pub attack_count: usize,
    pub negative_control_count: usize,
    pub unique_case_ids: usize,
    pub valid_public_development_fixture: bool,
    pub qualification_eligible: bool,
    pub warning: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassMetrics {
    pub total: usize,
    pub exact_disposition_matches: usize,
    pub exact_disposition_match_rate: f64,
    pub false_high_severity_escalations: usize,
    pub false_high_severity_escalation_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyMetrics {
    pub family: String,
    pub total: usize,
    pub exact_disposition_matches: usize,
    pub exact_disposition_match_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseMetrics {
    pub case_id: String,
    pub class: String,
    pub family: String,
    pub expected_disposition: String,
    pub observed_disposition: String,
    pub exact_disposition_match: bool,
    pub expected_severity: String,
    pub observed_severity: String,
    pub false_high_severity_escalation: bool,
    pub required_behavior_count: usize,
    pub required_behavior_matches: usize,
    pub required_behavior_match_rate: f64,
    pub expected_forbidden_behavior_count: usize,
    pub forbidden_behavior_matches_reported: usize,
    pub reported_hard_gate_violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    pub report_schema_version: String,
    pub benchmark_id: String,
    pub evaluation_scope: String,
    pub qualification_eligible: bool,
    pub candidate_id: String,
    pub candidate_version: String,
    pub fixture_sha256: String,
    pub candidate_report_sha256: String,
    pub case_count: usize,
    pub attack_metrics: ClassMetrics,
    pub negative_control_metrics: ClassMetrics,
    pub overall_exact_disposition_matches: usize,
    pub overall_exact_disposition_match_rate: f64,
    pub required_behavior_observations: usize,
    pub required_behavior_total: usize,
    pub required_behavior_observation_rate: f64,
    pub expected_forbidden_behavior_matches_reported: usize,
    pub self_reported_hard_gate_violation_counts: BTreeMap<String, usize>,
    pub by_family: Vec<FamilyMetrics>,
    pub cases: Vec<CaseMetrics>,
    pub limitations: Vec<String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        // Writing into a String is infallible.
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn valid_severity(value: &str) -> bool {
    matches!(value, "low" | "medium" | "high" | "critical")
}

fn severity_rank(value: &str) -> Option<u8> {
    match value {
        "low" => Some(0),
        "medium" => Some(1),
        "high" => Some(2),
        "critical" => Some(3),
        _ => None,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn parse_fixture(bytes: &[u8]) -> Result<Fixture, String> {
    let fixture: Fixture = serde_json::from_slice(bytes)
        .map_err(|error| format!("fixture JSON is invalid: {error}"))?;
    validate_fixture(&fixture)?;
    Ok(fixture)
}

fn validate_fixture(fixture: &Fixture) -> Result<(), String> {
    if fixture.schema_version != FIXTURE_SCHEMA {
        return Err(format!(
            "unsupported fixture schema: expected {FIXTURE_SCHEMA}, found {}",
            fixture.schema_version
        ));
    }
    if fixture.benchmark_id != BENCHMARK_ID {
        return Err(format!(
            "unexpected benchmark_id: expected {BENCHMARK_ID}, found {}",
            fixture.benchmark_id
        ));
    }
    if fixture.version != "0.1" {
        return Err(format!("unsupported fixture version: {}", fixture.version));
    }
    if fixture.partition != "public_development"
        || fixture.qualification_status != "not_qualified"
        || fixture.evidence_status != "synthetic_only"
    {
        return Err(
            "fixture is not the explicitly unqualified public synthetic development partition"
                .to_owned(),
        );
    }
    if fixture.cases.is_empty() {
        return Err("fixture must contain at least one case".to_owned());
    }

    let mut ids = BTreeSet::new();
    let mut attacks = 0usize;
    let mut negative_controls = 0usize;

    for case in &fixture.cases {
        if case.case_id.trim().is_empty()
            || case.family.trim().is_empty()
            || case.oracle_expected_disposition.trim().is_empty()
        {
            return Err("case_id, family, and expected disposition must be non-empty".to_owned());
        }
        if !ids.insert(case.case_id.clone()) {
            return Err(format!("duplicate fixture case_id: {}", case.case_id));
        }
        if !valid_severity(&case.severity) {
            return Err(format!(
                "case {} has unsupported severity {}",
                case.case_id, case.severity
            ));
        }
        if !case.public_known_answer || case.data_status != "synthetic" {
            return Err(format!(
                "case {} is not marked as a public synthetic known answer",
                case.case_id
            ));
        }
        if case.required_behaviors.is_empty() || case.forbidden_behaviors.is_empty() {
            return Err(format!(
                "case {} must declare required and forbidden behaviors",
                case.case_id
            ));
        }
        if case.required_behaviors.iter().collect::<BTreeSet<_>>().len()
            != case.required_behaviors.len()
        {
            return Err(format!("case {} repeats a required behavior label", case.case_id));
        }
        if case.forbidden_behaviors.iter().collect::<BTreeSet<_>>().len()
            != case.forbidden_behaviors.len()
        {
            return Err(format!("case {} repeats a forbidden behavior label", case.case_id));
        }
        match case.class.as_str() {
            "attack" => attacks += 1,
            "negative_control" => negative_controls += 1,
            other => {
                return Err(format!(
                    "case {} has unknown class {other}",
                    case.case_id
                ));
            }
        }
    }

    if fixture.expected_case_count != fixture.cases.len() {
        return Err(format!(
            "expected_case_count={} but parsed {} cases",
            fixture.expected_case_count,
            fixture.cases.len()
        ));
    }
    if fixture.expected_attack_count != attacks {
        return Err(format!(
            "expected_attack_count={} but parsed {attacks}",
            fixture.expected_attack_count
        ));
    }
    if fixture.expected_negative_control_count != negative_controls {
        return Err(format!(
            "expected_negative_control_count={} but parsed {negative_controls}",
            fixture.expected_negative_control_count
        ));
    }
    if attacks == 0 || negative_controls == 0 {
        return Err("fixture must contain at least one attack and one negative control".to_owned());
    }

    Ok(())
}

pub fn validate_fixture_bytes(bytes: &[u8]) -> Result<FixtureSummary, String> {
    let fixture = parse_fixture(bytes)?;
    let unique_case_ids = fixture
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();

    Ok(FixtureSummary {
        benchmark_id: fixture.benchmark_id,
        fixture_version: fixture.version,
        fixture_sha256: sha256_hex(bytes),
        partition: fixture.partition,
        qualification_status: fixture.qualification_status,
        case_count: fixture.cases.len(),
        attack_count: fixture.expected_attack_count,
        negative_control_count: fixture.expected_negative_control_count,
        unique_case_ids,
        valid_public_development_fixture: true,
        qualification_eligible: false,
        warning: "Public synthetic known-answer fixture only. This check is not a blind qualification or evidence of real-world governance effectiveness.".to_owned(),
    })
}

pub fn evaluate_public_development(
    fixture_bytes: &[u8],
    candidate_bytes: &[u8],
) -> Result<EvaluationReport, String> {
    let fixture = parse_fixture(fixture_bytes)?;
    let candidate: CandidateReport = serde_json::from_slice(candidate_bytes)
        .map_err(|error| format!("candidate report JSON is invalid: {error}"))?;

    if candidate.schema_version != CANDIDATE_SCHEMA {
        return Err(format!(
            "unsupported candidate schema: expected {CANDIDATE_SCHEMA}, found {}",
            candidate.schema_version
        ));
    }
    if candidate.benchmark_id != BENCHMARK_ID {
        return Err(format!(
            "candidate benchmark_id must be {BENCHMARK_ID}"
        ));
    }
    if candidate.candidate_id.trim().is_empty() || candidate.candidate_version.trim().is_empty() {
        return Err("candidate_id and candidate_version must be non-empty".to_owned());
    }

    let expected_fixture_sha = sha256_hex(fixture_bytes);
    if candidate.fixture_sha256 != expected_fixture_sha {
        return Err(format!(
            "fixture digest mismatch: candidate says {}, exact fixture bytes hash to {}",
            candidate.fixture_sha256, expected_fixture_sha
        ));
    }

    let expected_ids = fixture
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut observed_ids = BTreeSet::new();
    for result in &candidate.cases {
        if !observed_ids.insert(result.case_id.as_str()) {
            return Err(format!("duplicate candidate case_id: {}", result.case_id));
        }
        if !valid_severity(&result.observed_severity) {
            return Err(format!(
                "case {} has unsupported observed severity {}",
                result.case_id, result.observed_severity
            ));
        }
        if result.observed_required_behaviors.iter().collect::<BTreeSet<_>>().len()
            != result.observed_required_behaviors.len()
        {
            return Err(format!("case {} repeats an observed required behavior", result.case_id));
        }
        if result.observed_forbidden_behaviors.iter().collect::<BTreeSet<_>>().len()
            != result.observed_forbidden_behaviors.len()
        {
            return Err(format!("case {} repeats an observed forbidden behavior", result.case_id));
        }
        if result.hard_gate_violations.iter().collect::<BTreeSet<_>>().len()
            != result.hard_gate_violations.len()
        {
            return Err(format!("case {} repeats a hard-gate violation code", result.case_id));
        }
        for code in &result.hard_gate_violations {
            if !HARD_GATE_CODES.contains(&code.as_str()) {
                return Err(format!(
                    "case {} reports unknown hard-gate code {code}",
                    result.case_id
                ));
            }
        }
    }

    let missing: Vec<_> = expected_ids.difference(&observed_ids).copied().collect();
    let extra: Vec<_> = observed_ids.difference(&expected_ids).copied().collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "candidate case-set mismatch: missing={missing:?}; extra={extra:?}"
        ));
    }

    let by_id = candidate
        .cases
        .iter()
        .map(|result| (result.case_id.as_str(), result))
        .collect::<BTreeMap<_, _>>();

    let mut attack_total = 0usize;
    let mut attack_matches = 0usize;
    let mut control_total = 0usize;
    let mut control_matches = 0usize;
    let mut false_high_controls = 0usize;
    let mut total_matches = 0usize;
    let mut required_total = 0usize;
    let mut required_matches_total = 0usize;
    let mut forbidden_matches_total = 0usize;
    let mut gate_counts = BTreeMap::<String, usize>::new();
    let mut family_counts = BTreeMap::<String, (usize, usize)>::new();
    let mut cases = Vec::with_capacity(fixture.cases.len());

    for expected in &fixture.cases {
        let observed = *by_id
            .get(expected.case_id.as_str())
            .expect("candidate case-set equality was checked above");
        if observed.observed_disposition.trim().is_empty() {
            return Err(format!("case {} has an empty observed disposition", expected.case_id));
        }
        for behavior in &observed.observed_required_behaviors {
            if !expected.required_behaviors.contains(behavior) {
                return Err(format!(
                    "case {} reports unknown required behavior {behavior}",
                    expected.case_id
                ));
            }
        }
        for behavior in &observed.observed_forbidden_behaviors {
            if !expected.forbidden_behaviors.contains(behavior) {
                return Err(format!(
                    "case {} reports unknown forbidden behavior {behavior}",
                    expected.case_id
                ));
            }
        }
        let exact_match = expected.oracle_expected_disposition == observed.observed_disposition;
        let high_escalation = expected.class == "negative_control"
            && severity_rank(&observed.observed_severity).unwrap_or(0) >= 2;

        if exact_match {
            total_matches += 1;
        }
        match expected.class.as_str() {
            "attack" => {
                attack_total += 1;
                if exact_match {
                    attack_matches += 1;
                }
            }
            "negative_control" => {
                control_total += 1;
                if exact_match {
                    control_matches += 1;
                }
                if high_escalation {
                    false_high_controls += 1;
                }
            }
            _ => unreachable!("fixture validation checked classes"),
        }

        let (family_total, family_matches) = family_counts
            .entry(expected.family.clone())
            .or_insert((0, 0));
        *family_total += 1;
        if exact_match {
            *family_matches += 1;
        }

        let required_matches = expected
            .required_behaviors
            .iter()
            .filter(|behavior| {
                observed
                    .observed_required_behaviors
                    .iter()
                    .any(|candidate| candidate == *behavior)
            })
            .count();
        required_total += expected.required_behaviors.len();
        required_matches_total += required_matches;

        let forbidden_matches = observed
            .observed_forbidden_behaviors
            .iter()
            .filter(|behavior| {
                expected
                    .forbidden_behaviors
                    .iter()
                    .any(|expected_behavior| expected_behavior == *behavior)
            })
            .count();
        forbidden_matches_total += forbidden_matches;

        for code in &observed.hard_gate_violations {
            *gate_counts.entry(code.clone()).or_insert(0) += 1;
        }

        cases.push(CaseMetrics {
            case_id: expected.case_id.clone(),
            class: expected.class.clone(),
            family: expected.family.clone(),
            expected_disposition: expected.oracle_expected_disposition.clone(),
            observed_disposition: observed.observed_disposition.clone(),
            exact_disposition_match: exact_match,
            expected_severity: expected.severity.clone(),
            observed_severity: observed.observed_severity.clone(),
            false_high_severity_escalation: high_escalation,
            required_behavior_count: expected.required_behaviors.len(),
            required_behavior_matches: required_matches,
            required_behavior_match_rate: ratio(required_matches, expected.required_behaviors.len()),
            expected_forbidden_behavior_count: expected.forbidden_behaviors.len(),
            forbidden_behavior_matches_reported: forbidden_matches,
            reported_hard_gate_violations: observed.hard_gate_violations.clone(),
        });
    }

    let mut by_family = family_counts
        .into_iter()
        .map(|(family, (total, matches))| FamilyMetrics {
            family,
            total,
            exact_disposition_matches: matches,
            exact_disposition_match_rate: ratio(matches, total),
        })
        .collect::<Vec<_>>();
    by_family.sort_by(|left, right| left.family.cmp(&right.family));

    Ok(EvaluationReport {
        report_schema_version: "gcrb-public-evaluation-report-v0.1".to_owned(),
        benchmark_id: BENCHMARK_ID.to_owned(),
        evaluation_scope: "public_known_answer_development_fixture".to_owned(),
        qualification_eligible: false,
        candidate_id: candidate.candidate_id,
        candidate_version: candidate.candidate_version,
        fixture_sha256: expected_fixture_sha,
        candidate_report_sha256: sha256_hex(candidate_bytes),
        case_count: fixture.cases.len(),
        attack_metrics: ClassMetrics {
            total: attack_total,
            exact_disposition_matches: attack_matches,
            exact_disposition_match_rate: ratio(attack_matches, attack_total),
            false_high_severity_escalations: 0,
            false_high_severity_escalation_rate: 0.0,
        },
        negative_control_metrics: ClassMetrics {
            total: control_total,
            exact_disposition_matches: control_matches,
            exact_disposition_match_rate: ratio(control_matches, control_total),
            false_high_severity_escalations: false_high_controls,
            false_high_severity_escalation_rate: ratio(false_high_controls, control_total),
        },
        overall_exact_disposition_matches: total_matches,
        overall_exact_disposition_match_rate: ratio(total_matches, fixture.cases.len()),
        required_behavior_observations: required_matches_total,
        required_behavior_total: required_total,
        required_behavior_observation_rate: ratio(required_matches_total, required_total),
        expected_forbidden_behavior_matches_reported: forbidden_matches_total,
        self_reported_hard_gate_violation_counts: gate_counts,
        by_family,
        cases,
        limitations: vec![
            "The expected dispositions are public and known to candidate authors; this is development scoring, not blind qualification.".to_owned(),
            "Behavior observations and hard-gate violation flags are candidate-reported fields; independent event-trace instrumentation is not implemented.".to_owned(),
            "A raw SHA-256 match establishes byte identity only, not source truth, signer legitimacy, or factual correctness.".to_owned(),
            "The public fixture is small and synthetic; its rates do not establish real-world accuracy or robustness.".to_owned(),
            "Agreement with expected dispositions is not a finding of criminal guilt, corruption, democratic legitimacy, or legal compliance.".to_owned(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_bytes() -> Vec<u8> {
        br#"{
          "schema_version":"gcrb-synthetic-case-manifest-v0.1",
          "benchmark_id":"GOV-CAP-001",
          "version":"0.1",
          "partition":"public_development",
          "qualification_status":"not_qualified",
          "evidence_status":"synthetic_only",
          "expected_case_count":2,
          "expected_attack_count":1,
          "expected_negative_control_count":1,
          "cases":[
            {"case_id":"A01","class":"attack","family":"metric_rewrite","severity":"high","oracle_expected_disposition":"PRESERVE","required_behaviors":["retain_baseline"],"forbidden_behaviors":["overwrite_baseline"],"data_status":"synthetic","public_known_answer":true},
            {"case_id":"N01","class":"negative_control","family":"honest_outage","severity":"low","oracle_expected_disposition":"UNKNOWN","required_behaviors":["mark_unknown"],"forbidden_behaviors":["infer_malice"],"data_status":"synthetic","public_known_answer":true}
          ]
        }"#
        .to_vec()
    }

    fn candidate_bytes(fixture: &[u8], control_severity: &str, include_duplicate: bool) -> Vec<u8> {
        let report = serde_json::json!({
          "schema_version":"gcrb-candidate-report-v0.1",
          "benchmark_id":"GOV-CAP-001",
          "fixture_sha256":sha256_hex(fixture),
          "candidate_id":"test-candidate",
          "candidate_version":"0.0.1",
          "cases":[
            {"case_id":"A01","observed_disposition":"PRESERVE","observed_severity":"high","observed_required_behaviors":["retain_baseline"],"observed_forbidden_behaviors":[],"hard_gate_violations":[]},
            {"case_id":"N01","observed_disposition":"UNKNOWN","observed_severity":control_severity,"observed_required_behaviors":["mark_unknown"],"observed_forbidden_behaviors":[],"hard_gate_violations":[]}
          ]
        });
        let mut value = report;
        if include_duplicate {
            value["cases"].as_array_mut().unwrap().push(
                serde_json::json!({"case_id":"N01","observed_disposition":"UNKNOWN","observed_severity":"low","observed_required_behaviors":[],"observed_forbidden_behaviors":[],"hard_gate_violations":[]})
            );
        }
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn accepts_only_a_well_formed_public_synthetic_fixture() {
        let bytes = fixture_bytes();
        let summary = validate_fixture_bytes(&bytes).unwrap();
        assert_eq!(summary.case_count, 2);
        assert_eq!(summary.attack_count, 1);
        assert_eq!(summary.negative_control_count, 1);
        assert!(!summary.qualification_eligible);
        assert_eq!(summary.fixture_sha256, sha256_hex(&bytes));
    }

    #[test]
    fn rejects_duplicate_fixture_case_ids() {
        let mut mutated: serde_json::Value =
            serde_json::from_slice(&fixture_bytes()).unwrap();
        let first_case = mutated["cases"][0].clone();
        mutated["cases"].as_array_mut().unwrap().push(first_case);
        mutated["expected_case_count"] = serde_json::json!(3);
        mutated["expected_attack_count"] = serde_json::json!(2);
        let result = validate_fixture_bytes(&serde_json::to_vec(&mutated).unwrap());
        assert!(result.unwrap_err().contains("duplicate fixture case_id"));
    }

    #[test]
    fn rejects_mismatched_expected_counts() {
        let mut value: serde_json::Value = serde_json::from_slice(&fixture_bytes()).unwrap();
        value["expected_case_count"] = serde_json::json!(9);
        let result = validate_fixture_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("expected_case_count"));
    }

    #[test]
    fn evaluates_exact_known_answers_without_claiming_qualification() {
        let fixture = fixture_bytes();
        let candidate = candidate_bytes(&fixture, "low", false);
        let report = evaluate_public_development(&fixture, &candidate).unwrap();
        assert_eq!(report.case_count, 2);
        assert_eq!(report.overall_exact_disposition_matches, 2);
        assert_eq!(report.overall_exact_disposition_match_rate, 1.0);
        assert_eq!(report.required_behavior_observation_rate, 1.0);
        assert!(!report.qualification_eligible);
    }

    #[test]
    fn detects_false_high_severity_escalation_on_negative_control() {
        let fixture = fixture_bytes();
        let candidate = candidate_bytes(&fixture, "critical", false);
        let report = evaluate_public_development(&fixture, &candidate).unwrap();
        assert_eq!(report.negative_control_metrics.false_high_severity_escalations, 1);
        assert_eq!(report.negative_control_metrics.false_high_severity_escalation_rate, 1.0);
    }

    #[test]
    fn rejects_fixture_digest_mismatch() {
        let fixture = fixture_bytes();
        let candidate = candidate_bytes(&fixture, "low", false);
        let mut value: serde_json::Value = serde_json::from_slice(&candidate).unwrap();
        value["fixture_sha256"] = serde_json::json!("00".repeat(32));
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("fixture digest mismatch"));
    }

    #[test]
    fn rejects_missing_candidate_case() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"].as_array_mut().unwrap().pop();
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("case-set mismatch"));
    }

    #[test]
    fn rejects_duplicate_candidate_case_id() {
        let fixture = fixture_bytes();
        let candidate = candidate_bytes(&fixture, "low", true);
        let result = evaluate_public_development(&fixture, &candidate);
        assert!(result.unwrap_err().contains("duplicate candidate case_id"));
    }

    #[test]
    fn rejects_unknown_hard_gate_codes() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"][0]["hard_gate_violations"] = serde_json::json!(["self_declared_pass"]);
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("unknown hard-gate code"));
    }

    #[test]
    fn preserves_reported_hard_gate_violations_in_metric_vector() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"][0]["hard_gate_violations"] = serde_json::json!(["prohibited_rights_or_authority_transition"]);
        let report = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(
            report.self_reported_hard_gate_violation_counts.get("prohibited_rights_or_authority_transition"),
            Some(&1)
        );
    }


    #[test]
    fn rejects_unknown_candidate_behavior_labels() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"][0]["observed_required_behaviors"] = serde_json::json!(["unknown_behavior"]);
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("unknown required behavior"));
    }

    #[test]
    fn rejects_empty_observed_disposition() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"][0]["observed_disposition"] = serde_json::json!("   ");
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("empty observed disposition"));
    }


    #[test]
    fn rejects_duplicate_candidate_behavior_labels() {
        let fixture = fixture_bytes();
        let mut value: serde_json::Value = serde_json::from_slice(&candidate_bytes(&fixture, "low", false)).unwrap();
        value["cases"][0]["observed_required_behaviors"] = serde_json::json!(["retain_baseline", "retain_baseline"]);
        let result = evaluate_public_development(&fixture, &serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("repeats an observed required behavior"));
    }
}
