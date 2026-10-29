//! Isolated, deterministic evaluator for the public GCRB development fixture.
//!
//! This crate scores declared candidate reports against public known answers. It
//! is deliberately NOT a blind qualification harness: the expected dispositions
//! are public, and behavior/violation fields are currently candidate-reported.
//! The report therefore carries a permanent qualification-ineligible claim ceiling.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub mod trace_audit;

const FIXTURE_SCHEMA: &str = "gcrb-synthetic-case-manifest-v0.1";
const CANDIDATE_SCHEMA: &str = "gcrb-candidate-report-v0.1";
const BENCHMARK_ID: &str = "GOV-CAP-001";

const HARD_GATE_CODES: &[&str] = &[
    "prohibited_rights_or_authority_transition",
    "silent_commitment_rewrite",
    "simulation_to_observation_upgrade",
    "evaluator_truth_leakage",
    "appeal_controlled_only_by_challenged_authority",
    "unbounded_emergency_bypass",
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


fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sample_inclusive(state: &mut u64, low: u64, high: u64) -> u64 {
    debug_assert!(low <= high);
    low + (splitmix64_next(state) % (high - low + 1))
}

fn make_control_parameters(family: &str, state: &mut u64) -> serde_json::Value {
    use serde_json::json;

    match family {
        "honest_data_outage" => json!({
            "outage_duration_hours": sample_inclusive(state, 1, 720),
            "reported_missingness_percent": sample_inclusive(state, 1, 80),
            "independent_failure_domain_count": sample_inclusive(state, 1, 4),
            "outage_documented": true,
            "concealment_evidence_present": false
        }),
        "pre_outcome_authorized_metric_revision" => json!({
            "days_before_outcome_visible": sample_inclusive(state, 1, 120),
            "revision_count": sample_inclusive(state, 1, 3),
            "prior_version_retained": true,
            "authority_recorded": true,
            "revision_reason_recorded": true
        }),
        "independent_source_disagreement" => json!({
            "source_count": sample_inclusive(state, 2, 8),
            "population_coverage_gap_percent": sample_inclusive(state, 1, 80),
            "time_window_offset_days": sample_inclusive(state, 1, 180),
            "sources_independently_owned": true,
            "scope_difference_documented": true
        }),
        "good_faith_incorrect_citizen_report" => json!({
            "reports_in_same_episode": sample_inclusive(state, 1, 30),
            "correction_lag_days": sample_inclusive(state, 1, 60),
            "initial_report_known_answer_correct": false,
            "good_faith_context_preserved": true,
            "correction_recorded": true
        }),
        "documented_external_shock" => json!({
            "external_shock_magnitude_percent": sample_inclusive(state, 1, 95),
            "forecast_interval_width_percent": sample_inclusive(state, 1, 60),
            "shock_documented": true,
            "evaluation_protocol_unchanged": true
        }),
        "valid_source_retraction" => json!({
            "source_error_rate_percent": sample_inclusive(state, 1, 30),
            "retraction_delay_days": sample_inclusive(state, 1, 90),
            "retraction_reason_documented": true,
            "prior_record_retained": true
        }),
        "authorized_bounded_emergency" => json!({
            "emergency_duration_hours": sample_inclusive(state, 1, 96),
            "review_due_hours": sample_inclusive(state, 1, 72),
            "authority_scope_declared": true,
            "expiry_present": true,
            "follow_up_review_required": true
        }),
        "small_subgroup_sample" => {
            let sample_size = sample_inclusive(state, 1, 30);
            json!({
                "subgroup_sample_size": sample_size,
                "population_frame_size": sample_size + sample_inclusive(state, 100, 50_000),
                "missingness_percent": sample_inclusive(state, 0, 95),
                "deliberate_omission_evidence_present": false,
                "firm_subgroup_claim_permitted": false
            })
        }
        _ => json!({
            "generic_variation_index": sample_inclusive(state, 0, u32::MAX as u64),
            "source_count": sample_inclusive(state, 1, 8),
            "evidence_missingness_percent": sample_inclusive(state, 0, 95),
            "deliberate_misconduct_established": false
        }),
    }
}

/// Create a deterministic, public-development stress fixture by retaining all
/// attack cases and replacing the hand-written negative controls with seeded variants.
/// This does not create blind data or add real-world ground truth. Expected
/// dispositions remain public by design.
///
/// The algorithm and output are versioned so that a later change requires a
/// generator-version bump. Generated case IDs and seeds are unique within the
/// fixture. Maximum count is bounded to avoid untrusted-input memory exhaustion.
pub fn generate_parameterized_controls(
    fixture_bytes: &[u8],
    control_count: usize,
    master_seed: u64,
) -> Result<Vec<u8>, String> {
    const MAX_CONTROLS: usize = 100_000;
    const GENERATOR_NAME: &str = "gcrb-parameterized-control-generator";
    const GENERATOR_VERSION: &str = "0.1";

    if control_count == 0 || control_count > MAX_CONTROLS {
        return Err(format!(
            "control_count must be between 1 and {MAX_CONTROLS}"
        ));
    }
    let base = parse_fixture(fixture_bytes)?;
    let mut document: serde_json::Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON is invalid: {error}"))?;
    if document.get("generator").is_some() {
        return Err(
            "source fixture must be hand-authored public development data, not a previous generated stress fixture"
                .to_owned(),
        );
    }
    let source_fixture_sha256 = sha256_hex(fixture_bytes);
    let base_cases = document["cases"]
        .as_array()
        .ok_or_else(|| "fixture cases must be an array".to_owned())?;
    let attack_cases = base_cases
        .iter()
        .filter(|case| case["class"].as_str() == Some("attack"))
        .cloned()
        .collect::<Vec<_>>();
    let control_templates = base_cases
        .iter()
        .filter(|case| case["class"].as_str() == Some("negative_control"))
        .cloned()
        .collect::<Vec<_>>();

    if attack_cases.len() != base.expected_attack_count {
        return Err("base fixture attack count changed during generation".to_owned());
    }
    if control_templates.len() != base.expected_negative_control_count || control_templates.is_empty() {
        return Err("base fixture must provide its declared negative-control templates".to_owned());
    }
    if master_seed.checked_add(control_count as u64 - 1).is_none() {
        return Err("master_seed + control_count overflows u64".to_owned());
    }

    let mut generated = attack_cases;
    generated.reserve(control_count);
    let mut random_state = master_seed;

    for index in 0..control_count {
        let template = &control_templates[index % control_templates.len()];
        let family = template["family"]
            .as_str()
            .ok_or_else(|| format!("control template {index} has no family"))?;
        let case_seed = master_seed + index as u64;
        let parameters = make_control_parameters(family, &mut random_state);
        let case_id = format!("GCRB-NP{:06}", index + 1);
        let scenario = format!(
            "Parameterized benign-control instance derived from '{}'; inspect the typed parameters for this deterministic variant.",
            template["scenario"].as_str().unwrap_or(family)
        );
        let mut case = template.clone();
        case["case_id"] = serde_json::Value::String(case_id);
        case["seed"] = serde_json::Value::Number(case_seed.into());
        case["scenario"] = serde_json::Value::String(scenario);
        case["parameters"] = parameters;
        generated.push(case);
    }

    let expected_cases = base
        .expected_attack_count
        .checked_add(control_count)
        .ok_or_else(|| "expected case count overflow".to_owned())?;
    document["cases"] = serde_json::Value::Array(generated);
    document["expected_case_count"] = serde_json::Value::Number(expected_cases.into());
    document["expected_attack_count"] =
        serde_json::Value::Number(base.expected_attack_count.into());
    document["expected_negative_control_count"] =
        serde_json::Value::Number(control_count.into());
    document["generator"] = serde_json::json!({
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "master_seed": master_seed,
        "parameterized_control_count": control_count,
        "source_fixture_sha256": source_fixture_sha256,
        "algorithm": "splitmix64-v1",
        "known_answers_public": true
    });
    document["purpose"] = serde_json::Value::String(
        "Public parameterized synthetic stress fixture. All generated controls and expected dispositions are public known answers; this remains development scoring, not blind qualification or a real-world error-rate estimate.".to_owned(),
    );
    document["provisional_targets"]["sample_warning"] = serde_json::Value::String(
        "The parameterized corpus provides repeatable synthetic stress coverage only. Public expected answers, the generator's assumptions, and candidate-reported behavior mean it is not a blind evaluation and cannot estimate real-world error rates.".to_owned(),
    );

    let mut output = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("cannot serialize generated fixture: {error}"))?;
    output.push(b'\n');
    let generated_fixture = parse_fixture(&output)?;
    if generated_fixture.cases.len() != expected_cases
        || generated_fixture.expected_negative_control_count != control_count
    {
        return Err("generated fixture failed its post-serialization count checks".to_owned());
    }
    Ok(output)
}


const COMBINED_CAMPAIGN_SCHEMA: &str = "gcrb-combined-attack-campaign-v0.1";
const COMBINED_CAMPAIGN_PATH: &str =
    "docs/research/fixtures/governance-capture-resistance-v0.1.json";
const CAMPAIGN_HARD_GATE_IDS: &[&str] = &[
    "zero_prohibited_rights_or_authority_transitions",
    "zero_silent_commitment_rewrites",
    "zero_simulation_to_observation_upgrades",
    "zero_evaluator_truth_leakage_in_blind_qualification",
    "appeal_route_must_not_be_controlled_only_by_challenged_authority",
    "unbounded_emergency_bypass",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CombinedCampaign {
    schema_version: String,
    benchmark_id: String,
    version: String,
    status: String,
    visibility: String,
    source_fixture_path: String,
    source_fixture_sha256: String,
    purpose: String,
    expected_case_count: usize,
    expected_composite_attack_count: usize,
    expected_composite_negative_control_count: usize,
    limits: Vec<String>,
    cases: Vec<CompositeCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompositeCase {
    case_id: String,
    class: String,
    family: String,
    severity: String,
    component_case_ids: Vec<String>,
    steps: Vec<CompositeStep>,
    expected_disposition: String,
    required_behaviors: Vec<String>,
    forbidden_behaviors: Vec<String>,
    hard_gates_triggered: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompositeStep {
    sequence: u64,
    attack_family: String,
    operation: String,
    actor_role: String,
    expected_observable: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CombinedCampaignSummary {
    pub benchmark_id: String,
    pub campaign_schema_version: String,
    pub source_fixture_sha256: String,
    pub campaign_sha256: String,
    pub case_count: usize,
    pub composite_attack_count: usize,
    pub composite_negative_control_count: usize,
    pub unique_case_ids: usize,
    pub component_reference_count: usize,
    pub resolved_component_reference_count: usize,
    pub valid_public_development_campaign: bool,
    pub qualification_eligible: bool,
    pub warning: String,
}

fn valid_case_id(value: &str, prefix: &str, digits: usize) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| suffix.len() == digits && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn validate_combined_campaign(
    source_fixture_bytes: &[u8],
    campaign: &CombinedCampaign,
) -> Result<(usize, usize, usize), String> {
    if campaign.schema_version != COMBINED_CAMPAIGN_SCHEMA {
        return Err(format!(
            "unsupported combined campaign schema: {}",
            campaign.schema_version
        ));
    }
    if campaign.benchmark_id != BENCHMARK_ID || campaign.version != "0.1" {
        return Err("unexpected combined campaign benchmark_id or version".to_owned());
    }
    if campaign.status != "public_development_not_qualified"
        || campaign.visibility != "public_known_answers"
    {
        return Err("combined campaign must remain public development with public known answers".to_owned());
    }
    if campaign.source_fixture_path != COMBINED_CAMPAIGN_PATH {
        return Err("combined campaign source fixture path is not the declared public fixture".to_owned());
    }
    let source_sha = sha256_hex(source_fixture_bytes);
    if campaign.source_fixture_sha256 != source_sha {
        return Err(format!(
            "source fixture digest mismatch: campaign says {}, exact source bytes hash to {}",
            campaign.source_fixture_sha256, source_sha
        ));
    }
    if campaign.purpose.trim().is_empty() || campaign.limits.len() < 4 {
        return Err("combined campaign must declare purpose and at least four limits".to_owned());
    }
    if campaign.cases.is_empty() {
        return Err("combined campaign must contain cases".to_owned());
    }

    let source_fixture = parse_fixture(source_fixture_bytes)?;
    let source_ids = source_fixture
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut campaign_ids = BTreeSet::new();
    let mut attack_count = 0usize;
    let mut control_count = 0usize;
    let mut component_reference_count = 0usize;
    let mut resolved_reference_count = 0usize;

    for case in &campaign.cases {
        if !valid_case_id(&case.case_id, "GCRB-C", 2) {
            return Err(format!("invalid combined case_id: {}", case.case_id));
        }
        if !campaign_ids.insert(case.case_id.clone()) {
            return Err(format!("duplicate combined case_id: {}", case.case_id));
        }
        if case.family.trim().is_empty() || case.expected_disposition.trim().is_empty() {
            return Err(format!("case {} has an empty family/disposition", case.case_id));
        }
        if !valid_severity(&case.severity) {
            return Err(format!("case {} has unsupported severity {}", case.case_id, case.severity));
        }
        if case.component_case_ids.len() < 2 {
            return Err(format!("case {} must reference at least two component cases", case.case_id));
        }
        let component_ids = case.component_case_ids.iter().collect::<BTreeSet<_>>();
        if component_ids.len() != case.component_case_ids.len() {
            return Err(format!("case {} repeats a component case reference", case.case_id));
        }
        component_reference_count += case.component_case_ids.len();
        for component_id in &case.component_case_ids {
            if !source_ids.contains(component_id.as_str()) {
                return Err(format!(
                    "case {} references unknown component case {}",
                    case.case_id, component_id
                ));
            }
            resolved_reference_count += 1;
        }
        if case.steps.len() < 2 {
            return Err(format!("case {} must contain at least two ordered steps", case.case_id));
        }
        let mut prior_sequence = 0u64;
        for step in &case.steps {
            if step.sequence == 0 || step.sequence <= prior_sequence {
                return Err(format!(
                    "case {} step sequence must be positive and strictly increasing",
                    case.case_id
                ));
            }
            prior_sequence = step.sequence;
            if step.attack_family.trim().is_empty()
                || step.operation.trim().is_empty()
                || step.expected_observable.trim().is_empty()
            {
                return Err(format!("case {} has an incomplete step", case.case_id));
            }
            if !matches!(
                step.actor_role.as_str(),
                "model" | "public_official" | "system" | "citizen" | "independent_reviewer" | "unknown"
            ) {
                return Err(format!(
                    "case {} has an unsupported step actor role {}",
                    case.case_id, step.actor_role
                ));
            }
        }
        if case.required_behaviors.is_empty() || case.forbidden_behaviors.is_empty() {
            return Err(format!("case {} must declare required and forbidden behaviors", case.case_id));
        }
        if case.required_behaviors.iter().collect::<BTreeSet<_>>().len()
            != case.required_behaviors.len()
        {
            return Err(format!("case {} repeats a required behavior", case.case_id));
        }
        if case.forbidden_behaviors.iter().collect::<BTreeSet<_>>().len()
            != case.forbidden_behaviors.len()
        {
            return Err(format!("case {} repeats a forbidden behavior", case.case_id));
        }
        if case.hard_gates_triggered.iter().collect::<BTreeSet<_>>().len()
            != case.hard_gates_triggered.len()
        {
            return Err(format!("case {} repeats a hard-gate label", case.case_id));
        }
        for gate in &case.hard_gates_triggered {
            if !CAMPAIGN_HARD_GATE_IDS.contains(&gate.as_str()) {
                return Err(format!("case {} has unknown hard-gate label {}", case.case_id, gate));
            }
        }
        match case.class.as_str() {
            "composite_attack" => attack_count += 1,
            "composite_negative_control" => {
                if !case.hard_gates_triggered.is_empty() {
                    return Err(format!(
                        "negative-control case {} must not declare expected hard-gate triggers",
                        case.case_id
                    ));
                }
                control_count += 1;
            }
            other => return Err(format!("case {} has unknown class {}", case.case_id, other)),
        }
    }

    if campaign.expected_case_count != campaign.cases.len() {
        return Err(format!(
            "expected_case_count={} but parsed {} cases",
            campaign.expected_case_count,
            campaign.cases.len()
        ));
    }
    if campaign.expected_composite_attack_count != attack_count {
        return Err(format!(
            "expected_composite_attack_count={} but parsed {} attacks",
            campaign.expected_composite_attack_count, attack_count
        ));
    }
    if campaign.expected_composite_negative_control_count != control_count {
        return Err(format!(
            "expected_composite_negative_control_count={} but parsed {} controls",
            campaign.expected_composite_negative_control_count, control_count
        ));
    }
    if attack_count == 0 || control_count == 0 {
        return Err("campaign must contain at least one composite attack and one benign control".to_owned());
    }

    Ok((attack_count, control_count, resolved_reference_count))
}

/// Validate a public composite-attack campaign against the exact source fixture.
/// This validates structure, known references and sequence order; it does not
/// run the attack, authenticate evidence, or qualify a candidate system.
pub fn validate_combined_campaign_bytes(
    source_fixture_bytes: &[u8],
    campaign_bytes: &[u8],
) -> Result<CombinedCampaignSummary, String> {
    let campaign: CombinedCampaign = serde_json::from_slice(campaign_bytes)
        .map_err(|error| format!("combined campaign JSON is invalid: {error}"))?;
    let (attack_count, control_count, resolved_references) =
        validate_combined_campaign(source_fixture_bytes, &campaign)?;
    let unique_case_ids = campaign
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let component_reference_count = campaign
        .cases
        .iter()
        .map(|case| case.component_case_ids.len())
        .sum();

    Ok(CombinedCampaignSummary {
        benchmark_id: campaign.benchmark_id,
        campaign_schema_version: campaign.schema_version,
        source_fixture_sha256: sha256_hex(source_fixture_bytes),
        campaign_sha256: sha256_hex(campaign_bytes),
        case_count: campaign.cases.len(),
        composite_attack_count: attack_count,
        composite_negative_control_count: control_count,
        unique_case_ids,
        component_reference_count,
        resolved_component_reference_count: resolved_references,
        valid_public_development_campaign: true,
        qualification_eligible: false,
        warning: "Public composed scenarios are development known answers. Validation does not execute scenarios, establish real-world incidence, authenticate sources, or qualify a governance system.".to_owned(),
    })
}


#[derive(Debug, Clone, Serialize)]
pub struct CrossSystemEvidenceBundleSummary {
    pub schema_version: String,
    pub benchmark_id: String,
    pub bundle_sha256: String,
    pub record_count: usize,
    pub unique_evidence_ids: usize,
    pub dependency_count: usize,
    pub resolved_dependency_count: usize,
    pub class_counts: BTreeMap<String, usize>,
    pub source_system_counts: BTreeMap<String, usize>,
    pub dependency_graph_acyclic: bool,
    pub valid_synthetic_evidence_bundle: bool,
    pub qualification_eligible: bool,
    pub warning: String,
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    record_id: &str,
) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("record {record_id} requires non-empty {field}"))
}

fn validate_cross_system_source_type(
    source_system: &str,
    source_record_type: &str,
    evidence_id: &str,
) -> Result<(), String> {
    let compatible = match source_system {
        "mycelix_justice" => matches!(
            source_record_type,
            "justice_case" | "justice_evidence" | "justice_decision" | "justice_appeal"
        ),
        "mycelix_desci" => matches!(
            source_record_type,
            "mycelix_desci_signed_event"
                | "mycelix_desci_authority_receipt"
                | "mycelix_desci_transparency_checkpoint"
        ),
        "symtropy_world" => {
            matches!(source_record_type, "symtropy_sim_snapshot" | "symtropy_sim_event")
        }
        "independent_measurement" => source_record_type == "independent_measurement",
        "gcrb_evaluator" => {
            matches!(source_record_type, "public_report" | "gcrb_derived_claim")
        }
        _ => false,
    };
    if compatible {
        Ok(())
    } else {
        Err(format!(
            "record {evidence_id} has incompatible source_system={source_system} and source_record_type={source_record_type}"
        ))
    }
}

fn validate_cross_system_record(
    record: &serde_json::Value,
    known_ids: &BTreeSet<String>,
) -> Result<(String, String, Vec<String>), String> {
    let evidence_id = required_string(record, "evidence_id", "<unknown>")?.to_owned();
    let id_bytes = evidence_id.as_bytes();
    if !(3..=96).contains(&id_bytes.len())
        || !id_bytes[0].is_ascii_uppercase()
        || !id_bytes.iter().all(|byte| {
            byte.is_ascii_uppercase() || byte.is_ascii_digit() || *byte == b'_' || *byte == b'-'
        })
    {
        return Err(format!("record has invalid evidence_id {evidence_id}"));
    }
    let source_system = required_string(record, "source_system", &evidence_id)?;
    let source_type = required_string(record, "source_record_type", &evidence_id)?;
    let assertion_class = required_string(record, "assertion_class", &evidence_id)?;
    let _claim = required_string(record, "claim", &evidence_id)?;
    let source_record_ref = required_string(record, "source_record_ref", &evidence_id)?;
    let limitations = record["limitations"]
        .as_array()
        .ok_or_else(|| format!("record {evidence_id} requires a limitations array"))?;
    if limitations.is_empty()
        || limitations.iter().any(|item| {
            item.as_str()
                .is_none_or(|text| text.trim().is_empty())
        })
    {
        return Err(format!(
            "record {evidence_id} must preserve explicit non-empty limitations"
        ));
    }
    if !source_record_ref.starts_with("synthetic:") {
        return Err(format!(
            "record {evidence_id} must use synthetic: source references because the bundle is synthetic_only"
        ));
    }
    validate_cross_system_source_type(source_system, source_type, &evidence_id)?;
    if !matches!(
        assertion_class,
        "observed" | "reported" | "inferred" | "simulated" | "unknown"
    ) {
        return Err(format!(
            "record {evidence_id} has unsupported assertion_class {assertion_class}"
        ));
    }

    let type_definition = record["type_definition_ref"]
        .as_object()
        .ok_or_else(|| format!("record {evidence_id} requires type_definition_ref"))?;
    for key in ["owner", "path", "revision"] {
        if type_definition
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_none_or(|text| text.trim().is_empty())
        {
            return Err(format!(
                "record {evidence_id} requires non-empty type_definition_ref.{key}"
            ));
        }
    }
    if let Some(blob_sha) = type_definition.get("git_blob_sha") {
        let value = blob_sha
            .as_str()
            .ok_or_else(|| format!("record {evidence_id} git_blob_sha must be a string"))?;
        if !is_lower_hex(value, 40) {
            return Err(format!(
                "record {evidence_id} type_definition_ref.git_blob_sha must be 40 lowercase hex chars"
            ));
        }
    }
    if let Some(digest) = record.get("source_record_digest_sha256") {
        let value = digest
            .as_str()
            .ok_or_else(|| format!("record {evidence_id} source digest must be a string"))?;
        if !is_lower_hex(value, 64) {
            return Err(format!(
                "record {evidence_id} source_record_digest_sha256 must be 64 lowercase hex chars"
            ));
        }
    }

    let depends = record["depends_on"]
        .as_array()
        .ok_or_else(|| format!("record {evidence_id} requires depends_on array"))?;
    let mut dependencies = Vec::with_capacity(depends.len());
    let mut seen = BTreeSet::new();
    for dependency in depends {
        let dependency_id = dependency.as_str().ok_or_else(|| {
            format!("record {evidence_id} has non-string dependency ID")
        })?;
        if dependency_id == evidence_id {
            return Err(format!("record {evidence_id} cannot depend on itself"));
        }
        if !seen.insert(dependency_id.to_owned()) {
            return Err(format!(
                "record {evidence_id} repeats dependency {dependency_id}"
            ));
        }
        if !known_ids.contains(dependency_id) {
            return Err(format!(
                "record {evidence_id} has unresolved dependency {dependency_id}"
            ));
        }
        dependencies.push(dependency_id.to_owned());
    }

    let has_simulation_context = record.get("simulation_context").is_some();
    let has_observation_basis = record.get("observation_basis").is_some();
    let has_derivation = record.get("derivation").is_some();

    match assertion_class {
        "simulated" => {
            if source_system != "symtropy_world" || !has_simulation_context {
                return Err(format!(
                    "simulated record {evidence_id} must originate from symtropy_world and include simulation_context"
                ));
            }
            if has_observation_basis {
                return Err(format!(
                    "simulated record {evidence_id} cannot carry observation_basis as if it were independently observed"
                ));
            }
            let sim = &record["simulation_context"];
            let simulator_id = required_string(sim, "simulator_id", &evidence_id)?;
            let source_revision = required_string(sim, "source_revision", &evidence_id)?;
            if !(7..=64).contains(&source_revision.len())
                || !source_revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(format!(
                    "simulated record {evidence_id} has invalid simulator source_revision"
                ));
            }
            for field in [
                "build_digest_sha256",
                "scenario_digest_sha256",
                "config_digest_sha256",
                "output_digest_sha256",
            ] {
                let digest = required_string(sim, field, &evidence_id)?;
                if !is_lower_hex(digest, 64) {
                    return Err(format!(
                        "simulated record {evidence_id} has invalid simulation_context.{field}"
                    ));
                }
            }
            let seed = sim["seed"]
                .as_u64()
                .ok_or_else(|| format!("simulated record {evidence_id} requires u64 seed"))?;
            let tick_start = sim["tick_start"]
                .as_u64()
                .ok_or_else(|| format!("simulated record {evidence_id} requires tick_start"))?;
            let tick_end = sim["tick_end"]
                .as_u64()
                .ok_or_else(|| format!("simulated record {evidence_id} requires tick_end"))?;
            let _ = (simulator_id, seed);
            if tick_end < tick_start {
                return Err(format!(
                    "simulated record {evidence_id} has tick_end before tick_start"
                ));
            }
            for field in ["horizon", "validity_scope"] {
                let _ = required_string(sim, field, &evidence_id)?;
            }
            if sim["limitations"]
                .as_array()
                .is_none_or(|items| items.is_empty())
            {
                return Err(format!(
                    "simulated record {evidence_id} must preserve simulation limitations"
                ));
            }
        }
        "observed" => {
            if source_system != "independent_measurement" || !has_observation_basis {
                return Err(format!(
                    "observed record {evidence_id} must originate from independent_measurement and include observation_basis"
                ));
            }
            if has_simulation_context {
                return Err(format!(
                    "observed record {evidence_id} cannot use simulation_context as its observation basis"
                ));
            }
            let basis = &record["observation_basis"];
            for field in [
                "measurement_ref",
                "measurement_method",
                "measured_at",
                "location_scope",
                "independent_corroboration_ref",
            ] {
                let _ = required_string(basis, field, &evidence_id)?;
            }
            for field in ["measurement_ref", "independent_corroboration_ref"] {
                let value = required_string(basis, field, &evidence_id)?;
                if !value.starts_with("synthetic:") {
                    return Err(format!(
                        "observed record {evidence_id} has non-synthetic {field} in a synthetic bundle"
                    ));
                }
            }
            if basis["quality_limits"]
                .as_array()
                .is_none_or(|items| items.is_empty())
            {
                return Err(format!(
                    "observed record {evidence_id} must declare quality_limits"
                ));
            }
        }
        "inferred" => {
            if !has_derivation {
                return Err(format!(
                    "inferred record {evidence_id} requires a derivation object"
                ));
            }
            if has_observation_basis || has_simulation_context {
                return Err(format!(
                    "inferred record {evidence_id} cannot be relabeled directly as observed or simulated; record inputs and derivation separately"
                ));
            }
            let derivation = &record["derivation"];
            for field in ["method_id", "method_version"] {
                let _ = required_string(derivation, field, &evidence_id)?;
            }
            let inputs = derivation["input_evidence_ids"]
                .as_array()
                .ok_or_else(|| {
                    format!("inferred record {evidence_id} requires input_evidence_ids")
                })?;
            let mut input_ids = BTreeSet::new();
            for input in inputs {
                let input_id = input
                    .as_str()
                    .ok_or_else(|| format!("inferred record {evidence_id} has non-string input ID"))?;
                if !known_ids.contains(input_id) {
                    return Err(format!(
                        "inferred record {evidence_id} has unresolved derivation input {input_id}"
                    ));
                }
                input_ids.insert(input_id.to_owned());
            }
            let dependency_ids = dependencies.iter().cloned().collect::<BTreeSet<_>>();
            if input_ids.is_empty() || input_ids != dependency_ids {
                return Err(format!(
                    "inferred record {evidence_id} derivation inputs must exactly match depends_on"
                ));
            }
            for field in ["assumptions", "limitations"] {
                if derivation[field]
                    .as_array()
                    .is_none_or(|items| items.is_empty())
                {
                    return Err(format!(
                        "inferred record {evidence_id} requires derivation.{field}"
                    ));
                }
            }
        }
        "reported" | "unknown" => {
            if has_simulation_context && assertion_class != "simulated" {
                return Err(format!(
                    "record {evidence_id} cannot carry simulation_context unless assertion_class is simulated"
                ));
            }
            if has_observation_basis && assertion_class != "observed" {
                return Err(format!(
                    "record {evidence_id} cannot carry observation_basis unless assertion_class is observed"
                ));
            }
            if has_derivation && assertion_class != "inferred" {
                return Err(format!(
                    "record {evidence_id} cannot carry derivation unless assertion_class is inferred"
                ));
            }
        }
        _ => unreachable!("assertion classes were checked"),
    }

    Ok((source_system.to_owned(), assertion_class.to_owned(), dependencies))
}

fn validate_dependency_acyclic(
    evidence_id: &str,
    dependency_map: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), String> {
    if visited.contains(evidence_id) {
        return Ok(());
    }
    if !visiting.insert(evidence_id.to_owned()) {
        return Err(format!("cross-system evidence dependency cycle contains {evidence_id}"));
    }
    if let Some(dependencies) = dependency_map.get(evidence_id) {
        for dependency in dependencies {
            validate_dependency_acyclic(dependency, dependency_map, visiting, visited)?;
        }
    }
    visiting.remove(evidence_id);
    visited.insert(evidence_id.to_owned());
    Ok(())
}

/// Validate a synthetic cross-system evidence bundle without granting its claims
/// real-world truth. This validator checks source/class compatibility, required
/// simulation/observation/derivation fields, closed dependencies and acyclicity.
pub fn validate_cross_system_evidence_bundle_bytes(
    bytes: &[u8],
) -> Result<CrossSystemEvidenceBundleSummary, String> {
    let document: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("cross-system evidence JSON is invalid: {error}"))?;
    if document["schema_version"].as_str()
        != Some("gcrb-cross-system-evidence-envelope-v0.1")
        || document["benchmark_id"].as_str() != Some(BENCHMARK_ID)
        || document["dataset_status"].as_str() != Some("synthetic_only")
    {
        return Err("cross-system bundle must use the v0.1 schema, GOV-CAP-001, and synthetic_only status".to_owned());
    }
    if document["purpose"].as_str().is_none_or(|value| value.trim().is_empty()) {
        return Err("cross-system bundle requires a non-empty purpose".to_owned());
    }
    if document["limits"]
        .as_array()
        .is_none_or(|items| items.len() < 4)
    {
        return Err("cross-system bundle requires at least four explicit limits".to_owned());
    }
    let records = document["records"]
        .as_array()
        .ok_or_else(|| "cross-system bundle requires records array".to_owned())?;
    if records.is_empty() {
        return Err("cross-system bundle must contain at least one record".to_owned());
    }
    let mut known_ids = BTreeSet::new();
    for record in records {
        let id = record["evidence_id"]
            .as_str()
            .ok_or_else(|| "record has missing evidence_id".to_owned())?;
        if !known_ids.insert(id.to_owned()) {
            return Err(format!("duplicate cross-system evidence_id: {id}"));
        }
    }

    let mut dependency_map = BTreeMap::<String, Vec<String>>::new();
    let mut class_counts = BTreeMap::<String, usize>::new();
    let mut source_system_counts = BTreeMap::<String, usize>::new();
    let mut dependency_count = 0usize;
    for record in records {
        let id = record["evidence_id"].as_str().expect("IDs validated above");
        let (source_system, assertion_class, dependencies) =
            validate_cross_system_record(record, &known_ids)?;
        dependency_count += dependencies.len();
        *class_counts.entry(assertion_class).or_insert(0) += 1;
        *source_system_counts.entry(source_system).or_insert(0) += 1;
        dependency_map.insert(id.to_owned(), dependencies);
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for evidence_id in &known_ids {
        validate_dependency_acyclic(evidence_id, &dependency_map, &mut visiting, &mut visited)?;
    }
    Ok(CrossSystemEvidenceBundleSummary {
        schema_version: "gcrb-cross-system-evidence-envelope-v0.1".to_owned(),
        benchmark_id: BENCHMARK_ID.to_owned(),
        bundle_sha256: sha256_hex(bytes),
        record_count: records.len(),
        unique_evidence_ids: known_ids.len(),
        dependency_count,
        resolved_dependency_count: dependency_count,
        class_counts,
        source_system_counts,
        dependency_graph_acyclic: true,
        valid_synthetic_evidence_bundle: true,
        qualification_eligible: false,
        warning: "This validator checks type boundaries and references in synthetic fixtures only. It does not authenticate source records, verify source truth, validate real measurement quality, prove capture completeness, or qualify cross-system runtime integration.".to_owned(),
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
            Some(&1usize)
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


    #[test]
    fn generated_parameterized_controls_are_reproducible() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let first = generate_parameterized_controls(fixture, 32, 20_261_029).unwrap();
        let second = generate_parameterized_controls(fixture, 32, 20_261_029).unwrap();
        let other_seed = generate_parameterized_controls(fixture, 32, 20_261_030).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, other_seed);
        let summary = validate_fixture_bytes(&first).unwrap();
        assert_eq!(summary.attack_count, 18);
        assert_eq!(summary.negative_control_count, 32);
        assert!(!summary.qualification_eligible);
    }

    #[test]
    fn generated_stress_fixture_contains_one_thousand_controls() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let bytes = generate_parameterized_controls(fixture, 1_000, 20_261_029).unwrap();
        let summary = validate_fixture_bytes(&bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(summary.case_count, 1_018);
        assert_eq!(summary.attack_count, 18);
        assert_eq!(summary.negative_control_count, 1_000);
        assert_eq!(value["generator"]["algorithm"], "splitmix64-v1");
        assert_eq!(value["generator"]["known_answers_public"], true);
        let source_sha = sha256_hex(fixture);
        assert_eq!(
            value["generator"]["source_fixture_sha256"].as_str(),
            Some(source_sha.as_str())
        );
        let ids = value["cases"].as_array().unwrap().iter()
            .map(|case| case["case_id"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 1_018);
        let parameterized = value["cases"].as_array().unwrap().iter()
            .filter(|case| case["case_id"].as_str().unwrap().starts_with("GCRB-NP"))
            .collect::<Vec<_>>();
        assert_eq!(parameterized.len(), 1_000);
        assert!(parameterized.iter().all(|case| case["parameters"].is_object()));
    }

    #[test]
    fn generator_rejects_zero_excessive_and_overflowing_inputs() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        assert!(generate_parameterized_controls(fixture, 0, 1).unwrap_err().contains("between 1"));
        assert!(generate_parameterized_controls(fixture, 100_001, 1).unwrap_err().contains("between 1"));
        assert!(generate_parameterized_controls(fixture, 2, u64::MAX).unwrap_err().contains("overflows"));
    }

    #[test]
    fn generator_rejects_a_generated_fixture_as_its_own_source() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let generated = generate_parameterized_controls(fixture, 8, 41).unwrap();
        let result = generate_parameterized_controls(&generated, 8, 42);
        assert!(result.unwrap_err().contains("not a previous generated stress fixture"));
    }


    #[test]
    fn generator_matches_committed_stress_fixture_semantically() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let golden = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-stress-1000-v0.1.json");
        let generated = generate_parameterized_controls(fixture, 1_000, 20_261_029).unwrap();
        let golden_value: serde_json::Value = serde_json::from_slice(golden).unwrap();
        let generated_value: serde_json::Value = serde_json::from_slice(&generated).unwrap();
        assert_eq!(generated_value, golden_value);
    }



    #[test]
    fn validates_combined_attack_campaign_and_all_component_references() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let campaign = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let summary = validate_combined_campaign_bytes(fixture, campaign).unwrap();
        assert_eq!(summary.case_count, 13);
        assert_eq!(summary.composite_attack_count, 11);
        assert_eq!(summary.composite_negative_control_count, 2);
        assert_eq!(summary.component_reference_count, summary.resolved_component_reference_count);
        assert_eq!(summary.unique_case_ids, 13);
        assert!(!summary.qualification_eligible);
    }

    #[test]
    fn rejects_combined_campaign_source_fixture_digest_mismatch() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut campaign: serde_json::Value = serde_json::from_slice(source).unwrap();
        campaign["source_fixture_sha256"] = serde_json::Value::String("00".repeat(32));
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&campaign).unwrap());
        assert!(result.unwrap_err().contains("source fixture digest mismatch"));
    }

    #[test]
    fn rejects_unresolved_combined_component_reference() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut campaign: serde_json::Value = serde_json::from_slice(source).unwrap();
        campaign["cases"][0]["component_case_ids"][0] = serde_json::Value::String("GCRB-A99".to_owned());
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&campaign).unwrap());
        assert!(result.unwrap_err().contains("unknown component case"));
    }

    #[test]
    fn rejects_non_monotonic_composite_step_sequence() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut campaign: serde_json::Value = serde_json::from_slice(source).unwrap();
        campaign["cases"][0]["steps"][1]["sequence"] = serde_json::json!(1);
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&campaign).unwrap());
        assert!(result.unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn rejects_duplicate_combined_case_ids() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut campaign: serde_json::Value = serde_json::from_slice(source).unwrap();
        campaign["cases"][1]["case_id"] = campaign["cases"][0]["case_id"].clone();
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&campaign).unwrap());
        assert!(result.unwrap_err().contains("duplicate combined case_id"));
    }

    #[test]
    fn rejects_control_cases_that_claim_hard_gate_triggers() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut campaign: serde_json::Value = serde_json::from_slice(source).unwrap();
        campaign["cases"][11]["hard_gates_triggered"] = serde_json::json!(["zero_silent_commitment_rewrites"]);
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&campaign).unwrap());
        assert!(result.unwrap_err().contains("must not declare expected hard-gate triggers"));
    }

    #[test]
    fn rejects_duplicate_and_unknown_hard_gate_labels() {
        let fixture = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-v0.1.json");
        let source = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-combined-attacks-v0.1.json");
        let mut duplicate: serde_json::Value = serde_json::from_slice(source).unwrap();
        duplicate["cases"][0]["hard_gates_triggered"] =
            serde_json::json!(["unbounded_emergency_bypass", "unbounded_emergency_bypass"]);
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&duplicate).unwrap());
        assert!(result.unwrap_err().contains("repeats a hard-gate label"));

        let mut unknown: serde_json::Value = serde_json::from_slice(source).unwrap();
        unknown["cases"][0]["hard_gates_triggered"] = serde_json::json!(["fake_gate"]);
        let result = validate_combined_campaign_bytes(fixture, &serde_json::to_vec(&unknown).unwrap());
        assert!(result.unwrap_err().contains("unknown hard-gate label"));
    }


    #[test]
    fn validates_cross_system_evidence_demo_and_preserves_class_separation() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let summary = validate_cross_system_evidence_bundle_bytes(bytes).unwrap();
        assert_eq!(summary.record_count, 5);
        assert_eq!(summary.unique_evidence_ids, 5);
        assert_eq!(summary.dependency_count, 2);
        assert_eq!(summary.resolved_dependency_count, 2);
        assert!(summary.dependency_graph_acyclic);
        assert_eq!(summary.class_counts.get("reported"), Some(&2usize));
        assert_eq!(summary.class_counts.get("simulated"), Some(&1usize));
        assert_eq!(summary.class_counts.get("observed"), Some(&1usize));
        assert_eq!(summary.class_counts.get("inferred"), Some(&1usize));
        assert!(!summary.qualification_eligible);
    }

    #[test]
    fn rejects_a_symtropy_snapshot_labeled_observed() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][2]["assertion_class"] = serde_json::Value::String("observed".to_owned());
        value["records"][2]["observation_basis"] = serde_json::json!({
            "measurement_ref":"synthetic:measurement-bogus",
            "measurement_method":"made up",
            "measured_at":"synthetic-time",
            "location_scope":"synthetic",
            "independent_corroboration_ref":"synthetic:corr-bogus",
            "quality_limits":["not real"]
        });
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("must originate from independent_measurement"));
    }

    #[test]
    fn rejects_observed_record_without_independent_corroboration() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][3]["observation_basis"]["independent_corroboration_ref"] = serde_json::Value::String(String::new());
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("requires non-empty independent_corroboration_ref"));
    }

    #[test]
    fn rejects_inferred_claim_without_exact_dependency_inputs() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][4]["derivation"]["input_evidence_ids"] = serde_json::json!(["MYCELIX-CASE-001"]);
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("must exactly match depends_on"));
    }

    #[test]
    fn rejects_unresolved_cross_system_dependency() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][4]["depends_on"] = serde_json::json!(["NO-SUCH-RECORD"]);
        value["records"][4]["derivation"]["input_evidence_ids"] = serde_json::json!(["NO-SUCH-RECORD"]);
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("unresolved dependency"));
    }

    #[test]
    fn rejects_cross_system_dependency_cycles() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][0]["depends_on"] = serde_json::json!(["GCRB-FORECAST-ERROR-001"]);
        value["records"][4]["depends_on"] = serde_json::json!(["MYCELIX-CASE-001"]);
        value["records"][4]["derivation"]["input_evidence_ids"] = serde_json::json!(["MYCELIX-CASE-001"]);
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("dependency cycle"));
    }

    #[test]
    fn rejects_non_synthetic_source_reference_in_synthetic_bundle() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][0]["source_record_ref"] = serde_json::Value::String("https://example.invalid/real-record".to_owned());
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("must use synthetic: source references"));
    }

    #[test]
    fn rejects_bad_source_definition_blob_sha() {
        let bytes = include_bytes!("../../../docs/research/fixtures/governance-capture-resistance-cross-system-evidence-demo-v0.1.json");
        let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        value["records"][0]["type_definition_ref"]["git_blob_sha"] = serde_json::Value::String("bad".to_owned());
        let result = validate_cross_system_evidence_bundle_bytes(&serde_json::to_vec(&value).unwrap());
        assert!(result.unwrap_err().contains("must be 40 lowercase hex chars"));
    }

}
