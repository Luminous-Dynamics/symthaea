//! Evaluator-only D6.12 scorer for benchmark-oracle integrity.
//!
//! The evaluator checks two-sided known-answer outcomes and derives oracle
//! fault classes from frozen state digests, replay records, cache identities,
//! repeat outcomes and declared observation coverage. It does not authenticate
//! the observer, prove OS-level isolation, or certify arbitrary benchmark
//! implementations.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

const FAULT_CLASSES: &[&str] = &[
    "candidate_state_drift",
    "state_mutation",
    "replay_disagreement",
    "cache_mismatch",
    "nondeterminism",
    "observer_coverage_gap",
    "false_positive",
    "false_negative",
];

const OUTCOMES: &[&str] = &["pass", "fail"];
const CLAIMED_OUTCOMES: &[&str] = &["pass", "fail", "unknown"];
const INTEGRITY_STATES: &[&str] = &["valid", "invalid", "unknown"];

const CLAIM_CEILING: &str = "Synthetic known-answer evaluation of declared oracle observations, state digests, replay records, cache identities, repeat outcomes and observer coverage. Does not authenticate telemetry, prove OS-level isolation, independently implement arbitrary benchmark semantics, or establish universal benchmark integrity.";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OracleCaseOutcome {
    pub case_id: String,
    pub control_role: String,
    pub expected_semantic_outcome: String,
    pub claimed_semantic_outcome: String,
    pub outcome_correct: bool,
    pub false_accept: bool,
    pub false_reject: bool,
    pub unnecessary_abstention: bool,
    pub native_verifier_outcome: String,
    pub actual_fault_classes: Vec<String>,
    pub detected_fault_classes: Vec<String>,
    pub missed_fault_classes: Vec<String>,
    pub false_alarm_fault_classes: Vec<String>,
    pub claimed_integrity_state: String,
    pub expected_integrity_state: String,
    pub integrity_claim_correct: bool,
    pub false_integrity_valid_assertion: bool,
    pub false_integrity_invalid_assertion: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OracleTraceReport {
    pub trace_id: String,
    pub cases: Vec<OracleCaseOutcome>,
    pub case_count: usize,
    pub positive_control_count: usize,
    pub negative_control_count: usize,
    pub sham_control_count: usize,
    pub fault_injection_case_count: usize,
    pub false_accept_count: usize,
    pub false_reject_count: usize,
    pub unnecessary_abstention_count: usize,
    pub total_fault_instances: usize,
    pub detected_fault_instances: usize,
    pub missed_fault_instances: usize,
    pub false_alarm_fault_instances: usize,
    pub false_integrity_valid_assertions: usize,
    pub false_integrity_invalid_assertions: usize,
    pub state_mutation_case_count: usize,
    pub replay_disagreement_case_count: usize,
    pub cache_mismatch_case_count: usize,
    pub nondeterminism_case_count: usize,
    pub observer_coverage_gap_case_count: usize,
    pub verifier_false_positive_case_count: usize,
    pub verifier_false_negative_case_count: usize,
    pub candidate_state_drift_case_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OracleIntegrityReport {
    pub scenario_id: String,
    pub adjudication_status: String,
    pub oracle_case_count: usize,
    pub traces: Vec<OracleTraceReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct OracleRun {
    control_role: String,
    candidate_input_digest: String,
    post_candidate_state_digest: String,
    pre_grade_state_digest: String,
    post_grade_state_digest: String,
    replay_state_digest: String,
    verifier_outcome: String,
    repeat_verifier_outcome: String,
    cache_hit: bool,
    cache_input_digest: Option<String>,
    observed_surfaces: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct CaseTruth {
    expected_outcome: String,
    expected_post_candidate_state_digest: String,
    required_surfaces: BTreeSet<String>,
    expected_faults: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct CaseClaim {
    claimed_outcome: String,
    detected_faults: BTreeSet<String>,
    claimed_integrity_state: String,
}

fn allowed_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field {key:?}"));
        }
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_bool(value: &Value, key: &str, label: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{key} must be boolean"))
}

fn string_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a non-empty string"))?;
        if !output.insert(id.to_owned()) {
            return Err(format!("{label}.{key} contains duplicate value {id:?}"));
        }
    }
    Ok(output)
}

fn valid_digest(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest(value: &Value, key: &str, label: &str) -> Result<String, String> {
    let raw = required_str(value, key, label)?;
    if !valid_digest(raw) {
        return Err(format!(
            "{label}.{key} must be a lowercase 64-character hexadecimal digest"
        ));
    }
    Ok(raw.to_owned())
}

fn parse_runs(candidate: &Value) -> Result<BTreeMap<String, OracleRun>, String> {
    let values = candidate
        .get("oracle_runs")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.oracle_runs must be an array".to_owned())?;
    if values.is_empty() {
        return Err("candidate_view.oracle_runs must not be empty".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.oracle_runs[{index}]");
        allowed_keys(
            value,
            &[
                "case_id",
                "control_role",
                "candidate_input_digest",
                "post_candidate_state_digest",
                "pre_grade_state_digest",
                "post_grade_state_digest",
                "replay_state_digest",
                "verifier_outcome",
                "repeat_verifier_outcome",
                "cache_hit",
                "cache_input_digest",
                "observed_surfaces",
            ],
            &label,
        )?;
        let case_id = required_str(value, "case_id", &label)?.to_owned();
        let control_role = required_str(value, "control_role", &label)?.to_owned();
        if !["positive", "negative", "sham", "fault_injection"].contains(&control_role.as_str()) {
            return Err(format!("{label}.control_role is unsupported"));
        }
        let candidate_input_digest = digest(value, "candidate_input_digest", &label)?;
        let post_candidate_state_digest = digest(value, "post_candidate_state_digest", &label)?;
        let pre_grade_state_digest = digest(value, "pre_grade_state_digest", &label)?;
        let post_grade_state_digest = digest(value, "post_grade_state_digest", &label)?;
        let replay_state_digest = digest(value, "replay_state_digest", &label)?;
        let verifier_outcome = required_str(value, "verifier_outcome", &label)?.to_owned();
        let repeat_verifier_outcome =
            required_str(value, "repeat_verifier_outcome", &label)?.to_owned();
        if !OUTCOMES.contains(&verifier_outcome.as_str())
            || !OUTCOMES.contains(&repeat_verifier_outcome.as_str())
        {
            return Err(format!("{label} verifier outcomes must be pass or fail"));
        }
        let cache_hit = required_bool(value, "cache_hit", &label)?;
        let cache_input_digest = match value.get("cache_input_digest") {
            Some(Value::String(raw)) if valid_digest(raw) => Some(raw.clone()),
            Some(Value::Null) | None => None,
            Some(Value::String(_)) => {
                return Err(format!("{label}.cache_input_digest must be a lowercase 64-character digest"));
            }
            _ => return Err(format!("{label}.cache_input_digest must be a string or null")),
        };
        if cache_hit && cache_input_digest.is_none() {
            return Err(format!("{label}.cache_input_digest is required when cache_hit is true"));
        }
        let observed_surfaces = string_set(value, "observed_surfaces", &label)?;
        if output
            .insert(
                case_id.clone(),
                OracleRun {
                    control_role,
                    candidate_input_digest,
                    post_candidate_state_digest,
                    pre_grade_state_digest,
                    post_grade_state_digest,
                    replay_state_digest,
                    verifier_outcome,
                    repeat_verifier_outcome,
                    cache_hit,
                    cache_input_digest,
                    observed_surfaces,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate oracle run for case_id {case_id:?}"));
        }
    }
    Ok(output)
}

fn parse_truths(
    gold: &Value,
    runs: &BTreeMap<String, OracleRun>,
) -> Result<BTreeMap<String, CaseTruth>, String> {
    let values = gold
        .get("case_truths")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.case_truths must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.case_truths[{index}]");
        allowed_keys(
            value,
            &[
                "case_id",
                "expected_semantic_outcome",
                "expected_post_candidate_state_digest",
                "required_observer_surfaces",
                "expected_fault_classes",
            ],
            &label,
        )?;
        let case_id = required_str(value, "case_id", &label)?.to_owned();
        if !runs.contains_key(&case_id) {
            return Err(format!("{label} references unknown oracle run {case_id:?}"));
        }
        let expected_semantic_outcome =
            required_str(value, "expected_semantic_outcome", &label)?.to_owned();
        if !OUTCOMES.contains(&expected_semantic_outcome.as_str()) {
            return Err(format!("{label}.expected_semantic_outcome must be pass or fail"));
        }
        let expected_post_candidate_state_digest =
            digest(value, "expected_post_candidate_state_digest", &label)?;
        let required_surfaces = string_set(value, "required_observer_surfaces", &label)?;
        if required_surfaces.is_empty() {
            return Err(format!("{label}.required_observer_surfaces must not be empty"));
        }
        let expected_faults = string_set(value, "expected_fault_classes", &label)?;
        if expected_faults
            .iter()
            .any(|fault| !FAULT_CLASSES.contains(&fault.as_str()))
        {
            return Err(format!("{label}.expected_fault_classes contains an unsupported fault"));
        }
        if output
            .insert(
                case_id.clone(),
                CaseTruth {
                    expected_outcome: expected_semantic_outcome,
                    expected_post_candidate_state_digest,
                    required_surfaces,
                    expected_faults,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate case truth {case_id:?}"));
        }
    }
    if output.len() != runs.len() || runs.keys().any(|case_id| !output.contains_key(case_id)) {
        return Err("gold_overlay.case_truths must cover every oracle run exactly once".to_owned());
    }
    Ok(output)
}

fn derive_faults(run: &OracleRun, truth: &CaseTruth) -> BTreeSet<String> {
    let mut faults = BTreeSet::new();
    if run.post_candidate_state_digest != truth.expected_post_candidate_state_digest {
        faults.insert("candidate_state_drift".to_owned());
    }
    if run.pre_grade_state_digest != run.post_grade_state_digest {
        faults.insert("state_mutation".to_owned());
    }
    if run.replay_state_digest != run.post_candidate_state_digest {
        faults.insert("replay_disagreement".to_owned());
    }
    if run.cache_hit
        && run
            .cache_input_digest
            .as_ref()
            .is_some_and(|digest| digest != &run.candidate_input_digest)
    {
        faults.insert("cache_mismatch".to_owned());
    }
    if run.verifier_outcome != run.repeat_verifier_outcome {
        faults.insert("nondeterminism".to_owned());
    }
    if !truth.required_surfaces.is_subset(&run.observed_surfaces) {
        faults.insert("observer_coverage_gap".to_owned());
    }
    if run.verifier_outcome == "pass" && truth.expected_outcome == "fail" {
        faults.insert("false_positive".to_owned());
    }
    if run.verifier_outcome == "fail" && truth.expected_outcome == "pass" {
        faults.insert("false_negative".to_owned());
    }
    faults
}

fn parse_claims(
    trace: &Value,
    runs: &BTreeMap<String, OracleRun>,
) -> Result<BTreeMap<String, CaseClaim>, String> {
    allowed_keys(trace, &["trace_id", "case_claims"], "candidate_trace")?;
    required_str(trace, "trace_id", "candidate_trace")?;
    let values = trace
        .get("case_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.case_claims must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_trace.case_claims[{index}]");
        allowed_keys(
            value,
            &["case_id", "claimed_semantic_outcome", "detected_fault_classes", "claimed_integrity_state"],
            &label,
        )?;
        let case_id = required_str(value, "case_id", &label)?.to_owned();
        if !runs.contains_key(&case_id) {
            return Err(format!("{label} references unknown case {case_id:?}"));
        }
        let claimed_outcome = required_str(value, "claimed_semantic_outcome", &label)?.to_owned();
        if !CLAIMED_OUTCOMES.contains(&claimed_outcome.as_str()) {
            return Err(format!("{label}.claimed_semantic_outcome is unsupported"));
        }
        let detected_faults = string_set(value, "detected_fault_classes", &label)?;
        if detected_faults
            .iter()
            .any(|fault| !FAULT_CLASSES.contains(&fault.as_str()))
        {
            return Err(format!("{label}.detected_fault_classes contains an unsupported fault"));
        }
        let claimed_integrity_state =
            required_str(value, "claimed_integrity_state", &label)?.to_owned();
        if !INTEGRITY_STATES.contains(&claimed_integrity_state.as_str()) {
            return Err(format!("{label}.claimed_integrity_state is unsupported"));
        }
        if output
            .insert(
                case_id.clone(),
                CaseClaim {
                    claimed_outcome,
                    detected_faults,
                    claimed_integrity_state,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate case claim {case_id:?}"));
        }
    }
    if output.len() != runs.len() || runs.keys().any(|case_id| !output.contains_key(case_id)) {
        return Err("candidate trace must claim every oracle run exactly once".to_owned());
    }
    Ok(output)
}

fn score_trace(
    trace: &Value,
    runs: &BTreeMap<String, OracleRun>,
    truths: &BTreeMap<String, CaseTruth>,
) -> Result<OracleTraceReport, String> {
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
    let claims = parse_claims(trace, runs)?;
    let mut cases = Vec::new();

    for (case_id, run) in runs {
        let truth = truths
            .get(case_id)
            .ok_or_else(|| format!("missing gold truth {case_id:?}"))?;
        let claim = claims
            .get(case_id)
            .ok_or_else(|| format!("missing candidate claim {case_id:?}"))?;
        let actual_faults = derive_faults(run, truth);
        if actual_faults != truth.expected_faults {
            return Err(format!(
                "gold expected_fault_classes mismatch for {case_id:?}: computed {actual_faults:?}, declared {:?}",
                truth.expected_faults
            ));
        }
        let missed_faults: BTreeSet<String> = actual_faults
            .difference(&claim.detected_faults)
            .cloned()
            .collect();
        let false_alarms: BTreeSet<String> = claim
            .detected_faults
            .difference(&actual_faults)
            .cloned()
            .collect();
        let expected_integrity_state = if actual_faults.is_empty() {
            "valid"
        } else {
            "invalid"
        };
        let false_integrity_valid_assertion =
            claim.claimed_integrity_state == "valid" && expected_integrity_state != "valid";
        let false_integrity_invalid_assertion =
            claim.claimed_integrity_state == "invalid" && expected_integrity_state != "invalid";
        let outcome_correct = claim.claimed_outcome == truth.expected_outcome;
        let false_accept =
            truth.expected_outcome == "fail" && claim.claimed_outcome == "pass";
        let false_reject =
            truth.expected_outcome == "pass" && claim.claimed_outcome == "fail";
        let unnecessary_abstention = claim.claimed_outcome == "unknown";
        cases.push(OracleCaseOutcome {
            case_id: case_id.clone(),
            control_role: run.control_role.clone(),
            expected_semantic_outcome: truth.expected_outcome.clone(),
            claimed_semantic_outcome: claim.claimed_outcome.clone(),
            outcome_correct,
            false_accept,
            false_reject,
            unnecessary_abstention,
            native_verifier_outcome: run.verifier_outcome.clone(),
            actual_fault_classes: actual_faults.iter().cloned().collect(),
            detected_fault_classes: claim.detected_faults.iter().cloned().collect(),
            missed_fault_classes: missed_faults.iter().cloned().collect(),
            false_alarm_fault_classes: false_alarms.iter().cloned().collect(),
            claimed_integrity_state: claim.claimed_integrity_state.clone(),
            expected_integrity_state: expected_integrity_state.to_owned(),
            integrity_claim_correct: claim.claimed_integrity_state == expected_integrity_state,
            false_integrity_valid_assertion,
            false_integrity_invalid_assertion,
        });
    }

    let count_role = |role: &str| cases.iter().filter(|case| case.control_role == role).count();
    let count_fault = |fault: &str| {
        cases
            .iter()
            .filter(|case| case.actual_fault_classes.iter().any(|item| item == fault))
            .count()
    };

    Ok(OracleTraceReport {
        trace_id,
        case_count: cases.len(),
        positive_control_count: count_role("positive"),
        negative_control_count: count_role("negative"),
        sham_control_count: count_role("sham"),
        fault_injection_case_count: count_role("fault_injection"),
        false_accept_count: cases.iter().filter(|case| case.false_accept).count(),
        false_reject_count: cases.iter().filter(|case| case.false_reject).count(),
        unnecessary_abstention_count: cases.iter().filter(|case| case.unnecessary_abstention).count(),
        total_fault_instances: cases.iter().map(|case| case.actual_fault_classes.len()).sum(),
        detected_fault_instances: cases.iter().map(|case| {
            case.actual_fault_classes.len() - case.missed_fault_classes.len()
        }).sum(),
        missed_fault_instances: cases.iter().map(|case| case.missed_fault_classes.len()).sum(),
        false_alarm_fault_instances: cases.iter().map(|case| case.false_alarm_fault_classes.len()).sum(),
        false_integrity_valid_assertions: cases.iter().filter(|case| case.false_integrity_valid_assertion).count(),
        false_integrity_invalid_assertions: cases.iter().filter(|case| case.false_integrity_invalid_assertion).count(),
        state_mutation_case_count: count_fault("state_mutation"),
        replay_disagreement_case_count: count_fault("replay_disagreement"),
        cache_mismatch_case_count: count_fault("cache_mismatch"),
        nondeterminism_case_count: count_fault("nondeterminism"),
        observer_coverage_gap_case_count: count_fault("observer_coverage_gap"),
        verifier_false_positive_case_count: count_fault("false_positive"),
        verifier_false_negative_case_count: count_fault("false_negative"),
        candidate_state_drift_case_count: count_fault("candidate_state_drift"),
        cases,
    })
}

/// Evaluate a D6.12 oracle-integrity bundle. Candidate traces must remain
/// separate from evaluator-only case truths and expected fault classes.
pub fn evaluate_oracle_integrity_json(input: &str) -> Result<OracleIntegrityReport, String> {
    let bundle: Value =
        serde_json::from_str(input).map_err(|error| format!("bundle JSON parse failed: {error}"))?;
    allowed_keys(
        &bundle,
        &["bundle_id", "bundle_visibility", "candidate_view", "gold_overlay", "candidate_traces"],
        "bundle",
    )?;
    required_str(&bundle, "bundle_id", "bundle")?;
    if required_str(&bundle, "bundle_visibility", "bundle")?
        != "evaluator_only_never_candidate_visible"
    {
        return Err("D6.12 bundle must be explicitly evaluator-only".to_owned());
    }
    let candidate = bundle
        .get("candidate_view")
        .ok_or_else(|| "candidate_view is required".to_owned())?;
    allowed_keys(candidate, &["scenario_id", "oracle_runs"], "candidate_view")?;
    let scenario_id = required_str(candidate, "scenario_id", "candidate_view")?.to_owned();
    let runs = parse_runs(candidate)?;

    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    allowed_keys(gold, &["adjudication_status", "case_truths"], "gold_overlay")?;
    let adjudication_status = required_str(gold, "adjudication_status", "gold_overlay")?.to_owned();
    if adjudication_status != "synthetic_known_answer_not_independently_adjudicated" {
        return Err(format!("unsupported D6.12 v0.1 adjudication status {adjudication_status:?}"));
    }
    let truths = parse_truths(gold, &runs)?;

    let traces = bundle
        .get("candidate_traces")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_traces must be an array".to_owned())?;
    if traces.is_empty() {
        return Err("candidate_traces must not be empty".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut reports = Vec::new();
    for trace in traces {
        let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
        if !trace_ids.insert(trace_id) {
            return Err("candidate trace identifiers must be unique".to_owned());
        }
        reports.push(score_trace(trace, &runs, &truths)?);
    }
    reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    Ok(OracleIntegrityReport {
        scenario_id,
        adjudication_status,
        oracle_case_count: runs.len(),
        traces: reports,
        claim_ceiling: CLAIM_CEILING.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-oracle-integrity-v0.1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-oracle-integrity-v0.1.expected.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.12 fixture JSON")
    }

    fn report_for(value: &Value) -> OracleIntegrityReport {
        evaluate_oracle_integrity_json(&value.to_string()).expect("D6.12 score")
    }

    fn trace<'a>(report: &'a OracleIntegrityReport, id: &str) -> &'a OracleTraceReport {
        report.traces.iter().find(|item| item.trace_id == id).expect("trace")
    }

    #[test]
    fn matches_complete_machine_readable_known_answer() {
        let actual = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let actual_value = serde_json::to_value(actual).expect("serialize report");
        let expected_value: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn detects_state_mutation_by_the_verifier() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.state_mutation_case_count, 1);
        assert!(auditor.cases.iter().any(|case| {
            case.case_id == "mutating-oracle"
                && case.actual_fault_classes.contains(&"state_mutation".to_owned())
        }));
    }

    #[test]
    fn detects_stale_cache_reuse_and_false_acceptance() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.cache_mismatch_case_count, 1);
        assert_eq!(auditor.false_accept_count, 1);
        assert!(auditor.cases.iter().any(|case| {
            case.case_id == "stale-cache"
                && case.actual_fault_classes.contains(&"cache_mismatch".to_owned())
                && case.false_accept
        }));
    }

    #[test]
    fn preserves_positive_and_negative_controls_against_always_reject() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let always = trace(&report, "always-reject");
        assert_eq!(always.false_reject_count, 7);
        assert_eq!(always.false_accept_count, 0);
    }

    #[test]
    fn detects_false_rejection_of_valid_alternative_implementation() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.verifier_false_negative_case_count, 1);
        assert!(auditor.cases.iter().any(|case| {
            case.case_id == "valid-alternative"
                && case.actual_fault_classes.contains(&"false_negative".to_owned())
        }));
    }

    #[test]
    fn detects_non_deterministic_repeat_outcomes() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.nondeterminism_case_count, 1);
    }

    #[test]
    fn detects_independent_replay_disagreement() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.replay_disagreement_case_count, 1);
    }

    #[test]
    fn detects_missing_observer_surface_coverage() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.observer_coverage_gap_case_count, 1);
    }

    #[test]
    fn auditor_detects_every_registered_fault_instance() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.total_fault_instances, 8);
        assert_eq!(auditor.detected_fault_instances, 8);
        assert_eq!(auditor.missed_fault_instances, 0);
        assert_eq!(auditor.false_alarm_fault_instances, 0);
        assert_eq!(auditor.false_integrity_valid_assertions, 0);
        assert_eq!(auditor.false_integrity_invalid_assertions, 0);
    }

    #[test]
    fn native_verifier_only_baseline_misses_faults_and_both_sides_of_accuracy() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let native = trace(&report, "native-verifier-only");
        assert_eq!(native.false_accept_count, 1);
        assert_eq!(native.false_reject_count, 1);
        assert_eq!(native.missed_fault_instances, 8);
        assert_eq!(native.false_integrity_valid_assertions, 7);
    }

    #[test]
    fn detects_candidate_state_drift_before_grading() {
        let report = evaluate_oracle_integrity_json(FIXTURE).expect("D6.12 fixture");
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.candidate_state_drift_case_count, 1);
        assert!(auditor.cases.iter().any(|case| {
            case.case_id == "candidate-state-drift"
                && case.actual_fault_classes.contains(&"candidate_state_drift".to_owned())
        }));
    }

    #[test]
    fn rejects_gold_fault_summary_that_does_not_match_raw_evidence() {
        let mut value = fixture_value();
        value["gold_overlay"]["case_truths"][2]["expected_fault_classes"] = json!([]);
        let error = evaluate_oracle_integrity_json(&value.to_string()).unwrap_err();
        assert!(error.contains("expected_fault_classes mismatch"));
    }

    #[test]
    fn rejects_claimed_valid_integrity_when_verifier_mutated_state() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["case_claims"][2]["claimed_integrity_state"] =
            json!("valid");
        let report = report_for(&value);
        let auditor = trace(&report, "independent-auditor");
        assert_eq!(auditor.false_integrity_valid_assertions, 1);
    }

    #[test]
    fn rejects_unknown_nested_fields() {
        let mut value = fixture_value();
        value["candidate_view"]["oracle_runs"][0]["grader_hint"] = json!("pass");
        let error = evaluate_oracle_integrity_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn requires_cache_digest_when_cache_hit_is_declared() {
        let mut value = fixture_value();
        value["candidate_view"]["oracle_runs"][3]["cache_input_digest"] = Value::Null;
        let error = evaluate_oracle_integrity_json(&value.to_string()).unwrap_err();
        assert!(error.contains("cache_input_digest is required"));
    }

    #[test]
    fn rejects_missing_case_claim() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["case_claims"].as_array_mut().expect("claims").pop();
        let error = evaluate_oracle_integrity_json(&value.to_string()).unwrap_err();
        assert!(error.contains("must claim every oracle run exactly once"));
    }
}
