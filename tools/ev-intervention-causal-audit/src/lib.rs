use jsonschema::{draft202012, Validator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-intervention-causal-scenario/0.1";
const SCENARIO_VERSION: &str = "ev-intervention-causal-scenario/0.1";
const CORPUS_VERSION: &str = "ev-intervention-causal-corpus/0.1";
const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expected_disposition: String,
    expected_did_effect: Option<f64>,
    expected_tolerance: Option<f64>,
    scenario: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_version: String,
    scenario_id: String,
    evidence_class: String,
    estimand: String,
    outcome_metric: String,
    outcome_unit: String,
    outcome_direction: String,
    parallel_trends_tolerance: f64,
    matched_sets: Vec<MatchedSet>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchedSet {
    match_set_id: String,
    matching_profile_digest: String,
    treated: UnitRecord,
    comparator: UnitRecord,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnitRecord {
    unit_id: String,
    vehicle_ref: String,
    pack_ref: String,
    matching_profile_digest: String,
    pre_configuration_digest: String,
    post_configuration_digest: String,
    follow_up_status: String,
    censoring_reason: Option<String>,
    pretrend: Measurement,
    pre: Measurement,
    post: Option<Measurement>,
    intervention: InterventionRecord,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    value: f64,
    observation_digest: String,
    protocol_digest: String,
    condition_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InterventionRecord {
    status: String,
    family: Option<String>,
    procedure_ref: Option<String>,
    service_event_ref: Option<String>,
    authority_ref: Option<String>,
    completion_digest: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AuditReport {
    pub scenario_count: usize,
    pub estimated_count: usize,
    pub abstain_count: usize,
    pub rejected_count: usize,
    pub report_sha256: String,
    pub cases: Vec<CaseReport>,
}

#[derive(Debug, Serialize)]
pub struct CaseReport {
    pub case_id: String,
    pub scenario_id: Option<String>,
    pub disposition: String,
    pub reason: Option<String>,
    pub matched_pair_count: usize,
    pub censored_unit_count: usize,
    pub censored_unit_ids: Vec<String>,
    pub treated_mean_change: Option<f64>,
    pub comparator_mean_change: Option<f64>,
    pub did_effect: Option<f64>,
    pub expected_did_effect: Option<f64>,
    pub absolute_effect_error: Option<f64>,
}

#[derive(Debug)]
struct Evaluation {
    scenario_id: String,
    matched_pair_count: usize,
    censored_unit_ids: Vec<String>,
    treated_mean_change: Option<f64>,
    comparator_mean_change: Option<f64>,
    did_effect: Option<f64>,
    disposition: String,
    reason: Option<String>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(schema_bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(schema_bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match frozen intervention contract".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_unit(unit: &UnitRecord, treated: bool) -> Result<(), String> {
    if !nonempty(&unit.unit_id) || !nonempty(&unit.vehicle_ref) || !nonempty(&unit.pack_ref) {
        return Err("unit identity is incomplete".to_owned());
    }
    if !is_digest(&unit.matching_profile_digest)
        || !is_digest(&unit.pre_configuration_digest)
        || !is_digest(&unit.post_configuration_digest)
    {
        return Err(format!("unit {} has invalid configuration or matching digest", unit.unit_id));
    }
    if !unit.pretrend.value.is_finite()
        || !is_digest(&unit.pretrend.observation_digest)
        || !is_digest(&unit.pretrend.protocol_digest)
        || !is_digest(&unit.pretrend.condition_digest)
        || unit.pretrend.protocol_digest != unit.pre.protocol_digest
        || unit.pretrend.condition_digest != unit.pre.condition_digest
    {
        return Err(format!("unit {} has invalid pre-trend reference measurement", unit.unit_id));
    }
    if !unit.pre.value.is_finite()
        || !is_digest(&unit.pre.observation_digest)
        || !is_digest(&unit.pre.protocol_digest)
        || !is_digest(&unit.pre.condition_digest)
    {
        return Err(format!("unit {} has invalid pre-intervention measurement", unit.unit_id));
    }
    match unit.follow_up_status.as_str() {
        "complete" => {
            let post = unit
                .post
                .as_ref()
                .ok_or_else(|| format!("unit {} lacks its required post measurement", unit.unit_id))?;
            if unit.censoring_reason.is_some() {
                return Err(format!("unit {} marks complete follow-up as censored", unit.unit_id));
            }
            if !post.value.is_finite()
                || !is_digest(&post.observation_digest)
                || !is_digest(&post.protocol_digest)
                || !is_digest(&post.condition_digest)
            {
                return Err(format!("unit {} has invalid post-intervention measurement", unit.unit_id));
            }
            if post.protocol_digest != unit.pre.protocol_digest
                || post.condition_digest != unit.pre.condition_digest
            {
                return Err(format!(
                    "unit {} changed reference protocol or conditions between measurements",
                    unit.unit_id
                ));
            }
        }
        "administrative_censoring" | "lost_to_follow_up" | "retired_due_to_fault" => {
            if unit.post.is_some() || unit.censoring_reason.as_deref().is_none_or(|v| !nonempty(v)) {
                return Err(format!(
                    "unit {} must preserve a censoring reason and must not fabricate a post measurement",
                    unit.unit_id
                ));
            }
        }
        _ => return Err(format!("unit {} has an unknown follow-up status", unit.unit_id)),
    }

    if treated {
        let intervention = &unit.intervention;
        if intervention.status != "verified_complete" {
            return Err(format!(
                "unit {} is not an independently verified completed intervention",
                unit.unit_id
            ));
        }
        if intervention.family.as_deref().is_none_or(|v| !nonempty(v))
            || intervention.procedure_ref.as_deref().is_none_or(|v| !nonempty(v))
            || intervention.service_event_ref.as_deref().is_none_or(|v| !nonempty(v))
            || intervention.authority_ref.as_deref().is_none_or(|v| !nonempty(v))
            || intervention.completion_digest.as_deref().is_none_or(|v| !is_digest(v))
        {
            return Err(format!(
                "unit {} lacks procedure, service, authority or completion evidence",
                unit.unit_id
            ));
        }
    } else if intervention_status(&unit.intervention) != "no_intervention" {
        return Err(format!(
            "comparator {} is contaminated by an intervention",
            unit.unit_id
        ));
    }
    Ok(())
}

fn intervention_status(intervention: &InterventionRecord) -> &str {
    intervention.status.as_str()
}

fn evaluate(scenario: Scenario) -> Result<Evaluation, String> {
    if scenario.scenario_version != SCENARIO_VERSION || !nonempty(&scenario.scenario_id) {
        return Err("scenario version or identity is invalid".to_owned());
    }
    if scenario.evidence_class != "synthetic_fixture" {
        return Err("v0.1 accepts synthetic fixtures only".to_owned());
    }
    if scenario.estimand != "matched_difference_in_differences" {
        return Err("unsupported estimand".to_owned());
    }
    if !nonempty(&scenario.outcome_metric) || !nonempty(&scenario.outcome_unit) {
        return Err("outcome metric and unit are required".to_owned());
    }
    if !scenario.parallel_trends_tolerance.is_finite() || scenario.parallel_trends_tolerance < 0.0 {
        return Err("parallel-trends tolerance must be finite and non-negative".to_owned());
    }
    let direction = match scenario.outcome_direction.as_str() {
        "higher_is_better" => 1.0,
        "lower_is_better" => -1.0,
        _ => return Err("unknown outcome direction".to_owned()),
    };
    if scenario.matched_sets.len() < 2 {
        return Err("at least two matched sets are required".to_owned());
    }

    let mut sets = scenario.matched_sets;
    sets.sort_by(|left, right| left.match_set_id.cmp(&right.match_set_id));
    let mut set_ids = BTreeSet::new();
    let mut unit_ids = BTreeSet::new();
    let mut censored = Vec::new();
    let mut reference: Option<(String, String)> = None;

    for set in &sets {
        if !nonempty(&set.match_set_id) || !set_ids.insert(set.match_set_id.clone()) {
            return Err("matched-set identity is empty or duplicated".to_owned());
        }
        if !is_digest(&set.matching_profile_digest)
            || set.treated.matching_profile_digest != set.matching_profile_digest
            || set.comparator.matching_profile_digest != set.matching_profile_digest
        {
            return Err(format!("matched set {} has inconsistent matching evidence", set.match_set_id));
        }
        validate_unit(&set.treated, true)?;
        validate_unit(&set.comparator, false)?;

        for unit in [&set.treated, &set.comparator] {
            if !unit_ids.insert(unit.unit_id.clone()) {
                return Err(format!("unit {} is reused across matched sets", unit.unit_id));
            }
            let key = (
                unit.pre.protocol_digest.clone(),
                unit.pre.condition_digest.clone(),
            );
            if let Some(expected) = &reference {
                if expected != &key {
                    return Err("matched units do not share the same reference protocol and conditions".to_owned());
                }
            } else {
                reference = Some(key);
            }
            if unit.follow_up_status != "complete" {
                censored.push(unit.unit_id.clone());
            }
        }
    }

    censored.sort();
    if !censored.is_empty() {
        return Ok(Evaluation {
            scenario_id: scenario.scenario_id,
            matched_pair_count: sets.len(),
            censored_unit_ids: censored,
            treated_mean_change: None,
            comparator_mean_change: None,
            did_effect: None,
            disposition: "abstain".to_owned(),
            reason: Some("censored follow-up prevents the v0.1 fixed-horizon DiD estimate".to_owned()),
        });
    }

    let mut pretrend_violation = false;
    for set in &sets {
        let treated_pretrend =
            direction * (set.treated.pre.value - set.treated.pretrend.value);
        let comparator_pretrend =
            direction * (set.comparator.pre.value - set.comparator.pretrend.value);
        let gap = (treated_pretrend - comparator_pretrend).abs();
        if !gap.is_finite() {
            return Err("non-finite pre-trend diagnostic".to_owned());
        }
        if gap > scenario.parallel_trends_tolerance {
            pretrend_violation = true;
        }
    }
    if pretrend_violation {
        return Ok(Evaluation {
            scenario_id: scenario.scenario_id,
            matched_pair_count: sets.len(),
            censored_unit_ids: Vec::new(),
            treated_mean_change: None,
            comparator_mean_change: None,
            did_effect: None,
            disposition: "abstain".to_owned(),
            reason: Some("pre-intervention trends differ beyond the declared tolerance".to_owned()),
        });
    }

    let mut treated_changes = Vec::with_capacity(sets.len());
    let mut comparator_changes = Vec::with_capacity(sets.len());
    let mut effects = Vec::with_capacity(sets.len());
    for set in &sets {
        let treated_post = set.treated.post.as_ref().expect("validated complete follow-up");
        let comparator_post = set.comparator.post.as_ref().expect("validated complete follow-up");
        let treated_change = direction * (treated_post.value - set.treated.pre.value);
        let comparator_change = direction * (comparator_post.value - set.comparator.pre.value);
        let effect = treated_change - comparator_change;
        if !treated_change.is_finite() || !comparator_change.is_finite() || !effect.is_finite() {
            return Err("non-finite effect arithmetic".to_owned());
        }
        treated_changes.push(treated_change);
        comparator_changes.push(comparator_change);
        effects.push(effect);
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    Ok(Evaluation {
        scenario_id: scenario.scenario_id,
        matched_pair_count: sets.len(),
        censored_unit_ids: Vec::new(),
        treated_mean_change: Some(mean(&treated_changes)),
        comparator_mean_change: Some(mean(&comparator_changes)),
        did_effect: Some(mean(&effects)),
        disposition: "estimated".to_owned(),
        reason: None,
    })
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus: Corpus =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION {
        return Err("unsupported intervention corpus version".to_owned());
    }
    corpus.cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    for case in &mut corpus.cases {
        if let Some(sets) = case.scenario.get_mut("matched_sets").and_then(Value::as_array_mut) {
            sets.sort_by(|left, right| {
                left.get("match_set_id")
                    .and_then(Value::as_str)
                    .cmp(&right.get("match_set_id").and_then(Value::as_str))
            });
        }
    }

    let canonical = serde_json::to_vec(&corpus).map_err(|error| format!("canonical corpus: {error}"))?;
    let report_sha256 = sha256(&canonical);
    let mut reports = Vec::new();
    let mut seen = BTreeSet::new();
    let mut estimated_count = 0;
    let mut abstain_count = 0;
    let mut rejected_count = 0;

    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expected_disposition.as_str(), "estimated" | "abstain" | "rejected") {
            return Err(format!("case {} has an unknown expected disposition", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.scenario)
            .map(|error| error.to_string())
            .collect();
        if !schema_errors.is_empty() {
            return Err(format!(
                "case {} fails the scenario JSON Schema: {}",
                case.case_id,
                schema_errors.join("; ")
            ));
        }
        let scenario_id = case.scenario.get("scenario_id").and_then(Value::as_str).map(str::to_owned);
        let parsed: Result<Scenario, _> = serde_json::from_value(case.scenario);
        let evaluated = parsed
            .map_err(|error| format!("scenario decode: {error}"))
            .and_then(evaluate);

        match (case.expected_disposition.as_str(), evaluated) {
            ("rejected", Err(reason)) => {
                rejected_count += 1;
                reports.push(CaseReport {
                    case_id: case.case_id,
                    scenario_id,
                    disposition: "rejected".to_owned(),
                    reason: Some(reason),
                    matched_pair_count: 0,
                    censored_unit_count: 0,
                    censored_unit_ids: Vec::new(),
                    treated_mean_change: None,
                    comparator_mean_change: None,
                    did_effect: None,
                    expected_did_effect: case.expected_did_effect,
                    absolute_effect_error: None,
                });
            }
            ("estimated", Ok(result)) if result.disposition == "estimated" => {
                let expected = case.expected_did_effect.ok_or_else(|| {
                    format!("case {} is missing its known-truth effect", case.case_id)
                })?;
                let tolerance = case.expected_tolerance.unwrap_or(EPSILON);
                let actual = result.did_effect.expect("estimated disposition has an effect");
                let error = (actual - expected).abs();
                if !error.is_finite() || error > tolerance {
                    return Err(format!(
                        "case {} effect mismatch: actual {actual}, expected {expected}, error {error}",
                        case.case_id
                    ));
                }
                estimated_count += 1;
                reports.push(CaseReport {
                    case_id: case.case_id,
                    scenario_id: Some(result.scenario_id),
                    disposition: result.disposition,
                    reason: result.reason,
                    matched_pair_count: result.matched_pair_count,
                    censored_unit_count: result.censored_unit_ids.len(),
                    censored_unit_ids: result.censored_unit_ids,
                    treated_mean_change: result.treated_mean_change,
                    comparator_mean_change: result.comparator_mean_change,
                    did_effect: result.did_effect,
                    expected_did_effect: Some(expected),
                    absolute_effect_error: Some(error),
                });
            }
            ("abstain", Ok(result)) if result.disposition == "abstain" => {
                abstain_count += 1;
                reports.push(CaseReport {
                    case_id: case.case_id,
                    scenario_id: Some(result.scenario_id),
                    disposition: result.disposition,
                    reason: result.reason,
                    matched_pair_count: result.matched_pair_count,
                    censored_unit_count: result.censored_unit_ids.len(),
                    censored_unit_ids: result.censored_unit_ids,
                    treated_mean_change: result.treated_mean_change,
                    comparator_mean_change: result.comparator_mean_change,
                    did_effect: result.did_effect,
                    expected_did_effect: None,
                    absolute_effect_error: None,
                });
            }
            (expected, Ok(result)) => {
                return Err(format!(
                    "case {} expected {expected} but evaluator returned {}",
                    case.case_id, result.disposition
                ));
            }
            (expected, Err(error)) => {
                return Err(format!(
                    "case {} expected {expected} but evaluator rejected it: {error}",
                    case.case_id
                ));
            }
        }
    }

    Ok(AuditReport {
        scenario_count: reports.len(),
        estimated_count,
        abstain_count,
        rejected_count,
        report_sha256,
        cases: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_intervention_causal_scenario_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/intervention_causal_scenarios_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen synthetic causal corpus")
    }

    #[test]
    fn frozen_corpus_checks_positive_null_harm_censoring_and_rejection() {
        let report = report();
        assert_eq!(report.scenario_count, 8);
        assert_eq!(report.estimated_count, 4);
        assert_eq!(report.abstain_count, 2);
        assert_eq!(report.rejected_count, 2);
    }

    #[test]
    fn regression_to_mean_does_not_become_a_treatment_effect() {
        let report = report();
        let case = report.cases.iter()
            .find(|case| case.case_id == "regression-to-mean-negative-control")
            .expect("negative control");
        assert_eq!(case.disposition, "estimated");
        assert!((case.treated_mean_change.unwrap() - 4.5).abs() < EPSILON);
        assert!((case.comparator_mean_change.unwrap() - 4.5).abs() < EPSILON);
        assert!(case.did_effect.unwrap().abs() < EPSILON);
    }

    #[test]
    fn censoring_is_preserved_and_forces_abstention() {
        let report = report();
        let case = report.cases.iter()
            .find(|case| case.case_id == "censoring-forces-abstention")
            .expect("censored fixture");
        assert_eq!(case.disposition, "abstain");
        assert_eq!(case.censored_unit_count, 1);
        assert_eq!(case.censored_unit_ids, vec!["treated-2"]);
        assert!(case.did_effect.is_none());
    }

    #[test]
    fn lower_is_better_metric_direction_normalizes_benefit_positive() {
        let report = report();
        let case = report.cases.iter()
            .find(|case| case.case_id == "lower-is-better-resistance-effect")
            .expect("lower-is-better fixture");
        assert_eq!(case.disposition, "estimated");
        assert!((case.did_effect.unwrap() - 7.0).abs() < EPSILON);
    }

    #[test]
    fn divergent_preintervention_trends_force_abstention() {
        let report = report();
        let case = report.cases.iter()
            .find(|case| case.case_id == "pretrend-violation-abstains")
            .expect("pre-trend negative control");
        assert_eq!(case.disposition, "abstain");
        assert!(case.reason.as_deref().unwrap().contains("pre-intervention trends"));
        assert!(case.did_effect.is_none());
    }

    #[test]
    fn work_order_without_verified_completion_is_rejected() {
        let report = report();
        let case = report.cases.iter()
            .find(|case| case.case_id == "work-order-is-not-completion")
            .expect("work order fixture");
        assert_eq!(case.disposition, "rejected");
        assert!(case.reason.as_deref().unwrap().contains("verified completed intervention"));
    }

    #[test]
    fn corpus_and_matched_set_permutations_preserve_the_report_digest() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("valid corpus JSON");
        value["cases"].as_array_mut().expect("cases").reverse();
        for case in value["cases"].as_array_mut().expect("cases") {
            case["scenario"]["matched_sets"]
                .as_array_mut().expect("matched sets").reverse();
        }
        let permuted = serde_json::to_vec(&value).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &permuted).expect("permuted replay");
        assert_eq!(original.report_sha256, replay.report_sha256);
    }
}
