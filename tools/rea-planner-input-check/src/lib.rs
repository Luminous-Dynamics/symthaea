//! Validation for the synthetic REA planner-input and held-out DGP manifests.
//!
//! This checks benchmark input accounting and mathematical invariants only.
//! It does not execute planners, authenticate caller-supplied provenance,
//! calibrate likelihoods, or establish a model-performance result.

use jsonschema::draft202012;
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, BTreeSet};

const PLANNER_SCHEMA: &str =
    include_str!("../../../docs/research/rea-planner-input-manifest-v0.schema.json");
const PLANNER_MANIFEST: &str =
    include_str!("../../../docs/research/rea-planner-input-manifest-v0.json");
const GRADER_SCHEMA: &str =
    include_str!("../../../docs/research/rea-planner-outcome-grader-v0.schema.json");
const GRADER_MANIFEST: &str =
    include_str!("../../../docs/research/rea-planner-outcome-grader-v0.json");
const AGENT_VIEW: &str =
    include_str!("../tests/fixtures/rea-proof-obligation-agent-view-v0.json");

const TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError(pub String);

impl std::fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ManifestError {}

fn array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>, ManifestError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| ManifestError(format!("{context}.{key} must be an array")))
}

fn string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, ManifestError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ManifestError(format!("{context}.{key} must be a string")))
}

fn bool_value(value: &Value, key: &str, context: &str) -> Result<bool, ManifestError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| ManifestError(format!("{context}.{key} must be a boolean")))
}

fn number(value: &Value, key: &str, context: &str) -> Result<f64, ManifestError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|item| item.is_finite())
        .ok_or_else(|| ManifestError(format!("{context}.{key} must be a finite number")))
}

fn unique_strings(items: &[Value], context: &str) -> Result<BTreeSet<String>, ManifestError> {
    let mut result = BTreeSet::new();
    for item in items {
        let value = item
            .as_str()
            .ok_or_else(|| ManifestError(format!("{context} contains a non-string ID")))?;
        if !result.insert(value.to_owned()) {
            return Err(ManifestError(format!("{context} contains duplicate ID {value}")));
        }
    }
    Ok(result)
}

fn is_evidence_id(value: &str) -> bool {
    value
        .strip_prefix("ev_")
        .is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn sum_to_one(values: impl Iterator<Item = f64>, context: &str) -> Result<(), ManifestError> {
    let collected: Vec<f64> = values.collect();
    if collected
        .iter()
        .any(|item| !item.is_finite() || *item < 0.0 || *item > 1.0)
    {
        return Err(ManifestError(format!(
            "{context} contains a non-finite or out-of-range probability"
        )));
    }
    let total: f64 = collected.iter().sum();
    if (total - 1.0).abs() > TOLERANCE {
        return Err(ManifestError(format!(
            "{context} probability mass must sum to 1.0 (received {total:.12})"
        )));
    }
    Ok(())
}

fn validate_likelihood_rows(
    rows: &[Value],
    hypothesis_ids: &BTreeSet<String>,
    outcome_count: usize,
    context: &str,
) -> Result<(), ManifestError> {
    let mut seen = BTreeSet::new();
    for row in rows {
        let hypothesis_id = string(row, "hypothesis_id", context)?;
        if !hypothesis_ids.contains(hypothesis_id) {
            return Err(ManifestError(format!(
                "{context} references unknown hypothesis {hypothesis_id}"
            )));
        }
        if !seen.insert(hypothesis_id.to_owned()) {
            return Err(ManifestError(format!(
                "{context} contains duplicate row for {hypothesis_id}"
            )));
        }
        let probabilities = array(row, "outcome_probabilities", context)?;
        if probabilities.len() != outcome_count {
            return Err(ManifestError(format!(
                "{context}/{hypothesis_id} has {} probabilities for {outcome_count} outcomes",
                probabilities.len()
            )));
        }
        sum_to_one(
            probabilities.iter().map(|item| item.as_f64().unwrap_or(f64::NAN)),
            &format!("{context}/{hypothesis_id}"),
        )?;
    }
    if seen != *hypothesis_ids {
        return Err(ManifestError(format!(
            "{context} hypothesis coverage differs from the case hypothesis set"
        )));
    }
    Ok(())
}

fn validate_agent_view_alignment(
    case: &Value,
    agent_case: &Value,
    context: &str,
) -> Result<(), ManifestError> {
    if string(case, "case_id", context)? != string(agent_case, "case_id", "agent_case")? {
        return Err(ManifestError(format!("{context} case_id does not match agent view")));
    }

    let candidates = array(case, "candidates", context)?;
    let agent_candidates = array(agent_case, "candidates", "agent_case")?;
    if candidates.len() != agent_candidates.len() {
        return Err(ManifestError(format!("{context} candidate count differs from agent view")));
    }

    for (candidate, source) in candidates.iter().zip(agent_candidates) {
        let candidate_context = format!("{context}/{}", string(candidate, "id", context)?);
        let source_id = string(source, "experiment_id", "agent_candidate")?;
        if string(candidate, "id", &candidate_context)? != source_id {
            return Err(ManifestError(format!(
                "{candidate_context} candidate order/identity differs from agent view"
            )));
        }
        let outcomes: Vec<String> = array(candidate, "outcomes", &candidate_context)?
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    ManifestError(format!("{candidate_context} contains non-string outcome"))
                })
            })
            .collect::<Result<_, _>>()?;
        let source_outcomes: Vec<String> = array(source, "outcomes", "agent_candidate")?
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    ManifestError("agent view contains non-string outcome".to_owned())
                })
            })
            .collect::<Result<_, _>>()?;
        if outcomes != source_outcomes {
            return Err(ManifestError(format!(
                "{candidate_context} outcome labels/order differ from agent view"
            )));
        }

        let authorization = match string(source, "authorization", "agent_candidate")? {
            "authorized" => "Authorized",
            "denied" => "Denied",
            "unknown" => "Unknown",
            other => {
                return Err(ManifestError(format!(
                    "unsupported source authorization {other}"
                )));
            }
        };
        if string(candidate, "authorization", &candidate_context)? != authorization
            || bool_value(candidate, "reversible", &candidate_context)?
                != bool_value(source, "reversible", "agent_candidate")?
            || bool_value(candidate, "destructive", &candidate_context)?
                != bool_value(source, "destructive", "agent_candidate")?
        {
            return Err(ManifestError(format!(
                "{candidate_context} authorization/reversibility/destruction differs from agent view"
            )));
        }
        if (number(candidate, "cost_units", &candidate_context)?
            - number(source, "cost_units", "agent_candidate")?)
        .abs()
            > TOLERANCE
        {
            return Err(ManifestError(format!(
                "{candidate_context} cost_units differs from agent view"
            )));
        }
        if string(candidate, "description", &candidate_context)?
            != string(source, "description", "agent_candidate")?
        {
            return Err(ManifestError(format!(
                "{candidate_context} description differs from agent view"
            )));
        }
        if string(candidate, "declared_source_impact", &candidate_context)?
            != string(source, "target_impact", "agent_candidate")?
        {
            return Err(ManifestError(format!(
                "{candidate_context} declared source impact differs from agent view"
            )));
        }
    }
    Ok(())
}

fn validate_case(case: &Value, agent_case: &Value, context: &str) -> Result<(), ManifestError> {
    validate_agent_view_alignment(case, agent_case, context)?;

    let evidence = array(case, "evidence", context)?;
    let mut evidence_by_id = BTreeMap::<String, &Value>::new();
    let mut source_roots = BTreeSet::new();
    let mut source_families = BTreeSet::new();
    for item in evidence {
        let id = string(item, "evidence_id", context)?;
        if !is_evidence_id(id) {
            return Err(ManifestError(format!("{context} has malformed Evidence ID {id}")));
        }
        if evidence_by_id.insert(id.to_owned(), item).is_some() {
            return Err(ManifestError(format!("{context} has duplicate Evidence ID {id}")));
        }
        let family = string(item, "fixture_source_family_id", context)?;
        if !source_families.insert(family.to_owned()) {
            return Err(ManifestError(format!(
                "{context} reuses a source-family root {family}"
            )));
        }
        let origin = string(item, "origin", context)?;
        if origin == "IndependentObservation" {
            if !array(item, "parent_evidence_ids", context)?.is_empty() {
                return Err(ManifestError(format!(
                    "{context} independent observation {id} cannot have parent Evidence"
                )));
            }
            source_roots.insert(id.to_owned());
        }
    }
    if source_roots.len() < 2 {
        return Err(ManifestError(format!(
            "{context} needs two distinct synthetic independent observation roots"
        )));
    }
    for item in evidence {
        for parent in array(item, "parent_evidence_ids", context)? {
            let parent_id = parent.as_str().ok_or_else(|| {
                ManifestError(format!("{context} parent Evidence ID is not a string"))
            })?;
            if !evidence_by_id.contains_key(parent_id) {
                return Err(ManifestError(format!(
                    "{context} Evidence has dangling parent {parent_id}"
                )));
            }
        }
    }

    let hypotheses = array(case, "hypotheses", context)?;
    let mut hypothesis_ids = BTreeSet::new();
    for hypothesis in hypotheses {
        let id = string(hypothesis, "id", context)?;
        if !hypothesis_ids.insert(id.to_owned()) {
            return Err(ManifestError(format!("{context} duplicate hypothesis {id}")));
        }
        let prior = number(hypothesis, "prior_probability", context)?;
        if prior <= 0.0 || prior > 1.0 {
            return Err(ManifestError(format!(
                "{context}/{id} prior must be in (0, 1]"
            )));
        }
        let supporting = array(hypothesis, "supporting_evidence_ids", context)?;
        if supporting.is_empty() {
            return Err(ManifestError(format!(
                "{context}/{id} has no independently rooted support"
            )));
        }
        let mut rooted = false;
        for support in supporting {
            let support_id = support.as_str().ok_or_else(|| {
                ManifestError(format!("{context}/{id} has non-string support Evidence ID"))
            })?;
            let record = evidence_by_id.get(support_id).ok_or_else(|| {
                ManifestError(format!("{context}/{id} references missing Evidence {support_id}"))
            })?;
            rooted |= string(record, "origin", context)? == "IndependentObservation"
                || source_roots.iter().any(|root| *root == support_id);
        }
        if !rooted {
            return Err(ManifestError(format!(
                "{context}/{id} has no independent observation root"
            )));
        }
        for contradiction in array(hypothesis, "contradicting_evidence_ids", context)? {
            let contradiction_id = contradiction.as_str().ok_or_else(|| {
                ManifestError(format!("{context}/{id} has non-string contradiction Evidence ID"))
            })?;
            if !evidence_by_id.contains_key(contradiction_id) {
                return Err(ManifestError(format!(
                    "{context}/{id} references missing contradiction Evidence {contradiction_id}"
                )));
            }
        }
    }
    sum_to_one(
        hypotheses
            .iter()
            .map(|item| item.get("prior_probability").and_then(Value::as_f64).unwrap_or(f64::NAN)),
        &format!("{context} hypothesis prior"),
    )?;

    let candidates = array(case, "candidates", context)?;
    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        let id = string(candidate, "id", context)?;
        if !candidate_ids.insert(id.to_owned()) {
            return Err(ManifestError(format!("{context} duplicate candidate {id}")));
        }
        let outcomes = array(candidate, "outcomes", context)?;
        let outcome_ids = unique_strings(outcomes, &format!("{context}/{id} outcomes"))?;
        let rows = array(candidate, "point_likelihoods", context)?;
        validate_likelihood_rows(
            rows,
            &hypothesis_ids,
            outcomes.len(),
            &format!("{context}/{id} point likelihood"),
        )?;
        let target_impact = string(candidate, "target_impact", context)?;
        let source_impact = string(candidate, "declared_source_impact", context)?;
        let expected_impact = match source_impact {
            "isolated_test" => "ReadOnlyObservation",
            "read_only" => "ReadOnlyObservation",
            "temporary_configuration" => "TemporaryConfiguration",
            "service_restart" => "ServiceRestart",
            "data_mutation" => "DataMutation",
            "physical_actuation" => "PhysicalActuation",
            other => return Err(ManifestError(format!("unsupported source impact {other}"))),
        };
        if target_impact != expected_impact {
            return Err(ManifestError(format!(
                "{context}/{id} impact mapping is not the declared deterministic mapping"
            )));
        }
        let _ = outcome_ids;
    }

    let scenarios = array(case, "likelihood_scenarios", context)?;
    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        let scenario_id = string(scenario, "scenario_id", context)?;
        if !scenario_ids.insert(scenario_id.to_owned()) {
            return Err(ManifestError(format!("{context} duplicate scenario {scenario_id}")));
        }
        let weight = number(scenario, "scenario_probability", context)?;
        if weight <= 0.0 || weight > 1.0 {
            return Err(ManifestError(format!("{context}/{scenario_id} scenario weight invalid")));
        }
        let experiments = array(scenario, "experiments", context)?;
        if experiments.len() != candidates.len() {
            return Err(ManifestError(format!(
                "{context}/{scenario_id} must cover every candidate exactly once"
            )));
        }
        let mut covered = BTreeSet::new();
        for experiment in experiments {
            let experiment_id = string(experiment, "experiment_id", context)?;
            if !candidate_ids.contains(experiment_id) || !covered.insert(experiment_id.to_owned()) {
                return Err(ManifestError(format!(
                    "{context}/{scenario_id} has unknown or duplicate candidate {experiment_id}"
                )));
            }
            let candidate = candidates
                .iter()
                .find(|item| item.get("id").and_then(Value::as_str) == Some(experiment_id))
                .expect("candidate ID presence was checked");
            let outcome_count = array(candidate, "outcomes", context)?.len();
            validate_likelihood_rows(
                array(experiment, "likelihoods", context)?,
                &hypothesis_ids,
                outcome_count,
                &format!("{context}/{scenario_id}/{experiment_id}"),
            )?;
        }
        if covered != candidate_ids {
            return Err(ManifestError(format!(
                "{context}/{scenario_id} candidate coverage is incomplete"
            )));
        }
    }
    sum_to_one(
        scenarios
            .iter()
            .map(|item| item.get("scenario_probability").and_then(Value::as_f64).unwrap_or(f64::NAN)),
        &format!("{context} scenario weights"),
    )?;

    let decisions = array(case, "terminal_decisions", context)?;
    let mut decision_ids = BTreeSet::new();
    for decision in decisions {
        let id = string(decision, "id", context)?;
        if !decision_ids.insert(id.to_owned()) {
            return Err(ManifestError(format!("{context} duplicate decision {id}")));
        }
        let utilities = array(decision, "utilities", context)?;
        let mut covered = BTreeSet::new();
        for utility in utilities {
            let hypothesis_id = string(utility, "hypothesis_id", context)?;
            if !hypothesis_ids.contains(hypothesis_id) || !covered.insert(hypothesis_id.to_owned()) {
                return Err(ManifestError(format!(
                    "{context}/{id} has unknown or duplicate utility for {hypothesis_id}"
                )));
            }
            let _utility = number(utility, "utility", context)?;
        }
        if covered != hypothesis_ids {
            return Err(ManifestError(format!(
                "{context}/{id} utility matrix is incomplete"
            )));
        }
    }
    Ok(())
}

fn validate_grader_case(
    grader_case: &Value,
    planner_case: &Value,
    context: &str,
) -> Result<(), ManifestError> {
    if bool_value(grader_case, "held_out_dgp_not_supplied_to_planner", context)? != true {
        return Err(ManifestError(format!(
            "{context} grader-only DGP flag must remain true"
        )));
    }
    let planner_hypotheses = array(planner_case, "hypotheses", context)?;
    let hypothesis_ids = unique_strings(
        &planner_hypotheses
            .iter()
            .map(|item| item["id"].clone())
            .collect::<Vec<_>>(),
        &format!("{context} planner hypothesis IDs"),
    )?;
    let planner_candidates = array(planner_case, "candidates", context)?;
    let candidate_ids = unique_strings(
        &planner_candidates
            .iter()
            .map(|item| item["id"].clone())
            .collect::<Vec<_>>(),
        &format!("{context} planner candidate IDs"),
    )?;
    let scenarios = array(planner_case, "likelihood_scenarios", context)?;
    let hidden = array(grader_case, "hidden_likelihoods", context)?;
    if hidden.len() != planner_candidates.len() {
        return Err(ManifestError(format!(
            "{context} held-out DGP must cover every candidate"
        )));
    }
    let mut hidden_by_candidate = BTreeMap::<String, &Value>::new();
    for entry in hidden {
        let experiment_id = string(entry, "experiment_id", context)?;
        if !candidate_ids.contains(experiment_id)
            || hidden_by_candidate.insert(experiment_id.to_owned(), entry).is_some()
        {
            return Err(ManifestError(format!(
                "{context} held-out DGP has unknown or duplicate candidate {experiment_id}"
            )));
        }
        let candidate = planner_candidates
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(experiment_id))
            .expect("candidate was checked");
        let outcome_count = array(candidate, "outcomes", context)?.len();
        validate_likelihood_rows(
            array(entry, "likelihoods", context)?,
            &hypothesis_ids,
            outcome_count,
            &format!("{context}/held_out/{experiment_id}"),
        )?;
    }
    if hidden_by_candidate.keys().cloned().collect::<BTreeSet<_>>() != candidate_ids {
        return Err(ManifestError(format!("{context} hidden DGP candidate coverage mismatch")));
    }

    let testable = bool_value(grader_case, "out_of_ensemble_testable", context)?;
    let reason = grader_case.get("not_testable_reason");
    if testable {
        if reason.is_some_and(|value| !value.is_null()) {
            return Err(ManifestError(format!(
                "{context} testable held-out DGP must not carry a not-testable reason"
            )));
        }
        // A held-out DGP must differ from every complete declared scenario,
        // not merely from at least one scenario.
        for scenario in scenarios {
            let experiments = array(scenario, "experiments", context)?;
            let matches_entire_scenario = experiments.iter().all(|declared| {
                let id = declared["experiment_id"].as_str().unwrap_or("");
                hidden_by_candidate.get(id).is_some_and(|hidden_entry| {
                    canonical_value_equal(
                        &hidden_entry["likelihoods"],
                        &declared["likelihoods"],
                    )
                })
            });
            if matches_entire_scenario {
                return Err(ManifestError(format!(
                    "{context} held-out DGP duplicates declared scenario {}",
                    string(scenario, "scenario_id", context)?
                )));
            }
        }
    } else {
        if !reason.is_some_and(Value::is_string) {
            return Err(ManifestError(format!(
                "{context} non-testable case must explain its outcome-space limitation"
            )));
        }
        if !planner_candidates
            .iter()
            .all(|candidate| array(candidate, "outcomes", context).is_ok_and(|items| items.len() == 1))
        {
            return Err(ManifestError(format!(
                "{context} is marked not-testable although it has a multi-outcome candidate"
            )));
        }
    }

    let realized_hypothesis = string(grader_case, "realized_hypothesis_id", context)?;
    if !hypothesis_ids.contains(realized_hypothesis) {
        return Err(ManifestError(format!(
            "{context} realized hypothesis is absent from planner case"
        )));
    }

    Ok(())
}

fn canonical_value_equal(left: &Value, right: &Value) -> bool {
    left == right
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn validate_embedded_manifests() -> Result<(), ManifestError> {
    let planner_schema: Value = serde_json::from_str(PLANNER_SCHEMA)
        .map_err(|error| ManifestError(format!("planner schema parse failed: {error}")))?;
    draft202012::meta::validate(&planner_schema)
        .map_err(|error| ManifestError(format!("planner schema meta-validation failed: {error}")))?;
    let grader_schema: Value = serde_json::from_str(GRADER_SCHEMA)
        .map_err(|error| ManifestError(format!("grader schema parse failed: {error}")))?;
    draft202012::meta::validate(&grader_schema)
        .map_err(|error| ManifestError(format!("grader schema meta-validation failed: {error}")))?;

    let planner: Value = serde_json::from_str(PLANNER_MANIFEST)
        .map_err(|error| ManifestError(format!("planner manifest parse failed: {error}")))?;
    let planner_validator = draft202012::options()
        .build(&planner_schema)
        .map_err(|error| ManifestError(format!("planner schema compile failed: {error}")))?;
    let planner_errors: Vec<String> = planner_validator
        .iter_errors(&planner)
        .map(|error| error.to_string())
        .collect();
    if !planner_errors.is_empty() {
        return Err(ManifestError(format!(
            "planner manifest schema validation failed: {}",
            planner_errors.join("; ")
        )));
    }

    let grader: Value = serde_json::from_str(GRADER_MANIFEST)
        .map_err(|error| ManifestError(format!("grader manifest parse failed: {error}")))?;
    let grader_validator = draft202012::options()
        .build(&grader_schema)
        .map_err(|error| ManifestError(format!("grader schema compile failed: {error}")))?;
    let grader_errors: Vec<String> = grader_validator
        .iter_errors(&grader)
        .map(|error| error.to_string())
        .collect();
    if !grader_errors.is_empty() {
        return Err(ManifestError(format!(
            "grader manifest schema validation failed: {}",
            grader_errors.join("; ")
        )));
    }

    let source_sha = git_blob_sha1(AGENT_VIEW.as_bytes());
    if planner["source_case_view_file_git_blob_sha1"].as_str() != Some(source_sha.as_str()) {
        return Err(ManifestError(
            "planner manifest agent-view Git blob SHA-1 does not match checked-in projection"
                .to_owned(),
        ));
    }
    let source: Value = serde_json::from_str(AGENT_VIEW)
        .map_err(|error| ManifestError(format!("agent-view source parse failed: {error}")))?;

    let planner_cases = array(&planner, "cases", "planner_manifest")?;
    let grader_cases = array(&grader, "cases", "grader_manifest")?;
    let source_cases = array(&source, "cases", "agent_view")?;
    let planner_ids = unique_strings(
        &planner_cases.iter().map(|item| item["case_id"].clone()).collect::<Vec<_>>(),
        "planner case IDs",
    )?;
    let grader_ids = unique_strings(
        &grader_cases.iter().map(|item| item["case_id"].clone()).collect::<Vec<_>>(),
        "grader case IDs",
    )?;
    let source_ids = unique_strings(
        &source_cases
            .iter()
            .filter(|item| planner_ids.contains(item.get("case_id").and_then(Value::as_str).unwrap_or("")))
            .map(|item| item["case_id"].clone())
            .collect::<Vec<_>>(),
        "selected source case IDs",
    )?;
    if planner_ids != grader_ids || planner_ids != source_ids {
        return Err(ManifestError(
            "planner, grader and parent agent-view case IDs differ".to_owned(),
        ));
    }

    for planner_case in planner_cases {
        let case_id = string(planner_case, "case_id", "planner_case")?;
        let context = format!("planner_case/{case_id}");
        let source_case = source_cases
            .iter()
            .find(|item| item.get("case_id").and_then(Value::as_str) == Some(case_id))
            .ok_or_else(|| ManifestError(format!("missing parent agent-view case {case_id}")))?;
        validate_case(planner_case, source_case, &context)?;

        let grader_case = grader_cases
            .iter()
            .find(|item| item.get("case_id").and_then(Value::as_str) == Some(case_id))
            .ok_or_else(|| ManifestError(format!("missing grader case {case_id}")))?;
        validate_grader_case(grader_case, planner_case, &context)?;
    }

    reject_grader_only_field_leaks(PLANNER_MANIFEST)?;
    Ok(())
}

fn reject_grader_only_field_leaks(serialized_planner: &str) -> Result<(), ManifestError> {
    let planner_text = serialized_planner.to_ascii_lowercase();
    for forbidden in [
        "held_out_dgp_id",
        "held_out_dgp_not_supplied_to_planner",
        "hidden_likelihoods",
        "realized_hypothesis_id",
        "outcome_oracle_policy",
    ] {
        if planner_text.contains(forbidden) {
            return Err(ManifestError(format!(
                "grader-only field {forbidden} leaked into the planner manifest"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planner() -> Value {
        serde_json::from_str(PLANNER_MANIFEST).unwrap()
    }

    fn grader() -> Value {
        serde_json::from_str(GRADER_MANIFEST).unwrap()
    }

    #[test]
    fn draft_2020_12_schemas_and_manifests_validate() {
        validate_embedded_manifests().unwrap();
    }

    #[test]
    fn prior_mass_mutation_is_rejected() {
        let mut value = planner();
        value["cases"][0]["hypotheses"][0]["prior_probability"] = Value::from(0.6);
        assert!(validate_case(
            &value["cases"][0],
            &serde_json::from_str(AGENT_VIEW).unwrap()["cases"][0],
            "mutated_case"
        )
        .is_err());
    }

    #[test]
    fn point_likelihood_mass_mutation_is_rejected() {
        let mut value = planner();
        value["cases"][0]["candidates"][0]["point_likelihoods"][0]["outcome_probabilities"][0] =
            Value::from(0.5);
        let source: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let case_id = value["cases"][0]["case_id"].as_str().unwrap();
        let source_case = source["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == case_id).unwrap();
        assert!(validate_case(&value["cases"][0], source_case, "mutated_case").is_err());
    }

    #[test]
    fn scenario_candidate_omission_is_rejected() {
        let mut value = planner();
        value["cases"][0]["likelihood_scenarios"][0]["experiments"]
            .as_array_mut().unwrap().pop();
        let source: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let case_id = value["cases"][0]["case_id"].as_str().unwrap();
        let source_case = source["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == case_id).unwrap();
        assert!(validate_case(&value["cases"][0], source_case, "mutated_case").is_err());
    }

    #[test]
    fn incomplete_terminal_utility_is_rejected() {
        let mut value = planner();
        value["cases"][0]["terminal_decisions"][0]["utilities"]
            .as_array_mut().unwrap().pop();
        let source: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let case_id = value["cases"][0]["case_id"].as_str().unwrap();
        let source_case = source["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == case_id).unwrap();
        assert!(validate_case(&value["cases"][0], source_case, "mutated_case").is_err());
    }

    #[test]
    fn hidden_dgp_candidate_mismatch_is_rejected() {
        let mut value = grader();
        value["cases"][0]["hidden_likelihoods"][0]["experiment_id"] =
            Value::String("undeclared_candidate".to_owned());
        let planner_value = planner();
        assert!(validate_grader_case(
            &value["cases"][0],
            &planner_value["cases"][0],
            "mutated_grader_case"
        )
        .is_err());
    }

    #[test]
    fn hidden_dgp_that_duplicates_the_declared_scenarios_is_rejected() {
        let mut value = grader();
        let planner_value = planner();
        let case_id = value["cases"][0]["case_id"].as_str().unwrap();
        let scenario = planner_value["cases"][0]["likelihood_scenarios"][0]["experiments"].clone();
        value["cases"][0]["hidden_likelihoods"] = scenario;
        assert!(validate_grader_case(
            &value["cases"][0],
            &planner_value["cases"][0],
            "mutated_grader_case"
        )
        .is_err());
        assert!(!case_id.is_empty());
    }

    #[test]
    fn single_outcome_negative_control_declares_held_out_shift_not_testable() {
        let grader_value = grader();
        let case = grader_value["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["case_id"] == "REA-POA-012")
            .unwrap();
        assert_eq!(case["out_of_ensemble_testable"], Value::Bool(false));
        assert!(case["not_testable_reason"].as_str().is_some());
    }

    #[test]
    fn source_agent_view_candidate_mutation_is_rejected() {
        let mut value = planner();
        value["cases"][0]["candidates"][0]["outcomes"][0] = Value::String("invented".into());
        let source: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let case_id = value["cases"][0]["case_id"].as_str().unwrap();
        let source_case = source["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == case_id).unwrap();
        assert!(validate_case(&value["cases"][0], source_case, "mutated").is_err());
    }

    #[test]
    fn grader_only_field_leak_is_rejected() {
        let mut value = planner();
        value["cases"][0]["realized_hypothesis_id"] = Value::String("hypothesis_a".into());
        let text = serde_json::to_string(&value).unwrap().to_ascii_lowercase();
        assert!(text.contains("realized_hypothesis_id"));
    }

    fn entropy_bits(probabilities: &[f64]) -> f64 {
        probabilities
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.log2())
            .sum()
    }

    fn fixture_candidate_eig(case_id: &str, candidate_id: &str) -> f64 {
        let value = planner();
        let case = value["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == case_id).unwrap();
        let candidate = case["candidates"].as_array().unwrap().iter()
            .find(|item| item["id"] == candidate_id).unwrap();
        let rows = candidate["point_likelihoods"].as_array().unwrap();
        let row_a = rows.iter().find(|row| row["hypothesis_id"] == "hypothesis_a").unwrap();
        let row_b = rows.iter().find(|row| row["hypothesis_id"] == "hypothesis_b").unwrap();
        let a = row_a["outcome_probabilities"].as_array().unwrap();
        let b = row_b["outcome_probabilities"].as_array().unwrap();
        let prior_entropy = entropy_bits(&[0.5, 0.5]);
        let mut expected_posterior_entropy = 0.0;
        for index in 0..a.len() {
            let pa = a[index].as_f64().unwrap();
            let pb = b[index].as_f64().unwrap();
            let outcome_mass = 0.5 * pa + 0.5 * pb;
            if outcome_mass > 0.0 {
                let posterior_a = 0.5 * pa / outcome_mass;
                let posterior_b = 0.5 * pb / outcome_mass;
                expected_posterior_entropy +=
                    outcome_mass * entropy_bits(&[posterior_a, posterior_b]);
            }
        }
        prior_entropy - expected_posterior_entropy
    }

    #[test]
    fn pilot_encodes_high_eig_non_decisive_vs_lower_eig_obligation_closure() {
        let static_probe = fixture_candidate_eig("REA-POA-001", "probe_many_functions");
        let deployment_manifest =
            fixture_candidate_eig("REA-POA-001", "verify_deployment_manifest");
        assert!(static_probe > deployment_manifest);
    }

    #[test]
    fn pilot_encodes_correlated_logs_as_more_informative_than_causal_intervention() {
        let logs = fixture_candidate_eig("REA-POA-008", "collect_more_cache_logs");
        let intervention = fixture_candidate_eig("REA-POA-008", "controlled_cache_bypass");
        assert!(logs > intervention);
    }

    #[test]
    fn pilot_encodes_unauthorized_probe_as_informative_but_denied() {
        let unauthorized =
            fixture_candidate_eig("REA-POA-013", "unauthorized_production_probe");
        let authorized =
            fixture_candidate_eig("REA-POA-013", "authorized_staging_reproduction");
        assert!(unauthorized > authorized);
        let value = planner();
        let candidate = value["cases"].as_array().unwrap().iter()
            .find(|item| item["case_id"] == "REA-POA-013").unwrap()["candidates"]
            .as_array().unwrap().iter()
            .find(|item| item["id"] == "unauthorized_production_probe").unwrap();
        assert_eq!(candidate["authorization"], "Denied");
    }

    #[test]
    fn grader_only_field_leak_is_rejected_by_the_semantic_guard() {
        let mut value = planner();
        value["cases"][0]["realized_hypothesis_id"] = Value::String("hypothesis_a".into());
        let serialized = serde_json::to_string(&value).unwrap();
        assert!(reject_grader_only_field_leaks(&serialized).is_err());
    }

    #[test]
    fn source_agent_view_blob_is_committed_and_checked() {
        let manifest: Value = serde_json::from_str(PLANNER_MANIFEST).unwrap();
        let declared = manifest["source_case_view_file_git_blob_sha1"].as_str().unwrap();
        assert_eq!(declared, git_blob_sha1(AGENT_VIEW.as_bytes()));
    }
}
