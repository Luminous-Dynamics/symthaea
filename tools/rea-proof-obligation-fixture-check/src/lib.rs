//! Structural and semantic-consistency checks for REA claim-obligation fixtures.
//!
//! This crate validates the authored research corpus. It does not adjudicate
//! the world-state oracle, execute experiments, or evaluate a planning policy.

pub mod score;

use jsonschema::draft202012;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA: &str = include_str!(
    "../../../docs/research/rea-proof-obligation-acquisition-fixtures-v0.schema.json"
);
const FIXTURE: &str =
    include_str!("../../../docs/research/rea-proof-obligation-acquisition-fixtures-v0.json");
const AGENT_SCHEMA: &str = include_str!(
    "../../../docs/research/rea-proof-obligation-acquisition-agent-view-v0.schema.json"
);
const AGENT_VIEW: &str = include_str!(
    "../../../docs/research/rea-proof-obligation-acquisition-agent-view-v0.json"
);
const GRADER_SCHEMA: &str = include_str!(
    "../../../docs/research/rea-proof-obligation-acquisition-grader-v0.schema.json"
);
const GRADER_VIEW: &str = include_str!(
    "../../../docs/research/rea-proof-obligation-acquisition-grader-v0.json"
);

const EXPECTED_CASES: usize = 18;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureError(pub String);

impl std::fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FixtureError {}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, FixtureError> {
    let mut current = value;
    for segment in path {
        current = current
            .get(*segment)
            .ok_or_else(|| FixtureError(format!("missing JSON path segment: {segment}")))?;
    }
    current.as_str().ok_or_else(|| {
        FixtureError(format!("JSON path is not a string: {}", path.join(".")))
    })
}

fn bool_at(value: &Value, path: &[&str]) -> Result<bool, FixtureError> {
    let mut current = value;
    for segment in path {
        current = current
            .get(*segment)
            .ok_or_else(|| FixtureError(format!("missing JSON path segment: {segment}")))?;
    }
    current
        .as_bool()
        .ok_or_else(|| FixtureError(format!("JSON path is not a boolean: {}", path.join("."))))
}

fn array_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Vec<Value>, FixtureError> {
    let mut current = value;
    for segment in path {
        current = current
            .get(*segment)
            .ok_or_else(|| FixtureError(format!("missing JSON path segment: {segment}")))?;
    }
    current.as_array().ok_or_else(|| {
        FixtureError(format!("JSON path is not an array: {}", path.join(".")))
    })
}

fn candidate_has_admissible_obligation_update(
    candidate: &Value,
    obligations: &BTreeMap<String, &Value>,
) -> Result<bool, FixtureError> {
    let eligible = string_at(candidate, &["authorization"])? == "authorized"
        && bool_at(candidate, &["reversible"])?
        && !bool_at(candidate, &["destructive"])?;
    if !eligible {
        return Ok(false);
    }

    for effect in array_at(candidate, &["effects"])? {
        let obligation_id = string_at(effect, &["obligation_id"])?;
        let Some(obligation) = obligations.get(obligation_id) else {
            return Err(FixtureError(format!(
                "effect references undeclared obligation {obligation_id}"
            )));
        };
        if !bool_at(obligation, &["required"])? {
            continue;
        }
        let evidence_class = string_at(effect, &["evidence_class"])?;
        let resulting_status = string_at(effect, &["resulting_status"])?;
        let initial_status = string_at(obligation, &["status"])?;
        let admissible = array_at(obligation, &["admissible_evidence_classes"])?
            .iter()
            .any(|item| item.as_str() == Some(evidence_class));
        if admissible && resulting_status != initial_status && resulting_status != "unknown" {
            return Ok(true);
        }
    }
    Ok(false)
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimReadiness {
    Ready,
    NotReady,
}

/// Evaluate readiness from current, independently assessed obligation states.
/// Predicted candidate outcomes are not inputs to this function.
pub fn evaluate_claim_readiness(case: &Value) -> Result<ClaimReadiness, FixtureError> {
    let obligations = array_at(case, &["obligations"])?;
    let mut has_mandatory_obligation = false;
    for obligation in obligations {
        if bool_at(obligation, &["required"])? {
            has_mandatory_obligation = true;
            if string_at(obligation, &["status"])? != "satisfied" {
                return Ok(ClaimReadiness::NotReady);
            }
        }
    }
    // Fail closed on vacuous proof: without a declared mandatory evidence
    // burden, the evaluator cannot promote the claim to Ready.
    if !has_mandatory_obligation {
        return Ok(ClaimReadiness::NotReady);
    }
    Ok(ClaimReadiness::Ready)
}


/// Apply only the effects associated with an already observed outcome.
/// This is a fixture reference simulator, not an acquisition planner or
/// production Evidence verifier.
#[cfg(test)]
fn apply_observed_fixture_outcome(
    agent_case: &Value,
    grader_case: &Value,
    experiment_id: &str,
    observed_outcome: &str,
) -> Result<Value, FixtureError> {
    let mut updated = agent_case.clone();
    let candidate = array_at(agent_case, &["candidates"])?
        .iter()
        .find(|candidate| {
            candidate.get("experiment_id").and_then(Value::as_str) == Some(experiment_id)
        })
        .ok_or_else(|| {
            FixtureError(format!("unknown agent-visible experiment {experiment_id}"))
        })?;

    let eligible = string_at(candidate, &["authorization"])? == "authorized"
        && bool_at(candidate, &["reversible"])?
        && !bool_at(candidate, &["destructive"])?;
    if !eligible {
        return Err(FixtureError(format!(
            "experiment {experiment_id} is not eligible for observed-outcome application"
        )));
    }

    if !array_at(candidate, &["outcomes"])?
        .iter()
        .any(|outcome| outcome.as_str() == Some(observed_outcome))
    {
        return Err(FixtureError(format!(
            "observed outcome {observed_outcome} was not declared for {experiment_id}"
        )));
    }

    let grader_candidate = array_at(grader_case, &["candidate_effects"])?
        .iter()
        .find(|entry| {
            entry.get("experiment_id").and_then(Value::as_str) == Some(experiment_id)
        })
        .ok_or_else(|| {
            FixtureError(format!("grader has no effects for experiment {experiment_id}"))
        })?;

    let matching_effects: Vec<&Value> = array_at(grader_candidate, &["effects"])?
        .iter()
        .filter(|effect| effect.get("outcome").and_then(Value::as_str) == Some(observed_outcome))
        .collect();

    for effect in matching_effects {
        let obligation_id = string_at(effect, &["obligation_id"])?;
        let evidence_class = string_at(effect, &["evidence_class"])?;
        let resulting_status = string_at(effect, &["resulting_status"])?;

        let obligation = array_at(&updated, &["obligations"])?
            .iter()
            .position(|item| {
                item.get("obligation_id").and_then(Value::as_str) == Some(obligation_id)
            })
            .ok_or_else(|| {
                FixtureError(format!("unknown obligation {obligation_id} in observed effect"))
            })?;

        let current = string_at(&updated["obligations"][obligation], &["status"])?;
        let admissible = array_at(
            &updated["obligations"][obligation],
            &["admissible_evidence_classes"],
        )?
        .iter()
        .any(|item| item.as_str() == Some(evidence_class));

        if !admissible {
            if resulting_status != current {
                return Err(FixtureError(format!(
                    "inadmissible observed evidence {evidence_class} cannot change {obligation_id}"
                )));
            }
            continue;
        }

        updated["obligations"][obligation]["status"] =
            Value::String(resulting_status.to_owned());
    }

    Ok(updated)
}

/// Check semantic relationships that JSON Schema alone cannot express:
/// unique identities, referential integrity, evidence admissibility, and
/// hard authorization/reversibility gates.
pub fn validate_fixture(value: &Value) -> Result<(), FixtureError> {
    if string_at(value, &["schema_version"])? != "0.1.0"
        || string_at(value, &["dataset_id"])? != "rea-proof-obligation-acquisition"
        || string_at(value, &["status"])? != "design_fixture_only"
        || string_at(value, &["source_policy"])? != "original_synthetic_text_only"
        || string_at(value, &["use"])? != "development_and_protocol_conformance"
        || string_at(value, &["oracle_policy"])? != "grader_only_not_for_agent_input"
        || value.get("not_human_adjudicated").and_then(Value::as_bool) != Some(true)
    {
        return Err(FixtureError(
            "fixture must preserve its design-only, synthetic, non-adjudicated and grader-only oracle declarations".into(),
        ));
    }

    let cases = array_at(value, &["cases"])?;
    if cases.len() != EXPECTED_CASES {
        return Err(FixtureError(format!(
            "expected {EXPECTED_CASES} cases, found {}",
            cases.len()
        )));
    }

    let mut case_ids = BTreeSet::new();
    for case in cases {
        let case_id = string_at(case, &["case_id"])?;
        if !case_ids.insert(case_id.to_owned()) {
            return Err(FixtureError(format!("duplicate case_id: {case_id}")));
        }

        let obligations = array_at(case, &["obligations"])?;
        if !obligations.iter().any(|obligation| {
            obligation
                .get("required")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        }) {
            return Err(FixtureError(format!(
                "case {case_id} must declare at least one mandatory proof obligation"
            )));
        }
        let mut obligation_by_id = BTreeMap::<String, &Value>::new();
        for obligation in obligations {
            let id = string_at(obligation, &["obligation_id"])?;
            if obligation_by_id.insert(id.to_owned(), obligation).is_some() {
                return Err(FixtureError(format!(
                    "duplicate obligation_id {id} in {case_id}"
                )));
            }
            let admissible = array_at(obligation, &["admissible_evidence_classes"])?;
            let mut classes = BTreeSet::new();
            for class in admissible {
                let name = class.as_str().ok_or_else(|| {
                    FixtureError(format!(
                        "admissible evidence class must be a string in {case_id}/{id}"
                    ))
                })?;
                if !classes.insert(name.to_owned()) {
                    return Err(FixtureError(format!(
                        "duplicate admissible evidence class {name} in {case_id}/{id}"
                    )));
                }
            }
        }

        let candidates = array_at(case, &["candidates"])?;
        let mut candidate_by_id = BTreeMap::<String, &Value>::new();
        for candidate in candidates {
            let experiment_id = string_at(candidate, &["experiment_id"])?;
            if candidate_by_id
                .insert(experiment_id.to_owned(), candidate)
                .is_some()
            {
                return Err(FixtureError(format!(
                    "duplicate experiment_id {experiment_id} in {case_id}"
                )));
            }

            let authorization = string_at(candidate, &["authorization"])?;
            let reversible = bool_at(candidate, &["reversible"])?;
            let destructive = bool_at(candidate, &["destructive"])?;
            let cost = candidate
                .get("cost_units")
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    FixtureError(format!(
                        "candidate cost_units must be numeric in {case_id}/{experiment_id}"
                    ))
                })?;
            if !cost.is_finite() || cost <= 0.0 {
                return Err(FixtureError(format!(
                    "candidate cost_units must be finite and positive in {case_id}/{experiment_id}"
                )));
            }

            let eligible = authorization == "authorized" && reversible && !destructive;
            let outcomes: BTreeSet<String> = array_at(candidate, &["outcomes"])?
                .iter()
                .map(|outcome| {
                    outcome.as_str().map(str::to_owned).ok_or_else(|| {
                        FixtureError(format!(
                            "outcome must be a string in {case_id}/{experiment_id}"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;

            for effect in array_at(candidate, &["effects"])? {
                let outcome = string_at(effect, &["outcome"])?;
                if !outcomes.contains(outcome) {
                    return Err(FixtureError(format!(
                        "effect references undeclared outcome {outcome} in {case_id}/{experiment_id}"
                    )));
                }

                let obligation_id = string_at(effect, &["obligation_id"])?;
                let obligation = obligation_by_id.get(obligation_id).ok_or_else(|| {
                    FixtureError(format!(
                        "effect references undeclared obligation {obligation_id} in {case_id}/{experiment_id}"
                    ))
                })?;
                let evidence_class = string_at(effect, &["evidence_class"])?;
                let resulting_status = string_at(effect, &["resulting_status"])?;

                let class_is_admissible = array_at(
                    obligation,
                    &["admissible_evidence_classes"],
                )?
                .iter()
                .any(|class| class.as_str() == Some(evidence_class));

                let initial_status = string_at(obligation, &["status"])?;

                // Inadmissible evidence cannot transition an obligation in
                // either direction; the state is unchanged.
                if !class_is_admissible
                    && resulting_status != initial_status
                {
                    return Err(FixtureError(format!(
                        "inadmissible evidence class {evidence_class} changes obligation {obligation_id} from {initial_status} to {resulting_status} in {case_id}/{experiment_id}"
                    )));
                }

                // An unauthorized, irreversible or destructive proposal may
                // be recorded as rejected, but may never satisfy an obligation.
                if !eligible && resulting_status != initial_status {
                    return Err(FixtureError(format!(
                        "ineligible experiment {experiment_id} attempts to change obligation {obligation_id} in {case_id}"
                    )));
                }

                if string_at(effect, &["source_family_id"])?.trim().is_empty() {
                    return Err(FixtureError(format!(
                        "effect source_family_id is empty in {case_id}/{experiment_id}"
                    )));
                }
            }
        }

        let mut any_candidate_can_update = false;
        for candidate in candidate_by_id.values() {
            any_candidate_can_update |=
                candidate_has_admissible_obligation_update(candidate, &obligation_by_id)?;
        }

        let preferred = string_at(case, &["expected_policy", "preferred_candidate_id"])?;
        if preferred == "abstain" {
            if any_candidate_can_update {
                return Err(FixtureError(format!(
                    "case {case_id} abstains although an admissible candidate can update a mandatory obligation"
                )));
            }
        } else {
            let candidate = candidate_by_id.get(preferred).ok_or_else(|| {
                FixtureError(format!(
                    "preferred candidate {preferred} is not declared in {case_id}"
                ))
            })?;
            let eligible = string_at(candidate, &["authorization"])? == "authorized"
                && bool_at(candidate, &["reversible"])?
                && !bool_at(candidate, &["destructive"])?;
            if !eligible {
                return Err(FixtureError(format!(
                    "preferred candidate {preferred} is not eligible in {case_id}"
                )));
            }
            if !candidate_has_admissible_obligation_update(candidate, &obligation_by_id)? {
                return Err(FixtureError(format!(
                    "preferred candidate {preferred} cannot update a mandatory obligation with admissible evidence in {case_id}"
                )));
            }
        }

        for blocking in array_at(case, &["expected_policy", "blocking_obligations"])? {
            let id = blocking.as_str().ok_or_else(|| {
                FixtureError(format!(
                    "blocking obligation must be a string in {case_id}"
                ))
            })?;
            if !obligation_by_id.contains_key(id) {
                return Err(FixtureError(format!(
                    "blocking obligation {id} is not declared in {case_id}"
                )));
            }
        }
    }

    Ok(())
}


/// Validate that the public development input and grader-only labels are
/// separated, mutually consistent, and free of hidden-outcome leakage.
pub fn validate_split_corpus() -> Result<(), FixtureError> {
    let agent: Value = serde_json::from_str(AGENT_VIEW)
        .map_err(|error| FixtureError(format!("agent-view fixture parse failed: {error}")))?;
    let grader: Value = serde_json::from_str(GRADER_VIEW)
        .map_err(|error| FixtureError(format!("grader fixture parse failed: {error}")))?;
    validate_split_values(&agent, &grader)
}

fn validate_split_values(agent: &Value, grader: &Value) -> Result<(), FixtureError> {
    let agent_schema: Value = serde_json::from_str(AGENT_SCHEMA)
        .map_err(|error| FixtureError(format!("agent schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&agent_schema).map_err(|error| {
        FixtureError(format!("agent Draft 2020-12 schema is invalid: {error}"))
    })?;
    let grader_schema: Value = serde_json::from_str(GRADER_SCHEMA)
        .map_err(|error| FixtureError(format!("grader schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&grader_schema).map_err(|error| {
        FixtureError(format!("grader Draft 2020-12 schema is invalid: {error}"))
    })?;

    for (label, schema, instance) in [
        ("agent-view", &agent_schema, agent),
        ("grader", &grader_schema, grader),
    ] {
        let validator = draft202012::options()
            .build(schema)
            .map_err(|error| FixtureError(format!("{label} schema compile failed: {error}")))?;
        let errors: Vec<String> = validator
            .iter_errors(instance)
            .map(|error| error.to_string())
            .collect();
        if !errors.is_empty() {
            return Err(FixtureError(format!(
                "{label} fixture failed JSON Schema validation: {}",
                errors.join("; ")
            )));
        }
    }

    if string_at(agent, &["use"])? != "agent_visible_development_input_only"
        || string_at(grader, &["access_policy"])? != "never_mount_in_agent_workspace"
        || !bool_at(agent, &["not_human_adjudicated"])?
        || !bool_at(grader, &["not_human_adjudicated"])?
    {
        return Err(FixtureError(
            "agent/grader views must remain development-only, non-adjudicated, and access-separated".into(),
        ));
    }

    let agent_cases = array_at(agent, &["cases"])?;
    let grader_cases = array_at(grader, &["cases"])?;
    if agent_cases.len() != EXPECTED_CASES || grader_cases.len() != EXPECTED_CASES {
        return Err(FixtureError(format!(
            "agent and grader views must each contain {EXPECTED_CASES} cases"
        )));
    }

    let mut agent_by_id = BTreeMap::<String, &Value>::new();
    for case in agent_cases {
        let id = string_at(case, &["case_id"])?;
        if agent_by_id.insert(id.to_owned(), case).is_some() {
            return Err(FixtureError(format!("duplicate agent-view case ID: {id}")));
        }
        if case.get("world_state_oracle").is_some() || case.get("expected_policy").is_some() {
            return Err(FixtureError(format!(
                "agent-visible case {id} leaks grader-only case labels"
            )));
        }
        for candidate in array_at(case, &["candidates"])? {
            if candidate.get("effects").is_some() {
                return Err(FixtureError(format!(
                    "agent-visible candidate in {id} leaks grader-only outcome effects"
                )));
            }
        }
    }

    let mut grader_ids = BTreeSet::new();
    for grader_case in grader_cases {
        let id = string_at(grader_case, &["case_id"])?;
        if !grader_ids.insert(id.to_owned()) {
            return Err(FixtureError(format!("duplicate grader case ID: {id}")));
        }
        let agent_case = agent_by_id.get(id).ok_or_else(|| {
            FixtureError(format!("grader case {id} has no corresponding agent case"))
        })?;

        let agent_obligations = array_at(agent_case, &["obligations"])?;
        let mut obligation_by_id = BTreeMap::<String, &Value>::new();
        for obligation in agent_obligations {
            obligation_by_id.insert(
                string_at(obligation, &["obligation_id"])?.to_owned(),
                obligation,
            );
        }

        let agent_candidates = array_at(agent_case, &["candidates"])?;
        let mut candidate_by_id = BTreeMap::<String, &Value>::new();
        for candidate in agent_candidates {
            candidate_by_id.insert(
                string_at(candidate, &["experiment_id"])?.to_owned(),
                candidate,
            );
        }

        let grader_effects = array_at(grader_case, &["candidate_effects"])?;
        let mut effect_candidate_ids = BTreeSet::new();
        for effect_candidate in grader_effects {
            let experiment_id = string_at(effect_candidate, &["experiment_id"])?;
            if !effect_candidate_ids.insert(experiment_id.to_owned()) {
                return Err(FixtureError(format!(
                    "duplicate grader candidate effects for {id}/{experiment_id}"
                )));
            }
            let agent_candidate = candidate_by_id.get(experiment_id).ok_or_else(|| {
                FixtureError(format!(
                    "grader effects reference undeclared agent candidate {id}/{experiment_id}"
                ))
            })?;
            let outcomes: BTreeSet<String> = array_at(agent_candidate, &["outcomes"])?
                .iter()
                .map(|value| {
                    value.as_str().map(str::to_owned).ok_or_else(|| {
                        FixtureError(format!("non-string candidate outcome in {id}/{experiment_id}"))
                    })
                })
                .collect::<Result<_, _>>()?;

            for effect in array_at(effect_candidate, &["effects"])? {
                let outcome = string_at(effect, &["outcome"])?;
                if !outcomes.contains(outcome) {
                    return Err(FixtureError(format!(
                        "grader effect outcome {outcome} is not in the agent-visible outcome set for {id}/{experiment_id}"
                    )));
                }
                let obligation_id = string_at(effect, &["obligation_id"])?;
                let obligation = obligation_by_id.get(obligation_id).ok_or_else(|| {
                    FixtureError(format!(
                        "grader effect references missing agent obligation {id}/{obligation_id}"
                    ))
                })?;
                let evidence_class = string_at(effect, &["evidence_class"])?;
                let resulting_status = string_at(effect, &["resulting_status"])?;
                let initial_status = string_at(obligation, &["status"])?;
                let admissible = array_at(
                    obligation,
                    &["admissible_evidence_classes"],
                )?
                .iter()
                .any(|item| item.as_str() == Some(evidence_class));

                if !admissible
                    && resulting_status != initial_status
                {
                    return Err(FixtureError(format!(
                        "inadmissible grader evidence {evidence_class} upgrades {id}/{obligation_id} from {initial_status} to {resulting_status}"
                    )));
                }
            }
        }
        if effect_candidate_ids != candidate_by_id.keys().cloned().collect() {
            return Err(FixtureError(format!(
                "agent/grader candidate sets differ for case {id}"
            )));
        }

        let preferred = string_at(grader_case, &["expected_policy", "preferred_candidate_id"])?;
        if preferred != "abstain" && !candidate_by_id.contains_key(preferred) {
            return Err(FixtureError(format!(
                "grader preferred candidate {preferred} is absent from agent case {id}"
            )));
        }
        for blocking in array_at(grader_case, &["expected_policy", "blocking_obligations"])? {
            let obligation_id = blocking.as_str().ok_or_else(|| {
                FixtureError(format!("grader blocking obligation is not a string in {id}"))
            })?;
            if !obligation_by_id.contains_key(obligation_id) {
                return Err(FixtureError(format!(
                    "grader blocking obligation {id}/{obligation_id} is absent from agent case"
                )));
            }
        }
    }

    if agent_by_id.keys().cloned().collect::<BTreeSet<_>>() != grader_ids {
        return Err(FixtureError("agent and grader case ID sets differ".into()));
    }
    Ok(())
}


fn validate_split_matches_authoring(
    authoring: &Value,
    agent: &Value,
    grader: &Value,
) -> Result<(), FixtureError> {
    let authoring_cases = array_at(authoring, &["cases"])?;
    let agent_cases = array_at(agent, &["cases"])?;
    let grader_cases = array_at(grader, &["cases"])?;

    let agent_by_id: BTreeMap<&str, &Value> = agent_cases
        .iter()
        .map(|case| Ok((string_at(case, &["case_id"])?, case)))
        .collect::<Result<_, FixtureError>>()?;
    let grader_by_id: BTreeMap<&str, &Value> = grader_cases
        .iter()
        .map(|case| Ok((string_at(case, &["case_id"])?, case)))
        .collect::<Result<_, FixtureError>>()?;

    for full in authoring_cases {
        let id = string_at(full, &["case_id"])?;
        let agent_case = agent_by_id.get(id).ok_or_else(|| {
            FixtureError(format!("authoring case {id} is missing from agent view"))
        })?;
        let grader_case = grader_by_id.get(id).ok_or_else(|| {
            FixtureError(format!("authoring case {id} is missing from grader view"))
        })?;

        let candidates = array_at(full, &["candidates"])?;
        let expected_agent_candidates: Vec<Value> = candidates
            .iter()
            .map(|candidate| {
                serde_json::json!({
                    "experiment_id": candidate["experiment_id"],
                    "authorization": candidate["authorization"],
                    "reversible": candidate["reversible"],
                    "destructive": candidate["destructive"],
                    "target_impact": candidate["target_impact"],
                    "cost_units": candidate["cost_units"],
                    "description": candidate["description"],
                    "outcomes": candidate["outcomes"]
                })
            })
            .collect();
        let expected_agent_case = serde_json::json!({
            "case_id": full["case_id"],
            "domain": full["domain"],
            "fault_family": full["fault_family"],
            "claim_text": full["claim_text"],
            "obligations": full["obligations"],
            "candidates": expected_agent_candidates
        });
        if **agent_case != expected_agent_case {
            return Err(FixtureError(format!(
                "agent-view projection differs from authoring case {id}"
            )));
        }

        let expected_candidate_effects: Vec<Value> = candidates
            .iter()
            .map(|candidate| {
                serde_json::json!({
                    "experiment_id": candidate["experiment_id"],
                    "effects": candidate["effects"]
                })
            })
            .collect();
        let expected_grader_case = serde_json::json!({
            "case_id": full["case_id"],
            "world_state_oracle": full["world_state_oracle"],
            "expected_policy": full["expected_policy"],
            "candidate_effects": expected_candidate_effects
        });
        if **grader_case != expected_grader_case {
            return Err(FixtureError(format!(
                "grader-view projection differs from authoring case {id}"
            )));
        }
    }
    Ok(())
}

/// Validate the embedded Draft 2020-12 schema, its instance, and semantic
/// cross-field invariants.
pub fn validate_embedded_corpus() -> Result<(), FixtureError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| FixtureError(format!("schema JSON parse failed: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| FixtureError(format!("Draft 2020-12 schema is invalid: {error}")))?;

    let fixture: Value = serde_json::from_str(FIXTURE)
        .map_err(|error| FixtureError(format!("fixture JSON parse failed: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| FixtureError(format!("schema compilation failed: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&fixture)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(FixtureError(format!(
            "fixture failed JSON Schema validation: {}",
            errors.join("; ")
        )));
    }

    validate_fixture(&fixture)?;
    validate_split_corpus()?;

    let agent: Value = serde_json::from_str(AGENT_VIEW)
        .map_err(|error| FixtureError(format!("agent-view fixture parse failed: {error}")))?;
    let grader: Value = serde_json::from_str(GRADER_VIEW)
        .map_err(|error| FixtureError(format!("grader fixture parse failed: {error}")))?;
    validate_split_matches_authoring(&fixture, &agent, &grader)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("checked-in fixture must parse")
    }

    #[test]
    fn draft_2020_12_schema_is_meta_valid_and_fixture_conforms() {
        validate_embedded_corpus().unwrap();
    }

    #[test]
    fn duplicate_case_id_is_rejected() {
        let mut value = fixture();
        let duplicate = value["cases"][0].clone();
        value["cases"].as_array_mut().unwrap().push(duplicate);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn dangling_effect_obligation_is_rejected() {
        let mut value = fixture();
        value["cases"][0]["candidates"][0]["effects"][0]["obligation_id"] =
            Value::String("missing_obligation".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn undeclared_candidate_outcome_is_rejected() {
        let mut value = fixture();
        value["cases"][0]["candidates"][0]["effects"][0]["outcome"] =
            Value::String("future_outcome_not_declared".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn inadmissible_evidence_cannot_upgrade_an_obligation() {
        let mut value = fixture();
        let case = value["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "REA-POA-008")
            .unwrap();
        let effect = &mut case["candidates"][0]["effects"][0];
        effect["resulting_status"] = Value::String("satisfied".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn unauthorized_experiment_cannot_satisfy_an_obligation() {
        let mut value = fixture();
        let case = value["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "REA-POA-013")
            .unwrap();
        case["candidates"][0]["effects"] = serde_json::json!([{
            "outcome":"recovered",
            "obligation_id":"fault_reproduction",
            "resulting_status":"satisfied",
            "evidence_class":"authorized_controlled_reproduction",
            "source_family_id":"synthetic:unauthorized-outcome"
        }]);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn hidden_oracle_policy_cannot_be_downgraded() {
        let mut value = fixture();
        value["oracle_policy"] = Value::String("agent_visible".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn preferred_candidate_must_exist_and_be_eligible() {
        let mut value = fixture();
        value["cases"][0]["expected_policy"]["preferred_candidate_id"] =
            Value::String("missing_candidate".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn blocking_obligation_must_exist() {
        let mut value = fixture();
        value["cases"][0]["expected_policy"]["blocking_obligations"][0] =
            Value::String("missing_obligation".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn zero_cost_candidate_is_rejected() {
        let mut value = fixture();
        value["cases"][0]["candidates"][0]["cost_units"] = serde_json::json!(0);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn non_adjudicated_status_cannot_be_removed() {
        let mut value = fixture();
        value["not_human_adjudicated"] = Value::Bool(false);
        assert!(validate_fixture(&value).is_err());
    }
    #[test]
    fn agent_and_grader_views_are_separated_and_consistent() {
        validate_split_corpus().unwrap();
    }

    #[test]
    fn agent_view_cannot_contain_world_state_oracle_labels() {
        let mut agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        agent["cases"][0]["world_state_oracle"] = Value::String("vulnerable".to_owned());
        assert!(validate_split_values(&agent, &grader).is_err());
    }

    #[test]
    fn agent_view_cannot_contain_candidate_effects() {
        let mut agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        agent["cases"][0]["candidates"][0]["effects"] =
            serde_json::json!([{"outcome":"leaked","obligation_id":"artifact_identity"}]);
        assert!(validate_split_values(&agent, &grader).is_err());
    }

    #[test]
    fn grader_cannot_upgrade_obligation_with_inadmissible_evidence() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let mut grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let case = grader["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "REA-POA-008")
            .unwrap();
        case["candidate_effects"][0]["effects"][0]["resulting_status"] =
            Value::String("satisfied".to_owned());
        assert!(validate_split_values(&agent, &grader).is_err());
    }

    #[test]
    fn agent_and_grader_case_sets_must_match() {
        let mut grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        grader["cases"][0]["case_id"] = Value::String("REA-POA-999".to_owned());
        assert!(validate_split_values(&serde_json::from_str(AGENT_VIEW).unwrap(), &grader).is_err());
    }


    #[test]
    fn split_views_must_be_exact_projections_of_authoring_fixture() {
        let authoring: Value = serde_json::from_str(FIXTURE).unwrap();
        let mut agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        agent["cases"][0]["claim_text"] = Value::String("silently changed claim".to_owned());
        assert!(validate_split_matches_authoring(&authoring, &agent, &grader).is_err());
    }


    #[test]
    fn preferred_candidate_must_have_an_admissible_mandatory_obligation_update() {
        let mut value = fixture();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|case| case["case_id"] == "REA-POA-001").unwrap();
        case["candidates"][1]["effects"] = serde_json::json!([]);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn abstention_is_rejected_when_an_admissible_candidate_can_update_an_obligation() {
        let mut value = fixture();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|case| case["case_id"] == "REA-POA-012").unwrap();
        case["candidates"][0]["effects"] = serde_json::json!([{
            "outcome":"no_network_event",
            "obligation_id":"complete_network_observation",
            "resulting_status":"satisfied",
            "evidence_class":"instrumented_full_session",
            "source_family_id":"synthetic:session-control-012"
        }]);
        assert!(validate_fixture(&value).is_err());
    }


    #[test]
    fn readiness_without_mandatory_obligations_fails_closed() {
        let mut case = fixture()["cases"][0].clone();
        for obligation in case["obligations"].as_array_mut().unwrap() {
            obligation["required"] = Value::Bool(false);
            obligation["status"] = Value::String("satisfied".to_owned());
        }
        assert_eq!(
            evaluate_claim_readiness(&case).unwrap(),
            ClaimReadiness::NotReady
        );

        case["obligations"] = serde_json::json!([]);
        assert_eq!(
            evaluate_claim_readiness(&case).unwrap(),
            ClaimReadiness::NotReady
        );
    }

    #[test]
    fn fixture_case_without_mandatory_obligation_is_rejected() {
        let mut value = fixture();
        for obligation in value["cases"][0]["obligations"].as_array_mut().unwrap() {
            obligation["required"] = Value::Bool(false);
        }
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn readiness_requires_every_mandatory_obligation_to_be_satisfied() {
        let mut case = fixture()["cases"][0].clone();
        assert_eq!(evaluate_claim_readiness(&case).unwrap(), ClaimReadiness::NotReady);
        case["obligations"][0]["status"] = Value::String("satisfied".to_owned());
        case["obligations"][1]["status"] = Value::String("satisfied".to_owned());
        case["obligations"][2]["status"] = Value::String("satisfied".to_owned());
        assert_eq!(evaluate_claim_readiness(&case).unwrap(), ClaimReadiness::Ready);
    }

    #[test]
    fn optional_unsatisfied_obligation_does_not_block_readiness() {
        let mut case = fixture()["cases"][0].clone();
        case["obligations"][0]["status"] = Value::String("satisfied".to_owned());
        case["obligations"][1]["status"] = Value::String("satisfied".to_owned());
        case["obligations"][2]["required"] = Value::Bool(false);
        case["obligations"][2]["status"] = Value::String("unknown".to_owned());
        assert_eq!(evaluate_claim_readiness(&case).unwrap(), ClaimReadiness::Ready);
    }

    #[test]
    fn every_non_satisfied_mandatory_status_blocks_readiness() {
        for status in ["unsatisfied", "conflicted", "unknown", "incomplete", "contradicted"] {
            let mut case = fixture()["cases"][0].clone();
            case["obligations"][0]["status"] = Value::String(status.to_owned());
            assert_eq!(
                evaluate_claim_readiness(&case).unwrap(),
                ClaimReadiness::NotReady,
                "mandatory status {status} must block readiness"
            );
        }
    }

    #[test]
    fn inadmissible_evidence_cannot_downgrade_or_upgrade_obligation_state() {
        let mut value = fixture();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|case| case["case_id"] == "REA-POA-001").unwrap();
        case["obligations"][2]["status"] = Value::String("satisfied".to_owned());
        case["candidates"][0]["effects"][0]["resulting_status"] =
            Value::String("unknown".to_owned());
        assert!(validate_fixture(&value).is_err());
    }


    #[test]
    fn ineligible_candidates_cannot_change_any_obligation_state() {
        let mut value = fixture();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|case| case["case_id"] == "REA-POA-013").unwrap();
        case["obligations"][0]["status"] = Value::String("satisfied".to_owned());
        case["candidates"][0]["effects"] = serde_json::json!([{
            "outcome":"recovered",
            "obligation_id":"fault_reproduction",
            "resulting_status":"unknown",
            "evidence_class":"authorized_controlled_reproduction",
            "source_family_id":"synthetic:unauthorized-observation"
        }]);
        assert!(validate_fixture(&value).is_err());
    }


    #[test]
    fn candidate_prediction_does_not_change_readiness_before_observation() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let case = &agent["cases"][0];
        assert_eq!(evaluate_claim_readiness(case).unwrap(), ClaimReadiness::NotReady);
        assert_eq!(
            string_at(&case["obligations"][1], &["status"]).unwrap(),
            "unknown",
            "candidate prediction is not an observation"
        );
    }

    #[test]
    fn high_information_static_probe_does_not_close_deployment_or_behavior_obligations() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let agent_case = &agent["cases"][0];
        let grader_case = &grader["cases"][0];

        let after = apply_observed_fixture_outcome(
            agent_case,
            grader_case,
            "probe_many_functions",
            "function_a",
        )
        .unwrap();
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::NotReady);
        assert_eq!(
            string_at(&after["obligations"][2], &["status"]).unwrap(),
            "unknown"
        );
    }

    #[test]
    fn exact_manifest_observation_closes_only_the_binding_obligation() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let after = apply_observed_fixture_outcome(
            &agent["cases"][0],
            &grader["cases"][0],
            "verify_deployment_manifest",
            "digest_match",
        )
        .unwrap();
        assert_eq!(after["obligations"][1]["status"], "satisfied");
        assert_eq!(after["obligations"][2]["status"], "unknown");
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::NotReady);
    }

    #[test]
    fn complete_page_reconciliation_closes_enumeration_and_acquisition() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let after = apply_observed_fixture_outcome(
            &agent["cases"][3],
            &grader["cases"][3],
            "reacquire_missing_page",
            "page_reconciled",
        )
        .unwrap();
        assert_eq!(after["obligations"][1]["status"], "satisfied");
        assert_eq!(after["obligations"][2]["status"], "satisfied");
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::Ready);
    }

    #[test]
    fn averaging_conflicted_scanner_scores_does_not_clear_the_conflict() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let after = apply_observed_fixture_outcome(
            &agent["cases"][8],
            &grader["cases"][8],
            "average_scanner_scores",
            "high_average",
        )
        .unwrap();
        assert_eq!(after["obligations"][0]["status"], "conflicted");
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::NotReady);
    }

    #[test]
    fn matched_configuration_reproduction_resolves_both_required_obligations() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let after = apply_observed_fixture_outcome(
            &agent["cases"][8],
            &grader["cases"][8],
            "reproduce_production_configuration",
            "pass",
        )
        .unwrap();
        assert_eq!(after["obligations"][0]["status"], "satisfied");
        assert_eq!(after["obligations"][1]["status"], "satisfied");
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::Ready);
    }

    #[test]
    fn bounded_no_event_observation_does_not_prove_universal_absence() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        let after = apply_observed_fixture_outcome(
            &agent["cases"][11],
            &grader["cases"][11],
            "run_one_short_session",
            "no_network_event",
        )
        .unwrap();
        assert_eq!(after["obligations"][0]["status"], "unknown");
        assert_eq!(evaluate_claim_readiness(&after).unwrap(), ClaimReadiness::NotReady);
    }

    #[test]
    fn irreversible_or_unauthorized_proposals_cannot_be_applied_as_observed_actions() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        assert!(apply_observed_fixture_outcome(
            &agent["cases"][12],
            &grader["cases"][12],
            "unauthorized_production_probe",
            "recovered"
        ).is_err());
        assert!(apply_observed_fixture_outcome(
            &agent["cases"][13],
            &grader["cases"][13],
            "run_irreversible_production_migration",
            "success"
        ).is_err());
    }

    #[test]
    fn undeclared_observed_outcome_is_rejected() {
        let agent: Value = serde_json::from_str(AGENT_VIEW).unwrap();
        let grader: Value = serde_json::from_str(GRADER_VIEW).unwrap();
        assert!(apply_observed_fixture_outcome(
            &agent["cases"][0],
            &grader["cases"][0],
            "verify_deployment_manifest",
            "invented_digest_match"
        ).is_err());
    }

}
