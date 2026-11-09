//! Deterministic checker for the REA anytime-evidence development corpus.
//!
//! It validates the agent/grader split, checks the finite transition ledger,
//! and computes readiness from current obligation/duty states. It does not
//! authenticate Evidence, execute experiments, or adjudicate real-world truth.

use jsonschema::draft202012;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const AGENT_SCHEMA: &str = include_str!(
    "../../../docs/research/rea-anytime-evidence-agent-view-v0.schema.json"
);
const AGENT_VIEW: &str =
    include_str!("../../../docs/research/rea-anytime-evidence-agent-view-v0.json");
const GRADER_SCHEMA: &str =
    include_str!("../../../docs/research/rea-anytime-evidence-grader-v0.schema.json");
const GRADER_VIEW: &str =
    include_str!("../../../docs/research/rea-anytime-evidence-grader-v0.json");
const EPISODE_COUNT: usize = 14;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckError(pub String);

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CheckError {}

type Result<T> = std::result::Result<T, CheckError>;

fn fail(message: impl Into<String>) -> CheckError {
    CheckError(message.into())
}

fn get<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value> {
    let mut current = value;
    for part in path {
        current = current
            .get(*part)
            .ok_or_else(|| fail(format!("missing JSON path {}", path.join("."))))?;
    }
    Ok(current)
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str> {
    get(value, path)?
        .as_str()
        .ok_or_else(|| fail(format!("expected string at {}", path.join("."))))
}

fn bool_at(value: &Value, path: &[&str]) -> Result<bool> {
    get(value, path)?
        .as_bool()
        .ok_or_else(|| fail(format!("expected boolean at {}", path.join("."))))
}

fn array_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Vec<Value>> {
    get(value, path)?
        .as_array()
        .ok_or_else(|| fail(format!("expected array at {}", path.join("."))))
}

fn validate_value_against_schema(schema_text: &str, instance: &Value, label: &str) -> Result<()> {
    let schema: Value = serde_json::from_str(schema_text)
        .map_err(|error| fail(format!("{label} schema JSON parse: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| fail(format!("{label} Draft 2020-12 meta-schema: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| fail(format!("{label} schema compile: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(fail(format!(
            "{label} instance does not conform: {}",
            errors.join("; ")
        )));
    }
    Ok(())
}

fn validate_schema_instance(schema_text: &str, instance_text: &str, label: &str) -> Result<Value> {
    let schema: Value = serde_json::from_str(schema_text)
        .map_err(|error| fail(format!("{label} schema JSON parse: {error}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|error| fail(format!("{label} Draft 2020-12 meta-schema: {error}")))?;
    let instance: Value = serde_json::from_str(instance_text)
        .map_err(|error| fail(format!("{label} instance JSON parse: {error}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|error| fail(format!("{label} schema compile: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&instance)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(fail(format!(
            "{label} instance does not conform: {}",
            errors.join("; ")
        )));
    }
    Ok(instance)
}

/// Current readiness is computed only from independently assessed states.
/// Candidate descriptions and predicted outcomes are intentionally ignored.
pub fn claim_is_ready(episode: &Value, claim_id: &str) -> Result<bool> {
    let claim = array_at(episode, &["claim_ladder"])?
        .iter()
        .find(|claim| claim.get("claim_id").and_then(Value::as_str) == Some(claim_id))
        .ok_or_else(|| fail(format!("unknown claim_id {claim_id}")))?;

    let required_obligations = array_at(claim, &["required_obligation_ids"])?;
    let required_duties = array_at(claim, &["required_defeater_duty_ids"])?;
    if required_obligations.is_empty() {
        return Ok(false);
    }

    for required in required_obligations {
        let id = required
            .as_str()
            .ok_or_else(|| fail("claim obligation reference must be a string"))?;
        let state = array_at(episode, &["initial_state", "obligations"])?
            .iter()
            .find(|item| item.get("obligation_id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| fail(format!("missing obligation {id}")))?;
        if string_at(state, &["status"])? != "satisfied" {
            return Ok(false);
        }
    }

    for required in required_duties {
        let id = required
            .as_str()
            .ok_or_else(|| fail("claim defeater duty reference must be a string"))?;
        let state = array_at(episode, &["initial_state", "defeater_duties"])?
            .iter()
            .find(|item| item.get("duty_id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| fail(format!("missing defeater duty {id}")))?;
        if string_at(state, &["status"])? != "satisfied" {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Return the strongest declared claim currently ready, using strength_rank.
/// Ties are rejected by corpus validation.
pub fn strongest_ready_claim(episode: &Value) -> Result<Option<String>> {
    let mut ready: Vec<(u64, String)> = Vec::new();
    for claim in array_at(episode, &["claim_ladder"])? {
        let id = string_at(claim, &["claim_id"])?;
        if claim_is_ready(episode, id)? {
            let rank = get(claim, &["strength_rank"])?
                .as_u64()
                .ok_or_else(|| fail(format!("invalid strength_rank for {id}")))?;
            ready.push((rank, id.to_owned()));
        }
    }
    ready.sort_by_key(|(rank, _)| *rank);
    Ok(ready.pop().map(|(_, id)| id))
}

/// Apply one grader event to a copied agent state. Used only by the finite
/// reference tests; real policy input never contains the grader event stream.
fn apply_event(agent_episode: &Value, grader_episode: &Value, index: usize) -> Result<Value> {
    let event = array_at(grader_episode, &["events"])?
        .get(index)
        .ok_or_else(|| fail(format!("missing grader event index {index}")))?;
    let mut updated = agent_episode.clone();

    for effect in array_at(event, &["obligation_updates"])? {
        let id = string_at(effect, &["obligation_id"])?;
        let class = string_at(effect, &["evidence_class"])?;
        let status = string_at(effect, &["resulting_status"])?;
        let target = updated["initial_state"]["obligations"]
            .as_array_mut()
            .ok_or_else(|| fail("agent obligations are not an array"))?
            .iter_mut()
            .find(|item| item.get("obligation_id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| fail(format!("grader updates undeclared obligation {id}")))?;
        let admissible = array_at(target, &["admissible_evidence_classes"])?
            .iter()
            .any(|item| item.as_str() == Some(class));
        if !admissible {
            return Err(fail(format!(
                "grader event uses inadmissible evidence class {class} for {id}"
            )));
        }
        target["status"] = Value::String(status.to_owned());
    }

    for effect in array_at(event, &["duty_updates"])? {
        let id = string_at(effect, &["duty_id"])?;
        let class = string_at(effect, &["evidence_class"])?;
        let status = string_at(effect, &["resulting_status"])?;
        let target = updated["initial_state"]["defeater_duties"]
            .as_array_mut()
            .ok_or_else(|| fail("agent defeater duties are not an array"))?
            .iter_mut()
            .find(|item| item.get("duty_id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| fail(format!("grader updates undeclared duty {id}")))?;
        let admissible = array_at(target, &["admissible_evidence_classes"])?
            .iter()
            .any(|item| item.as_str() == Some(class));
        if !admissible {
            return Err(fail(format!(
                "grader event uses inadmissible evidence class {class} for duty {id}"
            )));
        }
        target["status"] = Value::String(status.to_owned());
    }

    Ok(updated)
}

fn candidate_map<'a>(episode: &'a Value) -> Result<BTreeMap<String, &'a Value>> {
    let mut candidates = BTreeMap::new();
    for candidate in array_at(episode, &["candidate_actions"])? {
        let id = string_at(candidate, &["action_id"])?;
        if candidates.insert(id.to_owned(), candidate).is_some() {
            return Err(fail(format!("duplicate action_id {id}")));
        }
    }
    Ok(candidates)
}

fn is_action_eligible(action: &Value, budget: &Value) -> Result<bool> {
    let cost = get(action, &["cost_units"])?
        .as_f64()
        .ok_or_else(|| fail("action cost_units must be numeric"))?;
    let maximum = get(budget, &["max_cost_units"])?
        .as_f64()
        .ok_or_else(|| fail("max_cost_units must be numeric"))?;
    Ok(string_at(action, &["authorization"])? == "authorized"
        && bool_at(action, &["reversible"])?
        && !bool_at(action, &["destructive"])?
        && cost.is_finite()
        && cost > 0.0
        && cost <= maximum)
}

fn validate_claim_ladder(episode: &Value) -> Result<()> {
    let claims = array_at(episode, &["claim_ladder"])?;
    let obligations: BTreeSet<String> = array_at(episode, &["initial_state", "obligations"])?
        .iter()
        .map(|item| string_at(item, &["obligation_id"]).map(str::to_owned))
        .collect::<Result<_>>()?;
    let duties: BTreeSet<String> = array_at(episode, &["initial_state", "defeater_duties"])?
        .iter()
        .map(|item| string_at(item, &["duty_id"]).map(str::to_owned))
        .collect::<Result<_>>()?;

    let mut prior_rank = 0_u64;
    let mut prior_obligations = BTreeSet::new();
    let mut prior_duties = BTreeSet::new();
    let mut claim_ids = BTreeSet::new();

    for claim in claims {
        let id = string_at(claim, &["claim_id"])?;
        if !claim_ids.insert(id.to_owned()) {
            return Err(fail(format!("duplicate claim_id {id}")));
        }
        let rank = get(claim, &["strength_rank"])?
            .as_u64()
            .ok_or_else(|| fail(format!("invalid strength_rank in {id}")))?;
        if rank <= prior_rank {
            return Err(fail("claim strength_rank must be unique and strictly increasing"));
        }

        let current_obligations: BTreeSet<String> = array_at(claim, &["required_obligation_ids"])?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| fail("claim obligation ID must be a string"))
            })
            .collect::<Result<_>>()?;
        let current_duties: BTreeSet<String> = array_at(claim, &["required_defeater_duty_ids"])?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| fail("claim duty ID must be a string"))
            })
            .collect::<Result<_>>()?;

        if current_obligations.is_empty()
            || !current_obligations.is_subset(&obligations)
            || !current_duties.is_subset(&duties)
        {
            return Err(fail(format!("invalid proof burden in claim {id}")));
        }
        if !prior_obligations.is_subset(&current_obligations)
            || !prior_duties.is_subset(&current_duties)
        {
            return Err(fail(format!(
                "stronger claim {id} weakens a lower claim's declared proof burden"
            )));
        }

        prior_rank = rank;
        prior_obligations = current_obligations;
        prior_duties = current_duties;
    }
    Ok(())
}

/// Cross-check agent-visible actions against grader-only transitions and
/// verify that every expected stop decision matches the finite state machine.
pub fn validate_cross_corpus(agent: &Value, grader: &Value) -> Result<()> {
    // Revalidate caller-supplied values as well as the embedded corpus. This
    // keeps mutation tests honest: cross-view checks must reject hidden fields
    // even when invoked without validate_embedded_corpus().
    validate_value_against_schema(AGENT_SCHEMA, agent, "agent view")?;
    validate_value_against_schema(GRADER_SCHEMA, grader, "grader view")?;

    if string_at(agent, &["use"])? != "agent_visible_development_input_only"
        || string_at(grader, &["access_policy"])? != "never_mount_in_agent_workspace"
        || !bool_at(agent, &["not_human_adjudicated"])?
        || !bool_at(grader, &["not_human_adjudicated"])?
    {
        return Err(fail("agent/grader separation or non-adjudicated status was changed"));
    }

    let agent_episodes = array_at(agent, &["episodes"])?;
    let grader_episodes = array_at(grader, &["episodes"])?;
    if agent_episodes.len() != EPISODE_COUNT || grader_episodes.len() != EPISODE_COUNT {
        return Err(fail("unexpected agent/grader episode count"));
    }

    let mut agent_by_id = BTreeMap::new();
    for episode in agent_episodes {
        let id = string_at(episode, &["episode_id"])?;
        if agent_by_id.insert(id.to_owned(), episode).is_some() {
            return Err(fail(format!("duplicate agent episode {id}")));
        }
        validate_claim_ladder(episode)?;
    }

    let mut grader_ids = BTreeSet::new();
    for grader_episode in grader_episodes {
        let id = string_at(grader_episode, &["episode_id"])?;
        if !grader_ids.insert(id.to_owned()) {
            return Err(fail(format!("duplicate grader episode {id}")));
        }
        let agent_episode = agent_by_id
            .get(id)
            .ok_or_else(|| fail(format!("grader episode {id} has no agent view")))?;
        let candidates = candidate_map(agent_episode)?;
        let budget = get(agent_episode, &["budget"])?;
        let mut action_count = 0_u64;
        let mut action_cost = 0.0_f64;

        for event in array_at(grader_episode, &["events"])? {
            match string_at(event, &["trigger_kind"])? {
                "observed_action" => {
                    let action_id = string_at(event, &["action_id"])?;
                    let action = candidates
                        .get(action_id)
                        .ok_or_else(|| fail(format!("event uses undeclared action {action_id}")))?;
                    let outcome = string_at(event, &["outcome"])?;
                    if !array_at(action, &["outcomes"])?
                        .iter()
                        .any(|item| item.as_str() == Some(outcome))
                    {
                        return Err(fail(format!(
                            "event outcome {outcome} is not declared by action {action_id}"
                        )));
                    }
                    if !is_action_eligible(action, budget)? {
                        return Err(fail(format!(
                            "grader executes ineligible or over-budget action {action_id}"
                        )));
                    }
                    action_count += 1;
                    action_cost += get(action, &["cost_units"])?
                        .as_f64()
                        .ok_or_else(|| fail("invalid action cost"))?;
                }
                "external_update" => {
                    if !get(event, &["action_id"])?.is_null() {
                        return Err(fail("external update must not masquerade as an action"));
                    }
                }
                other => return Err(fail(format!("unsupported trigger_kind {other}"))),
            }

            for effect in array_at(event, &["obligation_updates"])? {
                let obligation_id = string_at(effect, &["obligation_id"])?;
                let evidence_class = string_at(effect, &["evidence_class"])?;
                let state = array_at(agent_episode, &["initial_state", "obligations"])?
                    .iter()
                    .find(|item| {
                        item.get("obligation_id").and_then(Value::as_str) == Some(obligation_id)
                    })
                    .ok_or_else(|| fail(format!("event updates undeclared obligation {obligation_id}")))?;
                if !array_at(state, &["admissible_evidence_classes"])?
                    .iter()
                    .any(|item| item.as_str() == Some(evidence_class))
                {
                    return Err(fail(format!(
                        "inadmissible event class {evidence_class} for obligation {obligation_id}"
                    )));
                }
            }
            for effect in array_at(event, &["duty_updates"])? {
                let duty_id = string_at(effect, &["duty_id"])?;
                let evidence_class = string_at(effect, &["evidence_class"])?;
                let state = array_at(agent_episode, &["initial_state", "defeater_duties"])?
                    .iter()
                    .find(|item| item.get("duty_id").and_then(Value::as_str) == Some(duty_id))
                    .ok_or_else(|| fail(format!("event updates undeclared duty {duty_id}")))?;
                if !array_at(state, &["admissible_evidence_classes"])?
                    .iter()
                    .any(|item| item.as_str() == Some(evidence_class))
                {
                    return Err(fail(format!(
                        "inadmissible event class {evidence_class} for duty {duty_id}"
                    )));
                }
            }

            for invalidated in array_at(event, &["invalidated_evidence_ids"])? {
                let evidence_id = invalidated
                    .as_str()
                    .ok_or_else(|| fail("invalidated Evidence ID must be a string"))?;
                if !array_at(agent_episode, &["initial_state", "observed_evidence_ids"])?
                    .iter()
                    .any(|item| item.as_str() == Some(evidence_id))
                {
                    return Err(fail(format!(
                        "event invalidates Evidence ID not present in initial state: {evidence_id}"
                    )));
                }
            }
        }

        let max_actions = get(budget, &["max_actions"])?
            .as_u64()
            .ok_or_else(|| fail("max_actions must be an integer"))?;
        let max_cost = get(budget, &["max_cost_units"])?
            .as_f64()
            .ok_or_else(|| fail("max_cost_units must be numeric"))?;
        if action_count > max_actions || action_cost > max_cost + 1.0e-12 {
            return Err(fail(format!("grader event stream exceeds budget in {id}")));
        }

        let mut current = (*agent_episode).clone();
        let events = array_at(grader_episode, &["events"])?;
        for checkpoint in array_at(grader_episode, &["checkpoints"])? {
            let event_index = get(checkpoint, &["after_event_index"])?
                .as_u64()
                .ok_or_else(|| fail("checkpoint event index must be an integer"))? as usize;
            if event_index > events.len() {
                return Err(fail(format!("checkpoint exceeds event stream in {id}")));
            }

            // Checkpoints are declared in chronological order; apply each
            // newly observed event exactly once.
            let already_applied = current
                .get("__checker_applied_events")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            if event_index < already_applied {
                return Err(fail(format!("checkpoint event indexes regress in {id}")));
            }
            for index in already_applied..event_index {
                current = apply_event(&current, grader_episode, index)?;
            }
            current["__checker_applied_events"] = Value::from(event_index as u64);

            let decision = string_at(checkpoint, &["decision"])?;
            match decision {
                "continue" => {
                    let action_id = string_at(checkpoint, &["action_id"])?;
                    let action = candidates
                        .get(action_id)
                        .ok_or_else(|| fail(format!("checkpoint continues with unknown action {action_id}")))?;
                    if !is_action_eligible(action, budget)? {
                        return Err(fail(format!("checkpoint selects ineligible action {action_id}")));
                    }
                    if get(checkpoint, &["claim_id"])?.is_null() == false {
                        return Err(fail("continue checkpoint must not declare a terminal claim"));
                    }
                }
                "stop_ready" | "stop_bounded" => {
                    let claim_id = string_at(checkpoint, &["claim_id"])?;
                    if !claim_is_ready(&current, claim_id)? {
                        return Err(fail(format!(
                            "checkpoint marks non-ready claim {claim_id} as {decision} in {id}"
                        )));
                    }
                    let strongest = strongest_ready_claim(&current)?;
                    if strongest.as_deref() != Some(claim_id) {
                        return Err(fail(format!(
                            "checkpoint selects {claim_id}, but strongest ready claim is {strongest:?} in {id}"
                        )));
                    }
                    if decision == "stop_bounded" {
                        let selected_rank = array_at(&current, &["claim_ladder"])?
                            .iter()
                            .find(|claim| claim.get("claim_id").and_then(Value::as_str) == Some(claim_id))
                            .and_then(|claim| claim.get("strength_rank"))
                            .and_then(Value::as_u64)
                            .unwrap_or(0);
                        if !array_at(&current, &["claim_ladder"])?.iter().any(|claim| {
                            claim.get("strength_rank").and_then(Value::as_u64).unwrap_or(0) > selected_rank
                        }) {
                            return Err(fail(format!(
                                "stop_bounded selected the strongest/only claim in {id}"
                            )));
                        }
                    }
                    if !get(checkpoint, &["action_id"])?.is_null() {
                        return Err(fail("terminal stop checkpoint must not select an action"));
                    }
                }
                "stop_unresolved" | "abstain" => {
                    if !get(checkpoint, &["claim_id"])?.is_null()
                        || !get(checkpoint, &["action_id"])?.is_null()
                    {
                        return Err(fail("unresolved/abstain checkpoint must not select a claim or action"));
                    }
                    if strongest_ready_claim(&current)?.is_some() {
                        return Err(fail(format!(
                            "{decision} used despite a declared ready claim in {id}"
                        )));
                    }
                }
                other => return Err(fail(format!("unknown checkpoint decision {other}"))),
            }
        }

        let expected_final = if get(grader_episode, &["final_ready_claim_id"])?.is_null() {
            None
        } else {
            Some(string_at(grader_episode, &["final_ready_claim_id"])?.to_owned())
        };
        let final_claim = strongest_ready_claim(&current)?;
        if expected_final != final_claim {
            return Err(fail(format!(
                "final ready claim mismatch in {id}: grader={expected_final:?}, computed={final_claim:?}"
            )));
        }
    }

    if grader_ids.len() != agent_by_id.len() {
        return Err(fail("agent/grader episode sets differ"));
    }
    Ok(())
}

/// Validate both Draft 2020-12 schemas, their instances and cross-view semantics.
pub fn validate_embedded_corpus() -> Result<()> {
    let agent = validate_schema_instance(AGENT_SCHEMA, AGENT_VIEW, "agent view")?;
    let grader = validate_schema_instance(GRADER_SCHEMA, GRADER_VIEW, "grader view")?;
    validate_cross_corpus(&agent, &grader)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_and_grader() -> (Value, Value) {
        (
            serde_json::from_str(AGENT_VIEW).unwrap(),
            serde_json::from_str(GRADER_VIEW).unwrap(),
        )
    }

    fn episode<'a>(value: &'a Value, id: &str) -> &'a Value {
        value["episodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["episode_id"] == id)
            .unwrap()
    }

    fn grader_episode<'a>(value: &'a Value, id: &str) -> &'a Value {
        value["episodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["episode_id"] == id)
            .unwrap()
    }

    fn apply_all(mut state: Value, grader: &Value, id: &str) -> Value {
        let events = grader_episode(grader, id)["events"].as_array().unwrap();
        for index in 0..events.len() {
            state = apply_event(&state, grader_episode(grader, id), index).unwrap();
        }
        state
    }

    #[test]
    fn both_schemas_are_draft_2020_12_and_instances_conform() {
        validate_embedded_corpus().unwrap();
    }

    #[test]
    fn candidates_do_not_discharge_initial_obligations() {
        let (agent, _) = agent_and_grader();
        let initial = episode(&agent, "REA-STOP-001");
        assert_eq!(strongest_ready_claim(initial).unwrap(), None);
    }

    #[test]
    fn negative_control_is_required_before_strong_firmware_claim() {
        let (agent, grader) = agent_and_grader();
        let mut state = episode(&agent, "REA-STOP-001").clone();
        let ge = grader_episode(&grader, "REA-STOP-001");
        for index in 0..3 {
            state = apply_event(&state, ge, index).unwrap();
        }
        assert_eq!(
            strongest_ready_claim(&state).unwrap().as_deref(),
            Some("observed_behavior_on_artifact")
        );
        state = apply_event(&state, ge, 3).unwrap();
        assert_eq!(
            strongest_ready_claim(&state).unwrap().as_deref(),
            Some("deployed_firmware_is_vulnerable")
        );
    }

    #[test]
    fn late_counterexample_revokes_strong_claim_but_preserves_historical_claim() {
        let (agent, grader) = agent_and_grader();
        let initial = episode(&agent, "REA-STOP-002").clone();
        assert_eq!(
            strongest_ready_claim(&initial).unwrap().as_deref(),
            Some("current_release_vulnerable")
        );
        let after = apply_event(&initial, grader_episode(&grader, "REA-STOP-002"), 0).unwrap();
        assert_eq!(
            strongest_ready_claim(&after).unwrap().as_deref(),
            Some("package_advisory_match")
        );
    }

    #[test]
    fn source_lineage_merge_revokes_independence_until_new_root_is_observed() {
        let (agent, grader) = agent_and_grader();
        let initial = episode(&agent, "REA-STOP-003").clone();
        let ge = grader_episode(&grader, "REA-STOP-003");
        assert_eq!(
            strongest_ready_claim(&initial).unwrap().as_deref(),
            Some("independent_sources_corroborate")
        );
        let conflicted = apply_event(&initial, ge, 0).unwrap();
        assert_eq!(
            strongest_ready_claim(&conflicted).unwrap().as_deref(),
            Some("two_pages_report_same_advisory")
        );
        let recovered = apply_event(&conflicted, ge, 1).unwrap();
        assert_eq!(
            strongest_ready_claim(&recovered).unwrap().as_deref(),
            Some("independent_sources_corroborate")
        );
    }

    #[test]
    fn supersession_revokes_current_claim_but_not_historical_fact() {
        let (agent, grader) = agent_and_grader();
        let initial = episode(&agent, "REA-STOP-004").clone();
        let updated = apply_event(&initial, grader_episode(&grader, "REA-STOP-004"), 0).unwrap();
        assert_eq!(
            strongest_ready_claim(&updated).unwrap().as_deref(),
            Some("historical_version_was_affected")
        );
    }

    #[test]
    fn artifact_mismatch_invalidates_descendant_behavior_evidence() {
        let (agent, grader) = agent_and_grader();
        let initial = episode(&agent, "REA-STOP-005").clone();
        let updated = apply_event(&initial, grader_episode(&grader, "REA-STOP-005"), 0).unwrap();
        assert_eq!(
            strongest_ready_claim(&updated).unwrap().as_deref(),
            Some("provider_report_reproduced")
        );
    }

    #[test]
    fn bounded_no_event_does_not_prove_universal_absence() {
        let (agent, grader) = agent_and_grader();
        let updated = apply_all(
            episode(&agent, "REA-STOP-006").clone(),
            &grader,
            "REA-STOP-006",
        );
        assert_eq!(
            strongest_ready_claim(&updated).unwrap().as_deref(),
            Some("no_network_event_in_session")
        );
    }

    #[test]
    fn challenge_counterexample_preserves_only_the_initial_test_fact() {
        let (agent, grader) = agent_and_grader();
        let updated = apply_all(
            episode(&agent, "REA-STOP-007").clone(),
            &grader,
            "REA-STOP-007",
        );
        assert_eq!(
            strongest_ready_claim(&updated).unwrap().as_deref(),
            Some("control_passed_initial_test")
        );
    }

    #[test]
    fn equivalent_final_evidence_states_are_order_invariant() {
        let (agent, grader) = agent_and_grader();
        let final_a = apply_all(
            episode(&agent, "REA-STOP-012").clone(),
            &grader,
            "REA-STOP-012",
        );
        let final_b = apply_all(
            episode(&agent, "REA-STOP-013").clone(),
            &grader,
            "REA-STOP-013",
        );
        assert_eq!(
            strongest_ready_claim(&final_a).unwrap(),
            strongest_ready_claim(&final_b).unwrap()
        );
        for id in ["artifact_identity", "source_independence", "current_validity"] {
            let status_a = final_a["initial_state"]["obligations"]
                .as_array().unwrap().iter()
                .find(|item| item["obligation_id"] == id).unwrap()["status"].clone();
            let status_b = final_b["initial_state"]["obligations"]
                .as_array().unwrap().iter()
                .find(|item| item["obligation_id"] == id).unwrap()["status"].clone();
            assert_eq!(status_a, status_b);
        }
    }

    #[test]
    fn all_declared_checkpoint_and_transition_mutations_are_rejected() {
        let (agent, grader) = agent_and_grader();
        let mut leaked = agent.clone();
        leaked["episodes"][0]["world_state_oracle"] = Value::String("leak".into());
        assert!(validate_cross_corpus(&leaked, &grader).is_err());

        let mut dangling = grader.clone();
        dangling["episodes"][0]["events"][0]["action_id"] = Value::String("missing_action".into());
        assert!(validate_cross_corpus(&agent, &dangling).is_err());

        let mut invalid_class = grader.clone();
        invalid_class["episodes"][0]["events"][1]["obligation_updates"][0]["evidence_class"] =
            Value::String("provider_self_report".into());
        assert!(validate_cross_corpus(&agent, &invalid_class).is_err());
    }
}
