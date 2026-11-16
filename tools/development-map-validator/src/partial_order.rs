//! Evaluator-only D6.10 scorer for partial-order event histories.
//!
//! The scorer distinguishes event time from candidate-visible availability,
//! and never invents happens-before edges from timestamps. It evaluates
//! supplied adjudicated/synthetic event relations, branch-scoped receipts,
//! branch joins, confluence claims and decision-cutoff leakage. It does not
//! authenticate logs, prove event completeness, or execute candidate code.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

const CLAIM_CEILING: &str = "Synthetic known-answer evaluation of declared partial-order relations, branch ancestry, receipt scope, join requalification, and candidate-visible availability. Does not establish complete causal reconstruction, authenticated event capture, production concurrency safety, or agent performance.";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CausalOrderReport {
    pub reference_ordered_pairs: usize,
    pub declared_ordered_pairs: usize,
    pub correctly_ordered_pairs: Vec<[String; 2]>,
    pub false_serialization_pairs: Vec<[String; 2]>,
    pub missing_order_pairs: Vec<[String; 2]>,
    pub wrong_direction_pairs: Vec<[String; 2]>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConfluenceReport {
    pub reference_commuting_pairs: Vec<[String; 2]>,
    pub claimed_confluent_pairs: Vec<[String; 2]>,
    pub true_confluence_pairs: Vec<[String; 2]>,
    pub false_confluence_pairs: Vec<[String; 2]>,
    pub missed_confluence_pairs: Vec<[String; 2]>,
    pub false_noncommuting_confluence_pairs: Vec<[String; 2]>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ReceiptUseIssue {
    pub receipt_id: String,
    pub target_branch_id: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JoinReport {
    pub branch_id: String,
    pub required_requalification_obligations: Vec<String>,
    pub freshly_requalified_obligations: Vec<String>,
    pub missing_requalification_obligations: Vec<String>,
    pub invalid_fresh_receipt_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PartialOrderTraceReport {
    pub trace_id: String,
    pub causal_order: CausalOrderReport,
    pub confluence: ConfluenceReport,
    pub future_evidence_leakage_event_ids: Vec<String>,
    pub future_evidence_leakage_receipt_ids: Vec<String>,
    pub out_of_scope_event_ids: Vec<String>,
    pub branch_receipt_contamination_ids: Vec<String>,
    pub invalid_receipt_uses: Vec<ReceiptUseIssue>,
    pub unsupported_current_claims: Vec<String>,
    pub critical_unsupported_current_claims: Vec<String>,
    pub join_reports: Vec<JoinReport>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PartialOrderReport {
    pub scenario_id: String,
    pub adjudication_status: String,
    pub event_count: usize,
    pub branch_count: usize,
    pub obligation_count: usize,
    pub receipt_count: usize,
    pub traces: Vec<PartialOrderTraceReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct Event {
    available_at: DateTime<FixedOffset>,
    branch_id: String,
    effect_dimensions: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct Branch {
    parent_branch_ids: BTreeSet<String>,
    event_ids: BTreeSet<String>,
    is_join: bool,
    merge_event_id: Option<String>,
    context_digest: String,
}

#[derive(Debug, Clone)]
struct Obligation {
    critical: bool,
    dependency_dimensions: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct Receipt {
    obligation_id: String,
    branch_id: String,
    scope: String,
    context_digest: String,
    source_event_ids: BTreeSet<String>,
    outcome: String,
    observed_at: DateTime<FixedOffset>,
    available_at: DateTime<FixedOffset>,
    valid_until: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
struct ReceiptAssessment {
    valid: bool,
    contamination: bool,
    future: bool,
    reasons: Vec<String>,
}

fn ensure_allowed_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
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

fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(raw)
        .map_err(|error| format!("{label}.{key} must be RFC 3339: {error}"))
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

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
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

fn parse_pairs(
    value: &Value,
    key: &str,
    label: &str,
    known_events: &BTreeSet<String>,
    unordered: bool,
) -> Result<BTreeSet<(String, String)>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, pair) in values.iter().enumerate() {
        let items = pair
            .as_array()
            .filter(|items| items.len() == 2)
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a two-element array"))?;
        let left = items[0]
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][0] must be a non-empty string"))?;
        let right = items[1]
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][1] must be a non-empty string"))?;
        if left == right {
            return Err(format!("{label}.{key}[{index}] cannot relate an event to itself"));
        }
        if !known_events.contains(left) || !known_events.contains(right) {
            return Err(format!(
                "{label}.{key}[{index}] references an unknown event"
            ));
        }
        let pair = if unordered && left > right {
            (right.to_owned(), left.to_owned())
        } else {
            (left.to_owned(), right.to_owned())
        };
        if !output.insert(pair) {
            return Err(format!("{label}.{key} contains a duplicate pair"));
        }
    }
    Ok(output)
}

fn transitive_closure(
    nodes: &BTreeSet<String>,
    edges: &BTreeSet<(String, String)>,
    label: &str,
) -> Result<BTreeSet<(String, String)>, String> {
    let mut outgoing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for node in nodes {
        outgoing.insert(node.clone(), BTreeSet::new());
    }
    for (before, after) in edges {
        if before == after || !nodes.contains(before) || !nodes.contains(after) {
            return Err(format!("{label} contains a self-edge or unknown event"));
        }
        outgoing
            .get_mut(before)
            .ok_or_else(|| format!("{label} has an unknown source event"))?
            .insert(after.clone());
    }

    let mut closure = BTreeSet::new();
    for start in nodes {
        let mut stack: Vec<String> = outgoing
            .get(start)
            .into_iter()
            .flat_map(|items| items.iter().cloned())
            .collect();
        let mut visited = BTreeSet::new();
        while let Some(next) = stack.pop() {
            if &next == start {
                return Err(format!("{label} contains a directed cycle"));
            }
            if !visited.insert(next.clone()) {
                continue;
            }
            closure.insert((start.clone(), next.clone()));
            if let Some(children) = outgoing.get(&next) {
                stack.extend(children.iter().cloned());
            }
        }
    }
    Ok(closure)
}

fn validate_branch_graph(branches: &BTreeMap<String, Branch>) -> Result<(), String> {
    let mut indegree: BTreeMap<String, usize> = branches
        .keys()
        .map(|id| (id.clone(), 0))
        .collect();
    let mut children: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (branch_id, branch) in branches {
        if branch.parent_branch_ids.contains(branch_id) {
            return Err(format!("branch {branch_id:?} cannot be its own parent"));
        }
        for parent in &branch.parent_branch_ids {
            if !branches.contains_key(parent) {
                return Err(format!(
                    "branch {branch_id:?} references missing parent {parent:?}"
                ));
            }
            *indegree
                .get_mut(branch_id)
                .ok_or_else(|| "branch indegree entry missing".to_owned())? += 1;
            children
                .entry(parent.clone())
                .or_default()
                .insert(branch_id.clone());
        }
        if branch.is_join && branch.parent_branch_ids.len() < 2 {
            return Err(format!(
                "join branch {branch_id:?} must have at least two parents"
            ));
        }
        if !branch.is_join && branch.parent_branch_ids.len() > 1 {
            return Err(format!(
                "non-join branch {branch_id:?} cannot have multiple parents"
            ));
        }
        if branch.is_join != branch.merge_event_id.is_some() {
            return Err(format!(
                "branch {branch_id:?} must provide merge_event_id exactly when is_join is true"
            ));
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut visited = 0usize;
    while let Some(branch_id) = ready.pop_first() {
        visited += 1;
        if let Some(next_branches) = children.get(&branch_id) {
            for child in next_branches {
                let degree = indegree
                    .get_mut(child)
                    .ok_or_else(|| "branch child indegree entry missing".to_owned())?;
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }
    if visited != branches.len() {
        return Err("branch ancestry graph contains a cycle".to_owned());
    }
    if !branches.is_empty() && !branches.values().any(|branch| branch.parent_branch_ids.is_empty()) {
        return Err("branch graph must contain at least one root branch".to_owned());
    }
    Ok(())
}

fn branch_ancestors(
    branch_id: &str,
    branches: &BTreeMap<String, Branch>,
) -> Result<BTreeSet<String>, String> {
    if !branches.contains_key(branch_id) {
        return Err(format!("unknown branch {branch_id:?}"));
    }
    let mut output = BTreeSet::new();
    let mut stack = vec![branch_id.to_owned()];
    while let Some(current) = stack.pop() {
        if !output.insert(current.clone()) {
            continue;
        }
        let branch = branches
            .get(&current)
            .ok_or_else(|| format!("unknown branch {current:?}"))?;
        stack.extend(branch.parent_branch_ids.iter().cloned());
    }
    Ok(output)
}

fn parse_events(
    candidate: &Value,
) -> Result<(BTreeMap<String, Event>, BTreeSet<String>), String> {
    let values = candidate
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.events must be an array".to_owned())?;
    if values.is_empty() {
        return Err("candidate_view.events must not be empty".to_owned());
    }
    let mut events = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.events[{index}]");
        ensure_allowed_keys(
            value,
            &[
                "event_id",
                "event_time",
                "available_at",
                "branch_id",
                "event_kind",
                "effect_dimensions",
            ],
            &label,
        )?;
        let id = required_str(value, "event_id", &label)?.to_owned();
        let _event_time = parse_time(value, "event_time", &label)?;
        let available_at = parse_time(value, "available_at", &label)?;
        let branch_id = required_str(value, "branch_id", &label)?.to_owned();
        let event_kind = required_str(value, "event_kind", &label)?;
        if ![
            "baseline",
            "source_change",
            "contract_change",
            "policy_change",
            "configuration_change",
            "rollback",
            "merge",
            "observation",
            "other",
        ]
        .contains(&event_kind)
        {
            return Err(format!("{label}.event_kind has unsupported value {event_kind:?}"));
        }
        let effect_dimensions = string_set(value, "effect_dimensions", &label)?;
        if events
            .insert(
                id.clone(),
                Event {
                    available_at,
                    branch_id,
                    effect_dimensions,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate event_id {id:?}"));
        }
    }
    let ids = events.keys().cloned().collect();
    Ok((events, ids))
}

fn parse_branches(
    candidate: &Value,
    events: &BTreeMap<String, Event>,
) -> Result<BTreeMap<String, Branch>, String> {
    let values = candidate
        .get("branches")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.branches must be an array".to_owned())?;
    let mut branches = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.branches[{index}]");
        ensure_allowed_keys(
            value,
            &[
                "branch_id",
                "parent_branch_ids",
                "event_ids",
                "is_join",
                "merge_event_id",
                "context_digest",
            ],
            &label,
        )?;
        let id = required_str(value, "branch_id", &label)?.to_owned();
        let parent_branch_ids = string_set(value, "parent_branch_ids", &label)?;
        let event_ids = string_set(value, "event_ids", &label)?;
        let is_join = required_bool(value, "is_join", &label)?;
        let merge_event_id = match value.get("merge_event_id") {
            Some(Value::String(raw)) if !raw.trim().is_empty() => Some(raw.clone()),
            Some(Value::Null) | None => None,
            _ => return Err(format!("{label}.merge_event_id must be a string or null")),
        };
        let context_digest = digest(value, "context_digest", &label)?;
        if branches
            .insert(
                id.clone(),
                Branch {
                    parent_branch_ids,
                    event_ids,
                    is_join,
                    merge_event_id,
                    context_digest,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate branch_id {id:?}"));
        }
    }
    validate_branch_graph(&branches)?;

    let mut assigned_events = BTreeSet::new();
    for (branch_id, branch) in &branches {
        for event_id in &branch.event_ids {
            let event = events
                .get(event_id)
                .ok_or_else(|| format!("branch {branch_id:?} references unknown event {event_id:?}"))?;
            if event.branch_id != *branch_id {
                return Err(format!(
                    "event {event_id:?} branch_id does not match its branch event_ids owner"
                ));
            }
            if !assigned_events.insert(event_id.clone()) {
                return Err(format!("event {event_id:?} is assigned to more than one branch"));
            }
        }
        if let Some(merge_event_id) = &branch.merge_event_id {
            if !branch.event_ids.contains(merge_event_id) {
                return Err(format!(
                    "join branch {branch_id:?} merge_event_id must be one of its event_ids"
                ));
            }
        }
    }
    if assigned_events.len() != events.len() {
        return Err("every event must be assigned to exactly one branch".to_owned());
    }
    Ok(branches)
}

fn parse_obligations(candidate: &Value) -> Result<BTreeMap<String, Obligation>, String> {
    let values = candidate
        .get("obligations")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.obligations must be an array".to_owned())?;
    let mut obligations = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.obligations[{index}]");
        ensure_allowed_keys(
            value,
            &["obligation_id", "critical", "dependency_dimensions"],
            &label,
        )?;
        let id = required_str(value, "obligation_id", &label)?.to_owned();
        let critical = required_bool(value, "critical", &label)?;
        let dependency_dimensions = string_set(value, "dependency_dimensions", &label)?;
        if dependency_dimensions.is_empty() {
            return Err(format!("{label}.dependency_dimensions must not be empty"));
        }
        if obligations
            .insert(
                id.clone(),
                Obligation {
                    critical,
                    dependency_dimensions,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate obligation_id {id:?}"));
        }
    }
    if obligations.is_empty() {
        return Err("candidate_view.obligations must not be empty".to_owned());
    }
    Ok(obligations)
}

fn parse_receipts(
    candidate: &Value,
    events: &BTreeMap<String, Event>,
    branches: &BTreeMap<String, Branch>,
    obligations: &BTreeMap<String, Obligation>,
) -> Result<BTreeMap<String, Receipt>, String> {
    let values = candidate
        .get("receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.receipts must be an array".to_owned())?;
    let mut receipts = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.receipts[{index}]");
        ensure_allowed_keys(
            value,
            &[
                "receipt_id",
                "obligation_id",
                "branch_id",
                "scope",
                "context_digest",
                "source_event_ids",
                "outcome",
                "observed_at",
                "available_at",
                "valid_until",
            ],
            &label,
        )?;
        let id = required_str(value, "receipt_id", &label)?.to_owned();
        let obligation_id = required_str(value, "obligation_id", &label)?.to_owned();
        if !obligations.contains_key(&obligation_id) {
            return Err(format!("{label} references unknown obligation {obligation_id:?}"));
        }
        let branch_id = required_str(value, "branch_id", &label)?.to_owned();
        if !branches.contains_key(&branch_id) {
            return Err(format!("{label} references unknown branch {branch_id:?}"));
        }
        let scope = required_str(value, "scope", &label)?.to_owned();
        if !["branch_only", "descendants"].contains(&scope.as_str()) {
            return Err(format!("{label}.scope has unsupported value {scope:?}"));
        }
        let context_digest = digest(value, "context_digest", &label)?;
        let source_event_ids = string_set(value, "source_event_ids", &label)?;
        if source_event_ids.is_empty() {
            return Err(format!("{label}.source_event_ids must not be empty"));
        }
        let outcome = required_str(value, "outcome", &label)?.to_owned();
        if !["pass", "fail", "unknown"].contains(&outcome.as_str()) {
            return Err(format!("{label}.outcome has unsupported value {outcome:?}"));
        }
        let observed_at = parse_time(value, "observed_at", &label)?;
        let available_at = parse_time(value, "available_at", &label)?;
        let valid_until = parse_time(value, "valid_until", &label)?;

        let receipt_ancestors = branch_ancestors(&branch_id, branches)?;
        for source_event_id in &source_event_ids {
            let source_event = events
                .get(source_event_id)
                .ok_or_else(|| format!("{label} references unknown source event {source_event_id:?}"))?;
            if !receipt_ancestors.contains(&source_event.branch_id) {
                return Err(format!(
                    "{label} source event {source_event_id:?} is outside receipt branch ancestry"
                ));
            }
        }
        if receipts
            .insert(
                id.clone(),
                Receipt {
                    obligation_id,
                    branch_id,
                    scope,
                    context_digest,
                    source_event_ids,
                    outcome,
                    observed_at,
                    available_at,
                    valid_until,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate receipt_id {id:?}"));
        }
    }
    Ok(receipts)
}

fn parse_join_requirements(
    gold: &Value,
    events: &BTreeMap<String, Event>,
    branches: &BTreeMap<String, Branch>,
    obligations: &BTreeMap<String, Obligation>,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let values = gold
        .get("join_required_obligations")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.join_required_obligations must be an array".to_owned())?;
    let mut declared = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.join_required_obligations[{index}]");
        ensure_allowed_keys(value, &["branch_id", "obligation_ids"], &label)?;
        let branch_id = required_str(value, "branch_id", &label)?.to_owned();
        if !branches.get(&branch_id).is_some_and(|branch| branch.is_join) {
            return Err(format!("{label}.branch_id must reference a join branch"));
        }
        let ids = string_set(value, "obligation_ids", &label)?;
        for id in &ids {
            if !obligations.contains_key(id) {
                return Err(format!("{label} references unknown obligation {id:?}"));
            }
        }
        if declared.insert(branch_id.clone(), ids).is_some() {
            return Err(format!("duplicate join requirement for {branch_id:?}"));
        }
    }

    let mut expected = BTreeMap::new();
    for (branch_id, branch) in branches {
        if !branch.is_join {
            continue;
        }
        let ancestors = branch_ancestors(branch_id, branches)?;
        let mut changed_dimensions = BTreeSet::new();
        for ancestor_id in ancestors.iter().filter(|id| *id != branch_id) {
            let ancestor = branches
                .get(ancestor_id)
                .ok_or_else(|| format!("missing ancestor branch {ancestor_id:?}"))?;
            for event_id in &ancestor.event_ids {
                let event = events
                    .get(event_id)
                    .ok_or_else(|| format!("missing event {event_id:?}"))?;
                changed_dimensions.extend(event.effect_dimensions.iter().cloned());
            }
        }
        let impacted: BTreeSet<String> = obligations
            .iter()
            .filter(|(_, obligation)| {
                !obligation
                    .dependency_dimensions
                    .is_disjoint(&changed_dimensions)
            })
            .map(|(id, _)| id.clone())
            .collect();
        expected.insert(branch_id.clone(), impacted);
    }

    if declared != expected {
        return Err(format!(
            "join_required_obligations mismatch: computed {expected:?}, declared {declared:?}"
        ));
    }
    Ok(expected)
}

fn assess_receipt_use(
    receipt: &Receipt,
    receipt_id: &str,
    target_branch_id: &str,
    cutoff: &DateTime<FixedOffset>,
    events: &BTreeMap<String, Event>,
    branches: &BTreeMap<String, Branch>,
    join_requirements: &BTreeMap<String, BTreeSet<String>>,
) -> Result<ReceiptAssessment, String> {
    let target = branches
        .get(target_branch_id)
        .ok_or_else(|| format!("unknown target branch {target_branch_id:?}"))?;
    let ancestors = branch_ancestors(target_branch_id, branches)?;
    let mut reasons = Vec::new();
    let mut contamination = false;
    let future = receipt.available_at > *cutoff;

    if !ancestors.contains(&receipt.branch_id) {
        reasons.push("receipt_branch_is_not_an_ancestor".to_owned());
        contamination = true;
    }
    if receipt.scope == "branch_only" && receipt.branch_id != target_branch_id {
        reasons.push("branch_only_receipt_used_outside_origin_branch".to_owned());
        contamination = true;
    }
    if receipt.context_digest != target.context_digest {
        reasons.push("receipt_context_digest_mismatch".to_owned());
        contamination = true;
    }
    if join_requirements
        .get(target_branch_id)
        .is_some_and(|required| required.contains(&receipt.obligation_id))
    {
        reasons.push("join_obligation_requires_fresh_requalification".to_owned());
        contamination = true;
    }
    if receipt.observed_at > receipt.available_at {
        reasons.push("receipt_observation_after_availability".to_owned());
    }
    if future {
        reasons.push("receipt_unavailable_at_decision_cutoff".to_owned());
    }
    if *cutoff > receipt.valid_until {
        reasons.push("receipt_expired_at_decision_cutoff".to_owned());
    }
    if receipt.outcome != "pass" {
        reasons.push(format!("receipt_outcome_{}", receipt.outcome));
    }
    for source_event_id in &receipt.source_event_ids {
        let source_event = events
            .get(source_event_id)
            .ok_or_else(|| format!("receipt {receipt_id:?} references missing event"))?;
        if !ancestors.contains(&source_event.branch_id) {
            reasons.push(format!("receipt_source_event_outside_target_ancestry:{source_event_id}"));
            contamination = true;
        }
        if source_event.available_at > *cutoff {
            reasons.push(format!("receipt_source_event_unavailable_at_cutoff:{source_event_id}"));
        }
    }
    reasons.sort();
    reasons.dedup();
    Ok(ReceiptAssessment {
        valid: reasons.is_empty(),
        contamination,
        future,
        reasons,
    })
}

fn assess_fresh_join_receipt(
    receipt: &Receipt,
    receipt_id: &str,
    branch_id: &str,
    cutoff: &DateTime<FixedOffset>,
    events: &BTreeMap<String, Event>,
    branches: &BTreeMap<String, Branch>,
) -> Result<Vec<String>, String> {
    let branch = branches
        .get(branch_id)
        .ok_or_else(|| format!("unknown join branch {branch_id:?}"))?;
    if !branch.is_join {
        return Err(format!("fresh requalification target {branch_id:?} is not a join branch"));
    }
    let merge_event_id = branch
        .merge_event_id
        .as_ref()
        .ok_or_else(|| format!("join branch {branch_id:?} lacks merge_event_id"))?;
    let merge_event = events
        .get(merge_event_id)
        .ok_or_else(|| format!("join branch {branch_id:?} references missing merge event"))?;

    let mut reasons = Vec::new();
    if receipt.branch_id != branch_id {
        reasons.push("fresh_receipt_not_issued_on_join_branch".to_owned());
    }
    if receipt.context_digest != branch.context_digest {
        reasons.push("fresh_receipt_context_digest_mismatch".to_owned());
    }
    if receipt.outcome != "pass" {
        reasons.push(format!("fresh_receipt_outcome_{}", receipt.outcome));
    }
    if !receipt.source_event_ids.contains(merge_event_id) {
        reasons.push("fresh_receipt_not_bound_to_merge_event".to_owned());
    }
    if receipt.observed_at < merge_event.available_at {
        reasons.push("fresh_receipt_observed_before_merge_available".to_owned());
    }
    if receipt.observed_at > receipt.available_at {
        reasons.push("fresh_receipt_observation_after_availability".to_owned());
    }
    if receipt.available_at > *cutoff {
        reasons.push("fresh_receipt_unavailable_at_decision_cutoff".to_owned());
    }
    if *cutoff > receipt.valid_until {
        reasons.push("fresh_receipt_expired_at_decision_cutoff".to_owned());
    }
    for event_id in &receipt.source_event_ids {
        let event = events
            .get(event_id)
            .ok_or_else(|| format!("fresh receipt {receipt_id:?} references missing event"))?;
        if event.available_at > *cutoff {
            reasons.push(format!("fresh_receipt_source_event_unavailable:{event_id}"));
        }
    }
    reasons.sort();
    reasons.dedup();
    Ok(reasons)
}

fn pair_array(pair: &(String, String)) -> [String; 2] {
    [pair.0.clone(), pair.1.clone()]
}

fn score_causal_order(
    nodes: &BTreeSet<String>,
    gold_closure: &BTreeSet<(String, String)>,
    candidate_edges: &BTreeSet<(String, String)>,
) -> Result<CausalOrderReport, String> {
    let candidate_closure =
        transitive_closure(nodes, candidate_edges, "candidate asserted_happens_before_edges")?;
    let mut correct = BTreeSet::new();
    let mut false_serialization = BTreeSet::new();
    let mut missing = BTreeSet::new();
    let mut wrong_direction = BTreeSet::new();

    for left in nodes {
        for right in nodes.range((std::ops::Bound::Excluded(left), std::ops::Bound::Unbounded)) {
            let gold_lr = gold_closure.contains(&(left.clone(), right.clone()));
            let gold_rl = gold_closure.contains(&(right.clone(), left.clone()));
            let candidate_lr = candidate_closure.contains(&(left.clone(), right.clone()));
            let candidate_rl = candidate_closure.contains(&(right.clone(), left.clone()));

            if gold_lr {
                if candidate_lr {
                    correct.insert((left.clone(), right.clone()));
                } else if candidate_rl {
                    wrong_direction.insert((left.clone(), right.clone()));
                } else {
                    missing.insert((left.clone(), right.clone()));
                }
            } else if gold_rl {
                if candidate_rl {
                    correct.insert((right.clone(), left.clone()));
                } else if candidate_lr {
                    wrong_direction.insert((left.clone(), right.clone()));
                } else {
                    missing.insert((right.clone(), left.clone()));
                }
            } else if candidate_lr || candidate_rl {
                false_serialization.insert((left.clone(), right.clone()));
            }
        }
    }

    Ok(CausalOrderReport {
        reference_ordered_pairs: gold_closure.len(),
        declared_ordered_pairs: candidate_closure.len(),
        correctly_ordered_pairs: correct.iter().map(pair_array).collect(),
        false_serialization_pairs: false_serialization.iter().map(pair_array).collect(),
        missing_order_pairs: missing.iter().map(pair_array).collect(),
        wrong_direction_pairs: wrong_direction.iter().map(pair_array).collect(),
    })
}

fn score_confluence(
    commuting: &BTreeSet<(String, String)>,
    non_commuting: &BTreeSet<(String, String)>,
    claimed: &BTreeSet<(String, String)>,
) -> ConfluenceReport {
    let true_pairs: BTreeSet<_> = commuting.intersection(claimed).cloned().collect();
    let false_pairs: BTreeSet<_> = claimed.difference(commuting).cloned().collect();
    let missed_pairs: BTreeSet<_> = commuting.difference(claimed).cloned().collect();
    let false_noncommuting: BTreeSet<_> = false_pairs.intersection(non_commuting).cloned().collect();
    ConfluenceReport {
        reference_commuting_pairs: commuting.iter().map(pair_array).collect(),
        claimed_confluent_pairs: claimed.iter().map(pair_array).collect(),
        true_confluence_pairs: true_pairs.iter().map(pair_array).collect(),
        false_confluence_pairs: false_pairs.iter().map(pair_array).collect(),
        missed_confluence_pairs: missed_pairs.iter().map(pair_array).collect(),
        false_noncommuting_confluence_pairs: false_noncommuting.iter().map(pair_array).collect(),
    }
}

fn score_trace(
    trace: &Value,
    events: &BTreeMap<String, Event>,
    event_ids: &BTreeSet<String>,
    branches: &BTreeMap<String, Branch>,
    obligations: &BTreeMap<String, Obligation>,
    receipts: &BTreeMap<String, Receipt>,
    gold_closure: &BTreeSet<(String, String)>,
    commuting: &BTreeSet<(String, String)>,
    non_commuting: &BTreeSet<(String, String)>,
    join_requirements: &BTreeMap<String, BTreeSet<String>>,
) -> Result<PartialOrderTraceReport, String> {
    ensure_allowed_keys(
        trace,
        &[
            "trace_id",
            "asserted_happens_before_edges",
            "claimed_confluent_pairs",
            "branch_decisions",
            "join_requalifications",
        ],
        "candidate_trace",
    )?;
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
    let candidate_edges = parse_pairs(
        trace,
        "asserted_happens_before_edges",
        "candidate_trace",
        event_ids,
        false,
    )?;
    let causal_order = score_causal_order(event_ids, gold_closure, &candidate_edges)?;
    let claimed_confluent_pairs = parse_pairs(
        trace,
        "claimed_confluent_pairs",
        "candidate_trace",
        event_ids,
        true,
    )?;
    for pair in &claimed_confluent_pairs {
        if !commuting.contains(pair) && !non_commuting.contains(pair) {
            return Err(format!(
                "candidate trace {trace_id:?} claims confluence for an unadjudicated pair {pair:?}"
            ));
        }
    }
    let confluence = score_confluence(commuting, non_commuting, &claimed_confluent_pairs);

    let decisions = trace
        .get("branch_decisions")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.branch_decisions must be an array".to_owned())?;
    let mut decision_by_branch = BTreeMap::new();
    for (index, decision) in decisions.iter().enumerate() {
        let label = format!("candidate_trace.branch_decisions[{index}]");
        ensure_allowed_keys(
            decision,
            &[
                "branch_id",
                "decision_cutoff",
                "considered_event_ids",
                "claimed_current_obligations",
                "receipt_ids",
            ],
            &label,
        )?;
        let branch_id = required_str(decision, "branch_id", &label)?.to_owned();
        if !branches.contains_key(&branch_id) {
            return Err(format!("{label} references unknown branch {branch_id:?}"));
        }
        if decision_by_branch.insert(branch_id.clone(), decision).is_some() {
            return Err(format!("candidate trace {trace_id:?} has duplicate decision for {branch_id:?}"));
        }
    }
    if decision_by_branch.len() != branches.len()
        || branches.keys().any(|branch_id| !decision_by_branch.contains_key(branch_id))
    {
        return Err(format!(
            "candidate trace {trace_id:?} must provide exactly one decision for every branch"
        ));
    }

    let fresh_entries = trace
        .get("join_requalifications")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.join_requalifications must be an array".to_owned())?;
    let mut fresh_ids_by_branch: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (index, entry) in fresh_entries.iter().enumerate() {
        let label = format!("candidate_trace.join_requalifications[{index}]");
        ensure_allowed_keys(entry, &["branch_id", "fresh_receipt_ids"], &label)?;
        let branch_id = required_str(entry, "branch_id", &label)?.to_owned();
        if !branches.get(&branch_id).is_some_and(|branch| branch.is_join) {
            return Err(format!("{label}.branch_id must reference a join branch"));
        }
        let fresh_ids = string_set(entry, "fresh_receipt_ids", &label)?;
        if fresh_ids_by_branch.insert(branch_id.clone(), fresh_ids).is_some() {
            return Err(format!("candidate trace {trace_id:?} has duplicate join requalification for {branch_id:?}"));
        }
    }
    for (branch_id, branch) in branches {
        if branch.is_join && !fresh_ids_by_branch.contains_key(branch_id) {
            return Err(format!(
                "candidate trace {trace_id:?} omits join requalification record for {branch_id:?}"
            ));
        }
        if !branch.is_join && fresh_ids_by_branch.contains_key(branch_id) {
            return Err(format!(
                "candidate trace {trace_id:?} provides join requalification for non-join branch {branch_id:?}"
            ));
        }
    }

    let mut future_events = BTreeSet::new();
    let mut future_receipts = BTreeSet::new();
    let mut out_of_scope_events = BTreeSet::new();
    let mut contamination_ids = BTreeSet::new();
    let mut invalid_receipt_uses = Vec::new();
    let mut unsupported_claims = BTreeSet::new();
    let mut critical_unsupported_claims = BTreeSet::new();
    let mut join_reports = Vec::new();

    let mut valid_fresh_by_branch: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (branch_id, fresh_ids) in &fresh_ids_by_branch {
        let decision = decision_by_branch
            .get(branch_id)
            .ok_or_else(|| format!("missing join decision for {branch_id:?}"))?;
        let label = format!("branch decision {branch_id}");
        let cutoff = parse_time(decision, "decision_cutoff", &label)?;
        let mut valid_by_obligation = BTreeMap::new();
        let mut invalid_ids = BTreeSet::new();
        for receipt_id in fresh_ids {
            let receipt = receipts.get(receipt_id).ok_or_else(|| {
                format!("join requalification references unknown receipt {receipt_id:?}")
            })?;
            let reasons = assess_fresh_join_receipt(
                receipt,
                receipt_id,
                branch_id,
                &cutoff,
                events,
                branches,
            )?;
            if reasons.is_empty() {
                if valid_by_obligation
                    .insert(receipt.obligation_id.clone(), receipt_id.clone())
                    .is_some()
                {
                    return Err(format!(
                        "join requalification for {branch_id:?} has duplicate fresh receipts for obligation {:?}",
                        receipt.obligation_id
                    ));
                }
            } else {
                invalid_ids.insert(receipt_id.clone());
                if reasons.iter().any(|reason| reason.contains("unavailable")) {
                    future_receipts.insert(receipt_id.clone());
                }
                invalid_receipt_uses.push(ReceiptUseIssue {
                    receipt_id: receipt_id.clone(),
                    target_branch_id: branch_id.clone(),
                    reasons,
                });
            }
        }
        valid_fresh_by_branch.insert(branch_id.clone(), valid_by_obligation.clone());

        let required = join_requirements
            .get(branch_id)
            .cloned()
            .unwrap_or_default();
        let fresh_obligations: BTreeSet<String> =
            valid_by_obligation.keys().cloned().collect();
        let missing: BTreeSet<String> = required.difference(&fresh_obligations).cloned().collect();
        join_reports.push(JoinReport {
            branch_id: branch_id.clone(),
            required_requalification_obligations: required.iter().cloned().collect(),
            freshly_requalified_obligations: fresh_obligations.iter().cloned().collect(),
            missing_requalification_obligations: missing.iter().cloned().collect(),
            invalid_fresh_receipt_ids: invalid_ids.iter().cloned().collect(),
        });
    }

    for (branch_id, decision) in &decision_by_branch {
        let label = format!("branch decision {branch_id}");
        let cutoff = parse_time(decision, "decision_cutoff", &label)?;
        let target_ancestors = branch_ancestors(branch_id, branches)?;
        let considered = string_set(decision, "considered_event_ids", &label)?;
        for event_id in considered {
            let event = events
                .get(&event_id)
                .ok_or_else(|| format!("{label} references unknown event {event_id:?}"))?;
            if !target_ancestors.contains(&event.branch_id) {
                out_of_scope_events.insert(event_id.clone());
            }
            if event.available_at > cutoff {
                future_events.insert(event_id);
            }
        }

        let reused_ids = string_set(decision, "receipt_ids", &label)?;
        let mut qualified_obligations = BTreeSet::new();
        for receipt_id in reused_ids {
            let receipt = receipts
                .get(&receipt_id)
                .ok_or_else(|| format!("{label} references unknown receipt {receipt_id:?}"))?;
            let assessment = assess_receipt_use(
                receipt,
                &receipt_id,
                branch_id,
                &cutoff,
                events,
                branches,
                join_requirements,
            )?;
            if assessment.future {
                future_receipts.insert(receipt_id.clone());
            }
            if assessment.contamination {
                contamination_ids.insert(receipt_id.clone());
            }
            if assessment.valid {
                qualified_obligations.insert(receipt.obligation_id.clone());
            } else {
                invalid_receipt_uses.push(ReceiptUseIssue {
                    receipt_id,
                    target_branch_id: branch_id.clone(),
                    reasons: assessment.reasons,
                });
            }
        }

        if let Some(fresh) = valid_fresh_by_branch.get(branch_id) {
            qualified_obligations.extend(fresh.keys().cloned());
        }

        let claimed = string_set(decision, "claimed_current_obligations", &label)?;
        for obligation_id in claimed {
            if !obligations.contains_key(&obligation_id) {
                return Err(format!("{label} claims unknown obligation {obligation_id:?}"));
            }
            if !qualified_obligations.contains(&obligation_id) {
                let claim = format!("{branch_id}:{obligation_id}");
                unsupported_claims.insert(claim.clone());
                if obligations
                    .get(&obligation_id)
                    .is_some_and(|obligation| obligation.critical)
                {
                    critical_unsupported_claims.insert(claim);
                }
            }
        }
    }

    for (branch_id, required) in join_requirements {
        let fresh = valid_fresh_by_branch
            .get(branch_id)
            .cloned()
            .unwrap_or_default();
        let fresh_obligations: BTreeSet<String> = fresh.keys().cloned().collect();
        let missing: BTreeSet<String> = required.difference(&fresh_obligations).cloned().collect();
        if !missing.is_empty() {
            for obligation_id in &missing {
                let decision = decision_by_branch
                    .get(branch_id)
                    .ok_or_else(|| format!("missing branch decision for join {branch_id:?}"))?;
                let claims = string_set(
                    decision,
                    "claimed_current_obligations",
                    &format!("branch decision {branch_id}"),
                )?;
                if claims.contains(obligation_id) {
                    let claim = format!("{branch_id}:{obligation_id}");
                    unsupported_claims.insert(claim.clone());
                    if obligations
                        .get(obligation_id)
                        .is_some_and(|obligation| obligation.critical)
                    {
                        critical_unsupported_claims.insert(claim);
                    }
                }
            }
        }
    }

    invalid_receipt_uses.sort_by(|left, right| {
        (&left.target_branch_id, &left.receipt_id).cmp(&(&right.target_branch_id, &right.receipt_id))
    });
    join_reports.sort_by(|left, right| left.branch_id.cmp(&right.branch_id));

    Ok(PartialOrderTraceReport {
        trace_id,
        causal_order,
        confluence,
        future_evidence_leakage_event_ids: future_events.into_iter().collect(),
        future_evidence_leakage_receipt_ids: future_receipts.into_iter().collect(),
        out_of_scope_event_ids: out_of_scope_events.into_iter().collect(),
        branch_receipt_contamination_ids: contamination_ids.into_iter().collect(),
        invalid_receipt_uses,
        unsupported_current_claims: unsupported_claims.into_iter().collect(),
        critical_unsupported_current_claims: critical_unsupported_claims.into_iter().collect(),
        join_reports,
    })
}

/// Evaluate a D6.10 partial-order bundle. Input and output contain gold
/// adjudication/metrics and must remain evaluator-only.
pub fn evaluate_partial_order_json(input: &str) -> Result<PartialOrderReport, String> {
    let bundle: Value =
        serde_json::from_str(input).map_err(|error| format!("bundle JSON parse failed: {error}"))?;
    ensure_allowed_keys(
        &bundle,
        &[
            "bundle_id",
            "bundle_visibility",
            "candidate_view",
            "gold_overlay",
            "candidate_traces",
        ],
        "bundle",
    )?;
    required_str(&bundle, "bundle_id", "bundle")?;
    if required_str(&bundle, "bundle_visibility", "bundle")?
        != "evaluator_only_never_candidate_visible"
    {
        return Err("D6.10 bundle must be explicitly evaluator-only".to_owned());
    }
    let candidate = bundle
        .get("candidate_view")
        .ok_or_else(|| "candidate_view is required".to_owned())?;
    ensure_allowed_keys(
        candidate,
        &["scenario_id", "events", "branches", "obligations", "receipts"],
        "candidate_view",
    )?;
    let scenario_id = required_str(candidate, "scenario_id", "candidate_view")?.to_owned();
    let (events, event_ids) = parse_events(candidate)?;
    let branches = parse_branches(candidate, &events)?;
    let obligations = parse_obligations(candidate)?;
    let receipts = parse_receipts(candidate, &events, &branches, &obligations)?;

    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    ensure_allowed_keys(
        gold,
        &[
            "adjudication_status",
            "happens_before_edges",
            "commuting_pairs",
            "non_commuting_pairs",
            "join_required_obligations",
        ],
        "gold_overlay",
    )?;
    let adjudication_status = required_str(gold, "adjudication_status", "gold_overlay")?.to_owned();
    if adjudication_status != "synthetic_known_answer_not_independently_adjudicated" {
        return Err(format!(
            "unsupported D6.10 v0.1 adjudication status {adjudication_status:?}"
        ));
    }

    let gold_edges = parse_pairs(gold, "happens_before_edges", "gold_overlay", &event_ids, false)?;
    let gold_closure = transitive_closure(&event_ids, &gold_edges, "gold happens-before graph")?;
    let commuting = parse_pairs(gold, "commuting_pairs", "gold_overlay", &event_ids, true)?;
    let non_commuting = parse_pairs(gold, "non_commuting_pairs", "gold_overlay", &event_ids, true)?;
    if !commuting.is_disjoint(&non_commuting) {
        return Err("commuting_pairs and non_commuting_pairs must be disjoint".to_owned());
    }
    for pair in commuting.iter().chain(non_commuting.iter()) {
        if gold_closure.contains(pair) || gold_closure.contains(&(pair.1.clone(), pair.0.clone())) {
            return Err(format!(
                "adjudicated concurrent pair {pair:?} is ordered in the gold happens-before graph"
            ));
        }
    }

    let join_requirements =
        parse_join_requirements(gold, &events, &branches, &obligations)?;

    let traces = bundle
        .get("candidate_traces")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_traces must be an array".to_owned())?;
    if traces.is_empty() {
        return Err("at least one candidate trace is required".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut reports = Vec::new();
    for trace in traces {
        let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
        if !trace_ids.insert(trace_id) {
            return Err("candidate trace identifiers must be unique".to_owned());
        }
        reports.push(score_trace(
            trace,
            &events,
            &event_ids,
            &branches,
            &obligations,
            &receipts,
            &gold_closure,
            &commuting,
            &non_commuting,
            &join_requirements,
        )?);
    }
    reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    Ok(PartialOrderReport {
        scenario_id,
        adjudication_status,
        event_count: events.len(),
        branch_count: branches.len(),
        obligation_count: obligations.len(),
        receipt_count: receipts.len(),
        traces: reports,
        claim_ceiling: CLAIM_CEILING.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-partial-order-v0.1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-partial-order-v0.1.expected.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.10 fixture JSON")
    }

    fn report_for(value: &Value) -> PartialOrderReport {
        evaluate_partial_order_json(&value.to_string()).expect("D6.10 score")
    }

    fn trace<'a>(report: &'a PartialOrderReport, id: &str) -> &'a PartialOrderTraceReport {
        report.traces.iter().find(|item| item.trace_id == id).expect("trace")
    }

    #[test]
    fn matches_complete_machine_readable_known_answer() {
        let actual = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let actual_value = serde_json::to_value(actual).expect("serialize report");
        let expected_value: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn does_not_infer_causality_from_timestamps() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let conformant = trace(&report, "conformant-partial-order");
        assert!(conformant.causal_order.false_serialization_pairs.is_empty());
        assert!(conformant.causal_order.missing_order_pairs.is_empty());
    }

    #[test]
    fn preserves_explicit_causal_edge_when_event_times_tie() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let conformant = trace(&report, "conformant-partial-order");
        assert!(conformant.causal_order.correctly_ordered_pairs.iter().any(|pair| {
            pair[0] == "tie-first" && pair[1] == "tie-second"
        }));
    }

    #[test]
    fn detects_false_serialization_of_independent_concurrent_events() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let adversarial = trace(&report, "adversarial-linearization");
        assert!(adversarial.causal_order.false_serialization_pairs.iter().any(|pair| {
            pair[0] == "branch-a-source" && pair[1] == "branch-b-docs"
        }));
    }

    #[test]
    fn detects_false_confluence_for_non_commuting_changes() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let adversarial = trace(&report, "adversarial-linearization");
        assert!(adversarial.confluence.false_noncommuting_confluence_pairs.iter().any(|pair| {
            pair[0] == "branch-a-policy" && pair[1] == "branch-b-policy"
        }));
    }

    #[test]
    fn rejects_future_use_of_backdated_late_arriving_event() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let adversarial = trace(&report, "adversarial-linearization");
        assert!(adversarial.future_evidence_leakage_event_ids.contains(&"late-backdated".to_owned()));
        let conformant = trace(&report, "conformant-partial-order");
        assert!(!conformant.future_evidence_leakage_event_ids.contains(&"late-backdated".to_owned()));
    }

    #[test]
    fn branch_local_receipts_cannot_cross_to_sibling_or_join() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let adversarial = trace(&report, "adversarial-linearization");
        assert!(adversarial.branch_receipt_contamination_ids.contains(&"receipt-build-a".to_owned()));
        assert!(adversarial.branch_receipt_contamination_ids.contains(&"receipt-docs-b".to_owned()));
    }

    #[test]
    fn join_requires_fresh_requalification_for_all_affected_obligations() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let conformant = trace(&report, "conformant-partial-order");
        assert_eq!(
            conformant.join_reports[0].freshly_requalified_obligations,
            vec!["build-repro", "docs-contract", "policy-authority"]
        );
        assert!(conformant.join_reports[0].missing_requalification_obligations.is_empty());

        let adversarial = trace(&report, "adversarial-linearization");
        assert_eq!(
            adversarial.join_reports[0].missing_requalification_obligations,
            vec!["docs-contract", "policy-authority"]
        );
    }

    #[test]
    fn rejects_cyclic_candidate_happens_before_graph() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["asserted_happens_before_edges"]
            .as_array_mut().expect("edges")
            .push(json!(["tie-second", "tie-first"]));
        let error = evaluate_partial_order_json(&value.to_string()).unwrap_err();
        assert!(error.contains("directed cycle"));
    }

    #[test]
    fn rejects_cyclic_branch_ancestry() {
        let mut value = fixture_value();
        value["candidate_view"]["branches"][0]["parent_branch_ids"] = json!(["branch-a"]);
        let error = evaluate_partial_order_json(&value.to_string()).unwrap_err();
        assert!(error.contains("branch ancestry graph contains a cycle"));
    }

    #[test]
    fn rejects_missing_event_parent_branch() {
        let mut value = fixture_value();
        value["candidate_view"]["branches"][1]["parent_branch_ids"] = json!(["missing-branch"]);
        let error = evaluate_partial_order_json(&value.to_string()).unwrap_err();
        assert!(error.contains("references missing parent"));
    }

    #[test]
    fn rejects_unknown_nested_fields() {
        let mut value = fixture_value();
        value["candidate_view"]["events"][0]["guessed_causal_parent"] = json!("branch-a-source");
        let error = evaluate_partial_order_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn rejects_join_gold_that_omits_dependency_implied_obligation() {
        let mut value = fixture_value();
        value["gold_overlay"]["join_required_obligations"][0]["obligation_ids"] = json!([
            "build-repro",
            "docs-contract"
        ]);
        let error = evaluate_partial_order_json(&value.to_string()).unwrap_err();
        assert!(error.contains("join_required_obligations mismatch"));
    }

    #[test]
    fn critical_join_claim_without_fresh_pass_is_unsupported() {
        let report = evaluate_partial_order_json(FIXTURE).expect("D6.10 fixture");
        let adversarial = trace(&report, "adversarial-linearization");
        assert!(adversarial.unsupported_current_claims.contains(&"joined:docs-contract".to_owned()));
        assert!(adversarial.unsupported_current_claims.contains(&"joined:policy-authority".to_owned()));
    }
}
