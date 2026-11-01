// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural verifier for PITB's two-level proof graph.
//!
//! Layer one is an AND/OR hypergraph within each declaration. Layer two is a
//! lemma-dependency DAG across declarations. This validates identities,
//! acyclicity, status closure, the symbolic ready frontier and resource
//! accounting. It does not authenticate receipt bytes or prove Lean terms.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde_json::Value;

const VERSION: &str = "pitb.proof-graph-state.v0";
const MAX_GOAL_GROUP: usize = 32;

#[derive(Debug, Clone)]
pub struct ProofGraphSummary {
    pub task_id: String,
    pub declaration_count: usize,
    pub goal_count: usize,
    pub proof_step_count: usize,
    pub ready_frontier_goal_ids: Vec<String>,
    pub selected_goal_ids: Vec<String>,
    pub all_roots_stable: bool,
    pub evidence_class: String,
}

fn field<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value> {
    value
        .pointer(pointer)
        .with_context(|| format!("missing proof graph field {pointer}"))
}

fn string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    field(value, pointer)?
        .as_str()
        .with_context(|| format!("{pointer} must be a string"))
}

fn integer(value: &Value, pointer: &str) -> Result<u64> {
    field(value, pointer)?
        .as_u64()
        .with_context(|| format!("{pointer} must be a nonnegative integer"))
}

fn boolean(value: &Value, pointer: &str) -> Result<bool> {
    field(value, pointer)?
        .as_bool()
        .with_context(|| format!("{pointer} must be boolean"))
}

fn array<'a>(value: &'a Value, pointer: &str) -> Result<&'a Vec<Value>> {
    field(value, pointer)?
        .as_array()
        .with_context(|| format!("{pointer} must be an array"))
}

fn string_array(value: &Value, pointer: &str) -> Result<Vec<String>> {
    let rows = array(value, pointer)?;
    let mut result = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let id = row
            .as_str()
            .with_context(|| format!("{pointer}[{index}] must be a string"))?;
        if id.trim().is_empty() {
            bail!("{pointer}[{index}] cannot be empty");
        }
        result.push(id.to_owned());
    }
    if result.iter().collect::<BTreeSet<_>>().len() != result.len() {
        bail!("{pointer} contains duplicate identifiers");
    }
    Ok(result)
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_digest(value: &str, pointer: &str, length: usize) -> Result<()> {
    if !lower_hex(value, length) {
        bail!("{pointer} must be {length} lowercase hexadecimal characters");
    }
    Ok(())
}

fn optional_digest(value: &Value, pointer: &str) -> Result<bool> {
    match field(value, pointer)? {
        Value::Null => Ok(false),
        Value::String(text) => {
            check_digest(text, pointer, 64)?;
            Ok(true)
        }
        _ => bail!("{pointer} must be a digest or null"),
    }
}

fn require_sorted(values: &[String], pointer: &str) -> Result<()> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("{pointer} must be lexically sorted and unique");
    }
    Ok(())
}

fn visit_declaration(
    node: &str,
    graph: &BTreeMap<String, Vec<String>>,
    states: &mut BTreeMap<String, u8>,
) -> Result<()> {
    match states.get(node).copied().unwrap_or(0) {
        1 => bail!("lemma dependency cycle includes declaration {node}"),
        2 => return Ok(()),
        _ => {}
    }
    states.insert(node.to_owned(), 1);
    if let Some(children) = graph.get(node) {
        for child in children {
            visit_declaration(child, graph, states)?;
        }
    }
    states.insert(node.to_owned(), 2);
    Ok(())
}

fn visit_goal(
    node: &str,
    graph: &BTreeMap<String, BTreeSet<String>>,
    states: &mut BTreeMap<String, u8>,
) -> Result<()> {
    match states.get(node).copied().unwrap_or(0) {
        1 => bail!("intra-declaration proof hypergraph cycle includes goal {node}"),
        2 => return Ok(()),
        _ => {}
    }
    states.insert(node.to_owned(), 1);
    if let Some(children) = graph.get(node) {
        for child in children {
            visit_goal(child, graph, states)?;
        }
    }
    states.insert(node.to_owned(), 2);
    Ok(())
}

pub fn validate_value(value: &Value) -> Result<ProofGraphSummary> {
    if string(value, "/schema_version")? != VERSION {
        bail!("unsupported proof graph schema version");
    }
    let evidence_class = string(value, "/evidence_class")?;
    if !matches!(evidence_class, "synthetic_contract_fixture" | "measured_snapshot") {
        bail!("unknown proof graph evidence class");
    }
    let task_id = string(value, "/task/task_id")?.to_owned();
    check_digest(&task_id, "/task/task_id", 64)?;
    check_digest(string(value, "/task/repository_commit")?, "/task/repository_commit", 40)?;
    check_digest(string(value, "/task/environment_sha256")?, "/task/environment_sha256", 64)?;
    check_digest(string(value, "/task/target_manifest_sha256")?, "/task/target_manifest_sha256", 64)?;

    let declaration_rows = array(value, "/declarations")?;
    let mut declarations = BTreeMap::<String, Value>::new();
    let mut declaration_graph = BTreeMap::<String, Vec<String>>::new();
    for (index, row) in declaration_rows.iter().enumerate() {
        let prefix = format!("/declarations/{index}");
        let id = string(row, &format!("{prefix}/declaration_id"))?.to_owned();
        if id.trim().is_empty() || declarations.insert(id.clone(), row.clone()).is_some() {
            bail!("declaration IDs must be nonempty and unique");
        }
        if !matches!(string(row, &format!("{prefix}/kind"))?, "target" | "helper") {
            bail!("declaration {id} has an unsupported kind");
        }
        check_digest(
            string(row, &format!("{prefix}/statement_sha256"))?,
            &format!("{prefix}/statement_sha256"),
            64,
        )?;
        let lifecycle = string(row, &format!("{prefix}/lifecycle_state"))?;
        if !matches!(lifecycle, "open" | "provisional" | "stable" | "discarded") {
            bail!("declaration {id} has an unsupported lifecycle state");
        }
        let dependencies = string_array(row, &format!("{prefix}/dependency_ids"))?;
        require_sorted(&dependencies, &format!("{prefix}/dependency_ids"))?;
        if dependencies.iter().any(|dependency| dependency == &id) {
            bail!("declaration {id} cannot depend on itself");
        }
        for dependency in &dependencies {
            if !declarations.contains_key(dependency) && !declaration_rows.iter().any(|item| {
                item.get("declaration_id").and_then(Value::as_str) == Some(dependency.as_str())
            }) {
                bail!("declaration {id} references missing dependency {dependency}");
            }
        }
        let proof_present = optional_digest(row, &format!("{prefix}/proof_receipt_sha256"))?;
        let kernel_verified = boolean(row, &format!("{prefix}/kernel_verified"))?;
        let contains_sorry = boolean(row, &format!("{prefix}/contains_sorry_ax"))?;
        let axioms = string_array(row, &format!("{prefix}/axiom_names"))?;
        require_sorted(&axioms, &format!("{prefix}/axiom_names"))?;
        if lifecycle == "stable" {
            if !proof_present || !kernel_verified || contains_sorry {
                bail!("stable declaration {id} needs a proof receipt, kernel verification and no sorryAx");
            }
        } else if proof_present || kernel_verified {
            bail!("non-stable declaration {id} cannot carry stable proof certification");
        }
        declaration_graph.insert(id, dependencies);
    }

    // A second pass validates cross-node status constraints after every node exists.
    for (id, declaration) in &declarations {
        let lifecycle = string(declaration, "/lifecycle_state")?;
        let dependencies = declaration_graph.get(id).context("dependency graph node missing")?;
        for dependency in dependencies {
            let dependency_node = declarations
                .get(dependency)
                .with_context(|| format!("missing dependency declaration {dependency}"))?;
            let dependency_lifecycle = string(dependency_node, "/lifecycle_state")?;
            if lifecycle == "stable" && dependency_lifecycle != "stable" {
                bail!("stable declaration {id} depends on non-stable declaration {dependency}");
            }
            if lifecycle != "discarded" && dependency_lifecycle == "discarded" {
                bail!("active declaration {id} still depends on discarded declaration {dependency}");
            }
        }
    }

    let mut declaration_states = BTreeMap::new();
    for id in declarations.keys() {
        visit_declaration(id, &declaration_graph, &mut declaration_states)?;
    }

    // Discarding a helper invalidates every transitive dependent; no active descendant survives.
    for (id, declaration) in &declarations {
        if string(declaration, "/lifecycle_state")? != "discarded" {
            continue;
        }
        for (dependent_id, dependent) in &declarations {
            if dependent_id == id || string(dependent, "/lifecycle_state")? == "discarded" {
                continue;
            }
            let mut stack = vec![dependent_id.clone()];
            let mut visited = BTreeSet::new();
            while let Some(current) = stack.pop() {
                if !visited.insert(current.clone()) {
                    continue;
                }
                for dependency in declaration_graph.get(&current).into_iter().flatten() {
                    if dependency == id {
                        bail!("discarded declaration {id} has active transitive dependent {dependent_id}");
                    }
                    stack.push(dependency.clone());
                }
            }
        }
    }

    let roots = string_array(value, "/root_declaration_ids")?;
    require_sorted(&roots, "/root_declaration_ids")?;
    if roots.is_empty() {
        bail!("at least one root declaration is required");
    }
    for root in &roots {
        let declaration = declarations
            .get(root)
            .with_context(|| format!("root declaration {root} is missing"))?;
        if string(declaration, "/kind")? != "target" {
            bail!("root declaration {root} must be a target theorem, not a helper");
        }
        if string(declaration, "/lifecycle_state")? == "discarded" {
            bail!("root declaration {root} cannot be discarded");
        }
    }

    let goal_rows = array(value, "/goal_nodes")?;
    let mut goals = BTreeMap::<String, Value>::new();
    let mut goal_graph = BTreeMap::<String, BTreeSet<String>>::new();
    for (index, row) in goal_rows.iter().enumerate() {
        let prefix = format!("/goal_nodes/{index}");
        let id = string(row, &format!("{prefix}/goal_id"))?.to_owned();
        if id.trim().is_empty() || goals.insert(id.clone(), row.clone()).is_some() {
            bail!("goal IDs must be nonempty and unique");
        }
        let declaration_id = string(row, &format!("{prefix}/declaration_id"))?;
        if !declarations.contains_key(declaration_id) {
            bail!("goal {id} references missing declaration {declaration_id}");
        }
        if string(&declarations[declaration_id], "/lifecycle_state")? == "discarded"
            && string(row, &format!("{prefix}/status"))? != "discarded"
        {
            bail!("goal {id} is active beneath a discarded declaration");
        }
        check_digest(
            string(row, &format!("{prefix}/normalized_goal_sha256"))?,
            &format!("{prefix}/normalized_goal_sha256"),
            64,
        )?;
        let status = string(row, &format!("{prefix}/status"))?;
        if !matches!(status, "open" | "closed" | "discarded") {
            bail!("goal {id} has unsupported status");
        }
        let proof = optional_digest(row, &format!("{prefix}/closure_proof_sha256"))?;
        let receipt = optional_digest(row, &format!("{prefix}/closure_receipt_sha256"))?;
        let kernel = boolean(row, &format!("{prefix}/closure_kernel_verified"))?;
        let contains_sorry = boolean(row, &format!("{prefix}/closure_contains_sorry_ax"))?;
        match status {
            "closed" if !(proof && receipt && kernel && !contains_sorry) => {
                bail!("closed goal {id} needs exact proof/receipt digests, kernel verification and no sorryAx");
            }
            "open" if proof || receipt || kernel || contains_sorry => {
                bail!("open goal {id} cannot carry a closed-goal certificate");
            }
            "discarded" if proof || receipt || kernel => {
                bail!("discarded goal {id} cannot remain proof-certified");
            }
            _ => {}
        }
        goal_graph.insert(id, BTreeSet::new());
    }

    let step_rows = array(value, "/proof_steps")?;
    let mut step_ids = BTreeSet::new();
    let mut applied_steps_by_goal = BTreeMap::<String, Vec<(String, Vec<String>)>>::new();
    for (index, row) in step_rows.iter().enumerate() {
        let prefix = format!("/proof_steps/{index}");
        let step_id = string(row, &format!("{prefix}/step_id"))?.to_owned();
        if step_id.trim().is_empty() || !step_ids.insert(step_id.clone()) {
            bail!("proof step IDs must be nonempty and unique");
        }
        let parent_id = string(row, &format!("{prefix}/parent_goal_id"))?;
        let parent = goals
            .get(parent_id)
            .with_context(|| format!("proof step {step_id} references missing parent goal {parent_id}"))?;
        let state_sha = string(row, &format!("{prefix}/goal_state_sha256"))?;
        check_digest(state_sha, &format!("{prefix}/goal_state_sha256"), 64)?;
        if state_sha != string(parent, "/normalized_goal_sha256")? {
            bail!("proof step {step_id} was proposed for a different goal-state identity");
        }
        let status = string(row, &format!("{prefix}/status"))?;
        let subgoals = string_array(row, &format!("{prefix}/subgoal_ids"))?;
        require_sorted(&subgoals, &format!("{prefix}/subgoal_ids"))?;
        let transition = optional_digest(row, &format!("{prefix}/transition_receipt_sha256"))?;
        let terminal_proof = optional_digest(row, &format!("{prefix}/terminal_proof_sha256"))?;
        let terminal_receipt = optional_digest(row, &format!("{prefix}/terminal_receipt_sha256"))?;
        let checker_id = field(row, &format!("{prefix}/transition_checker_id"))?;
        let kernel = boolean(row, &format!("{prefix}/terminal_kernel_verified"))?;
        let contains_sorry = boolean(row, &format!("{prefix}/terminal_contains_sorry_ax"))?;

        match status {
            "applied" => {
                if !transition || checker_id.as_str().filter(|id| !id.trim().is_empty()).is_none() {
                    bail!("applied proof step {step_id} needs a transition checker and receipt");
                }
                let parent_decl = string(parent, "/declaration_id")?;
                for subgoal_id in &subgoals {
                    let subgoal = goals
                        .get(subgoal_id)
                        .with_context(|| format!("proof step {step_id} references missing subgoal {subgoal_id}"))?;
                    if string(subgoal, "/declaration_id")? != parent_decl {
                        bail!("intra-declaration hyperedge {step_id} cannot cross declaration boundaries");
                    }
                    goal_graph
                        .get_mut(parent_id)
                        .context("parent goal missing from graph")?
                        .insert(subgoal_id.clone());
                }
                if subgoals.is_empty() {
                    if !(terminal_proof && terminal_receipt && kernel && !contains_sorry) {
                        bail!("terminal applied step {step_id} needs a kernel-verified proof receipt and no sorryAx");
                    }
                } else if terminal_proof || terminal_receipt || kernel || contains_sorry {
                    bail!("nonterminal applied step {step_id} cannot claim a completed kernel proof");
                }
                applied_steps_by_goal.insert(
                    parent_id.to_owned(),
                    applied_steps_by_goal
                        .get(parent_id)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .chain(std::iter::once((step_id, subgoals)))
                        .collect(),
                );
            }
            "candidate" | "rejected" | "failed" => {
                if !subgoals.is_empty()
                    || transition
                    || terminal_proof
                    || terminal_receipt
                    || kernel
                    || contains_sorry
                    || !checker_id.is_null()
                {
                    bail!("unapplied proof step {step_id} cannot claim verified transitions or subgoals");
                }
            }
            _ => bail!("proof step {step_id} has unsupported status"),
        }
    }

    let mut goal_states = BTreeMap::new();
    for id in goals.keys() {
        visit_goal(id, &goal_graph, &mut goal_states)?;
    }

    for (goal_id, goal) in &goals {
        let status = string(goal, "/status")?;
        if status == "closed" {
            let witness = applied_steps_by_goal.get(goal_id).is_some_and(|steps| {
                steps.iter().any(|(_, subgoals)| {
                    subgoals.iter().all(|subgoal| {
                        goals
                            .get(subgoal)
                            .and_then(|node| node.get("status"))
                            .and_then(Value::as_str)
                            == Some("closed")
                    })
                })
            });
            if !witness {
                bail!("closed goal {goal_id} has no applied AND/OR proof hyperedge with all subgoals closed");
            }
            let terminal_witness = applied_steps_by_goal
                .get(goal_id)
                .into_iter()
                .flatten()
                .find(|(_, subgoals)| subgoals.is_empty());
            if let Some((step_id, _)) = terminal_witness {
                let terminal = step_rows
                    .iter()
                    .find(|step| step.get("step_id").and_then(Value::as_str) == Some(step_id.as_str()))
                    .context("terminal proof step missing")?;
                if string(terminal, "/terminal_proof_sha256")? != string(goal, "/closure_proof_sha256")?
                    || string(terminal, "/terminal_receipt_sha256")? != string(goal, "/closure_receipt_sha256")?
                {
                    bail!("closed goal {goal_id} closure proof does not match its terminal Lean proof step");
                }
            }
        } else if status == "open" {
            let has_closed_witness = applied_steps_by_goal.get(goal_id).is_some_and(|steps| {
                steps.iter().any(|(_, subgoals)| {
                    subgoals.iter().all(|subgoal| {
                        goals
                            .get(subgoal)
                            .and_then(|node| node.get("status"))
                            .and_then(Value::as_str)
                            == Some("closed")
                    })
                })
            });
            if has_closed_witness {
                bail!("open goal {goal_id} has an already completed proof hyperedge");
            }
        }
    }

    for (declaration_id, declaration) in &declarations {
        let lifecycle = string(declaration, "/lifecycle_state")?;
        if lifecycle == "discarded"
            && goals.values().any(|goal| {
                goal.get("declaration_id").and_then(Value::as_str) == Some(declaration_id)
                    && goal.get("status").and_then(Value::as_str) != Some("discarded")
            })
        {
            bail!("discarded declaration {declaration_id} still has non-discarded goal nodes");
        }
        if lifecycle == "stable" {
            let declaration_goals: Vec<&Value> = goals
                .values()
                .filter(|goal| goal.get("declaration_id").and_then(Value::as_str) == Some(declaration_id))
                .collect();
            if declaration_goals.is_empty()
                || declaration_goals.iter().any(|goal| goal.get("status").and_then(Value::as_str) != Some("closed"))
            {
                bail!("stable declaration {declaration_id} has open or missing goal nodes");
            }
        }
    }

    // Build the symbolic ready frontier without model inference.
    let mut ready_frontier = Vec::new();
    for (goal_id, goal) in &goals {
        if string(goal, "/status")? != "open" {
            continue;
        }
        let declaration_id = string(goal, "/declaration_id")?;
        let declaration = declarations.get(declaration_id).context("goal declaration missing")?;
        if !matches!(string(declaration, "/lifecycle_state")?, "open" | "provisional") {
            continue;
        }
        let dependencies = declaration_graph.get(declaration_id).context("declaration dependencies missing")?;
        let ready = dependencies.iter().all(|dependency| {
            declarations
                .get(dependency)
                .and_then(|node| node.get("lifecycle_state"))
                .and_then(Value::as_str)
                .is_some_and(|state| matches!(state, "provisional" | "stable"))
        });
        if ready {
            ready_frontier.push(goal_id.clone());
        }
    }
    ready_frontier.sort();

    let declared_frontier = string_array(value, "/scheduler_decision/ready_frontier_goal_ids")?;
    require_sorted(&declared_frontier, "/scheduler_decision/ready_frontier_goal_ids")?;
    if declared_frontier != ready_frontier {
        bail!("declared ready frontier differs from the independently recomputed dependency-ready frontier");
    }
    let selected = string_array(value, "/scheduler_decision/selected_goal_ids")?;
    require_sorted(&selected, "/scheduler_decision/selected_goal_ids")?;
    if selected.is_empty() || selected.len() > MAX_GOAL_GROUP {
        bail!("selected goal group must contain 1..={MAX_GOAL_GROUP} goals");
    }
    let ready: BTreeSet<&str> = ready_frontier.iter().map(String::as_str).collect();
    if selected.iter().any(|goal| !ready.contains(goal.as_str())) {
        bail!("scheduler selected a goal outside the dependency-ready frontier");
    }

    let specialist = string(value, "/scheduler_decision/specialist")?;
    let reasoning_effort = string(value, "/scheduler_decision/reasoning_effort")?;
    let max_specialist_calls = integer(value, "/scheduler_decision/max_specialist_calls")?;
    check_digest(
        string(value, "/scheduler_decision/routing_policy_sha256")?,
        "/scheduler_decision/routing_policy_sha256",
        64,
    )?;
    check_digest(
        string(value, "/scheduler_decision/decision_receipt_sha256")?,
        "/scheduler_decision/decision_receipt_sha256",
        64,
    )?;
    if !matches!(specialist, "deterministic_automation" | "proof_writer" | "lemma_planner" | "recovery") {
        bail!("unsupported specialist choice");
    }
    if !matches!(reasoning_effort, "none" | "low" | "medium" | "high") {
        bail!("unsupported reasoning effort");
    }
    let actual = field(value, "/resource_ledger/actual")?;
    let budget = field(value, "/resource_ledger/budget")?;
    let router_calls = integer(actual, "/router_model_calls")?;
    let specialist_calls = integer(actual, "/specialist_model_calls")?;
    let model_calls = integer(actual, "/model_calls")?;
    if router_calls.checked_add(specialist_calls) != Some(model_calls) {
        bail!("router and specialist model-call counts do not reconcile");
    }
    if specialist == "deterministic_automation" {
        if specialist_calls != 0 || reasoning_effort != "none" || max_specialist_calls != 0 {
            bail!("deterministic automation must use no specialist model calls, no reasoning effort and zero specialist-call budget");
        }
    } else if specialist_calls == 0 || reasoning_effort == "none" || max_specialist_calls == 0 {
        bail!("model specialist requires positive call budget, calls and reasoning effort");
    }
    let router_cost = integer(actual, "/router_cost_microusd")?;
    let specialist_cost = integer(actual, "/specialist_cost_microusd")?;
    let lean_cost = integer(actual, "/lean_cost_microusd")?;
    let total_cost = integer(actual, "/cost_microusd")?;
    if router_cost
        .checked_add(specialist_cost)
        .and_then(|sum| sum.checked_add(lean_cost))
        != Some(total_cost)
    {
        bail!("router, specialist and Lean costs do not reconcile to total cost");
    }
    for (field_name, pointer) in [
        ("model_calls", "/model_calls"),
        ("tool_calls", "/tool_calls"),
        ("tokens", "/tokens"),
        ("wall_ms", "/wall_ms"),
        ("lean_cpu_ms", "/lean_cpu_ms"),
        ("cost_microusd", "/cost_microusd"),
    ] {
        if integer(actual, pointer)? > integer(budget, &format!("/{field_name}"))? {
            bail!("proof graph resource ledger exceeds {field_name} budget");
        }
    }

    let all_roots_stable = roots.iter().all(|root| {
        declarations
            .get(root)
            .and_then(|node| node.get("lifecycle_state"))
            .and_then(Value::as_str)
            == Some("stable")
    });
    Ok(ProofGraphSummary {
        task_id,
        declaration_count: declarations.len(),
        goal_count: goals.len(),
        proof_step_count: step_rows.len(),
        ready_frontier_goal_ids: ready_frontier,
        selected_goal_ids: selected,
        all_roots_stable,
        evidence_class: evidence_class.to_owned(),
    })
}
