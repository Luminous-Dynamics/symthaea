use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct PhysicalState {
    id: String,
    role: String,
    pairing: f64,
    coherence: f64,
    integrity: f64,
    phase_status: String,
}
#[derive(Clone, Debug)]
struct Decision {
    id: String,
    losses: Vec<f64>,
}
#[derive(Clone, Debug)]
struct Resource {
    id: String,
    unit: String,
    capacity: u64,
    semantics: String,
}
#[derive(Clone, Debug)]
struct Edge {
    next_state: usize,
    probability: f64,
    outcomes: [f64; 2],
}
#[derive(Clone, Debug)]
struct Action {
    id: String,
    kind: String,
    specimen_id: String,
    reversibility: String,
    requires: Vec<String>,
    provides: Vec<String>,
    eligible: bool,
    exclusion_reason: String,
    consumption: Vec<u64>,
    transitions: Vec<Vec<Edge>>,
}
#[derive(Clone, Debug)]
struct Branch {
    first_outcome: Option<usize>,
    second_action: Option<usize>,
    loss: f64,
    terminal_policy: Vec<usize>,
    additional_consumption: Vec<u64>,
}
#[derive(Clone, Debug)]
struct Plan {
    first_action: Option<usize>,
    branches: Vec<Branch>,
    loss: f64,
    consumption: Vec<u64>,
}
#[derive(Clone, Debug)]
struct Instance {
    instance_id: String,
    specimen_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    initial: Vec<f64>,
    states: Vec<PhysicalState>,
    decisions: Vec<Decision>,
    resources: Vec<Resource>,
    reference_resource: usize,
    outcomes: [String; 2],
    capabilities: HashSet<String>,
    actions: Vec<Action>,
}

fn finite(value: &Value, context: &str) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() { return Err(format!("{context} must be finite")); }
    Ok(number)
}
fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn strings(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let rows = value.as_array().ok_or_else(|| format!("{context} must be an array"))?;
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(rows.len());
    for row in rows {
        let item = row.as_str().filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("{context} entries must be non-empty strings"))?;
        if !seen.insert(item.to_owned()) { return Err(format!("{context} contains duplicate {item}")); }
        result.push(item.to_owned());
    }
    Ok(result)
}
fn add_vectors(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    if left.len() != right.len() { return None; }
    left.iter().zip(right.iter()).map(|(a, b)| a.checked_add(*b)).collect()
}
fn fits(consumption: &[u64], resources: &[Resource]) -> bool {
    consumption.len() == resources.len()
        && consumption.iter().zip(resources.iter()).all(|(used, resource)| *used <= resource.capacity)
}
fn state_mean(mass: &[f64], states: &[PhysicalState], field: fn(&PhysicalState) -> f64) -> f64 {
    mass.iter().zip(states.iter()).map(|(p, state)| p * field(state)).sum()
}
fn terminal_decision(mass: &[f64], decisions: &[Decision]) -> (f64, usize) {
    let mut best_loss = f64::INFINITY;
    let mut best_index = 0;
    for (index, decision) in decisions.iter().enumerate() {
        let loss: f64 = mass.iter().zip(decision.losses.iter()).map(|(p, l)| p * l).sum();
        if loss < best_loss - 1.0e-12
            || ((loss - best_loss).abs() <= 1.0e-12 && decision.id < decisions[best_index].id) {
            best_loss = loss;
            best_index = index;
        }
    }
    (best_loss, best_index)
}
fn apply_action(mass: &[f64], action: &Action, state_count: usize) -> Vec<Vec<f64>> {
    let mut outputs = vec![vec![0.0; state_count]; 2];
    for (from, probability_mass) in mass.iter().enumerate() {
        if *probability_mass == 0.0 { continue; }
        for edge in &action.transitions[from] {
            for outcome in 0..2 {
                outputs[outcome][edge.next_state] +=
                    probability_mass * edge.probability * edge.outcomes[outcome];
            }
        }
    }
    outputs
}
fn evaluate_branch(
    mass_after_first_outcome: &[f64],
    first_outcome: usize,
    second_action: Option<usize>,
    actions: &[Action],
    decisions: &[Decision],
    state_count: usize,
    resource_count: usize,
) -> Branch {
    match second_action {
        None => {
            let (loss, decision) = terminal_decision(mass_after_first_outcome, decisions);
            Branch {
                first_outcome: Some(first_outcome),
                second_action: None,
                loss,
                terminal_policy: vec![decision],
                additional_consumption: vec![0; resource_count],
            }
        }
        Some(action_index) => {
            let outcomes = apply_action(mass_after_first_outcome, &actions[action_index], state_count);
            let mut loss = 0.0;
            let mut policy = Vec::with_capacity(2);
            for outcome_mass in &outcomes {
                let (branch_loss, decision) = terminal_decision(outcome_mass, decisions);
                loss += branch_loss;
                policy.push(decision);
            }
            Branch {
                first_outcome: Some(first_outcome),
                second_action: Some(action_index),
                loss,
                terminal_policy: policy,
                additional_consumption: actions[action_index].consumption.clone(),
            }
        }
    }
}

impl Instance {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|e| format!("schema JSON parse failed: {e}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|e| format!("fixture JSON parse failed: {e}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator = validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!("causal transition schema validation failed: {error} at {}", error.instance_path()));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("causal transition auditor accepts synthetic fixtures only".to_owned());
        }
        if text_field(&fixture, "contract")? != "SC-CAUSAL-TRANSITION-0.1" {
            return Err("unsupported causal transition contract".to_owned());
        }
        if fixture.get("horizon").and_then(Value::as_u64) != Some(2) {
            return Err("v0.1 exact causal planner requires horizon exactly 2".to_owned());
        }
        let instance_id = text_field(&fixture, "instance_id")?.to_owned();
        let specimen_id = text_field(&fixture, "specimen_id")?.to_owned();

        let state_rows = fixture.get("physical_states").and_then(Value::as_array)
            .ok_or_else(|| "physical_states must be an array".to_owned())?;
        if state_rows.len() < 2 || state_rows.len() > 8 {
            return Err("physical state count must be in [2,8]".to_owned());
        }
        let mut states = Vec::with_capacity(state_rows.len());
        let mut state_index = HashMap::new();
        for row in state_rows {
            let id = text_field(row, "state_id")?.to_owned();
            if state_index.insert(id.clone(), states.len()).is_some() {
                return Err(format!("duplicate physical state {id}"));
            }
            let pairing = finite(row.get("pairing_strength").ok_or("missing pairing_strength")?, "pairing_strength")?;
            let coherence = finite(row.get("phase_coherence").ok_or("missing phase_coherence")?, "phase_coherence")?;
            let integrity = finite(row.get("sample_integrity").ok_or("missing sample_integrity")?, "sample_integrity")?;
            if [pairing, coherence, integrity].iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err(format!("physical state {id} feature values must be in [0,1]"));
            }
            states.push(PhysicalState {
                id,
                role: text_field(row, "state_role")?.to_owned(),
                pairing,
                coherence,
                integrity,
                phase_status: text_field(row, "phase_status")?.to_owned(),
            });
        }

        let initial_rows = fixture.get("initial_state_distribution").and_then(Value::as_array)
            .ok_or_else(|| "initial_state_distribution must be an array".to_owned())?;
        let mut initial = vec![0.0; states.len()];
        let mut initial_seen = HashSet::new();
        let mut prior_sum = 0.0;
        for row in initial_rows {
            let id = text_field(row, "state_id")?;
            let index = *state_index.get(id).ok_or_else(|| format!("initial distribution references unknown state {id}"))?;
            if !initial_seen.insert(id.to_owned()) { return Err(format!("duplicate initial state {id}")); }
            let probability = finite(row.get("probability").ok_or("missing initial probability")?, "initial probability")?;
            if !(0.0..=1.0).contains(&probability) { return Err("initial probabilities must be in [0,1]".to_owned()); }
            initial[index] = probability;
            prior_sum += probability;
        }
        if (prior_sum - 1.0).abs() > 1.0e-9 { return Err(format!("initial state probabilities sum to {prior_sum}, not 1")); }

        let resource_rows = fixture.get("resources").and_then(Value::as_array)
            .ok_or_else(|| "resources must be an array".to_owned())?;
        let mut resources = Vec::with_capacity(resource_rows.len());
        let mut resource_index = HashMap::new();
        for row in resource_rows {
            let id = text_field(row, "resource_id")?.to_owned();
            if resource_index.insert(id.clone(), resources.len()).is_some() {
                return Err(format!("duplicate resource {id}"));
            }
            let capacity = row.get("capacity").and_then(Value::as_u64)
                .filter(|v| *v > 0).ok_or_else(|| format!("resource {id} capacity must be positive integer"))?;
            resources.push(Resource {
                id,
                unit: text_field(row, "unit")?.to_owned(),
                capacity,
                semantics: text_field(row, "semantics")?.to_owned(),
            });
        }
        if resources.is_empty() { return Err("at least one resource is required".to_owned()); }
        let reference_id = text_field(&fixture, "reference_resource_id")?;
        let reference_resource = *resource_index.get(reference_id)
            .ok_or_else(|| format!("unknown reference resource {reference_id}"))?;

        let decision_rows = fixture.get("decisions").and_then(Value::as_array)
            .ok_or_else(|| "decisions must be an array".to_owned())?;
        let mut decisions = Vec::with_capacity(decision_rows.len());
        let mut decision_ids = HashSet::new();
        for row in decision_rows {
            let id = text_field(row, "decision_id")?.to_owned();
            if !decision_ids.insert(id.clone()) { return Err(format!("duplicate decision {id}")); }
            let loss_rows = row.get("losses").and_then(Value::as_array)
                .ok_or_else(|| format!("decision {id} losses must be an array"))?;
            let mut losses = vec![None; states.len()];
            for loss_row in loss_rows {
                let state_id = text_field(loss_row, "state_id")?;
                let index = *state_index.get(state_id)
                    .ok_or_else(|| format!("decision {id} references unknown state {state_id}"))?;
                if losses[index].is_some() { return Err(format!("decision {id} duplicates state loss {state_id}")); }
                let loss = finite(loss_row.get("loss").ok_or("missing decision loss")?, "decision loss")?;
                if loss < 0.0 { return Err("decision loss must be non-negative".to_owned()); }
                losses[index] = Some(loss);
            }
            if losses.iter().any(Option::is_none) {
                return Err(format!("decision {id} must define a loss for every physical state"));
            }
            decisions.push(Decision { id, losses: losses.into_iter().map(Option::unwrap).collect() });
        }
        if decisions.len() < 2 { return Err("at least two terminal decisions are required".to_owned()); }
        decisions.sort_by(|a, b| a.id.cmp(&b.id));

        let outcome_values = strings(fixture.get("outcome_labels").ok_or("missing outcome_labels")?, "outcome_labels")?;
        if outcome_values.len() != 2 { return Err("exactly two outcome labels are required".to_owned()); }
        let outcomes = [outcome_values[0].clone(), outcome_values[1].clone()];
        let capabilities: HashSet<String> = strings(
            fixture.get("initial_capabilities").ok_or("missing initial_capabilities")?,
            "initial_capabilities",
        )?.into_iter().collect();

        let action_rows = fixture.get("actions").and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_rows.is_empty() || action_rows.len() > 8 {
            return Err("action count must be in [1,8]".to_owned());
        }
        let mut actions = Vec::with_capacity(action_rows.len());
        let mut action_ids = HashSet::new();
        for row in action_rows {
            let id = text_field(row, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) { return Err(format!("duplicate action {id}")); }
            let action_specimen = text_field(row, "specimen_id")?.to_owned();
            if action_specimen != specimen_id {
                return Err(format!("action {id} specimen identity does not match instance specimen"));
            }
            let requires = strings(row.get("requires_capabilities").ok_or("missing requires_capabilities")?, "requires_capabilities")?;
            let provides = strings(row.get("provides_capabilities").ok_or("missing provides_capabilities")?, "provides_capabilities")?;
            let eligible = row.get("eligible").and_then(Value::as_bool).ok_or("eligible must be boolean")?;
            let exclusion_reason = row.get("exclusion_reason").and_then(Value::as_str)
                .ok_or("exclusion_reason must be text")?.to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!("ineligible action {id} needs an exclusion reason"));
            }
            let mut consumption = vec![0_u64; resources.len()];
            for item in row.get("resource_consumption").and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} resource_consumption must be array"))? {
                let resource_id = text_field(item, "resource_id")?;
                let index = *resource_index.get(resource_id)
                    .ok_or_else(|| format!("action {id} references unknown resource {resource_id}"))?;
                if consumption[index] != 0 { return Err(format!("action {id} duplicates resource {resource_id}")); }
                let unit = text_field(item, "unit")?;
                if unit != resources[index].unit { return Err(format!("action {id} unit mismatch for {resource_id}")); }
                consumption[index] = item.get("quantity").and_then(Value::as_u64)
                    .filter(|v| *v > 0).ok_or_else(|| format!("action {id} resource quantity must be positive integer"))?;
            }
            let mut transition_rows: Vec<Option<Vec<Edge>>> = vec![None; states.len()];
            for transition_row in row.get("transition_rows").and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} transition_rows must be array"))? {
                let from_id = text_field(transition_row, "from_state_id")?;
                let from = *state_index.get(from_id)
                    .ok_or_else(|| format!("action {id} references unknown from state {from_id}"))?;
                if transition_rows[from].is_some() { return Err(format!("action {id} duplicates transition row {from_id}")); }
                let mut edges = Vec::new();
                let mut transition_sum = 0.0;
                for edge in transition_row.get("transition_outcomes").and_then(Value::as_array)
                    .ok_or_else(|| format!("action {id} outcomes must be array"))? {
                    let next_id = text_field(edge, "next_state_id")?;
                    let next_state = *state_index.get(next_id)
                        .ok_or_else(|| format!("action {id} references unknown next state {next_id}"))?;
                    let probability = finite(edge.get("probability").ok_or("missing transition probability")?, "transition probability")?;
                    if probability <= 0.0 || probability > 1.0 { return Err(format!("action {id} transition probability must be in (0,1]")); }
                    let values = edge.get("outcome_probabilities").and_then(Value::as_array)
                        .ok_or_else(|| format!("action {id} outcome_probabilities must be array"))?;
                    if values.len() != 2 { return Err(format!("action {id} requires two observation probabilities")); }
                    let p0 = finite(&values[0], "observation probability")?;
                    let p1 = finite(&values[1], "observation probability")?;
                    if !(0.0..=1.0).contains(&p0) || !(0.0..=1.0).contains(&p1)
                        || (p0 + p1 - 1.0).abs() > 1.0e-9 {
                        return Err(format!("action {id} observation probabilities are not normalized"));
                    }
                    transition_sum += probability;
                    edges.push(Edge { next_state, probability, outcomes: [p0, p1] });
                }
                if (transition_sum - 1.0).abs() > 1.0e-9 {
                    return Err(format!("action {id} transition probabilities for {from_id} sum to {transition_sum}"));
                }
                transition_rows[from] = Some(edges);
            }
            if transition_rows.iter().any(Option::is_none) {
                return Err(format!("action {id} must define a transition row for every physical state"));
            }
            actions.push(Action {
                id,
                kind: text_field(row, "action_kind")?.to_owned(),
                specimen_id: action_specimen,
                reversibility: text_field(row, "reversibility")?.to_owned(),
                requires,
                provides,
                eligible,
                exclusion_reason,
                consumption,
                transitions: transition_rows.into_iter().map(Option::unwrap).collect(),
            });
        }
        actions.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self {
            instance_id,
            specimen_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            initial,
            states,
            decisions,
            resources,
            reference_resource,
            outcomes,
            capabilities,
            actions,
        })
    }

    fn baseline(&self) -> Plan {
        let (loss, decision) = terminal_decision(&self.initial, &self.decisions);
        Plan {
            first_action: None,
            branches: vec![Branch {
                first_outcome: None,
                second_action: None,
                loss,
                terminal_policy: vec![decision],
                additional_consumption: vec![0; self.resources.len()],
            }],
            loss,
            consumption: vec![0; self.resources.len()],
        }
    }

    fn branch_better(&self, candidate: &Branch, incumbent: &Branch) -> bool {
        if candidate.loss < incumbent.loss - 1.0e-12 { return true; }
        if (candidate.loss - incumbent.loss).abs() > 1.0e-12 { return false; }
        match (candidate.second_action, incumbent.second_action) {
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (Some(a), Some(b)) => self.actions[a].id < self.actions[b].id,
            (None, None) => false,
        }
    }

    fn evaluate_first_action(&self, first: usize) -> Result<Option<Plan>, String> {
        let action = &self.actions[first];
        if !action.eligible || !action.requires.iter().all(|c| self.capabilities.contains(c)) {
            return Ok(None);
        }
        if !fits(&action.consumption, &self.resources) { return Ok(None); }
        let first_outcomes = apply_action(&self.initial, action, self.states.len());
        let mut available = self.capabilities.clone();
        for capability in &action.provides { available.insert(capability.clone()); }
        let mut branches = Vec::with_capacity(2);
        let mut loss = 0.0;
        let mut worst_branch_consumption = vec![0_u64; self.resources.len()];
        for (outcome_index, state_mass) in first_outcomes.iter().enumerate() {
            let mut best = evaluate_branch(
                state_mass, outcome_index, None, &self.actions, &self.decisions,
                self.states.len(), self.resources.len()
            );
            for second in 0..self.actions.len() {
                if second == first { continue; }
                let candidate_action = &self.actions[second];
                if !candidate_action.eligible
                    || !candidate_action.requires.iter().all(|c| available.contains(c)) {
                    continue;
                }
                let Some(path_consumption) = add_vectors(&action.consumption, &candidate_action.consumption) else {
                    return Err("resource integer overflow while composing actions".to_owned());
                };
                if !fits(&path_consumption, &self.resources) { continue; }
                let candidate = evaluate_branch(
                    state_mass, outcome_index, Some(second), &self.actions, &self.decisions,
                    self.states.len(), self.resources.len()
                );
                if self.branch_better(&candidate, &best) { best = candidate; }
            }
            loss += best.loss;
            for (maximum, quantity) in worst_branch_consumption.iter_mut()
                .zip(best.additional_consumption.iter()) {
                *maximum = (*maximum).max(*quantity);
            }
            branches.push(best);
        }
        let consumption = add_vectors(&action.consumption, &worst_branch_consumption)
            .ok_or_else(|| "resource integer overflow in worst-case branch envelope".to_owned())?;
        if !fits(&consumption, &self.resources) {
            return Err("componentwise worst-case branch resource envelope is infeasible".to_owned());
        }
        Ok(Some(Plan { first_action: Some(first), branches, loss, consumption }))
    }

    fn plan_action_count(&self, plan: &Plan) -> usize {
        plan.first_action.map_or(0, |_| 1)
            + plan.branches.iter().filter(|branch| branch.second_action.is_some()).count()
    }

    fn plan_signature(&self, plan: &Plan) -> String {
        let first = plan.first_action.map(|i| self.actions[i].id.as_str()).unwrap_or("STOP");
        let branches = plan.branches.iter().map(|branch| {
            branch.second_action.map(|i| self.actions[i].id.as_str()).unwrap_or("STOP")
        }).collect::<Vec<_>>().join(",");
        format!("{first}|{branches}")
    }

    fn better_plan(&self, candidate: &Plan, incumbent: &Plan) -> bool {
        if candidate.loss < incumbent.loss - 1.0e-12 { return true; }
        if (candidate.loss - incumbent.loss).abs() > 1.0e-12 { return false; }
        let candidate_count = self.plan_action_count(candidate);
        let incumbent_count = self.plan_action_count(incumbent);
        if candidate_count != incumbent_count { return candidate_count < incumbent_count; }
        self.plan_signature(candidate) < self.plan_signature(incumbent)
    }

    fn exact_plan(&self) -> Result<Plan, String> {
        let mut best = self.baseline();
        for index in 0..self.actions.len() {
            if let Some(candidate) = self.evaluate_first_action(index)? {
                if self.better_plan(&candidate, &best) { best = candidate; }
            }
        }
        Ok(best)
    }

    fn single_action_baseline(&self) -> Result<Plan, String> {
        let mut best = self.baseline();
        for index in 0..self.actions.len() {
            let action = &self.actions[index];
            if !action.eligible || !action.requires.iter().all(|c| self.capabilities.contains(c))
                || !fits(&action.consumption, &self.resources) {
                continue;
            }
            let outputs = apply_action(&self.initial, action, self.states.len());
            let mut branches = Vec::new();
            let mut loss = 0.0;
            for (outcome_index, mass) in outputs.iter().enumerate() {
                let branch = evaluate_branch(
                    mass, outcome_index, None, &self.actions, &self.decisions,
                    self.states.len(), self.resources.len()
                );
                loss += branch.loss;
                branches.push(branch);
            }
            let candidate = Plan {
                first_action: Some(index),
                branches,
                loss,
                consumption: action.consumption.clone(),
            };
            if self.better_plan(&candidate, &best) { best = candidate; }
        }
        Ok(best)
    }

    fn fixed_sequence_loss(&self, first: usize, second: usize) -> Result<Option<f64>, String> {
        if first == second { return Ok(None); }
        let a = &self.actions[first];
        let b = &self.actions[second];
        let mut available = self.capabilities.clone();
        for capability in &a.provides { available.insert(capability.clone()); }
        if !a.eligible || !b.eligible
            || !a.requires.iter().all(|c| self.capabilities.contains(c))
            || !b.requires.iter().all(|c| available.contains(c)) {
            return Ok(None);
        }
        let Some(total) = add_vectors(&a.consumption, &b.consumption) else {
            return Err("resource overflow in fixed sequence".to_owned());
        };
        if !fits(&total, &self.resources) { return Ok(None); }
        let first_outcomes = apply_action(&self.initial, a, self.states.len());
        let mut expected_loss = 0.0;
        for (outcome_index, mass) in first_outcomes.iter().enumerate() {
            let branch = evaluate_branch(
                mass, outcome_index, Some(second), &self.actions, &self.decisions,
                self.states.len(), self.resources.len()
            );
            expected_loss += branch.loss;
        }
        Ok(Some(expected_loss))
    }

    fn action_effect_vs_sham(&self, action_index: usize, sham_index: Option<usize>) -> Value {
        let post_mean = |index: usize, field: fn(&PhysicalState) -> f64| -> f64 {
            let action = &self.actions[index];
            let mut next = vec![0.0; self.states.len()];
            for (from, prior) in self.initial.iter().enumerate() {
                for edge in &action.transitions[from] {
                    next[edge.next_state] += prior * edge.probability;
                }
            }
            state_mean(&next, &self.states, field)
        };
        let identity_mean = |field: fn(&PhysicalState) -> f64| -> f64 {
            self.initial.iter().zip(self.states.iter()).map(|(p, state)| p * field(state)).sum()
        };
        let reference = sham_index.unwrap_or(action_index);
        let baseline_pairing = if sham_index.is_some() { post_mean(reference, |s| s.pairing) } else { identity_mean(|s| s.pairing) };
        let baseline_coherence = if sham_index.is_some() { post_mean(reference, |s| s.coherence) } else { identity_mean(|s| s.coherence) };
        let baseline_integrity = if sham_index.is_some() { post_mean(reference, |s| s.integrity) } else { identity_mean(|s| s.integrity) };
        json!({
            "action_id":self.actions[action_index].id,
            "action_kind":self.actions[action_index].kind,
            "pairing_strength_delta_vs_sham":post_mean(action_index, |s| s.pairing)-baseline_pairing,
            "phase_coherence_delta_vs_sham":post_mean(action_index, |s| s.coherence)-baseline_coherence,
            "sample_integrity_delta_vs_sham":post_mean(action_index, |s| s.integrity)-baseline_integrity,
            "interpretation":"Model-implied synthetic interventional contrast only; not an empirically identified causal effect."
        })
    }

    fn action_descriptor(&self, index: usize) -> Value {
        let action = &self.actions[index];
        let resource_consumption = self.resources.iter().enumerate()
            .filter_map(|(resource_index, resource)| {
                let quantity = action.consumption[resource_index];
                (quantity > 0).then(|| json!({
                    "resource_id":resource.id,
                    "unit":resource.unit,
                    "quantity":quantity
                }))
            }).collect::<Vec<_>>();
        json!({
            "action_id":action.id,
            "action_kind":action.kind,
            "specimen_id":action.specimen_id,
            "reversibility":action.reversibility,
            "requires_capabilities":action.requires,
            "provides_capabilities":action.provides,
            "resource_consumption":resource_consumption
        })
    }

    fn branch_receipt(&self, branch: &Branch) -> Value {
        let first_label = branch.first_outcome.map(|i| self.outcomes[i].as_str()).unwrap_or("no_action");
        let second_policy = branch.terminal_policy.iter().enumerate().map(|(i, decision)| json!({
            "second_outcome": if branch.second_action.is_some() { self.outcomes[i].as_str() } else { "not_applicable" },
            "terminal_decision": self.decisions[*decision].id
        })).collect::<Vec<_>>();
        let second_details = branch.second_action.map(|index| self.action_descriptor(index));
        json!({
            "first_outcome":first_label,
            "second_action":branch.second_action.map(|i| self.actions[i].id.clone()),
            "second_action_details":second_details,
            "expected_terminal_loss_contribution":branch.loss,
            "terminal_policy":second_policy,
            "additional_resource_consumption":branch.additional_consumption
        })
    }

    fn plan_receipt(&self, plan: &Plan) -> Value {
        let usage = self.resources.iter().enumerate().map(|(i, resource)| json!({
            "resource_id":resource.id,
            "unit":resource.unit,
            "capacity":resource.capacity,
            "componentwise_worst_case_consumption":plan.consumption[i],
            "remaining":resource.capacity-plan.consumption[i],
            "binding":resource.capacity==plan.consumption[i],
            "semantics":resource.semantics
        })).collect::<Vec<_>>();
        let action_ids = plan.first_action.map(|i| self.actions[i].id.clone());
        let first_details = plan.first_action.map(|i| self.action_descriptor(i));
        json!({
            "first_action":action_ids,
            "first_action_details":first_details,
            "reference_resource_id":self.resources[self.reference_resource].id,
            "branches":plan.branches.iter().map(|b| self.branch_receipt(b)).collect::<Vec<_>>(),
            "expected_terminal_decision_loss":plan.loss,
            "action_count_on_policy_tree":self.plan_action_count(plan),
            "componentwise_worst_case_resource_envelope":usage,
            "policy_signature":self.plan_signature(plan)
        })
    }

    fn validate_transition_mass(&self) -> Result<(), String> {
        for action in &self.actions {
            for (state_index, edges) in action.transitions.iter().enumerate() {
                let transition_sum: f64 = edges.iter().map(|edge| edge.probability).sum();
                if (transition_sum-1.0).abs() > 1.0e-9 {
                    return Err(format!("action {} transition row {} is not normalized", action.id, self.states[state_index].id));
                }
                for edge in edges {
                    let outcome_sum = edge.outcomes.iter().sum::<f64>();
                    if (outcome_sum-1.0).abs() > 1.0e-9 {
                        return Err(format!("action {} has unnormalized observation probability", action.id));
                    }
                }
            }
        }
        Ok(())
    }
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance = Instance::from_bytes(schema_bytes, fixture_bytes)?;
    instance.validate_transition_mass()?;
    let exact = instance.exact_plan()?;
    let myopic = instance.single_action_baseline()?;
    let sham_index = instance.actions.iter().position(|a| a.kind == "sham_control");
    let interventional_effects = instance.actions.iter().enumerate().map(|(i, _)| {
        instance.action_effect_vs_sham(i, sham_index)
    }).collect::<Vec<_>>();
    let mut path_comparisons = Vec::new();
    for first_id in ["pressure_step", "strain_step"] {
        for second_id in ["pressure_step", "strain_step"] {
            if first_id == second_id { continue; }
            let first = instance.actions.iter().position(|a| a.id == first_id);
            let second = instance.actions.iter().position(|a| a.id == second_id);
            if let (Some(a), Some(b)) = (first, second) {
                path_comparisons.push(json!({
                    "sequence":[first_id,second_id],
                    "expected_terminal_loss":instance.fixed_sequence_loss(a,b)?
                }));
            }
        }
    }
    let excluded = instance.actions.iter().filter(|a| !a.eligible).map(|a| json!({
        "action_id":a.id,"exclusion_reason":a.exclusion_reason
    })).collect::<Vec<_>>();
    Ok(json!({
        "status":"synthetic_causal_contract_pass",
        "contract":"SC-CAUSAL-TRANSITION-AUDIT-0.1",
        "instance_id":instance.instance_id,
        "specimen_id":instance.specimen_id,
        "schema_sha256":instance.schema_sha256,
        "fixture_sha256":instance.fixture_sha256,
        "horizon":2,
        "physical_states":instance.states.iter().map(|s|json!({
            "state_id":s.id,"state_role":s.role,"pairing_strength":s.pairing,
            "phase_coherence":s.coherence,"sample_integrity":s.integrity,
            "phase_status":s.phase_status
        })).collect::<Vec<_>>(),
        "initial_state_distribution":instance.initial.iter().enumerate().map(|(i,p)|json!({
            "state_id":instance.states[i].id,"probability":p
        })).collect::<Vec<_>>(),
        "resource_registry":instance.resources.iter().map(|r|json!({
            "resource_id":r.id,"unit":r.unit,"capacity":r.capacity,"semantics":r.semantics
        })).collect::<Vec<_>>(),
        "action_count":instance.actions.len(),
        "excluded_actions":excluded,
        "no_intervention_baseline":instance.plan_receipt(&instance.baseline()),
        "myopic_single_action_baseline":instance.plan_receipt(&myopic),
        "exact_adaptive_two_step_policy":instance.plan_receipt(&exact),
        "path_dependence_comparisons":path_comparisons,
        "model_implied_interventional_contrasts_vs_sham":interventional_effects,
        "claim_ceiling":"Synthetic state-transition and adaptive planning contract only. No physical causal effect, superconducting property, instrument recipe, laboratory authorization or application result is established."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index+1 >= args.len() { usage(&args[0]); process::exit(2); }
                let path = PathBuf::from(&args[index+1]);
                match args[index].as_str() {
                    "--schema" => schema_path=Some(path),
                    "--fixture" => fixture_path=Some(path),
                    "--out" => output_path=Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => { usage(&args[0]); return; }
            unknown => { eprintln!("unknown argument: {unknown}"); usage(&args[0]); process::exit(2); }
        }
    }
    let (Some(schema_path),Some(fixture_path))=(schema_path,fixture_path) else {
        usage(&args[0]); process::exit(2);
    };
    let receipt=audit(&read(&schema_path),&read(&fixture_path)).unwrap_or_else(|e| {
        eprintln!("SC-CAUSAL-TRANSITION-FAIL: {e}");
        process::exit(1);
    });
    let rendered=serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path)=output_path {
        if let Err(e)=fs::write(&path,format!("{rendered}\n")) {
            eprintln!("could not write {}: {e}",path.display());
            process::exit(1);
        }
    } else { println!("{rendered}"); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs_for(filename: &str) -> (Vec<u8>,Vec<u8>) {
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_causal_transition_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research").join(filename)).unwrap()
        )
    }

    #[test]
    fn pressure_then_strain_is_better_than_strain_then_pressure() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_v0_1.json");
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let pressure=instance.actions.iter().position(|a|a.id=="pressure_step").unwrap();
        let strain=instance.actions.iter().position(|a|a.id=="strain_step").unwrap();
        let forward=instance.fixed_sequence_loss(pressure,strain).unwrap().unwrap();
        let reverse=instance.fixed_sequence_loss(strain,pressure).unwrap().unwrap();
        assert!(forward+0.25<reverse);
    }

    #[test]
    fn adaptive_policy_chooses_strain_after_supportive_pressure_observation() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_v0_1.json");
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let plan=instance.exact_plan().unwrap();
        assert_eq!(instance.actions[plan.first_action.unwrap()].id,"pressure_step");
        let supportive=plan.branches.iter().find(|b|b.first_outcome==Some(0)).unwrap();
        assert_eq!(instance.actions[supportive.second_action.unwrap()].id,"strain_step");
    }

    #[test]
    fn destructive_information_does_not_override_specimen_preservation_value() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_preserve_v0_1.json");
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let plan=instance.exact_plan().unwrap();
        assert!(plan.first_action.is_none());
        let baseline=instance.baseline();
        assert!((plan.loss-baseline.loss).abs()<1.0e-12);
    }

    #[test]
    fn specimen_identity_and_resource_constraints_fail_closed() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_v0_1.json");
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["actions"][0]["specimen_id"]=json!("different_specimen");
        assert!(Instance::from_bytes(&schema,&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["actions"][0]["resource_consumption"][0]["unit"]=json!("hour");
        assert!(Instance::from_bytes(&schema,&serde_json::to_vec(&value).unwrap()).is_err());
    }
    #[test]
    fn pairing_strength_and_phase_coherence_are_not_collapsed_into_one_score() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_v0_1.json");
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let strain=instance.actions.iter().position(|a|a.id=="strain_step").unwrap();
        let sham=instance.actions.iter().position(|a|a.id=="sham_control");
        let effect=instance.action_effect_vs_sham(strain,sham);
        let pairing=effect["pairing_strength_delta_vs_sham"].as_f64().unwrap();
        let coherence=effect["phase_coherence_delta_vs_sham"].as_f64().unwrap();
        assert!(pairing>0.0);
        assert!(coherence<0.0);
    }

    #[test]
    fn every_action_receipt_preserves_specimen_identity_and_reversibility() {
        let (schema,fixture)=inputs_for("superconductivity_causal_transition_fixture_v0_1.json");
        let receipt=audit(&schema,&fixture).unwrap();
        let details=&receipt["exact_adaptive_two_step_policy"]["first_action_details"];
        assert_eq!(details["specimen_id"],"specimen_A");
        assert!(details["reversibility"].as_str().is_some());
        assert!(receipt["exact_adaptive_two_step_policy"]["reference_resource_id"].as_str().is_some());
    }

}
