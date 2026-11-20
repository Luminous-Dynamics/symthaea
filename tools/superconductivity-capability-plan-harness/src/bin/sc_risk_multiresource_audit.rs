use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct WorldState {
    id: String,
    prior: f64,
    model_family: String,
    phase_realization: String,
    evidence_qualification: String,
    engineering_use: String,
}
#[derive(Clone, Debug)]
struct NuisanceState { id: String, prior: f64 }
#[derive(Clone, Debug)]
struct Decision { id: String, losses: Vec<f64> }
#[derive(Clone, Debug)]
struct Resource { id: String, unit: String, capacity: u64 }
#[derive(Clone, Debug)]
struct Action {
    id: String,
    action_type: String,
    requires: Vec<String>,
    provides: Vec<String>,
    eligible: bool,
    exclusion_reason: String,
    consumption: Vec<u64>,
    // [world state][one shared nuisance state][binary outcome]
    likelihoods: Vec<Vec<[f64; 2]>>,
}
#[derive(Clone, Debug)]
struct Plan {
    ordered_indices: Vec<usize>,
    consumption: Vec<u64>,
    expected_risk: f64,
    cvar_risk: f64,
    expected_regret: f64,
    cvar_regret: f64,
    expected_evsi: f64,
    cvar_evsi: f64,
    mutual_information_nats: f64,
    cvar_eta: f64,
    expected_policy: Vec<usize>,
    policy: Vec<usize>,
    outcome_count: usize,
}
#[derive(Clone, Debug)]
struct Instance {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    alpha: f64,
    reference_resource_index: usize,
    worlds: Vec<WorldState>,
    nuisance_states: Vec<NuisanceState>,
    decisions: Vec<Decision>,
    resources: Vec<Resource>,
    outcome_labels: [String; 2],
    initial_capabilities: HashSet<String>,
    actions: Vec<Action>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty text field {key}"))
}
fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() { return Err(format!("{context} must be finite")); }
    Ok(number)
}
fn unique_strings(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let values = value.as_array().ok_or_else(|| format!("{context} must be an array"))?;
    let mut seen = HashSet::with_capacity(values.len());
    let mut output = Vec::with_capacity(values.len());
    for item in values {
        let text = item.as_str().filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("{context} entries must be non-empty strings"))?;
        if !seen.insert(text.to_owned()) {
            return Err(format!("{context} contains duplicate value {text}"));
        }
        output.push(text.to_owned());
    }
    Ok(output)
}
fn entropy(probabilities: &[f64]) -> f64 {
    probabilities.iter().copied().filter(|p| *p > 0.0)
        .map(|p| -p * p.ln()).sum()
}

// Exact finite-support upper-tail CVaR. The Rockafellar-Uryasev form includes
// fractional probability mass at the tail boundary.
fn weighted_cvar(values: &[f64], probabilities: &[f64], alpha: f64) -> Result<f64, String> {
    if values.is_empty() || values.len() != probabilities.len() {
        return Err("CVaR values and probabilities must be non-empty and aligned".to_owned());
    }
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("CVaR alpha must be in [0, 1)".to_owned());
    }
    let mut total_probability = 0.0;
    for (value, probability) in values.iter().zip(probabilities.iter()) {
        if !value.is_finite() || !probability.is_finite() || *probability < 0.0 {
            return Err("CVaR support and probabilities must be finite and non-negative".to_owned());
        }
        total_probability += probability;
    }
    if (total_probability - 1.0).abs() > 1.0e-9 {
        return Err(format!("CVaR probabilities sum to {total_probability}, not 1"));
    }
    let mut thresholds = values.to_vec();
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
    let mut best = f64::INFINITY;
    for eta in thresholds {
        let excess: f64 = values.iter().zip(probabilities.iter())
            .map(|(value, probability)| probability * (value - eta).max(0.0)).sum();
        best = best.min(eta + excess / (1.0 - alpha));
    }
    if !best.is_finite() { return Err("CVaR calculation is non-finite".to_owned()); }
    Ok(best)
}

// Precommitment CVaR over the joint (world, observation) distribution.
// For fixed eta, each outcome's decision minimizes its joint probability-
// weighted positive excess loss. This is not E[CVaR(loss | observation)].
fn precommitment_cvar(
    masses: &[Vec<f64>],
    decisions: &[Decision],
    alpha: f64,
) -> Result<(f64, f64, Vec<usize>), String> {
    if masses.is_empty() || decisions.is_empty() {
        return Err("CVaR policy needs observations and decisions".to_owned());
    }
    let world_count = masses[0].len();
    if world_count == 0 || masses.iter().any(|row| row.len() != world_count) {
        return Err("joint observation masses have inconsistent world dimensions".to_owned());
    }
    if decisions.iter().any(|d| d.losses.len() != world_count) {
        return Err("decision loss table does not cover every world state".to_owned());
    }
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("precommitment CVaR alpha must be in [0, 1)".to_owned());
    }
    let mut thresholds: Vec<f64> = decisions.iter()
        .flat_map(|decision| decision.losses.iter().copied()).collect();
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
    let mut best_value = f64::INFINITY;
    let mut best_eta = 0.0;
    let mut best_policy = Vec::new();
    for eta in thresholds {
        let mut total_excess = 0.0;
        let mut policy = Vec::with_capacity(masses.len());
        for outcome_masses in masses {
            let mut best_decision = 0;
            let mut best_excess = f64::INFINITY;
            for (decision_index, decision) in decisions.iter().enumerate() {
                let excess: f64 = outcome_masses.iter().zip(decision.losses.iter())
                    .map(|(mass, loss)| mass * (loss - eta).max(0.0)).sum();
                if excess < best_excess - 1.0e-12
                    || ((excess - best_excess).abs() <= 1.0e-12
                        && decision.id < decisions[best_decision].id) {
                    best_excess = excess;
                    best_decision = decision_index;
                }
            }
            total_excess += best_excess;
            policy.push(best_decision);
        }
        let value = eta + total_excess / (1.0 - alpha);
        if value < best_value - 1.0e-12 {
            best_value = value;
            best_eta = eta;
            best_policy = policy;
        }
    }
    if !best_value.is_finite() { return Err("precommitment CVaR is non-finite".to_owned()); }
    Ok((best_value, best_eta, best_policy))
}

// Evaluate regret for a fixed outcome-contingent policy. This ensures the
// receipt's tail-regret metric describes the same decision rule as its EVSI
// policy, rather than silently optimizing a second, different policy.
fn policy_regret_metrics(
    masses: &[Vec<f64>],
    decisions: &[Decision],
    best_loss_by_world: &[f64],
    policy: &[usize],
    alpha: f64,
) -> Result<(f64, f64), String> {
    if masses.len() != policy.len() || masses.iter().any(|row| row.len() != best_loss_by_world.len()) {
        return Err("policy regret dimensions do not match joint world/observation masses".to_owned());
    }
    let mut values = Vec::with_capacity(masses.len() * best_loss_by_world.len());
    let mut probabilities = Vec::with_capacity(values.capacity());
    let mut expected_regret = 0.0;
    for (outcome_index, outcome_masses) in masses.iter().enumerate() {
        let decision = decisions.get(policy[outcome_index])
            .ok_or_else(|| "policy references an unknown decision".to_owned())?;
        for (world_index, mass) in outcome_masses.iter().enumerate() {
            let regret = (decision.losses[world_index] - best_loss_by_world[world_index]).max(0.0);
            expected_regret += mass * regret;
            values.push(regret);
            probabilities.push(*mass);
        }
    }
    let total_probability: f64 = probabilities.iter().sum();
    if (total_probability - 1.0).abs() > 1.0e-8 {
        return Err(format!("policy regret probability mass sums to {total_probability}, not 1"));
    }
    let cvar_regret = weighted_cvar(&values, &probabilities, alpha)?;
    Ok((expected_regret, cvar_regret))
}

fn expected_policy_risk(masses: &[Vec<f64>], decisions: &[Decision]) -> (f64, Vec<usize>) {
    let mut total = 0.0;
    let mut policy = Vec::with_capacity(masses.len());
    for outcome_masses in masses {
        let mut best_decision = 0;
        let mut best_loss = f64::INFINITY;
        for (decision_index, decision) in decisions.iter().enumerate() {
            let loss: f64 = outcome_masses.iter().zip(decision.losses.iter())
                .map(|(mass, value)| mass * value).sum();
            if loss < best_loss - 1.0e-12
                || ((loss - best_loss).abs() <= 1.0e-12
                    && decision.id < decisions[best_decision].id) {
                best_loss = loss;
                best_decision = decision_index;
            }
        }
        total += best_loss;
        policy.push(best_decision);
    }
    (total, policy)
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
            return Err(format!("risk-multiresource schema validation failed: {error} at {}", error.instance_path()));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("auditor accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-RISK-MULTIRESOURCE-0.1" {
            return Err("unsupported risk-multiresource contract".to_owned());
        }
        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let alpha = finite_number(fixture.get("risk_alpha").ok_or("missing risk_alpha")?, "risk_alpha")?;
        if !(0.5..1.0).contains(&alpha) { return Err("risk_alpha must be in [0.5, 1)".to_owned()); }

        let world_values = fixture.get("world_states").and_then(Value::as_array)
            .ok_or_else(|| "world_states must be an array".to_owned())?;
        let mut worlds = Vec::with_capacity(world_values.len());
        let mut world_index = HashMap::with_capacity(world_values.len());
        let mut world_prior_sum = 0.0;
        let mut has_none_of_above = false;
        for value in world_values {
            let id = required_text(value, "world_state_id")?.to_owned();
            if world_index.insert(id.clone(), worlds.len()).is_some() {
                return Err(format!("duplicate world_state_id {id}"));
            }
            let prior = finite_number(value.get("prior_probability").ok_or("missing world prior")?, "world prior")?;
            if prior <= 0.0 || prior > 1.0 { return Err(format!("world prior for {id} must be in (0,1]")); }
            let model_family = required_text(value, "model_family")?.to_owned();
            has_none_of_above |= model_family == "none_of_above";
            worlds.push(WorldState {
                id,
                prior,
                model_family,
                phase_realization: required_text(value, "phase_realization")?.to_owned(),
                evidence_qualification: required_text(value, "evidence_qualification")?.to_owned(),
                engineering_use: required_text(value, "engineering_use")?.to_owned(),
            });
            world_prior_sum += prior;
        }
        if (world_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("world-state priors sum to {world_prior_sum}, not 1"));
        }
        if !has_none_of_above { return Err("world states must include none_of_above".to_owned()); }

        let nuisance_values = fixture.get("nuisance_states").and_then(Value::as_array)
            .ok_or_else(|| "nuisance_states must be an array".to_owned())?;
        let mut nuisance_states = Vec::with_capacity(nuisance_values.len());
        let mut nuisance_index = HashMap::with_capacity(nuisance_values.len());
        let mut nuisance_prior_sum = 0.0;
        for value in nuisance_values {
            let id = required_text(value, "state_id")?.to_owned();
            if nuisance_index.insert(id.clone(), nuisance_states.len()).is_some() {
                return Err(format!("duplicate nuisance state {id}"));
            }
            let prior = finite_number(value.get("prior_probability").ok_or("missing nuisance prior")?, "nuisance prior")?;
            if prior <= 0.0 || prior > 1.0 { return Err(format!("nuisance prior for {id} must be in (0,1]")); }
            nuisance_prior_sum += prior;
            nuisance_states.push(NuisanceState { id, prior });
        }
        if (nuisance_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("nuisance-state priors sum to {nuisance_prior_sum}, not 1"));
        }

        let resource_values = fixture.get("resources").and_then(Value::as_array)
            .ok_or_else(|| "resources must be an array".to_owned())?;
        let mut resources = Vec::with_capacity(resource_values.len());
        let mut resource_index = HashMap::with_capacity(resource_values.len());
        for value in resource_values {
            let id = required_text(value, "resource_id")?.to_owned();
            if resource_index.insert(id.clone(), resources.len()).is_some() {
                return Err(format!("duplicate resource_id {id}"));
            }
            let capacity = value.get("capacity").and_then(Value::as_u64)
                .filter(|n| *n > 0).ok_or_else(|| format!("resource {id} capacity must be a positive integer"))?;
            resources.push(Resource { id, unit: required_text(value, "unit")?.to_owned(), capacity });
        }
        if resources.is_empty() { return Err("at least one resource is required".to_owned()); }
        let reference_id = required_text(&fixture, "reference_resource_id")?;
        let reference_resource_index = *resource_index.get(reference_id)
            .ok_or_else(|| format!("unknown reference resource {reference_id}"))?;

        let decision_values = fixture.get("decisions").and_then(Value::as_array)
            .ok_or_else(|| "decisions must be an array".to_owned())?;
        let mut decisions = Vec::with_capacity(decision_values.len());
        let mut decision_ids = HashSet::new();
        for value in decision_values {
            let id = required_text(value, "decision_id")?.to_owned();
            if !decision_ids.insert(id.clone()) { return Err(format!("duplicate decision_id {id}")); }
            let loss_rows = value.get("losses").and_then(Value::as_array)
                .ok_or_else(|| format!("decision {id} losses must be an array"))?;
            let mut by_world: Vec<Option<f64>> = vec![None; worlds.len()];
            for row in loss_rows {
                let world_id = required_text(row, "world_state_id")?;
                let Some(&wi) = world_index.get(world_id) else {
                    return Err(format!("decision {id} references unknown world {world_id}"));
                };
                if by_world[wi].is_some() { return Err(format!("decision {id} duplicates loss for {world_id}")); }
                let loss = finite_number(row.get("loss").ok_or("missing decision loss")?, "decision loss")?;
                if loss < 0.0 { return Err(format!("decision {id} loss must be non-negative")); }
                by_world[wi] = Some(loss);
            }
            if by_world.iter().any(Option::is_none) {
                return Err(format!("decision {id} must define a loss for every world state"));
            }
            decisions.push(Decision { id, losses: by_world.into_iter().map(Option::unwrap).collect() });
        }
        if decisions.len() < 2 { return Err("at least two downstream decisions are required".to_owned()); }
        decisions.sort_by(|a, b| a.id.cmp(&b.id));

        let outcome_values = unique_strings(fixture.get("outcome_labels").ok_or("missing outcome_labels")?, "outcome_labels")?;
        if outcome_values.len() != 2 { return Err("exactly two outcome labels are required".to_owned()); }
        let outcome_labels = [outcome_values[0].clone(), outcome_values[1].clone()];
        let initial_capabilities: HashSet<String> = unique_strings(
            fixture.get("initial_capabilities").ok_or("missing initial_capabilities")?,
            "initial_capabilities",
        )?.into_iter().collect();

        let action_values = fixture.get("actions").and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_values.is_empty() || action_values.len() > 12 {
            return Err("action count must be between 1 and 12".to_owned());
        }
        let mut actions = Vec::with_capacity(action_values.len());
        let mut action_ids = HashSet::new();
        for value in action_values {
            let id = required_text(value, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) { return Err(format!("duplicate action_id {id}")); }
            let action_type = required_text(value, "action_type")?.to_owned();
            let requires = unique_strings(value.get("requires_capabilities").ok_or("missing requirements")?, "requires_capabilities")?;
            let provides = unique_strings(value.get("provides_capabilities").ok_or("missing provisions")?, "provides_capabilities")?;
            let eligible = value.get("eligible").and_then(Value::as_bool).ok_or("eligible must be boolean")?;
            let exclusion_reason = value.get("exclusion_reason").and_then(Value::as_str)
                .ok_or("exclusion_reason must be text")?.to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!("ineligible action {id} requires an exclusion reason"));
            }

            let mut consumption = vec![0_u64; resources.len()];
            let consumption_rows = value.get("resource_consumption").and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} resource_consumption must be an array"))?;
            for row in consumption_rows {
                let resource_id = required_text(row, "resource_id")?;
                let Some(&ri) = resource_index.get(resource_id) else {
                    return Err(format!("action {id} references unknown resource {resource_id}"));
                };
                if consumption[ri] != 0 { return Err(format!("action {id} duplicates resource {resource_id}")); }
                let unit = required_text(row, "unit")?;
                if unit != resources[ri].unit {
                    return Err(format!("action {id} resource {resource_id} unit {unit} mismatches {}", resources[ri].unit));
                }
                consumption[ri] = row.get("quantity").and_then(Value::as_u64)
                    .filter(|n| *n > 0).ok_or_else(|| format!("action {id} quantity must be positive integer"))?;
            }

            let likelihood_rows = value.get("outcome_likelihoods").and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} outcome_likelihoods must be an array"))?;
            let mut by_world: Vec<Option<Vec<Option<[f64; 2]>>>> = vec![None; worlds.len()];
            for row in likelihood_rows {
                let world_id = required_text(row, "world_state_id")?;
                let Some(&wi) = world_index.get(world_id) else {
                    return Err(format!("action {id} references unknown world {world_id}"));
                };
                if by_world[wi].is_some() { return Err(format!("action {id} duplicates world likelihood {world_id}")); }
                let nuisance_rows = row.get("nuisance_likelihoods").and_then(Value::as_array)
                    .ok_or_else(|| format!("action {id} nuisance_likelihoods must be an array"))?;
                let mut by_nuisance: Vec<Option<[f64; 2]>> = vec![None; nuisance_states.len()];
                for nuisance_row in nuisance_rows {
                    let state_id = required_text(nuisance_row, "state_id")?;
                    let Some(&ni) = nuisance_index.get(state_id) else {
                        return Err(format!("action {id} references unknown nuisance state {state_id}"));
                    };
                    if by_nuisance[ni].is_some() { return Err(format!("action {id} duplicates nuisance state {state_id}")); }
                    let probs = nuisance_row.get("outcome_probabilities").and_then(Value::as_array)
                        .ok_or_else(|| format!("action {id} probabilities must be an array"))?;
                    if probs.len() != 2 { return Err(format!("action {id} requires two outcome probabilities")); }
                    let p0 = finite_number(&probs[0], "outcome probability")?;
                    let p1 = finite_number(&probs[1], "outcome probability")?;
                    if !(0.0..=1.0).contains(&p0) || !(0.0..=1.0).contains(&p1)
                        || (p0 + p1 - 1.0).abs() > 1.0e-9 {
                        return Err(format!("action {id} outcome probabilities must be normalized"));
                    }
                    by_nuisance[ni] = Some([p0, p1]);
                }
                if by_nuisance.iter().any(Option::is_none) {
                    return Err(format!("action {id} lacks a nuisance likelihood for {world_id}"));
                }
                by_world[wi] = Some(by_nuisance.into_iter().map(Option::unwrap).collect());
            }
            let likelihoods = if action_type == "capability" {
                if !likelihood_rows.is_empty() || provides.is_empty() {
                    return Err(format!("capability action {id} must provide capability and have no outcomes"));
                }
                Vec::new()
            } else {
                if by_world.iter().any(Option::is_none) {
                    return Err(format!("observation action {id} needs a likelihood for every world"));
                }
                by_world.into_iter().map(|entry| entry.unwrap().into_iter().map(Option::unwrap).collect()).collect()
            };
            actions.push(Action { id, action_type, requires, provides, eligible, exclusion_reason, consumption, likelihoods });
        }
        actions.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            alpha, reference_resource_index, worlds, nuisance_states, decisions, resources,
            outcome_labels, initial_capabilities, actions,
        })
    }

    fn feasible_order(&self, indices: &[usize]) -> Option<(Vec<usize>, Vec<u64>)> {
        let mut selected = indices.to_vec();
        selected.sort_unstable();
        selected.dedup();
        if selected.len() != indices.len() { return None; }
        let mut consumption = vec![0_u64; self.resources.len()];
        for index in &selected {
            let action = self.actions.get(*index)?;
            if !action.eligible { return None; }
            for (used, amount) in consumption.iter_mut().zip(action.consumption.iter()) {
                *used = used.checked_add(*amount)?;
            }
        }
        if consumption.iter().zip(self.resources.iter()).any(|(used, r)| *used > r.capacity) {
            return None;
        }
        let mut available = self.initial_capabilities.clone();
        let mut remaining = selected;
        let mut order = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            // Actions are sorted by ID, so this is a canonical topological ordering.
            let position = remaining.iter().position(|index| self.actions[*index].requires
                .iter().all(|capability| available.contains(capability)))?;
            let index = remaining.remove(position);
            for capability in &self.actions[index].provides { available.insert(capability.clone()); }
            order.push(index);
        }
        Some((order, consumption))
    }

    fn joint_masses(&self, order: &[usize]) -> Result<Vec<Vec<f64>>, String> {
        let measured: Vec<usize> = order.iter().copied()
            .filter(|index| !self.actions[*index].likelihoods.is_empty()).collect();
        if measured.len() > 10 { return Err("more than ten binary observations exceeds exact enumeration bound".to_owned()); }
        let mut masses = vec![vec![0.0; self.worlds.len()]; 1_usize << measured.len()];
        for outcome_mask in 0..masses.len() {
            for (world_index, world) in self.worlds.iter().enumerate() {
                // Integrate one shared nuisance state after multiplying the
                // conditional likelihoods of all observations.
                let likelihood: f64 = self.nuisance_states.iter().enumerate().map(|(ni, nuisance)| {
                    let conditional = measured.iter().enumerate().fold(1.0, |product, (oi, action_index)| {
                        let outcome = (outcome_mask >> oi) & 1;
                        product * self.actions[*action_index].likelihoods[world_index][ni][outcome]
                    });
                    nuisance.prior * conditional
                }).sum();
                masses[outcome_mask][world_index] = world.prior * likelihood;
            }
        }
        let total: f64 = masses.iter().flatten().sum();
        if !total.is_finite() || (total - 1.0).abs() > 1.0e-8 {
            return Err(format!("joint world/observation mass is {total}, not 1"));
        }
        Ok(masses)
    }
    fn information_gain(&self, masses: &[Vec<f64>]) -> f64 {
        let prior: Vec<f64> = self.worlds.iter().map(|w| w.prior).collect();
        let h_prior = entropy(&prior);
        let mut h_posterior = 0.0;
        for outcome_masses in masses {
            let p_y: f64 = outcome_masses.iter().sum();
            if p_y <= 0.0 { continue; }
            let posterior: Vec<f64> = outcome_masses.iter().map(|mass| mass / p_y).collect();
            h_posterior += p_y * entropy(&posterior);
        }
        (h_prior - h_posterior).max(0.0)
    }
    fn plan(&self, indices: &[usize]) -> Result<Option<Plan>, String> {
        let Some((ordered_indices, consumption)) = self.feasible_order(indices) else { return Ok(None); };
        let masses = self.joint_masses(&ordered_indices)?;
        let (expected_risk, expected_policy) = expected_policy_risk(&masses, &self.decisions);
        let (cvar_risk, eta, policy) = precommitment_cvar(&masses, &self.decisions, self.alpha)?;
        let prior = vec![self.worlds.iter().map(|world| world.prior).collect::<Vec<_>>()];
        let (prior_expected, _) = expected_policy_risk(&prior, &self.decisions);
        let (prior_cvar, _, _) = precommitment_cvar(&prior, &self.decisions, self.alpha)?;
        // Independently verify the no-observation baseline by minimizing each
        // decision's finite-support CVaR under the prior.
        let prior_probabilities: Vec<f64> = self.worlds.iter().map(|world| world.prior).collect();
        let direct_prior_cvars: Vec<f64> = self.decisions.iter()
            .map(|decision| weighted_cvar(&decision.losses, &prior_probabilities, self.alpha))
            .collect::<Result<Vec<_>, _>>()?;
        let direct_prior_cvar = direct_prior_cvars.into_iter().fold(f64::INFINITY, f64::min);
        if (prior_cvar - direct_prior_cvar).abs() > 1.0e-9 {
            return Err("precommitment baseline CVaR disagrees with direct decision CVaR".to_owned());
        }
        // Regret is measured against the best decision available with full
        // knowledge of each world state, under the policy actually reported
        // for each objective (rather than re-optimizing a different policy).
        let best_loss_by_world: Vec<f64> = (0..self.worlds.len()).map(|world_index| {
            self.decisions.iter().map(|decision| decision.losses[world_index])
                .fold(f64::INFINITY, f64::min)
        }).collect();
        let (expected_regret, _) = policy_regret_metrics(
            &masses, &self.decisions, &best_loss_by_world, &expected_policy, self.alpha
        )?;
        let (_, cvar_regret) = policy_regret_metrics(
            &masses, &self.decisions, &best_loss_by_world, &policy, self.alpha
        )?;
        let expected_evsi = prior_expected - expected_risk;
        let cvar_evsi = prior_cvar - cvar_risk;
        if expected_evsi < -1.0e-8 || cvar_evsi < -1.0e-8 {
            return Err("value of sample information is negative beyond numerical tolerance".to_owned());
        }
        Ok(Some(Plan {
            ordered_indices, consumption, expected_risk, cvar_risk,
            expected_regret, cvar_regret,
            expected_evsi: expected_evsi.max(0.0), cvar_evsi: cvar_evsi.max(0.0),
            mutual_information_nats: self.information_gain(&masses),
            cvar_eta: eta, expected_policy, policy, outcome_count: masses.len(),
        }))
    }
    fn reference_cost(&self, plan: &Plan) -> u64 {
        plan.consumption[self.reference_resource_index]
    }
    fn plan_ids(&self, plan: &Plan) -> Vec<String> {
        plan.ordered_indices.iter().map(|i| self.actions[*i].id.clone()).collect()
    }
    fn is_better(&self, candidate: &Plan, incumbent: &Plan, objective: Objective) -> bool {
        let a = score(candidate, objective);
        let b = score(incumbent, objective);
        if a > b + 1.0e-12 { return true; }
        if (a - b).abs() > 1.0e-12 { return false; }
        let ca = self.reference_cost(candidate);
        let cb = self.reference_cost(incumbent);
        if ca != cb { return ca < cb; }
        self.plan_ids(candidate) < self.plan_ids(incumbent)
    }
    fn exact_oracle(&self, objective: Objective) -> Result<(Plan, u64), String> {
        let count = 1_u64.checked_shl(self.actions.len() as u32)
            .ok_or_else(|| "subset count overflowed".to_owned())?;
        let mut best = self.plan(&[])?.ok_or("empty plan must be feasible")?;
        for mask in 1..count {
            let indices: Vec<usize> = (0..self.actions.len())
                .filter(|i| (mask & (1_u64 << *i)) != 0).collect();
            let Some(candidate) = self.plan(&indices)? else { continue; };
            if self.is_better(&candidate, &best, objective) { best = candidate; }
        }
        Ok((best, count))
    }
    fn greedy(&self, objective: Objective) -> Result<Plan, String> {
        let mut chosen = Vec::new();
        let mut available = self.initial_capabilities.clone();
        let mut selected = HashSet::new();
        loop {
            let current = self.plan(&chosen)?.ok_or("current greedy plan is infeasible")?;
            let current_score = score(&current, objective);
            let mut best: Option<(usize, f64)> = None;
            for index in 0..self.actions.len() {
                if selected.contains(&index) { continue; }
                let action = &self.actions[index];
                if !action.eligible || !action.requires.iter().all(|c| available.contains(c)) { continue; }
                let mut trial = chosen.clone();
                trial.push(index);
                let Some(candidate) = self.plan(&trial)? else { continue; };
                let marginal = score(&candidate, objective) - current_score;
                if marginal <= 1.0e-12 { continue; }
                let delta = candidate.consumption[self.reference_resource_index]
                    .saturating_sub(current.consumption[self.reference_resource_index]).max(1);
                let ratio = marginal / delta as f64;
                let replace = match best {
                    None => true,
                    Some((bi, br)) => ratio > br + 1.0e-12
                        || ((ratio - br).abs() <= 1.0e-12 && action.id < self.actions[bi].id),
                };
                if replace { best = Some((index, ratio)); }
            }
            let Some((index, _)) = best else { return Ok(current); };
            chosen.push(index);
            selected.insert(index);
            for capability in &self.actions[index].provides { available.insert(capability.clone()); }
        }
    }
    fn dominates_plan(&self, candidate: &Plan, other: &Plan) -> bool {
        let no_worse = candidate.expected_evsi + 1.0e-12 >= other.expected_evsi
            && candidate.cvar_evsi + 1.0e-12 >= other.cvar_evsi
            && candidate.cvar_regret <= other.cvar_regret + 1.0e-12
            && candidate.consumption.iter().zip(other.consumption.iter())
                .all(|(left, right)| left <= right);
        let strictly_better = candidate.expected_evsi > other.expected_evsi + 1.0e-12
            || candidate.cvar_evsi > other.cvar_evsi + 1.0e-12
            || candidate.cvar_regret + 1.0e-12 < other.cvar_regret
            || candidate.consumption.iter().zip(other.consumption.iter())
                .any(|(left, right)| left < right);
        no_worse && strictly_better
    }

    fn pareto_frontier(&self) -> Result<Vec<Plan>, String> {
        let subset_count = 1_u64.checked_shl(self.actions.len() as u32)
            .ok_or_else(|| "Pareto subset count overflowed".to_owned())?;
        let mut feasible = Vec::new();
        for mask in 0..subset_count {
            let indices: Vec<usize> = (0..self.actions.len())
                .filter(|index| (mask & (1_u64 << *index)) != 0).collect();
            if let Some(plan) = self.plan(&indices)? { feasible.push(plan); }
        }
        let mut frontier = Vec::new();
        for (index, candidate) in feasible.iter().enumerate() {
            let dominated = feasible.iter().enumerate().any(|(other_index, other)| {
                other_index != index && self.dominates_plan(other, candidate)
            });
            if !dominated { frontier.push(candidate.clone()); }
        }
        frontier.sort_by(|left, right| {
            right.cvar_evsi.total_cmp(&left.cvar_evsi)
                .then_with(|| right.expected_evsi.total_cmp(&left.expected_evsi))
                .then_with(|| left.cvar_regret.total_cmp(&right.cvar_regret))
                .then_with(|| self.reference_cost(left).cmp(&self.reference_cost(right)))
                .then_with(|| self.plan_ids(left).cmp(&self.plan_ids(right)))
        });
        Ok(frontier)
    }

    fn random_feasible(&self, seed: u64) -> Result<Plan, String> {
        let mut rng_state = seed;
        let mut chosen = Vec::new();
        let mut available = self.initial_capabilities.clone();
        let mut selected = HashSet::new();
        loop {
            let current = self.plan(&chosen)?.ok_or("current random plan is infeasible")?;
            let mut candidates = Vec::new();
            for index in 0..self.actions.len() {
                if selected.contains(&index) { continue; }
                let action = &self.actions[index];
                if !action.eligible
                    || !action.requires.iter().all(|capability| available.contains(capability)) {
                    continue;
                }
                let mut trial = chosen.clone();
                trial.push(index);
                if self.plan(&trial)?.is_some() {
                    candidates.push(index);
                }
            }
            if candidates.is_empty() {
                return Ok(current);
            }
            let random = splitmix64(&mut rng_state);
            let position = ((u128::from(random) * candidates.len() as u128) >> 64) as usize;
            let index = candidates[position];
            chosen.push(index);
            selected.insert(index);
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
        }
    }

    fn plan_receipt(&self, plan: &Plan) -> Value {
        let measured_ids: Vec<String> = plan.ordered_indices.iter()
            .filter(|i| !self.actions[**i].likelihoods.is_empty())
            .map(|i| self.actions[*i].id.clone()).collect();
        let render_policy = |decision_indices: &[usize]| -> Vec<Value> {
            decision_indices.iter().enumerate().map(|(mask, decision_index)| {
                let outcomes: Vec<Value> = (0..measured_ids.len()).map(|bit| json!({
                    "action_id": measured_ids[bit],
                    "outcome": self.outcome_labels[(mask >> bit) & 1]
                })).collect();
                json!({
                    "outcome_mask":mask,
                    "outcomes":outcomes,
                    "decision_id":self.decisions[*decision_index].id
                })
            }).collect()
        };
        let expected_policy = render_policy(&plan.expected_policy);
        let cvar_policy = render_policy(&plan.policy);
        let usage: Vec<Value> = self.resources.iter().enumerate().map(|(i, resource)| json!({
            "resource_id":resource.id,
            "unit":resource.unit,
            "capacity":resource.capacity,
            "consumed":plan.consumption[i],
            "remaining":resource.capacity-plan.consumption[i],
            "binding":resource.capacity==plan.consumption[i]
        })).collect();
        json!({
            "ordered_action_ids":self.plan_ids(plan),
            "ordered_action_types":plan.ordered_indices.iter().map(|i|self.actions[*i].action_type.as_str()).collect::<Vec<_>>(),
            "resource_usage":usage,
            "reference_resource_id":self.resources[self.reference_resource_index].id,
            "reference_resource_consumption":self.reference_cost(plan),
            "expected_posterior_decision_loss":plan.expected_risk,
            "precommitment_cvar_decision_risk":plan.cvar_risk,
            "expected_decision_regret":plan.expected_regret,
            "precommitment_cvar_decision_regret":plan.cvar_regret,
            "expected_loss_evsi":plan.expected_evsi,
            "precommitment_cvar_evsi":plan.cvar_evsi,
            "joint_world_state_information_gain_nats":plan.mutual_information_nats,
            "cvar_alpha":self.alpha,
            "rockafellar_uryasev_eta":plan.cvar_eta,
            "outcome_count":plan.outcome_count,
            "expected_loss_decision_policy":expected_policy,
            "precommitment_cvar_decision_policy":cvar_policy
        })
    }
}
#[derive(Clone, Copy)]
enum Objective { ExpectedValue, TailRisk, InformationGain }
fn score(plan: &Plan, objective: Objective) -> f64 {
    match objective {
        Objective::ExpectedValue => plan.expected_evsi,
        Objective::TailRisk => plan.cvar_evsi,
        Objective::InformationGain => plan.mutual_information_nats,
    }
}
fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance = Instance::from_bytes(schema_bytes, fixture_bytes)?;
    let (expected_oracle, subset_count) = instance.exact_oracle(Objective::ExpectedValue)?;
    let (tail_oracle, _) = instance.exact_oracle(Objective::TailRisk)?;
    let (information_oracle, _) = instance.exact_oracle(Objective::InformationGain)?;
    let expected_greedy = instance.greedy(Objective::ExpectedValue)?;
    let tail_greedy = instance.greedy(Objective::TailRisk)?;
    let information_greedy = instance.greedy(Objective::InformationGain)?;
    const RANDOM_SEEDS: [u64; 32] = [
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
        17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
    ];
    let random_trials: Vec<(u64, Plan)> = RANDOM_SEEDS.iter()
        .map(|seed| Ok((*seed, instance.random_feasible(*seed)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let pareto_frontier = instance.pareto_frontier()?;
    let excluded: Vec<Value> = instance.actions.iter().filter(|a| !a.eligible)
        .map(|a| json!({"action_id":a.id,"exclusion_reason":a.exclusion_reason})).collect();
    Ok(json!({
        "status":"synthetic_contract_pass",
        "contract":"SC-RISK-MULTIRESOURCE-AUDIT-0.1",
        "instance_id":instance.instance_id,
        "schema_sha256":instance.schema_sha256,
        "fixture_sha256":instance.fixture_sha256,
        "risk_alpha":instance.alpha,
        "world_states":instance.worlds.iter().map(|w|json!({
            "world_state_id":w.id,"prior_probability":w.prior,"model_family":w.model_family,
            "phase_realization":w.phase_realization,"evidence_qualification":w.evidence_qualification,
            "engineering_use":w.engineering_use
        })).collect::<Vec<_>>(),
        "shared_nuisance_states":instance.nuisance_states.iter().map(|n|json!({
            "state_id":n.id,"prior_probability":n.prior
        })).collect::<Vec<_>>(),
        "decision_loss_table":instance.decisions.iter().map(|d|json!({
            "decision_id":d.id,"losses_by_world_state":d.losses
        })).collect::<Vec<_>>(),
        "resource_registry":instance.resources.iter().map(|r|json!({
            "resource_id":r.id,"unit":r.unit,"capacity":r.capacity
        })).collect::<Vec<_>>(),
        "action_count":instance.actions.len(),
        "observation_action_count":instance.actions.iter().filter(|a|!a.likelihoods.is_empty()).count(),
        "excluded_actions":excluded,
        "enumerated_subsets":subset_count,
        "expected_loss_oracle":{
            "algorithm":"exact feasible subset enumeration; expected-loss EVSI; reference-resource tie-break",
            "plan":instance.plan_receipt(&expected_oracle)
        },
        "precommitment_cvar_oracle":{
            "algorithm":"exact feasible subset enumeration; joint world-observation CVaR; outcome-contingent policy",
            "plan":instance.plan_receipt(&tail_oracle)
        },
        "mutual_information_oracle":{
            "algorithm":"exact feasible subset enumeration; mutual information between joint world state and observations",
            "plan":instance.plan_receipt(&information_oracle)
        },
        "pareto_frontier":{
            "dominance":"maximize expected-loss EVSI and precommitment CVaR-EVSI; minimize precommitment CVaR regret and every resource consumption dimension",
            "frontier_size":pareto_frontier.len(),
            "plans":pareto_frontier.iter().map(|plan|instance.plan_receipt(plan)).collect::<Vec<_>>()
        },
        "random_feasible_baseline":{
            "algorithm":"uniformly sample among currently feasible next actions; stop when no further action fits; deterministic SplitMix64 seeds",
            "seed_count":random_trials.len(),
            "trials":random_trials.iter().map(|(seed,plan)|json!({
                "seed":seed,
                "plan":instance.plan_receipt(plan)
            })).collect::<Vec<_>>(),
            "mean_expected_loss_evsi":random_trials.iter().map(|(_,p)|p.expected_evsi).sum::<f64>() / random_trials.len() as f64,
            "mean_precommitment_cvar_evsi":random_trials.iter().map(|(_,p)|p.cvar_evsi).sum::<f64>() / random_trials.len() as f64,
            "mean_joint_world_state_information_nats":random_trials.iter().map(|(_,p)|p.mutual_information_nats).sum::<f64>() / random_trials.len() as f64
        },
        "greedy_expected_loss_baseline":{
            "algorithm":"myopic marginal expected-loss EVSI per reference resource",
            "plan":instance.plan_receipt(&expected_greedy)
        },
        "greedy_precommitment_cvar_baseline":{
            "algorithm":"myopic marginal precommitment CVaR-EVSI per reference resource",
            "plan":instance.plan_receipt(&tail_greedy)
        },
        "greedy_joint_world_state_information_baseline":{
            "algorithm":"myopic marginal mutual information per reference resource",
            "plan":instance.plan_receipt(&information_greedy)
        },
        "comparison":{
            "expected_oracle_evsi":expected_oracle.expected_evsi,
            "tail_risk_oracle_evsi":tail_oracle.cvar_evsi,
            "information_oracle_joint_world_state_information_nats":information_oracle.mutual_information_nats,
            "expected_oracle_and_tail_oracle_select_same_actions":instance.plan_ids(&expected_oracle)==instance.plan_ids(&tail_oracle),
            "tail_oracle_uses_less_reference_resource_than_information_oracle":instance.reference_cost(&tail_oracle)<instance.reference_cost(&information_oracle)
        },
        "claim_ceiling":"Synthetic multi-resource and precommitment tail-risk planning contract only; no physical likelihood, laboratory cost, synthesis yield, superconducting property, or application outcome is calibrated or claimed."
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
                if index + 1 >= args.len() { usage(&args[0]); process::exit(2); }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => { usage(&args[0]); return; }
            unknown => { eprintln!("unknown argument: {unknown}"); usage(&args[0]); process::exit(2); }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]); process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|e| {
        eprintln!("SC-RISK-MULTIRESOURCE-FAIL: {e}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path) = output_path {
        if let Err(e) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {e}", path.display());
            process::exit(1);
        }
    } else { println!("{rendered}"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_risk_multiresource_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research/superconductivity_risk_multiresource_fixture_v0_1.json")).unwrap(),
        )
    }
    fn mutated_fixture(mut mutation: impl FnMut(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        audit(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn cvar_alpha_zero_equals_expectation_and_fractional_tail_mass_is_exact() {
        assert!((weighted_cvar(&[0.0,10.0], &[0.8,0.2], 0.0).unwrap()-2.0).abs()<1.0e-12);
        assert!((weighted_cvar(&[0.0,10.0], &[0.8,0.2], 0.9).unwrap()-10.0).abs()<1.0e-12);
        assert!((weighted_cvar(&[0.0,10.0], &[0.8,0.2], 0.8).unwrap()-10.0).abs()<1.0e-12);
        // The 15% upper tail contains all 10% mass at loss 20 plus only
        // 5% of the mass at loss 10: (0.1*20 + 0.05*10) / 0.15.
        let fractional = weighted_cvar(&[0.0,10.0,20.0], &[0.7,0.2,0.1], 0.85).unwrap();
        assert!((fractional - (2.5 / 0.15)).abs() < 1.0e-12);
        assert!((weighted_cvar(&[0.0,10.0], &[0.8,0.2], 0.999_999).unwrap()-10.0).abs()<1.0e-9);
    }
    #[test]
    fn exact_oracle_respects_each_resource_and_capability_order() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let plan = &receipt["precommitment_cvar_oracle"]["plan"];
        let ids = plan["ordered_action_ids"].as_array().unwrap();
        let route = ids.iter().position(|v| v=="qualify_synthesis_route").unwrap();
        let trial = ids.iter().position(|v| v=="synthesis_trial").unwrap();
        assert!(route < trial);
        for resource in plan["resource_usage"].as_array().unwrap() {
            let used=resource["consumed"].as_u64().unwrap();
            let capacity=resource["capacity"].as_u64().unwrap();
            assert!(used<=capacity);
            assert_eq!(resource["remaining"].as_u64().unwrap(),capacity-used);
        }
        assert!(!ids.iter().any(|v| v=="unreviewed_high_field_probe"));
    }
    #[test]
    fn action_and_resource_json_order_does_not_change_canonical_plan() {
        let (schema, fixture) = inputs();
        let baseline = audit(&schema, &fixture).unwrap();
        let permuted = mutated_fixture(|f| {
            f["actions"].as_array_mut().unwrap().reverse();
            f["resources"].as_array_mut().unwrap().reverse();
        }).unwrap();
        assert_eq!(
            baseline["precommitment_cvar_oracle"]["plan"]["ordered_action_ids"],
            permuted["precommitment_cvar_oracle"]["plan"]["ordered_action_ids"]
        );
    }
    #[test]
    fn adding_a_constant_to_every_loss_preserves_tail_evsi_and_plan() {
        let (schema, fixture) = inputs();
        let baseline = audit(&schema, &fixture).unwrap();
        let shifted = mutated_fixture(|f| {
            for d in f["decisions"].as_array_mut().unwrap() {
                for row in d["losses"].as_array_mut().unwrap() {
                    row["loss"]=json!(row["loss"].as_f64().unwrap()+7.5);
                }
            }
        }).unwrap();
        let old=&baseline["precommitment_cvar_oracle"]["plan"];
        let new=&shifted["precommitment_cvar_oracle"]["plan"];
        assert!((old["precommitment_cvar_evsi"].as_f64().unwrap()-new["precommitment_cvar_evsi"].as_f64().unwrap()).abs()<1.0e-9);
        assert_eq!(old["ordered_action_ids"],new["ordered_action_ids"]);
    }
    #[test]
    fn positive_loss_scaling_scales_tail_evsi_without_changing_plan() {
        let (schema, fixture) = inputs();
        let baseline=audit(&schema,&fixture).unwrap();
        let scaled=mutated_fixture(|f| {
            for d in f["decisions"].as_array_mut().unwrap() {
                for row in d["losses"].as_array_mut().unwrap() {
                    row["loss"]=json!(row["loss"].as_f64().unwrap()*10.0);
                }
            }
        }).unwrap();
        let old=&baseline["precommitment_cvar_oracle"]["plan"];
        let new=&scaled["precommitment_cvar_oracle"]["plan"];
        assert!((10.0*old["precommitment_cvar_evsi"].as_f64().unwrap()-new["precommitment_cvar_evsi"].as_f64().unwrap()).abs()<1.0e-8);
        assert_eq!(old["ordered_action_ids"],new["ordered_action_ids"]);
    }
    #[test]
    fn identical_statewise_decision_losses_yield_zero_value_of_information() {
        let result=mutated_fixture(|f| {
            let losses=json!([
                {"world_state_id":"bulk_conventional","loss":2.0},
                {"world_state_id":"metastable_transport_only","loss":1.0},
                {"world_state_id":"finite_q_bulk","loss":4.0},
                {"world_state_id":"none_of_above_unknown","loss":3.0}
            ]);
            for d in f["decisions"].as_array_mut().unwrap(){d["losses"]=losses.clone();}
        }).unwrap();
        let plan=&result["precommitment_cvar_oracle"]["plan"];
        assert_eq!(plan["ordered_action_ids"].as_array().unwrap().len(),0);
        assert!(plan["precommitment_cvar_evsi"].as_f64().unwrap().abs()<1.0e-10);
    }
    #[test]
    fn wrong_units_unknown_resources_and_duplicate_resource_entries_fail_closed() {
        assert!(mutated_fixture(|f| {
            f["actions"][0]["resource_consumption"][0]["unit"]=json!("hour");
        }).is_err());
        assert!(mutated_fixture(|f| {
            f["actions"][0]["resource_consumption"][0]["resource_id"]=json!("unregistered_resource");
        }).is_err());
        assert!(mutated_fixture(|f| {
            let duplicate=f["actions"][0]["resource_consumption"][0].clone();
            f["actions"][0]["resource_consumption"].as_array_mut().unwrap().push(duplicate);
        }).is_err());
    }
    #[test]
    fn a_plan_over_capacity_in_one_dimension_is_infeasible() {
        let (schema, fixture)=inputs();
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let find=|id:&str| instance.actions.iter().position(|a|a.id==id).unwrap();
        let selected=vec![find("qualify_synthesis_route"),find("synthesis_trial"),
            find("bulk_thermodynamic_check"),find("stiffness_tensor")];
        assert!(instance.feasible_order(&selected).is_none());
    }
    #[test]
    fn tail_risk_and_expected_value_can_prefer_different_experiments() {
        // Synthetic control: a 0.99%-probability world carries catastrophic
        // decision loss. The expected-value and tail-risk objectives disagree.
        let priors=[0.6185428538145715,0.3715547962844521,0.009902349900976503];
        let losses=vec![
            Decision{id:"d0".into(),losses:vec![6.0,15.0,7.0]},
            Decision{id:"d1".into(),losses:vec![0.0,6.0,50.0]},
            Decision{id:"d2".into(),losses:vec![11.0,4.0,13.0]},
            Decision{id:"d3".into(),losses:vec![5.0,24.0,7.0]},
        ];
        let q_mean=[0.07173207,0.97299267,0.19905508];
        let q_tail=[0.83094999,0.98255861,0.18996299];
        let no_info=vec![priors.to_vec()];
        let (base_mean,_)=expected_policy_risk(&no_info,&losses);
        let (base_tail,_,_)=precommitment_cvar(&no_info,&losses,0.9).unwrap();
        let masses=|q:&[f64]|vec![
            (0..3).map(|i|priors[i]*q[i]).collect(),
            (0..3).map(|i|priors[i]*(1.0-q[i])).collect()
        ];
        let (mean1,_)=expected_policy_risk(&masses(&q_mean),&losses);
        let (mean2,_)=expected_policy_risk(&masses(&q_tail),&losses);
        let (tail1,_,_)=precommitment_cvar(&masses(&q_mean),&losses,0.9).unwrap();
        let (tail2,_,_)=precommitment_cvar(&masses(&q_tail),&losses,0.9).unwrap();
        assert!(base_mean-mean1 > base_mean-mean2+0.2);
        assert!(base_tail-tail2 > base_tail-tail1+0.2);
    }
    #[test]
    fn pareto_frontier_contains_only_nondominated_resource_and_risk_plans() {
        let (schema, fixture) = inputs();
        let instance = Instance::from_bytes(&schema, &fixture).unwrap();
        let frontier = instance.pareto_frontier().unwrap();
        assert!(!frontier.is_empty());
        for i in 0..frontier.len() {
            for j in 0..frontier.len() {
                if i != j {
                    assert!(!instance.dominates_plan(&frontier[i], &frontier[j]));
                }
            }
        }
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(
            receipt["pareto_frontier"]["frontier_size"].as_u64().unwrap() as usize,
            frontier.len()
        );
        let expected_policy = receipt["precommitment_cvar_oracle"]["plan"]["expected_loss_decision_policy"]
            .as_array().unwrap();
        let cvar_policy = receipt["precommitment_cvar_oracle"]["plan"]["precommitment_cvar_decision_policy"]
            .as_array().unwrap();
        assert_eq!(expected_policy.len(), cvar_policy.len());
        assert_eq!(receipt["random_feasible_baseline"]["seed_count"], 32);
        assert_eq!(receipt["random_feasible_baseline"]["trials"].as_array().unwrap().len(), 32);
        for plan in &frontier {
            assert!(plan.expected_regret.is_finite());
            assert!(plan.cvar_regret.is_finite());
        }
    }

    #[test]
    fn regret_diagnostics_are_evaluated_under_the_reported_policy() {
        let decisions = vec![
            Decision { id: "d0".into(), losses: vec![0.0, 100.0] },
            Decision { id: "d1".into(), losses: vec![40.0, 0.0] },
        ];
        let masses = vec![vec![0.45, 0.05], vec![0.05, 0.45]];
        let best_loss = [0.0, 0.0];
        let policy_always_d0 = [0, 0];
        let policy_adaptive = [0, 1];
        let (mean_fixed, tail_fixed) =
            policy_regret_metrics(&masses, &decisions, &best_loss, &policy_always_d0, 0.5).unwrap();
        let (mean_adaptive, tail_adaptive) =
            policy_regret_metrics(&masses, &decisions, &best_loss, &policy_adaptive, 0.5).unwrap();
        assert!((mean_fixed - 50.0).abs() < 1.0e-12);
        assert!((tail_fixed - 100.0).abs() < 1.0e-12);
        assert!((mean_adaptive - 7.0).abs() < 1.0e-12);
        assert!((tail_adaptive - 14.0).abs() < 1.0e-12);
    }

    #[test]
    fn deterministic_random_feasible_baseline_is_reproducible_and_never_overbooks_resources() {
        let (schema, fixture) = inputs();
        let instance = Instance::from_bytes(&schema, &fixture).unwrap();
        let first = instance.random_feasible(20261120).unwrap();
        let second = instance.random_feasible(20261120).unwrap();
        assert_eq!(instance.plan_ids(&first), instance.plan_ids(&second));
        assert_eq!(first.consumption, second.consumption);
        for (used, resource) in first.consumption.iter().zip(instance.resources.iter()) {
            assert!(*used <= resource.capacity);
        }
        assert!(first.ordered_indices.iter().all(|index| instance.actions[*index].eligible));
    }

    #[test]
    fn pressure_cell_capacity_rejects_action_even_when_other_gates_are_open() {
        let (schema, fixture_bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&fixture_bytes).unwrap();
        fixture["actions"].as_array_mut().unwrap().iter_mut()
            .find(|action| action["action_id"] == "unreviewed_high_field_probe")
            .unwrap()["eligible"] = json!(true);
        fixture["initial_capabilities"].as_array_mut().unwrap()
            .push(json!("high_field_facility"));
        fixture["resources"].as_array_mut().unwrap().iter_mut()
            .find(|resource| resource["resource_id"] == "magnetometer_B_time")
            .unwrap()["capacity"] = json!(80);
        let instance = Instance::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        let high_field = instance.actions.iter()
            .position(|action| action.id == "unreviewed_high_field_probe").unwrap();
        let pressure_resource = instance.resources.iter()
            .position(|resource| resource.id == "pressure_cell_A_time").unwrap();
        assert_eq!(instance.actions[high_field].consumption[pressure_resource], 60);
        assert_eq!(instance.resources[pressure_resource].capacity, 45);
        assert!(instance.feasible_order(&[high_field]).is_none());
    }

}
