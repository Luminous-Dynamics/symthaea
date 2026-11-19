use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct Model {
    id: String,
    prior: f64,
    none_of_above: bool,
}

#[derive(Clone, Debug)]
struct NuisanceState {
    id: String,
    prior: f64,
}

#[derive(Clone, Debug)]
struct Decision {
    id: String,
    // [model index] -> loss
    losses: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Action {
    id: String,
    action_type: String,
    cost: u64,
    requires: Vec<String>,
    provides: Vec<String>,
    eligible: bool,
    exclusion_reason: String,
    // [model index][shared nuisance state index][binary outcome]
    likelihoods: Vec<Vec<[f64; 2]>>,
}

#[derive(Clone, Debug)]
struct Plan {
    ordered_indices: Vec<usize>,
    cost: u64,
    evsi: f64,
    expected_posterior_risk: f64,
    mutual_information_nats: f64,
}

#[derive(Clone, Debug)]
struct Instance {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    models: Vec<Model>,
    nuisance_states: Vec<NuisanceState>,
    decisions: Vec<Decision>,
    outcome_labels: [String; 2],
    initial_capabilities: HashSet<String>,
    budget_units: u64,
    actions: Vec<Action>,
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() { return Err(format!("{context} must be finite")); }
    Ok(number)
}

fn strings(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let values = value.as_array().ok_or_else(|| format!("{context} must be an array"))?;
    let mut seen = HashSet::with_capacity(values.len());
    let mut result = Vec::with_capacity(values.len());
    for item in values {
        let text = item.as_str().filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("{context} entries must be non-empty strings"))?;
        if !seen.insert(text.to_owned()) {
            return Err(format!("{context} contains duplicate value {text}"));
        }
        result.push(text.to_owned());
    }
    Ok(result)
}

fn entropy(probabilities: impl IntoIterator<Item = f64>) -> f64 {
    probabilities.into_iter()
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

impl Instance {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|error| format!("schema JSON parse failed: {error}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator = validator_for(&schema)
            .map_err(|error| format!("schema compilation failed: {error}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!(
                "decision-value schema validation failed: {} at {}",
                error, error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("decision-value auditor accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-DECISION-VALUE-0.1" {
            return Err("unsupported decision-value contract".to_owned());
        }

        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let model_values = fixture.get("models").and_then(Value::as_array)
            .ok_or_else(|| "models must be an array".to_owned())?;
        let mut models = Vec::with_capacity(model_values.len());
        let mut model_index = HashMap::with_capacity(model_values.len());
        let mut model_prior_sum = 0.0;
        let mut none_count = 0;
        for (index, value) in model_values.iter().enumerate() {
            let id = required_text(value, "model_id")?.to_owned();
            if model_index.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate model_id {id}"));
            }
            let prior = finite_number(
                value.get("prior_probability").ok_or_else(|| {
                    format!("model {id} missing prior_probability")
                })?,
                &format!("model {id} prior_probability"),
            )?;
            if prior <= 0.0 || prior > 1.0 {
                return Err(format!("model prior for {id} must be in (0, 1]"));
            }
            let none_of_above = value.get("none_of_above").and_then(Value::as_bool)
                .ok_or_else(|| format!("model {id} none_of_above must be boolean"))?;
            if none_of_above { none_count += 1; }
            model_prior_sum += prior;
            models.push(Model { id, prior, none_of_above });
        }
        if none_count != 1 {
            return Err("exactly one none_of_above model is required".to_owned());
        }
        if (model_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("model priors sum to {model_prior_sum}, not 1"));
        }

        let nuisance_values = fixture.get("nuisance_states").and_then(Value::as_array)
            .ok_or_else(|| "nuisance_states must be an array".to_owned())?;
        let mut nuisance_states = Vec::with_capacity(nuisance_values.len());
        let mut nuisance_index = HashMap::with_capacity(nuisance_values.len());
        let mut nuisance_prior_sum = 0.0;
        for (index, value) in nuisance_values.iter().enumerate() {
            let id = required_text(value, "state_id")?.to_owned();
            if nuisance_index.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate nuisance state {id}"));
            }
            let prior = finite_number(
                value.get("prior_probability").ok_or_else(|| {
                    format!("nuisance state {id} missing prior_probability")
                })?,
                &format!("nuisance prior for {id}"),
            )?;
            if prior <= 0.0 || prior > 1.0 {
                return Err(format!("nuisance prior for {id} must be in (0, 1]"));
            }
            nuisance_prior_sum += prior;
            nuisance_states.push(NuisanceState { id, prior });
        }
        if (nuisance_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("nuisance priors sum to {nuisance_prior_sum}, not 1"));
        }

        let decision_values = fixture.get("decisions").and_then(Value::as_array)
            .ok_or_else(|| "decisions must be an array".to_owned())?;
        let mut decisions = Vec::with_capacity(decision_values.len());
        let mut decision_ids = HashSet::new();
        for value in decision_values {
            let id = required_text(value, "decision_id")?.to_owned();
            if !decision_ids.insert(id.clone()) {
                return Err(format!("duplicate decision_id {id}"));
            }
            let loss_rows = value.get("losses").and_then(Value::as_array)
                .ok_or_else(|| format!("decision {id} losses must be an array"))?;
            let mut by_model: Vec<Option<f64>> = vec![None; models.len()];
            for row in loss_rows {
                let model_id = required_text(row, "model_id")?;
                let Some(&mi) = model_index.get(model_id) else {
                    return Err(format!("decision {id} references unknown model {model_id}"));
                };
                if by_model[mi].is_some() {
                    return Err(format!("decision {id} duplicates loss for {model_id}"));
                }
                let loss = finite_number(
                    row.get("loss").ok_or_else(|| format!("decision {id} missing loss"))?,
                    "decision loss",
                )?;
                if loss < 0.0 { return Err(format!("decision {id} loss must be non-negative")); }
                by_model[mi] = Some(loss);
            }
            if by_model.iter().any(Option::is_none) {
                return Err(format!("decision {id} must define a loss for every model"));
            }
            decisions.push(Decision {
                id,
                losses: by_model.into_iter().map(Option::unwrap).collect(),
            });
        }

        let outcome_values = strings(
            fixture.get("outcome_labels").ok_or_else(|| "missing outcome_labels".to_owned())?,
            "outcome_labels",
        )?;
        if outcome_values.len() != 2 {
            return Err("this contract requires exactly two outcome labels".to_owned());
        }
        let outcome_labels = [outcome_values[0].clone(), outcome_values[1].clone()];
        let initial_capabilities = strings(
            fixture.get("initial_capabilities").ok_or_else(|| "missing initial_capabilities".to_owned())?,
            "initial_capabilities",
        )?.into_iter().collect();
        let budget_units = fixture.get("budget_units").and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| "budget_units must be positive".to_owned())?;

        let action_values = fixture.get("actions").and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_values.is_empty() || action_values.len() > 10 {
            return Err("action count must be between 1 and 10".to_owned());
        }
        let mut actions = Vec::with_capacity(action_values.len());
        let mut action_ids = HashSet::new();
        for value in action_values {
            let id = required_text(value, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) { return Err(format!("duplicate action_id {id}")); }
            let action_type = required_text(value, "action_type")?.to_owned();
            let cost = value.get("cost_units").and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("action {id} cost_units must be positive"))?;
            let requires = strings(
                value.get("requires_capabilities")
                    .ok_or_else(|| format!("action {id} missing requires_capabilities"))?,
                "requires_capabilities",
            )?;
            let provides = strings(
                value.get("provides_capabilities")
                    .ok_or_else(|| format!("action {id} missing provides_capabilities"))?,
                "provides_capabilities",
            )?;
            let eligible = value.get("eligible").and_then(Value::as_bool)
                .ok_or_else(|| format!("action {id} eligible must be boolean"))?;
            let exclusion_reason = value.get("exclusion_reason").and_then(Value::as_str)
                .ok_or_else(|| format!("action {id} exclusion_reason must be text"))?.to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!("ineligible action {id} requires an exclusion reason"));
            }
            let rows = value.get("model_likelihoods").and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} model_likelihoods must be an array"))?;
            let mut by_model: Vec<Option<Vec<Option<[f64; 2]>>>> = vec![None; models.len()];
            for row in rows {
                let model_id = required_text(row, "model_id")?;
                let Some(&mi) = model_index.get(model_id) else {
                    return Err(format!("action {id} references unknown model {model_id}"));
                };
                if by_model[mi].is_some() {
                    return Err(format!("action {id} duplicates model likelihood {model_id}"));
                }
                let nuisance_rows = row.get("nuisance_likelihoods").and_then(Value::as_array)
                    .ok_or_else(|| format!("action {id} nuisance_likelihoods must be an array"))?;
                let mut by_state: Vec<Option<[f64; 2]>> = vec![None; nuisance_states.len()];
                for nuisance_row in nuisance_rows {
                    let state_id = required_text(nuisance_row, "state_id")?;
                    let Some(&si) = nuisance_index.get(state_id) else {
                        return Err(format!("action {id} references unknown nuisance state {state_id}"));
                    };
                    if by_state[si].is_some() {
                        return Err(format!("action {id} duplicates nuisance state {state_id}"));
                    }
                    let probabilities = nuisance_row.get("outcome_probabilities")
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("action {id} probabilities must be a two-item array"))?;
                    if probabilities.len() != 2 {
                        return Err(format!("action {id} needs two outcome probabilities"));
                    }
                    let first = finite_number(&probabilities[0], "outcome probability")?;
                    let second = finite_number(&probabilities[1], "outcome probability")?;
                    if !(0.0..=1.0).contains(&first)
                        || !(0.0..=1.0).contains(&second)
                        || (first + second - 1.0).abs() > 1.0e-9
                    {
                        return Err(format!("action {id} probabilities must be normalized"));
                    }
                    by_state[si] = Some([first, second]);
                }
                if by_state.iter().any(Option::is_none) {
                    return Err(format!("action {id} lacks a nuisance likelihood for model {model_id}"));
                }
                by_model[mi] = Some(by_state.into_iter().map(Option::unwrap).collect());
            }
            let likelihoods = if action_type == "capability" {
                if !rows.is_empty() || provides.is_empty() {
                    return Err(format!("capability action {id} must provide capabilities and no outcomes"));
                }
                Vec::new()
            } else {
                if by_model.iter().any(Option::is_none) {
                    return Err(format!("action {id} requires likelihoods for every model"));
                }
                by_model.into_iter().map(Option::unwrap).collect()
            };
            actions.push(Action {
                id, action_type, cost, requires, provides, eligible, exclusion_reason, likelihoods
            });
        }

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            models,
            nuisance_states,
            decisions,
            outcome_labels,
            initial_capabilities,
            budget_units,
            actions,
        })
    }

    fn prior_probabilities(&self) -> Vec<f64> {
        self.models.iter().map(|model| model.prior).collect()
    }

    fn entropy_prior(&self) -> f64 {
        entropy(self.prior_probabilities())
    }

    fn prior_bayes_risk(&self) -> f64 {
        self.decisions.iter().map(|decision| {
            decision.losses.iter().zip(self.models.iter())
                .map(|(loss, model)| loss * model.prior).sum::<f64>()
        }).fold(f64::INFINITY, f64::min)
    }

    fn best_decision(&self, posterior: &[f64]) -> (usize, f64) {
        self.decisions.iter().enumerate().map(|(index, decision)| {
            let risk = decision.losses.iter().zip(posterior.iter())
                .map(|(loss, probability)| loss * probability).sum::<f64>();
            (index, risk)
        }).min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((0, f64::INFINITY))
    }

    fn measured_indices(&self, selected: &[usize]) -> Vec<usize> {
        selected.iter().copied()
            .filter(|index| !self.actions[*index].likelihoods.is_empty())
            .collect()
    }

    fn joint_model_likelihood(
        &self,
        measured: &[usize],
        outcome_mask: u64,
        model_index: usize,
    ) -> f64 {
        // A single shared nuisance state is drawn for the entire selected
        // measurement set, then outcome likelihoods are multiplied.
        self.nuisance_states.iter().enumerate().map(|(si, nuisance)| {
            let conditional = measured.iter().enumerate().fold(1.0, |product, (oi, action_index)| {
                let outcome = ((outcome_mask >> oi) & 1) as usize;
                product * self.actions[*action_index].likelihoods[model_index][si][outcome]
            });
            nuisance.prior * conditional
        }).sum()
    }

    fn metrics(&self, selected: &[usize]) -> Result<(f64, f64, f64), String> {
        let measured = self.measured_indices(selected);
        if measured.len() > 10 {
            return Err("too many observations for exact joint outcome enumeration".to_owned());
        }
        let prior_risk = self.prior_bayes_risk();
        let prior_entropy = self.entropy_prior();
        if measured.is_empty() {
            return Ok((0.0, prior_risk, 0.0));
        }
        let mut expected_posterior_risk = 0.0;
        let mut expected_posterior_entropy = 0.0;
        for outcome_mask in 0..(1_u64 << measured.len()) {
            let masses: Vec<f64> = self.models.iter().enumerate().map(|(mi, model)| {
                model.prior * self.joint_model_likelihood(&measured, outcome_mask, mi)
            }).collect();
            let outcome_probability: f64 = masses.iter().sum();
            if outcome_probability <= 0.0 { continue; }
            let posterior: Vec<f64> = masses.iter()
                .map(|mass| mass / outcome_probability).collect();
            let (_, posterior_risk) = self.best_decision(&posterior);
            expected_posterior_risk += outcome_probability * posterior_risk;
            expected_posterior_entropy += outcome_probability * entropy(posterior);
        }
        let evsi = (prior_risk - expected_posterior_risk).max(0.0);
        let information_gain = (prior_entropy - expected_posterior_entropy).max(0.0);
        if !evsi.is_finite() || !information_gain.is_finite()
            || !expected_posterior_risk.is_finite() {
            return Err("decision-value metrics are non-finite".to_owned());
        }
        Ok((evsi, expected_posterior_risk, information_gain))
    }

    fn feasible_order(&self, indices: &[usize]) -> Option<(u64, Vec<usize>)> {
        let mut selected = indices.to_vec();
        selected.sort_unstable();
        selected.dedup();
        if selected.len() != indices.len() { return None; }
        let mut cost = 0_u64;
        for index in &selected {
            let action = self.actions.get(*index)?;
            if !action.eligible { return None; }
            cost = cost.checked_add(action.cost)?;
            if cost > self.budget_units { return None; }
        }
        let mut available = self.initial_capabilities.clone();
        let mut remaining = selected;
        let mut order = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let position = remaining.iter().position(|index| {
                self.actions[*index].requires.iter()
                    .all(|capability| available.contains(capability))
            })?;
            let index = remaining.remove(position);
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
            order.push(index);
        }
        Some((cost, order))
    }

    fn plan(&self, indices: &[usize]) -> Result<Option<Plan>, String> {
        let Some((cost, order)) = self.feasible_order(indices) else { return Ok(None); };
        let (evsi, expected_posterior_risk, mutual_information_nats) = self.metrics(&order)?;
        Ok(Some(Plan {
            ordered_indices: order,
            cost,
            evsi,
            expected_posterior_risk,
            mutual_information_nats,
        }))
    }

    fn exact_oracle(&self, objective: Objective) -> Result<(Plan, u64), String> {
        let subsets = 1_u64.checked_shl(self.actions.len() as u32)
            .ok_or_else(|| "subset count overflowed".to_owned())?;
        let empty = self.plan(&[])?.ok_or_else(|| "empty plan must be feasible".to_owned())?;
        let mut best = empty;
        for mask in 1..subsets {
            let indices: Vec<usize> = (0..self.actions.len())
                .filter(|index| (mask & (1_u64 << *index)) != 0).collect();
            let Some(candidate) = self.plan(&indices)? else { continue; };
            if is_better(&candidate, &best, &self.actions, objective) { best = candidate; }
        }
        Ok((best, subsets))
    }

    fn greedy(&self, objective: Objective) -> Result<Plan, String> {
        let mut chosen = Vec::new();
        let mut available = self.initial_capabilities.clone();
        let mut selected = HashSet::new();
        let mut cost = 0_u64;
        loop {
            let current = self.plan(&chosen)?
                .ok_or_else(|| "greedy plan unexpectedly infeasible".to_owned())?;
            let current_score = score(&current, objective);
            let mut best: Option<(usize, f64, Plan)> = None;
            for index in 0..self.actions.len() {
                if selected.contains(&index) { continue; }
                let action = &self.actions[index];
                if !action.eligible
                    || action.cost > self.budget_units.saturating_sub(cost)
                    || !action.requires.iter().all(|capability| available.contains(capability)) {
                    continue;
                }
                let mut trial = chosen.clone();
                trial.push(index);
                let Some(candidate) = self.plan(&trial)? else { continue; };
                let marginal = score(&candidate, objective) - current_score;
                let ratio = marginal / action.cost as f64;
                if marginal <= 1.0e-12 { continue; }
                let replace = match &best {
                    None => true,
                    Some((best_index, best_ratio, _)) =>
                        ratio > *best_ratio + 1.0e-12
                        || ((ratio - *best_ratio).abs() <= 1.0e-12
                            && action.id < self.actions[*best_index].id),
                };
                if replace { best = Some((index, ratio, candidate)); }
            }
            let Some((index, _, candidate)) = best else { return Ok(current); };
            chosen.push(index);
            selected.insert(index);
            cost = candidate.cost;
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
        }
    }

    fn plan_receipt(&self, plan: &Plan) -> Value {
        let decision = self.best_decision(&self.prior_probabilities());
        json!({
            "ordered_action_ids": plan.ordered_indices.iter()
                .map(|index| self.actions[*index].id.as_str()).collect::<Vec<_>>(),
            "ordered_action_types": plan.ordered_indices.iter()
                .map(|index| self.actions[*index].action_type.as_str()).collect::<Vec<_>>(),
            "selected_capabilities": plan.ordered_indices.iter()
                .flat_map(|index| self.actions[*index].provides.iter().cloned())
                .collect::<Vec<_>>(),
            "cost_units": plan.cost,
            "remaining_budget_units": self.budget_units - plan.cost,
            "expected_value_of_sample_information": plan.evsi,
            "expected_posterior_decision_risk": plan.expected_posterior_risk,
            "expected_model_information_gain_nats": plan.mutual_information_nats,
            "prior_optimal_decision_id": self.decisions[decision.0].id
        })
    }
}

#[derive(Clone, Copy)]
enum Objective { DecisionValue, InformationGain }

fn score(plan: &Plan, objective: Objective) -> f64 {
    match objective {
        Objective::DecisionValue => plan.evsi,
        Objective::InformationGain => plan.mutual_information_nats,
    }
}

fn is_better(candidate: &Plan, incumbent: &Plan, actions: &[Action], objective: Objective) -> bool {
    let candidate_score = score(candidate, objective);
    let incumbent_score = score(incumbent, objective);
    if candidate_score > incumbent_score + 1.0e-12 { return true; }
    if (candidate_score - incumbent_score).abs() > 1.0e-12 { return false; }
    if candidate.cost != incumbent.cost { return candidate.cost < incumbent.cost; }
    let candidate_ids: Vec<&str> = candidate.ordered_indices.iter()
        .map(|index| actions[*index].id.as_str()).collect();
    let incumbent_ids: Vec<&str> = incumbent.ordered_indices.iter()
        .map(|index| actions[*index].id.as_str()).collect();
    candidate_ids < incumbent_ids
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance = Instance::from_bytes(schema_bytes, fixture_bytes)?;
    let (decision_oracle, subset_count) = instance.exact_oracle(Objective::DecisionValue)?;
    let (information_oracle, _) = instance.exact_oracle(Objective::InformationGain)?;
    let decision_greedy = instance.greedy(Objective::DecisionValue)?;
    let information_greedy = instance.greedy(Objective::InformationGain)?;
    let prior_decision = instance.best_decision(&instance.prior_probabilities());
    let decision_oracle_ids: Vec<usize> = decision_oracle.ordered_indices.clone();
    let information_oracle_ids: Vec<usize> = information_oracle.ordered_indices.clone();
    let selected_by_both = decision_oracle_ids.iter()
        .filter(|index| information_oracle_ids.contains(index)).count();
    let excluded_actions: Vec<Value> = instance.actions.iter()
        .filter(|action| !action.eligible)
        .map(|action| json!({
            "action_id": action.id,
            "exclusion_reason": action.exclusion_reason
        })).collect();

    Ok(json!({
        "status": "synthetic_contract_pass",
        "contract": "SC-DECISION-VALUE-AUDIT-0.1",
        "instance_id": instance.instance_id,
        "schema_sha256": instance.schema_sha256,
        "fixture_sha256": instance.fixture_sha256,
        "model_priors": instance.models.iter().map(|model| json!({
            "model_id": model.id,
            "prior_probability": model.prior,
            "none_of_above": model.none_of_above
        })).collect::<Vec<_>>(),
        "shared_nuisance_priors": instance.nuisance_states.iter().map(|state| json!({
            "state_id": state.id,
            "prior_probability": state.prior
        })).collect::<Vec<_>>(),
        "decisions": instance.decisions.iter().map(|decision| json!({
            "decision_id": decision.id,
            "losses_by_model": decision.losses
        })).collect::<Vec<_>>(),
        "prior_decision": {
            "decision_id": instance.decisions[prior_decision.0].id,
            "expected_loss": prior_decision.1
        },
        "budget_units": instance.budget_units,
        "action_count": instance.actions.len(),
        "eligible_action_count": instance.actions.iter().filter(|action| action.eligible).count(),
        "excluded_actions": excluded_actions,
        "decision_value_oracle": {
            "algorithm": "exact feasible subset enumeration; shared-nuisance marginalization; minimum posterior expected downstream loss",
            "enumerated_subsets": subset_count,
            "plan": instance.plan_receipt(&decision_oracle)
        },
        "myopic_decision_value_baseline": {
            "algorithm": "greedy marginal expected decision-risk reduction per cost",
            "plan": instance.plan_receipt(&decision_greedy)
        },
        "mutual_information_oracle": {
            "algorithm": "exact feasible subset enumeration maximizing I(M;Y_S)",
            "plan": instance.plan_receipt(&information_oracle)
        },
        "myopic_information_gain_baseline": {
            "algorithm": "greedy marginal model information gain per cost",
            "plan": instance.plan_receipt(&information_greedy)
        },
        "comparison": {
            "decision_value_oracle_minus_greedy": decision_oracle.evsi - decision_greedy.evsi,
            "information_oracle_minus_greedy_nats": information_oracle.mutual_information_nats - information_greedy.mutual_information_nats,
            "decision_value_oracle_evsi": decision_oracle.evsi,
            "information_oracle_evsi": information_oracle.evsi,
            "information_oracle_extra_cost_units": information_oracle.cost.saturating_sub(decision_oracle.cost),
            "same_decision_value_with_fewer_cost_units": decision_oracle.evsi >= information_oracle.evsi - 1.0e-10
                && decision_oracle.cost < information_oracle.cost,
            "shared_actions_between_exact_objectives": selected_by_both
        },
        "claim_ceiling": "Synthetic decision-value planning contract only; model likelihoods and downstream losses are illustrative, not calibrated to a real sample, instrument, or application."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
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
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
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
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-DECISION-VALUE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing decision-value receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_decision_value_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research/superconductivity_decision_value_fixture_v0_1.json")).unwrap(),
        )
    }

    fn mutated_fixture(mut mutation: impl FnMut(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        audit(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn decision_value_oracle_chooses_cheaper_downstream_useful_plan_than_information_oracle() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let decision = &receipt["decision_value_oracle"]["plan"];
        let information = &receipt["mutual_information_oracle"]["plan"];
        assert_eq!(decision["cost_units"], 5);
        assert!((decision["expected_value_of_sample_information"].as_f64().unwrap()
            - 0.8931875).abs() < 1.0e-10);
        assert_eq!(information["cost_units"], 7);
        assert!((decision["expected_value_of_sample_information"].as_f64().unwrap()
            - information["expected_value_of_sample_information"].as_f64().unwrap()).abs() < 1.0e-10);
        assert_eq!(receipt["comparison"]["same_decision_value_with_fewer_cost_units"], true);
        let ids: Vec<&str> = decision["ordered_action_ids"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap()).collect();
        assert!(ids.contains(&"qualify_thermodynamic_channel"));
        assert!(ids.contains(&"bulk_thermodynamic_observation"));
        assert!(ids.contains(&"stiffness_tensor"));
        assert!(!ids.contains(&"field_angle_hc2"));
        assert!(!ids.contains(&"unreviewed_high_field_probe"));
    }

    #[test]
    fn myopic_evsi_greedy_does_not_unlock_zero_immediate_information_capability() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let ids = receipt["myopic_decision_value_baseline"]["plan"]["ordered_action_ids"]
            .as_array().unwrap();
        assert!(!ids.iter().any(|value| value == "qualify_thermodynamic_channel"));
        assert!(!ids.iter().any(|value| value == "bulk_thermodynamic_observation"));
        assert!(receipt["comparison"]["decision_value_oracle_minus_greedy"]
            .as_f64().unwrap() > 0.2);
    }

    #[test]
    fn capability_actions_cannot_claim_observations() {
        assert!(mutated_fixture(|fixture| {
            fixture["actions"][0]["model_likelihoods"] =
                fixture["actions"][3]["model_likelihoods"].clone();
        }).is_err());
    }

    #[test]
    fn every_decision_must_score_every_model() {
        assert!(mutated_fixture(|fixture| {
            fixture["decisions"][0]["losses"].as_array_mut().unwrap().pop();
        }).is_err());
    }

    #[test]
    fn exactly_one_none_of_above_model_is_required() {
        assert!(mutated_fixture(|fixture| {
            fixture["models"][3]["none_of_above"] = json!(false);
        }).is_err());
    }

    #[test]
    fn unnormalized_nuisance_prior_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["nuisance_states"][0]["prior_probability"] = json!(0.2);
        }).is_err());
    }

    #[test]
    fn ineligible_high_information_action_never_enters_either_oracle() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        for key in ["decision_value_oracle", "mutual_information_oracle"] {
            assert!(!receipt[key]["plan"]["ordered_action_ids"].as_array().unwrap()
                .iter().any(|value| value == "unreviewed_high_field_probe"));
        }
    }

    #[test]
    fn shared_nuisance_model_is_deterministic() {
        let (schema, fixture) = inputs();
        assert_eq!(audit(&schema, &fixture).unwrap(), audit(&schema, &fixture).unwrap());
    }
}
