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
struct Action {
    id: String,
    action_type: String,
    cost: u64,
    requires: Vec<String>,
    provides: Vec<String>,
    eligible: bool,
    exclusion_reason: String,
    // [model index][shared nuisance-state index][binary outcome index]
    likelihoods: Vec<Vec<[f64; 2]>>,
}

#[derive(Clone, Debug)]
struct Plan {
    ordered_indices: Vec<usize>,
    cost: u64,
    information_gain_nats: f64,
}

#[derive(Clone, Debug)]
struct Instance {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    models: Vec<Model>,
    nuisance_states: Vec<NuisanceState>,
    outcome_labels: [String; 2],
    initial_capabilities: HashSet<String>,
    budget_units: u64,
    actions: Vec<Action>,
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty text field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{context} must be finite"));
    }
    Ok(number)
}

fn parse_string_set(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    let mut result = Vec::with_capacity(values.len());
    let mut seen = HashSet::with_capacity(values.len());
    for item in values {
        let text = item
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{context} entries must be non-empty strings"))?;
        if !seen.insert(text.to_owned()) {
            return Err(format!("{context} contains duplicate entry {text}"));
        }
        result.push(text.to_owned());
    }
    Ok(result)
}

fn entropy(probabilities: impl IntoIterator<Item = f64>) -> f64 {
    probabilities
        .into_iter()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
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
        let validator =
            validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!(
                "correlated-evidence schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("correlated-evidence auditor accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-CORRELATED-EVIDENCE-0.1" {
            return Err("unsupported correlated-evidence contract".to_owned());
        }

        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let model_values = fixture
            .get("models")
            .and_then(Value::as_array)
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
                return Err(format!("model {id} prior_probability must be in (0, 1]"));
            }
            let none_of_above = value
                .get("none_of_above")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("model {id} none_of_above must be boolean"))?;
            if none_of_above {
                none_count += 1;
            }
            model_prior_sum += prior;
            models.push(Model {
                id,
                prior,
                none_of_above,
            });
        }
        if none_count != 1 {
            return Err("exactly one none_of_above model is required".to_owned());
        }
        if (model_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("model priors sum to {model_prior_sum}, not 1"));
        }

        let nuisance_values = fixture
            .get("nuisance_states")
            .and_then(Value::as_array)
            .ok_or_else(|| "nuisance_states must be an array".to_owned())?;
        let mut nuisance_states = Vec::with_capacity(nuisance_values.len());
        let mut nuisance_index = HashMap::with_capacity(nuisance_values.len());
        let mut nuisance_prior_sum = 0.0;
        for (index, value) in nuisance_values.iter().enumerate() {
            let id = required_text(value, "state_id")?.to_owned();
            if nuisance_index.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate nuisance state_id {id}"));
            }
            let prior = finite_number(
                value.get("prior_probability").ok_or_else(|| {
                    format!("nuisance state {id} missing prior_probability")
                })?,
                &format!("nuisance state {id} prior_probability"),
            )?;
            if prior <= 0.0 || prior > 1.0 {
                return Err(format!("nuisance prior for {id} must be in (0, 1]"));
            }
            nuisance_prior_sum += prior;
            nuisance_states.push(NuisanceState { id, prior });
        }
        if (nuisance_prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!(
                "nuisance-state priors sum to {nuisance_prior_sum}, not 1"
            ));
        }

        let outcome_values = parse_string_set(
            fixture.get("outcome_labels").ok_or_else(|| "missing outcome_labels".to_owned())?,
            "outcome_labels",
        )?;
        if outcome_values.len() != 2 {
            return Err("this initial contract requires exactly two outcome labels".to_owned());
        }
        let outcome_labels = [outcome_values[0].clone(), outcome_values[1].clone()];
        let initial_capabilities = parse_string_set(
            fixture.get("initial_capabilities")
                .ok_or_else(|| "missing initial_capabilities".to_owned())?,
            "initial_capabilities",
        )?
        .into_iter()
        .collect();

        let budget_units = fixture.get("budget_units")
            .and_then(Value::as_u64)
            .filter(|budget| *budget > 0)
            .ok_or_else(|| "budget_units must be a positive integer".to_owned())?;
        let action_values = fixture.get("actions")
            .and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_values.is_empty() || action_values.len() > 10 {
            return Err("action count must be between 1 and 10".to_owned());
        }

        let mut actions = Vec::with_capacity(action_values.len());
        let mut action_ids = HashSet::with_capacity(action_values.len());
        for value in action_values {
            let id = required_text(value, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) {
                return Err(format!("duplicate action_id {id}"));
            }
            let action_type = required_text(value, "action_type")?.to_owned();
            let cost = value.get("cost_units")
                .and_then(Value::as_u64)
                .filter(|cost| *cost > 0)
                .ok_or_else(|| format!("action {id} cost_units must be positive"))?;
            let requires = parse_string_set(
                value.get("requires_capabilities")
                    .ok_or_else(|| format!("action {id} missing requires_capabilities"))?,
                &format!("action {id} requires_capabilities"),
            )?;
            let provides = parse_string_set(
                value.get("provides_capabilities")
                    .ok_or_else(|| format!("action {id} missing provides_capabilities"))?,
                &format!("action {id} provides_capabilities"),
            )?;
            let eligible = value.get("eligible")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("action {id} eligible must be boolean"))?;
            let exclusion_reason = value.get("exclusion_reason")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("action {id} exclusion_reason must be a string"))?
                .to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!("ineligible action {id} requires an exclusion reason"));
            }
            let likelihood_values = value.get("model_likelihoods")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} model_likelihoods must be an array"))?;

            let mut by_model: Vec<Option<Vec<Option<[f64; 2]>>>> =
                vec![None; models.len()];
            for row in likelihood_values {
                let model_id = required_text(row, "model_id")?;
                let Some(&mi) = model_index.get(model_id) else {
                    return Err(format!("action {id} references unknown model {model_id}"));
                };
                if by_model[mi].is_some() {
                    return Err(format!("action {id} duplicates model likelihood {model_id}"));
                }
                let state_rows = row.get("nuisance_likelihoods")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("action {id} nuisance_likelihoods must be an array"))?;
                let mut by_state: Vec<Option<[f64; 2]>> = vec![None; nuisance_states.len()];
                for state_row in state_rows {
                    let state_id = required_text(state_row, "state_id")?;
                    let Some(&si) = nuisance_index.get(state_id) else {
                        return Err(format!("action {id} references unknown nuisance state {state_id}"));
                    };
                    if by_state[si].is_some() {
                        return Err(format!("action {id} duplicates nuisance state {state_id}"));
                    }
                    let probabilities = state_row.get("outcome_probabilities")
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("action {id} probabilities must be an array"))?;
                    if probabilities.len() != 2 {
                        return Err(format!("action {id} must have two outcome probabilities"));
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
                    return Err(format!("action {id} is missing a nuisance-state likelihood for {model_id}"));
                }
                by_model[mi] = Some(by_state.into_iter().map(Option::unwrap).collect());
            }

            let likelihoods = if action_type == "capability" {
                if !likelihood_values.is_empty() || provides.is_empty() {
                    return Err(format!(
                        "capability action {id} must provide capabilities and cannot produce observations"
                    ));
                }
                Vec::new()
            } else {
                if by_model.iter().any(Option::is_none) {
                    return Err(format!("action {id} must define likelihoods for every model"));
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
            outcome_labels,
            initial_capabilities,
            budget_units,
            actions,
        })
    }

    fn prior_model_entropy(&self) -> f64 {
        entropy(self.models.iter().map(|model| model.prior))
    }

    fn measured_indices(&self, selected: &[usize]) -> Vec<usize> {
        selected.iter().copied()
            .filter(|index| !self.actions[*index].likelihoods.is_empty())
            .collect()
    }

    // Shared-state model: one nuisance state is drawn for the entire selected
    // observation set, then outcomes are conditionally independent given
    // (model, nuisance state). This captures a common calibration drift.
    fn mutual_information_shared(&self, selected: &[usize]) -> Result<f64, String> {
        let measured = self.measured_indices(selected);
        if measured.is_empty() { return Ok(0.0); }
        if measured.len() > 10 {
            return Err("too many measured actions for exact joint outcome enumeration".to_owned());
        }
        let mut expected_entropy = 0.0;
        for outcome_mask in 0..(1_u64 << measured.len()) {
            let mut joint_by_model = vec![0.0; self.models.len()];
            for (mi, model) in self.models.iter().enumerate() {
                let mut likelihood = 0.0;
                for (si, nuisance) in self.nuisance_states.iter().enumerate() {
                    let mut conditional = 1.0;
                    for (oi, action_index) in measured.iter().enumerate() {
                        let outcome = ((outcome_mask >> oi) & 1) as usize;
                        conditional *= self.actions[*action_index].likelihoods[mi][si][outcome];
                    }
                    likelihood += nuisance.prior * conditional;
                }
                joint_by_model[mi] = model.prior * likelihood;
            }
            let outcome_probability: f64 = joint_by_model.iter().sum();
            if outcome_probability <= 0.0 { continue; }
            expected_entropy += outcome_probability * entropy(
                joint_by_model.iter().map(|value| value / outcome_probability)
            );
        }
        let information = self.prior_model_entropy() - expected_entropy;
        if !information.is_finite() || information < -1.0e-9 {
            return Err("shared-nuisance information gain is invalid".to_owned());
        }
        Ok(information.clamp(0.0, self.prior_model_entropy()))
    }

    // Incorrect factorized comparator: marginalizes nuisance independently
    // for each action, thereby redrawing a shared calibration state each time.
    fn mutual_information_factorized_nuisance(&self, selected: &[usize]) -> Result<f64, String> {
        let measured = self.measured_indices(selected);
        if measured.is_empty() { return Ok(0.0); }
        if measured.len() > 10 {
            return Err("too many measured actions for exact factorized outcome enumeration".to_owned());
        }
        let mut expected_entropy = 0.0;
        for outcome_mask in 0..(1_u64 << measured.len()) {
            let mut joint_by_model = vec![0.0; self.models.len()];
            for (mi, model) in self.models.iter().enumerate() {
                let mut likelihood = 1.0;
                for (oi, action_index) in measured.iter().enumerate() {
                    let outcome = ((outcome_mask >> oi) & 1) as usize;
                    let marginal: f64 = self.nuisance_states.iter()
                        .enumerate()
                        .map(|(si, nuisance)| {
                            nuisance.prior * self.actions[*action_index].likelihoods[mi][si][outcome]
                        })
                        .sum();
                    likelihood *= marginal;
                }
                joint_by_model[mi] = model.prior * likelihood;
            }
            let outcome_probability: f64 = joint_by_model.iter().sum();
            if outcome_probability <= 0.0 { continue; }
            expected_entropy += outcome_probability * entropy(
                joint_by_model.iter().map(|value| value / outcome_probability)
            );
        }
        let information = self.prior_model_entropy() - expected_entropy;
        if !information.is_finite() || information < -1.0e-9 {
            return Err("factorized information gain is invalid".to_owned());
        }
        Ok(information.clamp(0.0, self.prior_model_entropy()))
    }

    fn plan_from_indices(&self, indices: &[usize]) -> Result<Option<Plan>, String> {
        let mut selected = indices.to_vec();
        selected.sort_unstable();
        selected.dedup();
        if selected.len() != indices.len() { return Ok(None); }

        let mut cost = 0_u64;
        for index in &selected {
            let Some(action) = self.actions.get(*index) else { return Ok(None); };
            if !action.eligible { return Ok(None); }
            cost = match cost.checked_add(action.cost) {
                Some(value) if value <= self.budget_units => value,
                _ => return Ok(None),
            };
        }

        let mut available = self.initial_capabilities.clone();
        let mut remaining = selected.clone();
        let mut ordered_indices = Vec::with_capacity(selected.len());
        while !remaining.is_empty() {
            let position = remaining.iter().position(|index| {
                self.actions[*index].requires.iter()
                    .all(|capability| available.contains(capability))
            });
            let Some(position) = position else { return Ok(None); };
            let index = remaining.remove(position);
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
            ordered_indices.push(index);
        }

        let information_gain_nats = self.mutual_information_shared(&selected)?;
        Ok(Some(Plan { ordered_indices, cost, information_gain_nats }))
    }

    fn exact_oracle(&self) -> Result<(Plan, u64), String> {
        let subset_count = 1_u64.checked_shl(self.actions.len() as u32)
            .ok_or_else(|| "action subset count overflowed".to_owned())?;
        let mut best = self.plan_from_indices(&[])?
            .ok_or_else(|| "empty plan should be feasible".to_owned())?;
        for mask in 1..subset_count {
            let indices: Vec<usize> = (0..self.actions.len())
                .filter(|index| (mask & (1_u64 << *index)) != 0).collect();
            let Some(candidate) = self.plan_from_indices(&indices)? else { continue; };
            if is_better_plan(&candidate, &best, &self.actions) { best = candidate; }
        }
        Ok((best, subset_count))
    }

    fn greedy_information_per_cost(&self) -> Result<Plan, String> {
        let mut chosen: Vec<usize> = Vec::new();
        let mut selected_set = HashSet::new();
        let mut available = self.initial_capabilities.clone();
        let mut cost = 0_u64;
        loop {
            let current = self.plan_from_indices(&chosen)?
                .ok_or_else(|| "greedy planner lost feasibility".to_owned())?;
            let mut best: Option<(usize, f64, Plan)> = None;
            for index in 0..self.actions.len() {
                if selected_set.contains(&index) { continue; }
                let action = &self.actions[index];
                if !action.eligible
                    || action.cost > self.budget_units.saturating_sub(cost)
                    || !action.requires.iter().all(|capability| available.contains(capability))
                { continue; }
                let mut trial = chosen.clone();
                trial.push(index);
                let Some(plan) = self.plan_from_indices(&trial)? else { continue; };
                let marginal = plan.information_gain_nats - current.information_gain_nats;
                let ratio = marginal / action.cost as f64;
                if marginal <= 1.0e-12 { continue; }
                let replace = match &best {
                    None => true,
                    Some((best_index, best_ratio, _)) =>
                        ratio > *best_ratio + 1.0e-12
                        || ((ratio - *best_ratio).abs() <= 1.0e-12
                            && action.id < self.actions[*best_index].id)
                };
                if replace { best = Some((index, ratio, plan)); }
            }
            let Some((index, _, plan)) = best else { return Ok(current); };
            selected_set.insert(index);
            chosen.push(index);
            cost = plan.cost;
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
        }
    }

    fn receipt_plan(&self, plan: &Plan) -> Value {
        json!({
            "ordered_action_ids": plan.ordered_indices.iter()
                .map(|index| self.actions[*index].id.as_str()).collect::<Vec<_>>(),
            "ordered_action_types": plan.ordered_indices.iter()
                .map(|index| self.actions[*index].action_type.as_str()).collect::<Vec<_>>(),
            "selected_capabilities": plan.ordered_indices.iter()
                .flat_map(|index| self.actions[*index].provides.iter().cloned()).collect::<Vec<_>>(),
            "cost_units": plan.cost,
            "remaining_budget_units": self.budget_units - plan.cost,
            "expected_model_information_gain_nats": plan.information_gain_nats
        })
    }
}

fn is_better_plan(candidate: &Plan, incumbent: &Plan, actions: &[Action]) -> bool {
    if candidate.information_gain_nats > incumbent.information_gain_nats + 1.0e-12 { return true; }
    if (candidate.information_gain_nats - incumbent.information_gain_nats).abs() > 1.0e-12 { return false; }
    if candidate.cost != incumbent.cost { return candidate.cost < incumbent.cost; }
    let candidate_ids: Vec<&str> = candidate.ordered_indices.iter()
        .map(|index| actions[*index].id.as_str()).collect();
    let incumbent_ids: Vec<&str> = incumbent.ordered_indices.iter()
        .map(|index| actions[*index].id.as_str()).collect();
    candidate_ids < incumbent_ids
}

fn audit_and_plan(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance = Instance::from_bytes(schema_bytes, fixture_bytes)?;
    let (oracle, subset_count) = instance.exact_oracle()?;
    let greedy = instance.greedy_information_per_cost()?;
    let oracle_indices = oracle.ordered_indices.clone();
    let independent_information =
        instance.mutual_information_factorized_nuisance(&oracle_indices)?;
    let factorization_delta = independent_information - oracle.information_gain_nats;

    let reference_index = instance.actions.iter()
        .position(|action| action.id == "shared_drift_reference_probe")
        .ok_or_else(|| "fixture missing shared-drift reference probe".to_owned())?;
    let heat_index = instance.actions.iter()
        .position(|action| action.id == "bulk_thermodynamic_observation_a")
        .ok_or_else(|| "fixture missing thermodynamic observation".to_owned())?;
    let reference_only = instance.mutual_information_shared(&[reference_index])?;
    let heat_only = instance.mutual_information_shared(&[heat_index])?;
    let reference_and_heat =
        instance.mutual_information_shared(&[reference_index, heat_index])?;

    let excluded_actions: Vec<Value> = instance.actions.iter()
        .filter(|action| !action.eligible)
        .map(|action| json!({
            "action_id": action.id,
            "exclusion_reason": action.exclusion_reason
        })).collect();

    let oracle_minus_greedy =
        (oracle.information_gain_nats - greedy.information_gain_nats).max(0.0);
    let greedy_fraction_of_oracle = if oracle.information_gain_nats > 1.0e-12 {
        greedy.information_gain_nats / oracle.information_gain_nats
    } else {
        1.0
    };

    Ok(json!({
        "status": "synthetic_contract_pass",
        "contract": "SC-CORRELATED-EVIDENCE-AUDIT-0.1",
        "instance_id": instance.instance_id,
        "schema_sha256": instance.schema_sha256,
        "fixture_sha256": instance.fixture_sha256,
        "models": instance.models.iter().map(|model| json!({
            "model_id": model.id,
            "prior_probability": model.prior,
            "none_of_above": model.none_of_above
        })).collect::<Vec<_>>(),
        "shared_nuisance_states": instance.nuisance_states.iter().map(|state| json!({
            "state_id": state.id,
            "prior_probability": state.prior
        })).collect::<Vec<_>>(),
        "outcome_labels": instance.outcome_labels,
        "budget_units": instance.budget_units,
        "action_count": instance.actions.len(),
        "eligible_action_count": instance.actions.iter().filter(|action| action.eligible).count(),
        "excluded_actions": excluded_actions,
        "joint_model": "one nuisance/calibration state is drawn once for the entire selected observation set; outcomes are conditionally independent only given both model and that shared state",
        "exact_oracle": {
            "algorithm": "exhaustive feasible-subset enumeration with exact shared-nuisance marginalization and capability-closure ordering",
            "enumerated_subsets": subset_count,
            "prior_model_entropy_nats": instance.prior_model_entropy(),
            "plan": instance.receipt_plan(&oracle)
        },
        "myopic_baseline": {
            "algorithm": "greedy marginal shared-nuisance model information gain per cost; zero-information capability actions are not selected"
            ,"plan": instance.receipt_plan(&greedy)
        },
        "correlation_diagnostics": {
            "reference_probe_information_alone_nats": reference_only,
            "thermodynamic_observation_information_alone_nats": heat_only,
            "joint_reference_and_thermodynamic_information_nats": reference_and_heat,
            "reference_probe_has_zero_standalone_model_information": reference_only < 1.0e-10,
            "reference_probe_adds_conditional_model_information": reference_and_heat > heat_only + 1.0e-4,
            "factorized_marginal_approximation_nats": independent_information,
            "factorized_minus_shared_nats": factorization_delta,
            "shared_nuisance_model_materially_changes_information": factorization_delta.abs() > 1.0e-3
        },
        "comparison": {
            "oracle_information_gain_minus_myopic_nats": oracle_minus_greedy,
            "myopic_fraction_of_oracle_information_gain": greedy_fraction_of_oracle,
            "capability_gating_gap_detected": oracle_minus_greedy > 1.0e-9
        },
        "claim_ceiling": "Synthetic shared-nuisance planning contract only; calibration states and likelihoods are illustrative, not fitted to a real specimen or physical instrument."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <instance.json> [--out <receipt.json>]");
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
    let receipt = audit_and_plan(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-CORRELATED-EVIDENCE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing correlated-evidence receipt should not fail");
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
            fs::read(root.join("data/schemas/superconductivity_correlated_evidence_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research/superconductivity_correlated_evidence_fixture_v0_1.json")).unwrap(),
        )
    }

    fn mutated_fixture(mut mutation: impl FnMut(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        audit_and_plan(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn exact_plan_uses_independent_reference_to_resolve_shared_drift() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let plan = &receipt["exact_oracle"]["plan"];
        assert!((plan["expected_model_information_gain_nats"].as_f64().unwrap()
            - 0.424973426561594).abs() < 1.0e-10);
        let ids: Vec<&str> = plan["ordered_action_ids"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap()).collect();
        for required in [
            "qualify_thermodynamic_channel",
            "bulk_thermodynamic_observation_a",
            "bulk_thermodynamic_observation_b",
            "qualify_reference_channel",
            "shared_drift_reference_probe",
            "field_angle_hc2",
            "stiffness_tensor"
        ] {
            assert!(ids.contains(&required), "missing selected action {required}");
        }
        assert!(
            ids.iter().position(|id| *id == "qualify_reference_channel").unwrap()
                < ids.iter().position(|id| *id == "shared_drift_reference_probe").unwrap()
        );
        assert!(!ids.contains(&"unreviewed_high_field_probe"));
    }

    #[test]
    fn myopic_policy_misses_both_capability_chains() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let plan = &receipt["myopic_baseline"]["plan"];
        let ids: Vec<&str> = plan["ordered_action_ids"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap()).collect();
        assert!(!ids.contains(&"qualify_thermodynamic_channel"));
        assert!(!ids.contains(&"qualify_reference_channel"));
        assert!(!ids.contains(&"shared_drift_reference_probe"));
        assert!(receipt["comparison"]["oracle_information_gain_minus_myopic_nats"]
            .as_f64().unwrap_or(0.0) > 0.05);
    }

    #[test]
    fn reference_control_has_zero_standalone_model_information_but_positive_joint_value() {
        let (schema, fixture) = inputs();
        let instance = Instance::from_bytes(&schema, &fixture).unwrap();
        let reference = instance.actions.iter().position(|a| a.id == "shared_drift_reference_probe").unwrap();
        let heat = instance.actions.iter().position(|a| a.id == "bulk_thermodynamic_observation_a").unwrap();
        let alone = instance.mutual_information_shared(&[reference]).unwrap();
        let heat_alone = instance.mutual_information_shared(&[heat]).unwrap();
        let joint = instance.mutual_information_shared(&[reference, heat]).unwrap();
        assert!(alone < 1.0e-10);
        assert!(joint > heat_alone + 0.02);
    }

    #[test]
    fn factorized_per_action_nuisance_marginalization_is_not_equivalent_to_shared_state() {
        let (schema, fixture) = inputs();
        let instance = Instance::from_bytes(&schema, &fixture).unwrap();
        let heat_a = instance.actions.iter().position(|a| a.id == "bulk_thermodynamic_observation_a").unwrap();
        let heat_b = instance.actions.iter().position(|a| a.id == "bulk_thermodynamic_observation_b").unwrap();
        let shared = instance.mutual_information_shared(&[heat_a, heat_b]).unwrap();
        let factorized = instance.mutual_information_factorized_nuisance(&[heat_a, heat_b]).unwrap();
        assert!((shared - factorized).abs() > 0.01);
    }

    #[test]
    fn missing_none_of_above_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["models"][3]["none_of_above"] = json!(false);
        }).is_err());
    }

    #[test]
    fn unnormalized_nuisance_prior_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["nuisance_states"][0]["prior_probability"] = json!(0.25);
        }).is_err());
    }

    #[test]
    fn incomplete_model_or_nuisance_table_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["actions"][1]["model_likelihoods"][0]["nuisance_likelihoods"].as_array_mut().unwrap().pop();
        }).is_err());
    }

    #[test]
    fn capability_action_cannot_claim_outcome_likelihoods() {
        assert!(mutated_fixture(|fixture| {
            let rows = fixture["actions"][1]["model_likelihoods"].clone();
            fixture["actions"][0]["model_likelihoods"] = rows;
        }).is_err());
    }

    #[test]
    fn ineligible_high_information_action_never_enters_oracle() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let ids = receipt["exact_oracle"]["plan"]["ordered_action_ids"].as_array().unwrap();
        assert!(!ids.iter().any(|value| value == "unreviewed_high_field_probe"));
    }

    #[test]
    fn receipt_is_deterministic() {
        let (schema, fixture) = inputs();
        assert_eq!(
            audit_and_plan(&schema, &fixture).unwrap(),
            audit_and_plan(&schema, &fixture).unwrap()
        );
    }
}
