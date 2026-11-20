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
struct Action {
    id: String,
    action_type: String,
    cost: u64,
    requires: Vec<String>,
    provides: Vec<String>,
    eligible: bool,
    exclusion_reason: String,
    // One binary outcome distribution per model, in model declaration order.
    likelihoods: Vec<[f64; 2]>,
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
                "model-discrimination schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("model-discrimination auditor accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-MODEL-DISCRIMINATION-0.1" {
            return Err("unsupported model-discrimination contract".to_owned());
        }

        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let model_values = fixture
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| "models must be an array".to_owned())?;
        let mut models = Vec::with_capacity(model_values.len());
        let mut model_index = HashMap::with_capacity(model_values.len());
        let mut prior_sum = 0.0;
        let mut none_count = 0;
        for (index, value) in model_values.iter().enumerate() {
            let id = required_text(value, "model_id")?.to_owned();
            if model_index.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate model_id {id}"));
            }
            let prior = finite_number(
                value
                    .get("prior_probability")
                    .ok_or_else(|| format!("model {id} missing prior_probability"))?,
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
            prior_sum += prior;
            models.push(Model {
                id,
                prior,
                none_of_above,
            });
        }
        if none_count != 1 {
            return Err("exactly one explicit none_of_above model is required".to_owned());
        }
        if (prior_sum - 1.0).abs() > 1.0e-9 {
            return Err(format!("model prior probabilities sum to {prior_sum}, not 1"));
        }

        let outcome_values = parse_string_set(
            fixture
                .get("outcome_labels")
                .ok_or_else(|| "missing outcome_labels".to_owned())?,
            "outcome_labels",
        )?;
        if outcome_values.len() != 2 {
            return Err("the initial contract requires exactly two outcome labels".to_owned());
        }
        let outcome_labels = [
            outcome_values[0].clone(),
            outcome_values[1].clone(),
        ];
        let initial_capabilities = parse_string_set(
            fixture
                .get("initial_capabilities")
                .ok_or_else(|| "missing initial_capabilities".to_owned())?,
            "initial_capabilities",
        )?
        .into_iter()
        .collect();

        let budget_units = fixture
            .get("budget_units")
            .and_then(Value::as_u64)
            .filter(|budget| *budget > 0)
            .ok_or_else(|| "budget_units must be a positive integer".to_owned())?;
        let action_values = fixture
            .get("actions")
            .and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_values.is_empty() || action_values.len() > 12 {
            return Err("action count must be between 1 and 12".to_owned());
        }

        let mut actions = Vec::with_capacity(action_values.len());
        let mut action_ids = HashSet::with_capacity(action_values.len());
        for value in action_values {
            let id = required_text(value, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) {
                return Err(format!("duplicate action_id {id}"));
            }
            let action_type = required_text(value, "action_type")?.to_owned();
            let cost = value
                .get("cost_units")
                .and_then(Value::as_u64)
                .filter(|cost| *cost > 0)
                .ok_or_else(|| format!("action {id} cost_units must be a positive integer"))?;
            let requires = parse_string_set(
                value
                    .get("requires_capabilities")
                    .ok_or_else(|| format!("action {id} missing requires_capabilities"))?,
                &format!("action {id} requires_capabilities"),
            )?;
            let provides = parse_string_set(
                value
                    .get("provides_capabilities")
                    .ok_or_else(|| format!("action {id} missing provides_capabilities"))?,
                &format!("action {id} provides_capabilities"),
            )?;
            let eligible = value
                .get("eligible")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("action {id} eligible must be boolean"))?;
            let exclusion_reason = value
                .get("exclusion_reason")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("action {id} exclusion_reason must be a string"))?
                .to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!(
                    "ineligible action {id} requires an explicit exclusion_reason"
                ));
            }

            let likelihood_values = value
                .get("model_likelihoods")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("action {id} model_likelihoods must be an array"))?;
            let mut likelihood_by_model: HashMap<String, [f64; 2]> =
                HashMap::with_capacity(likelihood_values.len());
            for row in likelihood_values {
                let model_id = required_text(row, "model_id")?.to_owned();
                if !model_index.contains_key(&model_id) {
                    return Err(format!("action {id} likelihood references unknown model {model_id}"));
                }
                let probabilities = row
                    .get("outcome_probabilities")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("action {id} likelihood for {model_id} needs two probabilities"))?;
                if probabilities.len() != 2 {
                    return Err(format!("action {id} likelihood for {model_id} must have two outcomes"));
                }
                let first = finite_number(&probabilities[0], "outcome probability")?;
                let second = finite_number(&probabilities[1], "outcome probability")?;
                if !(0.0..=1.0).contains(&first)
                    || !(0.0..=1.0).contains(&second)
                    || (first + second - 1.0).abs() > 1.0e-9
                {
                    return Err(format!(
                        "action {id} likelihood for {model_id} must be a normalized probability pair"
                    ));
                }
                if likelihood_by_model
                    .insert(model_id.clone(), [first, second])
                    .is_some()
                {
                    return Err(format!("action {id} has duplicate likelihood for model {model_id}"));
                }
            }

            let likelihoods = if action_type == "capability" {
                if !likelihood_by_model.is_empty() {
                    return Err(format!("capability action {id} cannot directly produce an observation"));
                }
                if provides.is_empty() {
                    return Err(format!("capability action {id} must provide at least one capability"));
                }
                Vec::new()
            } else {
                if likelihood_by_model.len() != models.len() {
                    return Err(format!(
                        "action {id} must define likelihoods for every model, including none_of_above"
                    ));
                }
                models
                    .iter()
                    .map(|model| {
                        likelihood_by_model
                            .get(&model.id)
                            .copied()
                            .ok_or_else(|| format!("action {id} is missing likelihood for {}", model.id))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };

            actions.push(Action {
                id,
                action_type,
                cost,
                requires,
                provides,
                eligible,
                exclusion_reason,
                likelihoods,
            });
        }

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            models,
            outcome_labels,
            initial_capabilities,
            budget_units,
            actions,
        })
    }

    fn prior_entropy_nats(&self) -> f64 {
        entropy(self.models.iter().map(|model| model.prior))
    }

    fn mutual_information_nats(&self, selected_indices: &[usize]) -> Result<f64, String> {
        let measured: Vec<usize> = selected_indices
            .iter()
            .copied()
            .filter(|index| !self.actions[*index].likelihoods.is_empty())
            .collect();
        if measured.is_empty() {
            return Ok(0.0);
        }
        if measured.len() > 12 {
            return Err("too many selected observations for exact outcome enumeration".to_owned());
        }

        // Exact I(M;Y_S) under the fixture's explicit conditional-independence assumption.
        let outcome_count = 1_u64 << measured.len();
        let mut expected_posterior_entropy = 0.0;
        for outcome_mask in 0..outcome_count {
            let mut unnormalized = vec![0.0; self.models.len()];
            for (model_index, model) in self.models.iter().enumerate() {
                let mut joint = model.prior;
                for (outcome_index, action_index) in measured.iter().enumerate() {
                    let outcome = ((outcome_mask >> outcome_index) & 1) as usize;
                    joint *= self.actions[*action_index].likelihoods[model_index][outcome];
                }
                unnormalized[model_index] = joint;
            }
            let outcome_probability: f64 = unnormalized.iter().sum();
            if outcome_probability <= 0.0 {
                continue;
            }
            let posterior_entropy = entropy(
                unnormalized
                    .iter()
                    .map(|probability| probability / outcome_probability),
            );
            expected_posterior_entropy += outcome_probability * posterior_entropy;
        }
        let information = self.prior_entropy_nats() - expected_posterior_entropy;
        if !information.is_finite() || information < -1.0e-9 {
            return Err("computed model information gain is invalid".to_owned());
        }
        Ok(information.clamp(0.0, self.prior_entropy_nats()))
    }

    fn plan_from_indices(&self, indices: &[usize]) -> Result<Option<Plan>, String> {
        let mut selected = indices.to_vec();
        selected.sort_unstable();
        selected.dedup();
        if selected.len() != indices.len() {
            return Ok(None);
        }

        let mut cost = 0_u64;
        for index in &selected {
            let Some(action) = self.actions.get(*index) else {
                return Ok(None);
            };
            if !action.eligible {
                return Ok(None);
            }
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
                self.actions[*index]
                    .requires
                    .iter()
                    .all(|capability| available.contains(capability))
            });
            let Some(position) = position else {
                return Ok(None);
            };
            let index = remaining.remove(position);
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
            ordered_indices.push(index);
        }

        let information_gain_nats = self.mutual_information_nats(&selected)?;
        Ok(Some(Plan {
            ordered_indices,
            cost,
            information_gain_nats,
        }))
    }

    fn exact_oracle(&self) -> Result<(Plan, u64), String> {
        let subset_count = 1_u64
            .checked_shl(self.actions.len() as u32)
            .ok_or_else(|| "action subset count overflowed".to_owned())?;
        let mut best = self
            .plan_from_indices(&[])?
            .ok_or_else(|| "empty plan should always be feasible".to_owned())?;
        for mask in 1..subset_count {
            let indices: Vec<usize> = (0..self.actions.len())
                .filter(|index| (mask & (1_u64 << *index)) != 0)
                .collect();
            let Some(candidate) = self.plan_from_indices(&indices)? else {
                continue;
            };
            if is_better_plan(&candidate, &best, &self.actions) {
                best = candidate;
            }
        }
        Ok((best, subset_count))
    }

    fn greedy_information_per_cost(&self) -> Result<Plan, String> {
        let mut chosen: Vec<usize> = Vec::new();
        let mut selected_set = HashSet::new();
        let mut available = self.initial_capabilities.clone();
        let mut cost = 0_u64;
        loop {
            let current_plan = self
                .plan_from_indices(&chosen)?
                .ok_or_else(|| "greedy planner lost feasibility".to_owned())?;
            let mut best_candidate: Option<(usize, f64, Plan)> = None;
            for index in 0..self.actions.len() {
                if selected_set.contains(&index) {
                    continue;
                }
                let action = &self.actions[index];
                if !action.eligible
                    || action.cost > self.budget_units.saturating_sub(cost)
                    || !action
                        .requires
                        .iter()
                        .all(|capability| available.contains(capability))
                {
                    continue;
                }
                let mut trial = chosen.clone();
                trial.push(index);
                let Some(trial_plan) = self.plan_from_indices(&trial)? else {
                    continue;
                };
                let marginal_gain =
                    trial_plan.information_gain_nats - current_plan.information_gain_nats;
                let ratio = marginal_gain / action.cost as f64;
                if marginal_gain <= 1.0e-12 {
                    continue;
                }
                let replace = match &best_candidate {
                    None => true,
                    Some((best_index, best_ratio, _)) => {
                        ratio > *best_ratio + 1.0e-12
                            || ((ratio - *best_ratio).abs() <= 1.0e-12
                                && action.id < self.actions[*best_index].id)
                    }
                };
                if replace {
                    best_candidate = Some((index, ratio, trial_plan));
                }
            }
            let Some((index, _, plan)) = best_candidate else {
                return Ok(current_plan);
            };
            selected_set.insert(index);
            chosen.push(index);
            cost = plan.cost;
            for capability in &self.actions[index].provides {
                available.insert(capability.clone());
            }
        }
    }

    fn plan_receipt(&self, plan: &Plan) -> Value {
        let action_ids: Vec<&str> = plan
            .ordered_indices
            .iter()
            .map(|index| self.actions[*index].id.as_str())
            .collect();
        let action_types: Vec<&str> = plan
            .ordered_indices
            .iter()
            .map(|index| self.actions[*index].action_type.as_str())
            .collect();
        let selected_capabilities: Vec<String> = plan
            .ordered_indices
            .iter()
            .flat_map(|index| self.actions[*index].provides.iter().cloned())
            .collect();
        json!({
            "ordered_action_ids": action_ids,
            "ordered_action_types": action_types,
            "selected_capabilities": selected_capabilities,
            "cost_units": plan.cost,
            "remaining_budget_units": self.budget_units - plan.cost,
            "expected_model_information_gain_nats": plan.information_gain_nats
        })
    }
}

fn is_better_plan(candidate: &Plan, incumbent: &Plan, actions: &[Action]) -> bool {
    if candidate.information_gain_nats > incumbent.information_gain_nats + 1.0e-12 {
        return true;
    }
    if (candidate.information_gain_nats - incumbent.information_gain_nats).abs() > 1.0e-12 {
        return false;
    }
    if candidate.cost != incumbent.cost {
        return candidate.cost < incumbent.cost;
    }
    let candidate_ids: Vec<&str> = candidate
        .ordered_indices
        .iter()
        .map(|index| actions[*index].id.as_str())
        .collect();
    let incumbent_ids: Vec<&str> = incumbent
        .ordered_indices
        .iter()
        .map(|index| actions[*index].id.as_str())
        .collect();
    candidate_ids < incumbent_ids
}

fn audit_and_plan(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance = Instance::from_bytes(schema_bytes, fixture_bytes)?;
    let (oracle, enumerated_subsets) = instance.exact_oracle()?;
    let greedy = instance.greedy_information_per_cost()?;
    let regret = (oracle.information_gain_nats - greedy.information_gain_nats).max(0.0);
    let greedy_fraction = if oracle.information_gain_nats > 1.0e-12 {
        greedy.information_gain_nats / oracle.information_gain_nats
    } else {
        1.0
    };
    let excluded_actions: Vec<Value> = instance
        .actions
        .iter()
        .filter(|action| !action.eligible)
        .map(|action| {
            json!({
                "action_id": action.id,
                "exclusion_reason": action.exclusion_reason
            })
        })
        .collect();
    let model_priors: Vec<Value> = instance
        .models
        .iter()
        .map(|model| {
            json!({
                "model_id": model.id,
                "prior_probability": model.prior,
                "none_of_above": model.none_of_above
            })
        })
        .collect();

    Ok(json!({
        "status": "synthetic_contract_pass",
        "contract": "SC-MODEL-DISCRIMINATION-AUDIT-0.1",
        "instance_id": instance.instance_id,
        "schema_sha256": instance.schema_sha256,
        "fixture_sha256": instance.fixture_sha256,
        "models": model_priors,
        "outcome_labels": instance.outcome_labels,
        "budget_units": instance.budget_units,
        "action_count": instance.actions.len(),
        "eligible_action_count": instance.actions.iter().filter(|action| action.eligible).count(),
        "excluded_actions": excluded_actions,
        "objective": "expected mutual information I(M;Y_selected|D) in nats, with exact enumeration of binary outcomes",
        "likelihood_assumption": "selected measurement outcomes are conditionally independent given the model identity; this is an explicit synthetic simplification, not a general physical assumption",
        "exact_oracle": {
            "algorithm": "exhaustive feasible subset enumeration with capability-closure ordering and exact joint-outcome marginalization",
            "enumerated_subsets": enumerated_subsets,
            "prior_model_entropy_nats": instance.prior_entropy_nats(),
            "plan": instance.plan_receipt(&oracle)
        },
        "myopic_baseline": {
            "algorithm": "greedy marginal expected model information gain per cost; capability-only actions have zero immediate score",
            "plan": instance.plan_receipt(&greedy)
        },
        "comparison": {
            "oracle_information_gain_minus_myopic_nats": regret,
            "myopic_fraction_of_oracle_information_gain": greedy_fraction,
            "capability_discrimination_gap_detected": regret > 1.0e-9
        },
        "claim_ceiling": "Synthetic model-discrimination planning contract only; likelihoods are illustrative and do not describe a real specimen, authorize an experiment, or establish a pairing mechanism."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <planning-instance.json> [--out <receipt.json>]"
    );
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
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
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
    let receipt =
        audit_and_plan(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
            eprintln!("SC-MODEL-DISCRIMINATION-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing model-discrimination receipt should not fail");
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
        let schema = fs::read(
            root.join("data/schemas/superconductivity_model_discrimination_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_model_discrimination_fixture_v0_1.json"),
        )
        .unwrap();
        (schema, fixture)
    }

    fn mutated_fixture(mut mutation: impl FnMut(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        audit_and_plan(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn exact_oracle_spends_budget_on_discriminating_measurements_and_capability() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let plan = &receipt["exact_oracle"]["plan"];
        assert_eq!(plan["cost_units"], 6);
        assert!(
            (plan["expected_model_information_gain_nats"]
                .as_f64()
                .unwrap()
                - 0.5569594230810538)
                .abs()
                < 1.0e-10
        );
        let ids: Vec<&str> = plan["ordered_action_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        for expected in [
            "qualify_thermodynamic_channel",
            "bulk_thermodynamic_crosscheck",
            "field_angle_hc2",
            "stiffness_tensor",
        ] {
            assert!(ids.contains(&expected), "missing selected action {expected}");
        }
        assert!(
            ids.iter()
                .position(|id| *id == "qualify_thermodynamic_channel")
                .unwrap()
                < ids.iter()
                    .position(|id| *id == "bulk_thermodynamic_crosscheck")
                    .unwrap()
        );
        assert!(!ids.contains(&"cheap_resistivity_repeat"));
        assert!(!ids.contains(&"unreviewed_high_field_probe"));
    }

    #[test]
    fn myopic_policy_misses_zero_information_capability_and_followup() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let plan = &receipt["myopic_baseline"]["plan"];
        let ids: Vec<&str> = plan["ordered_action_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert!(!ids.contains(&"qualify_thermodynamic_channel"));
        assert!(!ids.contains(&"bulk_thermodynamic_crosscheck"));
        assert!(
            receipt["comparison"]["oracle_information_gain_minus_myopic_nats"]
                .as_f64()
                .unwrap()
                > 0.15
        );
    }

    #[test]
    fn ineligible_action_is_excluded_even_when_its_likelihood_is_sharp() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        assert!(
            !receipt["exact_oracle"]["plan"]["ordered_action_ids"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "unreviewed_high_field_probe")
        );
        assert_eq!(
            receipt["excluded_actions"][0]["exclusion_reason"],
            "Independent facility and safety qualification is absent"
        );
    }

    #[test]
    fn explicit_none_of_above_is_mandatory() {
        assert!(
            mutated_fixture(|fixture| {
                fixture["models"][3]["none_of_above"] = json!(false);
            })
            .is_err()
        );
    }

    #[test]
    fn priors_must_be_normalized() {
        assert!(
            mutated_fixture(|fixture| {
                fixture["models"][0]["prior_probability"] = json!(0.45);
            })
            .is_err()
        );
    }

    #[test]
    fn every_measurement_must_score_every_model() {
        assert!(
            mutated_fixture(|fixture| {
                fixture["actions"][2]["model_likelihoods"].as_array_mut().unwrap().pop();
            })
            .is_err()
        );
    }

    #[test]
    fn likelihood_pairs_must_sum_to_one() {
        assert!(
            mutated_fixture(|fixture| {
                fixture["actions"][2]["model_likelihoods"][0]["outcome_probabilities"] =
                    json!([0.8, 0.4]);
            })
            .is_err()
        );
    }

    #[test]
    fn capability_action_cannot_smuggle_in_an_observation_model() {
        assert!(
            mutated_fixture(|fixture| {
                let measurement_likelihoods = fixture["actions"][2]["model_likelihoods"].clone();
                fixture["actions"][0]["model_likelihoods"] = measurement_likelihoods;
            })
            .is_err()
        );
    }

    #[test]
    fn identical_outcome_likelihoods_have_zero_model_information() {
        let (schema, fixture) = inputs();
        let instance = Instance::from_bytes(&schema, &fixture).unwrap();
        let cheap_index = instance
            .actions
            .iter()
            .position(|action| action.id == "cheap_resistivity_repeat")
            .unwrap();
        assert!(instance.mutual_information_nats(&[cheap_index]).unwrap() < 1.0e-12);
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
