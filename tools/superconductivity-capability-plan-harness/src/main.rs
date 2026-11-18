use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug, Default)]
struct Matrix2 {
    m00: f64,
    m01: f64,
    m11: f64,
}

impl Matrix2 {
    fn add(self, other: Self) -> Self {
        Self {
            m00: self.m00 + other.m00,
            m01: self.m01 + other.m01,
            m11: self.m11 + other.m11,
        }
    }

    fn determinant(self) -> f64 {
        self.m00 * self.m11 - self.m01 * self.m01
    }

    fn target_variance(self, target: [f64; 2]) -> Result<f64, String> {
        let determinant = self.determinant();
        if !determinant.is_finite() || determinant <= 1.0e-12 {
            return Err("posterior precision matrix is not positive definite".to_owned());
        }
        let numerator = self.m11 * target[0] * target[0]
            - 2.0 * self.m01 * target[0] * target[1]
            + self.m00 * target[1] * target[1];
        let variance = numerator / determinant;
        if !variance.is_finite() || variance < -1.0e-10 {
            return Err("target predictive variance is invalid".to_owned());
        }
        Ok(variance.max(0.0))
    }
}

#[derive(Clone, Debug)]
struct Action {
    id: String,
    action_type: String,
    cost: u64,
    requires: Vec<String>,
    provides: Vec<String>,
    information: Matrix2,
    eligible: bool,
    exclusion_reason: String,
}

#[derive(Clone, Debug)]
struct Plan {
    ordered_indices: Vec<usize>,
    cost: u64,
    posterior_precision: Matrix2,
    prior_target_variance: f64,
    posterior_target_variance: f64,
    variance_reduction: f64,
}

#[derive(Clone, Debug)]
struct Instance {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    parameter_names: Vec<String>,
    prior_precision: Matrix2,
    target_direction: [f64; 2],
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

fn parse_matrix(value: &Value, context: &str) -> Result<Matrix2, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a 2x2 matrix"))?;
    if rows.len() != 2 {
        return Err(format!("{context} must have two rows"));
    }
    let row0 = rows[0]
        .as_array()
        .ok_or_else(|| format!("{context} row 0 must be an array"))?;
    let row1 = rows[1]
        .as_array()
        .ok_or_else(|| format!("{context} row 1 must be an array"))?;
    if row0.len() != 2 || row1.len() != 2 {
        return Err(format!("{context} must be exactly 2x2"));
    }
    let m00 = finite_number(&row0[0], context)?;
    let m01 = finite_number(&row0[1], context)?;
    let m10 = finite_number(&row1[0], context)?;
    let m11 = finite_number(&row1[1], context)?;
    if (m01 - m10).abs() > 1.0e-10 {
        return Err(format!("{context} must be symmetric"));
    }
    let matrix = Matrix2 {
        m00,
        m01: 0.5 * (m01 + m10),
        m11,
    };
    if matrix.m00 < -1.0e-12 || matrix.m11 < -1.0e-12 || matrix.determinant() < -1.0e-10 {
        return Err(format!("{context} must be positive semidefinite"));
    }
    Ok(matrix)
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
                "capability-plan schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("initial capability planner accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-CAPABILITY-PLAN-0.1" {
            return Err("unsupported capability-planning contract".to_owned());
        }
        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let parameter_names = parse_string_set(
            fixture
                .get("parameter_names")
                .ok_or_else(|| "missing parameter_names".to_owned())?,
            "parameter_names",
        )?;
        if parameter_names.len() != 2 {
            return Err("initial planner requires exactly two target parameters".to_owned());
        }
        let prior_precision = parse_matrix(
            fixture
                .get("prior_precision")
                .ok_or_else(|| "missing prior_precision".to_owned())?,
            "prior_precision",
        )?;
        if prior_precision.determinant() <= 1.0e-12 {
            return Err("prior precision matrix must be positive definite".to_owned());
        }

        let target_values = fixture
            .get("target_direction")
            .and_then(Value::as_array)
            .ok_or_else(|| "target_direction must be an array".to_owned())?;
        if target_values.len() != 2 {
            return Err("target_direction must have two entries".to_owned());
        }
        let target_direction = [
            finite_number(&target_values[0], "target_direction[0]")?,
            finite_number(&target_values[1], "target_direction[1]")?,
        ];
        if target_direction.iter().all(|value| value.abs() <= f64::EPSILON) {
            return Err("target_direction must not be the zero vector".to_owned());
        }

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
            .ok_or_else(|| "budget_units must be a positive integer".to_owned())?;
        if budget_units == 0 {
            return Err("budget_units must be positive".to_owned());
        }

        let action_values = fixture
            .get("actions")
            .and_then(Value::as_array)
            .ok_or_else(|| "actions must be an array".to_owned())?;
        if action_values.is_empty() || action_values.len() > 20 {
            return Err("initial exact oracle supports between 1 and 20 actions".to_owned());
        }
        let mut action_ids = HashSet::with_capacity(action_values.len());
        let mut actions = Vec::with_capacity(action_values.len());
        for action_value in action_values {
            let id = required_text(action_value, "action_id")?.to_owned();
            if !action_ids.insert(id.clone()) {
                return Err(format!("duplicate action_id {id}"));
            }
            let action_type = required_text(action_value, "action_type")?.to_owned();
            let cost = action_value
                .get("cost_units")
                .and_then(Value::as_u64)
                .filter(|cost| *cost > 0)
                .ok_or_else(|| format!("action {id} cost_units must be positive"))?;
            let requires = parse_string_set(
                action_value
                    .get("requires_capabilities")
                    .ok_or_else(|| format!("action {id} missing requires_capabilities"))?,
                &format!("action {id} requires_capabilities"),
            )?;
            let provides = parse_string_set(
                action_value
                    .get("provides_capabilities")
                    .ok_or_else(|| format!("action {id} missing provides_capabilities"))?,
                &format!("action {id} provides_capabilities"),
            )?;
            let information = parse_matrix(
                action_value
                    .get("information_matrix")
                    .ok_or_else(|| format!("action {id} missing information_matrix"))?,
                &format!("action {id} information_matrix"),
            )?;
            let eligible = action_value
                .get("eligible")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("action {id} eligible must be boolean"))?;
            let exclusion_reason = action_value
                .get("exclusion_reason")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("action {id} exclusion_reason must be a string"))?
                .to_owned();
            if !eligible && exclusion_reason.trim().is_empty() {
                return Err(format!(
                    "ineligible action {id} requires an explicit exclusion_reason"
                ));
            }
            if action_type == "capability"
                && (information.m00.abs() > 1.0e-12
                    || information.m01.abs() > 1.0e-12
                    || information.m11.abs() > 1.0e-12)
            {
                return Err(format!(
                    "capability action {id} must not claim direct measurement information"
                ));
            }
            actions.push(Action {
                id,
                action_type,
                cost,
                requires,
                provides,
                information,
                eligible,
                exclusion_reason,
            });
        }

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            parameter_names,
            prior_precision,
            target_direction,
            initial_capabilities,
            budget_units,
            actions,
        })
    }

    fn prior_target_variance(&self) -> Result<f64, String> {
        self.prior_precision.target_variance(self.target_direction)
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

        let mut posterior_precision = self.prior_precision;
        for index in &selected {
            posterior_precision = posterior_precision.add(self.actions[*index].information);
        }
        let prior_target_variance = self.prior_target_variance()?;
        let posterior_target_variance =
            posterior_precision.target_variance(self.target_direction)?;
        let variance_reduction = (prior_target_variance - posterior_target_variance).max(0.0);
        Ok(Some(Plan {
            ordered_indices,
            cost,
            posterior_precision,
            prior_target_variance,
            posterior_target_variance,
            variance_reduction,
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
                .filter(|index| (mask & (1_u64 << index)) != 0)
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
                    trial_plan.variance_reduction - current_plan.variance_reduction;
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
            "prior_target_variance": plan.prior_target_variance,
            "posterior_target_variance": plan.posterior_target_variance,
            "target_variance_reduction": plan.variance_reduction,
            "posterior_precision": [
                [plan.posterior_precision.m00, plan.posterior_precision.m01],
                [plan.posterior_precision.m01, plan.posterior_precision.m11]
            ]
        })
    }
}

fn is_better_plan(candidate: &Plan, incumbent: &Plan, actions: &[Action]) -> bool {
    if candidate.variance_reduction > incumbent.variance_reduction + 1.0e-12 {
        return true;
    }
    if (candidate.variance_reduction - incumbent.variance_reduction).abs() > 1.0e-12 {
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
    let regret = (oracle.variance_reduction - greedy.variance_reduction).max(0.0);
    let ratio = if oracle.variance_reduction > 1.0e-12 {
        greedy.variance_reduction / oracle.variance_reduction
    } else {
        1.0
    };

    let excluded_actions: Vec<Value> = instance
        .actions
        .iter()
        .filter(|action| !action.eligible)
        .map(|action| json!({
            "action_id": action.id,
            "exclusion_reason": action.exclusion_reason
        }))
        .collect();

    Ok(json!({
        "status": "pass",
        "contract": "SC-CAPABILITY-PLAN-AUDIT-0.1",
        "instance_id": instance.instance_id,
        "schema_sha256": instance.schema_sha256,
        "fixture_sha256": instance.fixture_sha256,
        "parameter_names": instance.parameter_names,
        "target_direction": instance.target_direction,
        "budget_units": instance.budget_units,
        "action_count": instance.actions.len(),
        "eligible_action_count": instance.actions.iter().filter(|action| action.eligible).count(),
        "excluded_actions": excluded_actions,
        "objective": "reduction in target predictive variance g^T P^-1 g under additive 2x2 Gaussian/Fisher information",
        "exact_oracle": {
            "algorithm": "exhaustive feasible subset enumeration with capability-closure ordering",
            "enumerated_subsets": enumerated_subsets,
            "plan": instance.plan_receipt(&oracle)
        },
        "greedy_baseline": {
            "algorithm": "myopic marginal target-variance reduction per cost; capability-only actions have zero immediate score",
            "plan": instance.plan_receipt(&greedy)
        },
        "comparison": {
            "oracle_variance_reduction_minus_greedy": regret,
            "greedy_fraction_of_oracle_variance_reduction": ratio,
            "capability_gating_gap_detected": regret > 1.0e-9
        },
        "claim_ceiling": "Synthetic two-parameter planning benchmark only; no physical experiment is authorized or performed and no material property is established."
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
    let receipt = audit_and_plan(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-CAPABILITY-PLAN-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing capability-plan receipt should not fail");
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
            root.join("data/schemas/superconductivity_capability_experiment_plan_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_capability_experiment_plan_fixture_v0_1.json"),
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
    fn exact_oracle_selects_capability_then_measurement_and_amplitude() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let oracle = &receipt["exact_oracle"]["plan"];
        assert_eq!(oracle["cost_units"], 5);
        assert!(
            (oracle["target_variance_reduction"].as_f64().unwrap() - 1.6).abs() < 1.0e-12
        );
        let ids: Vec<&str> = oracle["ordered_action_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert!(ids.contains(&"heat_calibration"));
        assert!(ids.contains(&"heat_scan"));
        assert!(ids.contains(&"amp_map"));
        assert!(
            ids.iter().position(|id| *id == "heat_calibration").unwrap()
                < ids.iter().position(|id| *id == "heat_scan").unwrap()
        );
    }

    #[test]
    fn myopic_policy_misses_zero_immediate_information_capability() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let greedy = &receipt["greedy_baseline"]["plan"];
        let ids: Vec<&str> = greedy["ordered_action_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert!(!ids.contains(&"heat_calibration"));
        assert!(!ids.contains(&"heat_scan"));
        assert!(receipt["comparison"]["capability_gating_gap_detected"] == true);
        assert!(
            receipt["comparison"]["oracle_variance_reduction_minus_greedy"]
                .as_f64()
                .unwrap()
                > 0.1
        );
    }

    #[test]
    fn ineligible_high_information_action_is_never_selected() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_plan(&schema, &fixture).unwrap();
        let oracle_ids = receipt["exact_oracle"]["plan"]["ordered_action_ids"]
            .as_array()
            .unwrap();
        let greedy_ids = receipt["greedy_baseline"]["plan"]["ordered_action_ids"]
            .as_array()
            .unwrap();
        assert!(!oracle_ids.iter().any(|value| value == "unreviewed_high_field"));
        assert!(!greedy_ids.iter().any(|value| value == "unreviewed_high_field"));
        assert_eq!(receipt["excluded_actions"][0]["exclusion_reason"], "Missing independent safety and facility qualification");
    }

    #[test]
    fn required_capability_must_be_initial_or_acquired() {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        fixture["initial_capabilities"] = json!([]);
        fixture["actions"][0]["requires_capabilities"] = json!(["capability_never_provided"]);
        fixture["actions"][0]["provides_capabilities"] = json!([]);
        let receipt = audit_and_plan(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        let ids = receipt["exact_oracle"]["plan"]["ordered_action_ids"]
            .as_array()
            .unwrap();
        assert!(!ids.iter().any(|value| value == "amp_map"));
    }

    #[test]
    fn asymmetric_information_matrix_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["actions"][0]["information_matrix"] = json!([[1.0, 0.1], [0.2, 1.0]]);
        })
        .is_err());
    }

    #[test]
    fn non_positive_definite_prior_is_rejected() {
        assert!(mutated_fixture(|fixture| {
            fixture["prior_precision"] = json!([[1.0, 2.0], [2.0, 1.0]]);
        })
        .is_err());
    }

    #[test]
    fn capability_action_cannot_claim_direct_information() {
        assert!(mutated_fixture(|fixture| {
            fixture["actions"][1]["information_matrix"] = json!([[1.0, 0.0], [0.0, 0.0]]);
        })
        .is_err());
    }

    #[test]
    fn plan_is_deterministic_under_identical_input() {
        let (schema, fixture) = inputs();
        let first = audit_and_plan(&schema, &fixture).unwrap();
        let second = audit_and_plan(&schema, &fixture).unwrap();
        assert_eq!(first, second);
    }
}
