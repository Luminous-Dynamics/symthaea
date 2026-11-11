use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const BENCHMARK_ID: &str = "IG-CAUSAL-PRIMITIVES-001";
const SCHEMA_VERSION: &str = "ig-causal-primitives-fixtures-v0.1";
const EXPECTED_TASKS: usize = 40;
const PRIMITIVES: [&str; 4] = [
    "CausalPrior",
    "ForwardStateTracing",
    "BeliefUpdating",
    "AgencySensitivity",
];
const DOMAINS: [&str; 3] = [
    "mechanical_infrastructure",
    "institutional_resource",
    "abstract_systems",
];

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|text| !text.trim().is_empty())
}

fn unsigned_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key)?.as_u64()
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn checked_mul(a: u128, b: u128) -> Result<u128, String> {
    a.checked_mul(b).ok_or_else(|| "Bayesian oracle arithmetic overflow".to_string())
}

fn validate_schema(schema: &Value) -> Result<(), String> {
    if string_field(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema")
        || string_field(schema, "$id")
            != Some("https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-causal-primitives-001/CAUSAL_PRIMITIVES_FIXTURES_V0_1.schema.json")
        || string_field(schema, "type") != Some("object")
        || schema["additionalProperties"] != false
        || schema["properties"]["tasks"]["type"] != "array"
    {
        return Err("schema identity or bounded top-level contract mismatch".into());
    }
    Ok(())
}

fn validate_equations(equations: &Value) -> Result<(BTreeSet<String>, Vec<(String, String)>), String> {
    let rows = equations
        .as_array()
        .ok_or_else(|| "equations must be an array".to_string())?;
    if rows.is_empty() {
        return Err("equations cannot be empty".into());
    }

    let mut nodes = BTreeSet::new();
    let mut edges = Vec::new();

    for (index, equation) in rows.iter().enumerate() {
        let node = string_field(equation, "node")
            .ok_or_else(|| format!("equations[{index}] has no node identity"))?
            .to_string();
        if !nodes.insert(node.clone()) {
            return Err(format!("duplicate equation node {node}"));
        }
        let op = string_field(equation, "op")
            .ok_or_else(|| format!("equations[{index}] has no operation"))?;
        let inputs = equation["inputs"]
            .as_array()
            .ok_or_else(|| format!("equations[{index}].inputs must be an array"))?;
        let input_ids = inputs
            .iter()
            .map(|input| {
                input
                    .as_str()
                    .filter(|text| !text.is_empty())
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| format!("equations[{index}] has an invalid input identity"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let cardinality_ok = match op {
            "Const" => input_ids.is_empty() && equation["value"].is_boolean(),
            "Copy" | "Not" => input_ids.len() == 1,
            "And" | "Or" | "Xor" => input_ids.len() >= 2,
            _ => false,
        };
        if !cardinality_ok {
            return Err(format!("equations[{index}] has invalid operation/input arity"));
        }

        let mut unique_inputs = BTreeSet::new();
        for parent in input_ids {
            if !unique_inputs.insert(parent.clone()) {
                return Err(format!("equations[{index}] duplicates input {parent}"));
            }
            if !nodes.contains(&parent) {
                return Err(format!(
                    "equations[{index}] references {parent} before it is defined; order must be topological"
                ));
            }
            edges.push((parent, node.clone()));
        }
    }

    Ok((nodes, edges))
}

fn parse_edge_set(value: &Value) -> Result<BTreeSet<(String, String)>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "expected_edges must be an array".to_string())?;
    let mut edges = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        let pair = row
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("expected_edges[{index}] must contain two node IDs"))?;
        let source = pair[0]
            .as_str()
            .filter(|text| !text.is_empty())
            .ok_or_else(|| format!("expected_edges[{index}] has an invalid source"))?;
        let target = pair[1]
            .as_str()
            .filter(|text| !text.is_empty())
            .ok_or_else(|| format!("expected_edges[{index}] has an invalid target"))?;
        if source == target || !edges.insert((source.to_string(), target.to_string())) {
            return Err(format!("expected_edges[{index}] is a self-edge or duplicate"));
        }
    }
    Ok(edges)
}

fn evaluate_equations(
    equations: &Value,
    intervention: &Value,
) -> Result<BTreeMap<String, bool>, String> {
    let (nodes, _) = validate_equations(equations)?;
    let target = string_field(intervention, "target")
        .ok_or_else(|| "intervention target is missing".to_string())?;
    let intervention_value = intervention["value"]
        .as_bool()
        .ok_or_else(|| "intervention value must be boolean".to_string())?;
    if !nodes.contains(target) {
        return Err(format!("intervention target {target} is not in the SCM"));
    }

    let mut state = BTreeMap::<String, bool>::new();
    for equation in equations.as_array().expect("validated array") {
        let node = string_field(equation, "node").expect("validated node").to_string();
        if node == target {
            state.insert(node, intervention_value);
            continue;
        }
        let op = string_field(equation, "op").expect("validated op");
        let input_values = equation["inputs"]
            .as_array()
            .expect("validated inputs")
            .iter()
            .map(|input| {
                let input = input.as_str().expect("validated input");
                state
                    .get(input)
                    .copied()
                    .ok_or_else(|| format!("input {input} is unavailable during evaluation"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let result = match op {
            "Const" => equation["value"]
                .as_bool()
                .ok_or_else(|| "constant equation requires a boolean value".to_string())?,
            "Copy" => input_values[0],
            "Not" => !input_values[0],
            "And" => input_values.iter().all(|value| *value),
            "Or" => input_values.iter().any(|value| *value),
            "Xor" => input_values.iter().fold(false, |acc, value| acc ^ value),
            _ => return Err(format!("unsupported operation {op}")),
        };
        state.insert(node, result);
    }
    Ok(state)
}

fn parse_state(value: &Value) -> Result<BTreeMap<String, bool>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "expected_state must be an object".to_string())?;
    object
        .iter()
        .map(|(key, value)| {
            value
                .as_bool()
                .map(|boolean| (key.clone(), boolean))
                .ok_or_else(|| format!("expected_state.{key} must be boolean"))
        })
        .collect()
}

fn validate_belief_update(update: &Value) -> Result<(), String> {
    let prior_n = unsigned_field(&update["prior"], "numerator")
        .ok_or_else(|| "prior numerator is missing".to_string())? as u128;
    let prior_d = unsigned_field(&update["prior"], "denominator")
        .filter(|value| *value > 0)
        .ok_or_else(|| "prior denominator must be positive".to_string())? as u128;
    let l1_n = unsigned_field(&update["evidence_likelihood"], "numerator")
        .ok_or_else(|| "evidence likelihood numerator is missing".to_string())? as u128;
    let l1_d = unsigned_field(&update["evidence_likelihood"], "denominator")
        .filter(|value| *value > 0)
        .ok_or_else(|| "evidence likelihood denominator must be positive".to_string())? as u128;
    let l0_n = unsigned_field(&update["null_likelihood"], "numerator")
        .ok_or_else(|| "null likelihood numerator is missing".to_string())? as u128;
    let l0_d = unsigned_field(&update["null_likelihood"], "denominator")
        .filter(|value| *value > 0)
        .ok_or_else(|| "null likelihood denominator must be positive".to_string())? as u128;

    if prior_n > prior_d || l1_n > l1_d || l0_n > l0_d {
        return Err("prior and likelihood probabilities must lie in [0,1]".into());
    }

    let h1 = checked_mul(checked_mul(prior_n, l1_n)?, l0_d)?;
    let h0 = checked_mul(checked_mul(prior_d - prior_n, l0_n)?, l1_d)?;
    let denominator = h1
        .checked_add(h0)
        .filter(|value| *value > 0)
        .ok_or_else(|| "Bayesian update has zero total evidence probability".to_string())?;
    let divisor = gcd(h1, denominator);
    let expected_n = h1 / divisor;
    let expected_d = denominator / divisor;

    let actual_n = unsigned_field(&update["expected_posterior"], "numerator")
        .ok_or_else(|| "expected posterior numerator is missing".to_string())? as u128;
    let actual_d = unsigned_field(&update["expected_posterior"], "denominator")
        .filter(|value| *value > 0)
        .ok_or_else(|| "expected posterior denominator must be positive".to_string())? as u128;

    if actual_n != expected_n || actual_d != expected_d {
        return Err(format!(
            "Bayes oracle mismatch: expected {expected_n}/{expected_d}, got {actual_n}/{actual_d}"
        ));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct Report {
    tasks: usize,
    causal_prior: usize,
    forward_state_tracing: usize,
    belief_updating: usize,
    agency_sensitivity: usize,
    domains: usize,
}

fn validate_manifest(manifest: &Value) -> Result<Report, Vec<String>> {
    let mut errors = Vec::new();
    if string_field(manifest, "schema_version") != Some(SCHEMA_VERSION)
        || string_field(manifest, "benchmark_id") != Some(BENCHMARK_ID)
        || string_field(manifest, "status") != Some("synthetic_known_answer_unexecuted")
        || manifest["fixture_count"].as_u64() != Some(EXPECTED_TASKS as u64)
    {
        errors.push("manifest identity, status or fixture count mismatch".to_string());
    }

    let Some(tasks) = manifest["tasks"].as_array() else {
        return Err(vec!["tasks must be an array".into()]);
    };
    if tasks.len() != EXPECTED_TASKS {
        errors.push(format!("expected {EXPECTED_TASKS} tasks, got {}", tasks.len()));
    }

    let mut task_ids = BTreeSet::new();
    let mut primitive_counts = BTreeMap::<String, usize>::new();
    let mut domains = BTreeSet::<String>::new();
    let mut agency_pairs = BTreeMap::<String, Vec<&Value>>::new();

    for (index, task) in tasks.iter().enumerate() {
        let context = format!("tasks[{index}]");
        let task_id = string_field(task, "task_id").unwrap_or_default();
        if task_id.is_empty() || !task_ids.insert(task_id.to_string()) {
            errors.push(format!("{context}: task ID is empty or duplicated"));
        }
        let primitive = string_field(task, "primitive").unwrap_or_default();
        if !PRIMITIVES.contains(&primitive) {
            errors.push(format!("{context}: unknown primitive {primitive}"));
            continue;
        }
        *primitive_counts.entry(primitive.to_string()).or_default() += 1;

        let domain = string_field(task, "domain").unwrap_or_default();
        if !DOMAINS.contains(&domain) {
            errors.push(format!("{context}: unknown domain {domain}"));
        } else {
            domains.insert(domain.to_string());
        }

        let result = match primitive {
            "CausalPrior" => {
                let payload = &task["causal_prior"];
                let (_, derived_edges) = match validate_equations(&payload["equations"]) {
                    Ok(result) => result,
                    Err(error) => {
                        errors.push(format!("{context}: {error}"));
                        continue;
                    }
                };
                let expected = match parse_edge_set(&payload["expected_edges"]) {
                    Ok(result) => result,
                    Err(error) => {
                        errors.push(format!("{context}: {error}"));
                        continue;
                    }
                };
                if derived_edges.iter().cloned().collect::<BTreeSet<_>>() != expected {
                    Err("causal-prior edge oracle does not match structural equations".to_string())
                } else if payload["description"].as_str().is_none_or(|text| text.trim().is_empty()) {
                    Err("causal-prior description is missing".to_string())
                } else {
                    Ok(())
                }
            }
            "ForwardStateTracing" => {
                let payload = &task["forward_state_trace"];
                match evaluate_equations(&payload["equations"], &payload["intervention"]) {
                    Ok(derived) => match parse_state(&payload["expected_state"]) {
                        Ok(expected) if derived == expected => Ok(()),
                        Ok(_) => Err("forward-state intervention oracle mismatch".to_string()),
                        Err(error) => Err(error),
                    },
                    Err(error) => Err(error),
                }
            }
            "BeliefUpdating" => validate_belief_update(&task["belief_update"]),
            "AgencySensitivity" => {
                let payload = &task["agency_sensitivity"];
                let pair_id = string_field(payload, "pair_id").unwrap_or_default();
                if pair_id.is_empty()
                    || string_field(payload, "scm_ref").is_none()
                    || string_field(payload, "intervention_ref").is_none()
                    || string_field(payload, "agency_frame").is_none()
                    || string_field(payload, "expected_effect").is_none()
                    || string_field(payload, "narrative").is_none()
                {
                    Err("agency-sensitivity pair is missing a required identity or outcome".to_string())
                } else {
                    agency_pairs.entry(pair_id.to_string()).or_default().push(task);
                    Ok(())
                }
            }
            _ => unreachable!(),
        };
        if let Err(error) = result {
            errors.push(format!("{context}: {error}"));
        }
    }

    for primitive in PRIMITIVES {
        if primitive_counts.get(primitive).copied().unwrap_or_default() != 10 {
            errors.push(format!("primitive {primitive} must contain exactly ten tasks"));
        }
    }
    if domains.len() != DOMAINS.len() {
        errors.push("all three declared domains must be represented".into());
    }

    if agency_pairs.len() != 5 {
        errors.push(format!("expected five agency-framing pairs, got {}", agency_pairs.len()));
    }
    for (pair_id, members) in &agency_pairs {
        if members.len() != 2 {
            errors.push(format!("agency pair {pair_id} must have exactly two members"));
            continue;
        }
        let left = &members[0]["agency_sensitivity"];
        let right = &members[1]["agency_sensitivity"];
        if string_field(left, "scm_ref") != string_field(right, "scm_ref")
            || string_field(left, "intervention_ref") != string_field(right, "intervention_ref")
            || string_field(left, "expected_effect") != string_field(right, "expected_effect")
            || string_field(left, "agency_frame") == string_field(right, "agency_frame")
        {
            errors.push(format!(
                "agency pair {pair_id} must change framing only while holding SCM, intervention and expected effect constant"
            ));
        }
    }

    if errors.is_empty() {
        Ok(Report {
            tasks: tasks.len(),
            causal_prior: primitive_counts["CausalPrior"],
            forward_state_tracing: primitive_counts["ForwardStateTracing"],
            belief_updating: primitive_counts["BeliefUpdating"],
            agency_sensitivity: primitive_counts["AgencySensitivity"],
            domains: domains.len(),
        })
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let source = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&source).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ig-causal-primitives-qualifier <fixtures.json> <fixtures.schema.json>");
        return ExitCode::from(2);
    }
    let manifest = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let schema = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = validate_schema(&schema) {
        eprintln!("SCHEMA QUALIFICATION FAILED: {error}");
        return ExitCode::FAILURE;
    }
    match validate_manifest(&manifest) {
        Ok(report) => {
            println!(
                "QUALIFIED SYNTHETIC SCM FIXTURES: tasks={} causal_prior={} forward_state_tracing={} belief_updating={} agency_sensitivity={} domains={}",
                report.tasks,
                report.causal_prior,
                report.forward_state_tracing,
                report.belief_updating,
                report.agency_sensitivity,
                report.domains
            );
            println!("NOT EXECUTED: no model or human causal-reasoning trial is implied.");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("CAUSAL-PRIMITIVES QUALIFICATION FAILED ({} findings):", errors.len());
            for error in errors {
                eprintln!(" - {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../../CAUSAL_PRIMITIVES_FIXTURES_V0_1.json");

    fn parsed() -> Value {
        serde_json::from_str(FIXTURES).expect("fixture JSON must parse")
    }

    #[test]
    fn all_forty_known_answer_tasks_validate() {
        let report = validate_manifest(&parsed()).expect("synthetic corpus should validate");
        assert_eq!(report.tasks, 40);
        assert_eq!(report.causal_prior, 10);
        assert_eq!(report.forward_state_tracing, 10);
        assert_eq!(report.belief_updating, 10);
        assert_eq!(report.agency_sensitivity, 10);
        assert_eq!(report.domains, 3);
    }

    #[test]
    fn missing_task_cannot_shrink_the_declared_corpus() {
        let mut value = parsed();
        value["tasks"].as_array_mut().unwrap().pop();
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn duplicate_task_identity_is_rejected() {
        let mut value = parsed();
        value["tasks"][1]["task_id"] = value["tasks"][0]["task_id"].clone();
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn causal_prior_oracle_must_match_structural_equations() {
        let mut value = parsed();
        value["tasks"][0]["causal_prior"]["expected_edges"][0][0] =
            Value::String("UNKNOWN".into());
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn forward_state_oracle_must_apply_do_intervention() {
        let mut value = parsed();
        let previous = value["tasks"][10]["forward_state_trace"]["expected_state"]["X1"]
            .as_bool()
            .unwrap();
        value["tasks"][10]["forward_state_trace"]["expected_state"]["X1"] =
            Value::Bool(!previous);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn posterior_oracle_must_use_exact_bayes_arithmetic() {
        let mut value = parsed();
        value["tasks"][20]["belief_update"]["expected_posterior"]["numerator"] = Value::from(1);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn agency_pair_rejects_an_effect_change_under_framing_only() {
        let mut value = parsed();
        value["tasks"][31]["agency_sensitivity"]["expected_effect"] =
            Value::String("different_effect".into());
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn unknown_equation_parent_is_rejected() {
        let mut value = parsed();
        value["tasks"][0]["causal_prior"]["equations"][1]["inputs"][0] =
            Value::String("MISSING".into());
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn missing_domain_cannot_hide_behind_fixture_count() {
        let mut value = parsed();
        for task in value["tasks"].as_array_mut().unwrap() {
            task["domain"] = Value::String("mechanical_infrastructure".into());
        }
        assert!(validate_manifest(&value).is_err());
    }
}
