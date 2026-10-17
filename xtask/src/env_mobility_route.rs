use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;

const FIXTURE_PATH: &str = "docs/research/fixtures/env-mobility-microfactory-routing-v1.json";

#[derive(Debug, Serialize)]
pub struct RouteAudit {
    pub report_schema: &'static str,
    pub claim_ceiling: &'static str,
    pub routing_id: String,
    pub schema_version: String,
    pub fixture_path: String,
    pub raw_sha256: String,
    pub operation_count: usize,
    pub worker_count: usize,
    pub resource_count: usize,
    pub mutation_case_count: usize,
    pub all_ids_unique: bool,
    pub all_skills_have_a_qualified_worker_candidate: bool,
    pub all_capability_sets_have_a_resource_candidate: bool,
    pub all_predecessors_resolve: bool,
    pub dependency_graph_acyclic: bool,
    pub all_mandatory_operations_present: bool,
    pub critical_gate_order_valid: bool,
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value.as_object().with_context(|| format!("{context}: expected object"))
}

fn exact_keys(value: &Value, required: &[&str], optional: &[&str], context: &str) -> Result<()> {
    let map = object(value, context)?;
    let keys: HashSet<&str> = map.keys().map(String::as_str).collect();
    let required: HashSet<&str> = required.iter().copied().collect();
    let allowed: HashSet<&str> = required.iter().chain(optional.iter()).copied().collect();
    ensure!(required.is_subset(&keys) && keys.is_subset(&allowed),
        "{context}: required/allowed keys differ from schema");
    Ok(())
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str)
        .with_context(|| format!("{context}: {key} must be a string"))
}

fn positive_minutes(value: &Value, key: &str, context: &str) -> Result<u64> {
    let amount = value.get(key).and_then(Value::as_u64)
        .with_context(|| format!("{context}: {key} must be a nonnegative integer"))?;
    ensure!(amount > 0, "{context}: {key} must be positive");
    Ok(amount)
}

fn string_set(value: &Value, key: &str, context: &str) -> Result<HashSet<String>> {
    let values = value.get(key).and_then(Value::as_array)
        .with_context(|| format!("{context}: {key} must be an array"))?;
    let mut result = HashSet::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let value = item.as_str()
            .with_context(|| format!("{context}: {key}[{index}] must be a string"))?;
        ensure!(!value.trim().is_empty(), "{context}: {key}[{index}] is empty");
        ensure!(result.insert(value.to_owned()), "{context}: duplicate value {value:?} in {key}");
    }
    Ok(result)
}

fn array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>> {
    value.get(key).and_then(Value::as_array)
        .with_context(|| format!("{context}: {key} must be an array"))
}

fn reachable(start: &str, target: &str, graph: &BTreeMap<String, Vec<String>>) -> bool {
    let mut pending = vec![start.to_owned()];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if node == target {
            return true;
        }
        if !seen.insert(node.clone()) {
            continue;
        }
        if let Some(children) = graph.get(&node) {
            pending.extend(children.iter().cloned());
        }
    }
    false
}

pub fn validate_route(value: &Value, raw_sha256: String, context: &str) -> Result<RouteAudit> {
    exact_keys(value, &[
        "routing_id", "schema_version", "status", "article_id", "article_description",
        "source_class", "claim_ceiling", "duration_unit", "estimated_durations_are_observed",
        "route_api_mapping", "workers", "resources", "operations", "mandatory_operation_ids",
        "route_invariants", "mutation_cases",
    ], &[], context)?;

    ensure!(string_field(value, "schema_version", context)? == "env-mobility-routing-case-v1",
        "{context}: unsupported route schema version");
    ensure!(string_field(value, "status", context)? == "synthetic_planning_fixture",
        "{context}: fixture must remain synthetic");
    ensure!(string_field(value, "duration_unit", context)? == "minutes",
        "{context}: duration unit must be minutes");
    ensure!(value.get("estimated_durations_are_observed").and_then(Value::as_bool) == Some(false),
        "{context}: estimated durations must not masquerade as observations");
    for key in ["routing_id", "article_id", "article_description", "source_class", "claim_ceiling"] {
        ensure!(!string_field(value, key, context)?.trim().is_empty(),
            "{context}: {key} cannot be empty");
    }

    let mapping = value.get("route_api_mapping").context("route_api_mapping missing")?;
    exact_keys(mapping, &[
        "canonical_owner", "implementation_pr", "mapping_status", "do_not_create_second_scheduler",
    ], &[], "route_api_mapping")?;
    ensure!(string_field(mapping, "canonical_owner", context)? == "Luminous-Dynamics/symthaea#7677",
        "{context}: mapping must point to the existing scheduler owner");
    ensure!(string_field(mapping, "mapping_status", context)? == "pending API review against exact source head",
        "{context}: fixture cannot claim scheduler API conformance before review");
    ensure!(mapping.get("do_not_create_second_scheduler").and_then(Value::as_bool) == Some(true),
        "{context}: duplicate scheduler prohibition must remain explicit");

    let workers = array(value, "workers", context)?;
    let resources = array(value, "resources", context)?;
    let operations = array(value, "operations", context)?;
    let mutations = array(value, "mutation_cases", context)?;
    let mandatory = string_set(value, "mandatory_operation_ids", context)?;
    let invariants = string_set(value, "route_invariants", context)?;
    ensure!(workers.len() == 8, "{context}: v1 expects eight worker profiles");
    ensure!(resources.len() == 11, "{context}: v1 expects eleven resource profiles");
    ensure!(operations.len() == 10, "{context}: v1 expects ten operations");
    ensure!(mutations.len() == 10, "{context}: v1 expects ten mutation cases");
    ensure!(invariants.len() >= 5, "{context}: route invariants are missing");

    let mut worker_ids = HashSet::new();
    let mut worker_skills = Vec::with_capacity(workers.len());
    for (index, worker) in workers.iter().enumerate() {
        let at = format!("{context} worker[{index}]");
        exact_keys(worker, &["worker_id", "skill_ids", "available_minutes"], &[], &at)?;
        let id = string_field(worker, "worker_id", &at)?;
        ensure!(worker_ids.insert(id.to_owned()), "{at}: duplicate worker ID {id:?}");
        positive_minutes(worker, "available_minutes", &at)?;
        worker_skills.push(string_set(worker, "skill_ids", &at)?);
    }

    let mut resource_ids = HashSet::new();
    let mut resource_capabilities = Vec::with_capacity(resources.len());
    for (index, resource) in resources.iter().enumerate() {
        let at = format!("{context} resource[{index}]");
        exact_keys(resource,
            &["resource_id", "capability_ids", "available_minutes", "max_concurrent_operations"],
            &["restriction"], &at)?;
        let id = string_field(resource, "resource_id", &at)?;
        ensure!(resource_ids.insert(id.to_owned()), "{at}: duplicate resource ID {id:?}");
        positive_minutes(resource, "available_minutes", &at)?;
        positive_minutes(resource, "max_concurrent_operations", &at)?;
        if let Some(restriction) = resource.get("restriction") {
            ensure!(restriction.as_str().is_some_and(|s| !s.trim().is_empty()),
                "{at}: restriction must be a non-empty string");
        }
        resource_capabilities.push(string_set(resource, "capability_ids", &at)?);
    }

    let mut operation_ids = HashSet::new();
    let mut operation_map: BTreeMap<String, &Value> = BTreeMap::new();
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut indegree: BTreeMap<String, usize> = BTreeMap::new();
    for (index, operation) in operations.iter().enumerate() {
        let at = format!("{context} operation[{index}]");
        exact_keys(operation, &[
            "operation_id", "title", "duration_minutes_estimate", "predecessor_ids",
            "required_worker_skill_ids", "required_resource_capability_ids", "outputs",
        ], &[], &at)?;
        let id = string_field(operation, "operation_id", &at)?.to_owned();
        ensure!(!id.trim().is_empty(), "{at}: empty operation ID");
        ensure!(operation_ids.insert(id.clone()), "{at}: duplicate operation ID {id:?}");
        ensure!(!string_field(operation, "title", &at)?.trim().is_empty(), "{at}: empty title");
        let duration = positive_minutes(operation, "duration_minutes_estimate", &at)?;
        ensure!(duration <= 480, "{at}: duration exceeds declared 480-minute shift");
        let predecessors = string_set(operation, "predecessor_ids", &at)?;
        let skills = string_set(operation, "required_worker_skill_ids", &at)?;
        let capabilities = string_set(operation, "required_resource_capability_ids", &at)?;
        ensure!(!skills.is_empty(), "{at}: required worker skills cannot be empty");
        ensure!(!capabilities.is_empty(), "{at}: resource capabilities cannot be empty");
        ensure!(!string_set(operation, "outputs", &at)?.is_empty(), "{at}: operation outputs required");

        ensure!(workers.iter().zip(worker_skills.iter()).any(|(worker, available_skills)| {
            worker.get("available_minutes").and_then(Value::as_u64).unwrap_or(0) >= duration
                && skills.is_subset(available_skills)
        }), "{at}: no single available worker candidate has every required skill");
        ensure!(resources.iter().zip(resource_capabilities.iter()).any(|(resource, capabilities_available)| {
            resource.get("available_minutes").and_then(Value::as_u64).unwrap_or(0) >= duration
                && resource.get("max_concurrent_operations").and_then(Value::as_u64).unwrap_or(0) > 0
                && capabilities.is_subset(capabilities_available)
        }), "{at}: no single available resource candidate has every required capability");

        operation_map.insert(id.clone(), operation);
        children.insert(id.clone(), Vec::new());
        indegree.insert(id, predecessors.len());
    }

    for operation in operations {
        let id = string_field(operation, "operation_id", context)?.to_owned();
        for predecessor in string_set(operation, "predecessor_ids", context)? {
            ensure!(operation_ids.contains(&predecessor),
                "{context}: {id:?} references unknown predecessor {predecessor:?}");
            children.get_mut(&predecessor).context("known predecessor node missing")?.push(id.clone());
        }
    }
    for list in children.values_mut() {
        list.sort();
    }

    let mut ready: VecDeque<String> = indegree.iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect();
    let mut visited_count = 0usize;
    while let Some(id) = ready.pop_front() {
        visited_count += 1;
        if let Some(next) = children.get(&id) {
            for child in next {
                let degree = indegree.get_mut(child).context("known child node missing")?;
                *degree = degree.checked_sub(1).context("invalid dependency indegree")?;
                if *degree == 0 {
                    ready.push_back(child.clone());
                }
            }
        }
    }
    ensure!(visited_count == operations.len(), "{context}: dependency graph contains a cycle");
    ensure!(mandatory.len() == operations.len() && mandatory == operation_ids,
        "{context}: every operation must be mandatory exactly once for v1");

    let required_order = [
        ("OP-00-release-design", "OP-01-incoming-inspection"),
        ("OP-00-release-design", "OP-02-adapter-fabrication"),
        ("OP-01-incoming-inspection", "OP-03-harness-prep"),
        ("OP-02-adapter-fabrication", "OP-04-adapter-dimensional-check"),
        ("OP-03-harness-prep", "OP-05-module-assembly"),
        ("OP-04-adapter-dimensional-check", "OP-05-module-assembly"),
        ("OP-01-incoming-inspection", "OP-05-module-assembly"),
        ("OP-05-module-assembly", "OP-06-electrical-functional-check"),
        ("OP-06-electrical-functional-check", "OP-07-reference-measurement-check"),
        ("OP-07-reference-measurement-check", "OP-08-bind-build-manifest"),
        ("OP-08-bind-build-manifest", "OP-09-independent-acceptance"),
    ];
    for (before, after) in required_order {
        ensure!(operation_map.contains_key(before) && operation_map.contains_key(after),
            "{context}: mandatory gate {before:?} or {after:?} missing");
        ensure!(reachable(before, after, &children),
            "{context}: required ordering {before:?} -> {after:?} not guaranteed");
    }

    let acceptance = operation_map.get("OP-09-independent-acceptance").context("acceptance step missing")?;
    let acceptance_skills = string_set(acceptance, "required_worker_skill_ids", context)?;
    let acceptance_caps = string_set(acceptance, "required_resource_capability_ids", context)?;
    ensure!(acceptance_skills.contains("independent_acceptance_review")
            && acceptance_skills.contains("nonconformance_disposition"),
        "{context}: acceptance lacks required inspection skills");
    ensure!(acceptance_caps.contains("independent_acceptance_review")
            && acceptance_caps.contains("nonconformance_disposition"),
        "{context}: acceptance lacks required review resource");

    let measurement = operation_map.get("OP-07-reference-measurement-check").context("measurement step missing")?;
    let measurement_skills = string_set(measurement, "required_worker_skill_ids", context)?;
    let measurement_caps = string_set(measurement, "required_resource_capability_ids", context)?;
    for gate in ["calibration_check", "uncertainty_recording", "reference_instrument_use"] {
        ensure!(measurement_skills.contains(gate) && measurement_caps.contains(gate),
            "{context}: reference measurement lacks {gate}");
    }
    let harness = operation_map.get("OP-03-harness-prep").context("harness step missing")?;
    ensure!(string_set(harness, "required_worker_skill_ids", context)?.contains("continuity_polarity_test"),
        "{context}: harness preparation lacks continuity/polarity skill");

    let mut mutation_ids = HashSet::new();
    for (index, mutation) in mutations.iter().enumerate() {
        let at = format!("{context} mutation[{index}]");
        exact_keys(mutation, &["case_id", "mutation", "expected"], &[], &at)?;
        let id = string_field(mutation, "case_id", &at)?;
        ensure!(mutation_ids.insert(id.to_owned()), "{at}: duplicate mutation ID {id:?}");
        ensure!(!string_field(mutation, "mutation", &at)?.trim().is_empty(), "{at}: empty mutation description");
        ensure!(!string_field(mutation, "expected", &at)?.trim().is_empty(), "{at}: empty expected outcome");
    }

    Ok(RouteAudit {
        report_schema: "env-mobility-route-audit-v1",
        claim_ceiling: "Structural planning-fixture validation only; no schedule feasibility, factory qualification, manufactured article, cost, safety or field capability is established.",
        routing_id: string_field(value, "routing_id", context)?.to_owned(),
        schema_version: string_field(value, "schema_version", context)?.to_owned(),
        fixture_path: FIXTURE_PATH.to_owned(),
        raw_sha256,
        operation_count: operations.len(),
        worker_count: workers.len(),
        resource_count: resources.len(),
        mutation_case_count: mutations.len(),
        all_ids_unique: operation_ids.len() == operations.len()
            && worker_ids.len() == workers.len()
            && resource_ids.len() == resources.len()
            && mutation_ids.len() == mutations.len(),
        all_skills_have_a_qualified_worker_candidate: true,
        all_capability_sets_have_a_resource_candidate: true,
        all_predecessors_resolve: true,
        dependency_graph_acyclic: true,
        all_mandatory_operations_present: true,
        critical_gate_order_valid: true,
    })
}

pub fn run() -> Result<RouteAudit> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().context("xtask must be one level below workspace root")?;
    let path = root.join(FIXTURE_PATH);
    let bytes = fs::read(&path).with_context(|| format!("read route fixture {}", path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse route fixture {}", path.display()))?;
    validate_route(&value, digest, FIXTURE_PATH)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
        let bytes = fs::read(root.join(FIXTURE_PATH)).expect("read fixture");
        serde_json::from_slice(&bytes).expect("parse fixture")
    }

    #[test]
    fn routing_fixture_has_complete_acyclic_route_and_gates() {
        let report = run().expect("audit route fixture");
        assert_eq!(report.operation_count, 10);
        assert_eq!(report.worker_count, 8);
        assert_eq!(report.resource_count, 11);
        assert_eq!(report.mutation_case_count, 10);
        assert!(report.all_ids_unique);
        assert!(report.all_skills_have_a_qualified_worker_candidate);
        assert!(report.all_capability_sets_have_a_resource_candidate);
        assert!(report.all_predecessors_resolve);
        assert!(report.dependency_graph_acyclic);
        assert!(report.all_mandatory_operations_present);
        assert!(report.critical_gate_order_valid);
    }

    #[test]
    fn route_mutations_fail_closed() {
        let mut value = fixture();
        value["operations"][9]["required_worker_skill_ids"] = json!(["nonconformance_disposition"]);
        assert!(validate_route(&value, "mutation".into(), "test missing inspection skill").is_err());

        let mut value = fixture();
        value["resources"][8]["capability_ids"] = json!(["reference_instrument_use"]);
        assert!(validate_route(&value, "mutation".into(), "test missing metrology capability").is_err());

        let mut value = fixture();
        value["operations"][0]["predecessor_ids"] = json!(["OP-09-independent-acceptance"]);
        assert!(validate_route(&value, "mutation".into(), "test dependency cycle").is_err());

        let mut value = fixture();
        value["operations"][1]["predecessor_ids"] = json!(["OP-UNKNOWN"]);
        assert!(validate_route(&value, "mutation".into(), "test unknown predecessor").is_err());

        let mut value = fixture();
        value["operations"].as_array_mut().expect("operations array")
            .retain(|operation| operation["operation_id"] != "OP-07-reference-measurement-check");
        assert!(validate_route(&value, "mutation".into(), "test removed calibration step").is_err());
    }
}
