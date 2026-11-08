//! Semantic validation and canonical digesting for physical scenario manifest v1.
//!
//! Run the JSON Schema check first, then this semantic pass. This utility does
//! not infer physical reliability; it checks that the manifest is internally
//! referentially coherent and that key evidence/experiment invariants hold.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub canonical_sha256: String,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn bool_at(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn is_stable_identifier(id: &str) -> bool {
    id.len() >= 3
        && id.as_bytes().first().is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b".-_".contains(&byte)
        })
}

fn array_at<'a>(value: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    value.get(key).and_then(Value::as_array)
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort_unstable();
            let mut sorted = Map::new();
            for key in keys {
                if let Some(item) = object.get(key) {
                    sorted.insert(key.clone(), canonicalize(item));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(canonicalize).collect())
        }
        _ => value.clone(),
    }
}

/// Canonical JSON digest: object keys are sorted recursively; array order is
/// preserved. The digest identifies these exact manifest bytes/values, not the
/// physical truth of the claims they contain.
pub fn canonical_sha256(value: &Value) -> String {
    let normalized = canonicalize(value);
    let bytes = serde_json::to_vec(&normalized).unwrap_or_default();
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(7 + digest.len() * 2);
    output.push_str("sha256:");
    for byte in digest {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn parse_seed_range(
    value: Option<&Value>,
    path: &str,
    errors: &mut Vec<String>,
) -> Option<(u64, u64)> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) if values.len() == 2 => {
            let start = values[0].as_u64();
            let end = values[1].as_u64();
            match (start, end) {
                (Some(start), Some(end)) if start <= end => Some((start, end)),
                (Some(_), Some(_)) => {
                    errors.push(format!("{path}: range start must be <= range end"));
                    None
                }
                _ => {
                    errors.push(format!("{path}: expected two non-negative integer seeds"));
                    None
                }
            }
        }
        Some(_) => {
            errors.push(format!("{path}: expected null or [start, end]"));
            None
        }
    }
}

fn ranges_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

fn validate_parent_acyclic(
    parents: &BTreeMap<String, Option<String>>,
    errors: &mut Vec<String>,
) {
    for start in parents.keys() {
        let mut seen = BTreeSet::new();
        let mut current = Some(start.as_str());
        while let Some(domain_id) = current {
            if !seen.insert(domain_id.to_owned()) {
                errors.push(format!(
                    "fault_domains: parent cycle detected while following {start}"
                ));
                break;
            }
            current = parents
                .get(domain_id)
                .and_then(|parent| parent.as_deref());
        }
    }
}

/// Result of a deterministic domain-outage preflight replay.
///
/// This is deliberately not a task-execution result: it never claims that a
/// computation was correct or accepted. It only evaluates whether the declared
/// candidate path and verifier remain available under supported domain outages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightReplay {
    pub failed_domain_ids: Vec<String>,
    pub failed_component_ids: Vec<String>,
    pub candidate_available: bool,
    pub verifier_available: bool,
    pub classification: String,
}

/// Replay only deterministic `domain_outage` entries with parameters.state
/// == `failed`. Unsupported injection mechanisms fail closed rather than being
/// silently ignored. Parent-domain and declared dependency failures propagate
/// to descendants/dependent domains.
pub fn replay_domain_outages(value: &Value) -> Result<PreflightReplay, String> {
    let domains = array_at(value, "fault_domains")
        .ok_or_else(|| "fault_domains must be an array".to_owned())?;
    let dependencies = array_at(value, "domain_dependencies")
        .ok_or_else(|| "domain_dependencies must be an array".to_owned())?;
    let injections = array_at(value, "fault_injections")
        .ok_or_else(|| "fault_injections must be an array".to_owned())?;

    let mut domain_ids = BTreeSet::new();
    for domain in domains {
        let id = string_at(domain, "domain_id")
            .ok_or_else(|| "fault domain is missing domain_id".to_owned())?;
        if !domain_ids.insert(id.to_owned()) {
            return Err(format!("duplicate domain id {id}"));
        }
    }

    let mut failed_domains = BTreeSet::new();
    let mut failed_components = BTreeSet::new();
    let mut component_ids = BTreeSet::new();
    if let Some(components) = array_at(value, "components") {
        for component in components {
            let id = string_at(component, "component_id")
                .ok_or_else(|| "component is missing component_id".to_owned())?;
            if !component_ids.insert(id.to_owned()) {
                return Err(format!("duplicate component id {id}"));
            }
        }
    } else {
        return Err("components must be an array".to_owned());
    }

    for injection in injections {
        let mechanism = string_at(injection, "mechanism")
            .ok_or_else(|| "fault injection is missing mechanism".to_owned())?;
        if mechanism != "domain_outage" {
            return Err(format!("unsupported replay mechanism {mechanism}"));
        }
        let state = injection.get("parameters")
            .and_then(|p| string_at(p, "state"))
            .ok_or_else(|| "domain_outage is missing parameters.state".to_owned())?;
        if state != "failed" {
            return Err(format!("unsupported domain_outage state {state}"));
        }
        let target = string_at(injection, "target_id")
            .ok_or_else(|| "fault injection is missing target_id".to_owned())?;
        if domain_ids.contains(target) {
            failed_domains.insert(target.to_owned());
        } else if component_ids.contains(target) {
            return Err(format!(
                "domain_outage target {target} is a component; target a fault domain instead"
            ));
        } else {
            return Err(format!("unresolved domain_outage target {target}"));
        }
    }

    // Compute the closure of parent failures and explicit shared dependencies.
    loop {
        let before = failed_domains.len();
        for domain in domains {
            let id = string_at(domain, "domain_id").unwrap_or_default();
            let parent = domain.get("parent_domain_id").and_then(Value::as_str);
            if parent.is_some_and(|parent_id| failed_domains.contains(parent_id)) {
                failed_domains.insert(id.to_owned());
            }
        }
        for dependency in dependencies {
            let dependent = string_at(dependency, "dependent_domain_id")
                .ok_or_else(|| "domain dependency is missing dependent_domain_id".to_owned())?;
            let required = string_at(dependency, "dependency_domain_id")
                .ok_or_else(|| "domain dependency is missing dependency_domain_id".to_owned())?;
            if failed_domains.contains(required) {
                failed_domains.insert(dependent.to_owned());
            }
        }
        if failed_domains.len() == before {
            break;
        }
    }

    for domain in domains {
        let id = string_at(domain, "domain_id").unwrap_or_default();
        if failed_domains.contains(id) {
            if let Some(members) = array_at(domain, "member_component_ids") {
                for member in members {
                    if let Some(component_id) = member.as_str() {
                        failed_components.insert(component_id.to_owned());
                    }
                }
            }
        }
    }

    let required_components = array_at(
        value.get("workload").unwrap_or(&Value::Null),
        "required_component_ids",
    ).ok_or_else(|| "workload.required_component_ids must be an array".to_owned())?;
    let candidate_available = required_components
        .iter()
        .filter_map(Value::as_str)
        .all(|component| !failed_components.contains(component));

    let verifier_id = value.get("mitigation")
        .and_then(|m| string_at(m, "verifier_component_id"));
    let verifier_available = verifier_id
        .is_some_and(|component| !failed_components.contains(component));

    let classification = if !candidate_available {
        "hardware_unavailable"
    } else if !verifier_available {
        "verification_unavailable"
    } else {
        "preflight_ready"
    };

    Ok(PreflightReplay {
        failed_domain_ids: failed_domains.into_iter().collect(),
        failed_component_ids: failed_components.into_iter().collect(),
        candidate_available,
        verifier_available,
        classification: classification.to_owned(),
    })
}

/// Validate semantic constraints that JSON Schema cannot express conveniently.
///
/// This deliberately does not qualify probability distributions, physical
/// independence, calibration physics, or device behavior. Those require
/// separate evidence and domain-specific analysis.
pub fn validate_manifest(value: &Value) -> ValidationReport {
    let mut errors = Vec::new();

    if string_at(value, "manifest_version") != Some("1.0.0") {
        errors.push("manifest_version: expected exactly 1.0.0".to_owned());
    }

    let scenario_id = string_at(value, "scenario_id");
    if !scenario_id.is_some_and(is_stable_identifier) {
        errors.push("scenario_id: expected a lowercase stable identifier".to_owned());
    }

    let provenance = value.get("provenance").unwrap_or(&Value::Null);
    let source_kind = string_at(provenance, "source_kind");
    let source_digest = string_at(provenance, "source_data_digest");
    let scenario_class = string_at(value, "scenario_class");
    if matches!(source_kind, Some("measured" | "derived_from_measurements"))
        && source_digest.is_none()
    {
        errors.push("provenance.source_data_digest: required for measured/derived scenarios".to_owned());
    }

    let workload = value.get("workload").unwrap_or(&Value::Null);
    let oracle = workload.get("oracle").unwrap_or(&Value::Null);
    if bool_at(oracle, "independent_of_candidate") != Some(true) {
        errors.push("workload.oracle.independent_of_candidate: must be true".to_owned());
    }
    if bool_at(oracle, "evaluator_truth_withheld_from_verifier") != Some(true) {
        errors.push("workload.oracle.evaluator_truth_withheld_from_verifier: must be true".to_owned());
    }
    if string_at(workload, "numerical_semantics").is_none() {
        errors.push("workload.numerical_semantics: required".to_owned());
    }
    if workload.get("deadline_ticks").and_then(Value::as_u64).unwrap_or(0) == 0 {
        errors.push("workload.deadline_ticks: must be a positive integer".to_owned());
    }

    if let Some(condition) = workload.get("condition_number_range") {
        if !condition.is_null() {
            let min = condition.get("minimum").and_then(Value::as_f64);
            let max = condition.get("maximum").and_then(Value::as_f64);
            match (min, max) {
                (Some(min), Some(max)) if min > 0.0 && max > 0.0 && min <= max => {}
                _ => errors.push(
                    "workload.condition_number_range: minimum and maximum must be positive and ordered".to_owned()
                ),
            }
        }
    }

    let substrate = value.get("substrate").unwrap_or(&Value::Null);
    let calibration = substrate.get("calibration").unwrap_or(&Value::Null);
    let calibration_status = string_at(calibration, "status");
    let experiment = value.get("experiment").unwrap_or(&Value::Null);
    let experiment_phase = string_at(experiment, "phase");
    if matches!(experiment_phase, Some("confirmatory" | "reproduction"))
        && calibration_status == Some("measured_stale")
    {
        errors.push(
            "substrate.calibration.status: stale calibration cannot support confirmatory/reproduction evidence".to_owned()
        );
    }
    if matches!(scenario_class, Some("confirmatory" | "reproduction"))
        && matches!(source_kind, Some("measured" | "derived_from_measurements"))
        && calibration_status != Some("measured_current")
    {
        errors.push(
            "substrate.calibration.status: measured confirmatory/reproduction scenarios require measured_current".to_owned()
        );
    }

    let mut component_ids = BTreeSet::new();
    if let Some(components) = array_at(value, "components") {
        for (index, component) in components.iter().enumerate() {
            match string_at(component, "component_id") {
                Some(id) if !id.is_empty() => {
                    if !component_ids.insert(id.to_owned()) {
                        errors.push(format!("components[{index}]: duplicate component_id {id}"));
                    }
                }
                _ => errors.push(format!("components[{index}].component_id: missing or empty")),
            }
        }
    } else {
        errors.push("components: expected an array".to_owned());
    }

    if let Some(required) = array_at(workload, "required_component_ids") {
        if required.is_empty() {
            errors.push("workload.required_component_ids: must not be empty".to_owned());
        }
        let mut required_seen = BTreeSet::new();
        for (index, item) in required.iter().enumerate() {
            match item.as_str() {
                Some(id) if !id.is_empty() => {
                    if !required_seen.insert(id.to_owned()) {
                        errors.push(format!("workload.required_component_ids[{index}]: duplicate {id}"));
                    }
                    if !component_ids.contains(id) {
                        errors.push(format!("workload.required_component_ids[{index}]: unresolved component {id}"));
                    }
                }
                _ => errors.push(format!("workload.required_component_ids[{index}]: expected a non-empty string")),
            }
        }
    } else {
        errors.push("workload.required_component_ids: expected an array".to_owned());
    }

    let mut domain_ids = BTreeSet::new();
    let mut parents = BTreeMap::new();
    if let Some(domains) = array_at(value, "fault_domains") {
        for (index, domain) in domains.iter().enumerate() {
            let id = match string_at(domain, "domain_id") {
                Some(id) if !id.is_empty() => id.to_owned(),
                _ => {
                    errors.push(format!("fault_domains[{index}].domain_id: missing or empty"));
                    continue;
                }
            };
            if !domain_ids.insert(id.clone()) {
                errors.push(format!("fault_domains[{index}]: duplicate domain_id {id}"));
            }
            let parent = match domain.get("parent_domain_id") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
                _ => {
                    errors.push(format!("fault_domains[{index}].parent_domain_id: expected null or non-empty string"));
                    None
                }
            };
            parents.insert(id.clone(), parent);

            if let Some(members) = array_at(domain, "member_component_ids") {
                let mut local_members = BTreeSet::new();
                for member in members {
                    match member.as_str() {
                        Some(component_id) => {
                            if !local_members.insert(component_id.to_owned()) {
                                errors.push(format!(
                                    "fault_domains[{index}]: duplicate member component {component_id}"
                                ));
                            }
                            if !component_ids.contains(component_id) {
                                errors.push(format!(
                                    "fault_domains[{index}]: unresolved component reference {component_id}"
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "fault_domains[{index}].member_component_ids: every member must be a string"
                        )),
                    }
                }
            } else {
                errors.push(format!("fault_domains[{index}].member_component_ids: expected an array"));
            }
        }
    } else {
        errors.push("fault_domains: expected an array".to_owned());
    }

    for (domain_id, parent) in &parents {
        if let Some(parent_id) = parent {
            if !domain_ids.contains(parent_id) {
                errors.push(format!(
                    "fault_domains.{domain_id}.parent_domain_id: unresolved domain {parent_id}"
                ));
            }
            if parent_id == domain_id {
                errors.push(format!("fault_domains.{domain_id}: cannot parent itself"));
            }
        }
    }
    validate_parent_acyclic(&parents, &mut errors);

    if let Some(dependencies) = array_at(value, "domain_dependencies") {
        for (index, dependency) in dependencies.iter().enumerate() {
            let dependent = string_at(dependency, "dependent_domain_id");
            let required = string_at(dependency, "dependency_domain_id");
            match (dependent, required) {
                (Some(a), Some(b)) => {
                    if !domain_ids.contains(a) {
                        errors.push(format!("domain_dependencies[{index}]: unresolved dependent domain {a}"));
                    }
                    if !domain_ids.contains(b) {
                        errors.push(format!("domain_dependencies[{index}]: unresolved dependency domain {b}"));
                    }
                    if a == b {
                        errors.push(format!("domain_dependencies[{index}]: self-dependency is invalid"));
                    }
                }
                _ => errors.push(format!("domain_dependencies[{index}]: both domain references are required")),
            }
        }
    } else {
        errors.push("domain_dependencies: expected an array".to_owned());
    }

    if let Some(injections) = array_at(value, "fault_injections") {
        let mut injection_ids = BTreeSet::new();
        for (index, injection) in injections.iter().enumerate() {
            match string_at(injection, "injection_id") {
                Some(id) if !id.is_empty() => {
                    if !injection_ids.insert(id.to_owned()) {
                        errors.push(format!("fault_injections[{index}]: duplicate injection_id {id}"));
                    }
                }
                _ => errors.push(format!("fault_injections[{index}].injection_id: missing or empty")),
            }
            match string_at(injection, "target_id") {
                Some(target)
                    if component_ids.contains(target) || domain_ids.contains(target) => {}
                Some(target) => errors.push(format!(
                    "fault_injections[{index}].target_id: unresolved component/domain {target}"
                )),
                None => errors.push(format!("fault_injections[{index}].target_id: missing")),
            }
        }
    } else {
        errors.push("fault_injections: expected an array".to_owned());
    }

    let strategy = value.get("mitigation").unwrap_or(&Value::Null);
    for key in ["verifier_component_id", "fallback_component_id"] {
        if let Some(id) = strategy.get(key).and_then(Value::as_str) {
            if !component_ids.contains(id) {
                errors.push(format!("mitigation.{key}: unresolved component {id}"));
            }
        }
    }

    let phase = string_at(experiment, "phase");
    let seed_policy = string_at(experiment, "seed_policy");
    if experiment.get("planned_units").and_then(Value::as_u64).unwrap_or(0) == 0 {
        errors.push("experiment.planned_units: must be positive".to_owned());
    }
    if string_at(experiment, "independence_statement").is_none_or(str::is_empty) {
        errors.push("experiment.independence_statement: required".to_owned());
    }
    if phase == Some("confirmatory") && seed_policy != Some("independent_held_out") {
        errors.push("experiment.seed_policy: confirmatory phase requires independent_held_out".to_owned());
    }
    let expected_phase = match scenario_class {
        Some("screening") => Some("screening"),
        Some("confirmatory") => Some("confirmatory"),
        Some("negative_control") => Some("negative_control"),
        Some("reproduction") => Some("reproduction"),
        _ => None,
    };
    if expected_phase.is_some() && phase != expected_phase {
        errors.push("experiment.phase: must match scenario_class".to_owned());
    }
    let training = parse_seed_range(experiment.get("training_seed_range"), "experiment.training_seed_range", &mut errors);
    let validation = parse_seed_range(experiment.get("validation_seed_range"), "experiment.validation_seed_range", &mut errors);
    let confirmatory = parse_seed_range(experiment.get("confirmatory_seed_range"), "experiment.confirmatory_seed_range", &mut errors);
    let named_ranges = [
        ("training", training),
        ("validation", validation),
        ("confirmatory", confirmatory),
    ];
    for i in 0..named_ranges.len() {
        for j in (i + 1)..named_ranges.len() {
            if let (Some(a), Some(b)) = (named_ranges[i].1, named_ranges[j].1) {
                if ranges_overlap(a, b) {
                    errors.push(format!(
                        "experiment: {} and {} seed ranges overlap",
                        named_ranges[i].0, named_ranges[j].0
                    ));
                }
            }
        }
    }

    let measurement = value.get("measurement").unwrap_or(&Value::Null);
    let energy_boundary = string_at(measurement, "energy_boundary");
    if energy_boundary.is_some_and(|boundary| boundary != "not_measured") {
        if string_at(measurement, "energy_unit").is_none_or(str::is_empty) {
            errors.push("measurement.energy_unit: required when energy is measured".to_owned());
        }
        if string_at(measurement, "instrument_id").is_none_or(str::is_empty) {
            errors.push("measurement.instrument_id: required when energy is measured".to_owned());
        }
        if bool_at(measurement, "raw_receipt_required") != Some(true) {
            errors.push("measurement.raw_receipt_required: must be true when energy is measured".to_owned());
        }
    }
    let latency_boundary = string_at(measurement, "latency_boundary");
    if latency_boundary.is_some_and(|boundary| boundary != "not_measured")
        && string_at(measurement, "latency_unit").is_none_or(str::is_empty)
    {
        errors.push("measurement.latency_unit: required when latency is measured".to_owned());
    }

    let acceptance = value.get("acceptance").unwrap_or(&Value::Null);
    if bool_at(acceptance, "silent_corruption_allowed") != Some(false) {
        errors.push("acceptance.silent_corruption_allowed: must be false".to_owned());
    }
    if scenario_class == Some("confirmatory") && !bool_at(acceptance, "deadline_required").unwrap_or(false) {
        errors.push("acceptance.deadline_required: confirmatory scenarios must enforce a deadline".to_owned());
    }

    ValidationReport {
        errors,
        canonical_sha256: canonical_sha256(value),
    }
}

pub fn validate_json(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_manifest(&value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../scenarios/v1/common_cause_controller_outage.json");
    const SHARED_POWER_FIXTURE: &str =
        include_str!("../../scenarios/v1/common_cause_shared_power_outage.json");

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("fixture JSON must parse")
    }

    #[test]
    fn checked_in_controller_negative_control_is_semantically_valid() {
        let report = validate_manifest(&fixture_value());
        assert!(report.is_valid(), "{:?}", report.errors);
        assert!(report.canonical_sha256.starts_with("sha256:"));
        assert_eq!(report.canonical_sha256.len(), 71);
    }

    #[test]
    fn malformed_scenario_identifier_is_rejected() {
        let mut value = fixture_value();
        value["scenario_id"] = Value::String("Not A Stable ID".into());
        assert!(validate_manifest(&value).errors.iter().any(|e| e.contains("scenario_id")));
    }

    #[test]
    fn shared_power_negative_control_is_semantically_valid() {
        let value: Value = serde_json::from_str(SHARED_POWER_FIXTURE)
            .expect("shared-power fixture JSON must parse");
        let report = validate_manifest(&value);
        assert!(report.is_valid(), "{:?}", report.errors);
        assert_eq!(
            value["fault_injections"][0]["target_id"].as_str(),
            Some("shared-power-domain-0")
        );
    }

    #[test]
    fn controller_outage_replay_blocks_candidate_but_leaves_verifier_up() {
        let value = fixture_value();
        let replay = replay_domain_outages(&value).unwrap();
        assert!(!replay.candidate_available);
        assert!(replay.verifier_available);
        assert_eq!(replay.classification, "hardware_unavailable");
        assert!(replay.failed_component_ids.iter().any(|id| id == "controller-0"));
    }

    #[test]
    fn shared_power_outage_replay_blocks_candidate_and_verifier() {
        let value: Value = serde_json::from_str(SHARED_POWER_FIXTURE).unwrap();
        let replay = replay_domain_outages(&value).unwrap();
        assert!(!replay.candidate_available);
        assert!(!replay.verifier_available);
        assert_eq!(replay.classification, "hardware_unavailable");
        assert!(replay.failed_component_ids.iter().any(|id| id == "array-0"));
        assert!(replay.failed_component_ids.iter().any(|id| id == "verifier-0"));
    }

    #[test]
    fn domain_outage_cannot_target_a_component_directly() {
        let mut value = fixture_value();
        value["fault_injections"][0]["target_id"] = Value::String("controller-0".into());
        assert!(replay_domain_outages(&value).unwrap_err().contains("target a fault domain"));
    }

    #[test]
    fn unsupported_replay_mechanisms_fail_closed() {
        let mut value = fixture_value();
        value["fault_injections"][0]["mechanism"] = Value::String("read_noise".into());
        assert!(replay_domain_outages(&value).is_err());
    }

    #[test]
    fn duplicate_component_id_is_rejected() {
        let mut value = fixture_value();
        let first = value["components"][0].clone();
        value["components"].as_array_mut().unwrap().push(first);
        assert!(validate_manifest(&value).errors.iter().any(|e| e.contains("duplicate component_id")));
    }

    #[test]
    fn unresolved_domain_member_is_rejected() {
        let mut value = fixture_value();
        value["fault_domains"][0]["member_component_ids"]
            .as_array_mut().unwrap().push(Value::String("missing-component".into()));
        assert!(validate_manifest(&value).errors.iter().any(|e| e.contains("unresolved component reference")));
    }

    #[test]
    fn cyclic_parent_hierarchy_is_rejected() {
        let mut value = fixture_value();
        value["fault_domains"][0]["parent_domain_id"] = Value::String("board-0".into());
        assert!(validate_manifest(&value).errors.iter().any(|e| e.contains("parent cycle")));
    }

    #[test]
    fn overlapping_seed_partitions_are_rejected() {
        let mut value = fixture_value();
        value["experiment"]["training_seed_range"] = serde_json::json!([1, 100]);
        value["experiment"]["validation_seed_range"] = serde_json::json!([100, 200]);
        assert!(validate_manifest(&value).errors.iter().any(|e| e.contains("seed ranges overlap")));
    }

    #[test]
    fn energy_claim_without_instrument_and_receipt_is_rejected() {
        let mut value = fixture_value();
        value["measurement"]["energy_boundary"] = Value::String("whole_system_wall_input".into());
        value["measurement"]["energy_unit"] = Value::String("J".into());
        value["measurement"]["instrument_id"] = Value::Null;
        value["measurement"]["raw_receipt_required"] = Value::Bool(false);
        let errors = validate_manifest(&value).errors;
        assert!(errors.iter().any(|e| e.contains("instrument_id")));
        assert!(errors.iter().any(|e| e.contains("raw_receipt_required")));
    }

    #[test]
    fn key_order_does_not_change_canonical_digest() {
        let a: Value = serde_json::from_str(r#"{"b":2,"a":{"y":4,"x":3}}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":{"x":3,"y":4},"b":2}"#).unwrap();
        assert_eq!(canonical_sha256(&a), canonical_sha256(&b));
    }
}
