// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Structural contract checker for the Tier-B chemical-process candidate registry.
// It does not execute a simulator or establish chemical-engineering validity.

use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    process::ExitCode,
};

const ALLOWED_FREEZE_STATES: &[&str] =
    &["CANDIDATE_NOT_FROZEN", "DESIGN_ONLY", "FROZEN"];

const ALLOWED_RESULT_STATUSES: &[&str] = &[
    "PASS",
    "FAIL",
    "SKIPPED",
    "NOT_RUN",
    "OUT_OF_DOMAIN",
    "REFERENCE_UNAVAILABLE",
    "UNRESOLVED",
];

const RECEIPT_FIELD_NAMES: &[&str] = &[
    "primary_source_url_and_revision",
    "retrieved_at",
    "native_input_files_and_transitive_data_hashes",
    "component_identity_and_order",
    "units_and_composition_basis",
    "phase_and_reference_state_conventions",
    "property_package_or_mechanism_and_parameter_provenance",
    "solver_build_and_runtime_versions",
    "numerical_methods_tolerances_and_convergence_status",
    "stdout_stderr_and_raw_output_hashes",
    "independent_checker_version_and_result",
    "shared_dependency_audit",
    "preregistered_case_specific_tolerance_and_rationale",
    "claim_ceiling",
];

fn text<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn nonempty_text(value: &Value, field: &str) -> bool {
    text(value, field).is_some_and(|text| !text.trim().is_empty())
}

fn has_content(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        _ => true,
    }
}

fn is_sha256(value: &Value) -> bool {
    let Some(value) = value.as_str() else {
        return false;
    };
    let value = value.strip_prefix("sha256:").unwrap_or(value);
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

fn source_has_url(source: &Value) -> bool {
    source.as_object().is_some_and(|fields| {
        fields.values().any(|value| {
            value.as_str().is_some_and(|candidate| {
                candidate.starts_with("https://") || candidate.starts_with("http://")
            })
        })
    })
}

fn digest_set_with_empty(value: &Value, allow_empty: bool) -> Option<BTreeSet<String>> {
    let values = value.as_array()?;
    if !allow_empty && values.is_empty() {
        return None;
    }
    let mut digests = BTreeSet::new();
    for value in values {
        if !is_sha256(value) || !digests.insert(value.as_str()?.to_owned()) {
            return None;
        }
    }
    Some(digests)
}

fn digest_set(value: &Value) -> Option<BTreeSet<String>> {
    digest_set_with_empty(value, false)
}

fn visit_lineage_node(
    digest: &str,
    graph: &BTreeMap<String, BTreeSet<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), String> {
    if visited.contains(digest) {
        return Ok(());
    }
    if !visiting.insert(digest.to_owned()) {
        return Err(format!("lineage graph contains a cycle through {digest}"));
    }
    let parents = graph
        .get(digest)
        .ok_or_else(|| format!("lineage graph references undeclared digest {digest}"))?;
    for parent in parents {
        if !graph.contains_key(parent) {
            return Err(format!("{digest} derives from missing lineage node {parent}"));
        }
        visit_lineage_node(parent, graph, visiting, visited)?;
    }
    visiting.remove(digest);
    visited.insert(digest.to_owned());
    Ok(())
}

fn transitive_closure(
    roots: &BTreeSet<String>,
    graph: &BTreeMap<String, BTreeSet<String>>,
) -> Result<BTreeSet<String>, String> {
    let mut closure = BTreeSet::new();
    let mut pending = roots.iter().cloned().collect::<Vec<_>>();
    while let Some(digest) = pending.pop() {
        if !closure.insert(digest.clone()) {
            continue;
        }
        let parents = graph
            .get(&digest)
            .ok_or_else(|| format!("model dependency {digest} is absent from lineage_nodes"))?;
        pending.extend(parents.iter().cloned());
    }
    Ok(closure)
}

fn computed_dependency_closures(
    audit: &Value,
) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    if audit["lineage_complete"].as_bool() != Some(true) {
        return Err("lineage_complete must be true before independence can be assessed".to_owned());
    }
    let nodes = audit["lineage_nodes"]
        .as_array()
        .ok_or_else(|| "lineage_nodes must be an array".to_owned())?;
    if nodes.is_empty() {
        return Err("lineage_nodes must not be empty".to_owned());
    }

    let mut graph = BTreeMap::<String, BTreeSet<String>>::new();
    for (index, node) in nodes.iter().enumerate() {
        if !nonempty_text(node, "artifact") {
            return Err(format!("lineage node {index} requires an artifact identifier"));
        }
        if !is_sha256(&node["sha256"]) {
            return Err(format!("lineage node {index} requires a SHA-256 digest"));
        }
        let digest = node["sha256"]
            .as_str()
            .ok_or_else(|| format!("lineage node {index} digest is not a string"))?
            .to_owned();
        let parents = node["derived_from"]
            .as_array()
            .ok_or_else(|| format!("lineage node {digest} requires a derived_from array"))?;
        let mut parent_set = BTreeSet::new();
        for parent in parents {
            if !is_sha256(parent) {
                return Err(format!("lineage node {digest} has an invalid parent digest"));
            }
            let parent_digest = parent
                .as_str()
                .ok_or_else(|| format!("lineage node {digest} has a non-string parent digest"))?
                .to_owned();
            if parent_digest == digest {
                return Err(format!("lineage node {digest} cannot derive from itself"));
            }
            if !parent_set.insert(parent_digest) {
                return Err(format!("lineage node {digest} repeats a parent digest"));
            }
        }
        if graph.insert(digest.clone(), parent_set).is_some() {
            return Err(format!("lineage graph duplicates digest {digest}"));
        }
    }

    let mut visited = BTreeSet::<String>::new();
    for digest in graph.keys() {
        visit_lineage_node(digest, &graph, &mut BTreeSet::new(), &mut visited)?;
    }

    let model_a = digest_set(&audit["model_a_dependency_digests"])
        .ok_or_else(|| "model_a_dependency_digests must be nonempty, unique SHA-256 digests".to_owned())?;
    let model_b = digest_set(&audit["model_b_dependency_digests"])
        .ok_or_else(|| "model_b_dependency_digests must be nonempty, unique SHA-256 digests".to_owned())?;
    let closure_a = transitive_closure(&model_a, &graph)?;
    let closure_b = transitive_closure(&model_b, &graph)?;
    Ok((closure_a, closure_b))
}

fn validate_receipt_shape(
    receipt: &Value,
    raw_output_hash: &Value,
    errors: &mut Vec<String>,
    id: &str,
) {
    let artifacts = receipt["native_input_files_and_transitive_data_hashes"]
        .as_array();
    match artifacts {
        Some(artifacts) if !artifacts.is_empty() => {
            for (index, artifact) in artifacts.iter().enumerate() {
                if !nonempty_text(artifact, "artifact") {
                    errors.push(format!("{id}: input artifact {index} needs an artifact identifier"));
                }
                if !is_sha256(&artifact["sha256"]) {
                    errors.push(format!("{id}: input artifact {index} needs a SHA-256 digest"));
                }
            }
        }
        _ => errors.push(format!("{id}: input artifacts must be a nonempty digest-bearing array")),
    }

    let streams = &receipt["stdout_stderr_and_raw_output_hashes"];
    for field in ["stdout_sha256", "stderr_sha256", "raw_output_sha256"] {
        if !is_sha256(&streams[field]) {
            errors.push(format!("{id}: stdout/stderr receipt is missing valid {field}"));
        }
    }
    if text(streams, "raw_output_sha256") != raw_output_hash.as_str() {
        errors.push(format!("{id}: receipt raw-output digest disagrees with execution.raw_output_hash"));
    }

    let checker = &receipt["independent_checker_version_and_result"];
    if !nonempty_text(checker, "checker_version") || text(checker, "result") != Some("PASS") {
        errors.push(format!("{id}: freeze receipt requires a versioned independent checker with result=PASS"));
    }

    let dependency_audit = &receipt["shared_dependency_audit"];
    let declared_shared = digest_set_with_empty(&dependency_audit["shared_dependency_digests"], true);
    let closures = computed_dependency_closures(dependency_audit);
    let computed_shared = closures
        .as_ref()
        .map(|(a, b)| a.intersection(b).cloned().collect::<BTreeSet<_>>());
    if dependency_audit["complete"].as_bool() != Some(true)
        || dependency_audit["independence_assessed"].as_bool() != Some(true)
        || declared_shared.is_none()
    {
        errors.push(format!("{id}: dependency audit must be complete, assessed, and declare a valid unique shared dependency set"));
    }
    if let Err(reason) = &closures {
        errors.push(format!("{id}: cannot verify transitive dependency lineage: {reason}"));
    }
    if declared_shared.is_some() && computed_shared.is_some() && declared_shared != computed_shared {
        errors.push(format!("{id}: declared shared_dependency_digests do not equal the computed transitive intersection of model dependency closures"));
    }
    let independent_claimed = dependency_audit["independent_corroboration_claimed"].as_bool();
    if independent_claimed.is_none() {
        errors.push(format!("{id}: dependency audit must explicitly declare independent_corroboration_claimed"));
    } else if independent_claimed == Some(true)
        && computed_shared.as_ref().is_some_and(|shared| !shared.is_empty())
    {
        errors.push(format!("{id}: independent corroboration cannot be claimed when transitive source-data dependencies overlap"));
    }

    let tolerance = &receipt["preregistered_case_specific_tolerance_and_rationale"];
    let metrics = tolerance["metrics"].as_array();
    if tolerance["frozen_before_candidate_evaluation"].as_bool() != Some(true)
        || !nonempty_text(tolerance, "rationale")
        || !metrics.is_some_and(|items| {
            !items.is_empty()
                && items.iter().all(|item| {
                    nonempty_text(item, "metric") && nonempty_text(item, "acceptance_rule")
                })
        })
    {
        errors.push(format!("{id}: tolerances must be case-specific, justified, and frozen before candidate evaluation"));
    }

    if !receipt["claim_ceiling"]
        .as_str()
        .is_some_and(|value| !value.trim().is_empty())
    {
        errors.push(format!("{id}: claim_ceiling must be an explicit nonempty string"));
    }
}

fn validate_frozen_receipt(candidate: &Value, errors: &mut Vec<String>, id: &str) {
    let execution = &candidate["execution"];
    let outputs = &candidate["reference_outputs"];
    let freeze_state = text(candidate, "freeze_state").unwrap_or_default();
    let status = text(execution, "status").unwrap_or_default();
    let executed = execution["simulator_executed"].as_bool() == Some(true);
    let frozen = outputs["exact_simulator_outputs_frozen"].as_bool() == Some(true);

    if frozen {
        if freeze_state != "FROZEN" {
            errors.push(format!("{id}: frozen outputs require freeze_state=FROZEN"));
        }
        if text(outputs, "state") != Some("FROZEN") {
            errors.push(format!("{id}: frozen outputs require reference_outputs.state=FROZEN"));
        }
        if status != "PASS" {
            errors.push(format!("{id}: frozen outputs require execution.status=PASS"));
        }
        if !executed {
            errors.push(format!("{id}: frozen outputs require simulator_executed=true"));
        }
        if !is_sha256(&execution["raw_output_hash"]) {
            errors.push(format!("{id}: frozen outputs require a valid raw-output SHA-256"));
        }
        if !is_sha256(&outputs["output_digest"]) {
            errors.push(format!("{id}: frozen outputs require a valid output SHA-256"));
        }
        if !nonempty_text(execution, "simulator_version") {
            errors.push(format!("{id}: frozen outputs require a pinned simulator_version"));
        }

        let receipt = &outputs["freeze_receipt"];
        if !receipt.is_object() {
            errors.push(format!("{id}: frozen outputs require a freeze_receipt object"));
        } else {
            for field in RECEIPT_FIELD_NAMES {
                if !receipt.get(*field).is_some_and(has_content) {
                    errors.push(format!("{id}: freeze_receipt is missing {field}"));
                }
            }
            validate_receipt_shape(receipt, &execution["raw_output_hash"], errors, id);
        }
        if !candidate["freeze_blockers"]
            .as_array()
            .is_some_and(|blockers| blockers.is_empty())
        {
            errors.push(format!("{id}: a frozen candidate must have an explicit empty freeze_blockers array"));
        }
    } else {
        if freeze_state == "FROZEN" {
            errors.push(format!("{id}: freeze_state=FROZEN requires exact outputs to be frozen"));
        }
        if text(outputs, "state") == Some("FROZEN") {
            errors.push(format!("{id}: reference_outputs.state=FROZEN requires exact outputs to be frozen"));
        }
        if !outputs["output_digest"].is_null() {
            errors.push(format!("{id}: non-frozen outputs require output_digest=null"));
        }
        if status == "PASS" {
            errors.push(format!("{id}: PASS cannot be claimed while exact outputs are unfrozen"));
        }
        if !candidate["freeze_blockers"]
            .as_array()
            .is_some_and(|blockers| !blockers.is_empty())
        {
            errors.push(format!("{id}: an unfrozen candidate must enumerate freeze_blockers"));
        }
    }

    match status {
        "NOT_RUN" => {
            if executed {
                errors.push(format!("{id}: NOT_RUN conflicts with simulator_executed=true"));
            }
            if !execution["raw_output_hash"].is_null() {
                errors.push(format!("{id}: NOT_RUN requires raw_output_hash=null"));
            }
            if !execution["simulator_version"].is_null() {
                errors.push(format!("{id}: NOT_RUN requires simulator_version=null"));
            }
            if frozen {
                errors.push(format!("{id}: NOT_RUN may not freeze simulator outputs"));
            }
        }
        "SKIPPED" | "REFERENCE_UNAVAILABLE" => {
            if frozen {
                errors.push(format!("{id}: {status} cannot become a frozen-output pass"));
            }
            if executed {
                errors.push(format!("{id}: {status} conflicts with simulator_executed=true"));
            }
            if !execution["raw_output_hash"].is_null()
                || !execution["simulator_version"].is_null()
                || !outputs["output_digest"].is_null()
            {
                errors.push(format!("{id}: {status} may not claim simulator/output artifacts"));
            }
        }
        "PASS" | "FAIL" => {
            if !executed {
                errors.push(format!("{id}: {status} requires simulator_executed=true"));
            }
            if !is_sha256(&execution["raw_output_hash"]) {
                errors.push(format!("{id}: {status} requires a valid raw-output SHA-256"));
            }
            if !nonempty_text(execution, "simulator_version") {
                errors.push(format!("{id}: {status} requires simulator_version"));
            }
        }
        "OUT_OF_DOMAIN" | "UNRESOLVED" => {
            if frozen {
                errors.push(format!("{id}: {status} cannot be promoted to frozen gold outputs"));
            }
            if executed && !is_sha256(&execution["raw_output_hash"]) {
                errors.push(format!("{id}: an executed unresolved/out-of-domain run requires a raw-output SHA-256"));
            }
            if executed && !nonempty_text(execution, "simulator_version") {
                errors.push(format!("{id}: an executed unresolved/out-of-domain run requires simulator_version"));
            }
            if !executed
                && (!execution["raw_output_hash"].is_null()
                    || !execution["simulator_version"].is_null())
            {
                errors.push(format!("{id}: an unexecuted unresolved/out-of-domain case may not claim run artifacts"));
            }
        }
        _ => errors.push(format!("{id}: unknown execution status {status:?}")),
    }
}

fn validate_registry(registry: &Value) -> Vec<String> {
    let mut errors = Vec::<String>::new();

    if text(registry, "schema_id")
        != Some("org.luminousdynamics.symthaea.chemical-process-tier-b-candidates")
    {
        errors.push("schema_id is missing or unrecognized".to_owned());
    }
    if text(registry, "schema_version") != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if text(registry, "purpose").is_none_or(|value| value.trim().is_empty()) {
        errors.push("purpose must be explicit".to_owned());
    }
    if registry["safety_and_claim_policy"]["no_operational_authority"].as_bool() != Some(true) {
        errors.push("no_operational_authority must remain true".to_owned());
    }

    let declared_statuses = registry["result_statuses"]
        .as_array()
        .map(|statuses| {
            statuses
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let expected_statuses = ALLOWED_RESULT_STATUSES.iter().copied().collect::<BTreeSet<_>>();
    if declared_statuses != expected_statuses {
        errors.push("result_statuses must exactly match the fail-closed status vocabulary".to_owned());
    }

    let receipt_fields = registry["freeze_receipt_required_fields"]
        .as_array()
        .map(|fields| {
            fields.iter().filter_map(Value::as_str).collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let expected_receipt_fields = RECEIPT_FIELD_NAMES.iter().copied().collect::<BTreeSet<_>>();
    if receipt_fields != expected_receipt_fields {
        errors.push("freeze_receipt_required_fields must match the required receipt contract".to_owned());
    }

    let Some(candidates) = registry.get("candidates").and_then(Value::as_array) else {
        errors.push("candidates must be an array".to_owned());
        return errors;
    };
    if candidates.is_empty() {
        errors.push("candidates must not be empty".to_owned());
        return errors;
    }

    let mut ids = BTreeSet::<String>::new();
    for candidate in candidates {
        let id = text(candidate, "case_id").unwrap_or("<missing-case-id>");
        if id == "<missing-case-id>" || id.trim().is_empty() {
            errors.push("candidate case_id must be nonempty".to_owned());
        } else if !ids.insert(id.to_owned()) {
            errors.push(format!("duplicate candidate case_id {id}"));
        }

        let freeze_state = text(candidate, "freeze_state").unwrap_or_default();
        if !ALLOWED_FREEZE_STATES.contains(&freeze_state) {
            errors.push(format!("{id}: unknown freeze_state {freeze_state:?}"));
        }
        if !source_has_url(&candidate["source"]) {
            errors.push(format!("{id}: source must contain at least one HTTP(S) URL"));
        }
        if !nonempty_text(&candidate["source"], "publisher") {
            errors.push(format!("{id}: source publisher is required"));
        }

        let status = text(&candidate["execution"], "status").unwrap_or_default();
        if !declared_statuses.contains(status) {
            errors.push(format!("{id}: execution status {status:?} is not declared"));
        }
        if candidate["execution"]["simulator_executed"].as_bool().is_none() {
            errors.push(format!("{id}: simulator_executed must be an explicit boolean"));
        }
        if candidate["reference_outputs"]["exact_simulator_outputs_frozen"]
            .as_bool()
            .is_none()
        {
            errors.push(format!("{id}: exact_simulator_outputs_frozen must be an explicit boolean"));
        }
        if !nonempty_text(&candidate["reference_outputs"], "state") {
            errors.push(format!("{id}: reference_outputs.state is required"));
        }

        if !freeze_state.eq("FROZEN")
            && !candidate["freeze_blockers"]
                .as_array()
                .is_some_and(|blockers| !blockers.is_empty())
        {
            errors.push(format!("{id}: unfrozen case has no freeze_blockers"));
        }

        if let Some(hints) = candidate.get("source_hints_not_gold") {
            if let Some(hints) = hints.as_array() {
                for (index, hint) in hints.iter().enumerate() {
                    let hint_use = text(hint, "use").unwrap_or_default();
                    if !matches!(hint_use, "sanity_hint_only" | "broad_sanity_bound_only") {
                        errors.push(format!("{id}: source hint {index} lacks an explicit non-gold use"));
                    }
                    if !nonempty_text(hint, "qualifier") {
                        errors.push(format!("{id}: source hint {index} lacks its uncertainty qualifier"));
                    }
                }
            } else {
                errors.push(format!("{id}: source_hints_not_gold must be an array when present"));
            }
        }

        validate_frozen_receipt(candidate, &mut errors, id);
    }

    let targets = &registry["acceptance_targets"];
    for (field, expected) in [
        ("frozen_case_provenance_coverage", 1.0_f64),
        ("shared_dependency_declaration_coverage", 1.0_f64),
        ("missing_or_skipped_results_counted_as_pass", 0.0_f64),
    ] {
        let actual = targets[field]["target"].as_f64();
        if actual != Some(expected) || targets[field]["hard_gate"].as_bool() != Some(true) {
            errors.push(format!("acceptance target {field} must retain its hard-gated value {expected}"));
        }
    }
    if targets["frozen_matched_condition_cases_before_advanced_flowsheet_optimization"]["target"]
        .as_u64()
        .is_none_or(|target| target < 3)
    {
        errors.push("at least three matched-condition cases must precede advanced optimization".to_owned());
    }
    if targets["aggregate_scores_may_mask_a_failed_hard_invariant"].as_bool() != Some(false) {
        errors.push("aggregate scores must never mask a failed hard invariant".to_owned());
    }
    if !nonempty_text(&targets, "tolerance_policy") {
        errors.push("case-specific preregistered tolerance policy is required".to_owned());
    }

    errors
}

fn parse_registry_path() -> Result<String, String> {
    let mut path = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--registry" => {
                if path.is_some() {
                    return Err("--registry may only be specified once".to_owned());
                }
                path = Some(
                    args.next()
                        .ok_or_else(|| "--registry requires a path".to_owned())?,
                );
            }
            _ => return Err(format!("unknown argument {flag:?}; usage: --registry <path>")),
        }
    }
    path.ok_or_else(|| "usage: --registry <path>".to_owned())
}

fn run() -> Result<bool, Box<dyn Error>> {
    let path = parse_registry_path().map_err(|message| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
    })?;
    let bytes = fs::read(&path)?;
    let registry: Value = serde_json::from_slice(&bytes)?;
    let errors = validate_registry(&registry);
    let passed = errors.is_empty();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "validator": "chemical_process_tier_b_registry_contract_v1",
            "registry_path": path,
            "registry_bytes": bytes.len(),
            "schema_version": text(&registry, "schema_version"),
            "candidate_count": registry["candidates"].as_array().map_or(0, Vec::len),
            "passed": passed,
            "errors": errors,
            "claim_ceiling": "Registry structure only; no simulator execution or physical model validation."
        }))?
    );
    Ok(passed)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("Tier-B registry validator: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTRY: &str = include_str!(
        "../../../../docs/engineering/fixtures/CHEMICAL_PROCESS_TIER_B_CANDIDATES_V1.json"
    );

    fn registry() -> Value {
        serde_json::from_str(REGISTRY).expect("frozen Tier-B registry should parse")
    }

    #[test]
    fn current_candidate_registry_satisfies_contract() {
        let value = registry();
        assert_eq!(value["candidates"].as_array().unwrap().len(), 5);
        assert!(validate_registry(&value).is_empty(), "{:?}", validate_registry(&value));
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let mut value = registry();
        let duplicate_id = value["candidates"][0]["case_id"].clone();
        value["candidates"][1]["case_id"] = duplicate_id;
        assert!(validate_registry(&value).iter().any(|error| error.contains("duplicate candidate case_id")));
    }

    #[test]
    fn not_run_cannot_claim_execution_bytes_or_frozen_outputs() {
        let mut value = registry();
        value["candidates"][0]["execution"]["raw_output_hash"] =
            json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let errors = validate_registry(&value);
        assert!(errors.iter().any(|error| error.contains("NOT_RUN requires raw_output_hash=null")));
    }

    #[test]
    fn tutorial_hint_cannot_be_promoted_without_an_execution_receipt() {
        let mut value = registry();
        value["candidates"][0]["freeze_state"] = json!("FROZEN");
        value["candidates"][0]["execution"]["status"] = json!("PASS");
        value["candidates"][0]["execution"]["simulator_executed"] = json!(true);
        value["candidates"][0]["execution"]["simulator_version"] = json!("DWSIM pinned build");
        value["candidates"][0]["execution"]["raw_output_hash"] =
            json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        value["candidates"][0]["reference_outputs"]["exact_simulator_outputs_frozen"] = json!(true);
        value["candidates"][0]["reference_outputs"]["state"] = json!("FROZEN");
        value["candidates"][0]["reference_outputs"]["output_digest"] =
            json!("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        value["candidates"][0]["freeze_blockers"] = json!([]);
        let errors = validate_registry(&value);
        assert!(errors.iter().any(|error| error.contains("freeze_receipt")));
        assert!(!errors.is_empty());
    }

    #[test]
    fn explicit_gold_label_is_rejected_for_tutorial_hints() {
        let mut value = registry();
        value["candidates"][0]["source_hints_not_gold"][0]["use"] = json!("exact_golden_output");
        let errors = validate_registry(&value);
        assert!(
            errors.iter().any(|error| error.contains("explicit non-gold use")),
            "tutorial hints must not be relabeled as exact golden outputs"
        );
    }

    fn frozen_receipt_registry() -> Value {
        let mut value = registry();
        let candidate = &mut value["candidates"][0];
        candidate["freeze_state"] = json!("FROZEN");
        candidate["execution"]["status"] = json!("PASS");
        candidate["execution"]["simulator_executed"] = json!(true);
        candidate["execution"]["simulator_version"] = json!("DWSIM exact build pinned in receipt");
        candidate["execution"]["raw_output_hash"] =
            json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        candidate["reference_outputs"]["state"] = json!("FROZEN");
        candidate["reference_outputs"]["exact_simulator_outputs_frozen"] = json!(true);
        candidate["reference_outputs"]["output_digest"] =
            json!("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        candidate["freeze_blockers"] = json!([]);
        candidate["reference_outputs"]["freeze_receipt"] = json!({
            "primary_source_url_and_revision": "https://dwsim.org/tutorials/en/beginner/02-mixer-basics.html @ pinned revision",
            "retrieved_at": "2026-10-17",
            "native_input_files_and_transitive_data_hashes": [{
                "artifact": "mixer-project.dwxmz",
                "sha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            }],
            "component_identity_and_order": "water",
            "units_and_composition_basis": "SI; mass flow in kg/s",
            "phase_and_reference_state_conventions": "captured and documented",
            "property_package_or_mechanism_and_parameter_provenance": "pinned IAPWS-IF97 property data source",
            "solver_build_and_runtime_versions": "exact DWSIM build recorded",
            "numerical_methods_tolerances_and_convergence_status": "converged; case-specific tolerance preregistered",
            "stdout_stderr_and_raw_output_hashes": {
                "stdout_sha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "stderr_sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "raw_output_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "independent_checker_version_and_result": {
                "checker_version": "synthetic-checker-v1",
                "result": "PASS"
            },
            "shared_dependency_audit": {
                "complete": true,
                "independence_assessed": true,
                "lineage_complete": true,
                "lineage_nodes": [
                    {
                        "artifact": "model-a-property-table",
                        "sha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                        "derived_from": []
                    },
                    {
                        "artifact": "model-b-property-table",
                        "sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "derived_from": []
                    }
                ],
                "model_a_dependency_digests": [
                    "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                ],
                "model_b_dependency_digests": [
                    "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                ],
                "shared_dependency_digests": [],
                "independent_corroboration_claimed": true
            },
            "preregistered_case_specific_tolerance_and_rationale": {
                "frozen_before_candidate_evaluation": true,
                "rationale": "mass conservation is analytically exact under the declared bookkeeping assumptions",
                "metrics": [{
                    "metric": "total_mass_flow",
                    "acceptance_rule": "exact arithmetic invariant of 150 kg/s"
                }]
            },
            "claim_ceiling": "only matched-context water-mixer reference reproducibility"
        });
        value
    }

    #[test]
    fn complete_frozen_receipt_is_allowed_only_with_run_evidence() {
        let value = frozen_receipt_registry();
        assert!(
            validate_registry(&value).is_empty(),
            "complete frozen receipt should satisfy registry structure: {:?}",
            validate_registry(&value)
        );
    }

    #[test]
    fn shared_dependency_intersection_is_recomputed_not_trusted() {
        let mut value = frozen_receipt_registry();
        let audit = &mut value["candidates"][0]["reference_outputs"]["freeze_receipt"]["shared_dependency_audit"];
        audit["model_b_dependency_digests"][0] =
            json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        // A caller falsely reports an empty intersection despite identical digest sets.
        audit["shared_dependency_digests"] = json!([]);
        let errors = validate_registry(&value);
        assert!(
            errors.iter().any(|error| error.contains("do not equal the computed intersection")),
            "the checker must derive overlap from both declared manifests"
        );
    }

    #[test]
    fn shared_data_may_be_recorded_but_not_claimed_as_independent() {
        let mut value = frozen_receipt_registry();
        let audit = &mut value["candidates"][0]["reference_outputs"]["freeze_receipt"]["shared_dependency_audit"];
        audit["model_b_dependency_digests"][0] =
            json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        audit["shared_dependency_digests"] = json!([
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        ]);
        audit["independent_corroboration_claimed"] = json!(false);
        assert!(
            validate_registry(&value).is_empty(),
            "shared dependencies are admissible only when explicitly declared and not promoted: {:?}",
            validate_registry(&value)
        );
    }

    #[test]
    fn shared_data_cannot_be_promoted_to_independent_corroboration() {
        let mut value = frozen_receipt_registry();
        let audit = &mut value["candidates"][0]["reference_outputs"]["freeze_receipt"]["shared_dependency_audit"];
        audit["model_b_dependency_digests"][0] =
            json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        audit["shared_dependency_digests"] = json!([
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        ]);
        audit["independent_corroboration_claimed"] = json!(true);
        assert!(
            validate_registry(&value).iter().any(|error| error.contains("cannot be claimed")),
            "nonempty computed overlap must veto the independence claim"
        );
    }

    #[test]
    fn missing_blockers_are_rejected_for_unfrozen_cases() {
        let mut value = registry();
        value["candidates"][2]["freeze_blockers"] = json!([]);
        assert!(validate_registry(&value).iter().any(|error| error.contains("unfrozen")));
    }

    #[test]
    fn skipped_reference_cannot_be_promoted_to_pass() {
        let mut value = registry();
        value["candidates"][3]["execution"]["status"] = json!("SKIPPED");
        value["candidates"][3]["execution"]["simulator_executed"] = json!(false);
        value["candidates"][3]["reference_outputs"]["exact_simulator_outputs_frozen"] = json!(true);
        let errors = validate_registry(&value);
        assert!(errors.iter().any(|error| error.contains("frozen outputs require execution.status=PASS")));
        assert!(errors.iter().any(|error| error.contains("SKIPPED cannot become a frozen-output pass")));
    }
}
