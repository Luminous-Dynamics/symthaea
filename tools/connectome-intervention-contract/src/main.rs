//! Semantic verification for CONNECTOME-INTERVENTION-RUN-V1.
//!
//! JSON Schema validates document shape. This executable validates relationships
//! across the document: real split membership, digests, model references,
//! baseline coverage, paired score arithmetic, controls and claim ceilings.
//!
//! It intentionally does not fetch data, authenticate permission authorities,
//! or verify HMAC token derivation (the secret key must not be published).

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;

const EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_EXAMPLE_V1.json"
);
const TOLERANCE: f64 = 1.0e-8;

type MetricScoreKey = (String, String, String);
type MetricScoreRecord = (f64, f64, f64, usize, HashSet<String>);
type MetricScoreMap = HashMap<MetricScoreKey, MetricScoreRecord>;
type MetricScoreResult = (
    HashMap<String, &'static str>,
    HashMap<String, String>,
    MetricScoreMap,
);

#[derive(Debug, Clone, PartialEq)]
struct Report {
    disposition: &'static str,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Report {
    fn as_json(&self) -> Value {
        json!({
            "disposition": self.disposition,
            "errors": self.errors,
            "warnings": self.warnings,
            "biological_validation": false
        })
    }
}

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for part in path {
        cursor = cursor.get(*part)?;
    }
    Some(cursor)
}

fn str_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path)?.as_str()
}

fn arr_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Vec<Value>> {
    at(value, path)?.as_array()
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    is_lower_hex(value, 64)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// SHA256(sorted lowercase tokens, UTF-8, newline-delimited with trailing LF).
fn membership_digest(tokens: &[String]) -> String {
    let mut ordered = tokens.to_vec();
    ordered.sort_unstable();
    let mut bytes = ordered.join("\n");
    bytes.push('\n');
    sha256_hex(bytes.as_bytes())
}

fn require_sha(value: &Value, path: &[&str], label: &str, errors: &mut Vec<String>) {
    match str_at(value, path) {
        Some(digest) if is_sha256(digest) => {}
        Some(_) => errors.push(format!("{label}: expected lowercase 64-hex SHA-256")),
        None => errors.push(format!("{label}: missing SHA-256 digest")),
    }
}

fn check_digest(
    value: &Value,
    path: &[&str],
    label: &str,
    expected: &str,
    errors: &mut Vec<String>,
) {
    match str_at(value, path) {
        Some(actual) if actual == expected => {}
        Some(actual) => errors.push(format!(
            "{label}: digest mismatch; declared {actual}, recomputed {expected}"
        )),
        None => errors.push(format!("{label}: missing digest")),
    }
}

fn verify_claim_ceiling(value: &Value, errors: &mut Vec<String>) {
    let tier = str_at(value, &["evidence_tier"]);
    let ceiling = str_at(value, &["claim_ceiling"]);
    let source_kind = str_at(value, &["source", "source_kind"]);
    let expected = match tier {
        Some("synthetic_known_answer") => Some(("synthetic_only", "synthetic")),
        Some("observational_structure_function") => {
            Some(("observational_prediction_only", "observational"))
        }
        Some("in_silico_intervention") => Some(("model_counterfactual_only", "")),
        Some("measured_intervention") => {
            Some(("measured_intervention_prediction", "measured_intervention"))
        }
        _ => {
            errors.push("evidence_tier: missing or unsupported".to_owned());
            None
        }
    };
    if let Some((expected_ceiling, expected_source)) = expected {
        if ceiling != Some(expected_ceiling) {
            errors.push(format!(
                "claim_ceiling: evidence tier requires {expected_ceiling}"
            ));
        }
        if !expected_source.is_empty() && source_kind != Some(expected_source) {
            errors.push(format!(
                "source.source_kind: evidence tier requires {expected_source}"
            ));
        }
    }
    if tier == Some("measured_intervention") {
        if str_at(value, &["source", "rights_state"]) != Some("documented_permitted") {
            errors.push(
                "measured_intervention: rights_state must be documented_permitted".to_owned(),
            );
        }
        require_sha(
            value,
            &["source", "permission_receipt_sha256"],
            "source.permission_receipt_sha256",
            errors,
        );
    }
}

fn read_tokens(
    value: &Value,
    path: &[&str],
    label: &str,
    minimum: usize,
    errors: &mut Vec<String>,
) -> Vec<String> {
    let Some(items) = arr_at(value, path) else {
        errors.push(format!("{label}: missing token list"));
        return Vec::new();
    };
    if items.len() < minimum {
        errors.push(format!(
            "{label}: requires at least {minimum} tokens, found {}",
            items.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut tokens = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(token) if is_sha256(token) => {
                if !seen.insert(token.to_owned()) {
                    errors.push(format!("{label}[{index}]: duplicate token"));
                }
                tokens.push(token.to_owned());
            }
            Some(_) => errors.push(format!("{label}[{index}]: token must be lowercase 64-hex")),
            None => errors.push(format!("{label}[{index}]: token must be a string")),
        }
    }
    tokens
}

fn verify_split(value: &Value, errors: &mut Vec<String>) -> HashSet<String> {
    if str_at(value, &["split", "status"]) != Some("qualified") {
        errors.push("split.status: complete run requires qualified".to_owned());
    }
    if str_at(value, &["split", "unit_tokenization", "algorithm"]) != Some("HMAC-SHA256") {
        errors.push("split.unit_tokenization.algorithm: expected HMAC-SHA256".to_owned());
    }
    if str_at(
        value,
        &["split", "unit_tokenization", "canonicalization_version"],
    ) != Some("UNIT-TOKEN-V1")
    {
        errors.push(
            "split.unit_tokenization.canonicalization_version: expected UNIT-TOKEN-V1".to_owned(),
        );
    }
    require_sha(
        value,
        &["split", "unit_tokenization", "key_commitment_sha256"],
        "split.unit_tokenization.key_commitment_sha256",
        errors,
    );

    let train = read_tokens(
        value,
        &["split", "train_unit_tokens"],
        "split.train_unit_tokens",
        2,
        errors,
    );
    let validation = read_tokens(
        value,
        &["split", "validation_unit_tokens"],
        "split.validation_unit_tokens",
        1,
        errors,
    );
    let test = read_tokens(
        value,
        &["split", "test_unit_tokens"],
        "split.test_unit_tokens",
        2,
        errors,
    );
    let train_set: HashSet<String> = train.iter().cloned().collect();
    let validation_set: HashSet<String> = validation.iter().cloned().collect();
    let test_set: HashSet<String> = test.iter().cloned().collect();

    for token in train_set.intersection(&validation_set) {
        errors.push(format!(
            "split leakage: train/validation overlap at {token}"
        ));
    }
    for token in train_set.intersection(&test_set) {
        errors.push(format!("split leakage: train/test overlap at {token}"));
    }
    for token in validation_set.intersection(&test_set) {
        errors.push(format!("split leakage: validation/test overlap at {token}"));
    }

    check_digest(
        value,
        &["split", "train_membership_sha256"],
        "split.train_membership_sha256",
        &membership_digest(&train),
        errors,
    );
    check_digest(
        value,
        &["split", "validation_membership_sha256"],
        "split.validation_membership_sha256",
        &membership_digest(&validation),
        errors,
    );
    check_digest(
        value,
        &["split", "test_membership_sha256"],
        "split.test_membership_sha256",
        &membership_digest(&test),
        errors,
    );
    for field in [
        "preprocessing_fit_on_train_only",
        "feature_selection_fit_on_train_only",
        "model_selection_blind_to_test",
        "target_identity_split_before_preprocessing",
    ] {
        if at(value, &["split", field]).and_then(Value::as_bool) != Some(true) {
            errors.push(format!("split.{field}: must be true"));
        }
    }
    require_sha(
        value,
        &["split", "split_manifest_sha256"],
        "split.split_manifest_sha256",
        errors,
    );
    require_sha(
        value,
        &["split", "overlap_audit_sha256"],
        "split.overlap_audit_sha256",
        errors,
    );
    test_set
}

fn id_map<'a>(items: Option<&'a Vec<Value>>, id_key: &str) -> HashMap<String, &'a Value> {
    let mut result = HashMap::new();
    if let Some(items) = items {
        for item in items {
            if let (Some(id), Some(_)) =
                (item.get(id_key).and_then(Value::as_str), item.as_object())
            {
                result.insert(id.to_owned(), item);
            }
        }
    }
    result
}

fn verify_artifacts(value: &Value, errors: &mut Vec<String>) {
    for (path, label) in [
        (
            &["source", "source_manifest_sha256"][..],
            "source.source_manifest_sha256",
        ),
        (
            &["connectome", "canonical_identity_sha256"][..],
            "connectome.canonical_identity_sha256",
        ),
        (
            &["connectome", "structural_edge_manifest_sha256"][..],
            "connectome.structural_edge_manifest_sha256",
        ),
        (
            &["observation_model", "observation_kernel_parameters_sha256"][..],
            "observation_model.observation_kernel_parameters_sha256",
        ),
        (
            &["observation_model", "preprocessing_manifest_sha256"][..],
            "observation_model.preprocessing_manifest_sha256",
        ),
        (
            &["artifacts", "run_receipt_sha256"][..],
            "artifacts.run_receipt_sha256",
        ),
        (
            &["artifacts", "resolved_config_sha256"][..],
            "artifacts.resolved_config_sha256",
        ),
        (
            &["artifacts", "environment_sha256"][..],
            "artifacts.environment_sha256",
        ),
        (
            &["artifacts", "raw_output_manifest_sha256"][..],
            "artifacts.raw_output_manifest_sha256",
        ),
    ] {
        require_sha(value, path, label, errors);
    }
}

fn verify_models<'a>(
    value: &'a Value,
    errors: &mut Vec<String>,
) -> (HashMap<String, &'a Value>, HashSet<String>) {
    let hypothesis_items = arr_at(value, &["hypothesis_set"]);
    let hypotheses = id_map(hypothesis_items, "hypothesis_id");
    if hypotheses.len() < 2 || hypothesis_items.is_none_or(|items| items.len() != hypotheses.len())
    {
        errors.push(
            "hypothesis_set: requires at least two uniquely identified hypotheses".to_owned(),
        );
    }
    let mut prior_sum = 0.0;
    if let Some(items) = hypothesis_items {
        for (index, hypothesis) in items.iter().enumerate() {
            match hypothesis.get("prior_probability").and_then(Value::as_f64) {
                Some(p) if p.is_finite() && (0.0..=1.0).contains(&p) => prior_sum += p,
                _ => errors.push(format!("hypothesis_set[{index}].prior_probability: complete run requires a finite probability in [0,1]")),
            }
            if hypothesis
                .get("frozen_before_test")
                .and_then(Value::as_bool)
                != Some(true)
            {
                errors.push(format!(
                    "hypothesis_set[{index}].frozen_before_test: must be true"
                ));
            }
            if hypothesis
                .get("prior_source")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                errors.push(format!("hypothesis_set[{index}].prior_source: missing"));
            }
            if hypothesis
                .get("prediction_manifest_sha256")
                .and_then(Value::as_str)
                .is_none_or(|s| !is_sha256(s))
            {
                errors.push(format!(
                    "hypothesis_set[{index}].prediction_manifest_sha256: missing/malformed"
                ));
            }
        }
    }
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > 1.0e-9 {
        errors.push(format!(
            "hypothesis_set: prior probabilities sum to {prior_sum}, expected 1"
        ));
    }
    let model_items = arr_at(value, &["models"]);
    let models = id_map(model_items, "model_id");
    if models.len() < 4 || model_items.is_none_or(|items| items.len() != models.len()) {
        errors.push("models: require at least four uniquely identified model entries".to_owned());
    }

    let roles: HashSet<&str> = models
        .values()
        .filter_map(|model| model.get("role").and_then(Value::as_str))
        .collect();
    for role in ["candidate", "baseline", "negative_control"] {
        if !roles.contains(role) {
            errors.push(format!("models: missing required role {role}"));
        }
    }

    let mut families = HashSet::new();
    let mut training_scopes = HashSet::new();
    let mut budget_signatures = HashSet::new();
    for (model_id, model) in &models {
        let role = model.get("role").and_then(Value::as_str).unwrap_or("");
        if let Some(family) = model.get("model_family").and_then(Value::as_str) {
            families.insert(family.to_owned());
        } else {
            errors.push(format!("models.{model_id}.model_family: missing"));
        }
        if let Some(hypothesis_id) = model.get("hypothesis_id").and_then(Value::as_str)
            && !hypotheses.contains_key(hypothesis_id)
        {
            errors.push(format!(
                "models.{model_id}.hypothesis_id: unresolved {hypothesis_id}"
            ));
        }
        if model.get("observation_model_id").and_then(Value::as_str)
            != str_at(value, &["observation_model", "observation_model_id"])
        {
            errors.push(format!("models.{model_id}.observation_model_id: mismatch"));
        }
        if role == "candidate" || role == "baseline" || role == "negative_control" {
            match model
                .get("training_data_scope_sha256")
                .and_then(Value::as_str)
            {
                Some(hash) if is_sha256(hash) => {
                    training_scopes.insert(hash.to_owned());
                }
                _ => errors.push(format!(
                    "models.{model_id}.training_data_scope_sha256: missing/malformed"
                )),
            }
            let budget = model.get("training_budget").unwrap_or(&Value::Null);
            let signature = json!([
                budget.get("compute_units"),
                budget.get("wall_time_seconds"),
                budget.get("energy_joules"),
                budget.get("memory_bytes"),
                budget.get("training_samples")
            ])
            .to_string();
            budget_signatures.insert(signature);
            let has_resource = [
                "compute_units",
                "wall_time_seconds",
                "energy_joules",
                "memory_bytes",
                "training_samples",
            ]
            .iter()
            .any(|key| budget.get(*key).is_some_and(|v| !v.is_null()));
            if !has_resource {
                errors.push(format!(
                    "models.{model_id}.training_budget: no declared resource field"
                ));
            }
        }
        let code = model.get("code_commit_sha").and_then(Value::as_str);
        if code.is_none_or(|s| s.len() != 40 || !s.bytes().all(|b| b.is_ascii_hexdigit())) {
            errors.push(format!(
                "models.{model_id}.code_commit_sha: missing/malformed"
            ));
        }
        for field in ["config_sha256", "model_artifact_sha256"] {
            let valid = model
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(is_sha256);
            if !valid {
                errors.push(format!("models.{model_id}.{field}: missing/malformed"));
            }
        }
    }

    if str_at(value, &["budgets", "comparison_mode"]) == Some("matched_budget") {
        if str_at(value, &["budgets", "training_budget_policy"]) != Some("matched")
            || str_at(value, &["budgets", "evaluation_budget_policy"]) != Some("matched")
        {
            errors.push(
                "budgets: matched comparison requires matched training/evaluation policies"
                    .to_owned(),
            );
        }
        if training_scopes.len() > 1 {
            errors
                .push("budgets: matched comparison uses different training-data scopes".to_owned());
        }
        if budget_signatures.len() > 1 {
            errors.push(
                "budgets: matched comparison uses different training resource budgets".to_owned(),
            );
        }
    } else if str_at(value, &["budgets", "comparison_mode"])
        != Some("declared_unequal_budget_tradeoff")
    {
        errors.push("budgets.comparison_mode: complete run cannot be not_comparable".to_owned());
    }

    let mut unavailable = HashMap::new();
    if let Some(items) = arr_at(
        value,
        &["evaluation", "baseline_matrix", "unavailable_families"],
    ) {
        for item in items {
            if let (Some(family), Some(reason)) = (
                item.get("family").and_then(Value::as_str),
                item.get("reason").and_then(Value::as_str),
            ) {
                if reason.trim().is_empty() {
                    errors.push(format!(
                        "baseline_matrix.unavailable_families.{family}: empty reason"
                    ));
                } else {
                    unavailable.insert(family.to_owned(), reason.to_owned());
                }
            }
        }
    }
    let required = arr_at(
        value,
        &["evaluation", "baseline_matrix", "required_families"],
    );
    if let Some(required) = required {
        let mut seen = HashSet::new();
        if required.len() < 5 {
            errors.push(
                "baseline_matrix.required_families: requires at least five families".to_owned(),
            );
        }
        for family in required.iter().filter_map(Value::as_str) {
            if !seen.insert(family.to_owned()) {
                errors.push(format!(
                    "baseline_matrix: duplicate required family {family}"
                ));
            }
            if !families.contains(family) && !unavailable.contains_key(family) {
                errors.push(format!(
                    "baseline_matrix: {family} is neither implemented nor explicitly unavailable"
                ));
            }
            if families.contains(family) && unavailable.contains_key(family) {
                errors.push(format!(
                    "baseline_matrix: {family} is both implemented and unavailable"
                ));
            }
        }
    } else {
        errors.push("baseline_matrix.required_families: missing".to_owned());
    }
    require_sha(
        value,
        &["evaluation", "baseline_matrix", "coverage_sha256"],
        "baseline_matrix.coverage_sha256",
        errors,
    );
    (models, families)
}

fn verify_interventions(value: &Value, model_families: &HashSet<String>, errors: &mut Vec<String>) {
    if at(value, &["intervention_design", "preregistered_before_test"]).and_then(Value::as_bool)
        != Some(true)
    {
        errors.push("intervention_design.preregistered_before_test: must be true".to_owned());
    }
    if str_at(value, &["intervention_design", "selector"]) == Some("not_selected") {
        errors.push("intervention_design.selector: complete run requires a selector".to_owned());
    }
    require_sha(
        value,
        &["intervention_design", "candidate_pool_sha256"],
        "intervention_design.candidate_pool_sha256",
        errors,
    );
    require_sha(
        value,
        &["intervention_design", "feasibility_audit_sha256"],
        "intervention_design.feasibility_audit_sha256",
        errors,
    );
    let selected = arr_at(value, &["intervention_design", "selected_interventions"]);
    if selected.is_none_or(Vec::is_empty) {
        errors.push(
            "intervention_design.selected_interventions: at least one is required".to_owned(),
        );
    }
    let mut ids = HashSet::new();
    let mut control_roles = HashSet::new();
    let mut has_primary_intervention = false;
    if let Some(items) = selected {
        for (index, item) in items.iter().enumerate() {
            let id = item.get("intervention_id").and_then(Value::as_str);
            if id.is_none_or(str::is_empty) {
                errors.push(format!("selected_interventions[{index}]: missing ID"));
            } else if !ids.insert(id.unwrap().to_owned()) {
                errors.push(format!("selected_interventions[{index}]: duplicate ID"));
            }
            if item.get("feasibility_status").and_then(Value::as_str) != Some("feasible") {
                errors.push(format!(
                    "selected_interventions[{index}]: not marked feasible"
                ));
            }
            match item
                .get("control_role")
                .and_then(Value::as_str)
                .unwrap_or("")
            {
                "intervention" => has_primary_intervention = true,
                role @ ("sham"
                | "single_target_control"
                | "combination_control"
                | "washout"
                | "other") => {
                    control_roles.insert(role.to_owned());
                }
                _ => errors.push(format!(
                    "selected_interventions[{index}]: unknown control role"
                )),
            }
            if item
                .get("safety_constraints_sha256")
                .and_then(Value::as_str)
                .is_none_or(|s| !is_sha256(s))
            {
                errors.push(format!(
                    "selected_interventions[{index}]: missing safety-constraint digest"
                ));
            }
            let targets = item.get("target_ids").and_then(Value::as_array);
            if targets.is_none_or(Vec::is_empty) {
                errors.push(format!("selected_interventions[{index}]: no target IDs"));
            }
            if item
                .get("protocol_sha256")
                .and_then(Value::as_str)
                .is_none_or(|s| !is_sha256(s))
            {
                errors.push(format!(
                    "selected_interventions[{index}]: missing protocol digest"
                ));
            }
            let protocol_semantics = item.get("protocol_semantics").and_then(Value::as_str);
            let protocol_steps = item.get("protocol_steps").and_then(Value::as_array);
            if protocol_semantics == Some("ordered_steps")
                && protocol_steps.is_none_or(|steps| steps.len() < 2)
            {
                errors.push(format!(
                    "selected_interventions[{index}]: ordered_steps requires at least two protocol steps"
                ));
            }
            if protocol_steps.is_some() && protocol_semantics != Some("ordered_steps") {
                errors.push(format!(
                    "selected_interventions[{index}]: protocol_steps require protocol_semantics ordered_steps"
                ));
            }
            if let Some(steps) = protocol_steps {
                let mut ordered_targets = HashSet::new();
                for (step_index, step) in steps.iter().enumerate() {
                    if step.get("step_index").and_then(Value::as_u64) != Some(step_index as u64) {
                        errors.push(format!(
                            "selected_interventions[{index}].protocol_steps[{step_index}]: step indices must be contiguous and zero-based"
                        ));
                    }
                    let targets = step.get("target_ids").and_then(Value::as_array);
                    if targets.is_none_or(Vec::is_empty) {
                        errors.push(format!(
                            "selected_interventions[{index}].protocol_steps[{step_index}]: target list is empty"
                        ));
                    }
                    if let Some(targets) = targets {
                        let mut step_targets = HashSet::new();
                        for target in targets.iter().filter_map(Value::as_str) {
                            if !step_targets.insert(target.to_owned()) {
                                errors.push(format!(
                                    "selected_interventions[{index}].protocol_steps[{step_index}]: duplicate target {target}"
                                ));
                            }
                            ordered_targets.insert(target.to_owned());
                        }
                    }
                    match step.get("inter_step_interval_seconds").and_then(Value::as_f64) {
                        Some(interval) if interval.is_finite() && interval >= 0.0 => {}
                        _ => errors.push(format!(
                            "selected_interventions[{index}].protocol_steps[{step_index}]: invalid inter-step interval"
                        )),
                    }
                }
                let declared_targets: HashSet<String> = item.get("target_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect();
                if declared_targets != ordered_targets {
                    errors.push(format!(
                        "selected_interventions[{index}]: parent target_ids must equal the union of ordered protocol-step targets"
                    ));
                }
            }
            let state_context_id = item.get("state_context_id").and_then(Value::as_str);
            let state_digest = item.get("state_measurement_manifest_sha256").and_then(Value::as_str);
            let has_step_state = protocol_steps.into_iter().flatten()
                .any(|step| step.get("state_context_id").and_then(Value::as_str).is_some());
            if state_context_id.is_some() || has_step_state {
                if state_digest.is_none_or(|digest| !is_sha256(digest)) {
                    errors.push(format!(
                        "selected_interventions[{index}]: state-conditioned protocol requires a state-measurement manifest digest"
                    ));
                }
            } else if state_digest.is_some() {
                errors.push(format!(
                    "selected_interventions[{index}]: state-measurement digest has no declared state context"
                ));
            }
        }
    }
    let controls: HashSet<&str> = arr_at(value, &["intervention_design", "required_controls"])
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    for control in [
        "sham",
        "single_target",
        "combined_target",
        "rewired_graph",
        "activity_only",
    ] {
        if !controls.contains(control) {
            errors.push(format!(
                "intervention_design.required_controls: missing {control}"
            ));
        }
    }
    if !has_primary_intervention {
        errors.push("selected_interventions: no entry has control_role intervention".to_owned());
    }
    for (declared, actual) in [
        ("sham", "sham"),
        ("single_target", "single_target_control"),
        ("combined_target", "combination_control"),
    ] {
        if controls.contains(declared) && !control_roles.contains(actual) {
            errors.push(format!("selected_interventions: declared {declared} control has no materialized {actual} entry"));
        }
    }
    if controls.contains("activity_only") && !model_families.contains("activity_only") {
        errors.push("required activity_only control has no activity_only model family".to_owned());
    }
    if controls.contains("rewired_graph")
        && !model_families.contains("degree_type_rewired")
        && !model_families.contains("spatial_rewired")
    {
        errors.push("required rewired_graph control has no rewired model family".to_owned());
    }
}

fn verify_metrics_and_scores(
    value: &Value,
    models: &HashMap<String, &Value>,
    test_set: &HashSet<String>,
    errors: &mut Vec<String>,
) -> MetricScoreResult {
    let mut directions = HashMap::new();
    let mut metric_families = HashMap::new();
    if let Some(metrics) = arr_at(value, &["evaluation", "metrics"]) {
        for metric in metrics {
            if let (Some(id), Some(direction)) = (
                metric.get("metric_id").and_then(Value::as_str),
                metric.get("direction").and_then(Value::as_str),
            ) {
                let dir = match direction {
                    "higher_is_better" => "higher",
                    "lower_is_better" => "lower",
                    "target_range" => "target",
                    "descriptive_only" => "descriptive",
                    _ => "unknown",
                };
                if directions.insert(id.to_owned(), dir).is_some() {
                    errors.push(format!("evaluation.metrics: duplicate metric {id}"));
                }
                if let Some(family) = metric.get("metric_family").and_then(Value::as_str) {
                    metric_families.insert(id.to_owned(), family.to_owned());
                } else {
                    errors.push(format!("evaluation.metrics.{id}.metric_family: missing"));
                }
            }
        }
    }
    let primary: HashSet<String> = arr_at(value, &["evaluation", "primary_metric_ids"])
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    for metric in &primary {
        if !directions.contains_key(metric) {
            errors.push(format!(
                "evaluation.primary_metric_ids: unresolved metric {metric}"
            ));
        }
    }
    let strata: HashSet<String> = arr_at(value, &["evaluation", "strata"])
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let mut scores = HashMap::new();
    let mut score_count = 0;
    if let Some(items) = arr_at(value, &["results", "scorecards"]) {
        for (index, item) in items.iter().enumerate() {
            score_count += 1;
            let model = item.get("model_id").and_then(Value::as_str).unwrap_or("");
            let metric = item.get("metric_id").and_then(Value::as_str).unwrap_or("");
            let stratum = item.get("stratum").and_then(Value::as_str).unwrap_or("");
            if !models.contains_key(model) {
                errors.push(format!("scorecards[{index}].model_id: unresolved {model}"));
            }
            if !directions.contains_key(metric) {
                errors.push(format!(
                    "scorecards[{index}].metric_id: unresolved {metric}"
                ));
            }
            if !strata.contains(stratum) {
                errors.push(format!("scorecards[{index}].stratum: undeclared {stratum}"));
            }
            let state = item
                .get("score_state")
                .and_then(Value::as_str)
                .unwrap_or("");
            let score = item.get("score").and_then(Value::as_f64);
            let low = item.get("ci_low").and_then(Value::as_f64);
            let high = item.get("ci_high").and_then(Value::as_f64);
            let key = (model.to_owned(), metric.to_owned(), stratum.to_owned());
            let raw_units = item.get("unit_tokens").and_then(Value::as_array);
            let mut unit_tokens = HashSet::new();
            let mut token_values = Vec::new();
            if let Some(raw_units) = raw_units {
                for (unit_index, raw_unit) in raw_units.iter().enumerate() {
                    match raw_unit.as_str() {
                        Some(token) if is_sha256(token) => {
                            if !unit_tokens.insert(token.to_owned()) {
                                errors.push(format!("scorecards[{index}].unit_tokens[{unit_index}]: duplicate unit token"));
                            }
                            if !test_set.contains(token) {
                                errors.push(format!("scorecards[{index}].unit_tokens[{unit_index}]: token is outside held-out test membership"));
                            }
                            token_values.push(token.to_owned());
                        }
                        Some(_) => errors.push(format!(
                            "scorecards[{index}].unit_tokens[{unit_index}]: malformed token"
                        )),
                        None => errors.push(format!(
                            "scorecards[{index}].unit_tokens[{unit_index}]: token must be a string"
                        )),
                    }
                }
            } else {
                errors.push(format!("scorecards[{index}].unit_tokens: missing"));
            }
            if !token_values.is_empty() {
                check_digest(
                    item,
                    &["unit_manifest_sha256"],
                    &format!("scorecards[{index}].unit_manifest_sha256"),
                    &membership_digest(&token_values),
                    errors,
                );
            }
            match state {
                "measured" => {
                    let declared_n = item
                        .get("n_independent_units")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    if token_values.len() < 2 || declared_n != token_values.len() {
                        errors.push(format!("scorecards[{index}]: measured score token count must equal n_independent_units and be at least two"));
                    }
                    if score.is_none_or(|x| !x.is_finite())
                        || low.is_none_or(|x| !x.is_finite())
                        || high.is_none_or(|x| !x.is_finite())
                    {
                        errors.push(format!(
                            "scorecards[{index}]: measured score requires finite score and CI"
                        ));
                    } else if low.unwrap() > high.unwrap()
                        || score.unwrap() < low.unwrap()
                        || score.unwrap() > high.unwrap()
                    {
                        errors.push(format!(
                            "scorecards[{index}]: score/CI interval is inconsistent"
                        ));
                    } else if scores
                        .insert(
                            key,
                            (
                                score.unwrap(),
                                low.unwrap(),
                                high.unwrap(),
                                declared_n,
                                unit_tokens,
                            ),
                        )
                        .is_some()
                    {
                        errors.push(format!(
                            "scorecards[{index}]: duplicate model/metric/stratum"
                        ));
                    }
                }
                "abstained" | "failed" | "not_assessed" | "incomplete" => {
                    if score.is_some() || low.is_some() || high.is_some() {
                        errors.push(format!("scorecards[{index}]: non-measured state cannot carry a numeric score/CI"));
                    }
                }
                _ => errors.push(format!(
                    "scorecards[{index}].score_state: unsupported state"
                )),
            }
        }
    }
    if score_count == 0 {
        errors.push("results.scorecards: at least one scorecard is required".to_owned());
    }
    let required_families: HashSet<String> = arr_at(
        value,
        &["evaluation", "baseline_matrix", "required_families"],
    )
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .map(str::to_owned)
    .collect();
    for (model_id, model) in models {
        let family = model
            .get("model_family")
            .and_then(Value::as_str)
            .unwrap_or("");
        if required_families.contains(family) {
            for metric in &primary {
                for stratum in &strata {
                    if !scores.contains_key(&(model_id.clone(), metric.clone(), stratum.clone())) {
                        errors.push(format!("required baseline/candidate model {model_id} ({family}) lacks measured scorecard for {metric}/{stratum}"));
                    }
                }
            }
        }
    }
    for metric in &primary {
        for stratum in &strata {
            let has_candidate = scores.keys().any(|(model_id, metric_id, stratum_id)| {
                metric_id == metric
                    && stratum_id == stratum
                    && models
                        .get(model_id)
                        .and_then(|m| m.get("role"))
                        .and_then(Value::as_str)
                        == Some("candidate")
            });
            let has_baseline = scores.keys().any(|(model_id, metric_id, stratum_id)| {
                metric_id == metric
                    && stratum_id == stratum
                    && models
                        .get(model_id)
                        .and_then(|m| m.get("role"))
                        .and_then(Value::as_str)
                        == Some("baseline")
            });
            if !has_candidate {
                errors.push(format!("primary metric {metric}, stratum {stratum}: missing measured candidate scorecard"));
            }
            if !has_baseline {
                errors.push(format!("primary metric {metric}, stratum {stratum}: missing measured baseline scorecard"));
            }
        }
    }
    (directions, metric_families, scores)
}

fn verify_paired(
    value: &Value,
    models: &HashMap<String, &Value>,
    directions: &HashMap<String, &'static str>,
    scores: &MetricScoreMap,
    test_set: &HashSet<String>,
    errors: &mut Vec<String>,
) {
    let mut count = 0;
    let mut paired_keys = HashSet::new();
    if let Some(comparisons) = arr_at(value, &["results", "paired_comparisons"]) {
        for (index, item) in comparisons.iter().enumerate() {
            count += 1;
            let candidate = item
                .get("candidate_model_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let baseline = item
                .get("baseline_model_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let metric = item.get("metric_id").and_then(Value::as_str).unwrap_or("");
            let stratum = item.get("stratum").and_then(Value::as_str).unwrap_or("");
            if !paired_keys.insert((
                candidate.to_owned(),
                baseline.to_owned(),
                metric.to_owned(),
                stratum.to_owned(),
            )) {
                errors.push(format!("paired_comparisons[{index}]: duplicate candidate/baseline/metric/stratum comparison"));
            }
            if models
                .get(candidate)
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                != Some("candidate")
            {
                errors.push(format!(
                    "paired_comparisons[{index}]: candidate model ref/role invalid"
                ));
            }
            let baseline_role = models
                .get(baseline)
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str);
            if !matches!(
                baseline_role,
                Some("baseline" | "negative_control" | "oracle_reference")
            ) {
                errors.push(format!(
                    "paired_comparisons[{index}]: baseline model ref/role invalid"
                ));
            }
            let candidate_score =
                scores.get(&(candidate.to_owned(), metric.to_owned(), stratum.to_owned()));
            let baseline_score =
                scores.get(&(baseline.to_owned(), metric.to_owned(), stratum.to_owned()));
            let (Some(candidate_score), Some(baseline_score)) = (candidate_score, baseline_score)
            else {
                errors.push(format!(
                    "paired_comparisons[{index}]: paired candidate/baseline scorecards missing"
                ));
                continue;
            };
            match item
                .get("mean_candidate_minus_baseline")
                .and_then(Value::as_f64)
            {
                Some(observed) if observed.is_finite() => {
                    let expected = candidate_score.0 - baseline_score.0;
                    if (observed - expected).abs() > TOLERANCE * expected.abs().max(1.0) {
                        errors.push(format!("paired_comparisons[{index}]: mean delta does not reconcile with scorecards"));
                    }
                }
                _ => errors.push(format!(
                    "paired_comparisons[{index}]: missing finite mean delta"
                )),
            }
            let low = item.get("ci_low").and_then(Value::as_f64);
            let high = item.get("ci_high").and_then(Value::as_f64);
            if low.is_none_or(|x| !x.is_finite())
                || high.is_none_or(|x| !x.is_finite())
                || low.unwrap() > high.unwrap()
            {
                errors.push(format!(
                    "paired_comparisons[{index}]: CI missing, non-finite or reversed"
                ));
            }
            let tokens = item.get("paired_unit_tokens").and_then(Value::as_array);
            let mut token_values = Vec::new();
            let mut unique = HashSet::new();
            if let Some(tokens) = tokens {
                for token_value in tokens {
                    if let Some(token) = token_value.as_str() {
                        if !is_sha256(token) {
                            errors
                                .push(format!("paired_comparisons[{index}]: malformed unit token"));
                        }
                        if !unique.insert(token.to_owned()) {
                            errors.push(format!(
                                "paired_comparisons[{index}]: duplicate paired unit token"
                            ));
                        }
                        if !test_set.contains(token) {
                            errors.push(format!("paired_comparisons[{index}]: unit token is outside held-out test membership"));
                        }
                        token_values.push(token.to_owned());
                    } else {
                        errors.push(format!(
                            "paired_comparisons[{index}]: unit token must be string"
                        ));
                    }
                }
            } else {
                errors.push(format!(
                    "paired_comparisons[{index}]: missing paired_unit_tokens"
                ));
            }
            if token_values.len() < 2 {
                errors.push(format!(
                    "paired_comparisons[{index}]: requires at least two paired units"
                ));
            }
            check_digest(
                item,
                &["paired_unit_manifest_sha256"],
                &format!("paired_comparisons[{index}].paired_unit_manifest_sha256"),
                &membership_digest(&token_values),
                errors,
            );
            if item.get("n_independent_units").and_then(Value::as_u64)
                != Some(token_values.len() as u64)
            {
                errors.push(format!("paired_comparisons[{index}]: declared independent count differs from token count"));
            }
            if candidate_score.3 != token_values.len() || baseline_score.3 != token_values.len() {
                errors.push(format!("paired_comparisons[{index}]: paired token count must equal both scorecard independent-unit counts"));
            }
            if candidate_score.4 != baseline_score.4 || candidate_score.4 != unique {
                errors.push(format!("paired_comparisons[{index}]: paired units must exactly match the candidate and baseline scorecard cohorts"));
            }

            let outcome = item.get("outcome").and_then(Value::as_str).unwrap_or("");
            let direction = directions.get(metric).copied().unwrap_or("unknown");
            let min_effect = item
                .get("minimum_meaningful_effect")
                .and_then(Value::as_f64);
            let margin = item.get("non_inferiority_margin").and_then(Value::as_f64);
            if let (Some(lo), Some(hi)) = (low, high) {
                match (outcome, direction, min_effect, margin) {
                    ("outperforms_by_margin", "higher", Some(effect), _) if lo <= effect =>
                        errors.push(format!("paired_comparisons[{index}]: CI does not clear positive effect margin")),
                    ("outperforms_by_margin", "lower", Some(effect), _) if hi >= -effect =>
                        errors.push(format!("paired_comparisons[{index}]: CI does not clear lower-is-better effect margin")),
                    ("non_inferior", "higher", _, Some(limit)) if lo < -limit =>
                        errors.push(format!("paired_comparisons[{index}]: CI breaches non-inferiority margin")),
                    ("non_inferior", "lower", _, Some(limit)) if hi > limit =>
                        errors.push(format!("paired_comparisons[{index}]: CI breaches non-inferiority margin")),
                    ("regresses_beyond_margin", "higher", _, Some(limit)) if hi >= -limit =>
                        errors.push(format!("paired_comparisons[{index}]: CI does not establish regression")),
                    ("regresses_beyond_margin", "lower", _, Some(limit)) if lo <= limit =>
                        errors.push(format!("paired_comparisons[{index}]: CI does not establish regression")),
                    _ => {}
                }
            }
        }
    }
    if count == 0 {
        errors
            .push("results.paired_comparisons: at least one paired comparison required".to_owned());
    }
    let candidates: Vec<String> = models
        .iter()
        .filter(|(_, model)| model.get("role").and_then(Value::as_str) == Some("candidate"))
        .map(|(id, _)| id.clone())
        .collect();
    let baselines: Vec<String> = models
        .iter()
        .filter(|(_, model)| {
            matches!(
                model.get("role").and_then(Value::as_str),
                Some("baseline" | "negative_control" | "oracle_reference")
            )
        })
        .map(|(id, _)| id.clone())
        .collect();
    let metrics: Vec<String> = arr_at(value, &["evaluation", "primary_metric_ids"])
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let strata: Vec<String> = arr_at(value, &["evaluation", "strata"])
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    for candidate in &candidates {
        for baseline in &baselines {
            for metric in &metrics {
                for stratum in &strata {
                    if !paired_keys.contains(&(
                        candidate.clone(),
                        baseline.clone(),
                        metric.clone(),
                        stratum.clone(),
                    )) {
                        errors.push(format!("paired_comparisons: missing paired comparison {candidate} vs {baseline} for {metric}/{stratum}"));
                    }
                }
            }
        }
    }
}

fn verify_calibration(
    value: &Value,
    models: &HashMap<String, &Value>,
    metric_families: &HashMap<String, String>,
    errors: &mut Vec<String>,
) {
    let mut count = 0;
    if let Some(results) = arr_at(value, &["results", "calibration"]) {
        for (index, item) in results.iter().enumerate() {
            count += 1;
            let model = item.get("model_id").and_then(Value::as_str).unwrap_or("");
            let metric = item.get("metric_id").and_then(Value::as_str).unwrap_or("");
            if !models.contains_key(model) {
                errors.push(format!("calibration[{index}].model_id: unresolved"));
            }
            if !metric_families.contains_key(metric) {
                errors.push(format!("calibration[{index}].metric_id: unresolved"));
            } else if metric_families.get(metric).map(String::as_str)
                != Some("predictive_interval_coverage")
            {
                errors.push(format!(
                    "calibration[{index}].metric_id: metric is not predictive_interval_coverage"
                ));
            }
            for field in ["nominal_coverage", "observed_coverage"] {
                match item.get(field).and_then(Value::as_f64) {
                    Some(x) if (0.0..=1.0).contains(&x) => {}
                    _ => errors.push(format!(
                        "calibration[{index}].{field}: expected a proportion in [0,1]"
                    )),
                }
            }
            if item
                .get("interval_sharpness")
                .and_then(Value::as_f64)
                .is_none_or(|x| !x.is_finite() || x < 0.0)
            {
                errors.push(format!("calibration[{index}].interval_sharpness: invalid"));
            }
            if item
                .get("n_independent_units")
                .and_then(Value::as_u64)
                .is_none_or(|n| n < 2)
            {
                errors.push(format!(
                    "calibration[{index}]: requires at least two independent units"
                ));
            }
        }
    }
    if count == 0 {
        errors.push("results.calibration: at least one calibration result required".to_owned());
    }
}

fn verify_failures(value: &Value, models: &HashMap<String, &Value>, errors: &mut Vec<String>) {
    let intervention_ids: HashSet<String> =
        arr_at(value, &["intervention_design", "selected_interventions"])
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("intervention_id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
    if let Some(failures) = arr_at(value, &["results", "failures"]) {
        for (index, failure) in failures.iter().enumerate() {
            if let Some(model_id) = failure.get("model_id").and_then(Value::as_str)
                && !models.contains_key(model_id)
            {
                errors.push(format!("results.failures[{index}].model_id: unresolved"));
            }
            if let Some(intervention_id) = failure.get("intervention_id").and_then(Value::as_str)
                && !intervention_ids.contains(intervention_id)
            {
                errors.push(format!(
                    "results.failures[{index}].intervention_id: unresolved"
                ));
            }
            if failure
                .get("count")
                .and_then(Value::as_u64)
                .is_none_or(|count| count == 0)
            {
                errors.push(format!("results.failures[{index}].count: must be positive"));
            }
        }
    }
}

fn verify_complete(value: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    verify_claim_ceiling(value, &mut errors);
    verify_artifacts(value, &mut errors);
    let test_set = verify_split(value, &mut errors);
    if str_at(value, &["results", "status"]) != Some("complete") {
        errors
            .push("results.status: run_state complete requires results.status complete".to_owned());
    }
    let (models, families) = verify_models(value, &mut errors);
    verify_interventions(value, &families, &mut errors);
    if str_at(value, &["budgets", "comparison_mode"]) == Some("matched_budget")
        && (str_at(value, &["budgets", "training_budget_policy"]) != Some("matched")
            || str_at(value, &["budgets", "evaluation_budget_policy"]) != Some("matched"))
    {
        errors.push(
            "budgets: matched comparison requires matched training/evaluation policies".to_owned(),
        );
    }
    if str_at(value, &["evaluation", "uncertainty", "method_id"]) == Some("none_not_eligible") {
        errors
            .push("evaluation.uncertainty: complete run requires an uncertainty method".to_owned());
    }
    if at(
        value,
        &["evaluation", "uncertainty", "minimum_independent_units"],
    )
    .and_then(Value::as_u64)
    .is_none_or(|n| n < 2)
    {
        errors.push(
            "evaluation.uncertainty.minimum_independent_units: requires at least two".to_owned(),
        );
    }
    let (directions, metric_families, scores) =
        verify_metrics_and_scores(value, &models, &test_set, &mut errors);
    verify_paired(value, &models, &directions, &scores, &test_set, &mut errors);
    verify_calibration(value, &models, &metric_families, &mut errors);
    verify_failures(value, &models, &mut errors);
    errors
}

fn verify_manifest(value: &Value) -> Report {
    if str_at(value, &["schema_version"]) != Some("CONNECTOME-INTERVENTION-RUN-V1") {
        return Report {
            disposition: "INVALID",
            errors: vec!["schema_version: unsupported".to_owned()],
            warnings: Vec::new(),
        };
    }
    let mut claim_errors = Vec::new();
    verify_claim_ceiling(value, &mut claim_errors);
    if !claim_errors.is_empty() {
        return Report {
            disposition: "INVALID",
            errors: claim_errors,
            warnings: Vec::new(),
        };
    }
    match str_at(value, &["run_state"]) {
        Some("incomplete" | "planned" | "running") => Report {
            disposition: "INCOMPLETE",
            errors: Vec::new(),
            warnings: vec![
                "Planning/incomplete manifests are not completion passes or biological results."
                    .to_owned(),
            ],
        },
        Some("failed" | "invalid") => Report {
            disposition: "NOT_COMPLETE",
            errors: Vec::new(),
            warnings: vec!["Manifest explicitly records a failed or invalid run.".to_owned()],
        },
        Some("complete") => {
            let mut errors = verify_complete(value);
            errors.sort();
            errors.dedup();
            if errors.is_empty() {
                Report {
                    disposition: "COMPLETE",
                    errors,
                    warnings: vec![
                        "Semantic contract passed; biological validity is not established."
                            .to_owned(),
                    ],
                }
            } else {
                Report {
                    disposition: "INVALID",
                    errors,
                    warnings: Vec::new(),
                }
            }
        }
        _ => Report {
            disposition: "INVALID",
            errors: vec!["run_state: missing or unsupported".to_owned()],
            warnings: Vec::new(),
        },
    }
}

fn main() {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(EXAMPLE));
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read {}: {error}", path.display());
            std::process::exit(2);
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid JSON in {}: {error}", path.display());
            std::process::exit(2);
        }
    };
    let report = verify_manifest(&value);
    println!("{}", report.as_json());
    if report.disposition == "INVALID" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn token(n: u8) -> String {
        format!("{:064x}", n)
    }

    fn fixture_intervention(id: &str, role: &str, targets: &[&str]) -> Value {
        json!({
            "intervention_id":id,
            "target_ids":targets,
            "target_namespace":"synthetic-v1",
            "manipulation_type":"simulated_ablation",
            "waveform":{
                "shape":"not_applicable","amplitude":null,"amplitude_unit":null,
                "duration_seconds":0.0,"onset_seconds":0.0,"frequency_hz":null,
                "duty_cycle":null,"waveform_sha256":null
            },
            "protocol_sha256":digest(id),
            "randomization_block":"block-1",
            "control_role":role,
            "feasibility_status":"feasible",
            "safety_constraints_sha256":digest("safety")
        })
    }

    fn fixture_scorecard(model_id: &str, score: f64, low: f64, high: f64) -> Value {
        let units = vec![token(4), token(5)];
        json!({
            "model_id":model_id,"metric_id":"nll","stratum":"overall",
            "n_independent_units":2,"score":score,"ci_low":low,"ci_high":high,"score_state":"measured",
            "unit_tokens":units,"unit_manifest_sha256":membership_digest(&units)
        })
    }

    fn fixture_paired_comparison(
        baseline_id: &str,
        baseline_score: f64,
        low: f64,
        high: f64,
    ) -> Value {
        let pair = vec![token(4), token(5)];
        json!({
            "candidate_model_id":"candidate","baseline_model_id":baseline_id,"metric_id":"nll","stratum":"overall",
            "paired_unit_manifest_sha256":membership_digest(&pair),"n_independent_units":2,
            "mean_candidate_minus_baseline":0.8-baseline_score,"ci_low":low,"ci_high":high,
            "minimum_meaningful_effect":0.05,"non_inferiority_margin":0.1,
            "outcome":"outperforms_by_margin","paired_unit_tokens":pair
        })
    }

    fn complete_fixture() -> Value {
        let train = vec![token(1), token(2)];
        let validation = vec![token(3)];
        let test = vec![token(4), token(5)];
        let budget = json!({
            "compute_units":100.0,"wall_time_seconds":10.0,"energy_joules":2.0,
            "memory_bytes":4096,"training_samples":100,"notes":"matched synthetic budget"
        });
        let config = digest("config");
        let scope = digest("training-scope");
        let artifact = digest("model-artifact");
        let models = vec![
            json!({"model_id":"candidate","role":"candidate","model_family":"graph_constrained_state_space","hypothesis_id":"H_GRAPH","code_commit_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","config_sha256":config,"parameter_count":10,"training_data_scope_sha256":scope,"training_budget":budget.clone(),"observation_model_id":"obs","structural_channel_sha256":digest("structural"),"latent_channel_sha256":null,"model_artifact_sha256":artifact,"training_seed":1,"training_complete":true,"notes":"candidate"}),
            json!({"model_id":"activity","role":"baseline","model_family":"activity_only","hypothesis_id":null,"code_commit_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","config_sha256":config,"parameter_count":8,"training_data_scope_sha256":scope,"training_budget":budget.clone(),"observation_model_id":"obs","structural_channel_sha256":null,"latent_channel_sha256":null,"model_artifact_sha256":artifact,"training_seed":1,"training_complete":true,"notes":"activity baseline"}),
            json!({"model_id":"linear","role":"baseline","model_family":"linear_lds_var_glm","hypothesis_id":"H_LINEAR","code_commit_sha":"cccccccccccccccccccccccccccccccccccccccc","config_sha256":config,"parameter_count":9,"training_data_scope_sha256":scope,"training_budget":budget.clone(),"observation_model_id":"obs","structural_channel_sha256":null,"latent_channel_sha256":null,"model_artifact_sha256":artifact,"training_seed":1,"training_complete":true,"notes":"linear baseline"}),
            json!({"model_id":"rewired","role":"negative_control","model_family":"degree_type_rewired","hypothesis_id":null,"code_commit_sha":"dddddddddddddddddddddddddddddddddddddddd","config_sha256":config,"parameter_count":9,"training_data_scope_sha256":scope,"training_budget":budget.clone(),"observation_model_id":"obs","structural_channel_sha256":digest("rewired"),"latent_channel_sha256":null,"model_artifact_sha256":artifact,"training_seed":1,"training_complete":true,"notes":"negative control"}),
        ];
        let selected_interventions = vec![
            fixture_intervention("primary", "intervention", &["n-1"]),
            fixture_intervention("sham", "sham", &["n-1"]),
            fixture_intervention("single-control", "single_target_control", &["n-1"]),
            fixture_intervention(
                "combination-control",
                "combination_control",
                &["n-1", "n-2"],
            ),
        ];
        let scorecards = vec![
            fixture_scorecard("candidate", 0.8, 0.7, 0.9),
            fixture_scorecard("activity", 1.0, 0.95, 1.05),
            fixture_scorecard("linear", 1.2, 1.1, 1.3),
            fixture_scorecard("rewired", 1.4, 1.3, 1.5),
        ];
        let paired_comparisons = vec![
            fixture_paired_comparison("activity", 1.0, -0.3, -0.1),
            fixture_paired_comparison("linear", 1.2, -0.5, -0.3),
            fixture_paired_comparison("rewired", 1.4, -0.7, -0.5),
        ];
        json!({
            "schema_version":"CONNECTOME-INTERVENTION-RUN-V1","run_id":"synthetic-run-001",
            "run_state":"complete","evidence_tier":"synthetic_known_answer","claim_ceiling":"synthetic_only",
            "created_at_utc":"2026-11-02T00:00:00Z","parent_run_id":null,
            "source":{"source_kind":"synthetic","dataset_id":"synthetic-v1","dataset_version":"1","immutable_reference":"fixture://synthetic","source_manifest_sha256":digest("source"),"rights_state":"synthetic","permission_receipt_sha256":null,"provenance_notes":"synthetic only"},
            "connectome":{"taxon_id":"synthetic","taxon_name":"synthetic graph","connectome_id":"graph-1","reconstruction_version":"1","neuron_id_namespace":"synthetic-v1","row_axis_sha256":digest("rows"),"column_axis_sha256":digest("cols"),"canonical_identity_sha256":digest("identity"),"structural_edge_manifest_sha256":digest("edges"),"edge_semantics":"synthetic_graph","coverage_statement":"synthetic graph","source_artifacts":[]},
            "observation_model":{"observation_model_id":"obs","modality":"synthetic_latent","observation_kernel_id":"identity","observation_kernel_parameters_sha256":digest("kernel"),"sampling_interval_seconds":0.01,"time_origin":"synthetic t0","preprocessing_manifest_sha256":digest("preprocess"),"latent_observation_separation":true,"units":["arbitrary"],"indicator_or_sensor_id":null},
            "split":{
                "split_id":"split-1","split_strategy":"synthetic_known_answer","generalization_axes":["target_combination"],
                "split_manifest_sha256":digest("split"),"train_membership_sha256":membership_digest(&train),
                "validation_membership_sha256":membership_digest(&validation),"test_membership_sha256":membership_digest(&test),
                "preprocessing_fit_on_train_only":true,"feature_selection_fit_on_train_only":true,
                "model_selection_blind_to_test":true,"target_identity_split_before_preprocessing":true,
                "overlap_audit_sha256":digest("overlap"),"status":"qualified","grouping_keys":["target_combination"],
                "leakage_notes":"synthetic partition",
                "unit_tokenization":{"algorithm":"HMAC-SHA256","key_commitment_sha256":digest("key commitment"),"canonicalization_version":"UNIT-TOKEN-V1"},
                "train_unit_tokens":train,"validation_unit_tokens":validation,"test_unit_tokens":test
            },
            "hypothesis_set":[
                {"hypothesis_id":"H_GRAPH","model_family":"graph_constrained_state_space","mechanistic_channels":["direct_anatomical"],"prior_probability":0.5,"prior_source":"synthetic","frozen_before_test":true,"prediction_manifest_sha256":digest("pred-g")},
                {"hypothesis_id":"H_LINEAR","model_family":"linear_lds_var_glm","mechanistic_channels":["shared_input"],"prior_probability":0.5,"prior_source":"synthetic","frozen_before_test":true,"prediction_manifest_sha256":digest("pred-l")}
            ],
            "models":models,
            "intervention_design":{
                "candidate_pool_sha256":digest("pool"),"selector":"manual_preregistered","selection_objective":"model_discrimination",
                "preregistered_before_test":true,
                "experiment_budget":{"compute_units":10.0,"wall_time_seconds":5.0,"energy_joules":1.0,"memory_bytes":1024,"training_samples":0,"notes":"synthetic"},
                "selected_interventions":selected_interventions,
                "required_controls":["sham","single_target","combined_target","rewired_graph","activity_only"],
                "feasibility_audit_sha256":digest("feasibility"),"stopping_rule":"fixed"
            },
            "budgets":{"comparison_mode":"matched_budget","training_budget_policy":"matched","evaluation_budget_policy":"matched","hardware":"synthetic","wall_time_seconds_limit":60.0,"energy_joules_limit":10.0,"memory_bytes_limit":1048576,"budget_exception_rationale":""},
            "evaluation":{
                "task_ids":["synthetic-task"],"primary_metric_ids":["nll"],
                "metrics":[
                    {"metric_id":"nll","metric_family":"negative_log_predictive_density","definition_version":"1","unit":"nats/unit","direction":"lower_is_better","aggregation":"per_trial","primary":true,"formula_reference":null},
                    {"metric_id":"coverage","metric_family":"predictive_interval_coverage","definition_version":"1","unit":"proportion","direction":"target_range","aggregation":"per_trial","primary":false,"formula_reference":null}
                ],
                "uncertainty":{"method_id":"cluster_bootstrap_percentile","confidence_level":0.95,"resampling_unit":"intervention_block","cluster_keys":["animal-token","session-token"],"two_sided":true,"minimum_independent_units":2,"seed":1},
                "strata":["overall"],"promotion_targets":[{"metric_id":"nll","target_value":0.1,"comparison":"at_most","absolute_tolerance":null,"status":"aspirational_not_measured"}],
                "baseline_matrix":{"required_families":["activity_only","linear_lds_var_glm","graph_constrained_state_space","degree_type_rewired","pairwise_additive"],"unavailable_families":[{"family":"pairwise_additive","reason":"not applicable to this small synthetic linear fixture"}],"coverage_sha256":digest("baseline coverage")}
            },
            "results":{
                "status":"complete","observed_outcomes_sha256":digest("outcomes"),"heldout_prediction_sha256":digest("predictions"),
                "scorecards":scorecards,
                "paired_comparisons":paired_comparisons,
                "calibration":[{"model_id":"candidate","metric_id":"coverage","stratum":"overall","nominal_coverage":0.95,"observed_coverage":0.95,"interval_sharpness":0.1,"n_independent_units":2}],
                "failures":[]
            },
            "artifacts":{"run_receipt_sha256":digest("receipt"),"resolved_config_sha256":digest("resolved"),"environment_sha256":digest("environment"),"raw_output_manifest_sha256":digest("raw"),"logs_manifest_sha256":digest("logs"),"artifact_references":[]},
            "limitations":["synthetic fixture only; no biological claim"]
        })
    }

    #[test]
    fn membership_digest_is_order_independent_and_uses_trailing_newline() {
        let a = vec![token(2), token(1)];
        let b = vec![token(1), token(2)];
        assert_eq!(membership_digest(&a), membership_digest(&b));
        assert_ne!(
            membership_digest(&a),
            sha256_hex(b.to_vec().join("\n").as_bytes())
        );
    }

    #[test]
    fn incomplete_public_example_is_not_promoted_to_complete() {
        let value: Value = serde_json::from_str(include_str!(
            "../../../docs/neuroscience/CONNECTOME_INTERVENTION_BENCHMARK_EXAMPLE_V1.json"
        ))
        .unwrap();
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.errors.is_empty());
    }

    #[test]
    fn complete_synthetic_fixture_satisfies_semantic_contract() {
        let report = verify_manifest(&complete_fixture());
        assert_eq!(report.disposition, "COMPLETE", "{:?}", report.errors);
    }

    #[test]
    fn split_overlap_is_rejected_even_when_digest_is_recomputed() {
        let mut value = complete_fixture();
        let overlap = value["split"]["train_unit_tokens"][0].clone();
        value["split"]["test_unit_tokens"][0] = overlap;
        let test = value["split"]["test_unit_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        value["split"]["test_membership_sha256"] = json!(membership_digest(&test));
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("train/test overlap"))
        );
    }

    #[test]
    fn split_digest_mismatch_is_rejected() {
        let mut value = complete_fixture();
        value["split"]["test_membership_sha256"] = json!(digest("wrong"));
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.errors.iter().any(|e| e.contains("digest mismatch")));
    }

    #[test]
    fn paired_units_must_be_members_of_test_split() {
        let mut value = complete_fixture();
        value["results"]["paired_comparisons"][0]["paired_unit_tokens"][0] = json!(token(99));
        let pair = value["results"]["paired_comparisons"][0]["paired_unit_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        value["results"]["paired_comparisons"][0]["paired_unit_manifest_sha256"] =
            json!(membership_digest(&pair));
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("outside held-out test membership"))
        );
    }

    #[test]
    fn paired_scorecards_must_use_the_identical_unit_cohort() {
        let mut value = complete_fixture();
        value["split"]["test_unit_tokens"]
            .as_array_mut()
            .unwrap()
            .push(json!(token(6)));
        let test = value["split"]["test_unit_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        value["split"]["test_membership_sha256"] = json!(membership_digest(&test));

        // Candidate cohort is [4,5]; baseline cohort becomes [4,6]. Both have
        // n=2, both are in the declared test split, but they are not paired.
        value["results"]["scorecards"][1]["unit_tokens"][1] = json!(token(6));
        let baseline_units = value["results"]["scorecards"][1]["unit_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        value["results"]["scorecards"][1]["unit_manifest_sha256"] =
            json!(membership_digest(&baseline_units));

        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("paired units must exactly match"))
        );
    }

    #[test]
    fn prior_probabilities_must_be_normalized_before_complete_run() {
        let mut value = complete_fixture();
        value["hypothesis_set"][0]["prior_probability"] = json!(0.9);
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("prior probabilities sum"))
        );
    }

    #[test]
    fn declared_sham_control_must_be_materialized() {
        let mut value = complete_fixture();
        value["intervention_design"]["selected_interventions"][1]["control_role"] = json!("other");
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("declared sham control has no materialized"))
        );
    }

    #[test]
    fn duplicate_paired_comparison_is_rejected() {
        let mut value = complete_fixture();
        let duplicate = value["results"]["paired_comparisons"][0].clone();
        value["results"]["paired_comparisons"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("duplicate candidate/baseline/metric/stratum"))
        );
    }

    #[test]
    fn paired_mean_must_reconcile_with_scorecards() {
        let mut value = complete_fixture();
        value["results"]["paired_comparisons"][0]["mean_candidate_minus_baseline"] = json!(-0.8);
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("mean delta does not reconcile"))
        );
    }

    #[test]
    fn missing_baseline_explanation_is_rejected() {
        let mut value = complete_fixture();
        value["evaluation"]["baseline_matrix"]["unavailable_families"] = json!([]);
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("neither implemented nor explicitly unavailable"))
        );
    }

    #[test]
    fn unresolved_scorecard_model_is_rejected() {
        let mut value = complete_fixture();
        value["results"]["scorecards"][0]["model_id"] = json!("unresolved");
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("unresolved unresolved"))
        );
    }

    #[test]
    fn ordered_protocol_steps_and_state_context_are_verified() {
        let mut fixture = complete_fixture();
        let intervention = &mut fixture["intervention_design"]["selected_interventions"][0];
        intervention["target_ids"] = json!(["n-1", "n-2"]);
        intervention["protocol_semantics"] = json!("ordered_steps");
        intervention["state_context_id"] = json!("active-state");
        intervention["state_measurement_manifest_sha256"] = json!(digest("state-measurement"));
        let waveform = intervention["waveform"].clone();
        intervention["protocol_steps"] = json!([
            {
                "step_index":0,
                "target_ids":["n-1"],
                "manipulation_type":"simulated_ablation",
                "waveform":waveform,
                "inter_step_interval_seconds":0.0,
                "state_context_id":"active-state"
            },
            {
                "step_index":1,
                "target_ids":["n-2"],
                "manipulation_type":"simulated_ablation",
                "waveform":waveform,
                "inter_step_interval_seconds":0.5,
                "state_context_id":"active-state"
            }
        ]);
        let report = verify_manifest(&fixture);
        assert_eq!(report.disposition, "COMPLETE", "{:?}", report.errors);

        fixture["intervention_design"]["selected_interventions"][0]["protocol_steps"][1]["step_index"] = json!(2);
        let report = verify_manifest(&fixture);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.errors.iter().any(|error| error.contains("contiguous and zero-based")));
    }

    #[test]
    fn protocol_steps_cannot_disagree_with_parent_target_set() {
        let mut fixture = complete_fixture();
        let intervention = &mut fixture["intervention_design"]["selected_interventions"][0];
        intervention["target_ids"] = json!(["n-1", "n-2"]);
        intervention["protocol_semantics"] = json!("ordered_steps");
        let waveform = intervention["waveform"].clone();
        intervention["protocol_steps"] = json!([
            {"step_index":0,"target_ids":["n-1"],"manipulation_type":"simulated_ablation","waveform":waveform,"inter_step_interval_seconds":0.0},
            {"step_index":1,"target_ids":["n-3"],"manipulation_type":"simulated_ablation","waveform":waveform,"inter_step_interval_seconds":0.5}
        ]);
        let report = verify_manifest(&fixture);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.errors.iter().any(|error| error.contains("parent target_ids must equal the union")));
    }

    #[test]
    fn measured_intervention_requires_documented_permission() {
        let mut value = complete_fixture();
        value["evidence_tier"] = json!("measured_intervention");
        value["claim_ceiling"] = json!("measured_intervention_prediction");
        value["source"]["source_kind"] = json!("measured_intervention");
        value["source"]["rights_state"] = json!("unknown");
        let report = verify_manifest(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("documented_permitted"))
        );
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("permission_receipt_sha256"))
        );
    }
}
