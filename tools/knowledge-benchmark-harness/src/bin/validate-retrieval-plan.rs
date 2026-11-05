use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const SCHEMA_VERSION: &str = "retrieval-plan-v0.1";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalPlan {
    schema_version: String,
    plan_id: String,
    original_query: String,
    original_query_sha256: String,
    fallback_query: String,
    rewrite_policy: String,
    required_anchors: Vec<RequiredAnchor>,
    constraints: Vec<RetrievalConstraint>,
    steps: Vec<RetrievalStep>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequiredAnchor {
    anchor_id: String,
    kind: String,
    text: String,
    match_mode: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalConstraint {
    constraint_id: String,
    kind: String,
    value: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalStep {
    step_id: String,
    query: String,
    required_anchor_ids: Vec<String>,
    required_constraint_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationReport {
    report_schema_version: &'static str,
    plan_id: String,
    passed: bool,
    recommended_action: &'static str,
    checked_steps: usize,
    checked_anchors: usize,
    checked_constraints: usize,
    errors: Vec<String>,
    nonclaims: Vec<&'static str>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn has_non_whitespace(value: &str) -> bool {
    !value.trim().is_empty()
}

fn contains_anchor(query: &str, anchor: &RequiredAnchor) -> bool {
    match anchor.match_mode.as_str() {
        "exact" => query.contains(&anchor.text),
        "case_insensitive" => query.to_lowercase().contains(&anchor.text.to_lowercase()),
        _ => false,
    }
}

fn valid_anchor_kind(kind: &str) -> bool {
    matches!(
        kind,
        "exact_entity" | "identifier" | "numeric" | "date" | "rare_term" | "negation" | "modality"
    )
}

fn valid_constraint_kind(kind: &str) -> bool {
    matches!(
        kind,
        "language"
            | "geography"
            | "valid_time"
            | "known_at"
            | "relation"
            | "source_policy"
            | "rights_policy"
            | "source_family"
            | "domain"
    )
}

fn validate_plan(plan: &RetrievalPlan) -> ValidationReport {
    let mut errors = Vec::new();

    if plan.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "schema_version must be {SCHEMA_VERSION:?}, got {:?}",
            plan.schema_version
        ));
    }
    if !has_non_whitespace(&plan.plan_id) {
        errors.push("plan_id must be non-empty".to_string());
    }
    if !has_non_whitespace(&plan.original_query) {
        errors.push("original_query must be non-empty".to_string());
    }
    if plan.original_query_sha256 != sha256_hex(plan.original_query.as_bytes()) {
        errors.push(
            "original_query_sha256 does not match the exact original_query bytes".to_string(),
        );
    }
    if plan.fallback_query != plan.original_query {
        errors.push("fallback_query must equal the immutable original_query exactly".to_string());
    }
    if !matches!(
        plan.rewrite_policy.as_str(),
        "original_only" | "anchor_preserving"
    ) {
        errors.push(format!(
            "rewrite_policy must be original_only or anchor_preserving, got {:?}",
            plan.rewrite_policy
        ));
    }

    let mut anchors_by_id = HashMap::new();
    for anchor in &plan.required_anchors {
        if !has_non_whitespace(&anchor.anchor_id) {
            errors.push("anchor_id must be non-empty".to_string());
        } else if anchors_by_id
            .insert(anchor.anchor_id.as_str(), anchor)
            .is_some()
        {
            errors.push(format!("duplicate anchor_id {:?}", anchor.anchor_id));
        }
        if !valid_anchor_kind(&anchor.kind) {
            errors.push(format!(
                "anchor {:?} has unsupported kind {:?}",
                anchor.anchor_id, anchor.kind
            ));
        }
        if !has_non_whitespace(&anchor.text) {
            errors.push(format!("anchor {:?} has empty text", anchor.anchor_id));
        }
        if !matches!(anchor.match_mode.as_str(), "exact" | "case_insensitive") {
            errors.push(format!(
                "anchor {:?} has unsupported match_mode {:?}",
                anchor.anchor_id, anchor.match_mode
            ));
        } else if has_non_whitespace(&anchor.text) && !contains_anchor(&plan.original_query, anchor)
        {
            errors.push(format!(
                "required anchor {:?} is absent from immutable original_query",
                anchor.anchor_id
            ));
        }
    }

    let mut constraints_by_id = HashMap::new();
    for constraint in &plan.constraints {
        if !has_non_whitespace(&constraint.constraint_id) {
            errors.push("constraint_id must be non-empty".to_string());
        } else if constraints_by_id
            .insert(constraint.constraint_id.as_str(), constraint)
            .is_some()
        {
            errors.push(format!(
                "duplicate constraint_id {:?}",
                constraint.constraint_id
            ));
        }
        if !valid_constraint_kind(&constraint.kind) {
            errors.push(format!(
                "constraint {:?} has unsupported kind {:?}",
                constraint.constraint_id, constraint.kind
            ));
        }
        if !has_non_whitespace(&constraint.value) {
            errors.push(format!(
                "constraint {:?} has empty value",
                constraint.constraint_id
            ));
        }
    }

    if plan.steps.is_empty() {
        errors.push("at least one retrieval step is required".to_string());
    }

    let mut step_ids = HashSet::new();
    let mut covered_anchors = HashSet::new();
    let mut covered_constraints = HashSet::new();

    for step in &plan.steps {
        if !has_non_whitespace(&step.step_id) {
            errors.push("step_id must be non-empty".to_string());
        } else if !step_ids.insert(step.step_id.as_str()) {
            errors.push(format!("duplicate step_id {:?}", step.step_id));
        }
        if !has_non_whitespace(&step.query) {
            errors.push(format!("step {:?} has an empty query", step.step_id));
        }
        if plan.rewrite_policy == "original_only" && step.query != plan.original_query {
            errors.push(format!(
                "step {:?} rewrites the query under original_only policy",
                step.step_id
            ));
        }

        let mut step_anchor_ids = HashSet::new();
        for anchor_id in &step.required_anchor_ids {
            if !step_anchor_ids.insert(anchor_id.as_str()) {
                errors.push(format!(
                    "step {:?} repeats required anchor {:?}",
                    step.step_id, anchor_id
                ));
                continue;
            }
            match anchors_by_id.get(anchor_id.as_str()) {
                Some(anchor) => {
                    covered_anchors.insert(anchor_id.as_str());
                    if !contains_anchor(&step.query, anchor) {
                        errors.push(format!(
                            "step {:?} dropped required {:?} anchor {:?}",
                            step.step_id, anchor.kind, anchor.text
                        ));
                    }
                }
                None => errors.push(format!(
                    "step {:?} references unknown anchor {:?}",
                    step.step_id, anchor_id
                )),
            }
        }

        let mut step_constraint_ids = HashSet::new();
        for constraint_id in &step.required_constraint_ids {
            if !step_constraint_ids.insert(constraint_id.as_str()) {
                errors.push(format!(
                    "step {:?} repeats required constraint {:?}",
                    step.step_id, constraint_id
                ));
                continue;
            }
            if constraints_by_id.contains_key(constraint_id.as_str()) {
                covered_constraints.insert(constraint_id.as_str());
            } else {
                errors.push(format!(
                    "step {:?} references unknown constraint {:?}",
                    step.step_id, constraint_id
                ));
            }
        }
    }

    for anchor in &plan.required_anchors {
        if !covered_anchors.contains(anchor.anchor_id.as_str()) {
            errors.push(format!(
                "required anchor {:?} is not assigned to any retrieval step",
                anchor.anchor_id
            ));
        }
    }
    for constraint in &plan.constraints {
        if !covered_constraints.contains(constraint.constraint_id.as_str()) {
            errors.push(format!(
                "constraint {:?} is not assigned to any retrieval step",
                constraint.constraint_id
            ));
        }
    }

    let passed = errors.is_empty();
    ValidationReport {
        report_schema_version: "retrieval-plan-validation-report-v0.1",
        plan_id: plan.plan_id.clone(),
        passed,
        recommended_action: if passed {
            "plan_contract_valid_candidate_only"
        } else {
            "reject_rewrite_and_use_original_query"
        },
        checked_steps: plan.steps.len(),
        checked_anchors: plan.required_anchors.len(),
        checked_constraints: plan.constraints.len(),
        errors,
        nonclaims: vec![
            "does not execute retrieval or authorize network access",
            "does not prove a retrieval step was executed or completed",
            "does not establish evidence truth, entailment, independence, or source rights",
            "does not establish index, cache, embedding, or model materialization time",
            "does not prove temporal coverage or benchmark performance",
        ],
    }
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut args = env::args().skip(1);
    let mut schema = None;
    let mut input = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schema" => {
                schema = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--schema requires a path".to_string())?,
                ));
            }
            "--input" => {
                input = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--input requires a path".to_string())?,
                ));
            }
            "--help" | "-h" => {
                return Err(
                    "Usage: validate-retrieval-plan --schema SCHEMA.json --input PLAN.json\nValidates a local candidate plan only; no network or retrieval is performed."
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok((
        schema.ok_or_else(|| "--schema is required".to_string())?,
        input.ok_or_else(|| "--input is required".to_string())?,
    ))
}

fn schema_has_external_refs(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, child)| {
            (matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef")
                && !child.as_str().is_some_and(|target| target.starts_with('#')))
                || schema_has_external_refs(child)
        }),
        serde_json::Value::Array(items) => items.iter().any(schema_has_external_refs),
        _ => false,
    }
}

fn main() -> ExitCode {
    let (schema_path, path) = match parse_args() {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read {}: {error}", path.display());
            return ExitCode::from(2);
        }
    };
    let plan: RetrievalPlan = match serde_json::from_slice(&bytes) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("invalid retrieval plan JSON: {error}");
            return ExitCode::from(2);
        }
    };

    let input_value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid retrieval plan JSON: {error}");
            return ExitCode::from(2);
        }
    };
    let schema_bytes = match fs::read(&schema_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read schema {}: {error}", schema_path.display());
            return ExitCode::from(2);
        }
    };
    let schema: serde_json::Value = match serde_json::from_slice(&schema_bytes) {
        Ok(schema) => schema,
        Err(error) => {
            eprintln!("invalid schema JSON: {error}");
            return ExitCode::from(2);
        }
    };
    if schema.get("$schema").and_then(serde_json::Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        eprintln!("schema must declare JSON Schema Draft 2020-12 exactly");
        return ExitCode::from(2);
    }
    if schema_has_external_refs(&schema) {
        eprintln!("non-local JSON Schema references are refused");
        return ExitCode::from(2);
    }
    let meta_validator = match jsonschema::meta::validator_for(&schema) {
        Ok(validator) => validator,
        Err(error) => {
            eprintln!("cannot build schema meta-validator: {error}");
            return ExitCode::from(2);
        }
    };
    if !meta_validator.evaluate(&schema).flag().valid {
        eprintln!("schema does not validate against its detected meta-schema");
        return ExitCode::from(2);
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            eprintln!("cannot compile retrieval plan schema: {error}");
            return ExitCode::from(2);
        }
    };

    let mut report = validate_plan(&plan);
    report.errors.extend(
        validator
            .iter_errors(&input_value)
            .map(|error| error.to_string()),
    );
    report.passed = report.errors.is_empty();
    if !report.passed {
        report.recommended_action = "reject_rewrite_and_use_original_query";
    }
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("cannot serialize validation report: {error}");
            return ExitCode::from(2);
        }
    }
    if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> RetrievalPlan {
        let original_query = "What did the 2024 report say about KineticTrust v3.2 in isiZulu?";
        RetrievalPlan {
            schema_version: SCHEMA_VERSION.to_string(),
            plan_id: "synthetic-plan-001".to_string(),
            original_query: original_query.to_string(),
            original_query_sha256: sha256_hex(original_query.as_bytes()),
            fallback_query: original_query.to_string(),
            rewrite_policy: "anchor_preserving".to_string(),
            required_anchors: vec![
                RequiredAnchor {
                    anchor_id: "a-report-year".to_string(),
                    kind: "date".to_string(),
                    text: "2024".to_string(),
                    match_mode: "exact".to_string(),
                },
                RequiredAnchor {
                    anchor_id: "a-product".to_string(),
                    kind: "exact_entity".to_string(),
                    text: "KineticTrust".to_string(),
                    match_mode: "exact".to_string(),
                },
                RequiredAnchor {
                    anchor_id: "a-version".to_string(),
                    kind: "identifier".to_string(),
                    text: "v3.2".to_string(),
                    match_mode: "exact".to_string(),
                },
                RequiredAnchor {
                    anchor_id: "a-language".to_string(),
                    kind: "rare_term".to_string(),
                    text: "isiZulu".to_string(),
                    match_mode: "case_insensitive".to_string(),
                },
            ],
            constraints: vec![
                RetrievalConstraint {
                    constraint_id: "c-language".to_string(),
                    kind: "language".to_string(),
                    value: "isiZulu".to_string(),
                },
                RetrievalConstraint {
                    constraint_id: "c-time".to_string(),
                    kind: "valid_time".to_string(),
                    value: "2024-01-01/2024-12-31".to_string(),
                },
            ],
            steps: vec![RetrievalStep {
                step_id: "s-primary".to_string(),
                query: "What did the 2024 report say about KineticTrust v3.2 in isiZulu?"
                    .to_string(),
                required_anchor_ids: vec![
                    "a-report-year".to_string(),
                    "a-product".to_string(),
                    "a-version".to_string(),
                    "a-language".to_string(),
                ],
                required_constraint_ids: vec!["c-language".to_string(), "c-time".to_string()],
            }],
        }
    }

    #[test]
    fn valid_anchor_preserving_plan_passes() {
        let report = validate_plan(&sample_plan());
        assert!(report.passed, "{:?}", report.errors);
        assert_eq!(
            report.recommended_action,
            "plan_contract_valid_candidate_only"
        );
    }

    #[test]
    fn removed_named_entity_is_rejected() {
        let mut plan = sample_plan();
        plan.steps[0].query =
            "What did the 2024 report say about the product v3.2 in isiZulu?".to_string();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("KineticTrust"))
        );
        assert_eq!(
            report.recommended_action,
            "reject_rewrite_and_use_original_query"
        );
    }

    #[test]
    fn changed_numeric_or_version_anchor_is_rejected() {
        let mut plan = sample_plan();
        plan.steps[0].query =
            "What did the 2024 report say about KineticTrust v3.3 in isiZulu?".to_string();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(report.errors.iter().any(|error| error.contains("v3.2")));
    }

    #[test]
    fn removed_date_anchor_is_rejected() {
        let mut plan = sample_plan();
        plan.steps[0].query =
            "What did the report say about KineticTrust v3.2 in isiZulu?".to_string();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(report.errors.iter().any(|error| error.contains("2024")));
    }

    #[test]
    fn case_insensitive_anchor_accepts_case_only_change() {
        let mut plan = sample_plan();
        plan.steps[0].query =
            "What did the 2024 report say about KineticTrust v3.2 in ISIZULU?".to_string();
        let report = validate_plan(&plan);
        assert!(report.passed, "{:?}", report.errors);
    }

    #[test]
    fn required_anchors_must_be_present_in_immutable_original_query() {
        let mut plan = sample_plan();
        plan.original_query = plan.original_query.replace("KineticTrust", "the product");
        plan.original_query_sha256 = sha256_hex(plan.original_query.as_bytes());
        plan.fallback_query = plan.original_query.clone();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("absent from immutable original_query"))
        );
    }

    #[test]
    fn immutable_original_query_digest_is_checked() {
        let mut plan = sample_plan();
        plan.original_query_sha256 = "0".repeat(64);
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(report.errors.iter().any(|error| error.contains("sha256")));
    }

    #[test]
    fn fallback_must_equal_original_query() {
        let mut plan = sample_plan();
        plan.fallback_query.push_str(" altered");
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("fallback_query"))
        );
    }

    #[test]
    fn duplicate_anchor_ids_are_rejected() {
        let mut plan = sample_plan();
        let duplicate = plan.required_anchors[0].clone();
        plan.required_anchors.push(duplicate);
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("duplicate anchor_id"))
        );
    }

    #[test]
    fn unknown_anchor_references_are_rejected() {
        let mut plan = sample_plan();
        plan.steps[0]
            .required_anchor_ids
            .push("missing-anchor".to_string());
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("unknown anchor"))
        );
    }

    #[test]
    fn all_declared_anchors_must_be_covered_by_a_step() {
        let mut plan = sample_plan();
        plan.steps[0].required_anchor_ids.pop();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("not assigned"))
        );
    }

    #[test]
    fn unknown_constraint_references_are_rejected() {
        let mut plan = sample_plan();
        plan.steps[0]
            .required_constraint_ids
            .push("missing-constraint".to_string());
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("unknown constraint"))
        );
    }

    #[test]
    fn every_constraint_must_be_assigned_to_a_step() {
        let mut plan = sample_plan();
        plan.steps[0].required_constraint_ids.pop();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("not assigned"))
        );
    }

    #[test]
    fn original_only_policy_rejects_rewrites() {
        let mut plan = sample_plan();
        plan.rewrite_policy = "original_only".to_string();
        plan.steps[0].query = "different query".to_string();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("original_only"))
        );
    }

    #[test]
    fn empty_step_list_is_rejected() {
        let mut plan = sample_plan();
        plan.steps.clear();
        let report = validate_plan(&plan);
        assert!(!report.passed);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("at least one"))
        );
    }

    #[test]
    fn reports_never_claim_execution_or_truth() {
        let report = validate_plan(&sample_plan());
        assert!(report.passed);
        assert!(
            report
                .nonclaims
                .iter()
                .any(|claim| claim.contains("does not execute retrieval"))
        );
        assert!(
            report
                .nonclaims
                .iter()
                .any(|claim| claim.contains("evidence truth"))
        );
    }
}
