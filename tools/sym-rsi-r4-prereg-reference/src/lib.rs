//! Offline semantic readiness checks for Symthaea R4 preregistrations.
//!
//! This validator checks declarations and internal consistency. It does not
//! authenticate digests/signatures, inspect sealed split contents, prove
//! evaluator independence, execute research agents, or authorize execution.

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "sym-rsi-r4-ignition-preregistration-v0";
const PLACEHOLDERS: [&str; 2] = [
    "__REQUIRED_BEFORE_PREREGISTRATION__",
    "DETERMINE_AFTER_PILOT_AND_FREEZE_BEFORE_CONFIRMATION",
];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub schema_version: String,
    pub ready_for_independent_review: bool,
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            ready_for_independent_review: true,
            issues: Vec::new(),
            limitations: vec![
                "Digest syntax is checked; referenced bytes, signatures and custody are not authenticated."
                    .to_owned(),
                "Distinct split digests and a zero-overlap audit declaration do not prove that the underlying task/family populations are disjoint."
                    .to_owned(),
                "Shared evaluator bindings are checked as declarations; evaluator independence and sealed-data isolation require independent audit."
                    .to_owned(),
                "A ready report is not an execution receipt, scientific result, safety approval or promotion authorization."
                    .to_owned(),
            ],
        }
    }

    fn issue(&mut self, code: &str, detail: impl AsRef<str>) {
        self.ready_for_independent_review = false;
        self.issues.push(format!("{code}: {}", detail.as_ref()));
    }

    fn finish(mut self) -> Self {
        self.issues.sort();
        self.issues.dedup();
        self
    }
}

fn at<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(root, |value, key| value.get(*key))
}

fn string_at<'a>(root: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(root, path).and_then(Value::as_str)
}

fn bool_at(root: &Value, path: &[&str]) -> Option<bool> {
    at(root, path).and_then(Value::as_bool)
}

fn integer_at(root: &Value, path: &[&str]) -> Option<u64> {
    at(root, path)?.as_u64()
}

fn non_empty(value: Option<&str>) -> bool {
    value.is_some_and(|raw| !raw.trim().is_empty())
}

fn is_lower_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_string(report: &mut ValidationReport, root: &Value, path: &[&str]) {
    if !non_empty(string_at(root, path)) {
        report.issue("REQUIRED_STRING_MISSING", path.join("."));
    }
}

fn require_sha256(report: &mut ValidationReport, root: &Value, path: &[&str]) {
    match string_at(root, path) {
        Some(value) if is_lower_hex(value, 64) => {}
        _ => report.issue("INVALID_SHA256", path.join(".")),
    }
}

fn require_bool(report: &mut ValidationReport, root: &Value, path: &[&str], expected: bool) {
    if bool_at(root, path) != Some(expected) {
        report.issue(
            "BOOLEAN_GUARDRAIL_MISMATCH",
            format!("{} must be {expected}", path.join(".")),
        );
    }
}

fn collect_placeholders(value: &Value, path: &str, output: &mut Vec<String>) {
    match value {
        Value::String(raw) => {
            if PLACEHOLDERS.iter().any(|marker| raw.contains(marker)) {
                output.push(path.to_owned());
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                collect_placeholders(item, &format!("{path}[{index}]"), output);
            }
        }
        Value::Object(fields) => {
            for (key, item) in fields {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                collect_placeholders(item, &child, output);
            }
        }
        _ => {}
    }
}

fn check_arm_bindings(report: &mut ValidationReport, root: &Value) {
    let Some(arms) = at(root, &["arms"]).and_then(Value::as_array) else {
        report.issue("ARM_ARRAY_MISSING", "arms must be an array");
        return;
    };
    if arms.len() != 2 {
        report.issue("ARM_COUNT", format!("expected exactly two arms, found {}", arms.len()));
        return;
    }

    let expected = BTreeMap::from([
        (
            "evolved_outer_loop",
            "starting_state.outer_loop_arms.evolved_researcher",
        ),
        (
            "reference_outer_loop",
            "starting_state.outer_loop_arms.reference_researcher",
        ),
    ]);
    let mut seen = BTreeSet::new();
    for arm in arms {
        let id = arm.get("arm_id").and_then(Value::as_str).unwrap_or("");
        if !seen.insert(id.to_owned()) {
            report.issue("DUPLICATE_ARM_ID", id);
        }
        match expected.get(id) {
            Some(researcher_ref) => {
                if arm.get("researcher_ref").and_then(Value::as_str) != Some(*researcher_ref) {
                    report.issue("RESEARCHER_REFERENCE_MISMATCH", id);
                }
            }
            None => report.issue("UNKNOWN_ARM_ID", id),
        }
        if arm.get("inner_agent_start_ref").and_then(Value::as_str)
            != Some("starting_state.shared_inner_agent")
        {
            report.issue("START_REFERENCE_MISMATCH", id);
        }
    }
    if seen != expected.keys().map(|value| (*value).to_owned()).collect() {
        report.issue("ARM_SET_MISMATCH", "expected evolved_outer_loop and reference_outer_loop");
    }

    let left_backend = string_at(
        root,
        &[
            "starting_state",
            "outer_loop_arms",
            "evolved_researcher",
            "model_backend_version",
        ],
    );
    let right_backend = string_at(
        root,
        &[
            "starting_state",
            "outer_loop_arms",
            "reference_researcher",
            "model_backend_version",
        ],
    );
    if !non_empty(left_backend) || left_backend != right_backend {
        report.issue("BACKEND_MISMATCH", "both researcher arms must use the same model/backend version");
    }
    let identity_fields = [
        "artifact_digest",
        "source_tree_digest",
        "executable_digest",
        "prompt_tool_contract_digest",
        "memory_state_digest",
    ];
    let identities_equal = identity_fields.iter().all(|field| {
        string_at(
            root,
            &["starting_state", "outer_loop_arms", "evolved_researcher", field],
        ) == string_at(
            root,
            &["starting_state", "outer_loop_arms", "reference_researcher", field],
        )
    });
    if identities_equal {
        report.issue(
            "IDENTICAL_OUTER_RESEARCHERS",
            "evolved and reference outer-loop researcher artifact identities are identical",
        );
    }
    require_string(
        report,
        root,
        &["starting_state", "outer_loop_arms", "declared_differences"],
    );
    for field in [
        "execution_order_randomization",
        "common_random_numbers_policy",
        "permitted_arm_specific_differences",
    ] {
        require_string(report, root, &["matched_conditions", field]);
    }
    require_sha256(
        report,
        root,
        &["starting_state", "outer_loop_arms", "shared_frozen_context_digest"],
    );

    for arm in ["evolved_researcher", "reference_researcher"] {
        for field in [
            "artifact_digest",
            "source_tree_digest",
            "executable_digest",
            "prompt_tool_contract_digest",
            "memory_state_digest",
        ] {
            require_sha256(
                report,
                root,
                &["starting_state", "outer_loop_arms", arm, field],
            );
        }
        require_string(
            report,
            root,
            &["starting_state", "outer_loop_arms", arm, "lineage_evidence_ref"],
        );
    }
    for field in [
        "artifact_digest",
        "source_tree_digest",
        "executable_digest",
        "checkpoint_digest",
        "memory_research_state_digest",
    ] {
        require_sha256(report, root, &["starting_state", "shared_inner_agent", field]);
    }
    require_string(
        report,
        root,
        &["starting_state", "shared_inner_agent", "source_build_evidence_ref"],
    );
}

fn check_budget(report: &mut ValidationReport, root: &Value) {
    require_sha256(report, root, &["budget", "shared_envelope_digest"]);
    for field in [
        "same_envelope_for_all_arms",
        "inclusive_cost_boundary",
        "includes_failed_invalid_retries",
        "includes_human_review_and_rework",
    ] {
        require_bool(report, root, &["budget", field], true);
    }

    let shared = string_at(root, &["budget", "shared_envelope_digest"]);
    for arm in ["evolved_outer_loop", "reference_outer_loop"] {
        let binding = [
            "budget",
            "arm_budget_bindings",
            arm,
            "envelope_digest",
        ];
        require_sha256(report, root, &binding);
        if string_at(root, &binding) != shared {
            report.issue("UNEQUAL_ARM_BUDGET", arm);
        }
        require_bool(
            report,
            root,
            &["budget", "arm_budget_bindings", arm, "inclusive"],
            true,
        );
    }
    require_string(report, root, &["budget", "total_budget"]);
    require_string(report, root, &["budget", "currency_and_price_basis"]);
    require_string(report, root, &["budget", "accounting_boundary"]);

    let Some(caps) = at(root, &["budget", "per_arm_caps"]).and_then(Value::as_object) else {
        report.issue("BUDGET_CAPS_MISSING", "budget.per_arm_caps must be an object");
        return;
    };
    let required_caps = [
        "model_inference_tokens",
        "model_training_compute",
        "accelerator_hours_by_type",
        "cpu_hours",
        "evaluator_calls",
        "candidate_builds_and_executions",
        "retries",
        "wall_clock",
        "peak_memory",
        "energy",
        "human_active_minutes",
        "human_review_and_rework_minutes",
    ];
    for key in required_caps {
        match caps.get(key) {
            Some(value) if value.is_string() && non_empty(value.as_str()) => {}
            _ => report.issue("BUDGET_CAP_MISSING", key),
        }
    }
}

fn check_splits(report: &mut ValidationReport, root: &Value) {
    let paths: [&[&str]; 4] = [
        &["splits", "discovery", "manifest_digest"],
        &["splits", "validation", "manifest_digest"],
        &["splits", "sealed_confirmation", "manifest_digest_commitment"],
        &["splits", "fresh_family_transfer", "manifest_digest_commitment"],
    ];
    let mut digests = BTreeSet::new();
    for path in paths {
        require_sha256(report, root, path);
        if let Some(digest) = string_at(root, path) {
            if !digests.insert(digest.to_owned()) {
                report.issue("SPLIT_IDENTITY_COLLISION", path.join("."));
            }
        }
    }
    if digests.len() != 4 {
        report.issue("SPLIT_SET_INCOMPLETE", "four distinct split identities are required");
    }

    require_string(
        report,
        root,
        &["splits", "disjointness_audit", "independent_auditor"],
    );
    require_string(
        report,
        root,
        &["splits", "disjointness_audit", "evidence_ref"],
    );
    require_sha256(
        report,
        root,
        &["splits", "disjointness_audit", "audit_method_digest"],
    );
    if string_at(root, &["splits", "disjointness_audit", "overlap_unit"])
        != Some("task_instance_id_and_task_family_id")
        || integer_at(root, &["splits", "disjointness_audit", "expected_overlap_count"]) != Some(0)
    {
        report.issue(
            "SPLIT_DISJOINTNESS_NOT_DECLARED",
            "require a task-instance and task-family audit with expected overlap count zero",
        );
    }

    require_string(
        report,
        root,
        &["splits", "validation", "custodian"],
    );
    require_string(
        report,
        root,
        &["splits", "sealed_confirmation", "custodian_independent_of_candidate"],
    );
    require_string(
        report,
        root,
        &["splits", "sealed_confirmation", "one_shot_policy"],
    );
    require_string(
        report,
        root,
        &["splits", "fresh_family_transfer", "custodian_independent_of_candidate"],
    );
    require_bool(
        report,
        root,
        &["splits", "fresh_family_transfer", "retuning_prohibited"],
        true,
    );
    for field in [
        "task_manifest_digest",
        "task_family_manifest_digest",
        "cluster_definition_digest",
    ] {
        require_sha256(report, root, &["task_population", field]);
    }
    require_string(
        report,
        root,
        &["task_population", "licensing_and_data_rights_review_ref"],
    );
    let domains = at(root, &["task_population", "capability_domains"])
        .and_then(Value::as_array);
    if !domains.is_some_and(|items| {
        !items.is_empty()
            && items
                .iter()
                .all(|item| item.as_str().is_some_and(|name| !name.trim().is_empty()))
    }) {
        report.issue("CAPABILITY_DOMAINS_MISSING", "at least one non-empty capability domain is required");
    }
    for field in [
        "workspace_mount_audit_ref",
        "credentials_and_egress_test_ref",
        "canary_file_test_ref",
        "reviewer",
    ] {
        require_string(report, root, &["splits", "access_audit", field]);
    }
}

fn check_evaluator(report: &mut ValidationReport, root: &Value) {
    for field in [
        "implementation_digest",
        "configuration_digest",
        "task_contract_digest",
        "epoch_manifest_digest",
    ] {
        require_sha256(report, root, &["evaluator_binding", field]);
    }
    require_bool(report, root, &["evaluator_binding", "shared_across_arms"], true);
    require_bool(report, root, &["evaluator_binding", "candidate_write_access"], false);
    require_bool(
        report,
        root,
        &["evaluator_binding", "confirmation_material_write_access"],
        false,
    );
}

fn check_replication_and_rounds(report: &mut ValidationReport, root: &Value) {
    if integer_at(root, &["seed_and_replication_plan", "paired_pilot_seed_count"])
        .unwrap_or(0)
        < 5
    {
        report.issue("PILOT_SEED_FLOOR", "paired pilot must contain at least five seeds");
    }
    if integer_at(root, &["seed_and_replication_plan", "independent_full_loop_reruns_required"])
        .unwrap_or(0)
        < 2
    {
        report.issue("FULL_LOOP_RERUN_FLOOR", "at least two independent full-loop reruns are required");
    }
    if integer_at(root, &["seed_and_replication_plan", "confirmatory_seed_count"])
        .unwrap_or(0)
        == 0
    {
        report.issue("CONFIRMATORY_SEEDS_UNSET", "confirmatory seed count must be a positive frozen integer");
    }
    if integer_at(root, &["seed_and_replication_plan", "minimum_complete_outer_cycles"])
        .unwrap_or(0)
        < 2
    {
        report.issue("OUTER_CYCLE_FLOOR", "at least two complete outer cycles are required for recursive progress");
    }
    require_bool(
        report,
        root,
        &["seed_and_replication_plan", "all_trajectories_retained"],
        true,
    );

    if integer_at(root, &["rounds", "planned_outer_loop_steps"]).unwrap_or(0) == 0 {
        report.issue("ROUND_COUNT_UNSET", "planned outer-loop steps must be positive and frozen");
    }
    require_bool(report, root, &["rounds", "per_round_reporting_required"], true);
    let required = [
        "development score and objective",
        "independent audit score and objective",
        "candidate artifact/source/build identities",
        "accepted/rejected/invalid/failed/not-run changes",
        "promotion decision and policy/evaluator epoch",
        "budget consumed by category",
        "best-so-far and final state",
        "protected capability regressions",
        "human interventions and review time",
    ];
    let present: BTreeSet<String> = at(root, &["rounds", "per_round_fields"])
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default();
    for field in required {
        if !present.contains(field) {
            report.issue("PER_ROUND_FIELD_MISSING", field);
        }
    }
}

fn check_controls_and_analysis(report: &mut ValidationReport, root: &Value) {
    let required_controls = [
        "fixed_meta",
        "fixed_meta_archive",
        "noop_or_null",
        "random_matched_cost",
        "shuffled_feedback",
        "human_authored_reference",
        "candidate_evaluator_mismatch",
    ];
    let mut controls = BTreeSet::new();
    if let Some(items) = at(root, &["controls"]).and_then(Value::as_array) {
        for item in items {
            if let Some(id) = item.get("control_id").and_then(Value::as_str) {
                if !controls.insert(id.to_owned()) {
                    report.issue("DUPLICATE_CONTROL", id);
                }
            }
        }
    } else {
        report.issue("CONTROL_ARRAY_MISSING", "controls must be an array");
    }
    for id in required_controls {
        if !controls.contains(id) {
            report.issue("REQUIRED_CONTROL_MISSING", id);
        }
    }

    for field in [
        "statistical_unit",
        "clustering_and_dependence",
        "uncertainty_method",
        "confidence_level",
        "multiplicity_and_sequential_looks",
        "missingness_and_invalid_run_policy",
        "stopping_rule",
        "sensitivity_analyses",
    ] {
        require_string(report, root, &["analysis_plan", field]);
    }
    let commit = string_at(root, &["analysis_plan", "analysis_code_commit"]);
    if !commit.is_some_and(|value| is_lower_hex(value, 40)) {
        report.issue("ANALYSIS_COMMIT_INVALID", "analysis_code_commit must be a frozen 40-character lowercase commit SHA");
    }
    for field in [
        "claim_index_required",
        "raw_roundwise_data_required",
        "full_trajectory_and_failure_log_required",
        "cost_ledger_required",
        "best_so_far_and_final_required",
        "code_artifact_and_build_digests_required",
        "null_and_negative_control_results_required",
        "limitations_and_unidentified_effects_required",
    ] {
        require_bool(report, root, &["reporting", field], true);
    }
}

fn check_authority_and_claims(report: &mut ValidationReport, root: &Value) {
    for field in [
        "candidate_can_modify_evaluator",
        "candidate_can_access_confirmation_material",
        "candidate_can_change_preregistered_outcomes",
    ] {
        require_bool(report, root, &["governance", field], false);
    }

    let Some(claims) = at(root, &["experiment", "claim_profile"]) else {
        report.issue("CLAIM_PROFILE_MISSING", "experiment.claim_profile");
        return;
    };
    let value = |key: &str| claims.get(key).and_then(Value::as_bool).unwrap_or(false);
    if value("outer_loop_ignition")
        && !(value("matched_budget_better_improver")
            && value("repeatability")
            && value("fresh_family_transfer"))
    {
        report.issue(
            "IGNITION_CLAIM_DEPENDENCY",
            "outer-loop ignition requires matched-budget improver, repeatability and fresh-family transfer claims",
        );
    }
    if value("recursive_progress")
        && !(value("accumulation")
            && value("repeatability")
            && value("matched_budget_better_improver")
            && value("outer_loop_ignition")
            && value("fresh_family_transfer"))
    {
        report.issue(
            "RECURSIVE_PROGRESS_CLAIM_DEPENDENCY",
            "recursive progress requires accumulation, repeatability, matched-budget ignition and fresh-family transfer",
        );
    }
    if value("cross_domain_transfer") {
        let domains = at(root, &["task_population", "capability_domains"])
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if domains.len() < 2 {
            report.issue("CROSS_DOMAIN_POPULATION_TOO_SMALL", "cross-domain transfer requires at least two distinct declared domains");
        }
    }
    require_string(report, root, &["experiment", "estimand", "primary_outcome"]);
    require_string(report, root, &["experiment", "estimand", "target_population"]);
    require_string(report, root, &["experiment", "estimand", "practical_effect_threshold"]);
}

pub fn validate_preregistration(root: &Value) -> ValidationReport {
    let mut report = ValidationReport::new();

    if string_at(root, &["schema_version"]) != Some(SCHEMA_VERSION) {
        report.issue("SCHEMA_VERSION_MISMATCH", "unsupported or missing schema_version");
    }
    if string_at(root, &["document_status"]) != Some("PREREGISTERED") {
        report.issue(
            "DOCUMENT_NOT_FROZEN",
            "document_status must be PREREGISTERED; a template is never ready",
        );
    }

    let mut placeholders = Vec::new();
    collect_placeholders(root, "", &mut placeholders);
    for path in placeholders {
        report.issue("UNRESOLVED_PLACEHOLDER", path);
    }

    for field in [
        "experiment_id",
        "protocol_version",
        "preregistered_at_utc",
        "preregistration_commit",
    ] {
        require_string(&mut report, root, &["experiment", field]);
    }
    if !string_at(root, &["experiment", "preregistration_commit"])
        .is_some_and(|value| is_lower_hex(value, 40))
    {
        report.issue("PREREGISTRATION_COMMIT_INVALID", "expected a frozen 40-character lowercase commit SHA");
    }
    if !string_at(root, &["experiment", "preregistered_at_utc"])
        .is_some_and(|value| value.ends_with('Z'))
    {
        report.issue("PREREGISTRATION_TIME_INVALID", "preregistered_at_utc must be UTC and end in Z");
    }

    check_arm_bindings(&mut report, root);
    check_budget(&mut report, root);
    check_splits(&mut report, root);
    check_evaluator(&mut report, root);
    check_replication_and_rounds(&mut report, root);
    check_controls_and_analysis(&mut report, root);
    check_authority_and_claims(&mut report, root);

    report.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fill_placeholders(value: &mut Value, key: &str, next_digest: &mut u64) {
        match value {
            Value::String(raw) if PLACEHOLDERS.iter().any(|marker| raw.contains(marker)) => {
                let replacement = if key == "preregistered_at_utc" {
                    "2026-11-06T00:00:00Z".to_owned()
                } else if key == "preregistration_commit" || key == "analysis_code_commit" {
                    "b".repeat(40)
                } else if key == "confirmatory_seed_count" {
                    "10".to_owned()
                } else if key == "planned_outer_loop_steps" {
                    "4".to_owned()
                } else if key == "total_budget" {
                    "10000 USD inclusive".to_owned()
                } else if key.contains("digest") || key.contains("commitment") {
                    let digest = format!("{:064x}", *next_digest);
                    *next_digest += 1;
                    digest
                } else {
                    format!("synthetic-{key}")
                };
                *raw = replacement;
            }
            Value::Array(items) => {
                for item in items {
                    fill_placeholders(item, key, next_digest);
                }
            }
            Value::Object(fields) => {
                for (child_key, child) in fields {
                    fill_placeholders(child, child_key, next_digest);
                }
            }
            _ => {}
        }
    }

    fn valid_specimen() -> Value {
        let source = include_str!("../../../docs/research/templates/SYM_RSI_R4_IGNITION_PREREGISTRATION_V0_2026-11-06.json");
        let mut value: Value = serde_json::from_str(source).expect("template JSON parses");
        let mut next_digest = 1_u64;
        fill_placeholders(&mut value, "", &mut next_digest);
        value["document_status"] = Value::String("PREREGISTERED".to_owned());
        value["seed_and_replication_plan"]["confirmatory_seed_count"] =
            Value::Number(10_u64.into());
        value["rounds"]["planned_outer_loop_steps"] = Value::Number(4_u64.into());

        let shared_budget = value["budget"]["shared_envelope_digest"].clone();
        value["budget"]["arm_budget_bindings"]["evolved_outer_loop"]["envelope_digest"] =
            shared_budget.clone();
        value["budget"]["arm_budget_bindings"]["reference_outer_loop"]["envelope_digest"] =
            shared_budget;

        value
    }

    #[test]
    fn accepts_internally_consistent_synthetic_preregistration() {
        let report = validate_preregistration(&valid_specimen());
        assert!(
            report.ready_for_independent_review,
            "unexpected issues: {:?}",
            report.issues
        );
    }

    #[test]
    fn rejects_template_with_unresolved_placeholders() {
        let template: Value =
            serde_json::from_str(include_str!("../../../docs/research/templates/SYM_RSI_R4_IGNITION_PREREGISTRATION_V0_2026-11-06.json")).expect("template JSON parses");
        let report = validate_preregistration(&template);
        assert!(!report.ready_for_independent_review);
        assert!(report.issues.iter().any(|issue| issue.starts_with("DOCUMENT_NOT_FROZEN")));
        assert!(report.issues.iter().any(|issue| issue.starts_with("UNRESOLVED_PLACEHOLDER")));
    }

    #[test]
    fn rejects_mismatched_inner_start_reference() {
        let mut value = valid_specimen();
        value["arms"][1]["inner_agent_start_ref"] =
            Value::String("starting_state.other_candidate".to_owned());
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("START_REFERENCE_MISMATCH")));
    }

    #[test]
    fn rejects_identical_outer_loop_researcher_artifacts() {
        let mut value = valid_specimen();
        let evolved = value["starting_state"]["outer_loop_arms"]["evolved_researcher"].clone();
        value["starting_state"]["outer_loop_arms"]["reference_researcher"] = evolved;
        let report = validate_preregistration(&value);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.starts_with("IDENTICAL_OUTER_RESEARCHERS")));
    }

    #[test]
    fn rejects_missing_fresh_transfer_custodian() {
        let mut value = valid_specimen();
        value["splits"]["fresh_family_transfer"]["custodian_independent_of_candidate"] =
            Value::String(String::new());
        let report = validate_preregistration(&value);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.starts_with("REQUIRED_STRING_MISSING")));
    }

    #[test]
    fn rejects_missing_task_family_manifest() {
        let mut value = valid_specimen();
        value["task_population"]["task_family_manifest_digest"] =
            Value::String(String::new());
        let report = validate_preregistration(&value);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.starts_with("INVALID_SHA256")));
    }

    #[test]
    fn rejects_unequal_arm_budget_envelopes() {
        let mut value = valid_specimen();
        value["budget"]["arm_budget_bindings"]["reference_outer_loop"]["envelope_digest"] =
            Value::String("f".repeat(64));
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("UNEQUAL_ARM_BUDGET")));
    }

    #[test]
    fn rejects_split_identity_collision() {
        let mut value = valid_specimen();
        let discovery_digest = value["splits"]["discovery"]["manifest_digest"].clone();
        value["splits"]["validation"]["manifest_digest"] = discovery_digest;
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("SPLIT_IDENTITY_COLLISION")));
    }

    #[test]
    fn rejects_non_shared_evaluator_and_candidate_write_access() {
        let mut value = valid_specimen();
        value["evaluator_binding"]["shared_across_arms"] = Value::Bool(false);
        value["evaluator_binding"]["candidate_write_access"] = Value::Bool(true);
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("BOOLEAN_GUARDRAIL_MISMATCH")));
    }

    #[test]
    fn rejects_missing_null_control() {
        let mut value = valid_specimen();
        value["controls"]
            .as_array_mut()
            .expect("controls array")
            .retain(|item| item["control_id"] != "noop_or_null");
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("REQUIRED_CONTROL_MISSING")));
    }

    #[test]
    fn recursive_progress_requires_full_claim_profile() {
        let mut value = valid_specimen();
        value["experiment"]["claim_profile"]["recursive_progress"] = Value::Bool(true);
        let report = validate_preregistration(&value);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.starts_with("RECURSIVE_PROGRESS_CLAIM_DEPENDENCY")));
    }

    #[test]
    fn recursive_progress_requires_two_complete_outer_cycles() {
        let mut value = valid_specimen();
        value["experiment"]["claim_profile"]["recursive_progress"] = Value::Bool(true);
        value["experiment"]["claim_profile"]["accumulation"] = Value::Bool(true);
        value["experiment"]["claim_profile"]["repeatability"] = Value::Bool(true);
        value["experiment"]["claim_profile"]["matched_budget_better_improver"] = Value::Bool(true);
        value["experiment"]["claim_profile"]["outer_loop_ignition"] = Value::Bool(true);
        value["experiment"]["claim_profile"]["fresh_family_transfer"] = Value::Bool(true);
        value["seed_and_replication_plan"]["minimum_complete_outer_cycles"] = Value::Number(1.into());
        let report = validate_preregistration(&value);
        assert!(report.issues.iter().any(|issue| issue.starts_with("OUTER_CYCLE_FLOOR")));
    }
}
