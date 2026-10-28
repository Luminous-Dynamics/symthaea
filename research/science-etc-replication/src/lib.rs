//! Strict manifest validation for a matched-budget NewtonBench replication.
//!
//! This module validates the declared experiment design before scoring. It does
//! not execute NewtonBench, authenticate a hosted model snapshot, or establish
//! that a score demonstrates scientific discovery.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFEST_SCHEMA: &str = "symthaea.sci_etc_replication_manifest.v1";
pub const REPORT_SCHEMA: &str = "symthaea.sci_etc_replication_validation_report.v1";
const REQUIRED_ACQUISITIONS: [&str; 3] = ["variance", "pairwise", "random"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV1 {
    pub schema: String,
    pub study_id: String,
    pub purpose: String,
    pub stage: String,
    pub upstream: UpstreamV1,
    pub evaluator: EvaluatorV1,
    pub models: Vec<ModelIdentityV1>,
    pub task_slices: Vec<TaskSliceV1>,
    pub arms: Vec<ArmV1>,
    pub budget_profiles: BudgetProfilesV1,
    pub analysis_plan_sha256: String,
    pub primary_contrasts: Vec<PrimaryContrastV1>,
    pub missing_attempt_policy: String,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamV1 {
    pub repository: String,
    pub revision: String,
    pub benchmark_scope: String,
    pub native_metrics: Vec<String>,
    pub source_snapshot_status: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorV1 {
    pub evaluator_source_sha256: String,
    pub judge_identity: String,
    pub judge_snapshot_status: String,
    pub judge_prompt_sha256: Option<String>,
    pub prediction_coverage_audit: bool,
    pub coverage_complete_error: bool,
    pub sign_disagreement_audit: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentityV1 {
    pub model_id: String,
    pub provider: String,
    pub snapshot_status: String,
    pub revision_or_alias: String,
    pub temperature: f64,
    pub max_output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSliceV1 {
    pub model_id: String,
    pub modules: Vec<String>,
    pub difficulties: Vec<String>,
    pub model_system: String,
    pub law_version: String,
    pub noise_level: f64,
    pub rollouts_per_cell: u64,
    pub paired_rollout_seeds_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmV1 {
    pub arm_id: String,
    pub protocol: String,
    pub acquisition: String,
    pub interpreter: bool,
    pub automatic_diagnostics: bool,
    pub fresh_synthesis: bool,
    pub draft_retention: String,
    pub batch_limit_enforced: bool,
    pub role: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetProfilesV1 {
    pub measurement_matched: MeasurementMatchedBudgetV1,
    pub interaction_matched: InteractionMatchedBudgetV1,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementMatchedBudgetV1 {
    pub simulator_call_caps: Vec<u64>,
    pub max_measurements_per_action: u64,
    pub final_submission_calls: u64,
    pub reject_over_budget_action: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionMatchedBudgetV1 {
    pub max_model_action_calls: u64,
    pub max_tool_calls: u64,
    pub max_total_tokens: u64,
    pub max_wall_clock_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimaryContrastV1 {
    pub contrast_id: String,
    pub left_arm_id: String,
    pub right_arm_id: String,
    pub primary_metric: String,
    pub budget_profile: String,
    pub multiplicity_family: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReportV1 {
    pub schema: String,
    pub manifest_sha256: String,
    pub valid: bool,
    pub manifest_declares_scoring_eligible: bool,
    pub stage: String,
    pub expected_task_cells_per_budget_profile: u64,
    pub expected_attempts_per_budget_profile: u64,
    pub arm_count: u64,
    pub issues: Vec<String>,
    pub admission_reason: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_zero_digest(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte == b'0')
}

fn check_digest(field: &str, value: &str, qualified: bool, issues: &mut Vec<String>) {
    if !is_lower_hex(value, 64) {
        issues.push(format!("{field} must be 64 lowercase hexadecimal characters"));
    } else if qualified && is_zero_digest(value) {
        issues.push(format!("{field} is an all-zero placeholder in a qualified run"));
    }
}

fn check_git_revision(field: &str, value: &str, issues: &mut Vec<String>) {
    if !is_lower_hex(value, 40) || value.bytes().all(|byte| byte == b'0') {
        issues.push(format!(
            "{field} must be a nonzero, 40-character lowercase Git commit SHA"
        ));
    }
}

fn unique_nonempty(values: &[String], label: &str, issues: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            issues.push(format!("{label} entries must not be empty"));
        }
        if !seen.insert(value.as_str()) {
            issues.push(format!("duplicate {label} entry: {value}"));
        }
    }
}

fn expected_arm_grid(manifest: &RunManifestV1, issues: &mut Vec<String>) {
    let native: Vec<&ArmV1> = manifest
        .arms
        .iter()
        .filter(|arm| arm.protocol == "native_baseline")
        .collect();
    let planner: Vec<&ArmV1> = manifest
        .arms
        .iter()
        .filter(|arm| arm.protocol == "explore_then_commit")
        .collect();
    let ablations: Vec<&ArmV1> = manifest
        .arms
        .iter()
        .filter(|arm| arm.protocol == "component_ablation")
        .collect();

    if native.len() != 2 {
        issues.push(format!(
            "reference grid requires exactly two native baseline arms; found {}",
            native.len()
        ));
    }
    let native_interpreters: BTreeSet<bool> = native.iter().map(|arm| arm.interpreter).collect();
    if native_interpreters != BTreeSet::from([false, true]) {
        issues.push("native baseline grid must include interpreter=false and true".to_owned());
    }
    for arm in native {
        if arm.acquisition != "none"
            || arm.automatic_diagnostics
            || arm.fresh_synthesis
            || arm.role != "primary_baseline"
        {
            issues.push(format!(
                "native baseline {} has inconsistent protocol fields",
                arm.arm_id
            ));
        }
    }

    if planner.len() != 6 {
        issues.push(format!(
            "reference grid requires six explore-then-commit arms; found {}",
            planner.len()
        ));
    }
    let mut seen = BTreeSet::new();
    for arm in planner {
        let combination = (arm.interpreter, arm.acquisition.as_str());
        if !seen.insert(combination) {
            issues.push(format!(
                "duplicate explore-then-commit combination: interpreter={}, acquisition={}",
                arm.interpreter, arm.acquisition
            ));
        }
        if !REQUIRED_ACQUISITIONS.contains(&arm.acquisition.as_str()) {
            issues.push(format!(
                "planner arm {} uses unsupported acquisition {}",
                arm.arm_id, arm.acquisition
            ));
        }
        if !arm.automatic_diagnostics
            || !arm.fresh_synthesis
            || !arm.batch_limit_enforced
        {
            issues.push(format!(
                "planner arm {} must enable diagnostics, fresh synthesis, and enforced batch limits",
                arm.arm_id
            ));
        }
        let required_role = if arm.acquisition == "random" {
            "random_control"
        } else {
            "primary_candidate"
        };
        if arm.role != required_role {
            issues.push(format!(
                "planner arm {} must have role {required_role}",
                arm.arm_id
            ));
        }
        if !matches!(
            arm.draft_retention.as_str(),
            "native" | "retain_unless_disconfirmed" | "replace_if_valid" | "not_applicable"
        ) {
            issues.push(format!("arm {} has an unsupported draft_retention", arm.arm_id));
        }
    }
    let expected: BTreeSet<(bool, &str)> = [
        (false, "variance"),
        (false, "pairwise"),
        (false, "random"),
        (true, "variance"),
        (true, "pairwise"),
        (true, "random"),
    ]
    .into_iter()
    .collect();
    if seen != expected {
        issues.push("explore-then-commit arms must cover all six interpreter × acquisition combinations".to_owned());
    }

    if manifest.stage != "component_ablation" && !ablations.is_empty() {
        issues.push("component_ablation arms require stage=component_ablation".to_owned());
    }
    for arm in ablations {
        if arm.role != "secondary_ablation" {
            issues.push(format!(
                "component ablation {} must have role secondary_ablation",
                arm.arm_id
            ));
        }
    }
    if manifest.arms.iter().any(|arm| !arm.batch_limit_enforced) {
        issues.push("every arm must enforce the same hard experiment-batch limit".to_owned());
    }
    if manifest.arms.iter().any(|arm| {
        !matches!(arm.protocol.as_str(), "native_baseline" | "explore_then_commit" | "component_ablation")
    }) {
        issues.push("arm protocol must be native_baseline, explore_then_commit, or component_ablation".to_owned());
    }
    if manifest.arms.iter().any(|arm| {
        !matches!(arm.acquisition.as_str(), "none" | "random" | "variance" | "pairwise")
    }) {
        issues.push("arm acquisition must be none, random, variance, or pairwise".to_owned());
    }
}

fn count_cells(manifest: &RunManifestV1, issues: &mut Vec<String>) -> u64 {
    let mut total = 0_u64;
    for slice in &manifest.task_slices {
        let modules = match u64::try_from(slice.modules.len()) {
            Ok(value) => value,
            Err(_) => {
                issues.push("module count exceeds u64".to_owned());
                return 0;
            }
        };
        let difficulties = match u64::try_from(slice.difficulties.len()) {
            Ok(value) => value,
            Err(_) => {
                issues.push("difficulty count exceeds u64".to_owned());
                return 0;
            }
        };
        let Some(cells) = modules
            .checked_mul(difficulties)
            .and_then(|value| value.checked_mul(slice.rollouts_per_cell))
        else {
            issues.push("task-cell count overflows u64".to_owned());
            return 0;
        };
        let Some(updated) = total.checked_add(cells) else {
            issues.push("total task-cell count overflows u64".to_owned());
            return 0;
        };
        total = updated;
    }
    total
}

pub fn validate_manifest_bytes(bytes: &[u8]) -> Result<ValidationReportV1, String> {
    let manifest: RunManifestV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("manifest does not match the strict v1 schema: {error}"))?;
    let mut issues = Vec::new();
    let qualified = manifest.purpose == "qualified_run";

    if manifest.schema != MANIFEST_SCHEMA {
        issues.push(format!("schema must be {MANIFEST_SCHEMA}"));
    }
    if manifest.study_id.trim().is_empty() {
        issues.push("study_id must not be empty".to_owned());
    }
    if !matches!(manifest.purpose.as_str(), "protocol_fixture_not_for_scoring" | "qualified_run") {
        issues.push("unsupported purpose".to_owned());
    }
    if !matches!(manifest.stage.as_str(), "baseline_pilot" | "confirmatory" | "component_ablation") {
        issues.push("unsupported stage".to_owned());
    }
    if !matches!(
        manifest.claim_ceiling.as_str(),
        "protocol_fixture_only" | "baseline_pilot_only" | "confirmatory_comparison_pending" | "qualified_comparison"
    ) {
        issues.push("unsupported claim_ceiling".to_owned());
    }
    if qualified && manifest.claim_ceiling == "protocol_fixture_only" {
        issues.push("qualified_run cannot retain protocol_fixture_only claim ceiling".to_owned());
    }
    if !qualified && manifest.claim_ceiling != "protocol_fixture_only" {
        issues.push("protocol fixture must retain protocol_fixture_only claim ceiling".to_owned());
    }

    if manifest.upstream.repository.trim().is_empty() {
        issues.push("upstream.repository must not be empty".to_owned());
    }
    if !matches!(
        manifest.upstream.benchmark_scope.as_str(),
        "newtonbench_direct_output_v0_noise_free" | "newtonbench_extended"
    ) {
        issues.push("unsupported upstream benchmark_scope".to_owned());
    }
    if !matches!(manifest.upstream.source_snapshot_status.as_str(), "pinned" | "unresolved_fixture") {
        issues.push("unsupported upstream source_snapshot_status".to_owned());
    }
    let metrics: BTreeSet<&str> = manifest.upstream.native_metrics.iter().map(String::as_str).collect();
    if !metrics.contains("native_rmsle") || !metrics.contains("symbolic_accuracy") {
        issues.push("upstream.native_metrics must preserve native_rmsle and symbolic_accuracy".to_owned());
    }
    if metrics.len() != manifest.upstream.native_metrics.len()
        || metrics.iter().any(|metric| !matches!(*metric, "native_rmsle" | "symbolic_accuracy"))
    {
        issues.push("upstream.native_metrics must contain only unique native_rmsle and symbolic_accuracy entries".to_owned());
    }
    if qualified {
        check_git_revision("upstream.revision", &manifest.upstream.revision, &mut issues);
        if manifest.upstream.source_snapshot_status != "pinned" {
            issues.push("qualified_run requires a pinned upstream source snapshot".to_owned());
        }
    }

    if manifest.evaluator.judge_identity.trim().is_empty() {
        issues.push("evaluator.judge_identity must not be empty".to_owned());
    }
    if !matches!(
        manifest.evaluator.judge_snapshot_status.as_str(),
        "pinned" | "unresolved" | "not_applicable_fixture"
    ) {
        issues.push("unsupported evaluator judge_snapshot_status".to_owned());
    }
    check_digest(
        "evaluator.evaluator_source_sha256",
        &manifest.evaluator.evaluator_source_sha256,
        qualified,
        &mut issues,
    );
    if let Some(judge_prompt) = &manifest.evaluator.judge_prompt_sha256 {
        check_digest("evaluator.judge_prompt_sha256", judge_prompt, qualified, &mut issues);
    }
    if qualified
        && (manifest.evaluator.judge_snapshot_status != "pinned"
            || manifest.evaluator.judge_prompt_sha256.is_none())
    {
        issues.push("qualified_run requires a pinned judge snapshot and prompt digest".to_owned());
    }
    if !manifest.evaluator.prediction_coverage_audit
        || !manifest.evaluator.coverage_complete_error
        || !manifest.evaluator.sign_disagreement_audit
    {
        issues.push("prediction coverage, coverage-complete error, and sign disagreement audits must be enabled".to_owned());
    }

    if manifest.models.is_empty() {
        issues.push("models must not be empty".to_owned());
    }
    let mut model_ids = BTreeSet::new();
    for model in &manifest.models {
        if model.model_id.trim().is_empty() || !model_ids.insert(model.model_id.as_str()) {
            issues.push(format!("empty or duplicate model_id: {}", model.model_id));
        }
        if model.provider.trim().is_empty() || model.revision_or_alias.trim().is_empty() {
            issues.push(format!("model {} needs provider and revision_or_alias", model.model_id));
        }
        if !model.temperature.is_finite() || !(0.0..=2.0).contains(&model.temperature) {
            issues.push(format!("model {} temperature must be finite and within [0,2]", model.model_id));
        }
        if !matches!(model.snapshot_status.as_str(), "pinned" | "unresolved" | "not_run_fixture") {
            issues.push(format!("model {} has unsupported snapshot_status", model.model_id));
        }
        if qualified && model.snapshot_status != "pinned" {
            issues.push(format!("qualified_run requires immutable snapshot for model {}", model.model_id));
        }
        if model.max_output_tokens == Some(0) {
            issues.push(format!("model {} max_output_tokens must be positive", model.model_id));
        }
    }

    if manifest.task_slices.is_empty() {
        issues.push("task_slices must not be empty".to_owned());
    }
    let mut covered_model_ids = BTreeSet::new();
    let mut slice_identities = BTreeSet::new();
    for (index, slice) in manifest.task_slices.iter().enumerate() {
        let prefix = format!("task_slices[{index}]");
        if !model_ids.contains(slice.model_id.as_str()) {
            issues.push(format!("{prefix} references unknown model {}", slice.model_id));
        }
        covered_model_ids.insert(slice.model_id.as_str());
        let mut modules_key = slice.modules.clone();
        modules_key.sort();
        let mut difficulties_key = slice.difficulties.clone();
        difficulties_key.sort();
        let slice_identity = (
            slice.model_id.clone(),
            modules_key,
            difficulties_key,
            slice.model_system.clone(),
            slice.law_version.clone(),
            slice.noise_level.to_bits(),
            slice.rollouts_per_cell,
        );
        if !slice_identities.insert(slice_identity) {
            issues.push(format!("{prefix} duplicates an existing model/task-slice identity"));
        }
        if slice.modules.is_empty() || slice.difficulties.is_empty() {
            issues.push(format!("{prefix} requires modules and difficulties"));
        }
        unique_nonempty(&slice.modules, &format!("{prefix}.modules"), &mut issues);
        unique_nonempty(&slice.difficulties, &format!("{prefix}.difficulties"), &mut issues);
        if slice.difficulties.iter().any(|value| !matches!(value.as_str(), "easy" | "medium" | "hard")) {
            issues.push(format!("{prefix} has unsupported difficulty"));
        }
        if !matches!(
            slice.model_system.as_str(),
            "vanilla_equation" | "simple_system" | "complex_system"
        ) {
            issues.push(format!("{prefix} has unsupported model_system"));
        }
        if slice.law_version.trim().is_empty() {
            issues.push(format!("{prefix}.law_version must not be empty"));
        }
        if !slice.noise_level.is_finite() || slice.noise_level < 0.0 {
            issues.push(format!("{prefix}.noise_level must be finite and non-negative"));
        }
        if slice.rollouts_per_cell == 0 {
            issues.push(format!("{prefix}.rollouts_per_cell must be greater than zero"));
        }
        check_digest(
            &format!("{prefix}.paired_rollout_seeds_sha256"),
            &slice.paired_rollout_seeds_sha256,
            qualified,
            &mut issues,
        );
    }
    for model_id in &model_ids {
        if !covered_model_ids.contains(model_id) {
            issues.push(format!("model {model_id} has no task slice"));
        }
    }

    if manifest.arms.is_empty() {
        issues.push("arms must not be empty".to_owned());
    }
    let mut arm_ids = BTreeSet::new();
    for arm in &manifest.arms {
        if arm.arm_id.trim().is_empty() || !arm_ids.insert(arm.arm_id.as_str()) {
            issues.push(format!("empty or duplicate arm_id: {}", arm.arm_id));
        }
    }
    expected_arm_grid(&manifest, &mut issues);

    let caps = &manifest.budget_profiles.measurement_matched.simulator_call_caps;
    if caps.is_empty() || caps.iter().any(|cap| *cap == 0) {
        issues.push("measurement_matched simulator_call_caps must be nonempty and positive".to_owned());
    }
    if caps.windows(2).any(|pair| pair[1] <= pair[0]) {
        issues.push("measurement_matched simulator_call_caps must be strictly increasing".to_owned());
    }
    if let Some(minimum_cap) = caps.iter().min() {
        if manifest.budget_profiles.measurement_matched.max_measurements_per_action == 0
            || manifest.budget_profiles.measurement_matched.max_measurements_per_action > *minimum_cap
        {
            issues.push("max_measurements_per_action must be positive and no larger than the smallest simulator-call cap".to_owned());
        }
    }
    if manifest.budget_profiles.measurement_matched.final_submission_calls == 0 {
        issues.push("measurement_matched.final_submission_calls must be positive".to_owned());
    }
    if !manifest.budget_profiles.measurement_matched.reject_over_budget_action {
        issues.push("over-budget experiment actions must be rejected, not silently truncated".to_owned());
    }
    let interaction = &manifest.budget_profiles.interaction_matched;
    if interaction.max_model_action_calls == 0
        || interaction.max_total_tokens == 0
        || interaction.max_wall_clock_seconds == 0
    {
        issues.push("interaction_matched calls, tokens and wall-clock budgets must be positive".to_owned());
    }

    if manifest.analysis_plan_sha256.len() != 64
        || !is_lower_hex(&manifest.analysis_plan_sha256, 64)
        || (qualified && is_zero_digest(&manifest.analysis_plan_sha256))
    {
        issues.push("analysis_plan_sha256 must be a non-placeholder lowercase SHA-256 digest for qualified runs".to_owned());
    }
    if manifest.missing_attempt_policy != "retain_every_expected_cell_and_classify_failures" {
        issues.push("missing_attempt_policy must retain every expected cell and classify failures".to_owned());
    }

    let mut contrast_ids = BTreeSet::new();
    for contrast in &manifest.primary_contrasts {
        if contrast.contrast_id.trim().is_empty() || !contrast_ids.insert(contrast.contrast_id.as_str()) {
            issues.push(format!("empty or duplicate contrast_id: {}", contrast.contrast_id));
        }
        if !arm_ids.contains(contrast.left_arm_id.as_str()) {
            issues.push(format!("contrast {} references unknown left arm {}", contrast.contrast_id, contrast.left_arm_id));
        }
        if !arm_ids.contains(contrast.right_arm_id.as_str()) {
            issues.push(format!("contrast {} references unknown right arm {}", contrast.contrast_id, contrast.right_arm_id));
        }
        if contrast.left_arm_id == contrast.right_arm_id {
            issues.push(format!("contrast {} must compare two distinct arms", contrast.contrast_id));
        }
        if !matches!(
            contrast.primary_metric.as_str(),
            "coverage_complete_rmsle" | "native_rmsle" | "symbolic_accuracy"
                | "complete_prediction_rate" | "measurements_to_validated_law"
                | "total_tokens" | "wall_clock_seconds"
        ) {
            issues.push(format!("contrast {} has unsupported primary_metric", contrast.contrast_id));
        }
        if !matches!(contrast.budget_profile.as_str(), "measurement_matched" | "interaction_matched") {
            issues.push(format!("contrast {} has unsupported budget_profile", contrast.contrast_id));
        }
        if contrast.multiplicity_family.as_ref().is_some_and(|value| value.trim().is_empty()) {
            issues.push(format!("contrast {} has empty multiplicity_family", contrast.contrast_id));
        }
    }
    if manifest.primary_contrasts.is_empty() {
        issues.push("at least one primary contrast must be preregistered".to_owned());
    }

    let expected_cells = count_cells(&manifest, &mut issues);
    let arm_count = u64::try_from(manifest.arms.len()).unwrap_or(u64::MAX);
    let expected_attempts = match expected_cells.checked_mul(arm_count) {
        Some(value) => value,
        None => {
            issues.push("expected attempts per budget profile overflow u64".to_owned());
            0
        }
    };
    if qualified && manifest.stage == "confirmatory" && manifest.primary_contrasts.iter().any(|contrast| contrast.multiplicity_family.is_none()) {
        issues.push("confirmatory contrasts must declare a multiplicity_family".to_owned());
    }

    let valid = issues.is_empty();
    let manifest_declares_scoring_eligible = valid && qualified;
    let admission_reason = if !valid {
        "manifest failed one or more strict validation checks".to_owned()
    } else if !qualified {
        "protocol fixture is structurally valid but not admissible for scoring".to_owned()
    } else if manifest.stage == "baseline_pilot" {
        "manifest is structurally eligible for a baseline pilot; model/runner evidence still needs independent verification".to_owned()
    } else {
        "manifest is structurally eligible; run execution, evaluator validity, and scientific conclusions remain separate gates".to_owned()
    };

    Ok(ValidationReportV1 {
        schema: REPORT_SCHEMA.to_owned(),
        manifest_sha256: sha256_hex(bytes),
        valid,
        manifest_declares_scoring_eligible,
        stage: manifest.stage,
        expected_task_cells_per_budget_profile: expected_cells,
        expected_attempts_per_budget_profile: expected_attempts,
        arm_count,
        issues,
        admission_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn placeholder_digest() -> String {
        "0".repeat(64)
    }

    fn qualified_manifest() -> Value {
        let bytes = include_bytes!("../fixtures/run-manifest.example.json");
        let mut value: Value = serde_json::from_slice(bytes).unwrap();
        value["purpose"] = json!("qualified_run");
        value["claim_ceiling"] = json!("baseline_pilot_only");
        value["upstream"]["revision"] = json!("b".repeat(40));
        value["upstream"]["source_snapshot_status"] = json!("pinned");
        value["evaluator"]["evaluator_source_sha256"] = json!("c".repeat(64));
        value["evaluator"]["judge_snapshot_status"] = json!("pinned");
        value["evaluator"]["judge_prompt_sha256"] = json!("d".repeat(64));
        for model in value["models"].as_array_mut().unwrap() {
            model["snapshot_status"] = json!("pinned");
            let model_id = model["model_id"].as_str().unwrap().to_owned();
            model["revision_or_alias"] = json!(format!("fixture-pinned-{model_id}"));
        }
        for slice in value["task_slices"].as_array_mut().unwrap() {
            slice["paired_rollout_seeds_sha256"] = json!("e".repeat(64));
        }
        value["analysis_plan_sha256"] = json!("f".repeat(64));
        value
    }

    fn validate_value(value: &Value) -> Result<ValidationReportV1, String> {
        validate_manifest_bytes(&serde_json::to_vec(value).expect("manifest serializes"))
    }

    #[test]
    fn example_fixture_is_valid_but_not_scoring_admissible() {
        let report = validate_manifest_bytes(include_bytes!(
            "../fixtures/run-manifest.example.json"
        ))
        .expect("example manifest has strict v1 shape");
        assert!(report.valid, "{:?}", report.issues);
        assert!(!report.manifest_declares_scoring_eligible);
        assert_eq!(report.arm_count, 8);
        assert_eq!(report.expected_task_cells_per_budget_profile, 72);
        assert_eq!(report.expected_attempts_per_budget_profile, 576);
    }

    #[test]
    fn qualified_pilot_manifest_passes_structural_contract() {
        let report = validate_value(&qualified_manifest()).unwrap();
        assert!(report.valid, "{:?}", report.issues);
        assert!(report.manifest_declares_scoring_eligible);
    }

    #[test]
    fn unknown_top_level_field_is_rejected() {
        let mut value = qualified_manifest();
        value["oracle_secret"] = json!("must not be accepted");
        let error = validate_value(&value).expect_err("unknown field must be rejected");
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn duplicate_arm_id_is_rejected() {
        let mut value = qualified_manifest();
        let first_arm_id = value["arms"][0]["arm_id"].clone();
        value["arms"][1]["arm_id"] = first_arm_id;
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate arm_id")));
    }

    #[test]
    fn missing_random_interpreter_arm_is_rejected() {
        let mut value = qualified_manifest();
        let arms = value["arms"].as_array_mut().unwrap();
        arms.retain(|arm| arm["arm_id"] != "etc_random_interpreter");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("six explore-then-commit arms")));
    }

    #[test]
    fn model_snapshot_must_be_pinned_for_qualified_runs() {
        let mut value = qualified_manifest();
        value["models"][0]["snapshot_status"] = json!("unresolved");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("immutable snapshot")));
    }

    #[test]
    fn unknown_task_slice_model_is_rejected() {
        let mut value = qualified_manifest();
        value["task_slices"][0]["model_id"] = json!("unregistered-model");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("references unknown model")));
    }

    #[test]
    fn duplicate_module_in_slice_is_rejected() {
        let mut value = qualified_manifest();
        let first_module = value["task_slices"][0]["modules"][0].clone();
        value["task_slices"][0]["modules"][1] = first_module;
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate")));
    }

    #[test]
    fn unsorted_measurement_caps_are_rejected() {
        let mut value = qualified_manifest();
        value["budget_profiles"]["measurement_matched"]["simulator_call_caps"] = json!([8, 32, 16, 48]);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("strictly increasing")));
    }

    #[test]
    fn over_budget_actions_must_be_rejected_not_truncated() {
        let mut value = qualified_manifest();
        value["budget_profiles"]["measurement_matched"]["reject_over_budget_action"] = json!(false);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("rejected, not silently truncated")));
    }

    #[test]
    fn unknown_contrast_arm_is_rejected() {
        let mut value = qualified_manifest();
        value["primary_contrasts"][0]["right_arm_id"] = json!("imaginary-arm");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("unknown right arm")));
    }

    #[test]
    fn zero_analysis_digest_is_rejected_for_qualified_run() {
        let mut value = qualified_manifest();
        value["analysis_plan_sha256"] = json!(placeholder_digest());
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("analysis_plan_sha256")));
    }

    #[test]
    fn duplicate_acquisition_interpreter_combination_is_rejected() {
        let mut value = qualified_manifest();
        value["arms"][7]["acquisition"] = json!("variance");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("duplicate explore-then-commit combination")));
    }

    #[test]
    fn native_metrics_must_be_preserved() {
        let mut value = qualified_manifest();
        value["upstream"]["native_metrics"] = json!(["coverage_complete_rmsle"]);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("native_rmsle and symbolic_accuracy")));
    }

    #[test]
    fn pinned_upstream_revision_must_be_a_real_sha_shape() {
        let mut value = qualified_manifest();
        value["upstream"]["revision"] = json!("main");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("40-character lowercase Git commit SHA")));
    }

    #[test]
    fn duplicate_task_slice_identity_is_rejected() {
        let mut value = qualified_manifest();
        let duplicate = value["task_slices"][0].clone();
        value["task_slices"].as_array_mut().unwrap().push(duplicate);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("duplicates an existing model/task-slice identity")));
    }

    #[test]
    fn every_model_requires_at_least_one_task_slice() {
        let mut value = qualified_manifest();
        value["task_slices"]
            .as_array_mut()
            .unwrap()
            .retain(|slice| slice["model_id"] != "GPT-4.1");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.contains("has no task slice")));
    }

    #[test]
    fn duplicate_native_metric_is_rejected() {
        let mut value = qualified_manifest();
        value["upstream"]["native_metrics"] =
            json!(["native_rmsle", "native_rmsle", "symbolic_accuracy"]);
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("only unique native_rmsle")));
    }

    #[test]
    fn schema_document_is_json_and_declares_strict_manifest_fields() {
        let schema: Value = serde_json::from_str(include_str!(
            "../schema/run-manifest.schema.json"
        ))
        .expect("schema JSON");
        assert_eq!(
            schema["properties"]["schema"]["const"].as_str(),
            Some(MANIFEST_SCHEMA)
        );
        assert_eq!(schema["additionalProperties"].as_bool(), Some(false));
        let required = schema["required"].as_array().expect("required array");
        for name in [
            "schema",
            "purpose",
            "upstream",
            "task_slices",
            "arms",
            "budget_profiles",
            "primary_contrasts",
        ] {
            assert!(required.iter().any(|field| field.as_str() == Some(name)));
        }
    }

    #[test]
    fn confirmatory_run_requires_multiplicity_family_on_each_primary_contrast() {
        let mut value = qualified_manifest();
        value["stage"] = json!("confirmatory");
        value["claim_ceiling"] = json!("confirmatory_comparison_pending");
        value["primary_contrasts"][0]
            .as_object_mut()
            .unwrap()
            .remove("multiplicity_family");
        let report = validate_value(&value).unwrap();
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| {
            issue.contains("confirmatory contrasts must declare a multiplicity_family")
        }));
    }

    #[test]
    fn sha256_known_answer_matches() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
