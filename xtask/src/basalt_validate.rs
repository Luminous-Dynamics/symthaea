use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// A deterministic report about record validity. It is not a benchmark score.
#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub report_version: u32,
    pub input_digest: String,
    pub valid: bool,
    pub counts: Counts,
    pub errors: Vec<ValidationError>,
}

#[derive(Debug, Default, Serialize)]
pub struct Counts {
    pub tasks_total: usize,
    pub tasks_accepted: usize,
    pub tasks_quarantined: usize,
    pub tasks_invalid: usize,
    pub trials_expected: usize,
    pub trials_accounted: usize,
    pub trials_missing: usize,
    pub trials_duplicate: usize,
}

#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunRecord {
    schema_version: u32,
    program: String,
    run_id: String,
    created_at: String,
    source: Source,
    environment: Environment,
    suite: Suite,
    systems: Vec<System>,
    task_records: Vec<TaskRecord>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repository: String,
    base_revision: String,
    candidate_revision: String,
    manifest_digest: String,
    orchestrator_digest: String,
    verifier_bundle_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    image_digest: String,
    os: String,
    architecture: String,
    network_policy: String,
    workspace_policy: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    suite_id: String,
    version: String,
    split: String,
    seed_policy: SeedPolicy,
    attempts_per_task: usize,
    expected_task_count: usize,
    oracle_policy: OraclePolicy,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SeedPolicy {
    FixedPaired,
    PredeclaredRandom,
    Unseeded,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OraclePolicy {
    IndependentFrozen,
    HumanReviewed,
    Hybrid,
    NotQualified,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct System {
    system_id: String,
    role: SystemRole,
    agent_name: String,
    agent_version: String,
    config_digest: String,
    budget: Budget,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SystemRole {
    Baseline,
    Candidate,
    Ablation,
    ExternalReference,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Budget {
    max_wall_ms: u64,
    max_tool_calls: u64,
    max_cost_usd: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskRecord {
    task_id: String,
    family: String,
    languages: Vec<String>,
    base_tree_digest: String,
    spec_digest: String,
    #[serde(default)]
    paired_transform_id: Option<String>,
    quality_audit: QualityAudit,
    trials: Vec<Trial>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityAudit {
    status: TaskAuditStatus,
    audit_digest: String,
    oracle_independent: bool,
    ambiguity: Ambiguity,
    notes: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskAuditStatus {
    Accepted,
    Quarantined,
    Invalid,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Ambiguity {
    None,
    ResolvedBeforeRun,
    MaterialUnresolved,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Trial {
    system_id: String,
    attempt_index: usize,
    seed: Seed,
    paired_group_id: String,
    outcome: Outcome,
    verifier: Verifier,
    integrity: Integrity,
    usage: Usage,
    trajectory_digest: String,
    patch_digest: RequiredNullable<String>,
    final_tree_digest: RequiredNullable<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
enum Seed {
    Integer(i64),
    String(String),
}

// Option<T> fields can be omitted during Serde deserialization. This wrapper
// preserves the schema's distinction between a required nullable property and
// an optional property: patch_digest/final_tree_digest must exist, even if null.
#[derive(Debug)]
struct RequiredNullable<T>(Option<T>);

impl<'de, T> Deserialize<'de> for RequiredNullable<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verifier {
    status: VerifierStatus,
    bundle_digest: String,
    isolated_from_agent: bool,
    independent_of_generator: bool,
    evaluated_tree_digest: String,
    checks: Vec<Check>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum VerifierStatus {
    Pass,
    Fail,
    Unresolved,
    NotRun,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Check {
    check_id: String,
    kind: String,
    blocking: bool,
    status: CheckStatus,
    #[serde(default)]
    evidence_digest: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Pass,
    Fail,
    Unresolved,
    NotRun,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Integrity {
    status: IntegrityStatus,
    solution_material_access: EvidenceFlag,
    test_or_verifier_tampering: EvidenceFlag,
    scope_violation: EvidenceFlag,
    stale_evidence: EvidenceFlag,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum IntegrityStatus {
    Clean,
    Violation,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceFlag {
    NotDetected,
    Detected,
    NotChecked,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Usage {
    wall_ms: u64,
    tool_calls: u64,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    reasoning_tokens: Option<u64>,
    cost_usd: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Pass,
    Fail,
    Unresolved,
    InfraError,
    InvalidTask,
    NotRun,
}

pub fn validate_file(path: &Path) -> anyhow::Result<ValidationReport> {
    let bytes = fs::read(path)?;
    Ok(validate_bytes(&bytes))
}

fn validate_bytes(bytes: &[u8]) -> ValidationReport {
    let input_digest = format!("sha256:{}", sha256_hex(bytes));
    let mut report = ValidationReport {
        report_version: 1,
        input_digest,
        valid: false,
        counts: Counts::default(),
        errors: Vec::new(),
    };

    let run: RunRecord = match serde_json::from_slice(bytes) {
        Ok(run) => run,
        Err(error) => {
            push_error(
                &mut report.errors,
                "E_JSON_OR_SHAPE",
                "$",
                format!("invalid JSON, unknown field, duplicate field, or missing required field: {error}"),
            );
            return report;
        }
    };

    validate_run(&run, &mut report);
    report.valid = report.errors.is_empty();
    report
}

fn validate_run(run: &RunRecord, report: &mut ValidationReport) {
    if run.schema_version != 1 {
        error(report, "E_SCHEMA_VERSION", "schema_version", "only schema_version=1 is supported");
    }
    if run.program != "BASALT" {
        error(report, "E_PROGRAM", "program", "program must be BASALT");
    }
    if !valid_run_id(&run.run_id) {
        error(report, "E_RUN_ID", "run_id", "run_id must match the documented 3–128 character identifier grammar");
    }
    if !valid_rfc3339(&run.created_at) {
        error(report, "E_CREATED_AT", "created_at", "created_at must be an RFC3339 date-time");
    }

    if run.source.repository.trim().is_empty() {
        error(report, "E_REPOSITORY_EMPTY", "source.repository", "repository identity must not be empty");
    }
    if !valid_revision(&run.source.base_revision) {
        error(report, "E_BASE_REVISION", "source.base_revision", "base revision must be 40 or 64 lowercase hexadecimal characters");
    }
    if !valid_revision(&run.source.candidate_revision) {
        error(report, "E_CANDIDATE_REVISION", "source.candidate_revision", "candidate revision must be 40 or 64 lowercase hexadecimal characters");
    }
    validate_digest(report, "source.manifest_digest", &run.source.manifest_digest);
    validate_digest(report, "source.orchestrator_digest", &run.source.orchestrator_digest);
    validate_digest(report, "source.verifier_bundle_digest", &run.source.verifier_bundle_digest);

    validate_digest(report, "environment.image_digest", &run.environment.image_digest);
    if run.environment.os.trim().is_empty() || run.environment.architecture.trim().is_empty() {
        error(report, "E_ENVIRONMENT_IDENTITY", "environment", "OS and architecture must be non-empty");
    }
    if !matches!(run.environment.network_policy.as_str(), "disabled" | "allowlisted" | "unrestricted" | "not_recorded") {
        error(report, "E_NETWORK_POLICY", "environment.network_policy", "unknown network policy");
    }
    if !matches!(run.environment.workspace_policy.as_str(), "clean_checkout" | "prepared_workspace" | "persistent_workspace" | "not_recorded") {
        error(report, "E_WORKSPACE_POLICY", "environment.workspace_policy", "unknown workspace policy");
    }

    if run.suite.suite_id.trim().is_empty() || run.suite.version.trim().is_empty() {
        error(report, "E_SUITE_IDENTITY", "suite", "suite_id and version must be non-empty");
    }
    if !matches!(run.suite.split.as_str(), "development" | "validation" | "test" | "holdout" | "external") {
        error(report, "E_SPLIT", "suite.split", "unknown dataset split");
    }
    let oracle_qualified = match run.suite.oracle_policy {
        OraclePolicy::IndependentFrozen | OraclePolicy::HumanReviewed | OraclePolicy::Hybrid => true,
        OraclePolicy::NotQualified => false,
    };
    if run.suite.attempts_per_task == 0 {
        error(report, "E_ATTEMPTS_ZERO", "suite.attempts_per_task", "attempts_per_task must be at least one");
    }
    if run.suite.attempts_per_task > 10_000 {
        error(report, "E_ATTEMPTS_LIMIT", "suite.attempts_per_task", "attempt count exceeds validator safety limit (10000)");
    }

    report.counts.tasks_total = run.task_records.len();
    report.counts.tasks_accepted = run.task_records.iter().filter(|t| t.quality_audit.status == TaskAuditStatus::Accepted).count();
    report.counts.tasks_quarantined = run.task_records.iter().filter(|t| t.quality_audit.status == TaskAuditStatus::Quarantined).count();
    report.counts.tasks_invalid = run.task_records.iter().filter(|t| t.quality_audit.status == TaskAuditStatus::Invalid).count();
    report.counts.trials_accounted = run.task_records.iter().map(|t| t.trials.len()).sum();

    if run.task_records.is_empty() {
        error(report, "E_TASKS_EMPTY", "task_records", "at least one task record is required");
    }
    if run.suite.expected_task_count != run.task_records.len() {
        error(report, "E_TASK_COUNT_MISMATCH", "suite.expected_task_count", "expected_task_count must equal the number of retained task records, including quarantined and invalid records");
    }

    if run.systems.len() < 2 {
        error(report, "E_SYSTEMS_INSUFFICIENT", "systems", "paired evaluation requires at least two system arms");
    }
    let mut system_ids = BTreeSet::new();
    let mut has_baseline = false;
    let mut has_candidate = false;
    let mut first_budget: Option<&Budget> = None;
    for (index, system) in run.systems.iter().enumerate() {
        let path = format!("systems[{index}]");
        if system.system_id.trim().is_empty() || !system_ids.insert(system.system_id.clone()) {
            error(report, "E_DUPLICATE_SYSTEM_ID", format!("{path}.system_id"), "system_id must be non-empty and unique");
        }
        if system.agent_name.trim().is_empty() || system.agent_version.trim().is_empty() {
            error(report, "E_SYSTEM_IDENTITY", &path, "agent_name and agent_version must be non-empty");
        }
        validate_digest(report, &format!("{path}.config_digest"), &system.config_digest);
        if system.budget.max_wall_ms == 0 || system.budget.max_tool_calls == 0 || !system.budget.max_cost_usd.is_finite() || system.budget.max_cost_usd < 0.0 {
            error(report, "E_BUDGET_INVALID", format!("{path}.budget"), "budget limits must be positive and max_cost_usd must be finite and non-negative");
        }
        if let Some(budget) = first_budget {
            if budget.max_wall_ms != system.budget.max_wall_ms
                || budget.max_tool_calls != system.budget.max_tool_calls
                || budget.max_cost_usd != system.budget.max_cost_usd
            {
                error(report, "E_BUDGET_MISMATCH", format!("{path}.budget"), "all paired systems must use the same declared wall-time, tool-call, and cost limits");
            }
        } else {
            first_budget = Some(&system.budget);
        }
        has_baseline |= system.role == SystemRole::Baseline;
        has_candidate |= system.role == SystemRole::Candidate;
    }
    if !has_baseline {
        error(report, "E_BASELINE_MISSING", "systems", "at least one baseline system is required");
    }
    if !has_candidate {
        error(report, "E_CANDIDATE_MISSING", "systems", "at least one candidate system is required");
    }

    let per_task_expected = run.systems.len().saturating_mul(run.suite.attempts_per_task);
    report.counts.trials_expected = run.task_records.len().saturating_mul(per_task_expected);

    let mut task_ids = BTreeSet::new();
    let mut group_owners: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for (task_index, task) in run.task_records.iter().enumerate() {
        let task_path = format!("task_records[{task_index}]");
        if task.task_id.trim().is_empty() || !task_ids.insert(task.task_id.clone()) {
            error(report, "E_DUPLICATE_TASK_ID", format!("{task_path}.task_id"), "task_id must be non-empty and unique");
        }
        if task.family.trim().is_empty() {
            error(report, "E_TASK_FAMILY", format!("{task_path}.family"), "task family must be non-empty");
        }
        if task.languages.is_empty() || task.languages.iter().any(|l| l.trim().is_empty()) {
            error(report, "E_TASK_LANGUAGES", format!("{task_path}.languages"), "at least one non-empty language is required");
        }
        let mut languages = BTreeSet::new();
        for language in &task.languages {
            if !languages.insert(language) {
                error(report, "E_DUPLICATE_LANGUAGE", format!("{task_path}.languages"), "language labels must be unique within a task");
            }
        }
        validate_digest(report, &format!("{task_path}.base_tree_digest"), &task.base_tree_digest);
        validate_digest(report, &format!("{task_path}.spec_digest"), &task.spec_digest);
        validate_digest(report, &format!("{task_path}.quality_audit.audit_digest"), &task.quality_audit.audit_digest);
        if task.quality_audit.notes.trim().is_empty() {
            error(report, "E_AUDIT_NOTES_EMPTY", format!("{task_path}.quality_audit.notes"), "audit notes must explain the disposition");
        }
        if task.quality_audit.status == TaskAuditStatus::Accepted {
            if !task.quality_audit.oracle_independent {
                error(report, "E_ORACLE_NOT_INDEPENDENT", format!("{task_path}.quality_audit.oracle_independent"), "an accepted task requires an independent oracle");
            }
            if task.quality_audit.ambiguity == Ambiguity::MaterialUnresolved {
                error(report, "E_AMBIGUOUS_TASK_ACCEPTED", format!("{task_path}.quality_audit.ambiguity"), "material unresolved ambiguity is incompatible with an accepted task");
            }
            if !oracle_qualified {
                error(report, "E_ORACLE_POLICY_UNQUALIFIED", "suite.oracle_policy", "accepted tasks require a qualified oracle policy");
            }
        }

        let mut cells: BTreeMap<(usize, String), &Trial> = BTreeMap::new();
        let mut actual_by_attempt: BTreeMap<usize, Vec<&Trial>> = BTreeMap::new();
        for (trial_index, trial) in task.trials.iter().enumerate() {
            let trial_path = format!("{task_path}.trials[{trial_index}]");
            if !system_ids.contains(&trial.system_id) {
                error(report, "E_SYSTEM_UNDECLARED", format!("{trial_path}.system_id"), "trial references a system_id not declared in systems");
            }
            if trial.attempt_index == 0 || trial.attempt_index > run.suite.attempts_per_task {
                error(report, "E_ATTEMPT_OUT_OF_RANGE", format!("{trial_path}.attempt_index"), "attempt_index must be in 1..=attempts_per_task");
            }
            if let Seed::String(value) = &trial.seed {
                if value.trim().is_empty() {
                    error(report, "E_SEED_EMPTY", format!("{trial_path}.seed"), "string seeds must be non-empty");
                }
            }
            if trial.paired_group_id.trim().is_empty() {
                error(report, "E_PAIR_ID_EMPTY", format!("{trial_path}.paired_group_id"), "paired_group_id must be non-empty");
            } else {
                let owner = (task.task_id.clone(), trial.attempt_index);
                if let Some(previous) = group_owners.get(&trial.paired_group_id) {
                    if previous != &owner {
                        error(report, "E_PAIR_ID_REUSED", format!("{trial_path}.paired_group_id"), "paired_group_id must identify exactly one task and attempt");
                    }
                } else {
                    group_owners.insert(trial.paired_group_id.clone(), owner);
                }
            }

            let key = (trial.attempt_index, trial.system_id.clone());
            if cells.insert(key, trial).is_some() {
                report.counts.trials_duplicate += 1;
                error(report, "E_TRIAL_DUPLICATE", &trial_path, "duplicate task_id × attempt_index × system_id cell");
            }
            actual_by_attempt.entry(trial.attempt_index).or_default().push(trial);
            validate_trial(
                report,
                &trial_path,
                trial,
                &task.quality_audit,
                &run.source.verifier_bundle_digest,
            );
        }

        if run.suite.attempts_per_task <= 10_000 {
            for attempt in 1..=run.suite.attempts_per_task {
                for system in &run.systems {
                    let key = (attempt, system.system_id.clone());
                    if !cells.contains_key(&key) {
                        report.counts.trials_missing += 1;
                        error(report, "E_TRIAL_MISSING", format!("{task_path}.trials"), format!("missing record for attempt_index={attempt}, system_id={}", system.system_id));
                    }
                }
                if let Some(rows) = actual_by_attempt.get(&attempt) {
                    if let Some(first) = rows.first() {
                        for other in rows.iter().skip(1) {
                            if first.paired_group_id != other.paired_group_id {
                                error(report, "E_PAIR_ID_MISMATCH", format!("{task_path}.trials"), format!("all system arms in attempt {attempt} must share paired_group_id"));
                            }
                            if run.suite.seed_policy == SeedPolicy::FixedPaired && first.seed != other.seed {
                                error(report, "E_PAIR_SEED_MISMATCH", format!("{task_path}.trials"), format!("fixed_paired arms must share the same seed for attempt {attempt}"));
                            }
                        }
                    }
                }
            }
        }

        if task.quality_audit.status == TaskAuditStatus::Invalid {
            for (trial_index, trial) in task.trials.iter().enumerate() {
                if trial.outcome != Outcome::InvalidTask {
                    error(report, "E_INVALID_TASK_OUTCOME", format!("{task_path}.trials[{trial_index}].outcome"), "every trial for a task audited invalid must remain recorded as invalid_task");
                }
            }
        }
        for (trial_index, trial) in task.trials.iter().enumerate() {
            if trial.outcome == Outcome::InvalidTask && task.quality_audit.status != TaskAuditStatus::Invalid {
                error(report, "E_INVALID_TASK_UNAUDITED", format!("{task_path}.trials[{trial_index}].outcome"), "invalid_task outcome requires an invalid task-quality disposition");
            }
        }
    }

    // Read the optional note so the field is not silently treated as authoritative metadata.
    if let Some(note) = &run.notes {
        if note.len() > 16_384 {
            error(report, "E_NOTES_LIMIT", "notes", "notes exceed 16384 bytes");
        }
    }
}

fn validate_trial(
    report: &mut ValidationReport,
    path: &str,
    trial: &Trial,
    task_audit: &QualityAudit,
    pinned_verifier_digest: &str,
) {
    validate_digest(report, &format!("{path}.trajectory_digest"), &trial.trajectory_digest);
    if let Some(digest) = &trial.patch_digest.0 {
        validate_digest(report, &format!("{path}.patch_digest"), digest);
    }
    if let Some(digest) = &trial.final_tree_digest.0 {
        validate_digest(report, &format!("{path}.final_tree_digest"), digest);
    }

    let verifier_path = format!("{path}.verifier");
    validate_digest(report, &format!("{verifier_path}.bundle_digest"), &trial.verifier.bundle_digest);
    validate_digest(report, &format!("{verifier_path}.evaluated_tree_digest"), &trial.verifier.evaluated_tree_digest);
    if trial.verifier.bundle_digest != pinned_verifier_digest {
        error(report, "E_VERIFIER_BUNDLE_MISMATCH", format!("{verifier_path}.bundle_digest"), "trial verifier bundle differs from the run-pinned verifier bundle");
    }
    if let Some(final_digest) = &trial.final_tree_digest.0 {
        if trial.verifier.evaluated_tree_digest != *final_digest {
            error(report, "E_FINAL_TREE_DIGEST_MISMATCH", format!("{verifier_path}.evaluated_tree_digest"), "verifier must evaluate the exact recorded final tree");
        }
    }

    let mut check_ids = BTreeSet::new();
    let mut blocking_count = 0usize;
    for (check_index, check) in trial.verifier.checks.iter().enumerate() {
        let check_path = format!("{verifier_path}.checks[{check_index}]");
        if check.check_id.trim().is_empty() || !check_ids.insert(check.check_id.clone()) {
            error(report, "E_DUPLICATE_CHECK_ID", format!("{check_path}.check_id"), "check_id must be non-empty and unique within a verifier result");
        }
        if !matches!(check.kind.as_str(), "build" | "test" | "typecheck" | "lint" | "security" | "scope" | "regression" | "oracle" | "other") {
            error(report, "E_CHECK_KIND", format!("{check_path}.kind"), "unknown verifier check kind");
        }
        if let Some(digest) = &check.evidence_digest {
            validate_digest(report, &format!("{check_path}.evidence_digest"), digest);
        }
        if check.blocking {
            blocking_count += 1;
        }
        if check.status == CheckStatus::Pass && check.evidence_digest.is_none() {
            error(report, "E_CHECK_EVIDENCE_MISSING", format!("{check_path}.evidence_digest"), "a passing check requires a content digest for its evidence");
        }
        if trial.verifier.status == VerifierStatus::Pass && check.blocking && check.status != CheckStatus::Pass {
            error(report, "E_VERIFIER_BLOCKING_CHECK_FAILED", format!("{check_path}.status"), "verifier cannot pass while a blocking check is not pass");
        }
    }
    if trial.verifier.status == VerifierStatus::Pass {
        if !trial.verifier.isolated_from_agent || !trial.verifier.independent_of_generator {
            error(report, "E_VERIFIER_NOT_INDEPENDENT", &verifier_path, "a passing verifier must be isolated from the agent and independent of the generator");
        }
        if blocking_count == 0 {
            error(report, "E_NO_BLOCKING_CHECKS", format!("{verifier_path}.checks"), "a passing verifier requires at least one blocking check");
        }
    }

    validate_integrity(report, &format!("{path}.integrity"), &trial.integrity);
    if !trial.usage.cost_usd.is_finite() || trial.usage.cost_usd < 0.0 {
        error(report, "E_USAGE_COST", format!("{path}.usage.cost_usd"), "cost_usd must be finite and non-negative");
    }
    let _optional_token_fields = (
        trial.usage.input_tokens,
        trial.usage.output_tokens,
        trial.usage.reasoning_tokens,
    );

    match trial.outcome {
        Outcome::Pass => {
            if trial.verifier.status != VerifierStatus::Pass {
                error(report, "E_PASS_WITHOUT_VERIFIER_PASS", format!("{path}.outcome"), "pass outcome requires verifier status pass");
            }
            if trial.integrity.status != IntegrityStatus::Clean {
                error(report, "E_PASS_NOT_CLEAN", format!("{path}.integrity.status"), "pass outcome requires clean integrity status");
            }
            if trial.final_tree_digest.0.is_none() {
                error(report, "E_PASS_NO_FINAL_TREE", format!("{path}.final_tree_digest"), "pass outcome requires a final-tree digest");
            }
            if task_audit.status == TaskAuditStatus::Invalid {
                error(report, "E_PASS_INVALID_TASK", format!("{path}.outcome"), "an invalid task cannot be represented as a pass");
            }
        }
        Outcome::Fail => {
            if trial.verifier.status != VerifierStatus::Fail {
                error(report, "E_FAIL_VERIFIER_CONTRADICTION", format!("{path}.outcome"), "fail outcome requires verifier status fail");
            }
        }
        Outcome::Unresolved => {
            if !matches!(trial.verifier.status, VerifierStatus::Unresolved | VerifierStatus::NotRun) {
                error(report, "E_UNRESOLVED_VERIFIER_CONTRADICTION", format!("{path}.outcome"), "unresolved outcome requires unresolved or not_run verifier status");
            }
        }
        Outcome::InfraError | Outcome::NotRun => {
            if trial.verifier.status != VerifierStatus::NotRun {
                error(report, "E_NOT_RUN_VERIFIER_CONTRADICTION", format!("{path}.outcome"), "infra_error/not_run outcome requires verifier status not_run");
            }
        }
        Outcome::InvalidTask => {
            if trial.verifier.status != VerifierStatus::NotRun {
                error(report, "E_INVALID_TASK_VERIFIER_RAN", format!("{path}.verifier.status"), "invalid task disposition must not produce scored verifier execution");
            }
        }
    }
}

fn validate_integrity(report: &mut ValidationReport, path: &str, integrity: &Integrity) {
    let flags = [
        integrity.solution_material_access,
        integrity.test_or_verifier_tampering,
        integrity.scope_violation,
        integrity.stale_evidence,
    ];
    let any_detected = flags.contains(&EvidenceFlag::Detected);
    let all_not_detected = flags.iter().all(|f| *f == EvidenceFlag::NotDetected);

    match integrity.status {
        IntegrityStatus::Clean if !all_not_detected => {
            error(report, "E_INTEGRITY_CLAIM_MISMATCH", path, "clean integrity requires every integrity control to be not_detected");
        }
        IntegrityStatus::Violation if !any_detected => {
            error(report, "E_INTEGRITY_VIOLATION_UNSUPPORTED", path, "violation status requires at least one explicitly detected integrity violation");
        }
        IntegrityStatus::Unresolved if any_detected => {
            error(report, "E_INTEGRITY_VIOLATION_UNDERSTATED", path, "detected integrity violation cannot be labelled unresolved");
        }
        _ => {}
    }
}

fn validate_digest(report: &mut ValidationReport, path: &str, value: &str) {
    if !is_sha256_digest(value) {
        error(report, "E_DIGEST_FORMAT", path, "digest must be sha256: followed by 64 lowercase hexadecimal characters");
    }
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_revision(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_run_id(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (3..=128).contains(&value.len())
        && first.is_ascii_alphanumeric()
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'))
}

fn valid_rfc3339(value: &str) -> bool {
    let bytes = value.as_bytes();
    if !bytes.is_ascii()
        || bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || !matches!(bytes.get(10), Some(&b'T') | Some(&b't'))
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return false;
    }
    let Some(year) = decimal(&value[0..4]) else { return false };
    let Some(month) = decimal(&value[5..7]) else { return false };
    let Some(day) = decimal(&value[8..10]) else { return false };
    let Some(hour) = decimal(&value[11..13]) else { return false };
    let Some(minute) = decimal(&value[14..16]) else { return false };
    let Some(second) = decimal(&value[17..19]) else { return false };
    if year == 0 || !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 60 {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    if day == 0 || day > max_day {
        return false;
    }

    let rest = &value[19..];
    let zone_pos = if rest.ends_with('Z') || rest.ends_with('z') {
        value.len() - 1
    } else {
        let Some(relative) = rest.char_indices().find_map(|(i, c)| {
            if c == '+' || c == '-' { Some(i) } else { None }
        }) else {
            return false;
        };
        19 + relative
    };
    let fractional = &value[19..zone_pos];
    if !fractional.is_empty()
        && !(fractional.starts_with('.') && fractional.len() > 1 && fractional[1..].bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    if value.ends_with('Z') || value.ends_with('z') {
        return true;
    }
    let zone = &value[zone_pos..];
    if zone.len() != 6 || !matches!(zone.as_bytes()[0], b'+' | b'-') || zone.as_bytes()[3] != b':' {
        return false;
    }
    let Some(offset_hour) = decimal(&zone[1..3]) else { return false };
    let Some(offset_minute) = decimal(&zone[4..6]) else { return false };
    offset_hour <= 23 && offset_minute <= 59
}

fn decimal(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn error(report: &mut ValidationReport, code: &'static str, path: impl AsRef<str>, message: impl Into<String>) {
    push_error(&mut report.errors, code, path.as_ref(), message);
}

fn push_error(errors: &mut Vec<ValidationError>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    errors.push(ValidationError {
        code: code.to_string(),
        path: path.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const EXAMPLE: &str = include_str!("../../benchmarks/basalt/examples/run-record.example.json");

    fn validate_value(mutator: impl FnOnce(&mut Value)) -> ValidationReport {
        let mut value: Value = serde_json::from_str(EXAMPLE).expect("synthetic fixture parses");
        mutator(&mut value);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        validate_bytes(&bytes)
    }

    fn has_code(report: &ValidationReport, code: &str) -> bool {
        report.errors.iter().any(|error| error.code == code)
    }

    #[test]
    fn synthetic_paired_fixture_passes_structure_and_invariants() {
        let report = validate_bytes(EXAMPLE.as_bytes());
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(report.counts.tasks_total, 1);
        assert_eq!(report.counts.trials_expected, 2);
        assert_eq!(report.counts.trials_accounted, 2);
        assert_eq!(report.counts.trials_missing, 0);
    }

    #[test]
    fn duplicate_system_identity_is_rejected() {
        let report = validate_value(|v| {
            let candidate = v["systems"][1].clone();
            v["systems"][0] = candidate;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_DUPLICATE_SYSTEM_ID"));
    }

    #[test]
    fn undeclared_trial_system_is_rejected() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["system_id"] = Value::String("ghost".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_SYSTEM_UNDECLARED"));
    }

    #[test]
    fn missing_paired_arm_is_rejected_and_counted() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"].as_array_mut().unwrap().pop();
        });
        assert!(!report.valid);
        assert_eq!(report.counts.trials_missing, 1);
        assert!(has_code(&report, "E_TRIAL_MISSING"));
    }

    #[test]
    fn duplicate_trial_cell_is_rejected_and_counted() {
        let report = validate_value(|v| {
            let row = v["task_records"][0]["trials"][0].clone();
            v["task_records"][0]["trials"].as_array_mut().unwrap().push(row);
        });
        assert!(!report.valid);
        assert_eq!(report.counts.trials_duplicate, 1);
        assert!(has_code(&report, "E_TRIAL_DUPLICATE"));
    }

    #[test]
    fn paired_seed_mismatch_is_rejected() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][0]["seed"] = Value::from(999);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_PAIR_SEED_MISMATCH"));
    }

    #[test]
    fn verifier_bundle_mismatch_is_rejected() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["bundle_digest"] = Value::String(format!("sha256:{}", "f".repeat(64)));
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_VERIFIER_BUNDLE_MISMATCH"));
    }

    #[test]
    fn verifier_cannot_pass_with_a_failed_blocking_check() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][0]["status"] = Value::String("fail".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_VERIFIER_BLOCKING_CHECK_FAILED"));
    }

    #[test]
    fn integrity_violation_cannot_be_recorded_as_clean_or_passed() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["integrity"]["scope_violation"] = Value::String("detected".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_INTEGRITY_CLAIM_MISMATCH"));
    }

    #[test]
    fn exact_final_tree_must_be_evaluated() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["evaluated_tree_digest"] = Value::String(format!("sha256:{}", "f".repeat(64)));
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_FINAL_TREE_DIGEST_MISMATCH"));
    }

    #[test]
    fn unknown_schema_version_fails_closed() {
        let report = validate_value(|v| {
            v["schema_version"] = Value::from(2);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_SCHEMA_VERSION"));
    }

    #[test]
    fn report_is_deterministic_for_identical_input_bytes() {
        let first = serde_json::to_vec(&validate_bytes(EXAMPLE.as_bytes())).unwrap();
        let second = serde_json::to_vec(&validate_bytes(EXAMPLE.as_bytes())).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn accepted_task_cannot_have_unresolved_material_ambiguity() {
        let report = validate_value(|v| {
            v["task_records"][0]["quality_audit"]["ambiguity"] = Value::String("material_unresolved".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_AMBIGUOUS_TASK_ACCEPTED"));
    }

    #[test]
    fn passing_checks_require_content_addressed_evidence() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][0].as_object_mut().unwrap().remove("evidence_digest");
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_CHECK_EVIDENCE_MISSING"));
    }
}
