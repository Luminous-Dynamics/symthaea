use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

const MAX_RUN_RECORD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SUITE_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TASK_SPEC_BYTES: u64 = 4 * 1024 * 1024;

/// A deterministic report about record validity. It is not a benchmark score.
#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub report_version: u32,
    pub input_digest: String,
    pub manifest_digest: String,
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
    notes: OptionalNonNull<String>,
}


#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteManifest {
    schema_version: u32,
    program: String,
    suite_id: String,
    version: String,
    split: String,
    seed_policy: SeedPolicy,
    attempts_per_task: usize,
    attempt_seeds: Vec<Seed>,
    tasks: Vec<ManifestTask>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestTask {
    task_id: String,
    family: String,
    languages: Vec<String>,
    base_tree_digest: String,
    spec_digest: String,
    spec_byte_length: u64,
    #[serde(default)]
    paired_transform_id: RequiredNullable<String>,
    quality_audit: QualityAudit,
    required_checks: Vec<RequiredCheck>,
    quality_rubric: QualityRubric,
    requirements: Vec<TaskRequirement>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequiredCheck {
    check_id: String,
    kind: String,
    blocking: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskRequirement {
    requirement_id: String,
    source_span: RequirementSourceSpan,
    origin: RequirementOrigin,
    ambiguity: RequirementAmbiguity,
    statement_digest: String,
    expectation_digest: String,
    #[serde(default)]
    inference_rationale_digest: RequiredNullable<String>,
    blocking_check_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequirementSourceSpan {
    source_digest: String,
    start_byte: u64,
    end_byte: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RequirementOrigin {
    Explicit,
    Inferred,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RequirementAmbiguity {
    None,
    ResolvedBeforeRun,
    MaterialUnresolved,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityRubric {
    rubric_id: String,
    version: String,
    total_weight_basis_points: u32,
    criteria: Vec<QualityCriterion>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityCriterion {
    criterion_id: String,
    dimension: QualityCriterionDimension,
    applicability: QualityCriterionApplicability,
    weight_basis_points: u32,
    applicability_rationale: String,
    evidence_sources: Vec<QualityEvidenceSource>,
    reviewer_guidance: String,
    anchors: Vec<QualityAnchor>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum QualityCriterionDimension {
    Maintainability,
    ScopeDiscipline,
    ApiCompatibility,
    TestQuality,
    Documentation,
    Performance,
    ErrorHandling,
    IdiomaticDesign,
    SecurityHardening,
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum QualityCriterionApplicability {
    Scored,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum QualityEvidenceSource {
    Diff,
    TestResults,
    ApiAnalysis,
    StaticAnalysis,
    DocumentationReview,
    PerformanceMeasurements,
    HumanReview,
    SecurityScan,
    RuntimeLogs,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityAnchor {
    score: u8,
    description: String,
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
    #[serde(default)]
    patch_digest: RequiredNullable<String>,
    #[serde(default)]
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
struct RequiredNullable<T> {
    present: bool,
    value: Option<T>,
}

impl<T> Default for RequiredNullable<T> {
    fn default() -> Self {
        Self { present: false, value: None }
    }
}

impl<'de, T> Deserialize<'de> for RequiredNullable<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| Self { present: true, value })
    }
}

#[derive(Debug)]
struct OptionalNonNull<T> {
    value: Option<T>,
}

impl<T> Default for OptionalNonNull<T> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<'de, T> Deserialize<'de> for OptionalNonNull<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(|value| Self { value: Some(value) })
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
    evidence_digest: OptionalNonNull<String>,
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

pub fn validate_file(
    input_path: &Path,
    manifest_path: &Path,
    spec_root: &Path,
) -> anyhow::Result<ValidationReport> {
    let input_bytes = read_bounded(input_path, MAX_RUN_RECORD_BYTES)?;
    let manifest_bytes = read_bounded(manifest_path, MAX_SUITE_MANIFEST_BYTES)?;
    let spec_artifacts = load_spec_artifacts(&manifest_bytes, spec_root)?;
    Ok(validate_bytes(&input_bytes, &manifest_bytes, &spec_artifacts))
}

fn load_spec_artifacts(
    manifest_bytes: &[u8],
    spec_root: &Path,
) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let manifest: SuiteManifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| anyhow::anyhow!("cannot resolve task specifications from an invalid manifest: {error}"))?;
    let canonical_root = fs::canonicalize(spec_root)
        .map_err(|error| anyhow::anyhow!("cannot resolve BASALT specification root {}: {error}", spec_root.display()))?;
    if !canonical_root.is_dir() {
        anyhow::bail!("BASALT specification root {} is not a directory", canonical_root.display());
    }

    // Artifacts are keyed by their content identity, not task ID. This avoids
    // duplicate in-memory copies when multiple tasks intentionally share a spec.
    let mut artifacts = BTreeMap::new();
    for task in manifest.tasks {
        let Some(hex_digest) = task.spec_digest.strip_prefix("sha256:") else {
            continue;
        };
        if hex_digest.len() != 64
            || !hex_digest.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            continue;
        }
        if artifacts.contains_key(&task.spec_digest) {
            continue;
        }

        let path = spec_root.join(format!("{hex_digest}.txt"));
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "cannot inspect BASALT specification artifact {}: {error}",
                    path.display()
                ));
            }
        };
        if metadata.file_type().is_symlink() {
            anyhow::bail!(
                "BASALT specification artifact {} must not be a symbolic link",
                path.display()
            );
        }
        if !metadata.is_file() {
            anyhow::bail!(
                "BASALT specification artifact {} is not a regular file",
                path.display()
            );
        }

        let canonical_path = fs::canonicalize(&path).map_err(|error| {
            anyhow::anyhow!(
                "cannot resolve BASALT specification artifact {}: {error}",
                path.display()
            )
        })?;
        if !canonical_path.starts_with(&canonical_root) {
            anyhow::bail!(
                "BASALT specification artifact {} resolves outside --spec-root",
                path.display()
            );
        }

        let bytes = read_bounded(&canonical_path, MAX_TASK_SPEC_BYTES)?;
        artifacts.insert(task.spec_digest, bytes);
    }
    Ok(artifacts)
}

fn read_bounded(path: &Path, max_bytes: u64) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(max_bytes + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        anyhow::bail!(
            "{} exceeds the configured {} byte input limit",
            path.display(),
            max_bytes
        );
    }
    Ok(bytes)
}

fn validate_bytes(bytes: &[u8], manifest_bytes: &[u8], spec_artifacts: &BTreeMap<String, Vec<u8>>) -> ValidationReport {
    let input_digest = format!("sha256:{}", sha256_hex(bytes));
    let manifest_digest = format!("sha256:{}", sha256_hex(manifest_bytes));
    let mut report = ValidationReport {
        report_version: 1,
        input_digest,
        manifest_digest,
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
    let manifest: SuiteManifest = match serde_json::from_slice(manifest_bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            push_error(
                &mut report.errors,
                "E_MANIFEST_PARSE",
                "manifest",
                format!("invalid suite manifest, unknown field, duplicate field, or missing required field: {error}"),
            );
            return report;
        }
    };

    validate_run(&run, &manifest, manifest_bytes, spec_artifacts, &mut report);
    report.valid = report.errors.is_empty();
    report
}

fn validate_run(run: &RunRecord, manifest: &SuiteManifest, manifest_bytes: &[u8], spec_artifacts: &BTreeMap<String, Vec<u8>>, report: &mut ValidationReport) {
    validate_manifest_binding(run, manifest, manifest_bytes, report);
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

    let manifest_by_id: BTreeMap<String, &ManifestTask> = manifest
        .tasks
        .iter()
        .map(|task| (task.task_id.clone(), task))
        .collect();
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
        if task.spec_byte_length == 0 {
            error(report, "E_SPEC_BYTE_LENGTH", format!("{task_path}.spec_byte_length"), "task specification byte length must be non-zero");
        }
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
            } else if let Some(system) = run.systems.iter().find(|system| system.system_id == trial.system_id) {
                if trial.usage.wall_ms > system.budget.max_wall_ms {
                    error(report, "E_USAGE_WALL_BUDGET", format!("{trial_path}.usage.wall_ms"), "recorded wall time exceeds the declared system budget");
                }
                if trial.usage.tool_calls > system.budget.max_tool_calls {
                    error(report, "E_USAGE_TOOL_BUDGET", format!("{trial_path}.usage.tool_calls"), "recorded tool calls exceed the declared system budget");
                }
                if trial.usage.cost_usd > system.budget.max_cost_usd {
                    error(report, "E_USAGE_COST_BUDGET", format!("{trial_path}.usage.cost_usd"), "recorded cost exceeds the declared system budget");
                }
            }
            if trial.attempt_index == 0 || trial.attempt_index > run.suite.attempts_per_task {
                error(report, "E_ATTEMPT_OUT_OF_RANGE", format!("{trial_path}.attempt_index"), "attempt_index must be in 1..=attempts_per_task");
            }
            if let Seed::String(value) = &trial.seed {
                if value.trim().is_empty() {
                    error(report, "E_SEED_EMPTY", format!("{trial_path}.seed"), "string seeds must be non-empty");
                }
            }
            match manifest.attempt_seeds.get(trial.attempt_index.saturating_sub(1)) {
                Some(expected_seed) if expected_seed == &trial.seed => {}
                Some(_) => error(report, "E_TRIAL_SEED_SCHEDULE", format!("{trial_path}.seed"), "trial seed differs from the frozen pre-run seed schedule"),
                None => error(report, "E_TRIAL_SEED_INDEX", format!("{trial_path}.seed"), "attempt has no corresponding predeclared manifest seed"),
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
            let required_checks = manifest_by_id
                .get(&task.task_id)
                .map(|declared| declared.required_checks.as_slice())
                .unwrap_or(&[]);
            validate_trial(
                report,
                &trial_path,
                trial,
                &task.quality_audit,
                required_checks,
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
                            if first.seed != other.seed {
                                error(
                                    report,
                                    "E_PAIR_SEED_MISMATCH",
                                    format!("{task_path}.trials"),
                                    format!("paired system arms must use the same manifest-declared seed for attempt {attempt}"),
                                );
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
    if let Some(note) = &run.notes.value {
        if note.len() > 16_384 {
            error(report, "E_NOTES_LIMIT", "notes", "notes exceed 16384 bytes");
        }
    }
}


fn validate_manifest_binding(run: &RunRecord, manifest: &SuiteManifest, manifest_bytes: &[u8], report: &mut ValidationReport) {
    if manifest.schema_version != 3 {
        error(report, "E_MANIFEST_VERSION", "manifest.schema_version", "only suite-manifest schema_version=3 is supported");
    }
    if manifest.program != "BASALT" {
        error(report, "E_MANIFEST_PROGRAM", "manifest.program", "manifest program must be BASALT");
    }
    if run.source.manifest_digest != format!("sha256:{}", sha256_hex(manifest_bytes)) {
        error(report, "E_MANIFEST_DIGEST_MISMATCH", "source.manifest_digest", "source.manifest_digest must bind the exact bytes supplied with --manifest");
    }
    if manifest.suite_id != run.suite.suite_id || manifest.version != run.suite.version || manifest.split != run.suite.split {
        error(report, "E_MANIFEST_SUITE_MISMATCH", "manifest", "suite_id, version, and split must match the frozen suite manifest");
    }
    if manifest.seed_policy != run.suite.seed_policy {
        error(report, "E_MANIFEST_SEED_POLICY", "suite.seed_policy", "run seed policy must match the frozen manifest seed policy");
    }
    if manifest.attempts_per_task != run.suite.attempts_per_task {
        error(report, "E_MANIFEST_ATTEMPT_COUNT", "suite.attempts_per_task", "attempt count must match the frozen manifest");
    }
    if manifest.attempts_per_task == 0 || manifest.attempts_per_task > 10_000 {
        error(report, "E_MANIFEST_ATTEMPT_LIMIT", "manifest.attempts_per_task", "manifest attempts_per_task must be in 1..=10000");
    }
    if manifest.attempt_seeds.len() != manifest.attempts_per_task {
        error(report, "E_MANIFEST_SEED_SCHEDULE_LENGTH", "manifest.attempt_seeds", "attempt_seeds must contain exactly one predeclared seed per attempt");
    }
    for (index, seed) in manifest.attempt_seeds.iter().enumerate() {
        if let Seed::String(value) = seed {
            if value.trim().is_empty() {
                error(report, "E_MANIFEST_SEED_EMPTY", format!("manifest.attempt_seeds[{index}]"), "predeclared string seed must be non-empty");
            }
        }
        match manifest.seed_policy {
            SeedPolicy::Unseeded if seed != &Seed::String("unseeded".to_string()) => {
                error(report, "E_MANIFEST_UNSEEDED_SEED", format!("manifest.attempt_seeds[{index}]"), "unseeded policy requires the literal 'unseeded' marker");
            }
            SeedPolicy::NotApplicable if seed != &Seed::String("not_applicable".to_string()) => {
                error(report, "E_MANIFEST_NOT_APPLICABLE_SEED", format!("manifest.attempt_seeds[{index}]"), "not_applicable policy requires the literal 'not_applicable' marker");
            }
            _ => {}
        }
    }
    if manifest.tasks.len() != run.suite.expected_task_count {
        error(report, "E_MANIFEST_TASK_COUNT", "suite.expected_task_count", "expected_task_count must equal the task count in the frozen suite manifest");
    }

    let mut manifest_by_id: BTreeMap<String, &ManifestTask> = BTreeMap::new();
    for (index, task) in manifest.tasks.iter().enumerate() {
        let task_path = format!("manifest.tasks[{index}]");
        if task.task_id.trim().is_empty() {
            error(report, "E_MANIFEST_TASK_ID_EMPTY", format!("{task_path}.task_id"), "manifest task_id must be non-empty");
        }
        if manifest_by_id.insert(task.task_id.clone(), task).is_some() {
            error(report, "E_MANIFEST_DUPLICATE_TASK", format!("{task_path}.task_id"), "manifest task ids must be unique");
        }
        if task.family.trim().is_empty() || task.languages.is_empty() || task.languages.iter().any(|language| language.trim().is_empty()) {
            error(report, "E_MANIFEST_TASK_METADATA", &task_path, "manifest family and language list must be non-empty");
        }
        validate_digest(report, &format!("{task_path}.base_tree_digest"), &task.base_tree_digest);
        validate_digest(report, &format!("{task_path}.spec_digest"), &task.spec_digest);
        if !task.paired_transform_id.present {
            error(report, "E_MANIFEST_TRANSFORM_MISSING", format!("{task_path}.paired_transform_id"), "paired_transform_id property is required even when its value is null");
        }
        if let Some(transform_id) = &task.paired_transform_id.value {
            if transform_id.trim().is_empty() {
                error(report, "E_MANIFEST_TRANSFORM_ID", format!("{task_path}.paired_transform_id"), "paired_transform_id must be non-empty when present");
            }
        }
        validate_digest(report, &format!("{task_path}.quality_audit.audit_digest"), &task.quality_audit.audit_digest);
        if task.required_checks.is_empty() {
            error(report, "E_MANIFEST_REQUIRED_CHECKS_EMPTY", format!("{task_path}.required_checks"), "every manifest task must declare at least one required verifier check");
        }
        let mut required_ids = BTreeSet::new();
        for (check_index, check) in task.required_checks.iter().enumerate() {
            let check_path = format!("{task_path}.required_checks[{check_index}]");
            if check.check_id.trim().is_empty() || !required_ids.insert(check.check_id.as_str()) {
                error(report, "E_MANIFEST_REQUIRED_CHECK_DUPLICATE", format!("{check_path}.check_id"), "required verifier check IDs must be non-empty and unique");
            }
            if !matches!(check.kind.as_str(), "build" | "test" | "typecheck" | "lint" | "security" | "scope" | "regression" | "oracle" | "other") {
                error(report, "E_MANIFEST_REQUIRED_CHECK_KIND", format!("{check_path}.kind"), "unknown required verifier check kind");
            }
            if !check.blocking {
                error(report, "E_MANIFEST_REQUIRED_CHECK_NONBLOCKING", format!("{check_path}.blocking"), "required verifier obligations must be blocking");
            }
        }
        validate_quality_rubric(
            report,
            &format!("{task_path}.quality_rubric"),
            &task.quality_rubric,
        );
        validate_task_requirements(
            report,
            &task_path,
            task,
            spec_artifacts.get(&task.spec_digest).map(Vec::as_slice),
        );
    }

    if manifest.tasks.len() != run.task_records.len() {
        error(report, "E_TASK_SET_MISMATCH", "task_records", "run record must retain every frozen manifest task and no undeclared task");
    }
    for (index, task) in run.task_records.iter().enumerate() {
        let task_path = format!("task_records[{index}]");
        match manifest_by_id.get(&task.task_id) {
            None => error(report, "E_TASK_NOT_IN_MANIFEST", format!("{task_path}.task_id"), "task_id is absent from the frozen suite manifest"),
            Some(declared) => {
                let task_match = task.family == declared.family
                    && task.languages == declared.languages
                    && task.base_tree_digest == declared.base_tree_digest
                    && task.spec_digest == declared.spec_digest
                    && task.paired_transform_id.as_ref() == declared.paired_transform_id.value.as_ref()
                    && task.quality_audit.status == declared.quality_audit.status
                    && task.quality_audit.audit_digest == declared.quality_audit.audit_digest
                    && task.quality_audit.oracle_independent == declared.quality_audit.oracle_independent
                    && task.quality_audit.ambiguity == declared.quality_audit.ambiguity
                    && task.quality_audit.notes == declared.quality_audit.notes
                    && declared.required_checks.iter().all(|required| !required.check_id.trim().is_empty());
                if !task_match {
                    error(report, "E_TASK_MANIFEST_MISMATCH", &task_path, "task specification or quality-audit disposition differs from the frozen manifest");
                }
            }
        }
    }
}

fn validate_task_requirements(
    report: &mut ValidationReport,
    task_path: &str,
    task: &ManifestTask,
    spec_bytes: Option<&[u8]>,
) {
    let path = format!("{task_path}.requirements");
    let spec_text = match spec_bytes {
        None => {
            error(report, "E_SPEC_ARTIFACT_MISSING", &path, "the exact task specification artifact must be supplied from the offline --spec-root");
            None
        }
        Some(bytes) => {
            if bytes.len() as u64 != task.spec_byte_length {
                error(report, "E_SPEC_BYTE_LENGTH_MISMATCH", format!("{task_path}.spec_byte_length"), "actual specification artifact length differs from the frozen byte length");
            }
            let actual_digest = format!("sha256:{}", sha256_hex(bytes));
            if actual_digest != task.spec_digest {
                error(report, "E_SPEC_ARTIFACT_DIGEST", format!("{task_path}.spec_digest"), "actual specification artifact digest differs from the frozen specification digest");
            }
            match std::str::from_utf8(bytes) {
                Ok(text) => Some(text),
                Err(_) => {
                    error(report, "E_SPEC_ARTIFACT_UTF8", &path, "task specification artifact must be valid UTF-8");
                    None
                }
            }
        }
    };
    if task.requirements.is_empty() {
        error(report, "E_REQUIREMENTS_EMPTY", &path, "every task must freeze at least one atomic requirement");
    }

    let required_check_ids: BTreeSet<&str> = task
        .required_checks
        .iter()
        .map(|check| check.check_id.as_str())
        .collect();
    let mut requirement_ids = BTreeSet::new();
    let mut mapped_check_ids = BTreeSet::new();
    for (index, requirement) in task.requirements.iter().enumerate() {
        let requirement_path = format!("{path}[{index}]");
        if requirement.requirement_id.trim().is_empty()
            || !requirement_ids.insert(requirement.requirement_id.as_str())
        {
            error(
                report,
                "E_REQUIREMENT_ID",
                format!("{requirement_path}.requirement_id"),
                "requirement IDs must be non-empty and unique within a task",
            );
        }
        validate_digest(
            report,
            &format!("{requirement_path}.source_span.source_digest"),
            &requirement.source_span.source_digest,
        );
        if requirement.source_span.source_digest != task.spec_digest {
            error(
                report,
                "E_REQUIREMENT_SOURCE_MISMATCH",
                format!("{requirement_path}.source_span.source_digest"),
                "requirement source span must bind the task's frozen specification digest",
            );
        }
        if requirement.source_span.start_byte >= requirement.source_span.end_byte {
            error(
                report,
                "E_REQUIREMENT_SPAN_RANGE",
                format!("{requirement_path}.source_span"),
                "source span must be a non-empty half-open UTF-8 byte range [start_byte, end_byte)",
            );
        }
        if requirement.source_span.end_byte > task.spec_byte_length {
            error(
                report,
                "E_REQUIREMENT_SPAN_BOUNDS",
                format!("{requirement_path}.source_span.end_byte"),
                "source span end_byte must not exceed the frozen specification byte length",
            );
        }
        if let Some(spec_text) = spec_text {
            match (
                usize::try_from(requirement.source_span.start_byte),
                usize::try_from(requirement.source_span.end_byte),
            ) {
                (Ok(start), Ok(end)) => match spec_text.get(start..end) {
                    Some(source_span) => {
                        let actual_span_digest = format!("sha256:{}", sha256_hex(source_span.as_bytes()));
                        if actual_span_digest != requirement.statement_digest {
                            error(
                                report,
                                "E_REQUIREMENT_STATEMENT_MISMATCH",
                                format!("{requirement_path}.statement_digest"),
                                "statement_digest must hash the exact UTF-8 bytes selected by source_span",
                            );
                        }
                    }
                    None => error(
                        report,
                        "E_REQUIREMENT_SPAN_UTF8_BOUNDARY",
                        format!("{requirement_path}.source_span"),
                        "source span must select valid UTF-8 character boundaries in the actual specification",
                    ),
                },
                _ => error(
                    report,
                    "E_REQUIREMENT_SPAN_PLATFORM_RANGE",
                    format!("{requirement_path}.source_span"),
                    "source span offsets cannot be represented on this platform",
                ),
            }
        }
        validate_digest(
            report,
            &format!("{requirement_path}.statement_digest"),
            &requirement.statement_digest,
        );
        validate_digest(
            report,
            &format!("{requirement_path}.expectation_digest"),
            &requirement.expectation_digest,
        );
        if !requirement.inference_rationale_digest.present {
            error(
                report,
                "E_REQUIREMENT_RATIONALE_MISSING",
                format!("{requirement_path}.inference_rationale_digest"),
                "inference_rationale_digest is required as a property, even when null",
            );
        }
        match requirement.origin {
            RequirementOrigin::Explicit => {
                if requirement.inference_rationale_digest.value.is_some() {
                    error(
                        report,
                        "E_REQUIREMENT_EXPLICIT_RATIONALE",
                        format!("{requirement_path}.inference_rationale_digest"),
                        "explicit requirements must use a null inference_rationale_digest",
                    );
                }
            }
            RequirementOrigin::Inferred => {
                match &requirement.inference_rationale_digest.value {
                    Some(digest) => validate_digest(
                        report,
                        &format!("{requirement_path}.inference_rationale_digest"),
                        digest,
                    ),
                    None => error(
                        report,
                        "E_REQUIREMENT_INFERENCE_RATIONALE",
                        format!("{requirement_path}.inference_rationale_digest"),
                        "inferred requirements require a content-addressed rationale",
                    ),
                }
            }
        }
        if task.quality_audit.status == TaskAuditStatus::Accepted
            && requirement.ambiguity == RequirementAmbiguity::MaterialUnresolved
        {
            error(
                report,
                "E_REQUIREMENT_AMBIGUOUS_ACCEPTED",
                format!("{requirement_path}.ambiguity"),
                "accepted tasks cannot contain materially unresolved requirements",
            );
        }
        if requirement.blocking_check_ids.is_empty() {
            error(
                report,
                "E_REQUIREMENT_CHECKS_EMPTY",
                format!("{requirement_path}.blocking_check_ids"),
                "every requirement must map to at least one frozen blocking verifier check",
            );
        }
        let mut mapped_checks = BTreeSet::new();
        for check_id in &requirement.blocking_check_ids {
            if check_id.trim().is_empty() || !mapped_checks.insert(check_id.as_str()) {
                error(
                    report,
                    "E_REQUIREMENT_CHECK_DUPLICATE",
                    format!("{requirement_path}.blocking_check_ids"),
                    "blocking check IDs must be non-empty and unique within a requirement",
                );
            }
            if !required_check_ids.contains(check_id.as_str()) {
                error(
                    report,
                    "E_REQUIREMENT_CHECK_UNDECLARED",
                    format!("{requirement_path}.blocking_check_ids"),
                    "requirement mapping references a check not declared as required and blocking in this task manifest",
                );
            }
            mapped_check_ids.insert(check_id.as_str());
        }
    }
    for required_check in &task.required_checks {
        if !mapped_check_ids.contains(required_check.check_id.as_str()) {
            error(
                report,
                "E_REQUIRED_CHECK_UNMAPPED",
                format!("{task_path}.required_checks"),
                "every required blocking check must cover at least one frozen task requirement",
            );
        }
    }
}

fn validate_quality_rubric(report: &mut ValidationReport, path: &str, rubric: &QualityRubric) {
    if rubric.rubric_id.trim().is_empty() || rubric.version.trim().is_empty() {
        error(report, "E_QUALITY_RUBRIC_ID", path, "rubric_id and version must be non-empty");
    }
    if rubric.total_weight_basis_points != 10_000 {
        error(
            report,
            "E_QUALITY_RUBRIC_TOTAL",
            format!("{path}.total_weight_basis_points"),
            "the declared rubric total must be exactly 10000 basis points",
        );
    }
    if rubric.criteria.is_empty() {
        error(report, "E_QUALITY_RUBRIC_EMPTY", format!("{path}.criteria"), "at least one predeclared criterion is required");
    }

    let mut criterion_ids = BTreeSet::new();
    let mut scored_weight = 0u64;
    let mut scored_criteria = 0usize;
    for (index, criterion) in rubric.criteria.iter().enumerate() {
        let criterion_path = format!("{path}.criteria[{index}]");
        if criterion.criterion_id.trim().is_empty()
            || !criterion_ids.insert(criterion.criterion_id.as_str())
        {
            error(
                report,
                "E_QUALITY_CRITERION_ID",
                format!("{criterion_path}.criterion_id"),
                "criterion IDs must be non-empty and unique within a task rubric",
            );
        }
        if criterion.applicability_rationale.trim().is_empty() {
            error(
                report,
                "E_QUALITY_APPLICABILITY_RATIONALE",
                format!("{criterion_path}.applicability_rationale"),
                "every criterion requires a pre-run applicability rationale",
            );
        }
        if criterion.evidence_sources.is_empty() {
            error(
                report,
                "E_QUALITY_EVIDENCE_SOURCES",
                format!("{criterion_path}.evidence_sources"),
                "every criterion must predeclare at least one admissible evidence source",
            );
        } else {
            let mut sources = BTreeSet::new();
            for source in &criterion.evidence_sources {
                if !sources.insert(*source) {
                    error(
                        report,
                        "E_QUALITY_EVIDENCE_SOURCE_DUPLICATE",
                        format!("{criterion_path}.evidence_sources"),
                        "evidence sources must be unique within a criterion",
                    );
                    break;
                }
            }
        }
        if criterion.reviewer_guidance.trim().is_empty() {
            error(
                report,
                "E_QUALITY_REVIEWER_GUIDANCE",
                format!("{criterion_path}.reviewer_guidance"),
                "reviewer guidance must be non-empty and frozen before the run",
            );
        }
        if criterion.anchors.len() != 5 {
            error(
                report,
                "E_QUALITY_ANCHOR_COUNT",
                format!("{criterion_path}.anchors"),
                "each criterion must define exactly five ordered anchors for scores 0 through 4",
            );
        }
        for (anchor_index, anchor) in criterion.anchors.iter().enumerate() {
            if anchor.score as usize != anchor_index {
                error(
                    report,
                    "E_QUALITY_ANCHOR_ORDER",
                    format!("{criterion_path}.anchors[{anchor_index}].score"),
                    "anchor scores must be ordered and contiguous from 0 through 4",
                );
            }
            if anchor.description.trim().is_empty() {
                error(
                    report,
                    "E_QUALITY_ANCHOR_DESCRIPTION",
                    format!("{criterion_path}.anchors[{anchor_index}].description"),
                    "every score anchor requires a concrete description",
                );
            }
        }

        match criterion.applicability {
            QualityCriterionApplicability::Scored => {
                scored_criteria += 1;
                if criterion.weight_basis_points == 0 || criterion.weight_basis_points > 10_000 {
                    error(
                        report,
                        "E_QUALITY_CRITERION_WEIGHT",
                        format!("{criterion_path}.weight_basis_points"),
                        "a scored criterion must have a weight in 1..=10000 basis points",
                    );
                }
                scored_weight += criterion.weight_basis_points as u64;
            }
            QualityCriterionApplicability::NotApplicable => {
                if criterion.weight_basis_points != 0 {
                    error(
                        report,
                        "E_QUALITY_NOT_APPLICABLE_WEIGHT",
                        format!("{criterion_path}.weight_basis_points"),
                        "a not_applicable criterion must have zero weight; explain the exclusion in applicability_rationale",
                    );
                }
            }
        }
    }
    if scored_criteria == 0 || scored_weight != rubric.total_weight_basis_points as u64 {
        error(
            report,
            "E_QUALITY_WEIGHT_SUM",
            format!("{path}.criteria"),
            "the weights of scored criteria must sum exactly to the declared total; not_applicable criteria carry zero weight",
        );
    }
}

fn validate_trial(
    report: &mut ValidationReport,
    path: &str,
    trial: &Trial,
    task_audit: &QualityAudit,
    required_checks: &[RequiredCheck],
    pinned_verifier_digest: &str,
) {
    validate_digest(report, &format!("{path}.trajectory_digest"), &trial.trajectory_digest);
    if !trial.patch_digest.present {
        error(report, "E_REQUIRED_NULLABLE_MISSING", format!("{path}.patch_digest"), "patch_digest is required even when its value is null");
    }
    if !trial.final_tree_digest.present {
        error(report, "E_REQUIRED_NULLABLE_MISSING", format!("{path}.final_tree_digest"), "final_tree_digest is required even when its value is null");
    }
    if let Some(digest) = &trial.patch_digest.value {
        validate_digest(report, &format!("{path}.patch_digest"), digest);
    }
    if let Some(digest) = &trial.final_tree_digest.value {
        validate_digest(report, &format!("{path}.final_tree_digest"), digest);
    }

    let verifier_path = format!("{path}.verifier");
    validate_digest(report, &format!("{verifier_path}.bundle_digest"), &trial.verifier.bundle_digest);
    validate_digest(report, &format!("{verifier_path}.evaluated_tree_digest"), &trial.verifier.evaluated_tree_digest);
    if trial.verifier.bundle_digest != pinned_verifier_digest {
        error(report, "E_VERIFIER_BUNDLE_MISMATCH", format!("{verifier_path}.bundle_digest"), "trial verifier bundle differs from the run-pinned verifier bundle");
    }
    if let Some(final_digest) = &trial.final_tree_digest.value {
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
        if let Some(digest) = &check.evidence_digest.value {
            validate_digest(report, &format!("{check_path}.evidence_digest"), digest);
        }
        if check.blocking {
            blocking_count += 1;
        }
        if check.status != CheckStatus::NotRun && check.evidence_digest.value.is_none() {
            error(report, "E_CHECK_EVIDENCE_MISSING", format!("{check_path}.evidence_digest"), "an executed check requires a content digest for its evidence");
        }
        if trial.verifier.status == VerifierStatus::Pass && check.blocking && check.status != CheckStatus::Pass {
            error(report, "E_VERIFIER_BLOCKING_CHECK_FAILED", format!("{check_path}.status"), "verifier cannot pass while a blocking check is not pass");
        }
    }
    if required_checks.is_empty() {
        error(report, "E_REQUIRED_CHECKS_EMPTY", format!("{verifier_path}.checks"), "frozen manifest must declare at least one required verifier check");
    }
    let mut required_ids = BTreeSet::new();
    for (required_index, required) in required_checks.iter().enumerate() {
        let required_path = format!("{verifier_path}.required_checks[{required_index}]");
        if required.check_id.trim().is_empty() || !required_ids.insert(required.check_id.as_str()) {
            error(report, "E_REQUIRED_CHECK_DUPLICATE", format!("{required_path}.check_id"), "required check IDs must be non-empty and unique");
        }
        if !matches!(required.kind.as_str(), "build" | "test" | "typecheck" | "lint" | "security" | "scope" | "regression" | "oracle" | "other") {
            error(report, "E_REQUIRED_CHECK_KIND", format!("{required_path}.kind"), "unknown required check kind");
        }
        if !required.blocking {
            error(report, "E_REQUIRED_CHECK_NONBLOCKING", format!("{required_path}.blocking"), "required verifier obligations must be blocking");
        }
        match trial.verifier.checks.iter().find(|check| check.check_id == required.check_id) {
            None => error(report, "E_REQUIRED_CHECK_MISSING", format!("{verifier_path}.checks"), format!("required verifier check {:?} is absent", required.check_id)),
            Some(check) => {
                if check.kind != required.kind {
                    error(report, "E_REQUIRED_CHECK_KIND_MISMATCH", format!("{verifier_path}.checks"), format!("required check {:?} has a different kind", required.check_id));
                }
                if !check.blocking {
                    error(report, "E_REQUIRED_CHECK_NOT_BLOCKING", format!("{verifier_path}.checks"), format!("required check {:?} is not marked blocking", required.check_id));
                }
                if trial.outcome == Outcome::Pass && check.status != CheckStatus::Pass {
                    error(report, "E_REQUIRED_CHECK_NOT_PASSED", format!("{verifier_path}.checks"), format!("required check {:?} must pass for an outcome pass", required.check_id));
                }
            }
        }
    }

    if trial.verifier.status == VerifierStatus::NotRun && trial.verifier.checks.iter().any(|check| check.status != CheckStatus::NotRun) {
        error(report, "E_NOT_RUN_CHECK_CONTRADICTION", format!("{verifier_path}.checks"), "not_run verifier cannot contain checks reported as pass/fail/unresolved");
    }
    if trial.verifier.status == VerifierStatus::Fail
        && !trial.verifier.checks.iter().any(|check| check.blocking && check.status == CheckStatus::Fail)
    {
        error(report, "E_VERIFIER_FAIL_UNSUPPORTED", format!("{verifier_path}.checks"), "failed verifier requires at least one failed blocking check");
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
            if trial.final_tree_digest.value.is_none() {
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
    const MANIFEST: &str = include_str!("../../benchmarks/basalt/examples/suite-manifest.example.json");
    const EXAMPLE_SPEC: &str = include_str!("../../benchmarks/basalt/specifications/ae9535fc0aab3f958effde1a4b0a6f821740af7f999e012e37cd747023d8b2ac.txt");

    const EXAMPLE_SPEC_DIGEST: &str =
        "sha256:ae9535fc0aab3f958effde1a4b0a6f821740af7f999e012e37cd747023d8b2ac";

    fn synthetic_specs() -> BTreeMap<String, Vec<u8>> {
        BTreeMap::from([(EXAMPLE_SPEC_DIGEST.to_string(), EXAMPLE_SPEC.as_bytes().to_vec())])
    }

    struct TemporarySpecRoot(PathBuf);

    impl TemporarySpecRoot {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "basalt-spec-loader-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("temporary specification root is created");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write_example(&self) {
            let filename = format!("{}.txt", EXAMPLE_SPEC_DIGEST.trim_start_matches("sha256:"));
            fs::write(self.path().join(filename), EXAMPLE_SPEC.as_bytes())
                .expect("synthetic specification artifact is written");
        }
    }

    impl Drop for TemporarySpecRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn validate_value(mutator: impl FnOnce(&mut Value)) -> ValidationReport {
        let mut value: Value = serde_json::from_str(EXAMPLE).expect("synthetic fixture parses");
        mutator(&mut value);
        let bytes = serde_json::to_vec(&value).expect("fixture serializes");
        validate_bytes(&bytes, MANIFEST.as_bytes(), &synthetic_specs())
    }

    fn validate_value_and_manifest(
        mutator: impl FnOnce(&mut Value, &mut Value),
    ) -> ValidationReport {
        let mut value: Value = serde_json::from_str(EXAMPLE).expect("synthetic fixture parses");
        let mut manifest: Value = serde_json::from_str(MANIFEST).expect("synthetic manifest parses");
        mutator(&mut value, &mut manifest);
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest).expect("manifest serializes");
        manifest_bytes.push(b'\n');
        value["source"]["manifest_digest"] =
            Value::String(format!("sha256:{}", sha256_hex(&manifest_bytes)));
        let bytes = serde_json::to_vec_pretty(&value).expect("record serializes");
        validate_bytes(&bytes, &manifest_bytes, &synthetic_specs())
    }

    #[test]
    fn artifact_loader_reads_digest_named_file_and_indexes_by_digest() {
        let root = TemporarySpecRoot::new();
        root.write_example();

        let artifacts = load_spec_artifacts(MANIFEST.as_bytes(), root.path())
            .expect("content-addressed specification artifact loads");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(
            artifacts.get(EXAMPLE_SPEC_DIGEST).map(Vec::as_slice),
            Some(EXAMPLE_SPEC.as_bytes())
        );
    }

    #[test]
    fn artifact_loader_deduplicates_shared_specification_bytes() {
        let root = TemporarySpecRoot::new();
        root.write_example();

        let mut manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
        let mut second_task = manifest["tasks"][0].clone();
        second_task["task_id"] = Value::String("rust-repair-0002".into());
        manifest["tasks"].as_array_mut().unwrap().push(second_task);
        let bytes = serde_json::to_vec(&manifest).expect("manifest serializes");

        let artifacts = load_spec_artifacts(&bytes, root.path())
            .expect("shared content-addressed artifact loads");
        assert_eq!(artifacts.len(), 1);
        assert!(artifacts.contains_key(EXAMPLE_SPEC_DIGEST));
    }

    #[test]
    fn artifact_loader_rejects_directory_at_artifact_path() {
        let root = TemporarySpecRoot::new();
        let filename = format!("{}.txt", EXAMPLE_SPEC_DIGEST.trim_start_matches("sha256:"));
        fs::create_dir(root.path().join(filename)).expect("artifact path directory is created");

        let result = load_spec_artifacts(MANIFEST.as_bytes(), root.path());
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_loader_rejects_symbolic_link_artifact() {
        use std::os::unix::fs::symlink;

        let root = TemporarySpecRoot::new();
        let outside = TemporarySpecRoot::new();
        let outside_file = outside.path().join("outside.txt");
        fs::write(&outside_file, EXAMPLE_SPEC.as_bytes()).expect("outside artifact is written");
        let filename = format!("{}.txt", EXAMPLE_SPEC_DIGEST.trim_start_matches("sha256:"));
        symlink(&outside_file, root.path().join(filename)).expect("artifact symlink is created");

        let result = load_spec_artifacts(MANIFEST.as_bytes(), root.path());
        assert!(result.is_err());
    }

    #[test]
    fn artifact_loader_enforces_per_specification_size_limit() {
        let root = TemporarySpecRoot::new();
        let filename = format!("{}.txt", EXAMPLE_SPEC_DIGEST.trim_start_matches("sha256:"));
        fs::write(
            root.path().join(filename),
            vec![b'x'; MAX_TASK_SPEC_BYTES as usize + 1],
        )
        .expect("oversized artifact is written");

        let result = load_spec_artifacts(MANIFEST.as_bytes(), root.path());
        assert!(result.is_err());
    }

    fn has_code(report: &ValidationReport, code: &str) -> bool {
        report.errors.iter().any(|error| error.code == code)
    }

    #[test]
    fn synthetic_paired_fixture_passes_structure_and_invariants() {
        let report = validate_bytes(EXAMPLE.as_bytes(), MANIFEST.as_bytes(), &synthetic_specs());
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
    fn predeclared_random_seed_must_match_frozen_schedule_and_all_arms() {
        let report = validate_value_and_manifest(|run, manifest| {
            run["suite"]["seed_policy"] = Value::String("predeclared_random".into());
            manifest["seed_policy"] = Value::String("predeclared_random".into());
            run["task_records"][0]["trials"][1]["seed"] = Value::from(18);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_TRIAL_SEED_SCHEDULE"));
        assert!(has_code(&report, "E_PAIR_SEED_MISMATCH"));
    }

    #[test]
    fn unseeded_policy_uses_predeclared_unseeded_marker() {
        let report = validate_value_and_manifest(|run, manifest| {
            run["suite"]["seed_policy"] = Value::String("unseeded".into());
            manifest["seed_policy"] = Value::String("unseeded".into());
            manifest["attempt_seeds"] = Value::Array(vec![Value::String("unseeded".into())]);
            for trial in run["task_records"][0]["trials"].as_array_mut().unwrap() {
                trial["seed"] = Value::String("unseeded".into());
            }
        });
        assert!(report.valid, "{:?}", report.errors);
    }

    #[test]
    fn unseeded_policy_rejects_posthoc_numeric_seed() {
        let report = validate_value_and_manifest(|run, manifest| {
            run["suite"]["seed_policy"] = Value::String("unseeded".into());
            manifest["seed_policy"] = Value::String("unseeded".into());
            manifest["attempt_seeds"] = Value::Array(vec![Value::String("unseeded".into())]);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_TRIAL_SEED_SCHEDULE"));
    }

    #[test]
    fn trial_seed_cannot_be_changed_after_manifest_was_frozen() {
        let report = validate_value(|run| {
            run["task_records"][0]["trials"][1]["seed"] = Value::from(18);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_TRIAL_SEED_SCHEDULE"));
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
        let first = serde_json::to_vec(&validate_bytes(EXAMPLE.as_bytes(), MANIFEST.as_bytes(), &synthetic_specs())).unwrap();
        let second = serde_json::to_vec(&validate_bytes(EXAMPLE.as_bytes(), MANIFEST.as_bytes(), &synthetic_specs())).unwrap();
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

    #[test]
    fn null_is_rejected_for_optional_non_nullable_evidence_digest() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][0]["evidence_digest"] = Value::Null;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_JSON_OR_SHAPE"));
    }

    #[test]
    fn null_is_rejected_for_optional_non_nullable_run_notes() {
        let report = validate_value(|v| {
            v["notes"] = Value::Null;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_JSON_OR_SHAPE"));
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut value: Value = serde_json::from_str(EXAMPLE).unwrap();
        value["unexpected_top_level_field"] = Value::Bool(true);
        let bytes = serde_json::to_vec(&value).unwrap();
        let report = validate_bytes(&bytes, MANIFEST.as_bytes(), &synthetic_specs());
        assert!(!report.valid);
        assert!(has_code(&report, "E_JSON_OR_SHAPE"));
    }

    #[test]
    fn duplicate_json_object_keys_fail_closed() {
        let duplicate = EXAMPLE.replacen(
            "\"schema_version\": 1,",
            "\"schema_version\": 1,\n  \"schema_version\": 1,",
            1,
        );
        let report = validate_bytes(duplicate.as_bytes(), MANIFEST.as_bytes(), &synthetic_specs());
        assert!(!report.valid);
        assert!(has_code(&report, "E_JSON_OR_SHAPE"));
    }

    #[test]
    fn duplicate_task_identity_is_rejected() {
        let report = validate_value(|v| {
            let duplicate = v["task_records"][0].clone();
            v["task_records"].as_array_mut().unwrap().push(duplicate);
            v["suite"]["expected_task_count"] = Value::from(2);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_DUPLICATE_TASK_ID"));
    }

    #[test]
    fn predeclared_quarantined_task_is_retained_but_not_marked_accepted() {
        let report = validate_value_and_manifest(|run, manifest| {
            run["task_records"][0]["quality_audit"]["status"] = Value::String("quarantined".into());
            manifest["tasks"][0]["quality_audit"]["status"] = Value::String("quarantined".into());
            run["task_records"][0]["quality_audit"]["audit_digest"] = Value::String(format!("sha256:{}", "a".repeat(64)));
            manifest["tasks"][0]["quality_audit"]["audit_digest"] = Value::String(format!("sha256:{}", "a".repeat(64)));
        });
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(report.counts.tasks_total, 1);
        assert_eq!(report.counts.tasks_accepted, 0);
        assert_eq!(report.counts.tasks_quarantined, 1);
        assert_eq!(report.counts.trials_accounted, 2);
    }

    #[test]
    fn posthoc_task_reclassification_is_rejected_by_frozen_manifest() {
        let report = validate_value(|run| {
            run["task_records"][0]["quality_audit"]["status"] = Value::String("quarantined".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_TASK_MANIFEST_MISMATCH"));
    }

    #[test]
    fn stale_evidence_cannot_be_labelled_clean() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["integrity"]["stale_evidence"] = Value::String("detected".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_INTEGRITY_CLAIM_MISMATCH"));
    }

    #[test]
    fn unequal_system_budgets_are_rejected() {
        let report = validate_value(|v| {
            v["systems"][1]["budget"]["max_tool_calls"] = Value::from(31);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_BUDGET_MISMATCH"));
    }

    #[test]
    fn missing_required_nullable_tree_digest_is_rejected() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1].as_object_mut().unwrap().remove("final_tree_digest");
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_NULLABLE_MISSING"));
    }

    #[test]
    fn predeclared_invalid_task_trials_remain_accounted_without_becoming_successes() {
        let report = validate_value_and_manifest(|run, manifest| {
            run["task_records"][0]["quality_audit"]["status"] = Value::String("invalid".into());
            manifest["tasks"][0]["quality_audit"]["status"] = Value::String("invalid".into());
            run["task_records"][0]["quality_audit"]["audit_digest"] = Value::String(format!("sha256:{}", "b".repeat(64)));
            manifest["tasks"][0]["quality_audit"]["audit_digest"] = Value::String(format!("sha256:{}", "b".repeat(64)));
            for trial in run["task_records"][0]["trials"].as_array_mut().unwrap() {
                trial["outcome"] = Value::String("invalid_task".into());
                trial["verifier"]["status"] = Value::String("not_run".into());
                for check in trial["verifier"]["checks"].as_array_mut().unwrap() {
                    check["status"] = Value::String("not_run".into());
                }
            }
        });
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(report.counts.tasks_invalid, 1);
        assert_eq!(report.counts.trials_accounted, 2);
    }

    #[test]
    fn altered_manifest_bytes_are_rejected_without_pinned_digest_update() {
        let mut manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        manifest["tasks"][0]["family"] = Value::String("different_family".into());
        let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        bytes.push(b'\n');
        let report = validate_bytes(EXAMPLE.as_bytes(), &bytes, &synthetic_specs());
        assert!(!report.valid);
        assert!(has_code(&report, "E_MANIFEST_DIGEST_MISMATCH"));
    }


    #[test]
    fn passing_trial_must_satisfy_every_manifest_required_check() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"]
                .as_array_mut()
                .unwrap()
                .retain(|check| check["check_id"] != "scope");
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_CHECK_MISSING"));
    }

    #[test]
    fn required_check_kind_must_match_frozen_manifest() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][1]["kind"] =
                Value::String("other".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_CHECK_KIND_MISMATCH"));
    }

    #[test]
    fn required_check_cannot_be_downgraded_to_nonblocking() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][1]["blocking"] = Value::Bool(false);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_CHECK_NOT_BLOCKING"));
    }

    #[test]
    fn required_check_must_pass_even_if_verifier_claims_pass() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["verifier"]["checks"][1]["status"] =
                Value::String("unresolved".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_CHECK_NOT_PASSED"));
    }

    #[test]
    fn manifest_cannot_omit_all_required_checks() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["required_checks"] = Value::Array(Vec::new());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_MANIFEST_REQUIRED_CHECKS_EMPTY"));
    }

    #[test]
    fn observed_usage_over_declared_budget_is_rejected() {
        let report = validate_value(|v| {
            v["task_records"][0]["trials"][1]["usage"]["tool_calls"] = Value::from(31);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_USAGE_TOOL_BUDGET"));
    }

    #[test]
    fn run_record_cannot_lower_task_count_to_hide_a_manifest_task() {
        let report = validate_value(|v| {
            v["task_records"].as_array_mut().unwrap().clear();
            v["suite"]["expected_task_count"] = Value::from(0);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_MANIFEST_TASK_COUNT"));
        assert!(has_code(&report, "E_TASK_SET_MISMATCH"));
    }

    #[test]
    fn quality_rubric_requires_exact_weight_sum() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][1]["weight_basis_points"] = Value::from(2_999);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_WEIGHT_SUM"));
    }

    #[test]
    fn quality_rubric_rejects_duplicate_criterion_ids() {
        let report = validate_value_and_manifest(|_run, manifest| {
            let first = manifest["tasks"][0]["quality_rubric"]["criteria"][0]["criterion_id"].clone();
            manifest["tasks"][0]["quality_rubric"]["criteria"][1]["criterion_id"] = first;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_CRITERION_ID"));
    }

    #[test]
    fn quality_rubric_rejects_weight_on_not_applicable_criterion() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["applicability"] = Value::String("not_applicable".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_NOT_APPLICABLE_WEIGHT"));
    }

    #[test]
    fn quality_rubric_rejects_unordered_score_anchors() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["anchors"][0]["score"] = Value::from(1);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_ANCHOR_ORDER"));
    }

    #[test]
    fn quality_rubric_requires_admissible_evidence_sources() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["evidence_sources"] = Value::Array(Vec::new());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_EVIDENCE_SOURCES"));
    }

    #[test]
    fn quality_rubric_requires_nonempty_pre_run_reviewer_guidance() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["reviewer_guidance"] = Value::String("  ".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_QUALITY_REVIEWER_GUIDANCE"));
    }

    #[test]
    fn quality_rubric_accepts_explicit_not_applicable_criterion_at_zero_weight() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["applicability"] = Value::String("not_applicable".into());
            manifest["tasks"][0]["quality_rubric"]["criteria"][0]["weight_basis_points"] = Value::from(0);
            manifest["tasks"][0]["quality_rubric"]["criteria"][1]["weight_basis_points"] = Value::from(7_000);
        });
        assert!(report.valid, "{:?}", report.errors);
    }

    #[test]
    fn manifest_schema_version_one_is_not_silently_interpreted_as_version_three() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["schema_version"] = Value::from(1);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_MANIFEST_VERSION"));
    }

    #[test]
    fn task_requirement_must_map_to_declared_blocking_check() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["blocking_check_ids"][0] = Value::String("unreviewed-check".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_CHECK_UNDECLARED"));
    }

    #[test]
    fn duplicate_task_requirement_ids_are_rejected() {
        let report = validate_value_and_manifest(|_run, manifest| {
            let first = manifest["tasks"][0]["requirements"][0]["requirement_id"].clone();
            manifest["tasks"][0]["requirements"][1]["requirement_id"] = first;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_ID"));
    }

    #[test]
    fn task_requirement_source_must_bind_frozen_specification() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["source_span"]["source_digest"] =
                Value::String(format!("sha256:{}", "f".repeat(64)));
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_SOURCE_MISMATCH"));
    }

    #[test]
    fn task_requirement_requires_nonempty_source_span() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["source_span"]["end_byte"] = Value::from(0);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_SPAN_RANGE"));
    }

    #[test]
    fn inferred_requirement_requires_rationale_digest() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["origin"] = Value::String("inferred".into());
            manifest["tasks"][0]["requirements"][0]["inference_rationale_digest"] = Value::Null;
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_INFERENCE_RATIONALE"));
    }

    #[test]
    fn accepted_task_rejects_materially_unresolved_requirement() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["ambiguity"] = Value::String("material_unresolved".into());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_AMBIGUOUS_ACCEPTED"));
    }

    #[test]
    fn task_requirement_requires_at_least_one_check_mapping() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["blocking_check_ids"] = Value::Array(Vec::new());
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_CHECKS_EMPTY"));
    }

    #[test]
    fn requirement_span_is_hashed_against_actual_specification_bytes() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["statement_digest"] =
                Value::String(format!("sha256:{}", "f".repeat(64)));
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_STATEMENT_MISMATCH"));
    }

    #[test]
    fn actual_specification_bytes_must_match_frozen_digest() {
        let mut specs = synthetic_specs();
        specs.insert("rust-repair-0001".to_string(), b"tampered specification bytes".to_vec());
        let report = validate_bytes(EXAMPLE.as_bytes(), MANIFEST.as_bytes(), &specs);
        assert!(!report.valid);
        assert!(has_code(&report, "E_SPEC_BYTE_LENGTH_MISMATCH"));
        assert!(has_code(&report, "E_SPEC_ARTIFACT_DIGEST"));
    }

    #[test]
    fn missing_specification_artifact_is_rejected() {
        let report = validate_bytes(EXAMPLE.as_bytes(), MANIFEST.as_bytes(), &BTreeMap::new());
        assert!(!report.valid);
        assert!(has_code(&report, "E_SPEC_ARTIFACT_MISSING"));
    }

    #[test]
    fn spec_span_must_land_on_utf8_character_boundaries() {
        let mut manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        manifest["tasks"][0]["requirements"][1]["source_span"]["start_byte"] = Value::from(87);
        manifest["tasks"][0]["requirements"][1]["source_span"]["end_byte"] = Value::from(88);
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        manifest_bytes.push(b'\n');
        let mut run: Value = serde_json::from_str(EXAMPLE).unwrap();
        run["source"]["manifest_digest"] =
            Value::String(format!("sha256:{}", sha256_hex(&manifest_bytes)));
        let run_bytes = serde_json::to_vec_pretty(&run).unwrap();
        let report = validate_bytes(&run_bytes, &manifest_bytes, &synthetic_specs());
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_SPAN_UTF8_BOUNDARY"));
    }

    #[test]
    fn task_requirement_span_cannot_exceed_frozen_spec_length() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][0]["source_span"]["end_byte"] = Value::from(101);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIREMENT_SPAN_BOUNDS"));
    }

    #[test]
    fn every_required_check_must_cover_a_frozen_requirement() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["requirements"][1]["blocking_check_ids"] =
                Value::Array(vec![Value::String("cargo-test".into())]);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_REQUIRED_CHECK_UNMAPPED"));
    }

    #[test]
    fn zero_length_specification_is_rejected() {
        let report = validate_value_and_manifest(|_run, manifest| {
            manifest["tasks"][0]["spec_byte_length"] = Value::from(0);
        });
        assert!(!report.valid);
        assert!(has_code(&report, "E_SPEC_BYTE_LENGTH"));
    }


}
