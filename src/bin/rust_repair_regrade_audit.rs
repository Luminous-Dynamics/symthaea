//! Compare primary and independent fresh-image repository-repair grade reports.
//!
//! This binary performs evidence comparison only. It does not run Docker or tests,
//! and it does not authenticate the runner. Each input report must be independently
//! pinned by a SHA-256 obtained from a separately retained artifact receipt.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PROTOCOL_ID: &str = "symthaea-rust-repair";
const REPORT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Outcome {
    Pass,
    Fail,
    Unresolved,
}

impl Outcome {
    fn resolved(self) -> bool {
        self != Self::Unresolved
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ScoreTask {
    instance_id: String,
    outcome: Outcome,
    prediction_patch_sha256: Option<String>,
    scored_log_sha256: Option<String>,
    evaluation_evidence_sha256: Option<String>,
    container_image_id: Option<String>,
    required_test_count: usize,
    required_tests_passing: usize,
    runtime_platform_observed: Option<String>,
    workspace_clean_exit_code: Option<i64>,
    network_mode_observed: Option<String>,
    repo_head_sha_observed: Option<String>,
    scored_test_exit_code: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
struct ScoreReport {
    format_version: u32,
    protocol_id: String,
    task_manifest_sha256: String,
    task_manifest_file_sha256: String,
    candidate_jsonl_sha256: String,
    predictions_sha256: String,
    task_count: usize,
    scored_count: usize,
    success_count: usize,
    failure_count: usize,
    unresolved_count: usize,
    lockable: bool,
    pass_at_1: Option<f64>,
    tasks: Vec<ScoreTask>,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestTask {
    instance_id: String,
    repo_head_sha: String,
    runtime_platform: String,
    container_image_id: String,
    network_mode: String,
    workspace_clean: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct LockedTaskManifest {
    format_version: u32,
    protocol_id: String,
    task_records_sha256: String,
    candidate_jsonl_sha256: String,
    eligible_task_count: usize,
    tasks: Vec<ManifestTask>,
}

#[derive(Clone, Debug)]
struct LoadedReport {
    sha256: String,
    canonical_path: PathBuf,
    report: ScoreReport,
    tasks: BTreeMap<String, ScoreTask>,
}

#[derive(Clone, Debug)]
struct Cli {
    primary_report: PathBuf,
    expected_primary_report_sha256: String,
    regrade_report: PathBuf,
    expected_regrade_report_sha256: String,
    task_manifest: PathBuf,
    expected_task_manifest_file_sha256: String,
    expected_task_manifest_sha256: String,
    report_out: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct TaskComparison {
    instance_id: String,
    primary_outcome: Outcome,
    regrade_outcome: Outcome,
    relation: String,
    prediction_patch_sha256: Option<String>,
    primary_log_sha256: Option<String>,
    regrade_log_sha256: Option<String>,
    primary_evidence_sha256: Option<String>,
    regrade_evidence_sha256: Option<String>,
    container_image_id: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RegradeSummary {
    assigned_task_count: usize,
    primary_pass_count: usize,
    primary_fail_count: usize,
    primary_unresolved_count: usize,
    regrade_pass_count: usize,
    regrade_fail_count: usize,
    regrade_unresolved_count: usize,
    concordant_pass_count: usize,
    concordant_fail_count: usize,
    primary_only_pass_count: usize,
    regrade_only_pass_count: usize,
    unresolved_in_either_grade_count: usize,
    fully_resolved_in_both_count: usize,
    disagreement_count: usize,
    primary_false_positive_fraction: Option<f64>,
    primary_false_negative_fraction: Option<f64>,
    primary_pass_rate_full_denominator: f64,
    regrade_pass_rate_full_denominator: f64,
    double_verified_pass_rate_full_denominator: f64,
    strict_pass_at_1: Option<f64>,
    lockable: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RegradeAuditReport {
    format_version: u32,
    protocol_id: String,
    task_manifest_sha256: String,
    candidate_jsonl_sha256: String,
    predictions_sha256: String,
    primary_report_sha256: String,
    regrade_report_sha256: String,
    primary_report_path: String,
    regrade_report_path: String,
    summary: RegradeSummary,
    tasks: Vec<TaskComparison>,
    interpretation: String,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_image_id(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| is_hex(digest, 64))
}

fn valid_digest_field(value: &Option<String>, field: &str, instance_id: &str) -> Result<(), String> {
    if let Some(digest) = value
        && !is_hex(digest, 64)
    {
        return Err(format!(
            "task {instance_id:?} has malformed {field} SHA-256 digest"
        ));
    }
    Ok(())
}

fn read_pinned_report(
    path: &Path,
    expected_sha256: &str,
    expected_task_manifest_sha256: &str,
) -> Result<LoadedReport, String> {
    if !is_hex(expected_sha256, 64) {
        return Err("expected report SHA-256 must be 64 hexadecimal characters".to_string());
    }
    let canonical_path = fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve report {}: {error}", path.display()))?;
    if !canonical_path.is_file() {
        return Err(format!("score report {} is not a regular file", path.display()));
    }
    let bytes = fs::read(&canonical_path)
        .map_err(|error| format!("cannot read score report {}: {error}", canonical_path.display()))?;
    let observed_sha256 = sha256_bytes(&bytes);
    if observed_sha256 != expected_sha256.to_ascii_lowercase() {
        return Err(format!(
            "score report digest mismatch for {}: expected {}, observed {}",
            canonical_path.display(),
            expected_sha256,
            observed_sha256
        ));
    }
    let report: ScoreReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("score report {} is malformed JSON: {error}", canonical_path.display()))?;
    if report.format_version != REPORT_VERSION || report.protocol_id != PROTOCOL_ID {
        return Err(format!(
            "score report {} has unsupported format_version or protocol_id",
            canonical_path.display()
        ));
    }
    if !is_hex(&report.task_manifest_sha256, 64)
        || report.task_manifest_sha256 != expected_task_manifest_sha256.to_ascii_lowercase()
    {
        return Err(format!(
            "score report {} does not match the externally pinned task manifest",
            canonical_path.display()
        ));
    }
    for (field, digest) in [
        ("candidate_jsonl_sha256", &report.candidate_jsonl_sha256),
        ("predictions_sha256", &report.predictions_sha256),
    ] {
        if !is_hex(digest, 64) {
            return Err(format!(
                "score report {} has malformed {field}",
                canonical_path.display()
            ));
        }
    }
    if report.tasks.is_empty() || report.task_count != report.tasks.len() {
        return Err(format!(
            "score report {} task_count is empty or differs from task rows",
            canonical_path.display()
        ));
    }

    let mut tasks = BTreeMap::new();
    for task in &report.tasks {
        if task.instance_id.trim().is_empty() {
            return Err(format!(
                "score report {} has an empty instance_id",
                canonical_path.display()
            ));
        }
        if task.required_tests_passing > task.required_test_count {
            return Err(format!(
                "task {:?} required_tests_passing exceeds required_test_count",
                task.instance_id
            ));
        }
        valid_digest_field(&task.prediction_patch_sha256, "prediction patch", &task.instance_id)?;
        valid_digest_field(&task.scored_log_sha256, "scored log", &task.instance_id)?;
        valid_digest_field(
            &task.evaluation_evidence_sha256,
            "evaluation evidence",
            &task.instance_id,
        )?;
        if let Some(image_id) = &task.container_image_id
            && !valid_image_id(image_id)
        {
            return Err(format!("task {:?} has malformed container_image_id", task.instance_id));
        }
        if let Some(repo_head) = &task.repo_head_sha_observed
            && !(is_hex(repo_head, 40) || is_hex(repo_head, 64))
        {
            return Err(format!("task {:?} has malformed observed repository HEAD", task.instance_id));
        }
        if task.outcome.resolved() {
            if task.prediction_patch_sha256.is_none()
                || task.scored_log_sha256.is_none()
                || task.evaluation_evidence_sha256.is_none()
                || task.container_image_id.is_none()
                || task.runtime_platform_observed.as_deref() != Some("linux")
                || task.workspace_clean_exit_code != Some(0)
                || task.network_mode_observed.as_deref() != Some("none")
                || task.repo_head_sha_observed.is_none()
                || task.required_test_count == 0
            {
                return Err(format!(
                    "resolved task {:?} lacks patch/log/evidence/image/Linux/offline/clean-worktree/HEAD/test-count evidence",
                    task.instance_id
                ));
            }
            if task.outcome == Outcome::Pass
                && (task.required_tests_passing != task.required_test_count
                    || task.scored_test_exit_code != Some(0))
            {
                return Err(format!(
                    "passing task {:?} has incomplete required-test count or nonzero test exit",
                    task.instance_id
                ));
            }
        }
        if tasks.insert(task.instance_id.clone(), task.clone()).is_some() {
            return Err(format!(
                "score report {} repeats instance_id {:?}",
                canonical_path.display(),
                task.instance_id
            ));
        }
    }

    let success_count = tasks.values().filter(|task| task.outcome == Outcome::Pass).count();
    let failure_count = tasks.values().filter(|task| task.outcome == Outcome::Fail).count();
    let unresolved_count = tasks.values().filter(|task| task.outcome == Outcome::Unresolved).count();
    let scored_count = success_count + failure_count;
    let expected_lockable = !tasks.is_empty() && unresolved_count == 0;
    for (field, observed, expected) in [
        ("success_count", report.success_count, success_count),
        ("failure_count", report.failure_count, failure_count),
        ("unresolved_count", report.unresolved_count, unresolved_count),
        ("scored_count", report.scored_count, scored_count),
        ("task_count", report.task_count, tasks.len()),
    ] {
        if observed != expected {
            return Err(format!(
                "score report {} has inconsistent {field} aggregate",
                canonical_path.display()
            ));
        }
    }
    if report.lockable != expected_lockable {
        return Err(format!(
            "score report {} has inconsistent lockable aggregate",
            canonical_path.display()
        ));
    }
    let expected_pass_at_1 = expected_lockable.then(|| success_count as f64 / tasks.len() as f64);
    if !same_optional_float(report.pass_at_1, expected_pass_at_1) {
        return Err(format!(
            "score report {} has inconsistent pass_at_1 aggregate",
            canonical_path.display()
        ));
    }

    Ok(LoadedReport {
        sha256: observed_sha256,
        canonical_path,
        report,
        tasks,
    })
}

fn same_optional_float(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => a.is_finite() && b.is_finite() && (a - b).abs() <= 1.0e-12,
        _ => false,
    }
}

fn classify(primary: Outcome, regrade: Outcome) -> &'static str {
    match (primary, regrade) {
        (Outcome::Pass, Outcome::Pass) => "concordant_pass",
        (Outcome::Fail, Outcome::Fail) => "concordant_fail",
        (Outcome::Pass, Outcome::Fail) => "primary_only_pass",
        (Outcome::Fail, Outcome::Pass) => "regrade_only_pass",
        (Outcome::Unresolved, Outcome::Unresolved) => "both_unresolved",
        (Outcome::Unresolved, _) => "primary_unresolved",
        (_, Outcome::Unresolved) => "regrade_unresolved",
    }
}

fn audit_pair(primary: &ScoreTask, regrade: &ScoreTask) -> Result<(), String> {
    if primary.instance_id != regrade.instance_id {
        return Err("internal task pairing mismatch".to_string());
    }
    if primary.prediction_patch_sha256 != regrade.prediction_patch_sha256 {
        return Err(format!(
            "task {:?} patch digest differs between primary and regrade",
            primary.instance_id
        ));
    }
    if let (Some(left), Some(right)) = (&primary.container_image_id, &regrade.container_image_id)
        && left != right
    {
        return Err(format!(
            "task {:?} immutable image ID differs between primary and regrade",
            primary.instance_id
        ));
    }
    if let (Some(left), Some(right)) = (&primary.repo_head_sha_observed, &regrade.repo_head_sha_observed)
        && left.to_ascii_lowercase() != right.to_ascii_lowercase()
    {
        return Err(format!(
            "task {:?} repository HEAD differs between primary and regrade",
            primary.instance_id
        ));
    }
    if primary.outcome.resolved() && regrade.outcome.resolved() {
        if primary.scored_log_sha256 == regrade.scored_log_sha256 {
            return Err(format!(
                "task {:?} primary/regrade scored-log hashes are identical; fresh execution is unsubstantiated",
                primary.instance_id
            ));
        }
        if primary.evaluation_evidence_sha256 == regrade.evaluation_evidence_sha256 {
            return Err(format!(
                "task {:?} primary/regrade evidence hashes are identical; fresh execution is unsubstantiated",
                primary.instance_id
            ));
        }
    }
    Ok(())
}

fn build_report(
    primary: &LoadedReport,
    regrade: &LoadedReport,
    expected_task_manifest_sha256: &str,
    task_manifest_file_sha256: &str,
) -> Result<RegradeAuditReport, String> {
    if primary.canonical_path == regrade.canonical_path || primary.sha256 == regrade.sha256 {
        return Err("primary and regrade must be distinct, independently pinned report artifacts".to_string());
    }
    if primary.report.task_manifest_sha256 != regrade.report.task_manifest_sha256
        || primary.report.candidate_jsonl_sha256 != regrade.report.candidate_jsonl_sha256
        || primary.report.predictions_sha256 != regrade.report.predictions_sha256
        || primary.report.task_count != regrade.report.task_count
    {
        return Err("primary and regrade reports differ in task manifest, candidate, predictions, or task count".to_string());
    }
    if primary.report.task_manifest_sha256 != expected_task_manifest_sha256.to_ascii_lowercase() {
        return Err("primary/regrade task manifest does not match the expected external pin".to_string());
    }
    let primary_ids = primary.tasks.keys().cloned().collect::<BTreeSet<_>>();
    let regrade_ids = regrade.tasks.keys().cloned().collect::<BTreeSet<_>>();
    if primary_ids != regrade_ids {
        return Err("primary and regrade task-ID sets differ".to_string());
    }

    let mut summary = RegradeSummary {
        assigned_task_count: primary.tasks.len(),
        primary_pass_count: 0,
        primary_fail_count: 0,
        primary_unresolved_count: 0,
        regrade_pass_count: 0,
        regrade_fail_count: 0,
        regrade_unresolved_count: 0,
        concordant_pass_count: 0,
        concordant_fail_count: 0,
        primary_only_pass_count: 0,
        regrade_only_pass_count: 0,
        unresolved_in_either_grade_count: 0,
        fully_resolved_in_both_count: 0,
        disagreement_count: 0,
        primary_false_positive_fraction: None,
        primary_false_negative_fraction: None,
        primary_pass_rate_full_denominator: 0.0,
        regrade_pass_rate_full_denominator: 0.0,
        double_verified_pass_rate_full_denominator: 0.0,
        strict_pass_at_1: None,
        lockable: false,
    };
    let mut comparisons = Vec::with_capacity(summary.assigned_task_count);

    for (instance_id, primary_task) in &primary.tasks {
        let regrade_task = regrade.tasks.get(instance_id)
            .ok_or_else(|| format!("regrade missing task {instance_id:?}"))?;
        audit_pair(primary_task, regrade_task)?;
        match primary_task.outcome {
            Outcome::Pass => summary.primary_pass_count += 1,
            Outcome::Fail => summary.primary_fail_count += 1,
            Outcome::Unresolved => summary.primary_unresolved_count += 1,
        }
        match regrade_task.outcome {
            Outcome::Pass => summary.regrade_pass_count += 1,
            Outcome::Fail => summary.regrade_fail_count += 1,
            Outcome::Unresolved => summary.regrade_unresolved_count += 1,
        }
        let relation = classify(primary_task.outcome, regrade_task.outcome);
        match relation {
            "concordant_pass" => {
                summary.concordant_pass_count += 1;
                summary.fully_resolved_in_both_count += 1;
            }
            "concordant_fail" => {
                summary.concordant_fail_count += 1;
                summary.fully_resolved_in_both_count += 1;
            }
            "primary_only_pass" => {
                summary.primary_only_pass_count += 1;
                summary.disagreement_count += 1;
                summary.fully_resolved_in_both_count += 1;
            }
            "regrade_only_pass" => {
                summary.regrade_only_pass_count += 1;
                summary.disagreement_count += 1;
                summary.fully_resolved_in_both_count += 1;
            }
            _ => summary.unresolved_in_either_grade_count += 1,
        }
        comparisons.push(TaskComparison {
            instance_id: instance_id.clone(),
            primary_outcome: primary_task.outcome,
            regrade_outcome: regrade_task.outcome,
            relation: relation.to_string(),
            prediction_patch_sha256: primary_task.prediction_patch_sha256.clone(),
            primary_log_sha256: primary_task.scored_log_sha256.clone(),
            regrade_log_sha256: regrade_task.scored_log_sha256.clone(),
            primary_evidence_sha256: primary_task.evaluation_evidence_sha256.clone(),
            regrade_evidence_sha256: regrade_task.evaluation_evidence_sha256.clone(),
            container_image_id: primary_task.container_image_id.clone(),
        });
    }

    let denominator = summary.assigned_task_count as f64;
    summary.primary_pass_rate_full_denominator = summary.primary_pass_count as f64 / denominator;
    summary.regrade_pass_rate_full_denominator = summary.regrade_pass_count as f64 / denominator;
    summary.double_verified_pass_rate_full_denominator =
        summary.concordant_pass_count as f64 / denominator;
    summary.unresolved_in_either_grade_count = summary.assigned_task_count
        - summary.fully_resolved_in_both_count;
    if summary.primary_pass_count > 0 {
        summary.primary_false_positive_fraction =
            Some(summary.primary_only_pass_count as f64 / summary.primary_pass_count as f64);
    }
    if summary.regrade_pass_count > 0 {
        summary.primary_false_negative_fraction =
            Some(summary.regrade_only_pass_count as f64 / summary.regrade_pass_count as f64);
    }
    summary.lockable = summary.unresolved_in_either_grade_count == 0;
    if summary.lockable {
        summary.strict_pass_at_1 =
            Some(summary.concordant_pass_count as f64 / summary.assigned_task_count as f64);
    }

    comparisons.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    let interpretation = if summary.unresolved_in_either_grade_count > 0 {
        "INCOMPLETE: at least one assigned task is unresolved in one or both grades. Do not publish strict pass@1. Report both raw rates, concordance and missingness."
    } else if summary.disagreement_count > 0 {
        "GRADER DISAGREEMENT: at least one task changed between primary and fresh-image regrade. Use the double-verified pass rate for the conservative bound and investigate every disagreement before claiming performance."
    } else {
        "CONCORDANT: primary and fresh-image regrade agree on every assigned task. This supports repeatability for these artifacts, but does not authenticate the runner or establish state-of-the-art without an independent protocol-matched comparison."
    };
    Ok(RegradeAuditReport {
        format_version: REPORT_VERSION,
        protocol_id: "symthaea-rust-regrade-audit-v1".to_string(),
        task_manifest_sha256: expected_task_manifest_sha256.to_ascii_lowercase(),
        task_manifest_file_sha256: task_manifest_file_sha256.to_ascii_lowercase(),
        candidate_jsonl_sha256: primary.report.candidate_jsonl_sha256.clone(),
        predictions_sha256: primary.report.predictions_sha256.clone(),
        primary_report_sha256: primary.sha256.clone(),
        regrade_report_sha256: regrade.sha256.clone(),
        primary_report_path: primary.canonical_path.display().to_string(),
        regrade_report_path: regrade.canonical_path.display().to_string(),
        summary,
        tasks: comparisons,
        interpretation: interpretation.to_string(),
    })
}

fn parse_cli(args: &[String]) -> Result<Option<Cli>, String> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        println!(concat!(
            "Usage: rust-repair-regrade-audit ",
            "--primary-report PATH --expected-primary-report-sha256 HEX ",
            "--regrade-report PATH --expected-regrade-report-sha256 HEX ",
            "--task-manifest PATH --expected-task-manifest-file-sha256 HEX ",
            "--expected-task-manifest-sha256 HEX --report-out PATH"
        ));
        return Ok(None);
    }
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument {key:?}"));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if values.insert(key.clone(), value.clone()).is_some() {
            return Err(format!("duplicate argument {key}"));
        }
        index += 2;
    }
    let required = [
        "--primary-report",
        "--expected-primary-report-sha256",
        "--regrade-report",
        "--expected-regrade-report-sha256",
        "--task-manifest",
        "--expected-task-manifest-file-sha256",
        "--expected-task-manifest-sha256",
        "--report-out",
    ];
    for key in required {
        if !values.contains_key(key) {
            return Err(format!("missing required argument {key}"));
        }
    }
    if values.keys().any(|key| !required.contains(&key.as_str())) {
        return Err("unknown command-line argument".to_string());
    }
    let required_digest = |key: &str| -> Result<String, String> {
        let value = values.get(key).expect("required argument checked").to_ascii_lowercase();
        if !is_hex(&value, 64) {
            return Err(format!("{key} must be 64 hexadecimal characters"));
        }
        Ok(value)
    };
    Ok(Some(Cli {
        primary_report: PathBuf::from(values.get("--primary-report").expect("required argument")),
        expected_primary_report_sha256: required_digest("--expected-primary-report-sha256")?,
        regrade_report: PathBuf::from(values.get("--regrade-report").expect("required argument")),
        expected_regrade_report_sha256: required_digest("--expected-regrade-report-sha256")?,
        task_manifest: PathBuf::from(values.get("--task-manifest").expect("required argument")),
        expected_task_manifest_file_sha256: required_digest("--expected-task-manifest-file-sha256")?,
        expected_task_manifest_sha256: required_digest("--expected-task-manifest-sha256")?,
        report_out: PathBuf::from(values.get("--report-out").expect("required argument")),
    }))
}

fn read_pinned_manifest(
    path: &Path,
    expected_file_sha256: &str,
    expected_record_sha256: &str,
) -> Result<(LockedTaskManifest, BTreeMap<String, ManifestTask>), String> {
    if !is_hex(expected_file_sha256, 64) || !is_hex(expected_record_sha256, 64) {
        return Err("expected manifest file and task-record digests must be 64 hexadecimal characters".to_string());
    }
    let canonical_path = fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve task manifest {}: {error}", path.display()))?;
    if !canonical_path.is_file() {
        return Err(format!("task manifest {} is not a regular file", canonical_path.display()));
    }
    let bytes = fs::read(&canonical_path)
        .map_err(|error| format!("cannot read task manifest {}: {error}", canonical_path.display()))?;
    let observed_file_sha256 = sha256_bytes(&bytes);
    if observed_file_sha256 != expected_file_sha256.to_ascii_lowercase() {
        return Err(format!(
            "task manifest file digest mismatch: expected {}, observed {}",
            expected_file_sha256, observed_file_sha256
        ));
    }
    let manifest: LockedTaskManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("task manifest {} is malformed JSON: {error}", canonical_path.display()))?;
    if manifest.format_version != 1 || manifest.protocol_id != PROTOCOL_ID {
        return Err("task manifest has unsupported format_version or protocol_id".to_string());
    }
    if manifest.task_records_sha256 != expected_record_sha256.to_ascii_lowercase()
        || !is_hex(&manifest.task_records_sha256, 64)
        || !is_hex(&manifest.candidate_jsonl_sha256, 64)
    {
        return Err("task manifest record digest or candidate digest differs from the external pin".to_string());
    }
    if manifest.eligible_task_count != manifest.tasks.len() || manifest.tasks.is_empty() {
        return Err("task manifest eligible_task_count is empty or differs from task rows".to_string());
    }
    let mut tasks = BTreeMap::new();
    for task in &manifest.tasks {
        if task.instance_id.trim().is_empty()
            || !(is_hex(&task.repo_head_sha, 40) || is_hex(&task.repo_head_sha, 64))
            || !valid_image_id(&task.container_image_id)
            || task.runtime_platform != "linux"
            || task.network_mode != "none"
            || !task.workspace_clean
        {
            return Err(format!(
                "task manifest row {:?} violates repository/image/Linux/offline/clean requirements",
                task.instance_id
            ));
        }
        if tasks.insert(task.instance_id.clone(), task.clone()).is_some() {
            return Err(format!("task manifest repeats instance_id {:?}", task.instance_id));
        }
    }
    Ok((manifest, tasks))
}

fn validate_manifest_bindings(
    primary: &LoadedReport,
    regrade: &LoadedReport,
    manifest: &LockedTaskManifest,
    manifest_tasks: &BTreeMap<String, ManifestTask>,
) -> Result<(), String> {
    if primary.report.candidate_jsonl_sha256 != manifest.candidate_jsonl_sha256
        || regrade.report.candidate_jsonl_sha256 != manifest.candidate_jsonl_sha256
    {
        return Err("primary/regrade candidate split digest differs from the locked manifest".to_string());
    }
    let expected_ids = manifest_tasks.keys().cloned().collect::<BTreeSet<_>>();
    if primary.tasks.keys().cloned().collect::<BTreeSet<_>>() != expected_ids
        || regrade.tasks.keys().cloned().collect::<BTreeSet<_>>() != expected_ids
    {
        return Err("primary/regrade exact task-ID sets differ from the locked manifest".to_string());
    }
    for (instance_id, expected) in manifest_tasks {
        for (label, report) in [("primary", primary), ("regrade", regrade)] {
            let row = report.tasks.get(instance_id)
                .ok_or_else(|| format!("{label} report lacks manifest task {instance_id:?}"))?;
            if let Some(image_id) = &row.container_image_id
                && image_id != &expected.container_image_id
            {
                return Err(format!("{label} task {instance_id:?} image differs from locked manifest"));
            }
            if let Some(repo_head) = &row.repo_head_sha_observed
                && repo_head.to_ascii_lowercase() != expected.repo_head_sha.to_ascii_lowercase()
            {
                return Err(format!("{label} task {instance_id:?} repository HEAD differs from locked manifest"));
            }
            if let Some(platform) = &row.runtime_platform_observed
                && platform != &expected.runtime_platform
            {
                return Err(format!("{label} task {instance_id:?} platform differs from locked manifest"));
            }
            if let Some(network_mode) = &row.network_mode_observed
                && network_mode != &expected.network_mode
            {
                return Err(format!("{label} task {instance_id:?} network policy differs from locked manifest"));
            }
        }
    }
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    let primary = read_pinned_report(
        &cli.primary_report,
        &cli.expected_primary_report_sha256,
        &cli.expected_task_manifest_sha256,
    )?;
    let regrade = read_pinned_report(
        &cli.regrade_report,
        &cli.expected_regrade_report_sha256,
        &cli.expected_task_manifest_sha256,
    )?;
    let (manifest, manifest_tasks) = read_pinned_manifest(
        &cli.task_manifest,
        &cli.expected_task_manifest_file_sha256,
        &cli.expected_task_manifest_sha256,
    )?;
    validate_manifest_bindings(&primary, &regrade, &manifest, &manifest_tasks)?;
    let report = build_report(
        &primary,
        &regrade,
        &cli.expected_task_manifest_sha256,
        &cli.expected_task_manifest_file_sha256,
    )?;
    if let Some(parent) = cli.report_out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create output directory {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize regrade audit report: {error}"))?;
    bytes.push(b'\n');
    fs::write(&cli.report_out, &bytes)
        .map_err(|error| format!("cannot write report {}: {error}", cli.report_out.display()))?;
    println!(
        "RE-GRADE AUDIT: {} tasks, {} concordant passes, {} disagreements, {} unresolved; report sha256:{} ({})",
        report.summary.assigned_task_count,
        report.summary.concordant_pass_count,
        report.summary.disagreement_count,
        report.summary.unresolved_in_either_grade_count,
        sha256_bytes(&bytes),
        cli.report_out.display()
    );
    Ok(())
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let cli = match parse_cli(&args) {
        Ok(Some(cli)) => cli,
        Ok(None) => return,
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(cli) {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(instance_id: &str, outcome: Outcome, patch: &str, log: &str, evidence: &str) -> ScoreTask {
        ScoreTask {
            instance_id: instance_id.to_string(),
            outcome,
            prediction_patch_sha256: Some(patch.repeat(64 / patch.len())),
            scored_log_sha256: Some(log.repeat(64 / log.len())),
            evaluation_evidence_sha256: Some(evidence.repeat(64 / evidence.len())),
            container_image_id: Some(format!("sha256:{}", "d".repeat(64))),
            required_test_count: 2,
            required_tests_passing: if outcome == Outcome::Pass { 2 } else { 1 },
            runtime_platform_observed: Some("linux".to_string()),
            workspace_clean_exit_code: Some(0),
            network_mode_observed: Some("none".to_string()),
            repo_head_sha_observed: Some("a".repeat(40)),
            scored_test_exit_code: Some(if outcome == Outcome::Pass { 0 } else { 1 }),
        }
    }

    fn report(tasks: Vec<ScoreTask>) -> ScoreReport {
        let success_count = tasks.iter().filter(|task| task.outcome == Outcome::Pass).count();
        let failure_count = tasks.iter().filter(|task| task.outcome == Outcome::Fail).count();
        let unresolved_count = tasks.len() - success_count - failure_count;
        let lockable = !tasks.is_empty() && unresolved_count == 0;
        ScoreReport {
            format_version: REPORT_VERSION,
            protocol_id: PROTOCOL_ID.to_string(),
            task_manifest_sha256: "a".repeat(64),
            candidate_jsonl_sha256: "b".repeat(64),
            predictions_sha256: "c".repeat(64),
            task_count: tasks.len(),
            scored_count: success_count + failure_count,
            success_count,
            failure_count,
            unresolved_count,
            lockable,
            pass_at_1: lockable.then(|| success_count as f64 / tasks.len() as f64),
            tasks,
        }
    }

    fn loaded(report: ScoreReport, digest: &str) -> LoadedReport {
        let tasks = report.tasks.iter().cloned().map(|task| (task.instance_id.clone(), task)).collect();
        LoadedReport {
            sha256: digest.to_string(),
            canonical_path: PathBuf::from(format!("/{digest}.json")),
            report,
            tasks,
        }
    }

    #[test]
    fn sha256_uses_standard_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn outcome_relations_identify_both_directions_of_disagreement() {
        assert_eq!(classify(Outcome::Pass, Outcome::Pass), "concordant_pass");
        assert_eq!(classify(Outcome::Fail, Outcome::Fail), "concordant_fail");
        assert_eq!(classify(Outcome::Pass, Outcome::Fail), "primary_only_pass");
        assert_eq!(classify(Outcome::Fail, Outcome::Pass), "regrade_only_pass");
        assert_eq!(classify(Outcome::Unresolved, Outcome::Pass), "primary_unresolved");
        assert_eq!(classify(Outcome::Pass, Outcome::Unresolved), "regrade_unresolved");
    }

    #[test]
    fn patch_mismatch_is_a_protocol_error() {
        let left = task("task-1", Outcome::Pass, "1", "2", "3");
        let right = task("task-1", Outcome::Pass, "4", "5", "6");
        assert!(audit_pair(&left, &right).unwrap_err().contains("patch digest differs"));
    }

    #[test]
    fn reuse_of_primary_logs_is_not_independent_regrading() {
        let left = task("task-1", Outcome::Pass, "1", "2", "3");
        let right = task("task-1", Outcome::Pass, "1", "2", "4");
        assert!(audit_pair(&left, &right).unwrap_err().contains("scored-log hashes are identical"));
    }

    #[test]
    fn regrade_disagreement_reduces_double_verified_rate() {
        let primary = loaded(report(vec![
            task("task-a", Outcome::Pass, "1", "2", "3"),
            task("task-b", Outcome::Pass, "4", "5", "6"),
        ]), &"1".repeat(64));
        let regrade = loaded(report(vec![
            task("task-a", Outcome::Pass, "1", "7", "8"),
            task("task-b", Outcome::Fail, "4", "9", "a"),
        ]), &"2".repeat(64));
        let result = build_report(&primary, &regrade, &"a".repeat(64)).unwrap();
        assert_eq!(result.summary.concordant_pass_count, 1);
        assert_eq!(result.summary.primary_only_pass_count, 1);
        assert_eq!(result.summary.disagreement_count, 1);
        assert_eq!(result.summary.double_verified_pass_rate_full_denominator, 0.5);
        assert_eq!(result.summary.strict_pass_at_1, Some(0.5));
    }

    #[test]
    fn unresolved_in_either_grade_withholds_strict_pass_at_1() {
        let primary = loaded(report(vec![
            task("task-a", Outcome::Pass, "1", "2", "3"),
            task("task-b", Outcome::Unresolved, "4", "5", "6"),
        ]), &"1".repeat(64));
        let regrade = loaded(report(vec![
            task("task-a", Outcome::Pass, "1", "7", "8"),
            task("task-b", Outcome::Fail, "4", "9", "a"),
        ]), &"2".repeat(64));
        let result = build_report(&primary, &regrade, &"a".repeat(64)).unwrap();
        assert_eq!(result.summary.unresolved_in_either_grade_count, 1);
        assert_eq!(result.summary.strict_pass_at_1, None);
    }

    #[test]
    fn report_reuse_or_manifest_mismatch_is_rejected() {
        let left = loaded(report(vec![task("task-a", Outcome::Pass, "1", "2", "3")]), &"1".repeat(64));
        let mut right_report = report(vec![task("task-a", Outcome::Pass, "1", "7", "8")]);
        right_report.task_manifest_sha256 = "f".repeat(64);
        let right = loaded(right_report, &"2".repeat(64));
        assert!(build_report(&left, &right, &"a".repeat(64)).unwrap_err().contains("task manifest, candidate"));
        let reused = loaded(left.report.clone(), &"1".repeat(64));
        assert!(build_report(&left, &reused, &"a".repeat(64)).unwrap_err().contains("distinct"));
    }

    #[test]
    fn external_report_digest_is_required_by_cli() {
        let args = vec![
            "--primary-report".to_string(), "primary.json".to_string(),
            "--expected-primary-report-sha256".to_string(), "a".repeat(64),
            "--regrade-report".to_string(), "regrade.json".to_string(),
            "--expected-regrade-report-sha256".to_string(), "b".repeat(64),
            "--expected-task-manifest-sha256".to_string(), "c".repeat(64),
        ];
        assert!(parse_cli(&args).unwrap_err().contains("--report-out"));
    }

    #[test]
    fn report_aggregates_are_recomputed_not_trusted() {
        let temp = tempfile::tempdir().unwrap();
        let mut raw = serde_json::to_value(report(vec![
            task("task-a", Outcome::Pass, "1", "2", "3"),
        ])).unwrap();
        raw["success_count"] = Value::from(0);
        let bytes = serde_json::to_vec(&raw).unwrap();
        fs::write(temp.path().join("bad.json"), &bytes).unwrap();
        assert!(read_pinned_report(
            &temp.path().join("bad.json"),
            &sha256_bytes(&bytes),
            &"a".repeat(64)
        ).unwrap_err().contains("inconsistent success_count"));
    }
}
