//! Fail-closed, offline dataset intake gate for ConnectomeBench2 V2.
//!
//! PASS covers manifest consistency and local byte integrity only. It does not
//! establish biological truth, correct labels, independent samples, or model quality.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const DATASET_ID: &str = "jeffbbrown2/ConnectomeBench2";
const DATASET_REVISION: &str = "v2";
const UPSTREAM_REPO: &str = "https://github.com/timfarkas/ConnectomeBench2";
const ALLOWED_INPUT_COLUMNS: &[&str] =
    &["geometry", "geometry_single", "em_xy", "em_xz", "em_yz"];
const EM_INPUT_COLUMNS: &[&str] = &["em_xy", "em_xz", "em_yz"];

#[derive(Debug, Parser)]
#[command(
    name = "connectomics-intake-audit",
    version,
    about = "Validate a ConnectomeBench2 V2 intake manifest and hash local artifacts"
)]
struct Args {
    /// JSON intake manifest conforming to CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json.
    #[arg(long)]
    manifest: PathBuf,

    /// Root containing the files listed in input_files[].path.
    #[arg(long)]
    root: PathBuf,

    /// Write deterministic JSON report here (stdout if omitted).
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    dataset_id: String,
    dataset_revision: String,
    dataset_revision_commit_sha: String,
    upstream_repo: String,
    upstream_commit: String,
    task: String,
    task_route_value: String,
    label_column: String,
    label_true_semantics: String,
    label_mapping_reference: String,
    split: String,
    input_mode: String,
    model_input_columns: Vec<String>,
    input_files: Vec<InputFile>,
    #[serde(default)]
    geometry_preprocessing_profile: Option<String>,
    #[serde(default)]
    em_preprocessing_profile: Option<String>,
    #[serde(default)]
    reviewed_by: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    mapping_reviewed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputFile {
    path: String,
    sha256: String,
    bytes: u64,
    split: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    report_schema_version: u32,
    tool: String,
    tool_version: String,
    manifest_sha256: String,
    dataset_id: String,
    dataset_revision: String,
    task: String,
    declared_split: String,
    status: String,
    checks: Vec<Check>,
    files: Vec<FileAudit>,
    summary: BTreeMap<String, u64>,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct Check {
    code: String,
    passed: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct FileAudit {
    path: String,
    declared_split: String,
    declared_bytes: u64,
    declared_sha256: String,
    actual_bytes: Option<u64>,
    actual_sha256: Option<String>,
    status: String,
}

impl Check {
    fn pass(code: &str, detail: impl Into<String>) -> Self {
        Self { code: code.into(), passed: true, detail: detail.into() }
    }

    fn fail(code: &str, detail: impl Into<String>) -> Self {
        Self { code: code.into(), passed: false, detail: detail.into() }
    }
}

fn sha256_reader(mut reader: impl Read) -> std::io::Result<(u64, String)> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        hasher.update(&buffer[..read]);
    }
    Ok((total, hex::encode(hasher.finalize())))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, exact_len: usize) -> bool {
    value.len() == exact_len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn safe_relative_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path.components().all(|component| matches!(component, Component::Normal(_)))
}

fn semantic_checks(manifest: &Manifest) -> Vec<Check> {
    let mut checks = Vec::new();

    for (name, actual, expected) in [
        ("dataset_id", manifest.dataset_id.as_str(), DATASET_ID),
        ("dataset_revision", manifest.dataset_revision.as_str(), DATASET_REVISION),
        ("upstream_repo", manifest.upstream_repo.as_str(), UPSTREAM_REPO),
    ] {
        if actual == expected {
            checks.push(Check::pass(&format!("pinned_{name}"), format!("{name} matches the required value")));
        } else {
            checks.push(Check::fail(
                &format!("pinned_{name}"),
                format!("{name} must equal {expected:?}; got {actual:?}"),
            ));
        }
    }

    if manifest.schema_version == 1 {
        checks.push(Check::pass("manifest_schema_version", "schema_version is 1"));
    } else {
        checks.push(Check::fail("manifest_schema_version", "only manifest schema_version 1 is supported"));
    }

    for (name, value) in [
        ("dataset_revision_commit_sha", manifest.dataset_revision_commit_sha.as_str()),
        ("upstream_commit", manifest.upstream_commit.as_str()),
    ] {
        if is_lower_hex(value, 40) {
            checks.push(Check::pass(&format!("{name}_format"), format!("{name} is a lowercase 40-character SHA")));
        } else {
            checks.push(Check::fail(&format!("{name}_format"), format!("{name} must be a lowercase 40-character SHA")));
        }
    }

    let mapping_attested = manifest.mapping_reviewed
        && !manifest.label_mapping_reference.trim().is_empty()
        && !manifest.task_route_value.trim().is_empty()
        && manifest.reviewed_by.as_deref().is_some_and(|v| !v.trim().is_empty());
    if mapping_attested {
        checks.push(Check::pass(
            "mapping_attestation",
            "task route, label semantics, and exact mapping reference have a named reviewer attestation",
        ));
    } else {
        checks.push(Check::fail(
            "mapping_attestation",
            "mapping_reviewed=true, non-empty label_mapping_reference/task_route_value, and reviewed_by are required",
        ));
    }

    let expected = match manifest.task.as_str() {
        "merge_correction_false_split" => Some((
            "false_split_correction",
            "false_split_correction_label",
            "same_neuron",
            "dual_segment",
        )),
        "split_correction_false_merge" => Some((
            "false_merge_identification",
            "false_merge_identification_label",
            "different_neuron",
            "single_segment",
        )),
        _ => None,
    };
    match expected {
        Some((route, label_column, true_semantics, input_mode)) => {
            for (name, actual, expected) in [
                ("task_route_value", manifest.task_route_value.as_str(), route),
                ("label_column", manifest.label_column.as_str(), label_column),
                ("label_true_semantics", manifest.label_true_semantics.as_str(), true_semantics),
                ("input_mode", manifest.input_mode.as_str(), input_mode),
            ] {
                if actual == expected {
                    checks.push(Check::pass(&format!("task_contract_{name}"), format!("{name} matches the selected task")));
                } else {
                    checks.push(Check::fail(
                        &format!("task_contract_{name}"),
                        format!("{name} must be {expected:?} for task {:?}; got {actual:?}", manifest.task),
                    ));
                }
            }
        }
        None => checks.push(Check::fail("task_contract", format!("unsupported task {:?}", manifest.task))),
    }

    if matches!(manifest.split.as_str(), "train" | "validation" | "test" | "all") {
        checks.push(Check::pass("declared_split", "split is a supported value"));
    } else {
        checks.push(Check::fail("declared_split", "split must be train, validation, test, or all"));
    }

    let invalid_inputs: Vec<_> = manifest.model_input_columns
        .iter()
        .filter(|column| !ALLOWED_INPUT_COLUMNS.contains(&column.as_str()))
        .cloned()
        .collect();
    let mut unique_input_columns = BTreeSet::new();
    let duplicate_input_columns: Vec<_> = manifest.model_input_columns
        .iter()
        .filter(|column| !unique_input_columns.insert((*column).clone()))
        .cloned()
        .collect();
    if !duplicate_input_columns.is_empty() {
        checks.push(Check::fail(
            "unique_model_input_columns",
            format!("duplicate model input column(s): {}", duplicate_input_columns.join(", ")),
        ));
    } else {
        checks.push(Check::pass("unique_model_input_columns", "model input columns are unique"));
    }
    if manifest.model_input_columns.is_empty() {
        checks.push(Check::fail("model_input_allowlist", "at least one model input column must be declared"));
    } else if invalid_inputs.is_empty() {
        checks.push(Check::pass("model_input_allowlist", "all model input columns are allowlisted; em_best/labels/IDs/routing/metadata are excluded"));
    } else {
        checks.push(Check::fail(
            "model_input_allowlist",
            format!("disallowed input column(s): {}", invalid_inputs.join(", ")),
        ));
    }

    let has_geometry = manifest.model_input_columns.iter().any(|v| v == "geometry");
    let has_geometry_single = manifest.model_input_columns.iter().any(|v| v == "geometry_single");
    if manifest.input_mode == "single_segment" && has_geometry {
        checks.push(Check::fail("geometry_mode", "single_segment tasks cannot use dual-segment geometry; use geometry_single"));
    } else if manifest.input_mode == "dual_segment" && has_geometry_single {
        checks.push(Check::fail("geometry_mode", "dual_segment tasks cannot use geometry_single; use geometry"));
    } else if !matches!(manifest.input_mode.as_str(), "single_segment" | "dual_segment") {
        checks.push(Check::fail("geometry_mode", "input_mode must be single_segment or dual_segment"));
    } else {
        checks.push(Check::pass("geometry_mode", "geometry input matches the declared segment mode"));
    }

    let has_geometry_render = has_geometry || has_geometry_single;
    let geometry_profile = manifest.geometry_preprocessing_profile.as_deref().unwrap_or("");
    let expected_geometry_profile = if has_geometry_single {
        "single_segment_zero_mask_channels"
    } else if has_geometry {
        "dual_segment_native"
    } else {
        "not_applicable"
    };
    if (expected_geometry_profile == "not_applicable"
        && (geometry_profile.is_empty() || geometry_profile == expected_geometry_profile))
        || geometry_profile == expected_geometry_profile
    {
        checks.push(Check::pass(
            "geometry_preprocessing_profile",
            format!("geometry preprocessing profile is {expected_geometry_profile}"),
        ));
    } else {
        checks.push(Check::fail(
            "geometry_preprocessing_profile",
            format!("expected {expected_geometry_profile:?} for selected geometry columns; got {geometry_profile:?}"),
        ));
    }
    if !has_geometry_render && manifest.geometry_preprocessing_profile.as_deref().is_some_and(|v| v != "not_applicable") {
        checks.push(Check::fail(
            "geometry_preprocessing_profile",
            "geometry preprocessing profile must be not_applicable when no geometry column is a model input",
        ));
    }

    let has_em = manifest.model_input_columns.iter().any(|v| EM_INPUT_COLUMNS.contains(&v.as_str()));
    let em_profile = manifest.em_preprocessing_profile.as_deref().unwrap_or("");
    let expected_profile = match (has_em, manifest.input_mode.as_str()) {
        (false, _) => "not_applicable",
        (true, "single_segment") => "single_segment_fuse_ab_zero_b",
        (true, "dual_segment") => "dual_segment_native",
        (true, _) => "invalid_input_mode",
    };
    if (expected_profile == "not_applicable" && (em_profile.is_empty() || em_profile == expected_profile))
        || em_profile == expected_profile
    {
        checks.push(Check::pass("em_preprocessing_profile", format!("EM preprocessing profile is {expected_profile}")));
    } else {
        checks.push(Check::fail(
            "em_preprocessing_profile",
            format!("expected {expected_profile:?} for this task/input mode; got {em_profile:?}"),
        ));
    }

    if manifest.input_files.is_empty() {
        checks.push(Check::fail("input_files_nonempty", "at least one local input file must be declared"));
    } else {
        checks.push(Check::pass("input_files_nonempty", format!("{} input file(s) declared", manifest.input_files.len())));
    }

    let mut paths = BTreeSet::new();
    let mut invalid_paths = Vec::new();
    let mut duplicate_paths = Vec::new();
    let mut split_mismatches = Vec::new();
    for file in &manifest.input_files {
        if !safe_relative_path(&file.path) {
            invalid_paths.push(file.path.clone());
        }
        if !paths.insert(file.path.clone()) {
            duplicate_paths.push(file.path.clone());
        }
        if !is_lower_hex(&file.sha256, 64) {
            checks.push(Check::fail(
                "file_sha256_format",
                format!("{} must declare a lowercase 64-character SHA-256 digest", file.path),
            ));
        }
        if file.bytes == 0 {
            checks.push(Check::fail("file_bytes_positive", format!("{} has declared byte size 0", file.path)));
        }
        if manifest.split != "all" && file.split != manifest.split {
            split_mismatches.push(format!("{}: declared {}", file.path, file.split));
        }
        if !matches!(file.split.as_str(), "train" | "validation" | "test") {
            checks.push(Check::fail("file_split_value", format!("{} split must be train, validation, or test", file.path)));
        }
    }

    if invalid_paths.is_empty() {
        checks.push(Check::pass("relative_paths", "all input paths are relative and contain no parent traversal"));
    } else {
        checks.push(Check::fail("relative_paths", format!("unsafe input path(s): {}", invalid_paths.join(", "))));
    }
    if duplicate_paths.is_empty() {
        checks.push(Check::pass("unique_paths", "no duplicate input paths are declared"));
    } else {
        checks.push(Check::fail("unique_paths", format!("duplicate path(s): {}", duplicate_paths.join(", "))));
    }
    if split_mismatches.is_empty() {
        checks.push(Check::pass("input_split_consistency", "input file splits match the manifest split"));
    } else {
        checks.push(Check::fail("input_split_consistency", format!("split mismatch: {}", split_mismatches.join("; "))));
    }

    checks.sort_by(|a, b| a.code.cmp(&b.code).then_with(|| a.detail.cmp(&b.detail)));
    checks
}

fn audit_files(manifest: &Manifest, root: &Path) -> Vec<FileAudit> {
    let canonical_root = match fs::canonicalize(root) {
        Ok(path) if path.is_dir() => path,
        _ => {
            return manifest.input_files.iter().map(|file| FileAudit {
                path: file.path.clone(),
                declared_split: file.split.clone(),
                declared_bytes: file.bytes,
                declared_sha256: file.sha256.clone(),
                actual_bytes: None,
                actual_sha256: None,
                status: "invalid_root".to_string(),
            }).collect();
        }
    };

    manifest.input_files.iter().map(|file| {
        if !safe_relative_path(&file.path) {
            return FileAudit {
                path: file.path.clone(),
                declared_split: file.split.clone(),
                declared_bytes: file.bytes,
                declared_sha256: file.sha256.clone(),
                actual_bytes: None,
                actual_sha256: None,
                status: "unsafe_path".to_string(),
            };
        }

        let joined = canonical_root.join(&file.path);
        let canonical = match fs::canonicalize(&joined) {
            Ok(path) if path.starts_with(&canonical_root) && path.is_file() => path,
            _ => {
                return FileAudit {
                    path: file.path.clone(),
                    declared_split: file.split.clone(),
                    declared_bytes: file.bytes,
                    declared_sha256: file.sha256.clone(),
                    actual_bytes: None,
                    actual_sha256: None,
                    status: "missing_or_outside_root".to_string(),
                };
            }
        };

        match File::open(canonical).map(BufReader::new).and_then(sha256_reader) {
            Ok((bytes, digest)) => {
                let bytes_match = bytes == file.bytes;
                let hash_match = digest == file.sha256;
                let status = match (bytes_match, hash_match) {
                    (true, true) => "ok",
                    (false, true) => "byte_count_mismatch",
                    (true, false) => "sha256_mismatch",
                    (false, false) => "byte_count_and_sha256_mismatch",
                };
                FileAudit {
                    path: file.path.clone(),
                    declared_split: file.split.clone(),
                    declared_bytes: file.bytes,
                    declared_sha256: file.sha256.clone(),
                    actual_bytes: Some(bytes),
                    actual_sha256: Some(digest),
                    status: status.to_string(),
                }
            }
            Err(_) => FileAudit {
                path: file.path.clone(),
                declared_split: file.split.clone(),
                declared_bytes: file.bytes,
                declared_sha256: file.sha256.clone(),
                actual_bytes: None,
                actual_sha256: None,
                status: "read_error".to_string(),
            },
        }
    }).collect()
}

fn build_report(manifest: &Manifest, manifest_bytes: &[u8], root: &Path) -> AuditReport {
    let mut checks = semantic_checks(manifest);
    let files = audit_files(manifest, root);
    for file in &files {
        if file.status == "ok" {
            checks.push(Check::pass("input_file_integrity", format!("{} matches its declared byte count and SHA-256", file.path)));
        } else {
            checks.push(Check::fail("input_file_integrity", format!("{} integrity status: {}", file.path, file.status)));
        }
    }

    let mut seen_digests = BTreeMap::<String, String>::new();
    for file in &files {
        if file.status == "ok" {
            if let Some(first) = seen_digests.insert(file.declared_sha256.clone(), file.path.clone()) {
                checks.push(Check::fail(
                    "duplicate_file_digest",
                    format!("{} and {} have identical bytes; inspect for duplicate samples/shard aliasing", first, file.path),
                ));
            }
        }
    }
    checks.sort_by(|a, b| a.code.cmp(&b.code).then_with(|| a.detail.cmp(&b.detail)));

    let passed_checks = checks.iter().filter(|check| check.passed).count() as u64;
    let failed_checks = checks.len() as u64 - passed_checks;
    let ok_files = files.iter().filter(|file| file.status == "ok").count() as u64;
    let failed_files = files.len() as u64 - ok_files;
    let mut summary = BTreeMap::new();
    summary.insert("checks_total".to_string(), checks.len() as u64);
    summary.insert("checks_passed".to_string(), passed_checks);
    summary.insert("checks_failed".to_string(), failed_checks);
    summary.insert("files_total".to_string(), files.len() as u64);
    summary.insert("files_ok".to_string(), ok_files);
    summary.insert("files_failed".to_string(), failed_files);

    AuditReport {
        report_schema_version: 1,
        tool: "connectomics-intake-audit".to_string(),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        manifest_sha256: sha256_bytes(manifest_bytes),
        dataset_id: manifest.dataset_id.clone(),
        dataset_revision: manifest.dataset_revision.clone(),
        task: manifest.task.clone(),
        declared_split: manifest.split.clone(),
        status: if failed_checks == 0 { "PASS" } else { "FAIL" }.to_string(),
        checks,
        files,
        summary,
        scope_note: "PASS verifies manifest contracts and local bytes only; it does not verify Parquet contents, biological labels, sample independence, model quality, or scientific truth.",
    }
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    let manifest_bytes = fs::read(&args.manifest)
        .map_err(|err| format!("failed to read manifest {}: {err}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| format!("invalid manifest JSON/schema: {err}"))?;

    let report = build_report(&manifest, &manifest_bytes, &args.root);
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize audit report: {err}"))?
        + "\n";

    if let Some(output) = args.output {
        fs::write(&output, rendered.as_bytes())
            .map_err(|err| format!("failed to write report {}: {err}", output.display()))?;
    } else {
        print!("{rendered}");
    }

    Ok(if report.status == "PASS" { 0 } else { 1 })
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(message) => {
            eprintln!("connectomics-intake-audit: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_manifest() -> Manifest {
        Manifest {
            schema_version: 1,
            dataset_id: DATASET_ID.into(),
            dataset_revision: DATASET_REVISION.into(),
            dataset_revision_commit_sha: "a".repeat(40),
            upstream_repo: UPSTREAM_REPO.into(),
            upstream_commit: "b".repeat(40),
            task: "split_correction_false_merge".into(),
            task_route_value: "false_merge_identification".into(),
            label_column: "false_merge_identification_label".into(),
            label_true_semantics: "different_neuron".into(),
            label_mapping_reference: "https://github.com/timfarkas/ConnectomeBench2/blob/5679b39aa2641d8d0afeed88b441e696f87a66df/src/neurips_training/data.py#L201-L227".into(),
            split: "train".into(),
            input_mode: "single_segment".into(),
            model_input_columns: vec!["geometry_single".into(), "em_xy".into()],
            input_files: vec![InputFile {
                path: "train/shard-000.parquet".into(),
                sha256: "c".repeat(64),
                bytes: 123,
                split: "train".into(),
            }],
            geometry_preprocessing_profile: Some("single_segment_zero_mask_channels".into()),
            em_preprocessing_profile: Some("single_segment_fuse_ab_zero_b".into()),
            reviewed_by: Some("connectomics-reviewer".into()),
            notes: None,
            mapping_reviewed: true,
        }
    }

    #[test]
    fn valid_task_contract_passes() {
        let checks = semantic_checks(&base_manifest());
        assert!(checks.iter().all(|check| check.passed), "{checks:#?}");
    }

    #[test]
    fn wrong_task_label_mapping_fails_closed() {
        let mut manifest = base_manifest();
        manifest.label_column = "false_split_correction_label".into();
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "task_contract_label_column"));
    }

    #[test]
    fn dual_segment_task_contract_passes() {
        let mut manifest = base_manifest();
        manifest.task = "merge_correction_false_split".into();
        manifest.task_route_value = "false_split_correction".into();
        manifest.label_column = "false_split_correction_label".into();
        manifest.label_true_semantics = "same_neuron".into();
        manifest.input_mode = "dual_segment".into();
        manifest.model_input_columns = vec!["geometry".into(), "em_xy".into()];
        manifest.geometry_preprocessing_profile = Some("dual_segment_native".into());
        manifest.em_preprocessing_profile = Some("dual_segment_native".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().all(|check| check.passed), "{checks:#?}");
    }

    #[test]
    fn duplicate_model_input_columns_are_rejected() {
        let mut manifest = base_manifest();
        manifest.model_input_columns.push("geometry_single".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "unique_model_input_columns"));
    }

    #[test]
    fn single_segment_dual_geometry_is_rejected() {
        let mut manifest = base_manifest();
        manifest.model_input_columns.push("geometry".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "geometry_mode"));
    }

    #[test]
    fn known_label_leakage_field_is_rejected() {
        let mut manifest = base_manifest();
        manifest.model_input_columns.push("em_best".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "model_input_allowlist"));
    }

    #[test]
    fn single_segment_geometry_requires_zeroed_mask_channels() {
        let mut manifest = base_manifest();
        manifest.geometry_preprocessing_profile = Some("dual_segment_native".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "geometry_preprocessing_profile"));
    }

    #[test]
    fn geometry_profile_is_not_applicable_when_using_only_em() {
        let mut manifest = base_manifest();
        manifest.model_input_columns = vec!["em_xy".into()];
        manifest.geometry_preprocessing_profile = None;
        manifest.em_preprocessing_profile = Some("single_segment_fuse_ab_zero_b".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().all(|check| check.passed), "{checks:#?}");
    }

    #[test]
    fn em_masks_must_be_preprocessed_for_single_segment_tasks() {
        let mut manifest = base_manifest();
        manifest.em_preprocessing_profile = Some("dual_segment_native".into());
        let checks = semantic_checks(&manifest);
        assert!(checks.iter().any(|check| !check.passed && check.code == "em_preprocessing_profile"));
    }

    #[test]
    fn parent_traversal_paths_are_rejected() {
        assert!(!safe_relative_path("../test.parquet"));
        assert!(!safe_relative_path("train/../../test.parquet"));
        assert!(!safe_relative_path("/tmp/test.parquet"));
        assert!(safe_relative_path("train/shard-000.parquet"));
    }

    fn manifest_with_file_bytes(bytes: &[u8], suffix: &str) -> (PathBuf, Manifest) {
        let root = std::env::temp_dir().join(format!(
            "connectomics-intake-audit-{}-{suffix}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("train")).expect("create test root");
        fs::write(root.join("train/shard-000.parquet"), bytes).expect("write test shard");

        let mut manifest = base_manifest();
        manifest.input_files[0].bytes = bytes.len() as u64;
        manifest.input_files[0].sha256 = sha256_bytes(bytes);
        (root, manifest)
    }

    #[test]
    fn sha256_matches_known_digest() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn matching_local_file_bytes_produce_pass() {
        let (root, manifest) = manifest_with_file_bytes(b"tiny parquet fixture", "valid");
        let report = build_report(&manifest, br#"{"fixture":true}"#, &root);
        assert_eq!(report.status, "PASS", "{:#?}", report.checks);
        assert_eq!(report.files.len(), 1);
        assert_eq!(report.files[0].status, "ok");
        fs::remove_dir_all(root).expect("cleanup test root");
    }

    #[test]
    fn changed_local_file_bytes_fail_closed() {
        let (root, mut manifest) = manifest_with_file_bytes(b"actual bytes", "tampered");
        manifest.input_files[0].sha256 = sha256_bytes(b"different bytes");
        let report = build_report(&manifest, br#"{"fixture":true}"#, &root);
        assert_eq!(report.status, "FAIL");
        assert_eq!(report.files[0].status, "sha256_mismatch");
        fs::remove_dir_all(root).expect("cleanup test root");
    }
