//! HARDSEC-009 source-family split and leakage auditor.
//!
//! This tool validates a preregistered hardware-Trojan benchmark manifest before
//! training or threshold selection. It audits lineage and split integrity only;
//! it does not inspect netlist semantics, validate labels, or measure detection.
//!
//! Example:
//!   cargo run --example hardsec_family_split_audit --features hardsec-family-audit -- \
//!     --manifest /path/to/family-split-manifest.json \
//!     --receipt /path/outside/manifest/split-audit.json
//!
//! Confirmatory protocols are leave-one-base-family-out and
//! leave-one-insertion-source-out. Random splits are diagnostics only.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const SCHEMA: &str = "hardsec-family-split-v1";
const CLAIM_SCOPE: &str = "manifest_lineage_and_split_group_integrity_only";
const SPLITS: [&str; 3] = ["train", "validation", "test"];
const PROTOCOLS: [&str; 3] = [
    "leave_one_base_family_out",
    "leave_one_insertion_source_out",
    "diagnostic_random",
];
const LABELS: [&str; 2] = ["clean", "trojan"];

#[derive(Debug, Clone)]
struct Args {
    manifest: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone)]
struct Record {
    sample_id: String,
    design_id: String,
    host_family_id: String,
    insertion_source_id: String,
    source_uri: String,
    source_revision: String,
    source_sha256: String,
    netlist_sha256: String,
    normalized_structure_sha256: String,
    label: String,
    split: String,
}

#[derive(Debug, Clone, Default)]
struct SplitCounts {
    samples: usize,
    clean: usize,
    trojan: usize,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut manifest = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args.next().ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{field} must be a JSON object"))
}

fn required_string(value: &Value, field: &str) -> Result<String, String> {
    let text = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if text.trim().is_empty() || text.trim() != text {
        return Err(format!("{field} must be non-empty with no surrounding whitespace"));
    }
    Ok(text.to_owned())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn normalize_digest(value: &Value, field: &str) -> Result<String, String> {
    let digest = required_string(value, field)?.to_ascii_lowercase();
    if !valid_sha256(&digest) {
        return Err(format!("{field} must be a 64-character SHA-256 digest"));
    }
    Ok(digest)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn check_digest_binding(
    bindings: &mut BTreeMap<String, (String, String, usize)>,
    kind: &str,
    digest: &str,
    split: &str,
    label: &str,
) -> Result<usize, String> {
    match bindings.get_mut(digest) {
        Some((existing_split, existing_label, aliases)) => {
            if existing_split != split {
                return Err(format!(
                    "{kind} digest {digest} leaks across splits: {existing_split} and {split}"
                ));
            }
            if existing_label != label {
                return Err(format!(
                    "{kind} digest {digest} has conflicting labels {existing_label} and {label}"
                ));
            }
            *aliases += 1;
            Ok(1)
        }
        None => {
            bindings.insert(digest.to_owned(), (split.to_owned(), label.to_owned(), 1));
            Ok(0)
        }
    }
}

fn check_group_split(
    assignments: &mut BTreeMap<String, String>,
    kind: &str,
    group: &str,
    split: &str,
) -> Result<(), String> {
    match assignments.get(group) {
        Some(existing) if existing != split => Err(format!(
            "{kind} {group} crosses splits: {existing} and {split}"
        )),
        Some(_) => Ok(()),
        None => {
            assignments.insert(group.to_owned(), split.to_owned());
            Ok(())
        }
    }
}

fn parse_record(value: &Value, index: usize) -> Result<Record, String> {
    let prefix = format!("records[{index}]");
    let record = Record {
        sample_id: required_string(value, "sample_id")
            .map_err(|error| format!("{prefix}: {error}"))?,
        design_id: required_string(value, "design_id")
            .map_err(|error| format!("{prefix}: {error}"))?,
        host_family_id: required_string(value, "host_family_id")
            .map_err(|error| format!("{prefix}: {error}"))?,
        insertion_source_id: required_string(value, "insertion_source_id")
            .map_err(|error| format!("{prefix}: {error}"))?,
        source_uri: required_string(value, "source_uri")
            .map_err(|error| format!("{prefix}: {error}"))?,
        source_revision: required_string(value, "source_revision")
            .map_err(|error| format!("{prefix}: {error}"))?,
        source_sha256: normalize_digest(value, "source_sha256")
            .map_err(|error| format!("{prefix}: {error}"))?,
        netlist_sha256: normalize_digest(value, "netlist_sha256")
            .map_err(|error| format!("{prefix}: {error}"))?,
        normalized_structure_sha256: normalize_digest(value, "normalized_structure_sha256")
            .map_err(|error| format!("{prefix}: {error}"))?,
        label: required_string(value, "label")
            .map_err(|error| format!("{prefix}: {error}"))?,
        split: required_string(value, "split")
            .map_err(|error| format!("{prefix}: {error}"))?,
    };

    if !LABELS.contains(&record.label.as_str()) {
        return Err(format!(
            "{prefix}: label {} is not one of clean/trojan; unknown labels fail closed",
            record.label
        ));
    }
    if !SPLITS.contains(&record.split.as_str()) {
        return Err(format!("{prefix}: unsupported split {}", record.split));
    }
    if record.label == "clean" && record.insertion_source_id != "none" {
        return Err(format!(
            "{prefix}: clean samples must set insertion_source_id to the explicit sentinel 'none'"
        ));
    }
    if record.label == "trojan" && record.insertion_source_id == "none" {
        return Err(format!(
            "{prefix}: trojan samples must identify an insertion_source_id"
        ));
    }

    Ok(record)
}

fn validate_manifest(value: &Value, manifest_sha256: String) -> Result<Value, String> {
    if value.get("schema_version").and_then(Value::as_str) != Some(SCHEMA) {
        return Err(format!("schema_version must be {SCHEMA}"));
    }

    let manifest_id = required_string(value, "manifest_id")?;
    let dataset_id = required_string(value, "dataset_id")?;
    let protocol = required_string(value, "split_protocol")?;
    if !PROTOCOLS.contains(&protocol.as_str()) {
        return Err(format!("unsupported split_protocol: {protocol}"));
    }

    let normalization = object(value, "normalization")?;
    let normalization = Value::Object(normalization.clone());
    let normalizer_id = required_string(&normalization, "normalizer_id")?;
    let normalizer_version = required_string(&normalization, "normalizer_version")?;
    let cell_library_id = required_string(&normalization, "cell_library_id")?;
    let synthesis_toolchain_id = required_string(&normalization, "synthesis_toolchain_id")?;

    let records_value = value
        .get("records")
        .and_then(Value::as_array)
        .ok_or("records must be a JSON array")?;
    if records_value.is_empty() {
        return Err("records must not be empty".to_owned());
    }

    let mut records = Vec::with_capacity(records_value.len());
    let mut sample_ids = BTreeSet::new();
    let mut design_ids = BTreeSet::new();
    for (index, item) in records_value.iter().enumerate() {
        let record = parse_record(item, index)?;
        if !sample_ids.insert(record.sample_id.clone()) {
            return Err(format!("duplicate sample_id {}", record.sample_id));
        }
        if !design_ids.insert(record.design_id.clone()) {
            return Err(format!("duplicate design_id {}", record.design_id));
        }
        records.push(record);
    }

    let mut split_counts: BTreeMap<String, SplitCounts> = SPLITS
        .iter()
        .map(|split| (split.to_string(), SplitCounts::default()))
        .collect();
    let mut host_assignments = BTreeMap::new();
    let mut insertion_assignments = BTreeMap::new();
    let mut source_digests = BTreeMap::new();
    let mut netlist_digests = BTreeMap::new();
    let mut structure_digests = BTreeMap::new();
    let mut duplicate_aliases = 0usize;

    for record in &records {
        let counts = split_counts
            .get_mut(&record.split)
            .ok_or_else(|| format!("internal error: split {} missing", record.split))?;
        counts.samples += 1;
        match record.label.as_str() {
            "clean" => counts.clean += 1,
            "trojan" => counts.trojan += 1,
            _ => unreachable!("labels were checked in parse_record"),
        }

        duplicate_aliases += check_digest_binding(
            &mut source_digests,
            "source artifact",
            &record.source_sha256,
            &record.split,
            &record.label,
        )?;
        duplicate_aliases += check_digest_binding(
            &mut netlist_digests,
            "netlist",
            &record.netlist_sha256,
            &record.split,
            &record.label,
        )?;
        duplicate_aliases += check_digest_binding(
            &mut structure_digests,
            "normalized structure",
            &record.normalized_structure_sha256,
            &record.split,
            &record.label,
        )?;

        match protocol.as_str() {
            "leave_one_base_family_out" => check_group_split(
                &mut host_assignments,
                "host_family_id",
                &record.host_family_id,
                &record.split,
            )?,
            "leave_one_insertion_source_out" if record.label == "trojan" => check_group_split(
                &mut insertion_assignments,
                "insertion_source_id",
                &record.insertion_source_id,
                &record.split,
            )?,
            _ => {}
        }
    }

    for split in SPLITS {
        let counts = split_counts.get(split).expect("split initialized");
        if counts.samples == 0 {
            return Err(format!("split {split} is empty"));
        }
        if counts.clean == 0 || counts.trojan == 0 {
            return Err(format!(
                "split {split} must contain both clean and trojan samples"
            ));
        }
    }

    if protocol == "leave_one_base_family_out" && host_assignments.len() < 3 {
        return Err("leave_one_base_family_out requires at least three distinct host families".to_owned());
    }
    if protocol == "leave_one_insertion_source_out" && insertion_assignments.len() < 3 {
        return Err(
            "leave_one_insertion_source_out requires at least three distinct Trojan insertion sources"
                .to_owned(),
        );
    }

    records.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

    let split_summary = SPLITS
        .iter()
        .map(|split| {
            let counts = split_counts.get(*split).expect("split initialized");
            json!({
                "split": split,
                "samples": counts.samples,
                "clean": counts.clean,
                "trojan": counts.trojan,
            })
        })
        .collect::<Vec<_>>();

    let host_groups = host_assignments
        .iter()
        .map(|(group, split)| json!({"host_family_id": group, "split": split}))
        .collect::<Vec<_>>();
    let insertion_groups = insertion_assignments
        .iter()
        .map(|(group, split)| json!({"insertion_source_id": group, "split": split}))
        .collect::<Vec<_>>();

    let confirmatory = protocol != "diagnostic_random";
    Ok(json!({
        "schema_version": "hardsec-family-split-audit-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "manifest_sha256": manifest_sha256,
        "split_protocol": protocol,
        "confirmatory_generalization_protocol": confirmatory,
        "normalization": {
            "normalizer_id": normalizer_id,
            "normalizer_version": normalizer_version,
            "cell_library_id": cell_library_id,
            "synthesis_toolchain_id": synthesis_toolchain_id,
        },
        "sample_count": records.len(),
        "unique_sample_ids": sample_ids.len(),
        "unique_design_ids": design_ids.len(),
        "unique_source_sha256": source_digests.len(),
        "unique_netlist_sha256": netlist_digests.len(),
        "unique_normalized_structure_sha256": structure_digests.len(),
        "duplicate_same_split_content_aliases": duplicate_aliases,
        "split_counts": split_summary,
        "host_family_assignments": host_groups,
        "insertion_source_assignments": insertion_groups,
        "notes": [
            "This checks declared lineage and split integrity only; it does not validate labels or netlist semantics.",
            "The normalized_structure_sha256 is only comparable under the exact normalizer/version/library/toolchain identity recorded in the manifest.",
            "diagnostic_random is not a confirmatory generalization protocol.",
            "No model performance or hardware security verdict is produced."
        ]
    }))
}

fn receipt_target(manifest: &Path, receipt: &Path) -> Result<PathBuf, String> {
    let manifest = fs::canonicalize(manifest)
        .map_err(|error| format!("canonicalize manifest {}: {error}", manifest.display()))?;
    let manifest_parent = manifest
        .parent()
        .ok_or("manifest has no parent directory")?
        .to_path_buf();
    let parent = receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = receipt.file_name().ok_or("receipt path must include filename")?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = parent.join(filename);
    if target.starts_with(&manifest_parent) {
        return Err("receipt must be outside the manifest directory".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.manifest)
        .map_err(|error| format!("stat manifest {}: {error}", args.manifest.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("manifest must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(&args.manifest)
        .map_err(|error| format!("read manifest {}: {error}", args.manifest.display()))?;
    let manifest_sha256 = sha256_bytes(&bytes);
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse manifest JSON: {error}"))?;
    let receipt = validate_manifest(&value, manifest_sha256)?;
    let target = receipt_target(&args.manifest, &args.receipt)?;
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize receipt: {error}"))?;

    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target)
        .map_err(|error| format!("create new receipt {}: {error}", target.display()))?;
    let result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = result {
        let _ = fs::remove_file(&target);
        return Err(format!("write receipt {}: {error}", target.display()));
    }

    println!("HARDSEC-009 family split audit: PASS");
    println!("manifest SHA-256: {manifest_sha256}");
    println!("receipt: {}", target.display());
    println!("claim scope: {CLAIM_SCOPE}");
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC_FAMILY_SPLIT_AUDIT_ERROR: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!(
                "usage: hardsec_family_split_audit --manifest FILE --receipt NEW_FILE"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn hash(byte: u8) -> String {
        format!("{:02x}", byte).repeat(32)
    }

    fn record(
        sample: &str,
        host: &str,
        insertion: &str,
        split: &str,
        label: &str,
        digest_byte: u8,
    ) -> Value {
        json!({
            "sample_id": sample,
            "design_id": format!("design-{sample}"),
            "host_family_id": host,
            "insertion_source_id": insertion,
            "source_uri": format!("fixture://sources/{sample}.v"),
            "source_revision": "0123456789abcdef0123456789abcdef01234567",
            "source_sha256": hash(digest_byte),
            "netlist_sha256": hash(digest_byte.wrapping_add(1)),
            "normalized_structure_sha256": hash(digest_byte.wrapping_add(2)),
            "label": label,
            "split": split
        })
    }

    fn base_manifest(protocol: &str) -> Value {
        json!({
            "schema_version": SCHEMA,
            "manifest_id": "fixture-manifest",
            "dataset_id": "fixture-dataset",
            "split_protocol": protocol,
            "normalization": {
                "normalizer_id": "fixture-verilog-normalizer",
                "normalizer_version": "1.0.0",
                "cell_library_id": "fixture-library-1",
                "synthesis_toolchain_id": "yosys-fixture-1"
            },
            "records": [
                record("tr-a", "host-family-a", "insert-a", "train", "trojan", 1),
                record("cl-a", "host-family-a", "none", "train", "clean", 10),
                record("tr-b", "host-family-b", "insert-b", "validation", "trojan", 20),
                record("cl-b", "host-family-b", "none", "validation", "clean", 30),
                record("tr-c", "host-family-c", "insert-c", "test", "trojan", 40),
                record("cl-c", "host-family-c", "none", "test", "clean", 50)
            ]
        })
    }

    struct Fixture {
        base: PathBuf,
        manifest: PathBuf,
        output: PathBuf,
    }

    impl Fixture {
        fn new(value: &Value) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let base = env::temp_dir().join(format!(
                "hardsec-family-audit-{}-{id}",
                std::process::id()
            ));
            let input = base.join("input");
            let output = base.join("output");
            fs::create_dir_all(&input).unwrap();
            fs::create_dir_all(&output).unwrap();
            let manifest = input.join("manifest.json");
            fs::write(&manifest, serde_json::to_vec_pretty(value).unwrap()).unwrap();
            Self {
                base,
                manifest,
                output: output.join("receipt.json"),
            }
        }

        fn audit(&self, value: &Value) -> Result<Value, String> {
            validate_manifest(value, "0".repeat(64))
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    #[test]
    fn accepts_leave_one_base_family_out_manifest() {
        let fixture = Fixture::new(&base_manifest("leave_one_base_family_out"));
        let receipt = fixture.audit(&base_manifest("leave_one_base_family_out")).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["confirmatory_generalization_protocol"], true);
        assert_eq!(receipt["host_family_assignments"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn rejects_host_family_crossing_splits() {
        let fixture = Fixture::new(&base_manifest("leave_one_base_family_out"));
        let mut manifest = base_manifest("leave_one_base_family_out");
        manifest["records"][2]["host_family_id"] = json!("host-family-a");
        assert!(fixture.audit(&manifest).unwrap_err().contains("host_family_id"));
    }

    #[test]
    fn accepts_leave_one_insertion_source_out_with_shared_host_family() {
        let fixture = Fixture::new(&base_manifest("leave_one_insertion_source_out"));
        let mut manifest = base_manifest("leave_one_insertion_source_out");
        for item in manifest["records"].as_array_mut().unwrap() {
            item["host_family_id"] = json!("shared-host-family");
        }
        let receipt = fixture.audit(&manifest).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["insertion_source_assignments"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn rejects_insertion_source_crossing_splits() {
        let fixture = Fixture::new(&base_manifest("leave_one_insertion_source_out"));
        let mut manifest = base_manifest("leave_one_insertion_source_out");
        manifest["records"][2]["insertion_source_id"] = json!("insert-a");
        assert!(fixture.audit(&manifest).unwrap_err().contains("insertion_source_id"));
    }

    #[test]
    fn rejects_exact_netlist_hash_crossing_splits() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][2]["netlist_sha256"] = manifest["records"][0]["netlist_sha256"].clone();
        assert!(fixture.audit(&manifest).unwrap_err().contains("netlist digest"));
    }

    #[test]
    fn rejects_normalized_structure_crossing_splits() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][2]["normalized_structure_sha256"] =
            manifest["records"][0]["normalized_structure_sha256"].clone();
        assert!(fixture.audit(&manifest).unwrap_err().contains("normalized structure digest"));
    }

    #[test]
    fn rejects_source_artifact_hash_crossing_splits() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][2]["source_sha256"] = manifest["records"][0]["source_sha256"].clone();
        assert!(fixture.audit(&manifest).unwrap_err().contains("source artifact digest"));
    }

    #[test]
    fn rejects_unknown_labels() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][0]["label"] = json!("unknown");
        assert!(fixture.audit(&manifest).unwrap_err().contains("unknown labels fail closed"));
    }

    #[test]
    fn rejects_bad_sha256() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][0]["netlist_sha256"] = json!("1234");
        assert!(fixture.audit(&manifest).unwrap_err().contains("64-character SHA-256"));
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][1]["sample_id"] = json!("tr-a");
        assert!(fixture.audit(&manifest).unwrap_err().contains("duplicate sample_id"));
    }

    #[test]
    fn rejects_duplicate_design_ids() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][1]["design_id"] = manifest["records"][0]["design_id"].clone();
        assert!(fixture.audit(&manifest).unwrap_err().contains("duplicate design_id"));
    }

    #[test]
    fn rejects_unknown_split_protocol() {
        let fixture = Fixture::new(&base_manifest("unknown_protocol"));
        assert!(fixture.audit(&base_manifest("unknown_protocol")).unwrap_err().contains("unsupported split_protocol"));
    }

    #[test]
    fn rejects_missing_normalizer_version() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["normalization"].as_object_mut().unwrap().remove("normalizer_version");
        assert!(fixture.audit(&manifest).unwrap_err().contains("normalizer_version"));
    }

    #[test]
    fn rejects_missing_split() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"].as_array_mut().unwrap().retain(|record| {
            record["split"].as_str() != Some("test")
        });
        assert!(fixture.audit(&manifest).unwrap_err().contains("split test is empty"));
    }

    #[test]
    fn rejects_single_class_validation_or_test_split() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"].as_array_mut().unwrap().retain(|record| {
            !(record["split"].as_str() == Some("test") && record["label"].as_str() == Some("clean"))
        });
        assert!(fixture.audit(&manifest).unwrap_err().contains("both clean and trojan"));
    }

    #[test]
    fn diagnostic_random_is_not_a_confirmatory_protocol() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let receipt = fixture.audit(&base_manifest("diagnostic_random")).unwrap();
        assert_eq!(receipt["confirmatory_generalization_protocol"], false);
    }

    #[test]
    fn rejects_conflicting_labels_for_same_normalized_structure() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][1]["normalized_structure_sha256"] =
            manifest["records"][0]["normalized_structure_sha256"].clone();
        assert!(fixture.audit(&manifest).unwrap_err().contains("conflicting labels"));
    }

    #[test]
    fn counts_same_split_content_aliases() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][1]["netlist_sha256"] = manifest["records"][0]["netlist_sha256"].clone();
        manifest["records"][1]["label"] = json!("trojan");
        manifest["records"][1]["normalized_structure_sha256"] =
            manifest["records"][0]["normalized_structure_sha256"].clone();
        manifest["records"][1]["source_sha256"] = manifest["records"][0]["source_sha256"].clone();
        let receipt = fixture.audit(&manifest).unwrap();
        assert!(receipt["duplicate_same_split_content_aliases"].as_u64().unwrap() >= 3);
    }

    #[test]
    fn rejects_clean_sample_with_non_none_insertion_source() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][1]["insertion_source_id"] = json!("some-insertion");
        assert!(fixture.audit(&manifest).unwrap_err().contains("clean samples must set"));
    }

    #[test]
    fn rejects_trojan_sample_without_insertion_source() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let mut manifest = base_manifest("diagnostic_random");
        manifest["records"][0]["insertion_source_id"] = json!("none");
        assert!(fixture.audit(&manifest).unwrap_err().contains("must identify an insertion_source_id"));
    }

    #[test]
    fn receipt_target_must_be_outside_manifest_directory() {
        let fixture = Fixture::new(&base_manifest("diagnostic_random"));
        let inside = fixture.manifest.parent().unwrap().join("receipt.json");
        assert!(receipt_target(&fixture.manifest, &inside).unwrap_err().contains("outside the manifest directory"));
    }
}
