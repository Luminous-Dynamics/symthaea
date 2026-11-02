//! Exact replay of ConnectomeBench2's upstream per-species/per-bank split manifests.
//!
//! The existing dataset audit checks for cross-partition operation/cube overlap.
//! This tool adds a different claim: every observed sidecar row must be assigned
//! to the exact train/val/test partition recorded in the upstream split JSON files.
//! It also rejects duplicate bank keys that the upstream dict loader would overwrite.

use clap::Parser;
use md5::Md5;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

const DATASET_ID: &str = "jeffbbrown2/ConnectomeBench2";
const DATASET_REVISION: &str = "v2";
const UPSTREAM_COMMIT: &str = "5679b39aa2641d8d0afeed88b441e696f87a66df";
const EXPECTED_EDGE_NM: f64 = 50_000.0;
const SPLITS: [&str; 3] = ["train", "val", "test"];
const BANKS: [&str; 3] = [
    "operation_bank",
    "operation_bank_junction_controls",
    "operation_bank_synapse_controls",
];

#[derive(Debug, Parser)]
#[command(
    name = "connectome-split-replay-audit",
    version,
    about = "Replay exact ConnectomeBench2 V2 source split manifests against published metadata rows"
)]
struct Args {
    /// JSONL exported from metadata/{train,val,test}.parquet.
    #[arg(long)]
    sidecar_jsonl: PathBuf,

    /// Root containing <species>/{manifest.json,train.json,val.json,test.json}.
    #[arg(long)]
    split_root: PathBuf,

    /// Exact Hugging Face commit SHA resolved from revision=v2.
    #[arg(long)]
    dataset_revision_commit_sha: String,

    /// Upstream ConnectomeBench2 code commit. Must match the audited generator revision.
    #[arg(long, default_value = UPSTREAM_COMMIT)]
    upstream_commit: String,

    /// Write deterministic JSON report here (stdout when omitted).
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SidecarRow {
    combined_sample_hash: String,
    source_archive_sample_hash: String,
    source_archive: String,
    split: String,
    species: String,
    metadata: String,
}

#[derive(Debug)]
struct ParsedRow {
    row: SidecarRow,
    metadata: Value,
    bank: String,
    split: String,
    source_line: usize,
}

#[derive(Debug, serde::Serialize)]
struct Finding {
    code: String,
    species: Option<String>,
    bank: Option<String>,
    key: Option<String>,
    sample_id: Option<String>,
    detail: String,
}

impl Finding {
    fn global(code: &str, detail: impl Into<String>) -> Self {
        Self { code: code.into(), species: None, bank: None, key: None, sample_id: None, detail: detail.into() }
    }
    fn row(code: &str, row: &ParsedRow, key: Option<&str>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            species: Some(row.row.species.clone()),
            bank: Some(row.bank.clone()),
            key: key.map(str::to_owned),
            sample_id: Some(row.row.combined_sample_hash.clone()),
            detail: format!("line {}: {}", row.source_line, detail.into()),
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct FileDigest {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, serde::Serialize)]
struct SpeciesReplay {
    species: String,
    split_manifest_sha256: String,
    split_files: BTreeMap<String, FileDigest>,
    source_key_counts: BTreeMap<String, u64>,
    sidecar_row_counts: BTreeMap<String, u64>,
    matched_sidecar_rows: u64,
    mismatched_sidecar_rows: u64,
    missing_sidecar_keys: u64,
    duplicate_manifest_keys: u64,
}

#[derive(Debug, serde::Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    dataset_id: &'static str,
    dataset_revision: &'static str,
    dataset_revision_commit_sha: String,
    upstream_repo: &'static str,
    upstream_commit: &'static str,
    sidecar_jsonl_sha256: String,
    split_root_relative_path: String,
    species: BTreeMap<String, SpeciesReplay>,
    sidecar_rows: u64,
    matched_sidecar_rows: u64,
    mismatched_sidecar_rows: u64,
    missing_sidecar_keys: u64,
    duplicate_manifest_keys: u64,
    status: &'static str,
    errors: Vec<Finding>,
    warnings: Vec<Finding>,
    scope_note: &'static str,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn file_digest(path: &Path, root: &Path, relative: &str) -> Result<FileDigest, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {relative}: {error}"))?;
    Ok(FileDigest {
        relative_path: relative.to_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(&bytes),
    })
}

fn sample_key_md5(source_archive: &str, source_hash: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(format!("{source_archive}|{source_hash}").as_bytes());
    hex::encode(hasher.finalize())
}

fn normalize_split(value: &str) -> Option<&'static str> {
    match value {
        "train" => Some("train"),
        "val" | "validation" => Some("val"),
        "test" => Some("test"),
        _ => None,
    }
}

fn archive_bank(source_archive: &str) -> Option<&'static str> {
    if source_archive.starts_with("edits_and_adj_controls_") {
        Some("operation_bank")
    } else if source_archive.starts_with("junction_controls_") {
        Some("operation_bank_junction_controls")
    } else if source_archive.starts_with("synapse_controls_") {
        Some("operation_bank_synapse_controls")
    } else {
        None
    }
}

/// Upstream Python calls int(value) for integer lineage keys. Preserve 64-bit IDs
/// exactly; do not round them through f64.
fn python_int_key(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed == "None" {
                return None;
            }
            if let Ok(value) = trimmed.parse::<i128>() {
                return Some(value.to_string());
            }
            if let Ok(value) = trimmed.parse::<u128>() {
                return Some(value.to_string());
            }
            let float = trimmed.parse::<f64>().ok()?;
            if float.is_finite() && float.fract() == 0.0 && float >= i128::MIN as f64 && float < i128::MAX as f64 {
                return Some(format!("{:.0}", float));
            }
            None
        }
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Some(value.to_string())
            } else if let Some(value) = value.as_u64() {
                Some(value.to_string())
            } else {
                let float = value.as_f64()?;
                if float.is_finite() && float.fract() == 0.0 && float >= i128::MIN as f64 && float < i128::MAX as f64 {
                    Some(format!("{:.0}", float))
                } else {
                    None
                }
            }
        }
        _ => None,
    }
}

fn python_float_key(value: &Value) -> Option<String> {
    let number = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(value) => value.parse::<f64>().ok()?,
        _ => return None,
    };
    if !number.is_finite() {
        return None;
    }
    // Python tuple equality treats -0.0 and 0.0 as equal.
    Some(if number == 0.0 { "0".to_owned() } else { number.to_string() })
}

fn split_file_key(bank: &str, value: &Value) -> Option<String> {
    match bank {
        "operation_bank" | "operation_bank_junction_controls" => {
            Some(python_int_key(value)?)
        }
        "operation_bank_synapse_controls" => {
            let tuple = value.as_array()?;
            if tuple.len() != 3 {
                return None;
            }
            let pre = python_int_key(&tuple[0])?;
            let post = python_int_key(&tuple[1])?;
            let coords = tuple[2].as_array()?;
            if coords.len() != 3 {
                return None;
            }
            let x = python_float_key(&coords[0])?;
            let y = python_float_key(&coords[1])?;
            let z = python_float_key(&coords[2])?;
            Some(format!("{pre}|{post}|{x},{y},{z}"))
        }
        _ => None,
    }
}

fn row_lineage_key(row: &ParsedRow) -> Option<String> {
    let local = match row.bank.as_str() {
        "operation_bank" => python_int_key(row.metadata.get("operation_id")?)?,
        "operation_bank_junction_controls" => {
            python_int_key(row.metadata.get("source_operation_id")?)?
        }
        "operation_bank_synapse_controls" => {
            let pre = python_int_key(row.metadata.get("pre_pt_root_id")?)?;
            let post = python_int_key(row.metadata.get("post_pt_root_id")?)?;
            let center = row.metadata.get("synapse_ctr_pt_nm")?.as_array()?;
            if center.len() != 3 {
                return None;
            }
            let x = python_float_key(&center[0])?;
            let y = python_float_key(&center[1])?;
            let z = python_float_key(&center[2])?;
            format!("{pre}|{post}|{x},{y},{z}")
        }
        _ => return None,
    };
    Some(format!("{}|{}|{}", row.row.species, row.bank, local))
}

fn parse_sidecar(bytes: &[u8], errors: &mut Vec<Finding>) -> Vec<ParsedRow> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        errors.push(Finding::global("sidecar_not_utf8", "sidecar JSONL is not UTF-8"));
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = match serde_json::from_str::<SidecarRow>(line) {
            Ok(row) => row,
            Err(error) => {
                errors.push(Finding::global("sidecar_row_invalid", format!("line {}: {error}", line_index + 1)));
                continue;
            }
        };
        let metadata: Value = match serde_json::from_str(&row.metadata) {
            Ok(value) if value.is_object() => value,
            Ok(_) => {
                errors.push(Finding::global("sidecar_metadata_not_object", format!("line {} metadata must be a JSON object", line_index + 1)));
                continue;
            }
            Err(error) => {
                errors.push(Finding::global("sidecar_metadata_invalid", format!("line {}: {error}", line_index + 1)));
                continue;
            }
        };
        let bank = match archive_bank(&row.source_archive) {
            Some(bank) => bank.to_owned(),
            None => {
                errors.push(Finding::global("sidecar_archive_bank_unknown", format!("line {} has unsupported source_archive {:?}", line_index + 1, row.source_archive)));
                continue;
            }
        };
        let split = match normalize_split(&row.split) {
            Some(split) => split.to_owned(),
            None => {
                errors.push(Finding::global("sidecar_split_unknown", format!("line {} has unsupported split {:?}", line_index + 1, row.split)));
                continue;
            }
        };
        if row.combined_sample_hash != sample_key_md5(&row.source_archive, &row.source_archive_sample_hash) {
            errors.push(Finding::global("sidecar_sample_key_mismatch", format!("line {} combined_sample_hash does not match the upstream MD5 row-key formula", line_index + 1)));
        }
        rows.push(ParsedRow { row, metadata, bank, split, source_line: line_index + 1 });
    }
    rows
}

#[derive(Debug)]
struct LoadedSpecies {
    replay: SpeciesReplay,
    assignments: BTreeMap<String, String>,
}

fn load_species_split_index(
    root: &Path,
    species: &str,
    errors: &mut Vec<Finding>,
) -> Option<LoadedSpecies> {
    let species_root = root.join(species);
    let manifest_path = species_root.join("manifest.json");
    let manifest_bytes = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            errors.push(Finding::global("split_manifest_missing", format!("{species}/manifest.json: {error}")));
            return None;
        }
    };
    let manifest: Value = match serde_json::from_slice(&manifest_bytes) {
        Ok(value) if value.is_object() => value,
        Ok(_) => {
            errors.push(Finding::global("split_manifest_invalid", format!("{species}/manifest.json must be an object")));
            return None;
        }
        Err(error) => {
            errors.push(Finding::global("split_manifest_invalid", format!("{species}/manifest.json: {error}")));
            return None;
        }
    };
    if manifest.get("species").and_then(Value::as_str) != Some(species) {
        errors.push(Finding::global("split_manifest_species_mismatch", format!("{species}/manifest.json species does not match directory")));
    }
    let edge = manifest.get("edge_nm").and_then(Value::as_f64);
    if edge.is_none_or(|value| (value - EXPECTED_EDGE_NM).abs() > 1.0e-9) {
        errors.push(Finding::global("split_manifest_edge_mismatch", format!("{species}/manifest.json edge_nm must be {EXPECTED_EDGE_NM}")));
    }

    let mut assignments = BTreeMap::<String, String>::new();
    let mut counts = BTreeMap::<String, u64>::new();
    let mut split_files = BTreeMap::<String, FileDigest>::new();
    let mut duplicate_count = 0_u64;
    for split in SPLITS {
        let filename = format!("{split}.json");
        let path = species_root.join(&filename);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                errors.push(Finding::global("split_file_missing", format!("{species}/{filename}: {error}")));
                continue;
            }
        };
        let relative = format!("{species}/{filename}");
        match file_digest(&path, root, &relative) {
            Ok(digest) => { split_files.insert(split.to_owned(), digest); }
            Err(error) => errors.push(Finding::global("split_file_digest_failed", error)),
        }
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) if value.is_object() => value,
            Ok(_) => {
                errors.push(Finding::global("split_file_invalid", format!("{species}/{filename} must be an object")));
                continue;
            }
            Err(error) => {
                errors.push(Finding::global("split_file_invalid", format!("{species}/{filename}: {error}")));
                continue;
            }
        };
        let object = value.as_object().expect("object validated");
        for bank in BANKS {
            let Some(keys) = object.get(bank).and_then(Value::as_array) else {
                errors.push(Finding::global("split_bank_missing", format!("{species}/{filename} missing array for {bank}")));
                continue;
            };
            counts.insert(format!("{bank}|{split}"), keys.len() as u64);
            for (index, raw_key) in keys.iter().enumerate() {
                let Some(key) = split_file_key(bank, raw_key) else {
                    errors.push(Finding::global("split_key_invalid", format!("{species}/{filename} {bank}[{index}] has invalid upstream key shape")));
                    continue;
                };
                let full_key = format!("{species}|{bank}|{key}");
                if let Some(previous) = assignments.insert(full_key.clone(), split.to_owned()) {
                    duplicate_count += 1;
                    errors.push(Finding::global(
                        "split_key_duplicate_or_cross_split",
                        format!("{species}/{filename} key {full_key:?} already appears in {previous}; upstream dict loading would overwrite one assignment"),
                    ));
                }
            }
        }
    }

    if let Some(op_counts) = manifest.get("op_counts").and_then(Value::as_object) {
        for bank in BANKS {
            for split in SPLITS {
                let actual = counts.get(&format!("{bank}|{split}")).copied().unwrap_or(0);
                let declared = op_counts.get(bank).and_then(Value::as_object)
                    .and_then(|by_split| by_split.get(split)).and_then(Value::as_u64);
                if declared != Some(actual) {
                    errors.push(Finding::global(
                        "split_manifest_count_mismatch",
                        format!("{species}/manifest.json op_counts[{bank}][{split}] declares {declared:?}, actual JSON key count is {actual}"),
                    ));
                }
            }
        }
    } else {
        errors.push(Finding::global("split_manifest_counts_missing", format!("{species}/manifest.json missing op_counts object")));
    }

    let mut split_manifest_hasher = Sha256::new();
    split_manifest_hasher.update(b"connectomebench2-upstream-split-manifest-set-v1\n");
    split_manifest_hasher.update(sha256(&manifest_bytes).as_bytes());
    for split in SPLITS {
        if let Some(file) = split_files.get(split) {
            split_manifest_hasher.update(split.as_bytes());
            split_manifest_hasher.update(file.sha256.as_bytes());
        }
    }
    let replay = SpeciesReplay {
        species: species.to_owned(),
        split_manifest_sha256: hex::encode(split_manifest_hasher.finalize()),
        split_files,
        source_key_counts: counts,
        sidecar_row_counts: BTreeMap::new(),
        matched_sidecar_rows: 0,
        mismatched_sidecar_rows: 0,
        missing_sidecar_keys: 0,
        duplicate_manifest_keys: duplicate_count,
    };
    Some(LoadedSpecies { replay, assignments })
}

fn audit(
    sidecar_bytes: &[u8],
    sidecar_digest: String,
    split_root: &Path,
    dataset_revision_commit_sha: String,
) -> Report {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let rows = parse_sidecar(sidecar_bytes, &mut errors);
    let root = match split_root.canonicalize() {
        Ok(root) => root,
        Err(error) => {
            errors.push(Finding::global("split_root_unreadable", error.to_string()));
            split_root.to_path_buf()
        }
    };
    let species_names: BTreeSet<String> = rows.iter().map(|row| row.row.species.clone()).collect();
    let mut species_replays = BTreeMap::new();
    let mut total_matched = 0_u64;
    let mut total_mismatched = 0_u64;
    let mut total_missing = 0_u64;
    let mut total_duplicates = 0_u64;
    for species in species_names {
        let Some(mut loaded) = load_species_split_index(&root, &species, &mut errors) else {
            continue;
        };
        for row in rows.iter().filter(|row| row.row.species == species) {
            *loaded.replay.sidecar_row_counts.entry(format!("{}|{}", row.bank, row.split)).or_default() += 1;
            let Some(key) = row_lineage_key(row) else {
                loaded.replay.missing_sidecar_keys += 1;
                errors.push(Finding::row("sidecar_lineage_key_missing", row, None, "cannot reconstruct the exact upstream bank key from metadata"));
                continue;
            };
            match loaded.assignments.get(&key) {
                None => {
                    loaded.replay.missing_sidecar_keys += 1;
                    errors.push(Finding::row("sidecar_key_absent_from_split_manifests", row, Some(&key), "no exact key assignment exists in the supplied upstream split JSON files"));
                }
                Some(expected) if expected == &row.split => {
                    loaded.replay.matched_sidecar_rows += 1;
                }
                Some(expected) => {
                    loaded.replay.mismatched_sidecar_rows += 1;
                    errors.push(Finding::row("sidecar_partition_disagrees_with_upstream", row, Some(&key), format!("sidecar says {}, upstream split manifests say {expected}", row.split)));
                }
            }
        }
        total_matched += loaded.replay.matched_sidecar_rows;
        total_mismatched += loaded.replay.mismatched_sidecar_rows;
        total_missing += loaded.replay.missing_sidecar_keys;
        total_duplicates += loaded.replay.duplicate_manifest_keys;
        species_replays.insert(species, loaded.replay);
    }
    errors.sort_by(|a,b| a.code.cmp(&b.code).then(a.species.cmp(&b.species)).then(a.key.cmp(&b.key)).then(a.sample_id.cmp(&b.sample_id)));
    warnings.sort_by(|a,b| a.code.cmp(&b.code).then(a.species.cmp(&b.species)));
    Report {
        schema_version: 1,
        tool: "connectome-split-replay-audit",
        dataset_id: DATASET_ID,
        dataset_revision: DATASET_REVISION,
        dataset_revision_commit_sha,
        upstream_repo: "https://github.com/timfarkas/ConnectomeBench2",
        upstream_commit: UPSTREAM_COMMIT,
        sidecar_jsonl_sha256: sidecar_digest,
        split_root_relative_path: split_root.to_string_lossy().into_owned(),
        species: species_replays,
        sidecar_rows: rows.len() as u64,
        matched_sidecar_rows: total_matched,
        mismatched_sidecar_rows: total_mismatched,
        missing_sidecar_keys: total_missing,
        duplicate_manifest_keys: total_duplicates,
        status: if errors.is_empty() { "PASS" } else { "FAIL" },
        errors,
        warnings,
        scope_note: "PASS means every observed metadata-sidecar row maps to the exact partition in the supplied upstream per-species/per-bank split files, the source split-file counts match the source manifest, and no source split key is duplicated or overwritten across partitions. It does not prove that the supplied files are the ones used to publish the dataset unless their provenance is independently authenticated and their hashes are pinned; it does not prove sample independence, biological correctness, or benchmark representativeness.",
    }
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if args.dataset_revision_commit_sha.len() != 40
        || !args.dataset_revision_commit_sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("--dataset-revision-commit-sha must be the lowercase 40-character commit SHA resolved from Hugging Face revision=v2".into());
    }
    if args.upstream_commit != UPSTREAM_COMMIT {
        return Err(format!("--upstream-commit must equal the audited generator commit {UPSTREAM_COMMIT}"));
    }
    let bytes = fs::read(&args.sidecar_jsonl)
        .map_err(|error| format!("cannot read sidecar {}: {error}", args.sidecar_jsonl.display()))?;
    let digest = sha256(&bytes);
    let report = audit(&bytes, digest, &args.split_root, args.dataset_revision_commit_sha);
    let rendered = serde_json::to_string_pretty(&report).map_err(|error| format!("serialize report: {error}"))? + "\n";
    if let Some(path) = args.output {
        fs::write(&path, rendered.as_bytes()).map_err(|error| format!("write {}: {error}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(if report.status == "PASS" { 0 } else { 1 })
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(code as u8),
        Err(error) => {
            eprintln!("connectome-split-replay-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn root() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "connectome-split-replay-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("mouse")).expect("create fixture root");
        path
    }

    fn write_json(path: &Path, value: &Value) {
        fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }

    fn fixture() -> (PathBuf, Vec<u8>) {
        let root = root();
        let species = root.join("mouse");
        let mut manifest_counts = serde_json::Map::new();
        let bank_keys = [
            ("operation_bank", json!([11])),
            ("operation_bank_junction_controls", json!([12])),
            ("operation_bank_synapse_controls", json!([[101, 202, [1.5, 2.5, 3.5]]])),
        ];
        let mut train = serde_json::Map::new();
        let mut val = serde_json::Map::new();
        let mut test = serde_json::Map::new();
        for (bank, key) in &bank_keys {
            train.insert((*bank).to_owned(), json!([]));
            val.insert((*bank).to_owned(), json!([]));
            test.insert((*bank).to_owned(), json!([]));
            let count = if *bank == "operation_bank" {
                train.insert((*bank).to_owned(), key.clone());
                key.as_array().unwrap().len()
            } else if *bank == "operation_bank_junction_controls" {
                val.insert((*bank).to_owned(), key.clone());
                key.as_array().unwrap().len()
            } else {
                test.insert((*bank).to_owned(), key.clone());
                key.as_array().unwrap().len()
            };
            let mut values = serde_json::Map::new();
            values.insert("train".to_owned(), json!(if *bank == "operation_bank" { count } else { 0 }));
            values.insert("val".to_owned(), json!(if *bank == "operation_bank_junction_controls" { count } else { 0 }));
            values.insert("test".to_owned(), json!(if *bank == "operation_bank_synapse_controls" { count } else { 0 }));
            manifest_counts.insert((*bank).to_owned(), Value::Object(values));
        }
        write_json(&species.join("train.json"), &Value::Object(train));
        write_json(&species.join("val.json"), &Value::Object(val));
        write_json(&species.join("test.json"), &Value::Object(test));
        write_json(&species.join("manifest.json"), &json!({
            "species":"mouse",
            "edge_nm":50_000.0,
            "op_counts":manifest_counts
        }));
        let rows = vec![
            json!({
                "combined_sample_hash":sample_key_md5("edits_and_adj_controls_mouse","000000000011"),
                "source_archive_sample_hash":"000000000011",
                "source_archive":"edits_and_adj_controls_mouse",
                "split":"train",
                "species":"mouse",
                "metadata":json!({"operation_id":11}).to_string()
            }),
            json!({
                "combined_sample_hash":sample_key_md5("junction_controls_mouse","000000000012"),
                "source_archive_sample_hash":"000000000012",
                "source_archive":"junction_controls_mouse",
                "split":"val",
                "species":"mouse",
                "metadata":json!({"source_operation_id":12}).to_string()
            }),
            json!({
                "combined_sample_hash":sample_key_md5("synapse_controls_mouse","000000000013"),
                "source_archive_sample_hash":"000000000013",
                "source_archive":"synapse_controls_mouse",
                "split":"test",
                "species":"mouse",
                "metadata":json!({"pre_pt_root_id":101,"post_pt_root_id":202,"synapse_ctr_pt_nm":[1.5,2.5,3.5]}).to_string()
            })
        ];
        let bytes = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n").into_bytes();
        (root, bytes)
    }

    #[test]
    fn exact_manifest_assignment_replays_for_all_three_source_banks() {
        let (root, bytes) = fixture();
        let report = audit(&bytes, sha256(&bytes), &root, "a".repeat(40));
        assert_eq!(report.status, "PASS", "{:?}", report.errors);
        assert_eq!(report.sidecar_rows, 3);
        assert_eq!(report.matched_sidecar_rows, 3);
        assert_eq!(report.missing_sidecar_keys, 0);
    }

    #[test]
    fn rejects_sidecar_partition_that_differs_from_upstream_manifest() {
        let (root, bytes) = fixture();
        let altered = String::from_utf8(bytes).unwrap().replace("\"split\":\"train\"", "\"split\":\"test\"");
        let report = audit(altered.as_bytes(), sha256(altered.as_bytes()), &root, "a".repeat(40));
        assert!(report.errors.iter().any(|finding| finding.code == "sidecar_partition_disagrees_with_upstream"));
    }

    #[test]
    fn rejects_a_row_key_absent_from_all_source_split_files() {
        let (root, bytes) = fixture();
        let altered = String::from_utf8(bytes).unwrap().replace("\"operation_id\":11", "\"operation_id\":99");
        let report = audit(altered.as_bytes(), sha256(altered.as_bytes()), &root, "a".repeat(40));
        assert!(report.errors.iter().any(|finding| finding.code == "sidecar_key_absent_from_split_manifests"));
    }

    #[test]
    fn rejects_duplicate_key_overwritten_by_upstream_split_dictionary() {
        let (root, bytes) = fixture();
        let species = root.join("mouse");
        let mut train: Value = serde_json::from_slice(&fs::read(species.join("train.json")).unwrap()).unwrap();
        let mut val: Value = serde_json::from_slice(&fs::read(species.join("val.json")).unwrap()).unwrap();
        train["operation_bank"] = json!([11]);
        val["operation_bank"] = json!([11]);
        write_json(&species.join("train.json"), &train);
        write_json(&species.join("val.json"), &val);
        let mut manifest: Value = serde_json::from_slice(&fs::read(species.join("manifest.json")).unwrap()).unwrap();
        manifest["op_counts"]["operation_bank"]["train"] = json!(1);
        manifest["op_counts"]["operation_bank"]["val"] = json!(1);
        write_json(&species.join("manifest.json"), &manifest);
        let report = audit(&bytes, sha256(&bytes), &root, "a".repeat(40));
        assert!(report.errors.iter().any(|finding| finding.code == "split_key_duplicate_or_cross_split"));
    }

    #[test]
    fn rejects_manifest_counts_that_do_not_match_split_json() {
        let (root, bytes) = fixture();
        let path = root.join("mouse/manifest.json");
        let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest["op_counts"]["operation_bank"]["train"] = json!(88);
        write_json(&path, &manifest);
        let report = audit(&bytes, sha256(&bytes), &root, "a".repeat(40));
        assert!(report.errors.iter().any(|finding| finding.code == "split_manifest_count_mismatch"));
    }

    #[test]
    fn normalizes_val_alias_to_upstream_val_file() {
        let (root, bytes) = fixture();
        let altered = String::from_utf8(bytes).unwrap().replace("\"split\":\"val\"", "\"split\":\"validation\"");
        let report = audit(altered.as_bytes(), sha256(altered.as_bytes()), &root, "a".repeat(40));
        assert_eq!(report.status, "PASS", "{:?}", report.errors);
    }

    #[test]
    fn synapse_tuple_key_preserves_large_root_ids_without_f64_rounding() {
        let key = json!([864691128570256963u64, 864691128637601234u64, [0.0, -0.0, 1.25]]);
        assert_eq!(
            split_file_key("operation_bank_synapse_controls", &key).unwrap(),
            "864691128570256963|864691128637601234|0,0,1.25"
        );
    }

    #[test]
    fn rejects_invalid_source_revision_sha() {
        assert_eq!("x".repeat(39).len(), 39);
        assert!(!"x".repeat(40).bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }

    #[test]
    fn sample_hash_is_a_row_key_not_a_provenance_digest() {
        let value = sample_key_md5("edits_and_adj_controls_mouse", "000000000011");
        assert_eq!(value.len(), 32);
        assert_ne!(value, sha256(b"edits_and_adj_controls_mouse|000000000011"));
    }
}
