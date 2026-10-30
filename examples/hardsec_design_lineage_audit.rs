//! HARDSEC-005 design-lineage split auditor.
//!
//! Audits a strict TSV manifest before hardware-Trojan model training. It verifies
//! exact evaluated design artifacts (normally the netlist) and enforces the group
//! boundary declared by the split protocol. Shared upstream RTL ancestry belongs
//! in the lineage inventory, not in the evaluated-artifact path/hash columns.
//! It does not parse RTL/netlists, infer Trojan labels, prove semantic
//! graph similarity, or certify a design.
//!
//! cargo run --example hardsec_design_lineage_audit --features hardsec-design-audit -- \
//!   --manifest /path/to/design-samples.tsv --root /path/to/pinned-dataset \
//!   --receipt /path/outside/dataset/design-lineage-receipt.json
//!
//! Supported split_protocol values:
//!   random_gate               Diagnostic only; lineage overlaps are reported.
//!   loco_netlist              netlist_id cannot cross train/validation/test.
//!   lofo_base_family          base_design_family_id cannot cross splits.
//!   leave_trojan_family_out   trojan_family_id cannot cross splits for Trojan rows.
//!   leave_toolchain_out       synthesis_config_id cannot cross splits.
//!   leave_library_out         cell_library_id cannot cross splits.
//!   leave_benchmark_source_out benchmark_source_id cannot cross splits; identical
//!                              benign paired controls are counted and reported.
//!
//! TSV header (exact order):
//! dataset_id, source_artifact_id, source_path, source_sha256, sample_id,
//! design_id, base_design_family_id, netlist_id, trojan_family_id,
//! synthesis_config_id, cell_library_id, benchmark_source_id,
//! split_protocol, split, label

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const HEADER: [&str; 15] = [
    "dataset_id",
    "source_artifact_id",
    "source_path",
    "source_sha256",
    "sample_id",
    "design_id",
    "base_design_family_id",
    "netlist_id",
    "trojan_family_id",
    "synthesis_config_id",
    "cell_library_id",
    "benchmark_source_id",
    "split_protocol",
    "split",
    "label",
];

const CLAIM_SCOPE: &str = "source_integrity_and_declared_design_and_benchmark_source_split_integrity_only";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitProtocol {
    RandomGate,
    LocoNetlist,
    LofoBaseFamily,
    LeaveTrojanFamilyOut,
    LeaveToolchainOut,
    LeaveLibraryOut,
    LeaveBenchmarkSourceOut,
}

impl SplitProtocol {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "random_gate" => Ok(Self::RandomGate),
            "loco_netlist" => Ok(Self::LocoNetlist),
            "lofo_base_family" => Ok(Self::LofoBaseFamily),
            "leave_trojan_family_out" => Ok(Self::LeaveTrojanFamilyOut),
            "leave_toolchain_out" => Ok(Self::LeaveToolchainOut),
            "leave_library_out" => Ok(Self::LeaveLibraryOut),
            "leave_benchmark_source_out" => Ok(Self::LeaveBenchmarkSourceOut),
            _ => Err(format!("unsupported split_protocol: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RandomGate => "random_gate",
            Self::LocoNetlist => "loco_netlist",
            Self::LofoBaseFamily => "lofo_base_family",
            Self::LeaveTrojanFamilyOut => "leave_trojan_family_out",
            Self::LeaveToolchainOut => "leave_toolchain_out",
            Self::LeaveLibraryOut => "leave_library_out",
            Self::LeaveBenchmarkSourceOut => "leave_benchmark_source_out",
        }
    }

    fn group_field(self) -> Option<&'static str> {
        match self {
            Self::RandomGate => None,
            Self::LocoNetlist => Some("netlist_id"),
            Self::LofoBaseFamily => Some("base_design_family_id"),
            Self::LeaveTrojanFamilyOut => Some("trojan_family_id"),
            Self::LeaveToolchainOut => Some("synthesis_config_id"),
            Self::LeaveLibraryOut => Some("cell_library_id"),
            Self::LeaveBenchmarkSourceOut => Some("benchmark_source_id"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sample {
    dataset_id: String,
    source_artifact_id: String,
    source_path: String,
    source_sha256: String,
    sample_id: String,
    design_id: String,
    base_design_family_id: String,
    netlist_id: String,
    trojan_family_id: String,
    synthesis_config_id: String,
    cell_library_id: String,
    benchmark_source_id: String,
    split_protocol: SplitProtocol,
    split: String,
    label: String,
}

impl Sample {
    fn group_value(&self, field: &str) -> Option<&str> {
        match field {
            "netlist_id" => Some(&self.netlist_id),
            "base_design_family_id" => Some(&self.base_design_family_id),
            "trojan_family_id" if self.label == "trojan" => Some(&self.trojan_family_id),
            "synthesis_config_id" => Some(&self.synthesis_config_id),
            "cell_library_id" => Some(&self.cell_library_id),
            "benchmark_source_id" => Some(&self.benchmark_source_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    manifest: PathBuf,
    root: PathBuf,
    receipt: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut root = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Err(
                "usage: hardsec_design_lineage_audit --manifest FILE --root DIR --receipt NEW_FILE"
                    .to_owned(),
            );
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--root" => root = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        root: root.ok_or("--root is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    })
}

fn sha256_reader(mut reader: impl Read) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("read while hashing: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_file_stable(path: &Path) -> Result<String, String> {
    let before = fs::metadata(path)
        .map_err(|error| format!("stat {} before hashing: {error}", path.display()))?;
    if !before.is_file() {
        return Err(format!("source is not a regular file: {}", path.display()));
    }
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let digest = sha256_reader(file)?;
    let after = fs::metadata(path)
        .map_err(|error| format!("stat {} after hashing: {error}", path.display()))?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(format!("source changed while hashing: {}", path.display()));
    }
    Ok(digest)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty() || value.contains('\\') {
        return Err(format!(
            "source_path must be a non-empty slash-separated relative path: {value:?}"
        ));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(format!("absolute source_path is forbidden: {value}"));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(format!("unsafe source_path component in {value}")),
        }
    }
    Ok(path.to_path_buf())
}

fn resolve_source(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let rel = safe_relative_path(relative)?;
    let mut current = root.to_path_buf();
    let components: Vec<_> = rel.components().collect();

    for (index, component) in components.iter().enumerate() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("stat source {}: {error}", current.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("symlink source path is forbidden: {relative}"));
        }
        if index + 1 == components.len() {
            if !metadata.is_file() {
                return Err(format!("source is not a regular file: {relative}"));
            }
        } else if !metadata.is_dir() {
            return Err(format!("source parent is not a directory: {relative}"));
        }
    }
    Ok(current)
}

fn parse_manifest(bytes: &[u8]) -> Result<Vec<Sample>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("manifest is not UTF-8: {error}"))?;
    let mut lines = text.split_terminator('\n');
    let first = lines.next().ok_or("manifest is empty")?.trim_end_matches('\r');
    let actual_header: Vec<_> = first.split('\t').collect();
    if actual_header.as_slice() != HEADER.as_slice() {
        return Err(format!("manifest header must exactly match: {}", HEADER.join("\t")));
    }

    let mut samples = Vec::new();
    let mut sample_ids = BTreeSet::new();
    let mut source_id_map: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut dataset_id: Option<String> = None;
    let mut protocol: Option<SplitProtocol> = None;
    let mut splits = BTreeSet::new();

    for (line_index, raw_line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            return Err(format!("blank manifest row at line {line_number}"));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != HEADER.len() {
            return Err(format!(
                "line {line_number}: expected {} TSV fields, got {}",
                HEADER.len(),
                fields.len()
            ));
        }
        for (index, value) in fields.iter().enumerate() {
            if value.trim() != *value {
                return Err(format!(
                    "line {line_number}: {} has leading or trailing whitespace",
                    HEADER[index]
                ));
            }
        }
        let get = |index: usize| fields[index].to_owned();
        let row_dataset = get(0);
        let source_artifact_id = get(1);
        let source_path = get(2);
        let source_sha256 = get(3).to_ascii_lowercase();
        let sample_id = get(4);
        let design_id = get(5);
        let base_design_family_id = get(6);
        let netlist_id = get(7);
        let trojan_family_id = get(8);
        let synthesis_config_id = get(9);
        let cell_library_id = get(10);
        let benchmark_source_id = get(11);
        let row_protocol = SplitProtocol::parse(&get(12))
            .map_err(|error| format!("line {line_number}: {error}"))?;
        let split = get(13);
        let label = get(14);

        for (name, value) in [
            ("dataset_id", &row_dataset),
            ("source_artifact_id", &source_artifact_id),
            ("source_path", &source_path),
            ("sample_id", &sample_id),
            ("design_id", &design_id),
            ("base_design_family_id", &base_design_family_id),
            ("netlist_id", &netlist_id),
            ("split", &split),
            ("label", &label),
        ] {
            if value.is_empty() {
                return Err(format!("line {line_number}: {name} must be non-empty"));
            }
        }

        if !valid_sha256(&source_sha256) {
            return Err(format!(
                "line {line_number}: source_sha256 must be 64 hexadecimal characters"
            ));
        }
        safe_relative_path(&source_path)
            .map_err(|error| format!("line {line_number}: {error}"))?;

        if split != "train" && split != "validation" && split != "test" {
            return Err(format!("line {line_number}: split must be train, validation, or test"));
        }
        if label != "benign" && label != "trojan" {
            return Err(format!("line {line_number}: label must be benign or trojan"));
        }
        if label == "trojan" && trojan_family_id.is_empty() {
            return Err(format!("line {line_number}: Trojan row requires trojan_family_id"));
        }
        if label == "benign" && !trojan_family_id.is_empty() {
            return Err(format!("line {line_number}: benign row must leave trojan_family_id empty"));
        }
        match row_protocol {
            SplitProtocol::LeaveToolchainOut if synthesis_config_id.is_empty() => {
                return Err(format!(
                    "line {line_number}: leave_toolchain_out requires synthesis_config_id"
                ));
            }
            SplitProtocol::LeaveLibraryOut if cell_library_id.is_empty() => {
                return Err(format!(
                    "line {line_number}: leave_library_out requires cell_library_id"
                ));
            }
            SplitProtocol::LeaveBenchmarkSourceOut if benchmark_source_id.is_empty() => {
                return Err(format!(
                    "line {line_number}: leave_benchmark_source_out requires benchmark_source_id"
                ));
            }
            _ => {}
        }

        if let Some(previous) = &dataset_id {
            if previous != &row_dataset {
                return Err(format!("line {line_number}: mixed dataset_id values"));
            }
        } else {
            dataset_id = Some(row_dataset.clone());
        }
        if let Some(previous) = protocol {
            if previous != row_protocol {
                return Err(format!("line {line_number}: mixed split_protocol values"));
            }
        } else {
            protocol = Some(row_protocol);
        }

        if !sample_ids.insert(sample_id.clone()) {
            return Err(format!("line {line_number}: duplicate sample_id {sample_id}"));
        }
        let identity = (source_path.clone(), source_sha256.clone());
        if let Some(previous) = source_id_map.insert(source_artifact_id.clone(), identity.clone()) {
            if previous != identity {
                return Err(format!(
                    "line {line_number}: source_artifact_id maps to inconsistent path/hash"
                ));
            }
        }

        splits.insert(split.clone());
        samples.push(Sample {
            dataset_id: row_dataset,
            source_artifact_id,
            source_path,
            source_sha256,
            sample_id,
            design_id,
            base_design_family_id,
            netlist_id,
            trojan_family_id,
            synthesis_config_id,
            cell_library_id,
            benchmark_source_id,
            split_protocol: row_protocol,
            split,
            label,
        });
    }

    if samples.is_empty() {
        return Err("manifest has no sample rows".to_owned());
    }
    for required in ["train", "validation", "test"] {
        if !splits.contains(required) {
            return Err(format!("manifest missing required split: {required}"));
        }
    }

    samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    Ok(samples)
}

fn audit_groups(
    samples: &[Sample],
    protocol: SplitProtocol,
) -> Result<BTreeMap<String, usize>, String> {
    let Some(field) = protocol.group_field() else {
        return Ok(BTreeMap::new());
    };
    let mut group_to_split: BTreeMap<String, String> = BTreeMap::new();
    let mut split_groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for sample in samples {
        let Some(value) = sample.group_value(field) else {
            continue;
        };
        if value.is_empty() {
            return Err(format!(
                "sample {} requires non-empty {field} for {}",
                sample.sample_id,
                protocol.as_str()
            ));
        }
        if let Some(previous_split) = group_to_split.get(value) {
            if previous_split != &sample.split {
                return Err(format!(
                    "lineage leakage: {field}={value} appears in both {previous_split} and {}",
                    sample.split
                ));
            }
        } else {
            group_to_split.insert(value.to_owned(), sample.split.clone());
        }
        split_groups
            .entry(sample.split.clone())
            .or_default()
            .insert(value.to_owned());
    }

    Ok(split_groups
        .into_iter()
        .map(|(split, groups)| (split, groups.len()))
        .collect())
}

#[derive(Debug, Clone, Default)]
struct ExactHashAudit {
    cross_split_source_hash_count: usize,
    reused_benign_control_artifact_count: usize,
    reused_benign_control_artifacts: Vec<serde_json::Value>,
}

fn audit_exact_source_hash_splits(
    samples: &[Sample],
    protocol: SplitProtocol,
) -> Result<ExactHashAudit, String> {
    let mut hash_to_samples: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
    for sample in samples {
        hash_to_samples
            .entry(sample.source_sha256.clone())
            .or_default()
            .push(sample);
    }

    let mut audit = ExactHashAudit::default();
    for (digest, rows) in hash_to_samples {
        let labels = rows
            .iter()
            .map(|sample| sample.label.as_str())
            .collect::<BTreeSet<_>>();
        if labels.len() > 1 {
            return Err(format!(
                "identical evaluated artifact has inconsistent labels for sha256={digest}"
            ));
        }
        let splits = rows
            .iter()
            .map(|sample| sample.split.as_str())
            .collect::<BTreeSet<_>>();
        if splits.len() <= 1 {
            continue;
        }

        audit.cross_split_source_hash_count += 1;
        if protocol == SplitProtocol::RandomGate {
            continue;
        }
        if protocol == SplitProtocol::LeaveBenchmarkSourceOut
            && labels.len() == 1
            && labels.contains("benign")
        {
            audit.reused_benign_control_artifact_count += 1;
            audit.reused_benign_control_artifacts.push(serde_json::json!({
                "sha256": digest,
                "label": "benign",
                "sample_ids": rows.iter().map(|sample| sample.sample_id.clone()).collect::<BTreeSet<_>>(),
                "benchmark_source_ids": rows.iter().map(|sample| sample.benchmark_source_id.clone()).collect::<BTreeSet<_>>(),
                "splits": splits,
            }));
            continue;
        }

        return Err(format!(
            "exact source artifact hash leakage across splits for sha256={digest}"
        ));
    }

    Ok(audit)
}

fn lineage_overlap_counts(samples: &[Sample], field: &str) -> BTreeMap<String, usize> {
    let mut group_splits: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for sample in samples {
        let value = match field {
            "design_id" => &sample.design_id,
            "base_design_family_id" => &sample.base_design_family_id,
            "netlist_id" => &sample.netlist_id,
            "trojan_family_id" if sample.label == "trojan" => &sample.trojan_family_id,
            "synthesis_config_id" => &sample.synthesis_config_id,
            "cell_library_id" => &sample.cell_library_id,
            "benchmark_source_id" => &sample.benchmark_source_id,
            _ => continue,
        };
        if !value.is_empty() {
            group_splits.entry(value.clone()).or_default().insert(sample.split.clone());
        }
    }
    let overlapping_group_count = group_splits
        .values()
        .filter(|splits| splits.len() > 1)
        .count();
    BTreeMap::from([(field.to_owned(), overlapping_group_count)])
}

fn receipt_path_outside_root(root: &Path, receipt: &Path) -> Result<PathBuf, String> {
    let parent = receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = receipt.file_name().ok_or("receipt path must include a filename")?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = parent.join(filename);
    if target.starts_with(root) {
        return Err("receipt must be outside the dataset root".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let root = fs::canonicalize(&args.root)
        .map_err(|error| format!("canonicalize dataset root {}: {error}", args.root.display()))?;
    if !root.is_dir() {
        return Err("dataset root is not a directory".to_owned());
    }

    let manifest_bytes = fs::read(&args.manifest)
        .map_err(|error| format!("read manifest {}: {error}", args.manifest.display()))?;
    let manifest_sha256 = sha256_reader(manifest_bytes.as_slice())?;
    let samples = parse_manifest(&manifest_bytes)?;
    let protocol = samples[0].split_protocol;

    let mut source_hash_cache: BTreeMap<String, String> = BTreeMap::new();
    let mut source_records = BTreeSet::new();
    for sample in &samples {
        let actual = if let Some(value) = source_hash_cache.get(&sample.source_path) {
            value.clone()
        } else {
            let path = resolve_source(&root, &sample.source_path)?;
            let value = sha256_file_stable(&path)?;
            source_hash_cache.insert(sample.source_path.clone(), value.clone());
            value
        };
        if actual != sample.source_sha256 {
            return Err(format!(
                "source hash mismatch for {}: manifest {}, actual {}",
                sample.source_path, sample.source_sha256, actual
            ));
        }
        source_records.insert((
            sample.source_artifact_id.clone(),
            sample.source_path.clone(),
            actual,
        ));
    }

    let exact_hash_audit = audit_exact_source_hash_splits(&samples, protocol)?;
    let group_counts = audit_groups(&samples, protocol)?;
    let mut split_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut label_counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for sample in &samples {
        *split_counts.entry(sample.split.clone()).or_insert(0) += 1;
        *label_counts
            .entry(sample.split.clone())
            .or_default()
            .entry(sample.label.clone())
            .or_insert(0) += 1;
    }

    let overlap_fields = [
        "design_id",
        "base_design_family_id",
        "netlist_id",
        "trojan_family_id",
        "synthesis_config_id",
        "cell_library_id",
        "benchmark_source_id",
    ];
    let mut overlaps = BTreeMap::new();
    for field in overlap_fields {
        overlaps.extend(lineage_overlap_counts(&samples, field));
    }

    let receipt_path = receipt_path_outside_root(&root, &args.receipt)?;
    let receipt = serde_json::json!({
        "schema_version": "hardsec-design-lineage-audit-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "dataset_id": samples[0].dataset_id,
        "split_protocol": protocol.as_str(),
        "manifest_sha256": manifest_sha256,
        "sample_count": samples.len(),
        "split_sample_counts": split_counts,
        "split_label_counts": label_counts,
        "audited_group_sample_counts": group_counts,
        "cross_split_lineage_group_counts": overlaps,
        "cross_split_source_hash_count": exact_hash_audit.cross_split_source_hash_count,
        "reused_benign_control_artifact_count": exact_hash_audit.reused_benign_control_artifact_count,
        "reused_benign_control_artifacts": exact_hash_audit.reused_benign_control_artifacts,
        "verified_source_artifact_count": source_records.len(),
        "verified_source_file_count": source_hash_cache.len(),
        "verified_source_artifacts": source_records
            .iter()
            .map(|(artifact_id, path, sha256)| serde_json::json!({
                "source_artifact_id": artifact_id,
                "source_path": path,
                "sha256": sha256,
            }))
            .collect::<Vec<_>>(),
        "signature": null,
        "security_verdict": null,
        "lineage_metadata_status": "caller_declared_not_independently_inferred",
        "notes": [
            "Exact source file hashes and the declared split group were checked.",
            "source_path/source_sha256 bind the evaluated input artifact.",
            "Upstream parent RTL hashes belong in a separate lineage inventory.",
            "Design-lineage IDs are caller-declared metadata; ancestry is not inferred.",
            "Structural similarity and semantic netlist equivalence were not evaluated.",
            "A pass does not establish detector quality, Trojan absence, or design security.",
            "random_gate is diagnostic only; overlaps are reported, not rejected.",
            "leave_benchmark_source_out allows only identical benign paired-control reuse; every such digest is enumerated.",
            "benchmark_source_id is caller-declared cohort metadata and is not independently inferred or a model feature."
        ]
    });

    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize audit receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create new receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC-005 design-lineage audit: PASS");
    println!("protocol: {}", protocol.as_str());
    println!("samples: {}", samples.len());
    println!("verified source files: {}", source_hash_cache.len());
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-005 audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        root: PathBuf,
        manifest: PathBuf,
        receipt: PathBuf,
    }

    impl Fixture {
        fn new(protocol: &str, rows: &[(&str, &str, &str, &str)]) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let base = env::temp_dir().join(format!(
                "hardsec-design-audit-{}-{id}",
                std::process::id()
            ));
            let root = base.join("dataset");
            fs::create_dir_all(&root).unwrap();
            let manifest = base.join("manifest.tsv");
            let receipt = base.join("out").join("receipt.json");
            fs::create_dir_all(receipt.parent().unwrap()).unwrap();

            let mut text = format!("{}\n", HEADER.join("\t"));
            for (sample_id, split, family, trojan_family) in rows {
                let source_name = format!("design-{sample_id}.v");
                let source = root.join(&source_name);
                let source_bytes = format!("module fixture_{sample_id}; endmodule\n");
                fs::write(&source, source_bytes.as_bytes()).unwrap();
                let source_hash = sha256_file_stable(&source).unwrap();
                let source_artifact_id = format!("src-{sample_id}");
                let fields = vec![
                    "fixture-dataset".to_owned(),
                    source_artifact_id,
                    source_name,
                    source_hash,
                    (*sample_id).to_owned(),
                    "design-1".to_owned(),
                    (*family).to_owned(),
                    format!("netlist-{sample_id}"),
                    (*trojan_family).to_owned(),
                    "synth-v1".to_owned(),
                    "lib-v1".to_owned(),
                    format!("benchmark-{sample_id}"),
                    protocol.to_owned(),
                    (*split).to_owned(),
                    if trojan_family.is_empty() { "benign" } else { "trojan" }.to_owned(),
                ];
                text.push_str(&fields.join("\t"));
                text.push('\n');
            }
            fs::write(&manifest, text).unwrap();
            Self { root, manifest, receipt }
        }

        fn args(&self) -> Args {
            Args {
                manifest: self.manifest.clone(),
                root: self.root.clone(),
                receipt: self.receipt.clone(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(parent) = self.root.parent() {
                let _ = fs::remove_dir_all(parent);
            }
        }
    }

    #[test]
    fn accepts_valid_leave_one_family_out_manifest_and_writes_receipt() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["sample_count"], 3);
        assert_eq!(receipt["split_protocol"], "lofo_base_family");
    }

    #[test]
    fn rejects_base_family_leakage_across_splits() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-a", "trojan-c"),
            ],
        );
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("base_design_family_id=host-a"));
    }

    #[test]
    fn rejects_netlist_leakage_across_splits() {
        let fixture = Fixture::new(
            "loco_netlist",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let mut text = fs::read_to_string(&fixture.manifest).unwrap();
        text = text.replace("netlist-s3", "netlist-s1");
        fs::write(&fixture.manifest, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("netlist_id=netlist-s1"));
    }

    #[test]
    fn rejects_trojan_family_leakage() {
        let fixture = Fixture::new(
            "leave_trojan_family_out",
            &[
                ("s1", "train", "host-a", "flush"),
                ("s2", "validation", "host-b", "spectre"),
                ("s3", "test", "host-c", "flush"),
            ],
        );
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("trojan_family_id=flush"));
    }

    #[test]
    fn rejects_identical_source_bytes_crossing_group_splits() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let mut lines: Vec<String> = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let first: Vec<String> = lines[1].split('\t').map(str::to_owned).collect();
        let mut third: Vec<String> = lines[3].split('\t').map(str::to_owned).collect();
        third[2] = first[2].clone();
        third[3] = first[3].clone();
        lines[3] = third.join("\t");
        fs::write(&fixture.manifest, lines.join("\n") + "\n").unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("exact source artifact hash leakage"));
    }

    #[test]
    fn rejects_source_hash_mismatch() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        fs::write(fixture.root.join("design-s1.v"), b"tampered design\n").unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("source hash mismatch"));
    }

    #[test]
    fn rejects_path_traversal() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let text = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .replace("design-s1.v", "../design-s1.v");
        fs::write(&fixture.manifest, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("unsafe source_path"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_sources() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let real = fixture.root.join("design-s1.v");
        let link = fixture.root.join("linked-s1.v");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let text = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .replace("design-s1.v", "linked-s1.v");
        fs::write(&fixture.manifest, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("symlink source path"));
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("same", "train", "host-a", "trojan-a"),
                ("same", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("duplicate sample_id"));
    }

    #[test]
    fn rejects_missing_required_toolchain_identity() {
        let fixture = Fixture::new(
            "leave_toolchain_out",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let text = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .replace("synth-v1", "");
        fs::write(&fixture.manifest, text).unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("requires synthesis_config_id"));
    }

    #[test]
    fn refuses_to_overwrite_a_receipt() {
        let fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        fs::write(&fixture.receipt, b"do-not-overwrite").unwrap();
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("File exists"));
    }

    #[test]
    fn rejects_receipt_inside_dataset_root() {
        let mut fixture = Fixture::new(
            "lofo_base_family",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        fixture.receipt = fixture.root.join("receipt.json");
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("receipt must be outside"));
    }

    fn set_manifest_field(fixture: &Fixture, row_index: usize, column_index: usize, value: &str) {
        let mut lines: Vec<String> = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let mut fields: Vec<String> = lines[row_index]
            .split('\t')
            .map(str::to_owned)
            .collect();
        fields[column_index] = value.to_owned();
        lines[row_index] = fields.join("\t");
        fs::write(&fixture.manifest, lines.join("\n") + "\n").unwrap();
    }

    fn reuse_evaluated_artifact(fixture: &Fixture, source_row: usize, target_row: usize) {
        let mut lines: Vec<String> = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let source: Vec<String> = lines[source_row].split('\t').map(str::to_owned).collect();
        let mut target: Vec<String> = lines[target_row].split('\t').map(str::to_owned).collect();
        target[2] = source[2].clone();
        target[3] = source[3].clone();
        lines[target_row] = target.join("\t");
        fs::write(&fixture.manifest, lines.join("\n") + "\n").unwrap();
    }

    #[test]
    fn accepts_leave_one_benchmark_source_out_and_records_source_groups() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["split_protocol"], "leave_benchmark_source_out");
        assert_eq!(receipt["audited_group_sample_counts"]["train"], 1);
        assert_eq!(receipt["cross_split_lineage_group_counts"]["benchmark_source_id"], 0);
    }

    #[test]
    fn rejects_missing_benchmark_source_identity() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        set_manifest_field(&fixture, 1, 11, "");
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("requires benchmark_source_id"));
    }

    #[test]
    fn rejects_benchmark_source_leakage_across_splits() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        let train_source = "benchmark-s1";
        set_manifest_field(&fixture, 3, 11, train_source);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("benchmark_source_id=benchmark-s1"));
    }

    #[test]
    fn rejects_identical_trojan_artifact_across_source_splits() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        reuse_evaluated_artifact(&fixture, 1, 3);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("exact source artifact hash leakage"));
    }

    #[test]
    fn allows_and_reports_reused_benign_control_artifacts_for_source_holdout() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", ""),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", ""),
            ],
        );
        reuse_evaluated_artifact(&fixture, 1, 3);
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["reused_benign_control_artifact_count"], 1);
        assert_eq!(
            receipt["reused_benign_control_artifacts"][0]["label"],
            "benign"
        );
        assert_eq!(
            receipt["reused_benign_control_artifacts"][0]["benchmark_source_ids"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn rejects_identical_content_with_conflicting_labels() {
        let fixture = Fixture::new(
            "leave_benchmark_source_out",
            &[
                ("s1", "train", "host-a", ""),
                ("s2", "validation", "host-b", "trojan-b"),
                ("s3", "test", "host-c", "trojan-c"),
            ],
        );
        reuse_evaluated_artifact(&fixture, 1, 3);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("inconsistent labels"));
    }

    #[test]
    fn random_gate_is_explicitly_diagnostic_and_reports_overlap() {
        let fixture = Fixture::new(
            "random_gate",
            &[
                ("s1", "train", "host-a", "trojan-a"),
                ("s2", "validation", "host-a", "trojan-b"),
                ("s3", "test", "host-a", "trojan-c"),
            ],
        );
        let mut lines: Vec<String> = fs::read_to_string(&fixture.manifest)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let first: Vec<String> = lines[1].split('\t').map(str::to_owned).collect();
        for row_index in [2, 3] {
            let mut row: Vec<String> = lines[row_index].split('\t').map(str::to_owned).collect();
            row[2] = first[2].clone();
            row[3] = first[3].clone();
            lines[row_index] = row.join("\t");
        }
        fs::write(&fixture.manifest, lines.join("\n") + "\n").unwrap();
        run(fixture.args()).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["cross_split_lineage_group_counts"]["base_design_family_id"], 1);
        assert_eq!(receipt["cross_split_source_hash_count"], 1);
    }
}
