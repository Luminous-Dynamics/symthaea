//! HARDSEC-001 split-manifest audit.
//!
//! This tool audits identity and split integrity before model training. It does
//! not classify traces, verify vulnerabilities, or establish that a device is
//! secure. Run with:
//!
//! cargo run --example hardsec_split_audit --features hardsec-audit -- \
//!   --manifest /path/to/splits.tsv --root /path/to/dataset
//!
//! The TSV is deliberately strict and has one row per sample/window. Repeated
//! rows may reference the same source artifact, but all rows from that artifact
//! must remain in one split.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const HEADER: [&str; 16] = [
    "dataset_id",
    "domain",
    "source_artifact_id",
    "source_path",
    "source_sha256",
    "sample_id",
    "program_id",
    "execution_id",
    "attack_family_id",
    "device_config_id",
    "capture_config_id",
    "counter_set_id",
    "security_outcome",
    "split_protocol",
    "split",
    "label",
];

const DOMAINS: [&str; 4] = [
    "physical_hpc",
    "gem5_sim",
    "rtl_netlist",
    "physical_em",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sample {
    dataset_id: String,
    domain: String,
    artifact_id: String,
    source_path: String,
    source_sha256: String,
    sample_id: String,
    program_id: String,
    execution_id: String,
    attack_family_id: String,
    device_config_id: String,
    capture_config_id: String,
    counter_set_id: String,
    security_outcome: String,
    split_protocol: String,
    split: String,
    label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactBinding {
    source_path: String,
    source_sha256: String,
    device_config_id: String,
    capture_config_id: String,
    counter_set_id: String,
    program_id: String,
    execution_id: String,
    attack_family_id: String,
    split: String,
    label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AuditSummary {
    dataset_id: String,
    domain: String,
    split_protocol: String,
    samples: usize,
    artifacts: usize,
    train: usize,
    validation: usize,
    test: usize,
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some((manifest, root))) => match audit_manifest(&manifest, &root) {
            Ok(summary) => {
                println!(
                    "HARDSEC_AUDIT_PASS dataset_id={} domain={} protocol={} samples={} artifacts={} train={} validation={} test={}",
                    summary.dataset_id,
                    summary.domain,
                    summary.split_protocol,
                    summary.samples,
                    summary.artifacts,
                    summary.train,
                    summary.validation,
                    summary.test
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("HARDSEC_AUDIT_FAIL: {error}");
                ExitCode::from(2)
            }
        },
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC_AUDIT_FAIL: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_args() -> Result<Option<(PathBuf, PathBuf)>, String> {
    let mut args = env::args().skip(1);
    let mut manifest = None;
    let mut root = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "HARDSEC-001 split-manifest audit\n\n\
                     Usage: hardsec_split_audit --manifest PATH --root DATASET_ROOT\n\n\
                     Requires a strict 15-column TSV; see docs/research/HARDSEC_001_HARDWARE_ASSURANCE_BENCHMARK.md."
                );
                return Ok(None);
            }
            "--manifest" => {
                manifest = Some(PathBuf::from(
                    args.next().ok_or("--manifest requires a path")?,
                ));
            }
            "--root" => {
                root = Some(PathBuf::from(args.next().ok_or("--root requires a path")?));
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }

    let manifest = manifest.ok_or("--manifest is required")?;
    let root = root.ok_or("--root is required")?;
    Ok(Some((manifest, root)))
}

fn audit_manifest(manifest_path: &Path, dataset_root: &Path) -> Result<AuditSummary, String> {
    let root = fs::canonicalize(dataset_root).map_err(|error| {
        format!(
            "cannot resolve dataset root {}: {error}",
            dataset_root.display()
        )
    })?;
    if !fs::metadata(&root)
        .map_err(|error| format!("cannot inspect dataset root {}: {error}", root.display()))?
        .is_dir()
    {
        return Err(format!("dataset root is not a directory: {}", root.display()));
    }

    let text = fs::read_to_string(manifest_path).map_err(|error| {
        format!(
            "cannot read UTF-8 manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let mut lines = text.lines();
    let first = lines.next().ok_or("manifest is empty")?;
    let actual_header: Vec<&str> = first.trim_end_matches('\r').split('\t').collect();
    if actual_header.as_slice() != &HEADER[..] {
        return Err(format!(
            "header mismatch; expected exactly {} tab-separated columns in the documented order",
            HEADER.len()
        ));
    }

    let mut samples = Vec::new();
    for (zero_index, line) in lines.enumerate() {
        let line_number = zero_index + 2;
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        if line.contains('\0') {
            return Err(format!("line {line_number}: NUL byte is not allowed"));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != HEADER.len() {
            return Err(format!(
                "line {line_number}: expected {} columns, found {}",
                HEADER.len(),
                fields.len()
            ));
        }
        if fields.iter().any(|field| field.trim() != *field) {
            return Err(format!(
                "line {line_number}: leading/trailing whitespace is not allowed in fields"
            ));
        }
        let values: Vec<String> = fields.iter().map(|field| (*field).to_owned()).collect();
        samples.push(Sample {
            dataset_id: values[0].clone(),
            domain: values[1].clone(),
            artifact_id: values[2].clone(),
            source_path: values[3].clone(),
            source_sha256: values[4].clone(),
            sample_id: values[5].clone(),
            program_id: values[6].clone(),
            execution_id: values[7].clone(),
            attack_family_id: values[8].clone(),
            device_config_id: values[9].clone(),
            capture_config_id: values[10].clone(),
            counter_set_id: values[11].clone(),
            security_outcome: values[12].clone(),
            split_protocol: values[13].clone(),
            split: values[14].clone(),
            label: values[15].clone(),
        });
    }

    if samples.is_empty() {
        return Err("manifest has a header but no sample rows".into());
    }

    let dataset_id = samples[0].dataset_id.clone();
    let domain = samples[0].domain.clone();
    let split_protocol = samples[0].split_protocol.clone();
    if is_missing(&dataset_id) {
        return Err("dataset_id must not be empty or NA".into());
    }
    if !DOMAINS.contains(&domain.as_str()) {
        return Err(format!(
            "unsupported domain {domain:?}; expected one of {}",
            DOMAINS.join(", ")
        ));
    }
    if ![
        "program_holdout",
        "attack_family_holdout",
        "execution_holdout",
        "device_config_holdout",
        "capture_config_holdout",
    ]
    .contains(&split_protocol.as_str())
    {
        return Err(format!("unsupported split_protocol {split_protocol:?}"));
    }

    let mut sample_ids = BTreeSet::new();
    let mut groups: BTreeMap<String, String> = BTreeMap::new();
    let mut artifact_bindings: BTreeMap<String, ArtifactBinding> = BTreeMap::new();
    let mut verified_source_hashes: BTreeMap<String, String> = BTreeMap::new();
    let mut digest_to_artifact: BTreeMap<String, String> = BTreeMap::new();
    let mut split_counts = BTreeMap::from([
        ("train".to_owned(), 0usize),
        ("validation".to_owned(), 0usize),
        ("test".to_owned(), 0usize),
    ]);
    let mut labels_by_split: BTreeMap<String, BTreeSet<String>> = BTreeMap::from([
        ("train".to_owned(), BTreeSet::new()),
        ("validation".to_owned(), BTreeSet::new()),
        ("test".to_owned(), BTreeSet::new()),
    ]);

    for (index, sample) in samples.iter().enumerate() {
        let line_number = index + 2;
        if sample.dataset_id != dataset_id
            || sample.domain != domain
            || sample.split_protocol != split_protocol
        {
            return Err(format!(
                "line {line_number}: a manifest must contain exactly one dataset_id, domain and split_protocol"
            ));
        }
        for (name, value) in [
            ("source_artifact_id", sample.artifact_id.as_str()),
            ("source_path", sample.source_path.as_str()),
            ("sample_id", sample.sample_id.as_str()),
        ] {
            if is_missing(value) {
                return Err(format!("line {line_number}: {name} is required"));
            }
        }
        if !sample_ids.insert(sample.sample_id.clone()) {
            return Err(format!(
                "line {line_number}: duplicate sample_id {:?}",
                sample.sample_id
            ));
        }
        if !is_sha256(&sample.source_sha256) {
            return Err(format!(
                "line {line_number}: source_sha256 must be 64 lowercase hexadecimal characters"
            ));
        }
        if !["train", "validation", "test"].contains(&sample.split.as_str()) {
            return Err(format!(
                "line {line_number}: invalid split {:?}",
                sample.split
            ));
        }
        if !["benign", "attack"].contains(&sample.label.as_str()) {
            return Err(format!(
                "line {line_number}: label must be benign or attack"
            ));
        }
        let valid_outcome = match sample.label.as_str() {
            "benign" => sample.security_outcome == "not_applicable",
            "attack" => [
                "success_observed",
                "failed_observed",
                "unknown",
            ]
            .contains(&sample.security_outcome.as_str()),
            _ => false,
        };
        if !valid_outcome {
            return Err(format!(
                "line {line_number}: security_outcome is incompatible with the workload label"
            ));
        }
        *split_counts
            .get_mut(&sample.split)
            .expect("split was validated above") += 1;
        labels_by_split
            .get_mut(&sample.split)
            .expect("split was validated above")
            .insert(sample.label.clone());

        let relative = Path::new(&sample.source_path);
        if relative.is_absolute()
            || relative.components().any(|component| {
                !matches!(component, Component::Normal(_))
            })
        {
            return Err(format!(
                "line {line_number}: source_path must be a clean relative path without . or .. components"
            ));
        }
        verify_regular_file_without_symlinks(&root, relative, line_number)?;
        let resolved = root.join(relative);
        let actual_digest = if let Some(cached) = verified_source_hashes.get(&sample.source_path) {
            cached.clone()
        } else {
            let digest = sha256_file(&resolved)
                .map_err(|error| format!("line {line_number}: cannot hash source: {error}"))?;
            verified_source_hashes.insert(sample.source_path.clone(), digest.clone());
            digest
        };
        if actual_digest != sample.source_sha256 {
            return Err(format!(
                "line {line_number}: SHA-256 mismatch for {}: expected {}, got {}",
                sample.source_path, sample.source_sha256, actual_digest
            ));
        }

        let binding = ArtifactBinding {
            source_path: sample.source_path.clone(),
            source_sha256: sample.source_sha256.clone(),
            device_config_id: sample.device_config_id.clone(),
            capture_config_id: sample.capture_config_id.clone(),
            counter_set_id: sample.counter_set_id.clone(),
            program_id: sample.program_id.clone(),
            execution_id: sample.execution_id.clone(),
            attack_family_id: sample.attack_family_id.clone(),
            split: sample.split.clone(),
            label: sample.label.clone(),
        };
        if let Some(previous) = artifact_bindings.get(&sample.artifact_id) {
            if previous != &binding {
                return Err(format!(
                    "line {line_number}: source artifact {:?} has inconsistent path, digest, or split",
                    sample.artifact_id
                ));
            }
        } else {
            artifact_bindings.insert(sample.artifact_id.clone(), binding);
        }

        if let Some(previous_artifact) = digest_to_artifact.get(&sample.source_sha256) {
            if previous_artifact != &sample.artifact_id {
                return Err(format!(
                    "line {line_number}: exact duplicate artifact digest appears under distinct IDs {:?} and {:?}",
                    previous_artifact, sample.artifact_id
                ));
            }
        } else {
            digest_to_artifact.insert(sample.source_sha256.clone(), sample.artifact_id.clone());
        }

        let group_id = held_out_group(sample, &split_protocol, line_number)?;
        if let Some(previous_split) = groups.insert(group_id.clone(), sample.split.clone()) {
            if previous_split != sample.split {
                return Err(format!(
                    "line {line_number}: holdout group {group_id:?} crosses splits ({previous_split} and {})",
                    sample.split
                ));
            }
        }
    }

    for split in ["train", "validation", "test"] {
        if split_counts.get(split).copied().unwrap_or_default() == 0 {
            return Err(format!("split {split:?} is empty"));
        }
    }
    for split in ["validation", "test"] {
        let labels = labels_by_split.get(split).expect("split map is complete");
        if !labels.contains("benign") || !labels.contains("attack") {
            return Err(format!(
                "split {split:?} must contain both benign and attack labels to score binary detection"
            ));
        }
    }

    Ok(AuditSummary {
        dataset_id,
        domain,
        split_protocol,
        samples: samples.len(),
        artifacts: artifact_bindings.len(),
        train: split_counts["train"],
        validation: split_counts["validation"],
        test: split_counts["test"],
    })
}

fn held_out_group(sample: &Sample, protocol: &str, line_number: usize) -> Result<String, String> {
    let (kind, value) = match protocol {
        "program_holdout" => ("program", sample.program_id.as_str()),
        "attack_family_holdout" if sample.label == "attack" => {
            ("attack_family", sample.attack_family_id.as_str())
        }
        "attack_family_holdout" => ("benign_program", sample.program_id.as_str()),
        "execution_holdout" => ("execution", sample.execution_id.as_str()),
        "device_config_holdout" => ("device_config", sample.device_config_id.as_str()),
        "capture_config_holdout" => ("capture_config", sample.capture_config_id.as_str()),
        _ => return Err(format!("line {line_number}: unsupported split protocol")),
    };
    if is_missing(value) {
        return Err(format!(
            "line {line_number}: {kind} grouping key is missing for protocol {protocol:?}"
        ));
    }
    Ok(format!("{kind}:{value}"))
}

fn verify_regular_file_without_symlinks(
    root: &Path,
    relative: &Path,
    line_number: usize,
) -> Result<(), String> {
    let mut cursor = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(format!("line {line_number}: invalid source path component"));
        };
        cursor.push(part);
        let metadata = fs::symlink_metadata(&cursor).map_err(|error| {
            format!("line {line_number}: cannot inspect source path {}: {error}", cursor.display())
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "line {line_number}: symlinks are not allowed in source path {}",
                relative.display()
            ));
        }
        let last = index + 1 == components.len();
        if last && !metadata.is_file() {
            return Err(format!(
                "line {line_number}: source path is not a regular file: {}",
                relative.display()
            ));
        }
        if !last && !metadata.is_dir() {
            return Err(format!(
                "line {line_number}: non-directory path component in {}",
                relative.display()
            ));
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_missing(value: &str) -> bool {
    value.is_empty() || value.eq_ignore_ascii_case("na") || value == "-"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        rows: Vec<Vec<String>>,
    }

    impl Fixture {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before Unix epoch")
                .as_nanos();
            let root = env::temp_dir().join(format!(
                "hardsec-audit-{}-{}-{}",
                std::process::id(),
                nonce,
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).expect("create fixture root");
            Self { root, rows: Vec::new() }
        }

        fn add_row(
            &mut self,
            sample_id: &str,
            artifact_name: &str,
            program_id: &str,
            execution_id: &str,
            attack_family_id: &str,
            device_config_id: &str,
            split_protocol: &str,
            split: &str,
            label: &str,
        ) {
            self.add_row_with_capture(
                sample_id,
                artifact_name,
                program_id,
                execution_id,
                attack_family_id,
                device_config_id,
                "capture-config-default",
                "single-counter",
                split_protocol,
                split,
                label,
            );
        }

        fn add_row_with_capture(
            &mut self,
            sample_id: &str,
            artifact_name: &str,
            program_id: &str,
            execution_id: &str,
            attack_family_id: &str,
            device_config_id: &str,
            capture_config_id: &str,
            counter_set_id: &str,
            split_protocol: &str,
            split: &str,
            label: &str,
        ) {
            let source_path = format!("{artifact_name}.trace");
            fs::write(self.root.join(&source_path), format!("fixture:{artifact_name}"))
                .expect("write source fixture");
            let source_sha256 =
                sha256_file(&self.root.join(&source_path)).expect("hash fixture");
            self.rows.push(vec![
                "dataset-test".into(),
                "physical_hpc".into(),
                artifact_name.into(),
                source_path,
                source_sha256,
                sample_id.into(),
                program_id.into(),
                execution_id.into(),
                attack_family_id.into(),
                device_config_id.into(),
                capture_config_id.into(),
                counter_set_id.into(),
                if label == "benign" { "not_applicable" } else { "unknown" }.into(),
                split_protocol.into(),
                split.into(),
                label.into(),
            ]);
        }

        fn write_manifest(&self) -> PathBuf {
            let path = self.root.join("splits.tsv");
            let mut text = HEADER.join("\t");
            text.push('\n');
            for row in &self.rows {
                text.push_str(&row.join("\t"));
                text.push('\n');
            }
            fs::write(&path, text).expect("write manifest");
            path
        }

        fn audit(&self) -> Result<AuditSummary, String> {
            audit_manifest(&self.write_manifest(), &self.root)
        }
    }

    fn add_balanced_program_holdout(fixture: &mut Fixture) {
        fixture.add_row(
            "s-train-attack",
            "a",
            "program-a",
            "run-a",
            "family-a",
            "board-a",
            "program_holdout",
            "train",
            "attack",
        );
        fixture.add_row(
            "s-train-benign",
            "b",
            "program-b",
            "run-b",
            "NA",
            "board-a",
            "program_holdout",
            "train",
            "benign",
        );
        fixture.add_row(
            "s-val-attack",
            "c",
            "program-c",
            "run-c",
            "family-c",
            "board-a",
            "program_holdout",
            "validation",
            "attack",
        );
        fixture.add_row(
            "s-val-benign",
            "d",
            "program-d",
            "run-d",
            "NA",
            "board-a",
            "program_holdout",
            "validation",
            "benign",
        );
        fixture.add_row(
            "s-test-attack",
            "e",
            "program-e",
            "run-e",
            "family-e",
            "board-a",
            "program_holdout",
            "test",
            "attack",
        );
        fixture.add_row(
            "s-test-benign",
            "f",
            "program-f",
            "run-f",
            "NA",
            "board-a",
            "program_holdout",
            "test",
            "benign",
        );
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn sha256_uses_standard_sha256() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn valid_program_holdout_passes() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        let result = fixture.audit().expect("valid manifest should pass");
        assert_eq!(result.samples, 6);
        assert_eq!(result.artifacts, 6);
    }

    #[test]
    fn repeated_windows_from_one_artifact_are_allowed_within_one_split() {
        let mut fixture = Fixture::new();
        fixture.add_row(
            "s-train-1", "a", "program-a", "run-a", "family-a", "board-a",
            "program_holdout", "train", "attack",
        );
        fixture.add_row(
            "s-train-2", "a", "program-a", "run-a", "family-a", "board-a",
            "program_holdout", "train", "attack",
        );
        fixture.add_row(
            "s-train-benign", "b", "program-b", "run-b", "NA", "board-a",
            "program_holdout", "train", "benign",
        );
        fixture.add_row(
            "s-val-attack", "c", "program-c", "run-c", "family-c", "board-a",
            "program_holdout", "validation", "attack",
        );
        fixture.add_row(
            "s-val-benign", "d", "program-d", "run-d", "NA", "board-a",
            "program_holdout", "validation", "benign",
        );
        fixture.add_row(
            "s-test-attack", "e", "program-e", "run-e", "family-e", "board-a",
            "program_holdout", "test", "attack",
        );
        fixture.add_row(
            "s-test-benign", "f", "program-f", "run-f", "NA", "board-a",
            "program_holdout", "test", "benign",
        );
        let summary = fixture.audit().expect("repeated windows should pass");
        assert_eq!(summary.samples, 7);
        assert_eq!(summary.artifacts, 6);
    }

    #[test]
    fn same_program_cannot_cross_splits() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        fixture.rows[4][6] = "program-a".into();
        assert!(fixture.audit().unwrap_err().contains("crosses splits"));
    }

    #[test]
    fn attack_family_holdout_keeps_benign_programs_independently_grouped() {
        let mut fixture = Fixture::new();
        fixture.add_row(
            "s-train-attack", "a", "program-a", "run-a", "family-a", "board-a",
            "attack_family_holdout", "train", "attack",
        );
        fixture.add_row(
            "s-val-attack", "b", "program-b", "run-b", "family-b", "board-a",
            "attack_family_holdout", "validation", "attack",
        );
        fixture.add_row(
            "s-test-attack", "c", "program-c", "run-c", "family-c", "board-a",
            "attack_family_holdout", "test", "attack",
        );
        fixture.add_row(
            "s-train-benign", "d", "benign-a", "run-d", "NA", "board-a",
            "attack_family_holdout", "train", "benign",
        );
        fixture.add_row(
            "s-val-benign", "e", "benign-b", "run-e", "NA", "board-a",
            "attack_family_holdout", "validation", "benign",
        );
        fixture.add_row(
            "s-test-benign", "f", "benign-c", "run-f", "NA", "board-a",
            "attack_family_holdout", "test", "benign",
        );
        assert_eq!(fixture.audit().expect("valid family holdout").samples, 6);
    }

    #[test]
    fn capture_configuration_holdout_rejects_reuse_across_splits() {
        let mut fixture = Fixture::new();
        fixture.add_row_with_capture(
            "s-train-attack", "a", "program-a", "run-a", "family-a", "board-a",
            "capture-a", "counter-set-a", "capture_config_holdout", "train", "attack",
        );
        fixture.add_row_with_capture(
            "s-train-benign", "b", "benign-a", "run-b", "NA", "board-a",
            "capture-a", "counter-set-a", "capture_config_holdout", "train", "benign",
        );
        fixture.add_row_with_capture(
            "s-val-attack", "c", "program-c", "run-c", "family-c", "board-a",
            "capture-b", "counter-set-a", "capture_config_holdout", "validation", "attack",
        );
        fixture.add_row_with_capture(
            "s-val-benign", "d", "benign-b", "run-d", "NA", "board-a",
            "capture-b", "counter-set-a", "capture_config_holdout", "validation", "benign",
        );
        fixture.add_row_with_capture(
            "s-test-attack", "e", "program-e", "run-e", "family-e", "board-a",
            "capture-a", "counter-set-a", "capture_config_holdout", "test", "attack",
        );
        fixture.add_row_with_capture(
            "s-test-benign", "f", "benign-c", "run-f", "NA", "board-a",
            "capture-a", "counter-set-a", "capture_config_holdout", "test", "benign",
        );
        assert!(fixture.audit().unwrap_err().contains("crosses splits"));
    }

    #[test]
    fn one_source_artifact_cannot_be_assigned_to_two_splits() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        for field in [2usize, 3, 4, 6, 7, 8, 9, 10, 11] {
            fixture.rows[4][field] = fixture.rows[0][field].clone();
        }
        assert!(fixture.audit().unwrap_err().contains("inconsistent path, digest, or split"));
    }

    #[cfg(unix)]
    #[test]
    fn source_symlinks_are_rejected() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        let manifest = fixture.write_manifest();
        let target = fixture.root.join("target.trace");
        fs::write(&target, "fixture:a").unwrap();
        fs::remove_file(fixture.root.join("a.trace")).unwrap();
        std::os::unix::fs::symlink(&target, fixture.root.join("a.trace")).unwrap();
        assert!(audit_manifest(&manifest, &fixture.root)
            .unwrap_err()
            .contains("symlinks are not allowed"));
    }

    #[test]
    fn altered_source_bytes_fail_hash_check() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        let _ = fixture.write_manifest();
        fs::write(fixture.root.join("a.trace"), "tampered-after-freeze").unwrap();
        assert!(fixture.audit().unwrap_err().contains("SHA-256 mismatch"));
    }

    #[test]
    fn path_traversal_fails_closed() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        let path = fixture.write_manifest();
        let mut text = fs::read_to_string(&path).unwrap();
        text = text.replace("a.trace", "../a.trace");
        fs::write(&path, text).unwrap();
        assert!(fixture.audit().unwrap_err().contains("clean relative path"));
    }

    #[test]
    fn duplicate_sample_ids_are_rejected() {
        let mut fixture = Fixture::new();
        add_balanced_program_holdout(&mut fixture);
        fixture.rows[5][5] = fixture.rows[0][5].clone();
        assert!(fixture.audit().unwrap_err().contains("duplicate sample_id"));
    }

}
