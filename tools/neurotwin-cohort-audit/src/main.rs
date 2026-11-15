// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Privacy-minimizing BIDS cohort metadata preflight.
//!
//! This tool does not open scan contents. It reads dataset_description.json,
//! optional participants.tsv identifiers, and directory/file metadata only.
//! Its JSON receipt contains aggregate counts and finding codes, never subject
//! labels, participant-table values, scan paths, or file names.

use csv::ReaderBuilder;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MODALITIES: &[&str] = &[
    "anat", "beh", "dwi", "eeg", "fmap", "func", "ieeg", "meg", "micr", "motion", "nirs", "perf",
    "pet", "physio", "stim",
];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct Finding {
    code: &'static str,
    severity: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CohortReceipt {
    schema_version: &'static str,
    status: &'static str,
    bids_version: String,
    dataset_type: String,
    subject_directory_count: usize,
    participants_tsv_present: bool,
    participant_table_count: Option<usize>,
    participant_ids_matched_to_directories: Option<usize>,
    session_directory_count: usize,
    modality_file_counts: BTreeMap<String, u64>,
    findings: Vec<Finding>,
    limitations: Vec<&'static str>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("neurotwin-cohort-audit: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let Some(root) = args.next() else {
        eprintln!("Usage: neurotwin-cohort-audit <bids-dataset-root>");
        std::process::exit(64);
    };
    if args.next().is_some() {
        return Err("expected exactly one dataset-root argument".into());
    }

    let receipt = audit_dataset(Path::new(&root))?;
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, &receipt)?;
    writeln!(lock)?;
    if receipt.status == "BLOCKED" {
        std::process::exit(2);
    }
    Ok(())
}

fn audit_dataset(root: &Path) -> Result<CohortReceipt, Box<dyn Error>> {
    if !root.is_dir() {
        return Err("dataset root is not a directory".into());
    }

    let description_path = root.join("dataset_description.json");
    let description_bytes = fs::read(&description_path)
        .map_err(|_| "required dataset_description.json is missing or unreadable")?;
    let description: Value = serde_json::from_slice(&description_bytes)
        .map_err(|_| "dataset_description.json is not valid JSON")?;

    // Name is checked but intentionally never copied into the receipt: a local
    // cohort name may itself reveal sensitive study information.
    description
        .get("Name")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or("dataset_description.json must contain a non-empty Name string")?;
    let bids_version = description
        .get("BIDSVersion")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or("dataset_description.json must contain a non-empty BIDSVersion string")?
        .to_owned();

    let mut findings = Vec::new();
    let dataset_type = match description.get("DatasetType").and_then(Value::as_str) {
        Some("raw") => "raw".to_owned(),
        Some("derivative") => {
            if !description
                .get("GeneratedBy")
                .and_then(Value::as_array)
                .is_some_and(|items| !items.is_empty())
            {
                findings.push(Finding {
                    code: "DERIVATIVE_MISSING_GENERATED_BY",
                    severity: "error",
                });
            }
            "derivative".to_owned()
        }
        Some(_) => {
            findings.push(Finding {
                code: "INVALID_DATASET_TYPE",
                severity: "error",
            });
            "invalid".to_owned()
        }
        None => {
            findings.push(Finding {
                code: "DATASET_TYPE_OMITTED_DEFAULTED_RAW",
                severity: "info",
            });
            "raw (defaulted)".to_owned()
        }
    };

    let mut subject_dirs = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?; // Does not follow symlinks.
        if file_type.is_dir() {
            let label = entry.file_name().to_string_lossy().into_owned();
            if label.starts_with("sub-") {
                if !valid_bids_label(&label, "sub-") {
                    findings.push(Finding {
                        code: "INVALID_SUBJECT_DIRECTORY_LABEL",
                        severity: "error",
                    });
                }
                subject_dirs.insert(label);
            }
        }
    }
    if subject_dirs.is_empty() {
        findings.push(Finding {
            code: "NO_SUBJECT_DIRECTORIES",
            severity: "error",
        });
    }

    let participants_path = root.join("participants.tsv");
    let participants_tsv_present = participants_path.is_file();
    let mut participant_table_count = None;
    let mut participant_ids_matched_to_directories = None;
    if participants_tsv_present {
        let ids = read_participant_ids(&participants_path)?;
        let matched = subject_dirs.intersection(&ids).count();
        if !subject_dirs.is_subset(&ids) {
            findings.push(Finding {
                code: "PARTICIPANTS_TABLE_MISSING_SUBJECT_DIRECTORY",
                severity: "error",
            });
        }
        participant_table_count = Some(ids.len());
        participant_ids_matched_to_directories = Some(matched);
    } else {
        // BIDS 1.11.2 marks participants.tsv as recommended, not mandatory.
        findings.push(Finding {
            code: "PARTICIPANTS_TSV_ABSENT",
            severity: "warning",
        });
    }

    let mut session_directory_count = 0usize;
    let mut modality_file_counts = BTreeMap::<String, u64>::new();
    for subject_dir in &subject_dirs {
        let subject_path = root.join(subject_dir);
        for entry in fs::read_dir(&subject_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && entry.file_name().to_string_lossy().starts_with("ses-")
            {
                session_directory_count += 1;
            }
        }
        walk_for_modality_counts(&subject_path, None, &mut modality_file_counts)?;
    }

    findings.sort_by_key(|finding| (finding.severity, finding.code));
    let blocked = findings.iter().any(|finding| finding.severity == "error");
    let review = findings.iter().any(|finding| finding.severity == "warning");

    Ok(CohortReceipt {
        schema_version: "neurotwin-cohort-receipt-v0.1",
        status: if blocked {
            "BLOCKED"
        } else if review {
            "REVIEW"
        } else {
            "PRELIMINARY_PASS"
        },
        bids_version,
        dataset_type,
        subject_directory_count: subject_dirs.len(),
        participants_tsv_present,
        participant_table_count,
        participant_ids_matched_to_directories,
        session_directory_count,
        modality_file_counts,
        findings,
        limitations: vec![
            "Metadata preflight only; this is not the official BIDS Validator.",
            "No scan contents, clinical values, participant labels, filenames, or paths are included in the receipt.",
            "This tool does not validate consent, ethics approval, data-use rights, task labels, image quality, or clinical endpoints.",
            "A preliminary pass does not authorize model training or clinical use.",
        ],
    })
}

fn valid_bids_label(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(|label| {
        !label.is_empty() && label.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
}

fn read_participant_ids(path: &Path) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();
    let index = headers
        .iter()
        .position(|header| header == "participant_id")
        .ok_or("participants.tsv is missing the required participant_id column")?;
    let mut ids = BTreeSet::new();
    for row in reader.records() {
        let row = row?;
        let id = row
            .get(index)
            .filter(|value| valid_bids_label(value, "sub-"))
            .ok_or("participants.tsv contains an invalid participant_id value")?;
        if !ids.insert(id.to_owned()) {
            return Err("participants.tsv contains duplicate participant_id values".into());
        }
    }
    Ok(ids)
}

fn walk_for_modality_counts(
    directory: &Path,
    current_modality: Option<&str>,
    counts: &mut BTreeMap<String, u64>,
) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?; // Never recurse through symlinks.
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let modality = if MODALITIES.contains(&name.as_str()) {
                Some(name.as_str())
            } else {
                current_modality
            };
            walk_for_modality_counts(&entry.path(), modality, counts)?;
        } else if kind.is_file() {
            if let Some(modality) = current_modality {
                *counts.entry(modality.to_owned()).or_default() += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before Unix epoch")
                .as_nanos();
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neurotwin-cohort-audit-{}-{nonce}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_description(root: &Path, extra: &str) {
        let json = format!(r#"{{"Name":"Synthetic Test Cohort","BIDSVersion":"1.11.2"{extra}}}"#);
        fs::write(root.join("dataset_description.json"), json).expect("write description");
    }

    #[test]
    fn accepts_minimal_dataset_and_reports_only_aggregate_counts() {
        let dir = TestDir::new();
        write_description(dir.path(), ",\"DatasetType\":\"raw\"");
        fs::create_dir_all(dir.path().join("sub-01/func")).unwrap();
        fs::write(
            dir.path().join("sub-01/func/sub-01_task-rest_bold.nii.gz"),
            b"not opened",
        )
        .unwrap();
        fs::write(
            dir.path().join("participants.tsv"),
            "participant_id\tage\nsub-01\t42\n",
        )
        .unwrap();

        let receipt = audit_dataset(dir.path()).unwrap();
        assert_eq!(receipt.status, "PRELIMINARY_PASS");
        assert_eq!(receipt.subject_directory_count, 1);
        assert_eq!(receipt.participant_table_count, Some(1));
        assert_eq!(receipt.modality_file_counts.get("func"), Some(&1));
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains("sub-01"));
        assert!(!json.contains("42"));
        assert!(!json.contains("task-rest"));
    }

    #[test]
    fn participants_tsv_is_recommended_not_required() {
        let dir = TestDir::new();
        write_description(dir.path(), "");
        fs::create_dir_all(dir.path().join("sub-A")).unwrap();
        let receipt = audit_dataset(dir.path()).unwrap();
        assert_eq!(receipt.status, "REVIEW");
        assert!(!receipt.participants_tsv_present);
        assert!(
            receipt
                .findings
                .iter()
                .any(|f| f.code == "PARTICIPANTS_TSV_ABSENT")
        );
    }

    #[test]
    fn blocks_subject_directory_missing_from_participant_table() {
        let dir = TestDir::new();
        write_description(dir.path(), "");
        fs::create_dir_all(dir.path().join("sub-01")).unwrap();
        fs::write(
            dir.path().join("participants.tsv"),
            "participant_id\nsub-02\n",
        )
        .unwrap();
        let receipt = audit_dataset(dir.path()).unwrap();
        assert_eq!(receipt.status, "BLOCKED");
        assert!(
            receipt
                .findings
                .iter()
                .any(|f| f.code == "PARTICIPANTS_TABLE_MISSING_SUBJECT_DIRECTORY")
        );
    }

    #[test]
    fn blocks_derivative_without_generated_by() {
        let dir = TestDir::new();
        write_description(dir.path(), ",\"DatasetType\":\"derivative\"");
        fs::create_dir_all(dir.path().join("sub-01")).unwrap();
        let receipt = audit_dataset(dir.path()).unwrap();
        assert_eq!(receipt.status, "BLOCKED");
        assert!(
            receipt
                .findings
                .iter()
                .any(|f| f.code == "DERIVATIVE_MISSING_GENERATED_BY")
        );
    }

    #[test]
    fn rejects_duplicate_participant_ids() {
        let dir = TestDir::new();
        let path = dir.path().join("participants.tsv");
        fs::write(&path, "participant_id\nsub-01\nsub-01\n").unwrap();
        assert!(read_participant_ids(&path).is_err());
    }

    #[test]
    fn rejects_missing_required_dataset_description() {
        let dir = TestDir::new();
        assert!(audit_dataset(dir.path()).is_err());
    }
}
