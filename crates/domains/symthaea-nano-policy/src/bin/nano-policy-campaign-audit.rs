//! Reconcile all per-seed simulator-call ledgers in a frozen campaign manifest.
//!
//! Exit status:
//! - 0: each policy arm contains exactly one verified ledger for every expected seed.
//! - 3: the manifest is valid but one or more seed, identity, digest, or call-budget gates failed.
//! - 2: malformed input, unsafe path, or invalid campaign contract.
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use symthaea_nano_policy::sim_campaign::{
    audit_campaign_json, parse_campaign_manifest, MAX_CAMPAIGN_MANIFEST_BYTES,
    MAX_CAMPAIGN_TOTAL_LEDGER_BYTES,
};
use symthaea_nano_policy::sim_budget::MAX_SIMULATOR_LEDGER_BYTES;

const USAGE: &str = "nano-policy-campaign-audit --manifest campaign-manifest.json --ledger-root DIR

Checks that all policy arms use the same explicit seed roster and that every referenced
run ledger matches its SHA-256, declared policy/run identity, task/recipe contract, and
single-run actual simulator-call budget audit.

Options:
  --manifest PATH      Frozen campaign manifest JSON (required)
  --ledger-root DIR    Root directory for manifest-relative run-ledger files (required)
  --help               Print this help

The campaign manifest lists an expected seed roster and, for each arm, a relative path,
seed and raw-byte SHA-256 for every per-seed ledger. Missing files and mismatched rosters
produce an audit report with exit status 3. Unsafe paths and malformed contracts exit 2.
A successful audit proves only source/seed/budget accounting; it does not establish that
the statistical sample size is adequate or that the scientific result is valid.";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("nano-policy-campaign-audit: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return if args.is_empty() {
            Err("missing arguments; see --help".to_string())
        } else {
            Ok(ExitCode::SUCCESS)
        };
    }

    let mut manifest_path: Option<PathBuf> = None;
    let mut ledger_root: Option<PathBuf> = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        match flag {
            "--manifest" => {
                let value = next_value(&args, &mut index, flag)?;
                if manifest_path.replace(PathBuf::from(value)).is_some() {
                    return Err("--manifest was supplied more than once".to_string());
                }
            }
            "--ledger-root" => {
                let value = next_value(&args, &mut index, flag)?;
                if ledger_root.replace(PathBuf::from(value)).is_some() {
                    return Err("--ledger-root was supplied more than once".to_string());
                }
            }
            _ => return Err(format!("unknown argument {flag:?}; see --help")),
        }
    }

    let manifest_path = manifest_path.ok_or_else(|| "--manifest is required".to_string())?;
    let ledger_root = ledger_root.ok_or_else(|| "--ledger-root is required".to_string())?;
    let manifest_metadata = fs::metadata(&manifest_path)
        .map_err(|error| format!("cannot inspect campaign manifest: {error}"))?;
    if !manifest_metadata.is_file() {
        return Err("campaign manifest path must identify a regular file".to_string());
    }
    if manifest_metadata.len() > MAX_CAMPAIGN_MANIFEST_BYTES as u64 {
        return Err(format!(
            "campaign manifest exceeds the {MAX_CAMPAIGN_MANIFEST_BYTES}-byte limit"
        ));
    }
    let raw_manifest = fs::read(&manifest_path)
        .map_err(|error| format!("cannot read campaign manifest: {error}"))?;
    if raw_manifest.len() > MAX_CAMPAIGN_MANIFEST_BYTES {
        return Err(format!(
            "campaign manifest exceeds the {MAX_CAMPAIGN_MANIFEST_BYTES}-byte limit"
        ));
    }

    let manifest = parse_campaign_manifest(&raw_manifest)?;
    let root = fs::canonicalize(&ledger_root)
        .map_err(|error| format!("cannot resolve ledger root: {error}"))?;
    if !root.is_dir() {
        return Err("ledger root must identify a directory".to_string());
    }

    let mut ledger_bytes = BTreeMap::<String, Vec<u8>>::new();
    let mut total_ledger_bytes = 0usize;
    for arm in &manifest.arms {
        for reference in &arm.ledgers {
            if ledger_bytes.contains_key(&reference.relative_path) {
                continue;
            }
            let candidate_path = root.join(&reference.relative_path);
            let metadata = match fs::symlink_metadata(&candidate_path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(format!(
                        "cannot inspect ledger {:?}: {error}",
                        reference.relative_path
                    ));
                }
            };
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "ledger {:?} must not be a symbolic link",
                    reference.relative_path
                ));
            }
            if !metadata.is_file() {
                return Err(format!(
                    "ledger {:?} must identify a regular file",
                    reference.relative_path
                ));
            }
            let resolved = fs::canonicalize(&candidate_path).map_err(|error| {
                format!(
                    "cannot resolve ledger {:?}: {error}",
                    reference.relative_path
                )
            })?;
            if !resolved.starts_with(&root) {
                return Err(format!(
                    "ledger {:?} resolves outside the declared ledger root",
                    reference.relative_path
                ));
            }
            if metadata.len() > MAX_SIMULATOR_LEDGER_BYTES as u64 {
                return Err(format!(
                    "ledger {:?} exceeds the {MAX_SIMULATOR_LEDGER_BYTES}-byte limit",
                    reference.relative_path
                ));
            }
            let raw = fs::read(&resolved).map_err(|error| {
                format!("cannot read ledger {:?}: {error}", reference.relative_path)
            })?;
            if raw.len() > MAX_SIMULATOR_LEDGER_BYTES {
                return Err(format!(
                    "ledger {:?} exceeds the {MAX_SIMULATOR_LEDGER_BYTES}-byte limit",
                    reference.relative_path
                ));
            }
            total_ledger_bytes = total_ledger_bytes
                .checked_add(raw.len())
                .ok_or_else(|| "combined ledger byte count overflow".to_string())?;
            if total_ledger_bytes > MAX_CAMPAIGN_TOTAL_LEDGER_BYTES {
                return Err(format!(
                    "campaign ledger bytes exceed the {MAX_CAMPAIGN_TOTAL_LEDGER_BYTES}-byte limit"
                ));
            }
            ledger_bytes.insert(reference.relative_path.clone(), raw);
        }
    }

    let report = audit_campaign_json(&raw_manifest, &ledger_bytes)?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    serde_json::to_writer_pretty(&mut output, &report)
        .map_err(|error| format!("cannot serialize campaign audit report: {error}"))?;
    writeln!(output).map_err(|error| format!("cannot write campaign audit report: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("cannot flush campaign audit report: {error}"))?;

    if report.audit_pass {
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("nano-policy-campaign-audit: seed/budget campaign reconciliation failed");
        Ok(ExitCode::from(3))
    }
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(*index)
        .ok_or_else(|| format!("missing value after {flag}"))?
        .clone();
    *index += 1;
    Ok(value)
}

