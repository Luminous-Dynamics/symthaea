use mel_mmae_package_validator::{read_expected_ids, validate_package, DEFAULT_MANIFEST_NAME};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> &'static str {
    "Usage:\n  mel-mmae-package-validator validate --root PACKAGE_DIR --expected-ids IDS.txt [--manifest submission.jsonl] [--expected-count N] [--report REPORT.json]\n\nThe expected-ID file contains one released challenge sample ID per line."
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf, Option<usize>, Option<PathBuf>), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage().to_owned())?;
    if command == "--help" || command == "help" {
        return Err(usage().to_owned());
    }
    if command != "validate" {
        return Err(format!("unknown command {command:?}\n\n{}", usage()));
    }

    let mut root = None;
    let mut expected_ids = None;
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST_NAME);
    let mut expected_count = None;
    let mut report = None;

    while let Some(option) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {option}\n\n{}", usage()))?;
        match option.as_str() {
            "--root" => root = Some(PathBuf::from(value)),
            "--expected-ids" => expected_ids = Some(PathBuf::from(value)),
            "--manifest" => manifest = PathBuf::from(value),
            "--expected-count" => {
                expected_count = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --expected-count {value:?}"))?,
                )
            }
            "--report" => report = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {option:?}\n\n{}", usage())),
        }
    }

    Ok((
        root.ok_or_else(|| format!("--root is required\n\n{}", usage()))?,
        expected_ids.ok_or_else(|| format!("--expected-ids is required\n\n{}", usage()))?,
        manifest,
        expected_count,
        report,
    ))
}

fn run() -> Result<u8, String> {
    let (root, expected_path, manifest, expected_count, report_path) = parse_args()?;
    let expected_ids = read_expected_ids(&expected_path)?;
    let report = validate_package(&root, &manifest, &expected_ids, expected_count);
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot serialize report: {error}"))?;

    if let Some(path) = report_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create report directory: {error}"))?;
            }
        }
        fs::write(&path, json.as_bytes())
            .map_err(|error| format!("cannot write report {}: {error}", path.display()))?;
    }

    println!("{json}");
    Ok(if report.valid { 0 } else { 2 })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
