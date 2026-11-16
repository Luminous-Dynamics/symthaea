use benchmark_run_receipt_validator::{validate_comparable_arms, verify_receipt_json};
use serde_json::json;
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "Usage: benchmark-run-receipt-validate <receipt.json> [--compare <receipt.json> ...]"
    );
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(first) = args.next() else {
        usage();
        return ExitCode::from(2);
    };

    let first_path = first;
    let mut compare_paths = Vec::new();
    let mut compare_mode = false;
    while let Some(arg) = args.next() {
        if arg == std::ffi::OsStr::new("--compare") {
            compare_mode = true;
            continue;
        }
        if !compare_mode {
            usage();
            return ExitCode::from(2);
        }
        compare_paths.push(arg);
    }

    let mut verified = Vec::new();
    for path in std::iter::once(first_path).chain(compare_paths.into_iter()) {
        let display_path = path.to_string_lossy().into_owned();
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("cannot read {display_path}: {error}");
                return ExitCode::from(2);
            }
        };
        match verify_receipt_json(&bytes) {
            Ok(receipt) => {
                verified.push(receipt);
            }
            Err(error) => {
                eprintln!(
                    "invalid benchmark receipt {display_path}: {:?}: {}",
                    error.kind, error.message
                );
                return ExitCode::from(1);
            }
        }
    }

    if verified.len() > 1 {
        if let Err(error) = validate_comparable_arms(&verified) {
            eprintln!(
                "receipts are not a matched comparison group: {:?}: {}",
                error.kind, error.message
            );
            return ExitCode::from(1);
        }
    }

    let summaries: Vec<_> = verified
        .iter()
        .map(|receipt| {
            json!({
                "run_id": receipt.run_id,
                "raw_file_sha256": receipt.raw_file_sha256,
                "canonical_receipt_sha256": receipt.canonical_receipt_sha256,
                "benchmark_id": receipt.value["benchmark"]["id"],
                "benchmark_version": receipt.value["benchmark"]["version"],
                "access_class": receipt.value["benchmark"]["access_class"],
                "system_arm": receipt.value["system"]["arm"],
                "target_scope": receipt.value["target_scope"],
                "accounting": receipt.accounting,
            })
        })
        .collect();

    let output = json!({
        "schema": "luminous.benchmark-receipt-validation-result.v1",
        "valid": true,
        "comparison_checked": verified.len() > 1,
        "receipts": summaries,
        "claim_ceiling": "Accounting and consistency only; independent benchmark grader remains the correctness oracle. Hashes do not authenticate a producer."
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON summary serializes")
    );
    ExitCode::SUCCESS
}
