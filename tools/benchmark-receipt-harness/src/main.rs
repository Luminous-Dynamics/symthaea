use std::env;
use std::fs;
use std::path::PathBuf;

use benchmark_run_receipt_harness::validate_receipt_json;
use serde_json::json;

fn main() {
    if let Err(error) = run() {
        eprintln!("benchmark receipt validation failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut receipt_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => {
                receipt_path = Some(PathBuf::from(args.next().ok_or("--receipt requires a path")?));
            }
            "--out" => {
                output_path = Some(PathBuf::from(args.next().ok_or("--out requires a path")?));
            }
            "--help" | "-h" => {
                println!("Usage: benchmark_receipt_validate --receipt FILE [--out FILE]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let path = receipt_path.ok_or("--receipt is required")?;
    let bytes = fs::read(&path)?;
    let verified = validate_receipt_json(&bytes)?;
    let report = json!({
        "schema": "luminous.benchmark-receipt-validation-report.v1",
        "result": "structurally_and_semantically_valid",
        "run_id": verified.run_id,
        "raw_file_sha256": verified.raw_file_sha256,
        "canonical_receipt_sha256": verified.canonical_receipt_sha256,
        "accounting": {
            "eligible": verified.eligible,
            "attempted": verified.attempted,
            "not_attempted": verified.not_attempted,
            "graded": verified.graded,
            "passed": verified.passed
        },
        "authenticity_verified": false,
        "claim_ceiling": "receipt integrity and accounting only; no producer authentication or answer correctness claim"
    });
    let output = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output_path {
        fs::write(path, &output)?;
    } else {
        println!("{}", String::from_utf8(output)?);
    }
    Ok(())
}
