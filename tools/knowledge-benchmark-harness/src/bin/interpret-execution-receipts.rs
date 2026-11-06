#[path = "../../../../crates/domains/symthaea-epistemic-types/src/execution_receipt.rs"]
mod execution_receipt;

use execution_receipt::{
    ExecutionEvidenceInterpretationV1, ReceiptInterpretationError,
    interpret_tool_execution_receipt_json,
};
use serde::Serialize;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    receipt_corpus: PathBuf,
    profile_corpus: PathBuf,
    expected_profile_sha256: String,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ReceiptResult {
    receipt_id: String,
    interpreted: bool,
    interpretation: Option<ExecutionEvidenceInterpretationV1>,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReceiptInterpretationReport {
    schema_version: &'static str,
    authority_ceiling: &'static str,
    expected_profile_corpus_sha256: String,
    receipts_seen: usize,
    receipts_interpreted: usize,
    passed: bool,
    results: Vec<ReceiptResult>,
    nonclaims: Vec<&'static str>,
}

fn usage() -> &'static str {
    "Usage: interpret-execution-receipts --receipt-corpus FILE.json --profile-corpus FILE.json --expected-profile-sha256 SHA256 [--out FILE.json]\n\nOffline pure interpretation only. No network, filesystem traversal, credential, EPI-admission, or action authority is provided."
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut receipt_corpus = None;
    let mut profile_corpus = None;
    let mut expected_profile_sha256 = None;
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt-corpus" => {
                receipt_corpus =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--receipt-corpus requires a path".to_string()
                    })?));
            }
            "--profile-corpus" => {
                profile_corpus =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--profile-corpus requires a path".to_string()
                    })?));
            }
            "--expected-profile-sha256" => {
                expected_profile_sha256 = Some(
                    args.next()
                        .ok_or_else(|| "--expected-profile-sha256 requires a value".to_string())?,
                );
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out requires a path".to_string())?,
                ));
            }
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    Ok(Args {
        receipt_corpus: receipt_corpus
            .ok_or_else(|| format!("missing --receipt-corpus\n{}", usage()))?,
        profile_corpus: profile_corpus
            .ok_or_else(|| format!("missing --profile-corpus\n{}", usage()))?,
        expected_profile_sha256: expected_profile_sha256
            .ok_or_else(|| format!("missing --expected-profile-sha256\n{}", usage()))?,
        output,
    })
}

fn errors_to_strings(errors: Vec<ReceiptInterpretationError>) -> Vec<String> {
    errors
        .into_iter()
        .map(|error| format!("{} at {}: {}", error.code, error.path, error.message))
        .collect()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let receipt_bytes = match fs::read(&args.receipt_corpus) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!(
                "cannot read receipt corpus {}: {error}",
                args.receipt_corpus.display()
            );
            return ExitCode::from(2);
        }
    };
    let profile_bytes = match fs::read(&args.profile_corpus) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!(
                "cannot read profile corpus {}: {error}",
                args.profile_corpus.display()
            );
            return ExitCode::from(2);
        }
    };
    let corpus: Value = match serde_json::from_slice(&receipt_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid receipt corpus JSON: {error}");
            return ExitCode::from(2);
        }
    };
    let receipts = match corpus.get("receipts").and_then(Value::as_array) {
        Some(receipts) if !receipts.is_empty() => receipts,
        _ => {
            eprintln!("receipt corpus must contain a non-empty receipts array");
            return ExitCode::from(2);
        }
    };

    let mut results = Vec::with_capacity(receipts.len());
    for value in receipts {
        let receipt_id = value
            .get("receipt_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing-receipt-id>")
            .to_string();
        let receipt_json = match serde_json::to_vec(value) {
            Ok(bytes) => bytes,
            Err(error) => {
                results.push(ReceiptResult {
                    receipt_id,
                    interpreted: false,
                    interpretation: None,
                    errors: vec![format!("cannot serialize JSON value: {error}")],
                });
                continue;
            }
        };
        match interpret_tool_execution_receipt_json(
            &receipt_json,
            &profile_bytes,
            &args.expected_profile_sha256,
        ) {
            Ok(interpretation) => results.push(ReceiptResult {
                receipt_id,
                interpreted: true,
                interpretation: Some(interpretation),
                errors: Vec::new(),
            }),
            Err(errors) => results.push(ReceiptResult {
                receipt_id,
                interpreted: false,
                interpretation: None,
                errors: errors_to_strings(errors),
            }),
        }
    }

    let receipts_interpreted = results.iter().filter(|result| result.interpreted).count();
    let passed = receipts_interpreted == results.len();
    let report = ReceiptInterpretationReport {
        schema_version: "symthaea:execution-evidence-interpretation-report:v0.1",
        authority_ceiling: "CandidateAnalysisOnly",
        expected_profile_corpus_sha256: args.expected_profile_sha256,
        receipts_seen: results.len(),
        receipts_interpreted,
        passed,
        results,
        nonclaims: vec![
            "does not execute a tool or authorize network/browser access",
            "does not verify an EPI receipt signature or transparency inclusion proof",
            "does not establish source authenticity, factual truth, or independent corroboration",
            "does not admit results into Mycelix canonical evidence or Symthaea memory",
            "does not grant credential, collection, action, or policy authority",
        ],
    };
    let serialized = match serde_json::to_vec_pretty(&report) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot serialize interpretation report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, &serialized) {
            eprintln!("cannot write {}: {error}", path.display());
            return ExitCode::from(2);
        }
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
