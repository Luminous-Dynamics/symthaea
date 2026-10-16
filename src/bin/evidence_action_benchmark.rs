use serde_json::json;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::process::ExitCode;
use symthaea::knowledge::evidence_action_policy::EvidenceActionPolicy;
use symthaea::knowledge::evidence_benchmark::{
    audit_dataset_jsonl, evaluate_cases, score_receipts_jsonl,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, message)) => {
            eprintln!("{message}");
            ExitCode::from(code)
        }
    }
}

fn run() -> Result<(), (u8, String)> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("audit") if args.len() == 3 => {
            let file = File::open(&args[2]).map_err(input_error)?;
            let (_, report) = audit_dataset_jsonl(BufReader::new(file))
                .map_err(|e| (1, format!("dataset audit failed: {e}")))?;
            println!("{}", serde_json::to_string_pretty(&report).map_err(internal_error)?);
            Ok(())
        }
        Some("evaluate") if args.len() == 7 => {
            let dataset_file = File::open(&args[2]).map_err(input_error)?;
            let (cases, audit) = audit_dataset_jsonl(BufReader::new(dataset_file))
                .map_err(|e| (1, format!("dataset audit failed: {e}")))?;
            let minimum_origins = args[5].parse::<u32>()
                .map_err(|e| (2, format!("invalid minimum origin count: {e}")))?;
            let allow_scoped = args[6].parse::<bool>()
                .map_err(|e| (2, format!("allow-scoped must be true or false: {e}")))?;
            let policy = EvidenceActionPolicy::new(minimum_origins, allow_scoped)
                .map_err(|e| (2, e.to_owned()))?;
            if args[4].trim().is_empty() {
                return Err((2, "policy version cannot be empty".to_owned()));
            }
            let receipts = evaluate_cases(&cases, &args[4], &policy);
            let output = File::create(&args[3]).map_err(output_error)?;
            let mut writer = BufWriter::new(output);
            for receipt in &receipts {
                serde_json::to_writer(&mut writer, receipt).map_err(internal_error)?;
                writer.write_all(b"\n").map_err(output_error)?;
            }
            writer.flush().map_err(output_error)?;
            println!("{}", serde_json::to_string_pretty(&json!({
                "schema_version": audit.schema_version,
                "case_count": audit.total_cases,
                "policy_version": args[4],
                "receipt_count": receipts.len(),
                "receipt_path": args[3],
                "notice": "Receipts authored; run score to verify coverage, digest binding, and benchmark outcomes."
            })).map_err(internal_error)?);
            Ok(())
        }
        Some("score") if args.len() == 5 => {
            let dataset_file = File::open(&args[2]).map_err(input_error)?;
            let (cases, _) = audit_dataset_jsonl(BufReader::new(dataset_file))
                .map_err(|e| (1, format!("dataset audit failed: {e}")))?;
            let receipt_file = File::open(&args[3]).map_err(input_error)?;
            let score = score_receipts_jsonl(&cases, BufReader::new(receipt_file), &args[4])
                .map_err(|e| (1, format!("receipt verification/scoring failed: {e}")))?;
            println!("{}", serde_json::to_string_pretty(&score).map_err(internal_error)?);
            Ok(())
        }
        _ => Err((2, usage().to_owned())),
    }
}

fn input_error(error: std::io::Error) -> (u8, String) {
    (2, format!("input error: {error}"))
}
fn output_error(error: std::io::Error) -> (u8, String) {
    (2, format!("output error: {error}"))
}
fn internal_error(error: serde_json::Error) -> (u8, String) {
    (2, format!("serialization error: {error}"))
}
fn usage() -> &'static str {
    "Usage:\n  evidence_action_benchmark audit DATASET.jsonl\n  evidence_action_benchmark evaluate DATASET.jsonl RECEIPTS.jsonl POLICY_VERSION MIN_ORIGINS ALLOW_SCOPED\n  evidence_action_benchmark score DATASET.jsonl RECEIPTS.jsonl POLICY_VERSION"
}
