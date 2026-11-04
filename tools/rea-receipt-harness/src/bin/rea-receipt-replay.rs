//! Read-only independent replay for a versioned REA member-content receipt.
//!
//! Usage: rea-receipt-replay <archive.zip> <receipt.json>
//! Exit codes: 0 replayed, 1 rejected, 2 unsupported, 3 I/O/argument/schema error.

use rea_member_receipt_harness::{
    canonicalize_raw_receipt,
    replay::{ReplayError, replay_member_claim},
    validate_schema_document,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::process::ExitCode;

const RECEIPT_SCHEMA: &str =
    include_str!("../../../../docs/research/rea-zip-content-verification-receipt-v0.schema.json");
const MAX_RECEIPT_BYTES: u64 = 2 * 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, detail)) => {
            eprintln!("{detail}");
            ExitCode::from(code)
        }
    }
}

fn run() -> Result<(), (u8, String)> {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let archive = args.next().ok_or_else(|| {
        (
            3,
            format!(
                "usage: {} <archive.zip> <receipt.json>",
                program.to_string_lossy()
            ),
        )
    })?;
    let receipt_path = args.next().ok_or_else(|| {
        (
            3,
            format!(
                "usage: {} <archive.zip> <receipt.json>",
                program.to_string_lossy()
            ),
        )
    })?;
    if args.next().is_some() {
        return Err((3, "unexpected extra command-line arguments".into()));
    }

    let schema: Value = serde_json::from_str(RECEIPT_SCHEMA)
        .map_err(|error| (3, format!("embedded receipt schema is invalid: {error}")))?;
    validate_schema_document(&schema)
        .map_err(|error| (3, format!("receipt schema validation failed: {error}")))?;

    let receipt_metadata = std::fs::metadata(&receipt_path)
        .map_err(|error| (3, format!("cannot stat receipt file: {error}")))?;
    if receipt_metadata.len() > MAX_RECEIPT_BYTES {
        return Err((3, "receipt JSON exceeds the 2 MiB input bound".into()));
    }
    let raw_receipt = std::fs::read(&receipt_path)
        .map_err(|error| (3, format!("cannot read receipt file: {error}")))?;
    let (receipt, canonical) = canonicalize_raw_receipt(&schema, &raw_receipt)
        .map_err(|error| (1, format!("receipt is invalid: {error}")))?;
    let receipt_sha256 = format!("{:x}", Sha256::digest(&canonical));

    match replay_member_claim(&archive, &receipt) {
        Ok(report) => {
            let output = json!({
                "outcome": report.outcome,
                "claim": "independently_replayed_member_content_integrity",
                "receipt_sha256": receipt_sha256,
                "archive_sha256": report.archive_sha256,
                "payload_sha256": report.payload_sha256,
                "observed_uncompressed_bytes": report.observed_bytes.to_string(),
                "observed_crc32": format!("{:08x}", report.observed_crc32),
                "oracle": report.oracle,
                "codec": report.codec,
                "not_established": [
                    "producer_authenticity",
                    "producer_identity",
                    "apk_signature_validity",
                    "ipa_code_signature_validity",
                    "whole_archive_completeness",
                    "package_installability",
                    "runtime_behavior",
                    "masvs_control_pass",
                    "semantic_truth",
                    "absence_of_malicious_writer"
                ]
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output)
                    .map_err(|error| (3, format!("cannot serialize replay report: {error}")))?
            );
            Ok(())
        }
        Err(ReplayError::Unsupported(detail)) => {
            let output =
                json!({"outcome":"unsupported","reason":detail,"receipt_sha256":receipt_sha256});
            eprintln!(
                "{}",
                serde_json::to_string(&output)
                    .unwrap_or_else(|_| "{\"outcome\":\"unsupported\"}".into())
            );
            Err((2, format!("unsupported: {detail}")))
        }
        Err(ReplayError::Rejected(detail)) => {
            let output =
                json!({"outcome":"rejected","reason":detail,"receipt_sha256":receipt_sha256});
            eprintln!(
                "{}",
                serde_json::to_string(&output)
                    .unwrap_or_else(|_| "{\"outcome\":\"rejected\"}".into())
            );
            Err((1, format!("rejected: {detail}")))
        }
        Err(ReplayError::Io(detail)) => Err((3, detail)),
    }
}
