use nuclear_tea_reference::{validate_port_reference, PortReferenceDataset};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(Serialize)]
struct Receipt {
    receipt_schema: &'static str,
    tool_version: &'static str,
    build_commit: String,
    status: &'static str,
    input_path: String,
    input_sha256: String,
    dataset_id: String,
    source_class: String,
    source_count: usize,
    shore_power_row_count: usize,
    spare_capacity_row_count: usize,
    conversion_checks: usize,
    claim_ceiling: &'static str,
}

#[derive(Serialize)]
struct FailureReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    input_path: String,
    input_sha256: String,
    errors: Vec<String>,
    claim_ceiling: &'static str,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() -> ExitCode {
    let input_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "research/nuclear/port-reference-evidence-sa-2025.json".to_string());

    let bytes = match fs::read(&input_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read {input_path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let input_sha256 = sha256_hex(&bytes);
    let build_commit =
        env::var("GITHUB_SHA").unwrap_or_else(|_| "UNBOUND_LOCAL_BUILD".to_string());

    let dataset: PortReferenceDataset = match serde_json::from_slice(&bytes) {
        Ok(dataset) => dataset,
        Err(error) => {
            let receipt = FailureReceipt {
                receipt_schema: "nuclear-tea-port-reference-receipt-v1",
                tool_version: env!("CARGO_PKG_VERSION"),
                build_commit: build_commit.clone(),
                status: "FAIL",
                input_path,
                input_sha256,
                errors: vec![format!("JSON parse error: {error}")],
                claim_ceiling: "Source structure and arithmetic only; no engineering or licensing claim.",
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&receipt).expect("receipt serializes")
            );
            return ExitCode::FAILURE;
        }
    };

    match validate_port_reference(&dataset) {
        Ok(report) => {
            let receipt = Receipt {
                receipt_schema: "nuclear-tea-port-reference-receipt-v1",
                tool_version: env!("CARGO_PKG_VERSION"),
                build_commit: build_commit.clone(),
                status: "PASS_SOURCE_DATA_CONTRACT_ONLY",
                input_path,
                input_sha256,
                dataset_id: report.dataset_id,
                source_class: report.source_class,
                source_count: report.source_count,
                shore_power_row_count: report.shore_power_row_count,
                spare_capacity_row_count: report.spare_capacity_row_count,
                conversion_checks: report.conversion_checks,
                claim_ceiling: "Validates source structure and declared unit conversion only; does not establish metered load, grid capacity, site suitability, nuclear safety, licensing or economic feasibility.",
            };
            println!("{}", serde_json::to_string_pretty(&receipt).expect("receipt serializes"));
            ExitCode::SUCCESS
        }
        Err(validation_errors) => {
            let receipt = FailureReceipt {
                receipt_schema: "nuclear-tea-port-reference-receipt-v1",
                tool_version: env!("CARGO_PKG_VERSION"),
                build_commit: build_commit.clone(),
                status: "FAIL",
                input_path,
                input_sha256,
                errors: validation_errors.iter().map(ToString::to_string).collect(),
                claim_ceiling: "Source structure and arithmetic only; no engineering or licensing claim.",
            };
            println!("{}", serde_json::to_string_pretty(&receipt).expect("receipt serializes"));
            ExitCode::FAILURE
        }
    }
}
