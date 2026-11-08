use std::env;
use std::fs;
use std::process::ExitCode;

use physical_scenario_validator::{replay_domain_outages, validate_json};

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let path = match (args.next(), args.next()) {
        (Some(path), None) => path,
        _ => {
            eprintln!("usage: {} <scenario-manifest.json>", program.to_string_lossy());
            return ExitCode::from(2);
        }
    };

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {}: {error}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };

    let report = match validate_json(&input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid JSON: {error}");
            return ExitCode::from(2);
        }
    };

    if !report.is_valid() {
        for error in report.errors {
            eprintln!("ERROR: {error}");
        }
        eprintln!("SEMANTIC_INVALID");
        return ExitCode::FAILURE;
    }

    let value: serde_json::Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid JSON during replay: {error}");
            return ExitCode::from(2);
        }
    };
    let replay = match replay_domain_outages(&value) {
        Ok(replay) => replay,
        Err(error) => {
            eprintln!("PREFLIGHT_REPLAY_UNSUPPORTED: {error}");
            return ExitCode::FAILURE;
        }
    };

    let scenario_id = match value.get("scenario_id").and_then(serde_json::Value::as_str) {
        Some(id) => id,
        None => {
            eprintln!("validated manifest unexpectedly lacks scenario_id");
            return ExitCode::FAILURE;
        }
    };
    let receipt = serde_json::json!({
        "receipt_version": "1.0.0",
        "result_kind": "preflight_only",
        "scenario_id": scenario_id,
        "manifest_sha256": report.canonical_sha256,
        "replay_engine_version": "domain-outage-preflight/0.1.0",
        "failed_domain_ids": replay.failed_domain_ids,
        "failed_component_ids": replay.failed_component_ids,
        "candidate_available": replay.candidate_available,
        "verifier_available": replay.verifier_available,
        "preflight_classification": replay.classification,
        "correctness": "not_evaluated",
        "latency": "not_measured",
        "energy": "not_measured"
    });
    match serde_json::to_string_pretty(&receipt) {
        Ok(encoded) => println!("{encoded}"),
        Err(error) => {
            eprintln!("could not encode preflight receipt: {error}");
            return ExitCode::FAILURE;
        }
    }

    // A preflight receipt is not an execution result and must never be scored
    // as a correct accepted task.
    eprintln!("PREFLIGHT_ONLY: correctness, latency, energy and physical behavior were not evaluated");
    ExitCode::SUCCESS
}
