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

    println!("canonical_manifest_digest={}", report.canonical_sha256);
    if !report.is_valid() {
        for error in report.errors {
            eprintln!("ERROR: {error}");
        }
        println!("SEMANTIC_INVALID");
        return ExitCode::FAILURE;
    }
    println!("SEMANTIC_VALID");

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

    println!("preflight_failed_domains={}", replay.failed_domain_ids.join(","));
    println!("preflight_failed_components={}", replay.failed_component_ids.join(","));
    println!("candidate_available={}", replay.candidate_available);
    println!("verifier_available={}", replay.verifier_available);
    println!("preflight_classification={}", replay.classification);
    println!("PREFLIGHT_ONLY: no task arithmetic, correctness, timing, energy or physical behavior was simulated");
    ExitCode::SUCCESS
}
