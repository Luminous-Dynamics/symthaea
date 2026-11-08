use std::env;
use std::fs;
use std::process::ExitCode;

use physical_scenario_validator::validate_json;

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
    if report.is_valid() {
        println!("SEMANTIC_VALID");
        ExitCode::SUCCESS
    } else {
        for error in report.errors {
            eprintln!("ERROR: {error}");
        }
        println!("SEMANTIC_INVALID");
        ExitCode::FAILURE
    }
}
