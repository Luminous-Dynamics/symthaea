use std::env;
use std::fs;
use std::process::ExitCode;

use mel_agency_trace::{AgencyTraceV1, validate_trace};

fn main() -> ExitCode {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "tools/mel-agency-trace/fixtures/apply-approved-undo-v1.json".to_owned());

    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    let trace: AgencyTraceV1 = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid agency trace JSON: {error}");
            return ExitCode::FAILURE;
        }
    };

    let report = validate_trace(&trace);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize report: {error}");
            return ExitCode::FAILURE;
        }
    }

    if report.valid { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
