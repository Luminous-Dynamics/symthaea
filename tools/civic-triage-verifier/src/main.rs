use civic_triage_verifier::verify_manifest;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "docs/research/civic-triage/CIVIC_TRIAGE_CASES_V0_1.json".to_string());

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot_read_manifest {path}: {error}");
            return ExitCode::from(2);
        }
    };

    let report = verify_manifest(&bytes);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("cannot_encode_report: {error}");
            return ExitCode::from(2);
        }
    }

    if report.status == "CONFORMS" {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
