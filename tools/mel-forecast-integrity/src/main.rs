use std::env;
use std::fs;
use std::process::ExitCode;

use mel_forecast_integrity::{ForecastTraceV1, seal_trace, validate_trace};

fn read_trace(path: &str) -> Result<ForecastTraceV1, String> {
    let input = fs::read_to_string(path).map_err(|error| format!("failed to read {path}: {error}"))?;
    serde_json::from_str(&input).map_err(|error| format!("invalid forecast trace JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() == 4 && args[1] == "--seal" {
        let mut trace = match read_trace(&args[2]) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(error) = seal_trace(&mut trace) {
            eprintln!("failed to seal trace: {error}");
            return ExitCode::FAILURE;
        }
        let output = match serde_json::to_vec_pretty(&trace) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("failed to serialize sealed trace: {error}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(error) = fs::write(&args[3], output) {
            eprintln!("failed to write {}: {error}", args[3]);
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }
    if args.len() != 2 {
        eprintln!("usage: mel-forecast-integrity TRACE.json | --seal INPUT.json OUTPUT.json");
        return ExitCode::FAILURE;
    }
    let trace = match read_trace(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
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
    if report.valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
