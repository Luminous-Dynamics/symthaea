use std::env;
use std::fs;
use std::process::ExitCode;

use mel_rmi_expressive_v2::{RmiExpressiveTrialV2, seal_trial, validate_trial};

fn read_trial(path: &str) -> Result<RmiExpressiveTrialV2, String> {
    let input = fs::read_to_string(path).map_err(|error| format!("failed to read {path}: {error}"))?;
    serde_json::from_str(&input).map_err(|error| format!("invalid RMI expressive v2 JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() == 4 && args[1] == "--seal" {
        let mut trial = match read_trial(&args[2]) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(error) = seal_trial(&mut trial) {
            eprintln!("failed to seal trial: {error}");
            return ExitCode::FAILURE;
        }
        let output = match serde_json::to_vec_pretty(&trial) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("failed to serialize sealed trial: {error}");
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
        eprintln!("usage: mel-rmi-expressive-v2 TRIAL.json | --seal INPUT.json OUTPUT.json");
        return ExitCode::FAILURE;
    }
    let trial = match read_trial(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let report = validate_trial(&trial);
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
