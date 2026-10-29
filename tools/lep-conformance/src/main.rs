use lep_conformance::{verify_statement, ValidationReport};
use serde_json::Value;
use std::env;
use std::fs;
use std::process::ExitCode;

fn emit(report: &ValidationReport) {
    match serde_json::to_string_pretty(report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize validation report: {error}");
        }
    }
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        let report = ValidationReport::invalid_json("Usage: lep-conformance STATEMENT.json");
        emit(&report);
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        let report = ValidationReport::invalid_json("Exactly one statement path is required.");
        emit(&report);
        return ExitCode::from(2);
    }

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            let report = ValidationReport::invalid_json(format!(
                "Unable to read statement file: {error}"
            ));
            emit(&report);
            return ExitCode::from(2);
        }
    };

    let statement: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            let report = ValidationReport::invalid_json(format!("Invalid JSON statement: {error}"));
            emit(&report);
            return ExitCode::from(2);
        }
    };

    let report = verify_statement(&statement);
    let valid = report.valid;
    emit(&report);
    if valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
