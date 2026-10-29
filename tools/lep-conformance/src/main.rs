use lep_conformance::{ValidationReport, parse_json_unique_keys, verify_statement};
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
            let report =
                ValidationReport::invalid_json(format!("Unable to read statement file: {error}"));
            emit(&report);
            return ExitCode::from(2);
        }
    };

    let statement: Value = match parse_json_unique_keys(&bytes) {
        Ok(value) => value,
        Err(error) => {
            let report = ValidationReport::invalid_json(format!(
                "Invalid JSON statement (including duplicate-key rejection): {error}"
            ));
            emit(&report);
            return ExitCode::from(2);
        }
    };

    let report = verify_statement(&statement);
    let conforms = report.conforms;
    emit(&report);
    if conforms {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
