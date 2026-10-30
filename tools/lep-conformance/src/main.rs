use lep_conformance::{
    MAX_STATEMENT_BYTES, ValidationReport, parse_json_unique_keys, verify_statement,
};
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::Read;
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

    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            let report =
                ValidationReport::invalid_json(format!("Unable to read statement file: {error}"));
            emit(&report);
            return ExitCode::from(2);
        }
    };
    let mut bytes = Vec::new();
    if let Err(error) = file
        .take(MAX_STATEMENT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
    {
        let report =
            ValidationReport::invalid_json(format!("Unable to read statement file: {error}"));
        emit(&report);
        return ExitCode::from(2);
    }
    if bytes.len() > MAX_STATEMENT_BYTES {
        let report = ValidationReport::invalid_json(format!(
            "Statement exceeds maximum size of {MAX_STATEMENT_BYTES} bytes."
        ));
        emit(&report);
        return ExitCode::from(2);
    }

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
