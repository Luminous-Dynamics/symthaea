use lep_conformance::{MAX_STATEMENT_BYTES, ValidationReport, verify_statement_bytes};
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
    let report = verify_statement_bytes(&bytes);
    let parse_failed = report.violations.iter().any(|violation| {
        matches!(
            violation.code.as_str(),
            "INVALID_JSON" | "DUPLICATE_JSON_KEY" | "STATEMENT_TOO_LARGE"
        )
    });
    let conforms = report.conforms;
    emit(&report);
    if conforms {
        ExitCode::SUCCESS
    } else if parse_failed {
        ExitCode::from(2)
    } else {
        ExitCode::from(1)
    }
}
