use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_r4_prereg_reference::validate_preregistration;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: sym-rsi-r4-prereg-reference <preregistration.json>");
        return ExitCode::from(64);
    };

    let input = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };
    let document: serde_json::Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid JSON in {path}: {error}");
            return ExitCode::from(65);
        }
    };

    let report = validate_preregistration(&document);
    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize validation report: {error}");
            return ExitCode::from(70);
        }
    }

    if report.ready_for_independent_review {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}
