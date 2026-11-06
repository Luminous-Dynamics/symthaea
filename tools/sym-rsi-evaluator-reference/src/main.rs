use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_evaluator_reference::parse_and_validate;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: sym-rsi-evaluator-reference <fixture-pack.json>");
        return ExitCode::from(64);
    };

    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    let report = match parse_and_validate(&contents) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid fixture-pack JSON in {path}: {error}");
            return ExitCode::from(65);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize validation report: {error}");
            return ExitCode::from(70);
        }
    }

    if report.valid_fixture_pack {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}
