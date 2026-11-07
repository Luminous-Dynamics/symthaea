use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_evaluator_reference::{parse_and_validate, parse_and_validate_state_routing};

fn run_evaluator_fixture_pack(path: &str) -> ExitCode {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    let report = match parse_and_validate(&contents) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid evaluator fixture-pack JSON in {path}: {error}");
            return ExitCode::from(65);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize evaluator validation report: {error}");
            return ExitCode::from(70);
        }
    }

    if report.valid_fixture_pack {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn run_state_routing_fixture_pack(path: &str) -> ExitCode {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    let report = match parse_and_validate_state_routing(&contents) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid state-routing fixture-pack JSON in {path}: {error}");
            return ExitCode::from(65);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize state-routing validation report: {error}");
            return ExitCode::from(70);
        }
    }

    if report.valid_fixture_pack {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [path] => run_evaluator_fixture_pack(path),
        [mode, path] if mode == "--state-routing" => run_state_routing_fixture_pack(path),
        [mode, path] if mode == "--evaluator" => run_evaluator_fixture_pack(path),
        _ => {
            eprintln!(
                "usage:\n  sym-rsi-evaluator-reference <evaluator-fixtures.json>\n  sym-rsi-evaluator-reference --evaluator <evaluator-fixtures.json>\n  sym-rsi-evaluator-reference --state-routing <state-routing-fixtures.json>"
            );
            ExitCode::from(64)
        }
    }
}
