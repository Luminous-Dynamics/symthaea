use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_research_taste_reference::{parse_and_calculate_evsi, parse_and_validate_fixtures, parse_and_validate_pivotal_corrections};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(mode) = args.next() else {
        eprintln!("usage: sym-rsi-research-taste-reference <fixtures|pivotal|evsi> <input.json>");
        return ExitCode::from(64);
    };
    let Some(path) = args.next() else {
        eprintln!("usage: sym-rsi-research-taste-reference <fixtures|evsi> <input.json>");
        return ExitCode::from(64);
    };

    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    match mode.as_str() {
        "fixtures" => match parse_and_validate_fixtures(&contents) {
            Ok(report) => {
                println!("{}", serde_json::to_string_pretty(&report).expect("serializable report"));
                if report.valid_fixture_pack { ExitCode::SUCCESS } else { ExitCode::from(2) }
            }
            Err(error) => {
                eprintln!("invalid fixture JSON: {error}");
                ExitCode::from(65)
            }
        },
        "pivotal" => match parse_and_validate_pivotal_corrections(&contents) {
            Ok(report) => {
                println!("{}", serde_json::to_string_pretty(&report).expect("serializable report"));
                if report.valid_fixture_pack { ExitCode::SUCCESS } else { ExitCode::from(2) }
            }
            Err(error) => {
                eprintln!("invalid pivotal-correction fixture JSON: {error}");
                ExitCode::from(65)
            }
        },
        "evsi" => match parse_and_calculate_evsi(&contents) {
            Ok(report) => {
                println!("{}", serde_json::to_string_pretty(&report).expect("serializable report"));
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("invalid EVSI problem: {error}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("unknown mode {mode:?}; expected fixtures, pivotal or evsi");
            ExitCode::from(64)
        }
    }
}
