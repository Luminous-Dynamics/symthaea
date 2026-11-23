use std::env;
use std::fs;
use std::process::ExitCode;

use mel_judge_gap::parse_and_qualify;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: mel-judge-gap QUALIFICATION.json");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("usage: mel-judge-gap QUALIFICATION.json");
        return ExitCode::FAILURE;
    }
    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let report = match parse_and_qualify(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("qualification rejected: {error}");
            return ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize qualification report: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
