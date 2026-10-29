use gcrb_evaluator::{evaluate_public_development, validate_fixture_bytes};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "GCRB public-development evaluator\n\n\
         Usage:\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- validate-fixture <fixture.json>\n\
         cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- evaluate <fixture.json> <candidate-report.json> [report-output.json]\n\n\
         This is a public known-answer development scorer, not blind qualification."
    );
}

fn read_file(path: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|error| format!("cannot serialize JSON output: {error}"))?;
    println!("{output}");
    Ok(())
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, fixture_path] if command == "validate-fixture" => {
            let fixture_bytes = read_file(fixture_path)?;
            let summary = validate_fixture_bytes(&fixture_bytes)?;
            print_json(&summary)
        }
        [command, fixture_path, candidate_path] if command == "evaluate" => {
            let fixture_bytes = read_file(fixture_path)?;
            let candidate_bytes = read_file(candidate_path)?;
            let report = evaluate_public_development(&fixture_bytes, &candidate_bytes)?;
            print_json(&report)
        }
        [command, fixture_path, candidate_path, output_path] if command == "evaluate" => {
            let fixture_bytes = read_file(fixture_path)?;
            let candidate_bytes = read_file(candidate_path)?;
            let report = evaluate_public_development(&fixture_bytes, &candidate_bytes)?;
            let output = serde_json::to_string_pretty(&report)
                .map_err(|error| format!("cannot serialize evaluation report: {error}"))?;
            fs::write(output_path, format!("{output}\n"))
                .map_err(|error| format!("cannot write {output_path}: {error}"))?;
            eprintln!("wrote public-development evaluation report to {output_path}");
            Ok(())
        }
        _ => {
            usage();
            Err("invalid command-line arguments".to_owned())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gcrb-evaluator: {error}");
            ExitCode::from(2)
        }
    }
}
