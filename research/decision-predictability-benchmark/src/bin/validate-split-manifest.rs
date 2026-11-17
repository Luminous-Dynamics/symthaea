use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_decision_predictability::manifest::parse_and_validate;

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let manifest_path = arguments
        .next()
        .ok_or_else(|| "usage: validate-split-manifest <manifest.json>".to_owned())?;
    if arguments.next().is_some() {
        return Err("expected exactly one manifest path".to_owned());
    }

    let bytes = fs::read(&manifest_path)
        .map_err(|error| format!("failed to read {manifest_path}: {error}"))?;
    let validated = parse_and_validate(&bytes).map_err(|error| error.to_string())?;

    println!("schema={}", validated.manifest().schema);
    println!("scenario_count={}", validated.manifest().scenarios.len());
    println!("primary_independent_unit={:?}", validated.manifest().sampling.primary_independent_unit);
    println!("manifest_sha256={}", validated.canonical_sha256());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("validate-split-manifest failed: {error}");
            ExitCode::FAILURE
        }
    }
}
