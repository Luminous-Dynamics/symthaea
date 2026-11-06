use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_causal_reference::{validate_fixture_file, validate_manifest_json};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let command = arguments
        .next()
        .ok_or_else(usage)?
        .to_string_lossy()
        .into_owned();
    let path = arguments.next().ok_or_else(usage)?;

    if arguments.next().is_some() {
        return Err(usage());
    }

    let input = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.to_string_lossy()))?;

    match command.as_str() {
        "fixtures" => {
            let report = validate_fixture_file(&input).map_err(|error| error.to_string())?;
            println!(
                "PASS: {} inert causal-attribution fixtures validated.",
                report.fixture_ids.len()
            );
            for fixture_id in report.fixture_ids {
                println!("  {fixture_id}");
            }
            println!(
                "No candidate code was executed; this is not causal evidence or promotion authority."
            );
        }
        "manifest" => {
            validate_manifest_json(&input)?;
            println!("PASS: intervention manifest is structurally consistent.");
            println!(
                "Referenced digests and evidence are not authenticated by this structural check."
            );
        }
        _ => {
            return Err(format!("unknown command '{command}'\n{}", usage()));
        }
    }

    Ok(())
}

fn usage() -> String {
    "usage: sym-rsi-causal-reference <fixtures|manifest> <json-path>".to_owned()
}
