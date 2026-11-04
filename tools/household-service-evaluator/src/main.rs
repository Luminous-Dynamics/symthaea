use std::env;
use std::fs;
use std::process;

use symthaea_household_service_evaluator::batch::{
    BatchOutputEnvelope, evaluate_batch_from_paths_with_source_commit,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("household-service-evaluator: {error}");
        eprintln!(
            "usage: symthaea-household-service-evaluator --water PATH --repair PATH [--source-commit SHA] [--output PATH|-]"
        );
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut water_path = None;
    let mut repair_path = None;
    let mut output_path = None;
    let mut source_commit = None;
    let mut args = env::args().skip(1);

    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--water" if water_path.is_none() => water_path = Some(value),
            "--repair" if repair_path.is_none() => repair_path = Some(value),
            "--output" if output_path.is_none() => output_path = Some(value),
            "--source-commit" if source_commit.is_none() => source_commit = Some(value),
            "--water" | "--repair" | "--output" | "--source-commit" => {
                return Err(format!("duplicate argument: {argument}"));
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    let water_path = water_path.ok_or_else(|| "missing --water PATH".to_owned())?;
    let repair_path = repair_path.ok_or_else(|| "missing --repair PATH".to_owned())?;
    let output_path = output_path.unwrap_or_else(|| "-".to_owned());

    let report = evaluate_batch_from_paths_with_source_commit(
        &water_path,
        &repair_path,
        source_commit.as_deref(),
    )
    .map_err(|error| error.to_string())?;
    let envelope = BatchOutputEnvelope::from_report(report).map_err(|error| error.to_string())?;
    let rendered = serde_json::to_string_pretty(&envelope)
        .map_err(|error| format!("could not serialize batch output envelope: {error}"))?;

    if output_path == "-" {
        println!("{rendered}");
    } else {
        fs::write(&output_path, rendered)
            .map_err(|error| format!("could not write {}: {error}", output_path))?;
    }

    Ok(())
}
