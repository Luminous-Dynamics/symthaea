use std::env;
use std::fs;
use std::process;

use symthaea_household_service_evaluator::batch::{evaluate_batch_from_paths, sha256_hex};

fn main() {
    if let Err(error) = run() {
        eprintln!("household-service-evaluator: {error}");
        eprintln!(
            "usage: symthaea-household-service-evaluator --water PATH --repair PATH [--output PATH|-] [--digest-output PATH]"
        );
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut water_path = None;
    let mut repair_path = None;
    let mut output_path = None;
    let mut digest_output_path = None;
    let mut args = env::args().skip(1);

    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--water" if water_path.is_none() => water_path = Some(value),
            "--repair" if repair_path.is_none() => repair_path = Some(value),
            "--output" if output_path.is_none() => output_path = Some(value),
            "--digest-output" if digest_output_path.is_none() => digest_output_path = Some(value),
            "--water" | "--repair" | "--output" | "--digest-output" => {
                return Err(format!("duplicate argument: {argument}"));
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    let water_path = water_path.ok_or_else(|| "missing --water PATH".to_owned())?;
    let repair_path = repair_path.ok_or_else(|| "missing --repair PATH".to_owned())?;
    let output_path = output_path.unwrap_or_else(|| "-".to_owned());

    let report =
        evaluate_batch_from_paths(&water_path, &repair_path).map_err(|error| error.to_string())?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("could not serialize batch report: {error}"))?;
    let output_digest = sha256_hex(rendered.as_bytes());

    if output_path == "-" {
        print!("{rendered}");
    } else {
        fs::write(&output_path, rendered.as_bytes())
            .map_err(|error| format!("could not write {}: {error}", output_path))?;
    }

    let digest_path = digest_output_path.or_else(|| {
        if output_path == "-" {
            None
        } else {
            Some(format!("{output_path}.sha256"))
        }
    });
    if let Some(path) = digest_path {
        fs::write(&path, format!("{output_digest}\n"))
            .map_err(|error| format!("could not write {}: {error}", path))?;
    } else {
        eprintln!("sha256:{output_digest}");
    }

    Ok(())
}
