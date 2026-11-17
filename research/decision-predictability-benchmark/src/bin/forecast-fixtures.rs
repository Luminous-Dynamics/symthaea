use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use symthaea_decision_predictability::tournament::evaluate_fixture_csv;

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let input_path = args
        .next()
        .ok_or_else(|| "usage: forecast-fixtures <chaos-fixtures.csv>".to_owned())?;
    if args.next().is_some() {
        return Err("expected exactly one input CSV path".to_owned());
    }

    let input = fs::read_to_string(&input_path)
        .map_err(|error| format!("failed to read {input_path}: {error}"))?;
    let report = evaluate_fixture_csv(&input).map_err(|error| error.to_string())?;

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    lock.write_all(report.as_bytes())
        .map_err(|error| format!("failed to write forecast report: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("forecast-fixtures failed: {error}");
            ExitCode::FAILURE
        }
    }
}
