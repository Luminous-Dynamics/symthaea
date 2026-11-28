//! Run a deterministic exhaustive A0/A2 census over finite reference models.

use std::io::{self, Write};
use std::process::ExitCode;

use open_world_ansatz_microcosm::finite_campaign::run_exhaustive_finite_policy_campaign;

fn parse_budget(args: &[String]) -> Result<Option<usize>, String> {
    match args {
        [] => Ok(None),
        [flag, value] if flag == "--budget" => {
            let budget = value
                .parse::<usize>()
                .map_err(|error| format!("invalid action budget: {error}"))?;
            if budget == 0 {
                return Err("action budget must be greater than zero".to_owned());
            }
            Ok(Some(budget))
        }
        _ => Err("usage: finite_policy_campaign [--budget positive-integer]".to_owned()),
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let budget = parse_budget(&args)?;
    let report = run_exhaustive_finite_policy_campaign(budget)
        .map_err(|error| format!("finite policy campaign failed: {error}"))?;

    let stdout = io::stdout();
    let mut writer = stdout.lock();
    serde_json::to_writer_pretty(&mut writer, &report)
        .map_err(|error| format!("cannot serialize campaign report: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("cannot write report terminator: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}
