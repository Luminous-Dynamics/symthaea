use std::{env, error::Error, io, process};

use symthaea_architecture_engineering::functional_shift_benchmark::{
    run_functional_shift_benchmark, FunctionalShiftBenchmarkRequest, FunctionalShiftScenario,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("functional-shift benchmark failed: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut scenario = FunctionalShiftScenario::OrthogonalToDeclaredFunctionals;
    let mut seed = 20_261_102_u64;
    let mut replications = 20_000_usize;
    let mut samples_per_replication = 5_000_usize;
    let mut functional_delta = 0.01_f64;
    let mut tv_delta = 0.01_f64;

    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("missing value after {argument}"),
            )
        })?;
        match argument.as_str() {
            "--scenario" => {
                scenario = match value.as_str() {
                    "orthogonal" => FunctionalShiftScenario::OrthogonalToDeclaredFunctionals,
                    "aligned" => FunctionalShiftScenario::DecisionAligned,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("unknown scenario {value:?}; use orthogonal or aligned"),
                        )
                        .into());
                    }
                };
            }
            "--seed" => seed = value.parse()?,
            "--replications" => replications = value.parse()?,
            "--samples" => samples_per_replication = value.parse()?,
            "--functional-delta" => functional_delta = value.parse()?,
            "--tv-delta" => tv_delta = value.parse()?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {argument:?}"),
                )
                .into());
            }
        }
    }

    let scenario_id = match scenario {
        FunctionalShiftScenario::OrthogonalToDeclaredFunctionals => "f01-orthogonal",
        FunctionalShiftScenario::DecisionAligned => "f02-aligned",
    };
    let request = FunctionalShiftBenchmarkRequest::new(
        scenario_id,
        scenario,
        seed,
        replications,
        samples_per_replication,
        functional_delta,
        tv_delta,
    )?;
    let report = run_functional_shift_benchmark(&request)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
