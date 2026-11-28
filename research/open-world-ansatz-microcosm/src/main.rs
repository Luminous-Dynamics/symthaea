use std::env;
use std::process::ExitCode;

use open_world_ansatz_microcosm::{
    DiscoveryReport, HiddenWorld, MeasurementMap, Mechanism, ObservationLog, StructureAccess,
    discover, oracle_equivalence_classes,
};
use serde::Serialize;

#[derive(Serialize)]
struct OperatorOutput {
    observations: ObservationLog,
    discovery: DiscoveryReport,
    oracle_equivalence_class_count: usize,
}

fn parse_mechanism(value: &str) -> Result<Mechanism, String> {
    match value {
        "direct" => Ok(Mechanism::Direct),
        "hidden_lag" => Ok(Mechanism::HiddenLag),
        _ => Err(format!("unknown mechanism: {value}")),
    }
}

fn parse_measurement_map(value: &str) -> Result<MeasurementMap, String> {
    match value {
        "identity" => Ok(MeasurementMap::Identity),
        "invert" => Ok(MeasurementMap::Invert),
        "constant_zero" => Ok(MeasurementMap::ConstantZero),
        _ => Err(format!("unknown measurement map: {value}")),
    }
}

fn parse_initial_state(value: &str) -> Result<Option<bool>, String> {
    match value {
        "-" => Ok(None),
        "0" => Ok(Some(false)),
        "1" => Ok(Some(true)),
        _ => Err(format!(
            "initial hidden state must be -, 0 or 1; got {value}"
        )),
    }
}

fn parse_interventions(value: &str) -> Result<Vec<bool>, String> {
    let values = value
        .split(',')
        .map(|item| match item.trim() {
            "0" => Ok(false),
            "1" => Ok(true),
            other => Err(format!("intervention bits must be 0 or 1; got {other}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("at least one intervention is required".to_owned());
    }
    Ok(values)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: open-world-ansatz-microcosm <direct|hidden_lag> <identity|invert|constant_zero> <-|0|1> <bits: 1,0,0> <a0|a2>".to_owned());
    }

    // The first three arguments are evaluator-side scenario configuration.
    // Never pass them to the agent-facing process in a blind evaluation.
    let mechanism = parse_mechanism(&args[0])?;
    let measurement_map = parse_measurement_map(&args[1])?;
    let initial_hidden_state = parse_initial_state(&args[2])?;
    let interventions = parse_interventions(&args[3])?;
    let mut world = HiddenWorld::new(mechanism, measurement_map, initial_hidden_state)
        .map_err(|error| error.to_string())?;
    let observations = world.run(&interventions);

    let access = match args[4].as_str() {
        "a0" => StructureAccess::Supplied(mechanism),
        "a2" => StructureAccess::Withheld,
        other => return Err(format!("unknown access mode: {other}; expected a0 or a2")),
    };
    let discovery = discover(&observations, access).map_err(|error| error.to_string())?;
    let oracle_equivalence_class_count = oracle_equivalence_classes(interventions.len())
        .map_err(|error| error.to_string())?
        .len();

    let output = OperatorOutput {
        observations,
        discovery,
        oracle_equivalence_class_count,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize operator report: {error}"))?
    );
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
