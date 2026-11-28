use std::env;
use std::process::ExitCode;

use open_world_ansatz_microcosm::numeric::{
    DiscoveryReport, Mechanism, ModelHypothesis, NumericWorld, ObservationLog, SensorMap,
    StructureAccess, discover, oracle_equivalence_classes,
};
use serde::Serialize;

#[derive(Serialize)]
struct OperatorOutput {
    observations: ObservationLog,
    a0_parameter_control: DiscoveryReport,
    a2_structure_discovery: DiscoveryReport,
    oracle_equivalence_class_count: usize,
}

fn parse_mechanism(value: &str) -> Result<Mechanism, String> {
    match value {
        "direct" => Ok(Mechanism::DirectResponse),
        "leaky" => Ok(Mechanism::LeakyMemory),
        _ => Err(format!(
            "unknown mechanism: {value}; expected direct or leaky"
        )),
    }
}

fn parse_optional_integer(value: &str, label: &str) -> Result<Option<i64>, String> {
    if value == "-" {
        Ok(None)
    } else {
        value
            .parse::<i64>()
            .map(Some)
            .map_err(|error| format!("invalid {label} {value}: {error}"))
    }
}

fn parse_sensor(value: &str) -> Result<SensorMap, String> {
    match value {
        "identity" => Ok(SensorMap::Identity),
        "negate" => Ok(SensorMap::Negate),
        "double" => Ok(SensorMap::Double),
        "constant_zero" => Ok(SensorMap::ConstantZero),
        _ => Err(format!("unknown sensor map: {value}")),
    }
}

fn parse_interventions(value: &str) -> Result<Vec<i64>, String> {
    let inputs = value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<i64>()
                .map_err(|error| format!("invalid intervention {item}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if inputs.is_empty() {
        return Err("at least one intervention is required".to_owned());
    }
    Ok(inputs)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: numeric_microcosm <direct|leaky> <gain> <decay|-> <initial_state|-> <identity|negate|double|constant_zero> <inputs: -1,0,1>"
                .to_owned(),
        );
    }

    // These arguments define an evaluator-side scenario. They are not an
    // agent-visible interface and must not be exposed in a blind evaluation.
    let mechanism = parse_mechanism(&args[0])?;
    let gain = args[1]
        .parse::<i64>()
        .map_err(|error| format!("invalid gain {}: {error}", args[1]))?;
    let decay = parse_optional_integer(&args[2], "decay")?;
    let initial_state = parse_optional_integer(&args[3], "initial state")?;
    let sensor = parse_sensor(&args[4])?;
    let interventions = parse_interventions(&args[5])?;

    let truth = ModelHypothesis {
        mechanism,
        gain,
        decay,
        initial_state,
        sensor,
    };
    let mut world = NumericWorld::new(truth).map_err(|error| error.to_string())?;
    let observations = world
        .run(&interventions)
        .map_err(|error| error.to_string())?;

    let a0_parameter_control = discover(&observations, StructureAccess::Supplied(mechanism))
        .map_err(|error| error.to_string())?;
    let a2_structure_discovery =
        discover(&observations, StructureAccess::Withheld).map_err(|error| error.to_string())?;
    let oracle_equivalence_class_count = oracle_equivalence_classes(interventions.len())
        .map_err(|error| error.to_string())?
        .len();

    let output = OperatorOutput {
        observations,
        a0_parameter_control,
        a2_structure_discovery,
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
