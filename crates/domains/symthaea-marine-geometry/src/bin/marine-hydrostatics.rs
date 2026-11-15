use std::{env, fs, path::PathBuf, process};

use serde::Deserialize;
use symthaea_marine_geometry::hydrostatics::{
    calculate_hydrostatic_properties, HydrostaticProperties,
};
use symthaea_marine_geometry::loft::LoftedHullArtifact;

const USAGE: &str = concat!(
    "Usage: marine-hydrostatics <hydrostatics-request.json> [--output <properties.json>]\n",
    "The request contains { lofted_artifact, water_density_kg_m3 }."
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HydrostaticsRequest {
    lofted_artifact: LoftedHullArtifact,
    water_density_kg_m3: f64,
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(path));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(0);
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {value}\n{USAGE}"));
            }
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("exactly one input file is accepted\n{USAGE}"));
                }
            }
        }
    }

    let input = input.ok_or_else(|| USAGE.to_string())?;
    let bytes = fs::read(&input)
        .map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let request: HydrostaticsRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid hydrostatics request: {error}"))?;
    let properties: HydrostaticProperties = calculate_hydrostatic_properties(
        &request.lofted_artifact,
        request.water_density_kg_m3,
    )
    .map_err(|violations| format!("hydrostatics rejected: {violations:#?}"))?;
    let serialized = serde_json::to_vec_pretty(&properties)
        .map_err(|error| format!("cannot serialize hydrostatic properties: {error}"))?;

    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }

    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}
