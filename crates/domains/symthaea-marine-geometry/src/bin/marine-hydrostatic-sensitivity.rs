use std::{env, fs, path::PathBuf, process};

use serde::Deserialize;
use symthaea_marine_geometry::hydrostatics::HydrostaticSensitivity;
use symthaea_marine_geometry::sensitivity::{
    calculate_hydrostatic_sensitivity, verify_sensitivity_result, HydrostaticObservable,
    SectionParameter,
};
use symthaea_marine_geometry::{section::SectionHullSpec, AdmissionProfile};

const USAGE: &str = concat!(
    "Usage: marine-hydrostatic-sensitivity <sensitivity-request.json>\n",
    "The request contains section_spec, profile, parameter, step, observable, ",
    "water_density_kg_m3 and water_density_source."
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SensitivityRequest {
    section_spec: SectionHullSpec,
    profile: AdmissionProfile,
    parameter: SectionParameter,
    step: f64,
    observable: HydrostaticObservable,
    water_density_kg_m3: f64,
    water_density_source: String,
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let input = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(format!("exactly one request input is accepted\n{USAGE}"));
    }
    let input = input.ok_or_else(|| USAGE.to_string())?;
    let bytes = fs::read(&input)
        .map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let request: SensitivityRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid sensitivity request: {error}"))?;
    let result: HydrostaticSensitivity = calculate_hydrostatic_sensitivity(
        &request.section_spec,
        &request.profile,
        request.parameter,
        request.step,
        request.observable,
        request.water_density_kg_m3,
        &request.water_density_source,
    )
    .map_err(|violations| format!("sensitivity rejected: {violations:#?}"))?;
    verify_sensitivity_result(&result)
        .map_err(|error| format!("sensitivity receipt integrity check failed: {error:#?}"))?;
    let serialized = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("cannot serialize sensitivity result: {error}"))?;
    println!("{}", String::from_utf8_lossy(&serialized));
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
