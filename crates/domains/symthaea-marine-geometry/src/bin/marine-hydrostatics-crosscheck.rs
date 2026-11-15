use std::{env, fs, path::PathBuf, process};

use serde::{Deserialize, Serialize};
use symthaea_marine_geometry::{
    hydrostatics::{calculate_hydrostatic_properties, HydrostaticProperties},
    loft::LoftedHullArtifact,
    section::SectionFamilyArtifact,
    sectional_hydrostatics::{
        calculate_sectional_reference, compare_with_polyhedral_hydrostatics,
        SectionalCrossCheckReport, SectionalCrossCheckTolerance,
        SectionalHydrostaticsReference,
    },
};

const USAGE: &str = concat!(
    "Usage: marine-hydrostatics-crosscheck <request.json> [--output <report.json>]\n",
    "Exit codes: 0 consistent, 3 inconsistent, 4 incomparable, 2 invalid input/error.\n",
    "Request contains { section_family, lofted_artifact, water_density_kg_m3, tolerance }.\n"
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossCheckRequest {
    section_family: SectionFamilyArtifact,
    lofted_artifact: LoftedHullArtifact,
    water_density_kg_m3: f64,
    tolerance: SectionalCrossCheckTolerance,
}

#[derive(Serialize)]
struct CrossCheckOutput {
    sectional_reference: SectionalHydrostaticsReference,
    polyhedral_reference: HydrostaticProperties,
    comparison: SectionalCrossCheckReport,
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
    let request: CrossCheckRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid sectional cross-check request: {error}"))?;

    let polyhedral_reference = calculate_hydrostatic_properties(
        &request.lofted_artifact,
        request.water_density_kg_m3,
    )
    .map_err(|errors| format!("polyhedral hydrostatics rejected: {errors:#?}"))?;

    let sectional_reference = calculate_sectional_reference(
        &request.section_family,
        &request.lofted_artifact,
        request.water_density_kg_m3,
    )
    .map_err(|errors| format!("sectional hydrostatics rejected: {errors:#?}"))?;

    let comparison = compare_with_polyhedral_hydrostatics(
        &sectional_reference,
        &polyhedral_reference,
        &request.tolerance,
    )
    .map_err(|error| format!("cross-check could not be evaluated: {error:#?}"))?;

    let exit_code = match comparison.disposition {
        symthaea_marine_geometry::sectional_hydrostatics::CrossCheckDisposition::Consistent => 0,
        symthaea_marine_geometry::sectional_hydrostatics::CrossCheckDisposition::Inconsistent => 3,
        symthaea_marine_geometry::sectional_hydrostatics::CrossCheckDisposition::Incomparable => 4,
    };

    let result = CrossCheckOutput {
        sectional_reference,
        polyhedral_reference,
        comparison,
    };
    let serialized = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("cannot serialize cross-check report: {error}"))?;

    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }

    Ok(exit_code)
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
