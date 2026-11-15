use std::{env, fs, path::PathBuf, process};

use serde::Deserialize;
use symthaea_marine_geometry::{
    loft::loft_waterplane_capped_hull, AdmissionProfile, Disposition,
};
use symthaea_marine_geometry::section::SectionFamilyArtifact;

const USAGE: &str = concat!(
    "Usage: marine-waterplane-loft <loft-request.json> [--output <loft-artifact.json>]\n",
    "The request contains { section_family, profile }."
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoftRequest {
    section_family: SectionFamilyArtifact,
    profile: AdmissionProfile,
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
    let request: LoftRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid loft request: {error}"))?;
    let artifact = loft_waterplane_capped_hull(&request.section_family, request.profile)
        .map_err(|violations| format!("loft rejected: {violations:#?}"))?;
    let disposition = artifact.admission_report.disposition;
    let serialized = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("cannot serialize loft artifact: {error}"))?;

    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }

    Ok(if disposition == Disposition::AdmissibleForProfile {
        0
    } else {
        1
    })
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
