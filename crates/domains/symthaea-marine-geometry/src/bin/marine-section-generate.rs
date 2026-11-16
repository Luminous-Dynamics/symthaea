use std::{env, fs, path::PathBuf, process};

use symthaea_marine_geometry::section::{SectionHullSpec, generate_section_artifact};

const USAGE: &str = concat!(
    "Usage: marine-section-generate <section-spec.json> [--output <artifact.json>]\n",
    "Generates deterministic starboard-side keel-to-waterline section artifacts."
);

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
    let bytes =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let specification: SectionHullSpec = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid section specification: {error}"))?;
    let artifact = generate_section_artifact(&specification)
        .map_err(|violations| format!("section generation rejected: {violations:#?}"))?;
    let serialized = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("cannot serialize section artifact: {error}"))?;

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
