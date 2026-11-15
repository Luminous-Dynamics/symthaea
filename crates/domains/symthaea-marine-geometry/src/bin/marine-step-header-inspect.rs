use std::{env, fs, path::PathBuf, process};

use serde_json::json;
use symthaea_marine_geometry::step_header::{
    inspect_step_part21_header, StepHeaderProfile,
};

const USAGE: &str = concat!(
    "Usage: marine-step-header-inspect <step-file> <profile.json>\n",
    "Checks the Part 21 envelope and exact FILE_SCHEMA identifiers only."
);

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let step_path = args.next().map(PathBuf::from);
    let profile_path = args.next().map(PathBuf::from);
    if args.next().is_some() || step_path.is_none() || profile_path.is_none() {
        return Err(USAGE.to_string());
    }
    let step_path = step_path.unwrap();
    let profile_path = profile_path.unwrap();
    let step_bytes = fs::read(&step_path)
        .map_err(|error| format!("cannot read STEP file {}: {error}", step_path.display()))?;
    let profile_bytes = fs::read(&profile_path)
        .map_err(|error| format!("cannot read profile {}: {error}", profile_path.display()))?;
    let profile: StepHeaderProfile = serde_json::from_slice(&profile_bytes)
        .map_err(|error| format!("invalid STEP header profile: {error}"))?;

    match inspect_step_part21_header(&step_bytes, &profile) {
        Ok(inspection) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&inspection)
                    .map_err(|error| format!("cannot serialize STEP inspection: {error}"))?
            );
            Ok(0)
        }
        Err(violations) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "profile_id": profile.id,
                    "status": "not-qualified",
                    "violations": violations
                }))
                .map_err(|error| format!("cannot serialize STEP violations: {error}"))?
            );
            Ok(1)
        }
    }
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
