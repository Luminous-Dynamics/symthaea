use std::{env, fs, path::PathBuf, process};

use serde::Deserialize;
use symthaea_marine_geometry::cad_backend::{
    evaluate_backend_capabilities, BackendCapabilityManifest, CapabilityEvaluation,
    CapabilityRequirementProfile,
};

const USAGE: &str = concat!(
    "Usage: marine-cad-backend-evaluate <evaluation-request.json>\n",
    "The request contains { backend_manifest, requirement_profile }."
);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationRequest {
    backend_manifest: BackendCapabilityManifest,
    requirement_profile: CapabilityRequirementProfile,
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
    let request: EvaluationRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid backend evaluation request: {error}"))?;
    let evaluation: CapabilityEvaluation = evaluate_backend_capabilities(
        &request.backend_manifest,
        &request.requirement_profile,
    )
    .map_err(|violations| format!("backend manifest invalid: {violations:#?}"))?;
    let serialized = serde_json::to_vec_pretty(&evaluation)
        .map_err(|error| format!("cannot serialize backend evaluation: {error}"))?;
    println!("{}", String::from_utf8_lossy(&serialized));

    Ok(match evaluation.disposition {
        symthaea_marine_geometry::cad_backend::CapabilityDisposition::QualifiedWithinProfile => 0,
        symthaea_marine_geometry::cad_backend::CapabilityDisposition::Incomplete => 1,
        symthaea_marine_geometry::cad_backend::CapabilityDisposition::Rejected => 2,
    })
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(3);
        }
    }
}
