use std::{env, fs, path::PathBuf, process};

use serde::Serialize;
use symthaea_marine_geometry::sectional_reference::{
    SectionalReferenceIssue, SectionalReferenceRequest, evaluate_loading_sweep,
};

const USAGE: &str = concat!(
    "Usage: marine-sectional-hydrostatics-reference <loading-sweep-request.json> [--output <receipt.json>]\n",
    "Computes an independent piecewise-linear sectional-area reference for upright horizontal waterplanes.\n",
    "This does not generate displacement meshes or approve vessel stability.\n",
    "Exit codes: 0 = sweep evaluated; 1 = typed validation/integration rejection; 2 = CLI, I/O, or JSON parse failure."
);

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
enum CliReport {
    Evaluated {
        receipt: symthaea_marine_geometry::sectional_reference::LoadingSweepReceipt,
    },
    Rejected {
        issues: Vec<SectionalReferenceIssue>,
    },
}

fn write_report(output: Option<PathBuf>, report: &CliReport) -> Result<(), String> {
    let serialized = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("cannot serialize sectional reference report: {error}"))?;
    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }
    Ok(())
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
                    return Err(format!("exactly one request file is accepted\n{USAGE}"));
                }
            }
        }
    }

    let input = input.ok_or_else(|| USAGE.to_string())?;
    let bytes =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let request: SectionalReferenceRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid loading-sweep request: {error}"))?;

    match evaluate_loading_sweep(&request) {
        Ok(receipt) => {
            write_report(output, &CliReport::Evaluated { receipt })?;
            Ok(0)
        }
        Err(issues) => {
            write_report(output, &CliReport::Rejected { issues })?;
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
