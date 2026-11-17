use std::{env, fs, path::PathBuf, process};

use serde::Serialize;
use symthaea_marine_geometry::sectional_reference::{
    LoadingSweepReceipt, SectionalReferenceIssue, SectionalReferenceRequest,
    evaluate_loading_sweep, verify_loading_sweep_receipt,
};

const USAGE: &str = concat!(
    "Usage: marine-sectional-hydrostatics-reference <loading-sweep-request.json> [--output <report.json>] [--verify <receipt.json>]\n",
    "Computes an independent piecewise-linear sectional-area reference for upright horizontal waterplanes.\n",
    "This does not generate displacement meshes or approve vessel stability.\n",
    "Use --verify to independently recompute a supplied receipt against the exact request.\n",
    "Exit codes: 0 = evaluated or verified; 1 = typed validation/integration/integrity rejection; 2 = CLI, I/O, or JSON parse failure."
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
    Verified {
        receipt_schema_version: u32,
        sweep_digest_blake3: String,
        result_count: usize,
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
    let mut verify: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(path));
            }
            "--verify" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--verify requires a receipt path".to_string())?;
                verify = Some(PathBuf::from(path));
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

    if let Some(receipt_path) = verify {
        let receipt_bytes = fs::read(&receipt_path)
            .map_err(|error| format!("cannot read {}: {error}", receipt_path.display()))?;
        let receipt: LoadingSweepReceipt = serde_json::from_slice(&receipt_bytes)
            .map_err(|error| format!("invalid sectional reference receipt: {error}"))?;
        return match verify_loading_sweep_receipt(&request, &receipt) {
            Ok(()) => {
                write_report(
                    output,
                    &CliReport::Verified {
                        receipt_schema_version: receipt.receipt_schema_version,
                        sweep_digest_blake3: receipt.sweep_digest_blake3,
                        result_count: receipt.results.len(),
                    },
                )?;
                Ok(0)
            }
            Err(issues) => {
                write_report(output, &CliReport::Rejected { issues })?;
                Ok(1)
            }
        };
    }

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
