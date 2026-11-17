use std::{env, fs, path::PathBuf, process};

use serde::Serialize;
use symthaea_marine_geometry::static_attitude::{
    StaticAttitudeEvaluationReceipt, StaticAttitudeEvaluationRequest, StaticAttitudeIssue,
    evaluate_static_attitude, verify_static_attitude_evaluation_receipt,
};

const USAGE: &str = concat!(
    "Usage: marine-static-attitude-evaluate <attitude-request.json> [--output <report.json>] [--verify <receipt.json>]\n",
    "Evaluates one trial upright/pitch/roll waterplane using full-envelope sectional geometry.\n",
    "This is a trial force/moment evaluator, not an equilibrium solver and not a stability approval.\n",
    "Exit codes: 0 = evaluated or receipt verified; 1 = typed validation/integrity rejection; 2 = CLI, I/O, or JSON parse failure."
);

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
enum CliReport {
    Evaluated {
        receipt: Box<StaticAttitudeEvaluationReceipt>,
    },
    Rejected {
        issues: Vec<StaticAttitudeIssue>,
    },
    Verified {
        receipt_schema_version: u32,
        receipt_digest_blake3: String,
        displacement_volume_m3: f64,
        displaced_mass_tonnes: f64,
        mass_residual_tonnes: f64,
        moment_residual_1_nm: f64,
        moment_residual_2_nm: f64,
    },
}

fn write_report(output: Option<PathBuf>, report: &CliReport) -> Result<(), String> {
    let serialized = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("cannot serialize static-attitude report: {error}"))?;
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
    let request: StaticAttitudeEvaluationRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid static-attitude request: {error}"))?;

    if let Some(receipt_path) = verify {
        let receipt_bytes = fs::read(&receipt_path)
            .map_err(|error| format!("cannot read {}: {error}", receipt_path.display()))?;
        let receipt: StaticAttitudeEvaluationReceipt = serde_json::from_slice(&receipt_bytes)
            .map_err(|error| format!("invalid static-attitude receipt: {error}"))?;
        return match verify_static_attitude_evaluation_receipt(&request, &receipt) {
            Ok(()) => {
                write_report(
                    output,
                    &CliReport::Verified {
                        receipt_schema_version: receipt.receipt_schema_version,
                        receipt_digest_blake3: receipt.receipt_digest_blake3,
                        displacement_volume_m3: receipt.displacement_volume_m3,
                        displaced_mass_tonnes: receipt.displaced_mass_tonnes,
                        mass_residual_tonnes: receipt.residuals.mass_residual_tonnes,
                        moment_residual_1_nm: receipt.residuals.moment_residual_1_nm,
                        moment_residual_2_nm: receipt.residuals.moment_residual_2_nm,
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

    match evaluate_static_attitude(&request) {
        Ok(receipt) => {
            write_report(
                output,
                &CliReport::Evaluated {
                    receipt: Box::new(receipt),
                },
            )?;
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
