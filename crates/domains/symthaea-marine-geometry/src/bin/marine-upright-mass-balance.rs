use std::{env, fs, path::PathBuf, process};

use serde::Serialize;
use symthaea_marine_geometry::sectional_reference::{
    SectionalReferenceIssue, UprightMassBalanceReceipt, UprightMassBalanceRequest,
    solve_upright_mass_balance, verify_upright_mass_balance_receipt,
};

const USAGE: &str = concat!(
    "Usage: marine-upright-mass-balance <mass-balance-request.json> [--output <report.json>] [--verify <receipt.json>]\n",
    "Solves upright waterplane elevation for an explicit mass/density using bracketed bisection.\n",
    "This is sinkage-only: it does not solve trim, heel, tank free-surface, KG, or stability.\n",
    "Exit codes: 0 = solved or independently verified; 1 = typed validation/solve/integrity rejection; 2 = CLI, I/O, or JSON parse failure."
);

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
enum CliReport {
    Solved {
        receipt: Box<UprightMassBalanceReceipt>,
    },
    Rejected {
        issues: Vec<SectionalReferenceIssue>,
    },
    Verified {
        receipt_schema_version: u32,
        receipt_digest_blake3: String,
        equilibrium_waterplane_z_m: f64,
        achieved_mass_tonnes: f64,
        mass_residual_tonnes: f64,
        iteration_count: u32,
    },
}

fn write_report(output: Option<PathBuf>, report: &CliReport) -> Result<(), String> {
    let serialized = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("cannot serialize mass-balance report: {error}"))?;
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
    let request: UprightMassBalanceRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid upright mass-balance request: {error}"))?;

    if let Some(receipt_path) = verify {
        let receipt_bytes = fs::read(&receipt_path)
            .map_err(|error| format!("cannot read {}: {error}", receipt_path.display()))?;
        let receipt: UprightMassBalanceReceipt = serde_json::from_slice(&receipt_bytes)
            .map_err(|error| format!("invalid upright mass-balance receipt: {error}"))?;
        return match verify_upright_mass_balance_receipt(&request, &receipt) {
            Ok(()) => {
                write_report(
                    output,
                    &CliReport::Verified {
                        receipt_schema_version: receipt.receipt_schema_version,
                        receipt_digest_blake3: receipt.receipt_digest_blake3,
                        equilibrium_waterplane_z_m: receipt.equilibrium.waterplane_z_m,
                        achieved_mass_tonnes: receipt.equilibrium.displaced_mass_tonnes,
                        mass_residual_tonnes: receipt.mass_residual_tonnes,
                        iteration_count: receipt.iteration_count,
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

    match solve_upright_mass_balance(&request) {
        Ok(receipt) => {
            write_report(
                output,
                &CliReport::Solved {
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
