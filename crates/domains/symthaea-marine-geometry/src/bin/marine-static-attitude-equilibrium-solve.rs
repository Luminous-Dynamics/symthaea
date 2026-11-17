use std::{env, fs, path::PathBuf, process};

use serde_json::json;
use symthaea_marine_geometry::static_equilibrium::{
    EquilibriumDisposition, StaticAttitudeEquilibriumReceipt,
    StaticAttitudeEquilibriumRequest, solve_static_attitude_equilibrium,
    verify_static_attitude_equilibrium_receipt,
};

const USAGE: &str = concat!(
    "Usage: marine-static-attitude-equilibrium-solve <request.json> [--output <receipt.json>] [--verify <receipt.json>]\n",
    "Solves bounded static sinkage/pitch/roll equilibrium from the A0/A1 trial-pose evaluator.\n",
    "One converged root is a candidate, not proof of uniqueness or stability.\n",
    "Exit codes: 0 = one candidate; 1 = multiple roots observed; 2 = no converged root; 3 = request/CLI/I/O error; 4 = receipt verification failed."
);

fn write_json(output: Option<PathBuf>, value: &serde_json::Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize equilibrium output: {error}"))?;
    if let Some(path) = output {
        fs::write(&path, bytes)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut verify_path: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(value));
            }
            "--verify" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--verify requires a receipt path".to_string())?;
                verify_path = Some(PathBuf::from(value));
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
    let request_bytes =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let request: StaticAttitudeEquilibriumRequest = serde_json::from_slice(&request_bytes)
        .map_err(|error| format!("invalid static-attitude equilibrium request: {error}"))?;

    if let Some(receipt_path) = verify_path {
        let receipt_bytes = fs::read(&receipt_path)
            .map_err(|error| format!("cannot read {}: {error}", receipt_path.display()))?;
        let receipt: StaticAttitudeEquilibriumReceipt =
            serde_json::from_slice(&receipt_bytes)
                .map_err(|error| format!("invalid equilibrium receipt: {error}"))?;
        return match verify_static_attitude_equilibrium_receipt(&request, &receipt) {
            Ok(()) => {
                write_json(output, &json!({"status":"verified","receipt_digest_blake3":receipt.receipt_digest_blake3}))?;
                Ok(0)
            }
            Err(issues) => {
                write_json(output, &json!({"status":"rejected","issues":issues}))?;
                Ok(4)
            }
        };
    }

    let receipt = solve_static_attitude_equilibrium(&request)
        .map_err(|issues| format!("invalid equilibrium request: {}", issues[0].detail))?;
    let code = match receipt.disposition {
        EquilibriumDisposition::ConvergedCandidate => 0,
        EquilibriumDisposition::MultipleRootsObserved => 1,
        EquilibriumDisposition::NoConvergedRoot => 2,
    };
    let value = serde_json::to_value(&receipt)
        .map_err(|error| format!("cannot serialize equilibrium receipt: {error}"))?;
    write_json(output, &value)?;
    Ok(code)
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
