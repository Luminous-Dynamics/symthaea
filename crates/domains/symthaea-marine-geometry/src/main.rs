use std::{env, fs, path::PathBuf, process};

use symthaea_marine_geometry::{admit, compute_geometry_digest, CandidateInput, Disposition};

fn usage() -> &'static str {
    "Usage: marine-geometry-admit [--digest-only] <candidate.json> [--report <report.json>]\n\
     --digest-only computes the normalized geometry digest; it does not admit the candidate."
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut digest_only = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--digest-only" => digest_only = true,
            "--report" => {
                let value = args.next().ok_or_else(|| "--report requires a path".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(0);
            }
            value if value.starts_with('-') => return Err(format!("unknown option: {value}\n{}", usage())),
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("exactly one candidate input is accepted\n{}", usage()));
                }
            }
        }
    }

    let input_path = input.ok_or_else(|| usage().to_string())?;
    let bytes = fs::read(&input_path)
        .map_err(|e| format!("cannot read {}: {e}", input_path.display()))?;
    let candidate: CandidateInput = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid candidate JSON: {e}"))?;

    if digest_only {
        let digest = compute_geometry_digest(&candidate.geometry)
            .map_err(|e| format!("cannot serialize geometry for digest: {e}"))?;
        println!("{digest}");
        return Ok(0);
    }

    let report = admit(&candidate);
    let output = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("cannot serialize admission report: {e}"))?;

    if let Some(path) = report_path {
        fs::write(&path, &output)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&output));
    }

    Ok(if report.disposition == Disposition::AdmissibleForProfile { 0 } else { 1 })
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
