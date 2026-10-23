// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministically export validated synthetic energy-architecture candidates.
//!
//! This is an interchange utility, not the cognitive search loop or a validator
//! of physical feasibility. Exported bytes remain candidate hypotheses and must
//! be validated by downstream pinned contracts/evaluators.

use std::{env, error::Error, fs, io::{self, Write}, path::PathBuf};

use serde::Serialize;
use symthaea_engineering::energy_architecture::{
    EnergySystemCandidate, reference_service_hub_candidates, validate_candidate,
};

#[derive(Debug, Default)]
struct Args {
    candidate_id: Option<String>,
    output: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--candidate-id" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--candidate-id requires a value")
                })?;
                if parsed.candidate_id.replace(value).is_some() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "--candidate-id may be supplied once").into());
                }
            }
            "--output" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--output requires a path")
                })?;
                if parsed.output.replace(PathBuf::from(value)).is_some() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "--output may be supplied once").into());
                }
            }
            "-h" | "--help" => {
                println!("usage: energy_architecture_candidates [--candidate-id ID] [--output PATH]\n\nWithout --candidate-id, exports all candidates as a JSON array. With --candidate-id, exports one candidate object. Writes JSON to stdout unless --output is given.");
                std::process::exit(0);
            }
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown argument: {arg}; use --help"),
            ).into()),
        }
    }
    Ok(parsed)
}

fn export_value<T: Serialize>(value: &T) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn build_output(args: &Args) -> Result<Vec<u8>, Box<dyn Error>> {
    let candidates = reference_service_hub_candidates();
    if candidates.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "candidate generator returned no candidates").into());
    }
    for candidate in &candidates {
        validate_candidate(candidate)?;
    }

    if let Some(candidate_id) = &args.candidate_id {
        let candidate = candidates.iter().find(|item| &item.candidate_id == candidate_id)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("unknown candidate ID {candidate_id}; valid IDs: {}", candidates.iter().map(|item| item.candidate_id.as_str()).collect::<Vec<_>>().join(", ")),
            ))?;
        export_value(candidate)
    } else {
        export_value(&candidates)
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let output = build_output(&args)?;
    match args.output {
        Some(path) => fs::write(path, output)?,
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(&output)?;
            handle.flush()?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy architecture candidate export failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_export_is_deterministic_and_all_candidates_validate() {
        let args = Args::default();
        let first = build_output(&args).expect("valid template candidates export");
        let second = build_output(&args).expect("same templates export again");
        assert_eq!(first, second);
        let value: serde_json::Value = serde_json::from_slice(&first).expect("export is JSON");
        let candidates = value.as_array().expect("default export is an array");
        assert_eq!(candidates.len(), 3);
        for candidate in candidates {
            let parsed: EnergySystemCandidate = serde_json::from_value(candidate.clone()).expect("candidate has the interchange DTO shape");
            validate_candidate(&parsed).expect("exported candidate must pass structural validation");
        }
    }

    #[test]
    fn single_candidate_export_is_one_object_and_stable() {
        let args = Args { candidate_id: Some("synthetic-pv-battery-grid".to_owned()), output: None };
        let bytes = build_output(&args).expect("known candidate exists");
        let candidate: EnergySystemCandidate = serde_json::from_slice(&bytes).expect("single export is a candidate object");
        assert_eq!(candidate.candidate_id, "synthetic-pv-battery-grid");
        validate_candidate(&candidate).expect("single exported candidate remains structurally valid");
        assert_eq!(bytes, build_output(&args).unwrap());
    }

    #[test]
    fn unknown_candidate_id_fails_instead_of_substituting_a_default() {
        let args = Args { candidate_id: Some("not-a-candidate".to_owned()), output: None };
        assert!(build_output(&args).is_err());
    }
}
