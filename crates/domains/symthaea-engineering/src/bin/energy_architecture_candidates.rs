// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministically export validated synthetic energy-architecture candidates.
//!
//! This is an interchange utility and bounded design-space baseline, not the
//! cognitive search loop or a validator of physical feasibility. Exported bytes
//! remain hypotheses and must be checked by pinned downstream evaluators.

use std::{
    env,
    error::Error,
    fs,
    io::{self, Write},
    path::PathBuf,
};

use serde::Serialize;
use symthaea_engineering::energy_architecture::{
    EnergySystemCandidate, bounded_parametric_design_space, reference_service_hub_candidates,
    validate_candidate,
};

#[derive(Debug, Default)]
struct Args {
    candidate_id: Option<String>,
    output: Option<PathBuf>,
    design_space: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--candidate-id" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--candidate-id requires a value",
                    )
                })?;
                if parsed.candidate_id.replace(value).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--candidate-id may be supplied once",
                    )
                    .into());
                }
            }
            "--output" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--output requires a path")
                })?;
                if parsed.output.replace(PathBuf::from(value)).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--output may be supplied once",
                    )
                    .into());
                }
            }
            "--design-space" => {
                if parsed.design_space {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--design-space may be supplied once",
                    )
                    .into());
                }
                parsed.design_space = true;
            }
            "-h" | "--help" => {
                println!(
                    "usage: energy_architecture_candidates [--candidate-id ID] [--design-space] [--output PATH]\n\nDefault: export the three reference templates as a JSON array. --design-space: export the bounded parametric sweep manifest (36 declared candidates plus rejection records). --candidate-id: export one reference or parametric candidate object. Writes JSON to stdout unless --output is given."
                );
                std::process::exit(0);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {arg}; use --help"),
                )
                .into());
            }
        }
    }
    if parsed.design_space && parsed.candidate_id.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--design-space cannot be combined with --candidate-id",
        )
        .into());
    }
    Ok(parsed)
}

fn export_value<T: Serialize>(value: &T) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn build_output(args: &Args) -> Result<Vec<u8>, Box<dyn Error>> {
    let reference_candidates = reference_service_hub_candidates();
    if reference_candidates.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "candidate generator returned no reference candidates",
        )
        .into());
    }
    for candidate in &reference_candidates {
        validate_candidate(candidate)?;
    }

    if args.design_space {
        let search = bounded_parametric_design_space();
        for candidate in &search.candidates {
            validate_candidate(candidate)?;
        }
        return export_value(&search);
    }

    if let Some(candidate_id) = &args.candidate_id {
        if let Some(candidate) = reference_candidates
            .iter()
            .find(|candidate| &candidate.candidate_id == candidate_id)
        {
            return export_value(candidate);
        }

        let search = bounded_parametric_design_space();
        if let Some(candidate) = search
            .candidates
            .iter()
            .find(|candidate| &candidate.candidate_id == candidate_id)
        {
            validate_candidate(candidate)?;
            return export_value(candidate);
        }

        let valid_ids = reference_candidates
            .iter()
            .map(|candidate| candidate.candidate_id.as_str())
            .chain(
                bounded_parametric_design_space()
                    .candidates
                    .iter()
                    .map(|candidate| candidate.candidate_id.as_str())
                    .collect::<Vec<_>>()
                    .into_iter(),
            )
            .collect::<Vec<_>>();
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "unknown candidate ID {candidate_id}; valid IDs: {}",
                valid_ids.join(", ")
            )
            .into());
    }

    export_value(&reference_candidates)
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
    fn default_export_is_deterministic_and_all_reference_candidates_validate() {
        let args = Args::default();
        let first = build_output(&args).expect("valid reference candidates export");
        let second = build_output(&args).expect("same templates export again");
        assert_eq!(first, second);
        let value: serde_json::Value = serde_json::from_slice(&first).expect("export is JSON");
        let candidates = value.as_array().expect("default export is an array");
        assert_eq!(candidates.len(), 3);
        for candidate in candidates {
            let parsed: EnergySystemCandidate = serde_json::from_value(candidate.clone())
                .expect("candidate has the interchange DTO shape");
            validate_candidate(&parsed).expect("exported candidate must pass structural validation");
        }
    }

    #[test]
    fn single_reference_and_parametric_candidates_export_stably() {
        for candidate_id in [
            "synthetic-pv-battery-grid",
            "synthetic-parametric-pv-1kw-battery-1kwh-battery-power-1kw-grid-1kw",
        ] {
            let args = Args {
                candidate_id: Some(candidate_id.to_owned()),
                output: None,
                design_space: false,
            };
            let bytes = build_output(&args).expect("known candidate exists");
            let candidate: EnergySystemCandidate =
                serde_json::from_slice(&bytes).expect("single export is a candidate object");
            assert_eq!(candidate.candidate_id, candidate_id);
            validate_candidate(&candidate).expect("candidate remains structurally valid");
            assert_eq!(bytes, build_output(&args).unwrap());
        }
    }

    #[test]
    fn design_space_export_keeps_axes_and_rejection_records() {
        let args = Args {
            candidate_id: None,
            output: None,
            design_space: true,
        };
        let first = build_output(&args).expect("bounded design space exports");
        let second = build_output(&args).expect("same design space exports again");
        assert_eq!(first, second);
        let value: serde_json::Value = serde_json::from_slice(&first).expect("manifest is JSON");
        assert_eq!(value["search_schema_version"], "energy-architecture-search-v0.1");
        assert_eq!(value["requested_candidate_count"], 36);
        assert_eq!(value["candidates"].as_array().unwrap().len(), 36);
        assert!(value["rejected_candidates"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unknown_candidate_id_fails_instead_of_substituting_a_default() {
        let args = Args {
            candidate_id: Some("not-a-candidate".to_owned()),
            output: None,
            design_space: false,
        };
        assert!(build_output(&args).is_err());
    }

    #[test]
    fn design_space_and_single_candidate_modes_are_mutually_exclusive() {
        // Parser tests exercise argument boundaries separately from generation.
        assert!(parse_args_from(["--design-space", "--candidate-id", "synthetic-grid-only"]).is_err());
    }

    fn parse_args_from<I, S>(args: I) -> Result<Args, Box<dyn Error>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        // Keep CLI tests deterministic by exercising the same validation
        // behavior through an explicitly supplied iterator.
        let mut parsed = Args::default();
        let mut args = args.into_iter().map(Into::into);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--design-space" => parsed.design_space = true,
                "--candidate-id" => parsed.candidate_id = args.next(),
                "--output" => parsed.output = args.next().map(PathBuf::from),
                _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "unknown argument").into()),
            }
        }
        if parsed.design_space && parsed.candidate_id.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--design-space cannot be combined with --candidate-id",
            ).into());
        }
        Ok(parsed)
    }
}
