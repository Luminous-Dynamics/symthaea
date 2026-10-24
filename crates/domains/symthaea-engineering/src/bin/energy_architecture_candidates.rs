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
    DesignSpaceSearchResult, EnergySystemCandidate, bounded_parametric_design_space,
    reference_service_hub_candidates, validate_candidate,
};

#[derive(Debug, Default)]
struct Args {
    candidate_id: Option<String>,
    output: Option<PathBuf>,
    candidate_output_dir: Option<PathBuf>,
    design_space: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I, S>(values: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut parsed = Args::default();
    let mut args = values.into_iter().map(Into::into);
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
            "--candidate-output-dir" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--candidate-output-dir requires a directory",
                    )
                })?;
                if parsed
                    .candidate_output_dir
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--candidate-output-dir may be supplied once",
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
                    "usage: energy_architecture_candidates [--candidate-id ID] [--design-space] [--candidate-output-dir DIR] [--output PATH]\n\nDefault: export the three reference templates as a JSON array. --design-space: export the bounded parametric sweep manifest (36 declared candidates plus rejection records). --candidate-id: export one reference or parametric candidate object. With --design-space, --candidate-output-dir materializes all 36 candidate objects in stable <DIR>/<candidate-id>/candidate.json paths using the same generated in-memory candidates. Writes the manifest/candidate to stdout unless --output is given."
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
    if parsed.candidate_output_dir.is_some() && !parsed.design_space {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--candidate-output-dir requires --design-space",
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

fn write_design_space_candidates(
    search: &DesignSpaceSearchResult,
    output_dir: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    if !search.rejected_candidates.is_empty()
        || search.candidates.len() != search.requested_candidate_count
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "refusing partial design-space export: requested={}, emitted={}, rejected={}",
                search.requested_candidate_count,
                search.candidates.len(),
                search.rejected_candidates.len()
            ),
        )
        .into());
    }

    for candidate in &search.candidates {
        validate_candidate(candidate)?;
    }
    fs::create_dir_all(output_dir)?;
    for candidate in &search.candidates {
        let candidate_dir = output_dir.join(&candidate.candidate_id);
        fs::create_dir_all(&candidate_dir)?;
        fs::write(candidate_dir.join("candidate.json"), export_value(candidate)?)?;
    }
    Ok(())
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
                search
                    .candidates
                    .iter()
                    .map(|candidate| candidate.candidate_id.as_str()),
            )
            .collect::<Vec<_>>();
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "unknown candidate ID {candidate_id}; valid IDs: {}",
                valid_ids.join(", ")
            ),
        )
        .into());
    }

    export_value(&reference_candidates)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let output = if let Some(output_dir) = &args.candidate_output_dir {
        let search = bounded_parametric_design_space();
        write_design_space_candidates(&search, output_dir)?;
        export_value(&search)?
    } else {
        build_output(&args)?
    };

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
            validate_candidate(&parsed)
                .expect("exported candidate must pass structural validation");
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
                candidate_output_dir: None,
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
            candidate_output_dir: None,
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
    fn candidate_materialization_writes_manifest_equivalent_objects() {
        let search = bounded_parametric_design_space();
        let output_dir = env::temp_dir().join(format!(
            "symthaea-energy-design-space-export-test-{}",
            std::process::id()
        ));
        if output_dir.exists() {
            fs::remove_dir_all(&output_dir).expect("remove stale test output");
        }
        write_design_space_candidates(&search, &output_dir)
            .expect("complete design space materializes");
        for candidate in &search.candidates {
            let bytes = fs::read(output_dir.join(&candidate.candidate_id).join("candidate.json"))
                .expect("exported candidate file exists");
            let parsed: EnergySystemCandidate =
                serde_json::from_slice(&bytes).expect("materialized candidate is JSON");
            assert_eq!(&parsed, candidate, "materialized candidate differs from manifest object");
        }
        fs::remove_dir_all(output_dir).expect("clean temporary test output");
    }

    #[test]
    fn unknown_candidate_id_fails_instead_of_substituting_a_default() {
        let args = Args {
            candidate_id: Some("not-a-candidate".to_owned()),
            output: None,
            candidate_output_dir: None,
            design_space: false,
        };
        assert!(build_output(&args).is_err());
    }

    #[test]
    fn design_space_output_directory_requires_design_space_mode() {
        let args = parse_args_from(["--candidate-output-dir", "/tmp/candidates"]).unwrap_err();
        assert!(args.to_string().contains("--design-space"));
    }

    #[test]
    fn design_space_and_single_candidate_modes_are_mutually_exclusive() {
        assert!(parse_args_from([
            "--design-space",
            "--candidate-id",
            "synthetic-grid-only"
        ])
        .is_err());
    }
}
