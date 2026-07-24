// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audit and update a private study-book assignment registry.

use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Serialize;
use symthaea_music_theory::{
    CalibrationStudyAssignmentRegistry, CalibrationStudyEnrollment, assign_calibration_study_book,
    audit_calibration_study_assignment_registry, revoke_calibration_study_enrollment,
};

#[derive(Debug)]
enum Operation {
    Audit,
    Assign {
        token_file: Option<PathBuf>,
        write: PathBuf,
    },
    Revoke {
        assessor_pseudonym: String,
        write: PathBuf,
    },
}

#[derive(Debug)]
struct Arguments {
    registry: PathBuf,
    operation: Operation,
    write_report: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct AssignmentExport<'a> {
    registry_sha256: &'a str,
    enrollment: Option<&'a CalibrationStudyEnrollment>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments()?;
    let mut registry: CalibrationStudyAssignmentRegistry = read_json(&arguments.registry)?;
    let enrollment = match &arguments.operation {
        Operation::Audit => {
            let audit = audit_calibration_study_assignment_registry(&registry);
            if let Some(path) = arguments.write_report.as_deref() {
                atomic_json(path, &audit)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&audit)?);
            }
            if !audit.valid() {
                return Err(invalid_input(format!(
                    "assignment registry failed audit with {} issues",
                    audit.issues.len()
                ))
                .into());
            }
            return Ok(());
        }
        Operation::Assign { token_file, write } => {
            let token = read_secret(token_file.as_deref())?;
            let enrollment = assign_calibration_study_book(&mut registry, token.trim())?;
            atomic_json(write, &registry)?;
            Some(enrollment)
        }
        Operation::Revoke {
            assessor_pseudonym,
            write,
        } => {
            let enrollment =
                revoke_calibration_study_enrollment(&mut registry, assessor_pseudonym)?;
            atomic_json(write, &registry)?;
            Some(enrollment)
        }
    };
    let export = AssignmentExport {
        registry_sha256: &registry.registry_sha256,
        enrollment: enrollment.as_ref(),
    };
    if let Some(path) = arguments.write_report.as_deref() {
        atomic_json(path, &export)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&export)?);
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut registry = None;
    let mut write_report = None;
    let mut values = std::env::args().skip(1);
    let command = values
        .next()
        .ok_or_else(|| invalid_input("command is required: audit, assign, or revoke"))?;
    let mut token_file = None;
    let mut assessor_pseudonym = None;
    let mut write = None;
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--registry" => registry = Some(PathBuf::from(next_value(&mut values, &argument)?)),
            "--token-file" => token_file = Some(PathBuf::from(next_value(&mut values, &argument)?)),
            "--assessor-pseudonym" => {
                assessor_pseudonym = Some(next_value(&mut values, &argument)?)
            }
            "--write" => write = Some(PathBuf::from(next_value(&mut values, &argument)?)),
            "--write-report" => {
                write_report = Some(PathBuf::from(next_value(&mut values, &argument)?))
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
        }
    }
    let registry = registry.ok_or_else(|| invalid_input("--registry is required"))?;
    let operation = match command.as_str() {
        "audit" => Operation::Audit,
        "assign" => Operation::Assign {
            token_file,
            write: write.ok_or_else(|| invalid_input("assign requires --write"))?,
        },
        "revoke" => Operation::Revoke {
            assessor_pseudonym: assessor_pseudonym
                .ok_or_else(|| invalid_input("revoke requires --assessor-pseudonym"))?,
            write: write.ok_or_else(|| invalid_input("revoke requires --write"))?,
        },
        other => return Err(invalid_input(format!("unknown command: {other}")).into()),
    };
    Ok(Arguments {
        registry,
        operation,
        write_report,
    })
}

fn print_help() {
    println!(
        "usage:\n  evidence_study_assignment audit --registry PATH [--write-report PATH]\n  evidence_study_assignment assign --registry PATH [--token-file PATH] --write PATH [--write-report PATH]\n  evidence_study_assignment revoke --registry PATH --assessor-pseudonym SHA256 --write PATH [--write-report PATH]\n\nWithout --token-file, assign reads the raw token from standard input. Raw tokens are never written to the registry or report."
    );
}

fn read_secret(path: Option<&Path>) -> Result<String, Box<dyn Error>> {
    if let Some(path) = path {
        return Ok(std::fs::read_to_string(path)?);
    }
    let mut value = String::new();
    std::io::stdin().read_to_string(&mut value)?;
    if value.trim().is_empty() {
        return Err(invalid_input("assessor token is empty").into());
    }
    Ok(value)
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")).into())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error
    })?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
