// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit portable publication incident-response packages.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationIncidentResponsePackage, CalibrationPublicationRecoveredPolicyAnchor,
    CalibrationPublicationRecoveryBundle, audit_calibration_publication_incident_response_package,
    build_calibration_publication_incident_response_package,
    verify_calibration_publication_incident_response_package,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{invalid_input, next_value, read_json, required_path, write_json};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "build" => build(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn build(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationRecoveryBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let anchor: CalibrationPublicationRecoveredPolicyAnchor =
        read_json(&required_path(arguments.anchor, "--anchor")?)?;
    let verifier = verifier(&arguments)?;
    let package = build_calibration_publication_incident_response_package(
        bundle, anchor, &verifier, &verifier, &verifier, &verifier, &verifier,
    )?;
    write_json(arguments.write.as_deref(), &package)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let package: CalibrationPublicationIncidentResponsePackage =
        read_json(&required_path(arguments.package, "--package")?)?;
    let report = audit_calibration_publication_incident_response_package(&package);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let package: CalibrationPublicationIncidentResponsePackage =
        read_json(&required_path(arguments.package, "--package")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_incident_response_package(
        &package, &verifier, &verifier, &verifier, &verifier, &verifier,
    );
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

fn verifier(arguments: &Arguments) -> Result<CheckpointWitnessProcessVerifier, Box<dyn Error>> {
    Ok(CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier.clone(), "--verifier")?,
        args: arguments.verifier_args.clone(),
    })
}

struct Arguments {
    command: String,
    bundle: Option<PathBuf>,
    anchor: Option<PathBuf>,
    package: Option<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values
            .next()
            .ok_or_else(|| invalid_input("command is required"))?;
        let mut result = Self {
            command,
            bundle: None,
            anchor: None,
            package: None,
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--bundle" => {
                    result.bundle = Some(PathBuf::from(next_value(&mut values, &argument)?));
                }
                "--anchor" => {
                    result.anchor = Some(PathBuf::from(next_value(&mut values, &argument)?));
                }
                "--package" => {
                    result.package = Some(PathBuf::from(next_value(&mut values, &argument)?));
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?));
                }
                "--verifier-arg" => {
                    result
                        .verifier_args
                        .push(next_value(&mut values, &argument)?);
                }
                "--write" => {
                    result.write = Some(PathBuf::from(next_value(&mut values, &argument)?));
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(result)
    }
}

fn print_help() {
    eprintln!("evidence_publication_incident_response <command> [options]");
    eprintln!("  build --bundle FILE --anchor FILE --verifier PROGRAM [--write FILE]");
    eprintln!("  audit --package FILE [--write FILE]");
    eprintln!("  verify --package FILE --verifier PROGRAM [--write FILE]");
}
