// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit conservative publication incident reports.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationContinuityBundle, CalibrationPublicationIncidentReport,
    audit_calibration_publication_incident_report, build_calibration_publication_incident_report,
    verify_calibration_publication_incident_report,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    invalid_input, next_value, parse_u64, read_json, required_path, required_string, required_u64,
    write_json,
};

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
    let continuity: CalibrationPublicationContinuityBundle =
        read_json(&required_path(arguments.continuity, "--continuity")?)?;
    let verifier = verifier(&arguments)?;
    let report = build_calibration_publication_incident_report(
        required_string(arguments.incident_id, "--incident-id")?,
        required_u64(arguments.epoch, "--epoch")?,
        continuity,
        &verifier,
        &verifier,
        &verifier,
    )?;
    write_json(arguments.write.as_deref(), &report)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let report: CalibrationPublicationIncidentReport =
        read_json(&required_path(arguments.report, "--report")?)?;
    let audit = audit_calibration_publication_incident_report(&report);
    write_json(arguments.write.as_deref(), &audit)?;
    if !audit.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let report: CalibrationPublicationIncidentReport =
        read_json(&required_path(arguments.report, "--report")?)?;
    let verifier = verifier(&arguments)?;
    let audit =
        verify_calibration_publication_incident_report(&report, &verifier, &verifier, &verifier);
    write_json(arguments.write.as_deref(), &audit)?;
    if !audit.authenticated() {
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
    continuity: Option<PathBuf>,
    report: Option<PathBuf>,
    incident_id: Option<String>,
    epoch: Option<u64>,
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
            continuity: None,
            report: None,
            incident_id: None,
            epoch: None,
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--continuity" => {
                    result.continuity = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--report" => {
                    result.report = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--incident-id" => result.incident_id = Some(next_value(&mut values, &argument)?),
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--write" => {
                    result.write = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    eprintln!("evidence_publication_incident <command> [options]");
    eprintln!(
        "  build --continuity FILE --incident-id ID --epoch N --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
    eprintln!("  audit --report FILE [--write FILE]");
    eprintln!("  verify --report FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]");
}
