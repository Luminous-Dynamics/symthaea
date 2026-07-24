// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit fresh-checkpoint certification after exceptional recovery.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationContinuityBundle, CalibrationPublicationIncidentResponsePackage,
    CalibrationPublicationPostRecoveryCertification, CalibrationPublicationRecoveryAuthorityLedger,
    audit_calibration_publication_post_recovery_certification,
    build_calibration_publication_post_recovery_certification,
    verify_calibration_publication_post_recovery_certification,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    invalid_input, next_value, parse_u64, read_json, required_path, required_u64, write_json,
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
    let incident: CalibrationPublicationIncidentResponsePackage = read_json(&required_path(
        arguments.incident_response,
        "--incident-response",
    )?)?;
    let continuity: CalibrationPublicationContinuityBundle =
        read_json(&required_path(arguments.continuity, "--continuity")?)?;
    let authorities: CalibrationPublicationRecoveryAuthorityLedger = read_json(&required_path(
        arguments.authority_ledger,
        "--authority-ledger",
    )?)?;
    let verifier = verifier(&arguments)?;
    let certification = build_calibration_publication_post_recovery_certification(
        incident,
        continuity,
        authorities,
        required_u64(
            arguments.minimum_additional_events,
            "--minimum-additional-events",
        )?,
        required_u64(arguments.epoch, "--epoch")?,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
    )?;
    write_json(arguments.write.as_deref(), &certification)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let certification: CalibrationPublicationPostRecoveryCertification =
        read_json(&required_path(arguments.certification, "--certification")?)?;
    let report = audit_calibration_publication_post_recovery_certification(&certification);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let certification: CalibrationPublicationPostRecoveryCertification =
        read_json(&required_path(arguments.certification, "--certification")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_post_recovery_certification(
        &certification,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
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
    incident_response: Option<PathBuf>,
    continuity: Option<PathBuf>,
    authority_ledger: Option<PathBuf>,
    certification: Option<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    minimum_additional_events: Option<u64>,
    epoch: Option<u64>,
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
            incident_response: None,
            continuity: None,
            authority_ledger: None,
            certification: None,
            verifier: None,
            verifier_args: Vec::new(),
            minimum_additional_events: None,
            epoch: None,
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--incident-response" => {
                    result.incident_response =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--continuity" => {
                    result.continuity = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authority-ledger" => {
                    result.authority_ledger =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--certification" => {
                    result.certification = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--minimum-additional-events" => {
                    result.minimum_additional_events =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
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
    eprintln!("evidence_publication_post_recovery <command> [options]");
    eprintln!(
        "  build --incident-response FILE --continuity FILE --authority-ledger FILE --minimum-additional-events N --epoch N --verifier PROGRAM [--write FILE]"
    );
    eprintln!("  audit --certification FILE [--write FILE]");
    eprintln!("  verify --certification FILE --verifier PROGRAM [--write FILE]");
}
