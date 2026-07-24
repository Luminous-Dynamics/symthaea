// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build, rotate, audit, and query exceptional recovery-authority policies.

mod support;

use std::error::Error;
use std::path::PathBuf;

use serde::Serialize;
use symthaea_music_theory::{
    CalibrationPublicationCatalogCheckpoint, CalibrationPublicationRecoveryAuthorityEpoch,
    CalibrationPublicationRecoveryAuthorityLedger, CalibrationPublicationRecoveryAuthorityPolicy,
    CalibrationPublicationRecoveryAuthorityRotationPayload,
    CalibrationPublicationRecoveryAuthorityRotationSet,
    CalibrationSignedPublicationRecoveryAuthorityRotation, CalibrationSignerIdentity,
    active_calibration_publication_recovery_authority_epoch,
    append_calibration_publication_recovery_authority_rotation,
    audit_calibration_publication_recovery_authority_ledger,
    build_calibration_publication_recovery_authority_genesis,
    build_calibration_publication_recovery_authority_rotation_set,
    build_calibration_signed_publication_recovery_authority_rotation,
    plan_calibration_publication_recovery_authority_rotation,
    verify_calibration_publication_recovery_authority_ledger,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    atomic_bytes, invalid_input, next_value, parse_u64, read_json, required_path, required_string,
    required_u64, write_json,
};

#[derive(Serialize)]
struct RecoveryAuthorityRotationPlanExport {
    epoch: CalibrationPublicationRecoveryAuthorityEpoch,
    payload: CalibrationPublicationRecoveryAuthorityRotationPayload,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "genesis" => genesis(arguments),
        "plan" => plan(arguments),
        "wrap" => wrap(arguments),
        "set" => rotation_set(arguments),
        "append" => append(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        "active" => active(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn genesis(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let policy: CalibrationPublicationRecoveryAuthorityPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let ledger = build_calibration_publication_recovery_authority_genesis(
        policy,
        checkpoint,
        required_u64(arguments.epoch, "--epoch")?,
    )?;
    write_json(arguments.write.as_deref(), &ledger)
}

fn plan(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationRecoveryAuthorityLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let policy: CalibrationPublicationRecoveryAuthorityPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let (epoch, payload) = plan_calibration_publication_recovery_authority_rotation(
        &ledger,
        checkpoint,
        policy,
        required_u64(arguments.epoch, "--epoch")?,
    )?;
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &payload.canonical_bytes())?;
    }
    write_json(
        arguments.write.as_deref(),
        &RecoveryAuthorityRotationPlanExport { epoch, payload },
    )
}

fn wrap(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationRecoveryAuthorityRotationPayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let envelope = build_calibration_signed_publication_recovery_authority_rotation(
        payload,
        CalibrationSignerIdentity {
            key_id: required_string(arguments.key_id, "--key-id")?,
            algorithm: required_string(arguments.algorithm, "--algorithm")?,
            issuer: arguments.issuer,
        },
        &signature,
    );
    write_json(arguments.write.as_deref(), &envelope)
}

fn rotation_set(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationRecoveryAuthorityRotationPayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let outgoing_policy: CalibrationPublicationRecoveryAuthorityPolicy = read_json(
        &required_path(arguments.outgoing_policy, "--outgoing-policy")?,
    )?;
    let incoming_policy: CalibrationPublicationRecoveryAuthorityPolicy = read_json(
        &required_path(arguments.incoming_policy, "--incoming-policy")?,
    )?;
    let outgoing = read_many::<CalibrationSignedPublicationRecoveryAuthorityRotation>(
        &arguments.outgoing_statements,
    )?;
    let incoming = read_many::<CalibrationSignedPublicationRecoveryAuthorityRotation>(
        &arguments.incoming_statements,
    )?;
    let set = build_calibration_publication_recovery_authority_rotation_set(
        &payload,
        &outgoing_policy,
        &incoming_policy,
        outgoing,
        incoming,
    );
    write_json(arguments.write.as_deref(), &set)
}

fn append(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let mut ledger: CalibrationPublicationRecoveryAuthorityLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let epoch: CalibrationPublicationRecoveryAuthorityEpoch = read_json(&required_path(
        arguments.authority_epoch,
        "--authority-epoch",
    )?)?;
    let set: CalibrationPublicationRecoveryAuthorityRotationSet =
        read_json(&required_path(arguments.rotation_set, "--rotation-set")?)?;
    let verifier = verifier(&arguments)?;
    append_calibration_publication_recovery_authority_rotation(&mut ledger, epoch, set, &verifier)?;
    write_json(arguments.write.as_deref(), &ledger)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationRecoveryAuthorityLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let report = audit_calibration_publication_recovery_authority_ledger(&ledger);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationRecoveryAuthorityLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_recovery_authority_ledger(&ledger, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

fn active(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationRecoveryAuthorityLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let epoch = active_calibration_publication_recovery_authority_epoch(&ledger, &checkpoint)
        .ok_or_else(|| invalid_input("no recovery-authority epoch is active at this checkpoint"))?;
    write_json(arguments.write.as_deref(), epoch)
}

fn read_many<T: serde::de::DeserializeOwned>(paths: &[PathBuf]) -> Result<Vec<T>, Box<dyn Error>> {
    paths.iter().map(|path| read_json(path)).collect()
}

fn verifier(arguments: &Arguments) -> Result<CheckpointWitnessProcessVerifier, Box<dyn Error>> {
    Ok(CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier.clone(), "--verifier")?,
        args: arguments.verifier_args.clone(),
    })
}

struct Arguments {
    command: String,
    ledger: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    policy: Option<PathBuf>,
    outgoing_policy: Option<PathBuf>,
    incoming_policy: Option<PathBuf>,
    authority_epoch: Option<PathBuf>,
    payload: Option<PathBuf>,
    signature: Option<PathBuf>,
    rotation_set: Option<PathBuf>,
    outgoing_statements: Vec<PathBuf>,
    incoming_statements: Vec<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    epoch: Option<u64>,
    key_id: Option<String>,
    algorithm: Option<String>,
    issuer: Option<String>,
    write: Option<PathBuf>,
    write_bytes: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values
            .next()
            .ok_or_else(|| invalid_input("command is required"))?;
        let mut result = Self {
            command,
            ledger: None,
            checkpoint: None,
            policy: None,
            outgoing_policy: None,
            incoming_policy: None,
            authority_epoch: None,
            payload: None,
            signature: None,
            rotation_set: None,
            outgoing_statements: Vec::new(),
            incoming_statements: Vec::new(),
            verifier: None,
            verifier_args: Vec::new(),
            epoch: None,
            key_id: None,
            algorithm: None,
            issuer: None,
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--ledger" => {
                    result.ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--checkpoint" => {
                    result.checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--policy" => {
                    result.policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--outgoing-policy" => {
                    result.outgoing_policy =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--incoming-policy" => {
                    result.incoming_policy =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authority-epoch" => {
                    result.authority_epoch =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--payload" => {
                    result.payload = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--rotation-set" => {
                    result.rotation_set = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--outgoing-statement" => result
                    .outgoing_statements
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--incoming-statement" => result
                    .incoming_statements
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--key-id" => result.key_id = Some(next_value(&mut values, &argument)?),
                "--algorithm" => result.algorithm = Some(next_value(&mut values, &argument)?),
                "--issuer" => result.issuer = Some(next_value(&mut values, &argument)?),
                "--write" => {
                    result.write = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--write-bytes" => {
                    result.write_bytes = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    eprintln!("evidence_publication_recovery_authority <command> [options]");
    eprintln!("  genesis --checkpoint FILE --policy FILE --epoch N [--write FILE]");
    eprintln!(
        "  plan --ledger FILE --checkpoint FILE --policy FILE --epoch N [--write FILE] [--write-bytes FILE]"
    );
    eprintln!(
        "  wrap --payload FILE --signature FILE --key-id ID --algorithm NAME [--issuer NAME] [--write FILE]"
    );
    eprintln!(
        "  set --payload FILE --outgoing-policy FILE --incoming-policy FILE --outgoing-statement FILE... --incoming-statement FILE... [--write FILE]"
    );
    eprintln!(
        "  append --ledger FILE --authority-epoch FILE --rotation-set FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
    eprintln!("  audit --ledger FILE [--write FILE]");
    eprintln!("  verify --ledger FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]");
    eprintln!("  active --ledger FILE --checkpoint FILE [--write FILE]");
}
