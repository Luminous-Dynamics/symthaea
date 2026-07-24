// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build, wrap, audit, and externally verify checkpoint witness sets.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalogCheckpoint, CalibrationPublicationCheckpointWitnessPayload,
    CalibrationPublicationCheckpointWitnessPolicy, CalibrationPublicationCheckpointWitnessSet,
    CalibrationSignedPublicationCheckpointWitness, CalibrationSignerIdentity,
    audit_calibration_publication_checkpoint_witness_set,
    build_calibration_publication_checkpoint_witness_payload,
    build_calibration_publication_checkpoint_witness_policy,
    build_calibration_publication_checkpoint_witness_set,
    build_calibration_signed_publication_checkpoint_witness,
    verify_calibration_publication_checkpoint_witness_set,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    atomic_bytes, invalid_input, next_value, parse_u64, read_json, required_path, required_string,
    required_u64, write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "payload" => payload(arguments),
        "wrap" => wrap(arguments),
        "policy" => policy(arguments),
        "set" => set(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn payload(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let payload = build_calibration_publication_checkpoint_witness_payload(
        &checkpoint,
        required_u64(arguments.epoch, "--epoch")?,
    );
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &payload.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &payload)
}

fn wrap(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationCheckpointWitnessPayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let envelope = build_calibration_signed_publication_checkpoint_witness(
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

fn policy(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let policy = build_calibration_publication_checkpoint_witness_policy(
        required_u64(arguments.minimum_witnesses, "--minimum-witnesses")?,
        arguments.accepted_key_ids,
    );
    write_json(arguments.write.as_deref(), &policy)
}

fn set(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let policy: CalibrationPublicationCheckpointWitnessPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let statements = arguments
        .statements
        .iter()
        .map(|path| read_json::<CalibrationSignedPublicationCheckpointWitness>(path))
        .collect::<Result<Vec<_>, _>>()?;
    let set = build_calibration_publication_checkpoint_witness_set(&checkpoint, policy, statements);
    write_json(arguments.write.as_deref(), &set)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let set: CalibrationPublicationCheckpointWitnessSet =
        read_json(&required_path(arguments.set, "--set")?)?;
    let report = audit_calibration_publication_checkpoint_witness_set(&checkpoint, &set);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let set: CalibrationPublicationCheckpointWitnessSet =
        read_json(&required_path(arguments.set, "--set")?)?;
    let verifier = CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier, "--verifier")?,
        args: arguments.verifier_args,
    };
    let report =
        verify_calibration_publication_checkpoint_witness_set(&checkpoint, &set, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

struct Arguments {
    command: String,
    checkpoint: Option<PathBuf>,
    payload: Option<PathBuf>,
    signature: Option<PathBuf>,
    policy: Option<PathBuf>,
    set: Option<PathBuf>,
    statements: Vec<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    epoch: Option<u64>,
    minimum_witnesses: Option<u64>,
    accepted_key_ids: Vec<String>,
    key_id: Option<String>,
    algorithm: Option<String>,
    issuer: Option<String>,
    write: Option<PathBuf>,
    write_bytes: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values.next().ok_or_else(|| {
            invalid_input("command is required: payload, wrap, policy, set, audit, or verify")
        })?;
        let mut result = Self {
            command,
            checkpoint: None,
            payload: None,
            signature: None,
            policy: None,
            set: None,
            statements: Vec::new(),
            verifier: None,
            verifier_args: Vec::new(),
            epoch: None,
            minimum_witnesses: None,
            accepted_key_ids: Vec::new(),
            key_id: None,
            algorithm: None,
            issuer: None,
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--checkpoint" => {
                    result.checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--payload" => {
                    result.payload = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--policy" => {
                    result.policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--set" => result.set = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--statement" => result
                    .statements
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
                "--minimum-witnesses" => {
                    result.minimum_witnesses =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--accepted-key-id" => result
                    .accepted_key_ids
                    .push(next_value(&mut values, &argument)?),
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
    eprintln!("evidence_publication_witness <command> [options]");
    eprintln!("  payload --checkpoint FILE --epoch N [--write FILE] [--write-bytes FILE]");
    eprintln!(
        "  wrap --payload FILE --signature FILE --key-id ID --algorithm NAME [--issuer NAME] [--write FILE]"
    );
    eprintln!("  policy --minimum-witnesses N --accepted-key-id ID... [--write FILE]");
    eprintln!("  set --checkpoint FILE --policy FILE --statement FILE... [--write FILE]");
    eprintln!("  audit --checkpoint FILE --set FILE [--write FILE]");
    eprintln!(
        "  verify --checkpoint FILE --set FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
}
