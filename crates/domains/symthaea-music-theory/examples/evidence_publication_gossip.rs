// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build, authenticate, record, and inspect publication checkpoint gossip.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalogCheckpoint, CalibrationPublicationGossipConflictProof,
    CalibrationPublicationGossipLedger, CalibrationPublicationGossipPayload,
    CalibrationSignedPublicationGossip, CalibrationSignerIdentity,
    audit_calibration_publication_gossip_conflict_proof,
    audit_calibration_publication_gossip_ledger, build_calibration_publication_gossip_ledger,
    build_calibration_publication_gossip_payload, build_calibration_signed_publication_gossip,
    extract_calibration_publication_gossip_conflict_proofs,
    record_calibration_publication_gossip_statement, verify_calibration_publication_gossip_ledger,
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
        "ledger" => ledger(arguments),
        "record" => record(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        "conflicts" => conflicts(arguments),
        "audit-conflict" => audit_conflict(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn payload(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let payload = build_calibration_publication_gossip_payload(
        required_string(arguments.observer_id, "--observer-id")?,
        checkpoint,
        arguments.previous_checkpoint_sha256,
        required_string(arguments.policy_epoch_sha256, "--policy-epoch-sha256")?,
        required_u64(arguments.epoch, "--epoch")?,
    );
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &payload.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &payload)
}

fn wrap(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationGossipPayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let statement = build_calibration_signed_publication_gossip(
        payload,
        CalibrationSignerIdentity {
            key_id: required_string(arguments.key_id, "--key-id")?,
            algorithm: required_string(arguments.algorithm, "--algorithm")?,
            issuer: arguments.issuer,
        },
        &signature,
    );
    write_json(arguments.write.as_deref(), &statement)
}

fn ledger(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger = build_calibration_publication_gossip_ledger(
        required_string(arguments.catalog_id, "--catalog-id")?,
        required_string(arguments.authority_id, "--authority-id")?,
    );
    write_json(arguments.write.as_deref(), &ledger)
}

fn record(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let mut ledger: CalibrationPublicationGossipLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let statement: CalibrationSignedPublicationGossip =
        read_json(&required_path(arguments.statement, "--statement")?)?;
    let verifier = verifier(&arguments)?;
    record_calibration_publication_gossip_statement(&mut ledger, statement, &verifier)?;
    write_json(arguments.write.as_deref(), &ledger)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationGossipLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let report = audit_calibration_publication_gossip_ledger(&ledger);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.integrity_valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationGossipLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_gossip_ledger(&ledger, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.signatures_authenticated || !report.integrity_valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn conflicts(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationGossipLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let proofs = extract_calibration_publication_gossip_conflict_proofs(&ledger);
    write_json(arguments.write.as_deref(), &proofs)
}

fn audit_conflict(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let proof: CalibrationPublicationGossipConflictProof =
        read_json(&required_path(arguments.conflict, "--conflict")?)?;
    let report = audit_calibration_publication_gossip_conflict_proof(&proof);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
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
    checkpoint: Option<PathBuf>,
    payload: Option<PathBuf>,
    signature: Option<PathBuf>,
    ledger: Option<PathBuf>,
    statement: Option<PathBuf>,
    conflict: Option<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    observer_id: Option<String>,
    catalog_id: Option<String>,
    authority_id: Option<String>,
    previous_checkpoint_sha256: Option<String>,
    policy_epoch_sha256: Option<String>,
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
            checkpoint: None,
            payload: None,
            signature: None,
            ledger: None,
            statement: None,
            conflict: None,
            verifier: None,
            verifier_args: Vec::new(),
            observer_id: None,
            catalog_id: None,
            authority_id: None,
            previous_checkpoint_sha256: None,
            policy_epoch_sha256: None,
            epoch: None,
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
                "--ledger" => {
                    result.ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--statement" => {
                    result.statement = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--conflict" => {
                    result.conflict = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--observer-id" => result.observer_id = Some(next_value(&mut values, &argument)?),
                "--catalog-id" => result.catalog_id = Some(next_value(&mut values, &argument)?),
                "--authority-id" => result.authority_id = Some(next_value(&mut values, &argument)?),
                "--previous-checkpoint-sha256" => {
                    result.previous_checkpoint_sha256 = Some(next_value(&mut values, &argument)?)
                }
                "--policy-epoch-sha256" => {
                    result.policy_epoch_sha256 = Some(next_value(&mut values, &argument)?)
                }
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
    eprintln!("evidence_publication_gossip <command> [options]");
    eprintln!(
        "  payload --checkpoint FILE --observer-id ID --policy-epoch-sha256 HEX --epoch N [--previous-checkpoint-sha256 HEX] [--write FILE] [--write-bytes FILE]"
    );
    eprintln!(
        "  wrap --payload FILE --signature FILE --key-id ID --algorithm NAME [--issuer NAME] [--write FILE]"
    );
    eprintln!("  ledger --catalog-id ID --authority-id ID [--write FILE]");
    eprintln!(
        "  record --ledger FILE --statement FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
    eprintln!("  audit --ledger FILE [--write FILE]");
    eprintln!("  verify --ledger FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]");
    eprintln!("  conflicts --ledger FILE [--write FILE]");
    eprintln!("  audit-conflict --conflict FILE [--write FILE]");
}
