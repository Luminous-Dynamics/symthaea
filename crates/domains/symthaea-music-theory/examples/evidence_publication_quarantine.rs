// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Create, authenticate, append, and evaluate publication quarantine decisions.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCheckpointWitnessPolicy, CalibrationPublicationGossipLedger,
    CalibrationPublicationIncidentReport, CalibrationPublicationQuarantineAction,
    CalibrationPublicationQuarantineDecisionSet, CalibrationPublicationQuarantineLedger,
    CalibrationPublicationQuarantinePayload, CalibrationPublicationQuarantineReason,
    CalibrationPublicationQuarantineScope, CalibrationSignedPublicationQuarantineDecision,
    CalibrationSignerIdentity, append_calibration_publication_quarantine_decision,
    audit_calibration_publication_quarantine_ledger,
    build_calibration_publication_quarantine_decision_set,
    build_calibration_publication_quarantine_ledger,
    build_calibration_publication_quarantine_policy,
    build_calibration_signed_publication_quarantine_decision,
    evaluate_calibration_publication_quarantine, plan_calibration_publication_quarantine_decision,
    verify_calibration_publication_quarantine_ledger,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    invalid_input, next_value, parse_u64, read_json, required_path, required_string, required_u64,
    write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "ledger" => ledger(arguments),
        "payload" => payload(arguments),
        "wrap" => wrap(arguments),
        "set" => set(arguments),
        "append" => append(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        "evaluate" => evaluate(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn ledger(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let policy = build_calibration_publication_quarantine_policy(
        required_u64(arguments.threshold, "--threshold")?,
        csv(required_string(arguments.accepted_keys, "--accepted-keys")?),
    )?;
    let ledger = build_calibration_publication_quarantine_ledger(
        required_string(arguments.catalog_id, "--catalog-id")?,
        required_string(arguments.authority_id, "--authority-id")?,
        policy,
    )?;
    write_json(arguments.write.as_deref(), &ledger)
}

fn payload(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let incident: CalibrationPublicationIncidentReport =
        read_json(&required_path(arguments.incident, "--incident")?)?;
    let payload = plan_calibration_publication_quarantine_decision(
        &ledger,
        &incident,
        required_string(arguments.key_id, "--key-id")?,
        parse_scope(&required_string(arguments.scope, "--scope")?)?,
        parse_action(&required_string(arguments.action, "--action")?)?,
        parse_reason(&required_string(arguments.reason, "--reason")?)?,
        required_u64(arguments.epoch, "--epoch")?,
        arguments.expires_epoch,
    )?;
    if let Some(path) = arguments.write_bytes.as_deref() {
        std::fs::write(path, payload.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &payload)
}

fn wrap(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationQuarantinePayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let statement = build_calibration_signed_publication_quarantine_decision(
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

fn set(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationQuarantinePayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let statements = arguments
        .statement
        .iter()
        .map(read_json::<CalibrationSignedPublicationQuarantineDecision>)
        .collect::<Result<Vec<_>, _>>()?;
    let set = build_calibration_publication_quarantine_decision_set(&payload, statements);
    write_json(arguments.write.as_deref(), &set)
}

fn append(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let mut ledger: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let set: CalibrationPublicationQuarantineDecisionSet =
        read_json(&required_path(arguments.set, "--set")?)?;
    let verifier = verifier(&arguments)?;
    append_calibration_publication_quarantine_decision(&mut ledger, set, &verifier)?;
    write_json(arguments.write.as_deref(), &ledger)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let report = audit_calibration_publication_quarantine_ledger(&ledger);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_quarantine_ledger(&ledger, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

fn evaluate(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let witness_policy = arguments
        .witness_policy
        .as_deref()
        .map(read_json::<CalibrationPublicationCheckpointWitnessPolicy>)
        .transpose()?;
    let gossip = arguments
        .gossip
        .as_deref()
        .map(read_json::<CalibrationPublicationGossipLedger>)
        .transpose()?;
    let evaluation = evaluate_calibration_publication_quarantine(
        &ledger,
        required_u64(arguments.epoch, "--epoch")?,
        witness_policy.as_ref(),
        gossip.as_ref(),
    );
    write_json(arguments.write.as_deref(), &evaluation)?;
    if !evaluation.witness_quorum_available {
        std::process::exit(3);
    }
    Ok(())
}

fn verifier(arguments: &Arguments) -> Result<CheckpointWitnessProcessVerifier, Box<dyn Error>> {
    Ok(CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier.clone(), "--verifier")?,
        args: arguments.verifier_args.clone(),
    })
}

fn csv(value: String) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_scope(value: &str) -> Result<CalibrationPublicationQuarantineScope, Box<dyn Error>> {
    match value {
        "witness" => Ok(CalibrationPublicationQuarantineScope::Witness),
        "observer" => Ok(CalibrationPublicationQuarantineScope::Observer),
        "both" | "witness_and_observer" => {
            Ok(CalibrationPublicationQuarantineScope::WitnessAndObserver)
        }
        _ => Err(invalid_input("scope must be witness, observer, or both").into()),
    }
}

fn parse_action(value: &str) -> Result<CalibrationPublicationQuarantineAction, Box<dyn Error>> {
    match value {
        "quarantine" => Ok(CalibrationPublicationQuarantineAction::Quarantine),
        "release" => Ok(CalibrationPublicationQuarantineAction::Release),
        _ => Err(invalid_input("action must be quarantine or release").into()),
    }
}

fn parse_reason(value: &str) -> Result<CalibrationPublicationQuarantineReason, Box<dyn Error>> {
    match value {
        "direct_signer_contradiction" => {
            Ok(CalibrationPublicationQuarantineReason::DirectSignerContradiction)
        }
        "suspected_key_compromise" => {
            Ok(CalibrationPublicationQuarantineReason::SuspectedKeyCompromise)
        }
        "quorum_loss" => Ok(CalibrationPublicationQuarantineReason::QuorumLoss),
        "administrative_containment" => {
            Ok(CalibrationPublicationQuarantineReason::AdministrativeContainment)
        }
        "recovery_completed" => Ok(CalibrationPublicationQuarantineReason::RecoveryCompleted),
        _ => Err(invalid_input("unknown quarantine reason").into()),
    }
}

struct Arguments {
    command: String,
    ledger: Option<PathBuf>,
    incident: Option<PathBuf>,
    payload: Option<PathBuf>,
    signature: Option<PathBuf>,
    set: Option<PathBuf>,
    statement: Vec<PathBuf>,
    witness_policy: Option<PathBuf>,
    gossip: Option<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    catalog_id: Option<String>,
    authority_id: Option<String>,
    accepted_keys: Option<String>,
    threshold: Option<u64>,
    key_id: Option<String>,
    scope: Option<String>,
    action: Option<String>,
    reason: Option<String>,
    epoch: Option<u64>,
    expires_epoch: Option<u64>,
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
            incident: None,
            payload: None,
            signature: None,
            set: None,
            statement: Vec::new(),
            witness_policy: None,
            gossip: None,
            verifier: None,
            verifier_args: Vec::new(),
            catalog_id: None,
            authority_id: None,
            accepted_keys: None,
            threshold: None,
            key_id: None,
            scope: None,
            action: None,
            reason: None,
            epoch: None,
            expires_epoch: None,
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
                "--incident" => {
                    result.incident = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--payload" => {
                    result.payload = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--set" => result.set = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--statement" => result
                    .statement
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--witness-policy" => {
                    result.witness_policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--gossip" => {
                    result.gossip = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--catalog-id" => result.catalog_id = Some(next_value(&mut values, &argument)?),
                "--authority-id" => result.authority_id = Some(next_value(&mut values, &argument)?),
                "--accepted-keys" => {
                    result.accepted_keys = Some(next_value(&mut values, &argument)?)
                }
                "--threshold" => {
                    result.threshold =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--key-id" => result.key_id = Some(next_value(&mut values, &argument)?),
                "--scope" => result.scope = Some(next_value(&mut values, &argument)?),
                "--action" => result.action = Some(next_value(&mut values, &argument)?),
                "--reason" => result.reason = Some(next_value(&mut values, &argument)?),
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--expires-epoch" => {
                    result.expires_epoch =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
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
    eprintln!("evidence_publication_quarantine <command> [options]");
    eprintln!(
        "  ledger --catalog-id ID --authority-id ID --accepted-keys K1,K2 --threshold N [--write FILE]"
    );
    eprintln!(
        "  payload --ledger FILE --incident FILE --key-id ID --scope witness|observer|both --action quarantine|release --reason NAME --epoch N [--expires-epoch N] [--write FILE] [--write-bytes FILE]"
    );
    eprintln!(
        "  wrap --payload FILE --signature FILE --key-id ID --algorithm NAME [--issuer NAME] [--write FILE]"
    );
    eprintln!("  set --payload FILE --statement FILE... [--write FILE]");
    eprintln!(
        "  append --ledger FILE --set FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
    eprintln!("  audit --ledger FILE [--write FILE]");
    eprintln!("  verify --ledger FILE --verifier PROGRAM [--write FILE]");
    eprintln!(
        "  evaluate --ledger FILE --epoch N [--witness-policy FILE] [--gossip FILE] [--write FILE]"
    );
}
