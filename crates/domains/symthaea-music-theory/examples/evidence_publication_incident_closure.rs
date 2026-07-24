// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Plan, authorize, build, and audit operational incident closure.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCheckpointWitnessPolicy,
    CalibrationPublicationIncidentClosureAuthorizationSet,
    CalibrationPublicationIncidentClosureBundle, CalibrationPublicationIncidentClosurePlan,
    CalibrationPublicationIncidentClosurePolicy, CalibrationPublicationIncidentClosureSignerRole,
    CalibrationPublicationPostRecoveryCertification, CalibrationPublicationQuarantineLedger,
    CalibrationPublicationRecoveryAuthorityPolicy,
    CalibrationSignedPublicationIncidentClosureStatement, CalibrationSignerIdentity,
    audit_calibration_publication_incident_closure_bundle,
    build_calibration_publication_incident_closure_authorization_set,
    build_calibration_publication_incident_closure_bundle,
    build_calibration_publication_incident_closure_policy,
    build_calibration_signed_publication_incident_closure_statement,
    plan_calibration_publication_incident_closure,
    verify_calibration_publication_incident_closure_bundle,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    atomic_bytes, invalid_input, next_value, parse_u64, read_json, required_path, required_string,
    required_u64, write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "policy" => policy(arguments),
        "plan" => plan(arguments),
        "wrap" => wrap(arguments),
        "set" => authorization_set(arguments),
        "build" => build(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn policy(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let value = build_calibration_publication_incident_closure_policy(
        required_u64(
            arguments.minimum_additional_events,
            "--minimum-additional-events",
        )?,
        arguments.minimum_authority_signers,
        arguments.minimum_witness_signers,
        arguments.require_no_witness_quarantines,
        arguments.require_no_observer_quarantines,
    )?;
    write_json(arguments.write.as_deref(), &value)
}

fn plan(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let certification: CalibrationPublicationPostRecoveryCertification =
        read_json(&required_path(arguments.certification, "--certification")?)?;
    let quarantine: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.quarantine, "--quarantine")?)?;
    let policy: CalibrationPublicationIncidentClosurePolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let plan = plan_calibration_publication_incident_closure(
        &certification,
        &quarantine,
        &policy,
        required_u64(arguments.epoch, "--epoch")?,
    )?;
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &plan.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &plan)
}

fn wrap(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let plan: CalibrationPublicationIncidentClosurePlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let role = parse_role(&required_string(arguments.role, "--role")?)?;
    let statement = build_calibration_signed_publication_incident_closure_statement(
        &plan,
        role,
        CalibrationSignerIdentity {
            key_id: required_string(arguments.key_id, "--key-id")?,
            algorithm: required_string(arguments.algorithm, "--algorithm")?,
            issuer: arguments.issuer,
        },
        &signature,
    );
    write_json(arguments.write.as_deref(), &statement)
}

fn authorization_set(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let plan: CalibrationPublicationIncidentClosurePlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let authority_policy: CalibrationPublicationRecoveryAuthorityPolicy = read_json(
        &required_path(arguments.authority_policy, "--authority-policy")?,
    )?;
    let witness_policy: CalibrationPublicationCheckpointWitnessPolicy = read_json(&required_path(
        arguments.witness_policy,
        "--witness-policy",
    )?)?;
    let authority = read_many::<CalibrationSignedPublicationIncidentClosureStatement>(
        &arguments.authority_statements,
    )?;
    let witnesses = read_many::<CalibrationSignedPublicationIncidentClosureStatement>(
        &arguments.witness_statements,
    )?;
    let set = build_calibration_publication_incident_closure_authorization_set(
        &plan,
        &authority_policy,
        &witness_policy,
        authority,
        witnesses,
    );
    write_json(arguments.write.as_deref(), &set)
}

fn build(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let certification: CalibrationPublicationPostRecoveryCertification =
        read_json(&required_path(arguments.certification, "--certification")?)?;
    let quarantine: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.quarantine, "--quarantine")?)?;
    let policy: CalibrationPublicationIncidentClosurePolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let plan: CalibrationPublicationIncidentClosurePlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let set: CalibrationPublicationIncidentClosureAuthorizationSet = read_json(&required_path(
        arguments.authorization_set,
        "--authorization-set",
    )?)?;
    let verifier = verifier(&arguments)?;
    let bundle = build_calibration_publication_incident_closure_bundle(
        certification,
        quarantine,
        policy,
        plan,
        set,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
        &verifier,
    )?;
    write_json(arguments.write.as_deref(), &bundle)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationIncidentClosureBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let report = audit_calibration_publication_incident_closure_bundle(&bundle);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationIncidentClosureBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_incident_closure_bundle(
        &bundle, &verifier, &verifier, &verifier, &verifier, &verifier, &verifier, &verifier,
    );
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

fn parse_role(
    value: &str,
) -> Result<CalibrationPublicationIncidentClosureSignerRole, Box<dyn Error>> {
    match value {
        "recovery-authority" => {
            Ok(CalibrationPublicationIncidentClosureSignerRole::RecoveryAuthority)
        }
        "recovered-witness" => {
            Ok(CalibrationPublicationIncidentClosureSignerRole::RecoveredWitness)
        }
        other => Err(invalid_input(format!("unknown closure signer role: {other}")).into()),
    }
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
    certification: Option<PathBuf>,
    quarantine: Option<PathBuf>,
    policy: Option<PathBuf>,
    plan: Option<PathBuf>,
    signature: Option<PathBuf>,
    authority_policy: Option<PathBuf>,
    witness_policy: Option<PathBuf>,
    authorization_set: Option<PathBuf>,
    bundle: Option<PathBuf>,
    authority_statements: Vec<PathBuf>,
    witness_statements: Vec<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    minimum_additional_events: Option<u64>,
    minimum_authority_signers: Option<u64>,
    minimum_witness_signers: Option<u64>,
    epoch: Option<u64>,
    require_no_witness_quarantines: bool,
    require_no_observer_quarantines: bool,
    role: Option<String>,
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
            certification: None,
            quarantine: None,
            policy: None,
            plan: None,
            signature: None,
            authority_policy: None,
            witness_policy: None,
            authorization_set: None,
            bundle: None,
            authority_statements: Vec::new(),
            witness_statements: Vec::new(),
            verifier: None,
            verifier_args: Vec::new(),
            minimum_additional_events: None,
            minimum_authority_signers: None,
            minimum_witness_signers: None,
            epoch: None,
            require_no_witness_quarantines: false,
            require_no_observer_quarantines: false,
            role: None,
            key_id: None,
            algorithm: None,
            issuer: None,
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--certification" => {
                    result.certification = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--quarantine" => {
                    result.quarantine = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--policy" => {
                    result.policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--plan" => result.plan = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authority-policy" => {
                    result.authority_policy =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--witness-policy" => {
                    result.witness_policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authorization-set" => {
                    result.authorization_set =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--bundle" => {
                    result.bundle = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authority-statement" => result
                    .authority_statements
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--witness-statement" => result
                    .witness_statements
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
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
                "--minimum-authority-signers" => {
                    result.minimum_authority_signers =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--minimum-witness-signers" => {
                    result.minimum_witness_signers =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--require-no-witness-quarantines" => result.require_no_witness_quarantines = true,
                "--require-no-observer-quarantines" => {
                    result.require_no_observer_quarantines = true
                }
                "--role" => result.role = Some(next_value(&mut values, &argument)?),
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
    eprintln!("evidence_publication_incident_closure <command> [options]");
    eprintln!(
        "  policy --minimum-additional-events N [--minimum-authority-signers N] [--minimum-witness-signers N] [--require-no-witness-quarantines] [--require-no-observer-quarantines]"
    );
    eprintln!(
        "  plan --certification FILE --quarantine FILE --policy FILE --epoch N [--write-bytes FILE]"
    );
    eprintln!(
        "  wrap --plan FILE --role recovery-authority|recovered-witness --signature FILE --key-id ID --algorithm NAME"
    );
    eprintln!(
        "  set --plan FILE --authority-policy FILE --witness-policy FILE [--authority-statement FILE]... [--witness-statement FILE]..."
    );
    eprintln!(
        "  build --certification FILE --quarantine FILE --policy FILE --plan FILE --authorization-set FILE --verifier PROGRAM"
    );
    eprintln!("  audit --bundle FILE");
    eprintln!("  verify --bundle FILE --verifier PROGRAM");
}
