// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Plan, authenticate, and audit governed publication recovery.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalogLineageChain, CalibrationPublicationCheckpointWitnessPolicy,
    CalibrationPublicationIncidentReport, CalibrationPublicationQuarantineLedger,
    CalibrationPublicationRecoveredPolicyAnchor, CalibrationPublicationRecoveryAuthorityPolicy,
    CalibrationPublicationRecoveryAuthorizationSet, CalibrationPublicationRecoveryBundle,
    CalibrationPublicationRecoveryPlan, CalibrationPublicationRecoveryRationale,
    CalibrationPublicationRecoverySignerRole, CalibrationSignedPublicationRecoveryStatement,
    CalibrationSignerIdentity, audit_calibration_publication_recovered_policy_anchor,
    audit_calibration_publication_recovery_bundle,
    build_calibration_publication_recovered_policy_anchor,
    build_calibration_publication_recovery_authority_policy,
    build_calibration_publication_recovery_authorization_set,
    build_calibration_publication_recovery_bundle,
    build_calibration_signed_publication_recovery_statement, plan_calibration_publication_recovery,
    verify_calibration_publication_recovery_bundle,
};

use support::checkpoint_verifier::CheckpointWitnessProcessVerifier;
use support::publication_io::{
    invalid_input, next_value, parse_u64, read_json, required_path, required_string, required_u64,
    write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "policy" => policy(arguments),
        "plan" => plan(arguments),
        "statement" => statement(arguments),
        "set" => set(arguments),
        "build" => build(arguments),
        "audit" => audit(arguments),
        "verify" => verify(arguments),
        "anchor" => anchor(arguments),
        "audit-anchor" => audit_anchor(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn policy(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let policy = build_calibration_publication_recovery_authority_policy(
        required_u64(arguments.threshold, "--threshold")?,
        csv(required_string(arguments.accepted_keys, "--accepted-keys")?),
    )?;
    write_json(arguments.write.as_deref(), &policy)
}

fn plan(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let incident: CalibrationPublicationIncidentReport =
        read_json(&required_path(arguments.incident, "--incident")?)?;
    let quarantine: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.quarantine, "--quarantine")?)?;
    let lineage: CalibrationPublicationCatalogLineageChain =
        read_json(&required_path(arguments.lineage, "--lineage")?)?;
    let incoming: CalibrationPublicationCheckpointWitnessPolicy = read_json(&required_path(
        arguments.incoming_policy,
        "--incoming-policy",
    )?)?;
    let plan = plan_calibration_publication_recovery(
        &incident,
        &quarantine,
        required_string(arguments.disputed_policy_epoch, "--disputed-policy-epoch")?,
        lineage,
        incoming,
        required_u64(arguments.epoch, "--epoch")?,
        parse_rationale(&required_string(arguments.rationale, "--rationale")?)?,
    )?;
    if let Some(path) = arguments.write_bytes.as_deref() {
        std::fs::write(path, plan.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &plan)
}

fn statement(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let plan: CalibrationPublicationRecoveryPlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let statement = build_calibration_signed_publication_recovery_statement(
        &plan,
        parse_role(&required_string(arguments.role, "--role")?)?,
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
    let plan: CalibrationPublicationRecoveryPlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let policy: CalibrationPublicationRecoveryAuthorityPolicy = read_json(&required_path(
        arguments.recovery_policy,
        "--recovery-policy",
    )?)?;
    let recovery = arguments
        .recovery_statement
        .iter()
        .map(read_json::<CalibrationSignedPublicationRecoveryStatement>)
        .collect::<Result<Vec<_>, _>>()?;
    let incoming = arguments
        .incoming_statement
        .iter()
        .map(read_json::<CalibrationSignedPublicationRecoveryStatement>)
        .collect::<Result<Vec<_>, _>>()?;
    let set = build_calibration_publication_recovery_authorization_set(
        &plan, &policy, recovery, incoming,
    );
    write_json(arguments.write.as_deref(), &set)
}

fn build(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let incident: CalibrationPublicationIncidentReport =
        read_json(&required_path(arguments.incident, "--incident")?)?;
    let quarantine: CalibrationPublicationQuarantineLedger =
        read_json(&required_path(arguments.quarantine, "--quarantine")?)?;
    let policy: CalibrationPublicationRecoveryAuthorityPolicy = read_json(&required_path(
        arguments.recovery_policy,
        "--recovery-policy",
    )?)?;
    let plan: CalibrationPublicationRecoveryPlan =
        read_json(&required_path(arguments.plan, "--plan")?)?;
    let set: CalibrationPublicationRecoveryAuthorizationSet =
        read_json(&required_path(arguments.set, "--set")?)?;
    let verifier = verifier(&arguments)?;
    let bundle = build_calibration_publication_recovery_bundle(
        incident, quarantine, policy, plan, set, &verifier, &verifier, &verifier, &verifier,
        &verifier,
    )?;
    write_json(arguments.write.as_deref(), &bundle)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationRecoveryBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let report = audit_calibration_publication_recovery_bundle(&bundle);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationRecoveryBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let verifier = verifier(&arguments)?;
    let report = verify_calibration_publication_recovery_bundle(
        &bundle, &verifier, &verifier, &verifier, &verifier, &verifier,
    );
    write_json(arguments.write.as_deref(), &report)?;
    if !report.authorized() {
        std::process::exit(2);
    }
    Ok(())
}

fn anchor(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationRecoveryBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let anchor = build_calibration_publication_recovered_policy_anchor(&bundle)?;
    write_json(arguments.write.as_deref(), &anchor)
}

fn audit_anchor(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationRecoveryBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let anchor: CalibrationPublicationRecoveredPolicyAnchor =
        read_json(&required_path(arguments.anchor, "--anchor")?)?;
    let valid = audit_calibration_publication_recovered_policy_anchor(&anchor, &bundle);
    write_json(arguments.write.as_deref(), &valid)?;
    if !valid {
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

fn csv(value: String) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_role(value: &str) -> Result<CalibrationPublicationRecoverySignerRole, Box<dyn Error>> {
    match value {
        "recovery_authority" => Ok(CalibrationPublicationRecoverySignerRole::RecoveryAuthority),
        "incoming_witness" => Ok(CalibrationPublicationRecoverySignerRole::IncomingWitness),
        _ => Err(invalid_input("role must be recovery_authority or incoming_witness").into()),
    }
}

fn parse_rationale(value: &str) -> Result<CalibrationPublicationRecoveryRationale, Box<dyn Error>> {
    match value {
        "outgoing_quorum_unavailable" => {
            Ok(CalibrationPublicationRecoveryRationale::OutgoingQuorumUnavailable)
        }
        "suspected_outgoing_policy_compromise" => {
            Ok(CalibrationPublicationRecoveryRationale::SuspectedOutgoingPolicyCompromise)
        }
        "confirmed_signer_contradiction" => {
            Ok(CalibrationPublicationRecoveryRationale::ConfirmedSignerContradiction)
        }
        "catalog_fork_containment" => {
            Ok(CalibrationPublicationRecoveryRationale::CatalogForkContainment)
        }
        _ => Err(invalid_input("unknown recovery rationale").into()),
    }
}

struct Arguments {
    command: String,
    incident: Option<PathBuf>,
    quarantine: Option<PathBuf>,
    lineage: Option<PathBuf>,
    incoming_policy: Option<PathBuf>,
    recovery_policy: Option<PathBuf>,
    plan: Option<PathBuf>,
    signature: Option<PathBuf>,
    set: Option<PathBuf>,
    bundle: Option<PathBuf>,
    anchor: Option<PathBuf>,
    recovery_statement: Vec<PathBuf>,
    incoming_statement: Vec<PathBuf>,
    disputed_policy_epoch: Option<String>,
    accepted_keys: Option<String>,
    threshold: Option<u64>,
    epoch: Option<u64>,
    rationale: Option<String>,
    role: Option<String>,
    key_id: Option<String>,
    algorithm: Option<String>,
    issuer: Option<String>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
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
            incident: None,
            quarantine: None,
            lineage: None,
            incoming_policy: None,
            recovery_policy: None,
            plan: None,
            signature: None,
            set: None,
            bundle: None,
            anchor: None,
            recovery_statement: Vec::new(),
            incoming_statement: Vec::new(),
            disputed_policy_epoch: None,
            accepted_keys: None,
            threshold: None,
            epoch: None,
            rationale: None,
            role: None,
            key_id: None,
            algorithm: None,
            issuer: None,
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--incident" => {
                    result.incident = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--quarantine" => {
                    result.quarantine = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--lineage" => {
                    result.lineage = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--incoming-policy" => {
                    result.incoming_policy =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--recovery-policy" => {
                    result.recovery_policy =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--plan" => result.plan = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--set" => result.set = Some(PathBuf::from(next_value(&mut values, &argument)?)),
                "--bundle" => {
                    result.bundle = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--anchor" => {
                    result.anchor = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--recovery-statement" => result
                    .recovery_statement
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--incoming-statement" => result
                    .incoming_statement
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--disputed-policy-epoch" => {
                    result.disputed_policy_epoch = Some(next_value(&mut values, &argument)?)
                }
                "--accepted-keys" => {
                    result.accepted_keys = Some(next_value(&mut values, &argument)?)
                }
                "--threshold" => {
                    result.threshold =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--rationale" => result.rationale = Some(next_value(&mut values, &argument)?),
                "--role" => result.role = Some(next_value(&mut values, &argument)?),
                "--key-id" => result.key_id = Some(next_value(&mut values, &argument)?),
                "--algorithm" => result.algorithm = Some(next_value(&mut values, &argument)?),
                "--issuer" => result.issuer = Some(next_value(&mut values, &argument)?),
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
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
    eprintln!("evidence_publication_recovery <command> [options]");
    eprintln!("  policy --accepted-keys K1,K2 --threshold N [--write FILE]");
    eprintln!(
        "  plan --incident FILE --quarantine FILE --lineage FILE --incoming-policy FILE --disputed-policy-epoch SHA --epoch N --rationale NAME [--write FILE] [--write-bytes FILE]"
    );
    eprintln!(
        "  statement --plan FILE --role recovery_authority|incoming_witness --signature FILE --key-id ID --algorithm NAME [--issuer NAME] [--write FILE]"
    );
    eprintln!(
        "  set --plan FILE --recovery-policy FILE --recovery-statement FILE... --incoming-statement FILE... [--write FILE]"
    );
    eprintln!(
        "  build --incident FILE --quarantine FILE --recovery-policy FILE --plan FILE --set FILE --verifier PROGRAM [--write FILE]"
    );
    eprintln!("  audit --bundle FILE [--write FILE]");
    eprintln!("  verify --bundle FILE --verifier PROGRAM [--write FILE]");
    eprintln!("  anchor --bundle FILE [--write FILE]");
    eprintln!("  audit-anchor --bundle FILE --anchor FILE [--write FILE]");
}
