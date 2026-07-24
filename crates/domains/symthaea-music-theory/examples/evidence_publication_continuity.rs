// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit publication witness-policy continuity bundles.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalogHeadBundle, CalibrationPublicationContinuityBundle,
    CalibrationPublicationGossipLedger, CalibrationPublicationWitnessPolicyLedger,
    audit_calibration_publication_continuity_bundle,
    build_calibration_publication_continuity_bundle,
    extract_calibration_publication_gossip_conflict_proofs,
    verify_calibration_publication_continuity_bundle,
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
    let head: CalibrationPublicationCatalogHeadBundle =
        read_json(&required_path(arguments.head_bundle, "--head-bundle")?)?;
    let policy: CalibrationPublicationWitnessPolicyLedger =
        read_json(&required_path(arguments.policy_ledger, "--policy-ledger")?)?;
    let gossip = arguments
        .gossip_ledger
        .as_deref()
        .map(read_json::<CalibrationPublicationGossipLedger>)
        .transpose()?;
    let conflicts = gossip
        .as_ref()
        .map(extract_calibration_publication_gossip_conflict_proofs)
        .unwrap_or_default();
    let verifier = verifier(&arguments)?;
    let bundle = build_calibration_publication_continuity_bundle(
        head, policy, gossip, conflicts, &verifier, &verifier, &verifier,
    )?;
    write_json(arguments.write.as_deref(), &bundle)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationContinuityBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let report = audit_calibration_publication_continuity_bundle(&bundle);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationContinuityBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let verifier = verifier(&arguments)?;
    let report =
        verify_calibration_publication_continuity_bundle(&bundle, &verifier, &verifier, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.authenticated() {
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
    head_bundle: Option<PathBuf>,
    policy_ledger: Option<PathBuf>,
    gossip_ledger: Option<PathBuf>,
    bundle: Option<PathBuf>,
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
            head_bundle: None,
            policy_ledger: None,
            gossip_ledger: None,
            bundle: None,
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--head-bundle" => {
                    result.head_bundle = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--policy-ledger" => {
                    result.policy_ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--gossip-ledger" => {
                    result.gossip_ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--bundle" => {
                    result.bundle = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    eprintln!("evidence_publication_continuity <command> [options]");
    eprintln!(
        "  build --head-bundle FILE --policy-ledger FILE [--gossip-ledger FILE] --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]"
    );
    eprintln!("  audit --bundle FILE [--write FILE]");
    eprintln!("  verify --bundle FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]");
}
