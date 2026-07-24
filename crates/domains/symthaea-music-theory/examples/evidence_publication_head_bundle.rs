// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit portable witnessed publication-catalog head bundles.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationCatalogHeadBundle, CalibrationPublicationCatalogHeadPredecessor,
    CalibrationPublicationCheckpointStatusProof, CalibrationPublicationCheckpointWitnessSet,
    CalibrationPublicationMirrorLedger, audit_calibration_publication_catalog_head_bundle,
    build_calibration_publication_catalog_head_bundle,
    verify_calibration_publication_catalog_head_bundle,
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
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let witness_set: CalibrationPublicationCheckpointWitnessSet =
        read_json(&required_path(arguments.witness_set, "--witness-set")?)?;
    let predecessor = predecessor(&arguments)?;
    let mirror_ledger = arguments
        .mirror_ledger
        .as_deref()
        .map(read_json::<CalibrationPublicationMirrorLedger>)
        .transpose()?;
    let status_proofs = arguments
        .status_proofs
        .iter()
        .map(|path| read_json::<CalibrationPublicationCheckpointStatusProof>(path))
        .collect::<Result<Vec<_>, _>>()?;
    let verifier = CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier, "--verifier")?,
        args: arguments.verifier_args,
    };
    let bundle = build_calibration_publication_catalog_head_bundle(
        catalog,
        checkpoint,
        predecessor,
        witness_set,
        mirror_ledger,
        status_proofs,
        &verifier,
    )?;
    write_json(arguments.write.as_deref(), &bundle)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationCatalogHeadBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let report = audit_calibration_publication_catalog_head_bundle(&bundle);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let bundle: CalibrationPublicationCatalogHeadBundle =
        read_json(&required_path(arguments.bundle, "--bundle")?)?;
    let verifier = CheckpointWitnessProcessVerifier {
        program: required_path(arguments.verifier, "--verifier")?,
        args: arguments.verifier_args,
    };
    let report = verify_calibration_publication_catalog_head_bundle(&bundle, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.accepted() {
        std::process::exit(2);
    }
    Ok(())
}

fn predecessor(
    arguments: &Arguments,
) -> Result<Option<CalibrationPublicationCatalogHeadPredecessor>, Box<dyn Error>> {
    let supplied = [
        arguments.previous_catalog.is_some(),
        arguments.previous_checkpoint.is_some(),
        arguments.consistency_proof.is_some(),
    ];
    if supplied.iter().all(|value| !value) {
        return Ok(None);
    }
    if !supplied.iter().all(|value| *value) {
        return Err(invalid_input("--previous-catalog, --previous-checkpoint, and --consistency-proof must be supplied together").into());
    }
    Ok(Some(CalibrationPublicationCatalogHeadPredecessor {
        catalog: read_json(&required_path(
            arguments.previous_catalog.clone(),
            "--previous-catalog",
        )?)?,
        checkpoint: read_json(&required_path(
            arguments.previous_checkpoint.clone(),
            "--previous-checkpoint",
        )?)?,
        consistency_proof: read_json(&required_path(
            arguments.consistency_proof.clone(),
            "--consistency-proof",
        )?)?,
    }))
}

struct Arguments {
    command: String,
    bundle: Option<PathBuf>,
    catalog: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    previous_catalog: Option<PathBuf>,
    previous_checkpoint: Option<PathBuf>,
    consistency_proof: Option<PathBuf>,
    witness_set: Option<PathBuf>,
    mirror_ledger: Option<PathBuf>,
    status_proofs: Vec<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values
            .next()
            .ok_or_else(|| invalid_input("command is required: build, audit, or verify"))?;
        let mut result = Self {
            command,
            bundle: None,
            catalog: None,
            checkpoint: None,
            previous_catalog: None,
            previous_checkpoint: None,
            consistency_proof: None,
            witness_set: None,
            mirror_ledger: None,
            status_proofs: Vec::new(),
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--bundle" => {
                    result.bundle = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--catalog" => {
                    result.catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--checkpoint" => {
                    result.checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--previous-catalog" => {
                    result.previous_catalog =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--previous-checkpoint" => {
                    result.previous_checkpoint =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--consistency-proof" => {
                    result.consistency_proof =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--witness-set" => {
                    result.witness_set = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--mirror-ledger" => {
                    result.mirror_ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--status-proof" => result
                    .status_proofs
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
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
    eprintln!("evidence_publication_head_bundle <command> [options]");
    eprintln!(
        "  build --catalog FILE --checkpoint FILE --witness-set FILE --verifier PROGRAM [--previous-catalog FILE --previous-checkpoint FILE --consistency-proof FILE] [--mirror-ledger FILE] [--status-proof FILE]... [--write FILE]"
    );
    eprintln!("  audit --bundle FILE [--write FILE]");
    eprintln!("  verify --bundle FILE --verifier PROGRAM [--verifier-arg ARG]... [--write FILE]");
}
