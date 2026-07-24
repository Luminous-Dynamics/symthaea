// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit publication checkpoints, consistency proofs, and anchored status proofs.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationCatalogConsistencyProof, CalibrationPublicationCheckpointStatusProof,
    audit_calibration_publication_catalog_checkpoint,
    audit_calibration_publication_catalog_consistency_proof,
    audit_calibration_publication_checkpoint_status_proof,
    build_calibration_publication_catalog_checkpoint,
    build_calibration_publication_catalog_consistency_proof,
    build_calibration_publication_checkpoint_status_proof,
};

use support::publication_io::{
    invalid_input, next_value, parse_u64, read_json, required_path, required_string, required_u64,
    write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "checkpoint" => checkpoint(arguments),
        "audit-checkpoint" => audit_checkpoint(arguments),
        "consistency" => consistency(arguments),
        "audit-consistency" => audit_consistency(arguments),
        "status" => status(arguments),
        "audit-status" => audit_status(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn checkpoint(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let previous = arguments
        .previous_checkpoint
        .as_deref()
        .map(read_json)
        .transpose()?;
    let checkpoint = build_calibration_publication_catalog_checkpoint(
        &catalog,
        previous.as_ref(),
        required_u64(arguments.epoch, "--epoch")?,
    )?;
    write_json(arguments.write.as_deref(), &checkpoint)
}

fn audit_checkpoint(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let report = audit_calibration_publication_catalog_checkpoint(&catalog, &checkpoint);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn consistency(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let from_catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.from_catalog, "--from-catalog")?)?;
    let from_checkpoint: CalibrationPublicationCatalogCheckpoint = read_json(&required_path(
        arguments.from_checkpoint,
        "--from-checkpoint",
    )?)?;
    let to_catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.to_catalog, "--to-catalog")?)?;
    let to_checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.to_checkpoint, "--to-checkpoint")?)?;
    let proof = build_calibration_publication_catalog_consistency_proof(
        &from_catalog,
        &from_checkpoint,
        &to_catalog,
        &to_checkpoint,
    )?;
    write_json(arguments.write.as_deref(), &proof)
}

fn audit_consistency(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let from_catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.from_catalog, "--from-catalog")?)?;
    let from_checkpoint: CalibrationPublicationCatalogCheckpoint = read_json(&required_path(
        arguments.from_checkpoint,
        "--from-checkpoint",
    )?)?;
    let to_catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.to_catalog, "--to-catalog")?)?;
    let to_checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.to_checkpoint, "--to-checkpoint")?)?;
    let proof: CalibrationPublicationCatalogConsistencyProof =
        read_json(&required_path(arguments.proof, "--proof")?)?;
    let report = audit_calibration_publication_catalog_consistency_proof(
        &from_catalog,
        &from_checkpoint,
        &to_catalog,
        &to_checkpoint,
        &proof,
    );
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn status(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let proof = build_calibration_publication_checkpoint_status_proof(
        &catalog,
        &checkpoint,
        &required_string(arguments.publication_id, "--publication-id")?,
    )?;
    write_json(arguments.write.as_deref(), &proof)
}

fn audit_status(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let proof: CalibrationPublicationCheckpointStatusProof =
        read_json(&required_path(arguments.proof, "--proof")?)?;
    let report =
        audit_calibration_publication_checkpoint_status_proof(&catalog, &checkpoint, &proof);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

struct Arguments {
    command: String,
    catalog: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    previous_checkpoint: Option<PathBuf>,
    from_catalog: Option<PathBuf>,
    from_checkpoint: Option<PathBuf>,
    to_catalog: Option<PathBuf>,
    to_checkpoint: Option<PathBuf>,
    proof: Option<PathBuf>,
    publication_id: Option<String>,
    epoch: Option<u64>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values.next().ok_or_else(|| {
            invalid_input("command is required: checkpoint, audit-checkpoint, consistency, audit-consistency, status, or audit-status")
        })?;
        let mut result = Self {
            command,
            catalog: None,
            checkpoint: None,
            previous_checkpoint: None,
            from_catalog: None,
            from_checkpoint: None,
            to_catalog: None,
            to_checkpoint: None,
            proof: None,
            publication_id: None,
            epoch: None,
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--catalog" => {
                    result.catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--checkpoint" => {
                    result.checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--previous-checkpoint" => {
                    result.previous_checkpoint =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--from-catalog" => {
                    result.from_catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--from-checkpoint" => {
                    result.from_checkpoint =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--to-catalog" => {
                    result.to_catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--to-checkpoint" => {
                    result.to_checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--proof" => {
                    result.proof = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--publication-id" => {
                    result.publication_id = Some(next_value(&mut values, &argument)?)
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
    eprintln!("evidence_publication_checkpoint <command> [options]");
    eprintln!("  checkpoint --catalog FILE --epoch N [--previous-checkpoint FILE] [--write FILE]");
    eprintln!("  audit-checkpoint --catalog FILE --checkpoint FILE [--write FILE]");
    eprintln!(
        "  consistency --from-catalog FILE --from-checkpoint FILE --to-catalog FILE --to-checkpoint FILE [--write FILE]"
    );
    eprintln!(
        "  audit-consistency --from-catalog FILE --from-checkpoint FILE --to-catalog FILE --to-checkpoint FILE --proof FILE [--write FILE]"
    );
    eprintln!("  status --catalog FILE --checkpoint FILE --publication-id ID [--write FILE]");
    eprintln!("  audit-status --catalog FILE --checkpoint FILE --proof FILE [--write FILE]");
}
