// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Create, update, and audit publication mirror ledgers.

mod support;

use std::error::Error;
use std::path::PathBuf;

use symthaea_music_theory::{
    CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationMirrorLedger, audit_calibration_publication_mirror_ledger,
    build_calibration_publication_mirror_ledger, record_calibration_publication_mirror_observation,
};

use support::publication_io::{
    atomic_json, invalid_input, next_value, parse_u64, read_json, required_path, required_string,
    required_u64, write_json,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "create" => create(arguments),
        "observe" => observe(arguments),
        "audit" => audit(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn create(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger = build_calibration_publication_mirror_ledger(
        required_string(arguments.catalog_id, "--catalog-id")?,
        required_string(arguments.authority_id, "--authority-id")?,
    );
    write_json(arguments.write.as_deref(), &ledger)
}

fn observe(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger_path = required_path(arguments.ledger, "--ledger")?;
    let mut ledger: CalibrationPublicationMirrorLedger = read_json(&ledger_path)?;
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let checkpoint: CalibrationPublicationCatalogCheckpoint =
        read_json(&required_path(arguments.checkpoint, "--checkpoint")?)?;
    let observation = record_calibration_publication_mirror_observation(
        &mut ledger,
        &catalog,
        checkpoint,
        required_string(arguments.mirror_id, "--mirror-id")?,
        required_u64(arguments.epoch, "--epoch")?,
    )?;
    let destination = arguments.write.unwrap_or(ledger_path);
    atomic_json(&destination, &ledger)?;
    if let Some(path) = arguments.write_report.as_deref() {
        atomic_json(path, &observation)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&observation)?);
    }
    Ok(())
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let ledger: CalibrationPublicationMirrorLedger =
        read_json(&required_path(arguments.ledger, "--ledger")?)?;
    let report = audit_calibration_publication_mirror_ledger(&ledger);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

struct Arguments {
    command: String,
    ledger: Option<PathBuf>,
    catalog: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    catalog_id: Option<String>,
    authority_id: Option<String>,
    mirror_id: Option<String>,
    epoch: Option<u64>,
    write: Option<PathBuf>,
    write_report: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values
            .next()
            .ok_or_else(|| invalid_input("command is required: create, observe, or audit"))?;
        let mut result = Self {
            command,
            ledger: None,
            catalog: None,
            checkpoint: None,
            catalog_id: None,
            authority_id: None,
            mirror_id: None,
            epoch: None,
            write: None,
            write_report: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--ledger" => {
                    result.ledger = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--catalog" => {
                    result.catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--checkpoint" => {
                    result.checkpoint = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--catalog-id" => result.catalog_id = Some(next_value(&mut values, &argument)?),
                "--authority-id" => result.authority_id = Some(next_value(&mut values, &argument)?),
                "--mirror-id" => result.mirror_id = Some(next_value(&mut values, &argument)?),
                "--epoch" => {
                    result.epoch = Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--write" => {
                    result.write = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--write-report" => {
                    result.write_report = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    eprintln!("evidence_publication_mirror <command> [options]");
    eprintln!("  create --catalog-id ID --authority-id ID [--write FILE]");
    eprintln!(
        "  observe --ledger FILE --catalog FILE --checkpoint FILE --mirror-id ID --epoch N [--write FILE] [--write-report FILE]"
    );
    eprintln!("  audit --ledger FILE [--write FILE]");
}
