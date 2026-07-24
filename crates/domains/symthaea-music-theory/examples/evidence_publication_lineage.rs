// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build and audit exact multi-hop publication catalog lineage chains.

mod support;

use std::error::Error;
use std::path::PathBuf;

use serde::Serialize;
use symthaea_music_theory::{
    CalibrationPublicationCatalog, CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationCatalogLineageChain, audit_calibration_publication_catalog_lineage_chain,
    build_calibration_publication_catalog_lineage_chain,
    calibration_publication_catalog_lineage_checkpoint_sha256s,
    calibration_publication_catalog_lineage_terminal,
};

use support::publication_io::{invalid_input, next_value, read_json, required_path, write_json};

#[derive(Serialize)]
struct TerminalExport<'a> {
    checkpoint_sha256s: Vec<String>,
    terminal_catalog: &'a CalibrationPublicationCatalog,
    terminal_checkpoint: &'a CalibrationPublicationCatalogCheckpoint,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "build" => build(arguments),
        "audit" => audit(arguments),
        "terminal" => terminal(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn build(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    if arguments.extension_catalogs.len() != arguments.extension_checkpoints.len() {
        return Err(
            invalid_input("each --extension-catalog requires one --extension-checkpoint").into(),
        );
    }
    let anchor_catalog: CalibrationPublicationCatalog = read_json(&required_path(
        arguments.anchor_catalog,
        "--anchor-catalog",
    )?)?;
    let anchor_checkpoint: CalibrationPublicationCatalogCheckpoint = read_json(&required_path(
        arguments.anchor_checkpoint,
        "--anchor-checkpoint",
    )?)?;
    let extensions = arguments
        .extension_catalogs
        .iter()
        .zip(arguments.extension_checkpoints.iter())
        .map(|(catalog, checkpoint)| {
            Ok((
                read_json::<CalibrationPublicationCatalog>(catalog)?,
                read_json::<CalibrationPublicationCatalogCheckpoint>(checkpoint)?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let chain = build_calibration_publication_catalog_lineage_chain(
        anchor_catalog,
        anchor_checkpoint,
        extensions,
    )?;
    write_json(arguments.write.as_deref(), &chain)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let chain: CalibrationPublicationCatalogLineageChain =
        read_json(&required_path(arguments.chain, "--chain")?)?;
    let report = audit_calibration_publication_catalog_lineage_chain(&chain);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn terminal(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let chain: CalibrationPublicationCatalogLineageChain =
        read_json(&required_path(arguments.chain, "--chain")?)?;
    let (catalog, checkpoint) = calibration_publication_catalog_lineage_terminal(&chain);
    let export = TerminalExport {
        checkpoint_sha256s: calibration_publication_catalog_lineage_checkpoint_sha256s(&chain),
        terminal_catalog: catalog,
        terminal_checkpoint: checkpoint,
    };
    write_json(arguments.write.as_deref(), &export)
}

struct Arguments {
    command: String,
    anchor_catalog: Option<PathBuf>,
    anchor_checkpoint: Option<PathBuf>,
    extension_catalogs: Vec<PathBuf>,
    extension_checkpoints: Vec<PathBuf>,
    chain: Option<PathBuf>,
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
            anchor_catalog: None,
            anchor_checkpoint: None,
            extension_catalogs: Vec::new(),
            extension_checkpoints: Vec::new(),
            chain: None,
            write: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--anchor-catalog" => {
                    result.anchor_catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--anchor-checkpoint" => {
                    result.anchor_checkpoint =
                        Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--extension-catalog" => result
                    .extension_catalogs
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--extension-checkpoint" => result
                    .extension_checkpoints
                    .push(PathBuf::from(next_value(&mut values, &argument)?)),
                "--chain" => {
                    result.chain = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    eprintln!("evidence_publication_lineage <command> [options]");
    eprintln!(
        "  build --anchor-catalog FILE --anchor-checkpoint FILE [--extension-catalog FILE --extension-checkpoint FILE]... [--write FILE]"
    );
    eprintln!("  audit --chain FILE [--write FILE]");
    eprintln!("  terminal --chain FILE [--write FILE]");
}
