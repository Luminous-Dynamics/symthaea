// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Create, update, audit, and query the append-only publication catalog.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationGovernanceExport, CalibrationGovernanceExportAttestationPayload,
    CalibrationPublicationAuthorizationDecision, CalibrationPublicationCatalog,
    CalibrationPublicationPolicy, CalibrationSignedPublicationDelegation,
    audit_calibration_publication_catalog, build_calibration_publication_catalog,
    build_calibration_publication_status_proof, publish_calibration_governance_export,
    revoke_calibration_publication, supersede_calibration_publication,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    match arguments.command.as_str() {
        "create" => create(arguments),
        "audit" => audit(arguments),
        "publish" => publish(arguments),
        "supersede" => supersede(arguments),
        "revoke" => revoke(arguments),
        "proof" => proof(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn create(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog = build_calibration_publication_catalog(
        required(arguments.catalog_id, "--catalog-id")?,
        required(arguments.authority_id, "--authority-id")?,
    );
    write_json(arguments.write.as_deref(), &catalog)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let report = audit_calibration_publication_catalog(&catalog);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn publish(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog_path = required_path(arguments.catalog, "--catalog")?;
    let mut catalog: CalibrationPublicationCatalog = read_json(&catalog_path)?;
    let export: CalibrationGovernanceExport =
        read_json(&required_path(arguments.export, "--export")?)?;
    let attestation: CalibrationGovernanceExportAttestationPayload =
        read_json(&required_path(arguments.attestation, "--attestation")?)?;
    let policy: CalibrationPublicationPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let delegation: CalibrationSignedPublicationDelegation =
        read_json(&required_path(arguments.delegation, "--delegation")?)?;
    let authorization: CalibrationPublicationAuthorizationDecision =
        read_json(&required_path(arguments.authorization, "--authorization")?)?;
    let record = publish_calibration_governance_export(
        &mut catalog,
        required(arguments.publication_id, "--publication-id")?,
        &export,
        &attestation,
        &policy,
        &delegation,
        &authorization,
        required_u64(arguments.effective_epoch, "--effective-epoch")?,
    )?;
    let destination = arguments.write.unwrap_or(catalog_path);
    atomic_json(&destination, &catalog)?;
    if let Some(path) = arguments.write_report.as_deref() {
        atomic_json(path, &record)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&record)?);
    }
    Ok(())
}

fn supersede(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog_path = required_path(arguments.catalog, "--catalog")?;
    let mut catalog: CalibrationPublicationCatalog = read_json(&catalog_path)?;
    supersede_calibration_publication(
        &mut catalog,
        &required(arguments.publication_id, "--publication-id")?,
        &required(arguments.replacement_id, "--replacement-id")?,
        required_u64(arguments.effective_epoch, "--effective-epoch")?,
    )?;
    let destination = arguments.write.unwrap_or(catalog_path);
    atomic_json(&destination, &catalog)
}

fn revoke(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog_path = required_path(arguments.catalog, "--catalog")?;
    let mut catalog: CalibrationPublicationCatalog = read_json(&catalog_path)?;
    revoke_calibration_publication(
        &mut catalog,
        &required(arguments.publication_id, "--publication-id")?,
        required(arguments.reason, "--reason")?,
        required_u64(arguments.effective_epoch, "--effective-epoch")?,
    )?;
    let destination = arguments.write.unwrap_or(catalog_path);
    atomic_json(&destination, &catalog)
}

fn proof(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let catalog: CalibrationPublicationCatalog =
        read_json(&required_path(arguments.catalog, "--catalog")?)?;
    let proof = build_calibration_publication_status_proof(
        &catalog,
        &required(arguments.publication_id, "--publication-id")?,
    )?;
    write_json(arguments.write.as_deref(), &proof)
}

struct Arguments {
    command: String,
    catalog: Option<PathBuf>,
    catalog_id: Option<String>,
    authority_id: Option<String>,
    publication_id: Option<String>,
    replacement_id: Option<String>,
    reason: Option<String>,
    effective_epoch: Option<u64>,
    export: Option<PathBuf>,
    attestation: Option<PathBuf>,
    policy: Option<PathBuf>,
    delegation: Option<PathBuf>,
    authorization: Option<PathBuf>,
    write: Option<PathBuf>,
    write_report: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values.next().ok_or_else(|| {
            invalid_input(
                "command is required: create, audit, publish, supersede, revoke, or proof",
            )
        })?;
        let mut result = Self {
            command,
            catalog: None,
            catalog_id: None,
            authority_id: None,
            publication_id: None,
            replacement_id: None,
            reason: None,
            effective_epoch: None,
            export: None,
            attestation: None,
            policy: None,
            delegation: None,
            authorization: None,
            write: None,
            write_report: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--catalog" => {
                    result.catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--catalog-id" => result.catalog_id = Some(next_value(&mut values, &argument)?),
                "--authority-id" => result.authority_id = Some(next_value(&mut values, &argument)?),
                "--publication-id" => {
                    result.publication_id = Some(next_value(&mut values, &argument)?)
                }
                "--replacement-id" => {
                    result.replacement_id = Some(next_value(&mut values, &argument)?)
                }
                "--reason" => result.reason = Some(next_value(&mut values, &argument)?),
                "--effective-epoch" => {
                    result.effective_epoch =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--export" => {
                    result.export = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--attestation" => {
                    result.attestation = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--policy" => {
                    result.policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--delegation" => {
                    result.delegation = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--authorization" => {
                    result.authorization = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    println!(
        "usage:\n  evidence_publication_catalog create --catalog-id ID --authority-id ID [--write PATH]\n  evidence_publication_catalog audit --catalog PATH [--write REPORT]\n  evidence_publication_catalog publish --catalog PATH --publication-id ID --export PATH --attestation PATH --policy PATH --delegation PATH --authorization PATH --effective-epoch N [--write CATALOG] [--write-report RECORD]\n  evidence_publication_catalog supersede --catalog PATH --publication-id OLD --replacement-id NEW --effective-epoch N [--write CATALOG]\n  evidence_publication_catalog revoke --catalog PATH --publication-id ID --reason TEXT --effective-epoch N [--write CATALOG]\n  evidence_publication_catalog proof --catalog PATH --publication-id ID [--write PROOF]"
    );
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, Box<dyn Error>> {
    value
        .parse::<u64>()
        .map_err(|_| invalid_input(format!("{flag} requires an unsigned integer")).into())
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T, Box<dyn Error>> {
    value.ok_or_else(|| invalid_input(format!("{flag} is required")).into())
}

fn required_u64(value: Option<u64>, flag: &str) -> Result<u64, Box<dyn Error>> {
    required(value, flag)
}

fn required_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, Box<dyn Error>> {
    required(value, flag)
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")).into())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn write_json<T: serde::Serialize>(path: Option<&Path>, value: &T) -> Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec_pretty(value)?;
    match path {
        Some(path) => atomic_bytes(path, &bytes)?,
        None => println!("{}", String::from_utf8(bytes)?),
    }
    Ok(())
}

fn atomic_json(path: &Path, value: &impl serde::Serialize) -> Result<(), Box<dyn Error>> {
    atomic_bytes(path, &serde_json::to_vec_pretty(value)?)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error
    })?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
