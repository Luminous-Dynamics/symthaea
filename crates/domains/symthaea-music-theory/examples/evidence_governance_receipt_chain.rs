// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append exact governance receipts to a deterministic chain or audit a chain.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationGovernanceReceipt, CalibrationGovernanceReceiptChain,
    append_calibration_governance_receipt, audit_calibration_governance_receipt_chain,
    build_calibration_governance_receipt_chain,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    if let Some(audit_path) = arguments.audit {
        let chain: CalibrationGovernanceReceiptChain = read_json(&audit_path)?;
        let report = audit_calibration_governance_receipt_chain(&chain);
        write_json(arguments.write.as_deref(), &report)?;
        if !report.valid() {
            std::process::exit(2);
        }
        return Ok(());
    }
    let receipt_path = arguments
        .receipt
        .ok_or_else(|| invalid_input("--receipt is required"))?;
    let receipt: CalibrationGovernanceReceipt = read_json(&receipt_path)?;
    let mut chain = match arguments.chain {
        Some(path) => read_json(&path)?,
        None => build_calibration_governance_receipt_chain(
            receipt.source.clone(),
            receipt.engine_version.clone(),
        ),
    };
    append_calibration_governance_receipt(&mut chain, receipt)?;
    write_json(arguments.write.as_deref(), &chain)
}

struct Arguments {
    receipt: Option<PathBuf>,
    chain: Option<PathBuf>,
    audit: Option<PathBuf>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = Self {
            receipt: None,
            chain: None,
            audit: None,
            write: None,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--receipt" => {
                    values.receipt = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--chain" => {
                    values.chain = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--audit" => {
                    values.audit = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--write" => {
                    values.write = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--help" | "-h" => {
                    println!(
                        "usage: evidence_governance_receipt_chain --receipt RECEIPT [--chain EXISTING] [--write PATH]\n       evidence_governance_receipt_chain --audit CHAIN [--write PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(invalid_input(format!("unknown argument: {other}")).into()),
            }
        }
        Ok(values)
    }
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
