// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export or audit a private-safe calibration disclosure.

use std::error::Error;
use std::path::{Path, PathBuf};

use symthaea_music_theory::{
    CalibrationEvidenceBundle, CalibrationSelectiveDisclosure,
    CalibrationSelectiveDisclosurePolicy, audit_calibration_selective_disclosure,
    audit_calibration_selective_disclosure_against_bundle, build_calibration_selective_disclosure,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    if let Some(disclosure_path) = arguments.audit {
        let disclosure: CalibrationSelectiveDisclosure = read_json(&disclosure_path)?;
        let report = match arguments.bundle {
            Some(bundle_path) => {
                let bundle: CalibrationEvidenceBundle = read_json(&bundle_path)?;
                audit_calibration_selective_disclosure_against_bundle(&bundle, &disclosure)
            }
            None => audit_calibration_selective_disclosure(&disclosure),
        };
        write_json(arguments.write.as_deref(), &report)?;
        if !report.valid() {
            std::process::exit(2);
        }
        return Ok(());
    }
    let bundle_path = arguments
        .bundle
        .ok_or_else(|| invalid_input("--bundle is required when exporting"))?;
    let bundle: CalibrationEvidenceBundle = read_json(&bundle_path)?;
    let policy = match arguments.profile.as_deref().unwrap_or("public") {
        "public" => CalibrationSelectiveDisclosurePolicy::public_release(),
        "auditor" => CalibrationSelectiveDisclosurePolicy::auditor_minimal(),
        other => return Err(invalid_input(format!("unknown profile: {other}")).into()),
    };
    let disclosure = build_calibration_selective_disclosure(&bundle, policy)?;
    write_json(arguments.write.as_deref(), &disclosure)
}

struct Arguments {
    bundle: Option<PathBuf>,
    audit: Option<PathBuf>,
    profile: Option<String>,
    write: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = Self {
            bundle: None,
            audit: None,
            profile: None,
            write: None,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--bundle" => {
                    values.bundle = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--audit" => {
                    values.audit = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--profile" => values.profile = Some(next_value(&mut arguments, &argument)?),
                "--write" => {
                    values.write = Some(PathBuf::from(next_value(&mut arguments, &argument)?))
                }
                "--help" | "-h" => {
                    println!(
                        "usage: evidence_selective_disclosure --bundle BUNDLE [--profile public|auditor] [--write PATH]\n       evidence_selective_disclosure --audit DISCLOSURE [--bundle PRIVATE_BUNDLE] [--write PATH]"
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
