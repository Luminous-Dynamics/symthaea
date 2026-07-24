// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build or audit a self-contained third-party governance package.

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;
use symthaea_music_theory::{
    CalibrationGovernanceExport, CalibrationGovernanceExportAttestationPayload,
    CalibrationPublicationAuthorizationDecision, CalibrationPublicationCatalog,
    CalibrationPublicationDelegationVerifier, CalibrationPublicationPolicy,
    CalibrationPublicationStatusProof, CalibrationSignedPublicationDelegation,
    CalibrationSignerIdentity, CalibrationThirdPartyAuditPackage,
    audit_calibration_third_party_audit_package, build_calibration_third_party_audit_package,
    verify_calibration_third_party_audit_package,
};

#[derive(Serialize)]
struct VerifierRequest<'a> {
    payload_hex: String,
    key_id: &'a str,
    algorithm: &'a str,
    issuer: Option<&'a str>,
    signature_hex: String,
}

struct ProcessVerifier {
    program: PathBuf,
    args: Vec<String>,
}

impl CalibrationPublicationDelegationVerifier for ProcessVerifier {
    type Error = String;

    fn verify(
        &self,
        payload: &[u8],
        signer: &CalibrationSignerIdentity,
        signature: &[u8],
    ) -> Result<(), Self::Error> {
        let request = VerifierRequest {
            payload_hex: encode_hex(payload),
            key_id: &signer.key_id,
            algorithm: &signer.algorithm,
            issuer: signer.issuer.as_deref(),
            signature_hex: encode_hex(signature),
        };
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| error.to_string())?;
        child
            .stdin
            .as_mut()
            .ok_or_else(|| "verifier standard input is unavailable".to_owned())?
            .write_all(&serde_json::to_vec(&request).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let output = child
            .wait_with_output()
            .map_err(|error| error.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
        }
    }
}

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
    let verifier = process_verifier(&arguments)?;
    let package = build_calibration_third_party_audit_package(
        read_json(&required_path(arguments.export, "--export")?)?,
        read_json(&required_path(arguments.attestation, "--attestation")?)?,
        read_json(&required_path(arguments.policy, "--policy")?)?,
        read_json(&required_path(arguments.delegation, "--delegation")?)?,
        read_json(&required_path(arguments.authorization, "--authorization")?)?,
        read_json(&required_path(arguments.catalog, "--catalog")?)?,
        read_json(&required_path(arguments.status_proof, "--status-proof")?)?,
        &verifier,
    )?;
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &package.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &package)
}

fn audit(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let package: CalibrationThirdPartyAuditPackage =
        read_json(&required_path(arguments.package, "--package")?)?;
    let report = audit_calibration_third_party_audit_package(&package);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn verify(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let package: CalibrationThirdPartyAuditPackage =
        read_json(&required_path(arguments.package, "--package")?)?;
    let verifier = process_verifier(&arguments)?;
    let report = verify_calibration_third_party_audit_package(&package, &verifier);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() || !report.delegation_authenticated {
        std::process::exit(2);
    }
    Ok(())
}

fn process_verifier(arguments: &Arguments) -> Result<ProcessVerifier, Box<dyn Error>> {
    Ok(ProcessVerifier {
        program: required_path(arguments.verifier.clone(), "--verifier")?,
        args: arguments.verifier_args.clone(),
    })
}

struct Arguments {
    command: String,
    package: Option<PathBuf>,
    export: Option<PathBuf>,
    attestation: Option<PathBuf>,
    policy: Option<PathBuf>,
    delegation: Option<PathBuf>,
    authorization: Option<PathBuf>,
    catalog: Option<PathBuf>,
    status_proof: Option<PathBuf>,
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
            .ok_or_else(|| invalid_input("command is required: build, audit, or verify"))?;
        let mut result = Self {
            command,
            package: None,
            export: None,
            attestation: None,
            policy: None,
            delegation: None,
            authorization: None,
            catalog: None,
            status_proof: None,
            verifier: None,
            verifier_args: Vec::new(),
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--package" => {
                    result.package = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
                "--catalog" => {
                    result.catalog = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--status-proof" => {
                    result.status_proof = Some(PathBuf::from(next_value(&mut values, &argument)?))
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
    println!(
        "usage:\n  evidence_third_party_audit_package build --export PATH --attestation PATH --policy PATH --delegation PATH --authorization PATH --catalog PATH --status-proof PATH --verifier PROGRAM [--verifier-arg ARG ...] [--write JSON] [--write-bytes BIN]\n  evidence_third_party_audit_package audit --package PATH [--write REPORT]\n  evidence_third_party_audit_package verify --package PATH --verifier PROGRAM [--verifier-arg ARG ...] [--write REPORT]"
    );
}

fn required_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, Box<dyn Error>> {
    value.ok_or_else(|| invalid_input(format!("{flag} is required")).into())
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

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 0x0f) as usize] as char);
    }
    value
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
