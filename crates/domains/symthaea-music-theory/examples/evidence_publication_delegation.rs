// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build, wrap, audit, and exercise publication delegations.

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;
use symthaea_music_theory::{
    CalibrationGovernanceExport, CalibrationPublicationDelegationPayload,
    CalibrationPublicationDelegationVerifier, CalibrationPublicationPolicy,
    CalibrationSignedPublicationDelegation, CalibrationSignerIdentity,
    audit_calibration_signed_publication_delegation, authorize_calibration_publication,
    build_calibration_publication_delegation_payload,
    build_calibration_signed_publication_delegation,
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
        "payload" => build_payload(arguments),
        "wrap" => wrap_payload(arguments),
        "audit" => audit_envelope(arguments),
        "authorize" => authorize(arguments),
        other => Err(invalid_input(format!("unknown command: {other}")).into()),
    }
}

fn build_payload(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let policy: CalibrationPublicationPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let payload = build_calibration_publication_delegation_payload(
        required(arguments.delegation_id, "--delegation-id")?,
        required(arguments.delegator_id, "--delegator-id")?,
        required(arguments.delegate_id, "--delegate-id")?,
        &policy,
        arguments.source_revision,
        arguments.source_tree,
        required_u64(arguments.valid_from_epoch, "--valid-from")?,
        required_u64(arguments.valid_until_epoch, "--valid-until")?,
        required_u64(arguments.maximum_publications, "--max-publications")?,
        required(arguments.nonce, "--nonce")?,
    );
    if let Some(path) = arguments.write_bytes.as_deref() {
        atomic_bytes(path, &payload.canonical_bytes())?;
    }
    write_json(arguments.write.as_deref(), &payload)
}

fn wrap_payload(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let payload: CalibrationPublicationDelegationPayload =
        read_json(&required_path(arguments.payload, "--payload")?)?;
    let signature = std::fs::read(required_path(arguments.signature, "--signature")?)?;
    let envelope = build_calibration_signed_publication_delegation(
        payload,
        CalibrationSignerIdentity {
            key_id: required(arguments.key_id, "--key-id")?,
            algorithm: required(arguments.algorithm, "--algorithm")?,
            issuer: arguments.issuer,
        },
        &signature,
    );
    write_json(arguments.write.as_deref(), &envelope)
}

fn audit_envelope(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let envelope: CalibrationSignedPublicationDelegation =
        read_json(&required_path(arguments.envelope, "--envelope")?)?;
    let report = audit_calibration_signed_publication_delegation(&envelope);
    write_json(arguments.write.as_deref(), &report)?;
    if !report.valid() {
        std::process::exit(2);
    }
    Ok(())
}

fn authorize(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let export: CalibrationGovernanceExport =
        read_json(&required_path(arguments.export, "--export")?)?;
    let policy: CalibrationPublicationPolicy =
        read_json(&required_path(arguments.policy, "--policy")?)?;
    let delegation: CalibrationSignedPublicationDelegation =
        read_json(&required_path(arguments.envelope, "--envelope")?)?;
    let verifier = ProcessVerifier {
        program: required_path(arguments.verifier, "--verifier")?,
        args: arguments.verifier_args,
    };
    let decision = authorize_calibration_publication(
        &export,
        &policy,
        &delegation,
        &required(arguments.delegate_id, "--delegate-id")?,
        required_u64(arguments.current_epoch, "--current-epoch")?,
        required_u64(arguments.publication_ordinal, "--ordinal")?,
        &verifier,
    );
    write_json(arguments.write.as_deref(), &decision)?;
    if !decision.authorized {
        std::process::exit(2);
    }
    Ok(())
}

struct Arguments {
    command: String,
    policy: Option<PathBuf>,
    payload: Option<PathBuf>,
    envelope: Option<PathBuf>,
    export: Option<PathBuf>,
    signature: Option<PathBuf>,
    verifier: Option<PathBuf>,
    verifier_args: Vec<String>,
    delegation_id: Option<String>,
    delegator_id: Option<String>,
    delegate_id: Option<String>,
    source_revision: Option<String>,
    source_tree: Option<String>,
    valid_from_epoch: Option<u64>,
    valid_until_epoch: Option<u64>,
    maximum_publications: Option<u64>,
    current_epoch: Option<u64>,
    publication_ordinal: Option<u64>,
    nonce: Option<String>,
    key_id: Option<String>,
    algorithm: Option<String>,
    issuer: Option<String>,
    write: Option<PathBuf>,
    write_bytes: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut values = std::env::args().skip(1);
        let command = values.next().ok_or_else(|| {
            invalid_input("command is required: payload, wrap, audit, or authorize")
        })?;
        let mut result = Self {
            command,
            policy: None,
            payload: None,
            envelope: None,
            export: None,
            signature: None,
            verifier: None,
            verifier_args: Vec::new(),
            delegation_id: None,
            delegator_id: None,
            delegate_id: None,
            source_revision: None,
            source_tree: None,
            valid_from_epoch: None,
            valid_until_epoch: None,
            maximum_publications: None,
            current_epoch: None,
            publication_ordinal: None,
            nonce: None,
            key_id: None,
            algorithm: None,
            issuer: None,
            write: None,
            write_bytes: None,
        };
        while let Some(argument) = values.next() {
            match argument.as_str() {
                "--policy" => {
                    result.policy = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--payload" => {
                    result.payload = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--envelope" => {
                    result.envelope = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--export" => {
                    result.export = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--signature" => {
                    result.signature = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier" => {
                    result.verifier = Some(PathBuf::from(next_value(&mut values, &argument)?))
                }
                "--verifier-arg" => result
                    .verifier_args
                    .push(next_value(&mut values, &argument)?),
                "--delegation-id" => {
                    result.delegation_id = Some(next_value(&mut values, &argument)?)
                }
                "--delegator-id" => result.delegator_id = Some(next_value(&mut values, &argument)?),
                "--delegate-id" => result.delegate_id = Some(next_value(&mut values, &argument)?),
                "--source-revision" => {
                    result.source_revision = Some(next_value(&mut values, &argument)?)
                }
                "--source-tree" => result.source_tree = Some(next_value(&mut values, &argument)?),
                "--valid-from" => {
                    result.valid_from_epoch =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--valid-until" => {
                    result.valid_until_epoch =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--max-publications" => {
                    result.maximum_publications =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--current-epoch" => {
                    result.current_epoch =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--ordinal" => {
                    result.publication_ordinal =
                        Some(parse_u64(&next_value(&mut values, &argument)?, &argument)?)
                }
                "--nonce" => result.nonce = Some(next_value(&mut values, &argument)?),
                "--key-id" => result.key_id = Some(next_value(&mut values, &argument)?),
                "--algorithm" => result.algorithm = Some(next_value(&mut values, &argument)?),
                "--issuer" => result.issuer = Some(next_value(&mut values, &argument)?),
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
        "usage:\n  evidence_publication_delegation payload --policy POLICY --delegation-id ID --delegator-id ID --delegate-id ID --valid-from N --valid-until N --max-publications N --nonce NONCE [--source-revision REV] [--source-tree TREE] [--write JSON] [--write-bytes BIN]\n  evidence_publication_delegation wrap --payload JSON --signature BIN --key-id ID --algorithm NAME [--issuer NAME] [--write JSON]\n  evidence_publication_delegation audit --envelope JSON [--write JSON]\n  evidence_publication_delegation authorize --export EXPORT --policy POLICY --envelope JSON --delegate-id ID --current-epoch N --ordinal N --verifier PROGRAM [--verifier-arg ARG ...] [--write JSON]"
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
