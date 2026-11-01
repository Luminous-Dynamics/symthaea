use basalt_witness_verifier::{load_pinned_trust_store, sha256_prefixed, verify_witness_envelope, QuorumKind};
use serde::Serialize;
use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(Serialize)]
struct CliReport {
    report_version: u32,
    status: String,
    trust_store_sha256: String,
    envelope_sha256: String,
    payload_type: String,
    payload_sha256: Option<String>,
    accepted_key_ids: Vec<String>,
    error: Option<String>,
    claim_ceiling: &'static str,
}

fn main() -> ExitCode {
    let mut envelope_path = None;
    let mut trust_path = None;
    let mut trust_pin = None;
    let mut payload_type = None;
    let mut quorum = None;
    let mut out_path = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let target = match arg.as_str() {
            "--envelope" => &mut envelope_path,
            "--trust-store" => &mut trust_path,
            "--trust-store-sha256" => &mut trust_pin,
            "--payload-type" => &mut payload_type,
            "--quorum" => &mut quorum,
            "--out" => &mut out_path,
            _ => {
                eprintln!("unknown argument: {arg}");
                return ExitCode::from(2);
            }
        };
        *target = args.next();
    }

    let (Some(envelope_path), Some(trust_path), Some(trust_pin), Some(payload_type), Some(quorum)) =
        (envelope_path, trust_path, trust_pin, payload_type, quorum)
    else {
        eprintln!("usage: basalt-witness-verifier --envelope ENVELOPE.json --trust-store TRUST.json --trust-store-sha256 sha256:<64hex> --payload-type TYPE --quorum read|write [--out REPORT.json]");
        return ExitCode::from(2);
    };
    let quorum_kind = match quorum.as_str() {
        "read" => QuorumKind::Read,
        "write" => QuorumKind::Write,
        _ => {
            eprintln!("--quorum must be read or write");
            return ExitCode::from(2);
        }
    };

    let envelope_bytes = match fs::read(&envelope_path) {
        Ok(bytes) => bytes,
        Err(error) => return fail_cli("sha256:".to_string(), "sha256:".to_string(), payload_type, error.to_string(), out_path),
    };
    let trust_bytes = match fs::read(&trust_path) {
        Ok(bytes) => bytes,
        Err(error) => return fail_cli("sha256:".to_string(), sha256_prefixed(&envelope_bytes), payload_type, error.to_string(), out_path),
    };
    let actual_pin = sha256_prefixed(&trust_bytes);
    let envelope_sha = sha256_prefixed(&envelope_bytes);
    let trust = match load_pinned_trust_store(&trust_bytes, &trust_pin) {
        Ok(value) => value,
        Err(error) => return fail_cli(actual_pin, envelope_sha, payload_type, error, out_path),
    };
    match verify_witness_envelope(
        &envelope_bytes, &trust, &trust_bytes, &trust_pin, &payload_type, quorum_kind,
    ) {
        Ok(verified) => {
            let report = CliReport {
                report_version: 1,
                status: "pass".to_string(),
                trust_store_sha256: verified.trust_store_sha256,
                envelope_sha256: envelope_sha,
                payload_type: verified.payload_type,
                payload_sha256: Some(verified.payload_sha256),
                accepted_key_ids: verified.accepted_key_ids,
                error: None,
                claim_ceiling: "Authenticates canonical DSSE payload bytes against the exact pinned local trust-store bytes; does not verify trust-root rotation or witness-member durability, independence, quorum consensus, freshness challenges, or physical anti-rollback.",
            };
            emit(report, out_path, 0)
        }
        Err(error) => fail_cli(actual_pin, envelope_sha, payload_type, error, out_path),
    }
}

fn fail_cli(
    trust_sha: String,
    envelope_sha: String,
    payload_type: String,
    error: String,
    out_path: Option<String>,
) -> ExitCode {
    emit(CliReport {
        report_version: 1,
        status: "rejected".to_string(),
        trust_store_sha256: trust_sha,
        envelope_sha256: envelope_sha,
        payload_type,
        payload_sha256: None,
        accepted_key_ids: Vec::new(),
        error: Some(error),
        claim_ceiling: "No authentication claim is made for a rejected or unverified envelope.",
    }, out_path, 1)
}

fn emit(report: CliReport, path: Option<String>, status: u8) -> ExitCode {
    let bytes = match serde_json::to_vec_pretty(&report) {
        Ok(mut value) => { value.push(b'\n'); value }
        Err(error) => {
            eprintln!("cannot serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    match path {
        Some(path) => if let Err(error) = fs::write(&path, bytes) {
            eprintln!("cannot write report {path}: {error}");
            return ExitCode::from(2);
        },
        None => println!("{}", String::from_utf8_lossy(&bytes)),
    }
    if status == 0 { ExitCode::SUCCESS } else { ExitCode::from(status) }
}
