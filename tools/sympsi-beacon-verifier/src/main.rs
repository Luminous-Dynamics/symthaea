use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sympsi_beacon_verifier::{
    QuicknetBeaconResponse, VerificationReceipt, encode_hex, verify_quicknet_response,
};

const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize)]
struct RunReceipt {
    receipt_schema_version: String,
    tool_version: String,
    git_head: String,
    verification: VerificationReceipt,
}

#[derive(Serialize)]
struct ReceiptEnvelope {
    receipt: RunReceipt,
    receipt_sha256: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sympsi-beacon-verifier: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut requested_round: Option<u64> = None;
    let mut git_head: Option<String> = None;
    let mut input = PathBuf::from("tools/sympsi-beacon-verifier/fixtures/quicknet-round-1000.json");
    let mut output: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--requested-round" => {
                let value = next_value(&mut args, "--requested-round")?;
                requested_round = Some(
                    value.parse::<u64>()
                        .map_err(|error| format!("invalid --requested-round: {error}"))?,
                );
            }
            "--git-head" => {
                git_head = Some(next_value(&mut args, "--git-head")?);
            }
            "--input" => {
                input = PathBuf::from(next_value(&mut args, "--input")?);
            }
            "--out" => {
                output = Some(PathBuf::from(next_value(&mut args, "--out")?));
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let requested_round = requested_round
        .ok_or_else(|| "missing required --requested-round argument".to_string())?;
    let git_head = git_head
        .ok_or_else(|| "missing required --git-head exact-commit-sha argument".to_string())?;
    validate_git_head(&git_head)?;

    let bytes = std::fs::read(&input)
        .map_err(|error| format!("cannot read beacon response {}: {error}", input.display()))?;
    let response: QuicknetBeaconResponse = serde_json::from_slice(&bytes)
        .map_err(|error| format!("strict Quicknet response JSON failed: {error}"))?;
    let verification = verify_quicknet_response(requested_round, &response)
        .map_err(|error| error.to_string())?;

    let receipt = RunReceipt {
        receipt_schema_version: "sympsi.quicknet-run-receipt.v0.1".into(),
        tool_version: TOOL_VERSION.into(),
        git_head: git_head.to_ascii_lowercase(),
        verification,
    };
    let body = serde_json::to_vec(&receipt)
        .map_err(|error| format!("receipt body encoding failed: {error}"))?;
    let receipt_sha256 = encode_hex(&Sha256::digest(&body));
    let envelope = ReceiptEnvelope {
        receipt,
        receipt_sha256,
    };
    let mut encoded = serde_json::to_vec_pretty(&envelope)
        .map_err(|error| format!("receipt JSON encoding failed: {error}"))?;
    encoded.push(b'\n');

    if let Some(path) = output {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("cannot create receipt {}: {error}", path.display()))?;
        file.write_all(&encoded)
            .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync receipt {}: {error}", path.display()))?;
        println!("receipt_path={}", path.display());
        println!("receipt_sha256={}", envelope.receipt_sha256);
    } else {
        io::stdout()
            .write_all(&encoded)
            .map_err(|error| format!("cannot write receipt to stdout: {error}"))?;
    }
    Ok(())
}

fn validate_git_head(git_head: &str) -> Result<(), String> {
    if git_head.len() != 40 || !git_head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--git-head must be an exact 40-character hexadecimal commit SHA".into());
    }
    Ok(())
}

fn next_value(args: &mut impl Iterator<Item = String>, argument: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {argument}"))
}

fn print_help() {
    println!(
        "Offline, pinned Quicknet beacon verifier\n\
         Required: --requested-round <u64> --git-head <40-char-sha>\n\
         Optional: --input <relay-response.json> --out <new-path>\n\
         Verifies BLS signature and published randomness against pinned Quicknet.\n\
         Does not prove beacon freshness or response-before-target chronology."
    );
}
