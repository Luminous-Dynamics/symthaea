use mel_hear_009_qualifier::audit_panel;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(Serialize)]
struct ReplayOutput {
    schema: &'static str,
    panel_sha256: String,
    report: mel_hear_009_qualifier::AuditReport,
}

fn run() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "benchmarks/mel/registry/mel-hear-009-perception-reasoning-synthetic-v0.json",
            )
        });
    let bytes =
        fs::read(&path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let panel: mel_hear_009_qualifier::Panel = serde_json::from_slice(&bytes)
        .map_err(|error| format!("panel JSON is invalid: {error}"))?;
    let report = audit_panel(&panel)?;
    let digest = Sha256::digest(&bytes);
    let panel_sha256 = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("");
    let output = ReplayOutput {
        schema: "mel-hear-009-perception-reasoning-replay-v0",
        panel_sha256,
        report,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("could not serialize report: {error}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-hear-009-qualifier: {error}");
            ExitCode::FAILURE
        }
    }
}
