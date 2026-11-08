//! Validate a public, non-secret campaign design manifest from stdin.

use std::io::{self, Read};
use std::process::ExitCode;

use open_world_ansatz_microcosm::campaign::CampaignManifest;

const MAX_MANIFEST_BYTES: u64 = 1_048_576;

fn run() -> Result<(), String> {
    let mut bytes = Vec::new();
    io::stdin()
        .take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read manifest: {error}"))?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("campaign manifest exceeds 1 MiB".to_owned());
    }

    let manifest: CampaignManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid manifest JSON: {error}"))?;
    manifest
        .validate()
        .map_err(|error| format!("campaign manifest validation failed: {error}"))?;

    println!(
        "{{\"status\":\"valid\",\"campaign_id\":{},\"paired_cells\":{}}}",
        serde_json::to_string(&manifest.campaign_id)
            .map_err(|error| format!("cannot encode campaign id: {error}"))?,
        manifest.pairs.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}
